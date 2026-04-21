# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-conforming spawn validation with batched IK and viser visualization.

Thin CLI wrapper around :class:`RetargetPipeline`.  Generates terrain mesh,
runs the pipeline, and visualises accepted/rejected candidates in viser.

Usage::

    # ANYmal-C -- USD resolved from preset config
    ./isaaclab.sh -p scripts/tools/validate_spawn_points.py \\
        --terrain EXTREME_STAIR --preset anymal_c

    # Go2
    ./isaaclab.sh -p scripts/tools/validate_spawn_points.py \\
        --terrain STEPPING_STONE --preset go2

    # Custom robot USD (no preset)
    ./isaaclab.sh -p scripts/tools/validate_spawn_points.py \\
        --terrain FLAT --robot /path/to/robot.usd --base-height 0.5
"""

from __future__ import annotations

import sys

sys.path[:] = [p for p in sys.path if "pip_prebundle" not in p and "pip_archive" not in p]

import argparse
import builtins
import importlib
import socket
import time

import numpy as np
import trimesh
import warp as wp

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils.warp import convert_to_warp_mesh


# ---------------------------------------------------------------------------
# Terrain helpers
# ---------------------------------------------------------------------------


def _load_terrain_module():
    builtins._isaaclab_tasks_registered = True  # type: ignore[attr-defined]
    pkg = importlib.import_module("isaaclab_tasks.manager_based.locomotion.position.terrains")
    return sys.modules.get(
        "isaaclab_tasks.manager_based.locomotion.position.terrains.terrain_cfg", pkg
    )


def _list_presets(module) -> dict[str, SubTerrainBaseCfg]:
    return {
        n: getattr(module, n) for n in sorted(dir(module))
        if n.isupper() and isinstance(getattr(module, n), SubTerrainBaseCfg)
    }


def _generate_mesh(cfg, difficulty):
    cfg = cfg.copy()
    cfg.difficulty = difficulty
    meshes, origin = cfg.function(difficulty, cfg)
    mesh = trimesh.util.concatenate(meshes)
    tf = np.eye(4)
    tf[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
    mesh.apply_transform(tf)
    origin += tf[0:3, -1]
    return mesh, origin


# ---------------------------------------------------------------------------
# Robot preset registry
# ---------------------------------------------------------------------------

_ROBOT_PRESETS: dict[str, dict] = {}


def _make_preset(cfg, haa_pattern: str) -> dict:
    """Build a preset dict from an :class:`ArticulationCfg` + HAA pattern.

    All data comes from the cfg -- no redundant constants.
    """
    return {
        "usd_path": cfg.spawn.usd_path,
        "base_height": cfg.init_state.pos[2],
        "joint_pos": cfg.init_state.joint_pos,
        "haa_pattern": haa_pattern,
    }


def _register_presets():
    """Lazily populate preset defaults from robot preset modules.

    Each entry is built from the robot's :class:`ArticulationCfg` --
    USD path, base height, and default joint positions are all drawn
    from the same config the sim uses.
    """
    if _ROBOT_PRESETS:
        return

    from isaaclab_tasks.manager_based.locomotion.position.mdp_presets.robots.anymal_c import (
        ANYMAL_C_HAA_PATTERN,
    )
    from isaaclab_tasks.manager_based.locomotion.position.mdp_presets.robots.robot_presets import (
        RobotArticulationCfg,
    )

    _ROBOT_PRESETS["anymal_c"] = _make_preset(RobotArticulationCfg.anymal_c, ANYMAL_C_HAA_PATTERN)

    from isaaclab_tasks.manager_based.locomotion.position.mdp_presets.robots.go2 import (
        GO2_HAA_PATTERN,
    )

    _ROBOT_PRESETS["go2"] = _make_preset(RobotArticulationCfg.go2, GO2_HAA_PATTERN)

    from isaaclab_tasks.manager_based.locomotion.position.mdp_presets.robots.spot import (
        SPOT_HAA_PATTERN,
    )

    _ROBOT_PRESETS["spot"] = _make_preset(RobotArticulationCfg.spot, SPOT_HAA_PATTERN)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import newton
    from newton.viewer import ViewerViser

    from isaaclab_tasks.manager_based.locomotion.position.utils.kinematic import (
        IKObjectiveStabilityMargin,
        IKObjectiveTerrainCollision,
        NewtonKinematicsCfg,
    )
    from isaaclab_tasks.manager_based.locomotion.position.mdp.retarget import (
        RetargetPipeline,
        RetargetPipelineCfg,
    )
    from isaaclab_tasks.manager_based.locomotion.position.utils.criteria import (
        BaseZError,
        CollisionCheck,
        HaaLimit,
    )
    from isaaclab_tasks.manager_based.locomotion.position.utils.sampling import (
        SupportPolygonSamplerCfg,
    )

    _register_presets()

    parser = argparse.ArgumentParser(description="Terrain-conforming spawn validation.")
    parser.add_argument("--terrain", type=str, default="EXTREME_STAIR")
    parser.add_argument("--difficulty", type=float, default=0.8)
    parser.add_argument("--robot", type=str, default=None,
                        help="Path to robot USD.  Optional when --preset is given.")
    parser.add_argument("--preset", type=str, default=None,
                        choices=list(_ROBOT_PRESETS.keys()),
                        help="Robot preset (resolves USD, base height, joints, HAA indices).")
    parser.add_argument("--max-robots", type=int, default=300)
    parser.add_argument("--base-height", type=float, default=None)
    parser.add_argument("--default-joints", type=str, default=None)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    tcfg = _load_terrain_module()
    presets = _list_presets(tcfg)

    if args.list:
        print("Terrain presets:")
        for n in presets:
            print(f"  {n}")
        print(f"\nRobot presets: {', '.join(_ROBOT_PRESETS.keys())}")
        sys.exit(0)
    if args.terrain not in presets:
        print(f"Unknown terrain '{args.terrain}'. Use --list.")
        sys.exit(1)

    device = "cuda:0"

    # Resolve robot preset
    rp = _ROBOT_PRESETS.get(args.preset, {}) if args.preset else {}
    base_height = args.base_height or rp.get("base_height", 0.6)
    haa_pattern = rp.get("haa_pattern")

    # Resolve USD path: --robot flag > preset usd_path
    robot_usd = args.robot or rp.get("usd_path")
    if robot_usd is None:
        parser.error("Either --robot or --preset (with a known USD path) is required.")

    from isaaclab.utils.assets import check_file_path, retrieve_file_path

    file_status = check_file_path(robot_usd)
    if file_status == 0:
        raise FileNotFoundError(f"USD not found: {robot_usd}")
    if file_status == 2:
        robot_usd = retrieve_file_path(robot_usd, force_download=False)

    # --- Terrain ---
    print(f"Terrain : {args.terrain} (difficulty={args.difficulty})")
    mesh, origin = _generate_mesh(presets[args.terrain], args.difficulty)
    wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)
    print(f"  {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")

    # --- Robot ---
    print(f"Robot   : {robot_usd}")
    if args.preset:
        print(f"Preset  : {args.preset}")

    if args.default_joints:
        default_jpos: dict[str, float] | None = {".*": float(args.default_joints.split(",")[0])}
    else:
        default_jpos = rp.get("joint_pos")

    foot_names = [n for n in rp.get("foot_names", []) if n]
    if not foot_names:
        from isaaclab_tasks.manager_based.locomotion.position.utils.kinematic import NewtonKinematics
        _tmp = NewtonKinematics(NewtonKinematicsCfg(usd_path=robot_usd, device=device,
                                                     default_pos=(0.0, 0.0, base_height)))
        foot_names = [n for n in _tmp.body_names if "foot" in n.lower()]
        del _tmp

    kin_cfg = NewtonKinematicsCfg(
        usd_path=robot_usd,
        device=device,
        default_pos=(0.0, 0.0, base_height),
        default_joint_pos=default_jpos,
    )

    def extra_objectives(kin, foot_ids, n_problems, sampler, wp_mesh):
        col = IKObjectiveTerrainCollision(
            mesh_id=wp_mesh.id, builder=kin.builder,
            exclude_bodies=foot_ids, weight=3.0, margin=0.01, n_samples=4,
        )
        active_mask_wp = wp.from_torch(
            sampler.active_mask[:n_problems].contiguous().to(device), dtype=wp.int32,
        )
        stab = IKObjectiveStabilityMargin(
            model=kin.model, foot_body_indices=foot_ids,
            active_mask=active_mask_wp, weight=1.0,
        )
        return [col, stab]

    pipeline_cfg = RetargetPipelineCfg(
        kin=kin_cfg,
        sampler=SupportPolygonSamplerCfg(oversample_candidates=10),
        foot_body_names=foot_names,
        extra_objectives_factory=extra_objectives,
    )

    t0 = time.time()
    pipeline = RetargetPipeline(pipeline_cfg)
    kin = pipeline.kin
    foot_ids = pipeline.foot_body_ids

    geom = kin.foot_geometry(foot_ids)
    foot_offsets = geom["foot_offsets"]
    standing_height = geom["standing_height"]
    print(f"  Bodies={kin.model.body_count} Joints={kin.model.joint_count}")
    print(f"  Feet: {foot_names} -> ids {foot_ids}")
    foot_spread = np.linalg.norm(foot_offsets[:, :2].max(axis=0) - foot_offsets[:, :2].min(axis=0))
    print(f"  standing_height={standing_height:.3f}m foot_spread={foot_spread:.3f}m")

    print("\n--- Running retarget pipeline ---")

    import torch

    def cost_filter(buffer, N):
        """Reject candidates with solver cost > 3x median (unresolved collision)."""
        if not hasattr(pipeline, "_solver_costs"):
            return torch.ones(N, device=buffer.device, dtype=torch.bool)
        costs = pipeline._solver_costs[:N]
        median = costs.median()
        return costs < median * 3.0

    criteria = {
        "cost": cost_filter,
        "base_z_err": BaseZError(),
        "collision": CollisionCheck(
            kin=kin, wp_mesh=wp_mesh, exclude_bodies=foot_ids, n_samples=16, max_pen=0.02,
        ),
    }
    if haa_pattern:
        criteria["haa_limit"] = HaaLimit(kin=kin, joint_pattern=haa_pattern, max_angle=0.95)

    buf = pipeline.run(wp_mesh, origin, n_desired=args.max_robots, criteria=criteria)
    t_total = time.time() - t0
    if hasattr(pipeline, "_ik_iterations_used"):
        print(f"  IK converged in {pipeline._ik_iterations_used} iterations")

    print(pipeline.rejection_summary)
    if hasattr(pipeline, "_reject_val"):
        print(f"  Validation: {pipeline._reject_val}")
    print(f"  Time: {t_total:.2f}s")

    if buf.num_selected == 0:
        print("No valid candidates. Exiting.")
        sys.exit(1)

    # --- Visualization ---
    print(f"\n--- Visualization ---")
    sel = buf._selected[:buf.num_selected].cpu().numpy()
    jq_results = buf.joint_q_result_t.cpu().numpy()
    ct_np = buf.contact_targets_t.cpu().numpy()
    nc = len(foot_ids)
    cpc = kin.model.joint_coord_count

    solved_qs = [jq_results[idx] for idx in sel]
    selected_feet = [ct_np[idx * nc:(idx + 1) * nc] for idx in sel]

    vis_builder = newton.ModelBuilder()
    vis_builder.add_shape_mesh(
        body=-1,
        mesh=newton.Mesh(
            vertices=mesh.vertices.flatten().tolist(),
            indices=mesh.faces.flatten().tolist(),
        ),
        scale=wp.vec3(1, 1, 1),
    )

    if solved_qs:
        template = newton.ModelBuilder()
        template.add_usd(robot_usd, collapse_fixed_joints=False)
        for _ in solved_qs:
            vis_builder.add_world(template)

    vis_model = vis_builder.finalize(device=device)
    if solved_qs:
        vis_jq = vis_model.joint_q.numpy().copy()
        for i, sq in enumerate(solved_qs):
            vis_jq[i * cpc:(i + 1) * cpc] = sq
        vis_model.joint_q = wp.array(vis_jq, dtype=float, device=device)

    vis_state = vis_model.state()
    newton.eval_fk(
        vis_model, vis_model.joint_q,
        wp.zeros(vis_model.joint_dof_count, dtype=float, device=device),
        vis_state,
    )

    viewer = ViewerViser(port=args.port)
    viewer.set_model(vis_model)
    viewer.set_world_offsets((0.0, 0.0, 0.0))

    from isaaclab_tasks.manager_based.locomotion.position.terrains.utils.patch_sampling_cfg import (
        CircleFootprintCfg,
        MorphologicalPatchSamplingCfg,
    )
    import torch

    fc_cfg = MorphologicalPatchSamplingCfg(
        num_patches=5000,
        footprint=CircleFootprintCfg(radius=0.04),
        max_height_diff=0.03,
        horizontal_scale=0.03,
        oversample_ratio=3.0,
    )

    fp = fc_cfg.func(wp_mesh, origin, fc_cfg)
    ot = torch.tensor(origin, dtype=torch.float, device=fp.device)
    fp[:, :3] += ot
    foot_pts = fp[:, :3].cpu().numpy()
    fl = foot_pts.copy()
    fl[:, 2] += 0.02
    viewer.log_points(
        "foot_candidates",
        wp.array(fl.tolist(), dtype=wp.vec3, device=device),
        radii=0.01,
        colors=(0.15, 0.5, 0.15),
    )

    if selected_feet:
        sf = np.concatenate(selected_feet)
        sf_l = sf.copy()
        if sf_l.ndim == 2:
            sf_l[:, 2] += 0.02
        viewer.log_points(
            "selected_feet",
            wp.array(sf_l.tolist(), dtype=wp.vec3, device=device),
            radii=0.03,
            colors=(0, 1, 1),
        )

    # Visualize collision probe points on ALL robots
    from isaaclab_tasks.manager_based.locomotion.position.utils.kinematic import _build_collision_probes

    probe_bodies, probe_offsets = _build_collision_probes(kin.builder, foot_ids, n_samples=16)
    if solved_qs:
        from isaaclab.utils.math import quat_apply as _qa
        import torch as _torch

        all_probe_world = []
        for sq in solved_qs:
            fk_jq = wp.array(sq, dtype=float, device=device)
            fk_state = kin.eval_fk(fk_jq)
            bq_np = fk_state.body_q.numpy()
            for i in range(len(probe_bodies)):
                bid = probe_bodies[i]
                off = np.array(probe_offsets[i], dtype=np.float32)
                bq = bq_np[bid]
                pos = bq[:3]
                quat = bq[3:7]
                q_t = _torch.tensor(quat, dtype=_torch.float32).unsqueeze(0)
                o_t = _torch.tensor(off, dtype=_torch.float32).unsqueeze(0)
                rotated = _qa(q_t, o_t).squeeze(0).numpy()
                all_probe_world.append((pos + rotated).tolist())
        viewer.log_points(
            "collision_probes",
            wp.array(all_probe_world, dtype=wp.vec3, device=device),
            radii=0.005,
            colors=(1.0, 0.2, 0.2),
        )
        print(f"  Collision probes: {len(all_probe_world)} points on {len(solved_qs)} robots (red)")

    viewer.begin_frame(0.0)
    viewer.log_state(vis_state)
    viewer.end_frame()

    hn = socket.gethostname()
    print(f"\n  http://localhost:{args.port}")
    print(f"  http://{hn}:{args.port}")
    print("  Green=candidates, Cyan=selected feet, Red=collision probes")
    print(f"\n  {len(solved_qs)} robots placed.")
    print("\nPress Ctrl+C to stop.\n")

    try:
        while viewer.is_running():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        viewer.close()


if __name__ == "__main__":
    wp.init()
    main()
