# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-conforming spawn validation with batched IK and viser visualization.

Thin CLI wrapper around :class:`RetargetPipeline`.  Generates terrain mesh,
runs the pipeline, and visualises accepted/rejected candidates in viser.

No Isaac Sim required.

Usage::

    ./isaaclab.sh -p scripts/tools/validate_spawn_points.py \\
        --terrain EXTREME_STAIR --robot /path/to/anymal_c.usd

    ./isaaclab.sh -p scripts/tools/validate_spawn_points.py \\
        --terrain STEPPING_STONE --robot /path/to/anymal_c.usd \\
        --difficulty 0.5 --max-robots 100 --port 8080
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
# Main
# ---------------------------------------------------------------------------


def main():
    import newton
    from newton.viewer import ViewerViser

    import newton.ik as ik

    from isaaclab_tasks.manager_based.locomotion.position.mdp.kinematics import NewtonKinematics
    from isaaclab_tasks.manager_based.locomotion.position.mdp.retarget import (
        RetargetPipeline,
        RetargetPipelineCfg,
    )

    parser = argparse.ArgumentParser(description="Terrain-conforming spawn validation.")
    parser.add_argument("--terrain", type=str, default="EXTREME_STAIR")
    parser.add_argument("--difficulty", type=float, default=0.8)
    parser.add_argument("--robot", type=str, required=True, help="Path to robot USD.")
    parser.add_argument("--max-robots", type=int, default=300)
    parser.add_argument("--base-height", type=float, default=0.6)
    parser.add_argument("--default-joints", type=str, default=None)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    tcfg = _load_terrain_module()
    presets = _list_presets(tcfg)

    if args.list:
        for n in presets:
            print(f"  {n}")
        sys.exit(0)
    if args.terrain not in presets:
        print(f"Unknown terrain '{args.terrain}'. Use --list.")
        sys.exit(1)

    device = "cuda:0"

    # --- Terrain ---
    print(f"Terrain : {args.terrain} (difficulty={args.difficulty})")
    mesh, origin = _generate_mesh(presets[args.terrain], args.difficulty)
    wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)
    print(f"  {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")

    # --- Robot ---
    print(f"Robot   : {args.robot}")
    if args.default_joints:
        default_jpos = np.array([float(x) for x in args.default_joints.split(",")], dtype=np.float32)
    else:
        default_jpos = np.array([0, 0.4, -0.8, 0, -0.4, 0.8, 0, 0.4, -0.8, 0, -0.4, 0.8], dtype=np.float32)

    kin = NewtonKinematics(args.robot, device=device,
                           default_pos=(0.0, 0.0, args.base_height), default_joint_pos=default_jpos)
    foot_ids = [i for i, n in enumerate(kin.body_names) if "foot" in n.lower()]
    foot_names = [kin.body_names[i] for i in foot_ids]
    print(f"  Bodies={kin.model.body_count} Joints={kin.model.joint_count}")
    print(f"  Feet: {foot_names} -> ids {foot_ids}")

    base_pos = kin.default_body_q[0][:3]
    foot_pos_default = np.array([kin.default_body_q[fid][:3] for fid in foot_ids])
    foot_offsets = foot_pos_default - base_pos
    foot_ground_offset = float(foot_pos_default[:, 2].mean())
    standing_height = float(base_pos[2] - foot_pos_default[:, 2].mean())
    print(f"  standing_height={standing_height:.3f}m foot_ground_offset={foot_ground_offset:.3f}m")

    # --- Pipeline ---
    def objectives_factory(n_problems):
        co = [
            ik.IKObjectivePosition(
                link_index=fid, link_offset=wp.vec3(0, 0, 0),
                target_positions=wp.zeros(n_problems, dtype=wp.vec3, device=device), weight=1.0,
            )
            for fid in foot_ids
        ]
        bpo = ik.IKObjectivePosition(
            link_index=0, link_offset=wp.vec3(0, 0, 0),
            target_positions=wp.zeros(n_problems, dtype=wp.vec3, device=device), weight=0.05,
        )
        bro = ik.IKObjectiveRotation(
            link_index=0, link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.zeros(n_problems, dtype=wp.vec4, device=device), weight=0.5,
        )
        jlo = ik.IKObjectiveJointLimit(
            joint_limit_lower=kin.model.joint_limit_lower,
            joint_limit_upper=kin.model.joint_limit_upper, weight=10.0,
        )
        return [*co, bpo, bro, jlo], co, bpo, bro

    print("\n--- Running retarget pipeline ---")
    t0 = time.time()
    pipeline = RetargetPipeline(
        kin=kin,
        objectives_factory=objectives_factory,
        cfg=RetargetPipelineCfg(device=device),
        contact_body_ids=foot_ids,
        foot_offsets=foot_offsets,
        foot_ground_offset=foot_ground_offset,
        standing_height=standing_height,
        default_joint_q=kin.default_joint_q,
    )
    from isaaclab_tasks.manager_based.locomotion.position.mdp_presets.robots.anymal_c import (
        base_z_error,
        foot_position_error,
        haa_limit,
        joint_margin,
    )

    criteria = {
        "foot_err": foot_position_error(kin, foot_ids),
        "joint_margin": joint_margin(kin),
        "haa_limit": haa_limit(),
        "base_z_err": base_z_error(),
    }
    buf = pipeline.run(wp_mesh, origin, n_desired=args.max_robots, criteria=criteria)
    t_total = time.time() - t0

    print(pipeline.rejection_summary)
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
        template.add_usd(str(args.robot), collapse_fixed_joints=False)
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

    viewer.begin_frame(0.0)
    viewer.log_state(vis_state)
    viewer.end_frame()

    hn = socket.gethostname()
    print(f"\n  http://localhost:{args.port}")
    print(f"  http://{hn}:{args.port}")
    print("  Green=candidates, Cyan=selected feet")
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
