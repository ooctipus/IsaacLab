# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare IK results with and without gravity torque minimization.

Runs the retarget pipeline twice on the same terrain with the same seed:
1. Without gravity torque objective (baseline)
2. With gravity torque objective (energy-optimized)

Visualizes both side by side in viser (offset along Y axis).

Usage::

    ./isaaclab.sh -p scripts/tools/compare_gravity_ik.py --preset anymal_c
    ./isaaclab.sh -p scripts/tools/compare_gravity_ik.py --preset go2
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


def _load_terrain_module():
    builtins._isaaclab_tasks_registered = True  # type: ignore[attr-defined]
    pkg = importlib.import_module("isaaclab_tasks.manager_based.locomotion.position.terrains")
    return sys.modules.get(
        "isaaclab_tasks.manager_based.locomotion.position.terrains.terrain_cfg", pkg
    )


def main():
    import newton
    from newton.viewer import ViewerViser

    import newton.ik as ik

    from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
    from isaaclab.utils.assets import check_file_path, retrieve_file_path
    from isaaclab.utils.warp import convert_to_warp_mesh

    from isaaclab_tasks.manager_based.locomotion.position.mdp.kinematics import (
        IKObjectiveGravityTorque,
        NewtonKinematics,
    )
    from isaaclab_tasks.manager_based.locomotion.position.mdp.retarget import (
        RetargetPipeline,
        RetargetPipelineCfg,
    )
    from isaaclab_tasks.manager_based.locomotion.position.mdp_presets.criteria import (
        BaseZError,
        FootPositionError,
        HaaLimit,
    )
    from isaaclab_tasks.manager_based.locomotion.position.mdp_presets.sampling import (
        SupportPolygonSampler,
        SupportPolygonSamplerCfg,
    )

    # --- Presets ---
    from isaaclab_tasks.manager_based.locomotion.position.mdp_presets.robots.anymal_c import ANYMAL_C_HAA_PATTERN
    from isaaclab_tasks.manager_based.locomotion.position.mdp_presets.robots.go2 import GO2_HAA_PATTERN
    from isaaclab_tasks.manager_based.locomotion.position.mdp_presets.robots.robot_presets import RobotArticulationCfg
    from isaaclab_tasks.manager_based.locomotion.position.mdp_presets.robots.spot import SPOT_HAA_PATTERN

    presets = {
        "anymal_c": {"cfg": RobotArticulationCfg.anymal_c, "haa": ANYMAL_C_HAA_PATTERN},
        "go2": {"cfg": RobotArticulationCfg.go2, "haa": GO2_HAA_PATTERN},
        "spot": {"cfg": RobotArticulationCfg.spot, "haa": SPOT_HAA_PATTERN},
    }

    parser = argparse.ArgumentParser(description="Compare gravity torque IK on/off.")
    parser.add_argument("--preset", type=str, required=True, choices=list(presets.keys()))
    parser.add_argument("--terrain", type=str, default="EXTREME_STAIR")
    parser.add_argument("--difficulty", type=float, default=0.8)
    parser.add_argument("--max-robots", type=int, default=30)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    device = "cuda:0"
    rp = presets[args.preset]
    robot_cfg = rp["cfg"]
    haa_pattern = rp["haa"]

    # Resolve USD
    robot_usd = robot_cfg.spawn.usd_path
    file_status = check_file_path(robot_usd)
    if file_status == 0:
        raise FileNotFoundError(f"USD not found: {robot_usd}")
    if file_status == 2:
        robot_usd = retrieve_file_path(robot_usd, force_download=False)

    # Terrain
    tcfg = _load_terrain_module()
    terrain_presets = {
        n: getattr(tcfg, n) for n in sorted(dir(tcfg))
        if n.isupper() and isinstance(getattr(tcfg, n), SubTerrainBaseCfg)
    }
    cfg = terrain_presets[args.terrain].copy()
    cfg.difficulty = args.difficulty
    meshes, origin = cfg.function(args.difficulty, cfg)
    mesh = trimesh.util.concatenate(meshes)
    tf = np.eye(4)
    tf[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
    mesh.apply_transform(tf)
    origin += tf[0:3, -1]
    wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)

    # Robot
    base_height = robot_cfg.init_state.pos[2]
    joint_pos = robot_cfg.init_state.joint_pos
    kin = NewtonKinematics(robot_usd, device=device,
                           default_pos=(0, 0, base_height), default_joint_pos=joint_pos)
    foot_ids = [i for i, n in enumerate(kin.body_names) if "foot" in n.lower()]
    base_pos = kin.default_body_q[0][:3]
    foot_pos = np.array([kin.default_body_q[fid][:3] for fid in foot_ids])
    foot_offsets = foot_pos - base_pos
    foot_ground_offset = float(foot_pos[:, 2].mean())
    standing_height = float(base_pos[2] - foot_pos[:, 2].mean())
    foot_spread = np.linalg.norm(foot_offsets[:, :2].max(axis=0) - foot_offsets[:, :2].min(axis=0))

    print(f"Robot: {args.preset} | Terrain: {args.terrain}")
    print(f"  Feet: {[kin.body_names[i] for i in foot_ids]}")

    sampler_cfg = SupportPolygonSamplerCfg(
        search_radius=foot_spread * 0.8,
        min_diagonal_length=foot_spread * 0.3,
        min_longitudinal_spread=abs(foot_offsets[:, 0]).max() * 0.3,
        min_lateral_spread=abs(foot_offsets[:, 1]).max() * 0.3,
    )

    def make_factory(use_gravity: bool):
        def factory(n_problems):
            co = [
                ik.IKObjectivePosition(
                    link_index=fid, link_offset=wp.vec3(0, 0, 0),
                    target_positions=wp.zeros(n_problems, dtype=wp.vec3, device=device), weight=1.0,
                ) for fid in foot_ids
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
            objs = [*co, bpo, bro, jlo]
            if use_gravity:
                objs.append(IKObjectiveGravityTorque(kin.model, weight=0.001))
            return objs, co, bpo, bro
        return factory

    criteria = {
        "foot_err": FootPositionError(kin=kin, foot_ids=foot_ids),
        "base_z_err": BaseZError(),
    }
    if haa_pattern:
        criteria["haa_limit"] = HaaLimit(kin=kin, joint_pattern=haa_pattern)

    import torch

    results = {}
    for label, use_grav in [("baseline", False), ("gravity_opt", True)]:
        print(f"\n--- {label} ---")
        torch.manual_seed(42)
        np.random.seed(42)
        sampler = SupportPolygonSampler(
            sampler_cfg,
            foot_offsets=foot_offsets,
            foot_ground_offset=foot_ground_offset,
            standing_height=standing_height,
            default_joint_q=kin.default_joint_q,
        )
        pipeline = RetargetPipeline(
            kin=kin,
            sampler=sampler,
            objectives_factory=make_factory(use_grav),
            cfg=RetargetPipelineCfg(device=device, ik_iterations=200),
            contact_body_ids=foot_ids,
        )
        t0 = time.time()
        buf = pipeline.run(wp_mesh, origin, n_desired=args.max_robots, criteria=criteria)
        dt = time.time() - t0
        print(pipeline.rejection_summary)
        if hasattr(pipeline, "_reject_val"):
            print(f"  Validation: {pipeline._reject_val}")
        print(f"  Time: {dt:.2f}s")

        sel = buf._selected[:buf.num_selected].cpu().numpy()
        jq_results = buf.joint_q_result_t.cpu().numpy()
        results[label] = {"sel": sel, "jq": jq_results, "n": buf.num_selected}

    # --- Visualization: side by side ---
    Y_OFFSET = float(cfg.size[1]) + 2.0

    vis_builder = newton.ModelBuilder()
    # Terrain for baseline (y=0)
    vis_builder.add_shape_mesh(
        body=-1,
        mesh=newton.Mesh(
            vertices=mesh.vertices.flatten().tolist(),
            indices=mesh.faces.flatten().tolist(),
        ),
        scale=wp.vec3(1, 1, 1),
    )
    # Terrain for gravity_opt (y=Y_OFFSET)
    shifted_verts = mesh.vertices.copy()
    shifted_verts[:, 1] += Y_OFFSET
    vis_builder.add_shape_mesh(
        body=-1,
        mesh=newton.Mesh(
            vertices=shifted_verts.flatten().tolist(),
            indices=mesh.faces.flatten().tolist(),
        ),
        scale=wp.vec3(1, 1, 1),
    )

    template = newton.ModelBuilder()
    template.add_usd(robot_usd, collapse_fixed_joints=False)
    cpc = kin.model.joint_coord_count

    total_robots = 0
    for label in ["baseline", "gravity_opt"]:
        for _ in range(results[label]["n"]):
            vis_builder.add_world(template)
            total_robots += 1

    vis_model = vis_builder.finalize(device=device)
    vis_jq = vis_model.joint_q.numpy().copy()

    idx = 0
    for li, label in enumerate(["baseline", "gravity_opt"]):
        y_off = li * Y_OFFSET
        for si in range(results[label]["n"]):
            sel_i = results[label]["sel"][si]
            sq = results[label]["jq"][sel_i].copy()
            sq[1] += y_off  # shift Y
            vis_jq[idx * cpc:(idx + 1) * cpc] = sq
            idx += 1

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

    viewer.begin_frame(0.0)
    viewer.log_state(vis_state)
    viewer.end_frame()

    n_base = results["baseline"]["n"]
    n_grav = results["gravity_opt"]["n"]
    hn = socket.gethostname()
    print(f"\n  http://localhost:{args.port}")
    print(f"  http://{hn}:{args.port}")
    print(f"\n  Left (y=0): BASELINE ({n_base} robots)")
    print(f"  Right (y={Y_OFFSET:.0f}): GRAVITY OPTIMIZED ({n_grav} robots)")
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
