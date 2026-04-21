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

    from isaaclab_tasks.manager_based.locomotion.position.utils.kinematic import (
        IKObjectiveGravityTorque,
        NewtonKinematics, NewtonKinematicsCfg,
    )
    from isaaclab_tasks.manager_based.locomotion.position.mdp.retarget import (
        RetargetPipeline,
        RetargetPipelineCfg,
    )
    from isaaclab_tasks.manager_based.locomotion.position.utils.criteria import (
        BaseZError,
        FootPositionError,
        HaaLimit,
    )
    from isaaclab_tasks.manager_based.locomotion.position.utils.sampling import (
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
    kin_cfg = NewtonKinematicsCfg(
        usd_path=robot_usd, device=device,
        default_pos=(0, 0, base_height), default_joint_pos=joint_pos,
    )
    kin = NewtonKinematics(kin_cfg)
    foot_names = [n for n in kin.body_names if "foot" in n.lower()]
    foot_ids = kin.find_body_indices(foot_names)

    print(f"Robot: {args.preset} | Terrain: {args.terrain}")
    print(f"  Feet: {foot_names}")

    def gravity_extras(kin, foot_ids, n_problems, sampler, wp_mesh):
        return [IKObjectiveGravityTorque(kin.model, weight=0.001)]

    import torch

    criteria_base = {
        "foot_err": FootPositionError(num_bodies=kin.model.body_count, foot_ids=foot_ids),
        "base_z_err": BaseZError(),
    }
    if haa_pattern:
        criteria_base["haa_limit"] = HaaLimit(kin=kin, joint_pattern=haa_pattern)

    results = {}
    for label, extra_fn in [("baseline", None), ("gravity_opt", gravity_extras)]:
        print(f"\n--- {label} ---")
        torch.manual_seed(42)
        np.random.seed(42)
        pipeline = RetargetPipeline(RetargetPipelineCfg(
            kin=kin_cfg,
            sampler=SupportPolygonSamplerCfg(),
            foot_body_names=foot_names,
            ik_iterations=200,
            extra_objectives_factory=extra_fn,
        ))
        t0 = time.time()
        buf = pipeline.run(wp_mesh, origin, n_desired=args.max_robots, criteria=criteria_base)
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
