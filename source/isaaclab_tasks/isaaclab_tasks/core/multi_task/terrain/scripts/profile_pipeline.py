# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Profile the retarget pipeline stages."""

from __future__ import annotations

import sys

sys.path[:] = [p for p in sys.path if "pip_prebundle" not in p and "pip_archive" not in p]

import builtins
import importlib
import time

import numpy as np
import trimesh
import warp as wp


def main():
    import newton.ik as ik

    from isaaclab.utils.assets import check_file_path, retrieve_file_path
    from isaaclab.utils.warp import convert_to_warp_mesh

    from isaaclab_tasks.core.multi_task.kinematics import (
        IKObjectiveStabilityMargin,
        IKObjectiveTerrainCollision,
        NewtonKinematics,
        NewtonKinematicsCfg,
    )
    from isaaclab_tasks.core.multi_task.terrain.mdp_presets.robots.robot_presets import RobotArticulationCfg
    from isaaclab_tasks.core.multi_task.terrain.retarget.buffer import RetargetBuffer
    from isaaclab_tasks.core.multi_task.terrain.retarget.cfg import SupportPolygonSamplerCfg
    from isaaclab_tasks.core.multi_task.terrain.retarget.contact_sampling import SupportPolygonSampler

    device = "cuda:0"

    # Terrain
    builtins._isaaclab_tasks_registered = True
    tcfg = importlib.import_module("isaaclab_tasks.core.multi_task.terrain.terrains")
    terrain_mod = sys.modules.get("isaaclab_tasks.core.multi_task.terrain.terrains.terrain_cfg", tcfg)
    cfg = getattr(terrain_mod, "EXTREME_STAIR").copy()
    cfg.difficulty = 0.8
    meshes, origin = cfg.function(0.8, cfg)
    mesh = trimesh.util.concatenate(meshes)
    tf = np.eye(4)
    tf[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
    mesh.apply_transform(tf)
    origin += tf[0:3, -1]
    wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)

    # Robot
    robot_cfg = RobotArticulationCfg.anymal_c
    robot_usd = robot_cfg.spawn.usd_path
    fs = check_file_path(robot_usd)
    if fs == 2:
        robot_usd = retrieve_file_path(robot_usd, force_download=False)

    kin = NewtonKinematics(
        NewtonKinematicsCfg(
            usd_path=robot_usd,
            device=device,
            default_pos=(0, 0, 0.6),
            default_joint_pos=robot_cfg.init_state.joint_pos,
        )
    )
    foot_ids = [i for i, n in enumerate(kin.body_names) if "foot" in n.lower()]

    sampler = SupportPolygonSampler(
        SupportPolygonSamplerCfg(),
        kin=kin,
        foot_body_ids=foot_ids,
    )

    print("=== PROFILING RETARGET PIPELINE ===\n")

    # Stage 1: Sampling
    buf = RetargetBuffer(2000, kin.model.joint_coord_count, kin.model.body_count, len(foot_ids), device)
    t0 = time.time()
    n_written, reject = sampler(wp_mesh, origin, buf, 50)
    t_sample = time.time() - t0
    print(f"1. Sampling:        {t_sample:.3f}s  ({n_written} candidates)")

    # Stage 2: Objective construction
    N = buf.num_geometry_valid
    t0 = time.time()
    co = [
        ik.IKObjectivePosition(
            link_index=fid,
            link_offset=wp.vec3(0, 0, 0),
            target_positions=wp.zeros(N, dtype=wp.vec3, device=device),
            weight=1.0,
        )
        for fid in foot_ids
    ]
    bpo = ik.IKObjectivePosition(
        link_index=0,
        link_offset=wp.vec3(0, 0, 0),
        target_positions=wp.zeros(N, dtype=wp.vec3, device=device),
        weight=0.05,
    )
    bro = ik.IKObjectiveRotation(
        link_index=0,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.zeros(N, dtype=wp.vec4, device=device),
        weight=0.5,
    )
    jlo = ik.IKObjectiveJointLimit(
        joint_limit_lower=kin.model.joint_limit_lower, joint_limit_upper=kin.model.joint_limit_upper, weight=10.0
    )
    col = IKObjectiveTerrainCollision(
        mesh_id=wp_mesh.id, builder=kin.builder, exclude_bodies=foot_ids, weight=3.0, margin=0.01, n_samples=4
    )
    foot_ids_ccw = [int(foot_ids[j]) for j in sampler._foot_ccw_order]
    stab = IKObjectiveStabilityMargin(model=kin.model, foot_body_indices=foot_ids_ccw, weight=1.0)
    all_objs = [*co, bpo, bro, jlo, col, stab]
    t_obj = time.time() - t0
    n_residuals = sum(o.residual_dim() for o in all_objs)
    print(f"2. Obj construct:   {t_obj:.3f}s  ({len(all_objs)} objectives, {n_residuals} total residuals)")

    # Stage 3: Solver creation
    t0 = time.time()
    has_autodiff = any(not o.supports_analytic() for o in all_objs)
    jac_mode = ik.IKJacobianType.MIXED if has_autodiff else ik.IKJacobianType.ANALYTIC
    solver = kin.create_ik_solver(all_objs, N, jacobian_mode=jac_mode)
    t_solver = time.time() - t0
    print(f"3. Solver create:   {t_solver:.3f}s  (mode={jac_mode}, N={N})")

    # Stage 4: Fill targets
    t0 = time.time()
    buf.scatter_contact_targets(co, N)
    wp.copy(bpo.target_positions, buf.base_target_pos, count=N)
    wp.copy(bro.target_rotations, buf.base_target_rot, count=N)
    t_fill = time.time() - t0
    print(f"4. Fill targets:    {t_fill:.3f}s")

    # Stage 5: IK solve (200 iterations)
    jq_in = wp.from_torch(buf.joint_q_init_t[:N].contiguous())
    jq_out = wp.from_torch(buf.joint_q_result_t[:N].contiguous())
    wp.synchronize()
    t0 = time.time()
    solver.step(jq_in, jq_out, iterations=200)
    wp.synchronize()
    t_ik = time.time() - t0
    print(f"5. IK solve:        {t_ik:.3f}s  (200 iters x {N} problems)")

    # Stage 6: Per-iteration breakdown
    wp.synchronize()
    t0 = time.time()
    solver.step(jq_in, jq_out, iterations=10)
    wp.synchronize()
    t_10 = time.time() - t0
    print(f"   Per iteration:   {t_10 / 10 * 1000:.1f}ms")

    # Breakdown
    n_autodiff = sum(o.residual_dim() for o in all_objs if not o.supports_analytic())
    n_analytic = sum(o.residual_dim() for o in all_objs if o.supports_analytic())
    print(f"\n   Analytic residuals:  {n_analytic}")
    print(f"   Autodiff residuals:  {n_autodiff}")
    print(f"   Backward passes/iter: {n_autodiff}")
    print(f"   Collision probes:    {col.n_probes}")
    print(f"   Total residuals:     {n_residuals}")
    print(f"   Jacobian size:       {N} x {n_residuals} x {kin.model.joint_dof_count}")


if __name__ == "__main__":
    wp.init()
    main()
