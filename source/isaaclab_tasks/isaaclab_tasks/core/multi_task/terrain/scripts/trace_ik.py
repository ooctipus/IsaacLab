# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-iteration IK tracer for the retarget pipeline.

Headless diagnostic: for a single (sub-terrain, robot, foot-subset) case,
run the retarget sampler once, take the first ``--trace_samples`` geometry-
valid candidates, then step the IK solver one iteration at a time. After
each step: evaluate FK, evaluate every IK objective's residual into a
per-objective scratch buffer, and record per-problem metrics -- base
position, base rpy, per-foot position/error, and per-objective
sum-of-squares cost.

Exists because tuning IK weights by eyeballing viser is ambiguous -- the
trace exposes *what the solver is actually optimizing* (whether a weight
is dominating, whether targets are wrong, whether the solver converged).

Usage (same flags as ``validate_spawn_points`` where applicable)::

    SCRIPT=source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/position/utils/tools/trace_ik.py
    ./isaaclab.sh -p $SCRIPT --sub_terrain FLAT --robot anymal_c \\
        --exclude_feet "LH|RH" --trace_samples 2 --max_iters 100 \\
        --output /tmp/trace.json
"""

from __future__ import annotations

import sys

sys.path[:] = [p for p in sys.path if "pip_prebundle" not in p and "pip_archive" not in p]

import argparse
import builtins
import json
import re
import time

import numpy as np

# Required transitively: wp.to_torch → torch tensors used for sum/view below.
import torch  # noqa: F401
import warp as wp

from isaaclab.utils.warp import convert_to_warp_mesh

# Reuse terrain helpers from validate_spawn_points (same directory).
from isaaclab_tasks.core.multi_task.terrain.scripts.validate_spawn_points import (
    _ROBOT_NAMES,
    _sub_terrain_lookup,
    _sub_terrain_mesh,
    _terrain_preset_lookup,
    _terrain_preset_mesh,
)


def _register_position_task() -> None:
    builtins._isaaclab_tasks_registered = True  # type: ignore[attr-defined]
    import importlib

    importlib.import_module("isaaclab_tasks.core.multi_task.terrain.terrains")
    importlib.import_module("isaaclab_tasks.core.multi_task.terrain.mdp_presets.robots")


def _quat_to_rpy_deg(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert a batch of quats (xyzw) to roll/pitch/yaw in degrees."""
    q = np.asarray(q_xyzw, dtype=np.float64)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return np.rad2deg(np.stack([roll, pitch, yaw], axis=-1))


def _quat_angle_deg(q_a_xyzw: np.ndarray, q_b_xyzw: np.ndarray) -> np.ndarray:
    """Minimal rotation angle between two unit quats (xyzw), in degrees."""
    dot = np.abs(np.sum(q_a_xyzw * q_b_xyzw, axis=-1))
    dot = np.clip(dot, 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dot))


def _label_objectives(
    all_objs: list,
    foot_names: list[str],
    n_feet: int,
) -> list[str]:
    """Human-readable labels for objectives in pipeline-build order."""
    labels: list[str] = []
    for i in range(n_feet):
        labels.append(f"foot[{foot_names[i]}]")
    labels.extend(["base_pos", "base_rot", "joint_limit"])
    for obj in all_objs[n_feet + 3 :]:
        labels.append(type(obj).__name__)
    return labels


def main():
    import newton.ik as ik

    from isaaclab_tasks.core.multi_task.kinematics import (
        NewtonKinematics,
        NewtonKinematicsCfg,
    )
    from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.cfg import (
        IKObjectiveGravityTorqueCfg,
        IKObjectiveJointRegularizeCfg,
        IKObjectiveStabilityMarginCfg,
        IKObjectiveTerrainCollisionCfg,
    )
    from isaaclab_tasks.core.multi_task.terrain.retarget import (
        RetargetPipeline,
        RetargetPipelineCfg,
        resolve_foot_body_names,
    )
    from isaaclab_tasks.core.multi_task.terrain.retarget.cfg import (
        PatchSamplingCfg,
        SamplerCfg,
    )
    from isaaclab_tasks.core.multi_task.terrain.retarget.criteria_cfg import (
        CollisionCheckCfg,
        FootPositionErrorCfg,
        JointWithinLimitCfg,
        LateralHipLimitCfg,
        SolverCostOutlierCfg,
        SupportPolygonStabilityCfg,
    )

    _register_position_task()

    from isaaclab_tasks.core.multi_task.terrain.mdp_presets.robots.robot_presets import (
        FootBodyNamesCfg,
        RetargetJointRegularizeTargetsCfg,
        RetargetLateralHipJointPatternCfg,
        RobotArticulationCfg,
    )

    sub_terrains = _sub_terrain_lookup()
    terrain_presets = _terrain_preset_lookup()

    parser = argparse.ArgumentParser(description="Per-iteration IK tracer.")
    parser.add_argument("--terrain", type=str, default="all", choices=sorted(terrain_presets.keys()))
    parser.add_argument("--sub_terrain", type=str, default=None, choices=sorted(sub_terrains.keys()))
    parser.add_argument("--difficulty", type=float, default=0.8)
    parser.add_argument("--robot", type=str, default="anymal_c", choices=sorted(_ROBOT_NAMES))
    parser.add_argument("--exclude_feet", type=str, default=None)
    parser.add_argument("--trace_samples", type=int, default=1, help="Number of geometry-valid candidates to trace.")
    parser.add_argument("--max_iters", type=int, default=100, help="IK iterations to step (one at a time).")
    parser.add_argument(
        "--oversample",
        type=int,
        default=64,
        help="Sampler ``n_desired`` (raised until at least trace_samples geom-valid candidates land).",
    )
    parser.add_argument("--output", type=str, default=None, help="Write per-iter JSON trace here (optional).")
    parser.add_argument(
        "--override_base_pitch_deg",
        type=float,
        default=None,
        help=(
            "Add a body-frame pitch to the sampler's base_target_rot (deg). Positive"
            " pitches the nose down (handstand); negative pitches up (rearing)."
        ),
    )
    parser.add_argument(
        "--override_base_z",
        type=float,
        default=None,
        help="Replace base_target_pos.z with this value [m]. Other axes left as-sampled.",
    )
    parser.add_argument(
        "--compress_feet",
        type=float,
        default=None,
        help=(
            "Scale factor applied to each foot target around the foot-pair centroid (per-sample)."
            " 1.0 = unchanged; 0.3 = compress spread to 30 percent. Use with nc=2 to test whether IK"
            " can achieve a handstand when the feet are close enough together for it."
        ),
    )
    args = parser.parse_args()

    device = "cuda:0"

    # --- Robot ---
    robot_cfg = getattr(RobotArticulationCfg, args.robot)
    robot_usd = robot_cfg.spawn.usd_path
    base_height = robot_cfg.init_state.pos[2]
    default_jpos = robot_cfg.init_state.joint_pos
    lateral_hip_pattern = getattr(
        RetargetLateralHipJointPatternCfg, args.robot, RetargetLateralHipJointPatternCfg().default
    )

    from isaaclab.utils.assets import check_file_path, retrieve_file_path

    s = check_file_path(robot_usd)
    if s == 0:
        raise FileNotFoundError(f"USD not found: {robot_usd}")
    if s == 2:
        robot_usd = retrieve_file_path(robot_usd, force_download=False)

    # --- Terrain ---
    sampler_x_range = sampler_y_range = None
    if args.sub_terrain:
        print(f"Terrain : sub_terrain={args.sub_terrain} (difficulty={args.difficulty})")
        mesh, origin = _sub_terrain_mesh(sub_terrains[args.sub_terrain], args.difficulty)
    else:
        print(f"Terrain : preset={args.terrain}")
        mesh, origin, sampler_x_range, sampler_y_range = _terrain_preset_mesh(terrain_presets[args.terrain], device)
    wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)

    # --- Kin + feet ---
    kin_cfg = NewtonKinematicsCfg(
        usd_path=robot_usd,
        device=device,
        default_pos=(0.0, 0.0, base_height),
        default_joint_pos=default_jpos,
    )
    _tmp = NewtonKinematics(kin_cfg)
    foot_spec = getattr(FootBodyNamesCfg, args.robot, ".*foot")
    foot_names = resolve_foot_body_names(foot_spec, _tmp.body_names)
    del _tmp
    if args.exclude_feet:
        pat = re.compile(args.exclude_feet, re.IGNORECASE)
        foot_names = [n for n in foot_names if not pat.search(n)]
    print(f"Feet    : {foot_names}")

    # --- Pipeline cfg (mirrors validate_spawn_points) ---
    criteria_list = [CollisionCheckCfg(n_samples=16, max_pen=0.02), JointWithinLimitCfg(limit_ratio=0.9)]
    if lateral_hip_pattern:
        criteria_list.append(LateralHipLimitCfg(joint_pattern=lateral_hip_pattern, max_angle=1.05))
    criteria_list += [
        SupportPolygonStabilityCfg(),
        FootPositionErrorCfg(max_err=0.25, aggregate="sum"),
        SolverCostOutlierCfg(threshold_multiplier=3.0),
    ]
    joint_regularize_targets = getattr(RetargetJointRegularizeTargetsCfg, args.robot, {})

    base_rot_weight = 0.5
    base_pos_weight = 0.05
    gravity_weight = 0.02
    print(f"IK wts  : base_rot={base_rot_weight}, base_pos={base_pos_weight}, gravity={gravity_weight}")

    patch_cfg = PatchSamplingCfg(x_range=sampler_x_range, y_range=sampler_y_range)
    sampler_cfg = SamplerCfg(patch=patch_cfg)

    pipeline_cfg = RetargetPipelineCfg(
        kin=kin_cfg,
        sampler=sampler_cfg,
        foot_body_names=foot_names,
        lateral_hip_joint_pattern=lateral_hip_pattern,
        joint_regularize_targets=joint_regularize_targets,
        base_pos_weight=base_pos_weight,
        base_rot_weight=base_rot_weight,
        extra_objectives=[
            IKObjectiveTerrainCollisionCfg(weight=3.0, margin=0.05, n_samples=4),
            IKObjectiveStabilityMarginCfg(weight=1.0),
            IKObjectiveGravityTorqueCfg(weight=gravity_weight),
            IKObjectiveJointRegularizeCfg(weight=0.02),
        ],
        criteria=criteria_list,
        ik_iterations=args.max_iters,
    )

    pipeline = RetargetPipeline(pipeline_cfg)
    kin = pipeline.kin
    foot_ids = pipeline.foot_body_ids
    n_feet = len(foot_ids)
    print(f"Bodies={kin.model.body_count} Joints={kin.model.joint_count} Feet={n_feet}")

    # --- Sample ---
    n_desired = max(args.oversample, args.trace_samples * 16)
    pipeline._ensure_buffer(n_desired)
    pipeline.buffer.reset()
    t0 = time.time()
    sampler_out = pipeline.sampler(wp_mesh, origin, pipeline.buffer, n_desired)
    t_sample = time.time() - t0
    n_geom = pipeline.buffer.num_geometry_valid
    print(f"Sampler : n_desired={n_desired}, geom_valid={n_geom} ({t_sample:.2f}s)")
    print(f"          reject_stats={sampler_out.reject_stats}")
    if n_geom == 0:
        print("No geometry-valid candidates; nothing to trace.")
        sys.exit(1)
    N = min(n_geom, args.trace_samples)
    print(f"Tracing : {N} candidates, {args.max_iters} iterations each")

    # --- Apply target overrides (before objectives read targets from the buffer) ---
    if args.override_base_pitch_deg is not None:
        theta = np.deg2rad(args.override_base_pitch_deg)
        q_pitch = np.array([0.0, np.sin(theta / 2.0), 0.0, np.cos(theta / 2.0)], dtype=np.float32)
        q0 = pipeline.buffer.base_target_rot_t[:N].detach().cpu().numpy().copy()  # [N,4] xyzw
        # Body-frame composition: q_new = q0 * q_pitch.
        qx, qy, qz, qw = q0[:, 0], q0[:, 1], q0[:, 2], q0[:, 3]
        px, py, pz, pw = q_pitch
        nx = qw * px + qx * pw + qy * pz - qz * py
        ny = qw * py - qx * pz + qy * pw + qz * px
        nz = qw * pz + qx * py - qy * px + qz * pw
        nw = qw * pw - qx * px - qy * py - qz * pz
        q_new = np.stack([nx, ny, nz, nw], axis=-1).astype(np.float32)
        pipeline.buffer.base_target_rot_t[:N] = (
            __import__("torch").from_numpy(q_new).to(pipeline.buffer.base_target_rot_t.device)
        )
        print(f"Override: base_target pitch += {args.override_base_pitch_deg} deg")
    if args.override_base_z is not None:
        pipeline.buffer.base_target_pos_t[:N, 2] = float(args.override_base_z)
        print(f"Override: base_target_pos.z = {args.override_base_z}")
    if args.compress_feet is not None:
        nc = pipeline.buffer.num_contacts
        ft = pipeline.buffer.contact_targets_t[: N * nc].view(N, nc, 3)
        centroid = ft.mean(dim=1, keepdim=True)
        ft[:] = centroid + (ft - centroid) * float(args.compress_feet)
        print(f"Override: foot spread scaled by {args.compress_feet} around per-sample centroid")

    # --- Build objectives + solver for N ---
    all_objs, contact_objs, base_pos_obj, base_rot_obj = pipeline._build_objectives(N, wp_mesh)
    has_autodiff = any(not o.supports_analytic() for o in all_objs)
    jac_mode = ik.IKJacobianType.MIXED if has_autodiff else ik.IKJacobianType.ANALYTIC
    solver = kin.create_ik_solver(all_objs, N, jacobian_mode=jac_mode)

    if contact_objs:
        pipeline.buffer.scatter_contact_targets(contact_objs, N)
    wp.copy(base_pos_obj.target_positions, pipeline.buffer.base_target_pos, count=N)
    wp.copy(base_rot_obj.target_rotations, pipeline.buffer.base_target_rot, count=N)

    jq_in = wp.from_torch(pipeline.buffer.joint_q_init_t[:N].contiguous())
    jq_out = wp.from_torch(pipeline.buffer.joint_q_result_t[:N].contiguous())

    # --- Extract fixed targets for reporting ---
    base_target_pos_np = pipeline.buffer.base_target_pos_t[:N].detach().cpu().numpy().copy()
    base_target_rot_np = pipeline.buffer.base_target_rot_t[:N].detach().cpu().numpy().copy()  # xyzw
    nc = pipeline.buffer.num_contacts
    foot_target_np = pipeline.buffer.contact_targets_t[: N * nc].detach().cpu().numpy().copy().reshape(N, nc, 3)

    # --- Body-q scratch ---
    body_q_scratch = wp.zeros((N, kin.model.body_count), dtype=wp.transformf, device=device)

    def _snapshot(iter_idx: int, joint_q: wp.array, include_solver_cost: bool) -> list[dict]:
        """Eval FK and build per-problem records.

        Per-objective residual breakdown is intentionally skipped: Newton's
        built-in IK objectives share an internal residual buffer layout with
        the solver, and calling ``compute_residuals`` against a scratch
        array out-of-band tripped a CUDA illegal-memory-access. Total IK
        cost is read from ``solver.costs`` instead (populated by each
        ``solver.step``). Per-foot pos and base pos/rot errors give the
        structural breakdown that matters for weight tuning.
        """
        kin.eval_fk_batched(joint_q, body_q=body_q_scratch)
        if include_solver_cost:
            total_cost_np = wp.to_torch(solver.costs)[:N].detach().cpu().numpy().copy()
        else:
            total_cost_np = np.full(N, np.nan, dtype=np.float32)
        # body_q: (N, body_count) of transformf stored as (px,py,pz, qx,qy,qz,qw)
        body_q_np = wp.to_torch(body_q_scratch).detach().cpu().numpy()  # [N, body_count, 7]
        base_pos = body_q_np[:, 0, 0:3]
        base_q = body_q_np[:, 0, 3:7]  # xyzw
        base_rpy = _quat_to_rpy_deg(base_q)
        base_rot_angle = _quat_angle_deg(base_q, base_target_rot_np)
        base_pos_err = np.linalg.norm(base_pos - base_target_pos_np, axis=-1)
        # Per-foot pos and err:
        foot_pos = np.stack([body_q_np[:, foot_ids[f], 0:3] for f in range(n_feet)], axis=1)  # [N, nc, 3]
        foot_err_vec = foot_pos - foot_target_np
        foot_err_norm = np.linalg.norm(foot_err_vec, axis=-1)  # [N, nc]
        records = []
        for k in range(N):
            rec: dict = {
                "iter": iter_idx,
                "problem": k,
                "base_pos": base_pos[k].tolist(),
                "base_rpy_deg": base_rpy[k].tolist(),
                "base_pos_err": float(base_pos_err[k]),
                "base_rot_err_deg": float(base_rot_angle[k]),
                "foot_pos": foot_pos[k].tolist(),
                "foot_err_norm": foot_err_norm[k].tolist(),
                "foot_err_max": float(foot_err_norm[k].max()),
                "total_cost": float(total_cost_np[k]),
            }
            records.append(rec)
        return records

    # --- Iter 0: pre-IK snapshot (uses joint_q_init; solver.costs not yet populated). ---
    records: list[dict] = []
    records.extend(_snapshot(0, jq_in, include_solver_cost=False))

    # --- Step 1 iter at a time ---
    for it in range(1, args.max_iters + 1):
        solver.step(jq_in, jq_out, iterations=1)
        records.extend(_snapshot(it, jq_out, include_solver_cost=True))
        # Next iter feeds from latest result (same memory, so no-op assignment).
        jq_in = jq_out

    # --- Summary per problem ---
    print("\n--- Per-problem summary ---")
    for k in range(N):
        problem_records = [r for r in records if r["problem"] == k]
        r0, rf = problem_records[0], problem_records[-1]
        print(f"\nProblem #{k}")
        print(f"  base_target_pos  = {base_target_pos_np[k].round(3).tolist()}")
        print(f"  base_target_rpy  = {_quat_to_rpy_deg(base_target_rot_np[k : k + 1])[0].round(2).tolist()} deg")
        print("  foot_targets     :")
        for f, name in enumerate(foot_names):
            print(f"    {name:20s} -> {np.round(foot_target_np[k, f], 3).tolist()}")
        print(
            f"  iter=0   base_pos={np.round(r0['base_pos'], 3).tolist()}"
            f" rpy={np.round(r0['base_rpy_deg'], 1).tolist()} "
            f"base_pos_err={r0['base_pos_err']:.3f} base_rot_err={r0['base_rot_err_deg']:.1f}° "
            f"foot_err_max={r0['foot_err_max']:.3f}"
        )
        print(
            f"  iter={rf['iter']:<3d} base_pos={np.round(rf['base_pos'], 3).tolist()}"
            f" rpy={np.round(rf['base_rpy_deg'], 1).tolist()} "
            f"base_pos_err={rf['base_pos_err']:.3f} base_rot_err={rf['base_rot_err_deg']:.1f}° "
            f"foot_err_max={rf['foot_err_max']:.3f} cost={rf['total_cost']:.4f}"
        )
        print(f"  foot world pos (final): {[np.round(p, 3).tolist() for p in rf['foot_pos']]}")
        print(f"  foot err per-slot      : {[round(e, 3) for e in rf['foot_err_norm']]}")

    # --- Waterfall: iter-by-iter printout (compact) ---
    print("\n--- Iter waterfall (per problem) ---")
    # Print every 10th iter to keep console readable; full trace lives in JSON.
    stride = max(1, args.max_iters // 20)
    for k in range(N):
        print(f"\nProblem #{k}:")
        print(
            f"  {'iter':>4s} {'base_x':>7s} {'base_y':>7s} {'base_z':>7s} "
            f"{'roll':>6s} {'pitch':>6s} {'yaw':>6s} "
            f"{'fmax':>6s} {'cost':>9s}"
        )
        for rec in records:
            if rec["problem"] != k:
                continue
            if rec["iter"] % stride != 0 and rec["iter"] != args.max_iters:
                continue
            bp = rec["base_pos"]
            rp = rec["base_rpy_deg"]
            print(
                f"  {rec['iter']:>4d} {bp[0]:>7.3f} {bp[1]:>7.3f} {bp[2]:>7.3f} "
                f"{rp[0]:>6.1f} {rp[1]:>6.1f} {rp[2]:>6.1f} "
                f"{rec['foot_err_max']:>6.3f} {rec['total_cost']:>9.4f}"
            )

    # --- Save JSON ---
    if args.output:
        out = {
            "config": {
                "sub_terrain": args.sub_terrain,
                "terrain": args.terrain,
                "difficulty": args.difficulty,
                "robot": args.robot,
                "exclude_feet": args.exclude_feet,
                "gravity_weight": gravity_weight,
                "base_rot_weight": base_rot_weight,
                "base_pos_weight": base_pos_weight,
                "foot_names": foot_names,
                "base_targets_pos": base_target_pos_np.tolist(),
                "base_targets_rot_xyzw": base_target_rot_np.tolist(),
                "foot_targets": foot_target_np.tolist(),
            },
            "records": records,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.output} ({len(records)} records)")


if __name__ == "__main__":
    main()
