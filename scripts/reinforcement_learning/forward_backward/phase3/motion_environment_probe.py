# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the shared motion environment at a fixed resolved-reproduction boundary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--preset", choices=("smpl_cmu", "g1_lafan", "g1_cmu"), required=True)
parser.add_argument("--source_artifact_root", type=Path, required=True)
parser.add_argument("--reference_artifact_root", type=Path)
parser.add_argument("--motion_split", choices=("train", "evaluation"), default="evaluation")
parser.add_argument("--num_envs", type=int, required=True)
parser.add_argument("--num_steps", type=int, default=None)
parser.add_argument("--warmup_steps", type=int, default=8)
parser.add_argument("--benchmark_steps", type=int, default=256)
parser.add_argument("--benchmark_without_final_obs", action="store_true")
parser.add_argument("--action_bound", type=float, default=0.0)
parser.add_argument("--action_seed", type=int, default=0)
parser.add_argument("--action_period_steps", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--evidence_role",
    choices=("standalone", "repeatability", "capture_cost"),
    default="standalone",
)
parser.add_argument("--replicate", type=int, default=0)
parser.add_argument("--pair_index", type=int)
parser.add_argument("--pair_position", type=int, choices=(0, 1))
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from gpu_ownership import exclusive_physical_gpu_snapshot, validate_same_exclusive_gpu
from motion_environment_identity import (
    motion_action_term_cfg,
    motion_environment_axes,
    motion_environment_dependency_identity,
    motion_runner_axes,
)

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.core.multi_task.motion.config.agents import MotionForwardBackwardRunnerCfg
from isaaclab_tasks.core.multi_task.motion.data.sources import CmuHumEnvSmplClips, LafanG1JoblibClips
from isaaclab_tasks.core.multi_task.motion.mdp.commands import MotionTaskTable
from isaaclab_tasks.core.multi_task.motion.robots.g1.reference import G1LocalBodyPoseFrameBuilder, G1PoseFrameBuilder
from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import SmplGeneralizedCoordinateFrameBuilder
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets


def _tensor_shapes(values) -> dict[str, list[int]]:
    """Return the flat observation-group shapes."""
    return {str(name): list(value.shape) for name, value in values.items()}


def _synchronize(device: str) -> None:
    """Synchronize CUDA measurements when the selected simulator uses CUDA."""
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def _signature_buffers(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Allocate two float64 device scalars per named trajectory field."""
    return {name: torch.zeros(2, dtype=torch.float64, device=value.device) for name, value in values.items()}


def _signature_add_(
    buffers: dict[str, torch.Tensor],
    values: dict[str, torch.Tensor],
    rows: torch.Tensor | None = None,
) -> None:
    """Accumulate sums and squared sums without materializing host values."""
    if buffers.keys() != values.keys():
        raise RuntimeError("Trajectory-signature fields changed during one environment run.")
    for name, value in values.items():
        selected = value if rows is None else value[rows]
        selected_float64 = selected.to(dtype=torch.float64)
        buffers[name][0].add_(torch.sum(selected_float64))
        buffers[name][1].add_(torch.sum(torch.square(selected_float64)))


def _signature_result(buffers: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    """Materialize synchronized trajectory signatures once after the rollout."""
    return {name: {"sum": float(value[0]), "squared_sum": float(value[1])} for name, value in buffers.items()}


def _motion_state_tensors(env, observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return the observable, physical, and command state at one edge boundary."""
    payload = env.command_manager.get_term("motion").payload
    robot = payload.robot
    values = {f"observations.{name}": value for name, value in observations.items()}
    values.update(
        {
            "robot.root_link_pose_w": robot.data.root_link_pose_w.torch,
            "robot.root_link_velocity_w": robot.data.root_link_vel_w.torch,
            "robot.body_link_pose_w": robot.data.body_link_pose_w.torch,
            "robot.body_link_velocity_w": robot.data.body_link_vel_w.torch,
            "robot.joint_position": robot.data.joint_pos.torch,
            "robot.joint_velocity": robot.data.joint_vel.torch,
        }
    )
    values.update({f"motion_facts.{name}": value for name, value in payload.motion_facts.items()})
    values.update({f"reference.{name}": value for name, value in payload.reference.items()})
    return values


def _tensor_identity(values: dict[str, torch.Tensor]) -> dict[str, dict[str, object]]:
    """Hash exact tensor bytes after one untimed device-to-host materialization."""
    identity = {}
    for name, value in values.items():
        host = value.detach().cpu()
        identity[name] = {
            "shape": list(host.shape),
            "dtype": str(host.dtype),
            "sha256": hashlib.sha256(host.numpy().tobytes(order="C")).hexdigest(),
        }
    return identity


def _numerical_signature(values: dict[str, torch.Tensor]) -> dict[str, dict[str, int | float]]:
    """Reduce one edge to scale-independent numerical moments on the device."""
    result = {}
    for name, value in values.items():
        flattened = value.to(dtype=torch.float64).reshape(-1)
        result[name] = {
            "count": flattened.numel(),
            "mean": float(flattened.mean()),
            "mean_square": float(torch.mean(torch.square(flattened))),
            "maximum_absolute": float(torch.max(torch.abs(flattened))),
        }
    return result


def _source_sha256(value: object) -> str:
    """Hash the source file defining one runtime class."""
    path = inspect.getsourcefile(value)
    if path is None:
        raise RuntimeError(f"Cannot locate source for {value!r}.")
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_observation_routes(observations, routes: dict[str, list[str]]) -> None:
    """Require every independently configured learner route from the live observations."""
    for route_name, fields in routes.items():
        missing = tuple(name for name in fields if name not in observations)
        if missing:
            raise RuntimeError(f"{route_name} route is missing observation fields {missing}.")


def _semantic_pass(
    env,
    action_program: torch.Tensor,
    num_steps: int,
    expected_terminal_step: int,
    reset_source_names: tuple[str, ...],
) -> dict:
    """Run an untimed horizon pass and aggregate correctness entirely on device."""
    observations, _ = env.reset()
    reset_state_identity = _tensor_identity(_motion_state_tensors(env, observations))
    observation_signature = _signature_buffers(observations)
    final_observation_signature = _signature_buffers(observations)
    reward_signature = torch.zeros(2, dtype=torch.float64, device=env.device)
    _signature_add_(observation_signature, observations)
    reset_sources = env.command_manager.get_term("motion").payload.reset_source_indices.clone()
    row_finite = torch.stack(
        tuple(torch.isfinite(value).reshape(env.num_envs, -1).all(dim=-1) for value in observations.values())
    ).all(dim=0)
    finite_observations = row_finite.all()
    first_nonfinite_step = torch.full((env.num_envs,), -1, dtype=torch.int64, device=env.device)
    first_nonfinite_step.masked_fill_(~row_finite, 0)
    finite_rewards = torch.ones((), dtype=torch.bool, device=env.device)
    finite_final_observations = torch.ones((), dtype=torch.bool, device=env.device)
    counts = torch.zeros(5, dtype=torch.int64, device=env.device)
    episode_steps = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
    action_applied = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    no_final_rows = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    terminal_steps = []
    action_period = action_program.shape[0]
    one_edge_signature = None

    for step in range(num_steps):
        actions = action_program[step % action_period]
        episode_steps.add_(1)
        observations, rewards, terminated, truncated, extras = env.step(actions)
        done = terminated | truncated
        final = extras.get("final_obs")
        final_valid = done if final is not None else no_final_rows
        if step == 0:
            edge_values = _motion_state_tensors(env, observations)
            edge_values.update(
                {
                    "transition.reward": rewards,
                    "transition.terminated": terminated,
                    "transition.truncated": truncated,
                    "transition.action_applied": action_applied,
                    "transition.episode_steps": episode_steps,
                    "transition.final_observation_valid": final_valid,
                }
            )
            one_edge_signature = _numerical_signature(edge_values)
        row_finite = torch.stack(
            tuple(torch.isfinite(value).reshape(env.num_envs, -1).all(dim=-1) for value in observations.values())
        ).all(dim=0)
        finite_observations.logical_and_(row_finite.all())
        first_nonfinite_step.masked_fill_((first_nonfinite_step < 0) & ~row_finite, step + 1)
        finite_rewards.logical_and_(torch.isfinite(rewards).all())
        _signature_add_(observation_signature, observations)
        reward_float64 = rewards.to(dtype=torch.float64)
        reward_signature[0].add_(torch.sum(reward_float64))
        reward_signature[1].add_(torch.sum(torch.square(reward_float64)))

        captured = final_valid
        if final is not None:
            _signature_add_(final_observation_signature, final, captured)
            for value in final.values():
                row_final_finite = torch.isfinite(value).reshape(env.num_envs, -1).all(dim=-1)
                finite_final_observations.logical_and_((row_final_finite | ~captured).all())

        counts[0].add_(terminated.sum())
        counts[1].add_(truncated.sum())
        counts[2].add_(done.sum())
        counts[3].add_(captured.sum())
        counts[4].add_(env.num_envs)
        terminal_steps.append(episode_steps[done])
        episode_steps.masked_fill_(done, 0)

    _synchronize(str(env.device))
    terminated_rows, truncated_rows, final_rows, captured_final_rows, applied_rows = (
        int(value) for value in counts.tolist()
    )
    nonfinite_mask = first_nonfinite_step >= 0
    reset_source_counts = torch.bincount(reset_sources, minlength=len(reset_source_names))
    if reset_source_counts.numel() != len(reset_source_names):
        raise RuntimeError("A reset-source index lies outside the task table's declared names.")
    nonfinite_env_ids = torch.nonzero(first_nonfinite_step >= 0).flatten().cpu().tolist()
    nonfinite_steps = first_nonfinite_step[first_nonfinite_step >= 0]
    terminal_episode_steps = sorted(torch.unique(torch.cat(terminal_steps)).cpu().tolist()) if final_rows else []
    if one_edge_signature is None:
        raise RuntimeError("Semantic repeatability evidence requires at least one applied edge.")
    result = {
        "terminated_rows": terminated_rows,
        "truncated_rows": truncated_rows,
        "final_rows": final_rows,
        "captured_final_rows": captured_final_rows,
        "missing_final_rows": final_rows - captured_final_rows,
        "terminal_episode_steps": terminal_episode_steps,
        "applied_rows": applied_rows,
        "finite_observations": bool(finite_observations),
        "finite_rewards": bool(finite_rewards),
        "finite_final_observations": bool(finite_final_observations),
        "nonfinite_observation_rows": len(nonfinite_env_ids),
        "first_nonfinite_observation_step": int(nonfinite_steps.min()) if nonfinite_env_ids else None,
        "nonfinite_env_ids": nonfinite_env_ids,
        "reset_source_rows": dict(
            zip(reset_source_names, (int(value) for value in reset_source_counts.tolist()), strict=True)
        ),
        "nonfinite_reset_source_indices": reset_sources[nonfinite_mask].cpu().tolist(),
        "reset_state_identity": reset_state_identity,
        "one_edge_signature": one_edge_signature,
        "trajectory_signature": {
            "observations": _signature_result(observation_signature),
            "reward": {"sum": float(reward_signature[0]), "squared_sum": float(reward_signature[1])},
            "final_observations": _signature_result(final_observation_signature),
        },
    }
    if terminal_episode_steps and terminal_episode_steps != [expected_terminal_step]:
        raise RuntimeError(f"Shared motion timeout differs from the resolved environment contract: {result}")
    return result


def _benchmark_steps(
    env,
    action_program: torch.Tensor,
    warmup_steps: int,
    benchmark_steps: int,
    *,
    without_final_obs: bool,
) -> dict[str, int | float | bool]:
    """Measure a matched runtime window and count terminal captures on device."""
    if not env.cfg.compute_final_obs:
        raise RuntimeError("Production motion environments must construct with final capture enabled.")
    capture_final_obs = not without_final_obs
    action_period = action_program.shape[0]
    terminal_counts = torch.zeros(2, dtype=torch.int64, device=env.device)
    env.cfg.compute_final_obs = capture_final_obs
    try:
        env.reset()
        for step in range(warmup_steps):
            env.step(action_program[step % action_period])
        _synchronize(str(env.device))

        if str(env.device).startswith("cuda"):
            device = torch.device(env.device)
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline_allocated = 0
            baseline_reserved = 0

        start = time.perf_counter()
        for step in range(warmup_steps, warmup_steps + benchmark_steps):
            actions = action_program[step % action_period]
            _, _, terminated, truncated, extras = env.step(actions)
            done = terminated | truncated
            terminal_counts[0].add_(done.sum())
            if capture_final_obs and "final_obs" in extras:
                terminal_counts[1].add_(done.sum())
        _synchronize(str(env.device))
        measured_seconds = time.perf_counter() - start

        if str(env.device).startswith("cuda"):
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
        else:
            peak_allocated = 0
            peak_reserved = 0
    finally:
        env.cfg.compute_final_obs = True

    terminal_rows, captured_final_rows = (int(value) for value in terminal_counts.tolist())
    return {
        "capture_final_obs": capture_final_obs,
        "terminal_rows": terminal_rows,
        "captured_final_rows": captured_final_rows,
        "missing_final_rows": terminal_rows - captured_final_rows,
        "measured_seconds": measured_seconds,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_increment_bytes": max(peak_allocated - baseline_allocated, 0),
        "peak_reserved_increment_bytes": max(peak_reserved - baseline_reserved, 0),
    }


def _resolved_applied_action_horizon(env: ManagerBasedRLEnv) -> int:
    """Return the environment-owned applied-action horizon."""
    expected = math.ceil(env.cfg.episode_length_s / (env.cfg.sim.dt * env.cfg.decimation))
    if expected < 1 or env.max_episode_length != expected:
        raise RuntimeError("Resolved environment episode clock differs from its applied-action horizon.")
    return expected


def _validate_args(values: argparse.Namespace) -> None:
    """Validate the probe execution contract before constructing simulation."""
    if values.num_envs < 1:
        raise ValueError("num_envs must be positive.")
    if values.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if values.benchmark_steps < 1:
        raise ValueError("benchmark_steps must be positive.")
    if not 0.0 <= values.action_bound <= 1.0:
        raise ValueError("action_bound must lie in [0, 1].")
    if values.action_period_steps < 1:
        raise ValueError("action_period_steps must be positive.")
    if values.replicate < 0:
        raise ValueError("replicate must be non-negative.")
    if values.evidence_role == "capture_cost" and values.pair_index is None:
        raise ValueError("Capture-cost evidence requires pair_index.")
    if values.evidence_role == "capture_cost" and values.pair_position is None:
        raise ValueError("Capture-cost evidence requires pair_position.")
    if values.evidence_role != "capture_cost" and (values.pair_index is not None or values.pair_position is not None):
        raise ValueError("pair_index and pair_position are only valid for capture-cost evidence.")
    if values.pair_index is not None and values.pair_index < 0:
        raise ValueError("pair_index must be non-negative.")
    if values.preset != "smpl_cmu" and values.reference_artifact_root is None:
        raise ValueError(f"Preset {values.preset!r} requires --reference_artifact_root.")


def main() -> None:
    """Construct one resolved composition, run through a timeout, and persist exact evidence."""
    _validate_args(args)
    execution_started_unix_ns = time.time_ns()
    cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes(args.preset))
    runner_selection = motion_environment_axes(args.preset) | motion_runner_axes(args.preset)
    runner_cfg = resolve_presets(MotionForwardBackwardRunnerCfg(), selected=runner_selection)
    runner_values = runner_cfg.to_dict()
    table_cfg = cfg.commands.motion.task_table
    table_cfg.source_artifact_root = str(args.source_artifact_root.expanduser().resolve())
    table_cfg.reference_artifact_root = (
        "" if args.reference_artifact_root is None else str(args.reference_artifact_root.expanduser().resolve())
    )
    table_cfg.motion_split = args.motion_split
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    configured_cfg = copy.deepcopy(cfg)
    action_name, _action_cfg = motion_action_term_cfg(configured_cfg)

    construct_start = time.perf_counter()
    env = ManagerBasedRLEnv(cfg=cfg)
    _synchronize(args.device)
    if args.device.startswith("cuda"):
        device = torch.device(args.device)
        post_construction_allocated = int(torch.cuda.memory_allocated(device))
        post_construction_reserved = int(torch.cuda.memory_reserved(device))
    else:
        post_construction_allocated = 0
        post_construction_reserved = 0

    construction_seconds = time.perf_counter() - construct_start
    backend_details: dict[str, object] = {
        "manager": env.sim.physics_manager.__name__,
    }
    if args.preset == "smpl_cmu":
        model = env.sim.physics_manager.get_model()
        action = env.action_manager.get_term(action_name)
        backend_details.update(
            {
                "native_actuator_rows": int(model.custom_frequency_counts.get("mujoco:actuator", 0)),
                "native_action_writer": action.__class__.__name__,
            }
        )

    try:
        observations, _ = env.reset()
        semantic_steps = args.num_steps
        default_horizon = semantic_steps is None
        table_cfg = env.cfg.commands.motion.task_table
        payload = env.command_manager.get_term("motion").payload
        table = payload.table
        if not isinstance(table, MotionTaskTable):
            raise TypeError("The motion command did not construct a MotionTaskTable.")
        if table_cfg.source.identifier == "cmu_humenv_smpl":
            importer_type = CmuHumEnvSmplClips
        elif table_cfg.source.identifier == "lafan_g1_29dof":
            importer_type = LafanG1JoblibClips
        else:
            raise ValueError(f"Unsupported motion source identifier: {table_cfg.source.identifier!r}.")
        applied_actions_before_timeout = _resolved_applied_action_horizon(env)
        if semantic_steps is None:
            semantic_steps = applied_actions_before_timeout
        if semantic_steps < 1:
            raise ValueError("num_steps must be positive.")

        action_program = torch.empty(
            (args.action_period_steps, env.num_envs, env.action_manager.total_action_dim),
            dtype=torch.float32,
            device=env.device,
        )
        if args.action_bound == 0.0:
            action_program.zero_()
        else:
            generator = torch.Generator(device=env.device)
            generator.manual_seed(args.action_seed)
            action_program.uniform_(-args.action_bound, args.action_bound, generator=generator)
        action_bytes = action_program.detach().cpu().contiguous().numpy().tobytes()
        action_identity = {
            "seed": args.action_seed,
            "bound": args.action_bound,
            "period_steps": args.action_period_steps,
            "minimum": float(action_program.min()),
            "maximum": float(action_program.max()),
            "sha256": hashlib.sha256(action_bytes).hexdigest(),
        }
        observation_shapes = _tensor_shapes(observations)
        _validate_observation_routes(observations, runner_values["obs_groups"])
        semantic = _semantic_pass(
            env,
            action_program,
            semantic_steps,
            applied_actions_before_timeout,
            payload.sampler.reset_source_names,
        )
        ownership_before = (
            exclusive_physical_gpu_snapshot(args.device) if args.evidence_role == "capture_cost" else None
        )
        benchmark = _benchmark_steps(
            env,
            action_program,
            args.warmup_steps,
            args.benchmark_steps,
            without_final_obs=args.benchmark_without_final_obs,
        )
        ownership_after = exclusive_physical_gpu_snapshot(args.device) if args.evidence_role == "capture_cost" else None
        if ownership_before is not None and ownership_after is not None:
            validate_same_exclusive_gpu(ownership_before, ownership_after)
        measured_seconds = float(benchmark["measured_seconds"])
        measured_transitions = args.benchmark_steps * env.num_envs
        horizon_steps = applied_actions_before_timeout
        complete_timeout_vectors = (args.warmup_steps + args.benchmark_steps) // horizon_steps - (
            args.warmup_steps // horizon_steps
        )
        expected_terminal_rows = complete_timeout_vectors * env.num_envs
        report = {
            "schema": "forward_backward_phase3e_motion_environment_probe_v9",
            "code_identity": {
                "probe_sha256": _source_sha256(main),
                "gpu_ownership_sha256": _source_sha256(exclusive_physical_gpu_snapshot),
                "dependency_identity": motion_environment_dependency_identity(
                    preset=args.preset,
                    cfg=configured_cfg,
                    importer_type=importer_type,
                    frame_builder_type={
                        "smpl_cmu": SmplGeneralizedCoordinateFrameBuilder,
                        "g1_lafan": G1PoseFrameBuilder,
                        "g1_cmu": G1LocalBodyPoseFrameBuilder,
                    }[args.preset],
                    reference_artifact_root=configured_cfg.commands.motion.task_table.reference_artifact_root,
                ),
            },
            "evidence": {
                "role": args.evidence_role,
                "replicate": args.replicate,
                "pair_index": args.pair_index,
                "pair_position": args.pair_position,
            },
            "benchmark_gpu_ownership": {
                "required_scope": "physical_gpu" if args.evidence_role == "capture_cost" else None,
                "before_benchmark": ownership_before,
                "after_benchmark": ownership_after,
            },
            "execution_started_unix_ns": execution_started_unix_ns,
            "preset": args.preset,
            "physics_manager": env.sim.physics_manager.__name__,
            "physics_details": backend_details,
            "motion_split": args.motion_split,
            "source_artifact_root": table_cfg.source_artifact_root,
            "reference_artifact_root": table_cfg.reference_artifact_root,
            "seed": args.seed,
            "device": args.device,
            "num_envs": env.num_envs,
            "action_width": env.action_manager.total_action_dim,
            "observation_shapes": observation_shapes,
            "physics_dt": float(env.cfg.sim.dt),
            "control_decimation": env.cfg.decimation,
            "control_dt": float(env.step_dt),
            "configured_horizon_steps": env.max_episode_length,
            "applied_actions_before_timeout": applied_actions_before_timeout,
            "production_capture_invariant": env.cfg.compute_final_obs,
            "construction_seconds": construction_seconds,
            "post_construction_memory": {
                "allocated_bytes": post_construction_allocated,
                "reserved_bytes": post_construction_reserved,
            },
            "action": action_identity,
            "semantic_steps": semantic_steps,
            "benchmark": {
                "warmup_steps": args.warmup_steps,
                "measured_steps": args.benchmark_steps,
                "measured_transitions": measured_transitions,
                "complete_timeout_vectors": complete_timeout_vectors,
                "expected_terminal_rows": expected_terminal_rows,
                "transitions_per_second": measured_transitions / measured_seconds,
                **benchmark,
            },
            **semantic,
            "task_table_identity": table.cache_identity,
            "task_table_builder_identity": table.frame_builder_identity_sha256,
            "task_table_builder_version": table.frame_builder_version,
            "task_table_clip_count": len(table.clip_index.clips),
            "task_table_frame_count": table.clip_index.total_frames,
        }
        finite = (
            semantic["finite_observations"],
            semantic["finite_rewards"],
            semantic["finite_final_observations"],
        )
        if not all(finite):
            raise RuntimeError(f"Non-finite shared motion environment output: {report}")
        if semantic["applied_rows"] != semantic_steps * env.num_envs:
            raise RuntimeError(f"Shared motion environment dropped applied actions: {report}")
        if semantic["missing_final_rows"]:
            raise RuntimeError(f"Shared motion environment lost terminal observations: {report}")
        if default_horizon and (
            semantic["terminated_rows"] != 0
            or semantic["truncated_rows"] != env.num_envs
            or semantic["final_rows"] != env.num_envs
        ):
            raise RuntimeError(f"Default semantic pass did not produce one full-vector timeout: {report}")
        if not env.cfg.compute_final_obs:
            raise RuntimeError("The benchmark did not restore the production final-observation invariant.")
        if benchmark["terminal_rows"] != expected_terminal_rows:
            raise RuntimeError(f"Benchmark timeout count differs from its resolved timing contract: {report}")
        if benchmark["capture_final_obs"]:
            if benchmark["missing_final_rows"]:
                raise RuntimeError(f"Capture benchmark lost terminal observations: {report}")
        elif benchmark["captured_final_rows"]:
            raise RuntimeError(f"No-capture benchmark materialized terminal observations: {report}")
        if args.benchmark_without_final_obs and expected_terminal_rows == 0:
            raise RuntimeError("No-capture benchmarks must cross at least one natural timeout.")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output)
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    else:
        simulation_app.close()
