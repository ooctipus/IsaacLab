# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reject native G1 checkpoint semantic drift before a long policy evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

_FIELD_WIDTHS = {
    "joint_position": 29,
    "joint_velocity": 29,
    "projected_gravity": 3,
    "base_angular_velocity": 3,
    "last_action": 29,
    "history_actor": 372,
    "privileged_state": 463,
}
_NATIVE_TRACE = Path(__file__).parent / "fixtures" / "g1_lafan_same_step_trace_v1.npz"
_RUNTIME_LIMIT_SECONDS = 120.0
_RAW_ORDER_SENSITIVITY_MINIMUM = 1.0e-6


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one regular, non-symbolic file."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Preflight input must be a regular non-symbolic file: {path}.")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _tensor_sha256(value: Any) -> str:
    """Hash one small tensor after the preflight's final synchronization."""
    host = value.detach().cpu().contiguous()
    return hashlib.sha256(host.numpy().tobytes(order="C")).hexdigest()


def _maximum_absolute_error(first: Any, second: Any) -> float:
    """Return a finite maximum absolute tensor difference."""
    difference = (first - second).abs()
    if difference.numel() == 0 or not bool(difference.isfinite().all()):
        raise ValueError("Preflight tensor comparisons require finite nonempty tensors.")
    return float(difference.max())


def _ordered_axis(
    *,
    axis_name: str,
    physical_names: Sequence[str],
    behavior_names: Sequence[str],
    behavior_ids: Any,
) -> dict[str, object]:
    """Validate one behavior-to-physical name permutation and its resolved indices."""
    import torch

    physical = tuple(physical_names)
    behavior = tuple(behavior_names)
    if (
        not physical
        or len(physical) != len(set(physical))
        or len(behavior) != len(set(behavior))
        or set(physical) != set(behavior)
    ):
        raise ValueError(f"G1 {axis_name} behavior and physical names must be complete unique permutations.")
    expected = torch.tensor(
        [physical.index(name) for name in behavior],
        dtype=torch.int64,
        device=behavior_ids.device,
    )
    if (
        behavior_ids.dtype is not torch.int64
        or behavior_ids.shape != expected.shape
        or not torch.equal(behavior_ids, expected)
    ):
        raise ValueError(f"G1 behavior-to-physical {axis_name} map differs from its declared names.")
    indices = expected.cpu().tolist()
    return {
        "physical_names": list(physical),
        "behavior_names": list(behavior),
        "behavior_to_physical": indices,
        "is_identity": indices == list(range(len(indices))),
    }


def _behavior_axis_identity(
    *,
    physical_joint_names: Sequence[str],
    behavior_joint_names: Sequence[str],
    behavior_joint_ids: Any,
    physical_body_names: Sequence[str],
    behavior_body_names: Sequence[str],
    behavior_body_ids: Any,
) -> dict[str, object]:
    """Return the validated checkpoint-facing G1 joint and body axes."""
    return {
        "joint": _ordered_axis(
            axis_name="joint",
            physical_names=physical_joint_names,
            behavior_names=behavior_joint_names,
            behavior_ids=behavior_joint_ids,
        ),
        "body": _ordered_axis(
            axis_name="body",
            physical_names=physical_body_names,
            behavior_names=behavior_body_names,
            behavior_ids=behavior_body_ids,
        ),
    }


def _scatter_behavior_to_physical(values: Any, behavior_joint_ids: Any) -> Any:
    """Express behavior-ordered joint columns in raw physical-axis order."""
    result = values.clone()
    result.scatter_(1, behavior_joint_ids.view(1, -1).expand(values.shape[0], -1), values)
    return result


def _raw_physical_actor_observations(
    observations: Mapping[str, Any],
    behavior_joint_ids: Any,
) -> dict[str, Any]:
    """Reproduce the former error that exposed raw physical joint columns to the policy."""
    joint_position = _scatter_behavior_to_physical(observations["joint_position"], behavior_joint_ids)
    joint_velocity = _scatter_behavior_to_physical(observations["joint_velocity"], behavior_joint_ids)
    last_action = _scatter_behavior_to_physical(observations["last_action"], behavior_joint_ids)
    history = observations["history_actor"].clone()
    for field_offset in (0, 128, 244):
        for lag in range(4):
            start = field_offset + 29 * lag
            history[:, start : start + 29] = _scatter_behavior_to_physical(
                history[:, start : start + 29], behavior_joint_ids
            )
    return {
        "joint_position": joint_position,
        "joint_velocity": joint_velocity,
        "projected_gravity": observations["projected_gravity"],
        "base_angular_velocity": observations["base_angular_velocity"],
        "last_action": last_action,
        "history_actor": history,
        "privileged_state": observations["privileged_state"],
    }


def _load_native_trace_observations(path: Path, device: Any):
    """Load checkpoint inputs from the first edge of the frozen native BFM trace."""
    import numpy as np
    import torch
    from tensordict import TensorDict

    path = path.expanduser().resolve()
    digest = _sha256(path)
    with np.load(path, allow_pickle=False) as trace:
        state = torch.from_numpy(trace["current_state"][0]).to(device=device)
        joint_position, joint_velocity, projected_gravity, base_angular_velocity = torch.split(
            state, (29, 29, 3, 3), dim=-1
        )
        fields = {
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "projected_gravity": projected_gravity,
            "base_angular_velocity": base_angular_velocity,
            "last_action": torch.from_numpy(trace["current_last_action"][0]).to(device=device),
            "history_actor": torch.from_numpy(trace["current_history_actor"][0]).to(device=device),
            "privileged_state": torch.from_numpy(trace["current_privileged_state"][0]).to(device=device),
        }
    batch_size = next(iter(fields.values())).shape[0]
    for name, width in _FIELD_WIDTHS.items():
        if fields[name].shape != (batch_size, width) or fields[name].dtype is not torch.float32:
            raise ValueError(f"Native trace field {name!r} has an incompatible checkpoint contract.")
    return TensorDict(fields, batch_size=[batch_size], device=device), {
        "path": str(path),
        "sha256": digest,
        "step": 0,
        "batch_size": batch_size,
    }


def _as_tensordict(observations: object, num_envs: int):
    """Expose manager observations through the named model contract."""
    from tensordict import TensorDict, TensorDictBase

    if isinstance(observations, TensorDictBase):
        return observations
    if not isinstance(observations, Mapping):
        raise TypeError("G1 preflight observations must be a tensor mapping.")
    return TensorDict(dict(observations), batch_size=[num_envs])


def _checkpoint_semantics(model: Any, observations: Any, behavior_joint_ids: Any) -> dict[str, object]:
    """Compare named inference routes with direct networks and a raw-order counterexample."""
    import torch
    from tensordict import TensorDict

    model.observation_schema.assert_valid(observations)
    with torch.inference_mode():
        backward_input = model.get_normalized_observations(observations, "backward")
        backward_direct = model.backward_network(backward_input)
        backward_named = model.backward_map(observations)
        context_direct = model.context_project(backward_direct)
        context_named = model.context_project(backward_named)

        actor_input = model.get_normalized_observations(observations, "actor")
        actor_output = model.actor_network(actor_input, torch.cat((actor_input, context_named), dim=-1))
        model.action_distribution.update(actor_output)
        action_direct = model.action_distribution.mean.clone()
        action_named = model.action_sample(observations, context_named, deterministic=True)

        wrong_fields = _raw_physical_actor_observations(observations, behavior_joint_ids)
        wrong = TensorDict(wrong_fields, batch_size=observations.batch_size, device=observations.device)
        backward_wrong = model.backward_map(wrong)
        context_wrong = model.context_project(backward_wrong)
        action_wrong = model.action_sample(wrong, context_wrong, deterministic=True)

    errors = {
        "backward_named_vs_direct": _maximum_absolute_error(backward_named, backward_direct),
        "context_named_vs_direct": _maximum_absolute_error(context_named, context_direct),
        "action_named_vs_direct": _maximum_absolute_error(action_named, action_direct),
    }
    return {
        "route_max_abs_errors": errors,
        "route_max_abs_error": max(errors.values()),
        "raw_physical_order_counterexample": {
            "backward_max_abs_delta": _maximum_absolute_error(backward_named, backward_wrong),
            "context_max_abs_delta": _maximum_absolute_error(context_named, context_wrong),
            "action_max_abs_delta": _maximum_absolute_error(action_named, action_wrong),
        },
        "outputs": {
            "backward_sha256": _tensor_sha256(backward_named),
            "context_sha256": _tensor_sha256(context_named),
            "action_sha256": _tensor_sha256(action_named),
            "context_norm_mean": float(context_named.norm(dim=-1).mean()),
        },
        "context": context_named,
    }


def _evaluation_history_factory(replay: Mapping[str, object]) -> Callable[[Any], Any]:
    """Build direct-evaluation history from the learner's replay contract."""
    from rsl_rl.algorithms.forward_backward import ForwardBackward
    from rsl_rl.storage.forward_backward_replay import ForwardBackwardHistoryLayout

    value = replay.get("history_layout")
    if value is None:
        return lambda _observations: None
    if not isinstance(value, Mapping):
        raise TypeError("Replay history_layout must be a mapping or None.")
    options = dict(value)
    sources = tuple(ForwardBackwardHistoryLayout.Source(**dict(source)) for source in options.pop("sources"))
    layout = ForwardBackwardHistoryLayout(sources=sources, **options)
    return lambda observations: ForwardBackward.EvaluationHistory(layout, observations)


def _short_rollout(
    *,
    env: Any,
    evaluation_scope: Any,
    command: Any,
    domain_scope: Any,
    history_factory: Any,
    model: Any,
    context: Any,
    robot: Any,
    action_term: Any,
    seed: int,
    steps: int,
) -> dict[str, object]:
    """Run a few live edges and prove behavior actions reach the mapped physical targets."""
    import torch

    context = context[:1].expand(env.num_envs, -1)
    finite = torch.ones((), dtype=torch.bool, device=env.device)
    done_rows = torch.zeros((), dtype=torch.int64, device=env.device)
    final_rows = torch.zeros((), dtype=torch.int64, device=env.device)
    request_error = torch.zeros((), dtype=torch.float32, device=env.device)
    processed_error = torch.zeros_like(request_error)
    target_error = torch.zeros_like(request_error)
    action_l2 = torch.zeros((), dtype=torch.float64, device=env.device)
    with (
        torch.inference_mode(),
        evaluation_scope(
            env,
            command,
            domain_scope,
            seed,
            reset_source_name=None,
        ),
    ):
        reset = env.reset()
        observations = _as_tensordict(reset[0] if isinstance(reset, tuple) else reset, env.num_envs)
        history = history_factory(observations)
        if history is not None:
            observations = history.decorate_current(observations)
        for _ in range(steps):
            model.observation_schema.assert_valid(observations)
            actions = model.action_sample(observations, context, deterministic=True)
            returned, rewards, terminated, truncated, extras = env.step(actions)
            returned = _as_tensordict(returned, env.num_envs)
            done = terminated | truncated
            if history is not None:
                history.advance(observations, returned, done)
            observations = returned
            expected_processed = (actions * action_term.cfg.scale).clamp(*action_term.cfg.clip[".*"])
            physical_target = robot.data.joint_pos_target.torch.index_select(1, action_term.joint_ids)
            torch.maximum(request_error, (action_term.raw_actions - actions).abs().max(), out=request_error)
            torch.maximum(
                processed_error,
                (action_term.processed_actions - expected_processed).abs().max(),
                out=processed_error,
            )
            torch.maximum(
                target_error, (physical_target - action_term.joint_position_target).abs().max(), out=target_error
            )
            action_l2.add_(actions.to(dtype=torch.float64).norm(dim=-1).mean())
            finite.logical_and_(actions.isfinite().all())
            finite.logical_and_(rewards.isfinite().all())
            for _name, value in observations.items(include_nested=True, leaves_only=True):
                finite.logical_and_(value.isfinite().all())
            done_rows.add_(done.sum())
            if "final_obs" in extras:
                final_rows.add_(done.sum())
    if torch.device(env.device).type == "cuda":
        torch.cuda.synchronize(env.device)
    return {
        "steps": steps,
        "num_envs": env.num_envs,
        "finite": bool(finite),
        "done_rows": int(done_rows),
        "final_observation_rows": int(final_rows),
        "action_request_max_abs_error": float(request_error),
        "processed_action_max_abs_error": float(processed_error),
        "action_target_max_abs_error": float(target_error),
        "action_l2_mean": float(action_l2) / steps,
    }


def _require_preflight_pass(
    *,
    runtime_seconds: float,
    runtime_limit_seconds: float,
    checkpoint_strict_load: bool,
    axis_identity: bool,
    route_max_abs_error: float,
    raw_order_action_delta: float,
    raw_order_backward_delta: float,
    rollout_finite: bool,
    action_request_max_abs_error: float,
    processed_action_max_abs_error: float,
    action_target_max_abs_error: float,
    done_rows: int,
) -> dict[str, object]:
    """Require every fast semantic gate before authorizing the long evaluation."""
    if not math.isfinite(runtime_seconds) or runtime_seconds > runtime_limit_seconds:
        raise RuntimeError("G1 checkpoint semantic preflight exceeded its runtime gate.")
    if checkpoint_strict_load is not True:
        raise RuntimeError("G1 checkpoint did not pass strict checkpoint loading.")
    if axis_identity is not True:
        raise RuntimeError("G1 behavior axis identity is incomplete.")
    if route_max_abs_error != 0.0:
        raise RuntimeError("G1 named and direct model routes are not exact.")
    if min(raw_order_action_delta, raw_order_backward_delta) <= _RAW_ORDER_SENSITIVITY_MINIMUM:
        raise RuntimeError("G1 checkpoint is not demonstrably sensitive to the old raw ordering error.")
    if rollout_finite is not True:
        raise RuntimeError("G1 short checkpoint rollout is not finite.")
    if max(action_request_max_abs_error, processed_action_max_abs_error, action_target_max_abs_error) != 0.0:
        raise RuntimeError("G1 behavior action boundary or mapped physical target is not exact.")
    if done_rows != 0:
        raise RuntimeError("G1 short checkpoint rollout crossed a terminal boundary.")
    return {"status": "passed", "passed": True}


def _run(args: argparse.Namespace, process_started: float) -> dict[str, object]:
    """Construct the native environment and execute the seconds-scale semantic gate."""
    import torch
    from motion_environment_identity import motion_environment_axes, motion_runner_axes
    from rsl_rl.models.forward_backward_model import ForwardBackwardModel

    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_tasks.core.multi_task.motion.config.agents import MotionForwardBackwardRunnerCfg
    from isaaclab_tasks.core.multi_task.motion.robots.g1.articulation import (
        G1_BEHAVIOR_BODY_NAMES,
        G1_BEHAVIOR_JOINT_NAMES,
    )
    from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
    from isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_tracking import forward_backward_evaluation_scope
    from isaaclab_tasks.utils import resolve_presets

    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != args.checkpoint_sha256:
        raise ValueError("Preflight checkpoint SHA-256 differs from the accepted frozen policy.")
    trace, trace_identity = _load_native_trace_observations(args.native_trace, torch.device(args.device))
    runner_selection = motion_environment_axes("g1_lafan") | motion_runner_axes("g1_lafan")
    runner_values = resolve_presets(MotionForwardBackwardRunnerCfg(), selected=runner_selection).to_dict()
    cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes("g1_lafan"))
    cfg.sim.device = args.device
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    table_cfg = cfg.commands.motion.task_table
    table_cfg.source_artifact_root = str(args.source_artifact_root.expanduser().resolve())
    table_cfg.reference_artifact_root = str(args.reference_artifact_root.expanduser().resolve())
    table_cfg.motion_split = "train"

    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        table = env.command_manager.get_term("motion").table
        if len(table.clip_ids) != 862 or table.clip_index.total_frames != 258_600:
            raise RuntimeError("Preflight environment does not contain the frozen native BFM corpus.")
        reset = env.reset()
        observations = _as_tensordict(reset[0] if isinstance(reset, tuple) else reset, env.num_envs)
        history_factory = _evaluation_history_factory(runner_values["replay"])
        construction_history = history_factory(observations)
        if construction_history is not None:
            observations = construction_history.decorate_current(observations)
        model = ForwardBackwardModel.from_config(
            observations,
            runner_values["obs_groups"],
            env.action_manager.total_action_dim,
            runner_values["model"],
        ).to(env.device)
        saved = torch.load(checkpoint, map_location=env.device, weights_only=True)
        if set(saved) != {"model_state_dict"}:
            raise ValueError("Preflight checkpoint must contain exactly model_state_dict.")
        loaded = model.load_state_dict(saved["model_state_dict"], strict=True, assign=True)
        strict_load = not loaded.missing_keys and not loaded.unexpected_keys
        model.eval()

        command = env.command_manager.get_term("motion")
        robot = command.payload.robot
        action_term = env.action_manager.get_term("joint_position")
        physical_joint_names = tuple(robot.joint_names)
        if table.joint_names != physical_joint_names:
            raise ValueError("The G1 motion table must retain the physical articulation joint axis.")
        body_ids, observed_body_names = robot.find_bodies(list(G1_BEHAVIOR_BODY_NAMES), preserve_order=True)
        axis_identity = _behavior_axis_identity(
            physical_joint_names=physical_joint_names,
            behavior_joint_names=action_term.joint_names,
            behavior_joint_ids=action_term.joint_ids,
            physical_body_names=robot.body_names,
            behavior_body_names=observed_body_names,
            behavior_body_ids=torch.tensor(body_ids, dtype=torch.int64, device=env.device),
        )
        if tuple(action_term.joint_names) != tuple(G1_BEHAVIOR_JOINT_NAMES):
            raise ValueError("The G1 action term does not expose the released behavior joint names.")
        if tuple(observed_body_names) != tuple(G1_BEHAVIOR_BODY_NAMES):
            raise ValueError("The G1 privileged route does not expose the released behavior body names.")

        semantics = _checkpoint_semantics(model, trace, action_term.joint_ids)
        rollout = _short_rollout(
            env=env,
            evaluation_scope=forward_backward_evaluation_scope,
            command=command,
            domain_scope=command.payload.sampler.reset_sampling_scope,
            history_factory=history_factory,
            model=model,
            context=semantics.pop("context"),
            robot=robot,
            action_term=action_term,
            seed=args.seed,
            steps=args.steps,
        )
        runtime_seconds = time.perf_counter() - process_started
        counterexample = semantics["raw_physical_order_counterexample"]
        decision = _require_preflight_pass(
            runtime_seconds=runtime_seconds,
            runtime_limit_seconds=_RUNTIME_LIMIT_SECONDS,
            checkpoint_strict_load=strict_load,
            axis_identity=True,
            route_max_abs_error=semantics["route_max_abs_error"],
            raw_order_action_delta=counterexample["action_max_abs_delta"],
            raw_order_backward_delta=counterexample["backward_max_abs_delta"],
            rollout_finite=rollout["finite"],
            action_request_max_abs_error=rollout["action_request_max_abs_error"],
            processed_action_max_abs_error=rollout["processed_action_max_abs_error"],
            action_target_max_abs_error=rollout["action_target_max_abs_error"],
            done_rows=rollout["done_rows"],
        )
        return {
            "schema": "forward_backward_phase3_g1_checkpoint_semantic_preflight_v1",
            "decision": decision,
            "runtime": {
                "seconds": runtime_seconds,
                "limit_seconds": _RUNTIME_LIMIT_SECONDS,
                "num_envs": args.num_envs,
                "rollout_steps": args.steps,
            },
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_sha256,
                "strict_load": strict_load,
                "missing_keys": list(loaded.missing_keys),
                "unexpected_keys": list(loaded.unexpected_keys),
            },
            "native_trace": trace_identity,
            "model": {
                "observation_schema_sha256": model.observation_schema.schema_hash,
                "field_widths": dict(model.observation_schema.field_widths),
                "routes": {name: list(fields) for name, fields in model.observation_schema.routes},
            },
            "axes": axis_identity,
            "frozen_native_batch": semantics,
            "short_live_rollout": rollout,
            "source": {
                "preflight_sha256": _sha256(Path(__file__).resolve()),
                "source_artifact_root": table_cfg.source_artifact_root,
                "reference_artifact_root": table_cfg.reference_artifact_root,
            },
        }
    finally:
        env.close()


def main() -> None:
    """Parse one native G1 preflight request and atomically publish its evidence."""
    process_started = time.perf_counter()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_artifact_root", type=Path, required=True)
    parser.add_argument("--reference_artifact_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint_sha256", required=True)
    parser.add_argument("--native_trace", type=Path, default=_NATIVE_TRACE)
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 16 or not 1 <= args.steps <= 8:
        raise ValueError("G1 preflight requires 1-16 environments and 1-8 rollout steps.")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"G1 preflight output already exists: {output}.")

    launcher = AppLauncher(args)
    simulation_app = launcher.app
    try:
        report = _run(args, process_started)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)
        print(json.dumps(report, indent=2, sort_keys=True))
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
