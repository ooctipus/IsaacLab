# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure native expert attachment and one deterministic Phase 3 learner update."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import inspect
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from motion_environment_identity import motion_action_term_cfg, motion_environment_axes, motion_runner_axes
from rsl_rl.algorithms.forward_backward import ForwardBackward
from rsl_rl.models.forward_backward_model import ForwardBackwardObservationSchema
from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertBuffer
from rsl_rl.storage.forward_backward_replay import ForwardBackwardTransitionBatch
from tensordict import TensorDict

from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematics, NewtonKinematicsCfg
from isaaclab_tasks.core.multi_task.motion.config.agents import MotionForwardBackwardRunnerCfg
from isaaclab_tasks.core.multi_task.motion.identity import canonical_sha256
from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import build_motion_task_table
from isaaclab_tasks.core.multi_task.motion.robots.g1.articulation import G1_BEHAVIOR_JOINT_NAMES
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_expert import forward_backward_expert_buffer
from isaaclab_tasks.utils.hydra import resolve_presets

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"
TRACE_PATHS = {
    "smpl_cmu": FIXTURES / "meta_humenv_next_step_trace_v2.npz",
    "g1_lafan": FIXTURES / "g1_lafan_same_step_trace_v1.npz",
}
FIELD_WIDTHS = {
    "smpl_cmu": {"policy": 358},
    "g1_lafan": {
        "joint_position": 29,
        "joint_velocity": 29,
        "projected_gravity": 3,
        "base_angular_velocity": 3,
        "last_action": 29,
        "history_actor": 372,
        "privileged_state": 463,
    },
}
ACTION_WIDTHS = {"smpl_cmu": 69, "g1_lafan": 29}
EXPECTED_TRAIN_COUNTS = {
    "smpl_cmu": {"clips": 1_638, "source_frames": 730_307, "expert_frames": 730_307},
    "g1_lafan": {"clips": 862, "source_frames": 258_600, "expert_frames": 430_138},
}
CANARY_BATCH_SIZE = 8
CANARY_REPLAY_TRANSITIONS = 32


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_sha256(value: object) -> str:
    """Hash the exact Python source file that defines ``value``."""
    path = inspect.getsourcefile(value)
    if path is None:
        raise ValueError(f"Cannot locate source for {value!r}.")
    return _sha256(Path(path).resolve())


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor metadata and contiguous CPU bytes without a Python-byte copy."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(tuple(value.shape), separators=(",", ":")).encode())
    digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _module_sha256(module: torch.nn.Module) -> str:
    """Hash one module state in stable key order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(_tensor_sha256(tensor).encode())
    return digest.hexdigest()


def _model_component_hashes(learner: ForwardBackward) -> dict[str, str]:
    """Hash every independently mutated model owner."""
    model = learner.model
    modules: dict[str, torch.nn.Module] = {
        "actor": model.actor_network,
        "forward": model.forward_network,
        "backward": model.backward_network,
        "target_forward": model.forward_target_network,
        "target_backward": model.backward_target_network,
        **{f"value/{name}": module for name, module in model.value_networks.items()},
        **{f"target_value/{name}": module for name, module in model.value_target_networks.items()},
    }
    if model.discriminator_network is not None:
        modules["discriminator"] = model.discriminator_network
    return {name: _module_sha256(module) for name, module in sorted(modules.items())}


def _live_axes(
    preset: str, cfg: MotionImitationEnvCfg, device: torch.device
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the exact axes resolved by each native simulator articulation."""
    if preset == "smpl_cmu":
        asset = NewtonKinematics(
            NewtonKinematicsCfg(
                mjcf_path=cfg.scene.robot.spawn.asset_path,
                device=str(device),
                collapse_fixed_joints=False,
            )
        )
        body_names = tuple(asset.body_names)
        joint_names = tuple(f"{joint}:{component}" for joint in asset.joint_names[1:] for component in range(3))
        if len(body_names) != 24 or len(joint_names) != 69:
            raise ValueError("Packaged SMPL asset does not expose the expected ball-joint coordinate count.")
        return joint_names, body_names
    if preset == "g1_lafan":
        actuator = next(iter(cfg.scene.robot.actuators.values()))
        joint_names = tuple(actuator.joint_names_expr)
        skeleton = cfg.commands.motion.task_table.source.build_skeleton()
        roots = tuple(index for index, parent in enumerate(skeleton.parent_indices) if parent == -1)
        if len(roots) != 1 or len(joint_names) != skeleton.num_joints:
            raise ValueError("Resolved G1 preset does not define one complete articulated tree.")
        joint_by_name = {name: index for index, name in enumerate(skeleton.joint_names)}
        child_body_names = tuple(
            skeleton.body_names[skeleton.joint_child_body_indices[joint_by_name[name]]] for name in joint_names
        )
        body_names = (skeleton.body_names[roots[0]], *child_body_names)
        if len(set(body_names)) != skeleton.num_bodies:
            raise ValueError("Resolved G1 action axis does not map one-to-one onto physical bodies.")
        return joint_names, body_names
    raise ValueError(f"Unsupported learning-evidence preset: {preset!r}.")


def _construction_env(
    preset: str,
    source_artifact_root: Path,
    reference_artifact_root: Path,
    device: torch.device,
) -> SimpleNamespace:
    """Resolve one direct preset into the minimum table-construction environment."""
    cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes(preset))
    cfg.seed = 0
    cfg.sim.device = str(device)
    table_cfg = cfg.commands.motion.task_table
    table_cfg.source_artifact_root = str(source_artifact_root.expanduser().resolve())
    table_cfg.reference_artifact_root = str(reference_artifact_root.expanduser().resolve())
    table_cfg.motion_split = "train"
    joint_names, body_names = _live_axes(preset, cfg, device)
    return SimpleNamespace(
        cfg=cfg,
        device=device,
        scene={"robot": SimpleNamespace(joint_names=joint_names, body_names=body_names)},
    )


def _canonical_g1_defaults(cfg: MotionImitationEnvCfg, joint_names: tuple[str, ...]) -> torch.Tensor:
    """Read immutable asset defaults in the declared behavior axis."""
    values = cfg.scene.robot.init_state.joint_pos
    if set(values) != set(joint_names):
        raise ValueError("G1 articulation defaults do not exactly cover the behavior action axis.")
    return torch.tensor([values[name] for name in joint_names], dtype=torch.float32, device=cfg.sim.device)


def _learner_env(table, cfg: MotionImitationEnvCfg, preset: str) -> SimpleNamespace:
    """Expose the completed command table and canonical action facts to RSL-RL."""
    payload = SimpleNamespace(table=table)
    command = SimpleNamespace(payload=payload, table=table)
    command_manager = SimpleNamespace(get_term=lambda name: command if name == "motion" else None)
    action_joint_names = G1_BEHAVIOR_JOINT_NAMES if preset == "g1_lafan" else table.joint_names
    action = SimpleNamespace(joint_names=action_joint_names)
    if preset == "g1_lafan":
        action.joint_default_position = _canonical_g1_defaults(cfg, action_joint_names)
        action.default_joint_offset = torch.zeros(2, 29, device=table.device)
    action_name, _action_cfg = motion_action_term_cfg(cfg)
    action_manager = SimpleNamespace(get_term=lambda name: action if name == action_name else None)
    body_names = table.reference_frame_names[:-1] if preset == "g1_lafan" else table.reference_frame_names
    robot = SimpleNamespace(joint_names=table.joint_names, body_names=body_names)
    env = SimpleNamespace(
        num_envs=2,
        num_actions=ACTION_WIDTHS[preset],
        command_manager=command_manager,
        action_manager=action_manager,
        scene={"robot": robot},
    )
    env.unwrapped = env
    return env


def _expert_schema(preset: str, runner: dict[str, object]) -> ForwardBackwardObservationSchema:
    """Build the exact named observation routes selected for one native learner."""
    return ForwardBackwardObservationSchema.from_config(FIELD_WIDTHS[preset], runner["obs_groups"])


def _expert_buffer(preset: str, env: SimpleNamespace, runner: dict[str, object]) -> ForwardBackwardExpertBuffer:
    """Attach the command-owned table through the production expert provider."""
    schema = _expert_schema(preset, runner)
    expert_cfg = runner["expert"]
    return forward_backward_expert_buffer(
        env,
        schema,
        str(env.unwrapped.command_manager.get_term("motion").payload.table.device),
        source_bind=expert_cfg["source_bind"],
        priorities_bind=expert_cfg["priorities_bind"],
        sampling_mode=expert_cfg["sampling_mode"],
        sampling_step_seconds=expert_cfg["sampling_step_seconds"],
        target_projection=expert_cfg["target_projection"],
        target_projection_binds=tuple(expert_cfg["target_projection_binds"]),
        window_lengths=tuple(expert_cfg["window_lengths"]),
        seed=int(runner["seed"]),
    )


def _expert_audit(
    preset: str,
    table,
    expert: ForwardBackwardExpertBuffer,
    sampling_mode: str,
    sampling_step_seconds: float | None,
) -> dict[str, object]:
    """Verify clip-safe expert cardinality and retain exact content identities."""
    sampled = table.sample(sampling_mode, sampling_step_seconds)
    offsets = sampled.clip_offsets
    counts = tuple(end - start for start, end in zip(offsets[:-1], offsets[1:], strict=True))
    expected = EXPECTED_TRAIN_COUNTS[preset]
    if (
        len(counts) != expected["clips"]
        or table.clip_index.total_frames != expected["source_frames"]
        or offsets[-1] != expected["expert_frames"]
    ):
        raise RuntimeError("Resolved native expert cardinality differs from the frozen train split.")
    if tuple(expert.clip_offsets.tolist()) != offsets or expert.clip_ids != table.clip_ids:
        raise RuntimeError("Expert offsets or clip ids differ from the command-owned table clock.")
    window_counts = {
        str(length): sum(max(0, count - length) for count in counts) for length in expert.schema.window_lengths
    }
    return {
        "clip_count": len(counts),
        "source_frame_count": table.clip_index.total_frames,
        "expert_frame_count": expert.schema.num_frames,
        "expert_feature_width": expert.schema.expert_feature_width,
        "sample_grid": {"mode": sampling_mode, "step_seconds": sampling_step_seconds},
        "window_lengths": list(expert.schema.window_lengths),
        "window_counts": window_counts,
        "clip_offsets_sha256": _tensor_sha256(expert.clip_offsets),
        "expert_frames_sha256": _tensor_sha256(expert.frames),
        "expert_schema_sha256": expert.schema.schema_hash,
        "expert_data_sha256": expert.schema.data_hash,
        "zero_copy_native_observations": False,
    }


def _trace_observations(
    preset: str,
    trace: np.lib.npyio.NpzFile,
    prefix: str,
    step: int,
    device: torch.device,
) -> TensorDict:
    """Convert one frozen native trace state into the exact named learner groups."""
    if preset == "smpl_cmu":
        values = {"policy": torch.from_numpy(trace[f"{prefix}_observation"][step]).to(device, torch.float32)}
    else:
        state = torch.from_numpy(trace[f"{prefix}_state"][step]).to(device, torch.float32)
        joint_position, joint_velocity, projected_gravity, base_angular_velocity = torch.split(
            state, (29, 29, 3, 3), dim=-1
        )
        values = {
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "projected_gravity": projected_gravity,
            "base_angular_velocity": base_angular_velocity,
            "last_action": torch.from_numpy(trace[f"{prefix}_last_action"][step]).to(device, torch.float32),
            "privileged_state": torch.from_numpy(trace[f"{prefix}_privileged_state"][step]).to(device, torch.float32),
        }
        edge_evidence = trace["learner_auxiliary_raw_evidence"]
        if prefix == "current":
            evidence = np.zeros_like(edge_evidence[step]) if step == 0 else edge_evidence[step - 1]
        else:
            evidence = edge_evidence[step]
        values["transition"] = torch.from_numpy(evidence).to(device, torch.float32)
    return TensorDict(values, batch_size=[2])


def _clone_expert(expert: ForwardBackwardExpertBuffer, seed: int) -> ForwardBackwardExpertBuffer:
    """Create independent sampler state while sharing immutable expert frames."""
    return ForwardBackwardExpertBuffer(
        expert.frames,
        expert.clip_offsets,
        expert.priorities.clone(),
        expert.schema,
        seed=seed,
        clip_ids=expert.clip_ids,
        clip_length_values=expert.clip_length_values,
    )


def _canary_config(
    runner: dict[str, object],
    expert: ForwardBackwardExpertBuffer,
) -> dict[str, object]:
    """Shrink only batch/storage execution scale for one deterministic integration update."""
    cfg = copy.deepcopy(runner)
    cfg["replay"]["capacity_transitions"] = CANARY_REPLAY_TRANSITIONS
    cfg["algorithm"]["batch_size"] = CANARY_BATCH_SIZE
    cfg["torch_compile_mode"] = None
    expert_seed = int(cfg["seed"])

    def provider(_env, _observation_schema, _device, **_kwargs):
        return _clone_expert(expert, expert_seed)

    cfg["expert"]["provider"] = provider
    return cfg


def _append_trace(learner: ForwardBackward, preset: str, trace: np.lib.npyio.NpzFile) -> None:
    """Append logical edges after exact native-autoreset normalization."""
    device = learner.device
    for step in range(trace["terminated"].shape[0]):
        action_applied = torch.from_numpy(trace["action_applied"][step]).to(device).unsqueeze(-1)
        if not torch.any(action_applied):
            continue
        current = _trace_observations(preset, trace, "current", step, device)
        native_returned = _trace_observations(preset, trace, "returned", step, device)
        returned = native_returned
        done = torch.from_numpy(trace["terminated"][step] | trace["truncated"][step]).to(device).unsqueeze(-1)
        if preset == "smpl_cmu":
            final = native_returned
            final_valid = done
            if torch.any(done):
                reset_step = step + 1
                if reset_step >= trace["terminated"].shape[0] or np.any(trace["action_applied"][reset_step]):
                    raise RuntimeError("SMPL next-step trace lacks the reset-only row after a terminal edge.")
                post_reset = _trace_observations(preset, trace, "returned", reset_step, device)
                returned = TensorDict(
                    {
                        name: torch.where(done, post_reset[name], native_returned[name])
                        for name in native_returned.keys()
                    },
                    batch_size=[2],
                )
            actions = torch.from_numpy(trace["actions"][step]).to(device, torch.float32)
        else:
            final = _trace_observations(preset, trace, "final", step, device)
            final_valid = torch.from_numpy(trace["final_observation_valid"][step]).to(device).unsqueeze(-1)
            actions = torch.from_numpy(trace["behavior_action"][step]).to(device, torch.float32)
        collected_actions = learner.act_random(current)
        collected_actions.copy_(actions)
        rewards = (
            torch.from_numpy(trace["environment_reward"][step]).to(device, torch.float32)
            if preset == "g1_lafan"
            else torch.from_numpy(trace["reward"][step]).to(device, torch.float32)
        )
        truncated = torch.from_numpy(trace["truncated"][step]).to(device)
        learner.process_env_step(
            returned,
            rewards,
            done.squeeze(-1),
            {
                "time_outs": truncated,
                "final_obs": final,
                "final_obs_valid": final_valid,
            },
        )
    learner.replay.assert_no_errors()
    if not learner.ready_to_update:
        raise RuntimeError("Native trace did not provide one complete canary batch.")


def _run_canary_once(
    preset: str,
    env: SimpleNamespace,
    runner: dict[str, object],
    expert: ForwardBackwardExpertBuffer,
    trace_path: Path,
) -> dict[str, object]:
    """Construct and mutate one exact-topology learner from frozen native edges."""
    seed = int(runner["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = expert.device
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    trace = np.load(trace_path)
    try:
        observations = _trace_observations(preset, trace, "current", 0, device)
        learner = ForwardBackward.construct_algorithm(observations, env, _canary_config(runner, expert), str(device))
        _append_trace(learner, preset, trace)
        before = _model_component_hashes(learner)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        metrics = learner.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_seconds = time.perf_counter() - started
        after = _model_component_hashes(learner)
        metric_values = {name: float(value) for name, value in sorted(metrics.items())}
        if not all(math.isfinite(value) for value in metric_values.values()):
            raise RuntimeError("First-update canary produced nonfinite metrics.")
        changed = tuple(name for name in before if before[name] != after[name])
        if set(changed) != set(before):
            raise RuntimeError(f"First update left declared model owners unchanged: {set(before) - set(changed)}")
        result = {
            "checkpoint_schema_sha256": learner.checkpoint_header.schema_hash,
            "model_before": canonical_sha256(before),
            "model_after": canonical_sha256(after),
            "changed_components": list(changed),
            "metrics": metric_values,
            "metrics_sha256": canonical_sha256(metric_values),
            "update_step": learner.update_step,
            "versions": dict(sorted(learner.versions.items())),
            "context_buffer_size": learner.context_buffer_size,
            "replay_transitions": learner.replay.num_transitions,
            "update_seconds": update_seconds,
        }
        del learner
        gc.collect()
        return result
    finally:
        trace.close()


def _first_update_canary(
    preset: str,
    env: SimpleNamespace,
    runner: dict[str, object],
    expert: ForwardBackwardExpertBuffer,
    trace_path: Path,
) -> dict[str, object]:
    """Require two independent constructions to produce one identical mutation."""
    first = _run_canary_once(preset, env, runner, expert, trace_path)
    second = _run_canary_once(preset, env, runner, expert, trace_path)
    deterministic_fields = (
        "checkpoint_schema_sha256",
        "model_before",
        "model_after",
        "changed_components",
        "metrics",
        "metrics_sha256",
        "update_step",
        "versions",
        "context_buffer_size",
        "replay_transitions",
    )
    mismatches = tuple(name for name in deterministic_fields if first[name] != second[name])
    if mismatches:
        raise RuntimeError(f"Independent first updates differ: {mismatches}.")
    return {
        "status": "passed",
        "claim_scope": "deterministic_native_trace_integration_not_convergence_or_source_numerical_parity",
        "batch_size": CANARY_BATCH_SIZE,
        "replay_capacity_transitions": CANARY_REPLAY_TRANSITIONS,
        "repeat_count": 2,
        "deterministic_fields": list(deterministic_fields),
        "result": first,
        "repeat_update_seconds": [first["update_seconds"], second["update_seconds"]],
    }


def _normalized_runner_identity(runner: dict[str, object]) -> str:
    """Hash the complete resolved learner configuration."""
    return canonical_sha256(runner)


def measure(args: argparse.Namespace) -> dict[str, object]:
    """Run exact train-corpus attachment and one deterministic learner update."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    construction = _construction_env(args.preset, args.source_artifact_root, args.reference_artifact_root, device)
    table = build_motion_task_table(construction.cfg.commands.motion, construction)
    runner_selection = motion_environment_axes(args.preset) | motion_runner_axes(args.preset)
    runner_cfg = resolve_presets(MotionForwardBackwardRunnerCfg(), selected=runner_selection)
    runner = runner_cfg.to_dict()
    env = _learner_env(table, construction.cfg, args.preset)
    expert = _expert_buffer(args.preset, env, runner)
    trace_path = TRACE_PATHS[args.preset]
    expert_cfg = runner["expert"]
    audit = _expert_audit(
        args.preset,
        table,
        expert,
        expert_cfg["sampling_mode"],
        expert_cfg["sampling_step_seconds"],
    )
    canary = _first_update_canary(args.preset, env, runner, expert, trace_path)
    split = construction.cfg.commands.motion.task_table.source.train
    return {
        "schema": "forward_backward_phase3_motion_learning_evidence_v1",
        "status": "measured",
        "preset": args.preset,
        "claim_scope": {
            "expert_attachment": "exact_train_corpus",
            "first_update": "deterministic_integration_canary",
            "source_numerical_parity": "inherited_from_phase2_not_remeasured_here",
            "convergence_non_inferiority": "not_evaluated",
        },
        "source": {
            "identifier": construction.cfg.commands.motion.task_table.source.identifier,
            "split": split.name,
            "artifact": split.artifact,
            "artifact_sha256": split.artifact_sha256,
            "source_content_sha256": table.clip_index.source_content_sha256,
        },
        "task_table": {
            "identity_sha256": table.cache_identity,
            "frame_builder_version": table.frame_builder_version,
            "frame_builder_identity_sha256": table.frame_builder_identity_sha256,
            "joint_names": list(table.joint_names),
            "reference_frame_names": list(table.reference_frame_names),
        },
        "learner": {
            "runner_config_sha256": _normalized_runner_identity(runner),
            "observation_schema_sha256": _expert_schema(args.preset, runner).schema_hash,
            "routes": runner["obs_groups"],
            "model": runner["model"],
        },
        "expert": audit,
        "first_update_canary": canary,
        "code_identity": {
            "evidence_sha256": _sha256(Path(__file__).resolve()),
            "motion_expert_provider_sha256": _source_sha256(forward_backward_expert_buffer),
            "rsl_algorithm_sha256": _source_sha256(ForwardBackward),
            "rsl_expert_buffer_sha256": _source_sha256(ForwardBackwardExpertBuffer),
            "rsl_replay_sha256": _source_sha256(ForwardBackwardTransitionBatch),
            "native_trace_sha256": _sha256(trace_path),
        },
        "runtime": {
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }


def main() -> None:
    """Parse one native preset and atomically publish measured learning evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smpl_cmu", "g1_lafan"), required=True)
    parser.add_argument("--source_artifact_root", type=Path, required=True)
    parser.add_argument("--reference_artifact_root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = measure(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
