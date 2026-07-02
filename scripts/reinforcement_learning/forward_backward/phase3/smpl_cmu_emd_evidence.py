# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate one compact Phase 3 SMPL-CMU checkpoint on the native held-out split."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import torch
from rsl_rl.models.forward_backward_model import ForwardBackwardInferenceModel, ForwardBackwardModel
from tensordict import TensorDictBase

_SCHEMA = "forward_backward_phase3_native_motion_emd_v2"
_PACKED_SCHEMA = "forward_backward_phase3_packed_motion_emd_v3"
_COMPACT_SCHEMA = "forward_backward_evaluation_checkpoint_v1"
_PROFILE = "smpl_cmu"
_EXPECTED_CLIPS = 182
_EXPECTED_SOURCE_FRAMES = 88_364
_NATIVE_ENVIRONMENT_SOURCE_OWNERS = (
    "isaaclab_newton.cloner.newton_clone_utils",
    "isaaclab_newton.cloner.replicate",
    "isaaclab_newton.physics.mjwarp_manager",
    "isaaclab_newton.physics.newton_manager",
    "isaaclab_newton.sim.spawners.mjcf.mjcf",
    "isaaclab_newton.sim.spawners.mjcf.mjcf_cfg",
    "isaaclab_tasks.core.multi_task.motion.config.robots.smpl",
)


class _EvidenceRequest(NamedTuple):
    """Canonical checkpoint, output, and lane request closed before launch."""

    identities: tuple[dict[str, object], ...]
    outputs: tuple[Path, ...]
    lanes_per_policy_override: int | None
    evaluator_mode: str


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one regular non-symbolic file."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"EMD evidence input must be a regular non-symbolic file: {path}.")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical_sha256(value: object) -> str:
    """Hash one JSON-compatible value without presentation whitespace."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_sha256(value: object) -> str:
    """Hash the Python source file defining one measured runtime boundary."""
    path = inspect.getsourcefile(inspect.unwrap(value))
    if path is None:
        raise RuntimeError(f"Cannot locate source for {value!r}.")
    return _sha256(Path(path))


def _statistics(values: list[float]) -> dict[str, float | int]:
    """Return finite scalar statistics for one nonempty metric vector."""
    import torch

    tensor = torch.tensor(values, dtype=torch.float64)
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
        raise ValueError("EMD evidence statistics require finite nonempty values.")
    quantiles = torch.quantile(tensor, torch.tensor((0.5, 0.95, 0.99), dtype=torch.float64))
    return {
        "count": tensor.numel(),
        "minimum": float(tensor.min()),
        "mean": float(tensor.mean()),
        "q50": float(quantiles[0]),
        "q95": float(quantiles[1]),
        "q99": float(quantiles[2]),
        "maximum": float(tensor.max()),
    }


def _native_environment_owner_hashes(identity: Mapping[str, object]) -> dict[str, dict[str, str]]:
    """Retain the native MJCF, spawner, cloner, and simulator owner hashes explicitly."""
    python_sources = identity.get("python_sources")
    robot_assets = identity.get("robot_assets")
    if not isinstance(python_sources, Mapping) or not isinstance(robot_assets, Mapping):
        raise TypeError("SMPL environment identity must declare Python sources and robot assets.")
    missing = set(_NATIVE_ENVIRONMENT_SOURCE_OWNERS) - set(python_sources)
    if missing:
        raise ValueError(f"SMPL environment identity is missing native source owners: {sorted(missing)}.")
    selected_sources = {name: python_sources[name] for name in _NATIVE_ENVIRONMENT_SOURCE_OWNERS}
    selected_assets = {
        name: digest
        for name, digest in robot_assets.items()
        if isinstance(name, str) and (name.startswith("simulation/") or name.startswith("reference/"))
    }
    values = (*selected_sources.values(), *selected_assets.values())
    if (
        not selected_assets
        or not any(name.startswith("simulation/") for name in selected_assets)
        or not any(name.startswith("reference/") for name in selected_assets)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in values
        )
    ):
        raise ValueError("SMPL native environment owner hashes are incomplete or malformed.")
    return {
        "python_sources": dict(sorted(selected_sources.items())),
        "robot_assets": dict(sorted(selected_assets.items())),
    }


def _checkpoint_identity(checkpoint: Path) -> dict[str, object]:
    """Validate one model-only checkpoint against its compaction manifest."""
    checkpoint = checkpoint.expanduser().resolve()
    checkpoint_sha256 = _sha256(checkpoint)
    manifest_path = checkpoint.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, Mapping) or manifest.get("schema") != _COMPACT_SCHEMA:
        raise ValueError(f"Unsupported compact checkpoint manifest: {manifest_path}.")
    output = manifest.get("output")
    if not isinstance(output, Mapping):
        raise TypeError("Compact checkpoint manifest must declare its output identity.")
    if (
        output.get("filename") != checkpoint.name
        or output.get("bytes") != checkpoint.stat().st_size
        or output.get("sha256") != checkpoint_sha256
    ):
        raise ValueError("Compact checkpoint bytes differ from their manifest.")
    transition = manifest.get("collected_transitions")
    iteration = manifest.get("iteration")
    if (
        isinstance(transition, bool)
        or not isinstance(transition, int)
        or transition < 0
        or isinstance(iteration, bool)
        or not isinstance(iteration, int)
        or iteration < 0
    ):
        raise ValueError("Compact checkpoint iteration or transition identity is invalid.")
    return {
        "path": str(checkpoint),
        "sha256": checkpoint_sha256,
        "bytes": checkpoint.stat().st_size,
        "iteration": iteration,
        "transition": transition,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
    }


class _StackedForwardBackwardInference:
    """Functional checkpoint stack with checkpoint-major tensor semantics."""

    def __init__(
        self,
        models: tuple[ForwardBackwardInferenceModel, ...],
        *,
        device: torch.device | str,
    ) -> None:
        """Stack compatible deterministic inference views once.

        Args:
            models: Ordered independent inference views.
            device: Device that owns the functional parameter and buffer stack.
        """
        if not isinstance(models, tuple) or not models:
            raise ValueError("Packed inference requires a nonempty tuple of models.")
        if any(not isinstance(model, ForwardBackwardInferenceModel) for model in models):
            raise TypeError("Packed inference requires ForwardBackwardInferenceModel views.")
        self.policy_count = len(models)
        self.observation_schema = models[0].observation_schema
        self.action_dim = models[0].action_dim
        self.context_dim = models[0].context_dim
        self.context_normalization = models[0].context_normalization
        self.state_bytes_per_policy = tuple(
            sum(value.numel() * value.element_size() for value in model.state_dict().values()) for model in models
        )

        parameters, buffers = torch.func.stack_module_state(models)
        target_device = torch.device(device)
        self._parameters = {name: value.to(target_device) for name, value in parameters.items()}
        self._buffers = {name: value.to(target_device) for name, value in buffers.items()}
        self._base = copy.deepcopy(models[0]).to("meta")

        def backward_one(
            parameters: dict[str, torch.Tensor],
            buffers: dict[str, torch.Tensor],
            observations: TensorDictBase,
        ) -> torch.Tensor:
            return torch.func.functional_call(
                self._base,
                (parameters, buffers),
                (observations,),
                {"output": "backward"},
            )

        def action_one(
            parameters: dict[str, torch.Tensor],
            buffers: dict[str, torch.Tensor],
            observations: TensorDictBase,
            context: torch.Tensor,
        ) -> torch.Tensor:
            return torch.func.functional_call(
                self._base,
                (parameters, buffers),
                (observations, context),
                {"output": "action"},
            )

        self._backward_batched = torch.vmap(backward_one)
        self._action_batched = torch.vmap(action_one)
        self.stacked_state_bytes = sum(
            value.numel() * value.element_size()
            for state in (self._parameters, self._buffers)
            for value in state.values()
        )
        if self.stacked_state_bytes != sum(self.state_bytes_per_policy):
            raise RuntimeError("Functional inference state bytes differ from the independent views.")

    def backward_map(self, observations: TensorDictBase) -> torch.Tensor:
        """Evaluate backward features with shape [K, rows, latent]."""
        return self._backward_batched(
            self._parameters,
            self._buffers,
            observations,
        )

    def context_project(self, context: torch.Tensor) -> torch.Tensor:
        """Apply the shared configured context projection without learned state."""
        return self._base.context_project(context)

    def action_deterministic(
        self,
        observations: TensorDictBase,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate deterministic actions with shape [K, rows, action]."""
        return self._action_batched(self._parameters, self._buffers, observations, context)


def _load_stacked_inference(
    template: ForwardBackwardModel,
    identities: tuple[dict[str, object], ...],
    *,
    device: torch.device | str,
) -> _StackedForwardBackwardInference:
    """Load compatible full checkpoints sequentially and retain inference state only."""
    views = []
    for identity in identities:
        checkpoint = Path(identity["path"])
        saved = torch.load(
            checkpoint,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        if not isinstance(saved, dict) or set(saved) != {"model_state_dict"}:
            raise ValueError("SMPL evaluation checkpoint must contain exactly model_state_dict.")
        state = saved["model_state_dict"]
        if not isinstance(state, Mapping):
            raise TypeError("SMPL evaluation model_state_dict must be a tensor mapping.")
        incompatible = template.load_state_dict(state, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("Strict SMPL checkpoint load returned incompatible model keys.")
        template.eval()
        view = copy.deepcopy(template.as_inference_model()).eval()
        view.requires_grad_(False)
        views.append(view)
    return _StackedForwardBackwardInference(tuple(views), device=device)


def _canonical_checkpoint_batch(checkpoints: tuple[Path, ...]) -> tuple[dict[str, object], ...]:
    """Validate and canonically order one immutable compact-checkpoint batch."""
    if not isinstance(checkpoints, tuple) or not checkpoints:
        raise ValueError("Packed SMPL evidence requires a nonempty checkpoint tuple.")
    identities = tuple(_checkpoint_identity(checkpoint) for checkpoint in checkpoints)
    keys = tuple((identity["transition"], identity["sha256"]) for identity in identities)
    if len(set(keys)) != len(keys):
        raise ValueError("Packed checkpoint identities must be unique.")
    transitions = tuple(identity["transition"] for identity in identities)
    if len(set(transitions)) != len(transitions):
        raise ValueError("Packed checkpoint transitions must be unique.")
    return tuple(sorted(identities, key=lambda identity: (identity["transition"], identity["sha256"])))


def _checkpoint_batch_identity(
    identities: tuple[dict[str, object], ...],
) -> tuple[tuple[dict[str, object], ...], str]:
    """Return the exact serialized membership sequence and its canonical digest."""
    members = tuple({"transition": identity["transition"], "sha256": identity["sha256"]} for identity in identities)
    return members, _canonical_sha256(members)


def _batch_output_paths(
    output_dir: Path,
    identities: tuple[dict[str, object], ...],
) -> tuple[Path, ...]:
    """Preflight collision-free immutable output paths before simulator launch."""
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or output_dir.is_symlink()):
        raise ValueError(f"SMPL EMD output directory must be a non-symbolic directory: {output_dir}.")
    outputs = tuple(output_dir / f"{identity['transition']}.json" for identity in identities)
    if len(set(outputs)) != len(outputs):
        raise ValueError("Packed checkpoint identities resolved to duplicate output paths.")
    occupied = tuple(path for path in outputs if path.exists())
    if occupied:
        raise FileExistsError(f"SMPL EMD evidence already exists: {occupied[0]}.")
    return outputs


def _prepare_request(args: argparse.Namespace) -> _EvidenceRequest:
    """Close all artifact paths and output collisions before simulator launch."""
    checkpoints = getattr(args, "checkpoints", None)
    output_dir = getattr(args, "output_dir", None)
    lanes_override = getattr(args, "num_envs", None)
    evaluator_mode = getattr(args, "evaluator_mode", None)
    if lanes_override is not None and (
        isinstance(lanes_override, bool) or not isinstance(lanes_override, int) or lanes_override < 1
    ):
        raise ValueError("SMPL EMD --num_envs must be a positive per-policy lane override.")
    if checkpoints is None or output_dir is None:
        raise ValueError("SMPL EMD requires --checkpoints and --output_dir.")
    if evaluator_mode not in {"faithful", "packed"}:
        raise ValueError("SMPL EMD evaluator_mode must be 'faithful' or 'packed'.")
    identities = _canonical_checkpoint_batch(tuple(checkpoints))
    if evaluator_mode == "faithful" and len(identities) != 1:
        raise ValueError("Faithful SMPL EMD evaluation requires exactly one checkpoint.")
    outputs = _batch_output_paths(output_dir, identities)
    return _EvidenceRequest(identities, outputs, lanes_override, evaluator_mode)


def _publish(path: Path, value: Mapping[str, object]) -> None:
    """Publish one complete record with an atomic same-directory replacement."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"SMPL EMD evidence already exists: {path}.")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _run(args: argparse.Namespace, request: _EvidenceRequest) -> tuple[dict[str, object], ...]:
    """Construct one native environment and measure native or packed SMPL EMD."""

    from gpu_ownership import exclusive_physical_gpu_snapshot, validate_same_exclusive_gpu
    from motion_environment_identity import (
        motion_environment_dependency_identity,
        motion_environment_semantic_sha256,
    )
    from tensordict import TensorDict, TensorDictBase

    from isaaclab_tasks.core.multi_task.motion.config.agents import MotionForwardBackwardRunnerPresetsCfg
    from isaaclab_tasks.core.multi_task.motion.config.sources import SMPL_CMU_SOURCE_CFG
    from isaaclab_tasks.core.multi_task.motion.data.importers import HumEnvHdf5Clips
    from isaaclab_tasks.core.multi_task.motion.impl import uniform_emd_warp
    from isaaclab_tasks.core.multi_task.motion.rsl_rl import motion_expert_buffer_smpl_cmu
    from isaaclab_tasks.core.multi_task.motion.tracking import (
        motion_tracking_refill_lane_count,
        smpl_motion_tracking_evaluator,
        smpl_motion_tracking_evaluator_packed,
    )
    from isaaclab_tasks.core.multi_task.motion.trajectory.smpl import SmplHumEnvFrameBuilder
    from isaaclab_tasks.core.multi_task.motion_env import MotionImitationEnv
    from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
    from isaaclab_tasks.utils import resolve_presets

    checkpoints = request.identities
    policy_count = len(checkpoints)
    source_root = args.source_artifact_root.expanduser().resolve()
    faithful = request.evaluator_mode == "faithful"
    source = SMPL_CMU_SOURCE_CFG.open_split(source_root, SMPL_CMU_SOURCE_CFG.evaluation)
    try:
        source_index = source.inspect()
    finally:
        source.close()
    if len(source_index.clips) != _EXPECTED_CLIPS or source_index.total_frames != _EXPECTED_SOURCE_FRAMES:
        raise RuntimeError("SMPL-CMU evaluation source counts differ from the frozen held-out split.")
    evaluation_horizon = max(clip.frame_count for clip in source_index.clips)
    action_counts = tuple(clip.frame_count - 1 for clip in source_index.clips)
    theoretical_lanes = motion_tracking_refill_lane_count(action_counts)
    if request.lanes_per_policy_override is None:
        lanes_per_policy = 50 if faithful else theoretical_lanes
    else:
        lanes_per_policy = request.lanes_per_policy_override
    if not faithful and lanes_per_policy > len(source_index.clips):
        raise ValueError("Packed SMPL lanes per policy cannot exceed the held-out clip count.")
    environment_num_envs = lanes_per_policy if faithful else policy_count * lanes_per_policy
    checkpoint_batch, checkpoint_batch_sha256 = _checkpoint_batch_identity(checkpoints)

    cfg = resolve_presets(MotionImitationEnvCfg(), selected={_PROFILE})
    cfg.sim.device = args.device
    cfg.scene.num_envs = environment_num_envs
    cfg.seed = args.environment_seed
    table_cfg = cfg.commands.motion.task_table
    table_cfg.source_artifact_root = str(source_root)
    table_cfg.reference_artifact_root = str(args.reference_artifact_root.expanduser().resolve())
    table_cfg.motion_split = "evaluation"
    cfg.commands.motion.payload.episode_length_steps = evaluation_horizon
    cfg.terminations.time_out.params["applied_actions_before_timeout"] = evaluation_horizon
    cfg.episode_length_s = evaluation_horizon * cfg.sim.dt * cfg.decimation
    environment_dependency_identity = motion_environment_dependency_identity(
        preset=_PROFILE,
        cfg=cfg,
        importer_type=HumEnvHdf5Clips,
        frame_builder_type=SmplHumEnvFrameBuilder,
        reference_artifact_root=table_cfg.reference_artifact_root,
    )
    environment_semantic_sha256 = motion_environment_semantic_sha256(environment_dependency_identity)
    native_environment_owner_hashes = _native_environment_owner_hashes(environment_dependency_identity)

    runner_values = resolve_presets(MotionForwardBackwardRunnerPresetsCfg(), selected={_PROFILE}).to_dict()
    env = MotionImitationEnv(cfg=cfg)
    try:
        if env.max_episode_length < evaluation_horizon:
            raise RuntimeError("Resolved evaluation horizon is shorter than the longest held-out clip.")
        table = env.command_manager.get_term("motion").table
        if (
            table.clip_ids != source_index.clip_ids
            or len(table.clip_ids) != _EXPECTED_CLIPS
            or table.clip_index.total_frames != _EXPECTED_SOURCE_FRAMES
            or table.clip_index.content_identity_sha256 != source_index.content_identity_sha256
        ):
            raise RuntimeError("Materialized SMPL evaluation table differs from the inspected source identity.")

        reset = env.reset()
        observations = reset[0] if isinstance(reset, tuple) else reset
        if not isinstance(observations, TensorDictBase):
            observations = TensorDict(observations, batch_size=[env.num_envs])
        model_template = ForwardBackwardModel.from_config(
            observations,
            runner_values["obs_groups"],
            env.action_manager.total_action_dim,
            runner_values["model"],
        )
        if faithful:
            checkpoint = checkpoints[0]
            model = model_template.to(env.device)
            saved = torch.load(Path(checkpoint["path"]), map_location=env.device, weights_only=True)
            if not isinstance(saved, dict) or set(saved) != {"model_state_dict"}:
                raise ValueError("SMPL evaluation checkpoint must contain exactly model_state_dict.")
            incompatible = model.load_state_dict(saved["model_state_dict"], strict=True, assign=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError("Strict SMPL checkpoint load returned incompatible model keys.")
            model.eval()
        else:
            model = _load_stacked_inference(
                model_template,
                checkpoints,
                device=env.device,
            )

        expert = motion_expert_buffer_smpl_cmu(
            env,
            model.observation_schema,
            env.device,
            command_name="motion",
            window_lengths=(8,),
            seed=runner_values["seed"],
        )
        if expert.clip_ids != table.clip_ids or expert.frames.data_ptr() != table.field("observation").data_ptr():
            raise RuntimeError("SMPL evaluator did not retain the table-owned native observation rows.")

        before = exclusive_physical_gpu_snapshot(args.device)
        with env.evaluation_transaction(args.evaluation_seed):
            if faithful:
                evaluations = (smpl_motion_tracking_evaluator(model, env, expert, expert.clip_ids),)
            else:
                evaluations = smpl_motion_tracking_evaluator_packed(
                    model,
                    env,
                    expert,
                    expert.clip_ids,
                    policy_count=policy_count,
                )
        after = exclusive_physical_gpu_snapshot(args.device)
        physical_gpu_uuid = validate_same_exclusive_gpu(before, after)

        protocol = {
            "evaluator_mode": request.evaluator_mode,
            "motion_split": "evaluation",
            "evaluation_seed": args.evaluation_seed,
            "environment_seed": args.environment_seed,
            "num_envs": lanes_per_policy,
            "clip_count": len(expert.clip_ids),
            "source_frame_count": source_index.total_frames,
            "metric_frame_count": sum(action_counts),
            "maximum_source_frames_per_clip": evaluation_horizon,
            "resolved_episode_length": env.max_episode_length,
            "reset": ("source_qpos_qvel_frame_0" if faithful else "source_qpos_qvel_frame_0_on_each_lpt_clip_start"),
            "action": "deterministic_actor_mean",
            "context": "mean_next_up_to_8_backward_features_then_project",
            "metric_projection": "native_humenv_observation_frames_1_to_T_minus_1_columns_0_to_213",
            "transport": "exact_uniform_optimal_assignment_float32_euclidean_cost",
            "autoreset": (
                "uninterrupted_global_horizon_with_same_step_final_obs_fallback"
                if faithful
                else "lpt_selected_exact_resets_with_same_step_final_obs_fallback"
            ),
        }
        source_identity = {
            "split_artifact_sha256": SMPL_CMU_SOURCE_CFG.evaluation.artifact_sha256,
            "source_content_sha256": source_index.source_content_sha256,
            "source_index_content_identity_sha256": source_index.content_identity_sha256,
            "table_cache_identity": table.cache_identity,
            "frame_builder_identity_sha256": table.frame_builder_identity_sha256,
            "expert_data_hash": expert.schema.data_hash,
            "expert_feature_schema_hash": expert.schema.feature_schema_hash,
            "expert_clip_offsets_hash": expert.schema.clip_offsets_hash,
        }
        environment_identity = {
            "dependency_identity": environment_dependency_identity,
            "semantic_sha256": environment_semantic_sha256,
            "native_owner_hashes": native_environment_owner_hashes,
        }
        evaluator = smpl_motion_tracking_evaluator if faithful else smpl_motion_tracking_evaluator_packed
        implementation = {
            "evaluator_sha256": _source_sha256(evaluator),
            "uniform_emd_warp_sha256": _source_sha256(uniform_emd_warp),
            "producer_sha256": _sha256(Path(__file__)),
            "model_config_sha256": _canonical_sha256(runner_values["model"]),
            "observation_routes_sha256": _canonical_sha256(runner_values["obs_groups"]),
        }
        if not faithful:
            implementation["forward_backward_inference_model_sha256"] = _source_sha256(ForwardBackwardInferenceModel)
        gpu_ownership = {
            "physical_gpu_uuid": physical_gpu_uuid,
            "before": before,
            "after": after,
        }
        reports = []
        for checkpoint_index, (checkpoint, evaluation) in enumerate(zip(checkpoints, evaluations, strict=True)):
            metrics = evaluation.serializable_metrics()
            emd = [float(metrics[clip_id]["emd"]) for clip_id in evaluation.clip_ids]
            if any(not math.isfinite(value) or value < 0.0 for value in emd):
                raise ValueError("SMPL evaluation emitted nonfinite or negative EMD.")
            if any("obs_state_emd" in row for row in metrics.values()):
                raise ValueError("SMPL evaluation must not fabricate the G1-only obs_state_emd diagnostic.")

            report = {
                "schema": _SCHEMA if faithful else _PACKED_SCHEMA,
                "status": "measured",
                "profile": _PROFILE,
                "checkpoint": checkpoint,
                "protocol": protocol,
                "source": source_identity,
                "environment": environment_identity,
                "implementation": implementation,
                "gpu_ownership": gpu_ownership,
                "clip_ids": list(evaluation.clip_ids),
                "metrics": metrics,
                "emd": emd,
                "emd_statistics": _statistics(emd),
                "duration_seconds": evaluation.duration_seconds,
            }
            if not faithful:
                report["execution"] = {
                    "checkpoint_count": policy_count,
                    "checkpoint_index": checkpoint_index,
                    "checkpoint_batch": checkpoint_batch,
                    "checkpoint_batch_sha256": checkpoint_batch_sha256,
                    "lanes_per_checkpoint": lanes_per_policy,
                    "theoretical_lanes_per_checkpoint": theoretical_lanes,
                    "environment_num_envs": environment_num_envs,
                    "lane_layout": "checkpoint_major_flat_index_policy_times_lanes_plus_lane",
                    "inference": "torch_func_stack_module_state_vmap",
                    "inference_state_bytes": model.state_bytes_per_policy[checkpoint_index],
                    "stacked_inference_state_bytes": model.stacked_state_bytes,
                }
            reports.append(report)
        return tuple(reports)
    finally:
        env.close()


def main() -> None:
    """Preflight one native or packed evaluation request and publish immutable evidence."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_artifact_root", type=Path, required=True)
    parser.add_argument("--reference_artifact_root", type=Path, default=Path("."))
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--evaluator_mode", choices=("faithful", "packed"), required=True)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--environment_seed", type=int, default=0)
    parser.add_argument("--evaluation_seed", type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.environment_seed < 0 or args.evaluation_seed < 0:
        raise ValueError("SMPL EMD seeds must be non-negative.")
    request = _prepare_request(args)

    launcher = AppLauncher(args)
    simulation_app = launcher.app
    try:
        reports = _run(args, request)
        if len(reports) != len(request.outputs):
            raise RuntimeError("SMPL EMD produced a different record count than the preflighted outputs.")
        for output, report in zip(request.outputs, reports, strict=True):
            _publish(output, report)
        print(
            json.dumps(
                {
                    "schema": "forward_backward_phase3_emd_publication_v1",
                    "outputs": [str(path) for path in request.outputs],
                },
                indent=2,
                sort_keys=True,
            )
        )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
