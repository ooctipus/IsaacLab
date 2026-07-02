# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independently consume canonical SMPL-CMU native EMD records."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parent
PRODUCER = ROOT / "smpl_cmu_emd_evidence.py"
IDENTITY = ROOT / "motion_environment_identity.py"
MILESTONES = ROOT / "fixtures" / "native_emd_milestones_v1.json"
_SMPL_DOMAIN = "phase3-smpl-cmu-post-sampler-seed0"
_PACKED_SCHEMA = "forward_backward_phase3_packed_motion_emd_v3"
_COMPACT_SCHEMA = "forward_backward_evaluation_checkpoint_v1"
_EXPECTED_CLIPS = 182


def _sha256(path: Path) -> str:
    """Hash one required regular, non-symbolic file."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"SMPL EMD consumer input must be a regular non-symbolic file: {path}.")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical_sha256(value: object) -> str:
    """Hash one JSON value without presentation whitespace."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_sha256(value: object) -> str:
    """Hash the unwrapped Python source owner of one runtime boundary."""
    path = inspect.getsourcefile(inspect.unwrap(value))
    if path is None:
        raise RuntimeError(f"Cannot locate source for {value!r}.")
    return _sha256(Path(path))


def _load_module(path: Path, name: str):
    """Load one colocated evidence module without a package assumption."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Phase 3 evidence module: {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mapping(value: object, label: str) -> Mapping[str, object]:
    """Return one required mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    return value


def _statistics(values: list[float]) -> dict[str, float | int]:
    """Independently recompute the producer's declared EMD reduction."""
    tensor = torch.tensor(values, dtype=torch.float64)
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
        raise ValueError("SMPL EMD statistics require finite nonempty values.")
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


def _record_paths(milestones_path: Path) -> tuple[tuple[Path, Path], ...]:
    """Resolve the one canonical SMPL milestone series without path escape."""
    milestones_path = milestones_path.resolve()
    manifest = _mapping(json.loads(milestones_path.read_text()), "Native EMD milestone manifest")
    if manifest.get("schema") != "forward_backward_dashboard_native_emd_v1":
        raise ValueError("Native EMD milestone manifest has an unsupported schema.")
    series = manifest.get("series")
    if not isinstance(series, list):
        raise TypeError("Native EMD milestone series must be a list.")
    selected = [value for value in series if isinstance(value, Mapping) and value.get("domain") == _SMPL_DOMAIN]
    if len(selected) != 1 or selected[0].get("evidence") != "checkpoint_record_paired":
        raise ValueError("Native EMD milestones must declare exactly one paired SMPL series.")
    milestones = selected[0].get("milestones")
    if not isinstance(milestones, list) or not milestones:
        raise ValueError("Canonical SMPL native EMD milestones must be nonempty.")

    root = milestones_path.parent.resolve()
    pairs: list[tuple[Path, Path]] = []
    for index, value in enumerate(milestones):
        milestone = _mapping(value, f"SMPL milestone {index}")
        if set(milestone) != {"checkpoint_record", "evaluation_record"}:
            raise ValueError(f"SMPL milestone {index} has an unsupported field set.")
        resolved: list[Path] = []
        for field in ("checkpoint_record", "evaluation_record"):
            relative = milestone[field]
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
                raise ValueError(f"SMPL milestone {index} {field} must be a relative path.")
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"SMPL milestone {index} {field} escapes the fixture root.")
            _sha256(path)
            resolved.append(path)
        pairs.append((resolved[0], resolved[1]))
    if len(set(pairs)) != len(pairs):
        raise ValueError("Canonical SMPL milestones contain duplicate record pairs.")
    return tuple(pairs)


def _current_implementation_identity() -> dict[str, str]:
    """Rehash every locally available packed-evaluator code owner."""
    from rsl_rl.models.forward_backward_model import ForwardBackwardInferenceModel

    from isaaclab_tasks.core.multi_task.motion.config.agents import MotionForwardBackwardRunnerPresetsCfg
    from isaaclab_tasks.core.multi_task.motion.impl import uniform_emd_warp
    from isaaclab_tasks.core.multi_task.motion.tracking import smpl_motion_tracking_evaluator_packed
    from isaaclab_tasks.utils import resolve_presets

    runner = resolve_presets(MotionForwardBackwardRunnerPresetsCfg(), selected={"smpl_cmu"}).to_dict()
    return {
        "evaluator_sha256": _source_sha256(smpl_motion_tracking_evaluator_packed),
        "uniform_emd_warp_sha256": _source_sha256(uniform_emd_warp),
        "producer_sha256": _sha256(PRODUCER),
        "model_config_sha256": _canonical_sha256(runner["model"]),
        "observation_routes_sha256": _canonical_sha256(runner["obs_groups"]),
        "forward_backward_inference_model_sha256": _source_sha256(ForwardBackwardInferenceModel),
    }


def _validate_implementation(value: object) -> None:
    """Require the exact current packed evaluator, transport, and inference owners."""
    implementation = _mapping(value, "SMPL EMD implementation identity")
    expected = _current_implementation_identity()
    if set(implementation) != set(expected):
        raise ValueError("SMPL EMD implementation identity has an unsupported field set.")
    for name, digest in expected.items():
        if implementation[name] != digest:
            raise ValueError(f"SMPL EMD {name.removesuffix('_sha256').replace('_', ' ')} bytes are stale.")


def _current_environment_identity(record: Mapping[str, object]) -> dict[str, object]:
    """Rebuild the current environment identity with the record's declared horizon."""
    from isaaclab_tasks.core.multi_task.motion.data.importers import HumEnvHdf5Clips
    from isaaclab_tasks.core.multi_task.motion.trajectory.smpl import SmplHumEnvFrameBuilder
    from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
    from isaaclab_tasks.utils import resolve_presets

    protocol = _mapping(record.get("protocol"), "SMPL EMD protocol")
    horizon = protocol.get("maximum_source_frames_per_clip")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 2:
        raise ValueError("SMPL EMD protocol has an invalid evaluation horizon.")
    cfg = resolve_presets(MotionImitationEnvCfg(), selected={"smpl_cmu"})
    cfg.commands.motion.task_table.motion_split = "evaluation"
    cfg.commands.motion.payload.episode_length_steps = horizon
    cfg.terminations.time_out.params["applied_actions_before_timeout"] = horizon
    cfg.episode_length_s = horizon * cfg.sim.dt * cfg.decimation
    identity = _load_module(IDENTITY, "smpl_cmu_emd_environment_identity")
    return identity.motion_environment_dependency_identity(
        preset="smpl_cmu",
        cfg=cfg,
        importer_type=HumEnvHdf5Clips,
        frame_builder_type=SmplHumEnvFrameBuilder,
    )


def _validate_environment(record: Mapping[str, object]) -> None:
    """Close the dependency source set and semantic digest against current code."""
    environment = _mapping(record.get("environment"), "SMPL EMD environment identity")
    if set(environment) != {"dependency_identity", "semantic_sha256", "native_owner_hashes"}:
        raise ValueError("SMPL EMD environment identity has an unsupported field set.")
    dependency = _mapping(environment["dependency_identity"], "SMPL EMD dependency identity")
    identity = _load_module(IDENTITY, "smpl_cmu_emd_stored_environment_identity")
    stored_semantic = identity.motion_environment_semantic_sha256(dependency)
    if environment["semantic_sha256"] != stored_semantic:
        raise ValueError("SMPL EMD environment semantic digest is stale.")

    current = _current_environment_identity(record)
    if dependency.get("python_sources") != current["python_sources"]:
        raise ValueError("SMPL EMD dependency Python source closure differs from current bytes.")
    current_semantic = identity.motion_environment_semantic_sha256(current)
    if stored_semantic != current_semantic:
        raise ValueError("SMPL EMD environment semantics differ from the current preset.")

    owners = _mapping(environment["native_owner_hashes"], "SMPL native owner identity")
    expected_sources = {
        name: dependency["python_sources"][name]
        for name in (
            "isaaclab_newton.cloner.newton_clone_utils",
            "isaaclab_newton.cloner.replicate",
            "isaaclab_newton.physics.mjwarp_manager",
            "isaaclab_newton.physics.newton_manager",
            "isaaclab_newton.sim.spawners.mjcf.mjcf",
            "isaaclab_newton.sim.spawners.mjcf.mjcf_cfg",
            "isaaclab_tasks.core.multi_task.motion.config.robots.smpl",
        )
    }
    expected_assets = {
        name: digest
        for name, digest in dependency["robot_assets"].items()
        if name.startswith(("simulation/", "reference/"))
    }
    if owners != {"python_sources": expected_sources, "robot_assets": expected_assets}:
        raise ValueError("SMPL EMD native owner projection differs from the dependency identity.")


def _validate_checkpoint_pair(
    checkpoint_path: Path,
    evaluation_path: Path,
    record: Mapping[str, object],
) -> tuple[int, str]:
    """Cross-check local compact-manifest bytes with the evaluation identity."""
    compact = _mapping(json.loads(checkpoint_path.read_text()), "Compact checkpoint record")
    if compact.get("schema") != _COMPACT_SCHEMA:
        raise ValueError("SMPL compact checkpoint record has an unsupported schema.")
    output = _mapping(compact.get("output"), "Compact checkpoint output")
    checkpoint = _mapping(record.get("checkpoint"), "SMPL EMD checkpoint identity")
    transition = compact.get("collected_transitions")
    iteration = compact.get("iteration")
    if (
        isinstance(transition, bool)
        or not isinstance(transition, int)
        or transition < 0
        or isinstance(iteration, bool)
        or not isinstance(iteration, int)
        or iteration < 0
    ):
        raise ValueError("SMPL compact checkpoint record has an invalid transition identity.")
    expected = {
        "bytes": output.get("bytes"),
        "iteration": iteration,
        "sha256": output.get("sha256"),
        "transition": transition,
    }
    for name, value in expected.items():
        if checkpoint.get(name) != value:
            raise ValueError(f"SMPL EMD checkpoint {name} differs from its compact record.")
    if checkpoint.get("manifest_sha256") != _sha256(checkpoint_path):
        raise ValueError("SMPL EMD checkpoint manifest bytes differ from the configured compact record.")
    output_name = output.get("filename")
    if (
        not isinstance(output_name, str)
        or Path(str(checkpoint.get("path"))).name != output_name
        or Path(str(checkpoint.get("manifest_path"))).name != checkpoint_path.name
        or evaluation_path.name != f"{transition}.json"
    ):
        raise ValueError("SMPL EMD checkpoint filenames differ across the paired records.")
    digest = output.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("SMPL compact checkpoint output has an invalid SHA-256 digest.")
    return transition, digest


def _validate_emd_rows(record: Mapping[str, object]) -> float:
    """Recompute all 182 per-clip values and their aggregate statistics."""
    clip_ids = record.get("clip_ids")
    metrics = record.get("metrics")
    emd = record.get("emd")
    if (
        not isinstance(clip_ids, list)
        or len(clip_ids) != _EXPECTED_CLIPS
        or len(set(clip_ids)) != _EXPECTED_CLIPS
        or not all(isinstance(value, str) and value for value in clip_ids)
        or not isinstance(metrics, Mapping)
        or set(metrics) != set(clip_ids)
        or not isinstance(emd, list)
        or len(emd) != _EXPECTED_CLIPS
    ):
        raise ValueError("SMPL EMD record does not contain exactly 182 uniquely paired clip rows.")

    values: list[float] = []
    metric_frames = 0
    for index, clip_id in enumerate(clip_ids):
        row = _mapping(metrics[clip_id], f"SMPL EMD metric {clip_id!r}")
        if set(row) != {"coverage_fraction", "emd", "evaluated_num_frames", "num_frames", "source_num_frames"}:
            raise ValueError(f"SMPL EMD metric {clip_id!r} has an unsupported field set.")
        value = row["emd"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"SMPL EMD metric {clip_id!r} is not finite and non-negative.")
        if emd[index] != value:
            raise ValueError("SMPL EMD vector differs from its clip-keyed metric rows.")
        frames = row["num_frames"]
        if (
            isinstance(frames, bool)
            or not isinstance(frames, int)
            or frames < 1
            or row["evaluated_num_frames"] != frames
            or row["source_num_frames"] != frames
            or row["coverage_fraction"] != 1.0
        ):
            raise ValueError(f"SMPL EMD metric {clip_id!r} is not a complete held-out rollout.")
        values.append(float(value))
        metric_frames += frames

    if record.get("emd_statistics") != _statistics(values):
        raise ValueError("SMPL EMD statistics differ from the independently recomputed clip rows.")
    protocol = _mapping(record.get("protocol"), "SMPL EMD protocol")
    if (
        protocol.get("clip_count") != _EXPECTED_CLIPS
        or protocol.get("metric_frame_count") != metric_frames
        or protocol.get("source_frame_count") != metric_frames + _EXPECTED_CLIPS
    ):
        raise ValueError("SMPL EMD protocol counts differ from the independently consumed rows.")
    return _statistics(values)["mean"]


def consume_canonical_smpl_emd_records(milestones_path: Path = MILESTONES) -> dict[str, object]:
    """Validate every configured SMPL evaluation and its compact checkpoint record."""
    pairs = _record_paths(milestones_path)
    loaded: list[tuple[Path, Mapping[str, object], int, str]] = []
    for checkpoint_path, evaluation_path in pairs:
        record = _mapping(json.loads(evaluation_path.read_text()), "SMPL EMD evaluation record")
        transition, digest = _validate_checkpoint_pair(checkpoint_path, evaluation_path, record)
        loaded.append((evaluation_path, record, transition, digest))

    transitions = [value[2] for value in loaded]
    if transitions != sorted(transitions) or len(set(transitions)) != len(transitions):
        raise ValueError("Canonical SMPL EMD milestones must have unique increasing transitions.")
    checkpoint_batch = [{"transition": transition, "sha256": digest} for _path, _record, transition, digest in loaded]
    batch_sha256 = _canonical_sha256(checkpoint_batch)
    means: list[float] = []
    common: dict[str, object] | None = None
    state_bytes: list[int] = []

    for index, (_path, record, transition, _digest) in enumerate(loaded):
        if record.get("schema") != _PACKED_SCHEMA or record.get("status") != "measured":
            raise ValueError("Canonical SMPL EMD record is not measured packed evidence.")
        if record.get("profile") != "smpl_cmu":
            raise ValueError("Canonical SMPL EMD record has the wrong profile.")
        protocol = _mapping(record.get("protocol"), "SMPL EMD protocol")
        if protocol.get("evaluator_mode") != "packed":
            raise ValueError("Canonical SMPL EMD record must declare packed evaluator mode.")
        _validate_implementation(record.get("implementation"))
        _validate_environment(record)
        means.append(_validate_emd_rows(record))

        execution = _mapping(record.get("execution"), "SMPL EMD packed execution identity")
        expected_execution = {
            "checkpoint_count": len(loaded),
            "checkpoint_index": index,
            "checkpoint_batch": checkpoint_batch,
            "checkpoint_batch_sha256": batch_sha256,
        }
        for name, value in expected_execution.items():
            if execution.get(name) != value:
                raise ValueError(f"SMPL EMD packed execution {name} differs from the configured milestone batch.")
        lanes = execution.get("lanes_per_checkpoint")
        if (
            not isinstance(lanes, int)
            or lanes < 1
            or execution.get("environment_num_envs") != len(loaded) * lanes
            or protocol.get("num_envs") != lanes
            or execution.get("inference") != "torch_func_stack_module_state_vmap"
        ):
            raise ValueError("SMPL EMD packed lane or inference layout is inconsistent.")
        per_policy_bytes = execution.get("inference_state_bytes")
        if not isinstance(per_policy_bytes, int) or per_policy_bytes < 1:
            raise ValueError("SMPL EMD packed inference-state size is invalid.")
        state_bytes.append(per_policy_bytes)

        shared = {
            "protocol": protocol,
            "source": record.get("source"),
            "environment": record.get("environment"),
            "implementation": record.get("implementation"),
            "checkpoint_batch_sha256": execution.get("checkpoint_batch_sha256"),
        }
        if common is None:
            common = shared
        elif shared != common:
            raise ValueError("SMPL EMD milestone records do not share one evaluation identity.")

    stacked_bytes = sum(state_bytes)
    if any(
        _mapping(record.get("execution"), "SMPL EMD packed execution identity").get("stacked_inference_state_bytes")
        != stacked_bytes
        for _path, record, _transition, _digest in loaded
    ):
        raise ValueError("SMPL EMD stacked inference-state size differs from its checkpoint members.")
    return {
        "record_count": len(loaded),
        "transitions": transitions,
        "emd_means": means,
        "checkpoint_batch_sha256": batch_sha256,
    }


def test_emd_row_consumer_rejects_a_redeclared_stale_summary() -> None:
    """A producer-declared mean cannot replace independent reduction of all clip rows."""
    clip_ids = [f"clip_{index}" for index in range(_EXPECTED_CLIPS)]
    values = [float(index + 1) / 100.0 for index in range(_EXPECTED_CLIPS)]
    record = {
        "clip_ids": clip_ids,
        "metrics": {
            clip_id: {
                "coverage_fraction": 1.0,
                "emd": value,
                "evaluated_num_frames": 2,
                "num_frames": 2,
                "source_num_frames": 2,
            }
            for clip_id, value in zip(clip_ids, values, strict=True)
        },
        "emd": values,
        "emd_statistics": _statistics(values),
        "protocol": {
            "clip_count": _EXPECTED_CLIPS,
            "metric_frame_count": 2 * _EXPECTED_CLIPS,
            "source_frame_count": 3 * _EXPECTED_CLIPS,
        },
    }
    assert _validate_emd_rows(record) == _statistics(values)["mean"]
    record["emd_statistics"] = {**record["emd_statistics"], "mean": 0.0}
    with pytest.raises(ValueError, match="independently recomputed"):
        _validate_emd_rows(record)


def test_checkpoint_consumer_rejects_changed_compact_manifest_bytes(tmp_path: Path) -> None:
    """The configured compact record, transition, and model SHA must remain paired."""
    compact_path = tmp_path / "model_1000.json"
    evaluation_path = tmp_path / "500000.json"
    compact = {
        "schema": _COMPACT_SCHEMA,
        "iteration": 1000,
        "collected_transitions": 500_000,
        "output": {"filename": "model_1000.pt", "bytes": 17, "sha256": "a" * 64},
    }
    compact_path.write_text(json.dumps(compact))
    record = {
        "checkpoint": {
            "bytes": 17,
            "iteration": 1000,
            "manifest_path": "/remote/model_1000.json",
            "manifest_sha256": _sha256(compact_path),
            "path": "/remote/model_1000.pt",
            "sha256": "a" * 64,
            "transition": 500_000,
        }
    }
    assert _validate_checkpoint_pair(compact_path, evaluation_path, record) == (500_000, "a" * 64)
    compact_path.write_text(json.dumps({**compact, "model_tensor_count": 116}))
    with pytest.raises(ValueError, match="manifest bytes differ"):
        _validate_checkpoint_pair(compact_path, evaluation_path, record)


def test_implementation_consumer_rejects_each_stale_current_owner() -> None:
    """Producer, unwrapped evaluator, Warp kernel, and inference bytes are current."""
    current = _current_implementation_identity()
    for field in current:
        stale = {**current, field: "0" * 64}
        with pytest.raises(ValueError, match="bytes are stale"):
            _validate_implementation(stale)


def test_canonical_smpl_emd_records_close_current_code_and_recompute_every_clip() -> None:
    """All ten configured checkpoints must form one current, independently reduced batch."""
    receipt = consume_canonical_smpl_emd_records()
    assert receipt["record_count"] == 10
    assert receipt["transitions"] == list(range(500_000, 5_000_001, 500_000))
    assert len(receipt["emd_means"]) == 10
