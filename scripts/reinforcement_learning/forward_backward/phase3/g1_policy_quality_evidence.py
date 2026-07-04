# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate one accepted G1 policy on released G1-retargeted LAFAN or zero-shot CMU motion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import random
import time
import traceback
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

_POLICY_CORPORA = {
    "g1_lafan": {"split": "train", "clip_count": 862, "frame_count": 258_600},
    "g1_cmu": {"split": "evaluation", "clip_count": 182, "frame_count": 88_364},
}
_POLICY_QUALITY_GATE = Path(__file__).parent / "fixtures" / "g1_lafan_policy_quality_gate_v1.json"
_POLICY_QUALITY_PROTOCOL_AUDIT = Path(__file__).parent / "fixtures" / "g1_lafan_policy_quality_protocol_audit_v1.json"


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one required regular file."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Policy-quality input must be a regular non-symbolic file: {path}.")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_sha256(value: object) -> str:
    """Hash the Python file defining one runtime boundary."""
    path = inspect.getsourcefile(inspect.unwrap(value))
    if path is None:
        raise RuntimeError(f"Cannot locate source for {value!r}.")
    return _sha256(Path(path))


def _canonical_sha256(value: object) -> str:
    """Hash one JSON-compatible value without host formatting differences."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _mujoco_model_source_identity(entrypoint: Path) -> dict[str, object]:
    """Return a hash-closed identity for one self-contained MuJoCo model bundle."""
    entrypoint = entrypoint.expanduser()
    root = entrypoint.parent.resolve()
    pending = [entrypoint]
    documents: dict[str, ET.ElementTree] = {}
    files: dict[str, str] = {}

    def register(path: Path) -> tuple[Path, str]:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"MuJoCo model source must be a regular non-symbolic file: {path}.")
        resolved = path.resolve()
        try:
            name = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"MuJoCo model source escapes its bundle root: {path}.") from error
        files[name] = _sha256(path)
        return resolved, name

    while pending:
        xml_path, name = register(pending.pop())
        if name in documents:
            continue
        document = ET.parse(xml_path)
        documents[name] = document
        for include in document.getroot().iter("include"):
            filename = include.get("file")
            if not filename:
                raise ValueError(f"MuJoCo XML include has no file in {xml_path}.")
            pending.append(xml_path.parent / filename)

    directories: dict[str, str] = {}
    for document in documents.values():
        for compiler in document.getroot().iter("compiler"):
            for key in ("assetdir", "meshdir", "texturedir"):
                value = compiler.get(key)
                if value is None:
                    continue
                previous = directories.setdefault(key, value)
                if previous != value:
                    raise ValueError(f"MuJoCo model bundle declares conflicting {key} values.")

    assetdir = directories.get("assetdir", "")
    asset_directories = {
        "mesh": directories.get("meshdir", assetdir),
        "texture": directories.get("texturedir", assetdir),
        "hfield": assetdir,
        "skin": assetdir,
    }
    for document in documents.values():
        for tag, directory in asset_directories.items():
            for asset in document.getroot().iter(tag):
                filename = asset.get("file")
                if filename:
                    register(root / directory / filename)

    ordered_files = dict(sorted(files.items()))
    entrypoint_name = entrypoint.resolve().relative_to(root).as_posix()
    return {
        "schema": "forward_backward_phase3_mujoco_model_source_identity_v1",
        "entrypoint": entrypoint_name,
        "file_count": len(ordered_files),
        "files": ordered_files,
        "bundle_sha256": _canonical_sha256(ordered_files),
    }


def _python_source_identity_module():
    """Load the sibling source-identity authority without a package-path assumption."""
    path = Path(__file__).with_name("python_source_identity.py")
    spec = importlib.util.spec_from_file_location("forward_backward_phase3_python_source_identity", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Phase 3 Python source identity module: {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _python_package_bundle_sha256(package_root: Path) -> str:
    """Delegate complete package identity to the shared Phase 3 authority."""
    return _python_source_identity_module().python_package_bundle_sha256(package_root)


def _statistics(values: Any) -> dict[str, float | int]:
    """Return finite scalar statistics for one nonempty tensor."""
    import torch

    tensor = torch.as_tensor(values, dtype=torch.float64, device="cpu").reshape(-1)
    if tensor.numel() == 0 or not torch.all(torch.isfinite(tensor)):
        raise ValueError("Policy-quality statistics require finite nonempty values.")
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


def _finite(mapping: Mapping[str, object], name: str) -> float:
    """Return one required finite numeric field without accepting booleans."""
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Policy-quality gate field {name!r} must be finite.")
    return float(value)


def _nonnegative(mapping: Mapping[str, object], name: str) -> float:
    """Return one required finite value in the non-negative metric domain."""
    value = _finite(mapping, name)
    if value < 0.0:
        raise ValueError(f"Policy-quality field {name!r} must be non-negative.")
    return value


def _rate(mapping: Mapping[str, object], name: str) -> float:
    """Return one required finite rate in the closed unit interval."""
    value = _finite(mapping, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Policy-quality field {name!r} must lie in [0, 1].")
    return value


def _validate_policy_protocol(protocol: Mapping[str, object]) -> None:
    """Reject malformed counts, flags, seeds, or coverage before comparison."""
    for name in ("preset", "motion_split"):
        if not isinstance(protocol.get(name), str) or not protocol[name]:
            raise ValueError(f"Policy-quality protocol field {name!r} must be a nonempty string.")
    for name in ("clip_count", "frame_count", "reward_task_count", "episodes_per_task", "reward_horizon"):
        value = protocol.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Policy-quality protocol field {name!r} must be a positive integer.")
    seed = protocol.get("evaluation_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Policy-quality evaluation_seed must be a non-negative integer.")
    for name in ("domain_randomization", "observation_noise"):
        if not isinstance(protocol.get(name), bool):
            raise ValueError(f"Policy-quality protocol field {name!r} must be boolean.")
    _rate(protocol, "tracking_coverage_fraction")


def _validate_policy_checkpoint(checkpoint: Mapping[str, object]) -> None:
    """Reject malformed checkpoint identity before comparing policy bytes."""
    for name in ("transition", "training_seed"):
        value = checkpoint.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Policy-quality checkpoint field {name!r} must be a non-negative integer.")
    digest = checkpoint.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("Policy-quality checkpoint sha256 must be a lowercase hexadecimal digest.")


def _load_policy_quality_gate() -> tuple[dict[str, object], str]:
    """Load the one frozen native gate and return its content identity."""
    digest = _sha256(_POLICY_QUALITY_GATE)
    value = json.loads(_POLICY_QUALITY_GATE.read_text())
    if not isinstance(value, dict):
        raise TypeError("Policy-quality gate must contain one JSON object.")
    return value, digest


def _load_policy_quality_protocol_audit() -> tuple[dict[str, object], str]:
    """Load the frozen audit separating diagnostic reward from authoritative evidence."""
    digest = _sha256(_POLICY_QUALITY_PROTOCOL_AUDIT)
    value = json.loads(_POLICY_QUALITY_PROTOCOL_AUDIT.read_text())
    if not isinstance(value, dict):
        raise TypeError("Policy-quality protocol audit must contain one JSON object.")
    if (
        value.get("schema") != "forward_backward_phase3g_g1_policy_quality_protocol_audit_v1"
        or value.get("status") != "frozen_after_reset_corpus_audit"
    ):
        raise ValueError("Policy-quality protocol audit is not the frozen v1 contract.")
    return value, digest


def _lowercase_sha256(value: object, name: str) -> str:
    """Return one required lowercase SHA-256 digest."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Broad-reward comparison identity {name!r} must be a lowercase SHA-256 digest.")
    return value


def _broad_reward_identity_closure(
    audit: Mapping[str, object],
    comparison_identity: Mapping[str, object] | None,
) -> dict[str, object]:
    """Classify whether one future paired reward comparison closes realized-corpus identity."""
    future = audit.get("future_authority_contract")
    if not isinstance(future, Mapping) or future.get("schema") != "forward_backward_g1_broad_reward_episode_bank_v1":
        raise ValueError("Policy-quality protocol audit has no supported future episode-bank contract.")
    required = (
        "baseline_episode_bank_sha256",
        "candidate_episode_bank_sha256",
        "baseline_realized_assignment_sha256",
        "candidate_realized_assignment_sha256",
    )
    if comparison_identity is None:
        return {
            "status": "inconclusive_protocol_identity",
            "identity_closed": False,
            "missing": list(required),
        }
    digests = {name: _lowercase_sha256(comparison_identity.get(name), name) for name in required}
    bank_matches = digests["baseline_episode_bank_sha256"] == digests["candidate_episode_bank_sha256"]
    assignment_matches = (
        digests["baseline_realized_assignment_sha256"] == digests["candidate_realized_assignment_sha256"]
    )
    return {
        "status": "identity_closed" if bank_matches and assignment_matches else "inconclusive_protocol_identity",
        "identity_closed": bank_matches and assignment_matches,
        "episode_bank_sha256_matches": bank_matches,
        "realized_assignment_sha256_matches": assignment_matches,
        "identity": digests,
    }


def _validate_quality_gate_header(gate: Mapping[str, object]) -> None:
    """Reject a gate that is not the predeclared native Phase 3G contract."""
    if gate.get("schema") != "forward_backward_phase3g_g1_lafan_policy_quality_gate_v1":
        raise ValueError("Policy-quality gate has an unsupported schema.")
    if gate.get("status") != "frozen_before_corrected_policy_evaluation":
        raise ValueError("Policy-quality gate was not frozen before evaluation.")


def _native_quality_decision(
    gate: Mapping[str, object],
    *,
    checkpoint: Mapping[str, object],
    protocol: Mapping[str, object],
    tracking: Mapping[str, object],
    broad_reward: Mapping[str, object],
    broad_reward_comparison_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Apply tracking non-inferiority and retain broad reward as a diagnostic."""
    _validate_quality_gate_header(gate)
    protocol_audit, _protocol_audit_sha256 = _load_policy_quality_protocol_audit()
    broad_reward_identity = _broad_reward_identity_closure(protocol_audit, broad_reward_comparison_identity)
    expected_checkpoint = gate.get("checkpoint")
    expected_protocol = gate.get("protocol")
    if not isinstance(expected_checkpoint, Mapping) or not isinstance(expected_protocol, Mapping):
        raise TypeError("Native quality gate must declare checkpoint and protocol mappings.")
    _validate_policy_checkpoint(expected_checkpoint)
    _validate_policy_checkpoint(checkpoint)
    _validate_policy_protocol(expected_protocol)
    _validate_policy_protocol(protocol)
    if expected_checkpoint != checkpoint:
        raise ValueError("Native policy checkpoint differs from the frozen quality gate.")
    if expected_protocol != protocol:
        raise ValueError("Native policy evaluation protocol differs from the frozen quality gate.")
    baseline = gate.get("phase2_baseline")
    acceptance = gate.get("acceptance")
    if not isinstance(baseline, Mapping) or not isinstance(acceptance, Mapping):
        raise TypeError("Native quality gate must declare baseline and acceptance mappings.")
    baseline_tracking = baseline.get("tracking")
    baseline_reward = baseline.get("broad_reward")
    acceptance_tracking = acceptance.get("tracking")
    acceptance_reward = acceptance.get("broad_reward")
    if not all(
        isinstance(value, Mapping)
        for value in (baseline_tracking, baseline_reward, acceptance_tracking, acceptance_reward)
    ):
        raise TypeError("Native quality gate metric groups must be mappings.")

    def absolute_delta(
        actual_values: Mapping[str, object],
        baseline_values: Mapping[str, object],
        limits: Mapping[str, object],
        name: str,
        limit_name: str,
    ) -> dict[str, object]:
        actual = _nonnegative(actual_values, name)
        reference = _nonnegative(baseline_values, name)
        limit = _finite(limits, limit_name)
        if limit < 0.0:
            raise ValueError(f"Policy-quality limit {limit_name!r} must be non-negative.")
        delta = actual - reference
        return {
            "direction": "absolute_delta_max",
            "actual": actual,
            "baseline": reference,
            "delta": delta,
            "absolute_delta": abs(delta),
            "limit": limit,
            "passed": abs(delta) <= limit,
        }

    emd = absolute_delta(
        tracking,
        baseline_tracking,
        acceptance_tracking,
        "emd_mean",
        "emd_mean_absolute_delta_max",
    )
    obs_state_emd = absolute_delta(
        tracking,
        baseline_tracking,
        acceptance_tracking,
        "obs_state_emd_mean",
        "obs_state_emd_mean_absolute_delta_max",
    )
    coverage = _rate(tracking, "coverage_fraction")
    coverage_baseline = _rate(expected_protocol, "tracking_coverage_fraction")
    coverage_minimum = _rate(acceptance_tracking, "coverage_fraction_min")
    coverage_result = {
        "direction": "minimum",
        "actual": coverage,
        "baseline": coverage_baseline,
        "delta": coverage - coverage_baseline,
        "limit": coverage_minimum,
        "passed": coverage >= coverage_minimum,
    }

    return_actual = _finite(broad_reward, "return_mean")
    return_baseline = _finite(baseline_reward, "return_mean")
    return_ratio_minimum = _finite(acceptance_reward, "return_mean_ratio_min")
    if return_baseline <= 0.0 or return_ratio_minimum < 0.0:
        raise ValueError("Native return baseline must be positive and its ratio limit non-negative.")
    return_ratio = return_actual / return_baseline
    return_result = {
        "direction": "ratio_minimum",
        "actual": return_actual,
        "baseline": return_baseline,
        "delta": return_actual - return_baseline,
        "ratio": return_ratio,
        "limit": return_ratio_minimum,
        "point_gate_met": return_ratio >= return_ratio_minimum,
    }

    safety_actual = _rate(broad_reward, "safety_violation_rate_mean")
    safety_baseline = _rate(baseline_reward, "safety_violation_rate_mean")
    safety_limit = _finite(acceptance_reward, "safety_violation_rate_mean_increase_max")
    if safety_limit < 0.0:
        raise ValueError("Native safety-increase limit must be non-negative.")
    safety_delta = safety_actual - safety_baseline
    safety_result = {
        "direction": "increase_max",
        "actual": safety_actual,
        "baseline": safety_baseline,
        "delta": safety_delta,
        "limit": safety_limit,
        "point_gate_met": safety_delta <= safety_limit,
    }

    termination_actual = _rate(broad_reward, "termination_rate_mean")
    termination_baseline = _rate(baseline_reward, "termination_rate_mean")
    termination_limit = _rate(acceptance_reward, "termination_rate_mean_max")
    termination_result = {
        "direction": "maximum",
        "actual": termination_actual,
        "baseline": termination_baseline,
        "delta": termination_actual - termination_baseline,
        "limit": termination_limit,
        "point_gate_met": termination_actual <= termination_limit,
    }
    auxiliary_cost = absolute_delta(
        broad_reward,
        baseline_reward,
        acceptance_reward,
        "auxiliary_cost_mean",
        "auxiliary_cost_mean_absolute_delta_max",
    )
    action_l2 = absolute_delta(
        broad_reward,
        baseline_reward,
        acceptance_reward,
        "action_l2_mean",
        "action_l2_mean_absolute_delta_max",
    )

    def as_point_gate(result: Mapping[str, object]) -> dict[str, object]:
        value = dict(result)
        value["point_gate_met"] = value.pop("passed")
        return value

    tracking_metrics = {
        "emd_mean": emd,
        "obs_state_emd_mean": obs_state_emd,
        "coverage_fraction": coverage_result,
    }
    broad_reward_metrics = {
        "return_mean": return_result,
        "safety_violation_rate_mean": safety_result,
        "termination_rate_mean": termination_result,
        "auxiliary_cost_mean": as_point_gate(auxiliary_cost),
        "action_l2_mean": as_point_gate(action_l2),
    }
    tracking_passed = all(result["passed"] for result in tracking_metrics.values())
    reward_point_gate_met = all(result["point_gate_met"] for result in broad_reward_metrics.values())
    return {
        "kind": "native_tracking_non_inferiority",
        "threshold_applied": True,
        "passed": tracking_passed,
        "status": "passed" if tracking_passed else "failed",
        "completion_scope": {
            "phase2_owner": "convergence_and_policy_quality",
            "phase3_owner": "deterministic_environment_learner_integration_and_structural_composition",
            "broad_reward_role": "diagnostic_only",
        },
        "protocol": {"passed": True, "actual": dict(protocol)},
        "metrics": {"tracking": tracking_metrics},
        "diagnostics": {
            "broad_reward": {
                "authoritative_threshold_scope": "tracking",
                "status": "inconclusive",
                "classification": broad_reward_identity["status"],
                "authoritative": False,
                "episode_rows_per_metric": 380,
                "identity_closure": broad_reward_identity,
                "point_gate": {
                    "threshold_applied": True,
                    "result": "met" if reward_point_gate_met else "not_met",
                    "metrics": broad_reward_metrics,
                },
            }
        },
    }


def _cross_source_quality_decision(
    gate: Mapping[str, object],
    *,
    checkpoint: Mapping[str, object],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Close zero-shot G1-CMU evidence without inventing a native threshold."""
    _validate_quality_gate_header(gate)
    expected_checkpoint = gate.get("checkpoint")
    shared_protocol = gate.get("protocol")
    if not isinstance(expected_checkpoint, Mapping) or not isinstance(shared_protocol, Mapping):
        raise TypeError("Policy-quality gate must declare checkpoint and protocol mappings.")
    _validate_policy_checkpoint(expected_checkpoint)
    _validate_policy_checkpoint(checkpoint)
    _validate_policy_protocol(shared_protocol)
    _validate_policy_protocol(protocol)
    if checkpoint != expected_checkpoint:
        raise ValueError("Cross-source policy checkpoint differs from the frozen native policy.")
    expected = gate.get("cross_source_measurement")
    if not isinstance(expected, Mapping):
        raise TypeError("Policy-quality gate must declare the cross-source measurement contract.")
    for name in ("preset", "motion_split", "clip_count", "frame_count"):
        if protocol.get(name) != expected.get(name):
            raise ValueError(f"Cross-source policy protocol differs at {name!r}.")
    for name in (
        "evaluation_seed",
        "domain_randomization",
        "observation_noise",
        "reward_task_count",
        "episodes_per_task",
        "reward_horizon",
    ):
        if protocol.get(name) != shared_protocol.get(name):
            raise ValueError(f"Cross-source policy protocol differs at shared field {name!r}.")
    coverage = _rate(protocol, "tracking_coverage_fraction")
    coverage_minimum = _rate(expected, "required_tracking_coverage_fraction")
    complete = coverage >= coverage_minimum
    claim = expected.get("claim")
    if not isinstance(claim, str) or not claim:
        raise ValueError("Cross-source measurement contract must state its claim boundary.")
    return {
        "kind": "zero_shot_cross_source_measurement",
        "threshold_applied": False,
        "passed": None,
        "status": "measured" if complete else "incomplete",
        "measurement_complete": complete,
        "protocol": {"passed": True, "actual": dict(protocol)},
        "tracking_coverage_fraction": {
            "direction": "minimum",
            "actual": coverage,
            "limit": coverage_minimum,
            "passed": complete,
        },
        "claim": claim,
    }


def _policy_motion_split(preset: str) -> str:
    """Return the frozen behavior-evaluation corpus selected by one preset."""
    try:
        return _POLICY_CORPORA[preset]["split"]
    except KeyError as error:
        raise ValueError(f"Unsupported policy-evaluation preset: {preset!r}.") from error


def _as_tensordict(observations: object, num_envs: int):
    """Expose one manager observation mapping through the model contract."""
    from tensordict import TensorDict, TensorDictBase

    if isinstance(observations, TensorDictBase):
        return observations
    if not isinstance(observations, Mapping):
        raise TypeError("Policy-quality observations must be a tensor mapping.")
    return TensorDict(dict(observations), batch_size=[num_envs])


class _BFMRewardContextPolicy:
    """Expose only the model protocol consumed by frozen BFM reward inference."""

    def __init__(self, model: Any):
        self.model = model

    @property
    def device(self):
        """Return the model parameter device."""
        return next(self.model.parameters()).device

    def backward_map(self, observations: Mapping[str, Any]):
        """Encode reward-inference observations through the unified backward map."""
        import torch

        joint_position, joint_velocity, projected_gravity, base_angular_velocity = torch.split(
            observations["state"], (29, 29, 3, 3), dim=-1
        )
        fields = {
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "projected_gravity": projected_gravity,
            "base_angular_velocity": base_angular_velocity,
            "privileged_state": observations["privileged_state"],
        }
        with torch.no_grad():
            return self.model.backward_map(_as_tensordict(fields, next(iter(fields.values())).shape[0]))

    def project_z(self, context: Any):
        """Project inferred contexts with the unified model geometry."""
        return self.model.context_project(context)


def _g1_reward_state_sources(env: Any) -> tuple[Any, Any]:
    """Validate and return the physical robot and behavior-ordered action term."""
    import torch

    command = env.command_manager.get_term("motion")
    table = command.table
    robot = command.payload.robot
    action = env.action_manager.get_term("joint_position")
    physical_names = tuple(robot.joint_names)
    behavior_names = tuple(action.joint_names)
    if table.joint_names != physical_names:
        raise ValueError("The G1 trajectory table must retain the physical articulation joint axis.")
    if len(behavior_names) != len(physical_names) or set(behavior_names) != set(physical_names):
        raise ValueError("The G1 behavior and physical joint axes must be complete permutations.")
    expected_ids = torch.tensor(
        [physical_names.index(name) for name in behavior_names],
        dtype=action.joint_ids.dtype,
        device=action.joint_ids.device,
    )
    if action.joint_ids.shape != expected_ids.shape or not torch.equal(action.joint_ids, expected_ids):
        raise ValueError("The G1 action term has an invalid behavior-to-physical joint map.")
    return robot, action


def _g1_qpos_qvel(
    env: Any,
    robot: Any,
    action: Any,
    qpos: Any | None = None,
    qvel: Any | None = None,
) -> tuple[Any, Any]:
    """Write released qpos/qvel through the validated behavior-axis boundary."""
    import torch

    if qpos is None:
        qpos = torch.empty(env.num_envs, 36, dtype=torch.float32, device=env.device)
    if qvel is None:
        qvel = torch.empty(env.num_envs, 35, dtype=torch.float32, device=env.device)
    if qpos.shape != (env.num_envs, 36) or qvel.shape != (env.num_envs, 35):
        raise ValueError("G1 qpos/qvel outputs have incompatible shapes.")
    torch.sub(robot.data.root_pos_w.torch, env.scene.env_origins, out=qpos[:, :3])
    qpos[:, 3:7].copy_(robot.data.root_quat_w.torch)
    torch.index_select(robot.data.joint_pos.torch, 1, action.joint_ids, out=qpos[:, 7:])
    qvel[:, :3].copy_(robot.data.root_lin_vel_w.torch)
    qvel[:, 3:6].copy_(robot.data.root_ang_vel_b.torch)
    torch.index_select(robot.data.joint_vel.torch, 1, action.joint_ids, out=qvel[:, 6:])
    return qpos, qvel


def _load_bfm_reward_operators(root: Path) -> dict[str, object]:
    """Load and fingerprint the frozen broad-reward implementation."""
    from bfm_reward_runtime import (
        BFM_AUXILIARY_COST_COEFFICIENTS,
        BFM_AUXILIARY_EVIDENCE_NAMES,
        BFM_HARD_SAFETY_NAMES,
        BFM_REWARD_TASKS,
        BFM_REWARD_TASKS_SHA256,
        BfmRewardRuntime,
        bfm_reward_source_identity,
        infer_reward_contexts_from_dataset,
        reward_metric_rows,
    )

    return {
        "reward_context_policy": _BFMRewardContextPolicy,
        "runtime_type": BfmRewardRuntime,
        "auxiliary_names": BFM_AUXILIARY_EVIDENCE_NAMES,
        "auxiliary_coefficients": BFM_AUXILIARY_COST_COEFFICIENTS,
        "hard_safety_names": BFM_HARD_SAFETY_NAMES,
        "tasks": BFM_REWARD_TASKS,
        "tasks_sha256": BFM_REWARD_TASKS_SHA256,
        "infer_contexts": infer_reward_contexts_from_dataset,
        "metric_rows": reward_metric_rows,
        "runtime_sha256": _source_sha256(BfmRewardRuntime),
        "code_identity": bfm_reward_source_identity(root),
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


def _broad_reward_rollout(
    *,
    model: Any,
    env: Any,
    scope_env: Any,
    evaluation_scope: Any,
    command: Any,
    domain_scope: Any,
    history_factory: Any,
    dataset: Mapping[str, object],
    reward_runtime: Any,
    runtime_setup_seconds: float,
    operators: Mapping[str, object],
    auxiliary_evidence_names: tuple[str, ...],
    episodes_per_task: int,
    horizon: int,
    batch_size: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Run the same broad tasks and normalization on one motion-preset reset law."""
    import torch

    def synchronize() -> None:
        if torch.device(env.device).type == "cuda":
            torch.cuda.synchronize(env.device)

    total_started = time.perf_counter()
    policy = operators["reward_context_policy"](model)
    context_started = time.perf_counter()
    with torch.inference_mode():
        contexts = operators["infer_contexts"](
            policy,
            dataset,
            batch_size=batch_size,
            reference_config_sha256=dataset["reference_config_sha256"],
            data_sha256=dataset["data_sha256"],
            reward_model_sha256=dataset["reward_model_sha256"],
        )
    synchronize()
    context_seconds = time.perf_counter() - context_started
    tasks = operators["tasks"]
    task_count = len(tasks)
    num_envs = task_count * episodes_per_task
    if env.num_envs != num_envs:
        raise ValueError(f"Broad reward requires exactly {num_envs} environments, got {env.num_envs}.")
    contexts = contexts.repeat_interleave(episodes_per_task, dim=0)
    evidence_names = tuple(operators["auxiliary_names"])
    if auxiliary_evidence_names != evidence_names:
        raise ValueError("Learner and broad-reward auxiliary evidence orders differ.")
    evidence_count = len(evidence_names)
    robot, action_term = _g1_reward_state_sources(env)
    qpos = torch.empty(num_envs, 36, device=env.device)
    qvel = torch.empty(num_envs, 35, device=env.device)
    task_returns = torch.zeros(task_count, episodes_per_task, device=env.device)
    evidence_sum = torch.zeros(task_count, episodes_per_task, evidence_count, device=env.device)
    evidence_active_count = torch.zeros_like(evidence_sum)
    auxiliary_cost_sum = torch.zeros_like(task_returns)
    safety_violation_count = torch.zeros_like(task_returns)
    termination_count = torch.zeros_like(task_returns)
    action_l2_sum = torch.zeros_like(task_returns)
    done_seen = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    coefficients = operators["auxiliary_coefficients"].to(device=env.device)
    hard_columns = torch.tensor(
        [evidence_names.index(name) for name in operators["hard_safety_names"]],
        dtype=torch.long,
        device=env.device,
    )
    rollout_started = time.perf_counter()
    with (
        torch.inference_mode(),
        evaluation_scope(
            scope_env,
            command,
            domain_scope,
            seed,
            reset_source_name=None,
        ),
    ):
        reset = env.reset()
        observations = reset[0] if isinstance(reset, tuple) else reset
        observations = _as_tensordict(observations, num_envs)
        history = history_factory(observations)
        if history is not None:
            observations = history.decorate_current(observations)
        for _ in range(horizon):
            action = model.action_sample(observations, contexts, deterministic=True).detach()
            returned, _reward, terminated, truncated, _extras = env.step(action)
            returned = _as_tensordict(returned, num_envs)
            done = terminated | truncated
            if history is not None:
                history.advance(observations, returned, done)
            observations = returned
            _g1_qpos_qvel(env, robot, action_term, qpos, qvel)
            task_returns.add_(reward_runtime.evaluate(qpos, qvel))
            if evidence_names:
                transition = observations["transition"]
                values = tuple(transition[name] for name in evidence_names)
                values = tuple(value if value.ndim == 2 else value.unsqueeze(-1) for value in values)
                step_evidence = torch.cat(values, dim=-1).view(task_count, episodes_per_task, evidence_count)
            else:
                step_evidence = evidence_sum[:, :, :0]
            active = step_evidence.ne(0.0)
            evidence_sum.add_(step_evidence)
            evidence_active_count.add_(active)
            auxiliary_cost_sum.add_(torch.sum(step_evidence * coefficients, dim=-1))
            safety_violation_count.add_(active.index_select(-1, hard_columns).any(dim=-1))
            termination_count.add_((terminated & ~truncated).view(task_count, episodes_per_task))
            action_l2_sum.add_(torch.linalg.vector_norm(action.view(task_count, episodes_per_task, -1), dim=-1))
            done_seen.logical_or_(done)
    synchronize()
    rollout_seconds = time.perf_counter() - rollout_started
    if bool(done_seen.any().item()):
        raise RuntimeError("Broad-reward horizon crossed an autoreset; exact terminal physical state is unavailable.")

    serialization_started = time.perf_counter()
    rows = operators["metric_rows"](
        tasks,
        task_returns,
        evidence_sum,
        evidence_active_count,
        auxiliary_cost_sum,
        safety_violation_count,
        termination_count,
        action_l2_sum,
        step_count=horizon,
    )
    serialization_seconds = time.perf_counter() - serialization_started
    return rows, {
        "runtime_setup": runtime_setup_seconds,
        "context_inference": context_seconds,
        "simulation_and_reward": rollout_seconds,
        "scalar_serialization": serialization_seconds,
        "total": runtime_setup_seconds + time.perf_counter() - total_started,
    }


def _g1_cmu_companion_evidence(
    retarget_path: Path,
    simulator_path: Path,
    environment_dependency_identity: Mapping[str, object],
    composition_dependency_identity: Mapping[str, object],
    *,
    environment_semantic_sha256: Callable[[object], str],
    composition_semantic_sha256: Callable[[object], str],
) -> dict[str, dict[str, object]]:
    """Validate and summarize the linked retarget and simulator error layers."""
    retarget_path = retarget_path.expanduser().resolve()
    simulator_path = simulator_path.expanduser().resolve()
    retarget_sha256 = _sha256(retarget_path)
    simulator_sha256 = _sha256(simulator_path)
    retarget = json.loads(retarget_path.read_text())
    simulator = json.loads(simulator_path.read_text())
    if not isinstance(retarget, Mapping) or not isinstance(simulator, Mapping):
        raise TypeError("G1-CMU companion evidence must contain JSON objects.")

    retarget_schema = "forward_backward_phase3g_g1_cmu_composition_evidence_v3"
    simulator_schema = "forward_backward_phase3g_g1_cmu_reference_tracking_evidence_v3"
    if retarget.get("schema") != retarget_schema or simulator.get("schema") != simulator_schema:
        raise ValueError("G1-CMU companion evidence has an unsupported schema.")

    retarget_code = retarget["code_identity"]
    simulator_code = simulator["code_identity"]
    retarget_probe = Path(__file__).with_name("g1_cmu_composition_evidence.py")
    simulator_probe = Path(__file__).with_name("g1_cmu_reference_tracking_evidence.py")
    if retarget_code.get("probe_sha256") != _sha256(retarget_probe):
        raise ValueError("G1-CMU retarget evidence was not produced by the current probe.")
    if simulator_code.get("probe_sha256") != _sha256(simulator_probe):
        raise ValueError("G1-CMU simulator evidence was not produced by the current probe.")
    environment_semantics = environment_semantic_sha256(environment_dependency_identity)
    if environment_semantic_sha256(simulator_code.get("dependency_identity")) != environment_semantics:
        raise ValueError("G1-CMU simulator evidence has a stale environment semantic identity.")
    composition_semantics = composition_semantic_sha256(composition_dependency_identity)
    if composition_semantic_sha256(retarget_code.get("composition_dependency_identity")) != composition_semantics:
        raise ValueError("G1-CMU retarget evidence has a stale source-to-target semantic identity.")
    if composition_semantic_sha256(simulator_code.get("composition_dependency_identity")) != composition_semantics:
        raise ValueError("G1-CMU simulator evidence has a stale source-to-target semantic identity.")

    expected_composition = {"selected": "g1_cmu", "source": "smpl_cmu", "scene_robot": "g1_29dof"}
    retarget_composition = retarget["composition"]
    simulator_composition = simulator["composition"]
    for name, expected in expected_composition.items():
        if retarget_composition.get(name) != expected or simulator_composition.get(name) != expected:
            raise ValueError(f"G1-CMU companion evidence has inconsistent {name!r} composition.")
    builder_identity = retarget_composition.get("frame_builder_identity_sha256")
    if (
        not isinstance(builder_identity, str)
        or len(builder_identity) != 64
        or simulator_composition.get("frame_builder_identity_sha256") != builder_identity
    ):
        raise ValueError("G1-CMU companion evidence has inconsistent trajectory-builder identity.")
    for axis in ("joint_names", "reference_frame_names"):
        if simulator_composition.get(axis) != retarget_composition.get(axis):
            raise ValueError(f"G1-CMU companion evidence has inconsistent {axis!r}.")

    retarget_layer = retarget["error_layers"]["retarget_fit"]
    simulator_retarget_layer = simulator["error_layers"]["retarget_fit"]
    simulator_layer = simulator["error_layers"]["reference_controller_simulator"]
    if retarget_layer.get("status") != "measured":
        raise ValueError("G1-CMU retarget-fit evidence is not measured.")
    if simulator.get("status") != "measured" or simulator_layer.get("status") != "measured":
        raise ValueError("G1-CMU reference-controller simulator evidence is not measured.")
    if simulator_retarget_layer.get("status") != "measured_by_companion_source_composition_probe":
        raise ValueError("G1-CMU simulator evidence does not identify the measured retarget companion.")
    if simulator_retarget_layer.get("frame_builder_identity_sha256") != builder_identity:
        raise ValueError("G1-CMU simulator retarget layer has a different trajectory-builder identity.")
    if simulator_retarget_layer.get("evidence_sha256") != retarget_sha256:
        raise ValueError("G1-CMU simulator retarget layer does not hash the supplied retarget evidence.")
    if simulator_retarget_layer.get("composition_semantic_sha256") != composition_semantics:
        raise ValueError("G1-CMU simulator retarget layer has a stale source-to-target semantic identity.")

    source = retarget["source"]
    selection = simulator["selection"]
    retarget_clip_ids = source.get("selected_clip_ids")
    simulator_clip_ids = selection.get("clip_ids")
    clip_count = _POLICY_CORPORA["g1_cmu"]["clip_count"]
    if source.get("complete_split") is not True or source.get("split") != "test":
        raise ValueError("G1-CMU retarget evidence does not cover the complete source split.")
    if selection.get("split") != "evaluation":
        raise ValueError("G1-CMU simulator evidence does not use the evaluation split.")
    if (
        not isinstance(retarget_clip_ids, list)
        or not all(isinstance(value, str) and value for value in retarget_clip_ids)
        or len(set(retarget_clip_ids)) != len(retarget_clip_ids)
        or source.get("selected_clip_count") != clip_count
        or selection.get("num_clips") != clip_count
        or len(retarget_clip_ids) != clip_count
    ):
        raise ValueError("G1-CMU companion evidence does not cover the complete unique clip set.")
    if simulator_clip_ids != retarget_clip_ids:
        raise ValueError("G1-CMU companion evidence has a different ordered clip set.")
    if selection.get("unexpected_done_rows") != 0:
        raise ValueError("G1-CMU simulator evidence contains unexpected done rows.")

    return {
        "retarget_fit": {
            "path": str(retarget_path),
            "sha256": retarget_sha256,
            "schema": retarget_schema,
            "status": retarget_layer["status"],
            "clip_count": clip_count,
            "composition_semantic_sha256": composition_semantics,
        },
        "reference_controller_simulator": {
            "path": str(simulator_path),
            "sha256": simulator_sha256,
            "schema": simulator_schema,
            "status": simulator_layer["status"],
            "clip_count": clip_count,
            "composition_semantic_sha256": composition_semantics,
            "environment_semantic_sha256": environment_semantics,
        },
    }


def _write_csv(path: Path, rows: list[dict[str, object]], identity: Mapping[str, object]) -> None:
    """Write broad-reward rows with their policy/preset identity."""
    records = [{**identity, **row} for row in rows]
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _run(args: argparse.Namespace) -> dict[str, object]:
    """Construct, strictly load, and evaluate one frozen G1 policy."""
    import rsl_rl
    import torch
    from gpu_ownership import exclusive_physical_gpu_snapshot, validate_same_exclusive_gpu
    from motion_environment_identity import (
        motion_composition_dependency_identity,
        motion_composition_semantic_sha256,
        motion_environment_axes,
        motion_environment_dependency_identity,
        motion_environment_semantic_sha256,
        motion_runner_axes,
    )
    from motion_tracking_records import motion_tracking_metrics_to_dict
    from rsl_rl.models.forward_backward_model import ForwardBackwardModel
    from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertBuffer

    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematics, NewtonKinematicsCfg
    from isaaclab_tasks.core.multi_task.metrics.impl import uniform_assignment_warp
    from isaaclab_tasks.core.multi_task.motion.config.agents import MotionForwardBackwardRunnerCfg
    from isaaclab_tasks.core.multi_task.motion.data.sources import CmuHumEnvSmplClips, LafanG1JoblibClips
    from isaaclab_tasks.core.multi_task.motion.robots.g1.reference import (
        G1LocalBodyPoseFrameBuilder,
        G1PoseFrameBuilder,
    )
    from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
    from isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_expert import forward_backward_expert_buffer
    from isaaclab_tasks.core.multi_task.rl.rsl_rl.forward_backward_tracking import (
        forward_backward_evaluation_scope,
        forward_backward_tracking_evaluator,
    )
    from isaaclab_tasks.utils import resolve_presets

    importers = {"g1_lafan": LafanG1JoblibClips, "g1_cmu": CmuHumEnvSmplClips}
    frame_builders = {"g1_lafan": G1PoseFrameBuilder, "g1_cmu": G1LocalBodyPoseFrameBuilder}
    package_file = getattr(rsl_rl, "__file__", None)
    if not isinstance(package_file, str):
        raise RuntimeError("The imported rsl_rl package has no concrete source root.")
    package_root = Path(package_file).resolve().parent
    runner_selection = motion_environment_axes(args.preset) | motion_runner_axes(args.preset)
    runner_values = resolve_presets(MotionForwardBackwardRunnerCfg(), selected=runner_selection).to_dict()
    native_selection = motion_environment_axes("g1_lafan") | motion_runner_axes("g1_lafan")
    native_values = resolve_presets(MotionForwardBackwardRunnerCfg(), selected=native_selection).to_dict()
    if runner_values["model"] != native_values["model"] or runner_values["obs_groups"] != native_values["obs_groups"]:
        raise RuntimeError("G1-CMU model or observation routes differ from the accepted native checkpoint schema.")
    quality_gate, quality_gate_sha256 = _load_policy_quality_gate()
    quality_protocol_audit, quality_protocol_audit_sha256 = _load_policy_quality_protocol_audit()
    operators = _load_bfm_reward_operators(args.bfm_reward_source_root)
    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != args.checkpoint_sha256:
        raise ValueError("Checkpoint SHA-256 differs from the accepted policy identity.")
    dataset_path = args.reward_dataset.expanduser().resolve()
    dataset_sha256 = _sha256(dataset_path)
    if dataset_sha256 != args.reward_dataset_sha256:
        raise ValueError("Reward-inference dataset SHA-256 differs.")
    dataset = torch.load(dataset_path, map_location=args.device, weights_only=True)
    reward_model_path = args.reward_model_entrypoint.expanduser().resolve()
    reward_model_sha256 = _sha256(reward_model_path)
    reward_model_source_identity = _mujoco_model_source_identity(reward_model_path)
    if (
        reward_model_sha256 != args.reward_model_entrypoint_sha256
        or dataset["reward_model_sha256"] != reward_model_sha256
    ):
        raise ValueError("Reward model and reward-inference dataset identities differ.")
    if reward_model_source_identity["bundle_sha256"] != args.reward_model_bundle_sha256:
        raise ValueError("Reward model source bundle SHA-256 differs.")
    if dataset["reward_tasks"] != operators["tasks"] or dataset["reward_tasks"] is None:
        raise ValueError("Reward dataset and evaluator task orders differ.")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Policy-quality output already exists: {output}.")

    cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes(args.preset))
    cfg.sim.device = args.device
    table_cfg = cfg.commands.motion.task_table
    table_cfg.source_artifact_root = str(args.source_artifact_root.expanduser().resolve())
    table_cfg.reference_artifact_root = str(args.reference_artifact_root.expanduser().resolve())
    table_cfg.motion_split = _policy_motion_split(args.preset)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.environment_seed
    dependency_identity = motion_environment_dependency_identity(
        preset=args.preset,
        cfg=cfg,
        importer_type=importers[args.preset],
        frame_builder_type=frame_builders[args.preset],
        reference_artifact_root=table_cfg.reference_artifact_root,
    )
    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=cfg))
    try:
        corpus = _POLICY_CORPORA[args.preset]
        table = env.unwrapped.command_manager.get_term("motion").table
        if len(table.clip_ids) != corpus["clip_count"] or table.clip_index.total_frames != corpus["frame_count"]:
            raise RuntimeError("Policy-quality motion corpus differs from its frozen profile.")
        composition_dependency_identity = motion_composition_dependency_identity(
            preset=args.preset,
            cfg=cfg,
            importer_type=importers[args.preset],
            frame_builder_type=frame_builders[args.preset],
            frame_builder_identity_sha256=table.frame_builder_identity_sha256,
            reference_artifact_root=table_cfg.reference_artifact_root,
        )
        observations, _reset_info = env.reset()
        history_factory = _evaluation_history_factory(runner_values["replay"])
        construction_history = history_factory(observations)
        if construction_history is not None:
            observations = construction_history.decorate_current(observations)
        model = ForwardBackwardModel.from_config(
            observations,
            runner_values["obs_groups"],
            env.num_actions,
            runner_values["model"],
        ).to(env.device)
        saved = torch.load(checkpoint, map_location=env.device, weights_only=True)
        if set(saved) != {"model_state_dict"}:
            raise ValueError("Accepted policy checkpoint must contain exactly model_state_dict.")
        loaded = model.load_state_dict(saved["model_state_dict"], strict=True, assign=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise RuntimeError("Strict checkpoint load returned incompatible state keys.")
        model.eval()

        companion_evidence: dict[str, object] = {}
        if args.preset == "g1_cmu":
            if args.retarget_evidence is None or args.simulator_evidence is None:
                raise ValueError("G1-CMU policy evidence requires separate retarget and simulator evidence.")
            companion_evidence = _g1_cmu_companion_evidence(
                args.retarget_evidence,
                args.simulator_evidence,
                dependency_identity,
                composition_dependency_identity,
                environment_semantic_sha256=motion_environment_semantic_sha256,
                composition_semantic_sha256=motion_composition_semantic_sha256,
            )

        before_tracking = exclusive_physical_gpu_snapshot(args.device)
        expert_cfg = runner_values["expert"]
        expert = forward_backward_expert_buffer(
            env,
            model.observation_schema,
            env.device,
            source_bind=expert_cfg["source_bind"],
            sampling_mode=expert_cfg["sampling_mode"],
            sampling_step_seconds=expert_cfg["sampling_step_seconds"],
            target_projection=expert_cfg["target_projection"],
            target_projection_binds=tuple(expert_cfg["target_projection_binds"]),
            window_lengths=tuple(expert_cfg["window_lengths"]),
            seed=expert_cfg["seed"],
        )
        command = env.unwrapped.command_manager.get_term("motion")
        with forward_backward_evaluation_scope(
            env,
            command,
            command.payload.evaluation_scope,
            args.evaluation_seed,
            reset_source_name="reference",
        ):
            lifecycle = runner_values["lifecycle_extension"]
            tracking = forward_backward_tracking_evaluator(
                model,
                env,
                expert,
                expert.clip_ids,
                command=command,
                history_factory=history_factory,
                sequence_start_rows=table.clip_start_rows,
                projections=tuple(lifecycle["projections"]),
                context_window_length=lifecycle["context_window_length"],
                include_reset_frame=lifecycle["include_reset_frame"],
                allow_horizon_truncation=lifecycle["allow_horizon_truncation"],
                shuffle_assignments=lifecycle["shuffle_assignments"],
                assignment_rng=random.Random(args.evaluation_seed),
            )
        after_tracking = exclusive_physical_gpu_snapshot(args.device)

        reward_runtime_started = time.perf_counter()
        reward_kinematics = NewtonKinematics(NewtonKinematicsCfg(mjcf_path=str(reward_model_path), device=env.device))
        reward_runtime = operators["runtime_type"](
            reward_kinematics,
            env.unwrapped.action_manager.get_term("joint_position").joint_names,
            args.episodes_per_task,
        )
        if torch.device(env.device).type == "cuda":
            torch.cuda.synchronize(env.device)
        reward_runtime_seconds = time.perf_counter() - reward_runtime_started
        reward_rows, broad_reward_timing = _broad_reward_rollout(
            model=model,
            env=env.unwrapped,
            scope_env=env,
            evaluation_scope=forward_backward_evaluation_scope,
            command=command,
            domain_scope=command.payload.evaluation_scope,
            history_factory=history_factory,
            dataset=dataset,
            reward_runtime=reward_runtime,
            runtime_setup_seconds=reward_runtime_seconds,
            operators=operators,
            auxiliary_evidence_names=tuple(runner_values["replay"]["auxiliary_evidence_names"]),
            episodes_per_task=args.episodes_per_task,
            horizon=args.reward_horizon,
            batch_size=args.inference_batch_size,
            seed=args.evaluation_seed,
        )
        after_broad_reward = exclusive_physical_gpu_snapshot(args.device)
        physical_gpu_uuid = validate_same_exclusive_gpu(before_tracking, after_tracking, after_broad_reward)

        tracking_metrics = motion_tracking_metrics_to_dict(tracking)
        tracking_emd = [row["emd"] for row in tracking_metrics.values()]
        tracking_payload = {
            "clip_ids": tracking.sequence_ids,
            "metrics": tracking_metrics,
            "emd": tracking_emd,
            "emd_statistics": _statistics(tracking_emd),
            "obs_state_emd_statistics": _statistics([row["obs_state_emd"] for row in tracking_metrics.values()]),
            "coverage_statistics": _statistics([row["coverage_fraction"] for row in tracking_metrics.values()]),
            "duration_seconds": tracking.duration_seconds,
        }
        row_identity = {
            "preset": args.preset,
            "training_seed": 4728,
            "evaluation_seed": args.evaluation_seed,
            "checkpoint_transition": 9_600_000,
        }
        by_metric: dict[str, list[float]] = {}
        for row in reward_rows:
            by_metric.setdefault(str(row["metric_name"]), []).append(float(row["metric_value"]))
        broad_summary = {name: _statistics(values) for name, values in sorted(by_metric.items())}
        required_reward_metrics = (
            "return",
            "safety_violation_rate",
            "termination_rate",
            "auxiliary_cost",
            "action_l2",
        )
        if any(name not in broad_summary for name in required_reward_metrics):
            raise ValueError("Broad-reward evidence is missing a frozen quality metric.")
        protocol = {
            "preset": args.preset,
            "motion_split": table_cfg.motion_split,
            "clip_count": len(table.clip_ids),
            "frame_count": table.clip_index.total_frames,
            "evaluation_seed": args.evaluation_seed,
            "domain_randomization": all(
                getattr(cfg.events, name) is not None for name in ("robot_material", "body_mass", "torso_com", "push")
            ),
            "observation_noise": bool(cfg.observations.joint_position.enable_corruption),
            "reward_task_count": len(operators["tasks"]),
            "episodes_per_task": args.episodes_per_task,
            "reward_horizon": args.reward_horizon,
            "tracking_coverage_fraction": tracking_payload["coverage_statistics"]["minimum"],
        }
        tracking_decision_input = {
            "emd_mean": tracking_payload["emd_statistics"]["mean"],
            "obs_state_emd_mean": tracking_payload["obs_state_emd_statistics"]["mean"],
            "coverage_fraction": tracking_payload["coverage_statistics"]["minimum"],
        }
        broad_decision_input = {f"{name}_mean": broad_summary[name]["mean"] for name in required_reward_metrics}
        if args.preset == "g1_lafan":
            decision = _native_quality_decision(
                quality_gate,
                checkpoint={
                    "transition": 9_600_000,
                    "training_seed": 4728,
                    "sha256": checkpoint_sha256,
                },
                protocol=protocol,
                tracking=tracking_decision_input,
                broad_reward=broad_decision_input,
            )
        else:
            decision = _cross_source_quality_decision(
                quality_gate,
                checkpoint={
                    "transition": 9_600_000,
                    "training_seed": 4728,
                    "sha256": checkpoint_sha256,
                },
                protocol=protocol,
            )

        output.mkdir(parents=True)
        tracking_path = output / "tracking.json"
        broad_reward_path = output / "broad_reward.csv"
        tracking_path.write_text(json.dumps(tracking_payload, indent=2, sort_keys=True) + "\n")
        _write_csv(broad_reward_path, reward_rows, row_identity)

        code_identity = {
            "evaluator_sha256": _source_sha256(_run),
            "tracking_evaluator_sha256": _source_sha256(forward_backward_tracking_evaluator),
            "emd_transport_kernel_sha256": _source_sha256(uniform_assignment_warp),
            "expert_provider_sha256": _source_sha256(forward_backward_expert_buffer),
            "expert_buffer_sha256": _source_sha256(ForwardBackwardExpertBuffer),
            "model_sha256": _source_sha256(ForwardBackwardModel),
            "learner_code_bundle_sha256": _python_package_bundle_sha256(package_root),
            "python_source_identity_sha256": _sha256(Path(__file__).with_name("python_source_identity.py")),
            "reward_kinematics_sha256": _source_sha256(NewtonKinematics),
            "reward_context_policy_sha256": _source_sha256(_BFMRewardContextPolicy),
            "bfm_reward_runtime_sha256": operators["runtime_sha256"],
            "gpu_ownership_sha256": _source_sha256(exclusive_physical_gpu_snapshot),
            "policy_quality_gate_sha256": quality_gate_sha256,
            "policy_quality_protocol_audit_sha256": quality_protocol_audit_sha256,
            "dependency_identity": dependency_identity,
            "composition_dependency_identity": composition_dependency_identity,
            "bfm_reward_sources": operators["code_identity"],
        }
        model_contract = {
            "observation_schema_sha256": model.observation_schema.schema_hash,
            "model_config_sha256": _canonical_sha256(runner_values["model"]),
            "observation_routes_sha256": _canonical_sha256(runner_values["obs_groups"]),
            "state_tensor_count": len(model.state_dict()),
            "parameter_count": sum(value.numel() for value in model.parameters()),
            "strict_load": True,
            "missing_keys": [],
            "unexpected_keys": [],
        }
        return {
            "schema": "forward_backward_phase3g_g1_policy_quality_evidence_v6",
            "status": "measured",
            "evaluation_kind": "zero_shot_cross_source" if args.preset == "g1_cmu" else "native_source",
            "preset": args.preset,
            "training_domain": "g1_lafan",
            "code_identity": code_identity,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_sha256,
                "transition": 9_600_000,
                "training_seed": 4728,
                "profile": "residual_6x1024",
            },
            "protocol_audit": {
                "schema": quality_protocol_audit["schema"],
                "sha256": quality_protocol_audit_sha256,
                "broad_reward_role": "diagnostic_only",
            },
            "model_contract": model_contract,
            "environment": {
                "source_artifact_root": table_cfg.source_artifact_root,
                "reference_artifact_root": table_cfg.reference_artifact_root,
                "motion_split": table_cfg.motion_split,
                "table_identity": table.cache_identity,
                "frame_builder_identity_sha256": table.frame_builder_identity_sha256,
                "clip_count": len(table.clip_ids),
                "frame_count": table.clip_index.total_frames,
                "num_envs": env.num_envs,
                "environment_seed": args.environment_seed,
                "evaluation_seed": args.evaluation_seed,
            },
            "physical_gpu_ownership": {
                "physical_gpu_uuid": physical_gpu_uuid,
                "before_tracking": before_tracking,
                "after_tracking": after_tracking,
                "after_broad_reward": after_broad_reward,
            },
            "policy_quality": {
                "tracking": {
                    "artifact": tracking_path.name,
                    "sha256": _sha256(tracking_path),
                    "clip_count": len(tracking.sequence_ids),
                    "emd": tracking_payload["emd_statistics"],
                    "obs_state_emd": tracking_payload["obs_state_emd_statistics"],
                    "coverage": tracking_payload["coverage_statistics"],
                    "duration_seconds": tracking.duration_seconds,
                },
                "broad_reward": {
                    "artifact": broad_reward_path.name,
                    "sha256": _sha256(broad_reward_path),
                    "task_count": len(operators["tasks"]),
                    "episodes_per_task": args.episodes_per_task,
                    "horizon": args.reward_horizon,
                    "row_count": len(reward_rows),
                    "duration_seconds": broad_reward_timing["total"],
                    "stage_durations_seconds": broad_reward_timing,
                    "metrics": broad_summary,
                    "reward_tasks_sha256": operators["tasks_sha256"],
                    "inference_dataset_path": str(dataset_path),
                    "inference_dataset_sha256": dataset_sha256,
                    "reward_model_entrypoint_path": str(reward_model_path),
                    "reward_model_entrypoint_sha256": reward_model_sha256,
                    "reward_model_source_identity": reward_model_source_identity,
                    "comparison_identity": None,
                },
            },
            "decision": decision,
            "error_layer_separation": {
                "policy_quality": "measured_here",
                "retarget_and_simulator_errors_are_not_policy_metrics": True,
                "companion_evidence": companion_evidence,
            },
        }
    finally:
        env.close()


def main() -> None:
    """Parse one frozen evaluation request and atomically publish its manifest."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("g1_lafan", "g1_cmu"), required=True)
    parser.add_argument("--source_artifact_root", type=Path, required=True)
    parser.add_argument("--reference_artifact_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint_sha256", required=True)
    parser.add_argument("--reward_dataset", type=Path, required=True)
    parser.add_argument("--reward_dataset_sha256", required=True)
    parser.add_argument("--reward_model_entrypoint", type=Path, required=True)
    parser.add_argument("--reward_model_entrypoint_sha256", required=True)
    parser.add_argument("--reward_model_bundle_sha256", required=True)
    parser.add_argument("--bfm_reward_source_root", type=Path, required=True)
    parser.add_argument("--retarget_evidence", type=Path)
    parser.add_argument("--simulator_evidence", type=Path)
    parser.add_argument("--num_envs", type=int, default=380)
    parser.add_argument("--episodes_per_task", type=int, default=10)
    parser.add_argument("--reward_horizon", type=int, default=500)
    parser.add_argument("--inference_batch_size", type=int, default=1024)
    parser.add_argument("--environment_seed", type=int, default=4728)
    parser.add_argument("--evaluation_seed", type=int, default=4728)
    parser.add_argument("--output_dir", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs != 38 * args.episodes_per_task:
        raise ValueError("num_envs must equal 38 broad tasks times episodes_per_task.")
    if min(args.episodes_per_task, args.reward_horizon, args.inference_batch_size) < 1:
        raise ValueError("Policy-quality evaluation counts must be positive.")
    if args.environment_seed != args.evaluation_seed:
        raise ValueError("Policy-quality evaluation uses one frozen seed for environment and evaluation transactions.")

    launcher = AppLauncher(args)
    simulation_app = launcher.app
    try:
        report = _run(args)
        output = args.output_dir.expanduser().resolve()
        temporary = output / "manifest.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(output / "manifest.json")
        print(json.dumps(report, indent=2, sort_keys=True))
        decision = report["decision"]
        if decision["threshold_applied"]:
            if decision["passed"] is not True:
                raise RuntimeError("Native G1-LAFAN policy quality failed its frozen non-inferiority gate.")
        elif decision["measurement_complete"] is not True:
            raise RuntimeError("Zero-shot G1-CMU policy quality measurement is incomplete.")
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
