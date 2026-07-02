# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independently consume native G1 policy-quality evidence from durable files."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

_TRACKING_MODULE = "isaaclab_tasks.core.multi_task.motion.tracking"
_EMD_TRANSPORT_MODULE = "isaaclab_tasks.core.multi_task.motion.impl.uniform_emd_warp"
_EXPERT_PROVIDER_MODULE = "isaaclab_tasks.core.multi_task.motion.rsl_rl"
_EXPERT_BUFFER_MODULE = "rsl_rl.storage.forward_backward_expert"
_MODEL_MODULE = "rsl_rl.models.forward_backward_model"
_REWARD_KINEMATICS_MODULE = "isaaclab_tasks.core.multi_task.kinematics.newton_kinematics"
_BFM_REWARD_SOURCE_IDENTITY = Path(__file__).with_name("fixtures") / "bfm_reward_source_identity_v1.json"
_LOCAL_POLICY_CODE_FIELDS = (
    "evaluator_sha256",
    "tracking_evaluator_sha256",
    "emd_transport_kernel_sha256",
    "expert_provider_sha256",
    "expert_buffer_sha256",
    "model_sha256",
    "learner_code_bundle_sha256",
    "python_source_identity_sha256",
    "reward_kinematics_sha256",
    "reward_context_policy_sha256",
    "bfm_reward_runtime_sha256",
    "gpu_ownership_sha256",
    "policy_quality_gate_sha256",
    "policy_quality_protocol_audit_sha256",
)
_POLICY_CODE_IDENTITY_FIELDS = frozenset(
    (
        *_LOCAL_POLICY_CODE_FIELDS,
        "dependency_identity",
        "composition_dependency_identity",
        "bfm_reward_sources",
    )
)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one regular, non-symbolic artifact."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Policy-quality evidence must be a regular non-symbolic file: {path}.")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_object(path: Path) -> dict[str, object]:
    """Load one JSON object from a required evidence file."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Policy-quality JSON must contain one object: {path}.")
    return value


def _statistics(values: Sequence[float]) -> dict[str, float | int]:
    """Recompute the producer's finite scalar summary from serialized values."""
    import torch

    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
        raise ValueError("Policy-quality evidence statistics require finite nonempty values.")
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


def _source_module(path: Path):
    """Load one colocated evidence module without a package assumption."""
    spec = importlib.util.spec_from_file_location(f"forward_backward_phase3_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Phase 3 evidence module: {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module_source_path(module_name: str) -> Path:
    """Return one concrete Python source path without importing its implementation."""
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(f"Policy-quality source module is missing: {module_name!r}.")
    path = Path(spec.origin).expanduser()
    if path.suffix != ".py":
        raise ValueError(f"Policy-quality source module is not Python source: {module_name!r}.")
    return path


def _current_policy_code_identity(gate_file: Path) -> dict[str, str]:
    """Recompute every producer code owner available in this checkout."""
    evaluator_path = Path(__file__).with_name("g1_policy_quality_evidence.py")
    source_identity_path = Path(__file__).with_name("python_source_identity.py")
    audit_path = Path(__file__).with_name("fixtures") / "g1_lafan_policy_quality_protocol_audit_v1.json"
    learner_package_root = _module_source_path("rsl_rl").parent
    source_identity = _source_module(source_identity_path)
    return {
        "evaluator_sha256": _sha256(evaluator_path),
        "tracking_evaluator_sha256": _sha256(_module_source_path(_TRACKING_MODULE)),
        "emd_transport_kernel_sha256": _sha256(_module_source_path(_EMD_TRANSPORT_MODULE)),
        "expert_provider_sha256": _sha256(_module_source_path(_EXPERT_PROVIDER_MODULE)),
        "expert_buffer_sha256": _sha256(_module_source_path(_EXPERT_BUFFER_MODULE)),
        "model_sha256": _sha256(_module_source_path(_MODEL_MODULE)),
        "learner_code_bundle_sha256": source_identity.python_package_bundle_sha256(learner_package_root),
        "python_source_identity_sha256": _sha256(source_identity_path),
        "reward_kinematics_sha256": _sha256(_module_source_path(_REWARD_KINEMATICS_MODULE)),
        "reward_context_policy_sha256": _sha256(evaluator_path),
        "bfm_reward_runtime_sha256": _sha256(Path(__file__).with_name("bfm_reward_runtime.py")),
        "gpu_ownership_sha256": _sha256(Path(__file__).with_name("gpu_ownership.py")),
        "policy_quality_gate_sha256": _sha256(gate_file),
        "policy_quality_protocol_audit_sha256": _sha256(audit_path),
    }


def _validate_local_policy_code_identity(code_identity: Mapping[str, object], gate_file: Path) -> None:
    """Require every locally recomputable producer code digest to match."""
    for name, digest in _current_policy_code_identity(gate_file).items():
        if code_identity.get(name) != digest:
            label = name.removesuffix("_sha256").replace("_", " ")
            raise ValueError(f"Policy-quality {label} bytes differ from the producer identity.")


def _validate_external_reward_source_identity(value: object) -> dict[str, object]:
    """Bind BFM reward sources to one frozen repository revision and tree."""
    if not isinstance(value, Mapping) or not value:
        raise TypeError("Policy-quality external BFM reward source identity must be a nonempty mapping.")
    declared: dict[str, str] = {}
    for name, digest in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Policy-quality external BFM reward source names must be nonempty strings.")
        path = PurePosixPath(name)
        if path.is_absolute() or path.suffix != ".py" or ".." in path.parts or path.as_posix() != name:
            raise ValueError(f"Policy-quality external BFM reward source path is invalid: {name!r}.")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Policy-quality external BFM reward source {name!r} has an invalid SHA-256 digest.")
        declared[name] = digest
    declared = dict(sorted(declared.items()))

    contract = _json_object(_BFM_REWARD_SOURCE_IDENTITY)
    if set(contract) != {"schema", "repository", "files"} or contract.get("schema") != (
        "forward_backward_phase3_bfm_reward_source_identity_v1"
    ):
        raise ValueError("Policy-quality BFM reward source contract has an invalid schema or field set.")
    repository = contract.get("repository")
    files = contract.get("files")
    if not isinstance(repository, Mapping) or set(repository) != {
        "url",
        "upstream_url",
        "revision",
        "tree",
        "source_root",
    }:
        raise ValueError("Policy-quality BFM reward source contract has an invalid repository identity.")
    revision = repository.get("revision")
    tree = repository.get("tree")
    if (
        repository.get("source_root") != "."
        or not isinstance(repository.get("url"), str)
        or not isinstance(repository.get("upstream_url"), str)
        or not isinstance(revision, str)
        or len(revision) != 40
        or not isinstance(tree, str)
        or len(tree) != 40
        or any(character not in "0123456789abcdef" for character in revision + tree)
    ):
        raise ValueError("Policy-quality BFM reward source contract has malformed repository provenance.")
    if not isinstance(files, Mapping) or dict(sorted(files.items())) != declared:
        raise ValueError("Policy-quality external BFM reward sources differ from the frozen repository identity.")
    return {
        "contract_sha256": _sha256(_BFM_REWARD_SOURCE_IDENTITY),
        "repository": dict(repository),
        "files": declared,
    }


def _tracking_statistics(
    tracking: Mapping[str, object], expected_clip_count: int = 862
) -> dict[str, dict[str, float | int]]:
    """Recompute tracking statistics in the declared clip order."""
    clip_ids = tracking.get("clip_ids")
    metrics = tracking.get("metrics")
    emd = tracking.get("emd")
    if not isinstance(clip_ids, list) or not isinstance(metrics, Mapping) or not isinstance(emd, list):
        raise TypeError("Tracking evidence must declare clip_ids, metrics, and emd.")
    if (
        len(clip_ids) != expected_clip_count
        or len(set(clip_ids)) != expected_clip_count
        or len(metrics) != expected_clip_count
        or len(emd) != expected_clip_count
    ):
        raise ValueError(f"Tracking evidence must cover exactly {expected_clip_count} unique clips.")
    try:
        emd_values = [float(metrics[clip_id]["emd"]) for clip_id in clip_ids]
        obs_state_values = [float(metrics[clip_id]["obs_state_emd"]) for clip_id in clip_ids]
        coverage_values = [float(metrics[clip_id]["coverage_fraction"]) for clip_id in clip_ids]
    except (KeyError, TypeError) as error:
        raise ValueError("Tracking evidence is missing a declared clip metric.") from error
    if emd_values != emd:
        raise ValueError("Tracking EMD vector differs from the declared clip metric order.")
    return {
        "emd": _statistics(emd_values),
        "obs_state_emd": _statistics(obs_state_values),
        "coverage": _statistics(coverage_values),
    }


def _broad_reward_statistics(
    path: Path, expected_rows: int, preset: str = "g1_lafan"
) -> dict[str, dict[str, float | int]]:
    """Recompute every broad-reward metric from the serialized episode rows."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_rows:
        raise ValueError(f"Broad-reward evidence has {len(rows)} rows, expected {expected_rows}.")
    by_metric: dict[str, list[float]] = {}
    for row in rows:
        try:
            if (
                row["preset"] != preset
                or int(row["training_seed"]) != 4728
                or int(row["evaluation_seed"]) != 4728
                or int(row["checkpoint_transition"]) != 9_600_000
            ):
                raise ValueError("Broad-reward row identity differs from the frozen policy.")
            by_metric.setdefault(row["metric_name"], []).append(float(row["metric_value"]))
        except KeyError as error:
            raise ValueError(f"Broad-reward evidence is missing column {error.args[0]!r}.") from error
    if any(len(values) != 380 for values in by_metric.values()):
        raise ValueError("Every broad-reward metric must contain 38 tasks times 10 episodes.")
    return {name: _statistics(values) for name, values in sorted(by_metric.items())}


def _validate_gpu_ownership(value: object) -> str:
    """Require one exclusive physical GPU owner across both evaluation stages."""
    if not isinstance(value, Mapping):
        raise TypeError("Policy-quality evidence must declare physical GPU ownership.")
    uuid = value.get("physical_gpu_uuid")
    if not isinstance(uuid, str) or not uuid.startswith("GPU-"):
        raise ValueError("Policy-quality evidence has an invalid physical GPU UUID.")
    for name in ("before_tracking", "after_tracking", "after_broad_reward"):
        snapshot = value.get(name)
        if not isinstance(snapshot, Mapping):
            raise TypeError(f"Policy-quality evidence is missing GPU snapshot {name!r}.")
        if snapshot.get("exclusive") is not True or snapshot.get("physical_gpu_uuid") != uuid:
            raise ValueError("Policy-quality evaluation did not retain one exclusive physical GPU.")
        owner = snapshot.get("owner_pid")
        if snapshot.get("competing_compute_pids") != [] or snapshot.get("compute_pids") != [owner]:
            raise ValueError("Policy-quality GPU snapshot contains a competing or ambiguous owner.")
    return uuid


def _companion_identity(value: object) -> dict[str, dict[str, object]]:
    """Return companion evidence identity without machine-local path provenance."""
    if not isinstance(value, Mapping):
        raise TypeError("Cross-source companion evidence must be a mapping.")
    expected_layers = {"retarget_fit", "reference_controller_simulator"}
    if set(value) != expected_layers:
        raise ValueError("Cross-source companion evidence has an unexpected layer set.")
    identity: dict[str, dict[str, object]] = {}
    for name in sorted(expected_layers):
        layer = value[name]
        if not isinstance(layer, Mapping):
            raise TypeError(f"Cross-source companion layer {name!r} must be a mapping.")
        path = layer.get("path")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError(f"Cross-source companion layer {name!r} must retain absolute path provenance.")
        identity[name] = {key: nested for key, nested in layer.items() if key != "path"}
    return identity


def _validate_policy_python_sources(
    owner: str,
    sources: object,
    identity_path: Path,
) -> None:
    """Require every stored module digest to match current source bytes."""
    if not isinstance(sources, Mapping) or not sources:
        raise TypeError(f"Policy {owner} identity must declare nonempty Python sources.")
    if not all(isinstance(module_name, str) and module_name for module_name in sources):
        raise ValueError(f"Policy {owner} Python source names must be nonempty strings.")
    for module_name in sorted(sources):
        expected = sources[module_name]
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError(f"Policy {owner} Python source {module_name!r} has an invalid SHA-256 digest.")
        path = identity_path if module_name == "motion_environment_identity" else _module_source_path(module_name)
        if _sha256(path) != expected:
            raise ValueError(f"Policy {owner} Python source {module_name!r} bytes differ from current source.")


def _policy_dependency_identities(
    code_identity: Mapping[str, object],
    identity_path: Path,
    preset: str,
):
    """Validate the closed environment and composition identities in one policy artifact."""
    environment = code_identity.get("dependency_identity")
    composition = code_identity.get("composition_dependency_identity")
    if not isinstance(environment, Mapping) or not isinstance(composition, Mapping):
        raise TypeError("Policy evidence must declare environment and composition dependency identities.")
    if environment.get("preset") != preset or composition.get("preset") != preset:
        raise ValueError("Policy dependency identities differ from the evaluated preset.")
    for name, value in (("environment", environment), ("composition", composition)):
        _validate_policy_python_sources(name, value.get("python_sources"), identity_path)
    identity_module = _source_module(identity_path)
    identity_module.motion_environment_semantic_sha256(environment)
    identity_module.motion_composition_semantic_sha256(composition)
    return environment, composition, identity_module


def _validate_policy_code_identity(code_identity: Mapping[str, object], gate_file: Path, preset: str):
    """Validate the closed local code graph and declared external provenance."""
    actual_fields = frozenset(code_identity)
    if actual_fields != _POLICY_CODE_IDENTITY_FIELDS:
        missing = sorted(_POLICY_CODE_IDENTITY_FIELDS - actual_fields)
        unexpected = sorted(actual_fields - _POLICY_CODE_IDENTITY_FIELDS)
        raise ValueError(f"Policy-quality code-identity fields differ: missing={missing}, unexpected={unexpected}.")
    _validate_local_policy_code_identity(code_identity, gate_file)
    _validate_external_reward_source_identity(code_identity["bfm_reward_sources"])
    identity_path = Path(__file__).with_name("motion_environment_identity.py")
    return _policy_dependency_identities(code_identity, identity_path, preset)


def _validate_protocol_audit(manifest: Mapping[str, object], code_identity: Mapping[str, object]) -> None:
    """Require one frozen protocol audit in both manifest and code identity."""
    audit_path = Path(__file__).with_name("fixtures") / "g1_lafan_policy_quality_protocol_audit_v1.json"
    audit = _json_object(audit_path)
    digest = _sha256(audit_path)
    declaration = manifest.get("protocol_audit")
    if (
        not isinstance(declaration, Mapping)
        or declaration.get("schema") != audit.get("schema")
        or declaration.get("sha256") != digest
        or declaration.get("broad_reward_role") != "diagnostic_only"
        or code_identity.get("policy_quality_protocol_audit_sha256") != digest
    ):
        raise ValueError("Policy-quality artifact does not close the frozen protocol audit.")


def _validate_broad_reward_decision_boundary(
    decision: Mapping[str, object],
    comparison_identity: Mapping[str, object] | None,
) -> dict[str, object]:
    """Reject any authoritative broad-reward claim without the future paired protocol."""
    diagnostics = decision.get("diagnostics")
    diagnostic = diagnostics.get("broad_reward") if isinstance(diagnostics, Mapping) else None
    if not isinstance(diagnostic, dict):
        raise TypeError("Policy-quality decision must retain a broad-reward diagnostic.")
    if "passed" in diagnostic or diagnostic.get("status") == "passed":
        raise ValueError("Broad reward cannot be called passed without a paired realized-corpus protocol.")
    if diagnostic.get("authoritative") is not False:
        raise ValueError("Broad reward cannot be authoritative under the diagnostic-only Phase 3 protocol.")
    closure = diagnostic.get("identity_closure")
    if not isinstance(closure, Mapping):
        raise TypeError("Broad-reward diagnostic must retain realized-corpus identity closure.")
    if comparison_identity is None and closure.get("identity_closed") is not False:
        raise ValueError("Broad-reward identity cannot be closed without comparison identities.")
    return diagnostic


def validate_native_policy_quality_artifact(
    artifact_dir: str | Path,
    gate_path: str | Path,
) -> dict[str, object]:
    """Recompute and validate one native G1 policy-quality artifact.

    Args:
        artifact_dir: Directory containing ``manifest.json``, ``tracking.json``, and
            ``broad_reward.csv``.
        gate_path: Frozen native non-inferiority gate JSON.

    Returns:
        Compact independent validation receipt.
    """
    artifact = Path(artifact_dir).expanduser().resolve()
    gate_file = Path(gate_path).expanduser().resolve()
    manifest_path = artifact / "manifest.json"
    tracking_path = artifact / "tracking.json"
    broad_reward_path = artifact / "broad_reward.csv"
    manifest = _json_object(manifest_path)
    tracking = _json_object(tracking_path)
    gate = _json_object(gate_file)

    if (
        manifest.get("schema") != "forward_backward_phase3g_g1_policy_quality_evidence_v6"
        or manifest.get("status") != "measured"
        or manifest.get("evaluation_kind") != "native_source"
        or manifest.get("preset") != "g1_lafan"
        or manifest.get("training_domain") != "g1_lafan"
    ):
        raise ValueError("Policy-quality manifest is not the native G1-LAFAN v6 evidence contract.")
    checkpoint = manifest.get("checkpoint")
    expected_checkpoint = gate.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or not isinstance(expected_checkpoint, Mapping):
        raise TypeError("Policy-quality manifest and gate must declare checkpoint mappings.")
    if {name: checkpoint.get(name) for name in expected_checkpoint} != expected_checkpoint:
        raise ValueError("Policy-quality artifact uses a checkpoint different from the frozen gate.")
    if checkpoint.get("profile") != "residual_6x1024":
        raise ValueError("Policy-quality artifact uses an unexpected model profile.")

    code_identity = manifest.get("code_identity")
    if not isinstance(code_identity, Mapping):
        raise TypeError("Policy-quality manifest must declare code identity.")
    _validate_policy_code_identity(code_identity, gate_file, "g1_lafan")
    _validate_protocol_audit(manifest, code_identity)

    policy_quality = manifest.get("policy_quality")
    if not isinstance(policy_quality, Mapping):
        raise TypeError("Policy-quality manifest must declare tracking and broad-reward evidence.")
    tracking_manifest = policy_quality.get("tracking")
    broad_manifest = policy_quality.get("broad_reward")
    if not isinstance(tracking_manifest, Mapping) or not isinstance(broad_manifest, Mapping):
        raise TypeError("Policy-quality manifest has malformed evidence declarations.")
    if tracking_manifest.get("artifact") != tracking_path.name or tracking_manifest.get("sha256") != _sha256(
        tracking_path
    ):
        raise ValueError("Tracking artifact bytes differ from the manifest.")
    if broad_manifest.get("artifact") != broad_reward_path.name or broad_manifest.get("sha256") != _sha256(
        broad_reward_path
    ):
        raise ValueError("Broad-reward artifact bytes differ from the manifest.")

    tracking_statistics = _tracking_statistics(tracking)
    if (
        tracking_statistics["emd"] != tracking.get("emd_statistics")
        or tracking_statistics["obs_state_emd"] != tracking.get("obs_state_emd_statistics")
        or tracking_statistics["coverage"] != tracking.get("coverage_statistics")
        or tracking_statistics["emd"] != tracking_manifest.get("emd")
        or tracking_statistics["obs_state_emd"] != tracking_manifest.get("obs_state_emd")
        or tracking_statistics["coverage"] != tracking_manifest.get("coverage")
    ):
        raise ValueError("Tracking summaries differ from the serialized per-clip evidence.")
    row_count = broad_manifest.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int):
        raise TypeError("Broad-reward row_count must be an integer.")
    broad_statistics = _broad_reward_statistics(broad_reward_path, row_count)
    if broad_statistics != broad_manifest.get("metrics"):
        raise ValueError("Broad-reward summaries differ from the serialized episode rows.")
    if (
        broad_manifest.get("task_count") != 38
        or broad_manifest.get("episodes_per_task") != 10
        or broad_manifest.get("horizon") != 500
    ):
        raise ValueError("Broad-reward protocol differs from the frozen 38-by-10, 500-step evaluation.")

    decision = manifest.get("decision")
    if not isinstance(decision, Mapping):
        raise TypeError("Policy-quality manifest must declare a decision.")
    protocol_result = decision.get("protocol")
    if not isinstance(protocol_result, Mapping) or protocol_result.get("passed") is not True:
        raise ValueError("Policy-quality decision did not accept its frozen protocol.")
    protocol = protocol_result.get("actual")
    if protocol != gate.get("protocol"):
        raise ValueError("Policy-quality protocol differs from the frozen gate.")
    model_contract = manifest.get("model_contract")
    if (
        not isinstance(model_contract, Mapping)
        or model_contract.get("strict_load") is not True
        or model_contract.get("missing_keys") != []
        or model_contract.get("unexpected_keys") != []
    ):
        raise ValueError("Policy-quality checkpoint was not loaded strictly.")
    physical_gpu_uuid = _validate_gpu_ownership(manifest.get("physical_gpu_ownership"))

    required_broad_metrics = (
        "return",
        "safety_violation_rate",
        "termination_rate",
        "auxiliary_cost",
        "action_l2",
    )
    if any(name not in broad_statistics for name in required_broad_metrics):
        raise ValueError("Broad-reward evidence is missing a frozen decision metric.")
    policy_module = _source_module(Path(__file__).with_name("g1_policy_quality_evidence.py"))
    recomputed = policy_module._native_quality_decision(
        gate,
        checkpoint={name: checkpoint[name] for name in ("transition", "training_seed", "sha256")},
        protocol=protocol,
        tracking={
            "emd_mean": tracking_statistics["emd"]["mean"],
            "obs_state_emd_mean": tracking_statistics["obs_state_emd"]["mean"],
            "coverage_fraction": tracking_statistics["coverage"]["minimum"],
        },
        broad_reward={f"{name}_mean": broad_statistics[name]["mean"] for name in required_broad_metrics},
        broad_reward_comparison_identity=broad_manifest.get("comparison_identity"),
    )
    broad_diagnostic = _validate_broad_reward_decision_boundary(decision, broad_manifest.get("comparison_identity"))
    if recomputed != decision or recomputed.get("passed") is not True:
        raise ValueError("Policy-quality decision differs from independently recomputed evidence.")
    return {
        "schema": "forward_backward_phase3_g1_policy_quality_consumer_v2",
        "status": "passed",
        "manifest_sha256": _sha256(manifest_path),
        "tracking_sha256": _sha256(tracking_path),
        "broad_reward_sha256": _sha256(broad_reward_path),
        "physical_gpu_uuid": physical_gpu_uuid,
        "tracking_emd_mean": tracking_statistics["emd"]["mean"],
        "tracking_obs_state_emd_mean": tracking_statistics["obs_state_emd"]["mean"],
        "broad_return_mean": broad_statistics["return"]["mean"],
        "decision": "tracking_non_inferiority_passed",
        "broad_reward_status": broad_diagnostic["classification"],
        "broad_reward_point_gate": broad_diagnostic["point_gate"]["result"],
    }


def validate_cross_policy_quality_artifact(
    artifact_dir: str | Path,
    gate_path: str | Path,
    retarget_path: str | Path,
    simulator_path: str | Path,
) -> dict[str, object]:
    """Recompute one zero-shot G1-CMU policy measurement and its companion closure.

    Args:
        artifact_dir: Directory containing the cross-source policy artifacts.
        gate_path: Frozen native gate with the cross-source measurement contract.
        retarget_path: Canonical CMU-to-G1 retarget evidence.
        simulator_path: Canonical G1-on-CMU simulator evidence.

    Returns:
        Compact independent measurement receipt.
    """
    artifact = Path(artifact_dir).expanduser().resolve()
    gate_file = Path(gate_path).expanduser().resolve()
    retarget_file = Path(retarget_path).expanduser().resolve()
    simulator_file = Path(simulator_path).expanduser().resolve()
    manifest_path = artifact / "manifest.json"
    tracking_path = artifact / "tracking.json"
    broad_reward_path = artifact / "broad_reward.csv"
    manifest = _json_object(manifest_path)
    tracking = _json_object(tracking_path)
    gate = _json_object(gate_file)

    if (
        manifest.get("schema") != "forward_backward_phase3g_g1_policy_quality_evidence_v6"
        or manifest.get("status") != "measured"
        or manifest.get("evaluation_kind") != "zero_shot_cross_source"
        or manifest.get("preset") != "g1_cmu"
        or manifest.get("training_domain") != "g1_lafan"
    ):
        raise ValueError("Policy-quality manifest is not the zero-shot G1-CMU v6 evidence contract.")
    checkpoint = manifest.get("checkpoint")
    expected_checkpoint = gate.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or not isinstance(expected_checkpoint, Mapping):
        raise TypeError("Policy-quality manifest and gate must declare checkpoint mappings.")
    if {name: checkpoint.get(name) for name in expected_checkpoint} != expected_checkpoint:
        raise ValueError("Cross-source artifact uses a checkpoint different from the frozen native gate.")
    if checkpoint.get("profile") != "residual_6x1024":
        raise ValueError("Cross-source artifact uses an unexpected model profile.")

    code_identity = manifest.get("code_identity")
    if not isinstance(code_identity, Mapping):
        raise TypeError("Policy-quality manifest must declare code identity.")
    dependency_identity, composition_dependency_identity, identity_module = _validate_policy_code_identity(
        code_identity, gate_file, "g1_cmu"
    )
    _validate_protocol_audit(manifest, code_identity)

    policy_quality = manifest.get("policy_quality")
    if not isinstance(policy_quality, Mapping):
        raise TypeError("Policy-quality manifest must declare tracking and broad-reward evidence.")
    tracking_manifest = policy_quality.get("tracking")
    broad_manifest = policy_quality.get("broad_reward")
    if not isinstance(tracking_manifest, Mapping) or not isinstance(broad_manifest, Mapping):
        raise TypeError("Policy-quality manifest has malformed evidence declarations.")
    if tracking_manifest.get("artifact") != tracking_path.name or tracking_manifest.get("sha256") != _sha256(
        tracking_path
    ):
        raise ValueError("Tracking artifact bytes differ from the manifest.")
    if broad_manifest.get("artifact") != broad_reward_path.name or broad_manifest.get("sha256") != _sha256(
        broad_reward_path
    ):
        raise ValueError("Broad-reward artifact bytes differ from the manifest.")

    tracking_statistics = _tracking_statistics(tracking, 182)
    if (
        tracking_manifest.get("clip_count") != 182
        or tracking_statistics["emd"] != tracking.get("emd_statistics")
        or tracking_statistics["obs_state_emd"] != tracking.get("obs_state_emd_statistics")
        or tracking_statistics["coverage"] != tracking.get("coverage_statistics")
        or tracking_statistics["emd"] != tracking_manifest.get("emd")
        or tracking_statistics["obs_state_emd"] != tracking_manifest.get("obs_state_emd")
        or tracking_statistics["coverage"] != tracking_manifest.get("coverage")
    ):
        raise ValueError("Cross-source tracking summaries differ from serialized per-clip evidence.")
    row_count = broad_manifest.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int):
        raise TypeError("Broad-reward row_count must be an integer.")
    broad_statistics = _broad_reward_statistics(broad_reward_path, row_count, "g1_cmu")
    if broad_statistics != broad_manifest.get("metrics"):
        raise ValueError("Broad-reward summaries differ from the serialized episode rows.")
    if (
        broad_manifest.get("task_count") != 38
        or broad_manifest.get("episodes_per_task") != 10
        or broad_manifest.get("horizon") != 500
    ):
        raise ValueError("Broad-reward protocol differs from the frozen 38-by-10, 500-step evaluation.")

    model_contract = manifest.get("model_contract")
    if (
        not isinstance(model_contract, Mapping)
        or model_contract.get("strict_load") is not True
        or model_contract.get("missing_keys") != []
        or model_contract.get("unexpected_keys") != []
    ):
        raise ValueError("Cross-source checkpoint was not loaded strictly.")
    physical_gpu_uuid = _validate_gpu_ownership(manifest.get("physical_gpu_ownership"))

    policy_module = _source_module(Path(__file__).with_name("g1_policy_quality_evidence.py"))
    error_layers = manifest.get("error_layer_separation")
    if not isinstance(error_layers, Mapping):
        raise TypeError("Cross-source evidence must separate policy and companion error layers.")
    recomputed_companions = policy_module._g1_cmu_companion_evidence(
        retarget_file,
        simulator_file,
        dependency_identity,
        composition_dependency_identity,
        environment_semantic_sha256=identity_module.motion_environment_semantic_sha256,
        composition_semantic_sha256=identity_module.motion_composition_semantic_sha256,
    )
    if (
        error_layers.get("policy_quality") != "measured_here"
        or error_layers.get("retarget_and_simulator_errors_are_not_policy_metrics") is not True
        or _companion_identity(error_layers.get("companion_evidence")) != _companion_identity(recomputed_companions)
    ):
        raise ValueError("Cross-source companion evidence differs from independently revalidated artifacts.")

    decision = manifest.get("decision")
    if not isinstance(decision, Mapping):
        raise TypeError("Cross-source policy-quality manifest must declare a decision.")
    protocol_result = decision.get("protocol")
    if not isinstance(protocol_result, Mapping) or protocol_result.get("passed") is not True:
        raise ValueError("Cross-source decision did not accept its frozen protocol.")
    protocol = protocol_result.get("actual")
    if not isinstance(protocol, Mapping):
        raise TypeError("Cross-source decision must retain its actual protocol.")
    recomputed_decision = policy_module._cross_source_quality_decision(
        gate,
        checkpoint={name: checkpoint[name] for name in ("transition", "training_seed", "sha256")},
        protocol=protocol,
    )
    if (
        recomputed_decision != decision
        or recomputed_decision.get("measurement_complete") is not True
        or recomputed_decision.get("status") != "measured"
        or recomputed_decision.get("passed") is not None
    ):
        raise ValueError("Cross-source measurement decision differs from independently recomputed evidence.")

    required_broad_metrics = ("return", "safety_violation_rate", "termination_rate", "auxiliary_cost", "action_l2")
    if any(name not in broad_statistics for name in required_broad_metrics):
        raise ValueError("Cross-source broad-reward evidence is missing a frozen measurement metric.")
    return {
        "schema": "forward_backward_phase3_g1_cross_policy_quality_consumer_v1",
        "status": "measured",
        "manifest_sha256": _sha256(manifest_path),
        "tracking_sha256": _sha256(tracking_path),
        "broad_reward_sha256": _sha256(broad_reward_path),
        "retarget_sha256": _sha256(retarget_file),
        "simulator_sha256": _sha256(simulator_file),
        "physical_gpu_uuid": physical_gpu_uuid,
        "tracking_emd_mean": tracking_statistics["emd"]["mean"],
        "tracking_obs_state_emd_mean": tracking_statistics["obs_state_emd"]["mean"],
        "broad_return_mean": broad_statistics["return"]["mean"],
        "decision": "measurement_only",
    }


__all__ = ["validate_cross_policy_quality_artifact", "validate_native_policy_quality_artifact"]
