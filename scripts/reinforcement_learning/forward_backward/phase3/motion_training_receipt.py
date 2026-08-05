# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Closed launch, completion, and strict-load evidence for Phase 3F."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import enum
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import re
from collections.abc import Mapping
from functools import cache
from pathlib import Path

from rsl_rl.utils import resolve_callable

from isaaclab_tasks.core.multi_task.rl.rsl_rl import RslRlForwardBackwardRunnerCfg

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RECEIPT_SCHEMA = "forward_backward_phase3f_motion_training_receipt_v1"
_FREEZE_SCHEMA = "forward_backward_phase3f_identity_freeze_v1"
_LAUNCH_SCHEMA = "forward_backward_phase3f_motion_training_launch_v1"
_COMPLETE_SCHEMA = "forward_backward_phase3f_motion_training_complete_v1"
_VALIDATION_SCHEMA = "forward_backward_phase3f_motion_training_validation_v1"
_CONTRACT_SCHEMA = "forward_backward_phase3_motion_training_smoke_contract_v2"
_CONTRACT = Path(__file__).parent / "fixtures" / "motion_training_smoke_contract_v2.json"
_NATIVE_PRESETS = frozenset({"smpl_cmu", "g1_lafan"})
_DERIVED_PROFILE_FIELDS = frozenset({"closed_input_identity", "environment_semantic_sha256"})


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def expected_update_groups(collection: object) -> int:
    """Return update-bearing iterations under the runner's strict post-seed rule."""
    values = _mapping(collection, "collection")
    num_envs = _positive_int(values.get("num_envs"), "num_envs")
    steps = _positive_int(values.get("steps_per_iteration"), "steps_per_iteration")
    iterations = _positive_int(values.get("iterations"), "iterations")
    random_actions = values.get("random_action_transitions")
    if not isinstance(random_actions, int) or isinstance(random_actions, bool) or random_actions < 0:
        raise ValueError("random_action_transitions must be a non-negative integer.")
    block = num_envs * steps
    return sum(iteration * block > random_actions for iteration in range(iterations))


def _expected_identity(profile: Mapping[str, object]) -> dict[str, object]:
    """Combine host-bound inputs with the portable environment meaning."""
    identity = dict(_mapping(profile.get("closed_input_identity"), "closed input identity"))
    semantic = profile.get("environment_semantic_sha256")
    if not isinstance(semantic, str) or _SHA256.fullmatch(semantic) is None:
        raise ValueError("Profile environment semantics must be a lowercase SHA-256 digest.")
    if "environment_semantic_sha256" in identity:
        raise ValueError("Environment semantics must have one profile-level owner.")
    identity["environment_semantic_sha256"] = semantic
    return identity


def _validate_identity(
    expected: Mapping[str, object],
    launch: Mapping[str, object],
    complete: Mapping[str, object],
    validation: Mapping[str, object],
) -> dict[str, str]:
    identities = tuple(_mapping(record.get("identity"), "identity") for record in (launch, complete, validation))
    if any(identity != expected for identity in identities):
        raise ValueError("Phase 3F identity differs across contract, launch, completion, or validation.")
    result: dict[str, str] = {}
    for name, value in expected.items():
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"Identity {name!r} must be a lowercase SHA-256 digest.")
        result[name] = value
    return result


def _validate_collection(profile: Mapping[str, object], runner: Mapping[str, object]) -> dict[str, int]:
    collection = _mapping(profile.get("collection"), "collection")
    num_envs = _positive_int(collection.get("num_envs"), "num_envs")
    steps = _positive_int(collection.get("steps_per_iteration"), "steps_per_iteration")
    iterations = _positive_int(collection.get("iterations"), "iterations")
    expected_transitions = num_envs * steps * iterations
    if collection.get("expected_transitions") != expected_transitions:
        raise ValueError("Declared transition count differs from exact collection math.")
    groups = expected_update_groups(collection)
    if collection.get("expected_update_groups") != groups:
        raise ValueError("Declared update-group count differs from strict post-seed math.")
    updates_per_group = _positive_int(collection.get("updates_per_group"), "updates_per_group")
    update_calls = groups * updates_per_group
    if collection.get("expected_update_calls") != update_calls:
        raise ValueError("Declared update-call count differs from exact update math.")
    if (
        runner.get("completed_iterations") != iterations
        or runner.get("collected_transitions") != expected_transitions
        or runner.get("update_calls") != update_calls
    ):
        raise ValueError("Runner iteration, transition, or update counters differ from the contract.")
    return {
        "num_envs": num_envs,
        "steps_per_iteration": steps,
        "iterations": iterations,
        "expected_transitions": expected_transitions,
        "random_action_transitions": int(collection["random_action_transitions"]),
        "updates_per_group": updates_per_group,
        "expected_update_groups": groups,
        "expected_update_calls": update_calls,
    }


def _validate_metrics(profile: Mapping[str, object], runner: Mapping[str, object]) -> None:
    learner = _mapping(profile.get("learner"), "learner contract")
    required = learner.get("required_metrics")
    names = runner.get("metric_names")
    if (
        not isinstance(required, list)
        or not all(isinstance(name, str) and name for name in required)
        or not isinstance(names, list)
        or len(names) != len(set(names))
        or set(names) != set(required)
    ):
        raise ValueError("Observed metric keys differ from the required metric set.")
    if runner.get("all_metrics_finite") is not True:
        raise ValueError("Runner did not prove all metrics finite.")
    last = _mapping(runner.get("last_metrics"), "last metrics")
    if set(last) != set(required) or any(
        not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value)
        for value in last.values()
    ):
        raise ValueError("The final metric values are missing or non-finite.")
    if runner.get("action_statistics_finite") is not True:
        raise ValueError("Policy action statistics are not finite.")
    checks = _mapping(runner.get("actor_state_checks"), "actor state checks")
    expected_checks = {"actor_network_parameters", "action_distribution_cached_params"}
    if set(checks) != expected_checks:
        raise ValueError("Actor state checks do not name the complete declared owners.")
    for name in sorted(expected_checks):
        check = _mapping(checks[name], f"actor state check {name}")
        tensor_count = check.get("tensor_count")
        scalar_count = check.get("scalar_count")
        if (
            not isinstance(tensor_count, int)
            or isinstance(tensor_count, bool)
            or tensor_count < 1
            or not isinstance(scalar_count, int)
            or isinstance(scalar_count, bool)
            or scalar_count < tensor_count
            or check.get("all_finite") is not True
        ):
            raise ValueError(f"Actor state owner {name!r} is empty, malformed, or non-finite.")


def _validate_learner(profile: Mapping[str, object], learner: Mapping[str, object], update_calls: int) -> None:
    contract = _mapping(profile.get("learner"), "learner contract")
    if learner.get("update_step") != update_calls:
        raise ValueError("Learner update_step differs from the exact update count.")
    expected_versions = contract.get("expected_version_names")
    versions = _mapping(learner.get("versions"), "learner versions")
    if (
        not isinstance(expected_versions, list)
        or not all(isinstance(name, str) and name for name in expected_versions)
        or set(versions) != set(expected_versions)
        or any(value != update_calls for value in versions.values())
    ):
        raise ValueError("Learner version owners did not advance exactly once per update.")
    batch_size = _positive_int(contract.get("batch_size"), "batch_size")
    capacity = _positive_int(contract.get("context_buffer_capacity"), "context_buffer_capacity")
    if learner.get("context_buffer_size") != min(update_calls * batch_size, capacity):
        raise ValueError("Learner context buffer size differs from exact update append math.")
    if learner.get("replay_contract_errors") is not False or learner.get("replay_terminal_overflow") is not False:
        raise ValueError("Learner replay reports a contract or terminal-overflow error.")
    if learner.get("device_scope") != profile.get("device_scope"):
        raise ValueError("Learner, replay, expert, simulator, or dtype device scope differs.")


def validate_smoke_records(
    profile: object,
    launch: object,
    complete: object,
    validation: object,
) -> dict[str, object]:
    """Validate every Phase 3F claim and return the canonical passed receipt."""
    profile = _mapping(profile, "profile")
    launch = _mapping(launch, "launch record")
    complete = _mapping(complete, "completion record")
    validation = _mapping(validation, "validation record")
    expected_schemas = (
        (launch, _LAUNCH_SCHEMA, "launch"),
        (complete, _COMPLETE_SCHEMA, "completion"),
        (validation, _VALIDATION_SCHEMA, "validation"),
    )
    for record, expected_schema, name in expected_schemas:
        if record.get("schema") != expected_schema:
            raise ValueError(f"Phase 3F {name} record has an unsupported schema.")

    if launch.get("tracking_curriculum") is not None:
        raise ValueError("Phase 3F learner integration must disable tracking curriculum.")
    presets = {record.get("preset") for record in (launch, complete, validation)}
    if len(presets) != 1 or None in presets:
        raise ValueError("Phase 3F record preset identities differ.")
    contract_digests = {record.get("contract_declaration_sha256") for record in (launch, complete, validation)}
    if len(contract_digests) != 1:
        raise ValueError("Phase 3F contract declaration changed across callback stages.")
    contract_digest = contract_digests.pop()
    if not isinstance(contract_digest, str) or _SHA256.fullmatch(contract_digest) is None:
        raise ValueError("Phase 3F contract declaration digest is missing or malformed.")
    identity = _validate_identity(
        _expected_identity(profile),
        launch,
        complete,
        validation,
    )
    provenance = _validate_provenance(identity, launch.get("provenance"))
    runner = _mapping(complete.get("runner"), "runner summary")
    collection = _validate_collection(profile, runner)
    _validate_metrics(profile, runner)
    complete_learner = _mapping(complete.get("learner"), "completion learner")
    validation_learner = _mapping(validation.get("learner"), "validation learner")
    _validate_learner(profile, complete_learner, collection["expected_update_calls"])
    _validate_learner(profile, validation_learner, collection["expected_update_calls"])
    if validation_learner != complete_learner:
        raise ValueError("Strict-load learner state differs from the completion learner state.")

    checkpoint_contract = _mapping(profile.get("checkpoint"), "checkpoint contract")
    completed_checkpoint = _mapping(complete.get("checkpoint"), "completion checkpoint")
    loaded_checkpoint = _mapping(validation.get("checkpoint"), "validation checkpoint")
    path = completed_checkpoint.get("path")
    if (
        not isinstance(path, str)
        or Path(path).name != checkpoint_contract.get("filename")
        or loaded_checkpoint.get("path") != path
        or not isinstance(completed_checkpoint.get("bytes"), int)
        or completed_checkpoint["bytes"] < 1
        or loaded_checkpoint.get("bytes") != completed_checkpoint["bytes"]
    ):
        raise ValueError("Completion and validation checkpoint path or size differs.")
    if loaded_checkpoint.get("strict_load") is not True:
        raise ValueError("Checkpoint validation did not complete a strict load.")
    if loaded_checkpoint.get("map_location") != checkpoint_contract.get("strict_map_location"):
        raise ValueError("Checkpoint validation map_location is not the declared CPU boundary.")
    if loaded_checkpoint.get("mmap") is not checkpoint_contract.get("strict_mmap"):
        raise ValueError("Checkpoint validation mmap mode differs from the declared boundary.")
    if loaded_checkpoint.get("environment_resume") != checkpoint_contract.get("environment_resume"):
        raise ValueError("Checkpoint must retain explicit environment restart semantics.")
    if loaded_checkpoint.get("environment_state_dict_is_none") is not True:
        raise ValueError("Restart checkpoint unexpectedly contains environment state.")
    completed_digest = completed_checkpoint.get("sha256")
    loaded_digest = loaded_checkpoint.get("sha256")
    if (
        not isinstance(completed_digest, str)
        or _SHA256.fullmatch(completed_digest) is None
        or not isinstance(loaded_digest, str)
        or _SHA256.fullmatch(loaded_digest) is None
    ):
        raise ValueError("Checkpoint SHA-256 is missing or malformed.")
    if completed_digest != loaded_digest:
        raise ValueError("Completion and validation checkpoint digests differ.")

    canonical_checkpoint = dict(loaded_checkpoint)
    canonical_checkpoint["filename"] = Path(path).name
    canonical_checkpoint.pop("path")
    return {
        "schema": _RECEIPT_SCHEMA,
        "status": "passed",
        "preset": presets.pop(),
        "identity": identity,
        "contract_declaration_sha256": contract_digest,
        "provenance": provenance,
        "collection": collection,
        "runner": dict(runner),
        "learner": dict(complete_learner),
        "checkpoint": canonical_checkpoint,
    }


@cache
def _sibling_module(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"phase3f_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Phase 3 identity module: {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file_sha256(path: Path) -> str:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Identity input must be a regular non-symbolic file: {path}.")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _contract_declaration_sha256() -> str:
    """Hash the exact immutable contract bytes read by every callback stage."""
    return _file_sha256(_CONTRACT)


def _owner_source(owner: object) -> tuple[str, str]:
    path = inspect.getsourcefile(owner)
    module = getattr(owner, "__module__", None)
    if path is None or not isinstance(module, str):
        raise RuntimeError(f"Cannot identify source owner: {owner!r}.")
    return module, _file_sha256(Path(path))


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _static_profile_sha256(profile: object) -> str:
    """Hash every contract profile field except live-derived identities."""
    values = dict(_mapping(profile, "contract profile"))
    missing = _DERIVED_PROFILE_FIELDS - values.keys()
    if missing:
        raise ValueError(f"Contract profile is missing derived identity fields: {sorted(missing)}")
    for name in _DERIVED_PROFILE_FIELDS:
        values.pop(name)
    return _json_sha256(values)


def _identity_digests(value: object) -> dict[str, str]:
    """Validate one complete live identity without duplicating its producer's field list."""
    identity = dict(_mapping(value, "live identity"))
    if "environment_semantic_sha256" not in identity:
        raise ValueError("Live identity is missing environment_semantic_sha256.")
    if not identity:
        raise ValueError("Live identity must not be empty.")
    for name, digest in identity.items():
        if not isinstance(name, str) or not name or not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"Live identity {name!r} must be a lowercase SHA-256 digest.")
    return identity


def _canonical_config_value(value: object, path: str) -> object:
    """Normalize one resolved config tree without host-object representations."""
    if isinstance(value, Mapping):
        if not all(isinstance(name, str) for name in value):
            raise TypeError(f"Resolved agent config keys must be strings at {path}.")
        return {name: _canonical_config_value(value[name], f"{path}.{name}") for name in sorted(value)}
    if isinstance(value, list | tuple):
        return [_canonical_config_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonical_config_value(dataclasses.asdict(value), path)
    if isinstance(value, enum.Enum):
        return _canonical_config_value(value.value, path)
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Resolved agent config contains a non-finite float at {path}.")
        return value
    if callable(value):
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        if not isinstance(module, str) or not isinstance(qualname, str):
            raise TypeError(f"Resolved callable has no stable owner at {path}.")
        return {"callable": f"{module}:{qualname}"}
    raise TypeError(f"Resolved agent config contains unsupported {type(value).__name__} at {path}.")


def _resolved_agent_config_sha256(agent_cfg: object) -> str:
    """Hash post-override learner semantics independently of output routing."""
    to_dict = getattr(agent_cfg, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("Resolved agent configuration must expose to_dict().")
    values = dict(_mapping(to_dict(), "resolved agent configuration"))
    # ``run_name`` only chooses the output-directory suffix. It neither changes
    # collection nor learner behavior and must remain free for collision-safe runs.
    values.pop("run_name", None)
    return _json_sha256(_canonical_config_value(values, "agent"))


def _python_package_identity(package_root: Path) -> dict[str, object]:
    """Retain every Python member contributing to one package digest."""
    identity = _sibling_module("python_source_identity")
    value = identity.python_package_identity(package_root)
    if not isinstance(value, dict):
        raise TypeError("Python package identity must be a mapping.")
    return value


def _python_package_bundle_sha256(package_root: Path) -> str:
    """Return the digest from :func:`_python_package_identity`."""
    digest = _python_package_identity(package_root).get("bundle_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError("Python package identity has no valid bundle digest.")
    return digest


def _task_bridge_identity(env: object, agent_cfg: RslRlForwardBackwardRunnerCfg) -> dict[str, object]:
    """Retain the wrapper, runner-config, and expert-provider source owners."""
    provider = resolve_callable(str(agent_cfg.expert.provider))
    if not callable(provider):
        raise ValueError("Resolved agent expert provider must be callable.")
    owners = {
        "environment_wrapper": type(env),
        "motion_runner_config": type(agent_cfg),
        "motion_expert_provider": provider,
    }
    members: dict[str, dict[str, str]] = {}
    for role, owner in owners.items():
        module, digest = _owner_source(owner)
        qualname = getattr(owner, "__qualname__", None)
        if not isinstance(qualname, str) or not qualname:
            raise RuntimeError(f"Bridge source owner has no qualified name: {owner!r}.")
        members[role] = {"owner": f"{module}:{qualname}", "source_sha256": digest}
    payload = {"source_owner_count": len(members), "source_owners": members}
    return {**payload, "bundle_sha256": _json_sha256(payload)}


def _task_bridge_bundle_sha256(env: object, agent_cfg: object) -> str:
    """Return the digest from :func:`_task_bridge_identity`."""
    digest = _task_bridge_identity(env, agent_cfg).get("bundle_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError("Task bridge identity has no valid bundle digest.")
    return digest


def _runtime_package_member(module: object, distribution_name: str) -> dict[str, str]:
    """Return version and imported-module source identity for one package."""
    module_version = getattr(module, "__version__", None)
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_version, str) or not module_version:
        raise ValueError(f"Runtime package {distribution_name!r} does not expose a module version.")
    if not isinstance(module_file, str) or not module_file:
        raise ValueError(f"Runtime package {distribution_name!r} does not expose a source file.")
    distribution_version = importlib.metadata.version(distribution_name)
    return {
        "module_version": module_version,
        "distribution_version": distribution_version,
        "module_source_sha256": _file_sha256(Path(module_file)),
    }


def _learner_runtime_identity() -> dict[str, object]:
    """Retain packages defining learner tensors and environment edges."""
    import gymnasium
    import tensordict

    payload = {
        "schema": "forward_backward_phase3f_learner_runtime_identity_v1",
        "packages": {
            "gymnasium": _runtime_package_member(gymnasium, "gymnasium"),
            "tensordict": _runtime_package_member(tensordict, "tensordict"),
        },
    }
    return {**payload, "bundle_sha256": _json_sha256(payload)}


def _manifest_bundle_sha256(value: object, name: str) -> str:
    """Validate and return one canonical member-manifest digest."""
    manifest = dict(_mapping(value, name))
    digest = manifest.pop("bundle_sha256", None)
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{name} has no valid bundle digest.")
    if _json_sha256(manifest) != digest:
        raise ValueError(f"{name} member manifest differs from its bundle digest.")
    return digest


def _validate_python_package_identity(value: object) -> dict[str, object]:
    """Require one complete, ordered Python package member manifest."""
    identity = dict(_mapping(value, "learner code identity"))
    if set(identity) != {"python_file_count", "python_files", "bundle_sha256"}:
        raise ValueError("Learner code identity has an unsupported structure.")
    members = identity["python_files"]
    if not isinstance(members, list) or identity["python_file_count"] != len(members) or not members:
        raise ValueError("Learner code identity has an invalid member count.")
    paths: list[str] = []
    for member in members:
        item = _mapping(member, "learner code member")
        path = item.get("path")
        digest = item.get("sha256")
        if (
            set(item) != {"path", "sha256"}
            or not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise ValueError("Learner code identity contains an invalid member.")
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Learner code identity paths must be ordered and unique.")
    _manifest_bundle_sha256(identity, "learner code identity")
    return identity


def _validate_task_bridge_identity(value: object) -> dict[str, object]:
    """Require all three concrete task-to-learner source owners."""
    identity = dict(_mapping(value, "task bridge identity"))
    if set(identity) != {"source_owner_count", "source_owners", "bundle_sha256"}:
        raise ValueError("Task bridge identity has an unsupported structure.")
    owners = _mapping(identity["source_owners"], "task bridge source owners")
    expected = {"environment_wrapper", "motion_runner_config", "motion_expert_provider"}
    if identity["source_owner_count"] != len(owners) or set(owners) != expected:
        raise ValueError("Task bridge identity does not name its three source owners.")
    for owner in owners.values():
        item = _mapping(owner, "task bridge source owner")
        qualified = item.get("owner")
        digest = item.get("source_sha256")
        if (
            set(item) != {"owner", "source_sha256"}
            or not isinstance(qualified, str)
            or ":" not in qualified
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise ValueError("Task bridge identity contains an invalid source owner.")
    _manifest_bundle_sha256(identity, "task bridge identity")
    return identity


def _validate_learner_runtime_identity(value: object) -> dict[str, object]:
    """Require exact Gymnasium and TensorDict version/source provenance."""
    identity = dict(_mapping(value, "learner runtime identity"))
    if set(identity) != {"schema", "packages", "bundle_sha256"} or identity.get("schema") != (
        "forward_backward_phase3f_learner_runtime_identity_v1"
    ):
        raise ValueError("Learner runtime identity has an unsupported structure.")
    packages = _mapping(identity["packages"], "learner runtime packages")
    if set(packages) != {"gymnasium", "tensordict"}:
        raise ValueError("Learner runtime identity must name Gymnasium and TensorDict.")
    for package in packages.values():
        item = _mapping(package, "learner runtime package")
        if set(item) != {"module_version", "distribution_version", "module_source_sha256"} or any(
            not isinstance(item[name], str) or not item[name] for name in ("module_version", "distribution_version")
        ):
            raise ValueError("Learner runtime identity contains an invalid package version.")
        digest = item["module_source_sha256"]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError("Learner runtime identity contains an invalid source digest.")
    _manifest_bundle_sha256(identity, "learner runtime identity")
    return identity


def _validate_provenance(identity: Mapping[str, object], value: object) -> dict[str, object]:
    """Validate retained members against every aggregate identity digest."""
    provenance = dict(_mapping(value, "provenance"))
    if set(provenance) != {"environment", "learner_code", "learner_runtime", "task_bridge"}:
        raise ValueError("Phase 3F provenance has an unsupported structure.")
    environment = dict(_mapping(provenance["environment"], "environment provenance"))
    if set(environment) != {"dependency_identity", "semantic_sha256"}:
        raise ValueError("Environment provenance has an unsupported structure.")
    dependency = dict(_mapping(environment["dependency_identity"], "environment dependency identity"))
    dependency_digest = _manifest_bundle_sha256(dependency, "environment dependency identity")
    semantic = _sibling_module("motion_environment_identity").motion_environment_semantic_sha256(dependency)
    if environment["semantic_sha256"] != semantic:
        raise ValueError("Environment provenance semantic digest differs from its dependency members.")
    learner_code = _validate_python_package_identity(provenance["learner_code"])
    learner_runtime = _validate_learner_runtime_identity(provenance["learner_runtime"])
    task_bridge = _validate_task_bridge_identity(provenance["task_bridge"])
    expected = {
        "environment_dependency_bundle_sha256": dependency_digest,
        "environment_semantic_sha256": semantic,
        "learner_code_bundle_sha256": learner_code["bundle_sha256"],
        "learner_runtime_bundle_sha256": learner_runtime["bundle_sha256"],
        "task_bridge_code_bundle_sha256": task_bridge["bundle_sha256"],
    }
    if any(identity.get(name) != digest for name, digest in expected.items()):
        raise ValueError("Phase 3F aggregate identity differs from its retained provenance members.")
    return provenance


def _load_json(path: Path, name: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a regular non-symbolic JSON file: {path}.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} root must be a JSON object.")
    return value


def _write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp")
    if path.exists() or path.is_symlink() or staging.exists() or staging.is_symlink():
        raise FileExistsError(f"Phase 3F evidence path already exists: {path}.")
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor = os.open(staging, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    os.replace(staging, path)


def _contract_profiles(contract: object) -> dict[str, object]:
    """Validate the frozen declaration envelope and return its native profiles."""
    values = dict(_mapping(contract, "Phase 3F smoke contract"))
    if values.get("schema") != _CONTRACT_SCHEMA:
        raise ValueError("Phase 3F smoke contract does not use the version-two schema.")
    if values.get("status") != "prepared_not_launched":
        raise ValueError("Phase 3F smoke contract is not prepared and unlaunched.")
    if values.get("launch_gate") != "wait_for_final_phase3e_simulator_and_cloner_identity":
        raise ValueError("Phase 3F smoke contract does not retain its Phase 3E launch gate.")
    profiles = dict(_mapping(values.get("profiles"), "contract profiles"))
    if set(profiles) != _NATIVE_PRESETS:
        raise ValueError("Phase 3F smoke contract must contain exactly the two native presets.")

    return profiles


def _identity_freeze_record(
    preset: str,
    profile: Mapping[str, object],
    identity: object,
    provenance: object,
    contract_declaration_sha256: str,
) -> dict[str, object]:
    """Bind one live production identity to the non-derived profile declaration."""
    identity = _identity_digests(identity)
    provenance = _validate_provenance(identity, provenance)
    if _SHA256.fullmatch(contract_declaration_sha256) is None:
        raise ValueError("Identity freeze record requires the exact contract SHA-256.")
    return {
        "schema": _FREEZE_SCHEMA,
        "preset": preset,
        "contract_declaration_sha256": contract_declaration_sha256,
        "static_profile_sha256": _static_profile_sha256(profile),
        "identity": identity,
        "provenance": provenance,
    }


def _validated_freeze_record(
    value: object,
    preset: str,
    profile: Mapping[str, object],
    contract_declaration_sha256: str,
) -> dict[str, str]:
    """Validate one prepared identity against the exact immutable profile fields."""
    record = dict(_mapping(value, "identity freeze record"))
    expected_fields = {
        "schema",
        "preset",
        "contract_declaration_sha256",
        "static_profile_sha256",
        "identity",
        "provenance",
    }
    if set(record) != expected_fields or record.get("schema") != _FREEZE_SCHEMA:
        raise ValueError("Identity freeze record has an unsupported structure.")
    if record.get("preset") != preset:
        raise ValueError("Identity freeze record preset differs from its selected profile.")
    if record.get("contract_declaration_sha256") != contract_declaration_sha256:
        raise ValueError("Identity freeze record was not derived from the selected contract bytes.")
    if record.get("static_profile_sha256") != _static_profile_sha256(profile):
        raise ValueError("Identity freeze record differs from the contract static profile.")
    identity = _identity_digests(record.get("identity"))
    _validate_provenance(identity, record.get("provenance"))
    return identity


def freeze_contract(
    contract_path: Path,
    identity_record_paths: list[Path],
    output_path: Path,
) -> dict[str, object]:
    """Write a contract candidate whose only changes are live-derived identity fields."""
    contract_path = contract_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    contract = _load_json(contract_path, "Phase 3F smoke contract")
    profiles = _contract_profiles(contract)
    contract_declaration_sha256 = _file_sha256(contract_path)
    if len(identity_record_paths) != len(_NATIVE_PRESETS):
        raise ValueError("Contract freeze requires exactly one identity record per native preset.")

    identities: dict[str, dict[str, str]] = {}
    for path in identity_record_paths:
        record = _load_json(path.expanduser().resolve(), "Phase 3F identity freeze record")
        preset = record.get("preset")
        if not isinstance(preset, str) or preset not in profiles or preset in identities:
            raise ValueError("Contract freeze identity presets must be native, unique, and complete.")
        profile = _mapping(profiles[preset], f"contract profile {preset}")
        identities[preset] = _validated_freeze_record(
            record,
            preset,
            profile,
            contract_declaration_sha256,
        )
    if set(identities) != _NATIVE_PRESETS:
        raise ValueError("Contract freeze identity records do not cover both native presets.")

    frozen = copy.deepcopy(contract)
    frozen_profiles = dict(_mapping(frozen["profiles"], "frozen contract profiles"))
    for preset, identity in identities.items():
        original_profile = _mapping(profiles[preset], f"contract profile {preset}")
        profile = dict(_mapping(frozen_profiles[preset], f"frozen contract profile {preset}"))
        identity = dict(identity)
        profile["environment_semantic_sha256"] = identity.pop("environment_semantic_sha256")
        profile["closed_input_identity"] = identity
        if _static_profile_sha256(profile) != _static_profile_sha256(original_profile):
            raise RuntimeError("Contract freeze changed a non-derived profile field.")
        frozen_profiles[preset] = profile
    frozen["profiles"] = frozen_profiles
    _write_json_exclusive(output_path, frozen)
    return frozen


def _contract_profile(preset: str) -> dict[str, object]:
    contract = _load_json(_CONTRACT, "Phase 3F smoke contract")
    profiles = _contract_profiles(contract)
    profile = profiles.get(preset)
    if not isinstance(profile, dict):
        raise ValueError(f"Phase 3F contract does not declare preset {preset!r}.")
    return profile


def _preset(env_cfg: object) -> str:
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_physx.physics import PhysxCfg

    from isaaclab_tasks.core.multi_task.mdp import NativeMujocoControlActionCfg
    from isaaclab_tasks.core.multi_task.motion.robots.g1.actions_cfg import G1JointPositionActionCfg

    _action_name, action_cfg = _sibling_module("motion_environment_identity").motion_action_term_cfg(env_cfg)
    if isinstance(action_cfg, NativeMujocoControlActionCfg):
        robot = "smpl"
    elif isinstance(action_cfg, G1JointPositionActionCfg):
        robot = "g1"
    else:
        raise ValueError(f"Unsupported Phase 3F robot action configuration: {type(action_cfg)!r}.")

    table_cfg = env_cfg.commands.motion.task_table
    source_id = table_cfg.source.identifier
    if source_id == "cmu_humenv_smpl":
        source = "cmu"
    elif source_id == "lafan_g1_29dof":
        source = "lafan"
    else:
        raise ValueError(f"Unsupported Phase 3F motion source: {source_id!r}.")

    simulation_cfg = env_cfg.sim
    physics_cfg = simulation_cfg.physics
    if isinstance(physics_cfg, NewtonCfg) and isinstance(physics_cfg.solver_cfg, MJWarpSolverCfg):
        backend = "newton_mjwarp"
    elif isinstance(physics_cfg, PhysxCfg):
        backend = "physx"
    else:
        raise ValueError(f"Unsupported Phase 3F physics configuration: {type(physics_cfg)!r}.")

    horizon_steps = math.ceil(env_cfg.episode_length_s / (simulation_cfg.dt * env_cfg.decimation))
    semantics = (robot, source, backend, simulation_cfg.dt, env_cfg.decimation, horizon_steps, table_cfg.task_row_mode)
    native_profiles = {
        ("smpl", "cmu", "newton_mjwarp", 1.0 / 450.0, 15, 300, "source_frames"): "smpl_cmu",
        ("g1", "lafan", "physx", 1.0 / 200.0, 4, 501, "clip_time_ranges"): "g1_lafan",
    }
    if semantics not in native_profiles:
        raise ValueError(f"Phase 3F has no native reproduction contract for resolved semantics {semantics!r}.")
    return native_profiles[semantics]


def _native_types(preset: str) -> tuple[type, type]:
    # bfm-env-20260805 campaign patch: frame builders were renamed at HEAD
    # (G1PoseFrameBuilder -> G1FrameBuilder, SmplGeneralizedCoordinateFrameBuilder
    # -> SmplFrameBuilder); the receipt harness must name the live classes.
    from isaaclab_tasks.core.multi_task.motion.data.sources import CmuHumEnvSmplClips, LafanG1JoblibClips
    from isaaclab_tasks.core.multi_task.motion.robots.g1.reference import G1FrameBuilder
    from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import SmplFrameBuilder

    if preset == "smpl_cmu":
        return CmuHumEnvSmplClips, SmplFrameBuilder
    return LafanG1JoblibClips, G1FrameBuilder


def _motion_table(env: object):
    command = env.unwrapped.command_manager.get_term("motion")
    table = getattr(command, "table", None)
    if table is None:
        raise RuntimeError("The motion command does not expose its constructed task table.")
    return table


def _live_identity_evidence(
    preset: str,
    configured_env_cfg: object,
    env: object,
    runner: object,
    agent_cfg: object,
) -> tuple[dict[str, str], dict[str, object]]:
    importer_type, frame_builder_type = _native_types(preset)
    table_cfg = configured_env_cfg.commands.motion.task_table
    environment_identity_module = _sibling_module("motion_environment_identity")
    broad = environment_identity_module.motion_environment_dependency_identity(
        preset=preset,
        cfg=configured_env_cfg,
        importer_type=importer_type,
        frame_builder_type=frame_builder_type,
        reference_artifact_root=table_cfg.target_artifact_root,
    )
    runner_path = inspect.getsourcefile(type(runner))
    if runner_path is None:
        raise RuntimeError("The active runner has no concrete source file.")
    runner_path = Path(runner_path).resolve()
    if runner_path.parent.name != "runners" or runner_path.parents[1].name != "rsl_rl":
        raise RuntimeError("The active runner source is outside the concrete rsl_rl package root.")
    package_root = runner_path.parents[1]
    learner_code = _python_package_identity(package_root)
    learner_runtime = _learner_runtime_identity()
    task_bridge = _task_bridge_identity(env, agent_cfg)
    semantic = environment_identity_module.motion_environment_semantic_sha256(broad)
    _runner_module, runner_digest = _owner_source(type(runner))
    algorithm = runner.alg
    table = _motion_table(env)
    identity = {
        "environment_dependency_bundle_sha256": broad["bundle_sha256"],
        "environment_semantic_sha256": semantic,
        "resolved_axes_sha256": broad["resolved_axes_sha256"],
        "runner_source_sha256": runner_digest,
        "learner_code_bundle_sha256": learner_code["bundle_sha256"],
        "learner_runtime_bundle_sha256": learner_runtime["bundle_sha256"],
        "python_source_identity_sha256": _file_sha256(Path(__file__).with_name("python_source_identity.py")),
        "resolved_agent_config_sha256": _resolved_agent_config_sha256(agent_cfg),
        "task_bridge_code_bundle_sha256": task_bridge["bundle_sha256"],
        "training_cli_sha256": _file_sha256(Path(__file__).parents[2] / "rsl_rl" / "train_rsl_rl.py"),
        "receipt_code_sha256": _file_sha256(Path(__file__)),
        "checkpoint_schema_sha256": algorithm.checkpoint_header.schema_hash,
        "task_table_sha256": table.cache_identity,
        "expert_schema_sha256": algorithm.expert.schema.schema_hash,
        "observation_schema_sha256": algorithm.model.observation_schema.schema_hash,
    }
    provenance = {
        "environment": {"dependency_identity": broad, "semantic_sha256": semantic},
        "learner_code": learner_code,
        "learner_runtime": learner_runtime,
        "task_bridge": task_bridge,
    }
    _validate_provenance(identity, provenance)
    return identity, provenance


def _live_identity(
    preset: str,
    configured_env_cfg: object,
    env: object,
    runner: object,
    agent_cfg: object,
) -> dict[str, str]:
    """Return only aggregate digests for narrow static callers."""
    return _live_identity_evidence(preset, configured_env_cfg, env, runner, agent_cfg)[0]


def _assert_runtime_contract(
    profile: Mapping[str, object],
    env: object,
    runner: object,
    agent_cfg: object,
) -> None:
    """Require the exact frozen tracking, collection cadence, and update math."""
    if agent_cfg.tracking_curriculum is not None or runner.tracking_curriculum is not None:
        raise ValueError("Phase 3F learner integration must select the tracking_off curriculum preset.")
    collection = _mapping(profile.get("collection"), "collection")
    actual = {
        "num_envs": env.num_envs,
        "steps_per_iteration": int(runner.cfg["num_steps_per_env"]),
        "iterations": agent_cfg.max_iterations,
        "random_action_transitions": runner.alg.random_action_transitions,
        "updates_per_group": runner.num_updates_per_iteration,
    }
    if any(collection.get(name) != value for name, value in actual.items()):
        raise ValueError("Live Phase 3F runner cadence differs from the frozen collection contract.")
    groups = expected_update_groups(collection)
    if (
        collection.get("expected_transitions")
        != actual["num_envs"] * actual["steps_per_iteration"] * actual["iterations"]
        or collection.get("expected_update_groups") != groups
        or collection.get("expected_update_calls") != groups * actual["updates_per_group"]
    ):
        raise ValueError("Frozen Phase 3F collection and update math is internally inconsistent.")


def _assert_launch_contract(
    profile: Mapping[str, object],
    identity: Mapping[str, str],
    env: object,
    runner: object,
    agent_cfg: object,
) -> None:
    """Require runtime cadence and strict equality with the frozen live identity."""
    _assert_runtime_contract(profile, env, runner, agent_cfg)
    expected_identity = _expected_identity(profile)
    if identity != expected_identity:
        names = sorted(set(identity) | set(expected_identity))
        differences = {
            name: {"expected": expected_identity.get(name), "live": identity.get(name)}
            for name in names
            if identity.get(name) != expected_identity.get(name)
        }
        raise ValueError(f"Live Phase 3F identity differs from the frozen launch identity: {json.dumps(differences)}")


def _learner_snapshot(env: object, runner: object) -> dict[str, object]:
    import torch

    algorithm = runner.alg
    replay = algorithm.replay
    replay.assert_no_errors()
    contract_errors = bool(torch.any(replay.contract_errors).item())
    terminal_overflow = bool(torch.any(replay.terminal_overflow).item())
    return {
        "update_step": int(algorithm.update_step),
        "versions": {name: int(value) for name, value in sorted(algorithm.versions.items())},
        "context_buffer_size": int(algorithm.context_buffer_size),
        "replay_contract_errors": contract_errors,
        "replay_terminal_overflow": terminal_overflow,
        "device_scope": {
            "simulator": str(env.unwrapped.device),
            "learner": str(algorithm.device),
            "replay": str(replay.device),
            "expert": str(algorithm.expert.device),
            "dtype": str(replay.dtype),
        },
    }


def _runner_summary(runner: object) -> dict[str, object]:
    import torch

    def tensor_check(tensors: object) -> dict[str, object]:
        values = tuple(tensor.detach() for tensor in tensors)
        if not values:
            raise ValueError("Phase 3F actor state owner contains no tensors.")
        return {
            "tensor_count": len(values),
            "scalar_count": sum(tensor.numel() for tensor in values),
            "all_finite": bool(torch.stack(tuple(torch.all(torch.isfinite(tensor)) for tensor in values)).all().item()),
        }

    summary = dict(runner.training_summary())
    model = runner.alg.model
    action_std = runner.alg.action_std.detach()
    summary["action_statistics_finite"] = bool(torch.all(torch.isfinite(action_std)).item())
    summary["action_std_minimum"] = float(action_std.min().item())
    summary["action_std_maximum"] = float(action_std.max().item())
    summary["actor_state_checks"] = {
        "actor_network_parameters": tensor_check(model.actor_network.parameters()),
        "action_distribution_cached_params": tensor_check(model.action_distribution.params),
    }
    return summary


def _record_paths(root: Path) -> dict[str, Path]:
    return {
        "launch": root / "phase3f_launch.json",
        "complete": root / "phase3f_complete.json",
        "validation": root / "phase3f_validation.json",
        "receipt": root / "phase3f_receipt.json",
    }


def _launch_record(
    preset: str,
    profile: Mapping[str, object],
    identity: Mapping[str, str],
    env: object,
    runner: object,
    agent_cfg: object,
    provenance: Mapping[str, object],
    contract_declaration_sha256: str,
) -> dict[str, object]:
    _assert_launch_contract(profile, identity, env, runner, agent_cfg)
    _validate_provenance(identity, provenance)
    return {
        "schema": _LAUNCH_SCHEMA,
        "preset": preset,
        "identity": dict(identity),
        "contract_declaration_sha256": contract_declaration_sha256,
        "provenance": dict(provenance),
        "tracking_curriculum": None,
    }


def _complete_record(
    preset: str,
    profile: Mapping[str, object],
    identity: Mapping[str, str],
    env: object,
    runner: object,
    checkpoint_path: Path,
    contract_declaration_sha256: str,
) -> dict[str, object]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise ValueError(f"Final checkpoint is missing or symbolic: {checkpoint_path}.")
    runner_summary = _runner_summary(runner)
    collection = _validate_collection(profile, runner_summary)
    _validate_metrics(profile, runner_summary)
    learner = _learner_snapshot(env, runner)
    _validate_learner(profile, learner, collection["expected_update_calls"])
    return {
        "schema": _COMPLETE_SCHEMA,
        "preset": preset,
        "identity": dict(identity),
        "contract_declaration_sha256": contract_declaration_sha256,
        "runner": runner_summary,
        "learner": learner,
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": _file_sha256(checkpoint_path),
        },
    }


def _validation_record(
    preset: str,
    identity: Mapping[str, str],
    env: object,
    runner: object,
    checkpoint_path: Path,
    contract_declaration_sha256: str,
) -> dict[str, object]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise ValueError(f"Validation checkpoint is missing or symbolic: {checkpoint_path}.")
    load = runner.checkpoint_load_summary()
    return {
        "schema": _VALIDATION_SCHEMA,
        "preset": preset,
        "identity": dict(identity),
        "contract_declaration_sha256": contract_declaration_sha256,
        "learner": _learner_snapshot(env, runner),
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": _file_sha256(checkpoint_path),
            "mmap": load["mmap"],
            "strict_load": load["strict"],
            "map_location": load["map_location"],
            "environment_resume": load["environment_resume"],
            "environment_state_dict_is_none": load["environment_state_dict_is_none"],
        },
    }


def training_callback(
    *,
    stage: str,
    env_cfg: object,
    agent_cfg: object,
    configured_env_cfg: object,
    env: object,
    runner: object,
    log_dir: Path,
    checkpoint_path: Path | None = None,
) -> None:
    """Collect or validate one immutable Phase 3F learner-integration record."""
    preset = _preset(configured_env_cfg)
    if _preset(env_cfg) != preset:
        raise ValueError("Environment construction changed the selected motion preset.")
    profile = _contract_profile(preset)
    contract_declaration_sha256 = _contract_declaration_sha256()
    identity, provenance = _live_identity_evidence(preset, configured_env_cfg, env, runner, agent_cfg)

    if stage == "prepare":
        _assert_runtime_contract(profile, env, runner, agent_cfg)
        record = _identity_freeze_record(
            preset,
            profile,
            identity,
            provenance,
            contract_declaration_sha256,
        )
        _write_json_exclusive(log_dir / "phase3f_identity_freeze.json", record)
        return

    if stage == "launch":
        record = _launch_record(
            preset,
            profile,
            identity,
            env,
            runner,
            agent_cfg,
            provenance,
            contract_declaration_sha256,
        )
        _write_json_exclusive(_record_paths(log_dir)["launch"], record)
        return

    if checkpoint_path is None:
        raise ValueError(f"Phase 3F callback stage {stage!r} requires checkpoint_path.")
    evidence_root = checkpoint_path.expanduser().resolve().parent
    paths = _record_paths(evidence_root)
    launch = _load_json(paths["launch"], "Phase 3F launch record")
    if (
        launch.get("preset") != preset
        or launch.get("identity") != identity
        or launch.get("provenance") != provenance
        or launch.get("contract_declaration_sha256") != contract_declaration_sha256
    ):
        raise ValueError(
            "Phase 3F identity, provenance, or contract declaration changed before completion or validation."
        )

    if stage == "complete":
        complete = _complete_record(
            preset, profile, identity, env, runner, checkpoint_path, contract_declaration_sha256
        )
        _write_json_exclusive(paths["complete"], complete)
        return

    if stage != "validate":
        raise ValueError(f"Unsupported Phase 3F callback stage: {stage!r}.")
    complete = _load_json(paths["complete"], "Phase 3F completion record")
    validation = _validation_record(preset, identity, env, runner, checkpoint_path, contract_declaration_sha256)
    receipt = validate_smoke_records(profile, launch, complete, validation)
    receipt["records_sha256"] = {
        "launch": _json_sha256(launch),
        "complete": _json_sha256(complete),
        "validation": _json_sha256(validation),
    }
    _write_json_exclusive(paths["validation"], validation)
    _write_json_exclusive(paths["receipt"], receipt)


def main(argv: list[str] | None = None) -> int:
    """Freeze live native identities into a new immutable contract candidate."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("command", choices=("freeze_contract",))
    parser.add_argument("--contract", type=Path, default=_CONTRACT)
    parser.add_argument("--identity_record", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    freeze_contract(args.contract, args.identity_record, args.output)
    return 0


__all__ = ["expected_update_groups", "freeze_contract", "training_callback", "validate_smoke_records"]


if __name__ == "__main__":
    raise SystemExit(main())
