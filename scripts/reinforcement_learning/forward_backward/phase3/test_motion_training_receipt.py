# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the closed Phase 3F learner-integration receipt."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "motion_training_receipt.py"
CONTRACT = Path(__file__).parent / "fixtures" / "motion_training_smoke_contract_v2.json"
RUNTIME = Path(__file__).parent / "fixtures" / "runtime"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _bundle(value: dict[str, object]) -> dict[str, object]:
    return {**value, "bundle_sha256": _canonical_sha256(value)}


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


_EMPTY_SHA256 = _canonical_sha256({})
_ENVIRONMENT_PAYLOAD = {
    "schema": "forward_backward_phase3_motion_environment_dependency_identity_v9",
    "preset": "fixture",
    "resolved_axes": {},
    "resolved_axes_sha256": _EMPTY_SHA256,
    "resolved_configuration": {},
    "resolved_configuration_sha256": _EMPTY_SHA256,
    "runtime_dependencies": {},
    "runtime_dependencies_sha256": _EMPTY_SHA256,
    "python_sources": {},
    "robot_assets": {},
}
_ENVIRONMENT_IDENTITY = _bundle(_ENVIRONMENT_PAYLOAD)
_ENVIRONMENT_SEMANTIC_SHA256 = _canonical_sha256(
    {
        name: _ENVIRONMENT_PAYLOAD[name]
        for name in (
            "schema",
            "preset",
            "resolved_axes",
            "resolved_axes_sha256",
            "resolved_configuration",
            "resolved_configuration_sha256",
            "python_sources",
            "robot_assets",
        )
    }
)
_LEARNER_CODE_IDENTITY = _bundle(
    {"python_file_count": 1, "python_files": [{"path": "algorithm.py", "sha256": "a" * 64}]}
)
_LEARNER_RUNTIME_IDENTITY = _bundle(
    {
        "schema": "forward_backward_phase3f_learner_runtime_identity_v1",
        "packages": {
            name: {
                "module_version": "1.0",
                "distribution_version": "1.0",
                "module_source_sha256": digest * 64,
            }
            for name, digest in (("gymnasium", "b"), ("tensordict", "c"))
        },
    }
)
_TASK_BRIDGE_IDENTITY = _bundle(
    {
        "source_owner_count": 3,
        "source_owners": {
            name: {"owner": f"fixture:{name}", "source_sha256": digest * 64}
            for name, digest in (
                ("environment_wrapper", "d"),
                ("motion_expert_provider", "e"),
                ("motion_runner_config", "f"),
            )
        },
    }
)
PROVENANCE = {
    "environment": {
        "dependency_identity": _ENVIRONMENT_IDENTITY,
        "semantic_sha256": _ENVIRONMENT_SEMANTIC_SHA256,
    },
    "learner_code": _LEARNER_CODE_IDENTITY,
    "learner_runtime": _LEARNER_RUNTIME_IDENTITY,
    "task_bridge": _TASK_BRIDGE_IDENTITY,
}
IDENTITY = {
    "environment_dependency_bundle_sha256": _ENVIRONMENT_IDENTITY["bundle_sha256"],
    "resolved_axes_sha256": _EMPTY_SHA256,
    "runner_source_sha256": "4" * 64,
    "learner_code_bundle_sha256": _LEARNER_CODE_IDENTITY["bundle_sha256"],
    "learner_runtime_bundle_sha256": _LEARNER_RUNTIME_IDENTITY["bundle_sha256"],
    "task_bridge_code_bundle_sha256": _TASK_BRIDGE_IDENTITY["bundle_sha256"],
    "checkpoint_schema_sha256": "5" * 64,
    "task_table_sha256": "6" * 64,
    "expert_schema_sha256": "7" * 64,
    "observation_schema_sha256": "8" * 64,
}


def _module():
    spec = importlib.util.spec_from_file_location("motion_training_receipt", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _profile() -> dict[str, object]:
    return {
        "collection": {
            "num_envs": 4,
            "steps_per_iteration": 2,
            "iterations": 4,
            "expected_transitions": 32,
            "random_action_transitions": 8,
            "updates_per_group": 3,
            "expected_update_groups": 2,
            "expected_update_calls": 6,
        },
        "closed_input_identity": dict(IDENTITY),
        "environment_semantic_sha256": _ENVIRONMENT_SEMANTIC_SHA256,
        "learner": {
            "required_metrics": ["actor/loss", "fb/loss"],
            "expected_version_names": ["actor", "representation", "target"],
            "batch_size": 8,
            "context_buffer_capacity": 16,
        },
        "checkpoint": {
            "iteration": 4,
            "filename": "model_4.pt",
            "strict_map_location": "cpu",
            "strict_mmap": True,
            "environment_resume": "restart",
        },
        "device_scope": {
            "simulator": "cuda:0",
            "learner": "cuda:0",
            "replay": "cuda:0",
            "expert": "cuda:0",
            "dtype": "torch.float32",
        },
    }


def _records() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    stage_identity = {**IDENTITY, "environment_semantic_sha256": _ENVIRONMENT_SEMANTIC_SHA256}
    learner = {
        "update_step": 6,
        "versions": {"actor": 6, "representation": 6, "target": 6},
        "context_buffer_size": 16,
        "replay_contract_errors": False,
        "replay_terminal_overflow": False,
        "device_scope": {
            "simulator": "cuda:0",
            "learner": "cuda:0",
            "replay": "cuda:0",
            "expert": "cuda:0",
            "dtype": "torch.float32",
        },
    }
    launch = {
        "schema": "forward_backward_phase3f_motion_training_launch_v1",
        "preset": "fixture",
        "identity": dict(stage_identity),
        "contract_declaration_sha256": "0" * 64,
        "provenance": copy.deepcopy(PROVENANCE),
        "lifecycle_extension": None,
    }
    complete = {
        "schema": "forward_backward_phase3f_motion_training_complete_v1",
        "preset": "fixture",
        "identity": dict(stage_identity),
        "contract_declaration_sha256": "0" * 64,
        "runner": {
            "completed_iterations": 4,
            "collected_transitions": 32,
            "update_calls": 6,
            "metric_names": ["actor/loss", "fb/loss"],
            "all_metrics_finite": True,
            "last_metrics": {"actor/loss": 0.5, "fb/loss": -1.0},
            "action_statistics_finite": True,
            "actor_state_checks": {
                "actor_network_parameters": {
                    "tensor_count": 4,
                    "scalar_count": 32,
                    "all_finite": True,
                },
                "action_distribution_cached_params": {
                    "tensor_count": 2,
                    "scalar_count": 12,
                    "all_finite": True,
                },
            },
        },
        "learner": copy.deepcopy(learner),
        "checkpoint": {"path": "/tmp/model_4.pt", "bytes": 123, "sha256": "9" * 64},
    }
    validation = {
        "schema": "forward_backward_phase3f_motion_training_validation_v1",
        "preset": "fixture",
        "identity": dict(stage_identity),
        "contract_declaration_sha256": "0" * 64,
        "learner": copy.deepcopy(learner),
        "checkpoint": {
            "path": "/tmp/model_4.pt",
            "bytes": 123,
            "sha256": "9" * 64,
            "strict_load": True,
            "mmap": True,
            "map_location": "cpu",
            "environment_resume": "restart",
            "environment_state_dict_is_none": True,
        },
    }
    return launch, complete, validation


def test_update_group_math_uses_strict_post_seed_boundary() -> None:
    """The source seed boundary itself remains actor-only before the first updates."""
    assert _module().expected_update_groups(_profile()["collection"]) == 2


def test_valid_records_close_every_declared_phase3f_claim() -> None:
    launch, complete, validation = _records()

    receipt = _module().validate_smoke_records(_profile(), launch, complete, validation)

    assert receipt["schema"] == "forward_backward_phase3f_motion_training_receipt_v1"
    assert receipt["status"] == "passed"
    assert receipt["identity"] == {**IDENTITY, "environment_semantic_sha256": _ENVIRONMENT_SEMANTIC_SHA256}
    assert receipt["contract_declaration_sha256"] == "0" * 64
    assert receipt["provenance"] == PROVENANCE
    assert receipt["checkpoint"]["strict_load"] is True
    assert receipt["checkpoint"]["filename"] == "model_4.pt"
    assert "path" not in receipt["checkpoint"]


def test_expected_identity_includes_environment_semantics() -> None:
    """The portable environment meaning must travel through every callback stage."""
    module = _module()
    profile = _profile()
    profile["environment_semantic_sha256"] = "a" * 64

    identity = module._expected_identity(profile)

    assert identity["environment_semantic_sha256"] == "a" * 64
    assert set(identity) == {*IDENTITY, "environment_semantic_sha256"}


def test_contract_declaration_digest_binds_exact_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A callback stage must detect any declaration edit after launch."""
    module = _module()
    contract = tmp_path / "contract.json"
    contract.write_text('{"version":1}\n')
    monkeypatch.setattr(module, "_CONTRACT", contract)
    first = module._contract_declaration_sha256()

    contract.write_text('{"version":2}\n')

    assert module._contract_declaration_sha256() != first


def test_resolved_agent_config_identity_excludes_output_run_name() -> None:
    """Changing only the log-directory suffix must not change learner semantics."""
    module = _module()

    class _AgentCfg:
        def __init__(self, run_name: str) -> None:
            self.run_name = run_name

        def to_dict(self) -> dict[str, object]:
            return {"seed": 7, "run_name": self.run_name, "algorithm": {"batch_size": 1024}}

    first = module._resolved_agent_config_sha256(_AgentCfg("first-output"))
    second = module._resolved_agent_config_sha256(_AgentCfg("second-output"))

    assert first == second


@pytest.mark.parametrize(
    "preset,filename",
    (
        ("smpl_cmu", "motion_training_smpl_cmu_v1.json"),
        ("g1_lafan", "motion_training_g1_lafan_v1.json"),
    ),
)
def test_published_native_receipt_is_authentic_and_reports_current_contract_compatibility(
    preset: str, filename: str
) -> None:
    """Published receipts retain their measured contract even after the declaration advances."""
    module = _module()
    contract = json.loads(CONTRACT.read_text())
    receipt = json.loads((RUNTIME / filename).read_text())

    assert receipt["schema"] == "forward_backward_phase3f_motion_training_receipt_v1"
    assert receipt["status"] == "passed"
    assert receipt["preset"] == preset
    identity = module._identity_digests(receipt["identity"])
    assert module._validate_provenance(identity, receipt["provenance"]) == receipt["provenance"]
    assert _is_sha256(receipt["contract_declaration_sha256"])
    assert set(receipt["records_sha256"]) == {"launch", "complete", "validation"}
    assert all(_is_sha256(digest) for digest in receipt["records_sha256"].values())

    collection = receipt["collection"]
    assert collection["expected_transitions"] == (
        collection["num_envs"] * collection["steps_per_iteration"] * collection["iterations"]
    )
    assert collection["expected_update_groups"] == module.expected_update_groups(collection)
    assert collection["expected_update_calls"] == (
        collection["expected_update_groups"] * collection["updates_per_group"]
    )

    checkpoint = receipt["checkpoint"]
    assert checkpoint["filename"] == f"model_{collection['iterations']}.pt"
    assert checkpoint["map_location"] == "cpu"
    assert checkpoint["mmap"] is True
    assert checkpoint["environment_resume"] == "restart"
    assert checkpoint["environment_state_dict_is_none"] is True
    assert checkpoint["strict_load"] is True
    assert checkpoint["bytes"] > 0
    assert _is_sha256(checkpoint["sha256"])

    learner = receipt["learner"]
    update_calls = collection["expected_update_calls"]
    assert learner["update_step"] == update_calls
    assert set(learner["versions"].values()) == {update_calls}
    assert learner["replay_contract_errors"] is False
    assert learner["replay_terminal_overflow"] is False
    assert set(learner["device_scope"]) == {"simulator", "learner", "replay", "expert", "dtype"}

    runner = receipt["runner"]
    assert runner["completed_iterations"] == collection["iterations"]
    assert runner["collected_transitions"] == collection["expected_transitions"]
    assert runner["update_calls"] == update_calls
    assert runner["all_metrics_finite"] is True
    assert runner["action_statistics_finite"] is True
    assert set(runner["last_metrics"]) == set(runner["metric_names"])
    assert all(math.isfinite(value) for value in runner["last_metrics"].values())
    assert all(check["all_finite"] is True for check in runner["actor_state_checks"].values())

    current_contract_sha256 = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    compatibility = (
        "exact_contract_match"
        if receipt["contract_declaration_sha256"] == current_contract_sha256
        else "historical_contract_differs_requires_fresh_smoke"
    )
    assert compatibility in {"exact_contract_match", "historical_contract_differs_requires_fresh_smoke"}
    if compatibility == "exact_contract_match":
        profile = contract["profiles"][preset]
        assert receipt["identity"] == {
            **profile["closed_input_identity"],
            "environment_semantic_sha256": profile["environment_semantic_sha256"],
        }
        assert receipt["collection"] == profile["collection"]


@pytest.mark.parametrize(
    "record,path,value,match",
    (
        ("launch", ("schema",), "wrong", "launch record"),
        ("complete", ("schema",), "wrong", "completion record"),
        ("validation", ("schema",), "wrong", "validation record"),
        ("launch", ("lifecycle_extension",), {"class_name": "tracking"}, "lifecycle"),
        ("complete", ("contract_declaration_sha256",), "a" * 64, "contract declaration"),
        (
            "launch",
            ("provenance", "task_bridge", "source_owners", "environment_wrapper", "source_sha256"),
            "1" * 64,
            "member manifest",
        ),
        (
            "launch",
            ("provenance", "environment", "semantic_sha256"),
            "2" * 64,
            "semantic digest",
        ),
        ("complete", ("identity", "environment_dependency_bundle_sha256"), "a" * 64, "identity"),
        ("validation", ("identity", "resolved_axes_sha256"), "b" * 64, "identity"),
        ("complete", ("runner", "update_calls"), 5, "update"),
        ("complete", ("runner", "metric_names"), ["actor/loss"], "metric"),
        ("complete", ("runner", "all_metrics_finite"), False, "finite"),
        ("complete", ("runner", "last_metrics", "fb/loss"), math.nan, "finite"),
        ("complete", ("runner", "action_statistics_finite"), False, "action"),
        (
            "complete",
            ("runner", "actor_state_checks", "actor_network_parameters", "all_finite"),
            False,
            "Actor state",
        ),
        ("complete", ("learner", "versions", "actor"), 5, "version"),
        ("complete", ("learner", "context_buffer_size"), 8, "context"),
        ("complete", ("learner", "replay_contract_errors"), True, "replay"),
        ("validation", ("learner", "replay_terminal_overflow"), True, "replay"),
        ("validation", ("checkpoint", "mmap"), False, "mmap"),
        ("validation", ("checkpoint", "strict_load"), False, "strict"),
        ("validation", ("checkpoint", "map_location"), "cuda:0", "map_location"),
        ("validation", ("checkpoint", "environment_resume"), "exact", "restart"),
        ("validation", ("checkpoint", "environment_state_dict_is_none"), False, "environment state"),
        ("validation", ("checkpoint", "sha256"), "short", "SHA-256"),
        ("complete", ("checkpoint", "sha256"), "a" * 64, "checkpoint"),
        ("validation", ("learner", "device_scope", "replay"), "cpu", "device"),
    ),
)
def test_validator_rejects_each_unproven_or_drifted_claim(
    record: str,
    path: tuple[str, ...],
    value: object,
    match: str,
) -> None:
    """Each success field must be consumed rather than merely listed in JSON."""
    launch, complete, validation = _records()
    records = {"launch": launch, "complete": complete, "validation": validation}
    target = records[record]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=match):
        _module().validate_smoke_records(_profile(), launch, complete, validation)


def test_validator_rejects_checkpoint_and_post_load_state_drift() -> None:
    launch, complete, validation = _records()
    validation["checkpoint"]["path"] = "/tmp/other.pt"
    with pytest.raises(ValueError, match="checkpoint"):
        _module().validate_smoke_records(_profile(), launch, complete, validation)

    launch, complete, validation = _records()
    validation["learner"]["update_step"] = 5
    with pytest.raises(ValueError, match="update_step"):
        _module().validate_smoke_records(_profile(), launch, complete, validation)
