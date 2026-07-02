# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for persisted CPU and CUDA StateCommand systems evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
_PHASE3 = Path(__file__).parent
_FIXTURES = _PHASE3 / "fixtures"
_EVIDENCE = _FIXTURES / "state_command_systems_v1.json"
_CPU = _FIXTURES / "state_command_systems_cpu_v1.json"
_CUDA = _FIXTURES / "state_command_systems_cuda_v1.json"
_MIGRATION = _FIXTURES / "state_command_migration_v1.json"
_BENCHMARK = _PHASE3 / "benchmark_state_command_delegation.py"
_COMBINER = _PHASE3 / "combine_state_command_systems_evidence.py"
_STATE_COMMAND = (
    _ROOT / "source/isaaclab_tasks/isaaclab_tasks/core/multi_task/mdp/commands/state_command/state_command.py"
)
_STATE_COMMAND_CFG = (
    _ROOT / "source/isaaclab_tasks/isaaclab_tasks/core/multi_task/mdp/commands/state_command/state_command_cfg.py"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _common_configuration(record: dict) -> dict:
    configuration = dict(record["configuration"])
    configuration.pop("device")
    return configuration


def test_state_command_systems_record_closes_over_both_devices_and_exact_code() -> None:
    """The combined record must close over exact code plus CPU and CUDA bytes."""
    evidence = _load(_EVIDENCE)
    cpu = _load(_CPU)
    cuda = _load(_CUDA)
    migration = _load(_MIGRATION)
    systems = migration["systems_evidence"]
    identity = evidence["identity"]

    assert evidence["schema"] == "forward_backward_phase3_state_command_systems_v1"
    assert cpu["schema"] == cuda["schema"] == ("forward_backward_phase3_state_command_systems_measurement_v1")
    assert identity["historical_revision"] == migration["repository_revision"]
    assert (
        identity["historical_state_command_sha256"]
        == migration["pre_migration_hashes"][
            "source/isaaclab_tasks/isaaclab_tasks/core/multi_task/mdp/commands/state_command/state_command.py"
        ]
    )
    assert identity["current_state_command_sha256"] == _sha256(_STATE_COMMAND)
    assert identity["current_state_command_cfg_sha256"] == _sha256(_STATE_COMMAND_CFG)
    assert identity["benchmark_script_sha256"] == _sha256(_BENCHMARK)
    assert identity["combiner_script_sha256"] == _sha256(_COMBINER)
    assert identity["cpu_measurement_sha256"] == _sha256(_CPU)
    assert identity["cuda_measurement_sha256"] == _sha256(_CUDA)
    assert cpu["identity"]["benchmark_script_sha256"] == _sha256(_BENCHMARK)
    assert cuda["identity"]["benchmark_script_sha256"] == _sha256(_BENCHMARK)
    assert cpu["identity"]["configuration_sha256"] == _canonical_sha256(cpu["configuration"])
    assert cuda["identity"]["configuration_sha256"] == _canonical_sha256(cuda["configuration"])
    assert _common_configuration(cpu) == _common_configuration(cuda)

    assert systems["status"] == "passed"
    assert systems["record"] == _EVIDENCE.name
    assert systems["record_sha256"] == _sha256(_EVIDENCE)
    assert systems["device_records"] == {"cpu": _CPU.name, "cuda": _CUDA.name}


def test_state_command_combined_gate_separates_cuda_update_and_resample_allocations() -> None:
    """CUDA per-step update must allocate nothing; reset-time allocations stay visible."""
    evidence = _load(_EVIDENCE)
    cpu = _load(_CPU)
    cuda = _load(_CUDA)
    decision = evidence["decision"]

    assert evidence["status"] == "passed"
    assert all(
        decision[name] is True
        for name in (
            "gate_passed",
            "cpu_reference_gate_passed",
            "cuda_reference_gate_passed",
            "semantic_equivalence_passed",
            "owned_storage_pointer_stability_passed",
            "cuda_update_allocation_free",
            "cuda_resample_allocation_accounted",
        )
    )
    assert cpu["runtime"]["device"] == "cpu"
    assert cuda["runtime"]["device"].startswith("cuda")
    assert cpu["status"] == cuda["status"] == "passed"
    assert cpu["decision"]["reference_ratio_gate_passed"] is True
    assert cuda["decision"]["reference_ratio_gate_passed"] is True
    assert evidence["baseline"]["full_domain_before_after"]["status"] == (
        "waived_not_reconstructable_as_controlled_comparison"
    )

    for record in (cpu, cuda):
        threshold = record["decision"]["no_material_regression_threshold_ratio"]
        for profile_result in record["comparison"].values():
            for key in record["decision"]["reference_ratio_gate_applies_to"]:
                assert profile_result[key] <= threshold
            assert profile_result["state_owned_byte_ratio"] < 1.0
        for profile_result in record["results"].values():
            current = profile_result["current"]
            historical = profile_result["historical_reference"]
            assert (
                current["construction"]["storage"]["state_command_owned_bytes"]
                < (historical["construction"]["storage"]["state_command_owned_bytes"])
            )
            for operation in ("randomized_resample", "pinned_resample", "update"):
                assert current[operation]["env_rows_per_second"] > 0.0
            for operation in (
                "randomized_resample_allocations",
                "pinned_resample_allocations",
                "update_allocations",
            ):
                allocation = current[operation]
                assert allocation["owned_storage_pointers_stable"] is True
                assert allocation["python_peak_bytes"] > 0
                assert allocation["positive_self_cpu_memory_bytes"] >= 0
                assert allocation["positive_self_device_memory_bytes"] >= 0
                assert allocation["cuda_peak_additional_bytes"] >= 0

    for profile_result in cpu["results"].values():
        update = profile_result["current"]["update_allocations"]
        assert update["positive_self_cpu_memory_bytes"] == 0

    for profile_result in cuda["results"].values():
        current = profile_result["current"]
        update = current["update_allocations"]
        assert update["positive_self_device_memory_bytes"] == 0
        assert update["cuda_peak_additional_bytes"] == 0
        for operation in ("randomized_resample_allocations", "pinned_resample_allocations"):
            allocation = current[operation]
            assert allocation["positive_self_device_memory_bytes"] > 0
            assert allocation["cuda_peak_additional_bytes"] > 0


def test_state_command_benchmark_is_rerunnable_with_repository_python(tmp_path: Path) -> None:
    """A tiny CPU invocation must reproduce the measurement schema and identities."""
    output = tmp_path / "state_command_systems_smoke.json"
    command = [
        sys.executable,
        str(_BENCHMARK),
        "--output",
        str(output),
        "--num_envs",
        "32",
        "--num_tasks",
        "64",
        "--batch_size",
        "16",
        "--warmup",
        "1",
        "--iterations",
        "2",
        "--repeats",
        "2",
        "--init_warmup",
        "1",
        "--init_iterations",
        "2",
        "--profile_iterations",
        "1",
        "--torch_threads",
        "1",
    ]
    subprocess.run(command, cwd=_ROOT, check=True, capture_output=True, text=True)
    record = _load(output)
    assert record["schema"] == "forward_backward_phase3_state_command_systems_measurement_v1"
    assert all(record["semantic_equivalence"].values())
    assert record["identity"]["benchmark_script_sha256"] == _sha256(_BENCHMARK)
    assert record["decision"]["owned_storage_pointer_stability_passed"] is True


def test_state_command_combiner_is_rerunnable(tmp_path: Path) -> None:
    """The saved device measurements must deterministically regenerate the gate."""
    output = tmp_path / "state_command_systems_combined.json"
    subprocess.run(
        [
            sys.executable,
            str(_COMBINER),
            "--cpu",
            str(_CPU),
            "--cuda",
            str(_CUDA),
            "--output",
            str(output),
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    regenerated = _load(output)
    persisted = _load(_EVIDENCE)
    assert regenerated == persisted
