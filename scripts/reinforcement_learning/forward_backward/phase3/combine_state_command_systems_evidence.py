# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Combine CPU and CUDA StateCommand measurements into one systems gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

_MEASUREMENT_SCHEMA = "forward_backward_phase3_state_command_systems_measurement_v1"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _common_configuration(record: dict) -> dict:
    configuration = dict(record["configuration"])
    configuration.pop("device")
    return configuration


def _current_allocations(record: dict, operation: str):
    for profile_result in record["results"].values():
        yield profile_result["current"][operation]


def _link_migration_manifest(migration_path: Path, record_path: Path) -> None:
    """Relink the saved migration decision to one generated systems record."""
    migration = _load(migration_path)
    if migration.get("schema") != "forward_backward.phase3.state_command_migration.v1":
        raise ValueError("StateCommand migration manifest has an unsupported schema.")
    systems = migration.get("systems_evidence")
    if not isinstance(systems, dict) or systems.get("record") != record_path.name:
        raise ValueError("StateCommand migration manifest references a different systems record.")
    systems["record_sha256"] = _sha256(record_path)
    migration_path.write_text(json.dumps(migration, indent=2) + "\n")


def combine(cpu_path: Path, cuda_path: Path) -> dict:
    cpu = _load(cpu_path)
    cuda = _load(cuda_path)
    if cpu["schema"] != _MEASUREMENT_SCHEMA or cuda["schema"] != _MEASUREMENT_SCHEMA:
        raise ValueError("StateCommand systems inputs must be device measurement records.")
    if cpu["runtime"]["device"] != "cpu" or not cuda["runtime"]["device"].startswith("cuda"):
        raise ValueError("StateCommand systems inputs must be ordered CPU then CUDA.")
    identity_fields = (
        "historical_revision",
        "historical_state_command_sha256",
        "current_state_command_sha256",
        "current_state_command_cfg_sha256",
        "benchmark_script_sha256",
    )
    if any(cpu["identity"][name] != cuda["identity"][name] for name in identity_fields):
        raise ValueError("CPU and CUDA measurements must cover identical code.")
    if _common_configuration(cpu) != _common_configuration(cuda):
        raise ValueError("CPU and CUDA measurements must use identical non-device configuration.")

    cpu_passed = cpu["decision"]["gate_passed"]
    cuda_passed = cuda["decision"]["gate_passed"]
    semantic_passed = all(cpu["semantic_equivalence"].values()) and all(cuda["semantic_equivalence"].values())
    pointer_stability_passed = all(
        allocation["owned_storage_pointers_stable"]
        for record in (cpu, cuda)
        for operation in (
            "randomized_resample_allocations",
            "pinned_resample_allocations",
            "update_allocations",
        )
        for allocation in _current_allocations(record, operation)
    )
    cuda_update_allocation_free = all(
        allocation["positive_self_device_memory_bytes"] == 0 and allocation["cuda_peak_additional_bytes"] == 0
        for allocation in _current_allocations(cuda, "update_allocations")
    )
    cuda_resample_allocation_accounted = all(
        allocation["positive_self_device_memory_bytes"] >= 0 and allocation["cuda_peak_additional_bytes"] >= 0
        for operation in ("randomized_resample_allocations", "pinned_resample_allocations")
        for allocation in _current_allocations(cuda, operation)
    )
    gate_passed = (
        cpu_passed
        and cuda_passed
        and semantic_passed
        and pointer_stability_passed
        and cuda_update_allocation_free
        and cuda_resample_allocation_accounted
    )

    script_path = Path(__file__).resolve()
    identities = {name: cpu["identity"][name] for name in identity_fields}
    return {
        "schema": "forward_backward_phase3_state_command_systems_v1",
        "status": "passed" if gate_passed else "failed",
        "scope": {
            "measured": (
                "Exact historical/final StateCommand shell construction, resample, update, storage, and allocation "
                "behavior on CPU and CUDA."
            ),
            "domain_boundary": (
                "Position-like and Factory-like tensor shapes with identical synthetic payload work isolate shell "
                "ownership."
            ),
            "not_claimed": [
                "simulator reset-write throughput",
                "Position sensor and robot-state kernel throughput",
                "Factory symmetry and Warp kernel throughput",
                "historical observation-cache materialization throughput",
            ],
        },
        "baseline": cpu["baseline"],
        "identity": {
            **identities,
            "combiner_script_sha256": _sha256(script_path),
            "cpu_measurement_sha256": _sha256(cpu_path),
            "cuda_measurement_sha256": _sha256(cuda_path),
        },
        "measurements": {
            "cpu": {
                "file": cpu_path.name,
                "runtime": cpu["runtime"],
                "configuration_sha256": cpu["identity"]["configuration_sha256"],
                "comparison": cpu["comparison"],
                "decision": cpu["decision"],
            },
            "cuda": {
                "file": cuda_path.name,
                "runtime": cuda["runtime"],
                "configuration_sha256": cuda["identity"]["configuration_sha256"],
                "comparison": cuda["comparison"],
                "decision": cuda["decision"],
                "current_allocations": {
                    profile_name: {
                        operation: profile_result["current"][operation]
                        for operation in (
                            "randomized_resample_allocations",
                            "pinned_resample_allocations",
                            "update_allocations",
                        )
                    }
                    for profile_name, profile_result in cuda["results"].items()
                },
            },
        },
        "configuration": {
            **_common_configuration(cpu),
            "devices": ["cpu", cuda["runtime"]["device"]],
        },
        "decision": {
            "gate_passed": gate_passed,
            "cpu_reference_gate_passed": cpu_passed,
            "cuda_reference_gate_passed": cuda_passed,
            "semantic_equivalence_passed": semantic_passed,
            "owned_storage_pointer_stability_passed": pointer_stability_passed,
            "cuda_update_allocation_free": cuda_update_allocation_free,
            "cuda_resample_allocation_accounted": cuda_resample_allocation_accounted,
            "resample_allocation_policy": (
                "Reset-time resample allocations are measured and reported; the per-step update must allocate no CUDA "
                "tensor storage."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--migration_manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = combine(args.cpu, args.cuda)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if args.migration_manifest is not None:
        _link_migration_manifest(args.migration_manifest, args.output)
    print(json.dumps(record["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
