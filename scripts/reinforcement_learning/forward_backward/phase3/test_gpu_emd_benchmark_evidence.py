# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Identity and accounting checks for the measured GPU EMD evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[4]
_EVIDENCE = Path(__file__).with_name("fixtures") / "runtime" / "g1_lafan_gpu_emd_benchmark_v2.json"
_PRODUCER = Path(__file__).with_name("generate_g1_gpu_emd_benchmark.py")
_FILES = {
    "producer.py": _PRODUCER,
    "forward_backward_tracking.py": _ROOT
    / "source/isaaclab_tasks/isaaclab_tasks/core/multi_task/rl/rsl_rl/forward_backward_tracking.py",
    "uniform_assignment_warp.py": _ROOT
    / "source/isaaclab_tasks/isaaclab_tasks/core/multi_task/metrics/impl/uniform_assignment_warp.py",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gpu_emd_evidence_is_authentic_and_reports_current_implementation() -> None:
    evidence = json.loads(_EVIDENCE.read_text())

    assert evidence["schema"] == "g1_lafan_gpu_emd_benchmark_v2"
    stored = evidence["implementation_sha256"]
    assert set(stored) == {"producer.py", "tracking.py", "uniform_emd_warp.py"}
    assert all(
        isinstance(digest, str) and len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)
        for digest in stored.values()
    )
    current = {name: _sha256(path) for name, path in _FILES.items()}
    fields = set(stored) | set(current)
    mismatched = sorted(name for name in fields if stored.get(name) != current.get(name))
    compatibility = {
        "status": "exact_producer_match" if not mismatched else "producer_changed_requires_fresh_benchmark",
        "mismatched_fields": mismatched,
    }
    assert compatibility["status"] in {"exact_producer_match", "producer_changed_requires_fresh_benchmark"}


def test_gpu_emd_evidence_accounts_for_fixed_workspace_and_full_workload() -> None:
    evidence = json.loads(_EVIDENCE.read_text())
    workload = evidence["workload"]
    contract = evidence["contract"]
    capacity = workload["assignment_batch_size"]
    frames = workload["frame_count_per_clip"]
    workspace_bytes = (
        4 * capacity * frames * frames
        + 2 * 4 * capacity * frames
        + 2 * 4 * capacity * frames * 29
        + 3 * 8 * capacity * (frames + 1)
        + 3 * 4 * capacity * (frames + 1)
        + 3 * 8 * capacity
    )

    assert workload["clip_count"] == sum(workload["rollout_chunk_sizes"])
    assert workload["assignment_batch_size"] == workload["clip_count"] == 862
    assert workload["assignment_calls"] == len(workload["feature_widths"]) == 2
    assert contract["workspace_bytes"] == workspace_bytes
    assert contract["host_trajectory_copies"] == 0
    assert contract["per_step_host_synchronizations"] == 0
    assert contract["motion_assignment"] == "released_random_shuffle_first_env"
    assert contract["rollout_horizon"] == "maximum_unique_clip_length_per_chunk"

    measurement = evidence["measurements"]
    assert len(measurement["calls"]) == workload["assignment_calls"]
    assert {row["batch_size"] for row in measurement["calls"]} == {workload["clip_count"]}
    assert measurement["full_two_call_seconds"] == pytest.approx(
        sum(row["seconds"] for row in measurement["calls"]),
        rel=0.0,
        abs=0.01,
    )
