# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Phase 2 long-form evaluation evidence."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from evaluation_records import (
    EvaluationRecord,
    join_paired_records,
    normalized_trapezoid_auc,
    paired_upper_bound,
    read_records,
    validate_manifest,
    validate_records,
    write_records,
)


def _records(implementation: str, offset: float = 0.0) -> list[EvaluationRecord]:
    return [
        EvaluationRecord(
            implementation=implementation,
            training_seed=seed,
            evaluation_seed=17,
            motion_id=motion,
            checkpoint_transition=500_000,
            terminal_profile="next_step_exact",
            metric_name=metric,
            metric_value=float(index) + offset,
            run_id=f"{implementation}-{seed}",
            evaluator_hash="evaluator",
            dataset_hash="dataset",
        )
        for seed in (0, 1)
        for index, (motion, metric) in enumerate(
            (("motion-0", "emd"), ("motion-0", "success"), ("motion-1", "emd"), ("motion-1", "success"))
        )
    ]


def test_records_round_trip_with_frozen_columns(tmp_path) -> None:
    """CSV evidence should preserve every typed identity field."""
    records = _records("reference")
    path = tmp_path / "metrics.csv"

    write_records(path, records)

    assert read_records(path) == records
    with pytest.raises(FileExistsError):
        write_records(path, records)


def test_rectangular_cardinality_rejects_missing_and_duplicate_rows() -> None:
    """Every motion should have one finite row for every emitted metric."""
    records = _records("reference")[:4]
    assert (
        validate_records(
            records,
            expected_motion_ids=("motion-0", "motion-1"),
            expected_metric_names=("emd", "success"),
        )["num_records"]
        == 4
    )

    with pytest.raises(ValueError, match="same metric names"):
        validate_records(records[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_records([*records, records[0]])
    with pytest.raises(ValueError, match="finite"):
        replace(records[0], metric_value=float("nan"))


def test_pairing_fails_instead_of_dropping_one_side() -> None:
    """The paired join should reject missing or duplicated comparison keys."""
    reference = _records("reference")
    candidate = _records("candidate", 0.25)
    pairs = join_paired_records(reference, candidate)

    assert len(pairs) == len(reference)
    assert {pair.difference for pair in pairs} == {0.25}
    with pytest.raises(ValueError, match="Pairing keys differ"):
        join_paired_records(reference, candidate[:-1])


def test_bootstrap_and_auc_are_deterministic() -> None:
    """Frozen analysis primitives should return stable values."""
    pairs = join_paired_records(_records("reference"), _records("candidate", 0.25))

    assert paired_upper_bound(pairs) == pytest.approx(0.25)
    assert normalized_trapezoid_auc((0, 10, 20), (1.0, 2.0, 3.0)) == pytest.approx(2.0)


def test_manifest_must_match_complete_record_identity(tmp_path) -> None:
    """A manifest should name the exact rows it claims as complete."""
    records = _records("reference")[:4]
    manifest = {
        "schema": "forward_backward_phase2_manifest_v1",
        "run_id": "reference-0",
        "implementation": "reference",
        "training_seed": 0,
        "evaluation_seed": 17,
        "checkpoint_transition": 500_000,
        "terminal_profile": "next_step_exact",
        "evaluator_hash": "evaluator",
        "dataset_hash": "dataset",
        "expected_motion_count": 2,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    assert validate_manifest(path, records)["num_motions"] == 2
    manifest["dataset_hash"] = "other"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dataset_hash"):
        validate_manifest(path, records)
