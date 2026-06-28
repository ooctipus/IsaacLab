# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate and compare immutable Phase 2 evaluation records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

RECORD_FIELDS = (
    "implementation",
    "training_seed",
    "evaluation_seed",
    "motion_id",
    "checkpoint_transition",
    "terminal_profile",
    "metric_name",
    "metric_value",
    "run_id",
    "evaluator_hash",
    "dataset_hash",
)
PAIRING_FIELDS = (
    "training_seed",
    "evaluation_seed",
    "motion_id",
    "checkpoint_transition",
    "terminal_profile",
    "metric_name",
    "evaluator_hash",
    "dataset_hash",
)


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One native metric for one motion at one checkpoint."""

    implementation: str
    training_seed: int
    evaluation_seed: int
    motion_id: str
    checkpoint_transition: int
    terminal_profile: str
    metric_name: str
    metric_value: float
    run_id: str
    evaluator_hash: str
    dataset_hash: str

    def __post_init__(self) -> None:
        """Reject records that cannot be paired or audited."""
        text_fields = (
            self.implementation,
            self.motion_id,
            self.terminal_profile,
            self.metric_name,
            self.run_id,
            self.evaluator_hash,
            self.dataset_hash,
        )
        if any(not value for value in text_fields):
            raise ValueError("Evaluation record identity fields must not be empty.")
        if self.training_seed < 0 or self.evaluation_seed < 0 or self.checkpoint_transition < 0:
            raise ValueError("Seeds and checkpoint_transition must be non-negative.")
        if not math.isfinite(self.metric_value):
            raise ValueError("metric_value must be finite.")

    @property
    def pairing_key(self) -> tuple[object, ...]:
        """Return the exact implementation-independent pairing key."""
        return tuple(getattr(self, field) for field in PAIRING_FIELDS)

    @property
    def identity_key(self) -> tuple[object, ...]:
        """Return the complete row identity excluding its numeric value."""
        return (self.implementation, *self.pairing_key, self.run_id)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> EvaluationRecord:
        """Parse one CSV or JSON mapping into its typed representation."""
        unknown = set(value).difference(RECORD_FIELDS)
        missing = set(RECORD_FIELDS).difference(value)
        if missing or unknown:
            raise ValueError(f"Record fields differ: missing={sorted(missing)}, unknown={sorted(unknown)}.")
        return cls(
            implementation=str(value["implementation"]),
            training_seed=int(value["training_seed"]),
            evaluation_seed=int(value["evaluation_seed"]),
            motion_id=str(value["motion_id"]),
            checkpoint_transition=int(value["checkpoint_transition"]),
            terminal_profile=str(value["terminal_profile"]),
            metric_name=str(value["metric_name"]),
            metric_value=float(value["metric_value"]),
            run_id=str(value["run_id"]),
            evaluator_hash=str(value["evaluator_hash"]),
            dataset_hash=str(value["dataset_hash"]),
        )


@dataclass(frozen=True, slots=True)
class PairedMetric:
    """One exact reference/candidate metric pair."""

    key: tuple[object, ...]
    reference: float
    candidate: float

    @property
    def difference(self) -> float:
        """Return candidate minus reference."""
        return self.candidate - self.reference


def read_records(path: str | Path) -> list[EvaluationRecord]:
    """Read records from a CSV file with the frozen column order."""
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RECORD_FIELDS:
            raise ValueError(f"Unexpected evaluation columns: {reader.fieldnames}.")
        return [EvaluationRecord.from_mapping(row) for row in reader]


def write_records(path: str | Path, records: Iterable[EvaluationRecord]) -> None:
    """Write records once without silently replacing evidence."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Evaluation output already exists: {destination}")
    with destination.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def validate_records(
    records: Sequence[EvaluationRecord],
    *,
    expected_motion_ids: Iterable[str] | None = None,
    expected_metric_names: Iterable[str] | None = None,
) -> dict[str, object]:
    """Validate uniqueness, cardinality, finiteness, and rectangular motion metrics."""
    if not records:
        raise ValueError("Evaluation output is empty.")
    seen: set[tuple[object, ...]] = set()
    metrics_by_motion: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.identity_key in seen:
            raise ValueError(f"Duplicate evaluation record: {record.identity_key}")
        seen.add(record.identity_key)
        metrics_by_motion[record.motion_id].add(record.metric_name)

    actual_motions = set(metrics_by_motion)
    if expected_motion_ids is not None:
        expected_motions = set(expected_motion_ids)
        if actual_motions != expected_motions:
            raise ValueError(
                f"Motion cardinality differs: missing={sorted(expected_motions - actual_motions)}, "
                f"unexpected={sorted(actual_motions - expected_motions)}."
            )
    metric_sets = {frozenset(names) for names in metrics_by_motion.values()}
    if len(metric_sets) != 1:
        raise ValueError("Every motion must emit the same metric names.")
    actual_metrics = set(next(iter(metric_sets)))
    if expected_metric_names is not None and actual_metrics != set(expected_metric_names):
        raise ValueError(f"Metric names differ: {sorted(actual_metrics)}.")

    expected_rows = len(actual_motions) * len(actual_metrics)
    if len(records) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rectangular rows, got {len(records)}.")
    return {
        "num_records": len(records),
        "num_motions": len(actual_motions),
        "metric_names": sorted(actual_metrics),
    }


def join_paired_records(
    reference: Sequence[EvaluationRecord],
    candidate: Sequence[EvaluationRecord],
) -> list[PairedMetric]:
    """Inner-join two complete implementations and fail on every dropped row."""
    reference_by_key = _index_pairing_keys(reference, "reference")
    candidate_by_key = _index_pairing_keys(candidate, "candidate")
    if reference_by_key.keys() != candidate_by_key.keys():
        raise ValueError(
            "Pairing keys differ: "
            f"reference_only={len(reference_by_key.keys() - candidate_by_key.keys())}, "
            f"candidate_only={len(candidate_by_key.keys() - reference_by_key.keys())}."
        )
    return [
        PairedMetric(key, reference_by_key[key].metric_value, candidate_by_key[key].metric_value)
        for key in sorted(reference_by_key, key=repr)
    ]


def paired_upper_bound(
    pairs: Sequence[PairedMetric],
    *,
    quantile: float = 0.95,
    num_replicates: int = 10_000,
    seed: int = 20_260_626,
    chunk_size: int = 1_024,
) -> float:
    """Return a deterministic paired percentile-bootstrap upper bound."""
    if not pairs or num_replicates < 1 or chunk_size < 1:
        raise ValueError("Bootstrap requires pairs, replicates, and a positive chunk size.")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between zero and one.")
    differences = np.asarray([pair.difference for pair in pairs], dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.empty(num_replicates, dtype=np.float64)
    for start in range(0, num_replicates, chunk_size):
        stop = min(start + chunk_size, num_replicates)
        indices = generator.integers(0, differences.size, size=(stop - start, differences.size))
        means[start:stop] = differences[indices].mean(axis=1)
    return float(np.quantile(means, quantile))


def normalized_trapezoid_auc(transitions: Sequence[int], values: Sequence[float]) -> float:
    """Compute the frozen transition-normalized trapezoid AUC."""
    x = np.asarray(transitions, dtype=np.int64)
    y = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError("AUC requires equal one-dimensional arrays with at least two points.")
    if np.any(np.diff(x) <= 0) or not np.all(np.isfinite(y)):
        raise ValueError("AUC transitions must increase strictly and values must be finite.")
    return float(np.trapezoid(y, x=x) / (x[-1] - x[0]))


def validate_manifest(path: str | Path, records: Sequence[EvaluationRecord]) -> dict[str, object]:
    """Validate a run manifest against the complete numeric record identity."""
    with Path(path).open() as stream:
        manifest = json.load(stream)
    required = {
        "schema",
        "run_id",
        "implementation",
        "training_seed",
        "evaluation_seed",
        "checkpoint_transition",
        "terminal_profile",
        "evaluator_hash",
        "dataset_hash",
        "expected_motion_count",
    }
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"Manifest is missing fields: {sorted(missing)}.")
    if manifest["schema"] != "forward_backward_phase2_manifest_v1":
        raise ValueError(f"Unsupported manifest schema: {manifest['schema']!r}.")
    identity_fields = required.difference({"schema", "expected_motion_count"})
    for field in identity_fields:
        actual = {getattr(record, field) for record in records}
        if actual != {manifest[field]}:
            raise ValueError(f"Manifest field {field!r} does not match records: {actual}.")
    summary = validate_records(records)
    if summary["num_motions"] != int(manifest["expected_motion_count"]):
        raise ValueError("Manifest expected_motion_count does not match records.")
    return summary


def _index_pairing_keys(records: Sequence[EvaluationRecord], label: str) -> dict[tuple[object, ...], EvaluationRecord]:
    indexed: dict[tuple[object, ...], EvaluationRecord] = {}
    for record in records:
        if record.pairing_key in indexed:
            raise ValueError(f"Duplicate {label} pairing key: {record.pairing_key}")
        indexed[record.pairing_key] = record
    return indexed


def main() -> None:
    """Validate one Phase 2 record file and optional manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    records = read_records(args.records)
    summary = validate_manifest(args.manifest, records) if args.manifest else validate_records(records)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
