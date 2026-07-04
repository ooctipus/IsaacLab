# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate canonical Phase 3E repeatability-scale and final-capture evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from motion_environment_identity import motion_environment_axes

from isaaclab_tasks.core.multi_task.motion.data.sources import CmuHumEnvSmplClips, LafanG1JoblibClips
from isaaclab_tasks.core.multi_task.motion.robots.g1.reference import G1PoseFrameBuilder
from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import SmplGeneralizedCoordinateFrameBuilder
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets

ROOT = Path(__file__).parent
PROBE = ROOT / "motion_environment_probe.py"
GPU_OWNERSHIP = ROOT / "gpu_ownership.py"
AGGREGATOR = ROOT / "aggregate_motion_environment_systems.py"
IDENTITY = ROOT / "motion_environment_identity.py"
CONTRACT = ROOT / "fixtures/motion_environment_systems_contract_v4.json"
PRESETS = ("g1_lafan", "smpl_cmu")
IMPORTERS = {
    "g1_lafan": LafanG1JoblibClips,
    "smpl_cmu": CmuHumEnvSmplClips,
}
FRAME_BUILDERS = {
    "g1_lafan": G1PoseFrameBuilder,
    "smpl_cmu": SmplGeneralizedCoordinateFrameBuilder,
}
CAPTURE_STEPS = {"g1_lafan": 1002, "smpl_cmu": 600}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _module():
    spec = importlib.util.spec_from_file_location("aggregate_motion_environment_systems", AGGREGATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity_module():
    spec = importlib.util.spec_from_file_location("phase3_motion_environment_identity", IDENTITY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_records(preset: str) -> list[tuple[Path, dict]]:
    raw = ROOT / f"fixtures/runtime/{preset}_systems_v4"
    return [(path, json.loads(path.read_text())) for path in sorted(raw.glob("*.json"))]


def _summary(preset: str) -> dict:
    return json.loads((ROOT / f"fixtures/runtime/{preset}_systems_v4.json").read_text())


@pytest.mark.parametrize("preset", PRESETS)
def test_systems_summary_recomputes_exactly_from_all_frozen_raw_records(preset: str) -> None:
    """The scientific summary must recompute independently of producer-source drift."""
    records = _raw_records(preset)
    assert len(records) == 16
    expected_names = {
        *(f"scale_{num_envs}_r{replicate}.json" for num_envs in (1, 16, 1024) for replicate in range(2)),
        *(f"capture_p{pair}_{mode}.json" for pair in range(5) for mode in ("off", "on")),
    }
    assert {path.name for path, _record in records} == expected_names

    summary = _summary(preset)
    recomputed = _module().aggregate(records, preset, CONTRACT)
    assert recomputed["aggregation_identity"] == {"aggregator_sha256": _sha256(AGGREGATOR)}
    assert len(summary["aggregation_identity"]["aggregator_sha256"]) == 64
    assert {name: value for name, value in recomputed.items() if name != "aggregation_identity"} == {
        name: value for name, value in summary.items() if name != "aggregation_identity"
    }
    assert summary["raw_records"] == [{"name": path.name, "sha256": _sha256(path)} for path, _record in records]


@pytest.mark.parametrize("preset", PRESETS)
def test_systems_summary_is_authentic_and_reports_current_compatibility(preset: str) -> None:
    """Frozen systems facts remain valid while current compatibility is explicit."""
    summary = _summary(preset)
    cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes(preset))
    cfg.commands.motion.task_table.motion_split = "evaluation"
    assert summary["contract_identity"]["file_sha256"] == _sha256(CONTRACT)
    code_identity = summary["environment_identity"]["code_identity"]
    assert all(
        isinstance(code_identity[name], str) and len(code_identity[name]) == 64
        for name in ("probe_sha256", "gpu_ownership_sha256")
    )
    assert len(summary["aggregation_identity"]["aggregator_sha256"]) == 64
    identity = _identity_module()
    stored_dependency = code_identity["dependency_identity"]
    identity.motion_environment_semantic_sha256(stored_dependency)
    expected_dependency = _identity_module().motion_environment_dependency_identity(
        preset=preset,
        cfg=cfg,
        importer_type=IMPORTERS[preset],
        frame_builder_type=FRAME_BUILDERS[preset],
    )
    compatibility = identity.motion_environment_compatibility(stored_dependency, expected_dependency)
    assert compatibility["status"] in {
        "exact_producer_match",
        "declared_contract_match_requires_runtime_validation",
        "declared_contract_mismatch",
    }
    current_producers = {
        "aggregator_sha256": _sha256(AGGREGATOR),
        "probe_sha256": _sha256(PROBE),
        "gpu_ownership_sha256": _sha256(GPU_OWNERSHIP),
    }
    assert all(len(digest) == 64 for digest in current_producers.values())
    assert "source_artifact_root" not in summary["environment_identity"]
    assert "reference_artifact_root" not in summary["environment_identity"]


@pytest.mark.parametrize("preset", PRESETS)
def test_systems_summary_passes_declared_pair_local_repeatability_and_capture_gates(preset: str) -> None:
    """Each capture pair owns one GPU locally while all declared statistical gates pass."""
    summary = _summary(preset)
    assert summary["schema"] == "forward_backward_phase3e_motion_environment_systems_v4"
    assert summary["status"] == "passed"
    assert summary["preset"] == preset
    assert summary["raw_record_count"] == 16
    scale = summary["repeatability_scale"]
    assert [result["num_envs"] for result in scale["results"]] == [1, 16, 1024]
    assert all(result["reset_state_identity_passed"] for result in scale["results"])
    assert all(result["one_edge"]["passed"] for result in scale["results"])
    if preset == "g1_lafan":
        assert all(result["one_edge"]["comparison"] == "exact_signature" for result in scale["results"])
        assert all(result["one_edge"]["maximum_signature_absolute_error"] == 0.0 for result in scale["results"])
        assert all(result["full_horizon"]["comparison"] == "exact_signature" for result in scale["results"])
        assert all(result["full_horizon"]["maximum_signature_absolute_error"] == 0.0 for result in scale["results"])
    else:
        assert all(result["one_edge"]["comparison"] == "bounded_signature" for result in scale["results"])
        assert all(
            result["one_edge"]["maximum_signature_absolute_error"]
            <= scale["contract"]["one_edge"]["absolute_tolerance"]
            for result in scale["results"]
        )
        assert all(
            result["full_horizon"]
            == {
                "comparison": "semantic_finiteness",
                "maximum_signature_absolute_error": None,
                "passed": True,
            }
            for result in scale["results"]
        )

    capture = summary["capture_cost"]
    assert capture["measured_steps"] == CAPTURE_STEPS[preset]
    assert len(capture["pairs"]) == 5
    contract = json.loads(CONTRACT.read_text())
    assert [pair["capture_order"] for pair in capture["pairs"]] == contract["shared"]["capture_cost"][
        "pair_capture_order"
    ]
    assert all(pair["execution_started_unix_ns"][0] < pair["execution_started_unix_ns"][1] for pair in capture["pairs"])
    raw_capture = [record for _path, record in _raw_records(preset) if record["evidence"]["role"] == "capture_cost"]
    for pair in capture["pairs"]:
        pair_records = [record for record in raw_capture if record["evidence"]["pair_index"] == pair["pair_index"]]
        assert len(pair_records) == 2
        physical_gpu_uuids = set()
        for record in pair_records:
            ownership = record["benchmark_gpu_ownership"]
            assert ownership["required_scope"] == "physical_gpu"
            for boundary in ("before_benchmark", "after_benchmark"):
                snapshot = ownership[boundary]
                assert snapshot["physical_gpu_uuid"].startswith("GPU-")
                assert snapshot["compute_pids"] == [snapshot["owner_pid"]]
                assert snapshot["competing_compute_pids"] == []
                assert snapshot["exclusive"] is True
                physical_gpu_uuids.add(snapshot["physical_gpu_uuid"])
        assert physical_gpu_uuids == {pair["physical_gpu_uuid"]}
    assert all(pair["repeatability"]["reset_state_identity_passed"] for pair in capture["pairs"])
    limits = summary["gates"]["limits"]
    assert capture["throughput_loss_fraction"]["upper_95_student_t"] <= limits["throughput_loss_upper_95_fraction"]
    assert capture["peak_allocated_increment_delta_bytes"]["mean"] <= limits["mean_peak_allocated_delta_bytes"]
    assert summary["gates"]["passed"] is True
