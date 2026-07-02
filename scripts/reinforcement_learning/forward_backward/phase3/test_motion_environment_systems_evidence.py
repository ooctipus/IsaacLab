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

from isaaclab_tasks.core.multi_task.motion.data.importers import BfmG1JoblibClips, HumEnvHdf5Clips
from isaaclab_tasks.core.multi_task.motion.trajectory.g1 import G1LafanFrameBuilder
from isaaclab_tasks.core.multi_task.motion.trajectory.smpl import SmplHumEnvFrameBuilder
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
    "g1_lafan": BfmG1JoblibClips,
    "smpl_cmu": HumEnvHdf5Clips,
}
FRAME_BUILDERS = {
    "g1_lafan": G1LafanFrameBuilder,
    "smpl_cmu": SmplHumEnvFrameBuilder,
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
    """The summary must be a pure reproducible reduction of the closed 16-record matrix."""
    records = _raw_records(preset)
    assert len(records) == 16
    expected_names = {
        *(f"scale_{num_envs}_r{replicate}.json" for num_envs in (1, 16, 1024) for replicate in range(2)),
        *(f"capture_p{pair}_{mode}.json" for pair in range(5) for mode in ("off", "on")),
    }
    assert {path.name for path, _record in records} == expected_names

    summary = _summary(preset)
    recomputed = _module().aggregate(records, preset, CONTRACT)
    assert recomputed == summary
    assert summary["raw_records"] == [{"name": path.name, "sha256": _sha256(path)} for path, _record in records]


@pytest.mark.parametrize("preset", PRESETS)
def test_systems_summary_closes_over_current_data_and_environment_code(preset: str) -> None:
    """Importer, frame builder, environment, probe, aggregator, and contract must all be current."""
    summary = _summary(preset)
    cfg = resolve_presets(MotionImitationEnvCfg(), selected={preset})
    cfg.commands.motion.task_table.motion_split = "evaluation"
    assert summary["aggregation_identity"] == {"aggregator_sha256": _sha256(AGGREGATOR)}
    assert summary["contract_identity"]["file_sha256"] == _sha256(CONTRACT)
    code_identity = summary["environment_identity"]["code_identity"]
    assert code_identity["probe_sha256"] == _sha256(PROBE)
    assert code_identity["gpu_ownership_sha256"] == _sha256(GPU_OWNERSHIP)
    expected_dependency = _identity_module().motion_environment_dependency_identity(
        preset=preset,
        cfg=cfg,
        importer_type=IMPORTERS[preset],
        frame_builder_type=FRAME_BUILDERS[preset],
    )
    semantic_sha256 = _identity_module().motion_environment_semantic_sha256
    assert semantic_sha256(code_identity["dependency_identity"]) == semantic_sha256(expected_dependency)
    assert "source_artifact_root" not in summary["environment_identity"]
    assert "reference_artifact_root" not in summary["environment_identity"]


@pytest.mark.parametrize("preset", PRESETS)
def test_systems_summary_passes_declared_repeatability_and_capture_gates(preset: str) -> None:
    """Preset-appropriate repeatability and paired uncertainty must pass the declared contract."""
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
    assert len({pair["physical_gpu_uuid"] for pair in capture["pairs"]}) == 1
    assert all(pair["repeatability"]["reset_state_identity_passed"] for pair in capture["pairs"])
    limits = summary["gates"]["limits"]
    assert capture["throughput_loss_fraction"]["upper_95_student_t"] <= limits["throughput_loss_upper_95_fraction"]
    assert capture["peak_allocated_increment_delta_bytes"]["mean"] <= limits["mean_peak_allocated_delta_bytes"]
    assert summary["gates"]["passed"] is True
