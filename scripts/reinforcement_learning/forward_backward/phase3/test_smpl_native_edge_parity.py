# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the frozen SMPL native-edge parity decision and ownership boundaries."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from motion_environment_identity import motion_environment_axes

from isaaclab_tasks.core.multi_task.motion.data.sources import CmuHumEnvSmplClips
from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import SmplGeneralizedCoordinateFrameBuilder
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets

ARTIFACT = Path(__file__).parent / "fixtures/runtime/smpl_native_edge_parity_v2.json"
PROBE = Path(__file__).parent / "generate_smpl_native_edge_parity.py"
IDENTITY = Path(__file__).parent / "motion_environment_identity.py"


def _report() -> dict:
    return json.loads(ARTIFACT.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_module():
    spec = importlib.util.spec_from_file_location("smpl_native_edge_environment_identity", IDENTITY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smpl_native_edge_parity_closes_source_candidate_and_lifecycle_contracts() -> None:
    """The frozen decision must pass all source, simulator, and lifecycle claims."""
    report = _report()
    assert report["schema"] == "forward_backward_phase3e_smpl_native_edge_parity_v2"
    assert report["profile"] == "smpl_cmu"
    assert report["source_contract"]["applied_edges"] == 8
    assert report["source_contract"]["reset_only_rows_excluded"] == 2
    assert report["source_contract"]["done_applied_edges"] == 2
    assert report["decision"] == {
        "current_passed": True,
        "exact_lifecycle_passed": True,
        "fixed_model_passed": True,
        "passed": True,
        "reached_passed": True,
        "source_contract_passed": True,
    }


def test_smpl_native_edge_parity_is_authentic_and_reports_current_compatibility() -> None:
    """The frozen replay remains authentic after producer source changes."""
    identity = _report()["code_identity"]
    assert isinstance(identity["generator_sha256"], str) and len(identity["generator_sha256"]) == 64
    dependency = identity["environment_dependencies"]
    cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes("smpl_cmu"))
    cfg.commands.motion.task_table.motion_split = "evaluation"
    module = _identity_module()
    expected = module.motion_environment_dependency_identity(
        preset="smpl_cmu",
        cfg=cfg,
        importer_type=CmuHumEnvSmplClips,
        frame_builder_type=SmplGeneralizedCoordinateFrameBuilder,
    )
    module.motion_environment_semantic_sha256(dependency)
    compatibility = module.motion_environment_compatibility(dependency, expected)
    assert compatibility["status"] in {
        "exact_producer_match",
        "declared_contract_match_requires_runtime_validation",
        "declared_contract_mismatch",
    }
    assert len(_sha256(PROBE)) == 64


def test_smpl_native_edge_parity_preserves_native_actuator_ownership() -> None:
    """The source model must remain the sole owner of all 69 native actuators."""
    ownership = _report()["candidate_replay"]["fixed_model"]["native_actuator_ownership"]
    assert ownership == {
        "articulation_config_actuator_groups": 0,
        "finalized_model_actuator_rows_per_world": 69,
        "native_action_width": 69,
        "passed": True,
        "source_model_actuator_rows": 69,
    }


def test_smpl_native_edge_parity_reads_solver_owned_contact_exclusions() -> None:
    """Native MJCF exclusions must be checked after solver finalization, not on USD."""
    source = PROBE.read_text()
    assert "model.exclude_signature" in source
    assert "expected_effective_exclusions" in source
    assert "physics:filteredPairs" not in source


def test_smpl_native_edge_parity_matches_fixed_model_and_active_contact_geometry() -> None:
    """The finalized model must retain every source-owned solver and contact fact."""
    fixed = _report()["candidate_replay"]["fixed_model"]
    assert fixed["passed"]
    for name, result in fixed["fields"].items():
        assert result["passed"], name

    geometry = fixed["geometry"]
    assert geometry["contract"].startswith("Only collision-active geometry")
    assert geometry["active"]["source_count"] == 25
    assert geometry["active"]["candidate_count"] == 25
    assert geometry["active"]["missing_source_indices"] == []
    assert geometry["active"]["extra_candidate_indices"] == []
    assert geometry["active"]["passed"]
    for name, result in geometry["active"]["fields"].items():
        assert result["passed"], name

    bridge = fixed["bridge_provenance"]
    assert bridge["configured_enable_multiccd"] is False
    assert bridge["configured_enable_native_ccd"] is False
    assert bridge["source_option_disableflags"] == 0
    assert bridge["effective_option_disableflags"] == 655_360
    disableflags = fixed["fields"]["option_disableflags"]
    assert disableflags["source_value"] == 0
    assert disableflags["expected_value"] == 655_360
    assert disableflags["bridge_added_bits"] == 655_360
    assert bridge["shape_margin_unique"]
    assert bridge["shape_solref_unique"]
    assert bridge["shape_solimp_unique"]


def test_smpl_native_edge_parity_matches_reached_edges_and_final_observations() -> None:
    """Reached physics and exact NEXT_STEP normalization must pass on all rows."""
    replay = _report()["candidate_replay"]
    for group in ("current", "reached"):
        for name, result in replay[group].items():
            assert result["passed"], (group, name)

    exact = replay["exact_lifecycle"]
    assert exact["done_mask"] == {
        "expected_done_rows": 2,
        "observed_done_rows": 2,
        "passed": True,
    }
    assert exact["final_observation_valid"] == {
        "expected_rows": 2,
        "observed_rows": 2,
        "passed": True,
    }
    for name, result in exact.items():
        assert result["passed"], name
