# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the frozen native BFM-Zero G1 trace without importing BFM-Zero."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).parent / "fixtures"
TRACE_MANIFEST = FIXTURES / "native_trace_manifest_v1.json"
G1_METADATA = FIXTURES / "g1_lafan_same_step_trace_v1.json"
G1_TENSORS = FIXTURES / "g1_lafan_same_step_trace_v1.npz"
GENERATOR = Path(__file__).parent / "generate_g1_lafan_trace.py"
TRACE_MEMBERS = {
    "meta_humenv_next_step_trace_v1.json",
    "meta_humenv_next_step_trace_v1.npz",
    "g1_lafan_same_step_trace_v1.json",
    "g1_lafan_same_step_trace_v1.npz",
}
G1_KEYS = {
    "action_applied",
    "behavior_action",
    "controller_target_joint_position",
    "current_body_angular_velocity",
    "current_body_linear_velocity",
    "current_body_position",
    "current_body_rotation_xyzw",
    "current_default_joint_offset",
    "current_episode_step",
    "current_history_actor",
    "current_last_action",
    "current_motion_id",
    "current_privileged_state",
    "current_qpos",
    "current_qvel",
    "current_reference_time_seconds",
    "current_state",
    "environment_raw_evidence",
    "environment_reward",
    "environment_reward_recomposed",
    "estimated_pd_torque",
    "final_body_angular_velocity",
    "final_body_linear_velocity",
    "final_body_position",
    "final_body_rotation_xyzw",
    "final_history_actor",
    "final_last_action",
    "final_observation_valid",
    "final_privileged_state",
    "final_qpos",
    "final_qvel",
    "final_state",
    "learner_auxiliary_raw_evidence",
    "learner_auxiliary_reward",
    "penalty_scale",
    "processed_action",
    "returned_body_angular_velocity",
    "returned_body_linear_velocity",
    "returned_body_position",
    "returned_body_rotation_xyzw",
    "returned_episode_step",
    "returned_history_actor",
    "returned_last_action",
    "returned_motion_id",
    "returned_privileged_state",
    "returned_qpos",
    "returned_qvel",
    "returned_reference_time_seconds",
    "returned_state",
    "target_reference_time_seconds",
    "terminated",
    "truncated",
    "current_body_com_pose_xyzw",
    "current_body_inertia",
    "current_body_mass",
    "current_contact_force",
    "current_joint_armature",
    "current_joint_damping",
    "current_joint_effort_limit",
    "current_joint_friction",
    "current_joint_position_limit",
    "current_joint_stiffness",
    "current_joint_velocity_limit",
    "current_shape_material",
    "final_contact_force",
    "returned_contact_force",
    "substep_applied_pd_torque",
    "substep_body_angular_velocity",
    "substep_body_linear_velocity",
    "substep_body_position",
    "substep_body_rotation_xyzw",
    "substep_contact_force",
    "substep_qpos",
    "substep_qvel",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def test_native_trace_manifest_closes_over_both_exact_native_traces() -> None:
    """The trace manifest must identify the expanded Meta and G1 files by exact bytes."""
    manifest = json.loads(TRACE_MANIFEST.read_text())
    assert manifest["schema"] == "forward_backward_phase3_native_trace_manifest_v1"
    assert set(manifest["members"]) == TRACE_MEMBERS
    for name, identity in manifest["members"].items():
        path = FIXTURES / name
        assert identity == {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def test_g1_trace_closes_over_every_exact_tensor() -> None:
    """Every G1 tensor must match its independently recorded shape, dtype, and hash."""
    metadata = json.loads(G1_METADATA.read_text())
    assert metadata["schema"] == "forward_backward_phase3_g1_lafan_trace_v1"
    assert metadata["profile"] == "g1_lafan_50hz"
    assert metadata["trace"]["file_sha256"] == _sha256(G1_TENSORS)
    with np.load(G1_TENSORS, allow_pickle=False) as tensors:
        assert set(tensors.files) == G1_KEYS
        for name in G1_KEYS:
            value = tensors[name]
            summary = metadata["trace"]["tensors"][name]
            assert list(value.shape) == summary["shape"]
            assert str(value.dtype) == summary["dtype"]
            assert _tensor_hash(value) == summary["sha256"]


def test_g1_trace_records_exact_procedural_source_provenance() -> None:
    """The simulator trace must close over generated motion and no redistributed corpus values."""
    metadata = json.loads(G1_METADATA.read_text())
    declared = metadata["declaration"]["motion_data"]
    observed = metadata["source"]["motion_source"]

    assert declared["kind"] == observed["kind"] == "procedural_synthetic"
    assert declared["identifier"] == observed["identifier"] == "phase3_synthetic_g1_periodic_v1"
    assert declared["contains_dataset_values"] is observed["contains_dataset_values"] is False
    assert (
        declared["clip_ids"]
        == observed["clip_ids"]
        == [
            "synthetic_g1_periodic_00",
            "synthetic_g1_periodic_01",
        ]
    )
    assert (
        declared["ordered_fields"]
        == observed["ordered_fields"]
        == [
            "root_trans_offset",
            "pose_aa",
            "fps",
        ]
    )
    assert declared["content_sha256"] == observed["content_sha256"]
    assert observed["exact_recipe_match"] is True
    assert observed["errors"] == []
    assert observed["file"]["bytes"] > 0
    assert len(observed["file"]["sha256"]) == 64


def test_g1_trace_records_physics_drives_contacts_and_each_substep() -> None:
    """The source trace must close every exposed candidate transition input."""
    metadata = json.loads(G1_METADATA.read_text())
    assert metadata["source"]["physical_shape_count"] == 33
    generator = metadata["source"]["trace_generator"]
    assert len(generator["sha256"]) == 64
    assert generator["bytes"] > 0
    with np.load(G1_TENSORS, allow_pickle=False) as tensors:
        assert tensors["current_body_mass"].shape == (5, 2, 30)
        assert tensors["current_body_inertia"].shape == (5, 2, 30, 9)
        assert tensors["current_body_com_pose_xyzw"].shape == (5, 2, 30, 7)
        assert tensors["current_shape_material"].shape == (5, 2, 33, 3)
        assert tensors["current_contact_force"].shape == (5, 2, 30, 3)
        assert tensors["current_joint_stiffness"].shape == (5, 2, 29)
        assert tensors["current_joint_position_limit"].shape == (5, 2, 29, 2)
        assert tensors["substep_qpos"].shape == (5, 2, 4, 36)
        assert tensors["substep_qvel"].shape == (5, 2, 4, 35)
        assert tensors["substep_applied_pd_torque"].shape == (5, 2, 4, 29)
        assert tensors["substep_body_position"].shape == (5, 2, 4, 31, 3)
        assert tensors["substep_contact_force"].shape == (5, 2, 4, 30, 3)


def test_g1_trace_exposes_same_step_final_and_action_history_timing() -> None:
    """The true reached state, post-reset return, and processed-action history must stay distinct."""
    expected_timeout = np.array([[False, False], [False, False], [True, True], [False, False], [False, False]])
    with np.load(G1_TENSORS, allow_pickle=False) as tensors:
        assert tensors["behavior_action"].shape == (5, 2, 29)
        assert tensors["current_state"].shape == (5, 2, 64)
        assert tensors["current_history_actor"].shape == (5, 2, 372)
        assert tensors["current_privileged_state"].shape == (5, 2, 463)
        assert tensors["current_qpos"].shape == (5, 2, 36)
        assert tensors["current_qvel"].shape == (5, 2, 35)
        assert tensors["current_body_position"].shape == (5, 2, 31, 3)
        assert not tensors["terminated"].any()
        np.testing.assert_array_equal(tensors["truncated"], expected_timeout)
        np.testing.assert_array_equal(tensors["final_observation_valid"], expected_timeout)
        assert tensors["action_applied"].all()
        np.testing.assert_array_equal(
            tensors["current_episode_step"],
            [[0, 0], [1, 1], [2, 2], [0, 0], [1, 1]],
        )
        np.testing.assert_array_equal(
            tensors["returned_episode_step"],
            [[1, 1], [2, 2], [0, 0], [1, 1], [2, 2]],
        )

        np.testing.assert_array_equal(tensors["processed_action"], tensors["behavior_action"] * 5.0)
        np.testing.assert_array_equal(tensors["current_state"][1:], tensors["returned_state"][:-1])
        np.testing.assert_array_equal(tensors["current_qpos"][1:], tensors["returned_qpos"][:-1])
        np.testing.assert_array_equal(tensors["current_qvel"][1:], tensors["returned_qvel"][:-1])
        np.testing.assert_array_equal(
            tensors["returned_last_action"][[0, 1, 3, 4]],
            tensors["processed_action"][[0, 1, 3, 4]],
        )
        assert not tensors["returned_last_action"][2].any()
        assert not tensors["returned_history_actor"][2].any()
        assert tensors["final_history_actor"][2].any()
        assert not np.array_equal(tensors["returned_state"][2], tensors["final_state"][2])
        assert not np.array_equal(tensors["returned_qpos"][2], tensors["final_qpos"][2])

        invalid = np.array([0, 1, 3, 4])
        assert not tensors["final_state"][invalid].any()
        assert not tensors["final_qpos"][invalid].any()
        assert not tensors["final_body_position"][invalid].any()
        np.testing.assert_allclose(
            tensors["environment_reward"],
            tensors["environment_reward_recomposed"],
            rtol=1.0e-5,
            atol=1.0e-5,
        )


def test_g1_trace_binds_completed_environment_and_later_capture_hook() -> None:
    """The completed 0a environment and later f049 capture hook must not be conflated."""
    metadata = json.loads(G1_METADATA.read_text())
    declaration = metadata["declaration"]
    assert metadata["source"]["repository_revision"] == "f0495e864ffcf332f346bd7b55e9aa108cb8b38f"
    assert metadata["source"]["exact_final_capture_adapter_sha256"] == (
        "52a35f031b2808479cf0dbf75bdb1ead5c7c4292f33c6f3cb33ca8e367460cc7"
    )
    identity = declaration["native_environment_byte_identity"]
    assert identity["completed_capacity_run_revision"] == "0a4132620e5b588752cf7d77cacda1720d6e4b08"
    assert identity["later_cadence_revision"] == "f0495e864ffcf332f346bd7b55e9aa108cb8b38f"
    assert identity["later_cadence_revision"] == metadata["source"]["repository_revision"]
    assert identity["environment_files_unchanged_across_revisions"] is True
