# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the immutable Phase 3A native motion-environment contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
FIXTURES = ROOT / "fixtures"
MANIFEST = FIXTURES / "native_contract_manifest_v1.json"
PROFILES = FIXTURES / "native_profiles_v1.json"
MOTION_DATA = FIXTURES / "native_motion_data_v1.json"
TENSORS = FIXTURES / "native_tensor_layouts_v1.json"
EVIDENCE = FIXTURES / "native_evidence_channels_v1.json"
G1_SKELETON = FIXTURES / "g1_lafan_50hz_skeleton_v1.json"
PHASE2_ENVIRONMENT = FIXTURES / "g1_lafan_50hz_phase2_environment_v1.json"
GENERATOR = ROOT / "generate_g1_lafan_trace.py"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(map(_keys, value.values())))
    if isinstance(value, list):
        return set().union(*(map(_keys, value)))
    return set()


def _generator_module():
    spec = importlib.util.spec_from_file_location("generate_g1_lafan_trace", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_closes_over_exact_immutable_contract_bytes() -> None:
    """The closure manifest must identify every immutable member by exact bytes."""
    manifest = _load(MANIFEST)
    assert manifest["schema"] == "forward_backward_phase3a_native_contract_manifest_v1"
    expected = {path.name for path in (PROFILES, MOTION_DATA, TENSORS, EVIDENCE, G1_SKELETON, PHASE2_ENVIRONMENT)}
    assert set(manifest["members"]) == expected
    for name, identity in manifest["members"].items():
        path = FIXTURES / name
        assert identity == {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    forbidden = {"status", "ready", "readiness", "hostname", "generated_at", "launch_authorized", "blockers"}
    assert not (_keys(manifest) & forbidden)


def test_profiles_are_concrete_and_preserve_native_source_differences() -> None:
    """The two profiles must state source behavior instead of algorithm branding."""
    profiles = _load(PROFILES)
    assert profiles["schema"] == "forward_backward_phase3a_native_profiles_v1"
    assert set(profiles["profiles"]) == {"smpl_humenv_30hz", "g1_lafan_50hz"}
    g1 = profiles["profiles"]["g1_lafan_50hz"]
    assert g1["timing"] == {
        "physics_dt_seconds": 0.005,
        "control_decimation": 4,
        "control_dt_seconds": 0.02,
        "source_dt_seconds": 1 / 30,
        "nominal_horizon_seconds": 10.0,
        "configured_horizon_steps": 500,
        "applied_actions_before_timeout": 501,
    }
    assert g1["reset"]["lie_down_probability"] == 0.3
    assert g1["reset"]["reference_noise_multiplier"] == 0.0
    assert g1["observation_noise"]["enabled"] is True
    assert g1["domain_randomization"]["enabled"] is True
    assert g1["history"]["native_internal_autoreset_seed_enters_history"] is True
    assert g1["history"]["phase2_corrected_include_seed_observations"] is False
    assert g1["reference"]["observation_query_step_offset"] == 1
    correction_ids = {item["id"] for item in profiles["plan_corrections"]}
    assert correction_ids == {f"P3A-G1-{index:02d}" for index in range(1, 13)}


def test_g1_skeleton_distinguishes_physical_and_synthetic_bodies() -> None:
    """A 30-body robot plus synthetic head must never be called an unqualified 31-body asset."""
    skeleton = _load(G1_SKELETON)
    assert skeleton["schema"] == "forward_backward_phase3a_g1_lafan_skeleton_v1"
    physical = skeleton["physical_skeleton"]
    assert len(physical["body_names"]) == 30
    assert len(physical["parent_indices"]) == 30
    assert len(physical["joint_names"]) == 29
    assert len(physical["joint_axes"]) == 29
    assert physical["parent_indices"][0] == -1
    assert skeleton["synthetic_bodies"] == [
        {
            "name": "head_link",
            "parent": "torso_link",
            "translation_m": [0.0, 0.0, 0.35],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
    ]
    assert skeleton["reference_frame_count"] == 31


def test_motion_data_contract_freezes_formats_counts_hashes_and_license_evidence() -> None:
    """Robot-ready artifacts and canonical-source rights must remain separate facts."""
    data = _load(MOTION_DATA)
    assert data["schema"] == "forward_backward_phase3a_native_motion_data_v1"
    cmu = data["collections"]["smpl_humenv_30hz"]
    assert (cmu["train"]["clips"], cmu["train"]["frames"], cmu["test"]["clips"], cmu["test"]["frames"]) == (
        1638,
        730_307,
        182,
        88_364,
    )
    lafan = data["collections"]["g1_lafan_50hz"]
    assert (lafan["evaluation"]["clips"], lafan["evaluation"]["frames"]) == (40, 264_705)
    assert (lafan["training"]["clips"], lafan["training"]["frames"]) == (862, 258_600)
    assert lafan["training"]["frames_per_clip"] == 300
    assert lafan["fields"]["pose_aa"]["shape"] == ["frames", 30, 3]
    assert lafan["semantic_level"] == "robot_pose_g1_not_canonical_lafan"
    assert lafan["license"]["dataset_specific_evidence"] == "not_present_in_local_repository"
    assert lafan["license"]["redistribution_gate"] == "requires_dataset_specific_provenance_review"


def test_tensor_layouts_freeze_field_major_history_and_runtime_widths() -> None:
    """Declared YAML widths must not override the tensors the released policy actually consumed."""
    layouts = _load(TENSORS)
    g1 = layouts["profiles"]["g1_lafan_50hz"]
    assert g1["physical_body_count"] == 30
    assert g1["reference_frame_count"] == 31
    assert g1["privileged_state"]["declared_yaml_width"] == 462
    assert g1["privileged_state"]["runtime_width"] == 463
    assert g1["routes"] == {
        "actor": 465,
        "backward": 527,
        "critic_auxiliary": 928,
        "critic_discriminator": 928,
        "discriminator": 527,
        "forward": 928,
        "underlying_hydra_actor_concat": 928,
        "underlying_hydra_declared_actor_space": 927,
    }
    history = g1["history_actor"]
    assert history["layout"] == "field_major_then_newest_first_time"
    assert [field["name"] for field in history["fields"]] == [
        "processed_action",
        "base_angular_velocity_scaled_noisy",
        "joint_position_scaled_noisy",
        "joint_velocity_scaled_noisy",
        "projected_gravity_scaled_noisy",
    ]
    assert sum(field["width"] * field["length"] for field in history["fields"]) == 372
    transition = layouts["logical_transition"]
    assert {
        "behavior_action",
        "processed_action",
        "reached_observation",
        "returned_observation",
        "final_observation",
    }.issubset(transition)


def test_phase2_environment_json_is_exact_and_rebinds_only_runtime_location() -> None:
    """The Phase 2 environment object and ordered Hydra overrides are scientific evidence."""
    frozen = _load(PHASE2_ENVIRONMENT)
    assert frozen["schema"] == "forward_backward_phase3a_g1_lafan_phase2_environment_v1"
    assert frozen["reference_config_sha256"] == "96889c351a919a907f1f6e3001c5213fe5469ffc4fc2d87c0ef52e422eba7d0f"
    environment = frozen["environment"]
    assert environment["name"] == "humanoidverse_isaac"
    assert environment["disable_obs_noise"] is False
    assert environment["disable_domain_randomization"] is False
    assert environment["include_history_actor"] is True
    assert environment["root_height_obs"] is True
    assert environment["hydra_overrides"][-2:] == [
        "env.config.lie_down_init=True",
        "env.config.lie_down_init_prob=0.3",
    ]
    assert frozen["runtime_rebindings"] == ["device", "lafan_tail_path"]


def test_reward_contract_keeps_environment_and_learner_compositions_distinct() -> None:
    """Environment dt/curriculum and learner auxiliary weights must each apply exactly once."""
    contract = _load(EVIDENCE)
    g1 = contract["profiles"]["g1_lafan_50hz"]
    environment = g1["compositions"]["environment_scalar"]
    auxiliary = g1["compositions"]["learner_auxiliary_scalar"]
    assert environment["coefficient_interpretation"] == "reward_density_per_second_then_multiply_by_control_dt"
    assert environment["control_dt_seconds"] == 0.02
    assert environment["penalty_curriculum"]["enabled"] is True
    assert auxiliary["coefficient_interpretation"] == "per_transition_raw_evidence_weight_no_dt"
    assert auxiliary["normalization"] == "scale_only_running_standard_deviation_after_scalar_composition"
    assert auxiliary["coefficients"]["penalty_torques"] == 0.0
    assert auxiliary["coefficients"]["limits_torque"] == 0.0
    assert set(environment["coefficients"]) == set(g1["environment_raw_evidence"])
    assert set(auxiliary["coefficients"]) == set(g1["learner_raw_evidence"])

    module = _generator_module()
    values = {name: float(index + 1) for index, name in enumerate(g1["environment_raw_evidence"])}
    expected_environment = sum(
        environment["coefficients"][name]
        * 0.02
        * values[name]
        * (0.1 if name in environment["penalty_curriculum"]["channels"] else 1.0)
        for name in values
    )
    assert module.compose_environment_reward(values, penalty_scale=0.1) == expected_environment
    expected_auxiliary = sum(auxiliary["coefficients"][name] * values[name] for name in auxiliary["coefficients"])
    assert module.compose_learner_auxiliary_reward(values) == expected_auxiliary


def test_g1_trace_source_is_procedural_smooth_and_exactly_validated(tmp_path: Path) -> None:
    """The redistributed trace input must be generated data, never a motion-corpus excerpt."""
    module = _generator_module()
    source = module.synthetic_motion_source()
    declaration = module.synthetic_motion_source_declaration()

    assert tuple(source) == ("synthetic_g1_periodic_00", "synthetic_g1_periodic_01")
    assert declaration["kind"] == "procedural_synthetic"
    assert declaration["contains_dataset_values"] is False
    assert declaration["clip_ids"] == list(source)
    assert declaration["ordered_fields"] == ["root_trans_offset", "pose_aa", "fps"]
    assert len(declaration["content_sha256"]) == 64

    for clip in source.values():
        assert tuple(clip) == ("root_trans_offset", "pose_aa", "fps")
        assert clip["root_trans_offset"].shape == (360, 3)
        assert clip["root_trans_offset"].dtype == np.float32
        assert clip["pose_aa"].shape == (360, 30, 3)
        assert clip["pose_aa"].dtype == np.float32
        assert clip["fps"] == 30
        assert np.isfinite(clip["root_trans_offset"]).all()
        assert np.isfinite(clip["pose_aa"]).all()
        assert np.max(np.abs(np.diff(clip["pose_aa"], axis=0))) < 0.02
        joint_pose = clip["pose_aa"][:, 1:]
        np.testing.assert_array_equal(
            joint_pose,
            joint_pose.sum(axis=-1, keepdims=True) * module.G1_JOINT_AXES[None, :, :],
        )

    first = tmp_path / "synthetic_first.pkl"
    second = tmp_path / "synthetic_second.pkl"
    assert (
        module.write_synthetic_motion_source(first)["sha256"] == module.write_synthetic_motion_source(second)["sha256"]
    )
    inspection = module.inspect_synthetic_motion_source(first)
    assert inspection["exact_recipe_match"] is True
    assert inspection["errors"] == []
    assert inspection["content_sha256"] == declaration["content_sha256"]

    corrupted = module.synthetic_motion_source()
    corrupted["synthetic_g1_periodic_00"]["pose_aa"][0, 0, 0] += np.float32(1.0e-3)
    with second.open("wb") as stream:
        pickle.dump(corrupted, stream, protocol=4)
    rejected = module.inspect_synthetic_motion_source(second)
    assert rejected["exact_recipe_match"] is False
    assert rejected["errors"] == ["serialized bytes differ from the procedural recipe"]


def test_generator_separates_immutable_declaration_from_transient_readiness(tmp_path: Path) -> None:
    """Preparation identity must not depend on host paths, time, GPU state, or launch status."""
    module = _generator_module()
    immutable = module.capture_declaration()
    assert immutable["schema"] == "forward_backward_phase3a_g1_lafan_trace_declaration_v1"
    assert immutable["profile"] == "g1_lafan_50hz"
    assert immutable["motion_data"] == module.synthetic_motion_source_declaration()
    assert immutable["trace_closure_members"] == [
        "meta_humenv_next_step_trace_v1.json",
        "meta_humenv_next_step_trace_v1.npz",
        "g1_lafan_same_step_trace_v1.json",
        "g1_lafan_same_step_trace_v1.npz",
    ]
    forbidden = {"status", "ready", "readiness", "hostname", "generated_at", "launch_authorized", "blockers"}
    assert not (_keys(immutable) & forbidden)

    missing = tmp_path / "missing"
    readiness = module.inspect_readiness(
        bfm_repo=missing,
        reference_config=missing / "config.json",
        motion_source_path=missing / "data.pkl",
    )
    assert readiness["schema"] == "forward_backward_phase3a_g1_lafan_trace_readiness_v1"
    assert readiness["ready"] is False
    assert readiness["blockers"]
    assert _keys(readiness) & {"ready", "generated_at", "blockers"}


def test_fixed_trace_actions_cover_raw_and_processed_action_contract() -> None:
    """The trace must make the released normalization visible without saturating it."""
    module = _generator_module()
    actions = module.fixed_actions()
    processed = module.process_actions(actions)
    assert actions.shape == (5, 2, 29)
    assert np.max(np.abs(actions)) < 1.0
    np.testing.assert_allclose(processed, actions * 5.0, rtol=0.0, atol=0.0)
