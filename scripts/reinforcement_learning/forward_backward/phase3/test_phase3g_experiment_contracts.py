# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the structural proof and the falsifiable Phase 3G experiment contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).parent
FIXTURES = ROOT / "fixtures"
STRUCTURAL_ARCHIVE = FIXTURES / "human_motion_g1_structural_v1.npz"
STRUCTURAL_RECORD = FIXTURES / "human_motion_g1_structural_v1.json"
TIMING = FIXTURES / "motion_timing_ablation_v1.json"
PROVIDERS = FIXTURES / "motion_provider_migration_v1.json"
GENERATOR = ROOT / "generate_structural_cross_composition_fixture.py"
COMPOSITION_EVIDENCE = ROOT / "g1_cmu_composition_evidence.py"
REFERENCE_TRACKING_EVIDENCE = ROOT / "g1_cmu_reference_tracking_evidence.py"
ENVIRONMENT_IDENTITY = ROOT / "motion_environment_identity.py"
RUNTIME = FIXTURES / "runtime"
CANONICAL_RETARGET = RUNTIME / "g1_cmu_retarget_evaluation_v3.json"
CANONICAL_SIMULATOR = RUNTIME / "g1_cmu_reference_tracking_evaluation_v3.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _generator_module():
    spec = importlib.util.spec_from_file_location("generate_structural_cross_composition_fixture", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _environment_identity_module():
    spec = importlib.util.spec_from_file_location("motion_environment_identity", ENVIRONMENT_IDENTITY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_structural_archive_is_deterministic_and_closed_by_declared_tensors(tmp_path: Path) -> None:
    """The portable proof must be reproducible and independent of zip timestamps."""
    record = _load(STRUCTURAL_RECORD)
    archive_record = record["source"]["archive"]
    assert record["schema"] == "forward_backward_phase3g_structural_cross_composition_v3"
    assert record["claim"]["level"] == "structural_interface_proof"
    assert "not_a_canonical_dataset" in record["claim"]["canonicality"]
    assert archive_record["sha256"] == _sha256_file(STRUCTURAL_ARCHIVE)

    with np.load(STRUCTURAL_ARCHIVE, allow_pickle=False) as tensors:
        assert tensors.files == archive_record["ordered_fields"]
        for name in tensors.files:
            value = tensors[name]
            expected = archive_record["tensors"][name]
            assert list(value.shape) == expected["shape"]
            assert str(value.dtype) == expected["dtype"]
            assert _sha256_array(value) == expected["sha256"]

    module = _generator_module()
    regenerated_archive = tmp_path / STRUCTURAL_ARCHIVE.name
    regenerated_record = tmp_path / STRUCTURAL_RECORD.name
    tensors = module.arrays()
    module.write_archive(regenerated_archive, tensors)
    regenerated_record.write_text(
        json.dumps(module.record(regenerated_archive, tensors), indent=2, sort_keys=True) + "\n"
    )
    assert regenerated_archive.read_bytes() == STRUCTURAL_ARCHIVE.read_bytes()
    assert regenerated_record.read_bytes() == STRUCTURAL_RECORD.read_bytes()
    assert module.source_skeleton().identity_sha256 == record["source"]["skeleton"]["identity_sha256"]


def test_structural_record_separates_retarget_simulator_and_policy_errors() -> None:
    """A source-to-robot tensor proof must not be reported as policy quality."""
    record = _load(STRUCTURAL_RECORD)
    errors = record["error_ownership"]
    assert set(errors) == {"retarget", "simulator_tracking", "policy"}
    assert errors["retarget"]["status"] == "measured_by_structural_test"
    assert errors["simulator_tracking"]["status"] == "not_claimed_by_pure_data_fixture"
    assert errors["policy"]["status"] == "not_claimed_by_pure_data_fixture"
    assert set(record["composition"]) >= {
        "identifier",
        "selected_preset",
        "scene_robot",
        "source_identifier",
        "control_dt_seconds",
        "expert_grid",
    }


def test_timing_cells_preserve_physical_time_and_native_discount_horizons() -> None:
    """Every proposed clock must be internally exact and retime all Bellman discounts."""
    contract = _load(TIMING)
    assert contract["schema"] == "forward_backward_phase3g_motion_timing_ablation_v1"
    assert contract["status"] == "frozen_experiment_contract_not_executed"
    native_dt = {"smpl_cmu": 1.0 / 30.0, "g1_lafan": 1.0 / 50.0}
    ids = set()
    for cell in contract["cells"]:
        assert cell["id"] not in ids
        ids.add(cell["id"])
        control_dt = cell["physics_dt_seconds"] * cell["control_decimation"]
        assert math.isclose(control_dt, cell["control_dt_seconds"], rel_tol=0.0, abs_tol=1.0e-15)
        assert math.isclose(
            cell["horizon_steps"] * cell["control_dt_seconds"],
            contract["invariants"]["episode_duration_seconds"],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        expected_gamma = math.exp(
            math.log(contract["native_discount"]["gamma_per_step"])
            * cell["control_dt_seconds"]
            / native_dt[cell["robot"]]
        )
        assert math.isclose(cell["gamma"], expected_gamma, rel_tol=0.0, abs_tol=1.0e-15)

    common = [cell for cell in contract["cells"] if "common_physics" in cell["id"]]
    assert {cell["physics_dt_seconds"] for cell in common} == {1.0 / 300.0}
    assert {cell["control_dt_seconds"] for cell in common} == {1.0 / 30.0, 1.0 / 50.0}
    assert contract["decision"]["no_combined_score"] is True


def test_real_composition_probe_separates_error_owners_and_exact_statistics() -> None:
    """The source probe must measure only retarget facts and leave later layers open."""
    spec = importlib.util.spec_from_file_location("g1_cmu_composition_evidence", COMPOSITION_EVIDENCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    statistics = module._statistics(torch.tensor((0.0, 1.0, 2.0)))
    assert statistics == {
        "min": 0.0,
        "count": 3,
        "mean": 1.0,
        "q50": 1.0,
        "q95": 1.9,
        "q99": 1.98,
        "q999": 1.998,
        "max": 2.0,
    }
    layers = module._unmeasured_error_layers()
    assert set(layers) == {"reference_controller_simulator", "policy"}
    assert all(value["status"] == "not_measured_by_source_composition_probe" for value in layers.values())


def test_reference_tracking_probe_uses_production_limits_without_claiming_policy_quality() -> None:
    """The oracle controller must invert the real action law and remain distinct from a policy."""
    spec = importlib.util.spec_from_file_location("g1_cmu_reference_tracking_evidence", REFERENCE_TRACKING_EVIDENCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    joint_default_position = torch.zeros(29)
    position_lower_limit = torch.full((29,), -1.0)
    position_upper_limit = torch.full((29,), 1.0)
    action = SimpleNamespace(
        joint_default_position=joint_default_position,
        joint_stiffness=torch.full((29,), 40.0),
        joint_effort_limit=torch.full((29,), 20.0),
        joint_target_gain=torch.full((29,), 0.125),
        cfg=SimpleNamespace(action_scale=0.25, action_clip=5.0, normalize_to=5.0),
    )
    offset = torch.zeros(2, 29)
    reference_position = torch.stack(
        (joint_default_position, position_upper_limit + 1.0),
    )
    reference_velocity = torch.zeros_like(reference_position)
    lookahead, bounded, achievable, behavior = module._reference_pd_behavior_action(
        reference_position,
        reference_velocity,
        offset,
        action,
        position_lower_limit,
        position_upper_limit,
        0.02,
    )

    torch.testing.assert_close(lookahead, reference_position)
    torch.testing.assert_close(bounded[0], joint_default_position)
    torch.testing.assert_close(bounded[1], position_upper_limit)
    assert torch.all(behavior.abs() <= 1.0)
    torch.testing.assert_close(achievable[0], joint_default_position)
    assert torch.all(achievable >= position_lower_limit)
    assert torch.all(achievable <= position_upper_limit)

    identity = torch.tensor(((0.0, 0.0, 0.0, 1.0),))
    torch.testing.assert_close(module._quaternion_geodesic(identity, -identity), torch.zeros(1))
    policy = module._policy_error_layer()
    assert policy["status"] == "not_measured_no_real_checkpoint_supplied"
    assert policy["reference_controller_is_policy_evidence"] is False
    assert "evaluation_emd" in policy["required_metrics"]

    failed = module._failed_report(
        module.argparse.Namespace(motion_split="evaluation", num_clips=8, num_steps=16, seed=3),
        RuntimeError("simulator failed"),
    )
    assert failed["status"] == "failed"
    assert failed["request"]["row_lifecycle"] == module.ROW_LIFECYCLE
    assert failed["error_layers"]["retarget_fit"]["status"] == "not_reported_run_failed"
    assert failed["error_layers"]["reference_controller_simulator"]["status"] == "failed"
    assert failed["error_layers"]["policy"] == policy

    alive = torch.ones(3, dtype=torch.bool)
    active_before = alive & torch.tensor((True, True, True))
    reached = module._finish_active_rows(
        alive,
        active_before,
        done=torch.tensor((True, False, False)),
        reached_tail_valid=torch.tensor((True, False, True)),
    )
    torch.testing.assert_close(reached, torch.tensor((False, True, True)))
    torch.testing.assert_close(alive, torch.tensor((False, False, True)))
    # Same-Step may expose a valid post-reset clip on the next loop, but the
    # retired done row and exhausted-tail row must never become active again.
    torch.testing.assert_close(alive & torch.ones(3, dtype=torch.bool), torch.tensor((False, False, True)))


def test_persisted_canonical_evidence_separates_and_closes_error_layers() -> None:
    """The real CMU-to-G1 records must cover one identical complete canonical split."""
    from isaaclab_tasks.core.multi_task.motion.data.importers import HumEnvHdf5Clips
    from isaaclab_tasks.core.multi_task.motion.trajectory.g1_smpl import G1SmplHumEnvFrameBuilder
    from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
    from isaaclab_tasks.utils import resolve_presets

    retarget = _load(CANONICAL_RETARGET)
    simulator = _load(CANONICAL_SIMULATOR)

    assert retarget["schema"] == "forward_backward_phase3g_g1_cmu_composition_evidence_v3"
    assert simulator["schema"] == "forward_backward_phase3g_g1_cmu_reference_tracking_evidence_v3"
    assert simulator["status"] == "measured"
    cfg = resolve_presets(MotionImitationEnvCfg(), selected={"g1_cmu"})
    cfg.commands.motion.task_table.motion_split = "evaluation"
    expected_dependency_identity = _environment_identity_module().motion_environment_dependency_identity(
        preset="g1_cmu",
        cfg=cfg,
        importer_type=HumEnvHdf5Clips,
        frame_builder_type=G1SmplHumEnvFrameBuilder,
    )
    assert retarget["code_identity"]["probe_sha256"] == _sha256_file(COMPOSITION_EVIDENCE)
    assert simulator["code_identity"]["probe_sha256"] == _sha256_file(REFERENCE_TRACKING_EVIDENCE)
    identity_module = _environment_identity_module()
    environment_semantics = identity_module.motion_environment_semantic_sha256(expected_dependency_identity)
    assert (
        identity_module.motion_environment_semantic_sha256(simulator["code_identity"]["dependency_identity"])
        == environment_semantics
    )
    expected_composition_identity = identity_module.motion_composition_dependency_identity(
        preset="g1_cmu",
        cfg=cfg,
        importer_type=HumEnvHdf5Clips,
        frame_builder_type=G1SmplHumEnvFrameBuilder,
        frame_builder_identity_sha256=retarget["composition"]["frame_builder_identity_sha256"],
    )
    composition_semantics = identity_module.motion_composition_semantic_sha256(expected_composition_identity)
    for record in (retarget, simulator):
        assert (
            identity_module.motion_composition_semantic_sha256(
                record["code_identity"]["composition_dependency_identity"]
            )
            == composition_semantics
        )
    assert retarget["composition"]["selected"] == simulator["composition"]["selected"] == "g1_cmu"
    assert retarget["composition"]["source"] == simulator["composition"]["source"] == "smpl_cmu"
    assert retarget["composition"]["scene_robot"] == simulator["composition"]["scene_robot"] == "g1_29dof"
    assert (
        retarget["composition"]["frame_builder_identity_sha256"]
        == simulator["composition"]["frame_builder_identity_sha256"]
        == simulator["error_layers"]["retarget_fit"]["frame_builder_identity_sha256"]
    )

    source = retarget["source"]
    selection = simulator["selection"]
    assert source["split"] == "test"
    assert selection["split"] == "evaluation"
    assert source["complete_split"] is True
    assert source["selected_clip_count"] == selection["num_clips"] == 182
    assert source["selected_clip_ids"] == selection["clip_ids"]
    assert source["selected_frame_count"] == 88_364
    assert selection["active_reached_rows"] > 0
    assert selection["unexpected_done_rows"] == 0
    assert selection["row_lifecycle"] == "retire_after_done_or_reference_tail_exhaustion"
    execution = simulator["execution"]
    assert Path(source["artifact_root"]).is_absolute()
    assert Path(execution["source_artifact_root"]).is_absolute()
    assert Path(execution["reference_artifact_root"]).is_absolute()
    assert simulator["composition"]["resolved_environment_axes_unmodified"] is True

    controller = simulator["reference_controller"]
    physical_joint_names = simulator["composition"]["joint_names"]
    behavior_joint_names = list(cfg.actions.joint_position.joint_names)
    behavior_to_physical = [physical_joint_names.index(name) for name in behavior_joint_names]
    assert controller["trajectory_and_simulator_share_physical_axis"] is True
    assert controller["behavior_joint_names"] == behavior_joint_names
    assert controller["behavior_to_physical_joint_indices"] == behavior_to_physical
    assert sorted(behavior_to_physical) == list(range(29))
    assert behavior_to_physical != list(range(29))

    retarget_layers = retarget["error_layers"]
    simulator_layers = simulator["error_layers"]
    assert retarget_layers["retarget_fit"]["status"] == "measured"
    assert simulator_layers["retarget_fit"]["status"] == "measured_by_companion_source_composition_probe"
    assert simulator_layers["reference_controller_simulator"]["status"] == "measured"
    assert simulator_layers["policy"]["status"] == "not_measured_no_real_checkpoint_supplied"
    assert simulator_layers["policy"]["reference_controller_is_policy_evidence"] is False
    assert simulator_layers["retarget_fit"]["composition_semantic_sha256"] == composition_semantics

    assert simulator_layers["retarget_fit"]["evidence_sha256"] == _sha256_file(CANONICAL_RETARGET)
    for metric in (
        "root_position_l2_m",
        "body_position_l2_m",
        "body_root_relative_position_l2_m",
        "joint_position_abs_rad",
        "body_rotation_geodesic_rad",
    ):
        assert simulator_layers["reference_controller_simulator"]["tracking"][metric]["count"] > 0


def test_timing_contract_does_not_hide_reward_or_compute_retiming() -> None:
    """Changed control rates require channel semantics and both budget views."""
    contract = _load(TIMING)
    assert {view["name"] for view in contract["training_budget_views"]} == {
        "physical_exposure",
        "compute_exposure",
    }
    gate = contract["reward_discretization_gate"]
    assert "No blanket dt scaling" in gate["auxiliary_evidence"]
    assert "RewardManager multiplies by control_dt" in gate["environment_scalar"]
    assert set(contract["metrics"]) == {"stability", "tracking", "learning", "safety", "systems"}
    assert contract["decision"]["failure_ownership"] == {
        "physics_only_failure": "simulator_or_controller_preset",
        "open_loop_control_failure": "control_sampling_or_observation_history",
        "reward_discretization_failure": "channel_semantics",
        "training_only_failure": "learner_or_optimization",
    }


def test_provider_migration_keeps_one_transition_interface_and_explicit_data_laws() -> None:
    """Provider stages may change their data law without changing environment tensors."""
    contract = _load(PROVIDERS)
    assert contract["schema"] == "forward_backward_phase3g_motion_provider_migration_v1"
    interface = contract["unchanged_transition_interface"]
    assert interface["observations"] == ["current", "reached", "returned", "final"]
    assert {"episode_continuation", "bootstrap_continuation", "context_continuation"}.issubset(interface["boundary"])
    assert set(interface["reward_facts"]) == {"environment_reward", "raw_named_auxiliary_evidence"}

    stages = {stage["id"]: stage for stage in contract["stages"]}
    assert tuple(stages) == (
        "expert_discriminator_reproduction",
        "own_experience_discriminator",
        "physical_energy",
        "general_value_readout",
    )
    provider_fields = {"name", "output", "timing", "sign", "update", "freeze"}
    assert all(set(stage["provider"]) == provider_fields for stage in stages.values())
    assert stages["expert_discriminator_reproduction"]["expert_data"] == "required"
    assert all(
        stage["expert_data"] == "forbidden"
        for name, stage in stages.items()
        if name != "expert_discriminator_reproduction"
    )

    own = stages["own_experience_discriminator"]
    assert "same episode and unchanged context segment" in own["positive_law"]
    assert "another valid context segment" in own["negative_law"]
    assert any("shuffled-context control" in check for check in own["anti_shortcut_checks"])


def test_energy_and_value_stages_reject_unanchored_self_reward() -> None:
    """Energy and value helpers must remain anchored to declared reward facts."""
    stages = {stage["id"]: stage for stage in _load(PROVIDERS)["stages"]}
    energy = stages["physical_energy"]
    assert energy["provider"]["sign"] == -1
    assert "constant-minus-infinity" in energy["anti_collapse_rule"]
    assert energy["value_path"] == "vector successor over raw energy components, then a fixed linear readout"

    value = stages["general_value_readout"]
    assert value["provider"]["name"] == "none"
    assert "not defined by the value network" in value["reward_identity"]
    assert "not self-supervision" in value["residual_identity"]
    assert "detached V or advantage" in value["actor_path"]
