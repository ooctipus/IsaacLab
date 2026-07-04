# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the frozen G1 native-edge replay decision and claim boundary."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

from motion_environment_identity import motion_environment_axes

from isaaclab_tasks.core.multi_task.motion.data.sources import LafanG1JoblibClips
from isaaclab_tasks.core.multi_task.motion.robots.g1.reference import G1PoseFrameBuilder
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils import resolve_presets

ARTIFACT = Path(__file__).parent / "fixtures/runtime/g1_native_edge_replay_v3.json"
PROBE = Path(__file__).parent / "generate_g1_native_edge_replay.py"
IDENTITY = Path(__file__).with_name("motion_environment_identity.py")


def _report() -> dict:
    return json.loads(ARTIFACT.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_module():
    """Load the shared portable environment-identity validator."""
    spec = importlib.util.spec_from_file_location("g1_edge_motion_environment_identity", IDENTITY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _current_controlled_environment_identity() -> dict[str, object]:
    """Reconstruct the producer's controlled G1 environment semantics."""
    cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes("g1_lafan"))
    action = cfg.actions.joint_position
    action.default_joint_offset_range = (0.0, 0.0)
    for name in ("joint_position", "joint_velocity", "projected_gravity", "base_angular_velocity"):
        getattr(cfg.observations, name).enable_corruption = False
    assert cfg.events.robot_material is not None
    assert cfg.events.body_mass is not None
    assert cfg.events.torso_com is not None
    cfg.events.robot_material.params["static_friction_range"] = (1.0, 1.0)
    cfg.events.robot_material.params["dynamic_friction_range"] = (1.0, 1.0)
    cfg.events.robot_material.params["num_buckets"] = 1
    cfg.events.body_mass.params["mass_distribution_params"] = (1.0, 1.0)
    cfg.events.torso_com.params["com_range"] = {axis: (0.0, 0.0) for axis in "xyz"}
    cfg.events.push = None
    cfg.commands.motion.task_table.motion_split = "evaluation"
    return _identity_module().motion_environment_dependency_identity(
        preset="g1_lafan",
        cfg=cfg,
        importer_type=LafanG1JoblibClips,
        frame_builder_type=G1PoseFrameBuilder,
    )


def test_g1_native_edge_portable_semantics_exclude_runtime_build_identity() -> None:
    """Host package/build drift must not invalidate unchanged environment semantics."""
    module = _identity_module()
    current = _current_controlled_environment_identity()
    changed = copy.deepcopy(current)
    changed["runtime_dependencies"] = {"different_host_runtime": True}
    changed["runtime_dependencies_sha256"] = module._json_hash(changed["runtime_dependencies"])
    payload = dict(changed)
    payload.pop("bundle_sha256")
    changed["bundle_sha256"] = module._json_hash(payload)

    assert module.motion_environment_semantic_sha256(changed) == module.motion_environment_semantic_sha256(current)


def test_g1_native_edge_replay_closes_exact_and_current_contracts() -> None:
    """The frozen decision must pass every claim that has sufficient source facts."""
    report = _report()
    assert report["schema"] == "forward_backward_phase3e_g1_native_edge_replay_v3"
    assert report["profile"] == "g1_lafan"
    assert report["oracle"]["edges"] == 10
    assert report["decision"] == {
        "current_observation_contract_passed": True,
        "exact_contract_passed": True,
        "passed": True,
        "simulator_elementwise_parity_claimed": False,
    }


def test_g1_native_edge_replay_is_authentic_and_reports_current_compatibility() -> None:
    """The historical replay remains closed while current compatibility is explicit."""
    identity = _report()["code_identity"]
    assert isinstance(identity["probe_sha256"], str) and len(identity["probe_sha256"]) == 64
    dependency = identity["dependency_identity"]
    module = _identity_module()
    stored_semantic = module.motion_environment_semantic_sha256(dependency)
    assert identity["environment_semantic_sha256"] == stored_semantic
    compatibility = module.motion_environment_compatibility(
        dependency,
        _current_controlled_environment_identity(),
    )
    assert compatibility["status"] in {
        "exact_producer_match",
        "declared_contract_match_requires_runtime_validation",
        "declared_contract_mismatch",
    }
    assert len(_sha256(PROBE)) == 64


def test_g1_native_edge_probe_consumes_the_declared_behavior_axis() -> None:
    """The evidence producer must not remap source behavior tensors into simulator order."""
    source = PROBE.read_text()
    assert "table.joint_names != simulator_joint_names" in source
    assert "action.joint_names != source_joint_names" in source
    assert "joint_ids = action.joint_ids" in source
    assert "simulator_joint_ids = joint_ids.to(dtype=torch.int32)" in source
    assert "_state_source_to_live" not in source
    assert "_history_source_to_live" not in source


def test_g1_native_edge_replay_closes_physics_and_drive_facts() -> None:
    """Every exposed randomized physical fact and joint drive must read back."""
    exact = _report()["candidate_replay"]["exact"]
    physics = exact["injected_physics_fact_readback"]
    assert physics["passed"]
    for name in ("body_mass", "body_com_pose", "shape_material"):
        assert physics[name]["passed"]
        assert physics[name]["max_abs"] == 0.0
    inertia = physics["body_inertia"]
    assert inertia["passed"]
    assert inertia["atol"] == 1.0e-7
    assert inertia["max_abs"] <= inertia["atol"]

    drives = exact["joint_drive_readback"]
    assert drives["passed"]
    for name, result in drives.items():
        if name != "passed":
            assert result["passed"], name
            assert result["max_abs"] == 0.0


def test_g1_native_edge_replay_preserves_exact_same_step_lifecycle() -> None:
    """Done, exact-final, action, history, and evidence routing must all pass."""
    exact = _report()["candidate_replay"]["exact"]
    assert exact["done_mask"]["passed"]
    assert exact["done_mask"]["expected_done_rows"] == 2
    assert exact["final_observation_valid"]["passed"]
    assert exact["final_observation_valid"]["expected_rows"] == 2
    for name in (
        "injected_qpos_readback",
        "injected_qvel_readback",
        "controller_target_joint_position",
        "history_recurrence",
        "auxiliary_evidence_selection",
        "environment_reward_from_raw_evidence",
        "all_reached_rows_captured",
    ):
        assert exact[name]["passed"], name


def test_g1_native_edge_replay_keeps_cross_solver_residual_measured() -> None:
    """Matched inputs and control must not be overstated as elementwise physics parity."""
    report = _report()
    replay = report["candidate_replay"]
    substep = replay["substep_transition"]
    assert substep["claim"] == "measured_cross_simulator_physics_not_elementwise_parity"
    assert len(substep["substeps"]) == 4
    first = substep["substeps"][0]
    assert first["applied_pd_torque"]["max_abs"] <= 1.0e-5
    assert first["qvel"]["root_linear_velocity"]["max_abs"] <= 0.02

    reached = replay["reached_transition"]
    assert reached["claim"] == "measurement_after_all_exposed_source_transition_facts_injected"
    assert reached["qpos"]["max_abs"] > 0.0
    assert reached["contact_force"]["max_abs"] > 0.0
    assert not report["decision"]["simulator_elementwise_parity_claimed"]
