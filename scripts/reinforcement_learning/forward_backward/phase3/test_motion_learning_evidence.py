# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate exact-corpus attachment and deterministic first-update evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from motion_environment_identity import motion_environment_axes, motion_runner_axes

import isaaclab_tasks.core.multi_task.motion.config  # noqa: F401
from isaaclab_tasks.core.multi_task.motion.config.agents import MotionForwardBackwardRunnerCfg
from isaaclab_tasks.core.multi_task.motion.robots.g1.articulation import G1_BEHAVIOR_JOINT_NAMES
from isaaclab_tasks.utils.hydra import register_task, resolve_presets

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "motion_learning_evidence.py"
FIXTURES = {
    "smpl_cmu": ROOT / "fixtures" / "motion_learning_smpl_cmu_cpu_v1.json",
    "g1_lafan": ROOT / "fixtures" / "motion_learning_g1_lafan_cuda_v1.json",
}
TRACE_PATHS = {
    "smpl_cmu": ROOT / "fixtures" / "meta_humenv_next_step_trace_v2.npz",
    "g1_lafan": ROOT / "fixtures" / "g1_lafan_same_step_trace_v1.npz",
}
SMOKE_CONTRACT = ROOT / "fixtures" / "motion_training_smoke_contract_v2.json"
EXPECTED = {
    "smpl_cmu": {
        "clips": 1_638,
        "source_frames": 730_307,
        "expert_frames": 730_307,
        "feature_width": 358,
        "windows": {"8": 717_203},
        "changed": {
            "actor",
            "backward",
            "discriminator",
            "forward",
            "target_backward",
            "target_forward",
            "target_value/discriminator",
            "value/discriminator",
        },
    },
    "g1_lafan": {
        "clips": 862,
        "source_frames": 258_600,
        "expert_frames": 430_138,
        "feature_width": 527,
        "windows": {"8": 423_242, "257": 208_604},
        "changed": {
            "actor",
            "backward",
            "discriminator",
            "forward",
            "target_backward",
            "target_forward",
            "target_value/auxiliary",
            "target_value/discriminator",
            "value/auxiliary",
            "value/discriminator",
        },
    },
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _module():
    spec = importlib.util.spec_from_file_location("motion_learning_evidence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(preset: str) -> dict[str, object]:
    return json.loads(FIXTURES[preset].read_text())


def _runner_cfg(preset: str) -> MotionForwardBackwardRunnerCfg:
    """Resolve the typed runner from independent environment and algorithm axes."""
    selection = motion_environment_axes(preset) | motion_runner_axes(preset)
    return resolve_presets(MotionForwardBackwardRunnerCfg(), selected=selection)


@pytest.mark.parametrize("preset", ("smpl_cmu", "g1_lafan"))
def test_learning_evidence_is_authentic_and_reports_current_code_compatibility(preset: str) -> None:
    module = _module()
    record = _record(preset)
    source = resolve_presets(
        module.MotionImitationEnvCfg(), selected=motion_environment_axes(preset)
    ).commands.motion.task_table.source

    assert record["schema"] == "forward_backward_phase3_motion_learning_evidence_v1"
    assert record["status"] == "measured"
    code_identity = record["code_identity"]
    assert all(isinstance(digest, str) and len(digest) == 64 for digest in code_identity.values())
    assert record["code_identity"]["native_trace_sha256"] == _sha256(TRACE_PATHS[preset])
    current = {
        "evidence_sha256": _sha256(SCRIPT),
        "motion_expert_provider_sha256": module._source_sha256(module.forward_backward_expert_buffer),
        "rsl_algorithm_sha256": module._source_sha256(module.ForwardBackward),
    }
    mismatched = sorted(name for name, digest in current.items() if code_identity[name] != digest)
    compatibility = {
        "status": "exact_producer_match" if not mismatched else "producer_changed_requires_fresh_canary",
        "mismatched_fields": mismatched,
    }
    assert compatibility["status"] in {"exact_producer_match", "producer_changed_requires_fresh_canary"}
    assert record["source"] == {
        "identifier": source.identifier,
        "split": source.train.name,
        "artifact": source.train.artifact,
        "artifact_sha256": source.train.artifact_sha256,
        "source_content_sha256": source.train.source_content_sha256,
    }

    project_root = ROOT.parents[3]
    deployment = project_root.parent.parent / ("humenv" if preset == "smpl_cmu" else "BFM-Zero")
    artifact = deployment / source.train.artifact
    if artifact.is_file():
        assert _sha256(artifact) == source.train.artifact_sha256


@pytest.mark.parametrize("preset", ("smpl_cmu", "g1_lafan"))
def test_learning_evidence_has_exact_expert_cardinality_and_windows(preset: str) -> None:
    record = _record(preset)
    expert = record["expert"]
    expected = EXPECTED[preset]

    assert expert["clip_count"] == expected["clips"]
    assert expert["source_frame_count"] == expected["source_frames"]
    assert expert["expert_frame_count"] == expected["expert_frames"]
    assert expert["expert_feature_width"] == expected["feature_width"]
    assert expert["window_counts"] == expected["windows"]
    assert all(len(expert[name]) == 64 for name in ("clip_offsets_sha256", "expert_frames_sha256"))
    assert all(len(expert[name]) == 64 for name in ("expert_schema_sha256", "expert_data_sha256"))
    assert expert["zero_copy_native_observations"] is False


@pytest.mark.parametrize("preset", ("smpl_cmu", "g1_lafan"))
def test_learning_evidence_is_repeat_deterministic_and_never_claims_convergence(preset: str) -> None:
    record = _record(preset)
    scope = record["claim_scope"]
    canary = record["first_update_canary"]
    result = canary["result"]

    assert scope == {
        "expert_attachment": "exact_train_corpus",
        "first_update": "deterministic_integration_canary",
        "source_numerical_parity": "inherited_from_phase2_not_remeasured_here",
        "convergence_non_inferiority": "not_evaluated",
    }
    assert canary["status"] == "passed"
    assert canary["claim_scope"] == (
        "deterministic_native_trace_integration_not_convergence_or_source_numerical_parity"
    )
    assert canary["repeat_count"] == 2
    assert canary["batch_size"] == 8
    assert canary["replay_capacity_transitions"] == 32
    assert set(result["changed_components"]) == EXPECTED[preset]["changed"]
    assert result["update_step"] == 1
    assert result["context_buffer_size"] == 8
    assert all(math.isfinite(value) for value in result["metrics"].values())
    assert result["metrics_sha256"] == _module().canonical_sha256(result["metrics"])
    assert len(result["checkpoint_schema_sha256"]) == 64
    assert len(result["model_before"]) == len(result["model_after"]) == 64
    assert result["model_before"] != result["model_after"]


def test_g1_trace_reconstructs_non_model_transition_evidence() -> None:
    """The frozen trace should expose completed-edge evidence through the live observation route."""
    module = _module()
    runner = _runner_cfg("g1_lafan").to_dict()
    with np.load(TRACE_PATHS["g1_lafan"]) as trace:
        current0 = module._trace_observations("g1_lafan", trace, "current", 0, torch.device("cpu"))
        returned0 = module._trace_observations("g1_lafan", trace, "returned", 0, torch.device("cpu"))
        current1 = module._trace_observations("g1_lafan", trace, "current", 1, torch.device("cpu"))
        final2 = module._trace_observations("g1_lafan", trace, "final", 2, torch.device("cpu"))
        current3 = module._trace_observations("g1_lafan", trace, "current", 3, torch.device("cpu"))
        expected0 = torch.from_numpy(trace["learner_auxiliary_raw_evidence"][0].copy())
        expected2 = torch.from_numpy(trace["learner_auxiliary_raw_evidence"][2].copy())

    assert "transition" not in dict(module._expert_schema("g1_lafan", runner).field_widths)
    assert not current0["transition"].any()
    torch.testing.assert_close(returned0["transition"], expected0)
    torch.testing.assert_close(current1["transition"], expected0)
    torch.testing.assert_close(final2["transition"], expected2)
    torch.testing.assert_close(current3["transition"], expected2)


def test_g1_routes_preserve_source_matched_forward_and_critic_coordinate_order() -> None:
    routes = _record("g1_lafan")["learner"]["routes"]
    state = ["joint_position", "joint_velocity", "projected_gravity", "base_angular_velocity"]
    privileged = [*state, "privileged_state"]
    evaluator = [*privileged, "last_action", "history_actor"]

    assert routes["forward"] == evaluator
    assert routes["critic_discriminator"] == evaluator
    assert set(routes) >= {"actor", "forward", "backward", "discriminator", "critic_discriminator"}
    assert routes["actor"] == [*state, "last_action", "history_actor"]
    assert routes["backward"] == privileged
    assert routes["discriminator"] == privileged
    current_routes = _runner_cfg("g1_lafan").to_dict()["obs_groups"]
    route_compatibility = "exact_route_match" if routes == current_routes else "route_changed_requires_fresh_canary"
    assert route_compatibility in {"exact_route_match", "route_changed_requires_fresh_canary"}


def test_g1_learning_stub_separates_behavior_action_axis_from_physical_table_axis() -> None:
    """The learner-only environment must retain the same named axes as the live environment."""
    module = _module()
    cfg = resolve_presets(module.MotionImitationEnvCfg(), selected=motion_environment_axes("g1_lafan"))
    cfg.sim.device = "cpu"
    live_joint_names = tuple(cfg.scene.robot.actuators["motion"].joint_names_expr)
    table = SimpleNamespace(
        joint_names=live_joint_names,
        reference_frame_names=(),
        device=torch.device("cpu"),
    )

    env = module._learner_env(table, cfg, "g1_lafan")
    action = env.action_manager.get_term("joint_position")

    assert action.joint_names == G1_BEHAVIOR_JOINT_NAMES
    assert env.scene["robot"].joint_names == live_joint_names
    assert action.joint_names != env.scene["robot"].joint_names
    expected_defaults = torch.tensor([cfg.scene.robot.init_state.joint_pos[name] for name in action.joint_names])
    torch.testing.assert_close(action.joint_default_position, expected_defaults)


def test_smpl_live_axes_use_native_mjcf_joint_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CPU evidence probe must reproduce the live Newton articulation axis."""
    module = _module()
    body_names = ("Pelvis", *(f"Body_{index}" for index in range(23)))
    mjcf_joint_names = (
        "Pelvis",
        *(f"Body_{index}_x_Body_{index}_y_Body_{index}_z" for index in range(23)),
    )
    kinematics = SimpleNamespace(body_names=body_names, joint_names=mjcf_joint_names)
    monkeypatch.setattr(module, "NewtonKinematics", lambda _cfg: kinematics)
    cfg = SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(spawn=SimpleNamespace(asset_path="smpl.xml"))))

    joint_names, resolved_body_names = module._live_axes("smpl_cmu", cfg, torch.device("cpu"))

    assert resolved_body_names == body_names
    assert joint_names == tuple(f"{name}:{component}" for name in mjcf_joint_names[1:] for component in range(3))


@pytest.mark.parametrize("preset", ("smpl_cmu", "g1_lafan"))
def test_canary_transform_changes_only_execution_scale_and_provider(preset: str) -> None:
    module = _module()
    runner = _runner_cfg(preset).to_dict()
    transformed = module._canary_config(runner, None)
    expected = copy.deepcopy(runner)
    expected["replay"]["capacity_transitions"] = 32
    expected["algorithm"]["batch_size"] = 8
    expected["torch_compile_mode"] = None
    transformed["expert"]["provider"] = expected["expert"]["provider"]

    assert transformed == expected


@pytest.mark.parametrize("preset", ("smpl_cmu", "g1_lafan"))
def test_native_training_smoke_changes_only_declared_execution_scale(preset: str) -> None:
    """The smoke must preserve production learning semantics outside three execution controls."""
    contract = json.loads(SMOKE_CONTRACT.read_text())
    profile = contract["profiles"][preset]
    overrides = profile["execution_scale_overrides"]
    production = _runner_cfg(preset).to_dict()
    smoke = copy.deepcopy(production)
    smoke["max_iterations"] = overrides["max_iterations"]
    smoke["lifecycle_extension"] = overrides["lifecycle_extension"]
    smoke["replay"]["capacity_transitions"] = overrides["replay_capacity_transitions"]

    def changed_paths(left: object, right: object, prefix: str = "") -> set[str]:
        if isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                return {prefix or "<root>"}
            return {
                path
                for key in left
                for path in changed_paths(left[key], right[key], f"{prefix}.{key}" if prefix else key)
            }
        return set() if left == right else {prefix}

    expected = {"max_iterations", "replay.capacity_transitions"}
    if production["lifecycle_extension"] is not None:
        expected.add("lifecycle_extension")
    assert changed_paths(production, smoke) == expected
    assert overrides == {
        "max_iterations": profile["collection"]["iterations"],
        "lifecycle_extension": None,
        "replay_capacity_transitions": profile["collection"]["expected_transitions"],
    }
    assert overrides["replay_capacity_transitions"] % profile["collection"]["num_envs"] == 0
    command = profile["command"]
    selection = next(token for token in command if token.startswith("presets="))
    selected = set(selection.removeprefix("presets=").split(","))
    assert "tracking_off" in selected
    assert selected.isdisjoint({"tracking_source_edge", "tracking_reset_frame"})
    assert not any(token.startswith("agent.lifecycle_extension=") for token in command)
    assert f"agent.replay.capacity_transitions={overrides['replay_capacity_transitions']}" in command
    assert not any(token.startswith("agent.run_name=") for token in command)


@pytest.mark.parametrize("preset", ("smpl_cmu", "g1_lafan"))
def test_native_training_smoke_reports_current_learner_compatibility(preset: str) -> None:
    """A frozen training contract reports current drift without changing its identity."""
    import rsl_rl
    from rsl_rl.runners.off_policy_runner import OffPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    receipt_path = ROOT / "motion_training_receipt.py"
    spec = importlib.util.spec_from_file_location("motion_training_receipt_contract", receipt_path)
    assert spec is not None and spec.loader is not None
    receipt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(receipt)

    profile = json.loads(SMOKE_CONTRACT.read_text())["profiles"][preset]
    overrides = profile["execution_scale_overrides"]
    agent_cfg = _runner_cfg(preset)
    agent_cfg.max_iterations = overrides["max_iterations"]
    agent_cfg.lifecycle_extension = overrides["lifecycle_extension"]
    agent_cfg.replay.capacity_transitions = overrides["replay_capacity_transitions"]
    wrapper = object.__new__(RslRlVecEnvWrapper)
    package_root = Path(rsl_rl.__file__).resolve().parent
    identity = profile["closed_input_identity"]

    runtime_identity = receipt._learner_runtime_identity()
    assert set(runtime_identity["packages"]) == {"gymnasium", "tensordict"}
    bridge_identity = receipt._task_bridge_identity(wrapper, agent_cfg)
    assert set(bridge_identity["source_owners"]) == {
        "environment_wrapper",
        "motion_expert_provider",
        "motion_runner_config",
    }
    current = {
        "runner_source_sha256": receipt._owner_source(OffPolicyRunner)[1],
        "python_source_identity_sha256": receipt._file_sha256(ROOT / "python_source_identity.py"),
        "learner_code_bundle_sha256": receipt._python_package_bundle_sha256(package_root),
        "learner_runtime_bundle_sha256": runtime_identity["bundle_sha256"],
        "task_bridge_code_bundle_sha256": bridge_identity["bundle_sha256"],
        "resolved_agent_config_sha256": receipt._resolved_agent_config_sha256(agent_cfg),
        "training_cli_sha256": receipt._file_sha256(ROOT.parents[1] / "rsl_rl" / "train_rsl_rl.py"),
        "receipt_code_sha256": receipt._file_sha256(receipt_path),
    }
    assert all(isinstance(digest, str) and len(digest) == 64 for digest in identity.values())
    mismatched = sorted(name for name, digest in current.items() if identity[name] != digest)
    status = "exact_producer_match" if not mismatched else "producer_changed_requires_fresh_smoke"
    assert status in {"exact_producer_match", "producer_changed_requires_fresh_smoke"}


@pytest.mark.parametrize("preset", ("smpl_cmu", "g1_lafan"))
def test_native_training_identity_differs_from_phase3e_only_by_motion_split(preset: str) -> None:
    """Frozen Phase 3E evaluation identity must transfer to train with one declared semantic difference."""
    from isaaclab_tasks.core.multi_task.motion.data.sources import CmuHumEnvSmplClips, LafanG1JoblibClips
    from isaaclab_tasks.core.multi_task.motion.robots.g1.reference import G1PoseFrameBuilder
    from isaaclab_tasks.core.multi_task.motion.robots.smpl.reference import SmplGeneralizedCoordinateFrameBuilder
    from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg

    identity_path = ROOT / "motion_environment_identity.py"
    spec = importlib.util.spec_from_file_location("phase3f_environment_identity", identity_path)
    assert spec is not None and spec.loader is not None
    identity_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(identity_module)

    importer = CmuHumEnvSmplClips if preset == "smpl_cmu" else LafanG1JoblibClips
    frame_builder = SmplGeneralizedCoordinateFrameBuilder if preset == "smpl_cmu" else G1PoseFrameBuilder
    reference_root = None
    train_cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes(preset))
    evaluation_cfg = resolve_presets(MotionImitationEnvCfg(), selected=motion_environment_axes(preset))
    evaluation_cfg.commands.motion.task_table.motion_split = "evaluation"

    def environment_identity(cfg: object) -> dict[str, object]:
        return identity_module.motion_environment_dependency_identity(
            preset=preset,
            cfg=cfg,
            importer_type=importer,
            frame_builder_type=frame_builder,
            reference_artifact_root=reference_root,
        )

    train = environment_identity(train_cfg)
    evaluation = environment_identity(evaluation_cfg)
    contract = json.loads(SMOKE_CONTRACT.read_text())
    bridge = contract["phase3e_to_phase3f_environment_bridge"]
    profile = contract["profiles"][preset]
    frozen = profile["closed_input_identity"]

    assert train_cfg.commands.motion.task_table.motion_split == bridge["phase3f_motion_split"]
    assert evaluation_cfg.commands.motion.task_table.motion_split == bridge["phase3e_motion_split"]
    assert train["resolved_axes"] == evaluation["resolved_axes"]
    assert train["resolved_axes_sha256"] == evaluation["resolved_axes_sha256"]
    current_semantic = identity_module.motion_environment_semantic_sha256(train)
    compatibility = (
        "exact_environment_match"
        if train["resolved_axes_sha256"] == frozen["resolved_axes_sha256"]
        and current_semantic == profile["environment_semantic_sha256"]
        else "environment_changed_requires_fresh_smoke"
    )
    assert compatibility in {"exact_environment_match", "environment_changed_requires_fresh_smoke"}

    def changed_paths(left: object, right: object, prefix: str = "") -> set[str]:
        if isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                return {prefix or "<root>"}
            return {
                path
                for key in left
                for path in changed_paths(left[key], right[key], f"{prefix}.{key}" if prefix else key)
            }
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                return {prefix}
            return {
                path
                for index in range(len(left))
                for path in changed_paths(left[index], right[index], f"{prefix}[{index}]")
            }
        return set() if left == right else {prefix}

    assert changed_paths(train["resolved_configuration"], evaluation["resolved_configuration"]) == set(
        bridge["required_resolved_configuration_difference"]
    )
    assert bridge["required_resolved_axes_relation"] == "exact_equal"


@pytest.mark.parametrize("preset", ("smpl_cmu", "g1_lafan"))
def test_native_training_smoke_overrides_resolve_through_real_task_registry(
    preset: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every config token must be consumed by the production preset resolver."""
    profile = json.loads(SMOKE_CONTRACT.read_text())["profiles"][preset]
    source_root = f"/tmp/phase3f/{preset}/source"
    reference_root = f"/tmp/phase3f/{preset}/reference"
    for command_name in ("command", "validation_command_template"):
        command = profile[command_name]
        assert not any("/home/" in token for token in command)
        assert "env.commands.motion.task_table.source_artifact_root=<SOURCE_ARTIFACT_ROOT>" in command
        if preset == "g1_lafan":
            assert "env.commands.motion.task_table.reference_artifact_root=<REFERENCE_ARTIFACT_ROOT>" in command
    config_tokens = [
        token.replace("<SOURCE_ARTIFACT_ROOT>", source_root).replace("<REFERENCE_ARTIFACT_ROOT>", reference_root)
        for token in profile["command"]
        if token.startswith(("presets=", "env.", "agent."))
    ]
    monkeypatch.setattr(sys, "argv", ["phase3f-smoke", *config_tokens])

    env_cfg, agent_cfg, hydra_args = register_task("Isaac-Motion-Imitation-v0", "rsl_rl_cfg_entry_point")

    assert hydra_args == []
    table_cfg = env_cfg.commands.motion.task_table
    expected_source = "cmu_humenv_smpl" if preset == "smpl_cmu" else "lafan_g1_29dof"
    assert table_cfg.source.identifier == expected_source
    assert table_cfg.source_artifact_root == source_root
    assert table_cfg.reference_artifact_root == ("" if preset == "smpl_cmu" else reference_root)
    assert agent_cfg.lifecycle_extension is None
    assert agent_cfg.replay.capacity_transitions == profile["collection"]["expected_transitions"]


def test_native_training_smokes_are_predeclared_but_not_launched() -> None:
    contract = json.loads(SMOKE_CONTRACT.read_text())

    assert contract["schema"] == "forward_backward_phase3_motion_training_smoke_contract_v2"
    assert contract["status"] == "prepared_not_launched"
    assert contract["launch_gate"] == "wait_for_final_phase3e_simulator_and_cloner_identity"
    assert (
        contract["claim_scope"]
        == "native_environment_learner_integration_one_update_group_not_convergence_evaluator_or_systems"
    )
    for preset, profile in contract["profiles"].items():
        runner = _runner_cfg(preset)
        collection = profile["collection"]
        assert collection["num_envs"] == runner.schedule.num_envs
        assert collection["steps_per_iteration"] == runner.schedule.num_steps_per_env
        assert collection["random_action_transitions"] == runner.schedule.random_action_steps
        assert collection["expected_transitions"] == (
            collection["num_envs"] * collection["steps_per_iteration"] * collection["iterations"]
        )
        assert collection["updates_per_group"] == runner.schedule.num_updates_per_iteration
        assert collection["expected_update_groups"] == 1
        assert collection["expected_update_calls"] == (
            collection["expected_update_groups"] * collection["updates_per_group"]
        )
        command = profile["command"]
        assert command[:6] == [
            "./isaaclab.sh",
            "train",
            "--rl_library",
            "rsl_rl",
            "--task",
            "Isaac-Motion-Imitation-v0",
        ]
        selection = next(token for token in command if token.startswith("presets="))
        selected = set(selection.removeprefix("presets=").split(","))
        expected = set(motion_environment_axes(preset)) | {
            axis for axis in motion_runner_axes(preset) if not axis.startswith("tracking_")
        }
        expected.add("tracking_off")
        assert selected == expected
        evidence = _record(preset)
        identity = profile["closed_input_identity"]
        assert identity["task_table_sha256"] == evidence["task_table"]["identity_sha256"]
        assert identity["expert_schema_sha256"] == evidence["expert"]["expert_schema_sha256"]
        assert identity["observation_schema_sha256"] == evidence["learner"]["observation_schema_sha256"]
    assert "missing_or_nonfinite_metric_or_actor_state" in contract["failure"]
    assert "simulator_cloner_or_contact_identity_changes_after_launch" in contract["failure"]
    assert "final_observation_capture_and_systems_headroom_owned_by_phase3e" in contract["excluded_claims"]
