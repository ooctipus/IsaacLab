# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the hash-closed native/cross G1 policy-quality contract."""

from __future__ import annotations

import ast
import csv
import importlib.util
import inspect
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parent
EVALUATOR = ROOT / "g1_policy_quality_evidence.py"
IDENTITY = ROOT / "motion_environment_identity.py"
QUALITY_GATE = ROOT / "fixtures" / "g1_lafan_policy_quality_gate_v1.json"
QUALITY_PROTOCOL_AUDIT = ROOT / "fixtures" / "g1_lafan_policy_quality_protocol_audit_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("g1_policy_quality_evidence", EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity_module():
    spec = importlib.util.spec_from_file_location("motion_environment_identity", IDENTITY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_sha256_unwraps_a_decorated_callable_to_its_implementation() -> None:
    """Decorator infrastructure must not replace the callable's implementation identity."""
    module = _module()

    @torch.inference_mode()
    def decorated() -> None:
        pass

    wrapper_path = Path(inspect.getsourcefile(decorated) or "")
    assert wrapper_path != Path(__file__).resolve()
    assert module._source_sha256(decorated) == module._sha256(Path(__file__).resolve())


def test_native_policy_quality_gate_is_frozen_to_exact_phase2_evidence() -> None:
    """The corrected native rerun must be judged against one immutable baseline."""
    gate = json.loads(QUALITY_GATE.read_text())

    assert gate["schema"] == "forward_backward_phase3g_g1_lafan_policy_quality_gate_v1"
    assert gate["status"] == "frozen_before_corrected_policy_evaluation"
    assert gate["checkpoint"] == {
        "transition": 9_600_000,
        "training_seed": 4728,
        "sha256": "6be44a20885b4b40751d61b9c5b88cf7c27f98d2bdc2e33d3004b4360118d2cb",
    }
    protocol = gate["protocol"]
    assert protocol["evaluation_seed"] == 4728
    assert protocol["clip_count"] == 862
    assert protocol["frame_count"] == 258_600
    assert protocol["domain_randomization"] is True
    assert protocol["observation_noise"] is True
    assert protocol["reward_task_count"] * protocol["episodes_per_task"] == 380
    assert protocol["reward_horizon"] == 500
    baseline = gate["phase2_baseline"]
    for name in (
        "tracking_manifest_sha256",
        "tracking_metrics_sha256",
        "reward_manifest_sha256",
        "reward_metrics_sha256",
    ):
        assert len(baseline[name]) == 64
    assert baseline["tracking"]["emd_mean"] == pytest.approx(1.317105856436808)
    assert baseline["tracking"]["obs_state_emd_mean"] == pytest.approx(1.144622921994978)
    assert gate["acceptance"]["tracking"]["coverage_fraction_min"] == 1.0
    assert gate["acceptance"]["broad_reward"]["safety_violation_rate_mean_increase_max"] == 0.001
    assert "tautological" in gate["comparison"]
    assert gate["cross_source_measurement"]["clip_count"] == 182


def test_policy_quality_protocol_audit_freezes_the_future_episode_bank_boundary() -> None:
    """Future reward authority must start from one backend-neutral realized episode bank."""
    audit = json.loads(QUALITY_PROTOCOL_AUDIT.read_text())

    assert audit["schema"] == "forward_backward_phase3g_g1_policy_quality_protocol_audit_v1"
    assert audit["status"] == "frozen_after_reset_corpus_audit"
    assert audit["phase_ownership"] == {
        "phase2": "convergence_and_policy_quality",
        "phase3": "deterministic_environment_learner_integration_and_structural_composition",
        "broad_reward_role_in_phase3": "diagnostic_only",
    }
    evidence = audit["audited_evidence"]
    assert evidence["classification"] == "inconclusive_protocol_identity"
    assert evidence["authoritative"] is False
    assert evidence["episode_rows_per_metric"] == 380
    point_results = evidence["historical_point_results"]
    assert point_results["threshold_ratio_minimum"] == 0.95
    assert point_results["pre_sampler_result"] == "met"
    assert point_results["post_sampler_result"] == "not_met"
    future = audit["future_authority_contract"]
    assert future["schema"] == "forward_backward_g1_broad_reward_episode_bank_v1"
    assert set(future["episode_bank_fields"]) == {
        "motion",
        "reset",
        "domain_randomization",
        "observation_noise",
        "push",
    }
    assert future["episode_bank_fields"]["motion"]["clip_id"] == "exact_stable_source_clip_id"
    assert future["episode_bank_fields"]["motion"]["continuous_reset_time_seconds"] == "exact_binary64_value"
    assert future["episode_bank_fields"]["reset"]["fall_state"].startswith("exact_pose_velocity")
    assert future["episode_bank_fields"]["observation_noise"]["owner"] == "backend_neutral_counter_stream"


def _native_gate_inputs():
    gate = json.loads(QUALITY_GATE.read_text())
    baseline = gate["phase2_baseline"]
    return (
        gate,
        dict(gate["checkpoint"]),
        dict(gate["protocol"]),
        {
            **baseline["tracking"],
            "coverage_fraction": gate["protocol"]["tracking_coverage_fraction"],
        },
        dict(baseline["broad_reward"]),
    )


def test_native_quality_decision_passes_exact_baseline_and_exposes_every_direction() -> None:
    """The pure native consumer must report every frozen comparison explicitly."""
    module = _module()
    gate, checkpoint, protocol, tracking, broad_reward = _native_gate_inputs()

    decision = module._native_quality_decision(
        gate,
        checkpoint=checkpoint,
        protocol=protocol,
        tracking=tracking,
        broad_reward=broad_reward,
    )

    assert decision["passed"] is True
    assert decision["status"] == "passed"
    assert decision["threshold_applied"] is True
    assert {name: result["direction"] for name, result in decision["metrics"]["tracking"].items()} == {
        "emd_mean": "absolute_delta_max",
        "obs_state_emd_mean": "absolute_delta_max",
        "coverage_fraction": "minimum",
    }
    reward_metrics = decision["diagnostics"]["broad_reward"]["point_gate"]["metrics"]
    assert {name: result["direction"] for name, result in reward_metrics.items()} == {
        "return_mean": "ratio_minimum",
        "safety_violation_rate_mean": "increase_max",
        "termination_rate_mean": "maximum",
        "auxiliary_cost_mean": "absolute_delta_max",
        "action_l2_mean": "absolute_delta_max",
    }
    assert all(result["point_gate_met"] for result in reward_metrics.values())


def test_broad_reward_cannot_authorize_a_pass_without_shared_realized_corpus_identities() -> None:
    """A numeric point gate cannot become non-inferiority evidence without paired corpus identity."""
    module = _module()
    gate, checkpoint, protocol, tracking, broad_reward = _native_gate_inputs()

    decision = module._native_quality_decision(
        gate,
        checkpoint=checkpoint,
        protocol=protocol,
        tracking=tracking,
        broad_reward=broad_reward,
    )

    diagnostic = decision["diagnostics"]["broad_reward"]
    assert diagnostic["status"] == "inconclusive"
    assert diagnostic["authoritative"] is False
    assert "passed" not in diagnostic
    assert diagnostic["point_gate"]["result"] == "met"
    assert diagnostic["point_gate"]["metrics"]["return_mean"]["limit"] == 0.95


def test_fresh_broad_return_point_failure_remains_a_non_authorizing_fact() -> None:
    """The fresh hard point result must remain not met without failing the Phase 3 tracking gate."""
    module = _module()
    gate, checkpoint, protocol, tracking, broad_reward = _native_gate_inputs()
    broad_reward["return_mean"] = 51.10669467049539

    decision = module._native_quality_decision(
        gate,
        checkpoint=checkpoint,
        protocol=protocol,
        tracking=tracking,
        broad_reward=broad_reward,
    )

    diagnostic = decision["diagnostics"]["broad_reward"]
    return_point = diagnostic["point_gate"]["metrics"]["return_mean"]
    assert decision["passed"] is True
    assert diagnostic["classification"] == "inconclusive_protocol_identity"
    assert diagnostic["point_gate"]["result"] == "not_met"
    assert return_point["ratio"] == pytest.approx(0.9313453136217896)
    assert return_point["limit"] == 0.95


def test_broad_reward_identity_closure_requires_matching_bank_and_assignment_digests() -> None:
    """Seed equality must not substitute for shared bank and realized-assignment identities."""
    module = _module()
    audit = json.loads(QUALITY_PROTOCOL_AUDIT.read_text())
    digest_a = "a" * 64
    digest_b = "b" * 64
    closed = module._broad_reward_identity_closure(
        audit,
        {
            "baseline_episode_bank_sha256": digest_a,
            "candidate_episode_bank_sha256": digest_a,
            "baseline_realized_assignment_sha256": digest_b,
            "candidate_realized_assignment_sha256": digest_b,
        },
    )
    mismatched = module._broad_reward_identity_closure(
        audit,
        {
            "baseline_episode_bank_sha256": digest_a,
            "candidate_episode_bank_sha256": digest_b,
            "baseline_realized_assignment_sha256": digest_b,
            "candidate_realized_assignment_sha256": digest_b,
        },
    )

    assert closed["identity_closed"] is True
    assert mismatched["identity_closed"] is False
    assert mismatched["status"] == "inconclusive_protocol_identity"


def test_native_quality_decision_applies_each_authoritative_or_diagnostic_direction() -> None:
    """Every frozen direction must remain observable at its declared authority level."""
    module = _module()
    gate, checkpoint, protocol, baseline_tracking, baseline_reward = _native_gate_inputs()
    acceptance = gate["acceptance"]
    cases = (
        (
            "tracking",
            "emd_mean",
            baseline_tracking["emd_mean"] + acceptance["tracking"]["emd_mean_absolute_delta_max"] + 0.001,
        ),
        (
            "tracking",
            "obs_state_emd_mean",
            baseline_tracking["obs_state_emd_mean"]
            + acceptance["tracking"]["obs_state_emd_mean_absolute_delta_max"]
            + 0.001,
        ),
        ("tracking", "coverage_fraction", acceptance["tracking"]["coverage_fraction_min"] - 0.001),
        (
            "broad_reward",
            "return_mean",
            baseline_reward["return_mean"] * (acceptance["broad_reward"]["return_mean_ratio_min"] - 0.001),
        ),
        (
            "broad_reward",
            "termination_rate_mean",
            acceptance["broad_reward"]["termination_rate_mean_max"] + 0.001,
        ),
        (
            "broad_reward",
            "auxiliary_cost_mean",
            baseline_reward["auxiliary_cost_mean"]
            + acceptance["broad_reward"]["auxiliary_cost_mean_absolute_delta_max"]
            + 0.001,
        ),
        (
            "broad_reward",
            "action_l2_mean",
            baseline_reward["action_l2_mean"] + acceptance["broad_reward"]["action_l2_mean_absolute_delta_max"] + 0.001,
        ),
    )
    for group, name, failed_value in cases:
        tracking = dict(baseline_tracking)
        broad_reward = dict(baseline_reward)
        (tracking if group == "tracking" else broad_reward)[name] = failed_value

        decision = module._native_quality_decision(
            gate,
            checkpoint=checkpoint,
            protocol=protocol,
            tracking=tracking,
            broad_reward=broad_reward,
        )

        if group == "tracking":
            assert decision["passed"] is False, (group, name)
            assert decision["status"] == "failed"
            assert decision["metrics"][group][name]["passed"] is False
        else:
            diagnostic = decision["diagnostics"]["broad_reward"]
            assert decision["passed"] is True, (group, name)
            assert diagnostic["point_gate"]["result"] == "not_met"
            assert diagnostic["point_gate"]["metrics"][name]["point_gate_met"] is False


def test_native_quality_safety_point_rule_marks_the_maximum_valid_rate_not_met() -> None:
    """The frozen diagnostic safety point rule must mark a rate of one not met."""
    module = _module()
    gate, checkpoint, protocol, tracking, broad_reward = _native_gate_inputs()
    broad_reward["safety_violation_rate_mean"] = 1.0

    decision = module._native_quality_decision(
        gate,
        checkpoint=checkpoint,
        protocol=protocol,
        tracking=tracking,
        broad_reward=broad_reward,
    )

    safety = decision["diagnostics"]["broad_reward"]["point_gate"]["metrics"]["safety_violation_rate_mean"]
    assert safety["direction"] == "increase_max"
    assert safety["point_gate_met"] is False
    assert safety["baseline"] + safety["limit"] < 1.0
    assert decision["passed"] is True


@pytest.mark.parametrize(
    "group,name,bad_value",
    (
        ("tracking", "emd_mean", -0.001),
        ("tracking", "obs_state_emd_mean", -0.001),
        ("tracking", "coverage_fraction", -0.001),
        ("tracking", "coverage_fraction", 1.001),
        ("broad_reward", "safety_violation_rate_mean", -0.001),
        ("broad_reward", "safety_violation_rate_mean", 1.001),
        ("broad_reward", "termination_rate_mean", -0.001),
        ("broad_reward", "termination_rate_mean", 1.001),
        ("broad_reward", "auxiliary_cost_mean", -0.001),
        ("broad_reward", "action_l2_mean", -0.001),
    ),
)
def test_native_quality_decision_rejects_out_of_domain_metrics(group, name, bad_value) -> None:
    """Malformed signs and rates must fail validation rather than pass an upper bound."""
    module = _module()
    gate, checkpoint, protocol, tracking, broad_reward = _native_gate_inputs()
    (tracking if group == "tracking" else broad_reward)[name] = bad_value

    with pytest.raises(ValueError, match="non-negative|\\[0, 1\\]"):
        module._native_quality_decision(
            gate,
            checkpoint=checkpoint,
            protocol=protocol,
            tracking=tracking,
            broad_reward=broad_reward,
        )


@pytest.mark.parametrize(
    "name,bad_value,error",
    (
        ("clip_count", 0, "positive integer"),
        ("frame_count", True, "positive integer"),
        ("reward_task_count", 0, "positive integer"),
        ("episodes_per_task", 0, "positive integer"),
        ("reward_horizon", 0, "positive integer"),
        ("domain_randomization", 1, "boolean"),
        ("observation_noise", 1, "boolean"),
    ),
)
def test_native_quality_decision_rejects_malformed_protocol_fields(name, bad_value, error) -> None:
    """Protocol booleans and counts must retain their declared types and domains."""
    module = _module()
    gate, checkpoint, protocol, tracking, broad_reward = _native_gate_inputs()
    protocol[name] = bad_value

    with pytest.raises(ValueError, match=error):
        module._native_quality_decision(
            gate,
            checkpoint=checkpoint,
            protocol=protocol,
            tracking=tracking,
            broad_reward=broad_reward,
        )


def test_native_quality_decision_rejects_protocol_drift() -> None:
    """A metric match cannot rescue an evaluation run under a different protocol."""
    module = _module()
    gate, checkpoint, protocol, tracking, broad_reward = _native_gate_inputs()
    protocol["evaluation_seed"] += 1

    with pytest.raises(ValueError, match="protocol"):
        module._native_quality_decision(
            gate,
            checkpoint=checkpoint,
            protocol=protocol,
            tracking=tracking,
            broad_reward=broad_reward,
        )


def test_cross_source_quality_decision_is_measurement_only() -> None:
    """Zero-shot G1-CMU must close coverage without borrowing native thresholds."""
    module = _module()
    gate = json.loads(QUALITY_GATE.read_text())
    expected = gate["cross_source_measurement"]
    protocol = {
        "preset": expected["preset"],
        "motion_split": expected["motion_split"],
        "clip_count": expected["clip_count"],
        "frame_count": expected["frame_count"],
        "tracking_coverage_fraction": expected["required_tracking_coverage_fraction"],
    }
    for name in (
        "evaluation_seed",
        "domain_randomization",
        "observation_noise",
        "reward_task_count",
        "episodes_per_task",
        "reward_horizon",
    ):
        protocol[name] = gate["protocol"][name]

    decision = module._cross_source_quality_decision(gate, checkpoint=dict(gate["checkpoint"]), protocol=protocol)

    assert decision["threshold_applied"] is False
    assert decision["passed"] is None
    assert decision["measurement_complete"] is True
    assert decision["status"] == "measured"


def test_evaluator_uses_one_shared_tracker_reward_operator_and_strict_load() -> None:
    """Native and cross presets must differ only at the environment/data boundary."""
    source = EVALUATOR.read_text()
    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    run_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run")
    run_source = ast.get_source_segment(source, run_node)
    assert run_source is not None
    assert run_source.index("companion_evidence = _g1_cmu_companion_evidence") < run_source.index(
        "tracking = g1_motion_tracking_evaluator"
    )
    environment_index = run_source.index("env = MotionImitationEnv")
    for immutable_input in (
        "operators = _load_bfm_reward_operators",
        "checkpoint_sha256 = _sha256(checkpoint)",
        "dataset_sha256 = _sha256(dataset_path)",
        "reward_model_sha256 = _sha256(reward_model_path)",
        "reward_model_source_identity = _mujoco_model_source_identity(reward_model_path)",
        "output = args.output_dir.expanduser().resolve()",
    ):
        assert run_source.index(immutable_input) < environment_index
    assert "_load_bfm_reward_operators(args.bfm_reward_source_root)" in run_source
    assert "args.bfm_repo" not in run_source
    assert "g1_motion_tracking_evaluator" in calls
    assert "_broad_reward_rollout" in calls
    assert "motion_environment_dependency_identity" in calls
    assert "_native_quality_decision" in calls
    assert "_cross_source_quality_decision" in calls
    assert "exclusive_physical_gpu_snapshot" in calls
    assert "validate_same_exclusive_gpu" in calls
    assert "load_state_dict" in calls
    assert "strict=True" in source
    assert "assign=True" in source
    assert '"g1_lafan"' in source and '"g1_cmu"' in source
    assert "retarget_and_simulator_errors_are_not_policy_metrics" in source
    assert '"expert_provider_sha256": _source_sha256(motion_expert_buffer_g1)' in source
    assert '"expert_buffer_sha256": _source_sha256(ForwardBackwardExpertBuffer)' in source
    assert '"learner_code_bundle_sha256": _python_package_bundle_sha256(package_root)' in source
    assert '"reward_kinematics_sha256": _source_sha256(NewtonKinematics)' in source
    assert "from isaaclab_tasks.core.multi_task.motion.impl import uniform_emd_warp" in source
    assert '"emd_transport_kernel_sha256": _source_sha256(uniform_emd_warp)' in source
    assert "map_location=args.device" in source
    assert '"stage_durations_seconds": broad_reward_timing' in source
    broad_call = next(
        node
        for node in ast.walk(run_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_broad_reward_rollout"
    )
    broad_seed = next(keyword.value for keyword in broad_call.keywords if keyword.arg == "seed")
    assert ast.unparse(broad_seed) == "args.evaluation_seed"


def test_policy_identity_closes_entire_learner_package(tmp_path: Path) -> None:
    """Inference evidence must drift with any learner source byte."""
    module = _module()

    def package(root: Path) -> Path:
        package_root = root / "rsl_rl"
        (package_root / "models").mkdir(parents=True)
        (package_root / "modules").mkdir()
        (package_root / "__init__.py").write_text("VERSION = 1\n")
        (package_root / "models" / "forward_backward_model.py").write_text("def model(): return 1\n")
        (package_root / "modules" / "normalization.py").write_text("def normalize(x): return x\n")
        return package_root

    first = package(tmp_path / "checkout-a")
    second = package(tmp_path / "checkout-b")
    baseline = module._python_package_bundle_sha256(first)
    assert module._python_package_bundle_sha256(second) == baseline

    (first / "modules" / "normalization.py").write_text("def normalize(x): return 2 * x\n")
    changed = module._python_package_bundle_sha256(first)
    assert changed != baseline
    (first / "storage.py").write_text("class Replay: pass\n")
    assert module._python_package_bundle_sha256(first) != changed


def test_reward_context_policy_is_local_and_evaluator_import_is_dependency_free() -> None:
    """The model protocol must not import BFM's HumanoidVerse/rich policy graph."""
    source = EVALUATOR.read_text()
    assert "phase2_adapter.policy" not in source
    assert "from humanoidverse" not in source
    assert "import rich" not in source
    code = f"""
import importlib.util
import json
import sys
spec = importlib.util.spec_from_file_location("policy_quality_import_probe", {str(EVALUATOR)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name in sys.modules:
    assert name != "rich" and not name.startswith("rich.")
    assert name != "humanoidverse" and not name.startswith("humanoidverse.")
assert module._BFMRewardContextPolicy.__module__ == "policy_quality_import_probe"
"""
    completed = subprocess.run(
        (sys.executable, "-I", "-c", code),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_reward_context_policy_delegates_only_backward_map_and_projection() -> None:
    """The local protocol must preserve the two model operations used by reward inference."""
    module = _module()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def backward_map(self, observations):
            assert set(observations.keys()) == {"state", "privileged_state"}
            return observations["state"] + observations["privileged_state"]

        def context_project(self, context):
            return 2.0 * context

    policy = module._BFMRewardContextPolicy(Model())
    observations = {
        "state": torch.ones(3, 2),
        "privileged_state": 2.0 * torch.ones(3, 2),
        "unused": torch.full((3, 2), 99.0),
    }
    torch.testing.assert_close(policy.backward_map(observations), 3.0 * torch.ones(3, 2))
    torch.testing.assert_close(policy.project_z(torch.ones(3, 2)), 2.0 * torch.ones(3, 2))
    assert policy.device == torch.device("cpu")


def test_policy_evaluation_uses_each_frozen_behavior_corpus() -> None:
    """Native BFM uses all 862 training motions; zero-shot CMU uses its 182 test clips."""
    module = _module()
    assert module._policy_motion_split("g1_lafan") == "train"
    assert module._policy_motion_split("g1_cmu") == "evaluation"
    with pytest.raises(ValueError, match="preset"):
        module._policy_motion_split("unknown")


def test_g1_qpos_qvel_uses_semantic_action_axis_and_preserves_released_root_frames() -> None:
    """Reward FK must consume semantic joints and released [world-v,body-w] root fields."""
    module = _module()
    semantic_names = tuple(f"joint_{index}" for index in range(29))
    physical_names = tuple(reversed(semantic_names))
    data = SimpleNamespace(
        root_pos_w=SimpleNamespace(torch=torch.tensor(((11.0, 22.0, 33.0),))),
        root_quat_w=SimpleNamespace(torch=torch.tensor(((0.5, 0.1, 0.2, 0.3),))),
        joint_pos=SimpleNamespace(torch=torch.tensor(((20.0, 10.0),))),
        root_lin_vel_w=SimpleNamespace(torch=torch.tensor(((1.0, 2.0, 3.0),))),
        root_ang_vel_b=SimpleNamespace(torch=torch.tensor(((4.0, 5.0, 6.0),))),
        joint_vel=SimpleNamespace(torch=torch.tensor(((200.0, 100.0),))),
    )
    robot = SimpleNamespace(data=data, joint_names=physical_names)
    action = SimpleNamespace(
        joint_names=semantic_names,
        joint_ids=torch.tensor([physical_names.index(name) for name in semantic_names]),
        joint_position=torch.arange(29, dtype=torch.float32).unsqueeze(0),
        joint_velocity=(100.0 + torch.arange(29, dtype=torch.float32)).unsqueeze(0),
    )
    payload = SimpleNamespace(
        robot=robot,
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(env_origins=torch.tensor(((1.0, 2.0, 3.0),))),
        device="cpu",
        num_envs=1,
    )
    table = SimpleNamespace(joint_names=physical_names)
    env.command_manager = SimpleNamespace(
        get_term=lambda name: SimpleNamespace(payload=payload, table=table) if name == "motion" else None
    )
    env.action_manager = SimpleNamespace(get_term=lambda name: action if name == "joint_position" else None)

    resolved_robot, resolved_action = module._g1_reward_state_sources(env)
    qpos, qvel = module._g1_qpos_qvel(env, resolved_robot, resolved_action)

    torch.testing.assert_close(qpos[:, :7], torch.tensor(((10.0, 20.0, 30.0, 0.5, 0.1, 0.2, 0.3),)))
    torch.testing.assert_close(qpos[:, 7:], action.joint_position)
    torch.testing.assert_close(qvel[:, :6], torch.tensor(((1.0, 2.0, 3.0, 4.0, 5.0, 6.0),)))
    torch.testing.assert_close(qvel[:, 6:], action.joint_velocity)


def test_broad_reward_rollout_streams_gpu_reductions_without_autograd() -> None:
    """Deterministic evidence collection must not send grad-tracked actions into the environment."""
    module = _module()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action = torch.nn.Parameter(torch.ones(29))

        def action_sample(self, observations, contexts, *, deterministic):
            assert deterministic
            assert observations.batch_size == torch.Size((1,))
            assert contexts.shape == (1, 1)
            return self.action.unsqueeze(0)

    root = SimpleNamespace(torch=torch.zeros(1, 3))
    rotation = SimpleNamespace(torch=torch.tensor(((1.0, 0.0, 0.0, 0.0),)))
    joints = SimpleNamespace(torch=torch.zeros(1, 29))
    joint_names = tuple(f"joint_{index}" for index in range(29))
    robot = SimpleNamespace(
        joint_names=joint_names,
        data=SimpleNamespace(
            root_pos_w=root,
            root_quat_w=rotation,
            root_lin_vel_w=root,
            root_ang_vel_b=root,
        ),
    )
    action_term = SimpleNamespace(
        joint_names=joint_names,
        joint_ids=torch.arange(29),
        joint_position=joints.torch,
        joint_velocity=joints.torch,
    )
    payload = SimpleNamespace(
        robot=robot,
    )
    table = SimpleNamespace(joint_names=joint_names)

    class Env:
        num_envs = 1
        device = "cpu"
        scene = SimpleNamespace(env_origins=torch.zeros(1, 3))
        cfg = SimpleNamespace(
            commands=SimpleNamespace(motion=SimpleNamespace(payload=SimpleNamespace(auxiliary_evidence=())))
        )
        command_manager = SimpleNamespace(
            get_term=lambda _name: SimpleNamespace(payload=payload, table=table),
        )
        action_manager = SimpleNamespace(get_term=lambda _name: action_term)

        def __init__(self):
            self.transaction_seeds = []

        def evaluation_transaction(self, seed):
            self.transaction_seeds.append(seed)
            return nullcontext()

        def reset(self):
            return {"state": torch.zeros(1, 1)}

        def step(self, action):
            assert not action.requires_grad
            observations = {"state": torch.zeros(1, 1)}
            mask = torch.zeros(1, dtype=torch.bool)
            extras = {"auxiliary_reward_evidence": torch.empty(1, 0)}
            return observations, torch.zeros(1), mask, mask, extras

    operators = {
        "reward_context_policy": lambda model: model,
        "infer_contexts": lambda *_args, **_kwargs: torch.zeros(1, 1),
        "tasks": ("task",),
        "auxiliary_names": (),
        "auxiliary_coefficients": torch.empty(0),
        "hard_safety_names": (),
        "runtime_type": lambda *_args: SimpleNamespace(evaluate=lambda *_state: torch.zeros(1, 1)),
        "metric_rows": lambda *_args, **_kwargs: [{"metric_name": "return", "metric_value": 0.0}],
    }
    dataset = {
        "reference_config_sha256": "reference",
        "data_sha256": "data",
        "reward_model_sha256": "reward",
    }

    env = Env()
    rows, timing = module._broad_reward_rollout(
        model=Model(),
        env=env,
        dataset=dataset,
        reward_runtime=SimpleNamespace(evaluate=lambda *_state: torch.zeros(1, 1)),
        runtime_setup_seconds=0.25,
        operators=operators,
        episodes_per_task=1,
        horizon=1,
        batch_size=1,
        seed=4728,
    )

    assert rows == [{"metric_name": "return", "metric_value": 0.0}]
    assert set(timing) == {
        "runtime_setup",
        "context_inference",
        "simulation_and_reward",
        "scalar_serialization",
        "total",
    }
    assert env.transaction_seeds == [4728]


def test_broad_reward_hot_loop_has_no_host_transfer_or_synchronizing_branch() -> None:
    """Trajectory-scale reward work must remain device-resident until serialization."""
    source = EVALUATOR.read_text()
    tree = ast.parse(source)
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_broad_reward_rollout"
    )
    loop = next(node for node in ast.walk(function) if isinstance(node, ast.For))
    loop_source = ast.get_source_segment(source, loop)
    assert loop_source is not None
    assert ".cpu(" not in loop_source
    assert ".item(" not in loop_source
    assert "torch.any(" not in loop_source
    assert "ThreadPoolExecutor" not in source


def _test_composition_identity(builder_identity: str) -> dict[str, object]:
    """Build one internally closed composition identity for isolated companion tests."""
    identity_module = _identity_module()
    composition = {
        "preset": "g1_cmu",
        "source": {"evaluation_content_sha256": "a" * 64},
        "target": {"joint_count": 29},
        "construction": {"frame_builder_identity_sha256": builder_identity},
    }
    runtime_dependencies = {"numpy": {"module_version": "test"}}
    identity = {
        "schema": "forward_backward_phase3_motion_composition_dependency_identity_v1",
        "preset": "g1_cmu",
        "composition": composition,
        "composition_sha256": identity_module._json_hash(composition),
        "runtime_dependencies": runtime_dependencies,
        "runtime_dependencies_sha256": identity_module._json_hash(runtime_dependencies),
        "python_sources": {"motion_environment_identity": _module()._sha256(IDENTITY)},
        "reference_assets": {"reference/g1_29dof.xml": "b" * 64},
    }
    return {**identity, "bundle_sha256": identity_module._json_hash(identity)}


def _test_environment_identity() -> dict[str, object]:
    """Build one internally closed current environment identity for isolated tests."""
    identity_module = _identity_module()
    resolved_axes = {"preset": "g1_cmu", "joint_count": 29}
    resolved_configuration = {"decimation": 4}
    runtime_dependencies = {"torch": {"module_version": "test"}}
    identity = {
        "schema": identity_module._SCHEMA,
        "preset": "g1_cmu",
        "resolved_axes": resolved_axes,
        "resolved_axes_sha256": identity_module._json_hash(resolved_axes),
        "resolved_configuration": resolved_configuration,
        "resolved_configuration_sha256": identity_module._json_hash(resolved_configuration),
        "runtime_dependencies": runtime_dependencies,
        "runtime_dependencies_sha256": identity_module._json_hash(runtime_dependencies),
        "python_sources": {"motion_environment_identity": _module()._sha256(IDENTITY)},
        "robot_assets": {"simulation/g1.usd": "c" * 64},
    }
    return {**identity, "bundle_sha256": identity_module._json_hash(identity)}


def _current_companion_paths(tmp_path: Path):
    """Upgrade frozen numerical fixture payloads in memory without regenerating artifacts."""
    module = _module()
    runtime = ROOT / "fixtures" / "runtime"
    retarget = json.loads((runtime / "g1_cmu_retarget_evaluation_v3.json").read_text())
    simulator = json.loads((runtime / "g1_cmu_reference_tracking_evaluation_v3.json").read_text())
    builder_identity = retarget["composition"]["frame_builder_identity_sha256"]
    composition_identity = _test_composition_identity(builder_identity)
    composition_semantics = _identity_module().motion_composition_semantic_sha256(composition_identity)

    retarget["schema"] = "forward_backward_phase3g_g1_cmu_composition_evidence_v3"
    retarget["code_identity"] = {
        "probe_sha256": module._sha256(ROOT / "g1_cmu_composition_evidence.py"),
        "composition_dependency_identity": composition_identity,
    }
    retarget_path = tmp_path / "retarget.json"
    retarget_path.write_text(json.dumps(retarget, sort_keys=True))

    simulator["schema"] = "forward_backward_phase3g_g1_cmu_reference_tracking_evidence_v3"
    simulator["code_identity"]["probe_sha256"] = module._sha256(ROOT / "g1_cmu_reference_tracking_evidence.py")
    simulator["code_identity"]["dependency_identity"] = _test_environment_identity()
    simulator["code_identity"]["composition_dependency_identity"] = composition_identity
    simulator_retarget = simulator["error_layers"]["retarget_fit"]
    simulator_retarget["evidence_path"] = str(retarget_path)
    simulator_retarget["evidence_sha256"] = module._sha256(retarget_path)
    simulator_retarget["composition_semantic_sha256"] = composition_semantics
    simulator_path = tmp_path / "simulator.json"
    simulator_path.write_text(json.dumps(simulator, sort_keys=True))
    return retarget_path, simulator_path, simulator, composition_identity


def test_g1_cmu_companion_evidence_validates_complete_linked_error_layers(tmp_path: Path) -> None:
    """Policy evidence must bind the complete ordered retarget and simulator records."""
    module = _module()
    retarget_path, simulator_path, simulator, composition_identity = _current_companion_paths(tmp_path)
    identity_module = _identity_module()
    evidence = module._g1_cmu_companion_evidence(
        retarget_path,
        simulator_path,
        simulator["code_identity"]["dependency_identity"],
        composition_identity,
        environment_semantic_sha256=identity_module.motion_environment_semantic_sha256,
        composition_semantic_sha256=identity_module.motion_composition_semantic_sha256,
    )

    assert evidence["retarget_fit"]["schema"] == "forward_backward_phase3g_g1_cmu_composition_evidence_v3"
    assert evidence["retarget_fit"]["status"] == "measured"
    assert evidence["retarget_fit"]["clip_count"] == 182
    assert evidence["reference_controller_simulator"]["schema"] == (
        "forward_backward_phase3g_g1_cmu_reference_tracking_evidence_v3"
    )
    assert evidence["reference_controller_simulator"]["status"] == "measured"
    assert evidence["reference_controller_simulator"]["clip_count"] == 182
    composition_semantics = evidence["retarget_fit"]["composition_semantic_sha256"]
    assert len(composition_semantics) == 64
    assert evidence["reference_controller_simulator"]["composition_semantic_sha256"] == composition_semantics
    assert len(evidence["reference_controller_simulator"]["environment_semantic_sha256"]) == 64


def test_g1_cmu_companion_evidence_rejects_reordered_simulator_clips(tmp_path: Path) -> None:
    """Equal clip sets in a different order must not be accepted as the same experiment."""
    module = _module()
    retarget_path, simulator_path, simulator, composition_identity = _current_companion_paths(tmp_path)
    clip_ids = simulator["selection"]["clip_ids"]
    clip_ids[0], clip_ids[1] = clip_ids[1], clip_ids[0]
    simulator_path.write_text(json.dumps(simulator))
    identity_module = _identity_module()

    with pytest.raises(ValueError, match="ordered clip"):
        module._g1_cmu_companion_evidence(
            retarget_path,
            simulator_path,
            simulator["code_identity"]["dependency_identity"],
            composition_identity,
            environment_semantic_sha256=identity_module.motion_environment_semantic_sha256,
            composition_semantic_sha256=identity_module.motion_composition_semantic_sha256,
        )


def test_g1_cmu_companion_evidence_rejects_corrupt_environment_identity(tmp_path: Path) -> None:
    """Policy evidence must reject a dependency record whose full provenance is not closed."""
    module = _module()
    retarget_path, simulator_path, simulator, composition_identity = _current_companion_paths(tmp_path)
    stale = dict(simulator["code_identity"]["dependency_identity"])
    stale["bundle_sha256"] = "0" * 64
    identity_module = _identity_module()

    with pytest.raises(ValueError, match="not internally closed"):
        module._g1_cmu_companion_evidence(
            retarget_path,
            simulator_path,
            stale,
            composition_identity,
            environment_semantic_sha256=identity_module.motion_environment_semantic_sha256,
            composition_semantic_sha256=identity_module.motion_composition_semantic_sha256,
        )


def test_g1_cmu_companion_evidence_rejects_different_semantics(tmp_path: Path) -> None:
    """Runtime placement may differ, but resolved environment semantics may not."""
    module = _module()
    retarget_path, simulator_path, simulator, composition_identity = _current_companion_paths(tmp_path)
    expected = simulator["code_identity"]["dependency_identity"]
    changed = json.loads(json.dumps(expected))
    changed["resolved_configuration"]["decimation"] += 1
    identity_module = _identity_module()
    changed["resolved_configuration_sha256"] = identity_module._json_hash(changed["resolved_configuration"])
    payload = dict(changed)
    payload.pop("bundle_sha256")
    changed["bundle_sha256"] = identity_module._json_hash(payload)
    simulator["code_identity"]["dependency_identity"] = changed
    simulator_path.write_text(json.dumps(simulator))

    with pytest.raises(ValueError, match="stale environment semantic identity"):
        module._g1_cmu_companion_evidence(
            retarget_path,
            simulator_path,
            expected,
            composition_identity,
            environment_semantic_sha256=identity_module.motion_environment_semantic_sha256,
            composition_semantic_sha256=identity_module.motion_composition_semantic_sha256,
        )


def test_statistics_and_csv_are_finite_and_identity_closed(tmp_path: Path) -> None:
    module = _module()
    statistics = module._statistics(torch.tensor((1.0, 2.0, 3.0)))
    assert statistics["count"] == 3
    assert statistics["mean"] == 2.0
    with pytest.raises(ValueError, match="finite"):
        module._statistics(torch.tensor((float("nan"),)))

    destination = tmp_path / "broad.csv"
    module._write_csv(
        destination,
        [{"task": "move", "episode": 0, "metric_name": "return", "metric_value": 1.5}],
        {"preset": "g1_cmu", "checkpoint_transition": 9_600_000},
    )
    with destination.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {
            "preset": "g1_cmu",
            "checkpoint_transition": "9600000",
            "task": "move",
            "episode": "0",
            "metric_name": "return",
            "metric_value": "1.5",
        }
    ]


def test_mujoco_model_identity_closes_xml_includes_and_file_assets(tmp_path: Path) -> None:
    """A stable scene byte cannot hide drift in included MJCF or file-backed assets."""
    module = _module()
    root = tmp_path / "reward_model"
    (root / "meshes").mkdir(parents=True)
    (root / "textures").mkdir()
    (root / "assets").mkdir()
    scene = root / "scene.xml"
    robot = root / "robot.xml"
    scene.write_text('<mujoco><include file="robot.xml"/></mujoco>')
    robot.write_text(
        """<mujoco>
  <compiler assetdir="assets" meshdir="meshes" texturedir="textures"/>
  <asset>
    <mesh file="body.stl"/>
    <texture type="2d" file="skin.png"/>
    <hfield file="terrain.bin"/>
    <skin file="body.skn"/>
  </asset>
</mujoco>"""
    )
    assets = {
        "meshes/body.stl": b"mesh-v1",
        "textures/skin.png": b"texture-v1",
        "assets/terrain.bin": b"height-v1",
        "assets/body.skn": b"skin-v1",
    }
    for name, content in assets.items():
        (root / name).write_bytes(content)

    identity = module._mujoco_model_source_identity(scene)

    assert identity["schema"] == "forward_backward_phase3_mujoco_model_source_identity_v1"
    assert identity["entrypoint"] == "scene.xml"
    assert set(identity["files"]) == {"scene.xml", "robot.xml", *assets}
    assert identity["file_count"] == 6
    assert identity["bundle_sha256"] == module._canonical_sha256(identity["files"])
    scene_sha256 = module._sha256(scene)

    (root / "meshes/body.stl").write_bytes(b"mesh-v2")
    changed = module._mujoco_model_source_identity(scene)
    assert module._sha256(scene) == scene_sha256
    assert changed["files"]["scene.xml"] == identity["files"]["scene.xml"]
    assert changed["bundle_sha256"] != identity["bundle_sha256"]

    (root / "textures/skin.png").unlink()
    with pytest.raises(ValueError, match="regular non-symbolic file"):
        module._mujoco_model_source_identity(scene)
