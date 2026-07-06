# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ownership regressions for the readable motion environment root."""

import ast
from dataclasses import fields
from pathlib import Path

from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg

_MULTI_TASK_ROOT = Path(__file__).parents[1]


def test_motion_environment_root_has_exactly_framework_fields() -> None:
    motion_fields = {field.name for field in fields(MotionImitationEnvCfg)}
    framework_fields = {field.name for field in fields(ManagerBasedRLEnvCfg)}

    assert motion_fields == framework_fields


def test_motion_environment_root_owns_visible_composition() -> None:
    source = _MULTI_TASK_ROOT / "motion_env_cfg.py"
    text = source.read_text(encoding="utf-8")

    assert len(text.splitlines()) < 1_000
    for declaration in (
        "class MotionGroundCfg(PresetCfg):",
        "class MotionRobotCfg(PresetCfg):",
        "class MotionContactSensorBackendCfg(PresetCfg):",
        "class MotionContactSensorCfg(PresetCfg):",
        "class MotionActionsCfg(PresetCfg):",
        "class MotionObservationsCfg(PresetCfg):",
        "class MotionSourcesCfg(PresetCfg):",
        "class MotionTargetKinematicsCfg(PresetCfg):",
        "class MotionCommandsCfg:",
        "class MotionEventsCfg(PresetCfg):",
        "class MotionRewardsCfg:",
        "class MotionTerminationsCfg:",
        "class MotionCurriculumCfg:",
        "class MotionPhysicsCfg(PresetCfg):",
        "class MotionImitationEnvCfg(ManagerBasedRLEnvCfg):",
        "class MotionTimingCfg(PresetCfg):",
        "task_table=MotionTaskTableCfg(",
        "payload=PayloadCfg()",
        "sim: SimulationCfg = MotionTimingCfg()",
        "decimation: int = preset(default=15, timing_sim200_control50_horizon501=4)",
        "episode_length_s: float = preset(default=300 / 30.0, timing_sim200_control50_horizon501=501 / 50.0)",
    ):
        assert declaration in text
    for forbidden in (
        "resolve_presets",
        "_motion_product",
        "MotionImitationEnvPresetsCfg",
        "_actor_noise",
        "_smpl_reset_transform",
        "_g1_reset_transform",
        "def motion_ground",
        "__all__",
        "smpl_cmu =",
        "g1_lafan =",
        "g1_cmu =",
        "contact_forces: SensorBaseCfg | None = preset(",
        "episode_length_steps",
        "applied_action_time_out",
        '"physics_dt_seconds": 1.0 / 450.0',
        '"physics_steps_per_action": 15',
        "transition_factory",
        "G1TransitionFacts",
        "motion_command_state",
        "CommandStateReward",
        "history_actor",
        "dt=preset(default=1.0 / 450.0, g1=1.0 / 200.0)",
        "render_interval=preset(default=15, g1=4)",
        "decimation: int = preset(default=15, g1=4)",
        'task_row_mode="clip_time_ranges"',
        "update_period=preset(",
    ):
        assert forbidden not in text
    assert text.count('source_artifact_root=""') == 1
    assert text.count('reference_artifact_root=""') == 1
    assert text.count('motion_split="train"') == 1
    assert "control = multi_task_mdp.NativeMujocoControlActionCfg(" in text
    assert "joint_position = multi_task_mdp.NativeMujocoControlActionCfg(" not in text
    assert 'reset_sources=(("reference", 0.8), ("fall", 0.2))' in text
    assert 'reset_sources=(("motion", 0.8), ("fall", 0.2))' not in text


def test_motion_environment_validation_does_not_duplicate_robot_source_composition() -> None:
    """Frame-builder constructors own source-schema validation for every selected edge."""
    source = _MULTI_TASK_ROOT / "motion_env_cfg.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    environment = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MotionImitationEnvCfg"
    )
    validator = next(
        node for node in environment.body if isinstance(node, ast.FunctionDef) and node.name == "validate_config"
    )
    validator_source = ast.unparse(validator)

    forbidden = ("source.identifier", "frame_builder_factory", "frame_builders", "cmu_humenv_smpl", "lafan_g1_29dof")
    assert all(name not in validator_source for name in forbidden)


def test_motion_command_module_contains_only_reusable_schemas() -> None:
    source = _MULTI_TASK_ROOT / "motion" / "mdp" / "commands" / "commands_cfg.py"
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    table_cfg_source = ast.unparse(classes["MotionTaskTableCfg"])
    payload_cfg_source = ast.unparse(classes["MotionStatePayloadCfg"])

    assert "class MotionTaskTableCfg" in text
    assert "class MotionStatePayloadCfg" in text
    assert "class MotionCommandsCfg" not in text
    assert "source: MotionSourceCfg = MISSING" in text
    assert "class TargetKinematicsCfg" in text
    assert "frame_builder_factory: Callable[[MotionSkeleton, NewtonKinematics], MotionFrameBuilder] = MISSING" in text
    assert "reference_kinematics_factory: Callable[[str, str | torch.device], NewtonKinematics] = MISSING" in text
    assert "target_kinematics: TargetKinematicsCfg = MISSING" in text
    assert "reset_transform_factory: Callable[..., object] = MISSING" in text
    assert "reset_transform_binds: dict[str, str] = {}" in text
    assert "transition_factory" not in payload_cfg_source
    assert "reset_sources" not in table_cfg_source
    assert "reset_sources" in payload_cfg_source
    assert "PresetCfg" not in text
    assert 'class_type: Callable | str = "{DIR}.motion_task_table:build_motion_task_table"' in text
    assert 'class_type: type | str = "{DIR}.motion_state_payload:MotionStatePayload"' in text
    assert "from .motion_task_table import build_motion_task_table" not in text
    assert "from .motion_state_payload import MotionStatePayload" not in text
    assert "episode_length_steps" not in text
    assert "Smpl" not in text
    assert "G1" not in text


def test_motion_payload_has_no_learner_history_or_reward_transition_state() -> None:
    """The command owns reference/reset clocks, not learner history or helper rewards."""
    source = _MULTI_TASK_ROOT / "motion" / "mdp" / "commands" / "motion_state_payload.py"
    text = source.read_text(encoding="utf-8")

    for forbidden in (
        "_TransitionFacts",
        "transition_factory",
        "_get_transition",
        "self._transition",
        "env.obs_buf",
    ):
        assert forbidden not in text


def test_motion_manager_wiring_lives_only_in_root() -> None:
    source = _MULTI_TASK_ROOT / "motion_env_cfg.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    selectors = {
        "MotionPhysicsCfg",
        "MotionTimingCfg",
        "MotionGroundCfg",
        "MotionRobotCfg",
        "MotionActionsCfg",
        "MotionObservationsCfg",
        "MotionContactSensorBackendCfg",
        "MotionContactSensorCfg",
        "MotionSourcesCfg",
        "MotionTargetKinematicsCfg",
        "MotionEventsCfg",
    }

    assert (
        selectors | {"MotionSceneCfg", "MotionCommandsCfg", "MotionTerminationsCfg", "MotionImitationEnvCfg"}
        <= classes.keys()
    )
    for name in selectors:
        assert any(isinstance(base, ast.Name) and base.id == "PresetCfg" for base in classes[name].bases)
    command_classes = {node.name: node for node in classes["MotionCommandsCfg"].body if isinstance(node, ast.ClassDef)}
    assert command_classes.keys() == {"PayloadCfg"}
    for name in ("MotionRewardsCfg", "MotionTerminationsCfg", "MotionCurriculumCfg"):
        assert not classes[name].bases
    assert any(isinstance(base, ast.Name) and base.id == "PresetCfg" for base in command_classes["PayloadCfg"].bases)
    assigned_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        if isinstance(target, ast.Name)
    }
    assert {"smpl_cmu", "g1_lafan", "g1_cmu"}.isdisjoint(assigned_names)


def test_default_equivalent_preset_names_exist_only_on_axis_owners() -> None:
    """Robot and physics selectors consume their tokens without aliases on dependent nodes."""
    source = _MULTI_TASK_ROOT / "motion_env_cfg.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    classes = {node.name: ast.unparse(node) for node in tree.body if isinstance(node, ast.ClassDef)}

    assert "smpl = default" in classes["MotionRobotCfg"]
    assert "smpl = default" not in classes["MotionContactSensorCfg"]
    assert "newton_mjwarp = default" in classes["MotionPhysicsCfg"]
    assert "newton_mjwarp = default" not in classes["MotionGroundCfg"]
    assert "newton_mjwarp = default" not in classes["MotionContactSensorBackendCfg"]


def test_removed_motion_config_layers_stay_absent() -> None:
    motion_root = _MULTI_TASK_ROOT / "motion"
    for relative in (
        "config/environment.py",
        "config/physics.py",
        "config/registrations.py",
        "config/simulations.py",
        "config/source_presets.py",
        "config/ground.py",
        "config/legacy_reproductions.py",
        "config/presets.py",
        "config/profiles.py",
        "mdp/terminations.py",
        "config/robot_presets.py",
        "config/source_skeletons.py",
        "config/sources.py",
        "config/agents/expert_sampling.py",
        "config/agents/rsl_rl_expert.py",
        "config/agents/rsl_rl_tracking.py",
        "config/agents/rsl_rl_tracking_curriculum.py",
        "config/agents/tracking.py",
        "config/agents/tracking_curriculum.py",
        "data/_identity.py",
        "data/sample_grid.py",
        "data/sources/_hashing.py",
        "evaluation",
        "frames.py",
        "mdp/commands/observations.py",
        "metrics",
        "rsl_rl.py",
        "rsl_rl/g1_tracking.py",
        "tracking.py",
        "trajectory",
        "robots/g1/curriculum.py",
        "robots/g1/events.py",
        "robots/g1/rewards.py",
        "robots/g1/history.py",
        "robots/g1/transition.py",
        "robots/g1/preset.py",
        "robots/g1/tracking.py",
        "robots/smpl/frames.py",
        "robots/smpl/preset.py",
        "robots/smpl/tracking.py",
    ):
        assert not (motion_root / relative).exists()
    assert all(
        "MotionSampleGrid" not in path.read_text(encoding="utf-8")
        for path in motion_root.rglob("*")
        if path.suffix in {".py", ".pyi"}
    )


def test_generic_tracking_has_no_robot_geometry_or_unwrapped_rollout() -> None:
    """The generic evaluator consumes named projectors and only the RSL VecEnv boundary."""
    source = _MULTI_TASK_ROOT / "rl" / "rsl_rl" / "forward_backward_tracking.py"
    text = source.read_text(encoding="utf-8")

    for forbidden in (
        "feature_count",
        ".unwrapped",
        "evaluate_forward_backward_tracking",
        "g1_tracking",
        "smpl_tracking",
        'getattr(command, "table"',
        'getattr(command, "payload"',
        'getattr(payload, "sampler"',
        "class ForwardBackwardTrackingRunner",
    ):
        assert forbidden not in text
    assert "from rsl_rl.env import VecEnv" in text
    assert text.count("def forward_backward_tracking_evaluator(") == 1


def test_robot_reset_transforms_do_not_traverse_the_environment() -> None:
    """Robot reset math receives explicit values from the composition root."""
    for robot in ("g1", "smpl"):
        text = (_MULTI_TASK_ROOT / "motion" / "robots" / robot / "reset.py").read_text(encoding="utf-8")
        assert "env." not in text
        assert "ManagerBased" not in text


def test_motion_implementation_uses_declared_robot_and_action_bindings() -> None:
    """Motion internals must not hard-code product assets or crawl root config."""
    motion_root = _MULTI_TASK_ROOT / "motion"
    source = "\n".join(path.read_text(encoding="utf-8") for path in motion_root.rglob("*.py"))
    assert 'env.scene["robot"]' not in source
    assert "env.cfg.commands" not in source
    assert 'action_manager.get_term("joint_position")' not in source


def test_uniform_assignment_has_one_domain_neutral_owner() -> None:
    """Uniform assignment must not retain a live motion-owned import boundary."""
    repository_root = Path(__file__).parents[6]
    source_roots = (
        _MULTI_TASK_ROOT,
        repository_root / "scripts" / "reinforcement_learning" / "forward_backward" / "phase3",
    )
    forbidden = ".".join(("isaaclab_tasks", "core", "multi_task", "motion", "metrics"))
    forbidden_relative = "motion.metrics"
    violations: list[str] = []
    for source_root in source_roots:
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = tuple(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    modules = (node.module or "",)
                else:
                    continue
                if any(
                    module in (forbidden, forbidden_relative)
                    or module.startswith(f"{forbidden}.")
                    or module.startswith(f"{forbidden_relative}.")
                    for module in modules
                ):
                    violations.append(str(path.relative_to(repository_root)))

    metrics_root = _MULTI_TASK_ROOT / "metrics"
    assert violations == []
    assert (metrics_root / "uniform_assignment.py").is_file()
    assert (metrics_root / "impl" / "uniform_assignment_warp.py").is_file()


def test_motion_packages_declare_public_surfaces_in_stubs() -> None:
    motion_root = _MULTI_TASK_ROOT / "motion"
    packages = (
        "config/agents",
        "data",
        "data/sources",
        "mdp/commands",
        "robots/g1",
        "robots/smpl",
    )

    for relative in packages:
        package = motion_root / relative
        implementation = (package / "__init__.py").read_text(encoding="utf-8")
        stub = (package / "__init__.pyi").read_text(encoding="utf-8")
        assert "lazy_export()" in implementation
        assert "__all__" not in implementation
        assert "__all__ = [" in stub


def test_motion_config_is_registration_only_and_learner_owns_rsl_rl_runtime() -> None:
    """Environment registration and learner runtime remain in their owning domains."""
    motion_root = _MULTI_TASK_ROOT / "motion"
    learner_root = _MULTI_TASK_ROOT / "rl" / "rsl_rl"
    config = motion_root / "config"
    config_text = (config / "__init__.py").read_text(encoding="utf-8")

    assert "lazy_export()" not in config_text
    assert not (config / "__init__.pyi").exists()
    assert {path.name for path in (config / "agents").iterdir() if path.is_file()} == {
        "__init__.py",
        "__init__.pyi",
        "rsl_rl_fb_cfg.py",
    }
    assert not (motion_root / "rsl_rl").exists()
    assert not (learner_root / "motion_expert.py").exists()
    assert not (learner_root / "motion_tracking.py").exists()

    for name in ("forward_backward_expert.py", "forward_backward_tracking.py"):
        path = learner_root / name
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        imports = (
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import | ast.ImportFrom)
            for alias in (node.names if isinstance(node, ast.Import) else ())
        )
        imported_from = (node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))

        assert path.is_file()
        assert not any(token in text.lower() for token in ("g1", "smpl", "lafan", "cmu"))
        assert not any("multi_task.motion" in module for module in (*imports, *imported_from))


def test_motion_semantic_retarget_has_one_stateless_motion_owner() -> None:
    """Motion owns projection math and its task-table stage owns bounded execution."""
    kinematics = (_MULTI_TASK_ROOT / "kinematics" / "newton_kinematics.py").read_text(encoding="utf-8")
    retarget = (_MULTI_TASK_ROOT / "motion" / "retarget.py").read_text(encoding="utf-8")
    task_table = (_MULTI_TASK_ROOT / "motion" / "mdp" / "commands" / "motion_task_table.py").read_text(encoding="utf-8")

    for symbol in (
        "semantic_retarget",
        "semantic_retarget_policy",
        "MotionSemanticSolution",
        "MotionSemanticSolverPolicy",
    ):
        assert symbol not in kinematics + retarget + task_table
    assert "class MotionSemanticTargets:" in retarget
    assert "class MotionSemanticProjection:" in retarget
    assert "execute_ik_batches(" in task_table
    assert "class _MotionSemanticWorkspace:" in task_table
    assert "IKObjectivePosition" not in retarget
    assert "IKObjectiveRotation" not in retarget

    for robot in ("g1", "smpl"):
        reference = (_MULTI_TASK_ROOT / "motion" / "robots" / robot / "reference.py").read_text(encoding="utf-8")
        assert "solve_semantic_targets" not in reference
        assert "class _G1SemanticProjection" not in reference
        assert "class _SmplSemanticProjection" not in reference
        assert "kinematic_retarget_positions" not in reference


def test_motion_runtime_modules_do_not_define_manual_export_lists() -> None:
    motion_root = _MULTI_TASK_ROOT / "motion"
    actual = {
        path.relative_to(motion_root)
        for path in motion_root.rglob("*.py")
        if "\n__all__ =" in path.read_text(encoding="utf-8")
    }

    assert actual == set()
    assert "__all__" not in (_MULTI_TASK_ROOT / "motion_env_cfg.py").read_text(encoding="utf-8")


def test_motion_task_registration_does_not_import_agent_package() -> None:
    source = _MULTI_TASK_ROOT / "motion" / "config" / "__init__.py"
    text = source.read_text(encoding="utf-8")

    assert "from . import agents" not in text
    assert "isaaclab_tasks.core.multi_task.motion.config.agents.rsl_rl_fb_cfg:" in text


def test_retired_motion_scientific_names_stay_absent() -> None:
    """Robot and source conventions must remain explicit at live public boundaries."""
    retired = (
        "smpl_body_observation",
        "smpl_released_tracking_pose",
        "SmplMocapAndFallReset",
        "g1_released_observation_state_pose",
        "g1_expert_observation_fields",
        "g1_privileged_observation",
        "g1_privileged_body_observation",
        "g1_expert_target",
        "released_behavior_axes_v2",
    )
    roots = (_MULTI_TASK_ROOT / "motion_env_cfg.py", *_MULTI_TASK_ROOT.joinpath("motion").rglob("*.py"))
    live_source = "\n".join(path.read_text(encoding="utf-8") for path in roots)

    assert all(name not in live_source for name in retired)
