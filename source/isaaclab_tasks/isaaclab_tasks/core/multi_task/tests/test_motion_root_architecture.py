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
        "MotionCoordinateRouteCfg",
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
        "reference_artifact_root",
    ):
        assert forbidden not in text
    assert text.count('source_artifact_root=""') == 1
    assert text.count('target_artifact_root=""') == 1
    assert text.count('motion_split="train"') == 1
    assert text.count("MotionContactObjectiveCfg(),") == 1
    assert "g1_reference_kinematics" not in text
    assert "smpl_reference_kinematics" not in text
    assert "control = multi_task_mdp.NativeMujocoControlActionCfg(" in text
    assert "joint_position = multi_task_mdp.NativeMujocoControlActionCfg(" not in text
    assert 'reset_sources=(("reference", 0.8), ("fall", 0.2))' in text
    assert 'reset_sources=(("motion", 0.8), ("fall", 0.2))' not in text


def test_motion_environment_validation_does_not_duplicate_robot_source_composition() -> None:
    """Target and source factories own schema validation for every selected edge."""
    source = _MULTI_TASK_ROOT / "motion_env_cfg.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    environment = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MotionImitationEnvCfg"
    )
    validator = next(
        node for node in environment.body if isinstance(node, ast.FunctionDef) and node.name == "validate_config"
    )
    validator_source = ast.unparse(validator)

    forbidden = (
        "source.identifier",
        "frame_builder_factory",
        "frame_builders",
        "cmu_humenv_smpl",
        "lafan_g1_29dof",
        "MotionActionsCfg",
        "target_factory is",
        "g1_frame_target",
        "smpl_frame_target",
    )
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
    assert "class ContactChannelCfg:" in text
    assert "contact_channels: tuple[ContactChannelCfg, ...] = MISSING" in text
    assert "class ContactPatchCfg:" in text
    assert "contact_patches: tuple[ContactPatchCfg, ...] = MISSING" in text
    assert "MotionClipSource" in table_cfg_source
    assert "tuple[MotionTaskTableCfg.ContactChannelCfg, ...]" in table_cfg_source
    assert 'asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")' in text
    assert "kinematics: NewtonKinematicsBuildCfg = NewtonKinematicsBuildCfg()" in text
    assert "reference_kinematics_factory" not in text
    assert "reference_artifact_root" not in text
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
    assert "frame_builder_factory" not in text
    assert "MotionFrameBuilder" not in text

    builder = (_MULTI_TASK_ROOT / "motion" / "mdp" / "commands" / "motion_task_table_builder.py").read_text(
        encoding="utf-8"
    )
    assert "del scene_cfg" not in builder
    assert "NewtonKinematics.from_articulation(" in builder
    assert "getattr(scene_cfg, target_kinematics.asset_cfg.name)" in builder


def test_motion_corpus_scope_has_no_accepted_only_filter_or_production_cap() -> None:
    """Motion publication owns one complete source manifest without thinning."""
    commands = (_MULTI_TASK_ROOT / "motion" / "mdp" / "commands" / "commands_cfg.py").read_text(encoding="utf-8")
    builder = (_MULTI_TASK_ROOT / "motion" / "mdp" / "commands" / "motion_task_table_builder.py").read_text(
        encoding="utf-8"
    )
    root = (_MULTI_TASK_ROOT / "motion_env_cfg.py").read_text(encoding="utf-8")

    for forbidden in ("MotionClipSelectionCfg", "max_clips"):
        assert forbidden not in commands
        assert forbidden not in root
    for forbidden in (
        "motion_select_source_order",
        "source_indices[accepted]",
        "route.selected_indices",
        "record.selected",
        "selected_records",
        "max_clips",
    ):
        assert forbidden not in builder


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
    for path in motion_root.rglob("*"):
        if path.suffix not in {".py", ".pyi"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert "MotionSampleGrid" not in text
        assert "MotionSemantic" not in text
        assert "semantic_correct_support" not in text


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


def test_motion_public_route_contracts_and_private_implementations_are_separated() -> None:
    """Route contracts and deprecated boundaries are public while active implementations remain private."""
    motion_root = _MULTI_TASK_ROOT / "motion"
    kinematics_root = _MULTI_TASK_ROOT / "kinematics"
    kinematics_stub = (kinematics_root / "__init__.pyi").read_text(encoding="utf-8")
    data_stub = (motion_root / "data" / "__init__.pyi").read_text(encoding="utf-8")
    source_stub = (motion_root / "data" / "sources" / "__init__.pyi").read_text(encoding="utf-8")
    g1_stub = (motion_root / "robots" / "g1" / "__init__.pyi").read_text(encoding="utf-8")
    smpl_stub = (motion_root / "robots" / "smpl" / "__init__.pyi").read_text(encoding="utf-8")

    for name in (
        "MotionGeneralizedCoordinates",
        "MotionSourceProjection",
        "MotionSourceProjectionExact",
        "MotionSourceProjectionAnalytic",
        "MotionSourceProjectionTrajectory",
    ):
        assert name in data_stub
    assert "MotionFrameTarget" not in data_stub
    for name in ("SmplLbsModel", "load_smpl_lbs_model"):
        assert name not in data_stub
    for name in ("AmassSmplhClip", "LafanBvhClip", "open_amass_smplh_source", "open_lafan_bvh_source"):
        assert name not in source_stub
    for name in ("g1_frame_target", "g1_source_projection"):
        assert name not in g1_stub
    for name in ("smpl_frame_target", "smpl_source_projection"):
        assert name not in smpl_stub
    assert "smpl_live_joint_mujoco_names" in smpl_stub
    assert "MotionFrameBuilder" not in data_stub
    for name in ("G1FrameBuilder", "g1_frame_builder", "g1_reference_kinematics"):
        assert name in g1_stub
    for name in ("SmplFrameBuilder", "smpl_frame_builder", "smpl_reference_kinematics"):
        assert name in smpl_stub
    assert "kinematic_retarget_positions" not in kinematics_stub
    assert not (kinematics_root / "deprecated_retarget.py").exists()
    assert not (motion_root / "data" / "deprecated_frame_builder.py").exists()
    assert not (motion_root / "robots" / "g1" / "deprecated_frame_builder.py").exists()
    assert not (motion_root / "robots" / "smpl" / "deprecated_frame_builder.py").exists()


def test_motion_target_contract_and_collision_certificate_have_single_owners() -> None:
    """Target contracts and certificate collision probes must each have one owner."""
    motion_root = _MULTI_TASK_ROOT / "motion"
    builder_path = motion_root / "mdp" / "commands" / "motion_task_table_builder.py"
    target_path = motion_root / "robots" / "target.py"
    frames_path = motion_root / "data" / "frames.py"
    commands_path = motion_root / "mdp" / "commands" / "commands_cfg.py"
    trajectory_path = motion_root / "mdp" / "commands" / "motion_trajectory.py"

    target_source = target_path.read_text(encoding="utf-8")
    builder_source = builder_path.read_text(encoding="utf-8")
    frames_source = frames_path.read_text(encoding="utf-8")
    commands_source = commands_path.read_text(encoding="utf-8")
    trajectory_source = trajectory_path.read_text(encoding="utf-8")
    target_tree = ast.parse(target_source)
    frames_tree = ast.parse(frames_source)
    commands_tree = ast.parse(commands_source)

    class_owners = {
        path
        for path in motion_root.rglob("*.py")
        if any(
            isinstance(node, ast.ClassDef) and node.name == "MotionFrameTarget"
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        )
    }
    assert class_owners == {target_path}

    type_checking_blocks = [
        node
        for node in frames_tree.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
    ]
    assert len(type_checking_blocks) == 1
    frame_target_imports = [
        node
        for node in ast.walk(frames_tree)
        if isinstance(node, ast.ImportFrom) and any(alias.name == "MotionFrameTarget" for alias in node.names)
    ]
    assert len(frame_target_imports) == 1
    assert frame_target_imports[0] in tuple(ast.walk(type_checking_blocks[0]))
    assert (frame_target_imports[0].level, frame_target_imports[0].module) == (2, "robots.target")

    command_target_imports = [
        node
        for node in commands_tree.body
        if isinstance(node, ast.ImportFrom) and any(alias.name == "MotionFrameTarget" for alias in node.names)
    ]
    assert len(command_target_imports) == 1
    assert (command_target_imports[0].level, command_target_imports[0].module) == (3, "robots.target")
    assert "from ...data.frames import MotionFrameTarget" not in commands_source

    data_stub = (motion_root / "data" / "__init__.pyi").read_text(encoding="utf-8")
    assert "MotionFrameTarget" not in data_stub

    collision_sampler_owners = {
        path for path in motion_root.rglob("*.py") if "collision_probes_sample" in path.read_text(encoding="utf-8")
    }
    sampler_imports = [
        node
        for node in ast.walk(target_tree)
        if isinstance(node, ast.ImportFrom) and any(alias.name == "collision_probes_sample" for alias in node.names)
    ]
    sampler_calls = [
        node
        for node in ast.walk(target_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "collision_probes_sample"
    ]
    assert collision_sampler_owners == {target_path}
    assert len(sampler_imports) == 1
    assert len(sampler_calls) == 1
    for symbol in (
        "_motion_ground_penetration_frames",
        "validate_collision_probe_geometry",
        "write_ground_penetration",
    ):
        owners = {
            path
            for path in motion_root.rglob("*.py")
            if any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            )
        }
        assert owners == {target_path}
    assert "write_ground_penetration(" in builder_source
    assert "_motion_ground_penetration_frames" not in builder_source
    assert "collision_probe_body_indices" not in builder_source
    assert "collision_probe_body_indices" not in trajectory_source
    assert "validate_collision_probe_geometry" in trajectory_source
    assert "collision_probes_sample" not in trajectory_source
    assert "ground_penetration" not in trajectory_source


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


def test_motion_trajectory_retarget_has_one_stateless_motion_owner() -> None:
    """Motion owns projection, construction, runtime querying, and trajectory solving separately."""
    kinematics = (_MULTI_TASK_ROOT / "kinematics" / "newton_kinematics.py").read_text(encoding="utf-8")
    retarget = (_MULTI_TASK_ROOT / "motion" / "retarget.py").read_text(encoding="utf-8")
    command_root = _MULTI_TASK_ROOT / "motion" / "mdp" / "commands"
    runtime = (command_root / "motion_task_table.py").read_text(encoding="utf-8")
    builder = (command_root / "motion_task_table_builder.py").read_text(encoding="utf-8")
    trajectory = (command_root / "motion_trajectory.py").read_text(encoding="utf-8")
    motion = retarget + runtime + builder + trajectory

    for symbol in (
        "semantic_retarget",
        "semantic_retarget_policy",
        "MotionTrajectorySolution",
        "MotionTrajectorySolverPolicy",
    ):
        assert symbol not in kinematics + motion
    assert "class MotionTrajectoryTargets:" in retarget
    assert "class MotionTrajectoryProjection:" in retarget
    assert "IKTrajectorySolver(" in trajectory
    assert "plan_trajectory_memory(" in trajectory
    assert "execute_ik_batches(" not in motion
    assert "class _MotionTrajectoryWorkspace:" in trajectory
    assert "_MotionSourceRelativePointObjective" not in trajectory
    assert "_source_relative_point_" not in trajectory
    assert "ik.IKObjectivePosition(" in trajectory
    assert "IKObjectivePosition" not in retarget
    assert "IKObjectiveRotation" not in retarget

    for robot in ("g1", "smpl"):
        reference = (_MULTI_TASK_ROOT / "motion" / "robots" / robot / "reference.py").read_text(encoding="utf-8")
        assert "SemanticProjection" not in reference
        assert "solve_semantic_targets" not in reference
        assert "kinematic_retarget_positions" not in reference


def test_motion_contact_and_required_direction_rows_are_declared_separately() -> None:
    """Contact state and publication evidence select distal rows independently."""
    path = _MULTI_TASK_ROOT / "motion" / "retarget.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    target = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_MotionTrajectoryTarget"
    )
    post_init = next(node for node in target.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
    assignments = {
        target.id: node.value
        for node in ast.walk(post_init)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    contact_rows = ast.unparse(assignments["contact_direction_rows"])
    required_rows = ast.unparse(assignments["required_direction_rows"])
    coupled_comparisons = [
        ast.unparse(node)
        for node in ast.walk(post_init)
        if isinstance(node, ast.Compare)
        and "contact_channel_names" in ast.unparse(node)
        and "required_direction_roles" in ast.unparse(node)
    ]

    assert "contact_channel_names" in contact_rows
    assert "required_direction_roles" not in contact_rows
    assert "self.required_direction_roles" in required_rows
    assert "contact_channel_names" not in required_rows
    assert coupled_comparisons == []
    assert "any(channel not in self.required_direction_roles for channel in contact_channel_names)" in text


def test_motion_task_table_modules_have_one_way_ownership() -> None:
    """Runtime, builder, and trajectory files reject dependency inversion and product policy."""
    command_root = _MULTI_TASK_ROOT / "motion" / "mdp" / "commands"
    runtime = (command_root / "motion_task_table.py").read_text(encoding="utf-8")
    builder = (command_root / "motion_task_table_builder.py").read_text(encoding="utf-8")
    trajectory = (command_root / "motion_trajectory.py").read_text(encoding="utf-8")

    for forbidden in ("newton", "warp", "IKTrajectory", "MotionClipSource", "execute_task_family"):
        assert forbidden not in runtime
    for forbidden in ("newton.ik", "import warp", "robots.g1", "robots.smpl", "data.sources"):
        assert forbidden not in builder
    for forbidden in (
        "robots.g1",
        "robots.smpl",
        "data.sources",
        "motion_state_payload",
        "motion_sampler",
        "ManagerBased",
    ):
        assert forbidden not in trajectory
    assert "class MotionTaskTable:" in runtime
    assert "def _build_motion_task_table(" in builder
    assert "def motion_solve_trajectory(" in trajectory
    assert "def build_motion_task_table(" in runtime
    assert "kinematics.trajectory import _trajectory_free_root_velocity_backward" not in trajectory


def test_product_coordinate_policies_live_only_in_robot_leaves() -> None:
    """Production robot code cannot import G1 or HumEnv policies from shared kinematics."""
    kinematics_root = _MULTI_TASK_ROOT / "kinematics"
    shared_source = (kinematics_root / "kinematic_tree.py").read_text(encoding="utf-8")
    public_source = (kinematics_root / "__init__.pyi").read_text(encoding="utf-8")
    g1_frames = (_MULTI_TASK_ROOT / "motion" / "robots" / "g1" / "frames.py").read_text(encoding="utf-8")
    shared_modules = {path: path.read_text(encoding="utf-8") for path in kinematics_root.rglob("*.py")}
    shared_policy_sources = {
        path.relative_to(kinematics_root)
        for path, source in shared_modules.items()
        if "_time_forward_difference_segmented" in source or "time_select_euler_xyz_branches_segmented" in source
    }
    assert not shared_policy_sources

    g1_reference = (_MULTI_TASK_ROOT / "motion" / "robots" / "g1" / "reference.py").read_text(encoding="utf-8")
    smpl_reference = (_MULTI_TASK_ROOT / "motion" / "robots" / "smpl" / "reference.py").read_text(encoding="utf-8")

    assert "time_select_euler_xyz_branches_segmented" not in shared_source
    assert "time_select_euler_xyz_branches_segmented" not in public_source
    assert "motion.robots" not in shared_source
    assert "kinematic_tree_warp" not in shared_source
    assert "import warp as wp" not in shared_source
    assert not (kinematics_root / "impl" / "kinematic_tree_warp.py").exists()
    assert (_MULTI_TASK_ROOT / "motion" / "robots" / "smpl" / "reference_warp.py").is_file()

    wrapper = next(
        node
        for node in ast.parse(shared_source).body
        if isinstance(node, ast.FunctionDef) and node.name == "time_forward_difference_segmented"
    )
    wrapper_source = ast.unparse(wrapper)
    assert "warnings.warn" in wrapper_source
    assert "_time_segment_rows" in wrapper_source
    assert not any(isinstance(node, ast.ImportFrom) for node in ast.walk(wrapper))
    assert "def _time_forward_difference_segmented(" in g1_frames
    assert "def _time_select_euler_xyz_branches_segmented(" in smpl_reference

    for source, forbidden in (
        (g1_reference, "time_forward_difference_segmented"),
        (smpl_reference, "time_select_euler_xyz_branches_segmented"),
    ):
        imports = {
            alias.name
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom) and node.module is not None and node.module.endswith("kinematics")
            for alias in node.names
        }
        assert forbidden not in imports


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


def test_motion_contact_contract_has_no_packed_public_compatibility_api() -> None:
    """Only Motion owns source contact inference and target support-patch visualization policy."""
    commands = (_MULTI_TASK_ROOT / "motion" / "mdp" / "commands" / "commands_cfg.py").read_text(encoding="utf-8")
    builder = (_MULTI_TASK_ROOT / "motion" / "mdp" / "commands" / "motion_task_table_builder.py").read_text(
        encoding="utf-8"
    )
    public_kinematics = (_MULTI_TASK_ROOT / "kinematics" / "__init__.pyi").read_text(encoding="utf-8")

    for forbidden in (
        "MotionContactEvidenceCfg",
        "MotionStanceDriftCriterionCfg",
        "ContactPointCfg",
        "contact_points:",
    ):
        assert forbidden not in commands
    for forbidden in (
        "IKTrajectoryContactEvidence",
        "infer_trajectory_contact_schedule",
        "bind_trajectory_contact_anchors",
    ):
        assert forbidden not in public_kinematics
    assert '"contact_target_offsets"' in builder
    assert '"contact_patch_anchor_offsets"' not in builder
