# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for independent motion robot, source, backend, and agent axes."""

from dataclasses import MISSING, fields
from pathlib import Path

import pytest
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_newton.sim.schemas import MujocoCollisionPropertiesCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysxContactSensorCfg

from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.managers import ActionTermCfg, ObservationTermCfg
from isaaclab.sim.schemas import CollisionBaseCfg
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab.utils.string import string_to_callable

from isaaclab_tasks.core.multi_task.motion.config.agents import MotionForwardBackwardRunnerCfg
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils import resolve_presets

_META_RUNNER_AXES = (
    "helpers_discriminator",
    "tracking_off",
    "model_plain_2x1024",
    "replay_transition_uniform_2m",
    "schedule_50x10_5m",
    "optimization_lr1e4_implied0p1_actor0p01",
    "context_online_10k",
    "exploration_std0p2_range1",
    "seed_0",
    "expert_clock_source_rows",
)
_BFM_RUNNER_AXES = (
    "helpers_discriminator_auxiliary",
    "tracking_reset_frame",
    "tracking_interval_9p6m",
    "model_residual_6x1024",
    "replay_episode_uniform_5120k",
    "schedule_1024x1_211p2m",
    "optimization_lr3e4_implied0_actor0p05",
    "context_expert_half_8192",
    "exploration_std0p05_range5",
    "seed_4728",
    "expert_clock_50hz",
)
_SMPL_ENV_AXES = ("smpl", "humenv_cmu", "newton_mjwarp", "timing_sim450_control30_horizon300", "sampling_source_rows")
_G1_CAPABILITY_AXES = ("evidence_physical_auxiliary", "randomization_physics_observation_pose_push")
_G1_LAFAN_ENV_AXES = (
    "g1",
    "bfm_lafan",
    "physx",
    "timing_sim200_control50_horizon501",
    "sampling_clip_time",
    *_G1_CAPABILITY_AXES,
)
_G1_CMU_ENV_AXES = (
    "g1",
    "cmu",
    "physx",
    "timing_sim200_control50_horizon501",
    "sampling_clip_time",
    *_G1_CAPABILITY_AXES,
)
_PROFILES = {
    "smpl_cmu": (*_SMPL_ENV_AXES, *_META_RUNNER_AXES),
    "g1_lafan": (*_G1_LAFAN_ENV_AXES, *_BFM_RUNNER_AXES),
    "g1_cmu": (*_G1_CMU_ENV_AXES, *_BFM_RUNNER_AXES),
}


@pytest.mark.parametrize(
    ("tokens", "source", "target", "projection", "row_mode", "physics", "runner_axes"),
    (
        (
            set(_SMPL_ENV_AXES),
            "cmu_humenv_smpl",
            "smpl_frame_target",
            "smpl_source_projection",
            "source_frames",
            "NewtonCfg",
            _META_RUNNER_AXES,
        ),
        (
            set(_G1_LAFAN_ENV_AXES),
            "lafan_g1_29dof",
            "g1_frame_target",
            "g1_source_projection",
            "clip_time_ranges",
            "PhysxCfg",
            _BFM_RUNNER_AXES,
        ),
        (
            set(_G1_CMU_ENV_AXES),
            "amass_cmu_smplh",
            "g1_frame_target",
            "g1_source_projection",
            "clip_time_ranges",
            "PhysxCfg",
            _BFM_RUNNER_AXES,
        ),
    ),
)
def test_independent_axes_resolve_three_motion_profiles(
    tokens: set[str],
    source: str,
    target: str,
    projection: str,
    row_mode: str,
    physics: str,
    runner_axes: tuple[str, ...],
) -> None:
    env = resolve_presets(MotionImitationEnvCfg(), selected=tokens)
    runner = resolve_presets(MotionForwardBackwardRunnerCfg(), selected={*tokens, *runner_axes})
    table = env.commands.motion.task_table

    assert table.source.identifier == source
    assert table.target_kinematics.target_factory.__name__ == target
    assert table.target_kinematics.source_projection_factory.__name__ == projection
    assert table.task_row_mode == row_mode
    assert type(env.sim.physics).__name__ == physics
    assert isinstance(runner, MotionForwardBackwardRunnerCfg)
    assert env.scene.robot is not MISSING


def test_fused_internal_config_modules_are_deleted() -> None:
    config_dir = Path(__file__).parents[1] / "motion" / "config"

    assert not (config_dir / "environment.py").exists()
    assert not (config_dir / "presets.py").exists()
    assert not (config_dir / "profiles.py").exists()
    assert not (config_dir / "simulations.py").exists()
    assert not (config_dir / "robots").exists()


_PROFILE_SEMANTICS = {
    "smpl_cmu": {
        "source": "cmu_humenv_smpl",
        "target": "smpl_frame_target",
        "projection": "smpl_source_projection",
        "row_mode": "source_frames",
        "reset_sources": (("reference", 0.8), ("fall", 0.2)),
        "action_term": ("control", "NativeMujocoControlActionCfg"),
        "root_velocity_frame": "link",
        "horizon": 300,
        "sensor": None,
        "ground_material": "NewtonMaterialPropertiesCfg",
        "managers": (
            "SmplCfg",
            "SmplCfg",
            "EmptyCfg",
            "MotionRewardsCfg",
            "MotionTerminationsCfg",
            "MotionCurriculumCfg",
        ),
        "timing": (1.0 / 450.0, 15, 10.0),
        "physics": "NewtonCfg",
    },
    "g1_lafan": {
        "source": "lafan_g1_29dof",
        "target": "g1_frame_target",
        "projection": "g1_source_projection",
        "row_mode": "clip_time_ranges",
        "reset_sources": (("reference", 0.7), ("lie_down", 0.3)),
        "action_term": ("joint_position", "G1JointPositionActionCfg"),
        "root_velocity_frame": "center_of_mass",
        "horizon": 501,
        "sensor": PhysxContactSensorCfg,
        "ground_material": "RigidBodyMaterialBaseCfg",
        "managers": (
            "G1Cfg",
            "G1Cfg",
            "RandomizationCfg",
            "MotionRewardsCfg",
            "MotionTerminationsCfg",
            "MotionCurriculumCfg",
        ),
        "timing": (1.0 / 200.0, 4, 501 / 50.0),
        "physics": "PhysxCfg",
    },
    "g1_cmu": {
        "source": "amass_cmu_smplh",
        "target": "g1_frame_target",
        "projection": "g1_source_projection",
        "row_mode": "clip_time_ranges",
        "reset_sources": (("reference", 0.7), ("lie_down", 0.3)),
        "action_term": ("joint_position", "G1JointPositionActionCfg"),
        "root_velocity_frame": "center_of_mass",
        "horizon": 501,
        "sensor": PhysxContactSensorCfg,
        "ground_material": "RigidBodyMaterialBaseCfg",
        "managers": (
            "G1Cfg",
            "G1Cfg",
            "RandomizationCfg",
            "MotionRewardsCfg",
            "MotionTerminationsCfg",
            "MotionCurriculumCfg",
        ),
        "timing": (1.0 / 200.0, 4, 501 / 50.0),
        "physics": "PhysxCfg",
    },
}


@pytest.mark.parametrize("profile", tuple(_PROFILE_SEMANTICS))
def test_profiles_preserve_the_frozen_environment_semantics(profile: str) -> None:
    env = resolve_presets(MotionImitationEnvCfg(), selected=set(_PROFILES[profile]))
    expected = _PROFILE_SEMANTICS[profile]
    table = env.commands.motion.task_table
    payload = env.commands.motion.payload
    managers = (
        type(env.actions).__name__,
        type(env.observations).__name__,
        type(env.events).__name__,
        type(env.rewards).__name__,
        type(env.terminations).__name__,
        type(env.curriculum).__name__,
    )

    assert table.source.identifier == expected["source"]
    assert table.target_kinematics.target_factory.__name__ == expected["target"]
    assert table.target_kinematics.source_projection_factory.__name__ == expected["projection"]
    assert table.task_row_mode == expected["row_mode"]
    assert payload.reset_sources == expected["reset_sources"]
    action_terms = tuple(
        (field.name, type(value).__name__)
        for field in fields(env.actions)
        if isinstance(value := getattr(env.actions, field.name), ActionTermCfg)
    )
    assert action_terms == (expected["action_term"],)
    assert payload.root_velocity_frame == expected["root_velocity_frame"]
    assert round(env.episode_length_s / (env.sim.dt * env.decimation)) == expected["horizon"]
    if expected["sensor"] is None:
        assert env.scene.contact_forces is None
    else:
        assert isinstance(env.scene.contact_forces, expected["sensor"])
        assert env.scene.contact_forces.update_period == 0.0
    assert env.terminations.time_out.func is isaaclab_mdp.time_out
    assert env.terminations.time_out.params == {}
    assert type(env.scene.ground.spawn.physics_material).__name__ == expected["ground_material"]
    assert managers == expected["managers"]
    assert (env.sim.dt, env.decimation, env.episode_length_s) == expected["timing"]
    assert env.sim.render_interval == env.decimation
    assert type(env.sim.physics).__name__ == expected["physics"]
    assert env.compute_final_obs is True


def test_g1_transition_evidence_is_named_and_not_concatenated() -> None:
    """Replay evidence remains individually addressable and independent from command state."""
    env = resolve_presets(MotionImitationEnvCfg(), selected=set(_G1_LAFAN_ENV_AXES))
    evidence = env.observations.transition
    term_names = tuple(
        field.name for field in fields(evidence) if isinstance(getattr(evidence, field.name), ObservationTermCfg)
    )

    assert evidence.concatenate_terms is False
    assert term_names == (
        "penalty_torques",
        "penalty_action_rate",
        "limits_dof_pos",
        "limits_torque",
        "penalty_undesired_contact",
        "penalty_feet_ori",
        "penalty_ankle_roll",
        "penalty_slippage",
    )
    assert not hasattr(env.observations, "history_actor")


def test_default_motion_axes_are_smpl_raw_cmu() -> None:
    env = resolve_presets(MotionImitationEnvCfg(), selected=set())

    assert env.commands.motion.task_table.source.identifier == "amass_cmu_smplh"
    target = env.commands.motion.task_table.target_kinematics
    assert target.target_factory.__name__ == "smpl_frame_target"
    assert target.source_projection_factory.__name__ == "smpl_source_projection"
    assert tuple(physics_type.__name__ for physics_type in target.physics_types) == ("NewtonCfg",)
    assert not target.supports_physical_evidence
    assert not target.supports_randomization
    assert env.commands.motion.task_table.task_row_mode == "source_frames"
    assert type(env.actions).__name__ == "SmplCfg"
    assert type(env.sim.physics).__name__ == "NewtonCfg"
    assert env.scene.robot is not MISSING


def test_g1_cmu_resolves_from_independent_axes() -> None:
    env = resolve_presets(
        MotionImitationEnvCfg(),
        selected=set(_G1_CMU_ENV_AXES),
    )

    assert env.commands.motion.task_table.source.identifier == "amass_cmu_smplh"
    target = env.commands.motion.task_table.target_kinematics
    assert target.target_factory.__name__ == "g1_frame_target"
    assert target.source_projection_factory.__name__ == "g1_source_projection"
    assert tuple(physics_type.__name__ for physics_type in target.physics_types) == ("NewtonCfg", "PhysxCfg")
    assert target.supports_physical_evidence
    assert target.supports_randomization
    assert env.commands.motion.task_table.task_row_mode == "clip_time_ranges"
    assert type(env.actions).__name__ == "G1Cfg"
    assert type(env.sim.physics).__name__ == "PhysxCfg"


@pytest.mark.parametrize(
    ("backend", "collision_type"),
    (
        ("newton_mjwarp", MujocoCollisionPropertiesCfg),
        ("physx", CollisionBaseCfg),
    ),
)
def test_bare_g1_selects_only_robot_geometry_and_control(
    backend: str,
    collision_type: type,
) -> None:
    env = resolve_presets(MotionImitationEnvCfg(), selected={"g1", "humenv_cmu", backend})

    assert isinstance(env.scene.ground.spawn.collision_props, collision_type)
    assert env.scene.ground.spawn.physics_material.static_friction == 0.7
    assert env.scene.ground.spawn.physics_material.dynamic_friction == 0.7
    assert env.scene.contact_forces is None
    assert env.observations.transition is None
    assert type(env.events).__name__ == "EmptyCfg"
    assert env.actions.joint_position.default_joint_offset_range == (0.0, 0.0)
    for group_name in ("joint_position", "joint_velocity", "projected_gravity", "base_angular_velocity"):
        assert getattr(env.observations, group_name).value.noise is None


@pytest.mark.parametrize(
    ("backend", "sensor_type"),
    (("newton_mjwarp", NewtonContactSensorCfg), ("physx", PhysxContactSensorCfg)),
)
def test_physical_auxiliary_evidence_selects_only_evidence_and_sensor(
    backend: str,
    sensor_type: type,
) -> None:
    env = resolve_presets(
        MotionImitationEnvCfg(),
        selected={"g1", "humenv_cmu", backend, "evidence_physical_auxiliary"},
    )

    assert isinstance(env.scene.contact_forces, sensor_type)
    assert env.observations.transition is not None
    assert env.scene.ground.spawn.physics_material.static_friction == 0.7
    assert type(env.events).__name__ == "EmptyCfg"
    assert env.actions.joint_position.default_joint_offset_range == (0.0, 0.0)
    assert env.observations.joint_position.value.noise is None
    for name in ("penalty_torques", "penalty_action_rate", "limits_torque"):
        term = getattr(env.observations.transition, name)
        resolved = string_to_callable(term.func) if isinstance(term.func, str) else term.func
        assert resolved.__module__ == "isaaclab_tasks.core.multi_task.motion.robots.g1.actions"


def test_physical_observation_pose_push_randomization_selects_no_evidence() -> None:
    env = resolve_presets(
        MotionImitationEnvCfg(),
        selected={"g1", "humenv_cmu", "physx", "randomization_physics_observation_pose_push"},
    )

    assert env.scene.contact_forces is None
    assert env.observations.transition is None
    assert env.scene.ground.spawn.physics_material.static_friction == 1.0
    assert env.scene.ground.spawn.physics_material.dynamic_friction == 1.0
    assert type(env.events).__name__ == "RandomizationCfg"
    assert env.actions.joint_position.default_joint_offset_range == (-0.02, 0.02)
    for group_name in ("joint_position", "joint_velocity", "projected_gravity", "base_angular_velocity"):
        assert isinstance(getattr(env.observations, group_name).value.noise, UniformNoiseCfg)


@pytest.mark.parametrize(
    ("robot", "backend", "action_type", "physics_type"),
    (
        ("smpl", "newton_mjwarp", "MotionActionsCfg.SmplCfg", "NewtonCfg"),
        ("g1", "newton_mjwarp", "MotionActionsCfg.G1Cfg", "NewtonCfg"),
        ("g1", "physx", "MotionActionsCfg.G1Cfg", "PhysxCfg"),
    ),
)
def test_motion_environment_composes_supported_robot_backend_pairs(
    robot: str,
    backend: str,
    action_type: str,
    physics_type: str,
) -> None:
    env = resolve_presets(MotionImitationEnvCfg(), selected={robot, "humenv_cmu", backend})

    env.validate()
    assert f"{type(env.actions).__module__}.{type(env.actions).__qualname__}".endswith(action_type)
    assert type(env.sim.physics).__name__ == physics_type


def test_motion_environment_robot_identity_does_not_depend_on_action_config_class() -> None:
    """Target kinematics, rather than an action wrapper type, identifies the robot."""

    class CustomActionsCfg:
        pass

    env = resolve_presets(MotionImitationEnvCfg(), selected={"g1", "humenv_cmu", "physx"})
    env.actions = CustomActionsCfg()

    env.validate_config()


@pytest.mark.parametrize(
    ("tokens", "message"),
    (
        (
            {"smpl", "humenv_cmu", "newton_mjwarp", "evidence_physical_auxiliary"},
            "does not support physical auxiliary evidence",
        ),
        (
            {"smpl", "humenv_cmu", "physx"},
            "requires one of these physics types: NewtonCfg",
        ),
        (
            {"smpl", "humenv_cmu", "newton_mjwarp", "randomization_physics_observation_pose_push"},
            "does not support physical and observation randomization",
        ),
    ),
)
def test_motion_environment_rejects_incompatible_robot_capabilities_and_backends(
    tokens: set[str],
    message: str,
) -> None:
    env = resolve_presets(MotionImitationEnvCfg(), selected=tokens)

    with pytest.raises(ValueError, match=message):
        env.validate()
