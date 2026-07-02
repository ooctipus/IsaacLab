# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for direct Position-style motion preset composition."""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from isaaclab_newton.sim import NewtonMjcfFileCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg

import isaaclab_tasks.core.multi_task.motion.config.environment as motion_environment
from isaaclab_tasks.core.multi_task.motion.config.agents import (
    G1CmuForwardBackwardRunnerCfg,
    G1LafanForwardBackwardRunnerCfg,
    MotionForwardBackwardRunnerPresetsCfg,
    SmplCmuForwardBackwardRunnerCfg,
)
from isaaclab_tasks.core.multi_task.motion.config.presets import (
    G1_CMU_PROFILE_CFG,
    G1_LAFAN_PROFILE_CFG,
    SMPL_CMU_PROFILE_CFG,
)
from isaaclab_tasks.core.multi_task.motion.config.robots import (
    G1_MOTION_ARTICULATION_CFG,
    SMPL_MOTION_ARTICULATION_CFG,
)
from isaaclab_tasks.core.multi_task.motion.config.robots.g1 import _SIMULATOR_JOINT_NAMES
from isaaclab_tasks.core.multi_task.motion.config.robots.g1 import (
    G1_BEHAVIOR_BODY_NAMES as _G1_BEHAVIOR_BODY_NAMES,
)
from isaaclab_tasks.core.multi_task.motion.config.robots.g1 import (
    G1_BEHAVIOR_JOINT_NAMES as _G1_BEHAVIOR_JOINT_NAMES,
)
from isaaclab_tasks.core.multi_task.motion.config.simulations import MotionSimulationPresetsCfg
from isaaclab_tasks.core.multi_task.motion.config.sources import G1_LAFAN_SOURCE_CFG, SMPL_CMU_SOURCE_CFG
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils.hydra import collect_presets, resolve_presets

from isaaclab_assets.robots.smpl.smpl_constants import (
    SMPL_HUMENV_MJCF_PATH,
    SMPL_HUMENV_MJCF_SHA256,
    SMPL_ROBOT_MJCF_PATH,
    SMPL_ROBOT_MJCF_SHA256,
)


@pytest.mark.parametrize(
    ("name", "source", "builder", "joint_width", "dt", "decimation"),
    (
        ("smpl_cmu", "smpl_cmu", "smpl_humenv_frame_builder", 69, 1.0 / 450.0, 15),
        ("g1_lafan", "g1_lafan", "g1_lafan_frame_builder", 29, 1.0 / 200.0, 4),
        ("g1_cmu", "smpl_cmu", "g1_smpl_humenv_frame_builder", 29, 1.0 / 200.0, 4),
    ),
)
def test_one_name_broadcasts_every_direct_environment_axis(
    name: str,
    source: str,
    builder: str,
    joint_width: int,
    dt: float,
    decimation: int,
) -> None:
    """A single name resolves physical scene, source table, manager groups, and clocks."""
    cfg = resolve_presets(MotionImitationEnvCfg(), selected={name})
    table_cfg = cfg.commands.motion.task_table

    assert not hasattr(cfg, "motion")
    assert isinstance(cfg.scene.robot, ArticulationCfg)
    assert isinstance(cfg.sim, SimulationCfg)
    assert table_cfg.source.identifier == source
    assert table_cfg.frame_builder_factory.__name__ == builder
    assert table_cfg.task_row_mode in ("source_frames", "clip_time_ranges")
    assert table_cfg.reset_sources
    profile = {
        "smpl_cmu": SMPL_CMU_PROFILE_CFG,
        "g1_lafan": G1_LAFAN_PROFILE_CFG,
        "g1_cmu": G1_CMU_PROFILE_CFG,
    }[name]
    if name == "smpl_cmu":
        assert table_cfg.task_row_mode == "source_frames"
        assert table_cfg.task_sampling_law == "clip_categorical_then_discrete_source_frame_v1"
        assert cfg.commands.motion.payload.reset_transform_factory.__name__ == "_smpl_mocap_and_fall_reset"
        assert table_cfg.reset_sources == (
            ("motion", profile.reset.motion_frame_probability),
            ("fall", profile.reset.fall_probability),
        )
    else:
        assert table_cfg.task_row_mode == "clip_time_ranges"
        assert table_cfg.task_sampling_law == "clip_categorical_then_continuous_time_v1"
        assert cfg.commands.motion.payload.reset_transform_factory.__name__ == "_g1_reference_and_lie_down_reset"
        assert table_cfg.reset_sources == (
            ("reference", 1.0 - profile.reset.lie_down_probability),
            ("lie_down", profile.reset.lie_down_probability),
        )
        assert table_cfg.expert_sample_grid.step_seconds == profile.timing.control_dt
    assert cfg.commands.motion.payload.step_fields == ()
    assert cfg.decimation == decimation
    assert math.isclose(cfg.sim.dt, dt)
    assert cfg.sim.render_interval == decimation
    assert cfg.commands.motion.payload.episode_length_steps == profile.timing.applied_actions_before_timeout
    assert cfg.terminations.time_out.params == {
        "applied_actions_before_timeout": profile.timing.applied_actions_before_timeout
    }
    assert callable(table_cfg.reference_kinematics_factory)
    assert callable(cfg.commands.motion.payload.transition_state_factory)
    assert not any(
        hasattr(cfg, field)
        for field in (
            "source_artifact_root",
            "reference_artifact_root",
            "motion_split",
            "expert_sample_grid",
            "reference_kinematics_factory",
            "transition_state_factory",
            "applied_actions_before_timeout",
        )
    )
    if name == "smpl_cmu":
        assert cfg.actions.joint_position.action_width == joint_width
    else:
        assert cfg.actions.joint_position.joint_names == list(_G1_BEHAVIOR_JOINT_NAMES)
        assert cfg.actions.joint_position.preserve_order
        state = cfg.observations.state
        assert state.joint_position.params == state.joint_velocity.params == {"action_name": "joint_position"}
        noise = profile.observation_noise.uniform_half_ranges
        assert (state.joint_position.noise.n_min, state.joint_position.noise.n_max) == (
            -noise["joint_position_rad"],
            noise["joint_position_rad"],
        )
        assert (state.joint_velocity.noise.n_min, state.joint_velocity.noise.n_max) == (
            -noise["joint_velocity_rad_s"],
            noise["joint_velocity_rad_s"],
        )
        assert cfg.observations.privileged_state.enable_corruption is profile.observation_noise.privileged_enabled
        body_axis = cfg.observations.privileged_state.value.params["asset_cfg"]
        assert body_axis.body_names == list(_G1_BEHAVIOR_BODY_NAMES)
        assert body_axis.preserve_order


def test_reset_factories_project_profile_values_into_runtime_owners(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset implementations must consume profile values instead of repeating literals."""
    calls: dict[str, tuple[object, dict[str, object]]] = {}

    class SmplReset:
        def __init__(self, env: object, **kwargs: object) -> None:
            calls["smpl"] = (env, kwargs)

    class G1Reset:
        def __init__(self, env: object, **kwargs: object) -> None:
            calls["g1"] = (env, kwargs)

    monkeypatch.setattr(motion_environment, "SmplMocapAndFallReset", SmplReset)
    monkeypatch.setattr(motion_environment, "G1ReferenceAndLieDownReset", G1Reset)
    env = object()

    motion_environment._smpl_mocap_and_fall_reset(env)
    motion_environment._g1_reference_and_lie_down_reset(env)

    assert calls == {
        "smpl": (
            env,
            {
                "random_actions_high_exclusive": SMPL_CMU_PROFILE_CFG.reset.fall_random_actions_high_exclusive,
                "physics_dt_seconds": SMPL_CMU_PROFILE_CFG.timing.physics_dt,
                "physics_steps_per_action": SMPL_CMU_PROFILE_CFG.timing.control_decimation,
            },
        ),
        "g1": (
            env,
            {"lie_down_root_height_m": G1_LAFAN_PROFILE_CFG.reset.lie_down_root_height_m},
        ),
    }


def test_broadcast_resolves_matching_runner_without_environment_identity_wrapper() -> None:
    """Environment and learner resolve from the same name without cross-check patches."""
    expected = {
        "smpl_cmu": SmplCmuForwardBackwardRunnerCfg,
        "g1_lafan": G1LafanForwardBackwardRunnerCfg,
        "g1_cmu": G1CmuForwardBackwardRunnerCfg,
    }
    for name, runner_type in expected.items():
        env = resolve_presets(MotionImitationEnvCfg(), selected={name})
        runner = resolve_presets(MotionForwardBackwardRunnerPresetsCfg(), selected={name})
        assert isinstance(runner, runner_type)
        resolved_num_envs = runner.resolve_num_envs(None, env.scene.num_envs)
        assert resolved_num_envs == runner.num_envs
        assert resolved_num_envs > 0


def test_direct_preset_axes_expose_the_three_shared_names() -> None:
    """Every environment-facing PresetCfg axis accepts the same broadcast names."""
    cfg = MotionImitationEnvCfg()
    collected = collect_presets(cfg)
    required = {"default", "smpl_cmu", "g1_lafan", "g1_cmu"}
    assert any(required.issubset(set(alternatives)) for alternatives in collected.values())


def test_profiles_remain_the_single_authority_for_runtime_schema_and_clocks() -> None:
    """Environment projections are derived from the profile facts used by RSL-RL."""
    assert math.isclose(SMPL_CMU_PROFILE_CFG.timing.physics_dt, 1.0 / 450.0)
    assert SMPL_CMU_PROFILE_CFG.timing.control_decimation == 15
    assert SMPL_CMU_PROFILE_CFG.timing.configured_horizon_steps == 300
    assert SMPL_CMU_PROFILE_CFG.timing.applied_actions_before_timeout == 300
    assert SMPL_CMU_PROFILE_CFG.routes.behavior_action_width == 69
    assert callable(SMPL_CMU_PROFILE_CFG.routes.transition_state_factory)

    assert math.isclose(G1_LAFAN_PROFILE_CFG.timing.physics_dt, 1.0 / 200.0)
    assert G1_LAFAN_PROFILE_CFG.timing.control_decimation == 4
    assert G1_LAFAN_PROFILE_CFG.timing.configured_horizon_steps == 500
    assert G1_LAFAN_PROFILE_CFG.timing.applied_actions_before_timeout == 501
    assert G1_LAFAN_PROFILE_CFG.routes.behavior_action_width == 29
    assert callable(G1_LAFAN_PROFILE_CFG.routes.transition_state_factory)
    assert G1_CMU_PROFILE_CFG.routes.forward_width == G1_LAFAN_PROFILE_CFG.routes.forward_width


def test_source_artifact_contracts_retain_frozen_identity_and_capacity() -> None:
    """Source configs retain exact train/evaluation bytes, clips, and frame counts."""
    assert (
        SMPL_CMU_SOURCE_CFG.train.clip_count,
        SMPL_CMU_SOURCE_CFG.train.frame_count,
        SMPL_CMU_SOURCE_CFG.evaluation.clip_count,
        SMPL_CMU_SOURCE_CFG.evaluation.frame_count,
    ) == (1_638, 730_307, 182, 88_364)
    assert (
        G1_LAFAN_SOURCE_CFG.train.clip_count,
        G1_LAFAN_SOURCE_CFG.train.frame_count,
        G1_LAFAN_SOURCE_CFG.evaluation.clip_count,
        G1_LAFAN_SOURCE_CFG.evaluation.frame_count,
    ) == (862, 258_600, 40, 264_705)
    for source in (SMPL_CMU_SOURCE_CFG, G1_LAFAN_SOURCE_CFG):
        assert len(source.train.artifact_sha256) == 64
        assert len(source.evaluation.artifact_sha256) == 64
        assert source.build_skeleton().identity_sha256 == source.build_skeleton().identity_sha256


def test_simulation_articulations_are_packaged_while_reference_artifacts_are_narrow() -> None:
    """Live simulation assets are packaged; only the G1 reference MJCF is external."""
    smpl_path = Path(SMPL_MOTION_ARTICULATION_CFG.spawn.asset_path)
    g1_path = Path(G1_MOTION_ARTICULATION_CFG.spawn.usd_path)
    assert smpl_path.is_file()
    assert g1_path.is_file()
    assert "source/isaaclab_assets" in str(smpl_path)
    assert "source/isaaclab_assets" in str(g1_path)
    assert "humanoidverse" not in str(g1_path)


def test_g1_articulation_actuator_uses_live_simulator_joint_axis() -> None:
    """The direct articulation config must define controller actions in live simulator order."""
    actuator = G1_MOTION_ARTICULATION_CFG.actuators["motion"]

    assert tuple(actuator.joint_names_expr) == _SIMULATOR_JOINT_NAMES


def test_smpl_articulation_has_one_native_actuator_authority() -> None:
    """The native MuJoCo asset must own SMPL controls and passive joint terms alone."""
    assert SMPL_MOTION_ARTICULATION_CFG.actuators == {}
    assert isinstance(SMPL_MOTION_ARTICULATION_CFG.spawn, NewtonMjcfFileCfg)
    assert SMPL_MOTION_ARTICULATION_CFG.spawn.asset_path == SMPL_ROBOT_MJCF_PATH
    assert SMPL_MOTION_ARTICULATION_CFG.spawn.self_collision is True
    assert SMPL_MOTION_ARTICULATION_CFG.articulation_root_prim_path == "/humanoid"


def test_packaged_smpl_source_hashes_are_frozen() -> None:
    """Runtime and reference factories close over exact packaged MJCF bytes."""
    with Path(SMPL_HUMENV_MJCF_PATH).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    assert actual == SMPL_HUMENV_MJCF_SHA256
    with Path(SMPL_ROBOT_MJCF_PATH).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    assert actual == SMPL_ROBOT_MJCF_SHA256


def test_smpl_robot_source_excludes_world_state_without_diverging_robot_physics() -> None:
    """The runtime source owns only the robot while preserving HumEnv robot declarations."""
    robot = ET.parse(SMPL_ROBOT_MJCF_PATH).getroot()
    humenv = ET.parse(SMPL_HUMENV_MJCF_PATH).getroot()

    robot_world = robot.find("worldbody")
    humenv_world = humenv.find("worldbody")
    assert robot_world is not None and humenv_world is not None
    assert robot_world.find("geom[@name='floor']") is None
    assert robot_world.find("light") is None
    assert robot_world.find("camera") is None
    assert humenv_world.find("geom[@name='floor']") is not None
    for path in ("option", "default", "worldbody/body", "actuator", "contact", "sensor"):
        assert ET.tostring(robot.find(path)) == ET.tostring(humenv.find(path))


def test_smpl_ground_preserves_full_mujoco_friction_tuple() -> None:
    """The environment-owned plane reproduces tangential, torsional, and rolling friction."""
    ground = resolve_presets(motion_environment.MotionGroundCfg(), selected={"smpl_cmu"})
    material = ground.spawn.physics_material
    assert material.static_friction == pytest.approx(0.7)
    assert material.dynamic_friction == pytest.approx(0.7)
    assert material.torsional_friction == pytest.approx(0.005)
    assert material.rolling_friction == pytest.approx(0.0001)


def test_simulation_axis_contains_plain_simulation_cfg_values() -> None:
    """Simulation selection carries no duplicate motion identity."""
    for name in ("smpl_cmu", "g1_lafan", "g1_cmu"):
        simulation = resolve_presets(MotionSimulationPresetsCfg(), selected={name})
        assert type(simulation) is SimulationCfg
        assert not hasattr(simulation, "identifier")
