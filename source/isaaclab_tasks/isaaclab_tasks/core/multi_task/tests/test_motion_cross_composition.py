# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Structural tests for independent source and live-robot cross composition."""

from __future__ import annotations

import pytest

from isaaclab.envs import mdp as isaaclab_mdp

from isaaclab_tasks.core.multi_task.motion.robots.g1.articulation import G1_BEHAVIOR_JOINT_NAMES
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets

_PROFILE_AXES = {
    "smpl_cmu": {"smpl", "humenv_cmu", "newton_mjwarp", "timing_sim450_control30_horizon300", "sampling_source_rows"},
    "g1_lafan": {
        "g1",
        "bfm_lafan",
        "physx",
        "timing_sim200_control50_horizon501",
        "sampling_clip_time",
        "evidence_physical_auxiliary",
        "randomization_physics_observation_pose_push",
    },
    "g1_cmu": {
        "g1",
        "cmu",
        "physx",
        "timing_sim200_control50_horizon501",
        "sampling_clip_time",
        "evidence_physical_auxiliary",
        "randomization_physics_observation_pose_push",
    },
    "smpl_lafan": {
        "smpl",
        "lafan",
        "newton_mjwarp",
        "timing_sim450_control30_horizon300",
        "sampling_source_rows",
    },
}


@pytest.mark.parametrize(
    ("robot", "dataset", "source_identifier", "target_factory", "projection_factory"),
    (
        ("smpl", "cmu", "amass_cmu_smplh", "smpl_frame_target", "smpl_source_projection"),
        ("g1", "cmu", "amass_cmu_smplh", "g1_frame_target", "g1_source_projection"),
        ("smpl", "lafan", "lafan1_bvh_ground", "smpl_frame_target", "smpl_source_projection"),
        ("g1", "lafan", "lafan1_bvh_ground", "g1_frame_target", "g1_source_projection"),
    ),
)
def test_raw_motion_sources_compose_with_each_robot_target(
    robot: str,
    dataset: str,
    source_identifier: str,
    target_factory: str,
    projection_factory: str,
) -> None:
    """Each raw source resolves independently with either robot target and projection."""
    cfg = resolve_presets(MotionImitationEnvCfg(), selected={robot, dataset})
    table = cfg.commands.motion.task_table

    assert table.source.identifier == source_identifier
    assert table.target_kinematics.target_factory.__name__ == target_factory
    assert table.target_kinematics.source_projection_factory.__name__ == projection_factory


def _resolved(profile: str) -> MotionImitationEnvCfg:
    cfg = resolve_presets(MotionImitationEnvCfg(), selected=_PROFILE_AXES[profile])
    assert isinstance(cfg, MotionImitationEnvCfg)
    return cfg


def test_g1_cmu_composes_raw_amass_with_the_same_physical_g1_preset() -> None:
    """Raw cross composition changes source projection, not the live robot/control preset."""
    native = _resolved("g1_lafan")
    cross = _resolved("g1_cmu")

    assert native.scene.robot.spawn.usd_path == cross.scene.robot.spawn.usd_path
    assert native.scene.robot.init_state == cross.scene.robot.init_state
    assert native.scene.robot.actuators.keys() == cross.scene.robot.actuators.keys()
    assert (
        native.actions.joint_position.joint_names
        == cross.actions.joint_position.joint_names
        == list(G1_BEHAVIOR_JOINT_NAMES)
    )
    assert native.actions.joint_position.preserve_order and cross.actions.joint_position.preserve_order
    assert native.commands.motion.task_table.source.identifier == "lafan_g1_29dof"
    assert cross.commands.motion.task_table.source.identifier == "amass_cmu_smplh"
    native_target = native.commands.motion.task_table.target_kinematics
    cross_target = cross.commands.motion.task_table.target_kinematics
    assert native_target.target_factory.__name__ == cross_target.target_factory.__name__ == "g1_frame_target"
    assert (
        native_target.source_projection_factory.__name__
        == cross_target.source_projection_factory.__name__
        == "g1_source_projection"
    )
    expected_routes = ("exact", "analytic", "trajectory")
    assert not hasattr(native.commands.motion.task_table, "route")
    assert not hasattr(cross.commands.motion.task_table, "route")
    assert tuple(family.name for family in native.commands.motion.task_table.families) == expected_routes
    assert tuple(family.name for family in cross.commands.motion.task_table.families) == expected_routes


def test_smpl_lafan_composes_raw_bvh_with_the_same_physical_smpl_preset() -> None:
    """Raw reverse composition changes only source construction, not the live SMPL robot."""
    native = _resolved("smpl_cmu")
    cross = _resolved("smpl_lafan")
    cross.validate()

    assert native.scene.robot.spawn.asset_path == cross.scene.robot.spawn.asset_path
    assert native.scene.robot.init_state == cross.scene.robot.init_state
    assert native.scene.robot.actuators.keys() == cross.scene.robot.actuators.keys()
    assert type(native.actions) is type(cross.actions)
    assert type(native.observations) is type(cross.observations)
    assert native.commands.motion.task_table.source.identifier == "cmu_humenv_smpl"
    assert cross.commands.motion.task_table.source.identifier == "lafan1_bvh_ground"
    native_target = native.commands.motion.task_table.target_kinematics
    cross_target = cross.commands.motion.task_table.target_kinematics
    assert native_target.target_factory.__name__ == cross_target.target_factory.__name__ == "smpl_frame_target"
    assert (
        native_target.source_projection_factory.__name__
        == cross_target.source_projection_factory.__name__
        == "smpl_source_projection"
    )


def test_g1_robot_does_not_select_timing_or_task_sampling() -> None:
    """The same G1 robot composes with either declared timing and descriptor law."""
    faithful = _resolved("g1_lafan")
    alternate = resolve_presets(
        MotionImitationEnvCfg(),
        selected={
            "g1",
            "bfm_lafan",
            "physx",
            "timing_sim450_control30_horizon300",
            "sampling_source_rows",
            "evidence_physical_auxiliary",
            "randomization_physics_observation_pose_push",
        },
    )

    assert faithful.scene.robot.spawn.usd_path == alternate.scene.robot.spawn.usd_path
    assert type(faithful.actions) is type(alternate.actions)
    assert faithful.sim.dt == 1.0 / 200.0
    assert faithful.decimation == 4
    assert faithful.scene.contact_forces.update_period == 0.0
    assert faithful.commands.motion.task_table.task_row_mode == "clip_time_ranges"
    assert alternate.sim.dt == 1.0 / 450.0
    assert alternate.decimation == 15
    assert alternate.scene.contact_forces.update_period == 0.0
    assert alternate.commands.motion.task_table.task_row_mode == "source_frames"


def test_cross_composition_uses_source_provenance_without_a_second_robot_model() -> None:
    """Source skeletons describe decoded coordinates and do not own simulation assets."""
    cross = _resolved("g1_cmu")

    assert cross.commands.motion.task_table.source.identifier == "amass_cmu_smplh"
    assert cross.commands.motion.task_table.source.semantic_level == "smplh_pose_shape"
    assert cross.commands.motion.task_table.source.open_source.__module__.endswith(".amass_smplh")
    assert cross.commands.motion.task_table.source.decoder_version == "amass_smplh_clip_rows_v2"
    assert not hasattr(cross.commands.motion.task_table.source, "robot")
    assert not hasattr(cross.commands.motion.task_table.source, "asset")


def test_native_and_cross_g1_share_runtime_routes_and_timing() -> None:
    """Only source construction changes; policy observations and timeout semantics stay G1."""
    native = _resolved("g1_lafan")
    cross = _resolved("g1_cmu")

    assert type(native.observations) is type(cross.observations)
    assert type(native.events) is type(cross.events)
    assert type(native.curriculum) is type(cross.curriculum)
    assert native.decimation == cross.decimation == 4
    assert native.episode_length_s == cross.episode_length_s == 501 / 50.0
    assert round(native.episode_length_s / (native.sim.dt * native.decimation)) == 501
    assert round(cross.episode_length_s / (cross.sim.dt * cross.decimation)) == 501
    assert native.terminations.time_out.func is cross.terminations.time_out.func is isaaclab_mdp.time_out
    assert native.terminations.time_out.params == cross.terminations.time_out.params == {}
    assert native.commands.motion.payload.root_velocity_frame == "center_of_mass"
    assert cross.commands.motion.payload.root_velocity_frame == "center_of_mass"


def test_cross_composition_keeps_task_rows_separate_from_learner_sampling() -> None:
    """G1 reset rows stay environmental while expert clocks remain learner-owned."""
    native = _resolved("g1_lafan")
    cross = _resolved("g1_cmu")

    assert native.commands.motion.task_table.task_row_mode == "clip_time_ranges"
    assert cross.commands.motion.task_table.task_row_mode == "clip_time_ranges"
    assert native.commands.motion.payload.reset_sources == (("reference", 0.7), ("lie_down", 0.3))
    assert cross.commands.motion.payload.reset_sources == (("reference", 0.7), ("lie_down", 0.3))
    assert not hasattr(native.commands.motion.task_table, "expert_sample_grid")
    assert not hasattr(cross.commands.motion.task_table, "expert_sample_grid")


def test_cross_composition_has_no_post_resolution_composition_hook() -> None:
    """Every component is final after the shared preset resolver returns."""
    cross = _resolved("g1_cmu")

    assert not hasattr(cross, "compose_motion")
    assert not hasattr(cross, "motion")
    target = cross.commands.motion.task_table.target_kinematics
    assert callable(target.target_factory)
    assert callable(target.source_projection_factory)
    assert not hasattr(target, "reference_kinematics_factory")
    assert not hasattr(target, "frame_builder_factory")
    assert callable(cross.commands.motion.payload.reset_transform_factory)
    assert not hasattr(cross.commands.motion.payload, "transition_factory")
