# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Structural tests for independent source and live-robot cross composition."""

from __future__ import annotations

from isaaclab.envs import mdp as isaaclab_mdp

from isaaclab_tasks.core.multi_task.motion.data.sources import cmu_humenv_smpl_skeleton, lafan_g1_29dof_skeleton
from isaaclab_tasks.core.multi_task.motion.robots.g1.articulation import G1_BEHAVIOR_JOINT_NAMES
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets

_PROFILE_AXES = {
    "smpl_cmu": {"smpl", "cmu", "newton_mjwarp", "timing_sim450_control30_horizon300", "sampling_source_rows"},
    "g1_lafan": {
        "g1",
        "lafan",
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


def _resolved(profile: str) -> MotionImitationEnvCfg:
    cfg = resolve_presets(MotionImitationEnvCfg(), selected=_PROFILE_AXES[profile])
    assert isinstance(cfg, MotionImitationEnvCfg)
    return cfg


def test_g1_cmu_composes_smpl_source_with_the_same_physical_g1_preset() -> None:
    """Cross composition changes source and builder, not the live robot/control preset."""
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
    assert cross.commands.motion.task_table.source.identifier == "cmu_humenv_smpl"
    assert native.commands.motion.task_table.target_kinematics.frame_builder_factory.__name__ == "g1_frame_builder"
    assert cross.commands.motion.task_table.target_kinematics.frame_builder_factory.__name__ == "g1_frame_builder"
    assert native.commands.motion.task_table.route.exact_family == "exact_coordinates"
    assert cross.commands.motion.task_table.route.semantic_family == "semantic_sequence"


def test_smpl_lafan_composes_g1_coordinates_with_the_same_physical_smpl_preset() -> None:
    """Reverse cross composition changes only source construction, not the live SMPL robot."""
    native = _resolved("smpl_cmu")
    cross = _resolved("smpl_lafan")
    cross.validate()

    assert native.scene.robot.spawn.asset_path == cross.scene.robot.spawn.asset_path
    assert native.scene.robot.init_state == cross.scene.robot.init_state
    assert native.scene.robot.actuators.keys() == cross.scene.robot.actuators.keys()
    assert type(native.actions) is type(cross.actions)
    assert type(native.observations) is type(cross.observations)
    assert native.commands.motion.task_table.source.identifier == "cmu_humenv_smpl"
    assert cross.commands.motion.task_table.source.identifier == "lafan_g1_29dof"
    assert native.commands.motion.task_table.target_kinematics.frame_builder_factory.__name__ == "smpl_frame_builder"
    assert cross.commands.motion.task_table.target_kinematics.frame_builder_factory.__name__ == "smpl_frame_builder"


def test_g1_robot_does_not_select_timing_or_task_sampling() -> None:
    """The same G1 robot composes with either declared timing and descriptor law."""
    faithful = _resolved("g1_lafan")
    alternate = resolve_presets(
        MotionImitationEnvCfg(),
        selected={
            "g1",
            "lafan",
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
    g1_source = lafan_g1_29dof_skeleton()
    smpl_source = cmu_humenv_smpl_skeleton()
    cross = _resolved("g1_cmu")

    assert g1_source.num_bodies == 30
    assert g1_source.num_joints == 29
    assert smpl_source.num_bodies == 24
    assert smpl_source.num_joints == 69
    assert smpl_source.identity_sha256 == cross.commands.motion.task_table.source.build_skeleton().identity_sha256
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
    assert callable(cross.commands.motion.task_table.target_kinematics.frame_builder_factory)
    assert callable(cross.commands.motion.task_table.target_kinematics.reference_kinematics_factory)
    assert callable(cross.commands.motion.payload.reset_transform_factory)
    assert not hasattr(cross.commands.motion.payload, "transition_factory")
