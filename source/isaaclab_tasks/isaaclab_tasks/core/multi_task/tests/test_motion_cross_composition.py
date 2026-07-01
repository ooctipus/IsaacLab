# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Structural tests for direct SMPL-source to G1 cross composition."""

from __future__ import annotations

from isaaclab_tasks.core.multi_task.motion.config.robots import G1_BEHAVIOR_JOINT_NAMES
from isaaclab_tasks.core.multi_task.motion.config.source_skeletons import (
    g1_lafan_source_skeleton,
    smpl_humenv_source_skeleton,
)
from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets


def _resolved(name: str) -> MotionImitationEnvCfg:
    cfg = resolve_presets(MotionImitationEnvCfg(), selected={name})
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
    assert native.commands.motion.task_table.source.identifier == "g1_lafan"
    assert cross.commands.motion.task_table.source.identifier == "smpl_cmu"
    assert native.commands.motion.task_table.frame_builder_factory.__name__ == "g1_lafan_frame_builder"
    assert cross.commands.motion.task_table.frame_builder_factory.__name__ == "g1_smpl_humenv_frame_builder"


def test_cross_composition_uses_source_provenance_without_a_second_robot_model() -> None:
    """Source skeletons describe decoded coordinates and do not own simulation assets."""
    g1_source = g1_lafan_source_skeleton()
    smpl_source = smpl_humenv_source_skeleton()
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
    assert native.episode_length_s == cross.episode_length_s == 10.0
    assert (
        native.commands.motion.payload.episode_length_steps == cross.commands.motion.payload.episode_length_steps == 501
    )
    assert (
        native.terminations.time_out.params
        == cross.terminations.time_out.params
        == {"applied_actions_before_timeout": 501}
    )
    assert native.commands.motion.payload.root_velocity_frame == "center_of_mass"
    assert cross.commands.motion.payload.root_velocity_frame == "center_of_mass"


def test_cross_composition_keeps_task_rows_and_expert_time_grid_explicit() -> None:
    """G1 resets span each clip while expert sampling stays at 50 Hz."""
    native = _resolved("g1_lafan")
    cross = _resolved("g1_cmu")

    assert native.commands.motion.task_table.task_row_mode == "clip_time_ranges"
    assert cross.commands.motion.task_table.task_row_mode == "clip_time_ranges"
    assert native.commands.motion.task_table.reset_sources == (("reference", 0.7), ("lie_down", 0.3))
    assert cross.commands.motion.task_table.reset_sources == (("reference", 0.7), ("lie_down", 0.3))
    assert native.commands.motion.task_table.expert_sample_grid.mode.value == "uniform_before_source_end"
    assert cross.commands.motion.task_table.expert_sample_grid.mode.value == "uniform_before_source_end"
    assert (
        native.commands.motion.task_table.expert_sample_grid.step_seconds
        == cross.commands.motion.task_table.expert_sample_grid.step_seconds
        == 0.02
    )


def test_cross_composition_has_no_post_resolution_composition_hook() -> None:
    """Every component is final after the shared preset resolver returns."""
    cross = _resolved("g1_cmu")

    assert not hasattr(cross, "compose_motion")
    assert not hasattr(cross, "motion")
    assert callable(cross.commands.motion.task_table.frame_builder_factory)
    assert callable(cross.commands.motion.task_table.reference_kinematics_factory)
    assert callable(cross.commands.motion.payload.transition_state_factory)
