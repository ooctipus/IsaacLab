# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared MDP terms (observations, rewards, terminations) and utilities for multi-task envs."""

__all__ = [
    "NativeMujocoControlAction",
    "NativeMujocoControlActionCfg",
    "RewardScaled",
    "EpisodeLengthScaleCurriculum",
    "RootVelocityPushDiscrete",
    "success_rate_sampler",
    "vision_obs",
    "body_heading_local_observation",
    "time_left",
    "command_progress",
    "command_reach",
    "command_track",
    "command_active",
    "target_asset_pose_in_root_asset_frame",
    "asset_link_velocity_in_root_asset_frame",
    "root_pose_in_env_frame",
    "command_task_reward",
    "action_rate_l2_clamped",
    "action_l2_clamped",
    "mechanical_power",
    "contact_penalty",
    "joint_position_target_l2",
    "joint_position_limits",
    "contact_undesired",
    "body_orientation_contact",
    "body_contact_velocity",
    "body_heading_alignment",
    "abnormal_robot_state",
    "out_of_bound",
    "illegal_contact_ratio",
    "joint_reaction_overload",
    "BaseTerminationsCfg",
]


from .curriculums import EpisodeLengthScaleCurriculum, success_rate_sampler
from .events import RootVelocityPushDiscrete
from .native_mujoco_action import NativeMujocoControlAction
from .native_mujoco_action_cfg import NativeMujocoControlActionCfg
from .observations import (
    asset_link_velocity_in_root_asset_frame,
    body_heading_local_observation,
    command_active,
    command_progress,
    command_reach,
    command_track,
    root_pose_in_env_frame,
    target_asset_pose_in_root_asset_frame,
    time_left,
    vision_obs,
)
from .rewards import (
    RewardScaled,
    action_l2_clamped,
    action_rate_l2_clamped,
    body_contact_velocity,
    body_heading_alignment,
    body_orientation_contact,
    command_task_reward,
    contact_undesired,
    contact_penalty,
    joint_position_limits,
    joint_position_target_l2,
    mechanical_power,
)
from .terminations import BaseTerminationsCfg, abnormal_robot_state, illegal_contact_ratio, joint_reaction_overload, out_of_bound
