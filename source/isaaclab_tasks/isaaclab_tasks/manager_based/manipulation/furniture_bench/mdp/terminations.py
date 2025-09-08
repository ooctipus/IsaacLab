# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Optional

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
MDP terminations.
"""


def check_pose_deviation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Check if any object's pose has deviated too much from its initial state."""
    max_pos_deviation = 0.05  # meters
    max_rot_deviation = 1.0  # radians

    pose_deviation = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    # Check rigid bodies
    for rigid_body in env.scene.rigid_objects.values():
        # Reset initial positions on first check or after env reset
        if not hasattr(rigid_body, 'initial_pos'):
            rigid_body.initial_pos = rigid_body.data.root_pos_w.clone()
            rigid_body.initial_quat = rigid_body.data.root_quat_w.clone()
        else:
            # Update only for environments that need reset
            reset_mask = env.episode_length_buf <= 1
            rigid_body.initial_pos[reset_mask] = rigid_body.data.root_pos_w[reset_mask].clone()
            rigid_body.initial_quat[reset_mask] = rigid_body.data.root_quat_w[reset_mask].clone()

        # Skip if position or quaternion is NaN
        pos_is_nan = torch.isnan(rigid_body.data.root_pos_w).any(dim=1)
        quat_is_nan = torch.isnan(rigid_body.data.root_quat_w).any(dim=1)
        skip_check = pos_is_nan | quat_is_nan

        pos_deviation = (rigid_body.data.root_pos_w - rigid_body.initial_pos).norm(dim=1)
        quat_deviation = torch.acos(torch.abs((rigid_body.data.root_quat_w * rigid_body.initial_quat).sum(dim=1))) * 2

        # Only check deviation if not NaN
        valid_pos_deviation = torch.where(~skip_check, pos_deviation, torch.zeros_like(pos_deviation))
        valid_quat_deviation = torch.where(~skip_check, quat_deviation, torch.zeros_like(quat_deviation))

        pose_deviation = pose_deviation | ((valid_pos_deviation > max_pos_deviation) | (valid_quat_deviation > max_rot_deviation))

    # Check rigid body collections
    for rigid_body_collection in env.scene.rigid_object_collections.values():
        # Reset initial positions on first check or after env reset
        if not hasattr(rigid_body_collection, 'initial_pos'):
            rigid_body_collection.initial_pos = rigid_body_collection.data.object_pos_w.clone()
            rigid_body_collection.initial_quat = rigid_body_collection.data.object_quat_w.clone()
        else:
            # Update only for environments that need reset
            reset_mask = env.episode_length_buf <= 1
            rigid_body_collection.initial_pos[reset_mask] = rigid_body_collection.data.object_pos_w[reset_mask].clone()
            rigid_body_collection.initial_quat[reset_mask] = rigid_body_collection.data.object_quat_w[reset_mask].clone()

        # Skip if position or quaternion is NaN
        pos_is_nan = torch.isnan(rigid_body_collection.data.object_pos_w).any(dim=1)
        quat_is_nan = torch.isnan(rigid_body_collection.data.object_quat_w).any(dim=1)
        skip_check = pos_is_nan | quat_is_nan

        pos_deviation = (rigid_body_collection.data.object_pos_w - rigid_body_collection.initial_pos).norm(dim=1)
        quat_deviation = torch.acos(torch.abs((rigid_body_collection.data.object_quat_w * rigid_body_collection.initial_quat).sum(dim=1))) * 2

        # Only check deviation if not NaN
        valid_pos_deviation = torch.where(~skip_check, pos_deviation, torch.zeros_like(pos_deviation))
        valid_quat_deviation = torch.where(~skip_check, quat_deviation, torch.zeros_like(quat_deviation))

        pose_deviation = pose_deviation | ((valid_pos_deviation > max_pos_deviation) | (valid_quat_deviation > max_rot_deviation))
    return pose_deviation


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return (robot.data.joint_vel.abs() > (robot.data.joint_vel_limits * 2)).any(dim=1)


def consecutive_success_state(env: ManagerBasedRLEnv, num_consecutive_successes: int = 10):
    # Get the progress context to access assets and offsets
    context_term = env.reward_manager.get_term_cfg("progress_context").func  # type: ignore
    held_asset = getattr(context_term, "held_asset")
    fixed_asset = getattr(context_term, "fixed_asset")
    held_asset_offset = getattr(context_term, "held_asset_offset")
    fixed_asset_offset = getattr(context_term, "fixed_asset_offset")

    # Get task command for success threshold
    task_command = env.command_manager.get_term("task_command")

    # Compute poses
    held_asset_alignment_pos_w, held_asset_alignment_quat_w = held_asset_offset.apply(held_asset)
    fixed_asset_alignment_pos_w, fixed_asset_alignment_quat_w = fixed_asset_offset.apply(fixed_asset)
    held_asset_in_fixed_asset_frame_pos, held_asset_in_fixed_asset_frame_quat = (
        math_utils.subtract_frame_transforms(
            fixed_asset_alignment_pos_w,
            fixed_asset_alignment_quat_w,
            held_asset_alignment_pos_w,
            held_asset_alignment_quat_w,
        )
    )

    # Compute success conditions
    e_x, e_y, _ = math_utils.euler_xyz_from_quat(held_asset_in_fixed_asset_frame_quat)
    euler_xy_diff = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
    xy_distance = torch.norm(held_asset_in_fixed_asset_frame_pos[:, 0:2], dim=1)
    z_distance = held_asset_in_fixed_asset_frame_pos[:, 2]

    orientation_aligned = euler_xy_diff < 0.025
    position_centered = xy_distance < 0.0025
    z_distance_reached = z_distance < task_command.success_threshold
    success = orientation_aligned & position_centered & z_distance_reached

    # Initialize continuous success counter if not exists
    if not hasattr(consecutive_success_state, 'continuous_success_counter'):
        consecutive_success_state.continuous_success_counter = torch.zeros(
            env.num_envs, dtype=torch.int32, device=env.device
        )

    # Reset counter for environments that just started new episodes
    reset_mask = env.episode_length_buf <= 1
    consecutive_success_state.continuous_success_counter[reset_mask] = 0

    # Update continuous success counter
    consecutive_success_state.continuous_success_counter[:] = torch.where(
        success,
        consecutive_success_state.continuous_success_counter + 1,
        torch.zeros_like(consecutive_success_state.continuous_success_counter)
    )

    return consecutive_success_state.continuous_success_counter >= num_consecutive_successes


def check_position_range(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    x_range: tuple[float, float] | None = (-0.05, 0.05),
    y_range: tuple[float, float] | None = (-0.05, 0.05),
    z_range: tuple[float, float] | None = (-0.05, 0.05),
) -> torch.Tensor:
    """Check if an object's global position is within specified XYZ ranges.

    Args:
        env: The environment instance
        asset_cfg: Configuration for the asset to check
        x_range: Tuple of (min, max) for x position, or None to skip x check
        y_range: Tuple of (min, max) for y position, or None to skip y check
        z_range: Tuple of (min, max) for z position, or None to skip z check

    Returns:
        torch.Tensor: Boolean tensor indicating if position is out of range
    """
    # Get the asset
    asset = env.scene[asset_cfg.name]

    # Get global position
    pos = asset.data.root_pos_w

    # Initialize range checks as True
    x_in_range = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    y_in_range = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    z_in_range = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    # Check ranges if they are provided
    if x_range is not None:
        x_in_range = (pos[:, 0] >= x_range[0]) & (pos[:, 0] <= x_range[1])
    if y_range is not None:
        y_in_range = (pos[:, 1] >= y_range[0]) & (pos[:, 1] <= y_range[1])
    if z_range is not None:
        z_in_range = (pos[:, 2] >= z_range[0]) & (pos[:, 2] <= z_range[1])

    # Return True if position is out of range
    return ~(x_in_range & y_in_range & z_in_range)
