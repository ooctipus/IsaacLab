# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared derived-frame equations for motion references and observations."""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply

G1_HEAD_PARENT_BODY_NAME = "torso_link"
"""Physical body that owns the released derived G1 head frame."""
G1_HEAD_FRAME_NAME = "head_link"
"""Released name of the derived G1 head reference frame."""


G1_HEAD_OFFSET_M = (0.0, 0.0, 0.35)
"""Derived head translation in the torso frame [m]."""

G1_HEAD_RUNTIME_VELOCITY_POLICY = "parent_linear_plus_cross_unrotated_offset_v1"
"""Released live-observation velocity policy for the derived G1 head frame."""

G1_HEAD_POSE_POLICY = "torso_local_offset_pose_v1"
"""Geometry law shared by reference and live derived G1 head frames."""


def append_g1_head_pose(
    body_position_world: torch.Tensor,
    body_rotation_xyzw: torch.Tensor,
    *,
    parent_body_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append the shared derived G1 head pose to physical body tensors.

    Args:
        body_position_world: Physical body positions [m], shape ``[..., B, 3]``.
        body_rotation_xyzw: Physical body rotations, shape ``[..., B, 4]``.
        parent_body_index: Index of :data:`G1_HEAD_PARENT_BODY_NAME` in the shared body order.

    Returns:
        Body position and rotation with one derived head row appended.
    """
    body_count = body_position_world.shape[-2]
    if (
        body_position_world.shape[-1] != 3
        or body_rotation_xyzw.shape != (*body_position_world.shape[:-1], 4)
        or parent_body_index < 0
        or parent_body_index >= body_count
    ):
        raise ValueError("G1 derived-head pose inputs must share one valid physical body layout.")

    parent_rotation = body_rotation_xyzw[..., parent_body_index, :]
    offset = body_position_world.new_tensor(G1_HEAD_OFFSET_M).expand_as(body_position_world[..., parent_body_index, :])
    head_position = body_position_world[..., parent_body_index, :] + quat_apply(parent_rotation, offset)
    return (
        torch.cat((body_position_world, head_position.unsqueeze(-2)), dim=-2),
        torch.cat((body_rotation_xyzw, parent_rotation.unsqueeze(-2)), dim=-2),
    )


def append_g1_head_runtime_frame(
    body_position_world: torch.Tensor,
    body_rotation_xyzw: torch.Tensor,
    body_linear_velocity_world: torch.Tensor,
    body_angular_velocity_world: torch.Tensor,
    *,
    parent_body_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append the released derived G1 head frame to physical body tensors.

    Args:
        body_position_world: Physical body positions [m], shape ``[..., B, 3]``.
        body_rotation_xyzw: Physical body rotations, shape ``[..., B, 4]``.
        body_linear_velocity_world: Physical body linear velocities [m/s], shape ``[..., B, 3]``.
        body_angular_velocity_world: Physical body angular velocities [rad/s], shape ``[..., B, 3]``.
        parent_body_index: Index of :data:`G1_HEAD_PARENT_BODY_NAME` in the shared body order.

    Returns:
        Body position, rotation, linear velocity, and angular velocity with one
        derived head row appended to the body axis.
    """
    body_count = body_position_world.shape[-2]
    if (
        body_position_world.shape[-1] != 3
        or body_rotation_xyzw.shape != (*body_position_world.shape[:-1], 4)
        or body_linear_velocity_world.shape != body_position_world.shape
        or body_angular_velocity_world.shape != body_position_world.shape
        or parent_body_index < 0
        or parent_body_index >= body_count
    ):
        raise ValueError("G1 derived-head inputs must share one valid physical body layout.")

    body_position_world, body_rotation_xyzw = append_g1_head_pose(
        body_position_world, body_rotation_xyzw, parent_body_index=parent_body_index
    )
    offset = body_position_world.new_tensor(G1_HEAD_OFFSET_M).expand_as(body_position_world[..., parent_body_index, :])
    head_angular_velocity = body_angular_velocity_world[..., parent_body_index, :]
    # Preserve the released runtime law exactly: the cross-product uses the
    # declared torso-frame offset rather than its rotated world vector.
    head_linear_velocity = body_linear_velocity_world[..., parent_body_index, :] + torch.cross(
        head_angular_velocity,
        offset,
        dim=-1,
    )
    return (
        body_position_world,
        body_rotation_xyzw,
        torch.cat((body_linear_velocity_world, head_linear_velocity.unsqueeze(-2)), dim=-2),
        torch.cat((body_angular_velocity_world, head_angular_velocity.unsqueeze(-2)), dim=-2),
    )


__all__ = [
    "G1_HEAD_FRAME_NAME",
    "G1_HEAD_RUNTIME_VELOCITY_POLICY",
    "G1_HEAD_OFFSET_M",
    "G1_HEAD_PARENT_BODY_NAME",
    "G1_HEAD_POSE_POLICY",
    "append_g1_head_runtime_frame",
    "append_g1_head_pose",
]
