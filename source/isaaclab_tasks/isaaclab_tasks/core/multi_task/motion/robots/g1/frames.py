# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared derived-frame equations for motion references and observations."""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.utils.math import combine_frame_transforms

G1_HEAD_PARENT_BODY_NAME = "torso_link"
"""Physical body that owns BFM-Zero's derived G1 head frame."""
G1_HEAD_FRAME_NAME = "head_link"
"""BFM-Zero name of the derived G1 head reference frame."""


G1_HEAD_OFFSET_M = (0.0, 0.0, 0.35)
"""Derived head translation in the torso frame [m]."""


G1_HEAD_POSE_POLICY = "torso_local_offset_pose_v1"
"""Geometry law shared by reference and live derived G1 head frames."""


@wp.kernel
def _g1_joint_velocity_canonical_warp(
    joint_q: wp.array2d(dtype=wp.float32),
    clip_offsets: wp.array(dtype=wp.int32),
    step_seconds: wp.array(dtype=wp.float32),
    segment_count: int,
    frame_count: int,
    joint_count: int,
    output: wp.array2d(dtype=wp.float32),
):
    """Write destination-indexed scalar-hinge velocity edges."""
    frame, joint = wp.tid()
    if frame >= frame_count or joint >= joint_count:
        return
    low = int(0)
    high = segment_count
    while low + 1 < high:
        middle = (low + high) // 2
        if frame < clip_offsets[middle]:
            high = middle
        else:
            low = middle
    start = clip_offsets[low]
    current = frame
    previous = frame - 1
    if frame == start:
        current = start + 1
        previous = start
    output[frame, joint + 6] = (joint_q[current, joint + 7] - joint_q[previous, joint + 7]) / step_seconds[low]


def _time_forward_difference_segmented(
    values: torch.Tensor,
    offsets: torch.Tensor,
    step_seconds: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the released G1 forward-difference law independently per clip."""
    if values.ndim < 1 or values.shape[0] < 1:
        raise ValueError("Segmented time values require a nonempty leading frame axis.")
    if not values.is_floating_point():
        raise ValueError("Segmented time values must use a floating-point dtype.")
    if offsets.ndim != 1 or offsets.dtype is not torch.int64 or not offsets.is_contiguous() or offsets.shape[0] < 2:
        raise ValueError("Segment offsets must be contiguous int64 with at least two entries.")
    if offsets.device != values.device:
        raise ValueError("Segment values and offsets must share one device.")
    if int(offsets[0]) != 0 or int(offsets[-1]) != values.shape[0] or bool(torch.any(offsets[1:] - offsets[:-1] < 3)):
        raise ValueError("Segment offsets must span the values with at least 3 frames per segment.")
    if (
        step_seconds.shape != (offsets.shape[0] - 1,)
        or step_seconds.dtype is not torch.float32
        or not step_seconds.is_contiguous()
    ):
        raise ValueError("Segment sample intervals must be contiguous float32 with one value per segment.")
    if step_seconds.device != values.device:
        raise ValueError("Segment values and sample intervals must share one device.")
    if bool(torch.any(~torch.isfinite(step_seconds) | (step_seconds <= 0.0))):
        raise ValueError("Segment sample intervals must be finite and positive [s].")
    if out is None:
        out = torch.empty_like(values)
    elif out.shape != values.shape or out.dtype != values.dtype or out.device != values.device:
        raise ValueError("Segmented forward-difference output must match the input.")

    rows = torch.arange(values.shape[0], dtype=torch.int64, device=values.device)
    segments = torch.searchsorted(offsets[1:], rows, right=True)
    stops = offsets.index_select(0, segments + 1)
    steps = step_seconds.index_select(0, segments)
    tail = rows == stops - 1
    previous = torch.where(tail, stops - 3, rows)
    following = torch.where(tail, stops - 2, rows + 1)
    while steps.ndim < values.ndim:
        steps = steps.unsqueeze(-1)
    torch.sub(values.index_select(0, following), values.index_select(0, previous), out=out)
    return out.div_(steps)


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
        Body positions [m] and rotations with one derived head row appended.
    """
    body_count = body_position_world.shape[-2]
    if (
        body_position_world.shape[-1] != 3
        or body_rotation_xyzw.shape != (*body_position_world.shape[:-1], 4)
        or parent_body_index < 0
        or parent_body_index >= body_count
    ):
        raise ValueError("G1 derived-head pose inputs must share one valid physical body layout.")

    parent_position = body_position_world[..., parent_body_index, :]
    parent_rotation = body_rotation_xyzw[..., parent_body_index, :]
    offset = body_position_world.new_tensor(G1_HEAD_OFFSET_M).expand_as(parent_position)
    head_position, head_rotation = combine_frame_transforms(parent_position, parent_rotation, offset)
    return (
        torch.cat((body_position_world, head_position.unsqueeze(-2)), dim=-2),
        torch.cat((body_rotation_xyzw, head_rotation.unsqueeze(-2)), dim=-2),
    )


def append_g1_head_runtime_frame(
    body_position_world: torch.Tensor,
    body_rotation_xyzw: torch.Tensor,
    body_linear_velocity_world: torch.Tensor,
    body_angular_velocity_world: torch.Tensor,
    *,
    parent_body_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append BFM-Zero's derived G1 head frame to physical body tensors.

    Args:
        body_position_world: Physical body positions [m], shape ``[..., B, 3]``.
        body_rotation_xyzw: Physical body rotations, shape ``[..., B, 4]``.
        body_linear_velocity_world: Physical body linear velocities [m/s], shape ``[..., B, 3]``.
        body_angular_velocity_world: Physical body angular velocities [rad/s], shape ``[..., B, 3]``.
        parent_body_index: Index of :data:`G1_HEAD_PARENT_BODY_NAME` in the shared body order.

    Returns:
        Body positions [m], rotations, linear velocities [m/s], and angular velocities [rad/s], with one derived
        head row appended to the body axis.
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
    # Preserve BFM-Zero's runtime law exactly: the cross-product uses the
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
