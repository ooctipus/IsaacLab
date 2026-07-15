# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinematic-tree topology, ordered-hinge projection, and trajectory operators."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from isaaclab.utils.math import (
    axis_angle_from_quat,
    euler_xyz_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_rotation_vector,
    quat_mul,
)

if TYPE_CHECKING:
    from .newton_kinematics import NewtonKinematics

ORDERED_HINGE_OPERATOR_VERSION = "ordered_independent_hinge_fit_v3"


@dataclass(frozen=True, slots=True)
class KinematicTree:
    """Body topology and exact grouped rotational coordinates of one kinematic model.

    Attributes:
        body_names: Bodies in model order.
        parent_indices: Parent body per body, with ``-1`` for the root.
        joint_names: Non-root joints in model order.
        joint_child_body_indices: Child body owned by each non-root joint.
        joint_coordinate_ranges: Half-open coordinate range owned by each joint.
        coordinate_names: Rotational coordinates in generalized-position order.
        coordinate_axes: Unit axes of the rotational coordinates.
        coordinate_q_indices: Indices into the model generalized positions.
        coordinate_qd_indices: Indices into the model generalized velocities.
        coordinate_lower_limits_rad: Hard lower coordinate limits [rad].
        coordinate_upper_limits_rad: Hard upper coordinate limits [rad].
    """

    body_names: tuple[str, ...]
    parent_indices: tuple[int, ...]
    joint_names: tuple[str, ...]
    joint_child_body_indices: tuple[int, ...]
    joint_coordinate_ranges: tuple[tuple[int, int], ...]
    coordinate_names: tuple[str, ...]
    coordinate_axes: tuple[tuple[float, float, float], ...]
    coordinate_q_indices: tuple[int, ...]
    coordinate_qd_indices: tuple[int, ...]
    coordinate_lower_limits_rad: tuple[float, ...]
    coordinate_upper_limits_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate the topology and coordinate layout."""
        coordinate_count = len(self.coordinate_names)
        coordinate_fields = (
            self.coordinate_axes,
            self.coordinate_q_indices,
            self.coordinate_qd_indices,
            self.coordinate_lower_limits_rad,
            self.coordinate_upper_limits_rad,
        )
        if not self.body_names or len(set(self.body_names)) != len(self.body_names):
            raise ValueError("Kinematic-tree body names must be nonempty and unique.")
        if len(self.parent_indices) != len(self.body_names) or self.parent_indices.count(-1) != 1:
            raise ValueError("Kinematic-tree parents must contain exactly one root.")
        if any(parent >= body or parent < -1 for body, parent in enumerate(self.parent_indices)):
            raise ValueError("Kinematic-tree parents must be topologically ordered.")
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Kinematic-tree joint names must be nonempty and unique.")
        if len(self.joint_child_body_indices) != len(self.joint_names) or len(self.joint_coordinate_ranges) != len(
            self.joint_names
        ):
            raise ValueError("Every kinematic-tree joint must own one child and coordinate range.")
        root = self.parent_indices.index(-1)
        if any(child < 0 or child >= len(self.body_names) or child == root for child in self.joint_child_body_indices):
            raise ValueError("Kinematic-tree joints must own valid non-root bodies.")
        non_root_bodies = set(range(len(self.body_names))) - {root}
        if (
            len(set(self.joint_child_body_indices)) != len(self.joint_child_body_indices)
            or set(self.joint_child_body_indices) != non_root_bodies
        ):
            raise ValueError("Kinematic-tree joints must uniquely own every non-root body.")
        if len(set(self.coordinate_names)) != coordinate_count:
            raise ValueError("Kinematic-tree coordinate names must be unique.")
        if any(len(values) != coordinate_count for values in coordinate_fields):
            raise ValueError("Kinematic-tree coordinate fields must have equal lengths.")
        next_coordinate = 0
        for start, stop in self.joint_coordinate_ranges:
            if start != next_coordinate or stop < start:
                raise ValueError("Kinematic-tree joint coordinate ranges must be contiguous and nondecreasing.")
            next_coordinate = stop
        if next_coordinate != coordinate_count:
            raise ValueError("Kinematic-tree joint ranges must cover every coordinate.")
        if any(
            len(axis) != 3
            or any(not math.isfinite(component) for component in axis)
            or not math.isclose(sum(component * component for component in axis), 1.0, rel_tol=1.0e-5, abs_tol=1.0e-5)
            for axis in self.coordinate_axes
        ):
            raise ValueError("Kinematic-tree coordinate axes must be finite unit vectors.")
        if len(set(self.coordinate_q_indices)) != coordinate_count or any(
            index < 0 for index in self.coordinate_q_indices
        ):
            raise ValueError("Kinematic-tree generalized-position indices must be unique and nonnegative.")
        if len(set(self.coordinate_qd_indices)) != coordinate_count or any(
            index < 0 for index in self.coordinate_qd_indices
        ):
            raise ValueError("Kinematic-tree generalized-velocity indices must be unique and nonnegative.")
        if any(
            lower > upper for lower, upper in zip(self.coordinate_lower_limits_rad, self.coordinate_upper_limits_rad)
        ):
            raise ValueError("Kinematic-tree coordinate lower limits must not exceed upper limits.")

    @classmethod
    def from_newton(cls, kinematics: NewtonKinematics) -> KinematicTree:
        """Derive grouped rotational coordinates from canonical Newton topology.

        Args:
            kinematics: Finalized Newton kinematic model.

        Returns:
            Body topology and exact grouped rotational coordinates.
        """
        topology = kinematics.topology
        if len(kinematics.body_names) != topology.body_count or len(kinematics.joint_names) != topology.joint_count:
            raise ValueError("Newton model names and canonical topology counts differ.")
        if len(kinematics.joint_q_names) != topology.coordinate_count:
            raise ValueError("Newton generalized-position names and canonical ranges differ.")

        root = int(np.flatnonzero(topology.body_parent == -1)[0])
        joint_names: list[str] = []
        joint_children: list[int] = []
        joint_ranges: list[tuple[int, int]] = []
        coordinate_names: list[str] = []
        coordinate_axes: list[tuple[float, float, float]] = []
        coordinate_q_indices: list[int] = []
        coordinate_qd_indices: list[int] = []
        lower_limits: list[float] = []
        upper_limits: list[float] = []
        for joint_index, child_value in enumerate(topology.joint_child):
            child = int(child_value)
            if child < 0 or child == root:
                continue
            q_begin = int(topology.joint_q_start[joint_index])
            q_end = int(topology.joint_q_start[joint_index + 1])
            qd_begin = int(topology.joint_qd_start[joint_index])
            qd_end = int(topology.joint_qd_start[joint_index + 1])
            linear_count, angular_count = (int(value) for value in topology.joint_dof_dim[joint_index])
            coordinate_count = q_end - q_begin
            if coordinate_count == 0:
                if linear_count != 0 or angular_count != 0 or qd_end != qd_begin:
                    raise ValueError("Fixed joints must own no generalized coordinates or velocities.")
            elif (
                linear_count != 0
                or angular_count != coordinate_count
                or qd_end - qd_begin != coordinate_count
                or coordinate_count > 3
            ):
                raise ValueError("KinematicTree requires fixed or one-to-three-axis rotational non-root joints.")
            start = len(coordinate_names)
            joint_names.append(str(kinematics.joint_names[joint_index]))
            joint_children.append(child)
            coordinate_names.extend(str(name) for name in kinematics.joint_q_names[q_begin:q_end])
            coordinate_axes.extend(
                tuple(float(value) for value in axis) for axis in topology.joint_axis[qd_begin:qd_end]
            )
            coordinate_q_indices.extend(range(q_begin, q_end))
            coordinate_qd_indices.extend(range(qd_begin, qd_end))
            lower_limits.extend(float(value) for value in topology.joint_limit_lower[qd_begin:qd_end])
            upper_limits.extend(float(value) for value in topology.joint_limit_upper[qd_begin:qd_end])
            joint_ranges.append((start, len(coordinate_names)))
        return cls(
            body_names=tuple(str(name) for name in kinematics.body_names),
            parent_indices=tuple(int(parent) for parent in topology.body_parent),
            joint_names=tuple(joint_names),
            joint_child_body_indices=tuple(joint_children),
            joint_coordinate_ranges=tuple(joint_ranges),
            coordinate_names=tuple(coordinate_names),
            coordinate_axes=tuple(coordinate_axes),
            coordinate_q_indices=tuple(coordinate_q_indices),
            coordinate_qd_indices=tuple(coordinate_qd_indices),
            coordinate_lower_limits_rad=tuple(lower_limits),
            coordinate_upper_limits_rad=tuple(upper_limits),
        )

    @property
    def num_bodies(self) -> int:
        """Number of bodies."""
        return len(self.body_names)

    @property
    def num_joints(self) -> int:
        """Number of non-root joints."""
        return len(self.joint_names)

    @property
    def num_coordinates(self) -> int:
        """Number of non-root rotational coordinates."""
        return len(self.coordinate_names)

    @property
    def root_body_index(self) -> int:
        """Index of the unique root body."""
        return self.parent_indices.index(-1)

    @property
    def coordinate_child_body_indices(self) -> tuple[int, ...]:
        """Child body owned by each rotational coordinate."""
        return tuple(
            child
            for child, (start, stop) in zip(self.joint_child_body_indices, self.joint_coordinate_ranges)
            for _ in range(start, stop)
        )

    def coordinates_within_limits(self, coordinates_rad: torch.Tensor) -> torch.Tensor:
        """Test coordinates against the exact target bounds.

        Args:
            coordinates_rad: Joint coordinates [rad], shape ``[..., coordinate_count]``.

        Returns:
            Elementwise limit membership with the same shape as :paramref:`coordinates_rad`.
        """
        if coordinates_rad.shape[-1] != self.num_coordinates:
            raise ValueError("Joint coordinates have the wrong final dimension.")
        lower = coordinates_rad.new_tensor(self.coordinate_lower_limits_rad)
        upper = coordinates_rad.new_tensor(self.coordinate_upper_limits_rad)
        return (coordinates_rad >= lower) & (coordinates_rad <= upper)


def kinematic_tree_forward(
    local_translation_m: torch.Tensor,
    local_rotation_xyzw: torch.Tensor,
    parent_indices: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose parent-relative body transforms through a topologically ordered tree.

    Args:
        local_translation_m: Parent-relative body translations [m], shape ``[..., body_count, 3]``.
        local_rotation_xyzw: Parent-relative unit quaternions, shape ``[..., body_count, 4]``.
        parent_indices: Parent body per body, with ``-1`` for the root.

    Returns:
        World translations [m] and xyzw rotations with the same respective shapes as the inputs.
    """
    body_count = len(parent_indices)
    if local_translation_m.shape[:-1] != local_rotation_xyzw.shape[:-1]:
        raise ValueError("Local translations and rotations must have equal leading shapes.")
    if local_translation_m.shape[-2:] != (body_count, 3) or local_rotation_xyzw.shape[-2:] != (body_count, 4):
        raise ValueError("Local transform shapes must agree with the kinematic-tree body count.")
    if (
        local_translation_m.dtype != local_rotation_xyzw.dtype
        or local_translation_m.device != local_rotation_xyzw.device
    ):
        raise ValueError("Local translations and rotations must share one dtype and device.")
    if not local_translation_m.is_floating_point():
        raise ValueError("Local transforms must use a floating-point dtype.")
    roots = tuple(body for body, parent in enumerate(parent_indices) if parent == -1)
    if len(roots) != 1 or any(parent < -1 or parent >= body for body, parent in enumerate(parent_indices)):
        raise ValueError("Parent indices must define one topologically ordered tree.")

    world_translation_m = torch.empty_like(local_translation_m)
    world_rotation_xyzw = torch.empty_like(local_rotation_xyzw)
    root = roots[0]
    world_translation_m[..., root, :].copy_(local_translation_m[..., root, :])
    world_rotation_xyzw[..., root, :].copy_(local_rotation_xyzw[..., root, :])
    for body, parent in enumerate(parent_indices):
        if body == root:
            continue
        world_rotation_xyzw[..., body, :] = quat_mul(
            world_rotation_xyzw[..., parent, :], local_rotation_xyzw[..., body, :]
        )
        world_translation_m[..., body, :] = world_translation_m[..., parent, :] + quat_apply(
            world_rotation_xyzw[..., parent, :], local_translation_m[..., body, :]
        )
    return world_translation_m, world_rotation_xyzw


def kinematic_pose_forward(
    rest_translation_m: torch.Tensor,
    rest_rotation_xyzw: torch.Tensor,
    root_world_and_local_body_rotation_xyzw: torch.Tensor,
    root_position_m: torch.Tensor,
    parent_indices: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose one world root rotation and non-root local pose deltas through a kinematic tree.

    Args:
        rest_translation_m: Parent-relative rest translations [m], shape ``[body_count, 3]``.
        rest_rotation_xyzw: Parent-relative rest rotations, shape ``[body_count, 4]``.
        root_world_and_local_body_rotation_xyzw: World root rotation in row zero and parent-local
            non-root pose deltas from the rest pose, shape ``[..., body_count, 4]``.
        root_position_m: World root positions [m], shape ``[..., 3]``.
        parent_indices: Parent body per body, with ``-1`` for the root.

    Returns:
        World body positions [m] and xyzw rotations with the pose leading shape.
    """
    body_count = len(parent_indices)
    body_rotation = root_world_and_local_body_rotation_xyzw
    if rest_translation_m.shape != (body_count, 3) or rest_rotation_xyzw.shape != (body_count, 4):
        raise ValueError("Rest transforms must contain one xyz/quaternion row per body.")
    if body_rotation.shape[-2:] != (body_count, 4) or root_position_m.shape != body_rotation.shape[:-2] + (3,):
        raise ValueError("Body rotations and root positions must share one leading shape.")
    tensors = (rest_translation_m, rest_rotation_xyzw, body_rotation, root_position_m)
    if any(tensor.dtype != body_rotation.dtype or tensor.device != body_rotation.device for tensor in tensors):
        raise ValueError("Rest transforms, body rotations, and root positions must share one dtype and device.")

    world_position_m = torch.empty(
        body_rotation.shape[:-1] + (3,), dtype=body_rotation.dtype, device=body_rotation.device
    )
    world_rotation_xyzw = torch.empty_like(body_rotation)
    root = parent_indices.index(-1)
    world_position_m[..., root, :].copy_(root_position_m)
    world_rotation_xyzw[..., root, :].copy_(body_rotation[..., root, :])
    for body, parent in enumerate(parent_indices):
        if body == root:
            continue
        local_rotation = quat_mul(
            rest_rotation_xyzw[body].expand_as(body_rotation[..., body, :]), body_rotation[..., body, :]
        )
        world_rotation_xyzw[..., body, :] = quat_mul(world_rotation_xyzw[..., parent, :], local_rotation)
        world_position_m[..., body, :] = world_position_m[..., parent, :] + quat_apply(
            world_rotation_xyzw[..., parent, :], rest_translation_m[body].expand_as(world_position_m[..., parent, :])
        )
    return world_position_m, world_rotation_xyzw


def kinematic_root_basis(
    body_position_m: torch.Tensor,
    root_body_index: int,
    left_body_index: int,
    right_body_index: int,
    up_body_index: int,
) -> torch.Tensor:
    """Derive a right-handed root basis from bilateral and vertical body landmarks.

    Args:
        body_position_m: Body positions [m], shape ``[..., body_count, 3]``.
        root_body_index: Root landmark index.
        left_body_index: Left bilateral landmark index.
        right_body_index: Right bilateral landmark index.
        up_body_index: Landmark above the root.

    Returns:
        Rotation matrices whose columns are forward, left, and up, shape ``[..., 3, 3]``.
    """
    if body_position_m.ndim < 2 or body_position_m.shape[-1] != 3:
        raise ValueError("Body positions must end in [body_count, 3].")
    body_count = body_position_m.shape[-2]
    indices = (root_body_index, left_body_index, right_body_index, up_body_index)
    if any(index < 0 or index >= body_count for index in indices) or len(set(indices)) != len(indices):
        raise ValueError("Root-basis landmarks must be distinct valid body indices.")
    up = F.normalize(body_position_m[..., up_body_index, :] - body_position_m[..., root_body_index, :], dim=-1)
    left = body_position_m[..., left_body_index, :] - body_position_m[..., right_body_index, :]
    left = F.normalize(left - (left * up).sum(dim=-1, keepdim=True) * up, dim=-1)
    forward = torch.linalg.cross(left, up, dim=-1)
    return torch.stack((forward, left, up), dim=-1)


def time_gradient(values: torch.Tensor, step_seconds: float) -> torch.Tensor:
    """Differentiate time axis one with first-order edges and central interiors.

    Args:
        values: Batch-major values with shape ``[batch, time, ...]``.
        step_seconds: Uniform sample interval [s].

    Returns:
        Time derivative with the same shape as :paramref:`values`.
    """
    if values.ndim < 2 or values.shape[1] < 2:
        raise ValueError("Time gradients require at least two samples on axis one.")
    result = torch.empty_like(values)
    result[:, 0] = (values[:, 1] - values[:, 0]) / step_seconds
    result[:, -1] = (values[:, -1] - values[:, -2]) / step_seconds
    if values.shape[1] > 2:
        result[:, 1:-1] = (values[:, 2:] - values[:, :-2]) / (2.0 * step_seconds)
    return result


def time_quaternion_angular_velocity(rotation_xyzw: torch.Tensor, step_seconds: float) -> torch.Tensor:
    """Differentiate unit quaternions with the released finite-difference law.

    Args:
        rotation_xyzw: Batch-major unit quaternions in xyzw order with shape ``[batch, time, ..., 4]``.
        step_seconds: Uniform sample interval [s].

    Returns:
        Angular velocities [rad/s] with the same leading shape and a final xyz axis.
    """
    relative = quat_mul(rotation_xyzw[:, 1:], quat_conjugate(rotation_xyzw[:, :-1]))
    relative = relative / relative.norm(p=2, dim=-1, keepdim=True).clamp_min(1.0e-9)
    angular_velocity = torch.zeros_like(rotation_xyzw[..., :3])
    angular_velocity[:, :-1] = axis_angle_from_quat(relative) / step_seconds
    return angular_velocity


def time_gaussian_filter(values: torch.Tensor, *, sigma: float = 2.0) -> torch.Tensor:
    """Apply a nearest-edge Gaussian filter along time axis one.

    Args:
        values: Batch-major values with shape ``[batch, time, ...]``.
        sigma: Gaussian standard deviation in samples.

    Returns:
        Filtered values with the same shape and dtype as :paramref:`values`.
    """
    radius = round(4.0 * sigma)
    coordinate = torch.arange(-radius, radius + 1, dtype=torch.float64, device=values.device)
    kernel = torch.exp(-0.5 * (coordinate / sigma) ** 2)
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    output_dtype = values.dtype
    moved = values.to(torch.float64).movedim(1, -1)
    flattened = moved.reshape(-1, 1, moved.shape[-1])
    filtered = F.conv1d(F.pad(flattened, (radius, radius), mode="replicate"), kernel)
    return filtered.reshape(moved.shape).movedim(-1, 1).to(output_dtype)


def _time_segment_bounds(
    values: torch.Tensor,
    offsets: torch.Tensor,
    *,
    minimum_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate a flat segmented time axis and return its row-local bounds."""
    if values.ndim < 1 or values.shape[0] < 1:
        raise ValueError("Segmented time values require a nonempty leading frame axis.")
    if not values.is_floating_point():
        raise ValueError("Segmented time values must use a floating-point dtype.")
    if offsets.ndim != 1 or offsets.dtype is not torch.int64 or not offsets.is_contiguous() or offsets.shape[0] < 2:
        raise ValueError("Segment offsets must be contiguous int64 with at least two entries.")
    if offsets.device != values.device:
        raise ValueError("Segment values and offsets must share one device.")
    if (
        int(offsets[0]) != 0
        or int(offsets[-1]) != values.shape[0]
        or bool(torch.any(offsets[1:] - offsets[:-1] < minimum_frames))
    ):
        raise ValueError(f"Segment offsets must span the values with at least {minimum_frames} frames per segment.")

    rows = torch.arange(values.shape[0], dtype=torch.int64, device=values.device)
    segments = torch.searchsorted(offsets[1:], rows, right=True)
    starts = offsets.index_select(0, segments)
    stops = offsets.index_select(0, segments + 1)
    return rows, starts, stops, segments


def _time_segment_rows(
    values: torch.Tensor,
    offsets: torch.Tensor,
    step_seconds: torch.Tensor,
    *,
    minimum_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return flat segment bounds and each row's validated sample interval [s]."""
    rows, starts, stops, segments = _time_segment_bounds(values, offsets, minimum_frames=minimum_frames)
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
    return rows, starts, stops, step_seconds.index_select(0, segments)


def time_gradient_segmented(
    values: torch.Tensor,
    offsets: torch.Tensor,
    step_seconds: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiate a flat segmented time axis without crossing segment boundaries.

    Args:
        values: Values with shape ``[frame_count, ...]``.
        offsets: Segment offsets with shape ``[segment_count + 1]``.
        step_seconds: Uniform sample interval within each segment [s], shape ``[segment_count]``.
        out: Optional caller-owned derivative storage with the same shape as :paramref:`values`.

    Returns:
        Time derivative with the same shape as :paramref:`values`.
    """
    rows, starts, stops, steps = _time_segment_rows(values, offsets, step_seconds, minimum_frames=2)
    if out is None:
        out = torch.empty_like(values)
    elif out.shape != values.shape or out.dtype != values.dtype or out.device != values.device:
        raise ValueError("Segmented gradient output must match the input shape, dtype, and device.")
    previous = torch.maximum(rows - 1, starts)
    following = torch.minimum(rows + 1, stops - 1)
    denominator = (following - previous).to(step_seconds.dtype) * steps
    while denominator.ndim < values.ndim:
        denominator = denominator.unsqueeze(-1)
    torch.sub(values.index_select(0, following), values.index_select(0, previous), out=out)
    return out.div_(denominator)


def time_quaternion_angular_velocity_segmented(
    rotation_xyzw: torch.Tensor,
    offsets: torch.Tensor,
    step_seconds: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiate flat segmented unit quaternions with a zero segment tail.

    Args:
        rotation_xyzw: Unit quaternions in xyzw order with shape ``[frame_count, ..., 4]``.
        offsets: Segment offsets with shape ``[segment_count + 1]``.
        step_seconds: Uniform sample interval within each segment [s], shape ``[segment_count]``.
        out: Optional caller-owned angular-velocity storage ending in xyz.

    Returns:
        Angular velocities [rad/s] with the same leading shape and a final xyz axis.
    """
    if rotation_xyzw.ndim < 2 or rotation_xyzw.shape[-1] != 4:
        raise ValueError("Segmented quaternion velocities require values ending in four.")
    output_shape = (*rotation_xyzw.shape[:-1], 3)
    if out is None:
        out = rotation_xyzw.new_empty(output_shape)
    elif out.shape != output_shape or out.dtype != rotation_xyzw.dtype or out.device != rotation_xyzw.device:
        raise ValueError("Segmented quaternion-velocity output must match the input leading shape, dtype, and device.")

    rows, _starts, stops, steps = _time_segment_rows(rotation_xyzw, offsets, step_seconds, minimum_frames=1)
    following = torch.minimum(rows + 1, stops - 1)
    relative = quat_mul(rotation_xyzw.index_select(0, following), quat_conjugate(rotation_xyzw))
    relative = relative / relative.norm(p=2, dim=-1, keepdim=True).clamp_min(1.0e-9)
    angular_velocity = axis_angle_from_quat(relative)
    while steps.ndim < angular_velocity.ndim:
        steps = steps.unsqueeze(-1)
    angular_velocity = angular_velocity / steps
    angular_velocity[rows == stops - 1] = 0.0
    out.copy_(angular_velocity)
    return out


def time_gaussian_filter_segmented(
    values: torch.Tensor,
    offsets: torch.Tensor,
    *,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Filter a flat segmented time axis with nearest segment-boundary padding.

    Args:
        values: Values with shape ``[frame_count, ...]``.
        offsets: Segment offsets with shape ``[segment_count + 1]``.
        sigma: Gaussian standard deviation in samples.

    Returns:
        Filtered values with the same shape and dtype as :paramref:`values`.
    """
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Gaussian sigma must be finite and positive.")
    rows, starts, stops, _segments = _time_segment_bounds(values, offsets, minimum_frames=1)
    radius = round(4.0 * sigma)
    weights = [math.exp(-0.5 * (coordinate / sigma) ** 2) for coordinate in range(-radius, radius + 1)]
    weight_sum = sum(weights)
    source = values.to(torch.float64)
    filtered = torch.zeros_like(source)
    for coordinate, weight in zip(range(-radius, radius + 1), weights, strict=True):
        source_rows = torch.minimum(torch.maximum(rows + coordinate, starts), stops - 1)
        filtered.add_(source.index_select(0, source_rows), alpha=weight / weight_sum)
    return filtered.to(values.dtype)


def time_backward_difference_segmented(
    values: torch.Tensor,
    offsets: torch.Tensor,
    step_seconds: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiate canonical backward edges and repeat the first segment edge.

    Args:
        values: Values with shape ``[frame_count, ...]``.
        offsets: Segment offsets with shape ``[segment_count + 1]``.
        step_seconds: Uniform sample interval within each segment [s].
        out: Optional caller-owned derivative storage matching :paramref:`values`.

    Returns:
        Backward-edge derivatives with the same shape as :paramref:`values`.
    """
    rows, starts, _stops, steps = _time_segment_rows(values, offsets, step_seconds, minimum_frames=2)
    first = rows == starts
    current = torch.where(first, rows + 1, rows)
    previous = torch.where(first, rows, rows - 1)
    if out is None:
        out = torch.empty_like(values)
    elif out.shape != values.shape or out.dtype != values.dtype or out.device != values.device:
        raise ValueError("Segmented backward-difference output must match the input.")
    while steps.ndim < values.ndim:
        steps = steps.unsqueeze(-1)
    torch.sub(values.index_select(0, current), values.index_select(0, previous), out=out)
    return out.div_(steps)


def time_quaternion_angular_velocity_backward_segmented(
    rotation_xyzw: torch.Tensor,
    offsets: torch.Tensor,
    step_seconds: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiate canonical backward quaternion edges and repeat each first edge.

    Args:
        rotation_xyzw: Unit quaternions in xyzw order, shape ``[frame_count, ..., 4]``.
        offsets: Segment offsets with shape ``[segment_count + 1]``.
        step_seconds: Uniform sample interval within each segment [s].
        out: Optional caller-owned angular-velocity storage ending in xyz.

    Returns:
        World angular velocities [rad/s] with the same leading shape and final xyz axis.
    """
    if rotation_xyzw.ndim < 2 or rotation_xyzw.shape[-1] != 4:
        raise ValueError("Segmented quaternion velocities require values ending in four.")
    rows, starts, _stops, steps = _time_segment_rows(rotation_xyzw, offsets, step_seconds, minimum_frames=2)
    first = rows == starts
    current = torch.where(first, rows + 1, rows)
    previous = torch.where(first, rows, rows - 1)
    relative = quat_mul(rotation_xyzw.index_select(0, current), quat_conjugate(rotation_xyzw.index_select(0, previous)))
    relative.div_(relative.norm(p=2, dim=-1, keepdim=True).clamp_min_(1.0e-9))
    angular_velocity = axis_angle_from_quat(relative)
    while steps.ndim < angular_velocity.ndim:
        steps = steps.unsqueeze(-1)
    angular_velocity.div_(steps)
    output_shape = (*rotation_xyzw.shape[:-1], 3)
    if out is None:
        return angular_velocity
    if out.shape != output_shape or out.dtype != rotation_xyzw.dtype or out.device != rotation_xyzw.device:
        raise ValueError("Segmented backward quaternion-velocity output must match the input leading shape.")
    out.copy_(angular_velocity)
    return out


def time_forward_difference_segmented(
    values: torch.Tensor,
    offsets: torch.Tensor,
    step_seconds: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the deprecated released-G1 forward-difference policy.

    Args:
        values: Values with shape ``[frame_count, ...]``.
        offsets: Segment offsets with shape ``[segment_count + 1]``.
        step_seconds: Uniform sample interval within each segment [s], shape ``[segment_count]``.
        out: Optional caller-owned derivative storage matching :paramref:`values`.

    Returns:
        Time derivative with the same shape as :paramref:`values`.
    """
    warnings.warn(
        "time_forward_difference_segmented() is deprecated; "
        "the released derivative policy is owned by G1 frame construction.",
        DeprecationWarning,
        stacklevel=2,
    )
    rows, _starts, stops, steps = _time_segment_rows(values, offsets, step_seconds, minimum_frames=3)
    if out is None:
        out = torch.empty_like(values)
    elif out.shape != values.shape or out.dtype != values.dtype or out.device != values.device:
        raise ValueError("Segmented forward-difference output must match the input.")
    tail = rows == stops - 1
    previous = torch.where(tail, stops - 3, rows)
    following = torch.where(tail, stops - 2, rows + 1)
    while steps.ndim < values.ndim:
        steps = steps.unsqueeze(-1)
    torch.sub(values.index_select(0, following), values.index_select(0, previous), out=out)
    return out.div_(steps)


def ordered_hinge_rotation(coordinates: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
    """Compose ordered hinge coordinates into local rotations.

    Args:
        coordinates: Ordered hinge angles [rad], shape ``[..., hinge_count]``.
        axes: Unit hinge axes, shape ``[hinge_count, 3]`` or
            ``[..., hinge_count, 3]`` broadcastable into :paramref:`coordinates`.

    Returns:
        Unit quaternions in xyzw order, shape ``[..., 4]``.
    """
    if coordinates.ndim < 1 or coordinates.shape[-1] < 1:
        raise ValueError("Ordered hinge coordinates must contain at least one hinge.")
    if axes.ndim < 2 or axes.shape[-2:] != (coordinates.shape[-1], 3):
        raise ValueError("Ordered hinge axes must end in [hinge_count, 3].")
    if not coordinates.is_floating_point() or axes.dtype != coordinates.dtype:
        raise ValueError("Ordered hinge coordinates and axes must share one floating-point dtype.")
    if axes.device != coordinates.device:
        raise ValueError("Ordered hinge coordinates and axes must share one device.")
    try:
        leading_shape = torch.broadcast_shapes(coordinates.shape[:-1], axes.shape[:-2])
    except RuntimeError as error:
        raise ValueError("Ordered hinge axes are not broadcastable into the coordinates.") from error
    if leading_shape != coordinates.shape[:-1]:
        raise ValueError("Ordered hinge axes must not expand the coordinate batch shape.")

    rotation_vector = coordinates[..., 0, None] * axes[..., 0, :]
    quaternion = quat_from_rotation_vector(rotation_vector)
    for index in range(1, coordinates.shape[-1]):
        rotation_vector = coordinates[..., index, None] * axes[..., index, :]
        quaternion = quat_mul(quaternion, quat_from_rotation_vector(rotation_vector))
    return quaternion


def ordered_hinge_coordinate_velocity(
    coordinates: torch.Tensor, axes: torch.Tensor, angular_velocity: torch.Tensor
) -> torch.Tensor:
    """Resolve parent-frame angular velocity into three ordered hinge rates.

    Args:
        coordinates: Ordered hinge angles [rad], shape ``[..., 3]``.
        axes: Right-handed unit hinge axes, shape ``[3, 3]`` or
            ``[..., 3, 3]`` broadcastable into :paramref:`coordinates`.
        angular_velocity: Parent-frame angular velocity [rad/s], shape ``[..., 3]``.

    Returns:
        Ordered hinge coordinate velocities [rad/s], shape ``[..., 3]``.
    """
    if coordinates.ndim < 1 or coordinates.shape[-1] != 3:
        raise ValueError("Ordered hinge velocity conversion requires exactly three coordinates.")
    if axes.ndim < 2 or axes.shape[-2:] != (3, 3):
        raise ValueError("Ordered hinge velocity axes must end in [3, 3].")
    if angular_velocity.shape != coordinates.shape:
        raise ValueError("Ordered hinge angular velocity must match the coordinate shape.")
    if (
        not coordinates.is_floating_point()
        or axes.dtype != coordinates.dtype
        or angular_velocity.dtype != coordinates.dtype
    ):
        raise ValueError("Ordered hinge velocity inputs must share one floating-point dtype.")
    if axes.device != coordinates.device or angular_velocity.device != coordinates.device:
        raise ValueError("Ordered hinge velocity inputs must share one device.")
    try:
        leading_shape = torch.broadcast_shapes(coordinates.shape[:-1], axes.shape[:-2])
    except RuntimeError as error:
        raise ValueError("Ordered hinge velocity axes are not broadcastable into the coordinates.") from error
    if leading_shape != coordinates.shape[:-1]:
        raise ValueError("Ordered hinge velocity axes must not expand the coordinate batch shape.")

    axis_0 = torch.broadcast_to(axes[..., 0, :], coordinates.shape)
    axis_1_local = torch.broadcast_to(axes[..., 1, :], coordinates.shape)
    axis_2_local = torch.broadcast_to(axes[..., 2, :], coordinates.shape)
    first_rotation = quat_from_rotation_vector(coordinates[..., 0, None] * axis_0)
    axis_1 = quat_apply(first_rotation, axis_1_local)
    second_rotation = quat_mul(first_rotation, quat_from_rotation_vector(coordinates[..., 1, None] * axis_1_local))
    axis_2 = quat_apply(second_rotation, axis_2_local)
    cross_12 = torch.cross(axis_1, axis_2, dim=-1)
    cross_02 = torch.cross(axis_0, axis_2, dim=-1)
    cross_01 = torch.cross(axis_0, axis_1, dim=-1)
    return torch.stack(
        (
            torch.sum(angular_velocity * cross_12, dim=-1) / torch.sum(axis_0 * cross_12, dim=-1),
            torch.sum(angular_velocity * cross_02, dim=-1) / torch.sum(axis_1 * cross_02, dim=-1),
            torch.sum(angular_velocity * cross_01, dim=-1) / torch.sum(axis_2 * cross_01, dim=-1),
        ),
        dim=-1,
    )


def time_unwrap_angles(coordinates: torch.Tensor) -> torch.Tensor:
    """Choose continuous representatives of principal angles along time.

    Args:
        coordinates: Principal angles [rad], shape ``[frame_count, coordinate_count]``.

    Returns:
        Continuous angles [rad] with the same shape as :paramref:`coordinates`.
    """
    if coordinates.ndim != 2:
        raise ValueError("Time unwrapping requires [frame_count, hinge_count] coordinates.")
    if coordinates.shape[0] < 2:
        return coordinates
    difference = coordinates[1:] - coordinates[:-1]
    difference = torch.atan2(torch.sin(difference), torch.cos(difference))
    return torch.cat((coordinates[:1], coordinates[:1] + torch.cumsum(difference, dim=0)), dim=0)


def time_unwrap_angles_segmented(coordinates: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Choose continuous angle representatives independently within each clip.

    Args:
        coordinates: Principal angles [rad], shape ``[frame_count, coordinate_count]``.
        offsets: Clip offsets with shape ``[clip_count + 1]``.

    Returns:
        Clip-local continuous angles [rad] with the same shape as :paramref:`coordinates`.
    """
    if coordinates.ndim != 2:
        raise ValueError("Segmented time unwrapping requires [frame_count, hinge_count] coordinates.")
    _rows, _starts, _stops, segments = _time_segment_bounds(coordinates, offsets, minimum_frames=1)
    unwrapped = time_unwrap_angles(coordinates)
    segment_starts = offsets[:-1]
    shifts = coordinates.index_select(0, segment_starts) - unwrapped.index_select(0, segment_starts)
    return unwrapped + shifts.index_select(0, segments)


def _fit_independent_hinge_coordinates(rotation_xyzw: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
    """Fit one ordered product of independent, nonorthogonal hinge axes."""
    hinge_count = axes.shape[0]
    batch_shape = rotation_xyzw.shape[:-1]
    target = rotation_xyzw.reshape(-1, 4)
    coordinates = torch.zeros(target.shape[0], hinge_count, dtype=torch.float32, device=target.device)
    identity = torch.zeros(target.shape[0], 4, dtype=torch.float32, device=target.device)
    identity[:, 3] = 1.0
    damping = 1.0e-6 * torch.eye(hinge_count, dtype=torch.float32, device=target.device)
    for _ in range(8):
        fitted = identity.clone()
        spatial_axes = []
        for index in range(hinge_count):
            axis = axes[index].expand(target.shape[0], 3)
            spatial_axes.append(quat_apply(fitted, axis))
            fitted = quat_mul(
                fitted,
                quat_from_rotation_vector(coordinates[:, index, None] * axes[index]),
            )
        relative = quat_mul(quat_conjugate(fitted), target)
        error = axis_angle_from_quat(relative)
        spatial_axes = torch.stack(spatial_axes, dim=1)
        inverse_fitted = quat_conjugate(fitted)[:, None].expand(-1, hinge_count, -1)
        body_axes = quat_apply(inverse_fitted.reshape(-1, 4), spatial_axes.reshape(-1, 3)).view_as(spatial_axes)
        jacobian = body_axes.transpose(1, 2)
        normal = jacobian.transpose(1, 2) @ jacobian + damping
        right_hand_side = (jacobian.transpose(1, 2) @ error.unsqueeze(-1)).squeeze(-1)
        step = torch.linalg.solve(normal, right_hand_side).clamp_(-0.5, 0.5)
        coordinates.add_(step)
    return coordinates.reshape(*batch_shape, hinge_count)


def _fit_redundant_hinge_coordinates(
    rotation_xyzw: torch.Tensor,
    axes: torch.Tensor,
    initial_coordinates: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Fit a redundant serial path while preserving its caller-owned default posture."""
    target = torch.nn.functional.normalize(rotation_xyzw.reshape(-1, 4), dim=-1)
    coordinate_count = axes.shape[0]
    coordinates = initial_coordinates.reshape(-1, coordinate_count).clone()
    coordinates.clamp_(lower, upper)
    identity = torch.zeros(target.shape[0], 4, dtype=torch.float32, device=target.device)
    identity[:, 3] = 1.0
    damping = 1.0e-6 * torch.eye(3, dtype=torch.float32, device=target.device)
    for _ in range(12):
        fitted = identity.clone()
        spatial_axes = []
        for index in range(coordinate_count):
            axis = axes[index].expand(target.shape[0], 3)
            spatial_axes.append(quat_apply(fitted, axis))
            fitted = quat_mul(fitted, quat_from_rotation_vector(coordinates[:, index, None] * axes[index]))
        error = axis_angle_from_quat(quat_mul(quat_conjugate(fitted), target))
        spatial_axes = torch.stack(spatial_axes, dim=1)
        inverse_fitted = quat_conjugate(fitted)[:, None].expand(-1, coordinate_count, -1)
        body_axes = quat_apply(inverse_fitted.reshape(-1, 4), spatial_axes.reshape(-1, 3)).view_as(spatial_axes)
        jacobian = body_axes.transpose(1, 2)
        task_normal = jacobian @ jacobian.transpose(1, 2) + damping
        task_step = torch.linalg.solve(task_normal, error.unsqueeze(-1))
        step = (jacobian.transpose(1, 2) @ task_step).squeeze(-1)
        step.mul_(0.5 / step.abs().amax(dim=-1, keepdim=True).clamp_min_(0.5))
        coordinates.add_(step).clamp_(lower, upper)
    return coordinates.reshape_as(initial_coordinates)


def _fit_coupled_hinge_coordinates(
    target_rotations_xyzw: tuple[torch.Tensor, ...],
    axes_by_path: tuple[torch.Tensor, ...],
    coordinate_positions_by_path: tuple[torch.Tensor, ...],
    initial_coordinates: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Fit coupled semantic paths that share target coordinates."""
    path_count = len(target_rotations_xyzw)
    problem_count, coordinate_count = initial_coordinates.shape
    coordinates = initial_coordinates.clone()
    coordinates.clamp_(lower, upper)
    identity = torch.zeros(problem_count, 4, dtype=torch.float32, device=coordinates.device)
    identity[:, 3] = 1.0
    damping = 1.0e-6 * torch.eye(3 * path_count, dtype=torch.float32, device=coordinates.device)
    error = torch.empty(problem_count, 3 * path_count, dtype=torch.float32, device=coordinates.device)
    jacobian = torch.empty(
        problem_count, 3 * path_count, coordinate_count, dtype=torch.float32, device=coordinates.device
    )
    for _ in range(12):
        jacobian.zero_()
        for path_index, (target, axes, coordinate_positions) in enumerate(
            zip(target_rotations_xyzw, axes_by_path, coordinate_positions_by_path, strict=True)
        ):
            fitted = identity.clone()
            spatial_axes = []
            path_coordinates = coordinates.index_select(1, coordinate_positions)
            for coordinate_index in range(axes.shape[0]):
                axis = axes[coordinate_index].expand(problem_count, 3)
                spatial_axes.append(quat_apply(fitted, axis))
                fitted = quat_mul(
                    fitted,
                    quat_from_rotation_vector(path_coordinates[:, coordinate_index, None] * axes[coordinate_index]),
                )
            error[:, 3 * path_index : 3 * path_index + 3] = axis_angle_from_quat(
                quat_mul(quat_conjugate(fitted), target)
            )
            if spatial_axes:
                spatial_axes_tensor = torch.stack(spatial_axes, dim=1)
                inverse_fitted = quat_conjugate(fitted)[:, None].expand(-1, axes.shape[0], -1)
                body_axes = quat_apply(inverse_fitted.reshape(-1, 4), spatial_axes_tensor.reshape(-1, 3)).view_as(
                    spatial_axes_tensor
                )
                jacobian[:, 3 * path_index : 3 * path_index + 3].index_copy_(
                    2, coordinate_positions, body_axes.transpose(1, 2)
                )
        task_step = torch.linalg.solve(jacobian @ jacobian.transpose(1, 2) + damping, error.unsqueeze(-1))
        step = (jacobian.transpose(1, 2) @ task_step).squeeze(-1)
        step.mul_(0.5 / step.abs().amax(dim=-1, keepdim=True).clamp_min_(0.5))
        coordinates.add_(step).clamp_(lower, upper)
    return coordinates


def fit_ordered_hinge_coordinates(
    rotation_xyzw: torch.Tensor,
    axes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one local rotation to one, two, or three ordered independent hinges.

    Args:
        rotation_xyzw: Unit quaternions in xyzw order.
        axes: One to three ordered independent unit hinge axes.

    Returns:
        Principal hinge coordinates [rad] and geodesic residuals [rad].
    """
    if rotation_xyzw.shape[-1] != 4 or axes.ndim != 2 or axes.shape[-1] != 3:
        raise ValueError("Ordered-hinge inputs must end in four and [hinge_count, 3].")
    hinge_count = axes.shape[0]
    if hinge_count < 1 or hinge_count > 3:
        raise ValueError("Ordered-hinge fitting supports one to three coordinates.")
    if rotation_xyzw.dtype is not torch.float32 or axes.dtype is not torch.float32:
        raise ValueError("Ordered-hinge fitting requires float32 tensors.")
    if rotation_xyzw.device != axes.device:
        raise ValueError("Ordered-hinge rotations and axes must share one device.")
    gram = axes @ axes.transpose(0, 1)
    if not torch.allclose(torch.diagonal(gram), torch.ones(hinge_count, device=axes.device), atol=1.0e-6) or (
        hinge_count > 1 and bool(torch.linalg.det(gram) < 1.0e-6)
    ):
        raise ValueError("Ordered hinge axes must be independent unit vectors.")

    rotation_xyzw = torch.nn.functional.normalize(rotation_xyzw, dim=-1)
    orthogonal = torch.allclose(gram, torch.eye(hinge_count, dtype=axes.dtype, device=axes.device), atol=1.0e-6)
    if not orthogonal:
        coordinates = _fit_independent_hinge_coordinates(rotation_xyzw, axes)
        fitted_xyzw = ordered_hinge_rotation(coordinates, axes)
        relative = quat_mul(quat_conjugate(fitted_xyzw), rotation_xyzw)
        residual = 2.0 * torch.atan2(torch.linalg.vector_norm(relative[..., :3], dim=-1), relative[..., 3].abs())
        return coordinates, residual
    vector = rotation_xyzw[..., :3]
    scalar = rotation_xyzw[..., 3]
    if hinge_count == 1:
        coordinates = (2.0 * torch.atan2(vector @ axes[0], scalar)).unsqueeze(-1)
    elif hinge_count == 2:
        first, second = axes.unbind(0)
        cross = torch.linalg.cross(first, second)
        bilinear = torch.stack((scalar, vector @ second, vector @ first, vector @ cross), dim=-1).reshape(
            *rotation_xyzw.shape[:-1], 2, 2
        )
        left, _, right_transpose = torch.linalg.svd(bilinear)
        left = left[..., :, 0]
        right = right_transpose[..., 0, :]
        coordinates = 2.0 * torch.stack(
            (torch.atan2(left[..., 1], left[..., 0]), torch.atan2(right[..., 1], right[..., 0])), dim=-1
        )
    else:
        first, second, third = axes.unbind(0)
        cross = torch.linalg.cross(first, second)
        parity = torch.dot(cross, third)
        if not torch.isclose(parity.abs(), parity.new_tensor(1.0), atol=1.0e-6):
            raise ValueError("A three-hinge chain must contain distinct cardinal directions.")
        inverse_xyzw = torch.stack(
            (
                -(vector * first).sum(dim=-1),
                -(vector * second).sum(dim=-1),
                -(vector * cross).sum(dim=-1),
                scalar,
            ),
            dim=-1,
        )
        first_angle, second_angle, third_angle = euler_xyz_from_quat(inverse_xyzw.reshape(-1, 4))
        coordinates = torch.stack((-first_angle, -second_angle, -parity * third_angle), dim=-1).reshape(
            *rotation_xyzw.shape[:-1], 3
        )
    coordinates = torch.atan2(torch.sin(coordinates), torch.cos(coordinates))

    fitted_xyzw = ordered_hinge_rotation(coordinates, axes)
    relative = quat_mul(quat_conjugate(fitted_xyzw), rotation_xyzw)
    residual = 2.0 * torch.atan2(torch.linalg.vector_norm(relative[..., :3], dim=-1), relative[..., 3].abs())
    return coordinates, residual


def kinematic_seed_target_rotations(
    tree: KinematicTree,
    topology: NewtonKinematics.Topology,
    *,
    target_body_indices: tuple[int, ...],
    target_parent_rows: tuple[int, ...],
    target_rotation_xyzw: torch.Tensor,
    joint_q: torch.Tensor,
) -> torch.Tensor:
    """Fit complete semantic paths by zero-coordinate product of exponentials.

    Fixed joint transforms define each path's zero-coordinate rotation. The
    ordered hinge axes are expressed in the reached semantic-parent frame,
    fitted together when paths share coordinates, and written as absolute
    target coordinates.

    Args:
        tree: Exact grouped rotational target topology.
        topology: Canonical Newton topology containing exact joint transforms.
        target_body_indices: Target body for each semantic row.
        target_parent_rows: Semantic parent row, with row zero equal to -1.
        target_rotation_xyzw: Desired world rotations, shape
            ``[semantic_row_count, problem_count, 4]``.
        joint_q: Generalized positions
            ``[problem_count, joint_coordinate_count]`` [m or rad, depending
            on joint type], modified in place.

    Returns:
        Reached-to-desired semantic rotation residuals [rad], shape
        ``[semantic_row_count, problem_count]``.
    """
    row_count = len(target_body_indices)
    if (
        row_count < 1
        or len(target_parent_rows) != row_count
        or target_parent_rows[0] != -1
        or target_body_indices[0] != tree.root_body_index
        or target_rotation_xyzw.shape[:1] != (row_count,)
        or target_rotation_xyzw.ndim != 3
        or target_rotation_xyzw.shape[-1] != 4
        or joint_q.ndim != 2
        or tree.num_bodies != topology.body_count
        or joint_q.shape[1] != topology.coordinate_count
        or joint_q.shape[0] != target_rotation_xyzw.shape[1]
    ):
        raise ValueError("Semantic target rotations, parents, bodies, and coordinates are incompatible.")
    if (
        target_rotation_xyzw.dtype is not torch.float32
        or joint_q.dtype is not torch.float32
        or target_rotation_xyzw.device != joint_q.device
    ):
        raise ValueError("Semantic target rotations and coordinates must share float32 storage.")
    if (
        len(set(target_body_indices)) != row_count
        or any(body < 0 or body >= tree.num_bodies for body in target_body_indices)
        or any(parent < 0 or parent >= row for row, parent in enumerate(target_parent_rows[1:], start=1))
    ):
        raise ValueError("Semantic target rows must define one topologically ordered body tree.")
    rotation_norm = torch.linalg.vector_norm(target_rotation_xyzw, dim=-1)
    if not torch.all(torch.isfinite(target_rotation_xyzw)) or not torch.allclose(
        rotation_norm, torch.ones_like(rotation_norm), atol=1.0e-5, rtol=1.0e-5
    ):
        raise ValueError("Semantic target rotations must contain finite unit quaternions.")

    joint_by_child = {body: joint for joint, body in enumerate(tree.joint_child_body_indices)}
    device = joint_q.device
    problem_count = joint_q.shape[0]
    identity = torch.zeros(problem_count, 4, dtype=torch.float32, device=device)
    identity[:, 3] = 1.0
    path_coordinate_indices_by_row: list[tuple[int, ...]] = []
    path_coordinate_rows_by_row: list[tuple[int, ...]] = []
    path_axes_by_row: list[torch.Tensor] = []
    zero_rotations_by_row: list[torch.Tensor] = []
    for row, parent_row in enumerate(target_parent_rows[1:], start=1):
        parent_body = target_body_indices[parent_row]
        child_body = target_body_indices[row]
        path = []
        body = child_body
        while body != parent_body:
            if body == tree.root_body_index:
                raise ValueError("A semantic target parent must be an ancestor of its child body.")
            path.append(body)
            body = tree.parent_indices[body]
        path.reverse()
        if not path:
            raise ValueError("A semantic target edge must contain at least one target body.")

        zero_rotation = identity.clone()
        path_coordinate_indices: list[int] = []
        path_coordinate_rows: list[int] = []
        path_axes: list[torch.Tensor] = []
        for body in path:
            tree_joint = joint_by_child[body]
            coordinate_start, coordinate_stop = tree.joint_coordinate_ranges[tree_joint]
            joint = int(topology.body_joint[body])
            if (
                int(topology.joint_parent[joint]) != tree.parent_indices[body]
                or int(topology.joint_child[joint]) != body
            ):
                raise ValueError("Canonical Newton and grouped target topology disagree on a target joint.")
            parent_frame = torch.tensor(
                topology.joint_transform_parent[joint][3:7], dtype=torch.float32, device=device
            ).expand(problem_count, 4)
            child_frame = torch.tensor(
                topology.joint_transform_child[joint][3:7], dtype=torch.float32, device=device
            ).expand(problem_count, 4)
            joint_frame = quat_mul(zero_rotation, parent_frame)
            for coordinate_row in range(coordinate_start, coordinate_stop):
                axis = torch.tensor(tree.coordinate_axes[coordinate_row], dtype=torch.float32, device=device)
                path_axes.append(quat_apply(joint_frame[:1], axis.expand(1, 3)).squeeze(0))
                path_coordinate_indices.append(tree.coordinate_q_indices[coordinate_row])
                path_coordinate_rows.append(coordinate_row)
            zero_rotation = quat_mul(quat_mul(zero_rotation, parent_frame), quat_conjugate(child_frame))

        coordinate_indices_tuple = tuple(path_coordinate_indices)
        path_coordinate_indices_by_row.append(coordinate_indices_tuple)
        path_coordinate_rows_by_row.append(tuple(path_coordinate_rows))
        path_axes_by_row.append(
            torch.stack(path_axes) if path_axes else torch.empty((0, 3), dtype=torch.float32, device=device)
        )
        zero_rotations_by_row.append(zero_rotation)

    coordinate_sets = tuple(set(path) for path in path_coordinate_indices_by_row)
    unassigned = set(range(len(coordinate_sets)))
    components: list[tuple[int, ...]] = []
    while unassigned:
        first = min(unassigned)
        unassigned.remove(first)
        component = [first]
        component_coordinates = set(coordinate_sets[first])
        while True:
            linked = {index for index in unassigned if component_coordinates.intersection(coordinate_sets[index])}
            if not linked:
                break
            unassigned.difference_update(linked)
            component.extend(sorted(linked))
            for index in linked:
                component_coordinates.update(coordinate_sets[index])
        components.append(tuple(component))

    coordinate_row_by_index = dict(zip(tree.coordinate_q_indices, range(tree.num_coordinates), strict=True))
    for component in components:
        if len(component) == 1:
            path_index = component[0]
            coordinate_indices_tuple = path_coordinate_indices_by_row[path_index]
            if not coordinate_indices_tuple:
                continue
            row = path_index + 1
            axes = path_axes_by_row[path_index]
            coordinate_indices = torch.tensor(coordinate_indices_tuple, dtype=torch.int64, device=device)
            coordinate_rows = path_coordinate_rows_by_row[path_index]
            lower = torch.tensor(
                [tree.coordinate_lower_limits_rad[index] for index in coordinate_rows],
                dtype=torch.float32,
                device=device,
            )
            upper = torch.tensor(
                [tree.coordinate_upper_limits_rad[index] for index in coordinate_rows],
                dtype=torch.float32,
                device=device,
            )
            desired_relative = quat_mul(
                quat_conjugate(target_rotation_xyzw[target_parent_rows[row]]), target_rotation_xyzw[row]
            )
            desired_hinge_rotation = quat_mul(desired_relative, quat_conjugate(zero_rotations_by_row[path_index]))
            if len(coordinate_indices_tuple) <= 3:
                coordinates, _ = fit_ordered_hinge_coordinates(desired_hinge_rotation, axes)
            else:
                coordinates = _fit_redundant_hinge_coordinates(
                    desired_hinge_rotation,
                    axes,
                    joint_q.index_select(1, coordinate_indices),
                    lower,
                    upper,
                )
            gram = axes @ axes.transpose(0, 1)
            if len(coordinate_indices_tuple) == 3 and torch.allclose(
                gram, torch.eye(3, dtype=torch.float32, device=device), atol=1.0e-6
            ):
                alternate = torch.stack(
                    (
                        coordinates[:, 0] + math.pi,
                        math.pi - coordinates[:, 1],
                        coordinates[:, 2] + math.pi,
                    ),
                    dim=-1,
                )
                alternate = torch.atan2(torch.sin(alternate), torch.cos(alternate))
                projected = coordinates.clamp(lower, upper)
                alternate_projected = alternate.clamp(lower, upper)
                projection_error = torch.square(coordinates - projected).sum(dim=-1)
                alternate_error = torch.square(alternate - alternate_projected).sum(dim=-1)
                coordinates = torch.where((alternate_error < projection_error)[:, None], alternate, coordinates)
            joint_q.index_copy_(1, coordinate_indices, coordinates)
            continue

        unique_coordinate_indices = tuple(
            dict.fromkeys(index for path_index in component for index in path_coordinate_indices_by_row[path_index])
        )
        coordinate_indices = torch.tensor(unique_coordinate_indices, dtype=torch.int64, device=device)
        position_by_coordinate = {coordinate: position for position, coordinate in enumerate(unique_coordinate_indices)}
        coordinate_positions_by_path = tuple(
            torch.tensor(
                tuple(position_by_coordinate[index] for index in path_coordinate_indices_by_row[path_index]),
                dtype=torch.int64,
                device=device,
            )
            for path_index in component
        )
        lower = torch.tensor(
            [tree.coordinate_lower_limits_rad[coordinate_row_by_index[index]] for index in unique_coordinate_indices],
            dtype=torch.float32,
            device=device,
        )
        upper = torch.tensor(
            [tree.coordinate_upper_limits_rad[coordinate_row_by_index[index]] for index in unique_coordinate_indices],
            dtype=torch.float32,
            device=device,
        )
        target_hinge_rotations = tuple(
            quat_mul(
                quat_mul(
                    quat_conjugate(target_rotation_xyzw[target_parent_rows[path_index + 1]]),
                    target_rotation_xyzw[path_index + 1],
                ),
                quat_conjugate(zero_rotations_by_row[path_index]),
            )
            for path_index in component
        )
        coordinates = _fit_coupled_hinge_coordinates(
            target_hinge_rotations,
            tuple(path_axes_by_row[path_index] for path_index in component),
            coordinate_positions_by_path,
            joint_q.index_select(1, coordinate_indices),
            lower,
            upper,
        )
        joint_q.index_copy_(1, coordinate_indices, coordinates)

    world_rotation = torch.empty(problem_count, tree.num_bodies, 4, dtype=torch.float32, device=device)
    world_rotation[:, tree.root_body_index].copy_(target_rotation_xyzw[0])
    residual = torch.zeros(row_count, problem_count, dtype=torch.float32, device=device)
    for path_index, (parent_row, coordinate_indices_tuple, axes, zero_rotation) in enumerate(
        zip(
            target_parent_rows[1:],
            path_coordinate_indices_by_row,
            path_axes_by_row,
            zero_rotations_by_row,
            strict=True,
        )
    ):
        reached_relative = zero_rotation
        if coordinate_indices_tuple:
            coordinate_indices = torch.tensor(coordinate_indices_tuple, dtype=torch.int64, device=device)
            reached_relative = quat_mul(
                ordered_hinge_rotation(joint_q.index_select(1, coordinate_indices), axes), zero_rotation
            )
        row = path_index + 1
        reached_world = quat_mul(world_rotation[:, target_body_indices[parent_row]], reached_relative)
        world_rotation[:, target_body_indices[row]] = reached_world
        error = quat_mul(quat_conjugate(reached_world), target_rotation_xyzw[row])
        residual[row] = 2.0 * torch.atan2(torch.linalg.vector_norm(error[:, :3], dim=-1), error[:, 3].abs())
    return residual
