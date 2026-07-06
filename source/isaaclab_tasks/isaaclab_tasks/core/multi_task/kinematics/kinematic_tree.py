# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinematic-tree topology, ordered-hinge projection, and trajectory operators."""

from __future__ import annotations

import math
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

ORDERED_HINGE_OPERATOR_VERSION = "ordered_orthogonal_hinge_fit_v2"


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


def kinematic_retarget_positions(
    root_position_m: torch.Tensor,
    target_rotation: torch.Tensor,
    aligned_target_rest_rotation: torch.Tensor,
    aligned_target_rest_edges_m: torch.Tensor,
    parent_indices: tuple[int, ...],
) -> torch.Tensor:
    """Transfer target-parent rotation deltas onto exact target-rest marker edges.

    Args:
        root_position_m: Desired root positions [m], shape ``[frame_count, 3]``.
        target_rotation: Desired marker rotation matrices, shape ``[landmark_count, frame_count, 3, 3]``.
        aligned_target_rest_rotation: Target-rest rotations aligned into the source root basis, shape
            ``[landmark_count, 3, 3]``.
        aligned_target_rest_edges_m: Target-rest parent edges aligned into the source root basis [m], shape
            ``[landmark_count, 3]``.
        parent_indices: Parent row of each semantic landmark, with root row zero equal to ``-1``.

    Returns:
        Desired world marker positions [m], shape ``[landmark_count, frame_count, 3]``.
    """
    landmark_count, frame_count = target_rotation.shape[:2]
    expected = (landmark_count, frame_count, 3, 3)
    if (
        target_rotation.shape != expected
        or aligned_target_rest_rotation.shape != (landmark_count, 3, 3)
        or aligned_target_rest_edges_m.shape != (landmark_count, 3)
        or root_position_m.shape != (frame_count, 3)
        or len(parent_indices) != landmark_count
        or parent_indices[0] != -1
        or any(parent < 0 or parent >= row for row, parent in enumerate(parent_indices[1:], start=1))
    ):
        raise ValueError("Semantic retarget position inputs have incompatible shapes or topology.")
    position_m = torch.empty(landmark_count, frame_count, 3, dtype=root_position_m.dtype, device=root_position_m.device)
    position_m[0].copy_(root_position_m)
    for row, parent in enumerate(parent_indices[1:], start=1):
        parent_rotation_delta = target_rotation[parent] @ aligned_target_rest_rotation[parent].transpose(-1, -2)
        position_m[row] = position_m[parent] + parent_rotation_delta @ aligned_target_rest_edges_m[row]
    return position_m


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
) -> torch.Tensor:
    """Differentiate a flat segmented time axis without crossing segment boundaries.

    Args:
        values: Values with shape ``[frame_count, ...]``.
        offsets: Segment offsets with shape ``[segment_count + 1]``.
        step_seconds: Uniform sample interval within each segment [s], shape ``[segment_count]``.

    Returns:
        Time derivative with the same shape as :paramref:`values`.
    """
    rows, starts, stops, steps = _time_segment_rows(values, offsets, step_seconds, minimum_frames=2)
    previous = torch.maximum(rows - 1, starts)
    following = torch.minimum(rows + 1, stops - 1)
    denominator = (following - previous).to(step_seconds.dtype) * steps
    while denominator.ndim < values.ndim:
        denominator = denominator.unsqueeze(-1)
    return (values.index_select(0, following) - values.index_select(0, previous)) / denominator


def time_quaternion_angular_velocity_segmented(
    rotation_xyzw: torch.Tensor,
    offsets: torch.Tensor,
    step_seconds: torch.Tensor,
) -> torch.Tensor:
    """Differentiate flat segmented unit quaternions with a zero segment tail.

    Args:
        rotation_xyzw: Unit quaternions in xyzw order with shape ``[frame_count, ..., 4]``.
        offsets: Segment offsets with shape ``[segment_count + 1]``.
        step_seconds: Uniform sample interval within each segment [s], shape ``[segment_count]``.

    Returns:
        Angular velocities [rad/s] with the same leading shape and a final xyz axis.
    """
    if rotation_xyzw.ndim < 2 or rotation_xyzw.shape[-1] != 4:
        raise ValueError("Segmented quaternion velocities require values ending in four.")
    rows, _starts, stops, steps = _time_segment_rows(rotation_xyzw, offsets, step_seconds, minimum_frames=1)
    following = torch.minimum(rows + 1, stops - 1)
    relative = quat_mul(rotation_xyzw.index_select(0, following), quat_conjugate(rotation_xyzw))
    relative = relative / relative.norm(p=2, dim=-1, keepdim=True).clamp_min(1.0e-9)
    angular_velocity = axis_angle_from_quat(relative)
    while steps.ndim < angular_velocity.ndim:
        steps = steps.unsqueeze(-1)
    angular_velocity = angular_velocity / steps
    angular_velocity[rows == stops - 1] = 0.0
    return angular_velocity


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


def time_forward_difference_segmented(
    values: torch.Tensor,
    offsets: torch.Tensor,
    step_seconds: torch.Tensor,
) -> torch.Tensor:
    """Apply the released G1 forward difference independently within each segment.

    The final row repeats the segment's penultimate forward difference, matching
    the released three-frame-minimum construction law.

    Args:
        values: Values with shape ``[frame_count, ...]``.
        offsets: Segment offsets with shape ``[segment_count + 1]``.
        step_seconds: Uniform sample interval within each segment [s], shape ``[segment_count]``.

    Returns:
        Time derivative with the same shape as :paramref:`values`.
    """
    rows, _starts, stops, steps = _time_segment_rows(values, offsets, step_seconds, minimum_frames=3)
    tail = rows == stops - 1
    previous = torch.where(tail, stops - 3, rows)
    following = torch.where(tail, stops - 2, rows + 1)
    while steps.ndim < values.ndim:
        steps = steps.unsqueeze(-1)
    return (values.index_select(0, following) - values.index_select(0, previous)) / steps


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


def fit_ordered_hinge_coordinates(
    rotation_xyzw: torch.Tensor,
    axes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one local rotation to one, two, or three ordered orthogonal hinges.

    Args:
        rotation_xyzw: Unit quaternions in xyzw order.
        axes: One to three ordered orthonormal hinge axes.

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
    if not torch.allclose(gram, torch.eye(hinge_count, dtype=axes.dtype, device=axes.device), atol=1.0e-6):
        raise ValueError("Ordered hinge axes must be mutually orthonormal.")

    rotation_xyzw = torch.nn.functional.normalize(rotation_xyzw, dim=-1)
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
) -> None:
    """Seed direct target joints from desired semantic world rotations.

    A row is seeded only when its target body is connected to its declared
    semantic parent by one rotational joint represented by one to three
    ordered hinges. Rows spanning fixed or multiple joints remain at the
    caller-provided default coordinates for the shared IK solver to resolve.

    Args:
        tree: Exact grouped rotational target topology.
        topology: Canonical Newton topology containing exact joint transforms.
        target_body_indices: Target body for each semantic row.
        target_parent_rows: Semantic parent row, with row zero equal to -1.
        target_rotation_xyzw: Desired world rotations, shape
            [semantic_row_count, problem_count, 4].
        joint_q: Initial generalized positions
            [problem_count, joint_coordinate_count] [m or rad, depending on
            joint type], modified in place.
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
        raise ValueError("Semantic target rotations, parents, bodies, and initial coordinates are incompatible.")
    if (
        target_rotation_xyzw.dtype is not torch.float32
        or joint_q.dtype is not torch.float32
        or target_rotation_xyzw.device != joint_q.device
    ):
        raise ValueError("Semantic target rotations and initial coordinates must share float32 storage.")
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

    row_by_body = {body: row for row, body in enumerate(target_body_indices)}
    joint_by_child = {body: joint for joint, body in enumerate(tree.joint_child_body_indices)}
    device = joint_q.device
    problem_count = joint_q.shape[0]
    world_rotation = torch.empty(problem_count, tree.num_bodies, 4, dtype=torch.float32, device=device)
    world_rotation[:, tree.root_body_index].copy_(target_rotation_xyzw[0])

    for body, parent_body in enumerate(tree.parent_indices):
        if parent_body == -1:
            continue
        tree_joint = joint_by_child[body]
        coordinate_start, coordinate_stop = tree.joint_coordinate_ranges[tree_joint]
        coordinate_count = coordinate_stop - coordinate_start
        joint = int(topology.body_joint[body])
        if int(topology.joint_parent[joint]) != parent_body or int(topology.joint_child[joint]) != body:
            raise ValueError("Canonical Newton and grouped target topology disagree on a target joint.")
        parent_frame = torch.tensor(
            topology.joint_transform_parent[joint][3:7], dtype=torch.float32, device=device
        ).expand(problem_count, 4)
        child_frame = torch.tensor(
            topology.joint_transform_child[joint][3:7], dtype=torch.float32, device=device
        ).expand(problem_count, 4)

        if coordinate_count:
            coordinate_indices = torch.tensor(
                tree.coordinate_q_indices[coordinate_start:coordinate_stop], dtype=torch.int64, device=device
            )
            axes = torch.tensor(
                tree.coordinate_axes[coordinate_start:coordinate_stop], dtype=torch.float32, device=device
            )
            coordinates = joint_q.index_select(1, coordinate_indices)
            row = row_by_body.get(body)
            if row is not None and target_body_indices[target_parent_rows[row]] == parent_body:
                lower = torch.tensor(
                    tree.coordinate_lower_limits_rad[coordinate_start:coordinate_stop],
                    dtype=torch.float32,
                    device=device,
                )
                upper = torch.tensor(
                    tree.coordinate_upper_limits_rad[coordinate_start:coordinate_stop],
                    dtype=torch.float32,
                    device=device,
                )
                desired_joint = quat_mul(
                    quat_mul(
                        quat_mul(quat_conjugate(parent_frame), quat_conjugate(world_rotation[:, parent_body])),
                        target_rotation_xyzw[row],
                    ),
                    child_frame,
                )
                coordinates, _ = fit_ordered_hinge_coordinates(desired_joint, axes)
                if coordinate_count == 3:
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
                coordinates = coordinates.clamp(lower, upper)
                joint_q[:, coordinate_indices] = coordinates
            hinge_rotation = ordered_hinge_rotation(coordinates, axes)
        else:
            hinge_rotation = torch.zeros(problem_count, 4, dtype=torch.float32, device=device)
            hinge_rotation[:, 3] = 1.0

        world_rotation[:, body] = quat_mul(
            quat_mul(
                quat_mul(world_rotation[:, parent_body], parent_frame),
                hinge_rotation,
            ),
            quat_conjugate(child_frame),
        )
