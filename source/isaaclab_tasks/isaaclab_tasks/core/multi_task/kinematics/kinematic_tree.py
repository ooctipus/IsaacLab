# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinematic-tree topology, ordered-hinge projection, and trajectory operators."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from isaaclab.utils.math import (
    axis_angle_from_quat,
    convert_quat,
    euler_xyz_from_quat,
    quat_conjugate,
    quat_from_rotation_vector,
    quat_mul,
)

if TYPE_CHECKING:
    from .newton_kinematics import NewtonKinematics

ORDERED_HINGE_OPERATOR_VERSION = "ordered_orthogonal_hinge_fit_v2"


@dataclass(frozen=True, slots=True)
class KinematicTree:
    """Ordered scalar-hinge topology derived from a kinematic model."""

    body_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    parent_indices: tuple[int, ...]
    joint_child_body_indices: tuple[int, ...]
    joint_axes: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        """Validate names, topology, and unit hinge axes."""
        if not self.body_names or len(set(self.body_names)) != len(self.body_names):
            raise ValueError("Kinematic-tree body names must be nonempty and unique.")
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Kinematic-tree joint names must be nonempty and unique.")
        if len(self.parent_indices) != len(self.body_names):
            raise ValueError("Kinematic-tree parents must contain one entry per body.")
        roots = tuple(index for index, parent in enumerate(self.parent_indices) if parent == -1)
        if len(roots) != 1:
            raise ValueError("A kinematic tree must contain exactly one root body.")
        if any(
            parent < -1 or parent >= len(self.body_names) or parent == body
            for body, parent in enumerate(self.parent_indices)
        ):
            raise ValueError("Kinematic-tree parent indices are invalid.")
        if len(self.joint_child_body_indices) != len(self.joint_names):
            raise ValueError("Kinematic-tree child indices must contain one entry per joint.")
        if any(child < 0 or child >= len(self.body_names) for child in self.joint_child_body_indices):
            raise ValueError("Kinematic-tree joint child indices are invalid.")
        if len(self.joint_axes) != len(self.joint_names):
            raise ValueError("Kinematic-tree axes must contain one entry per joint.")
        if any(
            len(axis) != 3
            or any(not math.isfinite(component) for component in axis)
            or not math.isclose(sum(component * component for component in axis), 1.0, rel_tol=1.0e-5, abs_tol=1.0e-5)
            for axis in self.joint_axes
        ):
            raise ValueError("Kinematic-tree hinge axes must be finite unit vectors.")

    @classmethod
    def from_newton(cls, kinematics: NewtonKinematics) -> KinematicTree:
        """Derive one scalar-hinge tree from finalized Newton metadata.

        Args:
            kinematics: Finalized Newton kinematic model.

        Returns:
            The ordered body and scalar-hinge topology.

        Raises:
            ValueError: If any non-root joint is not a scalar hinge.
        """
        model = kinematics.model
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_axis = model.joint_axis.numpy()
        joint_indices = tuple(range(1, model.joint_count))
        if any(int(joint_qd_start[index + 1] - joint_qd_start[index]) != 1 for index in joint_indices):
            raise ValueError("KinematicTree requires scalar non-root hinge joints.")
        children = tuple(int(joint_child[index]) for index in joint_indices)
        parents = [-1] * model.body_count
        for index in range(model.joint_count):
            child = int(joint_child[index])
            if child >= 0:
                parents[child] = int(joint_parent[index])
        axes = tuple(
            tuple(float(component) for component in joint_axis[int(joint_qd_start[index])]) for index in joint_indices
        )
        return cls(
            body_names=tuple(kinematics.body_names),
            joint_names=tuple(kinematics.joint_names[1:]),
            parent_indices=tuple(parents),
            joint_child_body_indices=children,
            joint_axes=axes,
        )

    @property
    def num_bodies(self) -> int:
        """Number of bodies."""
        return len(self.body_names)

    @property
    def num_joints(self) -> int:
        """Number of scalar hinge coordinates."""
        return len(self.joint_names)

    @property
    def root_body_index(self) -> int:
        """Index of the unique root body."""
        return self.parent_indices.index(-1)


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
    difference = torch.zeros_like(rotation_xyzw)
    difference[..., 3] = 1.0
    relative = quat_mul(rotation_xyzw[:, 1:], quat_conjugate(rotation_xyzw[:, :-1]))
    difference[:, :-1] = relative / relative.norm(p=2, dim=-1, keepdim=True).clamp_min(1.0e-9)
    scalar = difference[..., 3]
    angle = torch.acos((2.0 * scalar**2 - 1.0).clamp(-1.0, 1.0))
    axis = difference[..., :3]
    axis = axis / axis.norm(p=2, dim=-1, keepdim=True).clamp_min(1.0e-9)
    return axis * angle[..., None] / step_seconds


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


@dataclass(frozen=True, slots=True)
class KinematicTreeRotationProjection:
    """Project source-body local rotations into a target scalar-hinge tree."""

    source_body_count: int
    target_tree: KinematicTree
    target_joint_source_body_indices: tuple[int, ...]
    device: str | torch.device
    source_root_body_index: int = 0
    _joint_groups: tuple[tuple[int, int, int], ...] = field(init=False, repr=False)
    _target_child_indices: torch.Tensor = field(init=False, repr=False)
    _target_axes: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate source readers and preallocate fixed target maps."""
        if self.source_body_count < 1:
            raise ValueError("A rotation projection requires at least one source body.")
        if len(self.target_joint_source_body_indices) != self.target_tree.num_joints:
            raise ValueError("The source-body map must contain one row per target joint.")
        source_indices = (self.source_root_body_index, *self.target_joint_source_body_indices)
        if any(index < 0 or index >= self.source_body_count for index in source_indices):
            raise ValueError("The rotation projection contains an out-of-range source body.")

        groups: list[tuple[int, int, int]] = []
        start = 0
        while start < len(self.target_joint_source_body_indices):
            source_body_index = self.target_joint_source_body_indices[start]
            stop = start + 1
            while (
                stop < len(self.target_joint_source_body_indices)
                and self.target_joint_source_body_indices[stop] == source_body_index
            ):
                stop += 1
            groups.append((start, stop, source_body_index))
            start = stop
        if len({source for _, _, source in groups}) != len(groups):
            raise ValueError("Each source body must own one contiguous target hinge chain.")

        children = self.target_tree.joint_child_body_indices
        for start, stop, _ in groups:
            if stop - start > 3:
                raise ValueError("A source body may drive at most three ordered target hinges.")
            for joint_index in range(start + 1, stop):
                if self.target_tree.parent_indices[children[joint_index]] != children[joint_index - 1]:
                    raise ValueError("Repeated source mappings must form one serial target chain.")
        object.__setattr__(self, "_joint_groups", tuple(groups))
        object.__setattr__(self, "_target_child_indices", torch.tensor(children, dtype=torch.int64, device=self.device))
        object.__setattr__(
            self,
            "_target_axes",
            torch.tensor(self.target_tree.joint_axes, dtype=torch.float32, device=self.device),
        )

    @property
    def joint_groups(self) -> tuple[tuple[int, int, int], ...]:
        """Contiguous target-joint ranges owned by each source body."""
        return self._joint_groups

    def project(self, local_rotation_wxyz: torch.Tensor) -> torch.Tensor:
        """Project source local rotations into target-body rotation vectors.

        Args:
            local_rotation_wxyz: Source-body local quaternions in wxyz order.

        Returns:
            Target-body rotation vectors [rad], shape ``[frame_count, target_body_count, 3]``.
        """
        frame_count = local_rotation_wxyz.shape[0]
        expected = (frame_count, self.source_body_count, 4)
        if local_rotation_wxyz.shape != expected or local_rotation_wxyz.dtype is not torch.float32:
            raise ValueError(f"Local rotations must be float32 with shape {expected}.")
        if self._target_axes.device != local_rotation_wxyz.device:
            raise ValueError("Local rotations and projection tensors must share one device.")

        local_rotation_xyzw = convert_quat(local_rotation_wxyz, to="xyzw")
        pose_axis_angle = torch.zeros(
            frame_count,
            self.target_tree.num_bodies,
            3,
            dtype=torch.float32,
            device=local_rotation_wxyz.device,
        )
        pose_axis_angle[:, self.target_tree.root_body_index].copy_(
            axis_angle_from_quat(local_rotation_xyzw[:, self.source_root_body_index])
        )
        for start, stop, source_body_index in self._joint_groups:
            coordinates, _ = fit_ordered_hinge_coordinates(
                local_rotation_xyzw[:, source_body_index], self._target_axes[start:stop]
            )
            coordinates = time_unwrap_angles(coordinates)
            children = self._target_child_indices[start:stop]
            pose_axis_angle[:, children] = coordinates[..., None] * self._target_axes[start:stop]
        return pose_axis_angle
