# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared semantic motion-retarget objective, solve, projection, and residual algebra."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import torch
import warp as wp

from isaaclab.utils.math import convert_quat, matrix_from_quat, quat_apply, quat_apply_inverse, quat_from_matrix

from ..kinematics import (
    kinematic_pose_forward,
    kinematic_retarget_positions,
    kinematic_root_basis,
    kinematic_tree_forward,
)
from .data.skeleton import MotionSkeleton
from .identity import canonical_sha256

if TYPE_CHECKING:
    from ..kinematics import KinematicTree, NewtonKinematics


class MotionSemanticTarget(Protocol):
    """Exact target-robot facts consumed by semantic projection."""

    target_tree: KinematicTree
    reference_kinematics: NewtonKinematics
    construction_identity_sha256: str


@wp.kernel
def _semantic_measure_branch_jump(
    joint_q: wp.array2d(dtype=wp.float32),
    quality: wp.array2d(dtype=wp.float32),
):
    frame = wp.tid()
    jump = float(0.0)
    if frame > 0:
        for coordinate in range(7, joint_q.shape[1]):
            jump = wp.max(jump, wp.abs(joint_q[frame, coordinate] - joint_q[frame - 1, coordinate]))
    quality[frame, 2] = jump


@wp.kernel
def _semantic_project_coordinates(
    joint_q: wp.array2d(dtype=wp.float32),
    root_position: wp.array1d(dtype=wp.vec3),
    root_rotation: wp.array1d(dtype=wp.quatf),
    coordinate_indices: wp.array1d(dtype=wp.int32),
    coordinate_lower: wp.array1d(dtype=wp.float32),
    coordinate_upper: wp.array1d(dtype=wp.float32),
):
    problem, slot = wp.tid()
    if slot < 3:
        joint_q[problem, slot] = root_position[problem][slot]
    elif slot < 7:
        joint_q[problem, slot] = root_rotation[problem][slot - 3]
    else:
        coordinate = slot - 7
        index = coordinate_indices[coordinate]
        joint_q[problem, index] = wp.clamp(
            joint_q[problem, index], coordinate_lower[coordinate], coordinate_upper[coordinate]
        )


@wp.kernel
def _semantic_correct_support(
    joint_q: wp.array2d(dtype=wp.float32),
    body_q: wp.array2d(dtype=wp.transformf),
    support_body_indices: wp.array1d(dtype=wp.int32),
    source_support_height_m: wp.array1d(dtype=wp.float32),
    target_position_m: wp.array2d(dtype=wp.vec3),
):
    problem = wp.tid()
    support_height = float(1.0e6)
    for support in range(support_body_indices.shape[0]):
        position = wp.transform_get_translation(body_q[problem, support_body_indices[support]])
        support_height = wp.min(support_height, position[2])
    translation_z = source_support_height_m[problem] - support_height
    joint_q[problem, 2] = joint_q[problem, 2] + translation_z
    for landmark in range(target_position_m.shape[0]):
        position = target_position_m[landmark, problem]
        target_position_m[landmark, problem] = wp.vec3(position[0], position[1], position[2] + translation_z)


@wp.kernel
def _semantic_measure_quality(
    body_q: wp.array2d(dtype=wp.transformf),
    joint_q: wp.array2d(dtype=wp.float32),
    body_indices: wp.array1d(dtype=wp.int32),
    target_position_m: wp.array2d(dtype=wp.vec3),
    target_rotation_xyzw: wp.array2d(dtype=wp.quatf),
    support_body_indices: wp.array1d(dtype=wp.int32),
    source_support_height_m: wp.array1d(dtype=wp.float32),
    quality: wp.array2d(dtype=wp.float32),
    frame_finite: wp.array1d(dtype=wp.uint8),
    support_error_m: wp.array1d(dtype=wp.float32),
):
    problem = wp.tid()
    position_error = float(0.0)
    rotation_error = float(0.0)
    finite = wp.uint8(1)
    for coordinate in range(joint_q.shape[1]):
        if not wp.isfinite(joint_q[problem, coordinate]):
            finite = wp.uint8(0)
    for landmark in range(1, body_indices.shape[0]):
        pose = body_q[problem, body_indices[landmark]]
        position_delta = wp.transform_get_translation(pose) - target_position_m[landmark, problem]
        current_position_error = wp.length(position_delta)
        actual_rotation = wp.transform_get_rotation(pose)
        target_rotation = target_rotation_xyzw[landmark, problem]
        cosine = wp.min(wp.abs(wp.dot(actual_rotation, target_rotation)), 1.0)
        current_rotation_error = 2.0 * wp.acos(cosine)
        position_error = wp.max(position_error, current_position_error)
        rotation_error = wp.max(rotation_error, current_rotation_error)
        if not wp.isfinite(current_position_error) or not wp.isfinite(current_rotation_error):
            finite = wp.uint8(0)
    support_height = float(1.0e6)
    for support in range(support_body_indices.shape[0]):
        position = wp.transform_get_translation(body_q[problem, support_body_indices[support]])
        support_height = wp.min(support_height, position[2])
    quality[problem, 0] = position_error
    quality[problem, 1] = rotation_error
    frame_finite[problem] = finite
    support_error_m[problem] = wp.abs(support_height - source_support_height_m[problem])


def semantic_project_coordinates(
    joint_q: wp.array,
    root_position: torch.Tensor,
    root_rotation_xyzw: torch.Tensor,
    coordinate_indices: wp.array,
    coordinate_lower: wp.array,
    coordinate_upper: wp.array,
    device: str | torch.device,
) -> None:
    """Project active semantic IK rows onto root targets and hard coordinate limits."""
    wp.launch(
        _semantic_project_coordinates,
        dim=(joint_q.shape[0], 7 + coordinate_indices.shape[0]),
        inputs=(
            joint_q,
            wp.from_torch(root_position, dtype=wp.vec3),
            wp.from_torch(root_rotation_xyzw, dtype=wp.quatf),
            coordinate_indices,
            coordinate_lower,
            coordinate_upper,
        ),
        device=device,
    )


def semantic_correct_support(
    joint_q: torch.Tensor,
    body_q: wp.array,
    support_body_indices: wp.array,
    source_support_height_m: torch.Tensor,
    target_position_m: torch.Tensor,
    device: str | torch.device,
) -> None:
    """Translate active coordinates and targets to the declared support height [m]."""
    wp.launch(
        _semantic_correct_support,
        dim=joint_q.shape[0],
        inputs=(
            wp.from_torch(joint_q),
            body_q,
            support_body_indices,
            wp.from_torch(source_support_height_m),
            wp.from_torch(target_position_m, dtype=wp.vec3),
        ),
        device=device,
    )


def semantic_measure_quality(
    body_q: wp.array,
    joint_q: torch.Tensor,
    body_indices: wp.array,
    target_position_m: torch.Tensor,
    target_rotation_xyzw: torch.Tensor,
    support_body_indices: wp.array,
    source_support_height_m: torch.Tensor,
    quality: torch.Tensor,
    frame_finite: torch.Tensor,
    support_error_m: torch.Tensor,
    device: str | torch.device,
) -> None:
    """Measure semantic residual maxima and support error without residual matrices."""
    wp.launch(
        _semantic_measure_quality,
        dim=quality.shape[0],
        inputs=(
            body_q,
            wp.from_torch(joint_q),
            body_indices,
            wp.from_torch(target_position_m, dtype=wp.vec3),
            wp.from_torch(target_rotation_xyzw, dtype=wp.quatf),
            support_body_indices,
            wp.from_torch(source_support_height_m),
        ),
        outputs=(wp.from_torch(quality), wp.from_torch(frame_finite), wp.from_torch(support_error_m)),
        device=device,
    )


def semantic_measure_branch_jump(
    joint_q: torch.Tensor,
    quality: torch.Tensor,
    device: str | torch.device,
) -> None:
    """Write incoming non-root coordinate jumps into quality column two [rad]."""
    wp.launch(
        _semantic_measure_branch_jump,
        dim=joint_q.shape[0],
        inputs=(wp.from_torch(joint_q),),
        outputs=(wp.from_torch(quality),),
        device=device,
    )


@dataclass(frozen=True, slots=True)
class MotionSemanticTargets:
    """Concrete world-space landmark targets and target-coordinate bounds."""

    body_indices: tuple[int, ...]
    parent_rows: tuple[int, ...]
    body_index_tensor: torch.Tensor
    position_m: torch.Tensor
    rotation_xyzw: torch.Tensor
    segment_lengths_m: torch.Tensor
    segment_length_values_m: tuple[float, ...]
    coordinate_indices: torch.Tensor
    coordinate_lower_limits_rad: torch.Tensor
    coordinate_upper_limits_rad: torch.Tensor
    support_body_indices: torch.Tensor
    source_support_height_m: torch.Tensor


@dataclass(frozen=True, slots=True)
class MotionSemanticProjection:
    """Map any landmark-compatible source skeleton onto one exact robot target."""

    source_skeleton: MotionSkeleton
    target: MotionSemanticTarget
    target_landmarks: tuple[tuple[str, str, int], ...]
    root_basis_roles: tuple[str, ...]
    support_roles: tuple[str, ...]
    version: str
    construction_identity_sha256: str = field(init=False)
    _source_rest_translation_m: torch.Tensor = field(init=False, repr=False)
    _source_rest_rotation_xyzw: torch.Tensor = field(init=False, repr=False)
    _target_coordinate_indices: torch.Tensor = field(init=False, repr=False)
    _target_coordinate_lower_limits_rad: torch.Tensor = field(init=False, repr=False)
    _target_coordinate_upper_limits_rad: torch.Tensor = field(init=False, repr=False)
    _source_marker_rotation_indices: torch.Tensor = field(init=False, repr=False)
    _target_marker_body_indices: tuple[int, ...] = field(init=False, repr=False)
    _target_marker_indices: torch.Tensor = field(init=False, repr=False)
    _marker_parent_rows: tuple[int, ...] = field(init=False, repr=False)
    _target_marker_lengths_m: torch.Tensor = field(init=False, repr=False)
    _target_marker_length_values_m: tuple[float, ...] = field(init=False, repr=False)
    _aligned_target_rest_edges_m: torch.Tensor = field(init=False, repr=False)
    _aligned_target_rest_rotation: torch.Tensor = field(init=False, repr=False)
    _marker_rotation_correction: torch.Tensor = field(init=False, repr=False)
    _root_motion_scale: float = field(init=False, repr=False)
    _source_support_indices: torch.Tensor = field(init=False, repr=False)
    _target_support_indices: torch.Tensor = field(init=False, repr=False)
    _target_support_offsets_root_m: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve semantic roles and rest-frame geometry once per source-target pair."""
        target_tree = self.target.target_tree
        reference = self.target.reference_kinematics
        source_by_name = {name: index for index, name in enumerate(self.source_skeleton.body_names)}
        source_landmarks = {landmark.name: landmark for landmark in self.source_skeleton.landmarks}
        target_by_name = {name: index for index, name in enumerate(target_tree.body_names)}
        target_by_role = {role: body_name for role, body_name, _ in self.target_landmarks}
        required_roles = tuple(role for role, _, _ in self.target_landmarks)
        if len(set(required_roles)) != len(required_roles):
            raise ValueError("Semantic target landmark roles must be unique.")
        missing_roles = tuple(role for role in required_roles if role not in source_landmarks)
        missing_target_bodies = tuple(body for body in target_by_role.values() if body not in target_by_name)
        if missing_roles or missing_target_bodies:
            raise ValueError(
                f"Semantic projection lacks source roles {missing_roles} or target bodies {missing_target_bodies}."
            )

        source_marker_position_indices = tuple(
            source_by_name[source_landmarks[role].position_body_name] for role in required_roles
        )
        source_marker_rotation_indices = tuple(
            source_by_name[source_landmarks[role].rotation_body_name] for role in required_roles
        )
        target_marker_indices = tuple(target_by_name[target_by_role[role]] for role in required_roles)
        marker_parent_rows = tuple(parent for _, _, parent in self.target_landmarks)
        root = target_tree.root_body_index
        if target_marker_indices[0] != root or marker_parent_rows[0] != -1:
            raise ValueError("Semantic landmark row zero must own the exact target root.")

        device = torch.device(reference.device)
        source_rest_translation = torch.tensor(
            self.source_skeleton.rest_translation_m, dtype=torch.float32, device=device
        )
        source_rest_rotation = convert_quat(
            torch.tensor(self.source_skeleton.rest_rotation_wxyz, dtype=torch.float32, device=device), to="xyzw"
        )
        source_rest_position, source_rest_world_rotation = kinematic_tree_forward(
            source_rest_translation, source_rest_rotation, self.source_skeleton.parent_indices
        )
        target_rest_world = torch.tensor(reference.default_body_q, dtype=torch.float32, device=device)
        target_rest_position = target_rest_world[:, :3]
        target_rest_world_rotation = target_rest_world[:, 3:7]

        source_basis_indices = tuple(
            source_by_name[source_landmarks[role].position_body_name] for role in self.root_basis_roles
        )
        target_basis_indices = tuple(target_by_name[target_by_role[role]] for role in self.root_basis_roles)
        root_alignment = kinematic_root_basis(source_rest_position, *source_basis_indices) @ kinematic_root_basis(
            target_rest_position, *target_basis_indices
        ).transpose(-1, -2)

        source_position_indices = torch.tensor(source_marker_position_indices, dtype=torch.int64, device=device)
        source_rotation_indices = torch.tensor(source_marker_rotation_indices, dtype=torch.int64, device=device)
        target_indices = torch.tensor(target_marker_indices, dtype=torch.int64, device=device)
        source_marker_lengths = torch.zeros(len(required_roles), dtype=torch.float32, device=device)
        target_marker_lengths = torch.zeros(len(required_roles), dtype=torch.float32, device=device)
        aligned_target_rest_edges = torch.zeros(len(required_roles), 3, dtype=torch.float32, device=device)
        for row, parent_row in enumerate(marker_parent_rows[1:], start=1):
            if parent_row < 0 or parent_row >= row:
                raise ValueError("Semantic landmark parents must precede their children.")
        parent_rows = torch.tensor(marker_parent_rows[1:], dtype=torch.int64, device=device)
        source_parent_indices = source_position_indices.index_select(0, parent_rows)
        target_parent_indices = target_indices.index_select(0, parent_rows)
        source_edges = source_rest_position[source_position_indices[1:]] - source_rest_position[source_parent_indices]
        target_edges = target_rest_position[target_indices[1:]] - target_rest_position[target_parent_indices]
        source_marker_lengths[1:] = torch.linalg.vector_norm(source_edges, dim=-1)
        target_marker_lengths[1:] = torch.linalg.vector_norm(target_edges, dim=-1)
        aligned_target_rest_edges[1:] = torch.matmul(root_alignment, target_edges.unsqueeze(-1)).squeeze(-1)
        torch._assert_async(
            torch.all(
                torch.isfinite(source_marker_lengths[1:])
                & torch.isfinite(target_marker_lengths[1:])
                & (source_marker_lengths[1:] > 1.0e-6)
                & (target_marker_lengths[1:] > 1.0e-6)
            ),
            "Every semantic landmark edge must have finite nonzero source and target length.",
        )
        target_marker_lengths[0] = target_marker_lengths[1:].mean()
        target_marker_length_values = tuple(float(value) for value in target_marker_lengths.cpu().tolist())

        source_marker_rest_rotation = matrix_from_quat(
            source_rest_world_rotation.index_select(0, source_rotation_indices)
        )
        target_marker_rest_rotation = matrix_from_quat(target_rest_world_rotation.index_select(0, target_indices))
        aligned_target_rest_rotation = root_alignment @ target_marker_rest_rotation
        marker_rotation_correction = source_marker_rest_rotation.transpose(-1, -2) @ aligned_target_rest_rotation

        support_rows = tuple(required_roles.index(role) for role in self.support_roles)
        source_support_indices = source_position_indices[list(support_rows)]
        target_support_indices = target_indices[list(support_rows)]
        source_support_length = source_marker_lengths.new_zeros(())
        target_support_length = target_marker_lengths.new_zeros(())
        for support_row in support_rows:
            row = support_row
            while row != 0:
                source_support_length.add_(source_marker_lengths[row])
                target_support_length.add_(target_marker_lengths[row])
                row = marker_parent_rows[row]
        root_motion_scale = float((target_support_length / source_support_length).cpu())
        target_support_offsets_root = quat_apply_inverse(
            target_rest_world_rotation[root].expand(len(support_rows), 4),
            target_rest_position.index_select(0, target_support_indices) - target_rest_position[root],
        )

        object.__setattr__(self, "_source_rest_translation_m", source_rest_translation)
        object.__setattr__(self, "_source_rest_rotation_xyzw", source_rest_rotation)
        object.__setattr__(
            self,
            "_target_coordinate_indices",
            torch.tensor(target_tree.coordinate_q_indices, dtype=torch.int64, device=device),
        )
        object.__setattr__(
            self,
            "_target_coordinate_lower_limits_rad",
            torch.tensor(target_tree.coordinate_lower_limits_rad, dtype=torch.float32, device=device),
        )
        object.__setattr__(
            self,
            "_target_coordinate_upper_limits_rad",
            torch.tensor(target_tree.coordinate_upper_limits_rad, dtype=torch.float32, device=device),
        )
        object.__setattr__(self, "_source_marker_rotation_indices", source_rotation_indices)
        object.__setattr__(self, "_target_marker_body_indices", target_marker_indices)
        object.__setattr__(self, "_target_marker_indices", target_indices)
        object.__setattr__(self, "_marker_parent_rows", marker_parent_rows)
        object.__setattr__(self, "_target_marker_lengths_m", target_marker_lengths)
        object.__setattr__(self, "_target_marker_length_values_m", target_marker_length_values)
        object.__setattr__(self, "_aligned_target_rest_edges_m", aligned_target_rest_edges)
        object.__setattr__(self, "_aligned_target_rest_rotation", aligned_target_rest_rotation)
        object.__setattr__(self, "_marker_rotation_correction", marker_rotation_correction)
        object.__setattr__(self, "_root_motion_scale", root_motion_scale)
        object.__setattr__(self, "_source_support_indices", source_support_indices)
        object.__setattr__(self, "_target_support_indices", target_support_indices)
        object.__setattr__(self, "_target_support_offsets_root_m", target_support_offsets_root)
        object.__setattr__(
            self,
            "construction_identity_sha256",
            canonical_sha256(
                {
                    "math_version": self.version,
                    "source_skeleton_sha256": self.source_skeleton.identity_sha256,
                    "target_construction_sha256": self.target.construction_identity_sha256,
                    "target_landmarks": self.target_landmarks,
                    "root_basis_roles": self.root_basis_roles,
                    "support_roles": self.support_roles,
                    "orientation_law": "R_source(t) @ R_source(rest).T @ A_root @ R_target(rest)",
                    "edge_law": (
                        "R_target_parent(t) @ (A_root @ R_target_parent(rest)).T @ A_root @ target_rest_parent_edge"
                    ),
                    "root_xy_law": "source_root_xy(0) + target/source support-chain scale * source displacement",
                    "root_motion_scale": root_motion_scale,
                    "input_representation": "world_root_and_parent_local_pose_delta_xyzw_v1",
                }
            ),
        )

    def generate_targets(
        self,
        source_root_position: torch.Tensor,
        source_body_rotation_xyzw: torch.Tensor,
    ) -> MotionSemanticTargets:
        """Generate concrete target-robot landmark tensors without solving."""
        if (
            source_root_position.device != self._source_rest_translation_m.device
            or source_body_rotation_xyzw.device != source_root_position.device
            or source_root_position.dtype is not torch.float32
            or source_body_rotation_xyzw.dtype is not torch.float32
        ):
            raise ValueError("Semantic source tensors must use the target kinematics device and float32.")
        source_position, source_rotation = kinematic_pose_forward(
            self._source_rest_translation_m,
            self._source_rest_rotation_xyzw,
            source_body_rotation_xyzw,
            source_root_position,
            self.source_skeleton.parent_indices,
        )
        source_marker_rotation = source_rotation.index_select(1, self._source_marker_rotation_indices)
        target_marker_rotation = (
            quat_from_matrix(matrix_from_quat(source_marker_rotation) @ self._marker_rotation_correction[None])
            .transpose(0, 1)
            .contiguous()
        )
        target_root_rotation = target_marker_rotation[0]
        source_support_height = source_position.index_select(1, self._source_support_indices)[..., 2].amin(dim=1)
        target_support_offset_world = quat_apply(
            target_root_rotation[:, None, :], self._target_support_offsets_root_m[None]
        )
        target_root_position = source_root_position[:1] + self._root_motion_scale * (
            source_root_position - source_root_position[:1]
        )
        target_root_position[:, 2] = source_support_height - target_support_offset_world[..., 2].amin(dim=1)
        target_marker_position = kinematic_retarget_positions(
            target_root_position,
            matrix_from_quat(target_marker_rotation),
            self._aligned_target_rest_rotation,
            self._aligned_target_rest_edges_m,
            self._marker_parent_rows,
        )
        return MotionSemanticTargets(
            body_indices=self._target_marker_body_indices,
            parent_rows=self._marker_parent_rows,
            body_index_tensor=self._target_marker_indices,
            position_m=target_marker_position,
            rotation_xyzw=target_marker_rotation,
            segment_lengths_m=self._target_marker_lengths_m,
            segment_length_values_m=self._target_marker_length_values_m,
            coordinate_indices=self._target_coordinate_indices,
            coordinate_lower_limits_rad=self._target_coordinate_lower_limits_rad,
            coordinate_upper_limits_rad=self._target_coordinate_upper_limits_rad,
            support_body_indices=self._target_support_indices,
            source_support_height_m=source_support_height,
        )
