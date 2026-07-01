# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic SMPL-to-G1 trajectory projection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.utils.math import (
    axis_angle_from_quat,
    convert_quat,
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_conjugate,
    quat_from_matrix,
    quat_from_rotation_vector,
    quat_mul,
)

from ..data import MotionSkeleton
from ..data._identity import canonical_sha256
from ..mdp.commands.motion_task_table import MotionTaskTable
from .g1 import G1LafanFrameBuilder

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_HUMENV_FIELDS = ("motion_id", "observation", "qpos", "qvel", "terminated", "truncated")
_XYZ_AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_G1_SOURCE_BODY_NAMES = (
    "L_Hip",
    "L_Hip",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Ankle",
    "R_Hip",
    "R_Hip",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Ankle",
    "Torso",
    "Torso",
    "Torso",
    "L_Shoulder",
    "L_Shoulder",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Wrist",
    "L_Wrist",
    "R_Shoulder",
    "R_Shoulder",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Wrist",
    "R_Wrist",
)


def _compose_ordered_hinges(coordinates: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
    """Compose local hinge rotations in declared parent-to-child order."""
    quaternion = torch.zeros((*coordinates.shape[:-1], 4), dtype=coordinates.dtype, device=coordinates.device)
    quaternion[..., 3] = 1.0
    for index in range(coordinates.shape[-1]):
        quaternion = quat_mul(
            quaternion,
            quat_from_rotation_vector(coordinates[..., index, None] * axes[index]),
        )
    return quaternion


def _unwrap_angle_time(coordinates: torch.Tensor) -> torch.Tensor:
    """Choose the continuous time representative of principal hinge angles."""
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
    """Fit one local rotation to one, two, or three ordered orthogonal hinges."""
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
        bilinear = torch.stack(
            (scalar, vector @ second, vector @ first, vector @ cross),
            dim=-1,
        ).reshape(*rotation_xyzw.shape[:-1], 2, 2)
        left, _, right_transpose = torch.linalg.svd(bilinear)
        left = left[..., :, 0]
        right = right_transpose[..., 0, :]
        coordinates = 2.0 * torch.stack(
            (
                torch.atan2(left[..., 1], left[..., 0]),
                torch.atan2(right[..., 1], right[..., 0]),
            ),
            dim=-1,
        )
    else:
        first, second, third = axes.unbind(0)
        cross = torch.linalg.cross(first, second)
        parity = torch.dot(cross, third)
        if not torch.isclose(parity.abs(), parity.new_tensor(1.0), atol=1.0e-6):
            raise ValueError("A three-hinge chain must contain distinct cardinal directions.")
        basis = torch.stack((first, second, cross), dim=-1)
        matrix = matrix_from_quat(rotation_xyzw)
        canonical_inverse = torch.matmul(
            torch.matmul(basis.transpose(0, 1), matrix),
            basis,
        ).transpose(-1, -2)
        inverse_xyzw = quat_from_matrix(canonical_inverse)
        first_angle, second_angle, third_angle = euler_xyz_from_quat(inverse_xyzw.reshape(-1, 4))
        coordinates = torch.stack(
            (-first_angle, -second_angle, -parity * third_angle),
            dim=-1,
        ).reshape(*rotation_xyzw.shape[:-1], 3)
    coordinates = torch.atan2(torch.sin(coordinates), torch.cos(coordinates))

    fitted_xyzw = _compose_ordered_hinges(coordinates, axes)
    relative = quat_mul(quat_conjugate(fitted_xyzw), rotation_xyzw)
    residual = 2.0 * torch.atan2(torch.linalg.vector_norm(relative[..., :3], dim=-1), relative[..., 3].abs())
    return coordinates, residual


def _validate_smpl_xyz_chains(skeleton: MotionSkeleton) -> None:
    """Require one declared XYZ hinge chain for every non-root SMPL body."""
    expected_children = tuple(body for body in range(1, skeleton.num_bodies) for _ in range(3))
    expected_axes = _XYZ_AXES * (skeleton.num_bodies - 1)
    if skeleton.joint_child_body_indices != expected_children or skeleton.joint_axes != expected_axes:
        raise ValueError("HumEnv SMPL coordinates must declare one ordered XYZ hinge chain per body.")
    if skeleton.root_rotation_convention != "wxyz":
        raise ValueError("HumEnv SMPL root rotations must use the declared wxyz convention.")


def smpl_humenv_local_rotation_wxyz(
    qpos: torch.Tensor,
    source_skeleton: MotionSkeleton,
) -> torch.Tensor:
    """Reconstruct SMPL parent-local rotations from HumEnv XYZ hinge coordinates."""
    _validate_smpl_xyz_chains(source_skeleton)
    expected_shape = (qpos.shape[0], 7 + source_skeleton.num_joints)
    if qpos.ndim != 2 or qpos.shape != expected_shape or qpos.dtype is not torch.float32:
        raise ValueError(f"HumEnv qpos must be float32 with shape {expected_shape}.")
    if not torch.all(torch.isfinite(qpos)):
        raise ValueError("HumEnv qpos must contain only finite values.")

    local_xyzw = torch.zeros(
        qpos.shape[0],
        source_skeleton.num_bodies,
        4,
        dtype=torch.float32,
        device=qpos.device,
    )
    local_xyzw[..., 3] = 1.0
    local_xyzw[:, 0].copy_(convert_quat(qpos[:, 3:7], to="xyzw"))
    axes = qpos.new_tensor(source_skeleton.joint_axes)
    coordinate_rotation = quat_from_rotation_vector(qpos[:, 7:, None] * axes[None])
    for body_index in range(1, source_skeleton.num_bodies):
        start = 3 * (body_index - 1)
        rotation = quat_mul(coordinate_rotation[:, start], coordinate_rotation[:, start + 1])
        local_xyzw[:, body_index].copy_(quat_mul(rotation, coordinate_rotation[:, start + 2]))
    return convert_quat(local_xyzw, to="wxyz")


@dataclass(frozen=True, slots=True)
class G1HumanFrameProjection:
    """Project declared human local rotations into exact target-G1 hinge chains."""

    source_skeleton: MotionSkeleton
    target_builder: G1LafanFrameBuilder
    target_joint_source_body_indices: tuple[int, ...]
    source_root_body_index: int = 0
    construction_identity_sha256: str = field(init=False)
    _joint_groups: tuple[tuple[int, int, int], ...] = field(init=False, repr=False)
    _target_child_indices: torch.Tensor = field(init=False, repr=False)
    _target_axes: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate explicit source-body readers and preallocate fixed target maps."""
        target_skeleton = self.target_builder.source_skeleton
        if len(self.target_joint_source_body_indices) != target_skeleton.num_joints:
            raise ValueError("The G1 source-body map must contain one row per target joint.")
        source_indices = (self.source_root_body_index, *self.target_joint_source_body_indices)
        if any(index < 0 or index >= self.source_skeleton.num_bodies for index in source_indices):
            raise ValueError("The G1 projection contains an out-of-range source body.")

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

        children = target_skeleton.joint_child_body_indices
        axes = target_skeleton.joint_axes
        for start, stop, _ in groups:
            if stop - start > 3:
                raise ValueError("A source body may drive at most three ordered target hinges.")
            for joint_index in range(start + 1, stop):
                if target_skeleton.parent_indices[children[joint_index]] != children[joint_index - 1]:
                    raise ValueError("Repeated source mappings must form one serial target chain.")
        device = self.target_builder.reference_kinematics.device
        object.__setattr__(self, "_joint_groups", tuple(groups))
        object.__setattr__(
            self,
            "_target_child_indices",
            torch.tensor(children, dtype=torch.int64, device=device),
        )
        object.__setattr__(
            self,
            "_target_axes",
            torch.tensor(axes, dtype=torch.float32, device=device),
        )
        object.__setattr__(
            self,
            "construction_identity_sha256",
            canonical_sha256(
                {
                    "policy": "ordered_orthogonal_hinge_fit_v1",
                    "source_skeleton_sha256": self.source_skeleton.identity_sha256,
                    "target_builder_sha256": self.target_builder.construction_identity_sha256,
                    "source_root_body_index": self.source_root_body_index,
                    "target_joint_source_body_indices": self.target_joint_source_body_indices,
                    "joint_groups": groups,
                }
            ),
        )

    def project_local_rotations(self, local_rotation_wxyz: torch.Tensor) -> torch.Tensor:
        """Fit source-body rotations to the target ordered hinge chains."""
        frame_count = local_rotation_wxyz.shape[0]
        expected = (frame_count, self.source_skeleton.num_bodies, 4)
        if local_rotation_wxyz.shape != expected or local_rotation_wxyz.dtype is not torch.float32:
            raise ValueError(f"Human local rotations must be float32 with shape {expected}.")
        if self._target_axes.device != local_rotation_wxyz.device:
            raise ValueError("Human trajectory tensors must use the target reference-kinematics device.")

        local_rotation_xyzw = convert_quat(local_rotation_wxyz, to="xyzw")
        target_skeleton = self.target_builder.source_skeleton
        pose_axis_angle = torch.zeros(
            frame_count,
            target_skeleton.num_bodies,
            3,
            dtype=torch.float32,
            device=local_rotation_wxyz.device,
        )
        pose_axis_angle[:, 0].copy_(axis_angle_from_quat(local_rotation_xyzw[:, self.source_root_body_index]))
        for start, stop, source_body_index in self._joint_groups:
            coordinates, _ = fit_ordered_hinge_coordinates(
                local_rotation_xyzw[:, source_body_index],
                self._target_axes[start:stop],
            )
            coordinates = _unwrap_angle_time(coordinates)
            children = self._target_child_indices[start:stop]
            pose_axis_angle[:, children] = coordinates[..., None] * self._target_axes[start:stop]
        return pose_axis_angle

    def project_frames(
        self,
        root_translation: torch.Tensor,
        local_rotation_wxyz: torch.Tensor,
        source_fps: float,
    ) -> MotionTaskTable.Frames:
        """Project declared local rotations and build exact target-G1 frames."""
        if root_translation.dtype is not torch.float32 or root_translation.shape != (local_rotation_wxyz.shape[0], 3):
            raise ValueError("Human root translations must be float32 [frame_count, 3].")
        if root_translation.device != local_rotation_wxyz.device:
            raise ValueError("Human pose tensors must share one device.")
        pose_axis_angle = self.project_local_rotations(local_rotation_wxyz)
        return self.target_builder.build_pose_frames(pose_axis_angle, root_translation, source_fps)


@dataclass(frozen=True, slots=True)
class G1SmplHumEnvFrameBuilder:
    """Reconstruct HumEnv SMPL rotations and project them into G1 frames."""

    source_skeleton: MotionSkeleton
    projection: G1HumanFrameProjection
    source_fps: float = 30.0
    version: str = "g1_smpl_humenv_ordered_hinge_fit_v1"
    construction_identity_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate source identity and freeze the complete cross-robot policy."""
        if self.projection.source_skeleton.identity_sha256 != self.source_skeleton.identity_sha256:
            raise ValueError("The SMPL projection and frame-builder source skeletons differ.")
        _validate_smpl_xyz_chains(self.source_skeleton)
        if not math.isfinite(self.source_fps) or self.source_fps <= 0.0:
            raise ValueError("HumEnv source_fps must be finite and positive [Hz].")
        object.__setattr__(
            self,
            "construction_identity_sha256",
            canonical_sha256(
                {
                    "math_version": self.version,
                    "source_skeleton_sha256": self.source_skeleton.identity_sha256,
                    "projection_sha256": self.projection.construction_identity_sha256,
                    "source_fps": self.source_fps,
                    "smpl_decode_policy": "ordered_xyz_local_rotation_v1",
                }
            ),
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Target-G1 articulation order of the output joint axis."""
        return self.projection.target_builder.joint_names

    @property
    def reference_frame_names(self) -> tuple[str, ...]:
        """Target-G1 physical and derived reference-frame axis."""
        return self.projection.target_builder.reference_frame_names

    def allocate(self, frame_count: int, *, device: str | torch.device) -> MotionTaskTable.Frames:
        """Allocate the exact target-G1 trajectory columns."""
        return self.projection.target_builder.allocate(frame_count, device=device)

    def build_frames(
        self,
        fields: Mapping[str, object],
        *,
        device: str | torch.device,
    ) -> MotionTaskTable.Frames:
        """Build one native HumEnv clip as target-G1 trajectory facts."""
        if tuple(fields) != _HUMENV_FIELDS:
            raise ValueError("SMPL-to-G1 fields differ from the HumEnv source contract.")
        arrays = (fields["qpos"], fields["qvel"], fields["observation"])
        if not all(isinstance(value, np.ndarray) for value in arrays):
            raise ValueError("HumEnv qpos, qvel, and observation must be NumPy arrays.")
        qpos_array, qvel_array, observation_array = arrays
        if qpos_array.dtype != np.float32 or qvel_array.dtype != np.float32:
            raise ValueError("Native HumEnv qpos and qvel must be float32 NumPy arrays.")
        if observation_array.dtype != np.float64:
            raise ValueError("Native HumEnv observation must be a float64 NumPy array.")
        qpos = torch.as_tensor(qpos_array, device=device)
        local_rotation_wxyz = smpl_humenv_local_rotation_wxyz(qpos, self.source_skeleton)
        return self.projection.project_frames(qpos[:, :3], local_rotation_wxyz, self.source_fps)


def g1_smpl_humenv_frame_builder(env: ManagerBasedRLEnv) -> G1SmplHumEnvFrameBuilder:
    """Build the declared SMPL-to-G1 projection from live and exact reference models."""
    from ...kinematics import NewtonKinematics
    from ..config.source_skeletons import g1_lafan_source_skeleton

    table_cfg = env.cfg.commands.motion.task_table
    source_skeleton = table_cfg.source.build_skeleton()
    reference = table_cfg.reference_kinematics_factory(env)
    if not isinstance(reference, NewtonKinematics) or not reference.mjcf_path:
        raise TypeError("G1 frame construction requires an MJCF-backed NewtonKinematics reference.")
    robot = env.scene["robot"]
    target_builder = G1LafanFrameBuilder(
        source_skeleton=g1_lafan_source_skeleton(),
        reference_kinematics=reference,
        live_joint_names=tuple(robot.joint_names),
        live_body_names=tuple(robot.body_names),
        version="g1_target_for_smpl_projection_v1",
    )
    source_by_name = {name: index for index, name in enumerate(source_skeleton.body_names)}
    projection = G1HumanFrameProjection(
        source_skeleton=source_skeleton,
        target_builder=target_builder,
        target_joint_source_body_indices=tuple(source_by_name[name] for name in _G1_SOURCE_BODY_NAMES),
    )
    return G1SmplHumEnvFrameBuilder(source_skeleton=source_skeleton, projection=projection)


__all__ = [
    "G1HumanFrameProjection",
    "G1SmplHumEnvFrameBuilder",
    "fit_ordered_hinge_coordinates",
    "g1_smpl_humenv_frame_builder",
    "smpl_humenv_local_rotation_wxyz",
]
