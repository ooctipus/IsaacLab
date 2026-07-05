# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact-MJCF SMPL trajectory construction from generalized coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_inverse, quat_mul

from isaaclab_assets.robots.smpl.smpl_constants import SMPL_HUMENV_MJCF_SHA256

from ....kinematics import (
    fit_ordered_hinge_coordinates,
    time_gradient,
    time_quaternion_angular_velocity,
    time_unwrap_angles,
)
from ...data import MotionFrames, MotionSkeleton
from ...data.source import MotionGeneralizedCoordinateClip, MotionLocalBodyPoseClip
from ...identity import canonical_sha256, file_sha256, validate_sha256
from .frames import smpl_live_joint_source_names

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from ....kinematics import NewtonKinematics

_ROOT_STATE_POLICY = "free_root_origin_velocity_to_newton_com_velocity_v1"


@dataclass(frozen=True, slots=True)
class _SmplTargetFrameBuilder:
    """Materialize exact target-SMPL frames from generalized-coordinate tensors."""

    reference_kinematics: NewtonKinematics
    reference_mjcf_sha256: str
    live_joint_names: tuple[str, ...]
    live_body_names: tuple[str, ...]
    construction_identity_sha256: str = field(init=False)
    _live_from_reference_indices: torch.Tensor = field(init=False, repr=False)
    _body_com: torch.Tensor = field(init=False, repr=False)
    _live_body_from_reference_indices: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve the live articulation against the exact target reference once."""
        reference = self.reference_kinematics
        reference_body_names = tuple(reference.body_names)
        reference_joint_names = tuple(reference.joint_names[1:])
        expected_joint_names = tuple(f"{body_name}_{axis}" for body_name in reference_body_names[1:] for axis in "xyz")
        if reference_joint_names != expected_joint_names:
            raise ValueError("The SMPL reference must expose one ordered XYZ coordinate chain per non-root body.")
        if (
            len(self.live_body_names) != len(reference_body_names)
            or self.live_body_names[0] != reference_body_names[0]
            or set(self.live_body_names) != set(reference_body_names)
        ):
            raise ValueError("The live SMPL bodies differ from the exact reference MJCF.")
        if reference.model.joint_coord_count != 7 + len(reference_joint_names):
            raise ValueError("The exact SMPL MJCF generalized-position width differs from its named coordinates.")
        if reference.model.joint_dof_count != 6 + len(reference_joint_names):
            raise ValueError("The exact SMPL MJCF generalized-velocity width differs from its named coordinates.")

        live_reference_names = smpl_live_joint_source_names(self.live_joint_names)
        if len(live_reference_names) != len(reference_joint_names) or set(live_reference_names) != set(
            reference_joint_names
        ):
            raise ValueError("The live SMPL joint coordinates differ from the exact reference MJCF.")
        live_from_reference = tuple(reference_joint_names.index(name) for name in live_reference_names)
        live_body_from_reference = tuple(reference_body_names.index(name) for name in self.live_body_names)
        body_com = wp.to_torch(reference.model.body_com)
        if body_com.shape != (len(reference_body_names), 3) or body_com.dtype is not torch.float32:
            raise ValueError("The exact SMPL MJCF must expose one float32 center-of-mass offset per body.")
        if body_com.device != torch.device(reference.device):
            raise ValueError("The SMPL center-of-mass offsets must share the reference-kinematics device.")

        validate_sha256("reference_mjcf_sha256", self.reference_mjcf_sha256)
        object.__setattr__(self, "_body_com", body_com)
        object.__setattr__(
            self,
            "_live_body_from_reference_indices",
            torch.tensor(live_body_from_reference, dtype=torch.int64, device=reference.device),
        )
        object.__setattr__(
            self,
            "_live_from_reference_indices",
            torch.tensor(live_from_reference, dtype=torch.int64, device=reference.device),
        )
        object.__setattr__(
            self,
            "construction_identity_sha256",
            canonical_sha256(
                {
                    "reference_mjcf_sha256": self.reference_mjcf_sha256,
                    "reference_body_names": reference_body_names,
                    "reference_joint_names": reference_joint_names,
                    "live_body_names": self.live_body_names,
                    "live_joint_names": self.live_joint_names,
                    "live_body_from_reference": live_body_from_reference,
                    "live_from_reference": live_from_reference,
                    "root_state_policy": _ROOT_STATE_POLICY,
                    "frame_policy": "target_smpl_physical_body_fields_v1",
                }
            ),
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Live-articulation order of the output joint axis."""
        return self.live_joint_names

    @property
    def reference_frame_names(self) -> tuple[str, ...]:
        """Live-articulation order of physical reference frames."""
        return self.live_body_names

    def allocate(self, frame_count: int, *, device: str | torch.device) -> MotionFrames:
        """Allocate exact-capacity SMPL trajectory columns in live simulator order."""
        joint_count = len(self.live_joint_names)
        body_count = len(self.live_body_names)
        return MotionFrames(
            joint_position=torch.empty(frame_count, joint_count, dtype=torch.float32, device=device),
            joint_velocity=torch.empty(frame_count, joint_count, dtype=torch.float32, device=device),
            body_position=torch.empty(frame_count, body_count, 3, dtype=torch.float32, device=device),
            body_rotation=torch.empty(frame_count, body_count, 4, dtype=torch.float32, device=device),
            body_linear_velocity=torch.empty(frame_count, body_count, 3, dtype=torch.float32, device=device),
            body_angular_velocity=torch.empty(frame_count, body_count, 3, dtype=torch.float32, device=device),
        )

    def build_generalized_frames(
        self, generalized_position: torch.Tensor, generalized_velocity: torch.Tensor
    ) -> MotionFrames:
        """Build target-SMPL frames from wxyz positions and root-local velocities."""
        frame_count = generalized_position.shape[0]
        coordinate_count = len(self.reference_kinematics.joint_names) - 1
        if generalized_position.shape != (frame_count, 7 + coordinate_count) or generalized_velocity.shape != (
            frame_count,
            6 + coordinate_count,
        ):
            raise ValueError("Generalized-coordinate tensors differ from the exact SMPL reference widths.")
        if generalized_position.dtype is not torch.float32 or generalized_velocity.dtype is not torch.float32:
            raise ValueError("SMPL generalized-coordinate tensors must use float32.")
        if generalized_position.device != generalized_velocity.device:
            raise ValueError("SMPL generalized positions and velocities must share one device.")
        if (
            self._live_from_reference_indices.device != generalized_position.device
            or self._body_com.device != generalized_position.device
        ):
            raise ValueError("SMPL trajectory tensors must use the reference-kinematics device.")

        reference = self.reference_kinematics
        joint_q = generalized_position.clone()
        root_rotation = convert_quat(generalized_position[:, 3:7], to="xyzw")
        joint_q[:, 3:7].copy_(root_rotation)
        joint_qd = generalized_velocity.clone()
        root_angular_velocity = quat_apply(root_rotation, generalized_velocity[:, 3:6])
        joint_qd[:, 3:6].copy_(root_angular_velocity)
        root_com_world = quat_apply(root_rotation, self._body_com[0].expand(frame_count, 3))
        joint_qd[:, :3].add_(torch.cross(root_angular_velocity, root_com_world, dim=-1))

        body_q = torch.empty(
            frame_count, reference.model.body_count, 7, dtype=torch.float32, device=generalized_position.device
        )
        body_qd = torch.empty(
            frame_count, reference.model.body_count, 6, dtype=torch.float32, device=generalized_position.device
        )
        reference.eval_fk_batched_torch(joint_q, joint_qd, body_q, body_qd)

        body_rotation = body_q[:, :, 3:7]
        body_com_world = quat_apply(body_rotation, self._body_com.expand(frame_count, -1, -1))
        body_linear_velocity = body_qd[:, :, :3] - torch.cross(body_qd[:, :, 3:], body_com_world, dim=-1)
        body_indices = self._live_body_from_reference_indices
        return MotionFrames(
            joint_position=generalized_position[:, 7:].index_select(1, self._live_from_reference_indices),
            joint_velocity=generalized_velocity[:, 6:].index_select(1, self._live_from_reference_indices),
            body_position=body_q[:, :, :3].index_select(1, body_indices),
            body_rotation=body_rotation.index_select(1, body_indices),
            body_linear_velocity=body_linear_velocity.index_select(1, body_indices),
            body_angular_velocity=body_qd[:, :, 3:].index_select(1, body_indices),
        )


@dataclass(frozen=True, slots=True)
class SmplGeneralizedCoordinateFrameBuilder:
    """Build simulator-ordered SMPL frames from native generalized coordinates."""

    source_skeleton: MotionSkeleton
    reference_kinematics: NewtonKinematics
    reference_mjcf_sha256: str
    live_joint_names: tuple[str, ...]
    live_body_names: tuple[str, ...]
    version: str = "smpl_generalized_coordinate_exact_mjcf_v1"
    construction_identity_sha256: str = field(init=False)
    _target: _SmplTargetFrameBuilder = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Bind the native source coordinates to the exact target-SMPL model."""
        target = _SmplTargetFrameBuilder(
            self.reference_kinematics,
            self.reference_mjcf_sha256,
            self.live_joint_names,
            self.live_body_names,
        )
        if tuple(self.reference_kinematics.body_names) != self.source_skeleton.body_names:
            raise ValueError("The SMPL source-body order differs from the exact reference MJCF.")
        if tuple(self.reference_kinematics.joint_names[1:]) != self.source_skeleton.joint_names:
            raise ValueError("The SMPL source-coordinate order differs from the exact reference MJCF.")
        object.__setattr__(self, "_target", target)
        object.__setattr__(
            self,
            "construction_identity_sha256",
            canonical_sha256(
                {
                    "math_version": self.version,
                    "source_skeleton_sha256": self.source_skeleton.identity_sha256,
                    "target_construction_sha256": target.construction_identity_sha256,
                }
            ),
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Live-articulation order of the output joint axis."""
        return self._target.joint_names

    @property
    def reference_frame_names(self) -> tuple[str, ...]:
        """Live-articulation order of physical reference frames."""
        return self._target.reference_frame_names

    def allocate(self, frame_count: int, *, device: str | torch.device) -> MotionFrames:
        """Allocate exact-capacity SMPL trajectory columns in live simulator order."""
        return self._target.allocate(frame_count, device=device)

    def build_frames(self, clip: MotionGeneralizedCoordinateClip, *, device: str | torch.device) -> MotionFrames:
        """Build one native generalized-coordinate clip into target-SMPL fields."""
        generalized_position = torch.as_tensor(clip.generalized_position, device=device)
        generalized_velocity = torch.as_tensor(clip.generalized_velocity, device=device)
        return self._target.build_generalized_frames(generalized_position, generalized_velocity)


_SMPL_TARGET_SOURCE_BODY_CHAINS = (
    ("L_Hip", ("left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link")),
    ("L_Knee", ("left_knee_link",)),
    ("L_Ankle", ("left_ankle_pitch_link", "left_ankle_roll_link")),
    ("L_Toe", ()),
    ("R_Hip", ("right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link")),
    ("R_Knee", ("right_knee_link",)),
    ("R_Ankle", ("right_ankle_pitch_link", "right_ankle_roll_link")),
    ("R_Toe", ()),
    ("Torso", ("waist_yaw_link", "waist_roll_link", "torso_link")),
    ("Spine", ()),
    ("Chest", ()),
    ("Neck", ()),
    ("Head", ()),
    ("L_Thorax", ()),
    ("L_Shoulder", ("left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link")),
    ("L_Elbow", ("left_elbow_link",)),
    ("L_Wrist", ("left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link")),
    ("L_Hand", ()),
    ("R_Thorax", ()),
    ("R_Shoulder", ("right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link")),
    ("R_Elbow", ("right_elbow_link",)),
    ("R_Wrist", ("right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link")),
    ("R_Hand", ()),
)


@dataclass(frozen=True, slots=True)
class _SmplG1HingeFrameBuilder:
    """Reconstruct target SMPL from already G1-retargeted local hinge rotations."""

    source_skeleton: MotionSkeleton
    reference_kinematics: NewtonKinematics
    reference_mjcf_sha256: str
    live_joint_names: tuple[str, ...]
    live_body_names: tuple[str, ...]
    version: str = "smpl_from_g1_hinge_local_rotation_minimum_norm_v1"
    construction_identity_sha256: str = field(init=False)
    _target: _SmplTargetFrameBuilder = field(init=False, repr=False)
    _source_body_chains: tuple[tuple[int, ...], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve the explicit lossy source-to-target body reconstruction."""
        target = _SmplTargetFrameBuilder(
            self.reference_kinematics,
            self.reference_mjcf_sha256,
            self.live_joint_names,
            self.live_body_names,
        )
        target_body_names = tuple(self.reference_kinematics.body_names)
        if target_body_names[1:] != tuple(name for name, _ in _SMPL_TARGET_SOURCE_BODY_CHAINS):
            raise ValueError("The target SMPL body order differs from the declared reconstruction policy.")
        if self.source_skeleton.body_names[0] != "pelvis":
            raise ValueError("The G1-hinge source reconstruction requires pelvis as its root body.")
        source_by_name = {name: index for index, name in enumerate(self.source_skeleton.body_names)}
        try:
            source_body_chains = tuple(
                tuple(source_by_name[source_name] for source_name in source_names)
                for _, source_names in _SMPL_TARGET_SOURCE_BODY_CHAINS
            )
        except KeyError as error:
            raise ValueError(f"The G1-hinge source is missing semantic body {error.args[0]!r}.") from error
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_source_body_chains", source_body_chains)
        object.__setattr__(
            self,
            "construction_identity_sha256",
            canonical_sha256(
                {
                    "math_version": self.version,
                    "source_skeleton_sha256": self.source_skeleton.identity_sha256,
                    "target_construction_sha256": target.construction_identity_sha256,
                    "target_source_body_chains": _SMPL_TARGET_SOURCE_BODY_CHAINS,
                    "unmapped_target_policy": "identity_local_rotation_minimum_norm",
                    "root_translation_policy": "preserve_source_pelvis_world_translation",
                    "source_rest_transform_policy": "provenance_only_not_baked_into_motion_rotation",
                }
            ),
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Live-articulation order of the output joint axis."""
        return self._target.joint_names

    @property
    def reference_frame_names(self) -> tuple[str, ...]:
        """Live-articulation order of physical reference frames."""
        return self._target.reference_frame_names

    def allocate(self, frame_count: int, *, device: str | torch.device) -> MotionFrames:
        """Allocate exact-capacity SMPL trajectory columns in live simulator order."""
        return self._target.allocate(frame_count, device=device)

    def build_frames(self, clip: MotionLocalBodyPoseClip, *, device: str | torch.device) -> MotionFrames:
        """Build target-SMPL frames from one declared G1-hinge local-rotation clip."""
        source_local_wxyz = clip.local_body_rotation_wxyz(self.source_skeleton, device=device)
        source_local_xyzw = convert_quat(source_local_wxyz, to="xyzw")
        frame_count = source_local_xyzw.shape[0]
        target_coordinates = torch.zeros(
            frame_count, len(self._source_body_chains), 3, dtype=source_local_xyzw.dtype, device=device
        )
        xyz_axes = torch.eye(3, dtype=torch.float32, device=device)
        for target_index, source_chain in enumerate(self._source_body_chains):
            if not source_chain:
                continue
            rotation = source_local_xyzw[:, source_chain[0]]
            for source_index in source_chain[1:]:
                rotation = quat_mul(rotation, source_local_xyzw[:, source_index])
            coordinates, _ = fit_ordered_hinge_coordinates(rotation, xyz_axes)
            target_coordinates[:, target_index].copy_(time_unwrap_angles(coordinates))

        root_translation = torch.as_tensor(clip.root_translation, device=device)
        root_rotation_xyzw = source_local_xyzw[:, 0]
        step_seconds = 1.0 / clip.source_fps
        generalized_position = torch.empty(
            frame_count, 7 + target_coordinates.shape[1] * 3, dtype=torch.float32, device=device
        )
        generalized_position[:, :3].copy_(root_translation)
        generalized_position[:, 3:7].copy_(convert_quat(root_rotation_xyzw, to="wxyz"))
        generalized_position[:, 7:].copy_(target_coordinates.flatten(1))

        root_linear_velocity = time_gradient(root_translation.unsqueeze(0), step_seconds).squeeze(0)
        root_angular_velocity_world = time_quaternion_angular_velocity(
            root_rotation_xyzw.unsqueeze(0), step_seconds
        ).squeeze(0)
        root_angular_velocity_local = quat_apply_inverse(root_rotation_xyzw, root_angular_velocity_world)
        joint_velocity = time_gradient(target_coordinates.flatten(1).unsqueeze(0), step_seconds).squeeze(0)
        generalized_velocity = torch.cat((root_linear_velocity, root_angular_velocity_local, joint_velocity), dim=-1)
        return self._target.build_generalized_frames(generalized_position, generalized_velocity)


def smpl_generalized_coordinate_frame_builder(
    source_skeleton: MotionSkeleton,
    reference: NewtonKinematics,
    robot: Articulation,
) -> SmplGeneralizedCoordinateFrameBuilder:
    """Build a generalized-coordinate trajectory policy from the live articulation and exact MJCF."""
    from ....kinematics import NewtonKinematics

    if not isinstance(reference, NewtonKinematics) or not reference.mjcf_path:
        raise TypeError("SMPL frame construction requires an MJCF-backed NewtonKinematics reference.")
    reference_mjcf_sha256 = file_sha256(reference.mjcf_path)
    if reference_mjcf_sha256 != source_skeleton.content_sha256:
        raise ValueError("The injected SMPL reference model differs from the declared source coordinates.")

    return SmplGeneralizedCoordinateFrameBuilder(
        source_skeleton=source_skeleton,
        reference_kinematics=reference,
        reference_mjcf_sha256=reference_mjcf_sha256,
        live_joint_names=tuple(robot.joint_names),
        live_body_names=tuple(robot.body_names),
    )


def smpl_g1_hinge_frame_builder(
    source_skeleton: MotionSkeleton,
    reference: NewtonKinematics,
    robot: Articulation,
) -> _SmplG1HingeFrameBuilder:
    """Build the declared G1-hinge-source to target-SMPL reconstruction."""
    from ....kinematics import NewtonKinematics

    if not isinstance(reference, NewtonKinematics) or not reference.mjcf_path:
        raise TypeError("SMPL frame construction requires an MJCF-backed NewtonKinematics reference.")
    reference_mjcf_sha256 = file_sha256(reference.mjcf_path)
    if reference_mjcf_sha256 != SMPL_HUMENV_MJCF_SHA256:
        raise ValueError("The injected SMPL target model differs from the packaged HumEnv coordinates.")
    return _SmplG1HingeFrameBuilder(
        source_skeleton=source_skeleton,
        reference_kinematics=reference,
        reference_mjcf_sha256=reference_mjcf_sha256,
        live_joint_names=tuple(robot.joint_names),
        live_body_names=tuple(robot.body_names),
    )
