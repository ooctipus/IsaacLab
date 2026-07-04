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

from isaaclab.utils.math import convert_quat, quat_apply

from ...data import MotionFrames, MotionSkeleton
from ...data.source import MotionGeneralizedCoordinateClip
from ...identity import canonical_sha256, file_sha256, validate_sha256
from .frames import smpl_live_joint_source_names

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from ....kinematics import NewtonKinematics

_ROOT_STATE_POLICY = "free_root_origin_velocity_to_newton_com_velocity_v1"


@dataclass(frozen=True, slots=True)
class SmplGeneralizedCoordinateFrameBuilder:
    """Build simulator-ordered SMPL frames from free-root generalized coordinates."""

    source_skeleton: MotionSkeleton
    reference_kinematics: NewtonKinematics
    reference_mjcf_sha256: str
    live_joint_names: tuple[str, ...]
    live_body_names: tuple[str, ...]
    version: str = "smpl_generalized_coordinate_exact_mjcf_v1"
    construction_identity_sha256: str = field(init=False)
    _live_from_source_indices: torch.Tensor = field(init=False, repr=False)
    _body_com: torch.Tensor = field(init=False, repr=False)
    _live_body_from_reference_indices: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve live/source ordering and freeze complete construction provenance."""
        reference = self.reference_kinematics
        reference_body_names = tuple(reference.body_names)
        if reference_body_names != self.source_skeleton.body_names:
            raise ValueError("The SMPL source-body order differs from the exact reference MJCF.")
        if (
            len(self.live_body_names) != len(reference_body_names)
            or self.live_body_names[0] != reference_body_names[0]
            or set(self.live_body_names) != set(reference_body_names)
        ):
            raise ValueError("The live SMPL bodies differ from the exact reference MJCF.")
        if reference.model.joint_coord_count != 7 + self.source_skeleton.num_joints:
            raise ValueError("The exact SMPL MJCF generalized-position width differs from the source schema.")
        if reference.model.joint_dof_count != 6 + self.source_skeleton.num_joints:
            raise ValueError("The exact SMPL MJCF generalized-velocity width differs from the source schema.")

        live_source_names = smpl_live_joint_source_names(self.live_joint_names)
        if len(live_source_names) != self.source_skeleton.num_joints or set(live_source_names) != set(
            self.source_skeleton.joint_names
        ):
            raise ValueError("The live SMPL joint coordinates differ from the source schema.")
        live_from_source = tuple(self.source_skeleton.joint_names.index(name) for name in live_source_names)
        live_body_from_reference = tuple(reference_body_names.index(name) for name in self.live_body_names)
        body_com = wp.to_torch(reference.model.body_com)
        if body_com.shape != (len(reference_body_names), 3) or body_com.dtype is not torch.float32:
            raise ValueError("The exact SMPL MJCF must expose one float32 center-of-mass offset per body.")
        if body_com.device != torch.device(reference.device):
            raise ValueError("The SMPL center-of-mass offsets must share the reference-kinematics device.")

        validate_sha256("reference_mjcf_sha256", self.reference_mjcf_sha256)
        identity = canonical_sha256(
            {
                "math_version": self.version,
                "source_skeleton_sha256": self.source_skeleton.identity_sha256,
                "reference_mjcf_sha256": self.reference_mjcf_sha256,
                "reference_body_names": reference_body_names,
                "live_body_names": self.live_body_names,
                "live_joint_names": self.live_joint_names,
                "live_source_joint_names": live_source_names,
                "live_from_source": live_from_source,
                "root_state_policy": _ROOT_STATE_POLICY,
                "live_body_from_reference": live_body_from_reference,
                "frame_policy": "target_smpl_physical_body_fields_v1",
            }
        )
        object.__setattr__(self, "_body_com", body_com)
        object.__setattr__(
            self,
            "_live_body_from_reference_indices",
            torch.tensor(live_body_from_reference, dtype=torch.int64, device=reference.device),
        )
        object.__setattr__(
            self,
            "_live_from_source_indices",
            torch.tensor(live_from_source, dtype=torch.int64, device=reference.device),
        )
        object.__setattr__(self, "construction_identity_sha256", identity)

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

    def build_frames(
        self,
        clip: MotionGeneralizedCoordinateClip,
        *,
        device: str | torch.device,
    ) -> MotionFrames:
        """Build one generalized-coordinate clip into target-SMPL physical fields."""
        generalized_position = torch.as_tensor(clip.generalized_position, device=device)
        generalized_velocity = torch.as_tensor(clip.generalized_velocity, device=device)
        frame_count = generalized_position.shape[0]
        if generalized_position.shape != (
            frame_count,
            7 + self.source_skeleton.num_joints,
        ) or generalized_velocity.shape != (
            frame_count,
            6 + self.source_skeleton.num_joints,
        ):
            raise ValueError("Generalized-coordinate widths differ from the declared SMPL source contract.")
        if (
            self._live_from_source_indices.device != generalized_position.device
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
            (frame_count, reference.model.body_count, 7), dtype=torch.float32, device=generalized_position.device
        )
        body_qd = torch.empty(
            (frame_count, reference.model.body_count, 6), dtype=torch.float32, device=generalized_position.device
        )
        reference.eval_fk_batched_torch(joint_q, joint_qd, body_q, body_qd)

        body_rotation = body_q[:, :, 3:7]
        body_com_world = quat_apply(body_rotation, self._body_com.expand(frame_count, -1, -1))
        body_linear_velocity = body_qd[:, :, :3] - torch.cross(body_qd[:, :, 3:], body_com_world, dim=-1)
        body_indices = self._live_body_from_reference_indices
        return MotionFrames(
            joint_position=generalized_position[:, 7:].index_select(1, self._live_from_source_indices),
            joint_velocity=generalized_velocity[:, 6:].index_select(1, self._live_from_source_indices),
            body_position=body_q[:, :, :3].index_select(1, body_indices),
            body_rotation=body_rotation.index_select(1, body_indices),
            body_linear_velocity=body_linear_velocity.index_select(1, body_indices),
            body_angular_velocity=body_qd[:, :, 3:].index_select(1, body_indices),
        )


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
