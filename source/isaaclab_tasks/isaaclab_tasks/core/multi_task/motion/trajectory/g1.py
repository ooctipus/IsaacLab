# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact-MJCF G1 trajectory construction for native LAFAN rows."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.utils.math import quat_conjugate, quat_from_rotation_vector, quat_mul

from ..data import MotionSkeleton
from ..data._identity import canonical_sha256
from ..frames import (
    G1_HEAD_FRAME_NAME,
    G1_HEAD_OFFSET_M,
    G1_HEAD_PARENT_BODY_NAME,
    G1_HEAD_POSE_POLICY,
    append_g1_head_pose,
)
from ..mdp.commands.motion_task_table import MotionTaskTable
from ._time import gaussian_filter_time, gradient_time

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ...kinematics import NewtonKinematics

_SOURCE_FIELDS = ("root_trans_offset", "pose_aa", "fps")
_DERIVATIVE_POLICY = "bfm_gradient_gaussian2_quaternion_acos_and_joint_forward_v1"


def _sha256(path: str) -> str:
    """Hash one exact reference artifact without retaining its contents."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True, slots=True)
class G1LafanFrameBuilder:
    """Build simulator-ordered G1 frames from native BFM pose-axis-angle clips."""

    source_skeleton: MotionSkeleton
    reference_kinematics: NewtonKinematics
    live_joint_names: tuple[str, ...]
    live_body_names: tuple[str, ...]
    version: str = "g1_lafan_exact_mjcf_v1"
    construction_identity_sha256: str = field(init=False)
    reference_mjcf_sha256: str = field(init=False)
    _live_joint_from_reference: tuple[int, ...] = field(init=False, repr=False)
    _live_joint_from_reference_indices: torch.Tensor = field(init=False, repr=False)
    _head_parent_body_index: int = field(init=False, repr=False)
    _live_body_from_reference: tuple[int, ...] = field(init=False, repr=False)
    _live_body_from_reference_indices: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve live/reference ordering and freeze complete construction provenance."""
        reference = self.reference_kinematics
        reference_joint_names = tuple(reference.joint_names[1:])
        reference_body_names = tuple(reference.body_names)
        if self.source_skeleton.joint_names != reference_joint_names:
            raise ValueError("The G1 source-coordinate order differs from the exact reference MJCF.")
        if self.source_skeleton.body_names != reference_body_names:
            raise ValueError("The G1 source-body order differs from the exact reference MJCF.")
        if len(self.live_body_names) != len(reference_body_names) or set(self.live_body_names) != set(
            reference_body_names
        ):
            raise ValueError("The live G1 body names differ from the exact reference MJCF.")
        if self.live_body_names[0] != reference_body_names[0]:
            raise ValueError("The live G1 root body must remain first for root-state views.")
        if len(self.live_joint_names) != len(reference_joint_names) or set(self.live_joint_names) != set(
            reference_joint_names
        ):
            raise ValueError("The live G1 joint names differ from the exact reference MJCF.")
        live_joint_from_reference = tuple(reference_joint_names.index(name) for name in self.live_joint_names)
        live_body_from_reference = tuple(reference_body_names.index(name) for name in self.live_body_names)
        head_parent = self.live_body_names.index(G1_HEAD_PARENT_BODY_NAME)
        reference_sha256 = _sha256(reference.mjcf_path)
        identity = canonical_sha256(
            {
                "math_version": self.version,
                "source_skeleton_sha256": self.source_skeleton.identity_sha256,
                "reference_mjcf_sha256": reference_sha256,
                "reference_joint_names": reference_joint_names,
                "reference_body_names": reference_body_names,
                "live_joint_names": self.live_joint_names,
                "live_body_names": self.live_body_names,
                "live_joint_from_reference": live_joint_from_reference,
                "live_body_from_reference": live_body_from_reference,
                "derived_frame": {
                    "name": G1_HEAD_FRAME_NAME,
                    "pose_policy": G1_HEAD_POSE_POLICY,
                    "parent": G1_HEAD_PARENT_BODY_NAME,
                    "parent_index": head_parent,
                    "offset_m": G1_HEAD_OFFSET_M,
                },
                "derivative_policy": _DERIVATIVE_POLICY,
            }
        )
        object.__setattr__(self, "_live_joint_from_reference", live_joint_from_reference)
        object.__setattr__(
            self,
            "_live_joint_from_reference_indices",
            torch.tensor(live_joint_from_reference, dtype=torch.int64, device=reference.device),
        )
        object.__setattr__(self, "_head_parent_body_index", head_parent)
        object.__setattr__(self, "_live_body_from_reference", live_body_from_reference)
        object.__setattr__(
            self,
            "_live_body_from_reference_indices",
            torch.tensor(live_body_from_reference, dtype=torch.int64, device=reference.device),
        )
        object.__setattr__(self, "reference_mjcf_sha256", reference_sha256)
        object.__setattr__(self, "construction_identity_sha256", identity)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Live-articulation order of the output joint axis."""
        return self.live_joint_names

    @property
    def reference_frame_names(self) -> tuple[str, ...]:
        """Live physical bodies followed by the released derived head frame."""
        return (*self.live_body_names, G1_HEAD_FRAME_NAME)

    def allocate(self, frame_count: int, *, device: str | torch.device) -> MotionTaskTable.Frames:
        """Allocate exact-capacity G1 trajectory columns in live simulator order."""
        joint_count = len(self.live_joint_names)
        body_count = len(self.live_body_names) + 1
        return MotionTaskTable.Frames(
            joint_position=torch.empty(frame_count, joint_count, dtype=torch.float32, device=device),
            joint_velocity=torch.empty(frame_count, joint_count, dtype=torch.float32, device=device),
            body_position=torch.empty(frame_count, body_count, 3, dtype=torch.float32, device=device),
            body_rotation=torch.empty(frame_count, body_count, 4, dtype=torch.float32, device=device),
            body_linear_velocity=torch.empty(frame_count, body_count, 3, dtype=torch.float32, device=device),
            body_angular_velocity=torch.empty(frame_count, body_count, 3, dtype=torch.float32, device=device),
        )

    @staticmethod
    def _angular_velocity_raw(rotation_xyzw: torch.Tensor, step_seconds: float) -> torch.Tensor:
        """Apply the released BFM finite-difference angular-velocity equation."""
        difference = torch.zeros_like(rotation_xyzw)
        difference[..., 3] = 1.0
        relative = quat_mul(rotation_xyzw[1:], quat_conjugate(rotation_xyzw[:-1]))
        difference[:-1] = relative / relative.norm(p=2, dim=-1, keepdim=True).clamp_min(1.0e-9)
        scalar = difference[..., 3]
        angle = torch.acos((2.0 * scalar**2 - 1.0).clamp(-1.0, 1.0))
        axis = difference[..., :3]
        axis = axis / axis.norm(p=2, dim=-1, keepdim=True).clamp_min(1.0e-9)
        return axis * angle[..., None] / step_seconds

    @classmethod
    def _angular_velocity(cls, rotation_xyzw: torch.Tensor, step_seconds: float) -> torch.Tensor:
        """Filter released raw angular velocities along the frame axis."""
        raw = cls._angular_velocity_raw(rotation_xyzw, step_seconds)
        return gaussian_filter_time(raw.unsqueeze(0)).squeeze(0).contiguous()

    def build_pose_frames(
        self,
        pose_axis_angle: torch.Tensor,
        root_translation: torch.Tensor,
        source_fps: float,
    ) -> MotionTaskTable.Frames:
        """Build one target-G1 trajectory sampled at ``source_fps`` [Hz]."""
        reference = self.reference_kinematics
        frame_count = pose_axis_angle.shape[0]
        expected_pose = (frame_count, reference.model.body_count, 3)
        if pose_axis_angle.shape != expected_pose or root_translation.shape != (frame_count, 3):
            raise ValueError("G1 pose and root translation shapes do not match the reference MJCF.")
        if pose_axis_angle.dtype != torch.float32 or root_translation.dtype != torch.float32:
            raise ValueError("G1 trajectory construction requires float32 pose tensors.")
        if pose_axis_angle.device != root_translation.device:
            raise ValueError("G1 pose and root translation must share one device.")
        if not math.isfinite(source_fps) or source_fps <= 0.0:
            raise ValueError("source_fps must be finite and positive [Hz].")

        joint_position_reference = pose_axis_angle.sum(dim=-1)[:, 1:]
        joint_q = torch.empty(
            frame_count,
            reference.model.joint_coord_count,
            dtype=torch.float32,
            device=pose_axis_angle.device,
        )
        joint_q[:, :3].copy_(root_translation)
        joint_q[:, 3:7].copy_(quat_from_rotation_vector(pose_axis_angle[:, 0]))
        joint_q[:, 7:].copy_(joint_position_reference)
        joint_qd = torch.zeros(
            frame_count,
            reference.model.joint_dof_count,
            dtype=torch.float32,
            device=pose_axis_angle.device,
        )
        body_q = torch.empty(
            frame_count,
            reference.model.body_count,
            7,
            dtype=torch.float32,
            device=pose_axis_angle.device,
        )
        body_qd_scratch = torch.empty(
            frame_count,
            reference.model.body_count,
            6,
            dtype=torch.float32,
            device=pose_axis_angle.device,
        )
        if (
            self._live_joint_from_reference_indices.device != pose_axis_angle.device
            or self._live_body_from_reference_indices.device != pose_axis_angle.device
        ):
            raise ValueError("G1 trajectory tensors must use the reference-kinematics device.")
        reference.eval_fk_batched_torch(joint_q, joint_qd, body_q, body_qd_scratch)
        body_position = body_q[..., :3].index_select(1, self._live_body_from_reference_indices).contiguous()
        body_rotation = body_q[..., 3:].index_select(1, self._live_body_from_reference_indices).contiguous()
        body_position, body_rotation = append_g1_head_pose(
            body_position, body_rotation, parent_body_index=self._head_parent_body_index
        )

        step_seconds = 1.0 / source_fps
        body_linear_velocity = (
            gaussian_filter_time(gradient_time(body_position.unsqueeze(0), step_seconds)).squeeze(0).contiguous()
        )
        body_angular_velocity = self._angular_velocity(body_rotation, step_seconds)

        difference = (joint_position_reference[1:] - joint_position_reference[:-1]) * source_fps
        if difference.shape[0] < 2:
            raise ValueError("G1 joint velocity construction requires at least three source frames.")
        joint_velocity_reference = torch.cat((difference, difference[-2:-1]), dim=0)
        return MotionTaskTable.Frames(
            joint_position=joint_position_reference.index_select(1, self._live_joint_from_reference_indices),
            joint_velocity=joint_velocity_reference.index_select(1, self._live_joint_from_reference_indices),
            body_position=body_position,
            body_rotation=body_rotation,
            body_linear_velocity=body_linear_velocity,
            body_angular_velocity=body_angular_velocity,
        )

    def build_frames(
        self,
        fields: Mapping[str, object],
        *,
        device: str | torch.device,
    ) -> MotionTaskTable.Frames:
        """Build one native BFM G1 clip through exact reference FK."""
        if tuple(fields) != _SOURCE_FIELDS:
            raise ValueError("BFM trajectory fields differ from the native ordered contract.")
        pose_array = fields["pose_aa"]
        translation_array = fields["root_trans_offset"]
        if not isinstance(pose_array, np.ndarray) or not isinstance(translation_array, np.ndarray):
            raise ValueError("BFM pose_aa and root_trans_offset must be NumPy arrays.")
        if pose_array.dtype != np.float32 or translation_array.dtype != np.float32:
            raise ValueError("BFM pose_aa and root_trans_offset must be float32 NumPy arrays.")
        if fields["fps"] != 30:
            raise ValueError("BFM native trajectory construction requires fps=30.")
        return self.build_pose_frames(
            torch.as_tensor(pose_array, device=device),
            torch.as_tensor(translation_array, device=device),
            30.0,
        )


def g1_lafan_frame_builder(env: ManagerBasedRLEnv) -> G1LafanFrameBuilder:
    """Build native G1 trajectory policy from the live articulation and exact MJCF."""
    from ...kinematics import NewtonKinematics

    table_cfg = env.cfg.commands.motion.task_table
    source_skeleton = table_cfg.source.build_skeleton()
    reference = table_cfg.reference_kinematics_factory(env)
    if not isinstance(reference, NewtonKinematics) or not reference.mjcf_path:
        raise TypeError("G1 frame construction requires an MJCF-backed NewtonKinematics reference.")
    robot = env.scene["robot"]
    return G1LafanFrameBuilder(
        source_skeleton=source_skeleton,
        reference_kinematics=reference,
        live_joint_names=tuple(robot.joint_names),
        live_body_names=tuple(robot.body_names),
    )


__all__ = ["G1LafanFrameBuilder", "g1_lafan_frame_builder"]
