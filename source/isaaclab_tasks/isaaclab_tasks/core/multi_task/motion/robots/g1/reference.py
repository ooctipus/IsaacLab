# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact-MJCF G1 trajectory construction from declared pose representations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_from_rotation_vector

from ....kinematics import (
    KinematicTree,
    KinematicTreeRotationProjection,
    time_gaussian_filter,
    time_gradient,
    time_quaternion_angular_velocity,
)
from ...data import MotionFrames, MotionSkeleton
from ...data.source import MotionLocalBodyPoseClip, MotionPoseAxisAngleClip
from ...identity import canonical_sha256, file_sha256, validate_sha256
from .frames import (
    G1_HEAD_FRAME_NAME,
    G1_HEAD_OFFSET_M,
    G1_HEAD_PARENT_BODY_NAME,
    G1_HEAD_POSE_POLICY,
    append_g1_head_pose,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from ....kinematics import NewtonKinematics

_G1_REFERENCE_MJCF_RELATIVE_PATH = "humanoidverse/data/robots/g1/g1_29dof.xml"
G1_REFERENCE_MJCF_SHA256 = "439c1ec0806583d73b492da9484b0cb9e9eae215e0d9506e3c2fa69016733532"


def g1_reference_kinematics(reference_artifact_root: str, device: str | torch.device) -> NewtonKinematics:
    """Build the hash-verified external BFM reference model on the requested device."""
    from ....kinematics import NewtonKinematics, NewtonKinematicsCfg

    path = Path(reference_artifact_root).expanduser() / _G1_REFERENCE_MJCF_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"G1 reference MJCF does not exist: {path}")
    actual = file_sha256(path)
    if actual != G1_REFERENCE_MJCF_SHA256:
        raise ValueError(f"G1 reference MJCF hash differs: expected {G1_REFERENCE_MJCF_SHA256}, got {actual}.")
    return NewtonKinematics(
        NewtonKinematicsCfg(usd_path=None, mjcf_path=str(path), device=str(device), collapse_fixed_joints=False)
    )


_DERIVATIVE_POLICY = "pose_gradient_gaussian2_quaternion_acos_and_joint_forward_v1"


def _g1_target_tree(reference: NewtonKinematics) -> KinematicTree:
    """Derive and validate the exact G1 scalar-hinge topology."""
    if not reference.mjcf_path:
        raise TypeError("The G1 target tree requires an MJCF-backed reference model.")
    tree = KinematicTree.from_newton(reference)
    if tree.num_bodies != 30 or tree.num_joints != 29:
        raise ValueError("The target G1 model must contain 30 bodies and 29 hinge coordinates.")
    if tree.root_body_index != 0 or tree.joint_child_body_indices != tuple(range(1, 30)):
        raise ValueError("The target G1 model must expose root-first, one-body-per-hinge coordinates.")
    return tree


def _g1_target_tree_identity(tree: KinematicTree, content_sha256: str) -> str:
    """Hash target topology and exact reference-model provenance."""
    return canonical_sha256(
        {
            "content_sha256": content_sha256,
            "body_names": tree.body_names,
            "joint_names": tree.joint_names,
            "parent_indices": tree.parent_indices,
            "joint_child_body_indices": tree.joint_child_body_indices,
            "joint_axes": tree.joint_axes,
        }
    )


def _validate_source_target_coordinates(
    source: MotionSkeleton, target: KinematicTree, reference_mjcf_sha256: str
) -> None:
    """Require one source row per target coordinate without aliasing either owner."""
    if source.content_sha256 != reference_mjcf_sha256:
        raise ValueError("The G1 source coordinates were not declared from the exact reference MJCF.")
    if source.joint_names != target.joint_names:
        raise ValueError("The G1 source-coordinate order differs from the exact reference MJCF.")
    if source.body_names != target.body_names:
        raise ValueError("The G1 source-body order differs from the exact reference MJCF.")
    if source.parent_indices != target.parent_indices:
        raise ValueError("The G1 source topology differs from the exact reference MJCF.")
    if source.joint_child_body_indices != target.joint_child_body_indices:
        raise ValueError("The G1 source joint-child mapping differs from the exact reference MJCF.")
    if source.joint_axes != target.joint_axes:
        raise ValueError("The G1 source hinge axes differ from the exact reference MJCF.")
    if (
        source.root_translation_frame != "world"
        or source.root_rotation_convention != "axis_angle"
        or source.position_unit != "m"
        or source.angle_unit != "rad"
    ):
        raise ValueError("The G1 source root or unit conventions differ from the pose-builder contract.")


@dataclass(frozen=True, slots=True)
class G1PoseFrameBuilder:
    """Build simulator-ordered G1 frames from target-body pose-axis-angle coordinates."""

    target_tree: KinematicTree
    pose_coordinate_identity_sha256: str
    reference_kinematics: NewtonKinematics
    reference_mjcf_sha256: str
    live_joint_names: tuple[str, ...]
    live_body_names: tuple[str, ...]
    version: str = "g1_pose_exact_mjcf_v1"
    construction_identity_sha256: str = field(init=False)
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
        if self.target_tree.joint_names != reference_joint_names:
            raise ValueError("The G1 target-coordinate order differs from the exact reference MJCF.")
        if self.target_tree.body_names != reference_body_names:
            raise ValueError("The G1 target-body order differs from the exact reference MJCF.")
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
        validate_sha256("pose_coordinate_identity_sha256", self.pose_coordinate_identity_sha256)
        validate_sha256("reference_mjcf_sha256", self.reference_mjcf_sha256)
        identity = canonical_sha256(
            {
                "math_version": self.version,
                "pose_coordinate_identity_sha256": self.pose_coordinate_identity_sha256,
                "reference_mjcf_sha256": self.reference_mjcf_sha256,
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
        object.__setattr__(self, "construction_identity_sha256", identity)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Live-articulation order of the output joint axis."""
        return self.live_joint_names

    @property
    def reference_frame_names(self) -> tuple[str, ...]:
        """Live physical bodies followed by BFM-Zero's derived head frame."""
        return (*self.live_body_names, G1_HEAD_FRAME_NAME)

    def allocate(self, frame_count: int, *, device: str | torch.device) -> MotionFrames:
        """Allocate exact-capacity G1 trajectory columns in live simulator order."""
        joint_count = len(self.live_joint_names)
        body_count = len(self.live_body_names) + 1
        return MotionFrames(
            joint_position=torch.empty(frame_count, joint_count, dtype=torch.float32, device=device),
            joint_velocity=torch.empty(frame_count, joint_count, dtype=torch.float32, device=device),
            body_position=torch.empty(frame_count, body_count, 3, dtype=torch.float32, device=device),
            body_rotation=torch.empty(frame_count, body_count, 4, dtype=torch.float32, device=device),
            body_linear_velocity=torch.empty(frame_count, body_count, 3, dtype=torch.float32, device=device),
            body_angular_velocity=torch.empty(frame_count, body_count, 3, dtype=torch.float32, device=device),
        )

    def build_pose_frames(
        self,
        pose_axis_angle: torch.Tensor,
        root_translation: torch.Tensor,
        source_fps: float,
    ) -> MotionFrames:
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
            time_gaussian_filter(time_gradient(body_position.unsqueeze(0), step_seconds)).squeeze(0).contiguous()
        )
        body_angular_velocity = (
            time_gaussian_filter(time_quaternion_angular_velocity(body_rotation.unsqueeze(0), step_seconds))
            .squeeze(0)
            .contiguous()
        )

        difference = (joint_position_reference[1:] - joint_position_reference[:-1]) * source_fps
        if difference.shape[0] < 2:
            raise ValueError("G1 joint velocity construction requires at least three source frames.")
        joint_velocity_reference = torch.cat((difference, difference[-2:-1]), dim=0)
        return MotionFrames(
            joint_position=joint_position_reference.index_select(1, self._live_joint_from_reference_indices),
            joint_velocity=joint_velocity_reference.index_select(1, self._live_joint_from_reference_indices),
            body_position=body_position,
            body_rotation=body_rotation,
            body_linear_velocity=body_linear_velocity,
            body_angular_velocity=body_angular_velocity,
        )

    def build_frames(
        self,
        clip: MotionPoseAxisAngleClip,
        *,
        device: str | torch.device,
    ) -> MotionFrames:
        """Build one target pose-axis-angle clip through exact reference FK."""
        return self.build_pose_frames(
            torch.as_tensor(clip.pose_axis_angle, device=device),
            torch.as_tensor(clip.root_translation, device=device),
            clip.source_fps,
        )


def g1_pose_frame_builder(
    source_skeleton: MotionSkeleton,
    reference: NewtonKinematics,
    robot: Articulation,
) -> G1PoseFrameBuilder:
    """Build the target-pose trajectory edge from the live articulation and exact MJCF."""
    from ....kinematics import NewtonKinematics

    if not isinstance(reference, NewtonKinematics) or not reference.mjcf_path:
        raise TypeError("G1 frame construction requires an MJCF-backed NewtonKinematics reference.")
    target_tree = _g1_target_tree(reference)
    _validate_source_target_coordinates(source_skeleton, target_tree, G1_REFERENCE_MJCF_SHA256)
    return G1PoseFrameBuilder(
        target_tree=target_tree,
        pose_coordinate_identity_sha256=source_skeleton.identity_sha256,
        reference_kinematics=reference,
        reference_mjcf_sha256=G1_REFERENCE_MJCF_SHA256,
        live_joint_names=tuple(robot.joint_names),
        live_body_names=tuple(robot.body_names),
    )


_G1_TARGET_JOINT_LOCAL_BODY_NAMES = (
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


@dataclass(frozen=True, slots=True)
class G1LocalBodyPoseFrameBuilder:
    """Project parent-local body rotations into exact target-G1 frames."""

    source_skeleton: MotionSkeleton
    target_builder: G1PoseFrameBuilder
    projection: KinematicTreeRotationProjection
    target_tree_identity_sha256: str
    version: str = "g1_local_body_pose_ordered_hinge_fit_v1"
    construction_identity_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate source identity and freeze the complete source-to-target policy."""
        if self.projection.source_body_count != self.source_skeleton.num_bodies:
            raise ValueError("The SMPL projection and frame-builder source body counts differ.")
        if self.projection.target_tree != self.target_builder.target_tree:
            raise ValueError("The G1 projection and pose builder target trees differ.")
        projection_identity = canonical_sha256(
            {
                "policy": "ordered_orthogonal_hinge_fit_v1",
                "source_skeleton_sha256": self.source_skeleton.identity_sha256,
                "target_builder_sha256": self.target_builder.construction_identity_sha256,
                "target_skeleton_sha256": self.target_tree_identity_sha256,
                "source_root_body_index": self.projection.source_root_body_index,
                "target_joint_source_body_indices": self.projection.target_joint_source_body_indices,
                "joint_groups": self.projection.joint_groups,
            }
        )
        object.__setattr__(
            self,
            "construction_identity_sha256",
            canonical_sha256(
                {
                    "math_version": self.version,
                    "source_skeleton_sha256": self.source_skeleton.identity_sha256,
                    "projection_sha256": projection_identity,
                    "input_representation": "world_root_translation_and_parent_local_wxyz_v1",
                }
            ),
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Target-G1 articulation order of the output joint axis."""
        return self.target_builder.joint_names

    @property
    def reference_frame_names(self) -> tuple[str, ...]:
        """Target-G1 physical and derived reference-frame axis."""
        return self.target_builder.reference_frame_names

    def allocate(self, frame_count: int, *, device: str | torch.device) -> MotionFrames:
        """Allocate the exact target-G1 trajectory columns."""
        return self.target_builder.allocate(frame_count, device=device)

    def build_frames(
        self,
        clip: MotionLocalBodyPoseClip,
        *,
        device: str | torch.device,
    ) -> MotionFrames:
        """Build one local-body-pose clip as target-G1 trajectory facts."""
        root_translation = torch.as_tensor(clip.root_translation, device=device)
        local_rotation_wxyz = clip.local_body_rotation_wxyz(self.source_skeleton, device=device)
        pose_axis_angle = self.projection.project(local_rotation_wxyz)
        return self.target_builder.build_pose_frames(pose_axis_angle, root_translation, clip.source_fps)


def g1_local_body_pose_frame_builder(
    source_skeleton: MotionSkeleton,
    reference: NewtonKinematics,
    robot: Articulation,
) -> G1LocalBodyPoseFrameBuilder:
    """Build a local-body-pose projection from source and target owners."""
    from ....kinematics import NewtonKinematics

    if not isinstance(reference, NewtonKinematics) or not reference.mjcf_path:
        raise TypeError("G1 frame construction requires an MJCF-backed NewtonKinematics reference.")
    target_tree = _g1_target_tree(reference)
    target_tree_identity_sha256 = _g1_target_tree_identity(target_tree, G1_REFERENCE_MJCF_SHA256)
    target_builder = G1PoseFrameBuilder(
        target_tree=target_tree,
        pose_coordinate_identity_sha256=target_tree_identity_sha256,
        reference_kinematics=reference,
        reference_mjcf_sha256=G1_REFERENCE_MJCF_SHA256,
        live_joint_names=tuple(robot.joint_names),
        live_body_names=tuple(robot.body_names),
        version="g1_target_local_body_pose_projection_v1",
    )
    source_by_name = {name: index for index, name in enumerate(source_skeleton.body_names)}
    projection = KinematicTreeRotationProjection(
        source_body_count=source_skeleton.num_bodies,
        target_tree=target_tree,
        target_joint_source_body_indices=tuple(source_by_name[name] for name in _G1_TARGET_JOINT_LOCAL_BODY_NAMES),
        device=reference.device,
    )
    return G1LocalBodyPoseFrameBuilder(
        source_skeleton=source_skeleton,
        target_builder=target_builder,
        projection=projection,
        target_tree_identity_sha256=target_tree_identity_sha256,
    )
