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

from ....kinematics import (
    KinematicTree,
    time_forward_difference_segmented,
    time_gaussian_filter,
    time_gaussian_filter_segmented,
    time_gradient,
    time_gradient_segmented,
    time_quaternion_angular_velocity,
    time_quaternion_angular_velocity_segmented,
)
from ...data import MotionClipIndex, MotionFrames, MotionSkeleton
from ...identity import canonical_sha256, file_sha256, validate_sha256
from ...retarget import (
    MotionSemanticProjection,
    MotionSemanticTargets,
)
from .frames import (
    G1_HEAD_FRAME_NAME,
    G1_HEAD_OFFSET_M,
    G1_HEAD_PARENT_BODY_NAME,
    G1_HEAD_POSE_POLICY,
    append_g1_head_pose,
)

if TYPE_CHECKING:
    from ....kinematics import NewtonKinematics

_G1_REFERENCE_MJCF_RELATIVE_PATH = "humanoidverse/data/robots/g1/g1_29dof.xml"
G1_REFERENCE_MJCF_SHA256 = "439c1ec0806583d73b492da9484b0cb9e9eae215e0d9506e3c2fa69016733532"
G1_EXACT_COORDINATE_PROFILE_SHA256 = "b2cd2371cdc8cfff1caa1e2b502537a1d8be9fcbf42a6d3daf7b27c7880618ab"
"""Complete G1 axis-angle source-coordinate contract accepted by the exact route."""


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
    if tree.num_bodies != 30 or tree.num_coordinates != 29:
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
            "coordinate_axes": tree.coordinate_axes,
        }
    )


def _g1_coordinates_match(source: MotionSkeleton, target: KinematicTree) -> bool:
    """Return whether source rows already describe the exact target coordinates."""
    return (
        source.coordinate_identity_sha256 == G1_EXACT_COORDINATE_PROFILE_SHA256
        and source.joint_names == target.joint_names
        and source.body_names == target.body_names
        and source.parent_indices == target.parent_indices
        and source.joint_child_body_indices == target.joint_child_body_indices
        and source.joint_axes == target.coordinate_axes
        and source.root_translation_frame == "world"
        and source.root_rotation_convention == "axis_angle"
        and source.position_unit == "m"
        and source.angle_unit == "rad"
    )


@dataclass(frozen=True, slots=True)
class _G1TargetFrameBuilder:
    """Build simulator-ordered G1 frames from target-body pose-axis-angle coordinates."""

    target_tree: KinematicTree
    pose_coordinate_identity_sha256: str
    reference_kinematics: NewtonKinematics
    reference_mjcf_sha256: str
    live_joint_names: tuple[str, ...]
    live_body_names: tuple[str, ...]
    version: str = "g1_pose_exact_mjcf_v1"
    construction_identity_sha256: str = field(init=False)
    _live_joint_from_reference_indices: torch.Tensor = field(init=False, repr=False)
    _head_parent_body_index: int = field(init=False, repr=False)
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
                "exact_coordinate_profile_sha256": G1_EXACT_COORDINATE_PROFILE_SHA256,
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
        object.__setattr__(
            self,
            "_live_joint_from_reference_indices",
            torch.tensor(live_joint_from_reference, dtype=torch.int64, device=reference.device),
        )
        object.__setattr__(self, "_head_parent_body_index", head_parent)
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

    def build_generalized_frames(
        self,
        joint_q: torch.Tensor,
        source_fps: float,
    ) -> MotionFrames:
        """Build one target-G1 trajectory from free-root coordinates."""
        reference = self.reference_kinematics
        frame_count = joint_q.shape[0]
        if joint_q.shape != (frame_count, reference.model.joint_coord_count) or joint_q.dtype is not torch.float32:
            raise ValueError("G1 generalized positions differ from the exact reference MJCF.")
        if not math.isfinite(source_fps) or source_fps <= 0.0:
            raise ValueError("source_fps must be finite and positive [Hz].")

        joint_position_reference = joint_q[:, 7:]
        joint_qd = torch.zeros(
            frame_count,
            reference.model.joint_dof_count,
            dtype=torch.float32,
            device=joint_q.device,
        )
        body_q = torch.empty(
            frame_count,
            reference.model.body_count,
            7,
            dtype=torch.float32,
            device=joint_q.device,
        )
        body_qd_scratch = torch.empty(
            frame_count,
            reference.model.body_count,
            6,
            dtype=torch.float32,
            device=joint_q.device,
        )
        if (
            self._live_joint_from_reference_indices.device != joint_q.device
            or self._live_body_from_reference_indices.device != joint_q.device
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
        joint_position = joint_position_reference.index_select(1, self._live_joint_from_reference_indices)
        joint_velocity = joint_velocity_reference.index_select(1, self._live_joint_from_reference_indices)
        frames = MotionFrames(
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            body_position=body_position,
            body_rotation=body_rotation,
            body_linear_velocity=body_linear_velocity,
            body_angular_velocity=body_angular_velocity,
        )
        return frames

    def build_generalized_corpus(self, joint_q: torch.Tensor, clip_index: MotionClipIndex) -> MotionFrames:
        """Build one compact target-G1 corpus with segment-correct derivatives."""
        reference = self.reference_kinematics
        frame_count = joint_q.shape[0]
        if (
            joint_q.shape != (clip_index.total_frames, reference.model.joint_coord_count)
            or joint_q.dtype is not torch.float32
        ):
            raise ValueError("G1 semantic corpus coordinates must match the compact clip index and reference model.")
        offsets = torch.tensor(clip_index.offsets, dtype=torch.int64, device=joint_q.device)
        step_seconds = torch.tensor(
            [1.0 / clip.source_fps for clip in clip_index.clips], dtype=torch.float32, device=joint_q.device
        )
        if any(clip.frame_count < 3 for clip in clip_index.clips):
            raise ValueError("G1 semantic materialization requires at least three frames per segment.")
        if (
            self._live_joint_from_reference_indices.device != joint_q.device
            or self._live_body_from_reference_indices.device != joint_q.device
        ):
            raise ValueError("G1 trajectory tensors must use the reference-kinematics device.")

        joint_position_reference = joint_q[:, 7:]
        joint_qd = torch.zeros(frame_count, reference.model.joint_dof_count, dtype=torch.float32, device=joint_q.device)
        body_q = torch.empty(frame_count, reference.model.body_count, 7, dtype=torch.float32, device=joint_q.device)
        body_qd_scratch = torch.empty(
            frame_count, reference.model.body_count, 6, dtype=torch.float32, device=joint_q.device
        )
        reference.eval_fk_batched_torch(joint_q, joint_qd, body_q, body_qd_scratch)
        body_position = body_q[..., :3].index_select(1, self._live_body_from_reference_indices).contiguous()
        body_rotation = body_q[..., 3:].index_select(1, self._live_body_from_reference_indices).contiguous()
        body_position, body_rotation = append_g1_head_pose(
            body_position, body_rotation, parent_body_index=self._head_parent_body_index
        )
        body_linear_velocity = time_gaussian_filter_segmented(
            time_gradient_segmented(body_position, offsets, step_seconds), offsets
        ).contiguous()
        body_angular_velocity = time_gaussian_filter_segmented(
            time_quaternion_angular_velocity_segmented(body_rotation, offsets, step_seconds), offsets
        ).contiguous()
        joint_velocity_reference = time_forward_difference_segmented(joint_position_reference, offsets, step_seconds)
        return MotionFrames(
            joint_position=joint_position_reference.index_select(1, self._live_joint_from_reference_indices),
            joint_velocity=joint_velocity_reference.index_select(1, self._live_joint_from_reference_indices),
            body_position=body_position,
            body_rotation=body_rotation,
            body_linear_velocity=body_linear_velocity,
            body_angular_velocity=body_angular_velocity,
        )


_G1_RETARGET_TARGETS = (
    ("pelvis", "pelvis", -1),
    ("left_hip", "left_hip_yaw_link", 0),
    ("left_knee", "left_knee_link", 1),
    ("left_ankle", "left_ankle_roll_link", 2),
    ("right_hip", "right_hip_yaw_link", 0),
    ("right_knee", "right_knee_link", 4),
    ("right_ankle", "right_ankle_roll_link", 5),
    ("torso", "torso_link", 0),
    ("left_shoulder", "left_shoulder_yaw_link", 7),
    ("left_elbow", "left_elbow_link", 8),
    ("left_wrist", "left_wrist_yaw_link", 9),
    ("right_shoulder", "right_shoulder_yaw_link", 7),
    ("right_elbow", "right_elbow_link", 11),
    ("right_wrist", "right_wrist_yaw_link", 12),
)
_G1_ROOT_BASIS_ROLES = ("pelvis", "left_hip", "right_hip", "torso")
_G1_SUPPORT_ROLES = ("left_ankle", "right_ankle")
_G1_RETARGET_MATH_VERSION = "g1_semantic_landmark_newton_ik_v3"


@dataclass(frozen=True, slots=True)
class G1FrameBuilder:
    """One source-independent G1 builder exposing exact and semantic stages."""

    source_skeleton: MotionSkeleton
    target: _G1TargetFrameBuilder
    semantic: MotionSemanticProjection
    exact_coordinates: bool

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Live G1 joint order."""
        return self.target.joint_names

    @property
    def reference_frame_names(self) -> tuple[str, ...]:
        """Live and derived G1 reference-frame order."""
        return self.target.reference_frame_names

    @property
    def semantic_reference_kinematics(self) -> NewtonKinematics:
        """Exact G1 mechanics used by semantic IK."""
        return self.target.reference_kinematics

    @property
    def semantic_target_tree(self) -> KinematicTree:
        """Grouped G1 topology used to seed semantic IK."""
        return self.target.target_tree

    @property
    def version(self) -> str:
        """Selected coordinate-construction policy version."""
        return self.target.version if self.exact_coordinates else self.semantic.version

    @property
    def construction_identity_sha256(self) -> str:
        """Selected coordinate-construction identity."""
        return (
            self.target.construction_identity_sha256
            if self.exact_coordinates
            else self.semantic.construction_identity_sha256
        )

    def allocate(self, frame_count: int, *, device: str | torch.device) -> MotionFrames:
        """Allocate exact-capacity G1 trajectory columns."""
        return self.target.allocate(frame_count, device=device)

    def build_exact_coordinates(
        self,
        joint_q: torch.Tensor,
        joint_qd: torch.Tensor | None,
        source_fps: float,
    ) -> MotionFrames:
        """Materialize an exact-coordinate source clip."""
        del joint_qd
        return self.target.build_generalized_frames(joint_q, source_fps)

    def generate_semantic_targets(
        self,
        root_position: torch.Tensor,
        local_rotation_xyzw: torch.Tensor,
    ) -> MotionSemanticTargets:
        """Generate target-G1 semantic landmark tensors."""
        return self.semantic.generate_targets(root_position, local_rotation_xyzw)

    def build_semantic_corpus(self, joint_q: torch.Tensor, clip_index: MotionClipIndex) -> MotionFrames:
        """Materialize one compact solved target-G1 corpus."""
        return self.target.build_generalized_corpus(joint_q, clip_index)


def g1_frame_builder(
    source_skeleton: MotionSkeleton,
    reference: NewtonKinematics,
) -> G1FrameBuilder:
    """Build one G1 coordinate owner for exact and semantic sources."""
    from ....kinematics import NewtonKinematics

    if not isinstance(reference, NewtonKinematics) or not reference.mjcf_path:
        raise TypeError("G1 frame construction requires an MJCF-backed NewtonKinematics reference.")
    target_tree = _g1_target_tree(reference)
    target_tree_identity_sha256 = _g1_target_tree_identity(target_tree, G1_REFERENCE_MJCF_SHA256)
    target_builder = _G1TargetFrameBuilder(
        target_tree=target_tree,
        pose_coordinate_identity_sha256=target_tree_identity_sha256,
        reference_kinematics=reference,
        reference_mjcf_sha256=G1_REFERENCE_MJCF_SHA256,
        live_joint_names=tuple(reference.joint_names[1:]),
        live_body_names=tuple(reference.body_names),
        version="g1_target_local_body_pose_projection_v1",
    )
    semantic = MotionSemanticProjection(
        source_skeleton=source_skeleton,
        target=target_builder,
        target_landmarks=_G1_RETARGET_TARGETS,
        root_basis_roles=_G1_ROOT_BASIS_ROLES,
        support_roles=_G1_SUPPORT_ROLES,
        version=_G1_RETARGET_MATH_VERSION,
    )
    return G1FrameBuilder(
        source_skeleton=source_skeleton,
        target=target_builder,
        semantic=semantic,
        exact_coordinates=_g1_coordinates_match(source_skeleton, target_tree),
    )
