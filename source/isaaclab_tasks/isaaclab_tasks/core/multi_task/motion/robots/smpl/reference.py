# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact-MJCF SMPL trajectory construction from generalized coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import (
    convert_quat,
    quat_apply,
    quat_apply_inverse,
)

from isaaclab_assets.robots.smpl.smpl_constants import SMPL_HUMENV_MJCF_PATH, SMPL_HUMENV_MJCF_SHA256

from ....kinematics import (
    KinematicTree,
    time_gradient_segmented,
    time_quaternion_angular_velocity_segmented,
)
from ...data import MotionClipIndex, MotionFrames, MotionSkeleton
from ...identity import canonical_sha256, file_sha256, validate_sha256
from ...retarget import (
    MotionSemanticProjection,
    MotionSemanticTargets,
)
from .articulation import smpl_live_joint_mujoco_names

if TYPE_CHECKING:
    from ....kinematics import NewtonKinematics

_ROOT_STATE_POLICY = "free_root_origin_velocity_to_newton_com_velocity_v1"
SMPL_EXACT_COORDINATE_PROFILE_SHA256 = "2694fb6b394120bbbbf6166f0d206c3d37f629b2fc751ad19f9deb75a5232150"
"""Complete HumEnv SMPL generalized-coordinate contract accepted by the exact route."""


def smpl_reference_kinematics(reference_artifact_root: str, device: str | torch.device) -> NewtonKinematics:
    """Build the packaged hash-verified target-SMPL Newton model."""
    from ....kinematics import NewtonKinematics, NewtonKinematicsCfg

    del reference_artifact_root
    actual = file_sha256(SMPL_HUMENV_MJCF_PATH)
    if actual != SMPL_HUMENV_MJCF_SHA256:
        raise ValueError(f"SMPL reference MJCF hash differs: expected {SMPL_HUMENV_MJCF_SHA256}, got {actual}.")
    return NewtonKinematics(
        NewtonKinematicsCfg(
            usd_path=None,
            mjcf_path=str(SMPL_HUMENV_MJCF_PATH),
            device=str(device),
            collapse_fixed_joints=False,
        )
    )


@dataclass(frozen=True, slots=True)
class _SmplTargetFrameBuilder:
    """Materialize exact target-SMPL frames from generalized-coordinate tensors."""

    reference_kinematics: NewtonKinematics
    reference_mjcf_sha256: str
    live_joint_names: tuple[str, ...]
    live_body_names: tuple[str, ...]
    construction_identity_sha256: str = field(init=False)
    reference_coordinate_names: tuple[str, ...] = field(init=False)
    _target_tree: KinematicTree = field(init=False, repr=False)
    _live_from_reference_indices: torch.Tensor = field(init=False, repr=False)
    _body_com: torch.Tensor = field(init=False, repr=False)
    _live_body_from_reference_indices: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve the live articulation against the exact target reference once."""
        reference = self.reference_kinematics
        reference_body_names = tuple(reference.body_names)
        if (
            len(self.live_body_names) != len(reference_body_names)
            or self.live_body_names[0] != reference_body_names[0]
            or set(self.live_body_names) != set(reference_body_names)
        ):
            raise ValueError("The live SMPL bodies differ from the exact reference MJCF.")
        target_tree = KinematicTree.from_newton(reference)
        reference_body_names = target_tree.body_names
        coordinate_count = 3 * (len(reference_body_names) - 1)
        if (
            target_tree.num_bodies != len(reference_body_names)
            or target_tree.num_joints != len(reference_body_names) - 1
            or target_tree.num_coordinates != coordinate_count
            or target_tree.root_body_index != 0
        ):
            raise ValueError("The exact SMPL reference must expose one root and one three-coordinate joint per body.")
        if target_tree.coordinate_q_indices != tuple(range(7, 7 + coordinate_count)) or (
            target_tree.coordinate_qd_indices != tuple(range(6, 6 + coordinate_count))
        ):
            raise ValueError("The exact SMPL reference coordinates must follow its free root contiguously.")
        if sorted(target_tree.joint_child_body_indices) != list(range(1, len(reference_body_names))):
            raise ValueError("Every non-root SMPL body must own exactly one joint.")

        axis_names = {(1.0, 0.0, 0.0): "x", (0.0, 1.0, 0.0): "y", (0.0, 0.0, 1.0): "z"}
        reference_coordinate_names: list[str] = []
        for child_index, (coordinate_start, coordinate_stop) in zip(
            target_tree.joint_child_body_indices, target_tree.joint_coordinate_ranges
        ):
            if coordinate_stop - coordinate_start != 3:
                raise ValueError("Every non-root SMPL joint must expose three rotational coordinates.")
            coordinate_axes = target_tree.coordinate_axes[coordinate_start:coordinate_stop]
            try:
                coordinate_axis_names = tuple(axis_names[axis] for axis in coordinate_axes)
            except KeyError as error:
                raise ValueError("SMPL joint axes must be positive cardinal directions.") from error
            if set(coordinate_axis_names) != set("xyz"):
                raise ValueError("Every non-root SMPL joint must expose one positive X, Y, and Z axis.")
            reference_coordinate_names.extend(
                f"{reference_body_names[child_index]}_{axis_name}" for axis_name in coordinate_axis_names
            )
        reference_coordinate_names = tuple(reference_coordinate_names)
        canonical_coordinate_names = tuple(
            f"{body_name}_{axis}" for body_name in reference_body_names[1:] for axis in "xyz"
        )
        reference_from_canonical = tuple(canonical_coordinate_names.index(name) for name in reference_coordinate_names)

        live_reference_names = smpl_live_joint_mujoco_names(self.live_joint_names)
        if len(live_reference_names) != coordinate_count or set(live_reference_names) != set(
            reference_coordinate_names
        ):
            raise ValueError("The live SMPL joint coordinates differ from the exact reference MJCF.")
        live_from_reference = tuple(reference_coordinate_names.index(name) for name in live_reference_names)
        live_body_from_reference = tuple(reference_body_names.index(name) for name in self.live_body_names)
        body_com = torch.tensor(reference.topology.body_com, dtype=torch.float32, device=reference.device)
        if body_com.shape != (len(reference_body_names), 3) or body_com.dtype is not torch.float32:
            raise ValueError("The exact SMPL MJCF must expose one float32 center-of-mass offset per body.")
        if body_com.device != torch.device(reference.device):
            raise ValueError("The SMPL center-of-mass offsets must share the reference-kinematics device.")

        validate_sha256("reference_mjcf_sha256", self.reference_mjcf_sha256)
        object.__setattr__(self, "_body_com", body_com)
        object.__setattr__(self, "_target_tree", target_tree)
        object.__setattr__(self, "reference_coordinate_names", reference_coordinate_names)
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
                    "exact_coordinate_profile_sha256": SMPL_EXACT_COORDINATE_PROFILE_SHA256,
                    "reference_mjcf_sha256": self.reference_mjcf_sha256,
                    "reference_body_names": reference_body_names,
                    "reference_coordinate_names": reference_coordinate_names,
                    "reference_from_canonical": reference_from_canonical,
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

    @property
    def target_tree(self) -> KinematicTree:
        """Exact target-SMPL kinematic tree."""
        return self._target_tree

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
        coordinate_count = len(self.reference_coordinate_names)
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
        joint_position = generalized_position[:, 7:].index_select(1, self._live_from_reference_indices)
        joint_velocity = generalized_velocity[:, 6:].index_select(1, self._live_from_reference_indices)
        body_position = body_q[:, :, :3].index_select(1, body_indices)
        body_rotation = body_rotation.index_select(1, body_indices)
        body_linear_velocity = body_linear_velocity.index_select(1, body_indices)
        body_angular_velocity = body_qd[:, :, 3:].index_select(1, body_indices)
        frames = MotionFrames(
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            body_position=body_position,
            body_rotation=body_rotation,
            body_linear_velocity=body_linear_velocity,
            body_angular_velocity=body_angular_velocity,
        )
        return frames


_SMPL_RETARGET_TARGETS = (
    ("pelvis", "Pelvis", -1),
    ("left_hip", "L_Hip", 0),
    ("left_knee", "L_Knee", 1),
    ("left_ankle", "L_Ankle", 2),
    ("right_hip", "R_Hip", 0),
    ("right_knee", "R_Knee", 4),
    ("right_ankle", "R_Ankle", 5),
    ("torso", "Torso", 0),
    ("left_shoulder", "L_Shoulder", 7),
    ("left_elbow", "L_Elbow", 8),
    ("left_wrist", "L_Wrist", 9),
    ("right_shoulder", "R_Shoulder", 7),
    ("right_elbow", "R_Elbow", 11),
    ("right_wrist", "R_Wrist", 12),
)
_SMPL_ROOT_BASIS_ROLES = ("pelvis", "left_hip", "right_hip", "torso")
_SMPL_SUPPORT_ROLES = ("left_ankle", "right_ankle")
_SMPL_RETARGET_MATH_VERSION = "semantic_landmark_newton_ik_v4"
_SMPL_RETARGET_DERIVATIVE_POLICY = "first_order_edge_central_interior_no_smoothing_v1"


def _smpl_live_joint_names(reference: NewtonKinematics) -> tuple[str, ...]:
    """Return IsaacLab coordinate names from the grouped Newton joints."""
    joint_q_start = reference.topology.joint_q_start
    return tuple(
        f"{reference.joint_names[joint_index]}:{coordinate_index}"
        for joint_index in range(1, reference.topology.joint_count)
        for coordinate_index in range(int(joint_q_start[joint_index + 1]) - int(joint_q_start[joint_index]))
    )


def _smpl_coordinates_match(source: MotionSkeleton, target: _SmplTargetFrameBuilder) -> bool:
    """Return whether source rows already describe the exact target coordinates."""
    return (
        source.coordinate_identity_sha256 == SMPL_EXACT_COORDINATE_PROFILE_SHA256
        and source.body_names == tuple(target.reference_kinematics.body_names)
        and source.joint_names == target.reference_coordinate_names
        and source.root_translation_frame == "world"
        and source.root_rotation_convention == "wxyz"
        and source.position_unit == "m"
        and source.angle_unit == "rad"
    )


@dataclass(frozen=True, slots=True)
class SmplFrameBuilder:
    """One source-independent SMPL builder exposing exact and semantic stages."""

    source_skeleton: MotionSkeleton
    target: _SmplTargetFrameBuilder
    semantic: MotionSemanticProjection
    exact_coordinates: bool

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Live SMPL joint order."""
        return self.target.joint_names

    @property
    def reference_frame_names(self) -> tuple[str, ...]:
        """Live SMPL body-frame order."""
        return self.target.reference_frame_names

    @property
    def semantic_reference_kinematics(self) -> NewtonKinematics:
        """Exact SMPL mechanics used by semantic IK."""
        return self.target.reference_kinematics

    @property
    def semantic_target_tree(self) -> KinematicTree:
        """Grouped SMPL topology used to seed semantic IK."""
        return self.target.target_tree

    @property
    def version(self) -> str:
        """Selected coordinate-construction policy version."""
        return "smpl_generalized_coordinate_exact_mjcf_v1" if self.exact_coordinates else _SMPL_RETARGET_MATH_VERSION

    @property
    def construction_identity_sha256(self) -> str:
        """Selected coordinate-construction identity."""
        if not self.exact_coordinates:
            return canonical_sha256(
                {
                    "semantic_projection": self.semantic.construction_identity_sha256,
                    "derivative_policy": _SMPL_RETARGET_DERIVATIVE_POLICY,
                }
            )
        return canonical_sha256(
            {
                "math_version": self.version,
                "source_coordinates": self.source_skeleton.coordinate_identity_sha256,
                "target_construction": self.target.construction_identity_sha256,
            }
        )

    def allocate(self, frame_count: int, *, device: str | torch.device) -> MotionFrames:
        """Allocate exact-capacity SMPL trajectory columns."""
        return self.target.allocate(frame_count, device=device)

    def build_exact_coordinates(
        self,
        joint_q: torch.Tensor,
        joint_qd: torch.Tensor | None,
        source_fps: float,
    ) -> MotionFrames:
        """Materialize an exact-coordinate source clip."""
        del source_fps
        if joint_qd is None:
            raise ValueError("Exact SMPL coordinates require native generalized velocities.")
        generalized_position = torch.cat(
            (joint_q[:, :3], convert_quat(joint_q[:, 3:7], to="wxyz"), joint_q[:, 7:]), dim=-1
        )
        return self.target.build_generalized_frames(generalized_position, joint_qd)

    def generate_semantic_targets(
        self,
        root_position: torch.Tensor,
        local_rotation_xyzw: torch.Tensor,
    ) -> MotionSemanticTargets:
        """Generate target-SMPL semantic landmark tensors."""
        return self.semantic.generate_targets(root_position, local_rotation_xyzw)

    def build_semantic_corpus(self, joint_q: torch.Tensor, clip_index: MotionClipIndex) -> MotionFrames:
        """Materialize one compact solved target-SMPL corpus with segment-correct derivatives."""
        if joint_q.shape[0] != clip_index.total_frames or joint_q.dtype is not torch.float32:
            raise ValueError("SMPL semantic corpus coordinates must match the compact clip index.")
        offsets = torch.tensor(clip_index.offsets, dtype=torch.int64, device=joint_q.device)
        step_seconds = torch.tensor(
            [1.0 / clip.source_fps for clip in clip_index.clips], dtype=torch.float32, device=joint_q.device
        )
        root_position = joint_q[:, :3]
        root_rotation_xyzw = joint_q[:, 3:7]
        reference_coordinates = joint_q[:, 7:]
        root_linear_velocity = time_gradient_segmented(root_position, offsets, step_seconds)
        root_angular_velocity_world = time_quaternion_angular_velocity_segmented(
            root_rotation_xyzw, offsets, step_seconds
        )
        root_angular_velocity_local = quat_apply_inverse(root_rotation_xyzw, root_angular_velocity_world)
        joint_velocity = time_gradient_segmented(reference_coordinates, offsets, step_seconds)
        generalized_position = torch.cat(
            (root_position, convert_quat(root_rotation_xyzw, to="wxyz"), reference_coordinates), dim=-1
        )
        generalized_velocity = torch.cat((root_linear_velocity, root_angular_velocity_local, joint_velocity), dim=-1)
        return self.target.build_generalized_frames(generalized_position, generalized_velocity)


def smpl_frame_builder(
    source_skeleton: MotionSkeleton,
    reference: NewtonKinematics,
) -> SmplFrameBuilder:
    """Build one SMPL coordinate owner for exact and semantic sources."""
    from ....kinematics import NewtonKinematics

    if not isinstance(reference, NewtonKinematics) or not reference.mjcf_path:
        raise TypeError("SMPL frame construction requires an MJCF-backed NewtonKinematics reference.")
    reference_mjcf_sha256 = file_sha256(reference.mjcf_path)
    if reference_mjcf_sha256 != SMPL_HUMENV_MJCF_SHA256:
        raise ValueError("The injected SMPL target model differs from the packaged HumEnv coordinates.")
    live_joint_names = _smpl_live_joint_names(reference)
    live_body_names = tuple(reference.body_names)
    target = _SmplTargetFrameBuilder(reference, reference_mjcf_sha256, live_joint_names, live_body_names)
    semantic = MotionSemanticProjection(
        source_skeleton=source_skeleton,
        target=target,
        target_landmarks=_SMPL_RETARGET_TARGETS,
        root_basis_roles=_SMPL_ROOT_BASIS_ROLES,
        support_roles=_SMPL_SUPPORT_ROLES,
        version=_SMPL_RETARGET_MATH_VERSION,
    )
    return SmplFrameBuilder(
        source_skeleton=source_skeleton,
        target=target,
        semantic=semantic,
        exact_coordinates=_smpl_coordinates_match(source_skeleton, target),
    )
