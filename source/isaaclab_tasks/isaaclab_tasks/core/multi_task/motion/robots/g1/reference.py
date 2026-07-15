# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact-MJCF G1 trajectory construction from declared pose representations."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import warp as wp

from ....kinematics import (
    KinematicTree,
    time_gaussian_filter,
    time_gaussian_filter_segmented,
    time_gradient,
    time_gradient_segmented,
    time_quaternion_angular_velocity,
    time_quaternion_angular_velocity_segmented,
)
from ...data import MotionClipIndex, MotionFrames, MotionSkeleton
from ...data.frames import (
    MotionGeneralizedCoordinates,
    MotionSourceProjection,
    MotionSourceProjectionExact,
    MotionSourceProjectionTrajectory,
)
from ...identity import canonical_sha256, file_sha256, validate_sha256
from ...retarget import (
    MotionTrajectoryProjection,
    _MotionContactChannel,
    _MotionContactPatch,
    _MotionTargetCalibration,
    _MotionTrajectoryTarget,
)
from ..target import MotionFrameTarget, motion_collision_probe_geometry
from .articulation import G1_SIMULATOR_BODY_NAMES, G1_SIMULATOR_JOINT_NAMES
from .frames import (
    G1_HEAD_FRAME_NAME,
    G1_HEAD_OFFSET_M,
    G1_HEAD_PARENT_BODY_NAME,
    G1_HEAD_POSE_POLICY,
    _g1_joint_velocity_canonical_warp,
    _time_forward_difference_segmented,
    append_g1_head_pose,
)

if TYPE_CHECKING:
    from ....kinematics import NewtonKinematics
    from ...data.source import MotionClipSource

_G1_REFERENCE_MJCF_RELATIVE_PATH = "humanoidverse/data/robots/g1/g1_29dof.xml"
G1_REFERENCE_MJCF_SHA256 = "439c1ec0806583d73b492da9484b0cb9e9eae215e0d9506e3c2fa69016733532"
G1_EXACT_COORDINATE_PROFILE_SHA256 = "b2cd2371cdc8cfff1caa1e2b502537a1d8be9fcbf42a6d3daf7b27c7880618ab"
"""Complete G1 axis-angle source-coordinate contract accepted by the exact route."""


def g1_reference_kinematics(reference_artifact_root: str, device: str | torch.device) -> NewtonKinematics:
    """Build the deprecated standalone BFM reference model.

    Use :meth:`NewtonKinematics.from_articulation` with the selected scene
    articulation for production target mechanics.

    Args:
        reference_artifact_root: Root containing the released BFM-Zero artifacts.
        device: Device for model state and kinematic evaluation.

    Returns:
        Standalone Newton mechanics for the released BFM-Zero G1 MJCF.
    """
    from ....kinematics import NewtonKinematics, NewtonKinematicsCfg

    warnings.warn(
        "g1_reference_kinematics() is deprecated; build NewtonKinematics from the selected scene articulation.",
        DeprecationWarning,
        stacklevel=2,
    )
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
_CANONICAL_VELOCITY_POLICY = "newton_com_destination_edge_bounded_scalar_hinge_v1"


def _validate_g1_canonical_layout(reference: NewtonKinematics, tree: KinematicTree) -> None:
    """Require one contiguous scalar coordinate and velocity per non-root G1 joint."""
    joint_count = len(reference.joint_names) - 1
    if (
        reference.n_root_coords != 7
        or reference.model.joint_coord_count != 7 + joint_count
        or reference.model.joint_dof_count != 6 + joint_count
        or tree.num_coordinates != joint_count
        or tree.joint_coordinate_ranges != tuple((index, index + 1) for index in range(joint_count))
        or tree.coordinate_q_indices != tuple(range(7, 7 + joint_count))
        or tree.coordinate_qd_indices != tuple(range(6, 6 + joint_count))
    ):
        raise ValueError("G1 canonical velocity requires one contiguous scalar coordinate and dof per non-root joint.")


def _g1_target_tree(reference: NewtonKinematics) -> KinematicTree:
    """Derive and validate the selected G1 scalar-hinge topology."""
    tree = KinematicTree.from_newton(reference)
    if tree.num_bodies != 30 or tree.num_coordinates != 29:
        raise ValueError("The target G1 model must contain 30 bodies and 29 hinge coordinates.")
    if tree.root_body_index != 0 or tree.joint_child_body_indices != tuple(range(1, 30)):
        raise ValueError("The target G1 model must expose root-first, one-body-per-hinge coordinates.")
    _validate_g1_canonical_layout(reference, tree)
    for lower, upper in zip(tree.coordinate_lower_limits_rad, tree.coordinate_upper_limits_rad, strict=True):
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper or upper - lower >= 2.0 * math.pi:
            raise ValueError("Every G1 hinge must use one finite non-periodic coordinate chart narrower than 2 pi.")
    return tree


def _g1_target_tree_identity(tree: KinematicTree, mechanics_sha256: str) -> str:
    """Hash target topology and selected-articulation provenance."""
    return canonical_sha256(
        {
            "mechanics_sha256": mechanics_sha256,
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
class _G1FrameTarget:
    """Build simulator-ordered G1 frames from target-body pose-axis-angle coordinates."""

    kinematic_tree: KinematicTree
    pose_coordinate_identity_sha256: str
    kinematics: NewtonKinematics
    reference_mechanics_sha256: str
    live_joint_names: tuple[str, ...]
    live_body_names: tuple[str, ...]
    contact_patches: tuple[_MotionContactPatch, ...]
    version: str = "g1_scene_articulation_coordinates_v4"
    coordinate_profile_sha256: str = field(init=False, default=G1_EXACT_COORDINATE_PROFILE_SHA256)
    construction_identity_sha256: str = field(init=False)
    trajectory_target: _MotionTrajectoryTarget = field(init=False, repr=False)
    joint_q_indices: tuple[int, ...] = field(init=False)
    """Newton generalized-position indices matching :attr:`joint_names`."""
    _live_joint_from_reference_indices: torch.Tensor = field(init=False, repr=False)
    _head_parent_body_index: int = field(init=False, repr=False)
    _live_body_from_reference_indices: torch.Tensor = field(init=False, repr=False)
    collision_probe_body_indices: torch.Tensor = field(init=False, repr=False)
    collision_probe_offsets_m: torch.Tensor = field(init=False, repr=False)
    collision_probe_contact_slots: torch.Tensor = field(init=False, repr=False)
    collision_probe_normal_channel_slots: torch.Tensor = field(init=False, repr=False)
    collision_geometry_identity_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Resolve live/reference ordering and freeze complete construction provenance."""
        reference = self.kinematics
        reference_joint_names = tuple(reference.joint_names[1:])
        reference_body_names = tuple(reference.body_names)
        _validate_g1_canonical_layout(reference, self.kinematic_tree)
        if self.kinematic_tree.joint_names != reference_joint_names:
            raise ValueError("The G1 target-coordinate order differs from the selected G1 articulation.")
        if self.kinematic_tree.body_names != reference_body_names:
            raise ValueError("The G1 target-body order differs from the selected G1 articulation.")
        if len(self.live_body_names) != len(reference_body_names) or set(self.live_body_names) != set(
            reference_body_names
        ):
            raise ValueError("The live G1 body names differ from the selected G1 articulation.")
        if self.live_body_names[0] != reference_body_names[0]:
            raise ValueError("The live G1 root body must remain first for root-state views.")
        if len(self.live_joint_names) != len(reference_joint_names) or set(self.live_joint_names) != set(
            reference_joint_names
        ):
            raise ValueError("The live G1 joint names differ from the selected G1 articulation.")
        live_joint_from_reference = tuple(reference_joint_names.index(name) for name in self.live_joint_names)
        live_joint_q_indices = tuple(
            self.kinematic_tree.coordinate_q_indices[index] for index in live_joint_from_reference
        )
        live_body_from_reference = tuple(reference_body_names.index(name) for name in self.live_body_names)
        head_parent = self.live_body_names.index(G1_HEAD_PARENT_BODY_NAME)
        validate_sha256("pose_coordinate_identity_sha256", self.pose_coordinate_identity_sha256)
        validate_sha256("reference_mechanics_sha256", self.reference_mechanics_sha256)
        identity = canonical_sha256(
            {
                "math_version": self.version,
                "exact_coordinate_profile_sha256": self.coordinate_profile_sha256,
                "pose_coordinate_identity_sha256": self.pose_coordinate_identity_sha256,
                "reference_mechanics_sha256": self.reference_mechanics_sha256,
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
                "trajectory_seed_law": "target_default_nonroot_with_active_root_v1",
                "derivative_policy": _DERIVATIVE_POLICY,
                "canonical_velocity_policy": _CANONICAL_VELOCITY_POLICY,
            }
        )
        object.__setattr__(self, "joint_q_indices", live_joint_q_indices)
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
        object.__setattr__(
            self,
            "trajectory_target",
            _MotionTrajectoryTarget(
                frame_target=self,
                landmarks=_G1_RETARGET_TARGETS,
                rotation_landmarks=_G1_RETARGET_ROTATIONS,
                direction_points=_G1_RETARGET_DIRECTION_POINTS,
                leg_chains=_G1_RETARGET_LEG_CHAINS,
                source_root_policy="optimized",
                initializer_policy="batched_frame_ik",
                required_position_roles=_G1_REQUIRED_POSITION_ROLES,
                required_direction_roles=_G1_REQUIRED_DIRECTION_ROLES,
                contact_patches=self.contact_patches,
                support_up_frame="root",
                version=_G1_RETARGET_MATH_VERSION,
            ),
        )
        probe_bodies, probe_offsets, probe_contact_slots, probe_normal_slots = motion_collision_probe_geometry(
            reference,
            self.trajectory_target.support_body_indices,
            self.trajectory_target.support_point_body_m,
            self.trajectory_target.support_channel_slots,
            self.trajectory_target.body_normal_channel_slots,
        )
        object.__setattr__(self, "collision_probe_body_indices", probe_bodies)
        object.__setattr__(self, "collision_probe_offsets_m", probe_offsets)
        object.__setattr__(self, "collision_probe_contact_slots", probe_contact_slots)
        object.__setattr__(self, "collision_probe_normal_channel_slots", probe_normal_slots)
        object.__setattr__(
            self,
            "collision_geometry_identity_sha256",
            canonical_sha256(
                {
                    "law": "target_collision_probes_per_body_v2",
                    "body_indices": tuple(int(value) for value in probe_bodies.detach().cpu().tolist()),
                    "offsets_m": tuple(
                        tuple(float(component) for component in row) for row in probe_offsets.detach().cpu()
                    ),
                    "contact_slots": tuple(int(value) for value in probe_contact_slots.detach().cpu().tolist()),
                    "normal_slots": tuple(int(value) for value in probe_normal_slots.detach().cpu().tolist()),
                }
            ),
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Live-articulation order of the output joint axis."""
        return self.live_joint_names

    @property
    def reference_frame_names(self) -> tuple[str, ...]:
        """Live physical bodies followed by BFM-Zero's derived head frame."""
        return (*self.live_body_names, G1_HEAD_FRAME_NAME)

    @property
    def materialization_minimum_frames(self) -> int:
        """Minimum frames required by the released G1 output law."""
        return 3

    def trajectory_seed_joint_q(
        self,
        *,
        root_position_m: torch.Tensor,
        rotation_body_indices: tuple[int, ...],
        landmark_rotation_xyzw: torch.Tensor,
    ) -> torch.Tensor:
        """Seed G1 at its default nonroot pose with the active source root."""
        frame_count = root_position_m.shape[0] if root_position_m.ndim == 2 else -1
        root_rotation = landmark_rotation_xyzw[0] if landmark_rotation_xyzw.ndim == 3 else landmark_rotation_xyzw
        rotation_norm = (
            torch.linalg.vector_norm(root_rotation, dim=-1)
            if root_rotation.shape == (frame_count, 4)
            else root_position_m.new_empty(0)
        )
        if (
            frame_count < 1
            or root_position_m.shape != (frame_count, 3)
            or rotation_body_indices != (self.kinematic_tree.root_body_index,)
            or landmark_rotation_xyzw.shape != (1, frame_count, 4)
            or root_position_m.dtype is not torch.float32
            or landmark_rotation_xyzw.dtype is not torch.float32
            or root_position_m.device != torch.device(self.kinematics.device)
            or landmark_rotation_xyzw.device != root_position_m.device
            or not bool(torch.all(torch.isfinite(root_position_m)))
            or not bool(torch.all(torch.isfinite(landmark_rotation_xyzw)))
            or not torch.allclose(rotation_norm, torch.ones_like(rotation_norm), atol=1.0e-5, rtol=1.0e-5)
        ):
            raise ValueError("G1 trajectory seed requires one finite unit target-root rotation per frame.")
        joint_q = torch.tensor(self.kinematics.default_joint_q, dtype=torch.float32, device=root_position_m.device)
        joint_q = joint_q.expand(frame_count, -1).clone()
        joint_q[:, :3].copy_(root_position_m)
        joint_q[:, 3:7].copy_(root_rotation)
        return joint_q

    def allocate_coordinates(self, frame_count: int, *, device: str | torch.device) -> MotionGeneralizedCoordinates:
        """Allocate exact-capacity target generalized coordinates."""
        return MotionGeneralizedCoordinates(
            torch.empty(
                (frame_count, self.kinematics.model.joint_coord_count),
                dtype=torch.float32,
                device=device,
            ),
            None,
        )

    def coordinates_from_newton(
        self, joint_q: torch.Tensor, clip_index: MotionClipIndex
    ) -> MotionGeneralizedCoordinates:
        """Convert one Newton-coordinate G1 corpus to generalized coordinates."""
        if joint_q.shape[0] != clip_index.total_frames:
            raise ValueError("G1 solved coordinates must match the compact clip index.")
        return MotionGeneralizedCoordinates(joint_q, None)

    def write_nonroot_velocity_canonical(
        self,
        joint_q: torch.Tensor,
        clip_offsets: torch.Tensor,
        step_seconds: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Write destination-indexed G1 scalar-hinge velocities [rad/s]."""
        frame_count = joint_q.shape[0]
        joint_count = self.kinematic_tree.num_coordinates
        segment_count = clip_offsets.shape[0] - 1
        if (
            joint_q.shape != (frame_count, self.kinematics.model.joint_coord_count)
            or output.shape != (frame_count, self.kinematics.model.joint_dof_count)
            or frame_count < 2
            or segment_count < 1
            or step_seconds.shape != (segment_count,)
            or joint_q.dtype is not torch.float32
            or output.dtype is not torch.float32
            or clip_offsets.dtype is not torch.int32
            or step_seconds.dtype is not torch.float32
            or any(tensor.device != joint_q.device for tensor in (clip_offsets, step_seconds, output))
            or not all(tensor.is_contiguous() for tensor in (joint_q, clip_offsets, step_seconds, output))
        ):
            raise ValueError("G1 canonical velocity requires aligned contiguous Newton trajectory tensors.")
        wp.launch(
            _g1_joint_velocity_canonical_warp,
            dim=(frame_count, joint_count),
            inputs=[
                wp.from_torch(joint_q),
                wp.from_torch(clip_offsets),
                wp.from_torch(step_seconds),
                segment_count,
                frame_count,
                joint_count,
            ],
            outputs=[wp.from_torch(output)],
            device=str(joint_q.device),
        )

    def write_joint_position_newton(self, coordinates: MotionGeneralizedCoordinates, output: torch.Tensor) -> None:
        """Write G1 positions [m or rad, depending on joint type] in Newton order.

        Args:
            coordinates: Stored G1 generalized coordinates.
            output: Newton generalized positions, shape [frame_count, coordinate_count].
        """
        if (
            output.shape != coordinates.joint_q.shape
            or output.dtype is not torch.float32
            or not output.is_contiguous()
            or output.device != coordinates.device
        ):
            raise ValueError("G1 Newton-coordinate output must match the stored position tensor.")
        output.copy_(coordinates.joint_q)

    def materialize_coordinates(
        self, coordinates: MotionGeneralizedCoordinates, clip_index: MotionClipIndex
    ) -> MotionFrames:
        """Materialize G1 frames once from final source-ordered coordinates."""
        if coordinates.frame_count != clip_index.total_frames or coordinates.joint_qd is not None:
            raise ValueError("G1 materialization requires position-only generalized coordinates.")
        return self.build_generalized_corpus(coordinates.joint_q, clip_index)

    def build_generalized_frames(
        self,
        joint_q: torch.Tensor,
        source_fps: float,
    ) -> MotionFrames:
        """Build one target-G1 trajectory from free-root coordinates."""
        reference = self.kinematics
        frame_count = joint_q.shape[0]
        if joint_q.shape != (frame_count, reference.model.joint_coord_count) or joint_q.dtype is not torch.float32:
            raise ValueError("G1 generalized positions differ from the selected G1 articulation.")
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
        reference = self.kinematics
        frame_count = joint_q.shape[0]
        if (
            joint_q.shape != (clip_index.total_frames, reference.model.joint_coord_count)
            or joint_q.dtype is not torch.float32
        ):
            raise ValueError("G1 trajectory coordinates must match the clip index and reference model.")
        offsets = torch.tensor(clip_index.offsets, dtype=torch.int64, device=joint_q.device)
        step_seconds = torch.tensor(
            [1.0 / clip.source_fps for clip in clip_index.clips], dtype=torch.float32, device=joint_q.device
        )
        if any(clip.frame_count < 3 for clip in clip_index.clips):
            raise ValueError("G1 trajectory_projection materialization requires at least three frames per segment.")
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
        joint_velocity_reference = torch.empty_like(joint_position_reference)
        _time_forward_difference_segmented(joint_position_reference, offsets, step_seconds, joint_velocity_reference)
        return MotionFrames(
            joint_position=joint_position_reference.index_select(1, self._live_joint_from_reference_indices),
            joint_velocity=joint_velocity_reference.index_select(1, self._live_joint_from_reference_indices),
            body_position=body_position,
            body_rotation=body_rotation,
            body_linear_velocity=body_linear_velocity,
            body_angular_velocity=body_angular_velocity,
        )


_G1_RETARGET_TARGETS = (
    _MotionTrajectoryTarget.Landmark("pelvis", "pelvis", -1, 1.0),
    _MotionTrajectoryTarget.Landmark("left_hip", "left_hip_pitch_link", 0, 1.0),
    _MotionTrajectoryTarget.Landmark("left_knee", "left_knee_link", 1, 1.0),
    _MotionTrajectoryTarget.Landmark("left_ankle", "left_ankle_roll_link", 2, 4.0),
    _MotionTrajectoryTarget.Landmark("right_hip", "right_hip_pitch_link", 0, 1.0),
    _MotionTrajectoryTarget.Landmark("right_knee", "right_knee_link", 4, 1.0),
    _MotionTrajectoryTarget.Landmark("right_ankle", "right_ankle_roll_link", 5, 4.0),
    _MotionTrajectoryTarget.Landmark("torso", "torso_link", 0, 1.0),
    _MotionTrajectoryTarget.Landmark("left_shoulder", "left_shoulder_pitch_link", 7, 1.0),
    _MotionTrajectoryTarget.Landmark("left_elbow", "left_elbow_link", 8, 1.0),
    _MotionTrajectoryTarget.Landmark("left_wrist", "left_wrist_yaw_link", 9, 1.0),
    _MotionTrajectoryTarget.Landmark("right_shoulder", "right_shoulder_pitch_link", 7, 1.0),
    _MotionTrajectoryTarget.Landmark("right_elbow", "right_elbow_link", 11, 1.0),
    _MotionTrajectoryTarget.Landmark("right_wrist", "right_wrist_yaw_link", 12, 1.0),
)
_G1_RETARGET_LEG_CHAINS = (
    _MotionTrajectoryTarget.LegChain(
        "left_foot", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", (1.0, 0.0, 0.0)
    ),
    _MotionTrajectoryTarget.LegChain(
        "right_foot", "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", (1.0, 0.0, 0.0)
    ),
)
# ProtoMotions' raw-keypoint conversion preserves global root orientation and solves nonroot joints from positions.
_G1_RETARGET_ROTATIONS = (_MotionTrajectoryTarget.RotationLandmark("pelvis", "pelvis", 2.0),)
# ProtoMotions gives ankle origins and fixed robot-local foot endpoints equal lower-body authority.
# fmt: off
_G1_RETARGET_DIRECTION_POINTS = (
    _MotionTrajectoryTarget.DirectionPoint(
        "left_foot", "left_ankle", "left_ankle", "left_toe", "between_positions",
        "left_ankle_roll_link",
        (0.15, 0.0, 0.0),
        4.0,
    ),
    _MotionTrajectoryTarget.DirectionPoint(
        "right_foot", "right_ankle", "right_ankle", "right_toe", "between_positions",
        "right_ankle_roll_link",
        (0.15, 0.0, 0.0),
        4.0,
    ),
    _MotionTrajectoryTarget.DirectionPoint(
        "left_hand", "left_wrist", "left_elbow", "left_wrist", "wrist_forward",
        "left_wrist_yaw_link",
        (0.0, 0.0, 0.14),
        1.0,
    ),
    _MotionTrajectoryTarget.DirectionPoint(
        "right_hand", "right_wrist", "right_elbow", "right_wrist", "wrist_forward",
        "right_wrist_yaw_link",
        (0.0, 0.0, 0.14),
        1.0,
    ),
    _MotionTrajectoryTarget.DirectionPoint(
        "left_hand_endpoint", "left_wrist", "left_elbow", "left_wrist", "between_positions",
        "left_wrist_yaw_link",
        (0.12, 0.0, 0.0),
        1.0,
    ),
    _MotionTrajectoryTarget.DirectionPoint(
        "right_hand_endpoint", "right_wrist", "right_elbow", "right_wrist", "between_positions",
        "right_wrist_yaw_link",
        (0.12, 0.0, 0.0),
        1.0,
    ),
)
# fmt: on
_G1_REQUIRED_POSITION_ROLES = ("pelvis", "left_ankle", "right_ankle", "left_wrist", "right_wrist")
_G1_REQUIRED_DIRECTION_ROLES = ("left_foot", "right_foot", "left_hand_endpoint", "right_hand_endpoint")
_G1_RETARGET_MATH_VERSION = "g1_trajectory_landmark_newton_ik_v65"


def _g1_exact_coordinates(
    joint_q: torch.Tensor, joint_qd: torch.Tensor | None, source_fps: float
) -> MotionGeneralizedCoordinates:
    """Convert one exact G1 source clip to target generalized coordinates."""
    del joint_qd, source_fps
    return MotionGeneralizedCoordinates(joint_q, None)


def g1_frame_target(
    reference: NewtonKinematics,
    contact_patches: tuple[_MotionContactPatch, ...],
    *,
    calibration_artifact_root: str = "",
    calibration: _MotionTargetCalibration | None = None,
) -> _G1FrameTarget:
    """Build one source-independent G1 coordinate and frame target."""
    from ....kinematics import NewtonKinematics

    del calibration_artifact_root
    if calibration is not None:
        raise ValueError("G1 target mechanics do not use a calibration artifact.")
    if not isinstance(reference, NewtonKinematics) or not reference.asset_path:
        raise TypeError("G1 frame construction requires scene-derived NewtonKinematics.")
    reference_mechanics_sha256 = reference.mechanics_identity_sha256
    target_tree = _g1_target_tree(reference)
    target_tree_identity_sha256 = _g1_target_tree_identity(target_tree, reference_mechanics_sha256)
    return _G1FrameTarget(
        kinematic_tree=target_tree,
        pose_coordinate_identity_sha256=target_tree_identity_sha256,
        kinematics=reference,
        reference_mechanics_sha256=reference_mechanics_sha256,
        live_joint_names=G1_SIMULATOR_JOINT_NAMES,
        live_body_names=G1_SIMULATOR_BODY_NAMES,
        contact_patches=contact_patches,
        version="g1_target_scene_articulation_v3",
    )


def g1_source_projection(
    source_skeleton: MotionSkeleton,
    target: MotionFrameTarget,
    source: MotionClipSource,
    contact_channels: tuple[_MotionContactChannel, ...],
    contact_channel_probe_offsets: torch.Tensor,
) -> MotionSourceProjection:
    """Build one exact or trajectory source projection into a G1 target."""
    del source
    if not isinstance(target, _G1FrameTarget):
        raise TypeError("G1 source projection requires a G1 frame target.")
    if source_skeleton.coordinate_identity_sha256 == target.coordinate_profile_sha256 and _g1_coordinates_match(
        source_skeleton, target.kinematic_tree
    ):
        return MotionSourceProjectionExact(
            source_skeleton=source_skeleton,
            target=target,
            version=target.version,
            construction_identity_sha256=target.construction_identity_sha256,
            convert_coordinates=_g1_exact_coordinates,
        )
    projection = MotionTrajectoryProjection(
        source_skeleton,
        target.trajectory_target,
        contact_channels,
        contact_channel_probe_offsets,
    )
    return MotionSourceProjectionTrajectory(
        source_skeleton=source_skeleton,
        target=target,
        version=projection.target.version,
        construction_identity_sha256=projection.construction_identity_sha256,
        target_projection=projection,
    )


_FRAME_BUILDER_MIGRATION = (
    "Configure MotionTaskTableCfg.TargetKinematicsCfg with g1_frame_target, g1_source_projection, "
    "and explicit contact_patches; the removed builder cannot infer that policy."
)


class G1FrameBuilder:
    """Deprecated public boundary for the removed composite G1 builder."""

    def __init__(
        self,
        source_skeleton: MotionSkeleton,
        target: object,
        semantic: object,
        exact_coordinates: bool,
    ) -> None:
        """Reject construction after reporting the explicit migration boundary.

        Args:
            source_skeleton: Former source skeleton.
            target: Former target-frame builder.
            semantic: Former semantic projection.
            exact_coordinates: Former exact-route selector.
        """
        del source_skeleton, target, semantic, exact_coordinates
        warnings.warn("G1FrameBuilder is deprecated. " + _FRAME_BUILDER_MIGRATION, DeprecationWarning, stacklevel=2)
        raise RuntimeError(_FRAME_BUILDER_MIGRATION)


def g1_frame_builder(source_skeleton: MotionSkeleton, reference: NewtonKinematics) -> G1FrameBuilder:
    """Reject the deprecated composite G1 builder and report its migration.

    Args:
        source_skeleton: Former source skeleton.
        reference: Former standalone target mechanics.

    Raises:
        RuntimeError: Always; the required contact policy is not part of the deprecated signature.
    """
    del source_skeleton, reference
    warnings.warn("g1_frame_builder() is deprecated. " + _FRAME_BUILDER_MIGRATION, DeprecationWarning, stacklevel=2)
    raise RuntimeError(_FRAME_BUILDER_MIGRATION)
