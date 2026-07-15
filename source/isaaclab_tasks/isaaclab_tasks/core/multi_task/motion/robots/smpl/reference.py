# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact-MJCF SMPL trajectory construction from generalized coordinates."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import torch
import warp as wp

from isaaclab.utils.math import (
    convert_quat,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_matrix,
    quat_from_rotation_vector,
    quat_mul,
)

from isaaclab_assets.robots.smpl.smpl_constants import SMPL_HUMENV_MJCF_PATH, SMPL_HUMENV_MJCF_SHA256

from ....kinematics import (
    KinematicTree,
    fit_ordered_hinge_coordinates,
    time_gradient_segmented,
    time_quaternion_angular_velocity_segmented,
)
from ...data import MotionClipIndex, MotionFrames, MotionSkeleton
from ...data.frames import (
    MotionGeneralizedCoordinates,
    MotionSourceProjection,
    MotionSourceProjectionAnalytic,
    MotionSourceProjectionExact,
    MotionSourceProjectionTrajectory,
)
from ...data.smpl import SMPL_BODY_NAMES, SMPL_COMPATIBLE_POSE_PROFILE_SHA256, SmplLbsModel, load_smpl_lbs_model
from ...data.source import MotionClipSource, MotionSourceClip
from ...identity import canonical_sha256, file_sha256, validate_sha256
from ...retarget import (
    MotionTrajectoryProjection,
    _MotionContactChannel,
    _MotionContactPatch,
    _MotionTargetCalibration,
    _MotionTrajectoryTarget,
)
from ..target import MotionFrameTarget, motion_collision_probe_geometry
from .articulation import smpl_live_joint_mujoco_names
from .reference_warp import (
    smpl_joint_velocity_canonical_warp,
    smpl_joint_velocity_stored_warp,
    time_select_euler_xyz_branches_segmented_warp,
)

if TYPE_CHECKING:
    from ....kinematics import NewtonKinematics

_ROOT_STATE_POLICY = "free_root_origin_velocity_to_newton_com_velocity_v1"
_CANONICAL_VELOCITY_POLICY = "newton_com_destination_edge_ordered_d6_v1"
_SMPL_ANALYTIC_BRANCH_POLICY = "humenv_coupled_xyz_branch_two_pass_skip_tail_v1"
_SMPL_ANALYTIC_DERIVATIVE_POLICY = "humenv_root_body_angular_joint_coordinate_forward_repeat_tail_v1"
_SMPL_COORDINATE_PROFILE_SHA256 = "2694fb6b394120bbbbf6166f0d206c3d37f629b2fc751ad19f9deb75a5232150"
_SMPL_ANALYTIC_VERSION = "smpl_compatible_pose_to_humenv_coordinates_v3"
_SMPL_TARGET_FPS = 30.0


def _time_select_euler_xyz_branches_segmented(coordinates: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Apply the exact HumEnv equivalent-XYZ-branch recurrence per clip."""
    if (
        coordinates.ndim != 3
        or coordinates.shape[0] < 1
        or coordinates.shape[1] < 1
        or coordinates.shape[2] != 3
        or coordinates.dtype is not torch.float32
        or not coordinates.is_contiguous()
    ):
        raise ValueError("Euler branch selection requires contiguous float32 [frame, joint, xyz] coordinates.")
    if offsets.ndim != 1 or offsets.dtype is not torch.int64 or not offsets.is_contiguous() or offsets.shape[0] < 2:
        raise ValueError("Euler branch selection offsets must be contiguous int64 with at least two entries.")
    if offsets.device != coordinates.device:
        raise ValueError("Euler branch coordinates and offsets must share one device.")
    if int(offsets[0]) != 0 or int(offsets[-1]) != coordinates.shape[0] or bool(torch.any(offsets[1:] <= offsets[:-1])):
        raise ValueError("Euler branch offsets must strictly partition the complete frame axis.")

    wp.init()
    output = coordinates.clone()
    wp.launch(
        time_select_euler_xyz_branches_segmented_warp,
        dim=offsets.shape[0] - 1,
        inputs=[wp.from_torch(output), wp.from_torch(offsets), coordinates.shape[1]],
        device=str(output.device),
    )
    return output


@runtime_checkable
class _SmplCompatiblePoseClip(Protocol):
    """SMPL-family pose fields consumed by the target-owned analytic operator."""

    root_translation_m: np.ndarray
    local_axis_angle_rad: np.ndarray
    betas: np.ndarray
    gender: str
    source_fps: float

    @property
    def frame_count(self) -> int:
        """Number of native source frames."""


@runtime_checkable
class _SmplCompatiblePoseSource(Protocol):
    """Source mechanics required by the target-owned compatible-pose operator."""

    compatible_pose_profile_sha256: str

    def smpl_subject_model(
        self,
        skeleton_identity_sha256: str,
        device: str | torch.device,
    ) -> SmplLbsModel:
        """Return subject SMPL mechanics for one source skeleton."""


def smpl_reference_kinematics(reference_artifact_root: str, device: str | torch.device) -> NewtonKinematics:
    """Build the deprecated standalone target-SMPL model.

    Use :meth:`NewtonKinematics.from_articulation` with the selected scene
    articulation for production target mechanics.

    Args:
        reference_artifact_root: Deprecated artifact root; ignored for the packaged SMPL MJCF.
        device: Device for model state and kinematic evaluation.

    Returns:
        Standalone Newton mechanics for the packaged SMPL MJCF.
    """
    from ....kinematics import NewtonKinematics, NewtonKinematicsCfg

    warnings.warn(
        "smpl_reference_kinematics() is deprecated; build NewtonKinematics from the selected scene articulation.",
        DeprecationWarning,
        stacklevel=2,
    )
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
class _SmplFrameTarget:
    """Materialize selected target SMPL frames from generalized-coordinate tensors."""

    kinematics: NewtonKinematics
    reference_mechanics_sha256: str
    live_joint_names: tuple[str, ...]
    live_body_names: tuple[str, ...]
    contact_patches: tuple[_MotionContactPatch, ...]
    neutral_calibration_path: str | None = None
    neutral_calibration_artifact_sha256: str | None = None
    version: str = "smpl_scene_articulation_coordinates_v4"
    coordinate_profile_sha256: str = field(init=False, default=_SMPL_COORDINATE_PROFILE_SHA256)
    construction_identity_sha256: str = field(init=False)
    trajectory_target: _MotionTrajectoryTarget = field(init=False, repr=False)
    reference_coordinate_names: tuple[str, ...] = field(init=False)
    joint_q_indices: tuple[int, ...] = field(init=False)
    """Newton generalized-position indices matching :attr:`joint_names`."""
    _kinematic_tree: KinematicTree = field(init=False, repr=False)
    _coordinate_axes: torch.Tensor = field(init=False, repr=False)
    _live_from_reference_indices: torch.Tensor = field(init=False, repr=False)
    _body_com: torch.Tensor = field(init=False, repr=False)
    _live_body_from_reference_indices: torch.Tensor = field(init=False, repr=False)
    _reference_from_canonical_indices: torch.Tensor = field(init=False, repr=False)
    _target_default_local_rotation_xyzw: torch.Tensor = field(init=False, repr=False)
    _coordinate_lower_limits_rad: torch.Tensor = field(init=False, repr=False)
    _coordinate_upper_limits_rad: torch.Tensor = field(init=False, repr=False)
    _trajectory_seed_rotation_body_indices: tuple[int, ...] = field(init=False, repr=False)
    _hand_from_wrist_body_pairs: tuple[tuple[int, int], ...] = field(init=False, repr=False)
    collision_probe_body_indices: torch.Tensor = field(init=False, repr=False)
    collision_probe_offsets_m: torch.Tensor = field(init=False, repr=False)
    collision_probe_contact_slots: torch.Tensor = field(init=False, repr=False)
    collision_probe_normal_channel_slots: torch.Tensor = field(init=False, repr=False)
    collision_geometry_identity_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        """Resolve the live articulation against the selected target articulation once."""
        reference = self.kinematics
        reference_body_names = tuple(reference.body_names)
        if (
            len(self.live_body_names) != len(reference_body_names)
            or self.live_body_names[0] != reference_body_names[0]
            or set(self.live_body_names) != set(reference_body_names)
        ):
            raise ValueError("The live SMPL bodies differ from the selected SMPL articulation.")
        target_tree = KinematicTree.from_newton(reference)
        reference_body_names = target_tree.body_names
        coordinate_count = 3 * (len(reference_body_names) - 1)
        if (
            target_tree.num_bodies != len(reference_body_names)
            or target_tree.num_joints != len(reference_body_names) - 1
            or target_tree.num_coordinates != coordinate_count
            or target_tree.root_body_index != 0
        ):
            raise ValueError(
                "The selected SMPL articulation must expose one root and one three-coordinate joint per body."
            )
        if target_tree.coordinate_q_indices != tuple(range(7, 7 + coordinate_count)) or (
            target_tree.coordinate_qd_indices != tuple(range(6, 6 + coordinate_count))
        ):
            raise ValueError("The selected SMPL articulation coordinates must follow its free root contiguously.")
        if target_tree.joint_child_body_indices != tuple(range(1, len(reference_body_names))):
            raise ValueError("SMPL joints must own packaged child bodies in order 1 through 23.")

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
            if coordinate_axis_names != tuple("xyz"):
                raise ValueError("Every non-root SMPL joint must expose ordered positive X, Y, and Z axes.")
            reference_coordinate_names.extend(
                f"{reference_body_names[child_index]}_{axis_name}" for axis_name in coordinate_axis_names
            )
        reference_coordinate_names = tuple(reference_coordinate_names)
        canonical_coordinate_names = tuple(
            f"{body_name}_{axis}" for body_name in reference_body_names[1:] for axis in "xyz"
        )
        reference_from_canonical = tuple(canonical_coordinate_names.index(name) for name in reference_coordinate_names)
        if reference_from_canonical != tuple(range(coordinate_count)):
            raise ValueError("The SMPL reference and canonical coordinate orders must be identical.")
        default_joint_q = np.asarray(reference.default_joint_q)
        if default_joint_q.shape != (7 + coordinate_count,) or not np.all(np.isfinite(default_joint_q)):
            raise ValueError("The selected SMPL articulation must expose one finite default generalized pose.")
        if np.any(default_joint_q[7:] != 0.0):
            raise ValueError("The SMPL default non-root coordinates must be zero so the default pose is rest.")
        device = torch.device(reference.device)
        reference_from_canonical_indices = torch.tensor(reference_from_canonical, dtype=torch.int64, device=device)
        target_default_global_rotation = torch.tensor(
            reference.default_body_q[:, 3:7], dtype=torch.float32, device=device
        )
        target_default_global_rotation.div_(
            torch.linalg.vector_norm(target_default_global_rotation, dim=-1, keepdim=True)
        )
        target_default_local_rotation = target_default_global_rotation.clone()
        for body, parent in enumerate(target_tree.parent_indices[1:], start=1):
            target_default_local_rotation[body] = quat_mul(
                quat_conjugate(target_default_global_rotation[parent]), target_default_global_rotation[body]
            )
        coordinate_lower_limits = torch.tensor(
            target_tree.coordinate_lower_limits_rad, dtype=torch.float32, device=device
        )
        coordinate_upper_limits = torch.tensor(
            target_tree.coordinate_upper_limits_rad, dtype=torch.float32, device=device
        )
        trajectory_seed_rotation_body_indices = tuple(
            reference_body_names.index(item.body_name) for item in _SMPL_RETARGET_ROTATIONS
        )
        hand_from_wrist_body_pairs = (
            (reference_body_names.index("L_Hand"), reference_body_names.index("L_Wrist")),
            (reference_body_names.index("R_Hand"), reference_body_names.index("R_Wrist")),
        )
        mapped_bodies = set(trajectory_seed_rotation_body_indices)
        hand_bodies = {hand for hand, _ in hand_from_wrist_body_pairs}
        mapped_rows = {body: row for row, body in enumerate(trajectory_seed_rotation_body_indices)}
        root_body = target_tree.root_body_index
        if (
            len(trajectory_seed_rotation_body_indices) != len(reference_body_names) - len(hand_from_wrist_body_pairs)
            or len(mapped_bodies) != len(trajectory_seed_rotation_body_indices)
            or trajectory_seed_rotation_body_indices.count(root_body) != 1
            or trajectory_seed_rotation_body_indices[0] != root_body
            or len(hand_bodies) != len(hand_from_wrist_body_pairs)
            or not mapped_bodies.isdisjoint(hand_bodies)
            or mapped_bodies | hand_bodies != set(range(len(reference_body_names)))
        ):
            raise ValueError("The SMPL seed map must uniquely cover 22 mapped bodies plus two derived hands.")
        if any(
            wrist not in mapped_bodies or target_tree.parent_indices[hand] != wrist
            for hand, wrist in hand_from_wrist_body_pairs
        ) or any(
            body != root_body
            and (
                target_tree.parent_indices[body] not in mapped_rows
                or mapped_rows[target_tree.parent_indices[body]] >= mapped_rows[body]
            )
            for body in trajectory_seed_rotation_body_indices
        ):
            raise ValueError("The SMPL seed map must follow the target parent-before-child topology.")
        target_default_local_rotation_values = tuple(
            tuple(float(component) for component in rotation)
            for rotation in target_default_local_rotation.detach().cpu().tolist()
        )

        live_reference_names = smpl_live_joint_mujoco_names(self.live_joint_names)
        if len(live_reference_names) != coordinate_count or set(live_reference_names) != set(
            reference_coordinate_names
        ):
            raise ValueError("The live SMPL joint coordinates differ from the selected SMPL articulation.")
        live_from_reference = tuple(reference_coordinate_names.index(name) for name in live_reference_names)
        live_joint_q_indices = tuple(target_tree.coordinate_q_indices[index] for index in live_from_reference)
        live_body_from_reference = tuple(reference_body_names.index(name) for name in self.live_body_names)
        body_com = torch.tensor(reference.topology.body_com, dtype=torch.float32, device=reference.device)
        coordinate_axes = torch.tensor(target_tree.coordinate_axes, dtype=torch.float32, device=reference.device).view(
            target_tree.num_joints, 3, 3
        )
        if body_com.shape != (len(reference_body_names), 3) or body_com.dtype is not torch.float32:
            raise ValueError("The selected SMPL articulation must expose one float32 center-of-mass offset per body.")
        if body_com.device != torch.device(reference.device):
            raise ValueError("The SMPL center-of-mass offsets must share the reference-kinematics device.")

        validate_sha256("reference_mechanics_sha256", self.reference_mechanics_sha256)
        if (self.neutral_calibration_path is None) != (self.neutral_calibration_artifact_sha256 is None):
            raise ValueError("SMPL neutral calibration path and digest must be declared together.")
        if self.neutral_calibration_artifact_sha256 is not None:
            validate_sha256("neutral_calibration_artifact_sha256", self.neutral_calibration_artifact_sha256)
        object.__setattr__(self, "joint_q_indices", live_joint_q_indices)
        object.__setattr__(self, "_body_com", body_com)
        object.__setattr__(self, "_coordinate_axes", coordinate_axes)
        object.__setattr__(self, "_kinematic_tree", target_tree)
        object.__setattr__(self, "_reference_from_canonical_indices", reference_from_canonical_indices)
        object.__setattr__(self, "_target_default_local_rotation_xyzw", target_default_local_rotation)
        object.__setattr__(self, "_coordinate_lower_limits_rad", coordinate_lower_limits)
        object.__setattr__(self, "_coordinate_upper_limits_rad", coordinate_upper_limits)
        object.__setattr__(self, "_trajectory_seed_rotation_body_indices", trajectory_seed_rotation_body_indices)
        object.__setattr__(self, "_hand_from_wrist_body_pairs", hand_from_wrist_body_pairs)
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
                    "exact_coordinate_profile_sha256": self.coordinate_profile_sha256,
                    "reference_mechanics_sha256": self.reference_mechanics_sha256,
                    "reference_body_names": reference_body_names,
                    "reference_coordinate_names": reference_coordinate_names,
                    "reference_from_canonical": reference_from_canonical,
                    "trajectory_seed": {
                        "law": "mapped_global_to_target_rest_local_ordered_xyz_v1",
                        "rotation_body_indices": trajectory_seed_rotation_body_indices,
                        "hand_from_wrist_body_pairs": hand_from_wrist_body_pairs,
                        "target_default_local_rotation_xyzw": target_default_local_rotation_values,
                        "reference_from_canonical": reference_from_canonical,
                        "branch_policy": "complete_clip_humenv_equivalent_xyz",
                        "limit_policy": "clamp_nonroot_seed_only",
                    },
                    "live_body_names": self.live_body_names,
                    "live_joint_names": self.live_joint_names,
                    "live_body_from_reference": live_body_from_reference,
                    "live_from_reference": live_from_reference,
                    "root_state_policy": _ROOT_STATE_POLICY,
                    "canonical_velocity_policy": _CANONICAL_VELOCITY_POLICY,
                    "frame_policy": "target_smpl_physical_body_fields_v1",
                }
            ),
        )
        object.__setattr__(
            self,
            "trajectory_target",
            _MotionTrajectoryTarget(
                frame_target=self,
                landmarks=_SMPL_RETARGET_TARGETS,
                rotation_landmarks=_SMPL_RETARGET_ROTATIONS,
                direction_points=_SMPL_RETARGET_DIRECTION_POINTS,
                leg_chains=_SMPL_RETARGET_LEG_CHAINS,
                source_root_policy="fixed",
                initializer_policy="batched_frame_ik",
                required_position_roles=_SMPL_REQUIRED_POSITION_ROLES,
                required_direction_roles=_SMPL_REQUIRED_DIRECTION_ROLES,
                contact_patches=self.contact_patches,
                support_up_frame="anatomy",
                version=_SMPL_RETARGET_MATH_VERSION,
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

    def neutral_calibration_model(self) -> SmplLbsModel:
        """Load the target-owned neutral HumEnv calibration mechanics."""
        if self.neutral_calibration_path is None or self.neutral_calibration_artifact_sha256 is None:
            raise ValueError("Analytic SMPL conversion requires target-owned neutral calibration mechanics.")
        model = load_smpl_lbs_model(
            self.neutral_calibration_path,
            artifact_sha256=self.neutral_calibration_artifact_sha256,
            device=self.kinematics.device,
        )
        if model.gender != "neutral":
            raise ValueError("SMPL target calibration must use the neutral compact body model.")
        return model

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Live-articulation order of the output joint axis."""
        return self.live_joint_names

    @property
    def reference_frame_names(self) -> tuple[str, ...]:
        """Live-articulation order of physical reference frames."""
        return self.live_body_names

    @property
    def kinematic_tree(self) -> KinematicTree:
        """Exact target-SMPL kinematic tree."""
        return self._kinematic_tree

    @property
    def materialization_minimum_frames(self) -> int:
        """Minimum frames required by SMPL output velocity materialization."""
        return 2

    def trajectory_seed_joint_q(
        self,
        *,
        root_position_m: torch.Tensor,
        rotation_body_indices: tuple[int, ...],
        landmark_rotation_xyzw: torch.Tensor,
    ) -> torch.Tensor:
        """Seed SMPL from root-only evidence or the exact ProtoMotions 22-body map."""
        frame_count = root_position_m.shape[0] if root_position_m.ndim == 2 else -1
        rotation_count = len(rotation_body_indices)
        root_rotation = landmark_rotation_xyzw[0] if landmark_rotation_xyzw.ndim == 3 else landmark_rotation_xyzw
        rotation_norm = (
            torch.linalg.vector_norm(landmark_rotation_xyzw, dim=-1)
            if landmark_rotation_xyzw.shape == (rotation_count, frame_count, 4)
            else root_position_m.new_empty(0)
        )
        root_only = (self.kinematic_tree.root_body_index,)
        full_body = self._trajectory_seed_rotation_body_indices
        if (
            frame_count < 1
            or root_position_m.shape != (frame_count, 3)
            or rotation_body_indices not in (root_only, full_body)
            or landmark_rotation_xyzw.shape != (rotation_count, frame_count, 4)
            or root_position_m.dtype is not torch.float32
            or landmark_rotation_xyzw.dtype is not torch.float32
            or root_position_m.device != torch.device(self.kinematics.device)
            or landmark_rotation_xyzw.device != root_position_m.device
            or not bool(torch.all(torch.isfinite(root_position_m)))
            or not bool(torch.all(torch.isfinite(landmark_rotation_xyzw)))
            or not torch.allclose(rotation_norm, torch.ones_like(rotation_norm), atol=1.0e-5, rtol=1.0e-5)
        ):
            raise ValueError("SMPL trajectory seed requires root-only or exact 22-body finite unit rotations.")

        joint_q = torch.tensor(self.kinematics.default_joint_q, dtype=torch.float32, device=root_position_m.device)
        joint_q = joint_q.expand(frame_count, -1).clone()
        if rotation_body_indices == full_body:
            global_rotation = torch.empty(
                frame_count,
                self.kinematic_tree.num_bodies,
                4,
                dtype=torch.float32,
                device=root_position_m.device,
            )
            global_rotation[:, list(full_body)] = landmark_rotation_xyzw.transpose(0, 1)
            for hand, wrist in self._hand_from_wrist_body_pairs:
                global_rotation[:, hand].copy_(global_rotation[:, wrist])

            local_rotation = global_rotation.clone()
            for body, parent in enumerate(self.kinematic_tree.parent_indices[1:], start=1):
                local_rotation[:, body] = quat_mul(quat_conjugate(global_rotation[:, parent]), global_rotation[:, body])
            target_default_local = self._target_default_local_rotation_xyzw[None, 1:].expand(frame_count, -1, -1)
            joint_delta = quat_mul(quat_conjugate(target_default_local), local_rotation[:, 1:])
            principal, _ = fit_ordered_hinge_coordinates(
                joint_delta, torch.eye(3, dtype=torch.float32, device=root_position_m.device)
            )
            offsets = torch.tensor((0, frame_count), dtype=torch.int64, device=root_position_m.device)
            coordinate = _time_select_euler_xyz_branches_segmented(principal.contiguous(), offsets)
            coordinate = coordinate.flatten(1).index_select(1, self._reference_from_canonical_indices)
            coordinate.clamp_(self._coordinate_lower_limits_rad, self._coordinate_upper_limits_rad)
            joint_q[:, 7:].copy_(coordinate)

        joint_q[:, :3].copy_(root_position_m)
        joint_q[:, 3:7].copy_(root_rotation)
        return joint_q

    def allocate_coordinates(self, frame_count: int, *, device: str | torch.device) -> MotionGeneralizedCoordinates:
        """Allocate exact-capacity target generalized coordinates."""

        coordinate_count = len(self.reference_coordinate_names)
        return MotionGeneralizedCoordinates(
            torch.empty((frame_count, 7 + coordinate_count), dtype=torch.float32, device=device),
            torch.empty((frame_count, 6 + coordinate_count), dtype=torch.float32, device=device),
        )

    def coordinates_from_newton(
        self, joint_q: torch.Tensor, clip_index: MotionClipIndex
    ) -> MotionGeneralizedCoordinates:
        """Convert one Newton-coordinate target-SMPL corpus to generalized coordinates."""
        if joint_q.shape[0] != clip_index.total_frames or joint_q.dtype is not torch.float32:
            raise ValueError("SMPL trajectory corpus coordinates must match the compact clip index.")
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
        joint_velocity = torch.empty_like(reference_coordinates)
        joint_count = self._coordinate_axes.shape[0]
        wp.launch(
            smpl_joint_velocity_stored_warp,
            dim=(joint_q.shape[0], joint_count),
            inputs=[
                wp.from_torch(joint_q),
                wp.from_torch(self._coordinate_axes),
                wp.from_torch(offsets),
                wp.from_torch(step_seconds),
                len(clip_index.clips),
                joint_q.shape[0],
                joint_count,
            ],
            outputs=[wp.from_torch(joint_velocity)],
            device=str(joint_q.device),
        )
        generalized_position = torch.cat(
            (root_position, convert_quat(root_rotation_xyzw, to="wxyz"), reference_coordinates), dim=-1
        )
        generalized_velocity = torch.cat((root_linear_velocity, root_angular_velocity_local, joint_velocity), dim=-1)
        return MotionGeneralizedCoordinates(generalized_position, generalized_velocity)

    def write_nonroot_velocity_canonical(
        self,
        joint_q: torch.Tensor,
        clip_offsets: torch.Tensor,
        step_seconds: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Write destination-indexed ordered-D6 target velocities [rad/s]."""
        frame_count = joint_q.shape[0]
        joint_count = self._coordinate_axes.shape[0]
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
            raise ValueError("SMPL canonical velocity requires aligned contiguous Newton trajectory tensors.")
        wp.launch(
            smpl_joint_velocity_canonical_warp,
            dim=(frame_count, joint_count),
            inputs=[
                wp.from_torch(joint_q),
                wp.from_torch(self._coordinate_axes),
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
        """Write SMPL positions [m or rad, depending on joint type] in Newton order.

        Args:
            coordinates: Stored SMPL generalized coordinates with wxyz root rotation.
            output: Newton positions with xyzw root rotation, shape [frame_count, coordinate_count].
        """
        if (
            output.shape != coordinates.joint_q.shape
            or output.dtype is not torch.float32
            or not output.is_contiguous()
            or output.device != coordinates.device
        ):
            raise ValueError("SMPL Newton-coordinate output must match the stored position tensor.")
        output.copy_(coordinates.joint_q)
        output[:, 3:7].copy_(convert_quat(coordinates.joint_q[:, 3:7], to="xyzw"))

    def materialize_coordinates(
        self, coordinates: MotionGeneralizedCoordinates, clip_index: MotionClipIndex
    ) -> MotionFrames:
        """Materialize SMPL frames once from final source-ordered coordinates."""
        if coordinates.frame_count != clip_index.total_frames or coordinates.joint_qd is None:
            raise ValueError("SMPL materialization requires complete generalized positions and velocities.")
        return self.build_generalized_frames(coordinates.joint_q, coordinates.joint_qd)

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
            raise ValueError("Generalized-coordinate tensors differ from the selected SMPL articulation widths.")
        if generalized_position.dtype is not torch.float32 or generalized_velocity.dtype is not torch.float32:
            raise ValueError("SMPL generalized-coordinate tensors must use float32.")
        if generalized_position.device != generalized_velocity.device:
            raise ValueError("SMPL generalized positions and velocities must share one device.")
        if (
            self._live_from_reference_indices.device != generalized_position.device
            or self._body_com.device != generalized_position.device
        ):
            raise ValueError("SMPL trajectory tensors must use the reference-kinematics device.")

        reference = self.kinematics
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


_SMPL_RETARGET_TARGET_ROWS = (
    ("pelvis", "Pelvis", -1),
    ("left_hip", "L_Hip", 0),
    ("left_knee", "L_Knee", 1),
    ("left_ankle", "L_Ankle", 2),
    ("left_toe", "L_Toe", 3),
    ("right_hip", "R_Hip", 0),
    ("right_knee", "R_Knee", 5),
    ("right_ankle", "R_Ankle", 6),
    ("right_toe", "R_Toe", 7),
    ("torso", "Torso", 0),
    ("spine", "Spine", 9),
    ("chest", "Chest", 10),
    ("neck", "Neck", 11),
    ("head", "Head", 12),
    ("left_thorax", "L_Thorax", 11),
    ("left_shoulder", "L_Shoulder", 14),
    ("left_elbow", "L_Elbow", 15),
    ("left_wrist", "L_Wrist", 16),
    ("right_thorax", "R_Thorax", 11),
    ("right_shoulder", "R_Shoulder", 18),
    ("right_elbow", "R_Elbow", 19),
    ("right_wrist", "R_Wrist", 20),
)
_SMPL_RETARGET_TARGETS = tuple(
    _MotionTrajectoryTarget.Landmark(role, body_name, parent_row, 1.0)
    for role, body_name, parent_row in _SMPL_RETARGET_TARGET_ROWS
)
_SMPL_RETARGET_LEG_CHAINS = (
    _MotionTrajectoryTarget.LegChain("left_foot", "L_Hip", "L_Knee", "L_Ankle", (1.0, 0.0, 0.0)),
    _MotionTrajectoryTarget.LegChain("right_foot", "R_Hip", "R_Knee", "R_Ankle", (1.0, 0.0, 0.0)),
)
# ProtoMotions matches all 22 mapped body globals while fixing the root; keep uniform target role weights.
_SMPL_RETARGET_ROTATIONS = tuple(
    _MotionTrajectoryTarget.RotationLandmark(role, body_name, 10.0 if row == 0 else 1.0)
    for row, (role, body_name, _) in enumerate(_SMPL_RETARGET_TARGET_ROWS)
)
# fmt: off
_SMPL_RETARGET_DIRECTION_POINTS = (
    _MotionTrajectoryTarget.DirectionPoint(
        "left_foot", "left_ankle", "left_ankle", "left_toe", "between_positions",
        "L_Toe",
        (0.0, 0.0, 0.0),
        1.0,
    ),
    _MotionTrajectoryTarget.DirectionPoint(
        "right_foot", "right_ankle", "right_ankle", "right_toe", "between_positions",
        "R_Toe",
        (0.0, 0.0, 0.0),
        1.0,
    ),
)
_SMPL_REQUIRED_POSITION_ROLES = (
    "pelvis", "head", "left_ankle", "right_ankle", "left_toe", "right_toe", "left_wrist", "right_wrist"
)
# fmt: on
_SMPL_REQUIRED_DIRECTION_ROLES = ("left_foot", "right_foot")
_SMPL_RETARGET_MATH_VERSION = "trajectory_landmark_newton_ik_v60"
_SMPL_RETARGET_DERIVATIVE_POLICY = "solver_and_storage_ordered_d6_edge_rates_v3"


def _smpl_live_joint_names(reference: NewtonKinematics) -> tuple[str, ...]:
    """Return IsaacLab coordinate names from the grouped Newton joints."""
    joint_q_start = reference.topology.joint_q_start
    return tuple(
        f"{reference.joint_names[joint_index]}:{coordinate_index}"
        for joint_index in range(1, reference.topology.joint_count)
        for coordinate_index in range(int(joint_q_start[joint_index + 1]) - int(joint_q_start[joint_index]))
    )


def _smpl_coordinates_match(source: MotionSkeleton, target: _SmplFrameTarget) -> bool:
    """Return whether source rows already describe the exact target coordinates."""
    return (
        source.coordinate_identity_sha256 == target.coordinate_profile_sha256
        and source.body_names == tuple(target.kinematics.body_names)
        and source.joint_names == target.reference_coordinate_names
        and source.root_translation_frame == "world"
        and source.root_rotation_convention == "wxyz"
        and source.position_unit == "m"
        and source.angle_unit == "rad"
    )


def _compatible_pose_output_index(source: MotionClipIndex) -> MotionClipIndex:
    """Return the target 30 Hz output clock for compatible-pose clips."""
    clips = []
    for clip in source.clips:
        sample_ratio = clip.source_fps / _SMPL_TARGET_FPS
        stride = round(sample_ratio)
        if stride < 1 or not math.isclose(sample_ratio, stride, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError("Compatible-pose source_fps must be an integer multiple of 30 Hz.")
        if clip.source_clip_id is not None or clip.source_frame_start != 0:
            raise ValueError("Analytic compatible-pose conversion requires original full-span clips.")
        clips.append(
            dataclass_replace(
                clip,
                frame_count=(clip.frame_count + stride - 1) // stride,
                source_fps=_SMPL_TARGET_FPS,
            )
        )
    return MotionClipIndex(
        source_content_sha256=source.source_content_sha256,
        skeleton_identity_sha256s=source.skeleton_identity_sha256s,
        clips=tuple(clips),
    )


def _compatible_pose_coordinates(
    clip: MotionSourceClip,
    *,
    subject_model: SmplLbsModel,
    neutral_model: SmplLbsModel,
    target_body_names: tuple[str, ...],
    target_coordinate_names: tuple[str, ...],
) -> MotionGeneralizedCoordinates:
    """Convert one compatible SMPL-family pose to target generalized coordinates."""
    if not isinstance(clip, _SmplCompatiblePoseClip):
        raise TypeError("The analytic SMPL route requires a compatible SMPL-family pose clip.")
    if subject_model.gender != clip.gender or neutral_model.gender != "neutral":
        raise ValueError("Compatible-pose subject and neutral mechanics have incompatible genders.")
    if subject_model.device != neutral_model.device:
        raise ValueError("Compatible-pose subject and neutral mechanics must share one device.")
    if set(target_body_names) != set(SMPL_BODY_NAMES) or target_body_names[0] != SMPL_BODY_NAMES[0]:
        raise ValueError("Target SMPL body names differ from the compatible-pose profile.")

    sample_ratio = clip.source_fps / _SMPL_TARGET_FPS
    stride = round(sample_ratio)
    if stride < 1 or not math.isclose(sample_ratio, stride, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("Compatible-pose source_fps must be an integer multiple of 30 Hz.")
    sampled_translation = clip.root_translation_m[::stride]
    sampled_pose = clip.local_axis_angle_rad[::stride, :22]
    if sampled_translation.shape[0] < 2:
        raise ValueError("Analytic compatible-pose conversion requires at least two target-rate frames.")

    translation = torch.as_tensor(
        np.array(sampled_translation, copy=True, order="C"),
        dtype=torch.float32,
        device=neutral_model.device,
    ).clone()
    pose = torch.as_tensor(
        np.array(sampled_pose, copy=True, order="C"),
        dtype=torch.float32,
        device=neutral_model.device,
    )
    pose = torch.cat((pose, torch.zeros((pose.shape[0], 2, 3), dtype=torch.float32, device=pose.device)), dim=1)

    subject_betas = torch.as_tensor(clip.betas[:10].copy(), dtype=torch.float32, device=pose.device).unsqueeze(0)
    subject_ground = subject_model.vertices(pose[:1], subject_betas, translation[:1])[..., 2].amin()
    translation[:, 2].sub_(subject_ground)

    neutral_betas = torch.zeros((1, 10), dtype=torch.float32, device=pose.device)
    calibration_frames = min(30, pose.shape[0])
    neutral_ground = neutral_model.vertices(
        pose[:calibration_frames],
        neutral_betas,
        translation[:calibration_frames],
    )[..., 2].amin()
    translation[:, 2].sub_(neutral_ground)

    local_quaternion_xyzw = quat_from_rotation_vector(pose)
    target_body_indices = torch.tensor(
        tuple(SMPL_BODY_NAMES.index(name) for name in target_body_names),
        dtype=torch.int64,
        device=pose.device,
    )
    target_local_rotation = local_quaternion_xyzw.index_select(1, target_body_indices)
    principal, _ = fit_ordered_hinge_coordinates(
        target_local_rotation[:, 1:],
        torch.eye(3, dtype=torch.float32, device=pose.device),
    )
    canonical_coordinate_names = tuple(f"{body_name}_{axis}" for body_name in target_body_names[1:] for axis in "xyz")
    reference_from_canonical = torch.tensor(
        tuple(canonical_coordinate_names.index(name) for name in target_coordinate_names),
        dtype=torch.int64,
        device=pose.device,
    )
    branch_offsets = torch.tensor((0, pose.shape[0]), dtype=torch.int64, device=pose.device)
    dof_position = (
        _time_select_euler_xyz_branches_segmented(principal, branch_offsets)
        .flatten(1)
        .index_select(1, reference_from_canonical)
    )
    root_quaternion_xyzw = target_local_rotation[:, 0]
    root_quaternion_wxyz = torch.cat((root_quaternion_xyzw[:, 3:], root_quaternion_xyzw[:, :3]), dim=-1)
    root_offset = torch.round(neutral_model.shaped_joints(neutral_betas)[0, 0], decimals=5)
    root_position = translation + root_offset
    generalized_position = torch.cat((root_position, root_quaternion_wxyz, dof_position), dim=-1)
    generalized_velocity = _compatible_pose_velocity(
        root_position,
        matrix_from_quat(root_quaternion_xyzw),
        dof_position,
        step_seconds=1.0 / _SMPL_TARGET_FPS,
    )
    return MotionGeneralizedCoordinates(generalized_position, generalized_velocity)


def _compatible_pose_velocity(
    root_position: torch.Tensor,
    root_rotation: torch.Tensor,
    dof_position: torch.Tensor,
    *,
    step_seconds: float,
) -> torch.Tensor:
    """Apply released root and forward-coordinate velocity laws to one target-rate clip."""
    root_linear = (root_position[1:] - root_position[:-1]) / step_seconds
    root_linear = torch.cat((root_linear, root_linear[-1:]), dim=0)

    root_quaternion_xyzw = quat_from_matrix(root_rotation)
    relative = torch.nn.functional.normalize(
        quat_mul(root_quaternion_xyzw[1:], quat_conjugate(root_quaternion_xyzw[:-1])), dim=-1
    )
    relative = torch.where(relative[:, 3:] < 0.0, -relative, relative)
    angle = torch.acos((2.0 * relative[:, 3].square() - 1.0).clamp(-1.0, 1.0))
    axis = relative[:, :3] / relative[:, :3].norm(dim=-1, keepdim=True).clamp_min(1.0e-10)
    root_angular_world = axis * angle[:, None] / step_seconds
    root_angular_world = torch.cat((root_angular_world, torch.zeros_like(root_angular_world[:1])), dim=0)
    root_angular_local = torch.matmul(root_rotation.transpose(-1, -2), root_angular_world.unsqueeze(-1)).squeeze(-1)

    dof_velocity = (dof_position[1:] - dof_position[:-1]) / step_seconds
    dof_velocity = torch.cat((dof_velocity, dof_velocity[-1:]), dim=0)
    return torch.cat((root_linear, root_angular_local, dof_velocity), dim=-1)


def _smpl_exact_coordinates(
    joint_q: torch.Tensor, joint_qd: torch.Tensor | None, source_fps: float
) -> MotionGeneralizedCoordinates:
    """Convert one exact SMPL source clip to target generalized coordinates."""
    del source_fps
    if joint_qd is None:
        raise ValueError("Exact SMPL coordinates require native generalized velocities.")
    generalized_position = torch.cat((joint_q[:, :3], convert_quat(joint_q[:, 3:7], to="wxyz"), joint_q[:, 7:]), dim=-1)
    return MotionGeneralizedCoordinates(generalized_position, joint_qd)


def smpl_frame_target(
    reference: NewtonKinematics,
    contact_patches: tuple[_MotionContactPatch, ...],
    *,
    calibration_artifact_root: str = "",
    calibration: _MotionTargetCalibration | None = None,
) -> _SmplFrameTarget:
    """Build one source-independent SMPL coordinate, support, and frame target."""
    from ....kinematics import NewtonKinematics

    if not isinstance(reference, NewtonKinematics) or not reference.asset_path:
        raise TypeError("SMPL frame construction requires scene-derived NewtonKinematics.")
    reference_mechanics_sha256 = reference.mechanics_identity_sha256
    neutral_calibration_path = (
        str(Path(calibration_artifact_root) / calibration.artifact) if calibration is not None else None
    )
    return _SmplFrameTarget(
        kinematics=reference,
        reference_mechanics_sha256=reference_mechanics_sha256,
        live_joint_names=_smpl_live_joint_names(reference),
        live_body_names=tuple(reference.body_names),
        contact_patches=contact_patches,
        neutral_calibration_path=neutral_calibration_path,
        neutral_calibration_artifact_sha256=None if calibration is None else calibration.artifact_sha256,
    )


def smpl_source_projection(
    source_skeleton: MotionSkeleton,
    target: MotionFrameTarget,
    source: MotionClipSource,
    contact_channels: tuple[_MotionContactChannel, ...],
    contact_channel_probe_offsets: torch.Tensor,
) -> MotionSourceProjection:
    """Build one exact, compatible-pose analytic, or trajectory projection into an SMPL target."""
    if not isinstance(target, _SmplFrameTarget):
        raise TypeError("SMPL source projection requires an SMPL frame target.")
    if source_skeleton.coordinate_identity_sha256 == target.coordinate_profile_sha256 and _smpl_coordinates_match(
        source_skeleton, target
    ):
        construction_identity = canonical_sha256(
            {
                "math_version": target.version,
                "source_coordinates": source_skeleton.coordinate_identity_sha256,
                "target_construction": target.construction_identity_sha256,
            }
        )
        return MotionSourceProjectionExact(
            source_skeleton=source_skeleton,
            target=target,
            version=target.version,
            construction_identity_sha256=construction_identity,
            convert_coordinates=_smpl_exact_coordinates,
        )
    if (
        isinstance(source, _SmplCompatiblePoseSource)
        and source.compatible_pose_profile_sha256 == SMPL_COMPATIBLE_POSE_PROFILE_SHA256
    ):
        subject_model = source.smpl_subject_model(source_skeleton.identity_sha256, target.kinematics.device)
        neutral_model = target.neutral_calibration_model()
        converter_identity = canonical_sha256(
            {
                "version": _SMPL_ANALYTIC_VERSION,
                "branch_policy": _SMPL_ANALYTIC_BRANCH_POLICY,
                "derivative_policy": _SMPL_ANALYTIC_DERIVATIVE_POLICY,
                "source_skeleton_identity_sha256": source_skeleton.identity_sha256,
                "source_pose_profile_sha256": source.compatible_pose_profile_sha256,
                "target_coordinate_profile_sha256": target.coordinate_profile_sha256,
                "subject_model_sha256": subject_model.source_sha256,
                "subject_artifact_sha256": subject_model.artifact_sha256,
                "neutral_model_sha256": neutral_model.source_sha256,
                "neutral_artifact_sha256": neutral_model.artifact_sha256,
            }
        )
        construction_identity = canonical_sha256(
            {
                "math_version": _SMPL_ANALYTIC_VERSION,
                "source_coordinates": source_skeleton.coordinate_identity_sha256,
                "target_construction": target.construction_identity_sha256,
                "analytic_converter": converter_identity,
            }
        )
        return MotionSourceProjectionAnalytic(
            source_skeleton=source_skeleton,
            target=target,
            version=_SMPL_ANALYTIC_VERSION,
            construction_identity_sha256=construction_identity,
            output_clip_index=_compatible_pose_output_index,
            convert_clip=partial(
                _compatible_pose_coordinates,
                subject_model=subject_model,
                neutral_model=neutral_model,
                target_body_names=tuple(target.kinematics.body_names),
                target_coordinate_names=target.reference_coordinate_names,
            ),
        )
    projection = MotionTrajectoryProjection(
        source_skeleton,
        target.trajectory_target,
        contact_channels,
        contact_channel_probe_offsets,
    )
    construction_identity = canonical_sha256(
        {
            "trajectory_projection": projection.construction_identity_sha256,
            "derivative_policy": _SMPL_RETARGET_DERIVATIVE_POLICY,
        }
    )
    return MotionSourceProjectionTrajectory(
        source_skeleton=source_skeleton,
        target=target,
        version=_SMPL_RETARGET_MATH_VERSION,
        construction_identity_sha256=construction_identity,
        target_projection=projection,
    )


_FRAME_BUILDER_MIGRATION = (
    "Configure MotionTaskTableCfg.TargetKinematicsCfg with smpl_frame_target, smpl_source_projection, "
    "and explicit contact_patches; the removed builder cannot infer that policy."
)


class SmplFrameBuilder:
    """Deprecated public boundary for the removed composite SMPL builder."""

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
        warnings.warn("SmplFrameBuilder is deprecated. " + _FRAME_BUILDER_MIGRATION, DeprecationWarning, stacklevel=2)
        raise RuntimeError(_FRAME_BUILDER_MIGRATION)


def smpl_frame_builder(source_skeleton: MotionSkeleton, reference: NewtonKinematics) -> SmplFrameBuilder:
    """Reject the deprecated composite SMPL builder and report its migration.

    Args:
        source_skeleton: Former source skeleton.
        reference: Former standalone target mechanics.

    Raises:
        RuntimeError: Always; the required contact policy is not part of the deprecated signature.
    """
    del source_skeleton, reference
    warnings.warn("smpl_frame_builder() is deprecated. " + _FRAME_BUILDER_MIGRATION, DeprecationWarning, stacklevel=2)
    raise RuntimeError(_FRAME_BUILDER_MIGRATION)
