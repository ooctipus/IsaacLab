# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared landmark motion-retarget projection and target mechanics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import torch

from isaaclab.utils.math import convert_quat, quat_apply, quat_conjugate, quat_from_matrix, quat_mul, quat_unique

from ..kinematics import kinematic_pose_forward, kinematic_root_basis, kinematic_tree_forward
from .data.skeleton import MotionSkeleton
from .identity import canonical_sha256

if TYPE_CHECKING:
    from .robots.target import MotionFrameTarget


_ROOT_CLUSTER_ROLES = ("pelvis", "left_hip", "right_hip", "torso")
_TRAJECTORY_PROJECTION_MATH_VERSION = "trajectory_projection_ground_relative_root_v1"


def _anatomical_basis_indices(
    role_body_indices: tuple[int, int, int, int],
    parent_indices: tuple[int, ...],
    rest_position_m: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Select stable proximal bodies for pelvis, left, right, and up landmarks."""
    root = role_body_indices[0]
    basis = [root]
    for body in role_body_indices[1:]:
        branch = []
        while body != root:
            branch.append(body)
            body = parent_indices[body]
            if body < 0:
                raise ValueError("An anatomical landmark must descend from the declared root body.")
        for candidate in reversed(branch):
            if float(torch.linalg.vector_norm(rest_position_m[candidate] - rest_position_m[root])) > 1.0e-9:
                basis.append(candidate)
                break
        else:
            raise ValueError("An anatomical root branch must contain one nonzero rest displacement.")
    if len(set(basis)) != 4:
        raise ValueError("Anatomical root-basis bodies must be distinct.")
    return basis[0], basis[1], basis[2], basis[3]


class _MotionContactChannel(Protocol):
    """One canonical source contact channel formed from semantic probes."""

    name: str
    source_probe_roles: tuple[str, ...]


class _MotionContactPatch(Protocol):
    """One robot collider patch driven by a canonical source channel."""

    channel: str
    body_name: str
    points_per_body: int
    height_band_m: float


class _MotionTargetCalibration(Protocol):
    """Target-owned calibration artifact declaration."""

    artifact: str
    artifact_sha256: str


def motion_contact_probe_offsets(
    contact_channels: tuple[_MotionContactChannel, ...],
    device: str | torch.device,
) -> torch.Tensor:
    """Build one canonical source-probe offset tensor shared by every source skeleton."""
    offsets = [0]
    for channel in contact_channels:
        offsets.append(offsets[-1] + len(channel.source_probe_roles))
    return torch.tensor(offsets, dtype=torch.int32, device=device)


@dataclass(frozen=True, slots=True)
class MotionTrajectoryTargets:
    """Calibrated source evidence, a limit-valid robot initializer, and contact geometry."""

    position_body_indices: tuple[int, ...]
    root_body_index: int
    source_root_policy: Literal["fixed", "optimized"]
    """Whether source refinement preserves the target-owned root seed or optimizes it.

    Contact refinement freezes the height-aligned root for either source policy.
    """
    initializer_policy: Literal["direct", "batched_frame_ik"]
    """Evidence-resolved initializer executed before whole-trajectory refinement."""
    parent_rows: tuple[int, ...]
    parent_row_tensor: torch.Tensor
    position_weights: tuple[float, ...]
    required_position_rows: tuple[int, ...]
    required_position_row_tensor: torch.Tensor
    position_normal_channel_slots: torch.Tensor
    """Contact-normal owner per position row, or -1, shape ``[position_count]``."""
    position_body_index_tensor: torch.Tensor
    rotation_body_indices: tuple[int, ...]
    rotation_weights: tuple[float, ...]
    source_landmark_rotation_xyzw: torch.Tensor
    """Rest-calibrated target rotations, shape [rotation_count, frame_count, 4]."""
    direction_body_indices: tuple[int, ...]
    direction_position_rows: tuple[int, ...]
    direction_weights: tuple[float, ...]
    contact_direction_rows: tuple[int, ...]
    """Distal-direction rows owned by contact channels, in channel order."""
    contact_direction_row_tensor: torch.Tensor
    direction_contact_channel_slots: torch.Tensor
    """Contact channel per distal-direction row, or -1, shape ``[direction_count]``."""
    required_direction_rows: tuple[int, ...]
    required_direction_row_tensor: torch.Tensor
    direction_body_index_tensor: torch.Tensor
    direction_position_row_tensor: torch.Tensor
    direction_point_body_m: torch.Tensor
    """Robot-owned distal points [m], shape [direction_count, 3]."""
    source_landmark_position_m: torch.Tensor
    """Target-morphology source landmarks [m] in the ground-relative gauge, shape [landmark_count, frame_count, 3]."""
    source_direction_point_position_m: torch.Tensor
    """Calibrated source-driven distal points [m], shape [direction_count, frame_count, 3]."""
    direction_length_values_m: tuple[float, ...]
    initial_joint_q: torch.Tensor
    """Limit-valid robot initializer [m or rad, depending on coordinate type]; never a primary-fit target."""
    segment_lengths_m: torch.Tensor
    segment_length_values_m: tuple[float, ...]
    coordinate_indices: torch.Tensor
    coordinate_lower_limits_rad: torch.Tensor
    coordinate_upper_limits_rad: torch.Tensor
    source_contact_probe_position_m: torch.Tensor
    """Raw source contact probes [m], shape ``[probe_count, frame_count, 3]``."""
    contact_channel_probe_offsets: torch.Tensor
    """Canonical probe ranges per contact channel, shape ``[channel_count + 1]``."""
    target_support_position_m: torch.Tensor
    """Caller-owned target patch points [m], shape [point_count, frame_count, 3].

    Source evidence initializes seed geometry. The solve workspace overwrites
    certified stable intervals with rigid planted targets.
    """
    contact_body_indices: torch.Tensor
    """Target support body per contact channel, shape ``[channel_count]``."""
    contact_normal_body: torch.Tensor
    """Target support normal in body coordinates, shape ``[channel_count, 3]``."""
    contact_forward_body: torch.Tensor
    """Target support forward axis in body coordinates, shape ``[channel_count, 3]``."""
    contact_distal_point_body_m: torch.Tensor
    """Contact-normalized distal offset [m] in support-body coordinates, shape ``[channel_count, 3]``."""
    leg_chain_body_indices: torch.Tensor
    """Target hip, knee, and ankle body indices, shape ``[channel_count, 3]``."""
    leg_chain_parent_body_indices: torch.Tensor
    """Target hip-parent body indices, shape ``[channel_count]``."""
    leg_knee_hint_anatomy: torch.Tensor
    """Preferred knee pole directions in target anatomical coordinates, shape ``[channel_count, 3]``."""
    leg_knee_hint_root: torch.Tensor
    """Preferred knee pole directions in target root coordinates, shape ``[channel_count, 3]``."""
    leg_segment_lengths_m: torch.Tensor
    """Target rest hip-to-knee and knee-to-ankle lengths [m], shape ``[channel_count, 2]``."""
    support_patch_offsets: tuple[int, ...]
    """Prefix point offsets per contact patch, length ``channel_count + 1``."""
    support_body_indices: torch.Tensor
    """Target body per generated surface point, shape ``[target_point_count]``."""
    support_point_body_m: torch.Tensor
    """Target body-local surface offsets [m], shape ``[target_point_count, 3]``."""
    support_channel_slots: torch.Tensor
    """Source contact channel per target surface point, shape ``[target_point_count]``."""


@dataclass(frozen=True, slots=True)
class _MotionTrajectoryTarget:
    """Immutable target-robot trajectory layout shared by every source projection."""

    @dataclass(frozen=True, slots=True)
    class Landmark:
        """One semantic role in the target robot position tree."""

        role: str
        position_body_name: str
        parent_row: int
        weight: float

    @dataclass(frozen=True, slots=True)
    class RotationLandmark:
        """One independently weighted semantic target rotation."""

        role: str
        body_name: str
        weight: float

    @dataclass(frozen=True, slots=True)
    class DirectionPoint:
        """One semantic role bound to a robot-owned distal direction point."""

        name: str
        base_role: str
        source_from_role: str
        source_to_role: str
        source_direction_law: Literal["between_positions", "wrist_forward"]
        body_name: str
        point_body_m: tuple[float, float, float]
        weight: float

    @dataclass(frozen=True, slots=True)
    class LegChain:
        """One contact-channel leg and its target-anatomical knee pole."""

        channel: str
        hip_body_name: str
        knee_body_name: str
        ankle_body_name: str
        knee_hint_anatomy: tuple[float, float, float]

    frame_target: MotionFrameTarget
    landmarks: tuple[Landmark, ...]
    rotation_landmarks: tuple[RotationLandmark, ...]
    direction_points: tuple[DirectionPoint, ...]
    source_root_policy: Literal["fixed", "optimized"]
    initializer_policy: Literal["direct", "batched_frame_ik"]
    required_position_roles: tuple[str, ...]
    required_direction_roles: tuple[str, ...]
    contact_patches: tuple[_MotionContactPatch, ...]
    leg_chains: tuple[LegChain, ...]
    support_up_frame: Literal["root", "anatomy"]
    version: str
    construction_identity_sha256: str = field(init=False)
    roles: tuple[str, ...] = field(init=False)
    rotation_roles: tuple[str, ...] = field(init=False)
    direction_roles: tuple[str, ...] = field(init=False)
    required_position_rows: tuple[int, ...] = field(init=False)
    contact_direction_rows: tuple[int, ...] = field(init=False)
    required_direction_rows: tuple[int, ...] = field(init=False)
    position_body_indices: tuple[int, ...] = field(init=False)
    root_body_index: int = field(init=False)
    direction_body_indices: tuple[int, ...] = field(init=False)
    direction_position_rows: tuple[int, ...] = field(init=False)
    parent_rows: tuple[int, ...] = field(init=False)
    rotation_body_indices: tuple[int, ...] = field(init=False)
    rotation_weights: tuple[float, ...] = field(init=False)
    position_weights: tuple[float, ...] = field(init=False)
    direction_weights: tuple[float, ...] = field(init=False)
    position_body_index_tensor: torch.Tensor = field(init=False, repr=False)
    required_position_row_tensor: torch.Tensor = field(init=False, repr=False)
    body_normal_channel_slots: torch.Tensor = field(init=False, repr=False)
    position_normal_channel_slots: torch.Tensor = field(init=False, repr=False)
    direction_body_index_tensor: torch.Tensor = field(init=False, repr=False)
    contact_direction_row_tensor: torch.Tensor = field(init=False, repr=False)
    direction_contact_channel_slots: torch.Tensor = field(init=False, repr=False)
    required_direction_row_tensor: torch.Tensor = field(init=False, repr=False)
    rotation_rest_xyzw: torch.Tensor = field(init=False, repr=False)
    anatomical_basis_body_indices: tuple[int, int, int, int] = field(init=False)
    anatomical_rotation_xyzw: torch.Tensor = field(init=False, repr=False)
    direction_point_body_m: torch.Tensor = field(init=False, repr=False)
    direction_rows: torch.Tensor = field(init=False, repr=False)
    direction_lengths_m: torch.Tensor = field(init=False, repr=False)
    direction_length_values_m: tuple[float, ...] = field(init=False)
    root_cluster_rows: tuple[int, ...] = field(init=False)
    root_cluster_offset_m: torch.Tensor = field(init=False, repr=False)
    parent_row_tensor: torch.Tensor = field(init=False, repr=False)
    segment_lengths_m: torch.Tensor = field(init=False, repr=False)
    segment_length_values_m: tuple[float, ...] = field(init=False)
    coordinate_indices: torch.Tensor = field(init=False, repr=False)
    coordinate_lower_limits_rad: torch.Tensor = field(init=False, repr=False)
    coordinate_upper_limits_rad: torch.Tensor = field(init=False, repr=False)
    contact_channel_names: tuple[str, ...] = field(init=False)
    contact_body_indices: torch.Tensor = field(init=False, repr=False)
    contact_normal_body: torch.Tensor = field(init=False, repr=False)
    contact_forward_body: torch.Tensor = field(init=False, repr=False)
    contact_distal_point_body_m: torch.Tensor = field(init=False, repr=False)
    leg_chain_body_indices: torch.Tensor = field(init=False, repr=False)
    leg_chain_parent_body_indices: torch.Tensor = field(init=False, repr=False)
    leg_knee_hint_anatomy: torch.Tensor = field(init=False, repr=False)
    leg_knee_hint_root: torch.Tensor = field(init=False, repr=False)
    leg_segment_lengths_m: torch.Tensor = field(init=False, repr=False)
    support_patch_offsets: tuple[int, ...] = field(init=False)
    support_body_indices: torch.Tensor = field(init=False, repr=False)
    support_point_body_m: torch.Tensor = field(init=False, repr=False)
    support_point_root_m: torch.Tensor = field(init=False, repr=False)
    support_channel_slots: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:  # noqa: C901
        """Resolve target topology, rest geometry, bounds, and collider support once."""
        tree = self.frame_target.kinematic_tree
        reference = self.frame_target.kinematics
        if reference.n_root_coords != 7:
            raise ValueError("Trajectory targets require one free robot root.")
        roles = tuple(landmark.role for landmark in self.landmarks)
        rotation_roles = tuple(landmark.role for landmark in self.rotation_landmarks)
        direction_roles = tuple(point.name for point in self.direction_points)
        position_names = tuple(landmark.position_body_name for landmark in self.landmarks)
        rotation_names = tuple(landmark.body_name for landmark in self.rotation_landmarks)
        position_weights = tuple(landmark.weight for landmark in self.landmarks)
        rotation_weights = tuple(landmark.weight for landmark in self.rotation_landmarks)
        direction_weights = tuple(point.weight for point in self.direction_points)
        direction_names = tuple(point.body_name for point in self.direction_points)
        parent_rows = tuple(landmark.parent_row for landmark in self.landmarks)
        body_by_name = {name: index for index, name in enumerate(tree.body_names)}
        contact_channel_names = tuple(patch.channel for patch in self.contact_patches)
        contact_body_names = tuple(patch.body_name for patch in self.contact_patches)
        leg_channel_names = tuple(chain.channel for chain in self.leg_chains)
        leg_body_name_values = tuple(
            (chain.hip_body_name, chain.knee_body_name, chain.ankle_body_name) for chain in self.leg_chains
        )
        missing_position_bodies = tuple(name for name in position_names if name not in body_by_name)
        missing_rotation_bodies = tuple(name for name in rotation_names if name not in body_by_name)
        missing_direction_bodies = tuple(name for name in direction_names if name not in body_by_name)
        missing_support = tuple(name for name in contact_body_names if name not in body_by_name)
        missing_leg_bodies = tuple(
            name for chain_names in leg_body_name_values for name in chain_names if name not in body_by_name
        )
        if any(
            not math.isfinite(weight) or weight <= 0.0
            for weight in (*position_weights, *rotation_weights, *direction_weights)
        ):
            raise ValueError("Trajectory target objective weights must be finite and positive.")
        if (
            not roles
            or len(set(roles)) != len(roles)
            or not rotation_roles
            or len(set(rotation_roles)) != len(rotation_roles)
            or len(set(rotation_names)) != len(rotation_names)
            or missing_rotation_bodies
            or not self.direction_points
            or len(set(direction_roles)) != len(direction_roles)
            or any(point.base_role not in roles for point in self.direction_points)
            or any(
                not point.source_from_role or not point.source_to_role or point.source_from_role == point.source_to_role
                for point in self.direction_points
            )
            or any(
                point.source_direction_law not in ("between_positions", "wrist_forward")
                or (point.source_direction_law == "wrist_forward" and point.base_role != point.source_to_role)
                for point in self.direction_points
            )
            or missing_direction_bodies
            or self.source_root_policy not in ("fixed", "optimized")
            or self.initializer_policy not in ("direct", "batched_frame_ik")
            or not self.required_position_roles
            or len(set(self.required_position_roles)) != len(self.required_position_roles)
            or not self.required_direction_roles
            or len(set(self.required_direction_roles)) != len(self.required_direction_roles)
            or missing_position_bodies
            or not self.contact_patches
            or len(set(contact_channel_names)) != len(self.contact_patches)
            or len(set(contact_body_names)) != len(self.contact_patches)
            or missing_support
            or roles[0] not in self.required_position_roles
            or any(role not in roles for role in self.required_position_roles)
            or any(role not in direction_roles for role in self.required_direction_roles)
            or any(channel not in direction_roles for channel in contact_channel_names)
            or any(channel not in self.required_direction_roles for channel in contact_channel_names)
            or self.support_up_frame not in ("root", "anatomy")
        ):
            raise ValueError(
                f"Trajectory target has duplicate/missing roles, position bodies {missing_position_bodies}, "
                f"rotation bodies {missing_rotation_bodies}, direction bodies {missing_direction_bodies}, "
                f"or support bodies {missing_support}."
            )
        if leg_channel_names != contact_channel_names or missing_leg_bodies:
            raise ValueError(
                "Target leg chains must match contact channels in order and use existing target bodies; "
                f"missing bodies: {missing_leg_bodies}."
            )
        if any(
            len(chain.knee_hint_anatomy) != 3
            or any(not math.isfinite(component) for component in chain.knee_hint_anatomy)
            for chain in self.leg_chains
        ):
            raise ValueError("Target leg knee hints must be finite three-vectors in anatomical coordinates.")
        position_body_indices = tuple(body_by_name[name] for name in position_names)
        rotation_body_indices = tuple(body_by_name[name] for name in rotation_names)
        direction_body_indices = tuple(body_by_name[name] for name in direction_names)
        patch_body_values = tuple(body_by_name[name] for name in contact_body_names)
        leg_chain_body_values = tuple(
            tuple(body_by_name[name] for name in chain_names) for chain_names in leg_body_name_values
        )
        if any(bodies[2] != patch_body_values[channel] for channel, bodies in enumerate(leg_chain_body_values)):
            raise ValueError("Each target leg ankle role must resolve to its contact-patch body.")
        if len({body for bodies in leg_chain_body_values for body in bodies}) != 3 * len(leg_chain_body_values):
            raise ValueError("Target leg chains must own distinct hip, knee, and ankle bodies.")
        leg_chain_parent_values = tuple(tree.parent_indices[bodies[0]] for bodies in leg_chain_body_values)
        if any(parent < 0 for parent in leg_chain_parent_values):
            raise ValueError("Every target leg hip must have one parent body.")
        for hip, knee, ankle in leg_chain_body_values:
            body = knee
            while body >= 0 and body != hip:
                body = tree.parent_indices[body]
            if body != hip:
                raise ValueError("Every target leg hip must be an ancestor of its knee.")
            body = ankle
            while body >= 0 and body != knee:
                body = tree.parent_indices[body]
            if body != knee:
                raise ValueError("Every target leg knee must be an ancestor of its ankle.")
        root = tree.root_body_index
        if (
            position_body_indices[0] != root
            or parent_rows[0] != -1
            or rotation_roles[0] != roles[0]
            or rotation_body_indices[0] != root
        ):
            raise ValueError("Trajectory position and rotation row zero must own the exact target root role.")
        for row, parent in enumerate(parent_rows[1:], start=1):
            if parent < 0 or parent >= row:
                raise ValueError("Trajectory landmark parents must precede their children.")

        device = torch.device(reference.device)
        position_indices = torch.tensor(position_body_indices, dtype=torch.int64, device=device)
        direction_indices = torch.tensor(direction_body_indices, dtype=torch.int64, device=device)
        parent_row_tensor = torch.tensor(parent_rows, dtype=torch.int64, device=device)
        required_position_rows = tuple(roles.index(role) for role in self.required_position_roles)
        contact_direction_rows = tuple(direction_roles.index(channel) for channel in contact_channel_names)
        required_direction_rows = tuple(direction_roles.index(role) for role in self.required_direction_roles)
        required_position_row_tensor = torch.tensor(required_position_rows, dtype=torch.int64, device=device)
        contact_direction_row_tensor = torch.tensor(contact_direction_rows, dtype=torch.int64, device=device)
        required_direction_row_tensor = torch.tensor(required_direction_rows, dtype=torch.int64, device=device)
        direction_contact_channel_values = [-1] * len(direction_roles)
        for channel, row in enumerate(contact_direction_rows):
            direction_contact_channel_values[row] = channel
        direction_contact_channel_slots = torch.tensor(
            direction_contact_channel_values, dtype=torch.int64, device=device
        )

        target_scene_pose = torch.tensor(reference.default_body_q, dtype=torch.float64, device="cpu")
        target_scene_rotation = target_scene_pose[:, 3:7]
        target_scene_rotation.div_(torch.linalg.vector_norm(target_scene_rotation, dim=-1, keepdim=True))
        scene_root_rotation_inverse = quat_conjugate(target_scene_rotation[root])
        target_rest_position_calibration = quat_apply(
            scene_root_rotation_inverse.expand(len(target_scene_pose), 4),
            target_scene_pose[:, :3] - target_scene_pose[root, :3],
        )
        target_rest_rotation_calibration = quat_mul(
            scene_root_rotation_inverse.expand(len(target_scene_pose), 4), target_scene_rotation
        )
        rotation_rest_calibration = target_rest_rotation_calibration[list(rotation_body_indices)]
        rotation_rest_values = tuple(
            tuple(float(component) for component in rotation)
            for rotation in quat_unique(rotation_rest_calibration).tolist()
        )
        rotation_rest = quat_unique(rotation_rest_calibration)
        target_rest_calibration = torch.cat(
            (target_rest_position_calibration, target_rest_rotation_calibration), dim=-1
        )
        calibration_position_indices = torch.tensor(position_body_indices, dtype=torch.int64, device="cpu")
        calibration_parent_rows = torch.tensor(parent_rows[1:], dtype=torch.int64, device="cpu")
        calibration_parent_indices = calibration_position_indices.index_select(0, calibration_parent_rows)
        target_rest_edge_calibration = torch.zeros(len(roles), 3, dtype=torch.float64, device="cpu")
        target_rest_edge_calibration[1:] = (
            target_rest_position_calibration[calibration_position_indices[1:]]
            - target_rest_position_calibration[calibration_parent_indices]
        )
        lengths_calibration = torch.zeros(len(roles), dtype=torch.float64, device="cpu")
        lengths_calibration[1:] = torch.linalg.vector_norm(target_rest_edge_calibration[1:], dim=-1)
        torch._assert_async(
            torch.all(torch.isfinite(lengths_calibration[1:]) & (lengths_calibration[1:] > 1.0e-6)),
            "Every target landmark edge must have finite nonzero length.",
        )
        lengths_calibration[0] = lengths_calibration[1:].mean()
        length_values = tuple(float(value) for value in lengths_calibration.tolist())
        lengths = lengths_calibration.to(dtype=torch.float32, device=device)
        root_rest_position = target_rest_position_calibration[root]
        root_rest_rotation_inverse = quat_conjugate(target_rest_rotation_calibration[root])
        root_cluster_rows = tuple(roles.index(role) for role in _ROOT_CLUSTER_ROLES)
        target_basis_body_indices = tuple(position_body_indices[row] for row in root_cluster_rows)
        target_basis_indices = _anatomical_basis_indices(
            target_basis_body_indices, tree.parent_indices, target_rest_position_calibration
        )
        target_anatomical_rotation_calibration = quat_from_matrix(
            kinematic_root_basis(target_rest_position_calibration, *target_basis_indices)
        )
        target_anatomical_rotation_calibration.div_(torch.linalg.vector_norm(target_anatomical_rotation_calibration))
        target_anatomical_rotation_calibration = quat_unique(target_anatomical_rotation_calibration)
        target_anatomical_rotation_values = tuple(float(value) for value in target_anatomical_rotation_calibration)
        target_anatomical_rotation = target_anatomical_rotation_calibration.to(dtype=torch.float32, device=device)
        root_to_anatomy_rotation = quat_conjugate(target_anatomical_rotation_calibration)
        leg_chain_count = len(leg_chain_body_values)
        leg_chain_flat_values = tuple(body for bodies in leg_chain_body_values for body in bodies)
        leg_chain_rest_position = target_rest_position_calibration[list(leg_chain_flat_values)].view(
            leg_chain_count, 3, 3
        )
        leg_segment_calibration = torch.stack(
            (
                leg_chain_rest_position[:, 1] - leg_chain_rest_position[:, 0],
                leg_chain_rest_position[:, 2] - leg_chain_rest_position[:, 1],
            ),
            dim=1,
        )
        leg_segment_length_calibration = torch.linalg.vector_norm(leg_segment_calibration, dim=-1)
        if not bool(
            torch.all(torch.isfinite(leg_segment_length_calibration) & (leg_segment_length_calibration > 1.0e-6))
        ):
            raise ValueError("Every target leg segment must have finite nonzero rest length.")
        leg_knee_hint_anatomy_calibration = torch.tensor(
            [chain.knee_hint_anatomy for chain in self.leg_chains], dtype=torch.float64, device="cpu"
        )
        leg_knee_hint_length = torch.linalg.vector_norm(leg_knee_hint_anatomy_calibration, dim=-1)
        if not bool(torch.all(torch.isfinite(leg_knee_hint_length))) or not torch.allclose(
            leg_knee_hint_length, torch.ones_like(leg_knee_hint_length), rtol=1.0e-7, atol=1.0e-7
        ):
            raise ValueError("Every target-anatomical knee hint must be a finite unit vector.")
        leg_reach_anatomy = quat_apply(
            root_to_anatomy_rotation.expand(leg_chain_count, 4),
            leg_chain_rest_position[:, 2] - leg_chain_rest_position[:, 0],
        )
        leg_reach_length = torch.linalg.vector_norm(leg_reach_anatomy, dim=-1, keepdim=True)
        if not bool(torch.all(torch.isfinite(leg_reach_length) & (leg_reach_length > 1.0e-6))):
            raise ValueError("Every target leg must have finite nonzero hip-to-ankle rest reach.")
        leg_reach_axis = leg_reach_anatomy / leg_reach_length
        leg_knee_hint_perpendicular = (
            leg_knee_hint_anatomy_calibration
            - torch.sum(leg_knee_hint_anatomy_calibration * leg_reach_axis, dim=-1, keepdim=True) * leg_reach_axis
        )
        if not bool(torch.all(torch.linalg.vector_norm(leg_knee_hint_perpendicular, dim=-1) > 1.0e-6)):
            raise ValueError("Every target knee hint must define a nondegenerate bend plane for its rest leg.")
        leg_knee_hint_root_calibration = quat_apply(
            target_anatomical_rotation_calibration.expand(leg_chain_count, 4), leg_knee_hint_anatomy_calibration
        )
        leg_chain_body_indices = torch.tensor(leg_chain_body_values, dtype=torch.int64, device=device)
        leg_chain_parent_body_indices = torch.tensor(leg_chain_parent_values, dtype=torch.int64, device=device)
        leg_knee_hint_anatomy = leg_knee_hint_anatomy_calibration.to(dtype=torch.float32, device=device).contiguous()
        leg_knee_hint_root = leg_knee_hint_root_calibration.to(dtype=torch.float32, device=device).contiguous()
        leg_segment_lengths = leg_segment_length_calibration.to(dtype=torch.float32, device=device).contiguous()
        leg_knee_hint_root_values = tuple(
            tuple(float(component) for component in hint) for hint in leg_knee_hint_root_calibration.tolist()
        )
        leg_segment_length_values_m = tuple(
            tuple(float(component) for component in lengths) for lengths in leg_segment_length_calibration.tolist()
        )
        root_cluster_offset_calibration = quat_apply(
            root_rest_rotation_inverse.expand(len(root_cluster_rows), 4),
            target_rest_position_calibration[calibration_position_indices[list(root_cluster_rows)]]
            - root_rest_position,
        )
        root_cluster_offset_values_m = tuple(
            tuple(float(component) for component in offset) for offset in root_cluster_offset_calibration.tolist()
        )
        root_cluster_offset = root_cluster_offset_calibration.to(dtype=torch.float32, device=device)
        direction_row_values = tuple(roles.index(point.base_role) for point in self.direction_points)
        direction_rows = torch.tensor(direction_row_values, dtype=torch.int64, device=device)
        direction_semantic_body_indices = tuple(position_body_indices[row] for row in direction_row_values)
        body_normal_channel_values = [-1] * len(tree.body_names)
        for channel, direction_row in enumerate(contact_direction_rows):
            base_body = direction_semantic_body_indices[direction_row]
            ankle_body = patch_body_values[channel]
            if base_body != ankle_body:
                raise ValueError("Every required contact direction must start at its contact-patch and leg-ankle body.")
            chain = []
            body = direction_body_indices[direction_row]
            while body >= 0 and body != ankle_body:
                chain.append(body)
                body = tree.parent_indices[body]
            if body != ankle_body:
                raise ValueError("Every required contact distal body must descend from its contact-patch body.")
            chain.append(ankle_body)
            for body in chain:
                owner = body_normal_channel_values[body]
                if owner not in (-1, channel):
                    raise ValueError("Target contact-normal body chains must not overlap between channels.")
                body_normal_channel_values[body] = channel
        position_normal_channel_values = tuple(body_normal_channel_values[body] for body in position_body_indices)
        body_normal_channel_slots = torch.tensor(body_normal_channel_values, dtype=torch.int64, device=device)
        position_normal_channel_slots = torch.tensor(position_normal_channel_values, dtype=torch.int64, device=device)

        direction_point_body_calibration = torch.tensor(
            [point.point_body_m for point in self.direction_points], dtype=torch.float64, device="cpu"
        )
        direction_point_world_calibration = target_rest_position_calibration[list(direction_body_indices)] + quat_apply(
            target_rest_rotation_calibration[list(direction_body_indices)], direction_point_body_calibration
        )
        direction_world_calibration = (
            direction_point_world_calibration - target_rest_position_calibration[list(direction_semantic_body_indices)]
        )
        direction_length_calibration = torch.linalg.vector_norm(direction_world_calibration, dim=-1)
        direction_length_values = tuple(float(value) for value in direction_length_calibration.tolist())
        if any(not math.isfinite(value) or value <= 1.0e-6 for value in direction_length_values):
            raise ValueError("Every direction point must define one finite nonzero target-owned distal axis.")
        direction_lengths = direction_length_calibration.to(dtype=torch.float32, device=device)
        direction_point_body = direction_point_body_calibration.to(dtype=torch.float32, device=device)

        from ..kinematics.collider_geometry import model_body_collider_support_points

        if self.support_up_frame == "root":
            support_selection_position = target_rest_position_calibration
            support_selection_rotation = target_rest_rotation_calibration
        else:
            support_selection_position = quat_apply(
                root_to_anatomy_rotation.expand(len(target_rest_calibration), 4), target_rest_position_calibration
            )
            support_selection_rotation = quat_mul(
                root_to_anatomy_rotation.expand(len(target_rest_calibration), 4), target_rest_rotation_calibration
            )
        support_selection_pose = torch.cat((support_selection_position, support_selection_rotation), dim=-1)
        default_body_pose = support_selection_pose.to(dtype=torch.float32).numpy()
        support_parts: list[np.ndarray] = []
        support_normal_parts: list[torch.Tensor] = []
        for patch_index, patch in enumerate(self.contact_patches):
            selected, _ = model_body_collider_support_points(
                reference.builder,
                (patch_body_values[patch_index],),
                default_body_pose,
                points_per_body=patch.points_per_body,
                height_band_m=patch.height_band_m,
                support_up_world=np.asarray(((0.0, 0.0, 1.0),), dtype=np.float32),
            )
            if selected.shape != (3, 3) or not np.isfinite(selected).all():
                raise ValueError(f"Contact patch {patch.channel!r} must resolve exactly three finite low points.")
            triangle = torch.tensor(selected, dtype=torch.float64, device="cpu")
            normal = torch.linalg.cross(triangle[1] - triangle[0], triangle[2] - triangle[0], dim=-1)
            normal_length = torch.linalg.vector_norm(normal)
            if not bool(torch.isfinite(normal_length)) or float(normal_length) <= 1.0e-8:
                raise ValueError(f"Contact patch {patch.channel!r} low points must be non-collinear.")
            normal.div_(normal_length)
            body_rotation = support_selection_rotation[patch_body_values[patch_index]]
            if float(quat_apply(body_rotation, normal)[2]) < 0.0:
                selected = selected[[0, 2, 1]].copy()
                normal.neg_()
            support_parts.append(selected)
            support_normal_parts.append(normal.clone())
        support_points_np = np.concatenate(support_parts)
        points_per_patch = tuple(patch.points_per_body for patch in self.contact_patches)
        support_body_values = tuple(
            body for body, count in zip(patch_body_values, points_per_patch, strict=True) for _ in range(count)
        )
        support_channel_values = tuple(channel for channel, count in enumerate(points_per_patch) for _ in range(count))
        support_body_indices = torch.tensor(support_body_values, dtype=torch.int64, device=device)
        support_point_body_calibration = torch.tensor(support_points_np, dtype=torch.float64, device="cpu")
        support_point_body = support_point_body_calibration.to(dtype=torch.float32, device=device)
        support_world_calibration = target_rest_position_calibration[list(support_body_values)] + quat_apply(
            target_rest_rotation_calibration[list(support_body_values)], support_point_body_calibration
        )
        support_point_root_calibration = quat_apply(
            root_rest_rotation_inverse.expand(len(support_body_values), 4),
            support_world_calibration - root_rest_position,
        )
        support_point_root = support_point_root_calibration.to(dtype=torch.float32, device=device)
        contact_normal_body_calibration = torch.stack(support_normal_parts)
        contact_normal_body = contact_normal_body_calibration.to(dtype=torch.float32, device=device)
        contact_forward_body_calibration = quat_apply(
            quat_conjugate(target_rest_rotation_calibration[list(patch_body_values)]),
            direction_point_world_calibration[list(contact_direction_rows)]
            - target_rest_position_calibration[list(patch_body_values)],
        )
        contact_forward_body_calibration -= (
            torch.sum(contact_forward_body_calibration * contact_normal_body_calibration, dim=-1, keepdim=True)
            * contact_normal_body_calibration
        )
        contact_forward_length = torch.linalg.vector_norm(contact_forward_body_calibration, dim=-1, keepdim=True)
        if not bool(torch.all(torch.isfinite(contact_forward_length) & (contact_forward_length > 1.0e-8))):
            raise ValueError("Every contact channel must define one finite nonzero forward axis in its support plane.")
        contact_forward_body_calibration.div_(contact_forward_length)
        contact_forward_norm = torch.linalg.vector_norm(contact_forward_body_calibration, dim=-1)
        contact_forward_normal_dot = torch.sum(
            contact_forward_body_calibration * contact_normal_body_calibration, dim=-1
        )
        if not torch.allclose(
            contact_forward_norm, torch.ones_like(contact_forward_norm), rtol=1.0e-7, atol=1.0e-7
        ) or not torch.allclose(
            contact_forward_normal_dot,
            torch.zeros_like(contact_forward_normal_dot),
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise ValueError("Contact forward axes must be finite unit vectors orthogonal to their support normals.")
        contact_forward_body_values = tuple(
            tuple(float(component) for component in forward) for forward in contact_forward_body_calibration.tolist()
        )
        contact_direction_lengths = contact_forward_body_calibration.new_tensor(
            tuple(direction_length_values[row] for row in contact_direction_rows)
        )
        contact_distal_point_body_calibration = contact_forward_body_calibration * contact_direction_lengths.unsqueeze(
            -1
        )
        contact_distal_point_body_values = tuple(
            tuple(float(component) for component in point) for point in contact_distal_point_body_calibration.tolist()
        )
        contact_forward_body = contact_forward_body_calibration.to(dtype=torch.float32, device=device)
        contact_distal_point_body = contact_distal_point_body_calibration.to(dtype=torch.float32, device=device)
        support_channel_slots = torch.tensor(support_channel_values, dtype=torch.int64, device=device)
        support_patch_start_values = tuple(sum(points_per_patch[:index]) for index in range(len(points_per_patch)))
        support_patch_offsets = (*support_patch_start_values, sum(points_per_patch))
        contact_body_indices = torch.tensor(patch_body_values, dtype=torch.int64, device=device)

        coordinate_indices = torch.tensor(tree.coordinate_q_indices, dtype=torch.int64, device=device)
        coordinate_lower = torch.tensor(tree.coordinate_lower_limits_rad, dtype=torch.float32, device=device)
        coordinate_upper = torch.tensor(tree.coordinate_upper_limits_rad, dtype=torch.float32, device=device)
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "rotation_roles", rotation_roles)
        object.__setattr__(self, "rotation_body_indices", rotation_body_indices)
        object.__setattr__(self, "rotation_weights", rotation_weights)
        object.__setattr__(self, "position_weights", position_weights)
        object.__setattr__(self, "direction_weights", direction_weights)
        object.__setattr__(self, "direction_roles", direction_roles)
        object.__setattr__(self, "required_position_rows", required_position_rows)
        object.__setattr__(self, "contact_direction_rows", contact_direction_rows)
        object.__setattr__(self, "required_direction_rows", required_direction_rows)
        object.__setattr__(self, "position_body_indices", position_body_indices)
        object.__setattr__(self, "root_body_index", root)
        object.__setattr__(self, "direction_body_indices", direction_body_indices)
        object.__setattr__(self, "direction_position_rows", direction_row_values)
        object.__setattr__(self, "parent_rows", parent_rows)
        object.__setattr__(self, "position_body_index_tensor", position_indices)
        object.__setattr__(self, "required_position_row_tensor", required_position_row_tensor)
        object.__setattr__(self, "body_normal_channel_slots", body_normal_channel_slots)
        object.__setattr__(self, "position_normal_channel_slots", position_normal_channel_slots)
        object.__setattr__(self, "rotation_rest_xyzw", rotation_rest)
        object.__setattr__(self, "anatomical_basis_body_indices", target_basis_indices)
        object.__setattr__(self, "anatomical_rotation_xyzw", target_anatomical_rotation)
        object.__setattr__(self, "direction_body_index_tensor", direction_indices)
        object.__setattr__(self, "contact_direction_row_tensor", contact_direction_row_tensor)
        object.__setattr__(self, "direction_contact_channel_slots", direction_contact_channel_slots)
        object.__setattr__(self, "required_direction_row_tensor", required_direction_row_tensor)
        object.__setattr__(self, "direction_point_body_m", direction_point_body)
        object.__setattr__(self, "direction_rows", direction_rows)
        object.__setattr__(self, "direction_lengths_m", direction_lengths)
        object.__setattr__(self, "direction_length_values_m", direction_length_values)
        object.__setattr__(self, "parent_row_tensor", parent_row_tensor)
        object.__setattr__(self, "segment_lengths_m", lengths)
        object.__setattr__(self, "root_cluster_rows", root_cluster_rows)
        object.__setattr__(self, "root_cluster_offset_m", root_cluster_offset)
        object.__setattr__(self, "segment_length_values_m", length_values)
        object.__setattr__(self, "coordinate_indices", coordinate_indices)
        object.__setattr__(self, "coordinate_lower_limits_rad", coordinate_lower)
        object.__setattr__(self, "coordinate_upper_limits_rad", coordinate_upper)
        object.__setattr__(self, "contact_channel_names", contact_channel_names)
        object.__setattr__(self, "contact_body_indices", contact_body_indices)
        object.__setattr__(self, "contact_normal_body", contact_normal_body)
        object.__setattr__(self, "contact_forward_body", contact_forward_body)
        object.__setattr__(self, "contact_distal_point_body_m", contact_distal_point_body)
        object.__setattr__(self, "leg_chain_body_indices", leg_chain_body_indices)
        object.__setattr__(self, "leg_chain_parent_body_indices", leg_chain_parent_body_indices)
        object.__setattr__(self, "leg_knee_hint_anatomy", leg_knee_hint_anatomy)
        object.__setattr__(self, "leg_knee_hint_root", leg_knee_hint_root)
        object.__setattr__(self, "leg_segment_lengths_m", leg_segment_lengths)
        object.__setattr__(self, "support_patch_offsets", support_patch_offsets)
        object.__setattr__(self, "support_body_indices", support_body_indices)
        object.__setattr__(self, "support_point_body_m", support_point_body)
        object.__setattr__(self, "support_point_root_m", support_point_root)
        object.__setattr__(self, "support_channel_slots", support_channel_slots)
        object.__setattr__(
            self,
            "construction_identity_sha256",
            canonical_sha256(
                {
                    "source_root_policy": self.source_root_policy,
                    "initializer_policy": self.initializer_policy,
                    "math_version": self.version,
                    "frame_target_sha256": self.frame_target.construction_identity_sha256,
                    "optimization_weight_law": "global_cfg_times_target_role_weight_over_target_length",
                    "landmarks": tuple(
                        (item.role, item.position_body_name, item.parent_row, item.weight) for item in self.landmarks
                    ),
                    "rotation_landmarks": tuple(
                        (
                            item.role,
                            item.body_name,
                            rotation_body_indices[index],
                            item.weight,
                            rotation_rest_values[index],
                        )
                        for index, item in enumerate(self.rotation_landmarks)
                    ),
                    "direction_points": tuple(
                        (
                            item.name,
                            item.base_role,
                            item.source_from_role,
                            item.source_to_role,
                            item.source_direction_law,
                            position_names[direction_row_values[index]],
                            item.body_name,
                            direction_body_indices[index],
                            item.point_body_m,
                            direction_length_values[index],
                            item.weight,
                        )
                        for index, item in enumerate(self.direction_points)
                    ),
                    "rotation_laws": (
                        "source_world_rotation_times_source_rest_inverse_target_rest",
                        "source_world_anatomical_basis_times_target_root_anatomical_basis_inverse",
                        "anatomical_root_gauge_times_rest_calibrated_world_body_rotations",
                    ),
                    "anatomical_root_basis": {
                        "body_indices": target_basis_indices,
                        "rotation_xyzw": target_anatomical_rotation_values,
                        "law": "proximal_pelvis_hips_torso_forward_left_up",
                    },
                    "direction_geometry": {
                        "between_positions": "normalized_source_position_direction_times_target_endpoint_length",
                        "wrist_forward": "source_policy_calibrated_forward_times_target_endpoint_length",
                    },
                    "target_rest_frame": "scene_default_body_pose_factored_by_scene_free_root",
                    "edge_facts": "target_rest_segment_lengths",
                    "required_position_roles": self.required_position_roles,
                    "required_direction_roles": self.required_direction_roles,
                    "root_cluster_roles": _ROOT_CLUSTER_ROLES,
                    "root_cluster_offsets_m": root_cluster_offset_values_m,
                    "root_cluster_geometry": "target_default_fk_offsets_in_target_root_frame",
                    "contact_patches": tuple(
                        (
                            patch.channel,
                            patch.body_name,
                            patch.points_per_body,
                            patch.height_band_m,
                            patch_body_values[index],
                            tuple(tuple(float(component) for component in point) for point in support_parts[index]),
                        )
                        for index, patch in enumerate(self.contact_patches)
                    ),
                    "contact_normal_ownership": {
                        "law": "contact_ankle_to_distal_body_chain_v1",
                        "body_channel_slots": tuple(body_normal_channel_values),
                        "position_channel_slots": position_normal_channel_values,
                    },
                    "leg_chains": {
                        "law": "target_rest_lengths_and_anatomical_unit_knee_pole_resolved_to_root_v1",
                        "values": tuple(
                            (
                                chain.channel,
                                chain.hip_body_name,
                                chain.knee_body_name,
                                chain.ankle_body_name,
                                leg_chain_body_values[index],
                                leg_chain_parent_values[index],
                                chain.knee_hint_anatomy,
                                leg_knee_hint_root_values[index],
                                leg_segment_length_values_m[index],
                            )
                            for index, chain in enumerate(self.leg_chains)
                        ),
                    },
                    "support_up_frame": self.support_up_frame,
                    "support_law": "target_support_down_collider_patch_factored_to_root",
                    "contact_forward_body": {
                        "values": contact_forward_body_values,
                        "law": "contact_direction_distal_minus_contact_origin_projected_off_patch_normal",
                    },
                    "contact_distal_point_body_m": {
                        "values": contact_distal_point_body_values,
                        "law": "contact_forward_body_scaled_by_target_contact_direction_length_v1",
                    },
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class MotionTrajectoryProjection:
    """Calibrate one source skeleton into a target-owned trajectory layout."""

    source_skeleton: MotionSkeleton
    target: _MotionTrajectoryTarget
    contact_channels: tuple[_MotionContactChannel, ...]
    contact_channel_probe_offsets: torch.Tensor
    evidence_layout_identity_sha256: str = field(init=False)
    _target_rotation_body_indices: tuple[int, ...] = field(init=False, repr=False)
    _target_rotation_weights: tuple[float, ...] = field(init=False, repr=False)
    _initializer_policy: Literal["direct", "batched_frame_ik"] = field(init=False, repr=False)
    construction_identity_sha256: str = field(init=False)
    _source_rest_translation_m: torch.Tensor = field(init=False, repr=False)
    _source_rest_rotation_xyzw: torch.Tensor = field(init=False, repr=False)
    _source_marker_position_indices: torch.Tensor = field(init=False, repr=False)
    _source_direction_from_position_indices: torch.Tensor = field(init=False, repr=False)
    _source_direction_to_position_indices: torch.Tensor = field(init=False, repr=False)
    _source_wrist_forward_rows: torch.Tensor = field(init=False, repr=False)
    _source_wrist_forward_rotation_body_indices: torch.Tensor = field(init=False, repr=False)
    _source_wrist_forward_local_axis: torch.Tensor = field(init=False, repr=False)
    _source_rotation_body_indices: torch.Tensor = field(init=False, repr=False)
    _source_to_target_rotation_xyzw: torch.Tensor = field(init=False, repr=False)
    _source_anatomical_basis_body_indices: tuple[int, int, int, int] = field(init=False, repr=False)
    _root_translation_scale: float = field(init=False, repr=False)
    _source_contact_probe_position_indices: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve only source role indices, rest calibration, and scale."""
        target = self.target
        reference = target.frame_target.kinematics
        source_by_name = {name: index for index, name in enumerate(self.source_skeleton.body_names)}
        source_landmarks = {landmark.name: landmark for landmark in self.source_skeleton.landmarks}
        anatomical_root = self.source_skeleton.landmark_rotation_policy == "anatomical_root"
        active_rotation_rows = (0,) if anatomical_root else tuple(range(len(target.rotation_roles)))
        active_rotation_roles = tuple(target.rotation_roles[row] for row in active_rotation_rows)
        target_rotation_body_indices = tuple(target.rotation_body_indices[row] for row in active_rotation_rows)
        target_rotation_weights = tuple(target.rotation_weights[row] for row in active_rotation_rows)
        initializer_policy = target.initializer_policy
        missing_position_roles = tuple(role for role in target.roles if role not in source_landmarks)
        missing_rotation_roles = (
            () if anatomical_root else tuple(role for role in active_rotation_roles if role not in source_landmarks)
        )
        direction_source_roles = tuple(
            role for point in target.direction_points for role in (point.source_from_role, point.source_to_role)
        )
        missing_direction_roles = tuple(role for role in direction_source_roles if role not in source_landmarks)
        channel_names = tuple(channel.name for channel in self.contact_channels)
        source_probe_roles = tuple(role for channel in self.contact_channels for role in channel.source_probe_roles)
        missing_probes = tuple(role for role in source_probe_roles if role not in source_landmarks)
        if channel_names != target.contact_channel_names:
            raise ValueError("Source contact channels and target contact patches must match in declaration order.")
        if missing_position_roles or missing_rotation_roles or missing_direction_roles or missing_probes:
            raise ValueError(
                f"Trajectory source lacks position roles {missing_position_roles}, rotation roles "
                f"{missing_rotation_roles}, direction roles {missing_direction_roles}, or contact probes "
                f"{missing_probes}."
            )

        device = torch.device(reference.device)
        source_position_values = tuple(
            source_by_name[source_landmarks[role].position_body_name] for role in target.roles
        )
        source_rotation_values = (
            ()
            if anatomical_root
            else tuple(source_by_name[source_landmarks[role].rotation_body_name] for role in active_rotation_roles)
        )
        source_direction_from_values = tuple(
            source_by_name[source_landmarks[point.source_from_role].position_body_name]
            for point in target.direction_points
        )
        source_direction_to_values = tuple(
            source_by_name[source_landmarks[point.source_to_role].position_body_name]
            for point in target.direction_points
        )
        wrist_forward_row_values = tuple(
            row for row, point in enumerate(target.direction_points) if point.source_direction_law == "wrist_forward"
        )
        source_wrist_forward_rotation_values = (
            ()
            if anatomical_root
            else tuple(
                source_by_name[source_landmarks[target.direction_points[row].source_to_role].rotation_body_name]
                for row in wrist_forward_row_values
            )
        )
        source_root = self.source_skeleton.parent_indices.index(-1)
        if source_position_values[0] != source_root:
            raise ValueError("Trajectory source pelvis position carrier must be the exact source root body.")
        if not anatomical_root and source_rotation_values[0] != source_root:
            raise ValueError("Calibrated trajectory source pelvis rotation carrier must be the exact source root body.")
        source_contact_probe_values = tuple(
            source_by_name[source_landmarks[role].position_body_name] for role in source_probe_roles
        )
        source_position_indices = torch.tensor(source_position_values, dtype=torch.int64, device=device)
        source_rotation_indices = torch.tensor(source_rotation_values, dtype=torch.int64, device=device)
        source_direction_from_indices = torch.tensor(source_direction_from_values, dtype=torch.int64, device=device)
        source_direction_to_indices = torch.tensor(source_direction_to_values, dtype=torch.int64, device=device)
        wrist_forward_rows = torch.tensor(wrist_forward_row_values, dtype=torch.int64, device=device)
        source_wrist_forward_rotation_indices = torch.tensor(
            source_wrist_forward_rotation_values,
            dtype=torch.int64,
            device=device,
        )
        source_contact_probe_indices = torch.tensor(source_contact_probe_values, dtype=torch.int64, device=device)
        expected_contact_probe_offsets = motion_contact_probe_offsets(self.contact_channels, device)
        if (
            self.contact_channel_probe_offsets.shape != (len(self.contact_channels) + 1,)
            or self.contact_channel_probe_offsets.dtype is not torch.int32
            or self.contact_channel_probe_offsets.device != device
            or not self.contact_channel_probe_offsets.is_contiguous()
            or not torch.equal(self.contact_channel_probe_offsets, expected_contact_probe_offsets)
        ):
            raise ValueError("Contact probe offsets must equal the canonical target-device int32 prefix tensor.")

        source_rest_translation_calibration = torch.tensor(
            self.source_skeleton.rest_translation_m, dtype=torch.float64, device="cpu"
        )
        source_rest_rotation_calibration = convert_quat(
            torch.tensor(self.source_skeleton.rest_rotation_wxyz, dtype=torch.float64, device="cpu"),
            to="xyzw",
        )
        source_rest_rotation_calibration.div_(
            torch.linalg.vector_norm(source_rest_rotation_calibration, dim=-1, keepdim=True)
        )
        source_rest_position_calibration, source_rest_world_rotation_calibration = kinematic_tree_forward(
            source_rest_translation_calibration,
            source_rest_rotation_calibration,
            self.source_skeleton.parent_indices,
        )
        source_rest_translation = source_rest_translation_calibration.to(dtype=torch.float32, device=device)
        source_rest_rotation = source_rest_rotation_calibration.to(dtype=torch.float32, device=device)
        root_cluster_body_indices = tuple(
            source_position_values[target.roles.index(role)] for role in _ROOT_CLUSTER_ROLES
        )
        source_anatomical_basis_indices = _anatomical_basis_indices(
            root_cluster_body_indices,
            self.source_skeleton.parent_indices,
            source_rest_position_calibration,
        )
        source_rest_anatomical_basis = kinematic_root_basis(
            source_rest_position_calibration, *source_anatomical_basis_indices
        )
        source_rest_anatomy_forward = source_rest_anatomical_basis[..., 0]
        if anatomical_root or not wrist_forward_row_values:
            source_wrist_forward_local_axis_calibration = torch.empty((0, 3), dtype=torch.float64, device="cpu")
        else:
            calibration_wrist_indices = torch.tensor(
                source_wrist_forward_rotation_values, dtype=torch.int64, device="cpu"
            )
            source_wrist_rest_rotation = source_rest_world_rotation_calibration.index_select(
                0, calibration_wrist_indices
            )
            source_wrist_forward_local_axis_calibration = quat_apply(
                quat_conjugate(source_wrist_rest_rotation),
                source_rest_anatomy_forward.expand(len(wrist_forward_row_values), -1),
            )
            source_wrist_forward_local_norm = torch.linalg.vector_norm(
                source_wrist_forward_local_axis_calibration, dim=-1, keepdim=True
            )
            if not bool(
                torch.all(torch.isfinite(source_wrist_forward_local_norm) & (source_wrist_forward_local_norm > 1.0e-9))
            ):
                raise ValueError("Source wrist-forward calibration must contain finite nonzero axes.")
            source_wrist_forward_local_axis_calibration.div_(source_wrist_forward_local_norm)
        source_wrist_forward_local_axis_values = tuple(
            tuple(float(component) for component in axis)
            for axis in source_wrist_forward_local_axis_calibration.tolist()
        )
        source_wrist_forward_local_axis = source_wrist_forward_local_axis_calibration.to(
            dtype=torch.float32, device=device
        )
        if anatomical_root:
            calibrated_body_axis_identity = ()
        else:
            calibrated_body_axis_identity = tuple(
                (
                    target.direction_points[row].name,
                    target.direction_points[row].source_to_role,
                    source_landmarks[target.direction_points[row].source_to_role].rotation_body_name,
                    body_index,
                    local_axis,
                )
                for row, body_index, local_axis in zip(
                    wrist_forward_row_values,
                    source_wrist_forward_rotation_values,
                    source_wrist_forward_local_axis_values,
                    strict=True,
                )
            )
        wrist_forward_calibration_identity = {
            "row_indices": wrist_forward_row_values,
            "source_anatomical_basis_body_indices": source_anatomical_basis_indices,
            "anatomical_root_law": "anatomy_forward_projected_orthogonal_to_source_from_to_direction",
            "calibrated_body_axes": calibrated_body_axis_identity,
        }
        anatomical_root_identity = {
            "source_basis_body_indices": source_anatomical_basis_indices,
            "target_basis_body_indices": target.anatomical_basis_body_indices,
            "target_basis_rotation_xyzw": tuple(float(value) for value in target.anatomical_rotation_xyzw.cpu()),
        }
        if anatomical_root:
            source_to_target_rotation = torch.empty((0, 4), dtype=torch.float32, device=device)
            rotation_calibration_identity = anatomical_root_identity
            rotation_law = "source_world_anatomical_basis_times_target_root_anatomical_basis_inverse"
        else:
            calibration_rotation_indices = torch.tensor(source_rotation_values, dtype=torch.int64, device="cpu")
            calibration_target_rotation_rest = target.rotation_rest_xyzw[list(active_rotation_rows)]
            source_rotation_rest_calibration = source_rest_world_rotation_calibration.index_select(
                0, calibration_rotation_indices
            )
            source_to_target_rotation_calibration = quat_mul(
                quat_conjugate(source_rotation_rest_calibration),
                calibration_target_rotation_rest,
            )
            source_to_target_rotation_norm = torch.linalg.vector_norm(
                source_to_target_rotation_calibration, dim=-1, keepdim=True
            )
            if not bool(
                torch.all(torch.isfinite(source_to_target_rotation_norm) & (source_to_target_rotation_norm > 1.0e-9))
            ):
                raise ValueError("Source-to-target rotation calibration must contain finite nondegenerate quaternions.")
            source_to_target_rotation_calibration.div_(source_to_target_rotation_norm)
            source_to_target_rotation_calibration = quat_unique(source_to_target_rotation_calibration)
            source_rotation_rest_values = tuple(
                tuple(float(component) for component in rotation)
                for rotation in quat_unique(source_rotation_rest_calibration).tolist()
            )
            source_to_target_rotation_values = tuple(
                tuple(float(component) for component in rotation)
                for rotation in source_to_target_rotation_calibration.tolist()
            )
            source_to_target_rotation = source_to_target_rotation_calibration.to(dtype=torch.float32, device=device)
            body_rotation_calibration_identity = tuple(
                (
                    role,
                    source_landmarks[role].rotation_body_name,
                    source_rotation_values[index],
                    source_rotation_rest_values[index],
                    source_to_target_rotation_values[index],
                )
                for index, role in enumerate(active_rotation_roles)
            )
            rotation_calibration_identity = {
                "anatomical_root": anatomical_root_identity,
                "body_rest_transport": body_rotation_calibration_identity,
            }
            rotation_law = "anatomical_root_gauge_times_rest_calibrated_world_body_rotations"

        calibration_position_indices = torch.tensor(source_position_values, dtype=torch.int64, device="cpu")
        calibration_parent_rows = torch.tensor(target.parent_rows[1:], dtype=torch.int64, device="cpu")
        calibration_parent_indices = calibration_position_indices.index_select(0, calibration_parent_rows)
        source_rest_edge_calibration = torch.zeros(len(target.roles), 3, dtype=torch.float64, device="cpu")
        source_rest_edge_calibration[1:] = (
            source_rest_position_calibration[calibration_position_indices[1:]]
            - source_rest_position_calibration[calibration_parent_indices]
        )
        source_lengths_calibration = torch.zeros(len(target.roles), dtype=torch.float64, device="cpu")
        source_lengths_calibration[1:] = torch.linalg.vector_norm(source_rest_edge_calibration[1:], dim=-1)
        torch._assert_async(
            torch.all(torch.isfinite(source_lengths_calibration[1:]) & (source_lengths_calibration[1:] > 1.0e-6)),
            "Every source landmark edge must have finite nonzero length.",
        )
        target_lengths_calibration = torch.tensor(target.segment_length_values_m, dtype=torch.float64, device="cpu")
        support_rows = tuple(
            target.direction_position_rows[target.direction_roles.index(channel)]
            for channel in target.contact_channel_names
        )
        source_support_length = source_lengths_calibration.new_zeros(())
        target_support_length = target_lengths_calibration.new_zeros(())
        for support_row in support_rows:
            row = support_row
            while row != 0:
                source_support_length.add_(source_lengths_calibration[row])
                target_support_length.add_(target_lengths_calibration[row])
                row = target.parent_rows[row]
        root_translation_scale = float(target_support_length / source_support_length)
        if self.source_skeleton.coordinate_identity_sha256 == target.frame_target.coordinate_profile_sha256:
            root_translation_scale = 1.0
        if (
            not math.isfinite(root_translation_scale)
            or root_translation_scale <= 0.0
            or not bool(torch.isfinite(source_support_length))
            or float(source_support_length) <= 1.0e-12
        ):
            raise ValueError("Source and target support chains do not define a positive root translation scale.")
        evidence_layout_identity = canonical_sha256(
            {
                "target_trajectory_sha256": target.construction_identity_sha256,
                "rotation_body_indices": target_rotation_body_indices,
                "rotation_weights": target_rotation_weights,
                "initializer_policy": initializer_policy,
            }
        )
        object.__setattr__(self, "evidence_layout_identity_sha256", evidence_layout_identity)
        object.__setattr__(self, "_target_rotation_body_indices", target_rotation_body_indices)
        object.__setattr__(self, "_target_rotation_weights", target_rotation_weights)
        object.__setattr__(self, "_initializer_policy", initializer_policy)
        object.__setattr__(self, "_source_rest_translation_m", source_rest_translation)
        object.__setattr__(self, "_source_rest_rotation_xyzw", source_rest_rotation)
        object.__setattr__(self, "_source_marker_position_indices", source_position_indices)
        object.__setattr__(self, "_source_direction_from_position_indices", source_direction_from_indices)
        object.__setattr__(self, "_source_direction_to_position_indices", source_direction_to_indices)
        object.__setattr__(self, "_source_wrist_forward_rows", wrist_forward_rows)
        object.__setattr__(self, "_source_wrist_forward_rotation_body_indices", source_wrist_forward_rotation_indices)
        object.__setattr__(self, "_source_wrist_forward_local_axis", source_wrist_forward_local_axis)
        object.__setattr__(self, "_source_rotation_body_indices", source_rotation_indices)
        object.__setattr__(self, "_source_to_target_rotation_xyzw", source_to_target_rotation)
        object.__setattr__(self, "_source_anatomical_basis_body_indices", source_anatomical_basis_indices)
        object.__setattr__(self, "_root_translation_scale", root_translation_scale)
        object.__setattr__(self, "_source_contact_probe_position_indices", source_contact_probe_indices)
        object.__setattr__(
            self,
            "construction_identity_sha256",
            canonical_sha256(
                {
                    "target_trajectory_sha256": target.construction_identity_sha256,
                    "evidence_layout_sha256": evidence_layout_identity,
                    "projection_math_version": _TRAJECTORY_PROJECTION_MATH_VERSION,
                    "landmark_rotation_policy": self.source_skeleton.landmark_rotation_policy,
                    "source_skeleton_sha256": self.source_skeleton.identity_sha256,
                    "contact_channels": tuple(
                        (channel.name, channel.source_probe_roles) for channel in self.contact_channels
                    ),
                    "rotation_calibration": rotation_calibration_identity,
                    "rotation_law": rotation_law,
                    "position_law": "target_root_cluster_and_segment_lengths_with_source_directions",
                    "direction_calibration": tuple(
                        (
                            point.name,
                            point.source_direction_law,
                            point.source_from_role,
                            source_landmarks[point.source_from_role].position_body_name,
                            source_direction_from_values[index],
                            point.source_to_role,
                            source_landmarks[point.source_to_role].position_body_name,
                            source_direction_to_values[index],
                        )
                        for index, point in enumerate(target.direction_points)
                    ),
                    "wrist_forward_calibration": wrist_forward_calibration_identity,
                    "direction_point_laws": {
                        "between_positions": (
                            "target_base_plus_normalized_source_position_direction_times_target_length"
                        ),
                        "wrist_forward": "target_base_plus_source_policy_forward_times_target_length",
                    },
                    "initialization_law": "target_owned_trajectory_seed_from_active_rotation_evidence",
                    "root_translation_scale": root_translation_scale,
                    "root_translation_scale_law": "target_over_source_contact_base_root_path_length",
                    "source_ground_law": "minimum_raw_source_contact_probe_z",
                    "root_translation_law": "xy_source_0+s*(xy_source_t-xy_source_0);z=s*(z_source_t-z_source_ground)",
                    "input_representation": "world_root_and_parent_local_pose_delta_xyzw_v1",
                }
            ),
        )

    def generate_targets(
        self,
        source_root_position: torch.Tensor,
        source_body_rotation_xyzw: torch.Tensor,
    ) -> MotionTrajectoryTargets:
        """Project one source clip into independent morphology targets, coordinate seeds, and support geometry."""
        target = self.target
        frame_count = source_root_position.shape[0] if source_root_position.ndim == 2 else -1
        if (
            source_root_position.shape != (frame_count, 3)
            or source_body_rotation_xyzw.shape != (frame_count, self.source_skeleton.num_bodies, 4)
            or frame_count < 1
            or source_root_position.device != self._source_rest_translation_m.device
            or source_body_rotation_xyzw.device != source_root_position.device
            or source_root_position.dtype is not torch.float32
            or source_body_rotation_xyzw.dtype is not torch.float32
        ):
            raise ValueError("Trajectory source tensors must be nonempty target-device float32 pose rows.")
        torch._assert_async(
            torch.all(torch.isfinite(source_root_position)),
            "Trajectory source root positions must be finite [m].",
        )
        source_rotation_norm = torch.linalg.vector_norm(source_body_rotation_xyzw, dim=-1)
        torch._assert_async(
            torch.all(
                torch.isfinite(source_body_rotation_xyzw)
                & torch.isfinite(source_rotation_norm[..., None])
                & (torch.abs(source_rotation_norm[..., None] - 1.0) <= 1.0e-4)
            ),
            "Trajectory source body rotations must be finite unit quaternions.",
        )
        source_position, source_world_rotation = kinematic_pose_forward(
            self._source_rest_translation_m,
            self._source_rest_rotation_xyzw,
            source_body_rotation_xyzw,
            source_root_position,
            self.source_skeleton.parent_indices,
        )
        source_contact_probe_position = (
            source_position.index_select(1, self._source_contact_probe_position_indices).transpose(0, 1).contiguous()
        )
        source_ground_height_m = torch.amin(source_contact_probe_position[..., 2])
        source_anatomical_basis = kinematic_root_basis(source_position, *self._source_anatomical_basis_body_indices)
        source_anatomical_rotation = quat_from_matrix(source_anatomical_basis)
        target_anatomical_inverse = quat_conjugate(target.anatomical_rotation_xyzw).expand(frame_count, 4)
        anatomical_root_rotation = quat_mul(source_anatomical_rotation, target_anatomical_inverse)
        if self.source_skeleton.landmark_rotation_policy == "anatomical_root":
            source_landmark_rotation = anatomical_root_rotation.unsqueeze(1)
        else:
            source_landmark_rotation = source_world_rotation.index_select(1, self._source_rotation_body_indices)
            source_to_target_rotation = self._source_to_target_rotation_xyzw.unsqueeze(0).expand(frame_count, -1, -1)
            source_landmark_rotation = quat_mul(source_landmark_rotation, source_to_target_rotation)
            mapped_root_rotation = source_landmark_rotation[:, 0]
            mapped_root_rotation = mapped_root_rotation / torch.linalg.vector_norm(
                mapped_root_rotation, dim=-1, keepdim=True
            )
            anatomical_root_gauge = quat_mul(anatomical_root_rotation, quat_conjugate(mapped_root_rotation))
            source_landmark_rotation = quat_mul(
                anatomical_root_gauge.unsqueeze(1).expand_as(source_landmark_rotation),
                source_landmark_rotation,
            )
        source_landmark_rotation_norm = torch.linalg.vector_norm(source_landmark_rotation, dim=-1, keepdim=True)
        torch._assert_async(
            torch.all(torch.isfinite(source_landmark_rotation_norm) & (source_landmark_rotation_norm > 1.0e-8)),
            "Trajectory rotation targets must be finite nondegenerate quaternions.",
        )
        source_landmark_rotation.div_(source_landmark_rotation_norm)
        source_landmark_rotation = quat_unique(source_landmark_rotation)
        source_landmark_rotation = source_landmark_rotation.transpose(0, 1).contiguous()
        target_root_rotation = source_landmark_rotation[0]
        target_root_position = source_root_position[:1] + self._root_translation_scale * (
            source_root_position - source_root_position[:1]
        )
        target_root_position[:, 2] = self._root_translation_scale * (
            source_root_position[:, 2] - source_ground_height_m
        )
        source_marker_position = (
            source_position.index_select(1, self._source_marker_position_indices).transpose(0, 1).contiguous()
        )

        joint_q = target.frame_target.trajectory_seed_joint_q(
            root_position_m=target_root_position,
            rotation_body_indices=self._target_rotation_body_indices,
            landmark_rotation_xyzw=source_landmark_rotation,
        )
        coordinate_values = joint_q.index_select(1, target.coordinate_indices)
        if (
            joint_q.shape != (frame_count, target.frame_target.kinematics.model.joint_coord_count)
            or joint_q.dtype is not torch.float32
            or joint_q.device != source_root_position.device
            or not joint_q.is_contiguous()
            or not bool(torch.all(torch.isfinite(joint_q)))
            or not torch.equal(joint_q[:, :3], target_root_position)
            or not torch.equal(joint_q[:, 3:7], target_root_rotation)
            or not bool(
                torch.all(
                    (coordinate_values >= target.coordinate_lower_limits_rad)
                    & (coordinate_values <= target.coordinate_upper_limits_rad)
                )
            )
        ):
            raise ValueError(
                "Target trajectory seed must be exact at the root and finite, contiguous, and hard-limit-valid."
            )
        support_point_root = target.support_point_root_m[:, None].expand(-1, frame_count, -1)
        support_root_rotation = target_root_rotation[None].expand(len(target.support_point_root_m), -1, -1)
        target_support_position = target_root_position[None] + quat_apply(
            support_root_rotation.reshape(-1, 4), support_point_root.reshape(-1, 3)
        ).view_as(support_point_root)

        source_semantic_position = target_root_position[None] + source_marker_position - source_marker_position[:1]

        semantic_position = torch.empty_like(source_semantic_position)
        root_cluster_offset = target.root_cluster_offset_m[:, None].expand(-1, frame_count, -1)
        root_cluster_rotation = target_root_rotation[None].expand(len(target.root_cluster_rows), -1, -1)
        root_cluster_position = target_root_position[None] + quat_apply(
            root_cluster_rotation.reshape(-1, 4), root_cluster_offset.reshape(-1, 3)
        ).view_as(root_cluster_offset)
        for index, row in enumerate(target.root_cluster_rows):
            semantic_position[row].copy_(root_cluster_position[index])
        for row, parent in enumerate(target.parent_rows[1:], start=1):
            if row in target.root_cluster_rows:
                continue
            source_direction = source_semantic_position[row] - source_semantic_position[parent]
            source_direction_norm = torch.linalg.vector_norm(source_direction, dim=-1, keepdim=True)
            torch._assert_async(
                torch.all(torch.isfinite(source_direction_norm) & (source_direction_norm > 1.0e-8)),
                f"Source semantic direction row {row} must be finite and nondegenerate.",
            )
            source_direction.div_(source_direction_norm)
            semantic_position[row].copy_(semantic_position[parent] + target.segment_lengths_m[row] * source_direction)

        source_direction_from = source_position.index_select(1, self._source_direction_from_position_indices).transpose(
            0, 1
        )
        source_direction_to = source_position.index_select(1, self._source_direction_to_position_indices).transpose(
            0, 1
        )
        source_direction = source_direction_to - source_direction_from
        source_direction_norm = torch.linalg.vector_norm(source_direction, dim=-1, keepdim=True)
        torch._assert_async(
            torch.all(torch.isfinite(source_direction_norm) & (source_direction_norm > 1.0e-8)),
            "Source endpoint directions must be finite and nondegenerate.",
        )
        source_direction.div_(source_direction_norm)
        if self._source_wrist_forward_rows.numel() > 0:
            if self.source_skeleton.landmark_rotation_policy == "anatomical_root":
                wrist_forearm = source_direction.index_select(0, self._source_wrist_forward_rows)
                anatomy_forward = source_anatomical_basis[..., 0].unsqueeze(0).expand_as(wrist_forearm)
                wrist_direction = anatomy_forward - (
                    torch.sum(anatomy_forward * wrist_forearm, dim=-1, keepdim=True) * wrist_forearm
                )
            else:
                wrist_world_rotation = source_world_rotation.index_select(
                    1, self._source_wrist_forward_rotation_body_indices
                ).transpose(0, 1)
                wrist_local_axis = self._source_wrist_forward_local_axis[:, None].expand(-1, frame_count, -1)
                wrist_direction = quat_apply(
                    wrist_world_rotation.reshape(-1, 4), wrist_local_axis.reshape(-1, 3)
                ).view_as(wrist_local_axis)
            wrist_direction_norm = torch.linalg.vector_norm(wrist_direction, dim=-1, keepdim=True)
            torch._assert_async(
                torch.all(torch.isfinite(wrist_direction_norm) & (wrist_direction_norm > 1.0e-8)),
                "Source wrist-forward directions must be finite and nondegenerate.",
            )
            wrist_direction.div_(wrist_direction_norm)
            source_direction.index_copy_(0, self._source_wrist_forward_rows, wrist_direction)
        source_direction_point_position = semantic_position.index_select(0, target.direction_rows) + (
            target.direction_lengths_m[:, None, None] * source_direction
        )

        return MotionTrajectoryTargets(
            position_body_indices=target.position_body_indices,
            root_body_index=target.root_body_index,
            source_root_policy=target.source_root_policy,
            initializer_policy=self._initializer_policy,
            parent_rows=target.parent_rows,
            parent_row_tensor=target.parent_row_tensor,
            position_weights=target.position_weights,
            required_position_rows=target.required_position_rows,
            required_position_row_tensor=target.required_position_row_tensor,
            position_normal_channel_slots=target.position_normal_channel_slots,
            position_body_index_tensor=target.position_body_index_tensor,
            rotation_body_indices=self._target_rotation_body_indices,
            rotation_weights=self._target_rotation_weights,
            source_landmark_rotation_xyzw=source_landmark_rotation,
            direction_body_indices=target.direction_body_indices,
            direction_position_rows=target.direction_position_rows,
            direction_weights=target.direction_weights,
            contact_direction_rows=target.contact_direction_rows,
            contact_direction_row_tensor=target.contact_direction_row_tensor,
            direction_contact_channel_slots=target.direction_contact_channel_slots,
            required_direction_rows=target.required_direction_rows,
            required_direction_row_tensor=target.required_direction_row_tensor,
            direction_body_index_tensor=target.direction_body_index_tensor,
            direction_position_row_tensor=target.direction_rows,
            direction_point_body_m=target.direction_point_body_m,
            source_landmark_position_m=semantic_position,
            source_direction_point_position_m=source_direction_point_position,
            direction_length_values_m=target.direction_length_values_m,
            initial_joint_q=joint_q,
            segment_lengths_m=target.segment_lengths_m,
            segment_length_values_m=target.segment_length_values_m,
            coordinate_indices=target.coordinate_indices,
            coordinate_lower_limits_rad=target.coordinate_lower_limits_rad,
            coordinate_upper_limits_rad=target.coordinate_upper_limits_rad,
            source_contact_probe_position_m=source_contact_probe_position,
            contact_channel_probe_offsets=self.contact_channel_probe_offsets,
            target_support_position_m=target_support_position,
            contact_body_indices=target.contact_body_indices,
            contact_normal_body=target.contact_normal_body,
            contact_forward_body=target.contact_forward_body,
            contact_distal_point_body_m=target.contact_distal_point_body_m,
            leg_chain_body_indices=target.leg_chain_body_indices,
            leg_chain_parent_body_indices=target.leg_chain_parent_body_indices,
            leg_knee_hint_anatomy=target.leg_knee_hint_anatomy,
            leg_knee_hint_root=target.leg_knee_hint_root,
            leg_segment_lengths_m=target.leg_segment_lengths_m,
            support_patch_offsets=target.support_patch_offsets,
            support_body_indices=target.support_body_indices,
            support_point_body_m=target.support_point_body_m,
            support_channel_slots=target.support_channel_slots,
        )
