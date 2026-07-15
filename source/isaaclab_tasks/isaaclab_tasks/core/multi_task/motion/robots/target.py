# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared target-robot coordinate, velocity, and frame construction boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
import warp as wp

from ...kinematics.ik_objectives.mesh_collision import collision_probes_sample

if TYPE_CHECKING:
    from ...kinematics import KinematicTree, NewtonKinematics
    from ..data.clip_index import MotionClipIndex
    from ..data.frames import MotionFrames, MotionGeneralizedCoordinates


@runtime_checkable
class MotionFrameTarget(Protocol):
    """Robot-owned generalized-coordinate and frame construction."""

    @property
    def version(self) -> str:
        """Declared target-coordinate construction version."""

    @property
    def construction_identity_sha256(self) -> str:
        """Source-independent target-coordinate construction identity."""

    @property
    def coordinate_profile_sha256(self) -> str:
        """Identity of source coordinates already native to this target."""

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Ordered simulator joint names."""

    @property
    def joint_q_indices(self) -> tuple[int, ...]:
        """Newton generalized-position indices matching :attr:`joint_names`."""

    @property
    def reference_frame_names(self) -> tuple[str, ...]:
        """Ordered physical and derived reference-frame names."""

    @property
    def kinematics(self) -> NewtonKinematics:
        """Scene-derived target-robot mechanics used by every coordinate route."""

    @property
    def kinematic_tree(self) -> KinematicTree:
        """Grouped topology and hard coordinate limits of the selected target robot."""

    @property
    def materialization_minimum_frames(self) -> int:
        """Minimum complete-clip frames required by the target output law."""

    @property
    def collision_probe_body_indices(self) -> torch.Tensor:
        """Target body index per shared collision probe, shape [probe_count]."""

    @property
    def collision_probe_offsets_m(self) -> torch.Tensor:
        """Target body-local collision probe offsets [m], shape [probe_count, 3]."""

    @property
    def collision_probe_contact_slots(self) -> torch.Tensor:
        """Source contact channel per collision probe, or -1, shape [probe_count]."""

    @property
    def collision_probe_normal_channel_slots(self) -> torch.Tensor:
        """Rigid contact-normal owner per collision probe, or -1, shape [probe_count]."""

    @property
    def collision_geometry_identity_sha256(self) -> str:
        """Identity of the target-owned collision probe geometry."""

    def trajectory_seed_joint_q(
        self,
        *,
        root_position_m: torch.Tensor,
        rotation_body_indices: tuple[int, ...],
        landmark_rotation_xyzw: torch.Tensor,
    ) -> torch.Tensor:
        """Construct one target-owned, hard-limit-valid trajectory initializer.

        Args:
            root_position_m: Target root positions [m], shape [frame_count, 3].
            rotation_body_indices: Target body per semantic rotation row.
            landmark_rotation_xyzw: Desired world rotations, shape [rotation_count, frame_count, 4].

        Returns:
            Newton generalized positions [m or rad, depending on joint type], shape [frame_count, coordinate_count].
        """

    def allocate_coordinates(self, frame_count: int, *, device: str | torch.device) -> MotionGeneralizedCoordinates:
        """Allocate exact-capacity target generalized coordinates.

        Args:
            frame_count: Number of target frames.
            device: Device for every coordinate tensor.

        Returns:
            Empty target coordinates with exactly :paramref:`frame_count` rows.
        """

    def coordinates_from_newton(
        self,
        joint_q: torch.Tensor,
        clip_index: MotionClipIndex,
    ) -> MotionGeneralizedCoordinates:
        """Convert one Newton-coordinate corpus to target generalized coordinates.

        Args:
            joint_q: Newton generalized positions [m or rad, depending on joint type].
            clip_index: Complete source-ordered clip boundaries and sampling rates.

        Returns:
            Final target coordinates in their stored representation.
        """

    def write_joint_position_newton(self, coordinates: MotionGeneralizedCoordinates, output: torch.Tensor) -> None:
        """Write positions [m or rad, depending on joint type] in Newton order.

        Args:
            coordinates: Stored robot generalized coordinates.
            output: Newton generalized positions, shape [frame_count, coordinate_count].
        """

    def write_nonroot_velocity_canonical(
        self,
        joint_q: torch.Tensor,
        clip_offsets: torch.Tensor,
        step_seconds: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Write canonical non-root rates into full Newton velocity storage.

        The first six free-root columns of :paramref:`output` must be preserved.
        Every later row describes the edge reaching that row, and each complete
        clip head repeats its first real edge.

        Args:
            joint_q: Newton generalized positions [m or rad, depending on joint
                type], shape [frame_count, coordinate_count].
            clip_offsets: Half-open complete-clip offsets, shape [clip_count + 1].
            step_seconds: Sampling period per complete clip [s], shape [clip_count].
            output: Newton generalized velocities [m/s or rad/s, depending on
                joint type], shape [frame_count, dof_count].
        """

    def materialize_coordinates(
        self,
        coordinates: MotionGeneralizedCoordinates,
        clip_index: MotionClipIndex,
    ) -> MotionFrames:
        """Materialize robot frames once from final source-ordered coordinates.

        Args:
            coordinates: Certified final target coordinates.
            clip_index: Complete source-ordered clip boundaries and sampling rates.

        Returns:
            Robot-oriented state and reference-frame tensors.
        """


_MOTION_COLLISION_PROBES_PER_BODY = 4


def motion_collision_probe_geometry(
    reference: NewtonKinematics,
    support_body_indices: torch.Tensor,
    support_point_body_m: torch.Tensor,
    support_channel_slots: torch.Tensor,
    body_normal_channel_slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the one target-owned collision probe layout used by solve and certificate."""
    support_count = support_body_indices.shape[0]
    body_count = reference.model.body_count
    if (
        support_body_indices.shape != (support_count,)
        or support_body_indices.dtype is not torch.int64
        or support_point_body_m.shape != (support_count, 3)
        or support_point_body_m.dtype is not torch.float32
        or support_channel_slots.shape != (support_count,)
        or support_channel_slots.dtype is not torch.int64
        or body_normal_channel_slots.shape != (body_count,)
        or body_normal_channel_slots.dtype is not torch.int64
        or not body_normal_channel_slots.is_contiguous()
        or any(
            tensor.device != torch.device(reference.device)
            for tensor in (support_body_indices, support_point_body_m, support_channel_slots, body_normal_channel_slots)
        )
    ):
        raise ValueError("Target collision support geometry has an invalid shape, dtype, or device.")
    torch._assert_async(
        torch.all(body_normal_channel_slots >= -1),
        "Target body normal-channel slots must be -1 or a declared source contact channel.",
    )
    support_bodies = tuple(int(value) for value in support_body_indices.detach().cpu().tolist())
    probe_bodies, probe_offsets, probe_support_slots = collision_probes_sample(
        reference.builder,
        support_bodies,
        _MOTION_COLLISION_PROBES_PER_BODY,
        contact_points_body=support_point_body_m.detach().cpu().numpy(),
    )
    if not len(probe_bodies):
        raise ValueError("Target collision certification requires at least one collision probe.")
    support_channels = support_channel_slots.detach().cpu().numpy()
    probe_contact_slots = probe_support_slots.copy()
    contact_probes = probe_support_slots >= 0
    probe_contact_slots[contact_probes] = support_channels[probe_support_slots[contact_probes]]
    device = torch.device(reference.device)
    probe_body_tensor = torch.as_tensor(probe_bodies, dtype=torch.int64, device=device)
    return (
        probe_body_tensor,
        torch.as_tensor(probe_offsets, dtype=torch.float32, device=device),
        torch.as_tensor(probe_contact_slots, dtype=torch.int64, device=device),
        body_normal_channel_slots.index_select(0, probe_body_tensor),
    )


def validate_collision_probe_geometry(
    target: MotionFrameTarget,
    *,
    device: str | torch.device,
    contact_channel_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate and return one target-owned collision probe layout.

    Args:
        target: Target robot owning the collision geometry.
        device: Device required by the current consumer.
        contact_channel_count: Optional number of declared source contact channels.

    Returns:
        Body indices, body-local offsets [m], soft-contact slots, and rigid normal-owner slots.
    """
    probe_bodies = target.collision_probe_body_indices
    probe_offsets_m = target.collision_probe_offsets_m
    probe_contact_slots = target.collision_probe_contact_slots
    probe_normal_slots = target.collision_probe_normal_channel_slots
    probe_count = probe_bodies.shape[0]
    expected_device = torch.device(device)
    if contact_channel_count is not None and contact_channel_count < 0:
        raise ValueError("Target collision contact-channel count must be nonnegative.")
    if (
        probe_count < 1
        or probe_bodies.shape != (probe_count,)
        or probe_bodies.dtype is not torch.int64
        or probe_offsets_m.shape != (probe_count, 3)
        or probe_offsets_m.dtype is not torch.float32
        or probe_contact_slots.shape != (probe_count,)
        or probe_contact_slots.dtype is not torch.int64
        or probe_normal_slots.shape != (probe_count,)
        or probe_normal_slots.dtype is not torch.int64
        or any(
            tensor.device != expected_device
            for tensor in (probe_bodies, probe_offsets_m, probe_contact_slots, probe_normal_slots)
        )
        or not all(
            tensor.is_contiguous()
            for tensor in (probe_bodies, probe_offsets_m, probe_contact_slots, probe_normal_slots)
        )
    ):
        raise ValueError("Target collision probes must be aligned contiguous tensors on the consumer device.")
    torch._assert_async(
        torch.all((probe_bodies >= 0) & (probe_bodies < target.kinematics.model.body_count)),
        "Target collision probe body indices must lie inside the Newton body layout.",
    )
    torch._assert_async(
        torch.all(torch.isfinite(probe_offsets_m)),
        "Target collision probe offsets must be finite [m].",
    )
    contact_slots_valid = probe_contact_slots >= -1
    normal_slots_valid = probe_normal_slots >= -1
    if contact_channel_count is not None:
        contact_slots_valid.logical_and_(probe_contact_slots < contact_channel_count)
        normal_slots_valid.logical_and_(probe_normal_slots < contact_channel_count)
    torch._assert_async(
        torch.all(contact_slots_valid),
        "Target collision probe contact slots must be -1 or a declared source contact channel.",
    )
    torch._assert_async(
        torch.all(normal_slots_valid),
        "Target collision probe normal slots must be -1 or a declared source contact channel.",
    )
    return probe_bodies, probe_offsets_m, probe_contact_slots, probe_normal_slots


@wp.kernel
def _motion_ground_penetration_frames(
    body_q: wp.array3d(dtype=wp.float32),
    probe_bodies: wp.array(dtype=wp.int64),
    probe_offsets_m: wp.array2d(dtype=wp.float32),
    frame_count: int,
    probe_count: int,
    quality_column: int,
    frame_evidence: wp.array2d(dtype=wp.float32),
):
    """Measure maximum target-collider penetration below world ground z=0 [m]."""
    frame = wp.tid()
    if frame >= frame_count:
        return
    penetration = float(0.0)
    for probe in range(probe_count):
        body = probe_bodies[probe]
        position = wp.vec3(body_q[frame, body, 0], body_q[frame, body, 1], body_q[frame, body, 2])
        rotation = wp.quat(
            body_q[frame, body, 3],
            body_q[frame, body, 4],
            body_q[frame, body, 5],
            body_q[frame, body, 6],
        )
        offset = wp.vec3(probe_offsets_m[probe, 0], probe_offsets_m[probe, 1], probe_offsets_m[probe, 2])
        point = position + wp.quat_rotate(rotation, offset)
        if not wp.isfinite(point[0]) or not wp.isfinite(point[1]) or not wp.isfinite(point[2]):
            penetration = float(wp.inf)
        else:
            penetration = wp.max(penetration, wp.max(-point[2], 0.0))
    frame_evidence[frame, quality_column] = penetration


def write_ground_penetration(
    target: MotionFrameTarget,
    body_q: torch.Tensor,
    frame_evidence: torch.Tensor,
    *,
    quality_column: int,
) -> None:
    """Write target-owned ground penetration evidence into existing storage.

    Call validate_collision_probe_geometry once before a batched sequence.

    Args:
        target: Target robot owning the validated collision geometry.
        body_q: Newton body transforms, shape [frame_count, body_count, 7].
        frame_evidence: Existing per-frame quality storage.
        quality_column: Destination column in frame_evidence.
    """
    model = target.kinematics.model
    frame_count = body_q.shape[0]
    if (
        frame_count < 1
        or body_q.shape != (frame_count, model.body_count, 7)
        or body_q.dtype is not torch.float32
        or frame_evidence.ndim != 2
        or frame_evidence.shape[0] != frame_count
        or frame_evidence.dtype is not torch.float32
        or quality_column < 0
        or quality_column >= frame_evidence.shape[1]
        or body_q.device != torch.device(target.kinematics.device)
        or frame_evidence.device != body_q.device
        or not body_q.is_contiguous()
        or not frame_evidence.is_contiguous()
    ):
        raise ValueError("Ground penetration requires aligned contiguous float32 target evidence storage.")
    probe_bodies = target.collision_probe_body_indices
    probe_offsets_m = target.collision_probe_offsets_m
    wp.launch(
        _motion_ground_penetration_frames,
        dim=frame_count,
        inputs=[
            wp.from_torch(body_q),
            wp.from_torch(probe_bodies),
            wp.from_torch(probe_offsets_m),
            frame_count,
            probe_bodies.shape[0],
            quality_column,
        ],
        outputs=[wp.from_torch(frame_evidence)],
        device=str(body_q.device),
    )


@wp.kernel
def _free_root_velocity_canonical_warp(
    joint_q: wp.array2d(dtype=wp.float32),
    root_body_com: wp.array(dtype=wp.vec3),
    root_body_index: int,
    clip_offsets: wp.array(dtype=wp.int32),
    step_seconds: wp.array(dtype=wp.float32),
    clip_count: int,
    frame_count: int,
    joint_qd: wp.array2d(dtype=wp.float32),
):
    """Write destination-indexed Newton free-root velocities."""
    frame = wp.tid()
    if frame >= frame_count:
        return
    low = int(0)
    high = clip_count
    while low + 1 < high:
        middle = (low + high) // 2
        if frame < clip_offsets[middle]:
            high = middle
        else:
            low = middle
    start = clip_offsets[low]
    source = frame - 1
    destination = frame
    if frame == start:
        source = start
        destination = start + 1
    inverse_dt = 1.0 / step_seconds[low]

    source_rotation = wp.quat(joint_q[source, 3], joint_q[source, 4], joint_q[source, 5], joint_q[source, 6])
    destination_rotation = wp.quat(
        joint_q[destination, 3], joint_q[destination, 4], joint_q[destination, 5], joint_q[destination, 6]
    )
    relative = wp.mul(destination_rotation, wp.quat_inverse(source_rotation))
    relative = relative * (1.0 / wp.max(wp.sqrt(wp.dot(relative, relative)), 1.0e-9))
    if relative[3] < 0.0:
        relative = relative * -1.0
    magnitude = wp.sqrt(relative[0] * relative[0] + relative[1] * relative[1] + relative[2] * relative[2])
    angle = 2.0 * wp.atan2(magnitude, relative[3])
    scale = 2.0
    if wp.abs(angle) > 1.0e-6:
        scale = angle / magnitude
    else:
        scale = 1.0 / (0.5 - angle * angle / 48.0)
    angular_velocity = wp.vec3(relative[0], relative[1], relative[2]) * (scale * inverse_dt)
    state_rotation = wp.quat(joint_q[frame, 3], joint_q[frame, 4], joint_q[frame, 5], joint_q[frame, 6])
    state_rotation = state_rotation * (1.0 / wp.max(wp.sqrt(wp.dot(state_rotation, state_rotation)), 1.0e-9))
    root_com_world = wp.quat_rotate(state_rotation, root_body_com[root_body_index])
    origin_velocity = wp.vec3(
        (joint_q[destination, 0] - joint_q[source, 0]) * inverse_dt,
        (joint_q[destination, 1] - joint_q[source, 1]) * inverse_dt,
        (joint_q[destination, 2] - joint_q[source, 2]) * inverse_dt,
    )
    linear_velocity = origin_velocity + wp.cross(angular_velocity, root_com_world)
    for axis in range(3):
        joint_qd[frame, axis] = linear_velocity[axis]
        joint_qd[frame, axis + 3] = angular_velocity[axis]


def _validate_velocity_clock(
    frame_count: int,
    clip_offsets: torch.Tensor,
    step_seconds: torch.Tensor,
) -> None:
    """Validate one complete segmented output clock without a host synchronization."""
    clip_count = clip_offsets.shape[0] - 1
    if (
        frame_count < 2
        or clip_count < 1
        or clip_offsets.shape != (clip_count + 1,)
        or step_seconds.shape != (clip_count,)
        or clip_offsets.dtype is not torch.int32
        or step_seconds.dtype is not torch.float32
        or clip_offsets.device != step_seconds.device
        or not clip_offsets.is_contiguous()
        or not step_seconds.is_contiguous()
    ):
        raise ValueError("Canonical velocity clock tensors have an invalid shape, dtype, device, or stride.")
    torch._assert_async(clip_offsets[0] == 0, "Canonical velocity offsets must start at zero.")
    torch._assert_async(clip_offsets[-1] == frame_count, "Canonical velocity offsets must cover every frame.")
    torch._assert_async(
        torch.all(clip_offsets[1:] - clip_offsets[:-1] >= 2),
        "Canonical velocity clips must contain at least two frames.",
    )
    torch._assert_async(
        torch.all(torch.isfinite(step_seconds) & (step_seconds > 0.0)),
        "Canonical velocity sample intervals must be finite and positive [s].",
    )


def write_velocity_canonical(
    target: MotionFrameTarget,
    joint_q: torch.Tensor,
    clip_offsets: torch.Tensor,
    step_seconds: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Write one route-independent canonical target velocity corpus.

    Args:
        target: Target robot owning non-root coordinate-rate semantics.
        joint_q: Newton generalized positions [m or rad, depending on joint
            type], shape [frame_count, coordinate_count].
        clip_offsets: Half-open complete-clip offsets, shape [clip_count + 1].
        step_seconds: Sampling period per complete clip [s], shape [clip_count].
        output: Newton generalized velocities [m/s or rad/s, depending on
            joint type], shape [frame_count, dof_count].
    """
    model = target.kinematics.model
    frame_count = joint_q.shape[0]
    clip_count = clip_offsets.shape[0] - 1
    root_body_index = target.kinematic_tree.root_body_index
    if root_body_index < 0 or root_body_index >= model.body_count:
        raise ValueError("Canonical target velocity root body is outside the Newton body layout.")
    _validate_velocity_clock(frame_count, clip_offsets, step_seconds)
    if (
        joint_q.shape != (frame_count, model.joint_coord_count)
        or output.shape != (frame_count, model.joint_dof_count)
        or frame_count < 2
        or clip_count < 1
        or step_seconds.shape != (clip_count,)
        or joint_q.dtype is not torch.float32
        or clip_offsets.dtype is not torch.int32
        or step_seconds.dtype is not torch.float32
        or output.dtype is not torch.float32
        or any(tensor.device != joint_q.device for tensor in (clip_offsets, step_seconds, output))
        or not all(tensor.is_contiguous() for tensor in (joint_q, clip_offsets, step_seconds, output))
    ):
        raise ValueError("Canonical target velocity requires aligned contiguous float32/int32 tensors.")
    wp.launch(
        _free_root_velocity_canonical_warp,
        dim=frame_count,
        inputs=[
            wp.from_torch(joint_q),
            model.body_com,
            root_body_index,
            wp.from_torch(clip_offsets),
            wp.from_torch(step_seconds),
            clip_count,
            frame_count,
        ],
        outputs=[wp.from_torch(output)],
        device=str(joint_q.device),
    )
    target.write_nonroot_velocity_canonical(joint_q, clip_offsets, step_seconds, output)
