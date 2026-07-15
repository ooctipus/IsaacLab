# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose decoded source routes into one robot-oriented motion task table."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import torch

from ....kinematics import NewtonKinematics, plan_trajectory_memory
from ....mdp.commands.state_command import (
    ResetStateBank,
    ResetStateLayout,
    TaskTableKinematicView,
    TaskTableLineEvidence,
    TaskTablePointEvidence,
    TaskTableQuality,
    TaskTableSequenceIndex,
    TaskTableView,
    execute_task_family,
    make_task_table_rng,
)
from ...data.clip_index import MotionClipIndex
from ...data.frames import (
    MotionFrames,
    MotionGeneralizedCoordinates,
    MotionSourceProjection,
    MotionSourceProjectionAnalytic,
    MotionSourceProjectionExact,
    MotionSourceProjectionTrajectory,
)
from ...data.skeleton import MotionSkeleton
from ...data.source import MotionClipSource
from ...identity import canonical_sha256, validate_nonempty, validate_sha256
from ...retarget import motion_contact_probe_offsets
from ...robots.target import (
    MotionFrameTarget,
    validate_collision_probe_geometry,
    write_ground_penetration,
    write_velocity_canonical,
)
from .commands_cfg import (
    MotionAnalyticCoordinatesGenerateCfg,
    MotionExactCoordinatesGenerateCfg,
    MotionGroundPenetrationCriterionCfg,
    MotionTrajectorySolveCfg,
)
from .motion_task_table import (
    _DYNAMICS_QUALITY_START,
    _DYNAMICS_QUALITY_STOP,
    _QUALITY_ACCEPTED,
    _QUALITY_NAMES,
    _TARGET_COORDINATE_QUALITY_NAMES,
    _TARGET_COORDINATE_QUALITY_START,
    _TARGET_COORDINATE_QUALITY_STOP,
    _TRAJECTORY_INSPECTION_CAPTURE_STAGE_NAMES,
    _TRAJECTORY_INSPECTION_QUALITY_PREFIX,
    _TRAJECTORY_INSPECTION_STAGE_NAMES,
    _TRAJECTORY_METRIC_NAMES,
    _TRAJECTORY_METRIC_START,
    _TRAJECTORY_METRIC_STOP,
    MotionTaskTable,
)
from .motion_trajectory import (
    _MotionAcceptedContactEvidence,
    _MotionContactEvidence,
    _MotionTrajectoryViewEvidence,
    _populate_motion_contact_quality,
    _source_clips,
    _trajectory_corpus_quality,
)

if TYPE_CHECKING:
    from ....mdp.commands.state_command.state_command_cfg import StateCommandCfg
    from ...retarget import MotionTrajectoryTargets


@dataclass(frozen=True, slots=True)
class _MotionExactSourceCandidate:
    """One exact-coordinate source corpus before its single decode pass."""

    target: MotionFrameTarget
    projections: tuple[MotionSourceProjectionExact, ...]
    projection_indices: tuple[int, ...]
    source: MotionClipSource
    source_index: MotionClipIndex
    output_index: MotionClipIndex
    source_clip_indices: tuple[int, ...]
    device: str | torch.device
    coordinates: MotionGeneralizedCoordinates


@dataclass(frozen=True, slots=True)
class _MotionAnalyticSourceCandidate:
    """One analytically convertible source corpus before its single decode pass."""

    target: MotionFrameTarget
    projections: tuple[MotionSourceProjectionAnalytic, ...]
    projection_indices: tuple[int, ...]
    source: MotionClipSource
    source_index: MotionClipIndex
    output_index: MotionClipIndex
    source_clip_indices: tuple[int, ...]
    device: str | torch.device
    coordinates: MotionGeneralizedCoordinates


@dataclass(frozen=True, slots=True)
class _MotionTrajectorySourceCandidate:
    """One semantic source corpus before lazy target projection."""

    target: MotionFrameTarget
    projections: tuple[MotionSourceProjectionTrajectory, ...]
    projection_indices: tuple[int, ...]
    source: MotionClipSource
    source_index: MotionClipIndex
    output_index: MotionClipIndex
    source_clip_indices: tuple[int, ...]
    device: str | torch.device
    inspection: bool


_MotionSourceCandidate = _MotionExactSourceCandidate | _MotionAnalyticSourceCandidate | _MotionTrajectorySourceCandidate


@dataclass(frozen=True, slots=True)
class _MotionCoordinateCandidate:
    """Materialized exact or analytic coordinates with target-owned evidence."""

    target: MotionFrameTarget
    clip_index: MotionClipIndex
    coordinates: MotionGeneralizedCoordinates
    target_coordinate_evidence: torch.Tensor
    device: str | torch.device

    @property
    def num_rows(self) -> int:
        """Number of materialized clips entering criteria."""
        return self.target_coordinate_evidence.shape[0]


@dataclass(frozen=True, slots=True)
class _MotionTrajectoryTargetCandidate:
    """Lazy target trajectories awaiting one whole-corpus solve."""

    target: MotionFrameTarget
    clip_index: MotionClipIndex
    pending: Iterator[MotionTrajectoryTargets]
    source_body_counts: tuple[int, ...]
    device: str | torch.device
    inspection: bool


@dataclass(frozen=True, slots=True)
class _MotionTrajectorySolvedCandidate:
    """Solved target trajectories and their complete acceptance evidence."""

    target: MotionFrameTarget
    clip_index: MotionClipIndex
    coordinates: MotionGeneralizedCoordinates
    trajectory_quality: torch.Tensor
    target_coordinate_evidence: torch.Tensor
    constraint_geometry_feasible: torch.Tensor
    inner_solve_converged: torch.Tensor
    nonlinear_refinement_required: torch.Tensor
    nonlinear_phases_converged: torch.Tensor
    acceptance: MotionTrajectorySolveCfg.AcceptanceCfg
    device: str | torch.device
    contact_evidence: _MotionContactEvidence | None
    view_evidence: _MotionTrajectoryViewEvidence | None

    @property
    def num_rows(self) -> int:
        """Number of solved clips entering criteria."""
        return self.target_coordinate_evidence.shape[0]


_MotionFinishedCandidate = _MotionCoordinateCandidate | _MotionTrajectorySolvedCandidate

_EVIDENCE_COORDINATE_FINITE = _TARGET_COORDINATE_QUALITY_NAMES.index("coordinate_finite")
_EVIDENCE_ROOT_QUATERNION = _TARGET_COORDINATE_QUALITY_NAMES.index("root_quaternion_norm_error")
_EVIDENCE_POSITION_VIOLATION = _TARGET_COORDINATE_QUALITY_NAMES.index("joint_position_limit_violation_max_rad")
_EVIDENCE_POSITION_SATISFIED = _TARGET_COORDINATE_QUALITY_NAMES.index("joint_position_limits_satisfied")
_EVIDENCE_VELOCITY_RATIO = _TARGET_COORDINATE_QUALITY_NAMES.index("canonical_joint_velocity_limit_ratio")
_EVIDENCE_VELOCITY_SATISFIED = _TARGET_COORDINATE_QUALITY_NAMES.index("canonical_joint_velocity_limits_satisfied")
_EVIDENCE_FK_FINITE = _TARGET_COORDINATE_QUALITY_NAMES.index("fk_finite")
_EVIDENCE_GROUND_PENETRATION = _TARGET_COORDINATE_QUALITY_NAMES.index("ground_penetration_max_m")


def _certify_target_coordinates(
    target: MotionFrameTarget,
    coordinates: MotionGeneralizedCoordinates,
    clip_index: MotionClipIndex,
) -> torch.Tensor:
    """Measure immutable target-coordinate invariants in corpus-scale GPU work."""
    kinematics = target.kinematics
    tree = target.kinematic_tree
    model = kinematics.model
    validate_sha256("collision_geometry_identity_sha256", target.collision_geometry_identity_sha256)
    validate_collision_probe_geometry(target, device=coordinates.device)
    clip_count = len(clip_index.clips)
    if coordinates.frame_count != clip_index.total_frames or clip_count < 1:
        raise ValueError("Target coordinates must cover the declared complete clips exactly.")
    if any(clip.frame_count < 2 for clip in clip_index.clips):
        raise ValueError("Canonical target velocity requires at least two frames per complete clip.")
    if (
        kinematics.n_root_coords != 7
        or tree.num_coordinates < 1
        or len(tree.coordinate_q_indices) != tree.num_coordinates
        or len(tree.coordinate_qd_indices) != tree.num_coordinates
        or coordinates.device != torch.device(kinematics.device)
    ):
        raise ValueError("Target certification requires one free root and a complete target-tree coordinate map.")

    device = coordinates.device
    frame_count = coordinates.frame_count
    clip_lengths = tuple(clip.frame_count for clip in clip_index.clips)
    clip_offsets = torch.tensor(clip_index.offsets, dtype=torch.int32, device=device)
    clip_offsets_i64 = clip_offsets.to(torch.int64)
    step_seconds = torch.tensor(
        [1.0 / clip.source_fps for clip in clip_index.clips], dtype=torch.float32, device=device
    )
    joint_q_newton = torch.empty((frame_count, model.joint_coord_count), dtype=torch.float32, device=device)
    canonical_velocity = torch.empty((frame_count, model.joint_dof_count), dtype=torch.float32, device=device)
    target.write_joint_position_newton(coordinates, joint_q_newton)
    write_velocity_canonical(target, joint_q_newton, clip_offsets, step_seconds, canonical_velocity)

    coordinate_q_indices = torch.tensor(tree.coordinate_q_indices, dtype=torch.int64, device=device)
    coordinate_qd_indices = torch.tensor(tree.coordinate_qd_indices, dtype=torch.int64, device=device)
    canonical_joint_velocity = canonical_velocity.index_select(1, coordinate_qd_indices)
    position_lower = torch.tensor(tree.coordinate_lower_limits_rad, dtype=torch.float32, device=device)
    position_upper = torch.tensor(tree.coordinate_upper_limits_rad, dtype=torch.float32, device=device)
    velocity_lower = torch.tensor(
        kinematics.topology.joint_velocity_lower, dtype=torch.float32, device=device
    ).index_select(0, coordinate_qd_indices)
    velocity_upper = torch.tensor(
        kinematics.topology.joint_velocity_upper, dtype=torch.float32, device=device
    ).index_select(0, coordinate_qd_indices)
    epsilon = 32.0 * torch.finfo(torch.float32).eps
    position_rounding = epsilon * torch.maximum(
        torch.ones_like(position_lower), torch.maximum(position_lower.abs(), position_upper.abs())
    )
    velocity_rounding = epsilon * torch.maximum(
        torch.ones_like(velocity_lower), torch.maximum(velocity_lower.abs(), velocity_upper.abs())
    )

    frame_evidence = torch.empty(
        (frame_count, len(_TARGET_COORDINATE_QUALITY_NAMES)), dtype=torch.float32, device=device
    )
    values_finite = torch.isfinite(coordinates.joint_q).all(dim=1)
    values_finite.logical_and_(torch.isfinite(joint_q_newton).all(dim=1))
    values_finite.logical_and_(torch.isfinite(canonical_velocity).all(dim=1))
    frame_evidence[:, _EVIDENCE_COORDINATE_FINITE].copy_(~values_finite)
    frame_evidence[:, _EVIDENCE_ROOT_QUATERNION].copy_(
        (torch.linalg.vector_norm(joint_q_newton[:, 3:7], dim=-1) - 1.0).abs()
    )

    joint_position = joint_q_newton.index_select(1, coordinate_q_indices)
    position_violation = torch.maximum(position_lower - joint_position, joint_position - position_upper).clamp_min_(0.0)
    frame_evidence[:, _EVIDENCE_POSITION_VIOLATION].copy_(position_violation.amax(dim=1))
    position_satisfied = (joint_position >= position_lower - position_rounding) & (
        joint_position <= position_upper + position_rounding
    )
    frame_evidence[:, _EVIDENCE_POSITION_SATISFIED].copy_(~position_satisfied.all(dim=1))

    velocity_satisfied = (canonical_joint_velocity >= velocity_lower - velocity_rounding) & (
        canonical_joint_velocity <= velocity_upper + velocity_rounding
    )
    directional_limit = torch.where(canonical_joint_velocity >= 0.0, velocity_upper, -velocity_lower)
    velocity_ratio = canonical_joint_velocity.abs() / directional_limit.clamp_min(torch.finfo(torch.float32).tiny)
    velocity_ratio = torch.where(torch.isinf(directional_limit), torch.zeros_like(velocity_ratio), velocity_ratio)
    frame_evidence[:, _EVIDENCE_VELOCITY_RATIO].copy_(velocity_ratio.amax(dim=1))
    frame_evidence[:, _EVIDENCE_VELOCITY_SATISFIED].copy_(~velocity_satisfied.all(dim=1))
    frame_evidence[:, _EVIDENCE_FK_FINITE].zero_()
    frame_evidence[:, _EVIDENCE_GROUND_PENETRATION].zero_()

    float_bytes = torch.finfo(torch.float32).bits // 8

    def estimate_fk_memory(capacity: int) -> int:
        return float_bytes * capacity * (model.joint_dof_count + 13 * model.body_count)

    memory_plan = plan_trajectory_memory(clip_lengths, device, estimate_fk_memory)
    capacity = memory_plan.workspace_frame_capacity
    joint_qd_zero = torch.zeros((capacity, model.joint_dof_count), dtype=torch.float32, device=device)
    body_q = torch.empty((capacity, model.body_count, 7), dtype=torch.float32, device=device)
    body_qd = torch.empty((capacity, model.body_count, 6), dtype=torch.float32, device=device)
    for segment_start, segment_stop in zip(
        memory_plan.batch_segment_offsets[:-1], memory_plan.batch_segment_offsets[1:], strict=True
    ):
        frame_start = clip_index.offsets[segment_start]
        frame_stop = clip_index.offsets[segment_stop]
        batch_frames = frame_stop - frame_start
        kinematics.eval_fk_batched_torch(
            joint_q_newton[frame_start:frame_stop],
            joint_qd_zero[:batch_frames],
            body_q[:batch_frames],
            body_qd[:batch_frames],
        )
        fk_finite = torch.isfinite(body_q[:batch_frames]).all(dim=(1, 2)) & torch.isfinite(body_qd[:batch_frames]).all(
            dim=(1, 2)
        )
        frame_evidence[frame_start:frame_stop, _EVIDENCE_FK_FINITE].copy_(~fk_finite)
        write_ground_penetration(
            target,
            body_q[:batch_frames],
            frame_evidence[frame_start:frame_stop],
            quality_column=_EVIDENCE_GROUND_PENETRATION,
        )

    evidence = torch.segment_reduce(frame_evidence, "max", offsets=clip_offsets_i64)
    for column in (
        _EVIDENCE_COORDINATE_FINITE,
        _EVIDENCE_POSITION_SATISFIED,
        _EVIDENCE_VELOCITY_SATISFIED,
        _EVIDENCE_FK_FINITE,
    ):
        evidence[:, column].mul_(-1.0).add_(1.0)
    return evidence


def motion_generate_exact_coordinates(
    cfg: MotionExactCoordinatesGenerateCfg, candidate: _MotionExactSourceCandidate, _rng: object
) -> _MotionCoordinateCandidate:
    """Stream exact source coordinates into one compact target-coordinate bank."""
    del cfg
    for index, projection, clip in _source_clips(candidate):
        joint_q, joint_qd = clip.free_root_coordinates(projection.source_skeleton, device=candidate.device)
        coordinates = projection.convert_coordinates(joint_q, joint_qd, clip.source_fps)
        start, end = candidate.output_index.offsets[index : index + 2]
        candidate.coordinates._copy_clip_(start, end, coordinates)
    evidence = _certify_target_coordinates(candidate.target, candidate.coordinates, candidate.output_index)
    return _MotionCoordinateCandidate(
        target=candidate.target,
        clip_index=candidate.output_index,
        coordinates=candidate.coordinates,
        target_coordinate_evidence=evidence,
        device=candidate.device,
    )


def motion_generate_analytic_coordinates(
    cfg: MotionAnalyticCoordinatesGenerateCfg, candidate: _MotionAnalyticSourceCandidate, _rng: object
) -> _MotionCoordinateCandidate:
    """Stream a direct analytic map into its exact-size target-coordinate bank."""
    del cfg
    for index, projection, clip in _source_clips(candidate):
        coordinates = projection.convert_clip(clip)
        start, end = candidate.output_index.offsets[index : index + 2]
        candidate.coordinates._copy_clip_(start, end, coordinates)
    evidence = _certify_target_coordinates(candidate.target, candidate.coordinates, candidate.output_index)
    return _MotionCoordinateCandidate(
        target=candidate.target,
        clip_index=candidate.output_index,
        coordinates=candidate.coordinates,
        target_coordinate_evidence=evidence,
        device=candidate.device,
    )


def motion_criterion_target_coordinates(
    _cfg: object, candidate: _MotionFinishedCandidate, rows: torch.Tensor
) -> torch.Tensor:
    """Accept finite coordinates, a normalized root quaternion, and finite target FK."""
    evidence = candidate.target_coordinate_evidence[rows]
    quaternion_tolerance = 32.0 * torch.finfo(evidence.dtype).eps
    return (
        (evidence[:, _EVIDENCE_COORDINATE_FINITE] > 0.5)
        & torch.isfinite(evidence[:, _EVIDENCE_ROOT_QUATERNION])
        & (evidence[:, _EVIDENCE_ROOT_QUATERNION] <= quaternion_tolerance)
        & (evidence[:, _EVIDENCE_FK_FINITE] > 0.5)
    )


def motion_criterion_target_coordinate_limits(
    _cfg: object, candidate: _MotionFinishedCandidate, rows: torch.Tensor
) -> torch.Tensor:
    """Accept target joint positions and canonical target velocities within declared robot limits."""
    evidence = candidate.target_coordinate_evidence[rows]
    return (evidence[:, _EVIDENCE_POSITION_SATISFIED] > 0.5) & (evidence[:, _EVIDENCE_VELOCITY_SATISFIED] > 0.5)


def motion_criterion_ground_penetration(
    cfg: MotionGroundPenetrationCriterionCfg, candidate: _MotionFinishedCandidate, rows: torch.Tensor
) -> torch.Tensor:
    """Accept target collision geometry no more than the declared depth below world ground."""
    penetration = candidate.target_coordinate_evidence[rows, _EVIDENCE_GROUND_PENETRATION]
    return torch.isfinite(penetration) & (penetration <= cfg.upper_m)


def _motion_task_view(
    clip_index: MotionClipIndex,
    frames: MotionFrames,
    target: MotionFrameTarget,
    quality_values: torch.Tensor,
    view_evidence: _MotionTrajectoryViewEvidence | None,
) -> TaskTableView:
    """Expose exact stored frames through the shared canonical table boundary."""
    joint_names = target.joint_names
    reference = target.kinematics
    frames.validate_values()
    root_pose = torch.cat((frames.field("root_position"), frames.field("root_rotation")), dim=-1).unsqueeze(1)
    root_velocity = torch.cat(
        (frames.field("root_linear_velocity"), frames.field("root_angular_velocity")), dim=-1
    ).unsqueeze(1)
    layout = ResetStateLayout(
        names=("robot",),
        kinds=("articulation",),
        joint_names=(joint_names,),
        joint_offsets=(0, len(joint_names)),
    )
    state_bank = ResetStateBank(
        layout=layout,
        root_pose=root_pose,
        root_velocity=root_velocity,
        joint_position=frames.field("joint_position"),
        joint_velocity=frames.field("joint_velocity"),
    )

    model = reference.model
    if reference.n_root_coords != 7 or model.joint_coord_count != 7 + len(joint_names):
        raise ValueError("Motion reference coordinates must be one free root followed by every live robot joint.")
    device = frames.device
    kinematic_view = TaskTableKinematicView(
        model_builder_state=reference.builder,
        joint_q_default=torch.tensor(reference.default_joint_q, dtype=torch.float32, device=device),
        root_entity_names=("robot",),
        root_state_indices=torch.tensor((0,), dtype=torch.int64, device=device),
        root_q_indices=torch.arange(7, dtype=torch.int64, device=device).reshape(1, 7),
        joint_coordinate_names=tuple(("robot", name) for name in joint_names),
        joint_state_indices=torch.arange(len(joint_names), dtype=torch.int64, device=device),
        joint_q_indices=torch.tensor(target.joint_q_indices, dtype=torch.int64, device=device),
    )
    sequences = TaskTableSequenceIndex(
        offsets=torch.tensor(clip_index.offsets, dtype=torch.int64, device=device),
        frame_dt=torch.tensor(
            [1.0 / clip.source_fps for clip in clip_index.clips],
            dtype=torch.float32,
            device=device,
        ),
    )
    frame_counts = sequences.offsets[1:] - sequences.offsets[:-1]
    accepted_frames = torch.repeat_interleave(quality_values[:, 1] > 0.5, frame_counts)
    points = ()
    lines = ()
    if bool(torch.any(~accepted_frames)):
        points = (
            TaskTablePointEvidence(
                "rejected_frames",
                root_pose[:, :, :3].contiguous(),
                valid=(~accepted_frames).unsqueeze(1),
                color=(1.0, 0.1, 0.1),
                radius=0.035,
            ),
        )
    if view_evidence is not None:
        if (
            view_evidence.target_landmarks.shape != view_evidence.solved_robot_landmarks.shape
            or view_evidence.solved_robot_landmarks.shape[0] != frames.frame_count
        ):
            raise ValueError("Motion table-view evidence must share the final stored frame axis.")
        expected_stage_shape = (
            len(clip_index.clips),
            len(_TRAJECTORY_INSPECTION_CAPTURE_STAGE_NAMES),
            len(_TRAJECTORY_METRIC_NAMES),
        )
        if view_evidence.stage_quality.shape != expected_stage_shape:
            raise ValueError("Motion stage quality must contain every inspection stage, clip, and metric.")
        support_targets = view_evidence.target_support
        points = (
            *points,
            TaskTablePointEvidence(
                "target_landmarks", view_evidence.target_landmarks, color=(0.3, 1.0, 0.4), radius=0.014
            ),
            TaskTablePointEvidence(
                "solved_robot_landmarks", view_evidence.solved_robot_landmarks, color=(0.2, 0.7, 1.0), radius=0.018
            ),
            TaskTablePointEvidence(
                "contact_points",
                view_evidence.contact_points,
                valid=view_evidence.contact_valid,
                color=(1.0, 0.45, 0.1),
                radius=0.024,
            ),
        )
        lines = (
            TaskTableLineEvidence(
                "retarget_residuals",
                torch.stack((view_evidence.target_landmarks, view_evidence.solved_robot_landmarks), dim=2),
                color=(0.6, 0.6, 1.0),
                width=0.003,
            ),
            TaskTableLineEvidence(
                "contact_target_offsets",
                torch.stack((support_targets, view_evidence.contact_points), dim=2),
                valid=view_evidence.contact_valid,
                color=(1.0, 0.75, 0.1),
                width=0.008,
            ),
        )
    quality_names = _QUALITY_NAMES
    if view_evidence is not None:
        stage_names = tuple(
            f"{_TRAJECTORY_INSPECTION_QUALITY_PREFIX}{stage}/{metric}"
            for stage in _TRAJECTORY_INSPECTION_STAGE_NAMES
            for metric in _TRAJECTORY_METRIC_NAMES
        )
        stage_values = torch.cat(
            (
                view_evidence.stage_quality.flatten(1),
                quality_values[:, _TRAJECTORY_METRIC_START:_TRAJECTORY_METRIC_STOP],
            ),
            dim=1,
        )
        quality_names = (*quality_names, *stage_names)
        quality_values = torch.cat((quality_values, stage_values), dim=1)
    quality = TaskTableQuality(names=quality_names, values=quality_values, scope="sequence")
    return TaskTableView(
        sequences=sequences,
        state_bank=state_bank,
        kinematic_view=kinematic_view,
        points=points,
        lines=lines,
        quality=quality,
    )


def _stored_corpus_quality(candidate: _MotionCoordinateCandidate, accepted: torch.Tensor) -> torch.Tensor:
    """Materialize one quality row per exact or analytic source clip."""
    clip_count = len(candidate.clip_index.clips)
    if (
        candidate.target_coordinate_evidence.shape != (clip_count, len(_TARGET_COORDINATE_QUALITY_NAMES))
        or accepted.shape != (clip_count,)
        or accepted.dtype is not torch.bool
    ):
        raise ValueError("Stored-coordinate Motion acceptance must contain one boolean per source clip.")
    quality = torch.zeros(clip_count, len(_QUALITY_NAMES), dtype=torch.float32, device=candidate.coordinates.device)
    quality[:, _QUALITY_ACCEPTED].copy_(accepted)
    quality[:, 2:_TARGET_COORDINATE_QUALITY_START].fill_(torch.nan)
    quality[:, _QUALITY_NAMES.index("contact_applicable")].zero_()
    quality[:, _QUALITY_NAMES.index("contact_stable_frame_channel_count")].zero_()
    quality[:, _TARGET_COORDINATE_QUALITY_START:_TARGET_COORDINATE_QUALITY_STOP].copy_(
        candidate.target_coordinate_evidence
    )
    quality[:, _DYNAMICS_QUALITY_START:_DYNAMICS_QUALITY_STOP].fill_(torch.nan)
    quality[:, -1].fill_(1.0)
    return quality


@dataclass(frozen=True, slots=True)
class _MotionBuiltRoute:
    """One complete coordinate route after its family has completed once."""

    output_source_indices: tuple[int, ...]
    output_skeleton_ids: tuple[int, ...]
    clip_index: MotionClipIndex
    coordinates: MotionGeneralizedCoordinates
    quality: torch.Tensor
    contact_evidence: _MotionContactEvidence | None
    view_evidence: _MotionTrajectoryViewEvidence | None
    family_name: str
    family_identity_sha256: str


@dataclass(frozen=True, slots=True)
class _MotionGroupPlan:
    """One source-skeleton projection awaiting route composition."""

    skeleton_id: int
    source_clip_indices: tuple[int, ...]
    source_index: MotionClipIndex
    output_index: MotionClipIndex
    projection: MotionSourceProjection
    family: object
    family_name: str
    family_identity_sha256: str


@dataclass(frozen=True, slots=True)
class _MotionRoutePlan:
    """Common ownership for one composed family execution."""

    source_clip_indices: tuple[int, ...]
    source_skeleton_ids: tuple[int, ...]
    source_index: MotionClipIndex
    output_index: MotionClipIndex
    projection_indices: tuple[int, ...]
    family: object
    family_name: str
    family_identity_sha256: str


@dataclass(frozen=True, slots=True)
class _MotionExactRoutePlan(_MotionRoutePlan):
    """One exact-coordinate family execution."""

    projections: tuple[MotionSourceProjectionExact, ...]


@dataclass(frozen=True, slots=True)
class _MotionAnalyticRoutePlan(_MotionRoutePlan):
    """One analytic-coordinate family execution."""

    projections: tuple[MotionSourceProjectionAnalytic, ...]


@dataclass(frozen=True, slots=True)
class _MotionTrajectoryRoutePlan(_MotionRoutePlan):
    """One target-trajectory family execution."""

    projections: tuple[MotionSourceProjectionTrajectory, ...]


_MotionTypedRoutePlan = _MotionExactRoutePlan | _MotionAnalyticRoutePlan | _MotionTrajectoryRoutePlan


@dataclass(frozen=True, slots=True)
class _MotionStoredSequence:
    """One retained sequence stored inside its source clip's reserved capacity."""

    source_clip_index: int
    order_in_source: int
    skeleton_id: int
    clip: MotionClipIndex.Clip
    coordinate_start: int
    coordinate_stop: int
    contact_evidence: _MotionContactEvidence | None
    view_evidence: _MotionTrajectoryViewEvidence | None
    quality: torch.Tensor


def _resolve_source_skeletons(
    source: MotionClipSource,
    source_index: MotionClipIndex,
) -> tuple[MotionSkeleton, ...]:
    """Resolve and verify every declared skeleton before decoding any clip."""
    skeletons: list[MotionSkeleton] = []
    for skeleton_id, expected_identity in enumerate(source_index.skeleton_identity_sha256s):
        skeleton = source.skeleton(skeleton_id)
        if not isinstance(skeleton, MotionSkeleton):
            raise TypeError("MotionClipSource.skeleton() must return MotionSkeleton.")
        if skeleton.identity_sha256 != expected_identity:
            raise ValueError(
                f"Motion source skeleton {skeleton_id} identity changed after inspection: "
                f"expected {expected_identity}, got {skeleton.identity_sha256}."
            )
        skeletons.append(skeleton)
    return tuple(skeletons)


def _validate_motion_manifest(expected: MotionClipIndex, actual: MotionClipIndex) -> None:
    """Require exact declared clip identity, order, boundaries, clock, and provenance."""
    problems: list[str] = []
    if actual.source_content_sha256 != expected.source_content_sha256:
        problems.append("source_content_sha256")
    if actual.skeleton_identity_sha256s != expected.skeleton_identity_sha256s:
        problems.append("skeleton_identity_sha256s")
    if actual.offsets != expected.offsets:
        problems.append(f"offsets expected={expected.offsets} actual={actual.offsets}")
    if len(actual.clips) != len(expected.clips):
        problems.append(f"clip_count expected={len(expected.clips)} actual={len(actual.clips)}")
    fields = (
        "clip_id",
        "frame_count",
        "source_fps",
        "content_sha256",
        "skeleton_id",
        "source_clip_id",
        "source_frame_start",
    )
    for index, (expected_clip, actual_clip) in enumerate(zip(expected.clips, actual.clips, strict=False)):
        for field in fields:
            expected_value = getattr(expected_clip, field)
            actual_value = getattr(actual_clip, field)
            if actual_value != expected_value:
                problems.append(f"clip[{index}].{field} expected={expected_value!r} actual={actual_value!r}")
    if not problems:
        return
    details = "; ".join(problems[:8])
    remaining = len(problems) - min(len(problems), 8)
    suffix = "" if remaining == 0 else f"; and {remaining} more"
    raise ValueError("Motion output changed its explicitly declared clip manifest: " + details + suffix)


def _plan_motion_group(
    table_cfg: object,
    source: MotionClipSource,
    source_index: MotionClipIndex,
    skeleton_id: int,
    source_skeleton: MotionSkeleton,
    target: MotionFrameTarget,
    contact_channel_probe_offsets: torch.Tensor,
) -> _MotionGroupPlan:
    """Resolve one source projection and family without decoding source clips."""
    expected_skeleton_identity = source_index.skeleton_identity_sha256s[skeleton_id]
    source_clip_indices = source_index.for_skeleton(skeleton_id)
    group_index = MotionClipIndex(
        source_content_sha256=source_index.source_content_sha256,
        skeleton_identity_sha256s=(expected_skeleton_identity,),
        clips=tuple(replace(source_index.clips[index], skeleton_id=0) for index in source_clip_indices),
    )
    projection = table_cfg.target_kinematics.source_projection_factory(
        source_skeleton,
        target,
        source,
        table_cfg.contact_channels,
        contact_channel_probe_offsets,
    )
    if not isinstance(
        projection,
        (MotionSourceProjectionExact, MotionSourceProjectionAnalytic, MotionSourceProjectionTrajectory),
    ):
        raise TypeError("source_projection_factory must return a concrete MotionSourceProjection route.")
    if projection.source_skeleton.identity_sha256 != expected_skeleton_identity:
        raise ValueError("Motion source projection changed the selected source-skeleton identity.")
    validate_nonempty("construction_version", projection.version)
    validate_sha256("construction_identity_sha256", projection.construction_identity_sha256)
    if isinstance(projection, MotionSourceProjectionTrajectory):
        validate_sha256("evidence_layout_identity_sha256", projection.target_projection.evidence_layout_identity_sha256)
    matches = tuple(
        family
        for family in table_cfg.families
        if family.generate and isinstance(projection, family.generate[0].source_projection_type)
    )
    if len(matches) != 1:
        raise ValueError(
            f"Motion source projection matched {len(matches)} declared task families; expected exactly one."
        )
    family = matches[0]
    family_name = family.name
    if isinstance(projection, MotionSourceProjectionExact):
        output_index = group_index
    elif isinstance(projection, MotionSourceProjectionAnalytic):
        output_index = projection.output_clip_index(group_index)
    else:
        output_index = group_index
    if len(output_index.clips) != len(group_index.clips):
        raise ValueError("A Motion route must declare one output clip for every input clip.")
    if any(clip.frame_count < target.materialization_minimum_frames for clip in output_index.clips):
        raise ValueError(
            f"Motion target materialization requires at least {target.materialization_minimum_frames} frames per clip."
        )
    return _MotionGroupPlan(
        skeleton_id=skeleton_id,
        source_clip_indices=source_clip_indices,
        source_index=group_index,
        output_index=output_index,
        projection=projection,
        family=family,
        family_name=family_name,
        family_identity_sha256=canonical_sha256(family.to_dict()),
    )


def _compose_motion_routes(plans: tuple[_MotionGroupPlan, ...]) -> tuple[_MotionTypedRoutePlan, ...]:
    """Compose source projections sharing one family and trajectory evidence layout."""
    groups: dict[tuple[str, str, str], list[_MotionGroupPlan]] = {}
    for plan in plans:
        evidence_layout = (
            plan.projection.target_projection.evidence_layout_identity_sha256
            if isinstance(plan.projection, MotionSourceProjectionTrajectory)
            else ""
        )
        groups.setdefault((plan.family_name, plan.family_identity_sha256, evidence_layout), []).append(plan)

    routes: list[_MotionTypedRoutePlan] = []
    for members in groups.values():
        first = members[0]
        projections = tuple(member.projection for member in members)
        projection_indices = tuple(
            projection_index for projection_index, member in enumerate(members) for _ in member.source_index.clips
        )
        source_clip_indices = tuple(index for member in members for index in member.source_clip_indices)
        source_skeleton_ids = tuple(member.skeleton_id for member in members for _ in member.source_index.clips)
        skeleton_identities = tuple(projection.source_skeleton.identity_sha256 for projection in projections)
        source_index = MotionClipIndex(
            source_content_sha256=first.source_index.source_content_sha256,
            skeleton_identity_sha256s=skeleton_identities,
            clips=tuple(
                replace(clip, skeleton_id=projection_index)
                for projection_index, member in enumerate(members)
                for clip in member.source_index.clips
            ),
        )
        output_index = MotionClipIndex(
            source_content_sha256=first.output_index.source_content_sha256,
            skeleton_identity_sha256s=skeleton_identities,
            clips=tuple(
                replace(clip, skeleton_id=projection_index)
                for projection_index, member in enumerate(members)
                for clip in member.output_index.clips
            ),
        )
        if isinstance(first.projection, MotionSourceProjectionExact):
            if not all(isinstance(projection, MotionSourceProjectionExact) for projection in projections):
                raise ValueError("One Motion family cannot mix projection routes.")
            routes.append(
                _MotionExactRoutePlan(
                    source_clip_indices=source_clip_indices,
                    source_skeleton_ids=source_skeleton_ids,
                    source_index=source_index,
                    output_index=output_index,
                    projection_indices=projection_indices,
                    family=first.family,
                    family_name=first.family_name,
                    family_identity_sha256=first.family_identity_sha256,
                    projections=cast(tuple[MotionSourceProjectionExact, ...], projections),
                )
            )
        elif isinstance(first.projection, MotionSourceProjectionAnalytic):
            if not all(isinstance(projection, MotionSourceProjectionAnalytic) for projection in projections):
                raise ValueError("One Motion family cannot mix projection routes.")
            routes.append(
                _MotionAnalyticRoutePlan(
                    source_clip_indices=source_clip_indices,
                    source_skeleton_ids=source_skeleton_ids,
                    source_index=source_index,
                    output_index=output_index,
                    projection_indices=projection_indices,
                    family=first.family,
                    family_name=first.family_name,
                    family_identity_sha256=first.family_identity_sha256,
                    projections=cast(tuple[MotionSourceProjectionAnalytic, ...], projections),
                )
            )
        else:
            if not all(isinstance(projection, MotionSourceProjectionTrajectory) for projection in projections):
                raise ValueError("One Motion family cannot mix projection routes.")
            routes.append(
                _MotionTrajectoryRoutePlan(
                    source_clip_indices=source_clip_indices,
                    source_skeleton_ids=source_skeleton_ids,
                    source_index=source_index,
                    output_index=output_index,
                    projection_indices=projection_indices,
                    family=first.family,
                    family_name=first.family_name,
                    family_identity_sha256=first.family_identity_sha256,
                    projections=cast(tuple[MotionSourceProjectionTrajectory, ...], projections),
                )
            )
    return tuple(routes)


def _build_motion_route(
    plan: _MotionTypedRoutePlan,
    target: MotionFrameTarget,
    source: MotionClipSource,
    rng: object,
    device: str,
    coordinate_workspace: MotionGeneralizedCoordinates | None = None,
    *,
    inspection: bool = False,
) -> _MotionBuiltRoute:
    """Execute one coordinate family and optionally retain every inspected candidate."""
    route_contact_evidence = None
    route_view_evidence = None
    if isinstance(plan, _MotionExactRoutePlan):
        if coordinate_workspace is not None and coordinate_workspace.frame_count != plan.output_index.total_frames:
            raise ValueError("Motion route workspace capacity differs from its declared output clock.")
        coordinates = (
            target.allocate_coordinates(plan.output_index.total_frames, device=device)
            if coordinate_workspace is None
            else coordinate_workspace
        )
        initial: _MotionSourceCandidate = _MotionExactSourceCandidate(
            target=target,
            projections=plan.projections,
            projection_indices=plan.projection_indices,
            source=source,
            source_index=plan.source_index,
            output_index=plan.output_index,
            source_clip_indices=plan.source_clip_indices,
            device=device,
            coordinates=coordinates,
        )
    elif isinstance(plan, _MotionAnalyticRoutePlan):
        if coordinate_workspace is not None and coordinate_workspace.frame_count != plan.output_index.total_frames:
            raise ValueError("Motion route workspace capacity differs from its declared output clock.")
        coordinates = (
            target.allocate_coordinates(plan.output_index.total_frames, device=device)
            if coordinate_workspace is None
            else coordinate_workspace
        )
        initial = _MotionAnalyticSourceCandidate(
            target=target,
            projections=plan.projections,
            projection_indices=plan.projection_indices,
            source=source,
            source_index=plan.source_index,
            output_index=plan.output_index,
            source_clip_indices=plan.source_clip_indices,
            device=device,
            coordinates=coordinates,
        )
    else:
        initial = _MotionTrajectorySourceCandidate(
            target=target,
            projections=plan.projections,
            projection_indices=plan.projection_indices,
            source=source,
            source_index=plan.source_index,
            output_index=plan.output_index,
            source_clip_indices=plan.source_clip_indices,
            device=device,
            inspection=inspection,
        )
    execution = execute_task_family(plan.family, initial, None, rng)
    built = execution.candidates
    if execution.accepted_mask is None:
        raise TypeError("Motion family execution must return one accepted mask on its candidate axis.")
    if execution.selected_indices is not None:
        raise TypeError("Motion task families must not configure a selection stage.")
    if not inspection:
        rejected_indices = tuple(
            int(index)
            for index in torch.nonzero(~execution.accepted_mask, as_tuple=False).flatten().detach().cpu().tolist()
        )
        if rejected_indices:
            rejected_source_indices = tuple(plan.source_clip_indices[index] for index in rejected_indices)
            rejected_clip_ids = tuple(plan.output_index.clip_ids[index] for index in rejected_indices)
            raise ValueError(
                f"Motion production route {plan.family_name!r} rejected clips before publication: "
                f"route_indices={rejected_indices}, source_indices={rejected_source_indices}, "
                f"clip_ids={rejected_clip_ids}."
            )
    if isinstance(plan, (_MotionExactRoutePlan, _MotionAnalyticRoutePlan)):
        if not isinstance(built, _MotionCoordinateCandidate):
            raise TypeError("Stored-coordinate Motion family must return a coordinate candidate.")
        quality = _stored_corpus_quality(built, execution.accepted_mask)
    else:
        if not isinstance(built, _MotionTrajectorySolvedCandidate):
            raise TypeError("Trajectory Motion family must return a solved trajectory candidate.")
        quality = _trajectory_corpus_quality(built, execution.accepted_mask)
        if inspection:
            if built.contact_evidence is None:
                raise ValueError("Trajectory inspection did not retain source-contact evidence.")
            route_contact_evidence = built.contact_evidence
            route_view_evidence = built.view_evidence

    return _MotionBuiltRoute(
        output_source_indices=plan.source_clip_indices,
        output_skeleton_ids=plan.source_skeleton_ids,
        clip_index=built.clip_index,
        coordinates=built.coordinates,
        quality=quality,
        contact_evidence=route_contact_evidence,
        view_evidence=route_view_evidence,
        family_name=plan.family_name,
        family_identity_sha256=plan.family_identity_sha256,
    )


def _store_motion_route(
    source_index: MotionClipIndex,
    capacity_offsets: tuple[int, ...],
    used_frames: list[int],
    sequence_counts: list[int],
    coordinate_bank: MotionGeneralizedCoordinates,
    records: list[_MotionStoredSequence],
    route: _MotionBuiltRoute,
) -> None:
    """Copy one completed group into source-ordered reserved capacity."""
    if len(route.output_source_indices) != len(route.clip_index.clips) or len(route.output_skeleton_ids) != len(
        route.clip_index.clips
    ):
        raise ValueError("Motion route output origins must contain one source clip and skeleton per sequence.")
    if route.coordinates.joint_q.shape[1:] != coordinate_bank.joint_q.shape[1:] or (
        route.coordinates.joint_qd is None
    ) != (coordinate_bank.joint_qd is None):
        raise ValueError("Motion group coordinates differ from the corpus coordinate schema.")
    if records and route.quality.shape[1:] != records[0].quality.shape:
        raise ValueError("Motion groups produced incompatible quality schemas.")

    for output_index in range(len(route.output_source_indices)):
        source_clip_index = route.output_source_indices[output_index]
        skeleton_id = route.output_skeleton_ids[output_index]
        if source_index.clips[source_clip_index].skeleton_id != skeleton_id:
            raise ValueError("Motion route emitted a sequence for a different source skeleton.")
        clip = route.clip_index.clips[output_index]
        source_start, source_stop = route.clip_index.offsets[output_index : output_index + 2]
        destination_start = capacity_offsets[source_clip_index] + used_frames[source_clip_index]
        destination_stop = destination_start + clip.frame_count
        if destination_stop > capacity_offsets[source_clip_index + 1] or source_stop - source_start != clip.frame_count:
            raise ValueError("Motion route exceeded its declared per-source-clip coordinate capacity.")

        source_q = route.coordinates.joint_q[source_start:source_stop]
        destination_q = coordinate_bank.joint_q[destination_start:destination_stop]
        if source_q.data_ptr() != destination_q.data_ptr():
            destination_q.copy_(source_q)
        if coordinate_bank.joint_qd is not None and route.coordinates.joint_qd is not None:
            source_qd = route.coordinates.joint_qd[source_start:source_stop]
            destination_qd = coordinate_bank.joint_qd[destination_start:destination_stop]
            if source_qd.data_ptr() != destination_qd.data_ptr():
                destination_qd.copy_(source_qd)
        records.append(
            _MotionStoredSequence(
                source_clip_index=source_clip_index,
                order_in_source=sequence_counts[source_clip_index],
                skeleton_id=skeleton_id,
                clip=clip,
                coordinate_start=destination_start,
                coordinate_stop=destination_stop,
                quality=route.quality[output_index],
                contact_evidence=None
                if route.contact_evidence is None
                else _MotionContactEvidence(
                    route.contact_evidence.source_stable[source_start:source_stop],
                    route.contact_evidence.support_body_indices,
                    route.contact_evidence.support_point_body_m,
                    route.contact_evidence.support_channel_slots,
                    route.contact_evidence.policy,
                ),
                view_evidence=None
                if route.view_evidence is None
                else _MotionTrajectoryViewEvidence(
                    route.view_evidence.target_landmarks[source_start:source_stop],
                    route.view_evidence.solved_robot_landmarks[source_start:source_stop],
                    route.view_evidence.target_support[source_start:source_stop],
                    route.view_evidence.contact_points[source_start:source_stop],
                    route.view_evidence.contact_valid[source_start:source_stop],
                    route.view_evidence.stage_quality[output_index : output_index + 1],
                ),
            )
        )
        used_frames[source_clip_index] += clip.frame_count
        sequence_counts[source_clip_index] += 1


def _finish_motion_groups(
    source_index: MotionClipIndex,
    coordinate_bank: MotionGeneralizedCoordinates,
    coordinate_scratch: MotionGeneralizedCoordinates,
    records: list[_MotionStoredSequence],
) -> tuple[
    MotionClipIndex,
    MotionGeneralizedCoordinates,
    torch.Tensor,
    _MotionAcceptedContactEvidence | None,
    _MotionTrajectoryViewEvidence | None,
]:
    """Compact every route output into deterministic source-manifest order."""
    if not records:
        raise ValueError("Motion source produced no candidate sequences.")
    records.sort(key=lambda record: (record.source_clip_index, record.order_in_source))
    quality = torch.empty(
        (len(records), *records[0].quality.shape),
        dtype=records[0].quality.dtype,
        device=records[0].quality.device,
    )
    clips = []
    frame_cursor = 0
    for sequence_index, record in enumerate(records):
        clip = replace(record.clip, skeleton_id=record.skeleton_id)
        clips.append(clip)
        frame_count = record.coordinate_stop - record.coordinate_start
        if frame_count != clip.frame_count or frame_count > coordinate_scratch.frame_count:
            raise ValueError("Motion sequence differs from its compacting scratch capacity.")
        if record.coordinate_start < frame_cursor:
            raise ValueError("Motion reserved intervals are not monotonic in source order.")
        if record.coordinate_start != frame_cursor:
            coordinate_scratch.joint_q[:frame_count].copy_(
                coordinate_bank.joint_q[record.coordinate_start : record.coordinate_stop]
            )
            coordinate_bank.joint_q[frame_cursor : frame_cursor + frame_count].copy_(
                coordinate_scratch.joint_q[:frame_count]
            )
            if coordinate_bank.joint_qd is not None and coordinate_scratch.joint_qd is not None:
                coordinate_scratch.joint_qd[:frame_count].copy_(
                    coordinate_bank.joint_qd[record.coordinate_start : record.coordinate_stop]
                )
                coordinate_bank.joint_qd[frame_cursor : frame_cursor + frame_count].copy_(
                    coordinate_scratch.joint_qd[:frame_count]
                )
        quality[sequence_index].copy_(record.quality)
        frame_cursor += frame_count

    clip_index = MotionClipIndex(
        source_content_sha256=source_index.source_content_sha256,
        skeleton_identity_sha256s=source_index.skeleton_identity_sha256s,
        clips=tuple(clips),
    )
    coordinates = MotionGeneralizedCoordinates(
        coordinate_bank.joint_q[:frame_cursor],
        None if coordinate_bank.joint_qd is None else coordinate_bank.joint_qd[:frame_cursor],
    )
    evidence_rows = tuple(record.view_evidence for record in records)
    retained_evidence = tuple(item for item in evidence_rows if item is not None)
    if retained_evidence and len(retained_evidence) != len(evidence_rows):
        raise ValueError("Motion table-view evidence must be enabled consistently across all source groups.")
    view_evidence = None
    if retained_evidence:
        view_evidence = _MotionTrajectoryViewEvidence(
            torch.cat(tuple(item.target_landmarks for item in retained_evidence)),
            torch.cat(tuple(item.solved_robot_landmarks for item in retained_evidence)),
            torch.cat(tuple(item.target_support for item in retained_evidence)),
            torch.cat(tuple(item.contact_points for item in retained_evidence)),
            torch.cat(tuple(item.contact_valid for item in retained_evidence)),
            torch.cat(tuple(item.stage_quality for item in retained_evidence)),
        )
    contact_rows: list[tuple[int, _MotionContactEvidence]] = []
    for index, record in enumerate(records):
        if record.contact_evidence is not None:
            contact_rows.append((index, record.contact_evidence))
    contact_evidence = None
    if contact_rows:
        first = contact_rows[0][1]
        contact_channel_count = first.source_stable.shape[1]
        for _, item in contact_rows[1:]:
            if (
                item.support_body_indices != first.support_body_indices
                or item.policy != first.policy
                or item.source_stable.shape[1] != contact_channel_count
                or not torch.equal(item.support_point_body_m, first.support_point_body_m)
                or not torch.equal(item.support_channel_slots, first.support_channel_slots)
            ):
                raise ValueError("Trajectory groups require one shared support and dynamics schema.")
        contact_evidence = _MotionAcceptedContactEvidence(
            sequence_indices=tuple(index for index, _ in contact_rows),
            source_stable=torch.cat(tuple(item.source_stable for _, item in contact_rows)),
            support_body_indices=first.support_body_indices,
            support_point_body_m=first.support_point_body_m,
            support_channel_slots=first.support_channel_slots,
            policy=first.policy,
        )
    records.clear()
    return clip_index, coordinates, quality, contact_evidence, view_evidence


def _group_identity(
    target: MotionFrameTarget,
    source_index: MotionClipIndex,
    plans: tuple[_MotionGroupPlan, ...],
) -> tuple[str, str, str]:
    """Return aggregate target/projection version, construction identity, and family identity."""
    validate_nonempty("construction_version", target.version)
    validate_sha256("construction_identity_sha256", target.construction_identity_sha256)
    validate_sha256("collision_geometry_identity_sha256", target.collision_geometry_identity_sha256)
    projection_versions = tuple(dict.fromkeys(plan.projection.version for plan in plans))
    construction_version = (
        projection_versions[0] if len(projection_versions) == 1 else f"{target.version}+mixed_sources_v1"
    )
    construction_identity = canonical_sha256(
        {
            "target_sha256": target.construction_identity_sha256,
            "collision_geometry_sha256": target.collision_geometry_identity_sha256,
            "source_projections": [
                {
                    "source_skeleton_sha256": source_index.skeleton_identity_sha256s[plan.skeleton_id],
                    "projection_sha256": plan.projection.construction_identity_sha256,
                }
                for plan in plans
            ],
        }
    )
    family_identity = canonical_sha256(
        {
            "source_skeleton_families": [
                {
                    "source_skeleton_sha256": source_index.skeleton_identity_sha256s[plan.skeleton_id],
                    "family_name": plan.family_name,
                    "family_sha256": plan.family_identity_sha256,
                }
                for plan in plans
            ]
        }
    )
    return construction_version, construction_identity, family_identity


def _build_motion_task_table(
    command_cfg: StateCommandCfg,
    scene_cfg: object,
    device: str,
    *,
    inspection: bool,
    oracle: bool,
    sequence_limit: int | None,
) -> MotionTaskTable | TaskTableView:
    """Build one motion table under an explicit runtime or inspection policy."""
    if inspection:
        if oracle:
            raise ValueError("Motion table construction cannot be both inspection and oracle comparison.")
        if type(sequence_limit) is not int or sequence_limit < 1:
            raise ValueError("Motion inspection sequence_limit must be a positive integer.")
    elif sequence_limit is not None:
        raise ValueError("Runtime Motion table construction does not accept sequence_limit.")
    table_cfg = command_cfg.task_table
    source_cfg = table_cfg.source
    if oracle and source_cfg.purpose != "oracle":
        raise ValueError("Motion oracle comparison requires a source declared with oracle purpose.")
    if source_cfg.purpose == "oracle" and not (inspection or oracle):
        raise ValueError(
            f"Motion source {source_cfg.identifier!r} is oracle-only; "
            "use build_motion_task_table_inspection() to inspect released retargeting evidence."
        )
    split = source_cfg.train if table_cfg.motion_split == "train" else source_cfg.evaluation
    source = source_cfg.open_split(table_cfg.source_artifact_root, split)
    try:
        source_index = source.inspect()
        if (
            source_index.source_content_sha256 != split.source_content_sha256
            or len(source_index.clips) != split.clip_count
            or source_index.total_frames != split.frame_count
        ):
            raise ValueError(
                "Motion source identity/counts differ from the selected split: "
                f"hash={source_index.source_content_sha256}, clips={len(source_index.clips)}, "
                f"frames={source_index.total_frames}."
            )
        source_skeletons = _resolve_source_skeletons(source, source_index)
        if inspection and sequence_limit < len(source_index.clips):
            clips = source_index.clips[:sequence_limit]
            skeleton_count = 1 + max(clip.skeleton_id for clip in clips)
            source_index = MotionClipIndex(
                source_content_sha256=source_index.source_content_sha256,
                skeleton_identity_sha256s=source_index.skeleton_identity_sha256s[:skeleton_count],
                clips=clips,
            )
            source_skeletons = source_skeletons[:skeleton_count]
        target_kinematics = table_cfg.target_kinematics
        channel_names = tuple(channel.name for channel in table_cfg.contact_channels)
        patch_channel_names = tuple(patch.channel for patch in target_kinematics.contact_patches)
        if channel_names != patch_channel_names:
            raise ValueError("Source contact channels and target contact patches must match in declaration order.")
        articulation_cfg = getattr(scene_cfg, target_kinematics.asset_cfg.name)
        reference = NewtonKinematics.from_articulation(target_kinematics.kinematics, articulation_cfg, device)
        contact_channel_probe_offsets = motion_contact_probe_offsets(table_cfg.contact_channels, device)
        target = target_kinematics.target_factory(
            reference,
            target_kinematics.contact_patches,
            calibration_artifact_root=table_cfg.target_artifact_root,
            calibration=target_kinematics.calibration,
        )
        if not isinstance(target, MotionFrameTarget):
            raise TypeError("target_factory must return MotionFrameTarget.")
        plans = tuple(
            _plan_motion_group(
                table_cfg,
                source,
                source_index,
                skeleton_id,
                source_skeletons[skeleton_id],
                target,
                contact_channel_probe_offsets,
            )
            for skeleton_id in source_index.skeleton_ids
        )
        construction_version, frame_construction_identity, family_identity = _group_identity(
            target, source_index, plans
        )
        routes = _compose_motion_routes(plans)

        declared_clips: list[MotionClipIndex.Clip | None] = [None] * len(source_index.clips)
        for route in routes:
            route_index = route.output_index
            if route_index.source_content_sha256 != source_index.source_content_sha256:
                raise ValueError("Motion route changed the declared source-content identity.")
            for local_index, source_clip_index in enumerate(route.source_clip_indices):
                if declared_clips[source_clip_index] is not None:
                    raise ValueError("Motion planning assigned one source clip more than once.")
                declared_clips[source_clip_index] = replace(
                    route_index.clips[local_index],
                    skeleton_id=source_index.clips[source_clip_index].skeleton_id,
                )
        if any(clip is None for clip in declared_clips):
            raise ValueError("Every Motion source clip must declare exactly one output clip.")
        declared_output_index = MotionClipIndex(
            source_content_sha256=source_index.source_content_sha256,
            skeleton_identity_sha256s=source_index.skeleton_identity_sha256s,
            clips=tuple(clip for clip in declared_clips if clip is not None),
        )
        clip_capacities = [clip.frame_count for clip in declared_output_index.clips]
        capacity_offsets = [0]
        for capacity in clip_capacities:
            capacity_offsets.append(capacity_offsets[-1] + capacity)
        capacity_offsets = tuple(capacity_offsets)

        coordinate_bank = target.allocate_coordinates(capacity_offsets[-1], device=device)
        coordinate_scratch = target.allocate_coordinates(max(clip_capacities), device=device)
        used_frames = [0] * len(source_index.clips)
        sequence_counts = [0] * len(source_index.clips)
        records: list[_MotionStoredSequence] = []
        rng = make_task_table_rng(table_cfg.seed, device)
        with torch.jit.optimized_execution(False):
            for route in routes:
                coordinate_workspace = None
                if isinstance(route, (_MotionExactRoutePlan, _MotionAnalyticRoutePlan)):
                    declared_index = route.output_index
                    first_source = route.source_clip_indices[0]
                    last_source = route.source_clip_indices[-1]
                    if route.source_clip_indices == tuple(range(first_source, last_source + 1)):
                        workspace_start = capacity_offsets[first_source]
                        workspace_stop = capacity_offsets[last_source + 1]
                        if workspace_stop - workspace_start == declared_index.total_frames:
                            coordinate_workspace = MotionGeneralizedCoordinates(
                                coordinate_bank.joint_q[workspace_start:workspace_stop],
                                None
                                if coordinate_bank.joint_qd is None
                                else coordinate_bank.joint_qd[workspace_start:workspace_stop],
                            )
                route_result = _build_motion_route(
                    route, target, source, rng, device, coordinate_workspace, inspection=inspection
                )
                _store_motion_route(
                    source_index,
                    capacity_offsets,
                    used_frames,
                    sequence_counts,
                    coordinate_bank,
                    records,
                    route_result,
                )
                del route_result

            clip_index, coordinates, quality, contact_evidence, view_evidence = _finish_motion_groups(
                source_index, coordinate_bank, coordinate_scratch, records
            )
            _validate_motion_manifest(declared_output_index, clip_index)
            del coordinate_scratch
            coordinates.validate_values()
            if inspection:
                _populate_motion_contact_quality(target, clip_index, coordinates, quality, contact_evidence)
            frames = target.materialize_coordinates(coordinates, clip_index)
        del coordinates, coordinate_bank
        family_names = tuple(dict.fromkeys(route.family_name for route in routes))
        family_name = family_names[0] if len(family_names) == 1 else "+".join(family_names)
        view = _motion_task_view(clip_index, frames, target, quality, view_evidence)
        if inspection:
            return view
        return MotionTaskTable(
            clip_index,
            frames,
            target.joint_names,
            target.reference_frame_names,
            construction_version,
            frame_construction_identity,
            table_cfg.task_row_mode,
            source_cfg.decoder_version,
            family_name,
            family_identity,
            view,
        )
    finally:
        source.close()


def build_motion_task_table(command_cfg: StateCommandCfg, scene_cfg: object, device: str) -> MotionTaskTable:
    """Build the exact production motion task table."""
    return cast(
        MotionTaskTable,
        _build_motion_task_table(command_cfg, scene_cfg, device, inspection=False, oracle=False, sequence_limit=None),
    )


def build_motion_task_table_oracle(command_cfg: StateCommandCfg, scene_cfg: object, device: str) -> MotionTaskTable:
    """Build a comparison-only table from a source explicitly declared as oracle evidence."""
    return cast(
        MotionTaskTable,
        _build_motion_task_table(command_cfg, scene_cfg, device, inspection=False, oracle=True, sequence_limit=None),
    )


def build_motion_task_table_inspection(
    command_cfg: StateCommandCfg,
    scene_cfg: object,
    device: str,
    *,
    sequence_limit: int,
) -> TaskTableView:
    """Build the simulator-free view with accepted and rejected candidate evidence."""
    return cast(
        TaskTableView,
        _build_motion_task_table(
            command_cfg, scene_cfg, device, inspection=True, oracle=False, sequence_limit=sequence_limit
        ),
    )
