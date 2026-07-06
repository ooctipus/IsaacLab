# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion trajectory tensors and task descriptors owned by one command table."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import newton.ik as ik
import torch
import warp as wp

from isaaclab.utils.math import quat_slerp
from isaaclab.utils.string import string_to_callable

from ....kinematics import IKExecutionStatistics, execute_ik_batches, kinematic_seed_target_rotations
from ....mdp.commands.state_command import (
    ResetStateBank,
    ResetStateLayout,
    TaskTableKinematicView,
    TaskTableQuality,
    TaskTableSequenceIndex,
    TaskTableView,
    execute_task_family,
    make_task_table_rng,
)
from ...data.clip_index import MotionClipIndex
from ...data.frames import Interpolation, MotionFrameBuilder, MotionFrames
from ...data.source import MotionClipSource
from ...identity import canonical_sha256, validate_nonempty, validate_sha256
from ...retarget import (
    MotionSemanticTargets,
    semantic_correct_support,
    semantic_measure_branch_jump,
    semantic_measure_quality,
    semantic_project_coordinates,
)
from .commands_cfg import (
    MotionClipSelectionCfg,
    MotionExactCoordinatesGenerateCfg,
    MotionFrameFiniteCriterionCfg,
    MotionLandmarkPositionObjectiveCfg,
    MotionLandmarkRotationObjectiveCfg,
    MotionObjectiveMeasureCriterionCfg,
    MotionSemanticSegmentSelectionCfg,
    MotionSemanticSolveCfg,
    MotionSemanticTargetsGenerateCfg,
)

if TYPE_CHECKING:
    from ....mdp.commands.state_command.state_command_cfg import StateCommandCfg


_QUALITY_NAMES = (
    "semantic_route",
    "accepted",
    "landmark_position_max_m",
    "landmark_orientation_max_rad",
    "branch_jump_max_rad",
    "base_priority",
)


@dataclass(frozen=True, slots=True)
class _MotionCorpusCandidate:
    """One ordered source corpus moving through exactly one family execution."""

    builder: MotionFrameBuilder
    source: MotionClipSource | None
    clip_index: MotionClipIndex
    device: str | torch.device
    frames: MotionFrames | None
    pending: Iterator[MotionSemanticTargets] | None = None
    semantic_joint_q: torch.Tensor | None = None
    semantic_quality: torch.Tensor | None = None
    frame_finite: torch.Tensor | None = None
    solve_statistics: IKExecutionStatistics | None = None

    @property
    def num_rows(self) -> int:
        """Number of materialized clip or frame rows entering criteria."""
        if self.frame_finite is None:
            raise RuntimeError("Motion criteria require materialized finiteness evidence.")
        return self.frame_finite.shape[0]


@dataclass(slots=True)
class _MotionSemanticWorkspace:
    """One corpus-wide capacity workspace reused by every semantic solve interval."""

    solver: ik.IKSolver
    targets: MotionSemanticTargets
    joint_q: torch.Tensor
    joint_qd: wp.array
    body_q: wp.array
    body_qd: wp.array
    coordinate_indices: wp.array
    coordinate_lower: wp.array
    coordinate_upper: wp.array
    body_indices: wp.array
    support_body_indices: wp.array
    quality: torch.Tensor
    frame_finite: torch.Tensor
    support_error_m: torch.Tensor
    capacity: int


@dataclass(slots=True)
class _MotionSemanticTargetStream:
    """Monotonic clip stream packed into consecutive corpus workspace intervals."""

    iterator: Iterator[MotionSemanticTargets]
    clip_index: MotionClipIndex
    prototype: MotionSemanticTargets
    current: MotionSemanticTargets | None
    clip: int = 0
    local_frame: int = 0
    global_frame: int = 0

    def _validate(self, targets: MotionSemanticTargets) -> None:
        if self.clip >= len(self.clip_index.clips):
            raise ValueError("Motion semantic target stream exceeds the declared clip index.")
        expected_frames = self.clip_index.clips[self.clip].frame_count
        landmark_count = len(self.prototype.body_indices)
        if targets.position_m.shape != (landmark_count, expected_frames, 3) or targets.rotation_xyzw.shape != (
            landmark_count,
            expected_frames,
            4,
        ):
            raise ValueError("Motion semantic target shapes differ from the declared source clip.")
        if targets.body_indices != self.prototype.body_indices or targets.parent_rows != self.prototype.parent_rows:
            raise ValueError("Motion semantic landmark identity changed between source clips.")
        for name in (
            "body_index_tensor",
            "segment_lengths_m",
            "coordinate_indices",
            "coordinate_lower_limits_rad",
            "coordinate_upper_limits_rad",
            "support_body_indices",
        ):
            if getattr(targets, name).data_ptr() != getattr(self.prototype, name).data_ptr():
                raise ValueError(f"Motion semantic static tensor {name!r} changed between source clips.")

    def _next(self) -> MotionSemanticTargets:
        try:
            targets = next(self.iterator)
        except StopIteration as error:
            raise ValueError("Motion semantic target stream ended before the declared corpus.") from error
        self._validate(targets)
        return targets

    def fill(self, targets: MotionSemanticTargets, start: int, stop: int) -> None:
        """Pack global rows ``[start, stop)`` into the workspace leading prefix."""
        if start != self.global_frame or stop <= start:
            raise ValueError("Motion semantic executor must request monotonic nonempty intervals.")
        destination = 0
        while self.global_frame < stop:
            if self.current is None:
                self.current = self._next()
            frame_count = self.current.position_m.shape[1]
            count = min(stop - self.global_frame, frame_count - self.local_frame)
            source = slice(self.local_frame, self.local_frame + count)
            target = slice(destination, destination + count)
            targets.position_m[:, target].copy_(self.current.position_m[:, source])
            targets.rotation_xyzw[:, target].copy_(self.current.rotation_xyzw[:, source])
            targets.source_support_height_m[target].copy_(self.current.source_support_height_m[source])
            destination += count
            self.local_frame += count
            self.global_frame += count
            if self.local_frame == frame_count:
                self.current = None
                self.local_frame = 0
                self.clip += 1

    def finish(self) -> None:
        """Require exact declared corpus exhaustion with no hidden trailing clip."""
        if self.global_frame != self.clip_index.total_frames or self.clip != len(self.clip_index.clips):
            raise ValueError("Motion semantic target stream did not cover the declared corpus exactly.")
        try:
            next(self.iterator)
        except StopIteration:
            return
        raise ValueError("Motion semantic target stream contains an undeclared trailing clip.")


def _motion_workspace_targets(prototype: MotionSemanticTargets, capacity: int) -> MotionSemanticTargets:
    """Allocate only mutable semantic fields on the reusable capacity axis."""
    return replace(
        prototype,
        position_m=torch.empty(
            (len(prototype.body_indices), capacity, 3), dtype=torch.float32, device=prototype.position_m.device
        ),
        rotation_xyzw=torch.empty(
            (len(prototype.body_indices), capacity, 4), dtype=torch.float32, device=prototype.rotation_xyzw.device
        ),
        source_support_height_m=torch.empty(capacity, dtype=torch.float32, device=prototype.position_m.device),
    )


def _source_clips(candidate: _MotionCorpusCandidate):
    """Yield every declared source clip once in exact source order."""
    if candidate.source is None:
        raise ValueError("Motion source clips were already consumed by an earlier family stage.")
    count = 0
    for clip_id, clip in candidate.source.clips():
        if count == len(candidate.clip_index.clips):
            raise ValueError(f"Motion source yielded undeclared clip {clip_id!r}.")
        expected = candidate.clip_index.clips[count]
        if (
            clip_id != expected.clip_id
            or clip.frame_count != expected.frame_count
            or clip.source_fps != expected.source_fps
        ):
            raise ValueError(
                f"Motion source expected {expected.clip_id!r} with {expected.frame_count} frames at "
                f"{expected.source_fps} Hz, got {clip_id!r} with {clip.frame_count} frames at "
                f"{clip.source_fps} Hz; its sample rate or identity changed after inspection."
            )
        yield count, clip
        count += 1
    if count != len(candidate.clip_index.clips):
        raise ValueError(f"Motion source yielded {count} of {len(candidate.clip_index.clips)} declared clips.")


def _frame_is_finite(frames: MotionFrames) -> torch.Tensor:
    """Return one device boolean for all stored fields in one clip."""
    return torch.stack(tuple(torch.all(torch.isfinite(frames.field(name))) for name in frames.stored_fields)).all()


def motion_generate_exact_coordinates(
    cfg: MotionExactCoordinatesGenerateCfg,
    candidate: _MotionCorpusCandidate,
    _rng: object,
) -> _MotionCorpusCandidate:
    """Stream exact source coordinates into one corpus-sized robot frame bank."""
    del cfg
    if candidate.frames is None:
        raise ValueError("Exact-coordinate generation requires preallocated corpus frames.")
    frame_finite = torch.empty(len(candidate.clip_index.clips), dtype=torch.bool, device=candidate.device)
    for index, clip in _source_clips(candidate):
        joint_q, joint_qd = clip.free_root_coordinates(candidate.builder.source_skeleton, device=candidate.device)
        frames = candidate.builder.build_exact_coordinates(joint_q, joint_qd, clip.source_fps)
        start, end = candidate.clip_index.offsets[index : index + 2]
        candidate.frames._copy_clip_(start, end, frames)
        frame_finite[index].copy_(_frame_is_finite(frames))
    return replace(candidate, source=None, frame_finite=frame_finite)


def motion_generate_semantic_targets(
    cfg: MotionSemanticTargetsGenerateCfg,
    candidate: _MotionCorpusCandidate,
    _rng: object,
) -> _MotionCorpusCandidate:
    """Expose one-pass semantic targets without retaining the complete source corpus."""
    del cfg

    def targets() -> Iterator[MotionSemanticTargets]:
        for _index, clip in _source_clips(candidate):
            yield _motion_semantic_targets(candidate.builder, clip, candidate.device)

    return replace(candidate, source=None, pending=targets())


def _motion_semantic_targets(builder: MotionFrameBuilder, clip, device: str | torch.device) -> MotionSemanticTargets:
    """Decode and project one clip, releasing decoded source tensors before its solve."""
    root_position, local_rotation = clip.semantic_local_pose(builder.source_skeleton, device=device)
    return builder.generate_semantic_targets(root_position, local_rotation)


def motion_solve_semantic_sequence(
    cfg: MotionSemanticSolveCfg,
    candidate: _MotionCorpusCandidate,
) -> _MotionCorpusCandidate:
    """Solve one streamed semantic corpus through one reusable GPU workspace."""
    if candidate.pending is None:
        raise ValueError("Motion semantic solve requires a generated target stream.")
    if tuple(type(objective) for objective in cfg.objectives) != (
        MotionLandmarkPositionObjectiveCfg,
        MotionLandmarkRotationObjectiveCfg,
    ):
        raise ValueError("Motion semantic solve requires position then rotation objectives.")
    try:
        first_targets = next(candidate.pending)
    except StopIteration as error:
        raise ValueError("Motion semantic solve requires at least one source clip.") from error
    prototype = _motion_workspace_targets(first_targets, 1)
    stream = _MotionSemanticTargetStream(
        iterator=candidate.pending,
        clip_index=candidate.clip_index,
        prototype=prototype,
        current=first_targets,
    )
    stream._validate(first_targets)
    del first_targets

    reference = candidate.builder.semantic_reference_kinematics
    target_tree = candidate.builder.semantic_target_tree
    model = reference.model
    coordinate_count = model.joint_coord_count
    dof_count = model.joint_dof_count
    body_count = model.body_count
    default_joint_q = torch.as_tensor(reference.default_joint_q, dtype=torch.float32, device=reference.device)
    if default_joint_q.shape != (coordinate_count,):
        raise ValueError("Motion semantic IK requires one default value per target coordinate.")

    representative_objectives = motion_semantic_objectives(cfg, prototype)
    jacobian_mode = (
        ik.IKJacobianType.MIXED
        if any(not objective.supports_analytic() for objective in representative_objectives)
        else ik.IKJacobianType.ANALYTIC
    )
    float_bytes = wp.types.type_size_in_bytes(wp.float32)
    transform_bytes = wp.types.type_size_in_bytes(wp.transformf)
    spatial_bytes = wp.types.type_size_in_bytes(wp.spatial_vectorf)
    static_bytes = (
        3 * prototype.coordinate_indices.numel() + len(prototype.body_indices) + prototype.support_body_indices.numel()
    ) * wp.types.type_size_in_bytes(wp.int32)

    def estimate_memory(batch_size: int) -> int:
        solver_bytes = ik.IKSolver.estimate_memory(
            model,
            batch_size,
            representative_objectives,
            optimizer=ik.IKOptimizer.LM,
            jacobian_mode=jacobian_mode,
            sampler=ik.IKSampler.NONE,
            n_seeds=1,
        ).total_bytes
        workspace_bytes = batch_size * (
            coordinate_count * float_bytes
            + dof_count * float_bytes
            + body_count * (transform_bytes + spatial_bytes)
            + 4 * float_bytes
            + wp.types.type_size_in_bytes(wp.uint8)
        )
        seed_scratch_bytes = batch_size * (4 * target_tree.num_bodies + 8 * coordinate_count) * float_bytes
        return solver_bytes + workspace_bytes + seed_scratch_bytes + static_bytes

    def build_batch(batch_size: int) -> _MotionSemanticWorkspace:
        targets = _motion_workspace_targets(prototype, batch_size)
        objectives = motion_semantic_objectives(cfg, targets)
        solver = ik.IKSolver(
            model=model,
            n_problems=batch_size,
            objectives=objectives,
            optimizer=ik.IKOptimizer.LM,
            jacobian_mode=jacobian_mode,
            sampler=ik.IKSampler.NONE,
            n_seeds=1,
            lambda_initial=cfg.lambda_initial,
            lambda_factor=cfg.lambda_factor,
            lambda_min=cfg.lambda_min,
            lambda_max=cfg.lambda_max,
            rho_min=cfg.rho_min,
            history_len=cfg.history_length,
            h0_scale=cfg.h0_scale,
            line_search_alphas=None,
            wolfe_c1=cfg.wolfe_c1,
            wolfe_c2=cfg.wolfe_c2,
        )
        return _MotionSemanticWorkspace(
            solver=solver,
            targets=targets,
            joint_q=torch.empty((batch_size, coordinate_count), dtype=torch.float32, device=reference.device),
            joint_qd=wp.zeros((batch_size, dof_count), dtype=wp.float32, device=reference.device),
            body_q=wp.empty((batch_size, body_count), dtype=wp.transformf, device=reference.device),
            body_qd=wp.empty((batch_size, body_count), dtype=wp.spatial_vectorf, device=reference.device),
            coordinate_indices=wp.array(
                prototype.coordinate_indices.detach().cpu().numpy(), dtype=wp.int32, device=reference.device
            ),
            coordinate_lower=wp.from_torch(prototype.coordinate_lower_limits_rad),
            coordinate_upper=wp.from_torch(prototype.coordinate_upper_limits_rad),
            body_indices=wp.array(prototype.body_indices, dtype=wp.int32, device=reference.device),
            support_body_indices=wp.array(
                prototype.support_body_indices.detach().cpu().numpy(), dtype=wp.int32, device=reference.device
            ),
            quality=torch.empty((batch_size, 2), dtype=torch.float32, device=reference.device),
            frame_finite=torch.empty(batch_size, dtype=torch.uint8, device=reference.device),
            support_error_m=torch.empty(batch_size, dtype=torch.float32, device=reference.device),
            capacity=batch_size,
        )

    frame_count = candidate.clip_index.total_frames
    semantic_joint_q = torch.empty((frame_count, coordinate_count), dtype=torch.float32, device=reference.device)
    semantic_quality = torch.empty((frame_count, 3), dtype=torch.float32, device=reference.device)
    frame_finite = torch.empty(frame_count, dtype=torch.uint8, device=reference.device)

    def solve_batch(workspace, start, stop, max_iterations, tolerance, check_interval):
        active_count = stop - start
        stream.fill(workspace.targets, start, stop)
        active_joint_q = workspace.joint_q[:active_count]
        active_joint_q.copy_(default_joint_q)
        active_joint_q[:, :3].copy_(workspace.targets.position_m[0, :active_count])
        active_joint_q[:, 3:7].copy_(workspace.targets.rotation_xyzw[0, :active_count])
        kinematic_seed_target_rotations(
            target_tree,
            reference.topology,
            target_body_indices=workspace.targets.body_indices,
            target_parent_rows=workspace.targets.parent_rows,
            target_rotation_xyzw=workspace.targets.rotation_xyzw[:, :active_count],
            joint_q=active_joint_q,
        )

        def project(joint_q: wp.array) -> None:
            semantic_project_coordinates(
                joint_q,
                workspace.targets.position_m[0],
                workspace.targets.rotation_xyzw[0],
                workspace.coordinate_indices,
                workspace.coordinate_lower,
                workspace.coordinate_upper,
                reference.device,
            )

        result = workspace.solver.solve(
            wp.from_torch(workspace.joint_q),
            wp.from_torch(workspace.joint_q),
            max_iterations=max_iterations,
            active_problem_count=active_count,
            convergence_tolerance=tolerance,
            convergence_check_interval=check_interval,
            projection=project,
            projection_interval=cfg.projection_interval,
        )
        reference.eval_fk_batched(
            wp.from_torch(active_joint_q),
            workspace.joint_qd[:active_count],
            workspace.body_q[:active_count],
            workspace.body_qd[:active_count],
        )
        semantic_correct_support(
            active_joint_q,
            workspace.body_q[:active_count],
            workspace.support_body_indices,
            workspace.targets.source_support_height_m,
            workspace.targets.position_m,
            reference.device,
        )
        reference.eval_fk_batched(
            wp.from_torch(active_joint_q),
            workspace.joint_qd[:active_count],
            workspace.body_q[:active_count],
            workspace.body_qd[:active_count],
        )
        semantic_measure_quality(
            workspace.body_q[:active_count],
            active_joint_q,
            workspace.body_indices,
            workspace.targets.position_m,
            workspace.targets.rotation_xyzw,
            workspace.support_body_indices,
            workspace.targets.source_support_height_m,
            workspace.quality[:active_count],
            workspace.frame_finite[:active_count],
            workspace.support_error_m[:active_count],
            reference.device,
        )
        torch._assert_async(
            torch.all(workspace.support_error_m[:active_count] <= cfg.support_landmark_atol_m),
            "Motion semantic support correction exceeded its declared tolerance.",
        )
        semantic_joint_q[start:stop].copy_(active_joint_q)
        semantic_quality[start:stop, :2].copy_(workspace.quality[:active_count])
        frame_finite[start:stop].copy_(workspace.frame_finite[:active_count])
        return result

    solve_statistics = execute_ik_batches(
        problem_count=frame_count,
        device=reference.device,
        estimate_memory=estimate_memory,
        build_batch=build_batch,
        solve_batch=solve_batch,
        max_iterations=cfg.max_iterations,
        convergence_tolerance=cfg.convergence_tolerance,
        convergence_check_interval=cfg.convergence_check_interval,
    )
    stream.finish()
    semantic_measure_branch_jump(semantic_joint_q, semantic_quality, reference.device)
    clip_starts = torch.tensor(candidate.clip_index.offsets[:-1], dtype=torch.int64, device=reference.device)
    semantic_quality[clip_starts, 2] = 0.0
    return replace(
        candidate,
        pending=None,
        semantic_joint_q=semantic_joint_q,
        semantic_quality=semantic_quality,
        frame_finite=frame_finite.bool(),
        solve_statistics=solve_statistics,
    )


def motion_semantic_objectives(
    cfg: MotionSemanticSolveCfg,
    targets: MotionSemanticTargets,
) -> list[object]:
    """Construct the configured flat Newton objective list in declaration order."""
    objectives = []
    for objective_cfg in cfg.objectives:
        factory = objective_cfg.class_type
        if not callable(factory):
            factory = string_to_callable(factory)
        objectives.extend(factory(objective_cfg, targets))
    return objectives


def motion_objective_landmark_position(
    cfg: MotionLandmarkPositionObjectiveCfg,
    targets: MotionSemanticTargets,
) -> list[object]:
    """Build every length-normalized Newton landmark-position objective."""
    import newton.ik as ik
    import warp as wp

    lengths = targets.segment_length_values_m
    objectives = [
        ik.IKObjectivePosition(
            link_index=targets.body_indices[row],
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.from_torch(targets.position_m[row], dtype=wp.vec3),
            weight=cfg.weight / lengths[row],
        )
        for row in range(1, len(targets.body_indices))
    ]
    objectives.append(
        ik.IKObjectivePosition(
            link_index=targets.body_indices[0],
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.from_torch(targets.position_m[0], dtype=wp.vec3),
            weight=cfg.root_weight / lengths[0],
        )
    )
    return objectives


def motion_objective_landmark_rotation(
    cfg: MotionLandmarkRotationObjectiveCfg,
    targets: MotionSemanticTargets,
) -> list[object]:
    """Build every Newton landmark-orientation objective."""
    import newton.ik as ik
    import warp as wp

    objectives = [
        ik.IKObjectiveRotation(
            link_index=targets.body_indices[row],
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.from_torch(targets.rotation_xyzw[row], dtype=wp.vec4),
            canonicalize_quat_err=cfg.canonicalize_error,
            weight=cfg.weight,
        )
        for row in range(1, len(targets.body_indices))
    ]
    objectives.append(
        ik.IKObjectiveRotation(
            link_index=targets.body_indices[0],
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.from_torch(targets.rotation_xyzw[0], dtype=wp.vec4),
            canonicalize_quat_err=cfg.canonicalize_error,
            weight=cfg.root_weight,
        )
    )
    return objectives


def motion_criterion_frame_finite(
    _cfg: MotionFrameFiniteCriterionCfg,
    candidate: _MotionCorpusCandidate,
    rows: torch.Tensor,
) -> torch.Tensor:
    """Accept active materialized clips or frames only when every stored value is finite."""
    if candidate.frame_finite is None:
        raise ValueError("Frame finiteness requires materialized corpus evidence.")
    return candidate.frame_finite[rows]


def motion_criterion_objective_measure(
    cfg: MotionObjectiveMeasureCriterionCfg,
    candidate: _MotionCorpusCandidate,
    rows: torch.Tensor,
) -> torch.Tensor:
    """Accept active cached semantic objective measures when finite and bounded."""
    if candidate.semantic_quality is None:
        raise ValueError("Semantic objective criteria require a completed solve.")
    if cfg.objective == "landmark_position":
        values = candidate.semantic_quality[rows, 0]
    elif cfg.objective == "landmark_rotation":
        values = candidate.semantic_quality[rows, 1]
    else:
        raise ValueError(f"Unknown semantic objective measure: {cfg.objective!r}.")
    return torch.isfinite(values) & (values <= cfg.upper)


def _branch_jump_before(joint_q: torch.Tensor) -> torch.Tensor:
    """Return the maximum coordinate jump on each incoming edge [rad]."""
    jump = torch.zeros(joint_q.shape[0], dtype=torch.float32, device=joint_q.device)
    if joint_q.shape[0] > 1:
        jump[1:] = torch.abs(joint_q[1:, 7:] - joint_q[:-1, 7:]).amax(dim=1)
    return jump


@dataclass(frozen=True, slots=True)
class _MotionSegmentRuns:
    """Flat maximal semantic runs retained in source order."""

    starts: torch.Tensor
    stops: torch.Tensor
    source_indices: torch.Tensor

    @property
    def lengths(self) -> torch.Tensor:
        """Number of frames in each retained run."""
        return self.stops - self.starts

    def flat_indices(self) -> torch.Tensor:
        """Materialize retained source rows without losing explicit cuts."""
        lengths = self.lengths
        if not lengths.numel():
            return torch.empty(0, dtype=torch.int64, device=self.starts.device)
        output_starts = torch.cumsum(lengths, dim=0) - lengths
        return (
            torch.arange(int(lengths.sum()), dtype=torch.int64, device=self.starts.device)
            - torch.repeat_interleave(output_starts, lengths)
            + torch.repeat_interleave(self.starts, lengths)
        )


def _motion_semantic_runs(
    clip_index: MotionClipIndex,
    accepted: torch.Tensor,
    branch_jump_before: torch.Tensor,
    max_branch_jump_rad: float,
) -> _MotionSegmentRuns:
    """Return maximal valid source-ordered runs of at least three frames."""
    frame_count = clip_index.total_frames
    if (
        accepted.shape != (frame_count,)
        or accepted.dtype is not torch.bool
        or branch_jump_before.shape != (frame_count,)
        or branch_jump_before.device != accepted.device
    ):
        raise ValueError("Semantic node and incoming-edge evidence must share the complete flat frame axis.")
    offsets = torch.tensor(clip_index.offsets, dtype=torch.int64, device=accepted.device)
    clip_start = torch.zeros(frame_count, dtype=torch.bool, device=accepted.device)
    clip_start[offsets[:-1]] = True
    edge_cut = ~torch.isfinite(branch_jump_before) | (branch_jump_before > max_branch_jump_rad)
    edge_cut[clip_start] = True

    previous_valid = torch.zeros_like(accepted)
    previous_valid[1:] = accepted[:-1]
    next_valid = torch.zeros_like(accepted)
    next_valid[:-1] = accepted[1:]
    cut_before_next = torch.ones_like(accepted)
    cut_before_next[:-1] = edge_cut[1:]
    start_mask = accepted & (clip_start | ~previous_valid | edge_cut)
    stop_mask = accepted & (~next_valid | cut_before_next)
    starts = torch.nonzero(start_mask, as_tuple=False).squeeze(-1)
    stops = torch.nonzero(stop_mask, as_tuple=False).squeeze(-1) + 1
    if starts.shape != stops.shape:
        raise ValueError("Semantic node/edge evidence produced unpaired contiguous-run boundaries.")
    keep = stops - starts >= 3
    starts = starts[keep]
    stops = stops[keep]
    source_indices = torch.bucketize(starts, offsets[1:], right=True)
    return _MotionSegmentRuns(starts, stops, source_indices)


def motion_select_semantic_segments(
    cfg: MotionSemanticSegmentSelectionCfg,
    candidate: _MotionCorpusCandidate,
    accepted: torch.Tensor | None,
    target_count: int | None,
    _rng: object,
) -> torch.Tensor:
    """Retain all frames in maximal valid semantic runs while preserving edge cuts."""
    if target_count is not None:
        raise ValueError("Motion semantic segment selection retains every valid run.")
    if accepted is None or candidate.semantic_quality is None:
        raise ValueError("Motion semantic segment selection requires completed node and edge evidence.")
    runs = _motion_semantic_runs(
        candidate.clip_index,
        accepted,
        candidate.semantic_quality[:, 2],
        cfg.max_branch_jump_rad,
    )
    return runs.flat_indices()


def motion_select_source_order(
    cfg: MotionClipSelectionCfg,
    candidate: _MotionCorpusCandidate,
    accepted: torch.Tensor | None,
    target_count: int | None,
    _rng: object,
) -> torch.Tensor:
    """Retain accepted corpus clips once in deterministic source order."""
    if target_count is not None:
        raise ValueError("Motion source-order selection retains every accepted clip.")
    source_indices = torch.arange(len(candidate.clip_index.clips), dtype=torch.int64, device=candidate.device)
    selected = source_indices if accepted is None else source_indices[accepted]
    return selected if cfg.max_clips is None else selected[: cfg.max_clips]


def _table_identity(
    clip_index: MotionClipIndex,
    source_skeleton_identity_sha256: str,
    frames: MotionFrames,
    joint_names: tuple[str, ...],
    reference_frame_names: tuple[str, ...],
    frame_builder_version: str,
    frame_builder_identity_sha256: str,
    task_row_mode: Literal["source_frames", "clip_time_ranges"],
    family_name: str,
    family_identity_sha256: str,
) -> str:
    """Return deterministic trajectory-data provenance without robot ownership."""
    validate_nonempty("frame_builder_version", frame_builder_version)
    validate_sha256("source_skeleton_identity_sha256", source_skeleton_identity_sha256)
    validate_sha256("frame_builder_identity_sha256", frame_builder_identity_sha256)
    validate_nonempty("family_name", family_name)
    validate_sha256("family_identity_sha256", family_identity_sha256)
    stored_column_shapes = {name: tuple(frames.field(name).shape[1:]) for name in frames.stored_fields}
    return canonical_sha256(
        {
            "source_content_hash": clip_index.source_content_sha256,
            "source_clips": [
                {
                    "clip_id": clip.clip_id,
                    "frame_count": clip.frame_count,
                    "source_fps": clip.source_fps,
                    "content_sha256": clip.content_sha256,
                    "source_clip_id": clip.source_clip_id,
                    "source_frame_start": clip.source_frame_start,
                    "source_frame_stop": clip.source_frame_stop,
                }
                for clip in clip_index.clips
            ],
            "source_skeleton_hash": source_skeleton_identity_sha256,
            "frame_builder_version": frame_builder_version,
            "joint_names": joint_names,
            "reference_frame_names": reference_frame_names,
            "frame_builder_identity_sha256": frame_builder_identity_sha256,
            "family_name": family_name,
            "family_identity_sha256": family_identity_sha256,
            "task_row_mode": task_row_mode,
            "stored_columns": stored_column_shapes,
            "root_storage": frames.root_storage,
        }
    )


def _task_rows(
    clip_index: MotionClipIndex,
    device: torch.device,
    mode: Literal["source_frames", "clip_time_ranges"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic descriptor rows from one controlled mode."""
    if mode not in ("source_frames", "clip_time_ranges"):
        raise ValueError("task_row_mode must be 'source_frames' or 'clip_time_ranges'.")
    frame_counts = torch.tensor([clip.frame_count for clip in clip_index.clips], dtype=torch.int64, device=device)
    source_fps = torch.tensor([clip.source_fps for clip in clip_index.clips], dtype=torch.float32, device=device)
    clip_indices = torch.arange(len(clip_index.clips), dtype=torch.int64, device=device)
    if mode == "clip_time_ranges":
        end = (frame_counts - 1) / source_fps
        return clip_indices, torch.stack((torch.zeros_like(end), end), dim=-1)

    clips = torch.repeat_interleave(clip_indices, frame_counts)
    output_offsets = torch.cumsum(frame_counts, dim=0) - frame_counts
    repeated_output_offsets = torch.repeat_interleave(output_offsets, frame_counts)
    local_frames = torch.arange(clips.shape[0], device=device) - repeated_output_offsets
    times = local_frames / source_fps[clips]
    return clips, torch.stack((times, times), dim=-1)


def _motion_task_view(
    clip_index: MotionClipIndex,
    frames: MotionFrames,
    joint_names: tuple[str, ...],
    reference: object,
    quality_values: torch.Tensor,
) -> TaskTableView:
    """Expose exact stored frames through the shared canonical table boundary."""
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
        joint_q_indices=torch.arange(7, 7 + len(joint_names), dtype=torch.int64, device=device),
    )
    sequences = TaskTableSequenceIndex(
        offsets=torch.tensor(clip_index.offsets, dtype=torch.int64, device=device),
        frame_dt=torch.tensor(
            [1.0 / clip.source_fps for clip in clip_index.clips],
            dtype=torch.float32,
            device=device,
        ),
    )
    quality = TaskTableQuality(names=_QUALITY_NAMES, values=quality_values, scope="sequence")
    return TaskTableView(
        sequences=sequences,
        state_bank=state_bank,
        kinematic_view=kinematic_view,
        quality=quality,
    )


def _exact_corpus_quality(candidate: _MotionCorpusCandidate, accepted: torch.Tensor) -> torch.Tensor:
    """Materialize one quality row per exact-coordinate source clip."""
    clip_count = len(candidate.clip_index.clips)
    if accepted.shape != (clip_count,) or accepted.dtype is not torch.bool:
        raise ValueError("Exact Motion acceptance must contain one boolean per source clip.")
    if candidate.frames is None or candidate.frame_finite is None or candidate.semantic_quality is not None:
        raise ValueError("Exact Motion quality requires materialized frames and clip-level finite evidence.")
    quality = torch.zeros(clip_count, len(_QUALITY_NAMES), dtype=torch.float32, device=candidate.frames.device)
    quality[:, 1].copy_(accepted)
    quality[:, 5].fill_(1.0)
    return quality


def _compact_selected_exact_corpus(
    clip_index: MotionClipIndex,
    frames: MotionFrames,
    quality: torch.Tensor,
    selected_indices: torch.Tensor,
) -> tuple[MotionClipIndex, MotionFrames, torch.Tensor]:
    """Compact selected clips and their exact stored frames in source order."""
    selected = tuple(int(index) for index in selected_indices.detach().cpu().tolist())
    if not selected:
        raise ValueError("Motion criteria rejected every source clip.")
    if selected != tuple(sorted(set(selected))):
        raise ValueError("Motion clip selection must preserve unique source order.")
    if selected == tuple(range(len(clip_index.clips))):
        return clip_index, frames, quality

    selected_clips = tuple(clip_index.clips[index] for index in selected)
    selected_index = MotionClipIndex(source_content_sha256=clip_index.source_content_sha256, clips=selected_clips)
    frame_indices = torch.cat(
        tuple(
            torch.arange(clip_index.offsets[index], clip_index.offsets[index + 1], device=frames.device)
            for index in selected
        )
    )
    selected_frames = MotionFrames(
        **{name: frames.field(name).index_select(0, frame_indices).contiguous() for name in frames.stored_fields}
    )
    return selected_index, selected_frames, quality.index_select(0, selected_indices).contiguous()


def _derived_segment_clip(
    source: MotionClipIndex.Clip,
    local_start: int,
    local_stop: int,
) -> MotionClipIndex.Clip:
    """Return the original full-span clip or one stable source-range descriptor."""
    if local_start == 0 and local_stop == source.frame_count:
        return source
    source_clip_id = source.source_clip_id or source.clip_id
    source_frame_start = source.source_frame_start + local_start
    source_frame_stop = source.source_frame_start + local_stop
    content_sha256 = canonical_sha256(
        {
            "source_clip_sha256": source.content_sha256,
            "local_frame_start": local_start,
            "local_frame_stop": local_stop,
            "source_clip_id": source_clip_id,
            "source_frame_start": source_frame_start,
            "source_frame_stop": source_frame_stop,
        }
    )
    return MotionClipIndex.Clip(
        clip_id=f"{source_clip_id}__frames_{source_frame_start}_{source_frame_stop}__{content_sha256[:12]}",
        frame_count=local_stop - local_start,
        source_fps=source.source_fps,
        content_sha256=content_sha256,
        source_clip_id=source_clip_id,
        source_frame_start=source_frame_start,
    )


def _semantic_segment_quality(
    candidate: _MotionCorpusCandidate,
    runs: _MotionSegmentRuns,
    frame_indices: torch.Tensor,
) -> torch.Tensor:
    """Aggregate per-frame semantic evidence and source-balanced base mass per segment."""
    if candidate.semantic_quality is None:
        raise ValueError("Semantic segment quality requires per-frame solve evidence.")
    lengths = runs.lengths
    segment_count = lengths.shape[0]
    selected = candidate.semantic_quality.index_select(0, frame_indices).clone()
    compact_starts = torch.cumsum(lengths, dim=0) - lengths
    selected[compact_starts, 2] = 0.0
    segment_ids = torch.repeat_interleave(
        torch.arange(segment_count, dtype=torch.int64, device=frame_indices.device), lengths
    )
    maxima = torch.full((segment_count, 3), -torch.inf, dtype=torch.float32, device=frame_indices.device)
    maxima.scatter_reduce_(0, segment_ids[:, None].expand(-1, 3), selected, reduce="amax", include_self=True)
    source_counts = torch.bincount(runs.source_indices, minlength=len(candidate.clip_index.clips))
    base_priority = torch.reciprocal(source_counts.index_select(0, runs.source_indices).to(torch.float32))
    quality = torch.ones(segment_count, len(_QUALITY_NAMES), dtype=torch.float32, device=frame_indices.device)
    quality[:, 0].fill_(1.0)
    quality[:, 2:5].copy_(maxima)
    quality[:, 5].copy_(base_priority)
    return quality


def _finalize_semantic_corpus(
    candidate: _MotionCorpusCandidate,
    accepted: torch.Tensor,
    selected_indices: torch.Tensor,
    selection_cfg: MotionSemanticSegmentSelectionCfg,
) -> tuple[MotionClipIndex, MotionFrames, torch.Tensor]:
    """Compact semantic coordinates once, then materialize one segment-aware robot corpus."""
    if candidate.frames is not None or candidate.semantic_joint_q is None or candidate.semantic_quality is None:
        raise ValueError("Semantic finalization requires flat solved coordinates without pre-cut frames.")
    runs = _motion_semantic_runs(
        candidate.clip_index,
        accepted,
        candidate.semantic_quality[:, 2],
        selection_cfg.max_branch_jump_rad,
    )
    frame_indices = runs.flat_indices()
    if not frame_indices.numel():
        raise ValueError("Motion semantic criteria retained no builder-safe segment.")
    if not torch.equal(frame_indices, selected_indices):
        raise ValueError("Motion semantic selection and final compaction derived different source rows.")

    metadata = torch.stack((runs.source_indices, runs.starts, runs.stops), dim=-1).detach().cpu().numpy()
    clips = []
    for source_index, start, stop in metadata:
        source = candidate.clip_index.clips[int(source_index)]
        source_start = candidate.clip_index.offsets[int(source_index)]
        clips.append(_derived_segment_clip(source, int(start) - source_start, int(stop) - source_start))
    clip_index = MotionClipIndex(
        source_content_sha256=candidate.clip_index.source_content_sha256,
        clips=tuple(clips),
    )
    joint_q = candidate.semantic_joint_q.index_select(0, frame_indices).contiguous()
    frames = candidate.builder.build_semantic_corpus(joint_q, clip_index)
    return clip_index, frames, _semantic_segment_quality(candidate, runs, frame_indices)


class MotionTaskTable:
    """Exact-capacity trajectory tensors plus selectable motion descriptors.

    Robot identity, kinematic structure, control, and defaults remain owned by
    the selected preset and scene articulation. Builders resolve source data to
    live simulator order once; this table then owns only concrete task tensors,
    clip boundaries, and descriptor rows.
    """

    class ReferenceView:
        """Batched continuous-time reference lookup with explicit tail validity."""

        __slots__ = (
            "_alpha",
            "_global_frame0",
            "_global_frame1",
            "table",
            "clip_indices",
            "local_frame0",
            "local_frame1",
            "tail_valid",
            "time_seconds",
        )

        def __init__(
            self,
            table: MotionTaskTable,
            clip_indices: torch.Tensor,
            time_seconds: torch.Tensor,
            local_frame0: torch.Tensor,
            local_frame1: torch.Tensor,
            global_frame0: torch.Tensor,
            global_frame1: torch.Tensor,
            alpha: torch.Tensor,
            tail_valid: torch.Tensor,
        ) -> None:
            self.table = table
            self.clip_indices = clip_indices
            self.time_seconds = time_seconds
            self.local_frame0 = local_frame0
            self.local_frame1 = local_frame1
            self._global_frame0 = global_frame0
            self._global_frame1 = global_frame1
            self._alpha = alpha
            self.tail_valid = tail_valid

        def field(self, name: str) -> torch.Tensor:
            """Interpolate one named trajectory field at every requested time."""
            values = self.table.field(name)
            value0 = values[self._global_frame0]
            interpolation = self.table.interpolation(name)
            value1 = values[self._global_frame1]
            if interpolation == "slerp":
                return quat_slerp(value0, value1, self._alpha)
            fraction = self._alpha
            while fraction.ndim < value0.ndim:
                fraction = fraction.unsqueeze(-1)
            return torch.lerp(value0, value1, fraction)

    class SampledSequence:
        """Clip-safe trajectory view on one declared sample clock."""

        __slots__ = ("_field", "clip_ids", "clip_offsets", "data_hash", "dataset_id", "device", "source")

        def __init__(
            self,
            source: MotionTaskTable,
            clip_offsets: tuple[int, ...],
            sampling_mode: Literal["source_rows", "uniform_before_source_end"],
            sampling_step_seconds: float | None,
            field: Callable[[str], torch.Tensor],
        ) -> None:
            self.source = source
            self.device = source.device
            self.clip_ids = source.clip_ids
            self.clip_offsets = clip_offsets
            self.dataset_id = f"{source.clip_index.source_content_sha256}:{source.frame_builder_version}"
            self.data_hash = canonical_sha256(
                {
                    "source": source.cache_identity,
                    "sampling_mode": sampling_mode,
                    "sampling_step_seconds": sampling_step_seconds,
                }
            )
            self._field = field

        def field(self, name: str) -> torch.Tensor:
            """Return one sampled trajectory field."""
            return self._field(name)

    __slots__ = (
        "_cache_identity",
        "_clip_index",
        "_clip_offsets",
        "_clip_start_rows",
        "_frame_builder_identity_sha256",
        "_frame_builder_version",
        "_frame_counts",
        "_family_name",
        "_family_identity_sha256",
        "_frames",
        "_joint_names",
        "_reference_frame_names",
        "_sealed",
        "_task_row_mode",
        "_source_fps",
        "_view",
        "clip_indices",
        "reset_time_ranges_seconds",
    )

    def __init__(
        self,
        clip_index: MotionClipIndex,
        frames: MotionFrames,
        joint_names: tuple[str, ...],
        reference_frame_names: tuple[str, ...],
        frame_builder_version: str,
        frame_builder_identity_sha256: str,
        task_row_mode: Literal["source_frames", "clip_time_ranges"],
        source_skeleton_identity_sha256: str,
        family_name: str,
        family_identity_sha256: str,
        view: TaskTableView,
    ) -> None:
        validate_sha256("source_skeleton_identity_sha256", source_skeleton_identity_sha256)
        if frames.frame_count != clip_index.total_frames:
            raise ValueError("Trajectory capacity must equal the declared source frame count exactly.")
        validate_nonempty("frame_builder_version", frame_builder_version)
        if (
            not isinstance(joint_names, tuple)
            or not joint_names
            or any(not isinstance(name, str) or not name for name in joint_names)
            or len(set(joint_names)) != len(joint_names)
        ):
            raise ValueError("Trajectory joint_names must be a nonempty tuple of unique names.")
        if len(joint_names) != frames.field("joint_position").shape[1]:
            raise ValueError("Trajectory joint_names must match the joint-column axis exactly.")
        if (
            not isinstance(reference_frame_names, tuple)
            or any(not isinstance(name, str) or not name for name in reference_frame_names)
            or len(set(reference_frame_names)) != len(reference_frame_names)
        ):
            raise ValueError("Trajectory reference_frame_names must be a tuple of unique names.")
        body_position = frames.body_position
        if body_position is None:
            if reference_frame_names:
                raise ValueError("Trajectory reference-frame names require reference-frame columns.")
        elif len(reference_frame_names) != body_position.shape[1]:
            raise ValueError("Trajectory reference_frame_names must match the reference-frame column axis exactly.")

        validate_sha256("frame_builder_identity_sha256", frame_builder_identity_sha256)
        validate_nonempty("family_name", family_name)
        validate_sha256("family_identity_sha256", family_identity_sha256)
        if (
            view.state_bank.row_count != frames.frame_count
            or view.state_bank.layout.names != ("robot",)
            or view.state_bank.layout.joint_names != (joint_names,)
            or view.sequences.sequence_count != len(clip_index.clips)
            or view.sequences.frame_count != clip_index.total_frames
        ):
            raise ValueError("Motion task view must address the exact stored robot frames and clip offsets.")

        device = frames.device
        clip_offsets = torch.tensor(clip_index.offsets, dtype=torch.int64, device=device)
        frame_counts = torch.tensor([clip.frame_count for clip in clip_index.clips], dtype=torch.int64, device=device)
        source_fps = torch.tensor([clip.source_fps for clip in clip_index.clips], dtype=torch.float32, device=device)
        clip_indices, reset_time_ranges_seconds = _task_rows(clip_index, device, task_row_mode)
        num_tasks = clip_indices.shape[0]
        if task_row_mode == "source_frames":
            clip_start_rows = torch.tensor(clip_index.offsets[:-1], dtype=torch.int64, device=device)
        else:
            clip_start_rows = torch.arange(len(clip_index.clips), dtype=torch.int64, device=device)
        if (
            clip_indices.ndim != 1
            or clip_indices.dtype is not torch.int64
            or reset_time_ranges_seconds.shape != (num_tasks, 2)
            or not reset_time_ranges_seconds.is_floating_point()
            or num_tasks == 0
        ):
            raise ValueError("Motion tasks require clip [N] and reset-time-range [N, 2] tensors.")
        if clip_indices.device != device or reset_time_ranges_seconds.device != device:
            raise ValueError("Motion task and trajectory tensors must share one device.")

        torch._assert_async(
            torch.all((clip_indices >= 0) & (clip_indices < len(clip_index.clips))),
            "Motion task clip indices are outside the stored clips.",
        )
        low, high = reset_time_ranges_seconds.unbind(-1)
        clip_end = (frame_counts[clip_indices] - 1) / source_fps[clip_indices]
        torch._assert_async(
            torch.all(torch.isfinite(reset_time_ranges_seconds) & (reset_time_ranges_seconds >= 0.0)),
            "Motion reset-time ranges must be finite and nonnegative.",
        )
        torch._assert_async(
            torch.all((high >= low) & (high <= clip_end)),
            "A motion reset-time range crosses its clip boundary.",
        )
        expected = torch.arange(len(clip_index.clips), dtype=torch.int64, device=device)
        if not torch.equal(torch.unique(clip_indices, sorted=True), expected):
            raise ValueError("Motion task rows must cover every clip in stable source order.")

        object.__setattr__(self, "_clip_index", clip_index)
        object.__setattr__(self, "_frames", frames)
        object.__setattr__(self, "_frame_builder_version", frame_builder_version)
        object.__setattr__(self, "_frame_builder_identity_sha256", frame_builder_identity_sha256)
        object.__setattr__(self, "_joint_names", joint_names)
        object.__setattr__(self, "_reference_frame_names", reference_frame_names)
        object.__setattr__(self, "_clip_offsets", clip_offsets)
        object.__setattr__(self, "_clip_start_rows", clip_start_rows)
        object.__setattr__(self, "_frame_counts", frame_counts)
        object.__setattr__(self, "_source_fps", source_fps)
        object.__setattr__(self, "_view", view)
        object.__setattr__(self, "clip_indices", clip_indices)
        object.__setattr__(self, "reset_time_ranges_seconds", reset_time_ranges_seconds)
        object.__setattr__(
            self,
            "_cache_identity",
            _table_identity(
                clip_index,
                source_skeleton_identity_sha256,
                frames,
                joint_names,
                reference_frame_names,
                frame_builder_version,
                frame_builder_identity_sha256,
                task_row_mode,
                family_name,
                family_identity_sha256,
            ),
        )
        object.__setattr__(self, "_task_row_mode", task_row_mode)
        object.__setattr__(self, "_family_name", family_name)
        object.__setattr__(self, "_family_identity_sha256", family_identity_sha256)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("MotionTaskTable metadata is immutable.")
        object.__setattr__(self, name, value)

    @property
    def frames(self) -> MotionFrames:
        """Concrete trajectory tensor owner."""
        return self._frames

    @property
    def view(self) -> TaskTableView:
        """Canonical physical states, exact mechanics, and clip sequence view."""
        return self._view

    @property
    def base_priorities(self) -> torch.Tensor:
        """Immutable source-balanced sampling mass per retained sequence."""
        quality = self._view.quality
        if quality is None or quality.scope != "sequence" or "base_priority" not in quality.names:
            raise ValueError("Motion table quality must expose one sequence-scoped base_priority column.")
        return quality.values[:, quality.names.index("base_priority")]

    @property
    def family_name(self) -> str:
        """Exact or semantic coordinate family selected for this table."""
        return self._family_name

    @property
    def family_identity_sha256(self) -> str:
        """Complete selected generation, solve, acceptance, and selection policy identity."""
        return self._family_identity_sha256

    @property
    def clip_index(self) -> MotionClipIndex:
        """Ordered source clip metadata."""
        return self._clip_index

    @property
    def frame_builder_version(self) -> str:
        """Readable version of source-to-trajectory conversion."""
        return self._frame_builder_version

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Live-articulation order of every joint trajectory column."""
        return self._joint_names

    @property
    def reference_frame_names(self) -> tuple[str, ...]:
        """Ordered semantic labels of the optional reference-frame columns."""
        return self._reference_frame_names

    @property
    def frame_builder_identity_sha256(self) -> str:
        """Construction identity closing reference kinematics and ordered mappings."""
        return self._frame_builder_identity_sha256

    @property
    def task_row_mode(self) -> Literal["source_frames", "clip_time_ranges"]:
        """Controlled rule used to derive selectable descriptor rows."""
        return self._task_row_mode

    @property
    def cache_identity(self) -> str:
        """Deterministic source, builder, columns, and timing identity."""
        return self._cache_identity

    @property
    def clip_offsets(self) -> torch.Tensor:
        """Clip prefix offsets on the trajectory frame axis."""
        return self._clip_offsets

    @property
    def clip_start_rows(self) -> torch.Tensor:
        """First selectable task row per source clip."""
        return self._clip_start_rows

    @property
    def frame_counts(self) -> torch.Tensor:
        """Frames per clip."""
        return self._frame_counts

    @property
    def source_fps(self) -> torch.Tensor:
        """Source sample rate [Hz] per clip."""
        return self._source_fps

    @property
    def clip_ids(self) -> tuple[str, ...]:
        """Clip identifiers covered by selectable descriptor rows."""
        return self._clip_index.clip_ids

    @property
    def device(self) -> torch.device:
        """Shared trajectory and descriptor device."""
        return self._frames.device

    @property
    def num_tasks(self) -> int:
        """Number of selectable motion descriptors."""
        return self.clip_indices.shape[0]

    def field(self, name: str) -> torch.Tensor:
        """Return one concrete trajectory tensor without indirection or copying."""
        return self._frames.field(name)

    def interpolation(self, name: str) -> Interpolation:
        """Return the fixed temporal rule for one concrete trajectory field."""
        return self._frames.interpolation(name)

    def _validate_clip_indices(self, clip_indices: torch.Tensor) -> None:
        if clip_indices.ndim != 1 or clip_indices.dtype is not torch.int64 or clip_indices.device != self.device:
            raise ValueError("clip_indices must be a 1D int64 tensor on the table device.")
        torch._assert_async(
            torch.all((clip_indices >= 0) & (clip_indices < len(self._clip_index.clips))),
            "clip_indices are outside the motion table.",
        )

    def reference_view(self, clip_indices: torch.Tensor, time_seconds: torch.Tensor) -> ReferenceView:
        """Resolve clamped continuous-time reference interpolation."""
        self._validate_clip_indices(clip_indices)
        if (
            time_seconds.ndim != 1
            or not time_seconds.is_floating_point()
            or time_seconds.device != self.device
            or time_seconds.shape != clip_indices.shape
        ):
            raise ValueError("time_seconds must match clip_indices as a floating tensor on the table device.")
        torch._assert_async(torch.all(torch.isfinite(time_seconds)), "time_seconds must be finite.")

        frame_counts = self._frame_counts[clip_indices]
        position = time_seconds * self._source_fps[clip_indices]
        last = frame_counts - 1
        tail_valid = (position >= 0.0) & (position <= last)
        clamped = torch.minimum(position.clamp_min(0.0), last.to(position.dtype))
        local_frame0 = torch.floor(clamped).to(torch.int64)
        local_frame1 = torch.minimum(local_frame0 + 1, last)
        alpha = clamped - local_frame0.to(clamped.dtype)
        offset = self._clip_offsets[clip_indices]
        return self.ReferenceView(
            self,
            clip_indices,
            time_seconds,
            local_frame0,
            local_frame1,
            offset + local_frame0,
            offset + local_frame1,
            alpha,
            tail_valid,
        )

    def sample(
        self,
        mode: Literal["source_rows", "uniform_before_source_end"],
        step_seconds: float | None,
    ) -> SampledSequence:
        """Return clips on one source-row or uniform sample clock."""
        if mode == "source_rows":
            if step_seconds is not None:
                raise ValueError("Source-row sampling does not declare step_seconds.")
        elif mode == "uniform_before_source_end":
            if step_seconds is None or not math.isfinite(step_seconds) or step_seconds <= 0.0:
                raise ValueError("Uniform sampling requires finite positive step_seconds.")
        else:
            raise ValueError(f"Unsupported sampling mode: {mode!r}.")

        counts = tuple(
            clip.frame_count
            if mode == "source_rows"
            else math.ceil((clip.frame_count - 1) / clip.source_fps / step_seconds)
            for clip in self.clip_index.clips
        )
        if any(count < 1 for count in counts):
            raise ValueError("Every sampled clip must contain at least one sample before its source endpoint.")
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        clip_offsets = tuple(offsets)

        if mode == "source_rows":
            return self.SampledSequence(self, clip_offsets, mode, step_seconds, self.field)

        counts_tensor = torch.tensor(counts, dtype=torch.int64, device=self.device)
        clip_positions = torch.repeat_interleave(
            torch.arange(len(self.clip_index.clips), dtype=torch.int64, device=self.device),
            counts_tensor,
        )
        flat_indices = torch.arange(clip_offsets[-1], dtype=torch.int64, device=self.device)
        starts = torch.tensor(clip_offsets[:-1], dtype=torch.int64, device=self.device)
        local_samples = flat_indices - starts[clip_positions]
        clip_indices = clip_positions
        sample_times = local_samples * step_seconds
        reference = self.reference_view(clip_indices, sample_times)
        return self.SampledSequence(self, clip_offsets, mode, step_seconds, reference.field)


def build_motion_task_table(command_cfg: StateCommandCfg, scene_cfg: object, device: str) -> MotionTaskTable:
    """Stream the selected source split into a table from resolved configuration."""
    del scene_cfg
    table_cfg = command_cfg.task_table
    source_cfg = table_cfg.source
    split = source_cfg.train if table_cfg.motion_split == "train" else source_cfg.evaluation
    source = source_cfg.open_split(table_cfg.source_artifact_root, split)
    try:
        clip_index = source.inspect()
        if (
            clip_index.source_content_sha256 != split.source_content_sha256
            or len(clip_index.clips) != split.clip_count
            or clip_index.total_frames != split.frame_count
        ):
            raise ValueError(
                "Motion source identity/counts differ from the selected split: "
                f"hash={clip_index.source_content_sha256}, clips={len(clip_index.clips)}, "
                f"frames={clip_index.total_frames}."
            )
        source_skeleton = source_cfg.build_skeleton()
        target_kinematics = table_cfg.target_kinematics
        reference = target_kinematics.reference_kinematics_factory(table_cfg.reference_artifact_root, device)
        frame_builder = target_kinematics.frame_builder_factory(source_skeleton, reference)
        if not isinstance(frame_builder, MotionFrameBuilder):
            raise TypeError("frame_builder_factory must return a MotionFrameBuilder.")
        family_name = (
            table_cfg.route.exact_family if frame_builder.exact_coordinates else table_cfg.route.semantic_family
        )
        family = next((item for item in table_cfg.families if item.name == family_name), None)
        if family is None:
            raise ValueError(f"Motion coordinate route selected unknown family {family_name!r}.")
        family_identity_sha256 = canonical_sha256(family.to_dict())
        frames = (
            frame_builder.allocate(clip_index.total_frames, device=device) if frame_builder.exact_coordinates else None
        )
        rng = make_task_table_rng(table_cfg.seed, device)
        initial = _MotionCorpusCandidate(frame_builder, source, clip_index, device, frames)
        # Immutable corpus values and post-cut derivatives must not depend on TorchScript profiling/fusion.
        with torch.jit.optimized_execution(False):
            execution = execute_task_family(family, initial, None, rng)
            built = execution.candidates
            if not isinstance(built, _MotionCorpusCandidate) or execution.accepted_mask is None:
                raise TypeError("Motion family execution must return one accepted mask on its selected candidate axis.")
            if frame_builder.exact_coordinates:
                if built.frames is None:
                    raise ValueError("Exact Motion family did not materialize its preallocated frame bank.")
                quality_values = _exact_corpus_quality(built, execution.accepted_mask)
                clip_index, frames, quality_values = _compact_selected_exact_corpus(
                    clip_index,
                    built.frames,
                    quality_values,
                    execution.selected_indices,
                )
            else:
                if not isinstance(family.selection, MotionSemanticSegmentSelectionCfg):
                    raise TypeError("Semantic Motion family requires MotionSemanticSegmentSelectionCfg.")
                clip_index, frames, quality_values = _finalize_semantic_corpus(
                    built,
                    execution.accepted_mask,
                    execution.selected_indices,
                    family.selection,
                )
        view = _motion_task_view(clip_index, frames, frame_builder.joint_names, reference, quality_values)
        return MotionTaskTable(
            clip_index,
            frames,
            frame_builder.joint_names,
            frame_builder.reference_frame_names,
            frame_builder.version,
            frame_builder.construction_identity_sha256,
            table_cfg.task_row_mode,
            source_skeleton.identity_sha256,
            family_name,
            family_identity_sha256,
            view,
        )
    finally:
        source.close()
