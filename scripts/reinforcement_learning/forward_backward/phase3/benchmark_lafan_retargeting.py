# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare raw-LAFAN retargeting with the released G1 LAFAN trajectories.

The oracle pairs exact clip identifiers and source frames.  It removes one
constant yaw and translation per clip, but never fits scale, resamples time, or
compares target joint coordinates as if they were unique ground truth.
"""

from __future__ import annotations

import argparse
import builtins
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from benchmark_nonlinear_iterations import count_trajectory_nonlinear_iterations

from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_mul

_BVH_FRAME_TIME_QUANTIZATION_HALF_STEP_S = 0.5e-6
"""Half one microsecond, matching six-decimal BVH ``Frame Time`` serialization."""


if TYPE_CHECKING:
    from isaaclab_tasks.core.multi_task.kinematics import IKMemoryPlan, NewtonKinematics
    from isaaclab_tasks.core.multi_task.mdp.commands.state_command import TaskTableView
    from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex


@dataclass(frozen=True, slots=True)
class SemanticTrajectoryClip:
    """One target trajectory projected onto benchmark-owned semantic geometry."""

    clip_id: str
    frame_dt_s: float
    landmark_roles: tuple[str, ...]
    semantic_edges: tuple[tuple[str, str], ...]
    landmark_position_m: torch.Tensor
    rotation_roles: tuple[str, ...]
    rotation_xyzw: torch.Tensor
    support_roles: tuple[str, ...]
    support_position_m: torch.Tensor
    stance_active: torch.Tensor
    joint_names: tuple[str, ...]
    joint_position_rad: torch.Tensor
    joint_velocity_rad_s: torch.Tensor
    joint_lower_limit_rad: torch.Tensor
    joint_upper_limit_rad: torch.Tensor
    joint_velocity_limit_rad_s: torch.Tensor

    def __post_init__(self) -> None:
        """Require explicit, finite, frame-aligned benchmark tensors."""
        frame_count = self.landmark_position_m.shape[0] if self.landmark_position_m.ndim == 3 else -1
        landmark_count = len(self.landmark_roles)
        rotation_count = len(self.rotation_roles)
        support_count = len(self.support_roles)
        joint_count = len(self.joint_names)
        if not self.clip_id or not math.isfinite(self.frame_dt_s) or self.frame_dt_s <= 0.0:
            raise ValueError("Semantic clips require a nonempty identifier and positive frame period.")
        if (
            landmark_count < 2
            or len(set(self.landmark_roles)) != landmark_count
            or self.landmark_position_m.shape != (frame_count, landmark_count, 3)
            or rotation_count < 1
            or len(set(self.rotation_roles)) != rotation_count
            or self.rotation_xyzw.shape != (frame_count, rotation_count, 4)
            or support_count < 1
            or len(set(self.support_roles)) != support_count
            or self.support_position_m.shape != (frame_count, support_count, 3)
            or self.stance_active.shape != (frame_count, support_count)
            or self.stance_active.dtype is not torch.bool
            or joint_count < 1
            or len(set(self.joint_names)) != joint_count
            or self.joint_position_rad.shape != (frame_count, joint_count)
            or self.joint_velocity_rad_s.shape != (frame_count, joint_count)
            or self.joint_lower_limit_rad.shape != (joint_count,)
            or self.joint_upper_limit_rad.shape != (joint_count,)
            or self.joint_velocity_limit_rad_s.shape != (joint_count,)
        ):
            raise ValueError("Semantic clip tensors do not share their declared frame and feature axes.")
        if frame_count < 1:
            raise ValueError("Semantic clips require at least one frame.")
        floating = (
            self.landmark_position_m,
            self.rotation_xyzw,
            self.support_position_m,
            self.joint_position_rad,
            self.joint_velocity_rad_s,
            self.joint_lower_limit_rad,
            self.joint_upper_limit_rad,
            self.joint_velocity_limit_rad_s,
        )
        if any(not value.is_floating_point() for value in floating):
            raise ValueError("Semantic clip measurements and limits must use floating-point tensors.")
        measured = floating[:5]
        if any(not bool(torch.isfinite(value).all()) for value in measured):
            raise ValueError("Semantic clip measurements must be finite.")
        if (
            bool(torch.isnan(self.joint_lower_limit_rad).any())
            or bool(torch.isnan(self.joint_upper_limit_rad).any())
            or bool(torch.isnan(self.joint_velocity_limit_rad_s).any())
            or bool(torch.any(self.joint_lower_limit_rad > self.joint_upper_limit_rad))
            or bool(torch.any(self.joint_velocity_limit_rad_s <= 0.0))
        ):
            raise ValueError("Semantic clip hard limits are invalid.")
        role_set = set(self.landmark_roles)
        if (
            not self.semantic_edges
            or len(set(self.semantic_edges)) != len(self.semantic_edges)
            or any(
                parent not in role_set or child not in role_set or parent == child
                for parent, child in self.semantic_edges
            )
        ):
            raise ValueError("Semantic edges must be unique directed edges between declared landmark roles.")


@dataclass(frozen=True, slots=True)
class _SemanticCorpus:
    """Compact semantic and hard-limit inputs extracted from one inspection view."""

    landmark_position_m: torch.Tensor
    rotation_xyzw: torch.Tensor
    support_position_m: torch.Tensor
    joint_names: tuple[str, ...]
    joint_position_rad: torch.Tensor
    joint_velocity_rad_s: torch.Tensor


@dataclass(frozen=True, slots=True)
class _ClipMeasurements:
    """Unreduced values retained long enough for per-clip and corpus summaries."""

    alignment_yaw_rad: float
    alignment_translation_m: tuple[float, float, float]
    common_roles: tuple[str, ...]
    common_edges: tuple[tuple[str, str], ...]
    common_rotation_roles: tuple[str, ...]
    common_support_roles: tuple[str, ...]
    landmark_error_m: torch.Tensor
    landmark_by_role_m: dict[str, torch.Tensor]
    edge_angle_rad: torch.Tensor
    edge_by_role_rad: dict[str, torch.Tensor]
    orientation_error_rad: torch.Tensor
    orientation_by_role_rad: dict[str, torch.Tensor]
    stance_interval_count: int
    stance_anchor_error_m: torch.Tensor
    candidate_stance_slip_m: torch.Tensor
    reference_stance_slip_m: torch.Tensor
    candidate_penetration_m: torch.Tensor
    reference_penetration_m: torch.Tensor
    candidate_position_violation_rad: torch.Tensor
    candidate_velocity_violation_rad_s: torch.Tensor
    reference_position_violation_rad: torch.Tensor
    reference_velocity_violation_rad_s: torch.Tensor


def _cpu64(value: torch.Tensor) -> torch.Tensor:
    """Copy one measured tensor to stable CPU float64 arithmetic."""
    return value.detach().to(device="cpu", dtype=torch.float64)


def _statistics(values: torch.Tensor) -> dict[str, float | int | None]:
    """Summarize finite scalar samples without inventing values for an empty set."""
    values = _cpu64(values).reshape(-1)
    if values.numel() == 0:
        return {"count": 0, "mean": None, "rms": None, "p50": None, "p95": None, "max": None}
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Benchmark measurements must be finite.")
    quantiles = torch.quantile(values, torch.tensor((0.5, 0.95), dtype=torch.float64))
    return {
        "count": values.numel(),
        "mean": float(values.mean()),
        "rms": float(torch.sqrt(torch.mean(torch.square(values)))),
        "p50": float(quantiles[0]),
        "p95": float(quantiles[1]),
        "max": float(values.max()),
    }


def _build_performance(
    *,
    scope: str,
    included_stages: tuple[str, ...],
    wall_seconds: float,
    input_clip_count: int,
    input_frame_count: int,
    output_clip_count: int,
    output_frame_count: int,
    device: str | torch.device,
    cuda_allocated_bytes_before: int | None,
    cuda_peak_allocated_bytes: int | None,
    trajectory_nonlinear_iterations: dict[str, int],
) -> dict[str, object]:
    """Return one validated production-construction performance record."""
    counts = (input_clip_count, input_frame_count, output_clip_count, output_frame_count)
    if not scope or not included_stages or not math.isfinite(wall_seconds) or wall_seconds <= 0.0 or min(counts) < 1:
        raise ValueError("Build performance requires a named scope, positive duration, and nonempty input/output.")
    if (cuda_allocated_bytes_before is None) != (cuda_peak_allocated_bytes is None):
        raise ValueError("CUDA build memory requires both baseline and peak allocated bytes.")
    if cuda_allocated_bytes_before is not None and (
        cuda_allocated_bytes_before < 0 or cuda_peak_allocated_bytes < cuda_allocated_bytes_before
    ):
        raise ValueError("CUDA peak allocated bytes must be at least the nonnegative build baseline.")
    return {
        "scope": scope,
        "included_stages": list(included_stages),
        "device": str(device),
        "input_clip_count": input_clip_count,
        "input_frame_count": input_frame_count,
        "output_clip_count": output_clip_count,
        "output_frame_count": output_frame_count,
        "wall_seconds": wall_seconds,
        "input_frames_per_second": input_frame_count / wall_seconds,
        "output_frames_per_second": output_frame_count / wall_seconds,
        "cuda_allocated_bytes_before": cuda_allocated_bytes_before,
        "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
        "cuda_peak_incremental_allocated_bytes": (
            None if cuda_allocated_bytes_before is None else cuda_peak_allocated_bytes - cuda_allocated_bytes_before
        ),
        "trajectory_nonlinear_iterations": trajectory_nonlinear_iterations,
    }


def _fit_yaw_translation(candidate: torch.Tensor, reference: torch.Tensor) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Fit one least-squares yaw and translation, deliberately without scale."""
    candidate_flat = candidate.reshape(-1, 3)
    reference_flat = reference.reshape(-1, 3)
    candidate_center = candidate_flat.mean(dim=0)
    reference_center = reference_flat.mean(dim=0)
    centered_candidate = candidate_flat - candidate_center
    centered_reference = reference_flat - reference_center
    cosine_term = torch.sum(centered_candidate[:, 0] * centered_reference[:, 0]) + torch.sum(
        centered_candidate[:, 1] * centered_reference[:, 1]
    )
    sine_term = torch.sum(centered_candidate[:, 0] * centered_reference[:, 1]) - torch.sum(
        centered_candidate[:, 1] * centered_reference[:, 0]
    )
    yaw = 0.0 if float(torch.hypot(cosine_term, sine_term)) <= 1.0e-12 else float(torch.atan2(sine_term, cosine_term))
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = candidate.new_tensor(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
    translation = reference_center - candidate_center @ rotation.T
    return yaw, translation, candidate @ rotation.T + translation


def _apply_yaw_translation(points: torch.Tensor, yaw: float, translation: torch.Tensor) -> torch.Tensor:
    """Apply the already fitted rigid quotient to point rows."""
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = points.new_tensor(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
    return points @ rotation.T + translation


def _aligned_rotation(rotation_xyzw: torch.Tensor, yaw: float) -> torch.Tensor:
    """Apply a world-yaw change of basis to selected world orientations."""
    yaw_quaternion = rotation_xyzw.new_tensor((0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)))
    yaw_quaternion = yaw_quaternion.expand_as(rotation_xyzw)
    aligned = quat_mul(yaw_quaternion, rotation_xyzw)
    return aligned / torch.linalg.vector_norm(aligned, dim=-1, keepdim=True)


def _limit_violation(clip: SemanticTrajectoryClip) -> tuple[torch.Tensor, torch.Tensor]:
    """Return position and absolute-velocity hard-limit excesses."""
    position = _cpu64(clip.joint_position_rad)
    velocity = _cpu64(clip.joint_velocity_rad_s)
    lower = _cpu64(clip.joint_lower_limit_rad)
    upper = _cpu64(clip.joint_upper_limit_rad)
    speed_limit = _cpu64(clip.joint_velocity_limit_rad_s)
    position_violation = torch.maximum(lower - position, position - upper).clamp_min(0.0)
    velocity_violation = (velocity.abs() - speed_limit).clamp_min(0.0)
    return position_violation, velocity_violation


def _stance_values(
    candidate: torch.Tensor, reference: torch.Tensor, active: torch.Tensor
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Separate interval anchor disagreement from within-interval foot slip."""
    anchor_errors: list[torch.Tensor] = []
    candidate_slips: list[torch.Tensor] = []
    reference_slips: list[torch.Tensor] = []
    interval_count = 0
    for support in range(active.shape[1]):
        mask = active[:, support]
        starts = torch.nonzero(mask & ~torch.cat((mask.new_zeros(1), mask[:-1])), as_tuple=False).flatten()
        stops = torch.nonzero(mask & ~torch.cat((mask[1:], mask.new_zeros(1))), as_tuple=False).flatten() + 1
        for start_tensor, stop_tensor in zip(starts, stops, strict=True):
            start = int(start_tensor)
            stop = int(stop_tensor)
            candidate_anchor = candidate[start, support]
            reference_anchor = reference[start, support]
            anchor_errors.append(torch.linalg.vector_norm(candidate_anchor - reference_anchor).reshape(1))
            candidate_slips.append(torch.linalg.vector_norm(candidate[start:stop, support] - candidate_anchor, dim=-1))
            reference_slips.append(torch.linalg.vector_norm(reference[start:stop, support] - reference_anchor, dim=-1))
            interval_count += 1
    empty = candidate.new_empty(0)
    return (
        interval_count,
        torch.cat(anchor_errors) if anchor_errors else empty,
        torch.cat(candidate_slips) if candidate_slips else empty,
        torch.cat(reference_slips) if reference_slips else empty,
    )


def _measure_clip(candidate: SemanticTrajectoryClip, reference: SemanticTrajectoryClip) -> _ClipMeasurements:
    """Measure one exact-ID, exact-clock clip pair."""
    if candidate.clip_id != reference.clip_id:
        raise ValueError("Internal clip pairing must preserve exact clip identifiers.")
    if candidate.landmark_position_m.shape[0] != reference.landmark_position_m.shape[0]:
        raise ValueError(
            f"Clip {candidate.clip_id!r} has different candidate and released frame counts; time warp is forbidden."
        )
    if not math.isclose(
        candidate.frame_dt_s,
        reference.frame_dt_s,
        rel_tol=0.0,
        abs_tol=_BVH_FRAME_TIME_QUANTIZATION_HALF_STEP_S,
    ):
        raise ValueError(
            f"Clip {candidate.clip_id!r} has different candidate and released frame periods; resampling is forbidden."
        )

    candidate_role_row = {role: row for row, role in enumerate(candidate.landmark_roles)}
    reference_role_row = {role: row for row, role in enumerate(reference.landmark_roles)}
    common_roles = tuple(role for role in reference.landmark_roles if role in candidate_role_row)
    if len(common_roles) < 2:
        raise ValueError(f"Clip {candidate.clip_id!r} has fewer than two common semantic landmarks.")
    candidate_landmarks = _cpu64(candidate.landmark_position_m)[:, [candidate_role_row[role] for role in common_roles]]
    reference_landmarks = _cpu64(reference.landmark_position_m)[:, [reference_role_row[role] for role in common_roles]]
    yaw, translation, candidate_landmarks = _fit_yaw_translation(candidate_landmarks, reference_landmarks)
    landmark_error = torch.linalg.vector_norm(candidate_landmarks - reference_landmarks, dim=-1)
    landmark_by_role = {role: landmark_error[:, row] for row, role in enumerate(common_roles)}

    candidate_edges = set(candidate.semantic_edges)
    common_role_set = set(common_roles)
    common_edges = tuple(
        edge
        for edge in reference.semantic_edges
        if edge in candidate_edges and edge[0] in common_role_set and edge[1] in common_role_set
    )
    if not common_edges:
        raise ValueError(f"Clip {candidate.clip_id!r} has no common semantic edges.")
    edge_angles = []
    edge_by_role = {}
    for parent, child in common_edges:
        parent_row = common_roles.index(parent)
        child_row = common_roles.index(child)
        candidate_edge = candidate_landmarks[:, child_row] - candidate_landmarks[:, parent_row]
        reference_edge = reference_landmarks[:, child_row] - reference_landmarks[:, parent_row]
        if bool(
            torch.any(torch.linalg.vector_norm(candidate_edge, dim=-1) <= 1.0e-9)
            or torch.any(torch.linalg.vector_norm(reference_edge, dim=-1) <= 1.0e-9)
        ):
            raise ValueError(f"Clip {candidate.clip_id!r} contains a zero-length common semantic edge.")
        angle = torch.atan2(
            torch.linalg.vector_norm(torch.linalg.cross(candidate_edge, reference_edge, dim=-1), dim=-1),
            torch.sum(candidate_edge * reference_edge, dim=-1),
        )
        edge_angles.append(angle)
        edge_by_role[f"{parent}->{child}"] = angle
    edge_angle = torch.stack(edge_angles, dim=1)

    candidate_rotation_row = {role: row for row, role in enumerate(candidate.rotation_roles)}
    reference_rotation_row = {role: row for row, role in enumerate(reference.rotation_roles)}
    if set(candidate_rotation_row) != set(reference_rotation_row):
        raise ValueError(f"Clip {candidate.clip_id!r} does not share the selected orientation roles.")
    common_rotation_roles = reference.rotation_roles
    candidate_rotation = _cpu64(candidate.rotation_xyzw)[
        :, [candidate_rotation_row[role] for role in common_rotation_roles]
    ]
    candidate_rotation = _aligned_rotation(candidate_rotation, yaw)
    reference_rotation = _cpu64(reference.rotation_xyzw)[
        :, [reference_rotation_row[role] for role in common_rotation_roles]
    ]
    reference_rotation = reference_rotation / torch.linalg.vector_norm(reference_rotation, dim=-1, keepdim=True)
    orientation_error = quat_error_magnitude(candidate_rotation, reference_rotation)
    orientation_by_role = {role: orientation_error[:, row] for row, role in enumerate(common_rotation_roles)}

    candidate_support_row = {role: row for row, role in enumerate(candidate.support_roles)}
    reference_support_row = {role: row for row, role in enumerate(reference.support_roles)}
    common_support_roles = tuple(role for role in reference.support_roles if role in candidate_support_row)
    if not common_support_roles:
        raise ValueError(f"Clip {candidate.clip_id!r} has no common semantic support points.")
    candidate_support = _cpu64(candidate.support_position_m)[
        :, [candidate_support_row[role] for role in common_support_roles]
    ]
    candidate_support = _apply_yaw_translation(candidate_support, yaw, translation)
    reference_support = _cpu64(reference.support_position_m)[
        :, [reference_support_row[role] for role in common_support_roles]
    ]
    candidate_active = candidate.stance_active.detach().to("cpu")[
        :, [candidate_support_row[role] for role in common_support_roles]
    ]
    reference_active = reference.stance_active.detach().to("cpu")[
        :, [reference_support_row[role] for role in common_support_roles]
    ]
    if not torch.equal(candidate_active, reference_active):
        raise ValueError(f"Clip {candidate.clip_id!r} does not share one explicit stance mask across the comparison.")
    interval_count, anchor_error, candidate_slip, reference_slip = _stance_values(
        candidate_support, reference_support, reference_active
    )
    candidate_penetration = (-candidate_support[..., 2]).clamp_min(0.0).amax(dim=1)
    reference_penetration = (-reference_support[..., 2]).clamp_min(0.0).amax(dim=1)
    candidate_position_violation, candidate_velocity_violation = _limit_violation(candidate)
    reference_position_violation, reference_velocity_violation = _limit_violation(reference)
    return _ClipMeasurements(
        alignment_yaw_rad=yaw,
        alignment_translation_m=tuple(float(value) for value in translation),
        common_roles=common_roles,
        common_edges=common_edges,
        common_rotation_roles=common_rotation_roles,
        common_support_roles=common_support_roles,
        landmark_error_m=landmark_error,
        landmark_by_role_m=landmark_by_role,
        edge_angle_rad=edge_angle,
        edge_by_role_rad=edge_by_role,
        orientation_error_rad=orientation_error,
        orientation_by_role_rad=orientation_by_role,
        stance_interval_count=interval_count,
        stance_anchor_error_m=anchor_error,
        candidate_stance_slip_m=candidate_slip,
        reference_stance_slip_m=reference_slip,
        candidate_penetration_m=candidate_penetration,
        reference_penetration_m=reference_penetration,
        candidate_position_violation_rad=candidate_position_violation,
        candidate_velocity_violation_rad_s=candidate_velocity_violation,
        reference_position_violation_rad=reference_position_violation,
        reference_velocity_violation_rad_s=reference_velocity_violation,
    )


def _hard_limit_report(position: torch.Tensor, velocity: torch.Tensor) -> dict[str, object]:
    """Report joint-sample violations and per-frame maxima separately by limit kind."""
    position_count = int(torch.count_nonzero(position > 0.0))
    velocity_count = int(torch.count_nonzero(velocity > 0.0))
    return {
        "position": {
            "excess_rad": _statistics(position),
            "frame_max_excess_rad": _statistics(position.amax(dim=1)),
            "violating_samples": position_count,
            "violating_fraction": position_count / position.numel(),
        },
        "velocity": {
            "excess_rad_s": _statistics(velocity),
            "frame_max_excess_rad_s": _statistics(velocity.amax(dim=1)),
            "violating_samples": velocity_count,
            "violating_fraction": velocity_count / velocity.numel(),
        },
    }


def _measurement_report(values: _ClipMeasurements) -> dict[str, object]:
    """Reduce one clip's metric families without combining their meanings."""
    return {
        "alignment": {
            "yaw_rad": values.alignment_yaw_rad,
            "translation_m": list(values.alignment_translation_m),
        },
        "common_semantics": {
            "landmark_roles": list(values.common_roles),
            "edges": [list(edge) for edge in values.common_edges],
            "rotation_roles": list(values.common_rotation_roles),
            "support_roles": list(values.common_support_roles),
        },
        "landmark_error_m": {
            "all": _statistics(values.landmark_error_m),
            "by_role": {role: _statistics(error) for role, error in values.landmark_by_role_m.items()},
        },
        "semantic_edge_angle_rad": {
            "all": _statistics(values.edge_angle_rad),
            "by_edge": {edge: _statistics(error) for edge, error in values.edge_by_role_rad.items()},
        },
        "selected_orientation_error_rad": {
            "all": _statistics(values.orientation_error_rad),
            "by_role": {role: _statistics(error) for role, error in values.orientation_by_role_rad.items()},
        },
        "stance": {
            "interval_count": values.stance_interval_count,
            "anchor_error_m": _statistics(values.stance_anchor_error_m),
            "candidate_slip_m": _statistics(values.candidate_stance_slip_m),
            "released_reference_slip_m": _statistics(values.reference_stance_slip_m),
        },
        "frame_max_support_penetration_m": {
            "candidate": _statistics(values.candidate_penetration_m),
            "released_reference": _statistics(values.reference_penetration_m),
        },
        "hard_limits": {
            "candidate": _hard_limit_report(
                values.candidate_position_violation_rad, values.candidate_velocity_violation_rad_s
            ),
            "released_reference": _hard_limit_report(
                values.reference_position_violation_rad, values.reference_velocity_violation_rad_s
            ),
        },
    }


def _concatenate(values: list[_ClipMeasurements], field: str) -> torch.Tensor:
    """Concatenate one tensor metric across exact clip boundaries."""
    return torch.cat([getattr(value, field).reshape(-1) for value in values])


def _aggregate_report(values: list[_ClipMeasurements]) -> dict[str, object]:
    """Aggregate samples across clips while preserving metric-family boundaries."""
    candidate_position = torch.cat([value.candidate_position_violation_rad for value in values])
    candidate_velocity = torch.cat([value.candidate_velocity_violation_rad_s for value in values])
    reference_position = torch.cat([value.reference_position_violation_rad for value in values])
    reference_velocity = torch.cat([value.reference_velocity_violation_rad_s for value in values])
    rotation_roles = values[0].common_rotation_roles
    if any(value.common_rotation_roles != rotation_roles for value in values[1:]):
        raise ValueError("Selected orientation roles must remain identical across paired clips.")
    orientation_by_role = {
        role: _statistics(torch.cat([value.orientation_by_role_rad[role] for value in values]))
        for role in rotation_roles
    }
    return {
        "landmark_error_m": _statistics(_concatenate(values, "landmark_error_m")),
        "semantic_edge_angle_rad": _statistics(_concatenate(values, "edge_angle_rad")),
        "selected_orientation_error_rad": {
            "all": _statistics(_concatenate(values, "orientation_error_rad")),
            "by_role": orientation_by_role,
        },
        "stance": {
            "interval_count": sum(value.stance_interval_count for value in values),
            "anchor_error_m": _statistics(_concatenate(values, "stance_anchor_error_m")),
            "candidate_slip_m": _statistics(_concatenate(values, "candidate_stance_slip_m")),
            "released_reference_slip_m": _statistics(_concatenate(values, "reference_stance_slip_m")),
        },
        "frame_max_support_penetration_m": {
            "candidate": _statistics(_concatenate(values, "candidate_penetration_m")),
            "released_reference": _statistics(_concatenate(values, "reference_penetration_m")),
        },
        "hard_limits": {
            "candidate": _hard_limit_report(candidate_position, candidate_velocity),
            "released_reference": _hard_limit_report(reference_position, reference_velocity),
        },
    }


def compare_lafan_retargeting(
    candidate_clips: tuple[SemanticTrajectoryClip, ...],
    released_clips: tuple[SemanticTrajectoryClip, ...],
) -> dict[str, object]:
    """Compare exact clip IDs and frames under only a per-clip yaw/translation quotient."""
    candidate_by_id = {clip.clip_id: clip for clip in candidate_clips}
    released_by_id = {clip.clip_id: clip for clip in released_clips}
    if len(candidate_by_id) != len(candidate_clips) or len(released_by_id) != len(released_clips):
        raise ValueError("Candidate and released corpora require unique clip identifiers.")
    missing_candidate = sorted(set(released_by_id) - set(candidate_by_id))
    missing_released = sorted(set(candidate_by_id) - set(released_by_id))
    if missing_candidate or missing_released:
        raise ValueError(
            "Candidate and released corpora must have identical clip IDs; "
            f"missing candidate={missing_candidate}, missing released={missing_released}."
        )
    if not released_clips:
        raise ValueError("The LAFAN comparison requires at least one paired clip.")
    measurements = [_measure_clip(candidate_by_id[released.clip_id], released) for released in released_clips]
    frame_count = sum(clip.landmark_position_m.shape[0] for clip in released_clips)
    return {
        "comparison_policy": {
            "pairing": "exact_clip_id",
            "time_correspondence": "exact_source_frame_index_with_declared_period_precision",
            "frame_period_absolute_tolerance_s": _BVH_FRAME_TIME_QUANTIZATION_HALF_STEP_S,
            "frame_period_equivalence": "half_the_six_decimal_bvh_frame_time_serialization_step",
            "spatial_quotient": "one_constant_world_z_yaw_and_xyz_translation_per_clip",
            "spatial_fit": "least_squares_over_all_common_landmark_rows_without_scale",
            "scale_fitted": False,
            "time_warped_or_resampled": False,
            "geometry": "named_landmark_edge_support_intersection_plus_identical_selected_orientation_roles",
            "stance_mask": "candidate_source_contact_evidence_shared_by_pair",
            "stance_anchor": "first_active_support_point_per_contiguous_interval",
            "stance_slip": "3d_displacement_from_its_own_interval_anchor",
            "penetration_scope": "common_semantic_support_points_against_world_z_zero",
        },
        "clip_count": len(released_clips),
        "frame_count": frame_count,
        "aggregate": _aggregate_report(measurements),
        "clips": [
            {
                "clip_id": released.clip_id,
                "frame_count": released.landmark_position_m.shape[0],
                "candidate_frame_dt_s": candidate_by_id[released.clip_id].frame_dt_s,
                "released_frame_dt_s": released.frame_dt_s,
                **_measurement_report(values),
            }
            for released, values in zip(released_clips, measurements, strict=True)
        ],
    }


def _inspect_source_index(cfg: object) -> MotionClipIndex:
    """Inspect one configured source without retaining its decoded runtime state."""
    table_cfg = cfg.commands.motion.task_table
    source_cfg = table_cfg.source
    split = source_cfg.train if table_cfg.motion_split == "train" else source_cfg.evaluation
    source = source_cfg.open_split(table_cfg.source_artifact_root, split)
    try:
        return source.inspect()
    finally:
        source.close()


def _paired_prefix_indices(
    raw_index: MotionClipIndex,
    released_index: MotionClipIndex,
    max_clips: int | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Map one deterministic raw-source prefix to the same released clip IDs."""
    raw_ids = tuple(raw_index.clip_ids)
    released_ids = tuple(released_index.clip_ids)
    if len(set(raw_ids)) != len(raw_ids) or len(set(released_ids)) != len(released_ids):
        raise ValueError("Raw and released source inspections require unique clip IDs.")
    raw_id_set = set(raw_ids)
    released_id_set = set(released_ids)
    if raw_id_set != released_id_set:
        raise ValueError(
            "Raw and released source inspections must expose identical clip IDs; "
            f"missing raw={sorted(released_id_set - raw_id_set)}, "
            f"missing released={sorted(raw_id_set - released_id_set)}."
        )
    if max_clips is not None and (type(max_clips) is not int or max_clips < 1):
        raise ValueError("max_clips must be a positive integer when provided.")
    selected_count = len(raw_ids) if max_clips is None else max_clips
    if selected_count > len(raw_ids):
        raise ValueError(f"Requested {selected_count} clips but the evaluation split has {len(raw_ids)}.")

    raw_rows = tuple(range(selected_count))
    released_row_by_id = {clip_id: row for row, clip_id in enumerate(released_ids)}
    released_rows = tuple(released_row_by_id[raw_ids[row]] for row in raw_rows)
    for raw_row, released_row in zip(raw_rows, released_rows, strict=True):
        raw_clip = raw_index.clips[raw_row]
        released_clip = released_index.clips[released_row]
        if raw_clip.frame_count != released_clip.frame_count or not math.isclose(
            1.0 / raw_clip.source_fps,
            1.0 / released_clip.source_fps,
            rel_tol=0.0,
            abs_tol=_BVH_FRAME_TIME_QUANTIZATION_HALF_STEP_S,
        ):
            raise ValueError(f"Clip {raw_clip.clip_id!r} does not have exact released frame/time correspondence.")
    return raw_rows, released_rows


def _validate_inspection_index(view: TaskTableView, index: MotionClipIndex) -> None:
    """Require the inspection view to preserve its requested source prefix and clocks."""
    offsets = tuple(int(value) for value in view.sequences.offsets.detach().cpu())
    sequence_count = len(offsets) - 1
    if sequence_count > len(index.clips) or offsets != index.offsets[: sequence_count + 1]:
        raise ValueError("Inspection-view sequence offsets differ from the inspected source clips.")
    if view.sequences.frame_dt is None:
        raise ValueError("LAFAN inspection views require physical frame periods.")
    frame_dt = tuple(float(value) for value in view.sequences.frame_dt.detach().cpu())
    expected = tuple(1.0 / clip.source_fps for clip in index.clips[:sequence_count])
    if len(frame_dt) != len(expected) or any(
        not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1.0e-7)
        for actual, wanted in zip(frame_dt, expected, strict=True)
    ):
        raise ValueError("Inspection-view frame periods differ from the inspected source clips.")


def _inspection_corpus(
    view: TaskTableView,
    index: MotionClipIndex,
    sequence_indices: tuple[int, ...],
    reference: NewtonKinematics,
    trajectory: object,
) -> tuple[_SemanticCorpus, IKMemoryPlan, int | None]:
    """Run memory-planned shared-model FK over selected accepted or rejected inspection rows."""
    from isaaclab_tasks.core.multi_task.kinematics import plan_ik_memory

    _validate_inspection_index(view, index)
    view_sequence_count = view.sequences.offsets.numel() - 1
    if (
        not sequence_indices
        or len(set(sequence_indices)) != len(sequence_indices)
        or any(sequence < 0 or sequence >= view_sequence_count for sequence in sequence_indices)
    ):
        raise ValueError("Inspection corpus selection requires unique in-range sequence indices.")
    device = view.kinematic_view.joint_q_default.device
    if (
        str(device) != reference.device
        or view.kinematic_view.joint_q_default.numel() != reference.model.joint_coord_count
    ):
        raise ValueError("Inspection q mapping differs from the benchmark target mechanics.")
    layout = view.state_bank.layout
    if layout.names != ("robot",) or len(layout.joint_names) != 1:
        raise ValueError("LAFAN inspection requires one robot root and one robot joint axis.")

    flat_indices = torch.cat(
        tuple(
            torch.arange(index.offsets[sequence], index.offsets[sequence + 1], dtype=torch.int64, device=device)
            for sequence in sequence_indices
        )
    )
    state_rows = (
        flat_indices
        if view.sequences.state_indices is None
        else view.sequences.state_indices.index_select(0, flat_indices)
    )
    frame_count = state_rows.numel()
    landmark_rows = trajectory.position_body_index_tensor
    root_body_index = trajectory.root_body_index
    support_rows = trajectory.support_body_indices
    support_offsets = trajectory.support_point_body_m
    if (
        landmark_rows.device != device
        or type(root_body_index) is not int
        or root_body_index < 0
        or root_body_index >= reference.model.body_count
        or support_rows.device != device
        or support_offsets.device != device
        or support_offsets.shape != (support_rows.numel(), 3)
    ):
        raise ValueError("Target semantic geometry and inspection view must share one device.")

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
        memory_before = torch.cuda.memory_allocated(torch_device)
        torch.cuda.reset_peak_memory_stats(torch_device)
    else:
        memory_before = None

    landmark_count = landmark_rows.numel()
    support_count = support_rows.numel()
    body_count = reference.model.body_count
    landmark_position = torch.empty((frame_count, landmark_count, 3), dtype=torch.float32, device=device)
    rotation = torch.empty((frame_count, 1, 4), dtype=torch.float32, device=device)
    support_position = torch.empty((frame_count, support_count, 3), dtype=torch.float32, device=device)
    joint_position = view.state_bank.joint_position.index_select(0, state_rows)
    joint_velocity = view.state_bank.joint_velocity.index_select(0, state_rows)
    bytes_per_row = 4 * (
        reference.model.joint_coord_count
        + reference.model.joint_dof_count
        + 13 * body_count
        + 7 * landmark_count
        + 10 * support_count
    )
    memory_plan = plan_ik_memory(frame_count, device, lambda capacity: capacity * bytes_per_row)
    capacity = memory_plan.batch_capacity
    joint_q = torch.empty((capacity, reference.model.joint_coord_count), dtype=torch.float32, device=device)
    joint_qd = torch.zeros((capacity, reference.model.joint_dof_count), dtype=torch.float32, device=device)
    body_q = torch.empty((capacity, body_count, 7), dtype=torch.float32, device=device)
    body_qd = torch.empty((capacity, body_count, 6), dtype=torch.float32, device=device)
    landmark_pose = torch.empty((capacity, landmark_count, 7), dtype=torch.float32, device=device)
    support_pose = torch.empty((capacity, support_count, 7), dtype=torch.float32, device=device)
    for start in range(0, frame_count, capacity):
        stop = min(start + capacity, frame_count)
        batch_size = stop - start
        rows = state_rows[start:stop]
        active_joint_q = joint_q[:batch_size]
        active_body_q = body_q[:batch_size]
        view.kinematic_view.joint_q_into(view.state_bank, rows, active_joint_q)
        reference.eval_fk_batched_torch(active_joint_q, joint_qd[:batch_size], active_body_q, body_qd[:batch_size])
        torch.index_select(active_body_q, 1, landmark_rows, out=landmark_pose[:batch_size])
        landmark_position[start:stop].copy_(landmark_pose[:batch_size, :, :3])
        rotation[start:stop, 0].copy_(active_body_q[:, root_body_index, 3:])
        torch.index_select(active_body_q, 1, support_rows, out=support_pose[:batch_size])
        active_support_pose = support_pose[:batch_size]
        support_position[start:stop].copy_(
            active_support_pose[..., :3]
            + quat_apply(
                active_support_pose[..., 3:],
                support_offsets.unsqueeze(0).expand(batch_size, -1, -1),
            )
        )

    if memory_before is None:
        measured_peak_bytes = None
    else:
        torch.cuda.synchronize(torch_device)
        measured_peak_bytes = torch.cuda.max_memory_allocated(torch_device) - memory_before
    return (
        _SemanticCorpus(
            landmark_position_m=landmark_position,
            rotation_xyzw=rotation,
            support_position_m=support_position,
            joint_names=layout.joint_names[0],
            joint_position_rad=joint_position,
            joint_velocity_rad_s=joint_velocity,
        ),
        memory_plan,
        measured_peak_bytes,
    )


def _compact_index(index: MotionClipIndex, rows: tuple[int, ...]) -> MotionClipIndex:
    """Return exact selected clips with dense first-occurrence skeleton IDs."""
    from isaaclab_tasks.core.multi_task.motion.data import MotionClipIndex

    skeleton_rows: dict[int, int] = {}
    skeleton_identities = []
    clips = []
    for row in rows:
        clip = index.clips[row]
        skeleton_row = skeleton_rows.get(clip.skeleton_id)
        if skeleton_row is None:
            skeleton_row = len(skeleton_rows)
            skeleton_rows[clip.skeleton_id] = skeleton_row
            skeleton_identities.append(index.skeleton_identity_sha256s[clip.skeleton_id])
        clips.append(replace(clip, skeleton_id=skeleton_row))
    return MotionClipIndex(
        source_content_sha256=index.source_content_sha256,
        skeleton_identity_sha256s=tuple(skeleton_identities),
        clips=tuple(clips),
    )


def _inspection_joint_q(
    view: TaskTableView,
    index: MotionClipIndex,
    sequence_indices: tuple[int, ...],
    reference: NewtonKinematics,
) -> torch.Tensor:
    """Gather selected inspection coordinates into contiguous Newton order."""
    device = view.kinematic_view.joint_q_default.device
    flat_indices = torch.cat(
        tuple(
            torch.arange(index.offsets[row], index.offsets[row + 1], dtype=torch.int64, device=device)
            for row in sequence_indices
        )
    )
    state_rows = (
        flat_indices
        if view.sequences.state_indices is None
        else view.sequences.state_indices.index_select(0, flat_indices)
    )
    joint_q = torch.empty(
        (state_rows.numel(), reference.model.joint_coord_count),
        dtype=torch.float32,
        device=device,
    )
    view.kinematic_view.joint_q_into(view.state_bank, state_rows, joint_q)
    return joint_q


def _selected_source_targets(
    cfg: object,
    target: object,
    index: MotionClipIndex,
    rows: tuple[int, ...],
    device: str,
) -> object:
    """Generate one immutable raw-LAFAN semantic/contact target context."""
    from isaaclab_tasks.core.multi_task.motion.data import MotionSourceProjectionTrajectory
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import _motion_source_evidence
    from isaaclab_tasks.core.multi_task.motion.retarget import motion_contact_probe_offsets

    table_cfg = cfg.commands.motion.task_table
    source_cfg = table_cfg.source
    split = source_cfg.train if table_cfg.motion_split == "train" else source_cfg.evaluation
    source = source_cfg.open_split(table_cfg.source_artifact_root, split)
    try:
        skeleton_ids = {index.clips[row].skeleton_id for row in rows}
        if len(skeleton_ids) != 1:
            raise ValueError("The LAFAN paired certificate requires one shared raw source skeleton.")
        skeleton = source.skeleton(next(iter(skeleton_ids)))
        projection = table_cfg.target_kinematics.source_projection_factory(
            skeleton,
            target,
            source,
            table_cfg.contact_channels,
            motion_contact_probe_offsets(table_cfg.contact_channels, device),
        )
        if not isinstance(projection, MotionSourceProjectionTrajectory):
            raise TypeError("Raw LAFAN must select the trajectory projection for the paired certificate.")
        selected = tuple(_motion_source_evidence(projection, clip, device) for _, clip in source.clips(rows))
    finally:
        source.close()
    if len(selected) != len(rows):
        raise ValueError("Raw LAFAN target generation did not preserve the selected clip count.")
    first = selected[0]
    return replace(
        first,
        source_landmark_position_m=torch.cat(tuple(item.source_landmark_position_m for item in selected), dim=1),
        source_landmark_rotation_xyzw=torch.cat(tuple(item.source_landmark_rotation_xyzw for item in selected), dim=1),
        source_direction_point_position_m=torch.cat(
            tuple(item.source_direction_point_position_m for item in selected), dim=1
        ),
        initial_joint_q=torch.cat(tuple(item.initial_joint_q for item in selected)),
        source_contact_probe_position_m=torch.cat(
            tuple(item.source_contact_probe_position_m for item in selected), dim=1
        ),
        target_support_position_m=torch.cat(tuple(item.target_support_position_m for item in selected), dim=1),
    )


def _selected_released_joint_q(
    cfg: object,
    target: object,
    index: MotionClipIndex,
    rows: tuple[int, ...],
    device: str,
) -> torch.Tensor:
    """Decode exact released BFM G1 rows directly into Newton coordinates."""
    from isaaclab_tasks.core.multi_task.motion.data import MotionSourceProjectionExact
    from isaaclab_tasks.core.multi_task.motion.retarget import motion_contact_probe_offsets

    table_cfg = cfg.commands.motion.task_table
    source_cfg = table_cfg.source
    split = source_cfg.train if table_cfg.motion_split == "train" else source_cfg.evaluation
    source = source_cfg.open_split(table_cfg.source_artifact_root, split)
    values = []
    try:
        projections = {}
        for row, clip in source.clips(rows):
            skeleton_id = index.clips[row].skeleton_id
            projection = projections.get(skeleton_id)
            if projection is None:
                skeleton = source.skeleton(skeleton_id)
                projection = table_cfg.target_kinematics.source_projection_factory(
                    skeleton,
                    target,
                    source,
                    table_cfg.contact_channels,
                    motion_contact_probe_offsets(table_cfg.contact_channels, device),
                )
                if not isinstance(projection, MotionSourceProjectionExact):
                    raise TypeError("Released BFM G1 must select the exact coordinate projection.")
                projections[skeleton_id] = projection
            joint_q, joint_qd = clip.free_root_coordinates(projection.source_skeleton, device=device)
            coordinates = projection.convert_coordinates(joint_q, joint_qd, clip.source_fps)
            newton_q = torch.empty_like(coordinates.joint_q)
            target.write_joint_position_newton(coordinates, newton_q)
            values.append(newton_q)
    finally:
        source.close()
    if len(values) != len(rows):
        raise ValueError("Released BFM decoding did not preserve the selected clip count.")
    return torch.cat(tuple(values))


def _newton_corpus(
    target: object,
    index: MotionClipIndex,
    joint_q: torch.Tensor,
) -> tuple[_SemanticCorpus, IKMemoryPlan, int | None]:
    """Extract benchmark semantic geometry from explicit Newton coordinates."""
    from isaaclab_tasks.core.multi_task.kinematics import plan_ik_memory
    from isaaclab_tasks.core.multi_task.motion.robots.target import write_velocity_canonical

    reference = target.kinematics
    trajectory = target.trajectory_target
    device = joint_q.device
    frame_count = index.total_frames
    if joint_q.shape != (frame_count, reference.model.joint_coord_count):
        raise ValueError("Explicit Newton coordinates differ from the selected released clip index.")
    clip_offsets = torch.tensor(index.offsets, dtype=torch.int32, device=device)
    step_seconds = torch.tensor([1.0 / clip.source_fps for clip in index.clips], dtype=torch.float32, device=device)
    joint_qd = torch.empty((frame_count, reference.model.joint_dof_count), dtype=torch.float32, device=device)
    write_velocity_canonical(target, joint_q, clip_offsets, step_seconds, joint_qd)
    coordinates = target.coordinates_from_newton(joint_q, index)
    frames = target.materialize_coordinates(coordinates, index)
    landmark_rows = trajectory.position_body_index_tensor
    support_rows = trajectory.support_body_indices
    support_offsets = trajectory.support_point_body_m
    landmark_count = landmark_rows.numel()
    support_count = support_rows.numel()
    body_count = reference.model.body_count
    landmark_position = torch.empty((frame_count, landmark_count, 3), dtype=torch.float32, device=device)
    rotation = torch.empty((frame_count, 1, 4), dtype=torch.float32, device=device)
    support_position = torch.empty((frame_count, support_count, 3), dtype=torch.float32, device=device)
    bytes_per_row = 4 * (reference.model.joint_dof_count + 13 * body_count + 7 * landmark_count + 10 * support_count)
    memory_plan = plan_ik_memory(frame_count, device, lambda capacity: capacity * bytes_per_row)
    capacity = memory_plan.batch_capacity
    body_q = torch.empty((capacity, body_count, 7), dtype=torch.float32, device=device)
    body_qd = torch.empty((capacity, body_count, 6), dtype=torch.float32, device=device)
    landmark_pose = torch.empty((capacity, landmark_count, 7), dtype=torch.float32, device=device)
    support_pose = torch.empty((capacity, support_count, 7), dtype=torch.float32, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        memory_before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        memory_before = None
    for start in range(0, frame_count, capacity):
        stop = min(start + capacity, frame_count)
        count = stop - start
        reference.eval_fk_batched_torch(
            joint_q[start:stop],
            joint_qd[start:stop],
            body_q[:count],
            body_qd[:count],
        )
        torch.index_select(body_q[:count], 1, landmark_rows, out=landmark_pose[:count])
        landmark_position[start:stop].copy_(landmark_pose[:count, :, :3])
        rotation[start:stop, 0].copy_(body_q[:count, trajectory.root_body_index, 3:])
        torch.index_select(body_q[:count], 1, support_rows, out=support_pose[:count])
        support_position[start:stop].copy_(
            support_pose[:count, ..., :3]
            + quat_apply(
                support_pose[:count, ..., 3:],
                support_offsets.unsqueeze(0).expand(count, -1, -1),
            )
        )
    if memory_before is None:
        measured_peak_bytes = None
    else:
        torch.cuda.synchronize(device)
        measured_peak_bytes = torch.cuda.max_memory_allocated(device) - memory_before
    return (
        _SemanticCorpus(
            landmark_position_m=landmark_position,
            rotation_xyzw=rotation,
            support_position_m=support_position,
            joint_names=target.joint_names,
            joint_position_rad=frames.field("joint_position"),
            joint_velocity_rad_s=frames.field("joint_velocity"),
        ),
        memory_plan,
        measured_peak_bytes,
    )


def _raw_contact_context(
    solve_cfg: object,
    targets: object,
    index: MotionClipIndex,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Infer one immutable raw-source contact context shared by both outputs."""
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import (
        _motion_infer_contact_evidence,
    )

    device = targets.initial_joint_q.device
    frame_count = index.total_frames
    probe_count = targets.source_contact_probe_position_m.shape[0]
    channel_count = targets.contact_channel_probe_offsets.numel() - 1
    clip_offsets = torch.tensor(index.offsets, dtype=torch.int32, device=device)
    step_seconds = torch.tensor([1.0 / clip.source_fps for clip in index.clips], dtype=torch.float32, device=device)
    source_plane_height_m = torch.empty(len(index.clips), dtype=torch.float32, device=device)
    source_probe_active = torch.empty((frame_count, probe_count), dtype=torch.uint8, device=device)
    source_probe_stable = torch.empty_like(source_probe_active)
    confidence = torch.empty((frame_count, channel_count), dtype=torch.float32, device=device)
    stable = torch.empty((frame_count, channel_count), dtype=torch.uint8, device=device)
    edge_stable = torch.empty_like(stable)
    _motion_infer_contact_evidence(
        solve_cfg.contact,
        targets,
        clip_offsets,
        step_seconds,
        source_plane_height_m,
        source_probe_active,
        source_probe_stable,
        confidence,
        stable,
        edge_stable,
    )
    obstacle_pose = torch.zeros((frame_count, 7), dtype=torch.float32, device=device)
    obstacle_pose[:, 6] = 1.0
    return clip_offsets, step_seconds, confidence, stable, edge_stable, obstacle_pose


def _shared_output_certificate(
    target: object,
    index: MotionClipIndex,
    targets: object,
    joint_q: torch.Tensor,
    contact_context: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the production semantic/contact and target-coordinate certificates unchanged."""
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import _TRAJECTORY_METRIC_NAMES
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table_builder import (
        _certify_target_coordinates,
    )
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_trajectory import _trajectory_clip_quality
    from isaaclab_tasks.core.multi_task.motion.robots.target import write_velocity_canonical

    clip_offsets, step_seconds, confidence, stable, edge_stable, obstacle_pose = contact_context
    frame_count = index.total_frames
    device = joint_q.device
    joint_qd = torch.empty((frame_count, target.kinematics.model.joint_dof_count), dtype=torch.float32, device=device)
    body_q = torch.empty((frame_count, target.kinematics.model.body_count, 7), dtype=torch.float32, device=device)
    body_qd = torch.empty((frame_count, target.kinematics.model.body_count, 6), dtype=torch.float32, device=device)
    write_velocity_canonical(target, joint_q, clip_offsets, step_seconds, joint_qd)
    target.kinematics.eval_fk_batched_torch(joint_q, joint_qd, body_q, body_qd)
    frame_quality = torch.empty((frame_count, len(_TRAJECTORY_METRIC_NAMES)), dtype=torch.float32, device=device)
    quality = torch.empty((len(index.clips), len(_TRAJECTORY_METRIC_NAMES)), dtype=torch.float32, device=device)
    rotation_body_indices = torch.tensor(targets.rotation_body_indices, dtype=torch.int64, device=device)
    _trajectory_clip_quality(
        body_q,
        rotation_body_indices,
        frame_quality,
        confidence,
        stable,
        edge_stable,
        obstacle_pose,
        targets,
        clip_offsets,
        step_seconds,
        quality,
    )
    coordinates = target.coordinates_from_newton(joint_q, index)
    return quality, _certify_target_coordinates(target, coordinates, index)


def _finite_scalar(value: torch.Tensor) -> float | None:
    """Return one finite JSON scalar or explicit unavailability."""
    result = float(value)
    return result if math.isfinite(result) else None


def _shared_certificate_report(
    current_trajectory: torch.Tensor,
    current_coordinate: torch.Tensor,
    released_trajectory: torch.Tensor,
    released_coordinate: torch.Tensor,
    solve_cfg: object,
    ground_upper_m: float,
    pipeline_accepted: bool,
) -> dict[str, object]:
    """Build one source/current/BFM row table from identical certificate columns."""
    from isaaclab_tasks.core.multi_task.motion.mdp.commands.motion_task_table import (
        _TARGET_COORDINATE_QUALITY_NAMES,
        _TRAJECTORY_METRIC_NAMES,
    )

    values = (current_trajectory, current_coordinate, released_trajectory, released_coordinate)
    if any(value.shape[0] != 1 for value in values):
        raise ValueError("The paired production certificate currently requires exactly one selected clip.")
    semantic_source_rows = {
        "source_required_position_max_m",
        "source_required_distal_position_max_m",
        "source_required_distal_direction_max_rad",
        "source_root_rotation_max_rad",
        "source_all_position_max_m",
        "source_all_distal_position_max_m",
        "source_all_landmark_direction_max_rad",
        "source_all_distal_direction_max_rad",
        "source_nonroot_rotation_max_rad",
    }
    source_owned_rows = {
        "contact_applicable",
        "contact_stable_frame_channel_count",
        "source_contact_confidence_mean",
    }

    def row(current_value: torch.Tensor, released_value: torch.Tensor, source: float | None) -> dict[str, float | None]:
        current = _finite_scalar(current_value)
        released = _finite_scalar(released_value)
        return {
            "source": source,
            "current": current,
            "bfm": released,
            "current_minus_bfm": None if current is None or released is None else current - released,
        }

    metrics = {}
    for column, name in enumerate(_TRAJECTORY_METRIC_NAMES):
        current = current_trajectory[0, column]
        released = released_trajectory[0, column]
        if name in semantic_source_rows:
            source = 0.0
        elif name in source_owned_rows:
            source = _finite_scalar(current)
        else:
            source = None
        metrics[name] = row(current, released, source)
    for column, name in enumerate(_TARGET_COORDINATE_QUALITY_NAMES):
        metrics[name] = row(current_coordinate[0, column], released_coordinate[0, column], None)

    trajectory_row = {name: metrics[name] for name in _TRAJECTORY_METRIC_NAMES}
    coordinate_row = {name: metrics[name] for name in _TARGET_COORDINATE_QUALITY_NAMES}
    epsilon = 32.0 * torch.finfo(torch.float32).eps

    def output_contract(column: str) -> bool:
        def value(name: str) -> float | None:
            return metrics[name][column]

        applicable = value("contact_applicable")
        stable_count = value("contact_stable_frame_channel_count")
        source_valid = all(
            value(name) is not None and value(name) <= limit
            for name, limit in (
                ("source_required_position_max_m", solve_cfg.acceptance.source.required_position_upper_m),
                (
                    "source_required_distal_position_max_m",
                    solve_cfg.acceptance.source.required_distal_position_upper_m,
                ),
                (
                    "source_required_distal_direction_max_rad",
                    solve_cfg.acceptance.source.required_distal_direction_upper_rad,
                ),
                ("source_root_rotation_max_rad", solve_cfg.acceptance.source.root_rotation_upper_rad),
            )
        )
        contact_valid = (
            applicable == 1.0
            and stable_count is not None
            and stable_count > 0.0
            and all(
                value(name) is not None and value(name) <= limit
                for name, limit in (
                    ("contact_gap_max_m", solve_cfg.acceptance.contact.gap_upper_m),
                    ("contact_tilt_max_rad", solve_cfg.acceptance.contact.tilt_upper_rad),
                    ("contact_slip_speed_max_mps", solve_cfg.acceptance.contact.slip_speed_upper_mps),
                    (
                        "contact_cumulative_drift_max_m",
                        solve_cfg.acceptance.contact.cumulative_drift_upper_m,
                    ),
                )
            )
        )
        coordinate_valid = (
            value("coordinate_finite") == 1.0
            and value("root_quaternion_norm_error") is not None
            and value("root_quaternion_norm_error") <= epsilon
            and value("joint_position_limits_satisfied") == 1.0
            and value("canonical_joint_velocity_limits_satisfied") == 1.0
            and value("canonical_joint_velocity_limit_ratio") is not None
            and value("canonical_joint_velocity_limit_ratio") <= 1.0 + epsilon
            and value("fk_finite") == 1.0
            and value("ground_penetration_max_m") is not None
            and value("ground_penetration_max_m") <= ground_upper_m
        )
        return source_valid and contact_valid and coordinate_valid

    return {
        "schema": "lafan_g1_paired_baseline_v2",
        "columns": ["source", "current", "bfm"],
        "context": {
            "source_targets": "immutable_raw_lafan_projection_before_candidate_conditioned_contact_preparation",
            "contact_mask": "one_raw_lafan_confidence_stable_edge_context_shared_by_current_and_bfm",
            "clock": "raw_lafan_declared_period_shared_by_current_and_bfm",
            "spatial_alignment": "none_for_output_contract",
            "scale_fitted": False,
            "resampled": False,
            "time_warped": False,
        },
        "thresholds": {
            "source_required_position_max_m": solve_cfg.acceptance.source.required_position_upper_m,
            "source_required_distal_position_max_m": solve_cfg.acceptance.source.required_distal_position_upper_m,
            "source_required_distal_direction_max_rad": solve_cfg.acceptance.source.required_distal_direction_upper_rad,
            "source_root_rotation_max_rad": solve_cfg.acceptance.source.root_rotation_upper_rad,
            "contact_gap_max_m": solve_cfg.acceptance.contact.gap_upper_m,
            "contact_tilt_max_rad": solve_cfg.acceptance.contact.tilt_upper_rad,
            "contact_slip_speed_max_mps": solve_cfg.acceptance.contact.slip_speed_upper_mps,
            "contact_cumulative_drift_max_m": solve_cfg.acceptance.contact.cumulative_drift_upper_m,
            "ground_penetration_max_m": ground_upper_m,
        },
        "trajectory_metrics": trajectory_row,
        "target_coordinate_metrics": coordinate_row,
        "output_contract_pass": {
            "source": None,
            "current": output_contract("current"),
            "bfm": output_contract("bfm"),
        },
        "pipeline_accepted": {"source": None, "current": pipeline_accepted, "bfm": None},
        "algorithm_baselines": {
            "protomotions": {
                "status": "unavailable_for_lafan_to_g1",
                "reason": "Pinned ProtoMotions has no LAFAN route or paired LAFAN G1 output.",
            }
        },
    }


def _fk_memory_report(plan: IKMemoryPlan, measured_peak_bytes: int | None) -> dict[str, int | None]:
    """Serialize one shared-planner decision and measured extraction peak."""
    return {
        "problem_count": plan.problem_count,
        "max_safe_capacity": plan.max_safe_capacity,
        "batch_capacity": plan.batch_capacity,
        "batch_count": (plan.problem_count + plan.batch_capacity - 1) // plan.batch_capacity,
        "fixed_bytes": plan.fixed_bytes,
        "bytes_per_frame": plan.bytes_per_problem,
        "device_free_bytes": plan.device_free_bytes,
        "safety_reserve_bytes": plan.safety_reserve_bytes,
        "memory_budget_bytes": plan.memory_budget_bytes,
        "estimated_peak_additional_workspace_bytes": plan.peak_additional_workspace_bytes,
        "measured_peak_incremental_bytes": measured_peak_bytes,
    }


def _inspection_acceptance(
    view: TaskTableView, index: MotionClipIndex, sequence_indices: tuple[int, ...]
) -> dict[str, bool]:
    """Read production acceptance decisions while retaining every inspection candidate."""
    quality = view.quality
    if quality is None or quality.scope != "sequence" or "accepted" not in quality.names:
        raise ValueError("LAFAN inspection quality must expose sequence-scoped acceptance.")
    accepted_column = quality.names.index("accepted")
    rows = torch.tensor(sequence_indices, dtype=torch.int64, device=quality.values.device)
    accepted = quality.values.index_select(0, rows)[:, accepted_column] > 0.5
    return {
        index.clips[sequence].clip_id: bool(value)
        for sequence, value in zip(sequence_indices, accepted.detach().cpu(), strict=True)
    }


def _inspection_stance_by_id(
    view: TaskTableView,
    index: MotionClipIndex,
    sequence_indices: tuple[int, ...],
    support_patch_offsets: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    """Return the candidate source-contact mask expanded over each target patch."""
    contact_points = next((evidence for evidence in view.points if evidence.name == "contact_points"), None)
    if contact_points is None or contact_points.scope != "state" or contact_points.valid is None:
        raise ValueError("Candidate inspection must expose state-scoped contact_points validity.")
    point_count = contact_points.valid.shape[1]
    if (
        len(support_patch_offsets) < 2
        or support_patch_offsets[0] != 0
        or support_patch_offsets[-1] != point_count
        or any(start >= stop for start, stop in zip(support_patch_offsets[:-1], support_patch_offsets[1:], strict=True))
    ):
        raise ValueError("Target support-patch offsets must partition the candidate contact-point axis.")

    point_valid = []
    for start, stop in zip(support_patch_offsets[:-1], support_patch_offsets[1:], strict=True):
        patch_valid = contact_points.valid[:, start:stop]
        channel_valid = patch_valid[:, :1]
        if not torch.equal(patch_valid, channel_valid.expand_as(patch_valid)):
            raise ValueError("Every target support patch must carry one source-contact mask.")
        point_valid.append(channel_valid.expand(-1, stop - start))
    point_valid = torch.cat(point_valid, dim=1)

    stance_by_id = {}
    for sequence in sequence_indices:
        start = int(view.sequences.offsets[sequence])
        stop = int(view.sequences.offsets[sequence + 1])
        frame_rows = torch.arange(start, stop, dtype=torch.int64, device=point_valid.device)
        if view.sequences.state_indices is not None:
            frame_rows = view.sequences.state_indices.index_select(0, frame_rows)
        stance_by_id[index.clips[sequence].clip_id] = point_valid.index_select(0, frame_rows).contiguous()
    return stance_by_id


def _corpus_clips(
    corpus: _SemanticCorpus,
    clips: tuple[MotionClipIndex.Clip, ...],
    *,
    landmark_roles: tuple[str, ...],
    semantic_edges: tuple[tuple[str, str], ...],
    rotation_roles: tuple[str, ...],
    support_roles: tuple[str, ...],
    stance_by_id: dict[str, torch.Tensor],
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    velocity_limits: torch.Tensor,
) -> tuple[SemanticTrajectoryClip, ...]:
    """Split one compact inspection corpus into exact-ID semantic clips."""
    output = []
    cursor = 0
    for clip in clips:
        stop = cursor + clip.frame_count
        try:
            stance = stance_by_id[clip.clip_id]
        except KeyError as error:
            raise ValueError(f"Released reference does not contain candidate clip {clip.clip_id!r}.") from error
        output.append(
            SemanticTrajectoryClip(
                clip_id=clip.clip_id,
                frame_dt_s=1.0 / clip.source_fps,
                landmark_roles=landmark_roles,
                semantic_edges=semantic_edges,
                landmark_position_m=corpus.landmark_position_m[cursor:stop],
                rotation_roles=rotation_roles,
                rotation_xyzw=corpus.rotation_xyzw[cursor:stop],
                support_roles=support_roles,
                support_position_m=corpus.support_position_m[cursor:stop],
                stance_active=stance,
                joint_names=corpus.joint_names,
                joint_position_rad=corpus.joint_position_rad[cursor:stop],
                joint_velocity_rad_s=corpus.joint_velocity_rad_s[cursor:stop],
                joint_lower_limit_rad=lower_limits,
                joint_upper_limit_rad=upper_limits,
                joint_velocity_limit_rad_s=velocity_limits,
            )
        )
        cursor = stop
    if cursor != corpus.landmark_position_m.shape[0]:
        raise ValueError("Selected clip lengths do not cover the compact inspection corpus.")
    return tuple(output)


def _build_report(
    raw_artifact_root: Path,
    released_artifact_root: Path,
    motion_split: str,
    device: str,
    max_clips: int | None,
) -> dict[str, object]:
    """Materialize inspection candidates and run the simulator-free comparison."""
    import warp as wp

    from isaaclab_tasks.core.multi_task.kinematics import NewtonKinematics
    from isaaclab_tasks.core.multi_task.motion_env_cfg import MotionImitationEnvCfg
    from isaaclab_tasks.utils import resolve_presets

    def resolved(source: str, root: Path):
        cfg = resolve_presets(MotionImitationEnvCfg(), selected=frozenset(("g1", source)))
        cfg.commands.motion.task_table.source_artifact_root = str(root.expanduser().resolve())
        cfg.commands.motion.task_table.motion_split = motion_split
        return cfg

    raw_cfg = resolved("lafan", raw_artifact_root)
    released_cfg = resolved("bfm_lafan", released_artifact_root)
    raw_index = _inspect_source_index(raw_cfg)
    released_index = _inspect_source_index(released_cfg)
    raw_sequence_indices, released_sequence_indices = _paired_prefix_indices(raw_index, released_index, max_clips)
    selected_count = len(raw_sequence_indices)
    raw_clips = tuple(raw_index.clips[row] for row in raw_sequence_indices)
    released_clips = tuple(released_index.clips[row] for row in released_sequence_indices)
    selected_raw_index = _compact_index(raw_index, raw_sequence_indices)
    selected_released_index = _compact_index(released_index, released_sequence_indices)

    target_cfg = raw_cfg.commands.motion.task_table.target_kinematics
    articulation_cfg = getattr(raw_cfg.scene, target_cfg.asset_cfg.name)
    reference = NewtonKinematics.from_articulation(target_cfg.kinematics, articulation_cfg, device)
    target = target_cfg.target_factory(reference, target_cfg.contact_patches)
    trajectory = target.trajectory_target
    landmark_roles = tuple(item.role for item in trajectory.landmarks)
    semantic_edges = tuple((landmark_roles[item.parent_row], item.role) for item in trajectory.landmarks[1:])
    rotation_roles = (landmark_roles[0],)
    support_roles = tuple(
        f"{channel}[{point - start}]"
        for channel, start, stop in zip(
            trajectory.contact_channel_names,
            trajectory.support_patch_offsets[:-1],
            trajectory.support_patch_offsets[1:],
            strict=True,
        )
        for point in range(start, stop)
    )

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
        cuda_allocated_bytes_before = torch.cuda.memory_allocated(torch_device)
        torch.cuda.reset_peak_memory_stats(torch_device)
        torch.cuda.synchronize(torch_device)
    else:
        cuda_allocated_bytes_before = None
    with count_trajectory_nonlinear_iterations() as nonlinear_iterations:
        build_started = time.perf_counter()
        raw_view = raw_cfg.commands.motion.task_table.build_inspection_view(
            raw_cfg.commands.motion, raw_cfg.scene, device, sequence_limit=selected_count
        )
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        build_seconds = time.perf_counter() - build_started
    cuda_peak_allocated_bytes = torch.cuda.max_memory_allocated(torch_device) if torch_device.type == "cuda" else None
    build_performance = _build_performance(
        scope="full_production_task_table_construction_with_inspection_candidate_retention",
        included_stages=(
            "source_clip_decode",
            "scene_owned_target_kinematics",
            "source_landmark_projection",
            "trajectory_optimization",
            "criteria_and_selection",
            "target_robot_frame_materialization",
            "inspection_view_retaining_rejected_candidates",
        ),
        wall_seconds=build_seconds,
        input_clip_count=selected_count,
        input_frame_count=raw_index.offsets[selected_count],
        output_clip_count=raw_view.sequences.sequence_count,
        output_frame_count=raw_view.sequences.frame_count,
        device=torch_device,
        cuda_allocated_bytes_before=cuda_allocated_bytes_before,
        cuda_peak_allocated_bytes=cuda_peak_allocated_bytes,
        trajectory_nonlinear_iterations=nonlinear_iterations.report(),
    )
    acceptance_by_id = _inspection_acceptance(raw_view, raw_index, raw_sequence_indices)
    stance_by_id = _inspection_stance_by_id(raw_view, raw_index, raw_sequence_indices, trajectory.support_patch_offsets)
    candidate_corpus, candidate_memory_plan, candidate_peak_bytes = _inspection_corpus(
        raw_view, raw_index, raw_sequence_indices, reference, trajectory
    )
    if selected_count == 1:
        current_joint_q = _inspection_joint_q(raw_view, raw_index, raw_sequence_indices, reference)
        raw_targets = _selected_source_targets(raw_cfg, target, raw_index, raw_sequence_indices, device)
    else:
        current_joint_q = None
        raw_targets = None
    del raw_view

    released_joint_q = _selected_released_joint_q(
        released_cfg, target, released_index, released_sequence_indices, device
    )
    released_corpus, released_memory_plan, released_peak_bytes = _newton_corpus(
        target, selected_released_index, released_joint_q
    )

    tree = target.kinematic_tree
    if any(stop - start != 1 for start, stop in tree.joint_coordinate_ranges):
        raise ValueError("The G1 benchmark requires one hard-limit coordinate per named joint.")
    joint_row = {name: row for row, name in enumerate(tree.joint_names)}
    if (
        set(candidate_corpus.joint_names) != set(joint_row)
        or candidate_corpus.joint_names != released_corpus.joint_names
    ):
        raise ValueError("Candidate, released, and target G1 joint identities differ.")
    order = torch.tensor(
        [joint_row[name] for name in candidate_corpus.joint_names],
        dtype=torch.int64,
        device=reference.device,
    )
    lower_reference = torch.tensor(tree.coordinate_lower_limits_rad, dtype=torch.float32, device=reference.device)
    upper_reference = torch.tensor(tree.coordinate_upper_limits_rad, dtype=torch.float32, device=reference.device)
    velocity_reference = (
        wp.to_torch(reference.model.joint_velocity_limit)
        .abs()
        .index_select(
            0,
            torch.tensor(tree.coordinate_qd_indices, dtype=torch.int64, device=reference.device),
        )
    )
    velocity_reference.masked_fill_(~torch.isfinite(velocity_reference) | (velocity_reference <= 0.0), torch.inf)
    lower = lower_reference.index_select(0, order)
    upper = upper_reference.index_select(0, order)
    velocity = velocity_reference.index_select(0, order)

    trajectory_family = next(
        family for family in raw_cfg.commands.motion.task_table.families if family.name == "trajectory"
    )
    solve_cfg = trajectory_family.solve
    if solve_cfg is None:
        raise ValueError("The LAFAN trajectory family must own one solver configuration.")
    shared_certificate = None
    if selected_count == 1:
        from isaaclab_tasks.core.multi_task.motion.mdp.commands.commands_cfg import (
            MotionGroundPenetrationCriterionCfg,
        )

        if current_joint_q is None or raw_targets is None:
            raise RuntimeError("The selected LAFAN clip is missing its current coordinates or raw targets.")
        contact_context = _raw_contact_context(solve_cfg, raw_targets, selected_raw_index)
        current_trajectory, current_coordinate = _shared_output_certificate(
            target, selected_raw_index, raw_targets, current_joint_q, contact_context
        )
        released_trajectory, released_coordinate = _shared_output_certificate(
            target, selected_raw_index, raw_targets, released_joint_q, contact_context
        )
        ground_cfg = next(
            criterion
            for criterion in trajectory_family.criteria
            if isinstance(criterion, MotionGroundPenetrationCriterionCfg)
        )
        clip_id = raw_clips[0].clip_id
        shared_certificate = _shared_certificate_report(
            current_trajectory,
            current_coordinate,
            released_trajectory,
            released_coordinate,
            solve_cfg,
            ground_cfg.upper_m,
            acceptance_by_id[clip_id],
        )
    candidate_clips = _corpus_clips(
        candidate_corpus,
        raw_clips,
        landmark_roles=landmark_roles,
        semantic_edges=semantic_edges,
        rotation_roles=rotation_roles,
        support_roles=support_roles,
        stance_by_id=stance_by_id,
        lower_limits=lower,
        upper_limits=upper,
        velocity_limits=velocity,
    )
    released_semantic_clips = _corpus_clips(
        released_corpus,
        released_clips,
        landmark_roles=landmark_roles,
        semantic_edges=semantic_edges,
        rotation_roles=rotation_roles,
        support_roles=support_roles,
        stance_by_id=stance_by_id,
        lower_limits=lower,
        upper_limits=upper,
        velocity_limits=velocity,
    )
    report = compare_lafan_retargeting(candidate_clips, released_semantic_clips)
    if shared_certificate is not None:
        report["paired_output_certificate"] = shared_certificate
    accepted_ids = sorted(clip_id for clip_id, accepted in acceptance_by_id.items() if accepted)
    rejected_ids = sorted(clip_id for clip_id, accepted in acceptance_by_id.items() if not accepted)
    report["candidate_acceptance"] = {
        "accepted_clip_count": len(accepted_ids),
        "rejected_clip_count": len(rejected_ids),
        "accepted_clip_ids": accepted_ids,
        "rejected_clip_ids": rejected_ids,
    }
    for clip_report in report["clips"]:
        clip_report["candidate_accepted_by_config"] = acceptance_by_id[clip_report["clip_id"]]
    report["performance"] = build_performance
    report["inputs"] = {
        "candidate": "raw_lafan1_bvh_ground_to_g1_inspection_candidates",
        "released_reference": "released_bfm_lafan_g1_29dof",
        "split": motion_split,
        "device": device,
        "max_clips": max_clips,
        "fk_memory": {
            "candidate": _fk_memory_report(candidate_memory_plan, candidate_peak_bytes),
            "released_reference": _fk_memory_report(released_memory_plan, released_peak_bytes),
        },
    }
    report["stance_thresholds"] = {
        "enter_height_m": solve_cfg.contact.enter_height_m,
        "exit_height_m": solve_cfg.contact.exit_height_m,
        "enter_speed_mps": solve_cfg.contact.enter_speed_mps,
        "exit_speed_mps": solve_cfg.contact.exit_speed_mps,
        "persistence_seconds": solve_cfg.contact.persistence_seconds,
    }
    return report


def main() -> None:
    """Run the direct released-LAFAN comparison and print JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-artifact-root", type=Path, required=True)
    parser.add_argument("--released-artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--motion-split", choices=("train", "evaluation"), default="evaluation")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_clips is not None and args.max_clips < 1:
        parser.error("--max-clips must be positive")
    # This standalone analysis imports task internals but does not register Gym environments.
    builtins._isaaclab_tasks_registered = True
    report = _build_report(
        args.raw_artifact_root,
        args.released_artifact_root,
        args.motion_split,
        args.device,
        args.max_clips,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
