# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neutral retargeting measurements over explicit native and deployment mechanics.

The benchmark owns one immutable raw-source-to-selected-target semantic
projection.  It never compares a target robot directly with unscaled human
coordinates and never treats BFM, MetaMotivo, ProtoMotions, or SOMA as ground
truth.  Every method supplies two FK materializations:

* its native target mechanics, isolating output behavior under those mechanics;
* the selected Isaac target mechanics, for deployment quality and constraints.

Both views retain the common benchmark projection; neither recreates a method's private source calibration or objective.

Source fidelity removes one constant world-z yaw and horizontal translation per
clip.  It does not fit scale, vertical translation, time, or a per-frame pose.
Hard constraints and contact/ground measurements use unaligned world geometry.
The report deliberately contains no aggregate score or acceptance threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_FRAME_DT_ABSOLUTE_TOLERANCE_S = 0.5e-6
_NEAR_POSITION_LIMIT_TOLERANCE_RAD = 1.0e-4
_SCHEMA = "neutral_retarget_evaluation_v1"


def _validate_sha256(name: str, value: str) -> None:
    """Require one lowercase hexadecimal SHA-256 identity."""
    if len(value) != 64 or value.lower() != value:
        raise ValueError(f"{name} must be one lowercase SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be one lowercase SHA-256 digest.") from error


def _cpu64(value: torch.Tensor) -> torch.Tensor:
    """Copy measured values to stable CPU float64 arithmetic."""
    return value.detach().to(device="cpu", dtype=torch.float64)


def _strict_frame_indices(name: str, value: torch.Tensor) -> None:
    """Require nonempty, unique, increasing source-frame indices."""
    if value.ndim != 1 or value.dtype is not torch.int64 or value.numel() < 1:
        raise ValueError(f"{name} must be nonempty one-dimensional int64 source-frame indices.")
    rows = value.detach().cpu()
    if rows.numel() > 1 and bool(torch.any(rows[1:] <= rows[:-1])):
        raise ValueError(f"{name} must be strictly increasing.")


def _finite_reference(name: str, value: torch.Tensor) -> None:
    """Require finite floating-point benchmark-owned reference values."""
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain finite floating-point values.")


@dataclass(frozen=True, slots=True)
class RetargetEvaluationTarget:
    """One exact target-mechanics and collision-geometry contract."""

    geometry_sha256: str
    joint_names: tuple[str, ...]
    joint_lower_limit_rad: torch.Tensor
    joint_upper_limit_rad: torch.Tensor
    joint_velocity_limit_rad_s: torch.Tensor
    support_patch_roles: tuple[str, ...]
    support_patch_offsets: tuple[int, ...]
    collision_probe_count: int
    ground_height_m: float = 0.0

    def __post_init__(self) -> None:
        """Validate joint limits and target-owned support/collision layout."""
        _validate_sha256("geometry_sha256", self.geometry_sha256)
        joint_count = len(self.joint_names)
        patch_count = len(self.support_patch_roles)
        if joint_count < 1 or len(set(self.joint_names)) != joint_count:
            raise ValueError("Target joint names must be nonempty and unique.")
        if (
            self.joint_lower_limit_rad.shape != (joint_count,)
            or self.joint_upper_limit_rad.shape != (joint_count,)
            or self.joint_velocity_limit_rad_s.shape != (joint_count,)
        ):
            raise ValueError("Target hard limits must share the declared joint axis.")
        for name, value in (
            ("joint_lower_limit_rad", self.joint_lower_limit_rad),
            ("joint_upper_limit_rad", self.joint_upper_limit_rad),
            ("joint_velocity_limit_rad_s", self.joint_velocity_limit_rad_s),
        ):
            _finite_reference(name, value)
        if bool(torch.any(self.joint_lower_limit_rad >= self.joint_upper_limit_rad)):
            raise ValueError("Every target lower joint limit must be below its upper limit.")
        if bool(torch.any(self.joint_velocity_limit_rad_s <= 0.0)):
            raise ValueError("Every target joint velocity limit must be positive.")
        if (
            patch_count < 1
            or len(set(self.support_patch_roles)) != patch_count
            or len(self.support_patch_offsets) != patch_count + 1
            or self.support_patch_offsets[0] != 0
            or any(
                stop - start < 3
                for start, stop in zip(self.support_patch_offsets[:-1], self.support_patch_offsets[1:], strict=True)
            )
        ):
            raise ValueError("Every unique target support patch must contain at least three ordered points.")
        if self.collision_probe_count < 1:
            raise ValueError("Target ground checks require at least one target-owned collision probe.")
        if not math.isfinite(self.ground_height_m):
            raise ValueError("Target ground height must be finite.")


@dataclass(frozen=True, slots=True)
class RetargetEvaluationSourceClip:
    """Benchmark-owned common semantic projection for one raw source clip."""

    clip_id: str
    source_content_sha256: str
    semantic_projection_sha256: str
    selected_target_geometry_sha256: str
    frame_dt_s: float
    frame_indices: torch.Tensor
    landmark_roles: tuple[str, ...]
    root_role: str
    semantic_edges: tuple[tuple[str, str], ...]
    landmark_position_m: torch.Tensor
    rotation_roles: tuple[str, ...]
    rotation_xyzw: torch.Tensor
    support_patch_roles: tuple[str, ...]
    stance_active: torch.Tensor

    def __post_init__(self) -> None:
        """Require explicit finite target-space source semantics."""
        for name, value in (
            ("source_content_sha256", self.source_content_sha256),
            ("semantic_projection_sha256", self.semantic_projection_sha256),
            ("selected_target_geometry_sha256", self.selected_target_geometry_sha256),
        ):
            _validate_sha256(name, value)
        _strict_frame_indices("frame_indices", self.frame_indices)
        frame_count = self.frame_indices.numel()
        landmark_count = len(self.landmark_roles)
        rotation_count = len(self.rotation_roles)
        patch_count = len(self.support_patch_roles)
        if not self.clip_id or not math.isfinite(self.frame_dt_s) or self.frame_dt_s <= 0.0:
            raise ValueError("Source clips require a nonempty ID and positive frame period.")
        if (
            landmark_count < 2
            or len(set(self.landmark_roles)) != landmark_count
            or self.root_role not in self.landmark_roles
            or self.landmark_position_m.shape != (frame_count, landmark_count, 3)
            or rotation_count < 1
            or len(set(self.rotation_roles)) != rotation_count
            or self.rotation_xyzw.shape != (frame_count, rotation_count, 4)
            or patch_count < 1
            or len(set(self.support_patch_roles)) != patch_count
            or self.stance_active.shape != (frame_count, patch_count)
            or self.stance_active.dtype is not torch.bool
        ):
            raise ValueError("Source semantic tensors must share their declared frame and role axes.")
        _finite_reference("landmark_position_m", self.landmark_position_m)
        _finite_reference("rotation_xyzw", self.rotation_xyzw)
        rotation_norm = torch.linalg.vector_norm(self.rotation_xyzw, dim=-1)
        if bool(torch.any(rotation_norm <= 1.0e-9)):
            raise ValueError("Source semantic rotations must be nonzero quaternions.")
        role_set = set(self.landmark_roles)
        if (
            not self.semantic_edges
            or len(set(self.semantic_edges)) != len(self.semantic_edges)
            or any(parent not in role_set or child not in role_set or parent == child for parent, child in self.semantic_edges)
        ):
            raise ValueError("Semantic edges must be unique directed edges between declared landmarks.")


@dataclass(frozen=True, slots=True)
class RetargetEvaluationView:
    """One method output materialized through one explicit target mechanics."""

    geometry_sha256: str
    clip_id: str
    frame_dt_s: float
    frame_indices: torch.Tensor
    landmark_roles: tuple[str, ...]
    landmark_position_m: torch.Tensor
    rotation_roles: tuple[str, ...]
    rotation_xyzw: torch.Tensor
    support_point_position_m: torch.Tensor
    collision_probe_position_m: torch.Tensor
    joint_names: tuple[str, ...]
    joint_position_rad: torch.Tensor
    joint_velocity_rad_s: torch.Tensor

    def __post_init__(self) -> None:
        """Require frame-aligned materialized output tensors without hiding nonfinite output."""
        _validate_sha256("geometry_sha256", self.geometry_sha256)
        _strict_frame_indices("frame_indices", self.frame_indices)
        frame_count = self.frame_indices.numel()
        if not self.clip_id or not math.isfinite(self.frame_dt_s) or self.frame_dt_s <= 0.0:
            raise ValueError("Materialized views require a nonempty clip ID and positive frame period.")
        if (
            len(self.landmark_roles) < 2
            or len(set(self.landmark_roles)) != len(self.landmark_roles)
            or self.landmark_position_m.shape != (frame_count, len(self.landmark_roles), 3)
            or len(self.rotation_roles) < 1
            or len(set(self.rotation_roles)) != len(self.rotation_roles)
            or self.rotation_xyzw.shape != (frame_count, len(self.rotation_roles), 4)
            or self.support_point_position_m.ndim != 3
            or self.support_point_position_m.shape[0] != frame_count
            or self.support_point_position_m.shape[2] != 3
            or self.collision_probe_position_m.ndim != 3
            or self.collision_probe_position_m.shape[0] != frame_count
            or self.collision_probe_position_m.shape[2] != 3
            or len(self.joint_names) < 1
            or len(set(self.joint_names)) != len(self.joint_names)
            or self.joint_position_rad.shape != (frame_count, len(self.joint_names))
            or self.joint_velocity_rad_s.shape != (frame_count, len(self.joint_names))
        ):
            raise ValueError("Materialized output tensors must share their declared frame and feature axes.")
        measured = (
            self.landmark_position_m,
            self.rotation_xyzw,
            self.support_point_position_m,
            self.collision_probe_position_m,
            self.joint_position_rad,
            self.joint_velocity_rad_s,
        )
        if any(not value.is_floating_point() for value in measured):
            raise ValueError("Materialized output measurements must use floating-point tensors.")


@dataclass(frozen=True, slots=True)
class RetargetEvaluationRuntime:
    """Declared end-to-end runtime scope for one method corpus."""

    scope: str
    included_stages: tuple[str, ...]
    wall_seconds: float
    input_frame_count: int
    output_frame_count: int
    device: str
    peak_incremental_bytes: int | None = None

    def __post_init__(self) -> None:
        """Require enough provenance to decide whether runtimes are comparable."""
        if (
            not self.scope
            or not self.included_stages
            or len(set(self.included_stages)) != len(self.included_stages)
            or not math.isfinite(self.wall_seconds)
            or self.wall_seconds <= 0.0
            or self.input_frame_count < 1
            or self.output_frame_count < 1
            or not self.device
            or (self.peak_incremental_bytes is not None and self.peak_incremental_bytes < 0)
        ):
            raise ValueError("Runtime records require explicit stages, positive counts/duration, and valid memory.")


@dataclass(frozen=True, slots=True)
class RetargetEvaluationMethod:
    """One method's native and selected-target materializations over a corpus."""

    name: str
    native_target: RetargetEvaluationTarget
    native_clips: tuple[RetargetEvaluationView, ...]
    selected_target_clips: tuple[RetargetEvaluationView, ...]
    runtime: RetargetEvaluationRuntime

    def __post_init__(self) -> None:
        """Require unique, paired native and selected-target clip IDs."""
        native_ids = tuple(clip.clip_id for clip in self.native_clips)
        selected_ids = tuple(clip.clip_id for clip in self.selected_target_clips)
        if (
            not self.name
            or not native_ids
            or len(set(native_ids)) != len(native_ids)
            or len(set(selected_ids)) != len(selected_ids)
            or set(native_ids) != set(selected_ids)
        ):
            raise ValueError("Methods require a name and paired unique native/selected clip IDs.")


def _statistics(values: torch.Tensor) -> dict[str, float | int | None]:
    """Summarize finite samples while retaining every nonfinite output count."""
    values = _cpu64(values).reshape(-1)
    finite = torch.isfinite(values)
    measured = values[finite]
    result: dict[str, float | int | None] = {
        "count": values.numel(),
        "finite_count": measured.numel(),
        "nonfinite_count": values.numel() - measured.numel(),
        "mean": None,
        "rms": None,
        "p50": None,
        "p95": None,
        "max": None,
    }
    if measured.numel() == 0:
        return result
    quantiles = torch.quantile(measured, torch.tensor((0.5, 0.95), dtype=torch.float64))
    result.update(
        mean=float(measured.mean()),
        rms=float(torch.sqrt(torch.mean(torch.square(measured)))),
        p50=float(quantiles[0]),
        p95=float(quantiles[1]),
        max=float(measured.max()),
    )
    return result


def _margin_statistics(values: torch.Tensor) -> dict[str, float | int | None]:
    """Summarize signed distance to the nearest position limit."""
    result = _statistics(values)
    finite = _cpu64(values).reshape(-1)
    finite = finite[torch.isfinite(finite)]
    result["min"] = None
    result["p05"] = None
    if finite.numel():
        result["min"] = float(finite.min())
        result["p05"] = float(torch.quantile(finite, 0.05))
    return result


def _fit_yaw_xy(candidate: torch.Tensor, reference: torch.Tensor) -> tuple[float, torch.Tensor, torch.Tensor, int]:
    """Fit one constant world-z yaw and XY translation without scale or vertical shift."""
    candidate_flat = candidate.reshape(-1, 3)
    reference_flat = reference.reshape(-1, 3)
    valid = torch.isfinite(candidate_flat).all(dim=-1) & torch.isfinite(reference_flat).all(dim=-1)
    if int(valid.sum()) < 2:
        raise ValueError("Source-fidelity alignment requires at least two finite landmark samples.")
    candidate_fit = candidate_flat[valid]
    reference_fit = reference_flat[valid]
    candidate_center_xy = candidate_fit[:, :2].mean(dim=0)
    reference_center_xy = reference_fit[:, :2].mean(dim=0)
    centered_candidate = candidate_fit[:, :2] - candidate_center_xy
    centered_reference = reference_fit[:, :2] - reference_center_xy
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
    translation = candidate.new_zeros(3)
    translation[:2] = reference_center_xy - candidate_center_xy @ rotation[:2, :2].T
    return yaw, translation, candidate @ rotation.T + translation, int(valid.sum())


def _quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply quaternions in XYZW storage order."""
    left_xyz, left_w = left[..., :3], left[..., 3:]
    right_xyz, right_w = right[..., :3], right[..., 3:]
    xyz = left_w * right_xyz + right_w * left_xyz + torch.linalg.cross(left_xyz, right_xyz, dim=-1)
    w = left_w * right_w - torch.sum(left_xyz * right_xyz, dim=-1, keepdim=True)
    return torch.cat((xyz, w), dim=-1)


def _aligned_rotation(rotation_xyzw: torch.Tensor, yaw: float) -> torch.Tensor:
    """Apply the fitted world-yaw basis change and normalize valid quaternions."""
    yaw_quaternion = rotation_xyzw.new_tensor((0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)))
    aligned = _quat_mul(yaw_quaternion.expand_as(rotation_xyzw), rotation_xyzw)
    norm = torch.linalg.vector_norm(aligned, dim=-1, keepdim=True)
    return aligned / norm


def _orientation_error(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """Return shortest quaternion geodesic angle [rad]."""
    actual = actual / torch.linalg.vector_norm(actual, dim=-1, keepdim=True)
    expected = expected / torch.linalg.vector_norm(expected, dim=-1, keepdim=True)
    dot = torch.sum(actual * expected, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _paired_rows(source: RetargetEvaluationSourceClip, view: RetargetEvaluationView) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Pair exact source-frame indices without interpolation or time warp."""
    if source.clip_id != view.clip_id:
        raise ValueError("Source and output clip IDs differ.")
    if not math.isclose(
        source.frame_dt_s,
        view.frame_dt_s,
        rel_tol=0.0,
        abs_tol=_FRAME_DT_ABSOLUTE_TOLERANCE_S,
    ):
        raise ValueError(f"Clip {source.clip_id!r} changes frame period; resampling is forbidden.")
    source_values = tuple(int(value) for value in source.frame_indices.detach().cpu())
    output_values = tuple(int(value) for value in view.frame_indices.detach().cpu())
    output_row = {value: row for row, value in enumerate(output_values)}
    paired = tuple((row, output_row[value]) for row, value in enumerate(source_values) if value in output_row)
    if not paired:
        raise ValueError(f"Clip {source.clip_id!r} has no exact source-frame correspondence.")
    source_rows = torch.tensor(tuple(row[0] for row in paired), dtype=torch.int64)
    output_rows = torch.tensor(tuple(row[1] for row in paired), dtype=torch.int64)
    source_set = set(source_values)
    output_set = set(output_values)
    return source_rows, output_rows, {
        "source_frames": len(source_values),
        "output_frames": len(output_values),
        "matched_frames": len(paired),
        "matched_fraction": len(paired) / len(source_values),
        "missing_source_frame_count": len(source_set - output_set),
        "unexpected_output_frame_count": len(output_set - source_set),
        "complete": source_set == output_set,
    }


@dataclass(frozen=True, slots=True)
class _Measurements:
    """Unreduced samples for one clip and one target mechanics."""

    landmark_error_m: torch.Tensor
    root_relative_landmark_error_m: torch.Tensor
    root_z_error_m: torch.Tensor
    root_horizontal_path_ratio: torch.Tensor
    root_horizontal_displacement_ratio: torch.Tensor
    edge_angle_rad: torch.Tensor
    orientation_error_rad: torch.Tensor
    position_excess_rad: torch.Tensor
    position_margin_rad: torch.Tensor
    position_normalized_margin: torch.Tensor
    velocity_excess_rad_s: torch.Tensor
    velocity_utilization_ratio: torch.Tensor
    quaternion_norm_error: torch.Tensor
    collision_penetration_m: torch.Tensor
    stance_hover_m: torch.Tensor
    stance_support_penetration_m: torch.Tensor
    stance_tilt_rad: torch.Tensor
    stance_slip_speed_m_s: torch.Tensor
    stance_cumulative_drift_m: torch.Tensor
    stance_interval_count: int


def _source_fidelity(
    source: RetargetEvaluationSourceClip,
    view: RetargetEvaluationView,
    source_rows: torch.Tensor,
    output_rows: torch.Tensor,
) -> tuple[dict, tuple[torch.Tensor, ...]]:
    """Measure common projected source semantics through one explicit mechanics."""
    if view.landmark_roles != source.landmark_roles or view.rotation_roles != source.rotation_roles:
        raise ValueError(f"Clip {source.clip_id!r} changes the benchmark-owned semantic role set or order.")
    reference_landmark = _cpu64(source.landmark_position_m).index_select(0, source_rows)
    output_landmark = _cpu64(view.landmark_position_m).index_select(0, output_rows)
    yaw, translation, aligned_landmark, fit_count = _fit_yaw_xy(output_landmark, reference_landmark)
    landmark_error = torch.linalg.vector_norm(aligned_landmark - reference_landmark, dim=-1)
    role_row = {role: row for row, role in enumerate(source.landmark_roles)}
    root_row = role_row[source.root_role]
    output_root_relative = aligned_landmark - aligned_landmark[:, root_row : root_row + 1]
    reference_root_relative = reference_landmark - reference_landmark[:, root_row : root_row + 1]
    root_relative_error = torch.linalg.vector_norm(output_root_relative - reference_root_relative, dim=-1)
    root_z_error = aligned_landmark[:, root_row, 2] - reference_landmark[:, root_row, 2]
    paired_indices = source.frame_indices.detach().cpu().index_select(0, source_rows)
    valid_edges = paired_indices[1:] == paired_indices[:-1] + 1
    output_root_xy = aligned_landmark[:, root_row, :2]
    reference_root_xy = reference_landmark[:, root_row, :2]
    output_path = torch.linalg.vector_norm(output_root_xy[1:] - output_root_xy[:-1], dim=-1)[valid_edges].sum()
    reference_path = torch.linalg.vector_norm(reference_root_xy[1:] - reference_root_xy[:-1], dim=-1)[valid_edges].sum()
    output_displacement = torch.linalg.vector_norm(output_root_xy[-1] - output_root_xy[0])
    reference_displacement = torch.linalg.vector_norm(reference_root_xy[-1] - reference_root_xy[0])
    nan = landmark_error.new_tensor(float("nan"))
    path_ratio = torch.where(reference_path > 1.0e-12, output_path / reference_path, nan).reshape(1)
    displacement_ratio = torch.where(
        reference_displacement > 1.0e-12, output_displacement / reference_displacement, nan
    ).reshape(1)
    edge_values = []
    edge_by_role = {}
    for parent, child in source.semantic_edges:
        candidate_edge = aligned_landmark[:, role_row[child]] - aligned_landmark[:, role_row[parent]]
        reference_edge = reference_landmark[:, role_row[child]] - reference_landmark[:, role_row[parent]]
        cross = torch.linalg.vector_norm(torch.linalg.cross(candidate_edge, reference_edge, dim=-1), dim=-1)
        dot = torch.sum(candidate_edge * reference_edge, dim=-1)
        valid = (torch.linalg.vector_norm(candidate_edge, dim=-1) > 1.0e-9) & (
            torch.linalg.vector_norm(reference_edge, dim=-1) > 1.0e-9
        )
        angle = torch.where(valid, torch.atan2(cross, dot), torch.full_like(dot, torch.nan))
        edge_values.append(angle)
        edge_by_role[f"{parent}->{child}"] = _statistics(angle)
    edge_angle = torch.stack(edge_values, dim=1)
    output_rotation = _cpu64(view.rotation_xyzw).index_select(0, output_rows)
    output_rotation = _aligned_rotation(output_rotation, yaw)
    reference_rotation = _cpu64(source.rotation_xyzw).index_select(0, source_rows)
    orientation_error = _orientation_error(output_rotation, reference_rotation)
    return (
        {
            "alignment": {
                "yaw_rad": yaw,
                "translation_m": [float(value) for value in translation],
                "fit_landmark_samples": fit_count,
                "requested_landmark_samples": reference_landmark.shape[0] * reference_landmark.shape[1],
            },
            "landmark_error_m": {
                "all": _statistics(landmark_error),
                "by_role": {
                    role: _statistics(landmark_error[:, row]) for row, role in enumerate(source.landmark_roles)
                },
            },
            "root_relative_landmark_error_m": _statistics(root_relative_error),
            "root_z_error_m": _statistics(root_z_error),
            "root_z_absolute_error_m": _statistics(root_z_error.abs()),
            "root_horizontal_path_ratio": _statistics(path_ratio),
            "root_horizontal_displacement_ratio": _statistics(displacement_ratio),
            "semantic_edge_angle_rad": {"all": _statistics(edge_angle), "by_edge": edge_by_role},
            "selected_orientation_error_rad": {
                "all": _statistics(orientation_error),
                "by_role": {
                    role: _statistics(orientation_error[:, row]) for row, role in enumerate(source.rotation_roles)
                },
            },
        },
        (
            landmark_error,
            root_relative_error,
            root_z_error,
            path_ratio,
            displacement_ratio,
            edge_angle,
            orientation_error,
        ),
    )


def _hard_constraints(
    view: RetargetEvaluationView, target: RetargetEvaluationTarget
) -> tuple[dict, tuple[torch.Tensor, ...]]:
    """Measure coordinate limits using the same mechanics that materialized the view."""
    if view.geometry_sha256 != target.geometry_sha256 or view.joint_names != target.joint_names:
        raise ValueError(f"Clip {view.clip_id!r} is not materialized in its declared target mechanics.")
    if (
        view.support_point_position_m.shape[1] != target.support_patch_offsets[-1]
        or view.collision_probe_position_m.shape[1] != target.collision_probe_count
    ):
        raise ValueError(f"Clip {view.clip_id!r} does not expose its target-owned support/collision geometry.")
    position = _cpu64(view.joint_position_rad)
    velocity = _cpu64(view.joint_velocity_rad_s)
    lower = _cpu64(target.joint_lower_limit_rad)
    upper = _cpu64(target.joint_upper_limit_rad)
    speed = _cpu64(target.joint_velocity_limit_rad_s)
    position_excess = torch.maximum(lower - position, position - upper).clamp_min(0.0)
    position_margin = torch.minimum(position - lower, upper - position)
    position_normalized_margin = position_margin / (upper - lower)
    velocity_utilization = velocity.abs() / speed
    velocity_excess = (velocity.abs() - speed).clamp_min(0.0)
    quaternion_norm_error = (torch.linalg.vector_norm(_cpu64(view.rotation_xyzw), dim=-1) - 1.0).abs()

    def violation_report(values: torch.Tensor, unit: str) -> dict:
        finite = torch.isfinite(values)
        violating = finite & (values > 0.0)
        return {
            f"excess_{unit}": _statistics(values),
            "violating_samples": int(violating.sum()),
            "violating_fraction_of_all_samples": float(violating.sum()) / values.numel(),
        }

    coordinate_values = torch.cat((position.reshape(-1), velocity.reshape(-1)))
    position_report = violation_report(position_excess, "rad")
    position_report.update(
        {
            "minimum_margin_rad": _margin_statistics(position_margin),
            "normalized_margin_of_interval": _margin_statistics(position_normalized_margin),
            "near_limit_tolerance_rad": _NEAR_POSITION_LIMIT_TOLERANCE_RAD,
            "near_limit_samples": int(
                torch.sum(torch.isfinite(position_margin) & (position_margin.abs() <= _NEAR_POSITION_LIMIT_TOLERANCE_RAD))
            ),
        }
    )
    velocity_report = violation_report(velocity_excess, "rad_s")
    velocity_report["utilization_ratio"] = _statistics(velocity_utilization)
    return (
        {
            "coordinate_finite": {
                "count": coordinate_values.numel(),
                "finite_count": int(torch.isfinite(coordinate_values).sum()),
                "nonfinite_count": int((~torch.isfinite(coordinate_values)).sum()),
            },
            "position": position_report,
            "velocity": velocity_report,
            "selected_rotation_quaternion_norm_error": _statistics(quaternion_norm_error),
        },
        (
            position_excess,
            position_margin,
            position_normalized_margin,
            velocity_excess,
            velocity_utilization,
            quaternion_norm_error,
        ),
    )


def _contacts_ground(
    source: RetargetEvaluationSourceClip,
    view: RetargetEvaluationView,
    target: RetargetEvaluationTarget,
    source_rows: torch.Tensor,
    output_rows: torch.Tensor,
) -> tuple[dict, tuple[torch.Tensor, ...], int]:
    """Measure target-owned collision geometry under the source-owned stance mask."""
    if source.support_patch_roles != target.support_patch_roles:
        raise ValueError("Source stance roles and target support-patch roles differ.")
    support_all = _cpu64(view.support_point_position_m)
    collision = _cpu64(view.collision_probe_position_m)
    collision_penetration = (target.ground_height_m - collision[..., 2]).clamp_min(0.0).amax(dim=1)
    support = support_all.index_select(0, output_rows)
    stance = source.stance_active.detach().cpu().index_select(0, source_rows)
    frame_indices = source.frame_indices.detach().cpu().index_select(0, source_rows)
    hover_values: list[torch.Tensor] = []
    penetration_values: list[torch.Tensor] = []
    tilt_values: list[torch.Tensor] = []
    slip_values: list[torch.Tensor] = []
    drift_values: list[torch.Tensor] = []
    interval_count = 0
    for patch, (start, stop) in enumerate(
        zip(target.support_patch_offsets[:-1], target.support_patch_offsets[1:], strict=True)
    ):
        points = support[:, start:stop]
        center = points.mean(dim=1)
        first = points[:, 1] - points[:, 0]
        second = points[:, 2] - points[:, 0]
        normal = torch.linalg.cross(first, second, dim=-1)
        normal_norm = torch.linalg.vector_norm(normal, dim=-1)
        cosine = (normal[:, 2].abs() / normal_norm).clamp(max=1.0)
        tilt = torch.where(normal_norm > 1.0e-9, torch.acos(cosine), torch.full_like(cosine, torch.nan))
        active = stance[:, patch]
        hover_values.append((center[:, 2] - target.ground_height_m).clamp_min(0.0)[active])
        penetration_values.append((target.ground_height_m - points[..., 2]).clamp_min(0.0).amax(dim=1)[active])
        tilt_values.append(tilt[active])
        anchor = None
        previous_active = False
        previous_frame = None
        previous_center = None
        for row in range(center.shape[0]):
            frame = int(frame_indices[row])
            is_active = bool(active[row])
            contiguous = previous_frame is not None and frame == previous_frame + 1
            if is_active and (not previous_active or not contiguous):
                anchor = center[row, :2]
                interval_count += 1
            if is_active and previous_active and contiguous and previous_center is not None:
                slip_values.append(
                    (torch.linalg.vector_norm(center[row, :2] - previous_center[:2]) / source.frame_dt_s).reshape(1)
                )
            if is_active and anchor is not None:
                drift_values.append(torch.linalg.vector_norm(center[row, :2] - anchor).reshape(1))
            previous_active = is_active
            previous_frame = frame
            previous_center = center[row]
    empty = support.new_empty(0)
    values = (
        collision_penetration,
        torch.cat(hover_values) if hover_values else empty,
        torch.cat(penetration_values) if penetration_values else empty,
        torch.cat(tilt_values) if tilt_values else empty,
        torch.cat(slip_values) if slip_values else empty,
        torch.cat(drift_values) if drift_values else empty,
    )
    return (
        {
            "ground_height_m": target.ground_height_m,
            "collision_probe_frame_max_penetration_m": _statistics(values[0]),
            "source_stance_sample_count": int(stance.sum()),
            "stance_interval_count": interval_count,
            "stance_patch_hover_m": _statistics(values[1]),
            "stance_support_point_penetration_m": _statistics(values[2]),
            "stance_patch_tilt_rad": _statistics(values[3]),
            "stance_tangential_slip_speed_m_s": _statistics(values[4]),
            "stance_tangential_cumulative_drift_m": _statistics(values[5]),
        },
        values,
        interval_count,
    )


def _measure(
    source: RetargetEvaluationSourceClip,
    view: RetargetEvaluationView,
    target: RetargetEvaluationTarget,
) -> tuple[dict, _Measurements]:
    """Measure one source/output clip through one explicit target mechanics."""
    source_rows, output_rows, coverage = _paired_rows(source, view)
    fidelity, fidelity_values = _source_fidelity(source, view, source_rows, output_rows)
    hard, hard_values = _hard_constraints(view, target)
    contacts, contact_values, intervals = _contacts_ground(source, view, target, source_rows, output_rows)
    measurements = _Measurements(
        landmark_error_m=fidelity_values[0],
        root_relative_landmark_error_m=fidelity_values[1],
        root_z_error_m=fidelity_values[2],
        root_horizontal_path_ratio=fidelity_values[3],
        root_horizontal_displacement_ratio=fidelity_values[4],
        edge_angle_rad=fidelity_values[5],
        orientation_error_rad=fidelity_values[6],
        position_excess_rad=hard_values[0],
        position_margin_rad=hard_values[1],
        position_normalized_margin=hard_values[2],
        velocity_excess_rad_s=hard_values[3],
        velocity_utilization_ratio=hard_values[4],
        quaternion_norm_error=hard_values[5],
        collision_penetration_m=contact_values[0],
        stance_hover_m=contact_values[1],
        stance_support_penetration_m=contact_values[2],
        stance_tilt_rad=contact_values[3],
        stance_slip_speed_m_s=contact_values[4],
        stance_cumulative_drift_m=contact_values[5],
        stance_interval_count=intervals,
    )
    return {
        "coverage": coverage,
        "source_fidelity": fidelity,
        "hard_target_constraints": hard,
        "contacts_ground": contacts,
    }, measurements


def _concatenate(values: list[_Measurements], name: str) -> torch.Tensor:
    """Concatenate one sample family across exact clip boundaries."""
    tensors = [getattr(value, name).reshape(-1) for value in values]
    return torch.cat(tensors) if tensors else torch.empty(0, dtype=torch.float64)


def _aggregate(values: list[_Measurements]) -> dict:
    """Aggregate metric families without inventing one quality score."""
    position = _concatenate(values, "position_excess_rad")
    position_margin = _concatenate(values, "position_margin_rad")
    position_normalized_margin = _concatenate(values, "position_normalized_margin")
    velocity = _concatenate(values, "velocity_excess_rad_s")
    velocity_utilization = _concatenate(values, "velocity_utilization_ratio")
    return {
        "source_fidelity": {
            "landmark_error_m": _statistics(_concatenate(values, "landmark_error_m")),
            "root_relative_landmark_error_m": _statistics(
                _concatenate(values, "root_relative_landmark_error_m")
            ),
            "root_z_error_m": _statistics(_concatenate(values, "root_z_error_m")),
            "root_z_absolute_error_m": _statistics(_concatenate(values, "root_z_error_m").abs()),
            "root_horizontal_path_ratio": _statistics(_concatenate(values, "root_horizontal_path_ratio")),
            "root_horizontal_displacement_ratio": _statistics(
                _concatenate(values, "root_horizontal_displacement_ratio")
            ),
            "semantic_edge_angle_rad": _statistics(_concatenate(values, "edge_angle_rad")),
            "selected_orientation_error_rad": _statistics(_concatenate(values, "orientation_error_rad")),
        },
        "hard_target_constraints": {
            "position_excess_rad": _statistics(position),
            "position_violating_samples": int(torch.sum(torch.isfinite(position) & (position > 0.0))),
            "position_minimum_margin_rad": _margin_statistics(position_margin),
            "position_normalized_margin_of_interval": _margin_statistics(position_normalized_margin),
            "position_near_limit_tolerance_rad": _NEAR_POSITION_LIMIT_TOLERANCE_RAD,
            "position_near_limit_samples": int(
                torch.sum(torch.isfinite(position_margin) & (position_margin.abs() <= _NEAR_POSITION_LIMIT_TOLERANCE_RAD))
            ),
            "velocity_excess_rad_s": _statistics(velocity),
            "velocity_violating_samples": int(torch.sum(torch.isfinite(velocity) & (velocity > 0.0))),
            "velocity_utilization_ratio": _statistics(velocity_utilization),
            "selected_rotation_quaternion_norm_error": _statistics(
                _concatenate(values, "quaternion_norm_error")
            ),
        },
        "contacts_ground": {
            "collision_probe_frame_max_penetration_m": _statistics(
                _concatenate(values, "collision_penetration_m")
            ),
            "stance_interval_count": sum(value.stance_interval_count for value in values),
            "stance_patch_hover_m": _statistics(_concatenate(values, "stance_hover_m")),
            "stance_support_point_penetration_m": _statistics(
                _concatenate(values, "stance_support_penetration_m")
            ),
            "stance_patch_tilt_rad": _statistics(_concatenate(values, "stance_tilt_rad")),
            "stance_tangential_slip_speed_m_s": _statistics(
                _concatenate(values, "stance_slip_speed_m_s")
            ),
            "stance_tangential_cumulative_drift_m": _statistics(
                _concatenate(values, "stance_cumulative_drift_m")
            ),
        },
    }


def _runtime_report(runtime: RetargetEvaluationRuntime) -> dict:
    """Return declared runtime without mixing it into geometric quality."""
    return {
        "scope": runtime.scope,
        "included_stages": list(runtime.included_stages),
        "wall_seconds": runtime.wall_seconds,
        "input_frame_count": runtime.input_frame_count,
        "output_frame_count": runtime.output_frame_count,
        "input_frames_per_second": runtime.input_frame_count / runtime.wall_seconds,
        "output_frames_per_second": runtime.output_frame_count / runtime.wall_seconds,
        "device": runtime.device,
        "peak_incremental_bytes": runtime.peak_incremental_bytes,
    }


def evaluate_neutral_retargeting(
    source_clips: tuple[RetargetEvaluationSourceClip, ...],
    selected_target: RetargetEvaluationTarget,
    methods: tuple[RetargetEvaluationMethod, ...],
) -> dict:
    """Evaluate methods against common semantics with native and selected mechanics kept separate."""
    if not source_clips or len({clip.clip_id for clip in source_clips}) != len(source_clips):
        raise ValueError("Neutral evaluation requires unique nonempty source clip IDs.")
    if not methods or len({method.name for method in methods}) != len(methods):
        raise ValueError("Neutral evaluation requires unique nonempty method names.")
    if any(clip.selected_target_geometry_sha256 != selected_target.geometry_sha256 for clip in source_clips):
        raise ValueError("Every common source projection must name the selected target geometry.")
    projection_ids = {clip.semantic_projection_sha256 for clip in source_clips}
    source_ids = {clip.source_content_sha256 for clip in source_clips}
    if len(projection_ids) != 1 or len(source_ids) != 1:
        raise ValueError("One report must use one source corpus and one common semantic projection.")
    source_by_id = {clip.clip_id: clip for clip in source_clips}
    method_reports = {}
    runtime_scopes = set()
    for method in methods:
        native_by_id = {clip.clip_id: clip for clip in method.native_clips}
        selected_by_id = {clip.clip_id: clip for clip in method.selected_target_clips}
        available_ids = set(native_by_id) & set(selected_by_id)
        matched_ids = tuple(clip.clip_id for clip in source_clips if clip.clip_id in available_ids)
        if not matched_ids:
            raise ValueError(f"Method {method.name!r} has no source clip IDs in common with the benchmark.")
        native_measurements = []
        selected_measurements = []
        clip_reports = []
        for clip_id in matched_ids:
            source = source_by_id[clip_id]
            native = native_by_id[clip_id]
            selected = selected_by_id[clip_id]
            if (
                not torch.equal(native.frame_indices.detach().cpu(), selected.frame_indices.detach().cpu())
                or not math.isclose(native.frame_dt_s, selected.frame_dt_s, rel_tol=0.0, abs_tol=0.0)
            ):
                raise ValueError(f"Method {method.name!r} changes frames between native and selected mechanics.")
            native_report, native_values = _measure(source, native, method.native_target)
            selected_report, selected_values = _measure(source, selected, selected_target)
            native_measurements.append(native_values)
            selected_measurements.append(selected_values)
            clip_reports.append(
                {
                    "clip_id": clip_id,
                    "native_mechanics": native_report,
                    "selected_target_mechanics": selected_report,
                }
            )
        source_id_set = set(source_by_id)
        emitted_id_set = set(native_by_id)
        runtime_scopes.add((method.runtime.scope, method.runtime.included_stages))
        method_reports[method.name] = {
            "mechanics": {
                "native_geometry_sha256": method.native_target.geometry_sha256,
                "selected_target_geometry_sha256": selected_target.geometry_sha256,
                "same_geometry": method.native_target.geometry_sha256 == selected_target.geometry_sha256,
                "native_joint_names": list(method.native_target.joint_names),
                "selected_target_joint_names": list(selected_target.joint_names),
                "joint_order_equal": method.native_target.joint_names == selected_target.joint_names,
            },
            "corpus_coverage": {
                "source_clip_count": len(source_clips),
                "emitted_clip_count": len(method.native_clips),
                "evaluated_clip_count": len(matched_ids),
                "evaluated_fraction": len(matched_ids) / len(source_clips),
                "missing_clip_ids": sorted(source_id_set - emitted_id_set),
                "unexpected_clip_ids": sorted(emitted_id_set - source_id_set),
            },
            "aggregate": {
                "native_mechanics": _aggregate(native_measurements),
                "selected_target_mechanics": _aggregate(selected_measurements),
            },
            "clips": clip_reports,
            "runtime": _runtime_report(method.runtime),
        }
    return {
        "schema": _SCHEMA,
        "contract": {
            "source_reference": "one_immutable_raw_source_to_selected_target_semantic_projection",
            "source_content_sha256": next(iter(source_ids)),
            "semantic_projection_sha256": next(iter(projection_ids)),
            "selected_target_geometry_sha256": selected_target.geometry_sha256,
            "mechanics_views": {
                "native": "common-reference fidelity plus limits/contact geometry under method-native mechanics",
                "selected_target": "source fidelity and deployability reconstructed with the selected Isaac mechanics",
                "never_mixed": True,
            },
            "alignment": {
                "scope": "one_constant_transform_per_clip_and_mechanics_view",
                "fit": "least_squares_over_all_required_common_target_landmarks",
                "world_z_yaw": True,
                "horizontal_xy_translation": True,
                "vertical_z_translation": False,
                "scale": False,
                "per_frame_alignment": False,
            },
            "time": {
                "pairing": "exact_source_frame_index",
                "frame_period_absolute_tolerance_s": _FRAME_DT_ABSOLUTE_TOLERANCE_S,
                "resampling": False,
                "time_warp": False,
            },
            "ownership": {
                "semantic_target": "benchmark",
                "stance_mask": "benchmark_source",
                "joint_limits": "reported_mechanics_view",
                "support_and_collision_geometry": "reported_mechanics_view",
            },
            "quality_acceptance": "none_measurements_only",
            "aggregate_score": None,
        },
        "runtime_comparison": {
            "comparable": len(runtime_scopes) == 1,
            "reason": (
                "all methods declared identical scope and included stages"
                if len(runtime_scopes) == 1
                else "methods declared different runtime scopes or included stages"
            ),
        },
        "methods": method_reports,
    }
