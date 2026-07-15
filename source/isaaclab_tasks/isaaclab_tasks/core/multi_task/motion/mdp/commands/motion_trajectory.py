# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Whole-trajectory motion projection, IK solving, and quality evidence."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import newton.ik as ik
import numpy as np
import torch
import warp as wp

from isaaclab.utils.string import string_to_callable

from ....kinematics import (
    ContactFeasibilityWorkspace,
    IKObjectiveSupportPatch,
    IKTrajectorySolver,
    KinematicTree,
    plan_trajectory_memory,
    time_gradient_segmented,
)
from ....kinematics.ik_objectives.cfg import (
    IKObjectiveBaseCfg,
    IKObjectiveJointDefaultCfg,
    IKObjectiveJointPinCfg,
    IKObjectiveMeshCollisionCfg,
    IKObjectiveMeshNonpenetrationCfg,
)
from ....kinematics.ik_objectives.context import (
    IKJointPinObjectiveBuildContext,
    IKObjectiveBuild,
    IKObjectiveBuildContext,
    IKObjectiveMeshCollisionBuildContext,
)
from ....mdp.commands.state_command.state_command_cfg import StateCommandCfg
from ...data.clip_index import MotionClipIndex
from ...data.frames import (
    MotionGeneralizedCoordinates,
    MotionSourceProjectionTrajectory,
)
from ...retarget import MotionTrajectoryTargets
from ...robots.target import (
    _MOTION_COLLISION_PROBES_PER_BODY,
    MotionFrameTarget,
    validate_collision_probe_geometry,
    write_velocity_canonical,
)
from .commands_cfg import (
    MotionContactCriterionCfg,
    MotionContactObjectiveCfg,
    MotionSourceDirectionPointObjectiveCfg,
    MotionSourceEvidenceGenerateCfg,
    MotionSourceFidelityCriterionCfg,
    MotionSourceGlobalPositionObjectiveCfg,
    MotionSourceRotationObjectiveCfg,
    MotionTrajectorySolveCfg,
)
from .motion_task_table import (
    _DYNAMICS_QUALITY_START,
    _DYNAMICS_QUALITY_STOP,
    _QUALITY_ACCEPTED,
    _QUALITY_CONSTRAINT_GEOMETRY_FEASIBLE,
    _QUALITY_INNER_SOLVE_CONVERGED,
    _QUALITY_NAMES,
    _QUALITY_NONLINEAR_PHASES_CONVERGED,
    _QUALITY_NONLINEAR_REFINEMENT_REQUIRED,
    _QUALITY_TRAJECTORY_ROUTE,
    _TARGET_COORDINATE_QUALITY_NAMES,
    _TARGET_COORDINATE_QUALITY_START,
    _TARGET_COORDINATE_QUALITY_STOP,
    _TRAJECTORY_INSPECTION_CAPTURE_STAGE_NAMES,
    _TRAJECTORY_METRIC_NAMES,
    _TRAJECTORY_METRIC_START,
    _TRAJECTORY_METRIC_STOP,
)

if TYPE_CHECKING:
    from . import motion_task_table_builder as _builder


_TRAJECTORY_METRIC_COUNT = len(_TRAJECTORY_METRIC_NAMES)
_METRIC_SOURCE_REQUIRED_POSITION = _TRAJECTORY_METRIC_NAMES.index("source_required_position_max_m")
_METRIC_SOURCE_REQUIRED_DISTAL_POSITION = _TRAJECTORY_METRIC_NAMES.index("source_required_distal_position_max_m")
_METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION = _TRAJECTORY_METRIC_NAMES.index("source_required_distal_direction_max_rad")
_METRIC_SOURCE_ROOT_ROTATION = _TRAJECTORY_METRIC_NAMES.index("source_root_rotation_max_rad")
_METRIC_SOURCE_ALL_POSITION = _TRAJECTORY_METRIC_NAMES.index("source_all_position_max_m")
_METRIC_SOURCE_ALL_DISTAL_POSITION = _TRAJECTORY_METRIC_NAMES.index("source_all_distal_position_max_m")
_METRIC_SOURCE_ALL_LANDMARK_DIRECTION = _TRAJECTORY_METRIC_NAMES.index("source_all_landmark_direction_max_rad")
_METRIC_SOURCE_ALL_DISTAL_DIRECTION = _TRAJECTORY_METRIC_NAMES.index("source_all_distal_direction_max_rad")
_METRIC_SOURCE_NONROOT_ROTATION = _TRAJECTORY_METRIC_NAMES.index("source_nonroot_rotation_max_rad")
_METRIC_CONTACT_GAP = _TRAJECTORY_METRIC_NAMES.index("contact_gap_max_m")
_METRIC_CONTACT_TILT = _TRAJECTORY_METRIC_NAMES.index("contact_tilt_max_rad")
_METRIC_CONTACT_SLIP_SPEED = _TRAJECTORY_METRIC_NAMES.index("contact_slip_speed_max_mps")
_METRIC_CONTACT_CUMULATIVE_DRIFT = _TRAJECTORY_METRIC_NAMES.index("contact_cumulative_drift_max_m")
_METRIC_CONTACT_APPLICABLE = _TRAJECTORY_METRIC_NAMES.index("contact_applicable")
_METRIC_CONTACT_STABLE_COUNT = _TRAJECTORY_METRIC_NAMES.index("contact_stable_frame_channel_count")
_METRIC_SOURCE_CONTACT_CONFIDENCE = _TRAJECTORY_METRIC_NAMES.index("source_contact_confidence_mean")
_FRAME_SEED_GLOBAL_SEEDS = 64
_FRAME_SEED_GLOBAL_ITERATIONS = 200
_FRAME_SEED_LOCAL_ITERATIONS = 24
_FRAME_SEED_LOCAL_CANDIDATES = 2
_FRAME_SEED_GLOBAL_NOISE_STD = 0.1
_FRAME_SEED_GLOBAL_RNG_SEED = 12345
_TERMINAL_ACCEPT_CONSTRAINTS = 0
_TERMINAL_ACCEPT_SOURCE = 1
_TERMINAL_ACCEPT_SOURCE_CONTACT = 2
_SOLVER_RECOVERY_LIMIT = 12


@wp.kernel
def _motion_frame_seed_project(
    joint_q: wp.array2d(dtype=wp.float32),
    root_reference_joint_q: wp.array2d(dtype=wp.float32),
    coordinate_indices: wp.array(dtype=wp.int64),
    coordinate_lower: wp.array(dtype=wp.float32),
    coordinate_upper: wp.array(dtype=wp.float32),
    active_rows: int,
    coordinate_bound_count: int,
    seeds_per_problem: int,
    root_fixed: wp.uint8,
):
    """Project active frame seeds onto target root policy and coordinate bounds."""
    row, value = wp.tid()
    if row >= active_rows:
        return
    if value == 0:
        if root_fixed != wp.uint8(0):
            problem = row // seeds_per_problem
            for root_coordinate in range(7):
                joint_q[row, root_coordinate] = root_reference_joint_q[problem, root_coordinate]
        else:
            norm = wp.sqrt(
                joint_q[row, 3] * joint_q[row, 3]
                + joint_q[row, 4] * joint_q[row, 4]
                + joint_q[row, 5] * joint_q[row, 5]
                + joint_q[row, 6] * joint_q[row, 6]
            )
            if norm > 1.0e-9:
                inverse_norm = 1.0 / norm
                for axis in range(4):
                    joint_q[row, axis + 3] = joint_q[row, axis + 3] * inverse_norm
            else:
                joint_q[row, 3] = 0.0
                joint_q[row, 4] = 0.0
                joint_q[row, 5] = 0.0
                joint_q[row, 6] = 1.0
    if value < coordinate_bound_count:
        coordinate = coordinate_indices[value]
        joint_q[row, coordinate] = wp.clamp(joint_q[row, coordinate], coordinate_lower[value], coordinate_upper[value])


@wp.kernel
def _motion_frame_seed_global_gather(
    source_landmark_position_m: wp.array3d(dtype=wp.float32),
    source_landmark_rotation_xyzw: wp.array3d(dtype=wp.float32),
    source_direction_point_position_m: wp.array3d(dtype=wp.float32),
    source_initial_joint_q: wp.array2d(dtype=wp.float32),
    clip_offsets: wp.array(dtype=wp.int32),
    clip_count: int,
    position_count: int,
    rotation_count: int,
    direction_count: int,
    coordinate_count: int,
    target_landmark_position_m: wp.array3d(dtype=wp.float32),
    target_landmark_rotation_xyzw: wp.array3d(dtype=wp.float32),
    target_direction_point_position_m: wp.array3d(dtype=wp.float32),
    target_initial_joint_q: wp.array2d(dtype=wp.float32),
):
    """Gather the true first frame of every complete clip into the global IK batch."""
    clip = wp.tid()
    if clip >= clip_count:
        return
    frame = clip_offsets[clip]
    for position in range(position_count):
        for axis in range(3):
            target_landmark_position_m[position, clip, axis] = source_landmark_position_m[position, frame, axis]
    for rotation in range(rotation_count):
        for axis in range(4):
            target_landmark_rotation_xyzw[rotation, clip, axis] = source_landmark_rotation_xyzw[rotation, frame, axis]
    for direction in range(direction_count):
        for axis in range(3):
            target_direction_point_position_m[direction, clip, axis] = source_direction_point_position_m[
                direction, frame, axis
            ]
    for coordinate in range(coordinate_count):
        target_initial_joint_q[clip, coordinate] = source_initial_joint_q[frame, coordinate]


@wp.kernel
def _motion_frame_seed_global_scatter(
    global_joint_q: wp.array2d(dtype=wp.float32),
    clip_offsets: wp.array(dtype=wp.int32),
    clip_count: int,
    coordinate_count: int,
    joint_q: wp.array2d(dtype=wp.float32),
):
    """Scatter global IK results only to each clip's true first frame."""
    clip, coordinate = wp.tid()
    if clip < clip_count and coordinate < coordinate_count:
        joint_q[clip_offsets[clip], coordinate] = global_joint_q[clip, coordinate]


@wp.kernel
def _motion_frame_seed_local_gather(
    source_landmark_position_m: wp.array3d(dtype=wp.float32),
    source_landmark_rotation_xyzw: wp.array3d(dtype=wp.float32),
    source_direction_point_position_m: wp.array3d(dtype=wp.float32),
    selected_joint_q: wp.array2d(dtype=wp.float32),
    clip_offsets: wp.array(dtype=wp.int32),
    baseline_joint_q: wp.array2d(dtype=wp.float32),
    relative_frame: int,
    clip_count: int,
    position_count: int,
    rotation_count: int,
    direction_count: int,
    coordinate_count: int,
    target_landmark_position_m: wp.array3d(dtype=wp.float32),
    target_landmark_rotation_xyzw: wp.array3d(dtype=wp.float32),
    target_direction_point_position_m: wp.array3d(dtype=wp.float32),
    target_reference_joint_q: wp.array2d(dtype=wp.float32),
    target_candidate_joint_q: wp.array2d(dtype=wp.float32),
    frame_indices: wp.array(dtype=wp.int32),
    frame_active: wp.array(dtype=wp.int32),
):
    """Gather one frame's semantics, mapped reference, and two exact branch candidates."""
    clip = wp.tid()
    if clip >= clip_count:
        return
    start = clip_offsets[clip]
    stop = clip_offsets[clip + 1]
    frame = start + relative_frame
    active = int(frame < stop)
    if active == 0:
        frame = stop - 1
    previous_frame = frame - 1
    if active == 0:
        previous_frame = frame
    frame_indices[clip] = frame
    frame_active[clip] = active
    for position in range(position_count):
        for axis in range(3):
            target_landmark_position_m[position, clip, axis] = source_landmark_position_m[position, frame, axis]
    for rotation in range(rotation_count):
        for axis in range(4):
            target_landmark_rotation_xyzw[rotation, clip, axis] = source_landmark_rotation_xyzw[rotation, frame, axis]
    for direction in range(direction_count):
        for axis in range(3):
            target_direction_point_position_m[direction, clip, axis] = source_direction_point_position_m[
                direction, frame, axis
            ]
    for coordinate in range(coordinate_count):
        baseline = baseline_joint_q[frame, coordinate]
        target_reference_joint_q[clip, coordinate] = baseline
        target_candidate_joint_q[_FRAME_SEED_LOCAL_CANDIDATES * clip, coordinate] = selected_joint_q[
            previous_frame, coordinate
        ]
        target_candidate_joint_q[_FRAME_SEED_LOCAL_CANDIDATES * clip + 1, coordinate] = baseline


@wp.kernel
def _motion_frame_seed_local_scatter(
    candidate_joint_q: wp.array2d(dtype=wp.float32),
    candidate_cost: wp.array(dtype=wp.float32),
    baseline_joint_q: wp.array2d(dtype=wp.float32),
    frame_indices: wp.array(dtype=wp.int32),
    frame_active: wp.array(dtype=wp.int32),
    coordinate_indices: wp.array(dtype=wp.int64),
    coordinate_lower: wp.array(dtype=wp.float32),
    coordinate_upper: wp.array(dtype=wp.float32),
    clip_count: int,
    coordinate_count: int,
    coordinate_bound_count: int,
    root_fixed: wp.uint8,
    joint_q: wp.array2d(dtype=wp.float32),
):
    """Store the lowest-cost feasible branch, preferring continuation on exact ties."""
    clip = wp.tid()
    if clip >= clip_count or frame_active[clip] == 0:
        return
    frame = frame_indices[clip]
    selected = wp.int32(-1)
    selected_cost = wp.float32(0.0)
    for candidate in range(_FRAME_SEED_LOCAL_CANDIDATES):
        row = _FRAME_SEED_LOCAL_CANDIDATES * clip + candidate
        cost = candidate_cost[row]
        feasible = int(wp.isfinite(cost))
        for coordinate in range(coordinate_count):
            if not wp.isfinite(candidate_joint_q[row, coordinate]):
                feasible = 0
        for bound in range(coordinate_bound_count):
            bound_coordinate = coordinate_indices[bound]
            value = candidate_joint_q[row, bound_coordinate]
            if value < coordinate_lower[bound] or value > coordinate_upper[bound]:
                feasible = 0
        if root_fixed != wp.uint8(0):
            for coordinate in range(7):
                if wp.abs(candidate_joint_q[row, coordinate] - baseline_joint_q[frame, coordinate]) > 1.0e-6:
                    feasible = 0
        else:
            quaternion_norm = wp.sqrt(
                candidate_joint_q[row, 3] * candidate_joint_q[row, 3]
                + candidate_joint_q[row, 4] * candidate_joint_q[row, 4]
                + candidate_joint_q[row, 5] * candidate_joint_q[row, 5]
                + candidate_joint_q[row, 6] * candidate_joint_q[row, 6]
            )
            if wp.abs(quaternion_norm - 1.0) > 1.0e-5:
                feasible = 0
        if feasible != 0 and (selected < 0 or cost < selected_cost):
            selected = wp.int32(candidate)
            selected_cost = cost
    if selected < 0:
        for coordinate in range(coordinate_count):
            joint_q[frame, coordinate] = baseline_joint_q[frame, coordinate]
    else:
        row = _FRAME_SEED_LOCAL_CANDIDATES * clip + selected
        for coordinate in range(coordinate_count):
            joint_q[frame, coordinate] = candidate_joint_q[row, coordinate]


@wp.kernel
def _motion_initializer_validate_or_restore(
    joint_q: wp.array2d(dtype=wp.float32),
    baseline_joint_q: wp.array2d(dtype=wp.float32),
    joint_qd: wp.array2d(dtype=wp.float32),
    coordinate_indices: wp.array(dtype=wp.int64),
    coordinate_lower: wp.array(dtype=wp.float32),
    coordinate_upper: wp.array(dtype=wp.float32),
    segment_offsets: wp.array(dtype=wp.int32),
    segment_count: int,
    coordinate_count: int,
    dof_count: int,
    coordinate_bound_count: int,
    segment_valid: wp.array(dtype=wp.int32),
):
    """Keep finite target-selected initializers inside coordinate bounds before trajectory fitting."""
    segment = wp.tid()
    if segment >= segment_count:
        return
    start = segment_offsets[segment]
    stop = segment_offsets[segment + 1]
    valid = int(segment_valid[segment] != 0)
    for frame in range(start, stop):
        for coordinate in range(coordinate_count):
            if not wp.isfinite(joint_q[frame, coordinate]):
                valid = 0
        for bound in range(coordinate_bound_count):
            bound_coordinate = coordinate_indices[bound]
            value = joint_q[frame, bound_coordinate]
            if value < coordinate_lower[bound] or value > coordinate_upper[bound]:
                valid = 0
        for dof in range(dof_count):
            value = joint_qd[frame, dof]
            if not wp.isfinite(value):
                valid = 0
    segment_valid[segment] = valid
    if valid == 0:
        for frame in range(start, stop):
            for coordinate in range(coordinate_count):
                joint_q[frame, coordinate] = baseline_joint_q[frame, coordinate]


@wp.kernel
def _motion_phase_copy_selected(
    source_joint_q: wp.array2d(dtype=wp.float32),
    segment_offsets: wp.array(dtype=wp.int32),
    segment_selected: wp.array(dtype=wp.int32),
    segment_count: int,
    coordinate_count: int,
    destination_joint_q: wp.array2d(dtype=wp.float32),
):
    """Copy selected complete clips between explicit trajectory states."""
    segment, coordinate = wp.tid()
    if segment >= segment_count or coordinate >= coordinate_count or segment_selected[segment] == 0:
        return
    start = segment_offsets[segment]
    stop = segment_offsets[segment + 1]
    for frame in range(start, stop):
        destination_joint_q[frame, coordinate] = source_joint_q[frame, coordinate]


@wp.kernel
def _motion_solver_recovery_update(
    iteration_attempted: wp.array(dtype=wp.int32),
    iteration_geometry_feasible: wp.array(dtype=wp.bool),
    iteration_inner_converged: wp.array(dtype=wp.bool),
    iteration_globalization_succeeded: wp.array(dtype=wp.bool),
    iteration_residual_constraints_satisfied: wp.array(dtype=wp.bool),
    recovery_limit: int,
    segment_active: wp.array(dtype=wp.int32),
    recovery_count: wp.array(dtype=wp.int32),
    segment_damping: wp.array(dtype=wp.float32),
):
    """Retry only geometrically feasible direction or globalization failures."""
    segment = wp.tid()
    if iteration_attempted[segment] == 0:
        return
    if not iteration_geometry_feasible[segment]:
        segment_active[segment] = 0
        return
    if iteration_inner_converged[segment] and iteration_globalization_succeeded[segment]:
        return

    iteration_residual_constraints_satisfied[segment] = False
    if recovery_count[segment] >= recovery_limit:
        segment_active[segment] = 0
        return

    recovery_count[segment] = recovery_count[segment] + 1
    segment_damping[segment] = 2.0 * segment_damping[segment]
    segment_active[segment] = 1
    iteration_inner_converged[segment] = True
    iteration_globalization_succeeded[segment] = True


@wp.kernel
def _motion_scalar_velocity_box_witness(
    joint_q: wp.array2d(dtype=wp.float32),
    coordinate_indices: wp.array(dtype=wp.int32),
    dof_indices: wp.array(dtype=wp.int32),
    coordinate_lower: wp.array(dtype=wp.float32),
    coordinate_upper: wp.array(dtype=wp.float32),
    velocity_lower: wp.array(dtype=wp.float32),
    velocity_upper: wp.array(dtype=wp.float32),
    step_seconds: wp.array(dtype=wp.float32),
    segment_offsets: wp.array(dtype=wp.int32),
    segment_active: wp.array(dtype=wp.int32),
    segment_count: int,
    bound_count: int,
    reachable_lower: wp.array2d(dtype=wp.float32),
    reachable_upper: wp.array2d(dtype=wp.float32),
):
    """Write an exact scalar-coordinate box and velocity-feasible trajectory witness."""
    segment, bound = wp.tid()
    if segment >= segment_count or bound >= bound_count or segment_active[segment] == 0:
        return
    start = segment_offsets[segment]
    stop = segment_offsets[segment + 1]
    coordinate = coordinate_indices[bound]
    dof = dof_indices[bound]
    lower_bound = coordinate_lower[bound]
    upper_bound = coordinate_upper[bound]
    lower_rate = velocity_lower[dof]
    upper_rate = velocity_upper[dof]
    dt = step_seconds[segment]

    last = stop - 1
    reachable_lower[last, bound] = lower_bound
    reachable_upper[last, bound] = upper_bound
    for reverse in range(stop - start - 1):
        frame = stop - 2 - reverse
        lower = lower_bound
        upper = upper_bound
        if wp.isfinite(upper_rate):
            lower = wp.max(lower, reachable_lower[frame + 1, bound] - upper_rate * dt)
        if wp.isfinite(lower_rate):
            upper = wp.min(upper, reachable_upper[frame + 1, bound] - lower_rate * dt)
        if lower > upper:
            wp.atomic_min(segment_active, segment, 0)
            return
        reachable_lower[frame, bound] = lower
        reachable_upper[frame, bound] = upper

    value = wp.clamp(joint_q[start, coordinate], reachable_lower[start, bound], reachable_upper[start, bound])
    joint_q[start, coordinate] = value
    for offset in range(1, stop - start):
        frame = start + offset
        lower = reachable_lower[frame, bound]
        upper = reachable_upper[frame, bound]
        if wp.isfinite(lower_rate):
            lower = wp.max(lower, value + lower_rate * dt)
        if wp.isfinite(upper_rate):
            upper = wp.min(upper, value + upper_rate * dt)
        if lower > upper:
            wp.atomic_min(segment_active, segment, 0)
            return
        value = wp.clamp(joint_q[frame, coordinate], lower, upper)
        joint_q[frame, coordinate] = value


@wp.kernel
def _motion_terminal_acceptance_update(
    quality: wp.array2d(dtype=wp.float32),
    phase_attempted: wp.array(dtype=wp.bool),
    constraint_geometry_feasible: wp.array(dtype=wp.bool),
    residual_constraints_satisfied: wp.array(dtype=wp.bool),
    acceptance_mode: wp.uint8,
    clip_count: int,
    metric_required_position: int,
    metric_required_distal_position: int,
    metric_required_distal_direction: int,
    metric_root_rotation: int,
    metric_contact_gap: int,
    metric_contact_tilt: int,
    metric_contact_slip_speed: int,
    metric_contact_cumulative_drift: int,
    metric_contact_applicable: int,
    metric_contact_stable_count: int,
    metric_source_contact_confidence: int,
    required_position_upper_m: float,
    required_distal_position_upper_m: float,
    required_distal_direction_upper_rad: float,
    root_rotation_upper_rad: float,
    contact_gap_upper_m: float,
    contact_tilt_upper_rad: float,
    contact_slip_speed_upper_mps: float,
    contact_cumulative_drift_upper_m: float,
    terminal_accepted: wp.array(dtype=wp.bool),
    segment_active: wp.array(dtype=wp.int32),
):
    """Freeze the first phase iterate satisfying its explicit phase acceptance scope."""
    clip = wp.tid()
    if clip >= clip_count:
        return
    if terminal_accepted[clip]:
        segment_active[clip] = 0
        return
    if not phase_attempted[clip] or not constraint_geometry_feasible[clip] or not residual_constraints_satisfied[clip]:
        return

    if acceptance_mode == wp.uint8(_TERMINAL_ACCEPT_CONSTRAINTS):
        terminal_accepted[clip] = True
        segment_active[clip] = 0
        return
    required_position = quality[clip, metric_required_position]
    required_distal_position = quality[clip, metric_required_distal_position]
    required_distal_direction = quality[clip, metric_required_distal_direction]
    root_rotation = quality[clip, metric_root_rotation]
    source_valid = (
        wp.isfinite(required_position)
        and required_position >= 0.0
        and required_position <= required_position_upper_m
        and wp.isfinite(required_distal_position)
        and required_distal_position >= 0.0
        and required_distal_position <= required_distal_position_upper_m
        and wp.isfinite(required_distal_direction)
        and required_distal_direction >= 0.0
        and required_distal_direction <= required_distal_direction_upper_rad
        and wp.isfinite(root_rotation)
        and root_rotation >= 0.0
        and root_rotation <= root_rotation_upper_rad
    )
    if acceptance_mode == wp.uint8(_TERMINAL_ACCEPT_SOURCE):
        if source_valid:
            terminal_accepted[clip] = True
            segment_active[clip] = 0
        return

    gap = quality[clip, metric_contact_gap]
    tilt = quality[clip, metric_contact_tilt]
    slip_speed = quality[clip, metric_contact_slip_speed]
    cumulative_drift = quality[clip, metric_contact_cumulative_drift]
    applicable = quality[clip, metric_contact_applicable]
    stable_count = quality[clip, metric_contact_stable_count]
    confidence = quality[clip, metric_source_contact_confidence]
    metadata_valid = (
        wp.isfinite(applicable)
        and (applicable == 0.0 or applicable == 1.0)
        and wp.isfinite(stable_count)
        and stable_count >= 0.0
        and stable_count == wp.floor(stable_count)
        and ((applicable > 0.5) == (stable_count > 0.0))
        and wp.isfinite(confidence)
        and confidence >= 0.0
        and confidence <= 1.0
    )
    applicable_valid = (
        wp.isfinite(gap)
        and gap >= 0.0
        and gap <= contact_gap_upper_m
        and wp.isfinite(tilt)
        and tilt >= 0.0
        and tilt <= contact_tilt_upper_rad
        and wp.isfinite(slip_speed)
        and slip_speed >= 0.0
        and slip_speed <= contact_slip_speed_upper_mps
        and wp.isfinite(cumulative_drift)
        and cumulative_drift >= 0.0
        and cumulative_drift <= contact_cumulative_drift_upper_m
    )
    not_applicable_valid = (
        gap != gap and tilt != tilt and slip_speed != slip_speed and cumulative_drift != cumulative_drift
    )
    contact_valid = metadata_valid and (
        (applicable > 0.5 and applicable_valid) or (applicable <= 0.5 and not_applicable_valid)
    )
    if source_valid and contact_valid:
        terminal_accepted[clip] = True
        segment_active[clip] = 0


@wp.kernel
def _motion_contact_source_plane(
    source_probe_position_m: wp.array3d(dtype=wp.float32),
    clip_offsets: wp.array(dtype=wp.int32),
    probe_count: int,
    source_plane_height_m: wp.array(dtype=wp.float32),
):
    """Measure one source-relative support plane [m] per clip."""
    clip = wp.tid()
    start = clip_offsets[clip]
    stop = clip_offsets[clip + 1]
    plane = float(wp.inf)
    for probe in range(probe_count):
        for frame in range(start, stop):
            plane = wp.min(plane, source_probe_position_m[probe, frame, 2])
    source_plane_height_m[clip] = plane


@wp.kernel
def _motion_contact_probe_evidence_infer(
    source_probe_position_m: wp.array3d(dtype=wp.float32),
    channel_probe_offsets: wp.array(dtype=wp.int32),
    channel_count: int,
    clip_offsets: wp.array(dtype=wp.int32),
    step_seconds: wp.array(dtype=wp.float32),
    source_plane_height_m: wp.array(dtype=wp.float32),
    enter_height_m: float,
    exit_height_m: float,
    enter_speed_mps: float,
    exit_speed_mps: float,
    persistence_seconds: float,
    probe_active: wp.array2d(dtype=wp.uint8),
    probe_stable: wp.array2d(dtype=wp.uint8),
):
    """Infer ankle-or-toe source proximity with per-probe hysteretic speed evidence."""
    clip, probe = wp.tid()
    channel = int(-1)
    for candidate in range(channel_count):
        if probe >= channel_probe_offsets[candidate] and probe < channel_probe_offsets[candidate + 1]:
            channel = candidate
    if channel < 0:
        return
    channel_probe_start = channel_probe_offsets[channel]
    channel_probe_stop = channel_probe_offsets[channel + 1]
    start = clip_offsets[clip]
    stop = clip_offsets[clip + 1]
    dt = step_seconds[clip]
    persistence_frames = int(wp.ceil(persistence_seconds / dt))
    if persistence_frames < 1:
        persistence_frames = 1
    plane = source_plane_height_m[clip]
    active = wp.uint8(0)
    enter_start = int(start)
    enter_count = int(0)
    for frame in range(start, stop):
        position = wp.vec3(
            source_probe_position_m[probe, frame, 0],
            source_probe_position_m[probe, frame, 1],
            source_probe_position_m[probe, frame, 2],
        )
        velocity = wp.vec3(0.0, 0.0, 0.0)
        if stop - start > 1:
            if frame == start:
                velocity = (
                    wp.vec3(
                        source_probe_position_m[probe, frame + 1, 0],
                        source_probe_position_m[probe, frame + 1, 1],
                        source_probe_position_m[probe, frame + 1, 2],
                    )
                    - position
                ) / dt
            elif frame == stop - 1:
                velocity = (
                    position
                    - wp.vec3(
                        source_probe_position_m[probe, frame - 1, 0],
                        source_probe_position_m[probe, frame - 1, 1],
                        source_probe_position_m[probe, frame - 1, 2],
                    )
                ) / dt
            else:
                velocity = (
                    wp.vec3(
                        source_probe_position_m[probe, frame + 1, 0],
                        source_probe_position_m[probe, frame + 1, 1],
                        source_probe_position_m[probe, frame + 1, 2],
                    )
                    - wp.vec3(
                        source_probe_position_m[probe, frame - 1, 0],
                        source_probe_position_m[probe, frame - 1, 1],
                        source_probe_position_m[probe, frame - 1, 2],
                    )
                ) / (2.0 * dt)
        channel_height = float(wp.inf)
        for channel_probe in range(channel_probe_start, channel_probe_stop):
            channel_height = wp.min(channel_height, source_probe_position_m[channel_probe, frame, 2])
        # Source height contact is ankle OR toe; speed remains probe-local so planted evidence still requires rigidity.
        height = channel_height - plane
        speed = wp.length(velocity)
        probe_stable[frame, probe] = (
            wp.uint8(1) if height <= enter_height_m and speed <= enter_speed_mps else wp.uint8(0)
        )
        if active != wp.uint8(0):
            if height <= exit_height_m and speed <= exit_speed_mps:
                probe_active[frame, probe] = wp.uint8(1)
            else:
                active = wp.uint8(0)
                enter_count = 0
                probe_active[frame, probe] = wp.uint8(0)
        else:
            probe_active[frame, probe] = wp.uint8(0)
            if height <= enter_height_m and speed <= enter_speed_mps:
                if enter_count == 0:
                    enter_start = frame
                enter_count += 1
                if enter_count >= persistence_frames:
                    active = wp.uint8(1)
                    for onset_frame in range(enter_start, frame + 1):
                        probe_active[onset_frame, probe] = wp.uint8(1)
            else:
                enter_count = 0


@wp.kernel
def _motion_contact_channel_aggregate(
    source_probe_position_m: wp.array3d(dtype=wp.float32),
    probe_active: wp.array2d(dtype=wp.uint8),
    probe_stable: wp.array2d(dtype=wp.uint8),
    channel_probe_offsets: wp.array(dtype=wp.int32),
    clip_offsets: wp.array(dtype=wp.int32),
    step_seconds: wp.array(dtype=wp.float32),
    confidence_window_seconds: float,
    enter_speed_mps: float,
    channel_confidence: wp.array2d(dtype=wp.float32),
    channel_stable: wp.array2d(dtype=wp.uint8),
    channel_edge_stable: wp.array2d(dtype=wp.uint8),
):
    """Aggregate centered confidence and stability plus strict backward source edges."""
    clip, channel = wp.tid()
    start = clip_offsets[clip]
    stop = clip_offsets[clip + 1]
    probe_start = channel_probe_offsets[channel]
    probe_stop = channel_probe_offsets[channel + 1]
    dt = step_seconds[clip]
    radius = int(wp.floor(0.5 * confidence_window_seconds / dt + 1.0e-6))
    for frame in range(start, stop):
        window_start = wp.max(start, frame - radius)
        window_stop = wp.min(stop, frame + radius + 1)
        active = int(0)
        planted = wp.uint8(1)
        samples = (probe_stop - probe_start) * (window_stop - window_start)
        for probe in range(probe_start, probe_stop):
            for sample in range(window_start, window_stop):
                active += int(probe_active[sample, probe])
                if probe_active[sample, probe] == wp.uint8(0) or probe_stable[sample, probe] == wp.uint8(0):
                    planted = wp.uint8(0)
        channel_confidence[frame, channel] = float(active) / float(samples)
        channel_stable[frame, channel] = planted
        channel_edge_stable[frame, channel] = wp.uint8(0)

    for frame in range(start + 1, stop):
        if channel_stable[frame - 1, channel] != wp.uint8(0) and channel_stable[frame, channel] != wp.uint8(0):
            valid = wp.uint8(1)
            for probe in range(probe_start, probe_stop):
                current = wp.vec3(
                    source_probe_position_m[probe, frame, 0],
                    source_probe_position_m[probe, frame, 1],
                    source_probe_position_m[probe, frame, 2],
                )
                previous = wp.vec3(
                    source_probe_position_m[probe, frame - 1, 0],
                    source_probe_position_m[probe, frame - 1, 1],
                    source_probe_position_m[probe, frame - 1, 2],
                )
                if wp.length(current - previous) / dt > enter_speed_mps:
                    valid = wp.uint8(0)
            channel_edge_stable[frame, channel] = valid


@wp.kernel
def _motion_contact_activity(
    channel_stable: wp.array2d(dtype=wp.uint8),
    channel_edge_stable: wp.array2d(dtype=wp.uint8),
    channel_confidence: wp.array2d(dtype=wp.float32),
    channel_clearance_lift_m: wp.array2d(dtype=wp.float32),
    frame_count: int,
    channel_count: int,
    channel_normal_owned: wp.array2d(dtype=wp.float32),
    channel_activity: wp.array2d(dtype=wp.float32),
):
    """Write acceptance-normalized vertex and end-indexed edge activity."""
    frame, channel = wp.tid()
    if frame >= frame_count or channel >= channel_count:
        return
    confidence = channel_confidence[frame, channel]
    vertex_activity = confidence * confidence
    if channel_clearance_lift_m[frame, channel] > 0.0:
        vertex_activity = 1.0
    normal_owned = 1.0 if vertex_activity > 0.0 else 0.0
    channel_normal_owned[frame, channel] = normal_owned
    channel_activity[frame, channel] = vertex_activity
    channel_activity[frame, channel_count + channel] = (
        1.0 if channel_edge_stable[frame, channel] != wp.uint8(0) else 0.0
    )


def _motion_infer_contact_evidence(
    cfg: MotionTrajectorySolveCfg.ContactCfg,
    targets: MotionTrajectoryTargets,
    clip_offsets: torch.Tensor,
    step_seconds: torch.Tensor,
    source_plane_height_m: torch.Tensor,
    source_probe_active: torch.Tensor,
    source_probe_stable: torch.Tensor,
    source_channel_confidence: torch.Tensor,
    source_channel_stable: torch.Tensor,
    source_channel_edge_stable: torch.Tensor,
) -> None:
    """Infer source-owned contact evidence into caller-owned fixed buffers."""
    frame_capacity = targets.source_contact_probe_position_m.shape[1]
    probe_count = targets.source_contact_probe_position_m.shape[0]
    channel_count = targets.contact_channel_probe_offsets.numel() - 1
    clip_count = clip_offsets.numel() - 1
    device = targets.source_contact_probe_position_m.device
    if (
        targets.source_contact_probe_position_m.shape != (probe_count, frame_capacity, 3)
        or clip_count < 1
        or probe_count < 1
        or channel_count < 1
        or clip_offsets.shape != (clip_count + 1,)
        or step_seconds.shape != (clip_count,)
        or source_plane_height_m.shape[0] < clip_count
        or source_probe_active.shape != (frame_capacity, probe_count)
        or source_probe_stable.shape != source_probe_active.shape
        or source_channel_confidence.shape != (frame_capacity, channel_count)
        or source_channel_stable.shape != source_channel_confidence.shape
        or source_channel_edge_stable.shape != source_channel_confidence.shape
        or clip_offsets.dtype is not torch.int32
        or step_seconds.dtype is not torch.float32
        or source_plane_height_m.dtype is not torch.float32
        or source_probe_active.dtype is not torch.uint8
        or source_probe_stable.dtype is not torch.uint8
        or source_channel_confidence.dtype is not torch.float32
        or source_channel_stable.dtype is not torch.uint8
        or source_channel_edge_stable.dtype is not torch.uint8
        or any(
            tensor.device != device
            for tensor in (
                clip_offsets,
                step_seconds,
                source_plane_height_m,
                source_probe_active,
                source_probe_stable,
                source_channel_confidence,
                source_channel_stable,
                source_channel_edge_stable,
            )
        )
    ):
        raise ValueError("Motion contact inference requires aligned source evidence and caller-owned buffers.")
    wp.launch(
        _motion_contact_source_plane,
        dim=clip_count,
        inputs=[
            wp.from_torch(targets.source_contact_probe_position_m),
            wp.from_torch(clip_offsets),
            probe_count,
        ],
        outputs=[wp.from_torch(source_plane_height_m)],
        device=str(device),
    )
    wp.launch(
        _motion_contact_probe_evidence_infer,
        dim=(clip_count, probe_count),
        inputs=[
            wp.from_torch(targets.source_contact_probe_position_m),
            wp.from_torch(targets.contact_channel_probe_offsets),
            channel_count,
            wp.from_torch(clip_offsets),
            wp.from_torch(step_seconds),
            wp.from_torch(source_plane_height_m),
            cfg.enter_height_m,
            cfg.exit_height_m,
            cfg.enter_speed_mps,
            cfg.exit_speed_mps,
            cfg.persistence_seconds,
        ],
        outputs=[wp.from_torch(source_probe_active), wp.from_torch(source_probe_stable)],
        device=str(device),
    )
    wp.launch(
        _motion_contact_channel_aggregate,
        dim=(clip_count, channel_count),
        inputs=[
            wp.from_torch(targets.source_contact_probe_position_m),
            wp.from_torch(source_probe_active),
            wp.from_torch(source_probe_stable),
            wp.from_torch(targets.contact_channel_probe_offsets),
            wp.from_torch(clip_offsets),
            wp.from_torch(step_seconds),
            cfg.confidence_window_seconds,
            cfg.enter_speed_mps,
        ],
        outputs=[
            wp.from_torch(source_channel_confidence),
            wp.from_torch(source_channel_stable),
            wp.from_torch(source_channel_edge_stable),
        ],
        device=str(device),
    )


@wp.kernel
def _motion_target_ground_gauge(
    obstacle_pose: wp.array2d(dtype=wp.float32),
    body_q: wp.array3d(dtype=wp.float32),
    contact_body_indices: wp.array(dtype=wp.int64),
    support_point_body_m: wp.array2d(dtype=wp.float32),
    support_channel_slots: wp.array(dtype=wp.int64),
    source_channel_confidence: wp.array2d(dtype=wp.float32),
    source_channel_stable: wp.array2d(dtype=wp.uint8),
    clip_offsets: wp.array(dtype=wp.int32),
    segment_active: wp.array(dtype=wp.int32),
    segment_count: int,
    channel_count: int,
    landmark_count: int,
    direction_count: int,
    support_count: int,
    minimum_clearance_m: float,
    joint_q: wp.array2d(dtype=wp.float32),
    source_landmark_position_m: wp.array3d(dtype=wp.float32),
    source_direction_point_position_m: wp.array3d(dtype=wp.float32),
):
    """Align one signed clip gauge to confidence-weighted stable planar support."""
    segment = wp.tid()
    if segment >= segment_count or segment_active[segment] == 0:
        return
    start = clip_offsets[segment]
    stop = clip_offsets[segment + 1]

    shift_sum = float(0.0)
    weight_sum = float(0.0)
    valid = wp.uint8(1)
    for frame in range(start, stop):
        ground_height = obstacle_pose[frame, 2]
        if not wp.isfinite(ground_height):
            valid = wp.uint8(0)
        for channel in range(channel_count):
            if source_channel_stable[frame, channel] == wp.uint8(0):
                continue
            confidence = source_channel_confidence[frame, channel]
            body = contact_body_indices[channel]
            position = wp.vec3(
                body_q[frame, body, 0],
                body_q[frame, body, 1],
                body_q[frame, body, 2],
            )
            rotation = wp.quat(
                body_q[frame, body, 3], body_q[frame, body, 4], body_q[frame, body, 5], body_q[frame, body, 6]
            )
            minimum_height = float(wp.inf)
            point_count = int(0)
            for point in range(support_count):
                if support_channel_slots[point] == channel:
                    offset = wp.vec3(
                        support_point_body_m[point, 0],
                        support_point_body_m[point, 1],
                        support_point_body_m[point, 2],
                    )
                    support_position = position + wp.quat_rotate(rotation, offset)
                    minimum_height = wp.min(minimum_height, support_position[2])
                    point_count += 1
            if (
                not wp.isfinite(confidence)
                or confidence <= 0.0
                or not wp.isfinite(position[2])
                or not wp.isfinite(minimum_height)
                or point_count < 1
            ):
                valid = wp.uint8(0)
            else:
                required_shift = ground_height + minimum_clearance_m - minimum_height
                if wp.isfinite(required_shift):
                    shift_sum += confidence * required_shift
                    weight_sum += confidence
                else:
                    valid = wp.uint8(0)
    if valid == wp.uint8(0):
        segment_active[segment] = 0
        return
    if weight_sum <= 0.0:
        return
    shift = shift_sum / weight_sum
    if not wp.isfinite(shift):
        segment_active[segment] = 0
        return
    for frame in range(start, stop):
        joint_q[frame, 2] += shift
        for landmark in range(landmark_count):
            source_landmark_position_m[landmark, frame, 2] += shift
        for direction in range(direction_count):
            source_direction_point_position_m[direction, frame, 2] += shift


@wp.kernel
def _motion_clearance_lift(
    body_q: wp.array3d(dtype=wp.float32),
    obstacle_pose: wp.array2d(dtype=wp.float32),
    probe_bodies: wp.array(dtype=wp.int64),
    probe_offsets_m: wp.array2d(dtype=wp.float32),
    probe_normal_slots: wp.array(dtype=wp.int64),
    frame_count: int,
    channel_count: int,
    probe_count: int,
    accepted_clearance_m: float,
    target_clearance_m: float,
    channel_clearance_lift_m: wp.array2d(dtype=wp.float32),
):
    """Measure rigid normal lift from every probe in one contact-owned body chain."""
    frame, channel = wp.tid()
    if frame >= frame_count or channel >= channel_count:
        return
    origin = wp.vec3(obstacle_pose[frame, 0], obstacle_pose[frame, 1], obstacle_pose[frame, 2])
    support_rotation = wp.quat(
        obstacle_pose[frame, 3],
        obstacle_pose[frame, 4],
        obstacle_pose[frame, 5],
        obstacle_pose[frame, 6],
    )
    rotation_length = wp.sqrt(wp.dot(support_rotation, support_rotation))
    valid = (
        wp.isfinite(origin[0])
        and wp.isfinite(origin[1])
        and wp.isfinite(origin[2])
        and wp.isfinite(rotation_length)
        and rotation_length > 1.0e-8
        and wp.isfinite(accepted_clearance_m)
        and wp.isfinite(target_clearance_m)
        and target_clearance_m >= accepted_clearance_m
    )
    if valid:
        support_rotation = wp.quat(
            support_rotation[0] / rotation_length,
            support_rotation[1] / rotation_length,
            support_rotation[2] / rotation_length,
            support_rotation[3] / rotation_length,
        )
    support_inverse = wp.quat_inverse(support_rotation)
    minimum_height = float(wp.inf)
    owned_probe_count = int(0)
    for probe in range(probe_count):
        if probe_normal_slots[probe] != channel:
            continue
        body = probe_bodies[probe]
        position = wp.vec3(body_q[frame, body, 0], body_q[frame, body, 1], body_q[frame, body, 2])
        rotation = wp.quat(
            body_q[frame, body, 3],
            body_q[frame, body, 4],
            body_q[frame, body, 5],
            body_q[frame, body, 6],
        )
        offset = wp.vec3(probe_offsets_m[probe, 0], probe_offsets_m[probe, 1], probe_offsets_m[probe, 2])
        point_world = position + wp.quat_rotate(rotation, offset)
        point_support = wp.quat_rotate(support_inverse, point_world - origin)
        if not wp.isfinite(point_support[0]) or not wp.isfinite(point_support[1]) or not wp.isfinite(point_support[2]):
            valid = False
        minimum_height = wp.min(minimum_height, point_support[2])
        owned_probe_count += 1
    if not valid or owned_probe_count < 1 or not wp.isfinite(minimum_height):
        channel_clearance_lift_m[frame, channel] = float(wp.inf)
    elif minimum_height < accepted_clearance_m:
        channel_clearance_lift_m[frame, channel] = target_clearance_m - minimum_height
    else:
        channel_clearance_lift_m[frame, channel] = 0.0


@wp.kernel
def _motion_normal_ownership_segments(
    channel_normal_owned: wp.array2d(dtype=wp.float32),
    channel_stable: wp.array2d(dtype=wp.uint8),
    channel_clearance_lift_m: wp.array2d(dtype=wp.float32),
    clip_offsets: wp.array(dtype=wp.int32),
    segment_active: wp.array(dtype=wp.uint8),
    segment_count: int,
    channel_count: int,
    segment_refinement_required: wp.array(dtype=wp.uint8),
):
    """Force refinement for transitions and any contact-owned clearance lift."""
    segment = wp.tid()
    if segment >= segment_count or segment_active[segment] == wp.uint8(0):
        segment_refinement_required[segment] = wp.uint8(0)
        return
    required = wp.uint8(0)
    for frame in range(clip_offsets[segment], clip_offsets[segment + 1]):
        for channel in range(channel_count):
            transition_owned = channel_normal_owned[frame, channel] > 0.0 and channel_stable[
                frame, channel
            ] == wp.uint8(0)
            clearance_owned = channel_clearance_lift_m[frame, channel] > 0.0
            if transition_owned or clearance_owned:
                required = wp.uint8(1)
    segment_refinement_required[segment] = required


@wp.kernel
def _motion_contact_interval_targets(
    obstacle_pose: wp.array2d(dtype=wp.float32),
    source_landmark_position_m: wp.array3d(dtype=wp.float32),
    source_direction_point_position_m: wp.array3d(dtype=wp.float32),
    contact_direction_rows: wp.array(dtype=wp.int64),
    direction_position_rows: wp.array(dtype=wp.int64),
    contact_forward_body: wp.array2d(dtype=wp.float32),
    contact_normal_body: wp.array2d(dtype=wp.float32),
    support_point_body_m: wp.array2d(dtype=wp.float32),
    support_channel_slots: wp.array(dtype=wp.int64),
    channel_confidence: wp.array2d(dtype=wp.float32),
    channel_stable: wp.array2d(dtype=wp.uint8),
    channel_edge_stable: wp.array2d(dtype=wp.uint8),
    clip_offsets: wp.array(dtype=wp.int32),
    segment_active: wp.array(dtype=wp.int32),
    segment_count: int,
    channel_count: int,
    support_count: int,
    target_support_position_m: wp.array3d(dtype=wp.float32),
):
    """Plant stable intervals and construct full grounded confidence-transition seed patches."""
    segment, channel = wp.tid()
    if segment >= segment_count or channel >= channel_count or segment_active[segment] == 0:
        return
    start = clip_offsets[segment]
    stop = clip_offsets[segment + 1]
    point_count = int(0)
    for point in range(support_count):
        if support_channel_slots[point] == channel:
            point_count += 1

    direction = contact_direction_rows[channel]
    position = direction_position_rows[direction]
    forward = wp.vec3(
        contact_forward_body[channel, 0],
        contact_forward_body[channel, 1],
        contact_forward_body[channel, 2],
    )
    normal = wp.vec3(
        contact_normal_body[channel, 0],
        contact_normal_body[channel, 1],
        contact_normal_body[channel, 2],
    )
    forward_length = wp.length(forward)
    normal_length = wp.length(normal)
    basis_valid = (
        point_count > 0
        and wp.isfinite(forward_length)
        and wp.isfinite(normal_length)
        and wp.abs(forward_length - 1.0) <= 1.0e-4
        and wp.abs(normal_length - 1.0) <= 1.0e-4
        and wp.abs(wp.dot(forward, normal)) <= 1.0e-4
    )
    forward_unit = wp.vec3(0.0, 0.0, 0.0)
    normal_unit = wp.vec3(0.0, 0.0, 0.0)
    left_unit = wp.vec3(0.0, 0.0, 0.0)
    if basis_valid:
        forward_unit = forward / forward_length
        normal_unit = normal / normal_length
        left = wp.cross(normal_unit, forward_unit)
        left_length = wp.length(left)
        if wp.isfinite(left_length) and left_length > 1.0e-8:
            left_unit = left / left_length
        else:
            basis_valid = False

    frame = start
    while frame < stop:
        confidence = channel_confidence[frame, channel]
        stable = channel_stable[frame, channel] != wp.uint8(0)
        if not stable and (not wp.isfinite(confidence) or confidence <= 0.0):
            frame += 1
            continue
        interval_start = frame
        frame += 1
        if stable:
            while (
                frame < stop
                and channel_stable[frame, channel] != wp.uint8(0)
                and channel_edge_stable[frame, channel] != wp.uint8(0)
            ):
                frame += 1
        interval_stop = frame

        valid = basis_valid
        weight_sum = float(0.0)
        heading_sum = wp.vec3(0.0, 0.0, 0.0)
        centroid_sum = wp.vec3(0.0, 0.0, 0.0)
        for sample in range(interval_start, interval_stop):
            confidence = channel_confidence[sample, channel]
            origin = wp.vec3(obstacle_pose[sample, 0], obstacle_pose[sample, 1], obstacle_pose[sample, 2])
            rotation = wp.quat(
                obstacle_pose[sample, 3],
                obstacle_pose[sample, 4],
                obstacle_pose[sample, 5],
                obstacle_pose[sample, 6],
            )
            rotation_length = wp.sqrt(
                rotation[0] * rotation[0]
                + rotation[1] * rotation[1]
                + rotation[2] * rotation[2]
                + rotation[3] * rotation[3]
            )
            sample_valid = (
                basis_valid
                and wp.isfinite(confidence)
                and confidence > 0.0
                and wp.isfinite(origin[0])
                and wp.isfinite(origin[1])
                and wp.isfinite(origin[2])
                and wp.isfinite(rotation_length)
                and rotation_length > 1.0e-8
            )
            if sample_valid:
                inverse_rotation_length = 1.0 / rotation_length
                rotation = wp.quat(
                    rotation[0] * inverse_rotation_length,
                    rotation[1] * inverse_rotation_length,
                    rotation[2] * inverse_rotation_length,
                    rotation[3] * inverse_rotation_length,
                )
                support_inverse = wp.quat_inverse(rotation)
                heading_world = wp.vec3(
                    source_direction_point_position_m[direction, sample, 0]
                    - source_landmark_position_m[position, sample, 0],
                    source_direction_point_position_m[direction, sample, 1]
                    - source_landmark_position_m[position, sample, 1],
                    source_direction_point_position_m[direction, sample, 2]
                    - source_landmark_position_m[position, sample, 2],
                )
                heading_local_3d = wp.quat_rotate(support_inverse, heading_world)
                heading_local = wp.vec3(heading_local_3d[0], heading_local_3d[1], 0.0)
                heading_length = wp.length(heading_local)
                if wp.isfinite(heading_length) and heading_length > 1.0e-8:
                    heading_sum += confidence * heading_local / heading_length
                else:
                    sample_valid = False

                centroid = wp.vec3(0.0, 0.0, 0.0)
                for point in range(support_count):
                    if support_channel_slots[point] == channel:
                        point_world = wp.vec3(
                            target_support_position_m[point, sample, 0],
                            target_support_position_m[point, sample, 1],
                            target_support_position_m[point, sample, 2],
                        )
                        if wp.isfinite(point_world[0]) and wp.isfinite(point_world[1]) and wp.isfinite(point_world[2]):
                            centroid += wp.quat_rotate(support_inverse, point_world - origin)
                        else:
                            sample_valid = False
                if sample_valid:
                    centroid_sum += confidence * centroid / float(point_count)
                    weight_sum += confidence
            if not sample_valid:
                valid = False

        heading_length = wp.length(heading_sum)
        if (
            not wp.isfinite(weight_sum)
            or weight_sum <= 0.0
            or not wp.isfinite(heading_length)
            or heading_length <= 1.0e-8
        ):
            valid = False
        heading = wp.vec3(0.0, 0.0, 0.0)
        centroid = wp.vec3(0.0, 0.0, 0.0)
        if valid:
            heading = heading_sum / heading_length
            centroid = centroid_sum / weight_sum
        left_support = wp.vec3(-heading[1], heading[0], 0.0)
        up_support = wp.vec3(0.0, 0.0, 1.0)

        rotated_sum = wp.vec3(0.0, 0.0, 0.0)
        minimum_height = float(1.0e30)
        for point in range(support_count):
            if support_channel_slots[point] == channel:
                point_body = wp.vec3(
                    support_point_body_m[point, 0],
                    support_point_body_m[point, 1],
                    support_point_body_m[point, 2],
                )
                if not wp.isfinite(point_body[0]) or not wp.isfinite(point_body[1]) or not wp.isfinite(point_body[2]):
                    valid = False
                rotated = (
                    wp.dot(point_body, forward_unit) * heading
                    + wp.dot(point_body, left_unit) * left_support
                    + wp.dot(point_body, normal_unit) * up_support
                )
                rotated_sum += rotated
                minimum_height = wp.min(minimum_height, rotated[2])
        rotated_centroid = wp.vec3(0.0, 0.0, 0.0)
        if point_count > 0:
            rotated_centroid = rotated_sum / float(point_count)
        translation = wp.vec3(
            centroid[0] - rotated_centroid[0],
            centroid[1] - rotated_centroid[1],
            -minimum_height,
        )
        if not wp.isfinite(translation[0]) or not wp.isfinite(translation[1]) or not wp.isfinite(translation[2]):
            valid = False

        for sample in range(interval_start, interval_stop):
            origin = wp.vec3(obstacle_pose[sample, 0], obstacle_pose[sample, 1], obstacle_pose[sample, 2])
            rotation = wp.quat(
                obstacle_pose[sample, 3],
                obstacle_pose[sample, 4],
                obstacle_pose[sample, 5],
                obstacle_pose[sample, 6],
            )
            rotation_length = wp.sqrt(
                rotation[0] * rotation[0]
                + rotation[1] * rotation[1]
                + rotation[2] * rotation[2]
                + rotation[3] * rotation[3]
            )
            if valid and wp.isfinite(rotation_length) and rotation_length > 1.0e-8:
                inverse_rotation_length = 1.0 / rotation_length
                rotation = wp.quat(
                    rotation[0] * inverse_rotation_length,
                    rotation[1] * inverse_rotation_length,
                    rotation[2] * inverse_rotation_length,
                    rotation[3] * inverse_rotation_length,
                )
                for point in range(support_count):
                    if support_channel_slots[point] == channel:
                        point_body = wp.vec3(
                            support_point_body_m[point, 0],
                            support_point_body_m[point, 1],
                            support_point_body_m[point, 2],
                        )
                        rotated = (
                            wp.dot(point_body, forward_unit) * heading
                            + wp.dot(point_body, left_unit) * left_support
                            + wp.dot(point_body, normal_unit) * up_support
                        )
                        point_world = origin + wp.quat_rotate(rotation, translation + rotated)
                        target_support_position_m[point, sample, 0] = point_world[0]
                        target_support_position_m[point, sample, 1] = point_world[1]
                        target_support_position_m[point, sample, 2] = point_world[2]
            else:
                wp.atomic_min(segment_active, segment, 0)


@dataclass(frozen=True, slots=True)
class _MotionContactEvidence:
    """Irreducible source-channel activity plus one robot patch layout."""

    source_stable: torch.Tensor
    support_body_indices: tuple[int, ...]
    support_point_body_m: torch.Tensor
    support_channel_slots: torch.Tensor
    policy: MotionTrajectorySolveCfg.DynamicsCfg


@dataclass(frozen=True, slots=True)
class _MotionAcceptedContactEvidence:
    """Selected trajectory rows awaiting one post-selection certificate pass."""

    sequence_indices: tuple[int, ...]
    source_stable: torch.Tensor
    support_body_indices: tuple[int, ...]
    support_point_body_m: torch.Tensor
    support_channel_slots: torch.Tensor
    policy: MotionTrajectorySolveCfg.DynamicsCfg


@dataclass(frozen=True, slots=True)
class _MotionTrajectoryViewEvidence:
    """Optional frame-aligned geometry retained only for generic inspection."""

    target_landmarks: torch.Tensor
    solved_robot_landmarks: torch.Tensor
    target_support: torch.Tensor
    contact_points: torch.Tensor
    contact_valid: torch.Tensor
    stage_quality: torch.Tensor
    """Retained and pre-rollback attempt quality by physical clip and metric."""


@dataclass(frozen=True, slots=True)
class _MotionTrajectoryResidualLayout:
    """Explicit row ownership for one ordered motion residual vector."""

    source_global_position: slice
    source_rotation: slice
    source_direction_point: slice
    source_fidelity_guard: slice
    contact: slice
    activity_group_by_residual: torch.Tensor
    first_difference_group_by_residual: torch.Tensor
    joint_default: slice
    joint_reference: slice
    collision_objective: slice
    nonpenetration_objective: slice

    @property
    def residual_count(self) -> int:
        """Total residual width."""
        return max(
            self.source_global_position.stop,
            self.source_rotation.stop,
            self.source_direction_point.stop,
            self.source_fidelity_guard.stop,
            self.contact.stop,
            self.joint_default.stop,
            self.joint_reference.stop,
            self.collision_objective.stop,
            self.nonpenetration_objective.stop,
        )


@dataclass(slots=True)
class _MotionTrajectoryWorkspace:
    """One maximum whole-clip workspace reused by every source-ordered batch."""

    joint_q: torch.Tensor
    certified_joint_q: torch.Tensor
    segment_iteration_attempted: torch.Tensor
    segment_damping: torch.Tensor
    segment_recovery_count: torch.Tensor
    joint_qd: torch.Tensor
    achieved_direction_position_m: torch.Tensor
    joint_reference: torch.Tensor
    body_q: torch.Tensor
    body_qd: torch.Tensor
    velocity_reachable_lower: torch.Tensor
    velocity_reachable_upper: torch.Tensor
    frame_quality: torch.Tensor
    segment_active: torch.Tensor
    segment_phase_attempted: torch.Tensor
    segment_iteration_geometry_feasible: torch.Tensor
    segment_iteration_inner_converged: torch.Tensor
    segment_iteration_globalization_succeeded: torch.Tensor
    segment_iteration_residual_constraints_satisfied: torch.Tensor
    segment_phase_globalization_succeeded: torch.Tensor
    segment_phase_converged: torch.Tensor
    segment_contact_refinement_required: torch.Tensor
    source_plane_height_m: torch.Tensor
    source_probe_active: torch.Tensor
    source_probe_stable: torch.Tensor
    source_channel_confidence: torch.Tensor
    source_channel_normal_owned: torch.Tensor
    source_channel_clearance_lift_m: torch.Tensor
    source_channel_activity: torch.Tensor
    source_channel_stable: torch.Tensor
    source_channel_edge_stable: torch.Tensor
    obstacle_pose: torch.Tensor
    rotation_body_indices: torch.Tensor
    residual_layout: _MotionTrajectoryResidualLayout
    base_weights: torch.Tensor
    temporal_weights: torch.Tensor
    velocity_lower: torch.Tensor
    velocity_upper: torch.Tensor
    source_velocity_lower: torch.Tensor
    source_velocity_upper: torch.Tensor


@dataclass(slots=True)
class _MotionSourceEvidenceStream:
    """Monotonic clip stream packed into consecutive whole-clip batches."""

    iterator: Iterator[MotionTrajectoryTargets]
    clip_index: MotionClipIndex
    prototype: MotionTrajectoryTargets
    current: MotionTrajectoryTargets | None
    clip: int = 0
    local_frame: int = 0
    global_frame: int = 0

    def _validate(self, targets: MotionTrajectoryTargets) -> None:
        if self.clip >= len(self.clip_index.clips):
            raise ValueError("Motion landmark target stream exceeds the declared clip index.")
        expected_frames = self.clip_index.clips[self.clip].frame_count
        position_count = len(self.prototype.position_body_indices)
        rotation_count = len(self.prototype.rotation_body_indices)
        direction_count = len(self.prototype.direction_body_indices)
        if (
            targets.source_landmark_position_m.shape != (position_count, expected_frames, 3)
            or targets.source_landmark_rotation_xyzw.shape != (rotation_count, expected_frames, 4)
            or targets.source_direction_point_position_m.shape != (direction_count, expected_frames, 3)
            or targets.initial_joint_q.shape != (expected_frames, self.prototype.initial_joint_q.shape[1])
        ):
            raise ValueError("Motion landmark target shapes differ from the declared source clip.")
        if targets.source_contact_probe_position_m.shape != (
            self.prototype.source_contact_probe_position_m.shape[0],
            expected_frames,
            3,
        ) or targets.target_support_position_m.shape != (
            self.prototype.target_support_position_m.shape[0],
            expected_frames,
            3,
        ):
            raise ValueError("Motion support target shapes differ from the declared source clip.")
        if (
            targets.position_body_indices != self.prototype.position_body_indices
            or targets.position_weights != self.prototype.position_weights
            or targets.rotation_body_indices != self.prototype.rotation_body_indices
            or targets.rotation_weights != self.prototype.rotation_weights
            or targets.source_root_policy != self.prototype.source_root_policy
            or targets.initializer_policy != self.prototype.initializer_policy
            or targets.root_body_index != self.prototype.root_body_index
            or targets.parent_rows != self.prototype.parent_rows
            or targets.required_position_rows != self.prototype.required_position_rows
            or targets.contact_direction_rows != self.prototype.contact_direction_rows
            or targets.required_direction_rows != self.prototype.required_direction_rows
            or targets.direction_length_values_m != self.prototype.direction_length_values_m
            or targets.direction_body_indices != self.prototype.direction_body_indices
            or targets.direction_position_rows != self.prototype.direction_position_rows
            or targets.direction_weights != self.prototype.direction_weights
        ):
            raise ValueError("Motion landmark identity changed between source clips.")
        for name in (
            "position_body_index_tensor",
            "required_position_row_tensor",
            "position_normal_channel_slots",
            "direction_body_index_tensor",
            "contact_direction_row_tensor",
            "direction_contact_channel_slots",
            "required_direction_row_tensor",
            "direction_position_row_tensor",
            "direction_point_body_m",
            "parent_row_tensor",
            "segment_lengths_m",
            "coordinate_indices",
            "coordinate_lower_limits_rad",
            "coordinate_upper_limits_rad",
            "contact_channel_probe_offsets",
            "contact_body_indices",
            "contact_forward_body",
            "contact_distal_point_body_m",
            "contact_normal_body",
            "leg_chain_body_indices",
            "leg_chain_parent_body_indices",
            "leg_knee_hint_anatomy",
            "leg_knee_hint_root",
            "leg_segment_lengths_m",
            "support_body_indices",
            "support_point_body_m",
            "support_channel_slots",
        ):
            if getattr(targets, name).data_ptr() != getattr(self.prototype, name).data_ptr():
                raise ValueError(f"Motion landmark static tensor {name!r} changed between source clips.")

    def _next(self) -> MotionTrajectoryTargets:
        try:
            targets = next(self.iterator)
        except StopIteration as error:
            raise ValueError("Motion landmark target stream ended before the declared corpus.") from error
        self._validate(targets)
        return targets

    def fill(self, targets: MotionTrajectoryTargets, start: int, stop: int) -> None:
        """Pack global rows ``[start, stop)`` into the workspace leading prefix."""
        if start != self.global_frame or stop <= start:
            raise ValueError("Motion trajectory executor requires monotonic nonempty intervals.")
        destination = 0
        while self.global_frame < stop:
            if self.current is None:
                self.current = self._next()
            frame_count = self.current.source_landmark_position_m.shape[1]
            count = min(stop - self.global_frame, frame_count - self.local_frame)
            source = slice(self.local_frame, self.local_frame + count)
            target = slice(destination, destination + count)
            targets.initial_joint_q[target].copy_(self.current.initial_joint_q[source])
            targets.source_landmark_position_m[:, target].copy_(self.current.source_landmark_position_m[:, source])
            targets.source_landmark_rotation_xyzw[:, target].copy_(
                self.current.source_landmark_rotation_xyzw[:, source]
            )
            targets.source_direction_point_position_m[:, target].copy_(
                self.current.source_direction_point_position_m[:, source]
            )
            targets.source_contact_probe_position_m[:, target].copy_(
                self.current.source_contact_probe_position_m[:, source]
            )
            targets.target_support_position_m[:, target].copy_(self.current.target_support_position_m[:, source])
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
            raise ValueError("Motion landmark target stream did not cover the declared corpus exactly.")
        try:
            next(self.iterator)
        except StopIteration:
            return
        raise ValueError("Motion landmark target stream contains an undeclared trailing clip.")


def _motion_frame_seed_targets(prototype: MotionTrajectoryTargets, capacity: int) -> MotionTrajectoryTargets:
    """Allocate only semantic targets and coordinates required by frame-local IK."""
    return replace(
        prototype,
        source_landmark_position_m=torch.empty(
            (len(prototype.position_body_indices), capacity, 3),
            dtype=torch.float32,
            device=prototype.source_landmark_position_m.device,
        ),
        source_landmark_rotation_xyzw=torch.empty(
            (len(prototype.rotation_body_indices), capacity, 4),
            dtype=torch.float32,
            device=prototype.source_landmark_rotation_xyzw.device,
        ),
        source_direction_point_position_m=torch.empty(
            (len(prototype.direction_body_indices), capacity, 3),
            dtype=torch.float32,
            device=prototype.source_direction_point_position_m.device,
        ),
        initial_joint_q=torch.empty(
            (capacity, prototype.initial_joint_q.shape[1]),
            dtype=torch.float32,
            device=prototype.initial_joint_q.device,
        ),
    )


def _motion_workspace_targets(prototype: MotionTrajectoryTargets, capacity: int) -> MotionTrajectoryTargets:
    """Allocate only fixed-capacity calibrated solver evidence."""
    return replace(
        prototype,
        source_landmark_position_m=torch.empty(
            (len(prototype.position_body_indices), capacity, 3),
            dtype=torch.float32,
            device=prototype.source_landmark_position_m.device,
        ),
        source_landmark_rotation_xyzw=torch.empty(
            (len(prototype.rotation_body_indices), capacity, 4),
            dtype=torch.float32,
            device=prototype.source_landmark_rotation_xyzw.device,
        ),
        source_direction_point_position_m=torch.empty(
            (len(prototype.direction_body_indices), capacity, 3),
            dtype=torch.float32,
            device=prototype.source_direction_point_position_m.device,
        ),
        initial_joint_q=torch.empty(
            (capacity, prototype.initial_joint_q.shape[1]),
            dtype=torch.float32,
            device=prototype.initial_joint_q.device,
        ),
        source_contact_probe_position_m=torch.empty(
            (prototype.source_contact_probe_position_m.shape[0], capacity, 3),
            dtype=torch.float32,
            device=prototype.source_landmark_position_m.device,
        ),
        target_support_position_m=torch.empty(
            (prototype.target_support_position_m.shape[0], capacity, 3),
            dtype=torch.float32,
            device=prototype.source_landmark_position_m.device,
        ),
    )


def _source_clips(candidate: _builder._MotionSourceCandidate):
    """Yield every declared clip once while keeping source mechanics grouped."""
    clip_count = len(candidate.source_index.clips)
    if len(candidate.source_clip_indices) != clip_count or len(candidate.projection_indices) != clip_count:
        raise ValueError("Motion clip origins and source projections must cover the declared corpus exactly.")
    count = 0
    while count < clip_count:
        projection_index = candidate.projection_indices[count]
        if projection_index < 0 or projection_index >= len(candidate.projections):
            raise ValueError("Motion clip maps to an invalid source projection.")
        group_stop = count + 1
        while group_stop < clip_count and candidate.projection_indices[group_stop] == projection_index:
            group_stop += 1
        group_source_indices = candidate.source_clip_indices[count:group_stop]
        for source_index, clip in candidate.source.clips(group_source_indices):
            if count == group_stop:
                raise ValueError(f"Motion source projection yielded undeclared clip index {source_index!r}.")
            expected = candidate.source_index.clips[count]
            expected_source_index = candidate.source_clip_indices[count]
            if (
                source_index != expected_source_index
                or clip.frame_count != expected.frame_count
                or clip.source_fps != expected.source_fps
            ):
                raise ValueError(
                    f"Motion source expected index {expected_source_index} ({expected.clip_id!r}) with "
                    f"{expected.frame_count} frames at {expected.source_fps} Hz, got index {source_index!r} with "
                    f"{clip.frame_count} frames at {clip.source_fps} Hz; its clock or identity "
                    "changed after inspection."
                )
            yield count, candidate.projections[projection_index], clip
            count += 1
        if count != group_stop:
            raise ValueError(f"Motion source yielded {count} of {group_stop} clips for one source projection.")


def motion_generate_source_evidence(
    cfg: MotionSourceEvidenceGenerateCfg, candidate: _builder._MotionTrajectorySourceCandidate, _rng: object
) -> _builder._MotionTrajectoryTargetCandidate:
    """Stream calibrated source evidence and limit-valid robot initializers without retaining decoded clips."""
    from .motion_task_table_builder import _MotionTrajectoryTargetCandidate

    del cfg

    def targets() -> Iterator[MotionTrajectoryTargets]:
        for _index, projection, clip in _source_clips(candidate):
            yield _motion_source_evidence(projection, clip, candidate.device)

    return _MotionTrajectoryTargetCandidate(
        target=candidate.target,
        clip_index=candidate.output_index,
        pending=targets(),
        source_body_counts=tuple(
            candidate.projections[index].source_skeleton.num_bodies for index in candidate.projection_indices
        ),
        device=candidate.device,
        inspection=candidate.inspection,
    )


def _motion_source_evidence(
    projection: MotionSourceProjectionTrajectory,
    clip,
    device: str | torch.device,
) -> MotionTrajectoryTargets:
    """Decode one source clip and project calibrated target-owned evidence."""
    root_position, local_rotation = clip.local_pose(projection.source_skeleton, device=device)
    return projection.target_projection.generate_targets(root_position, local_rotation)


def _trajectory_ground_mesh(device: str) -> wp.Mesh:
    """Build one wide local ground plane retained by the trajectory workspace."""
    vertices = np.asarray(
        ((-100.0, -100.0, 0.0), (100.0, -100.0, 0.0), (100.0, 100.0, 0.0), (-100.0, 100.0, 0.0)),
        dtype=np.float32,
    )
    points = wp.array(vertices, dtype=wp.vec3, device=device)
    indices = wp.array(np.asarray((0, 1, 2, 0, 2, 3), dtype=np.int32), dtype=wp.int32, device=device)
    return wp.Mesh(points=points, indices=indices)


def motion_objective_source_global_position(
    cfg: MotionSourceGlobalPositionObjectiveCfg,
    targets: MotionTrajectoryTargets,
    source_channel_normal_owned: torch.Tensor | None = None,
    source_channel_confidence: torch.Tensor | None = None,
) -> list[object]:
    """Match body origins in 3-D unless contact or clearance owns their normal coordinate."""
    channel_count = len(targets.contact_direction_rows)
    if (source_channel_normal_owned is None) != (source_channel_confidence is None):
        raise ValueError("Source normal ownership and contact confidence must be provided together.")
    for name, tensor in (
        ("normal ownership", source_channel_normal_owned),
        ("contact confidence", source_channel_confidence),
    ):
        if tensor is not None and (
            tensor.shape != (targets.source_landmark_position_m.shape[1], channel_count)
            or tensor.dtype is not torch.float32
            or tensor.device != targets.source_landmark_position_m.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"Source {name} must be contiguous float32 [frame, contact channel].")
    if (
        targets.position_normal_channel_slots.shape != (len(targets.position_body_indices),)
        or targets.position_normal_channel_slots.dtype is not torch.int64
        or targets.position_normal_channel_slots.device != targets.source_landmark_position_m.device
        or not targets.position_normal_channel_slots.is_contiguous()
    ):
        raise ValueError("Source position normal-channel slots must be aligned contiguous target int64 rows.")
    lengths = targets.segment_length_values_m
    position_normal_channel_slots = tuple(int(value) for value in targets.position_normal_channel_slots.cpu().tolist())
    direction_point_body_values = targets.direction_point_body_m.detach().cpu().tolist()
    normal_owned_wp = None if source_channel_normal_owned is None else wp.from_torch(source_channel_normal_owned)
    confidence_wp = None if source_channel_confidence is None else wp.from_torch(source_channel_confidence)
    objectives = []
    for row in (*range(1, len(targets.position_body_indices)), 0):
        target_positions = wp.from_torch(targets.source_landmark_position_m[row], dtype=wp.vec3)
        weight = (cfg.root_weight if row == 0 else cfg.weight) * targets.position_weights[row] / lengths[row]
        contact_channel = position_normal_channel_slots[row]
        if normal_owned_wp is None or contact_channel < 0:
            objective = ik.IKObjectivePosition(
                link_index=targets.position_body_indices[row],
                link_offset=wp.vec3(0.0, 0.0, 0.0),
                target_positions=target_positions,
                weight=weight,
            )
        else:
            direction_row = targets.contact_direction_rows[contact_channel]
            direction_point_body = direction_point_body_values[direction_row]
            contact_point_owned = (
                targets.position_body_indices[row] == targets.direction_body_indices[direction_row]
                and sum(value * value for value in direction_point_body) <= 1.0e-16
            )
            objective = _IKObjectiveSourcePhasePosition(
                link_index=targets.position_body_indices[row],
                link_offset=wp.vec3(0.0, 0.0, 0.0),
                target_positions=target_positions,
                source_base_positions=target_positions,
                source_channel_confidence=confidence_wp,
                source_channel_normal_owned=normal_owned_wp,
                contact_tangent_length_m=0.0,
                contact_channel=contact_channel,
                contact_point_owned=contact_point_owned,
                weight=weight,
            )
        objectives.append(objective)
    return objectives


def motion_objective_source_rotation(
    _cfg: MotionSourceRotationObjectiveCfg, targets: MotionTrajectoryTargets
) -> list[object]:
    """Match target-owned robot bodies to calibrated source-landmark rotations."""
    return [
        ik.IKObjectiveRotation(
            link_index=body_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.from_torch(targets.source_landmark_rotation_xyzw[row], dtype=wp.vec4),
            canonicalize_quat_err=True,
            weight=weight,
        )
        for row, (body_index, weight) in enumerate(
            zip(targets.rotation_body_indices, targets.rotation_weights, strict=True)
        )
    ]


@wp.func
def _source_contact_planar_distal_target(
    source_base: wp.vec3, source_point: wp.vec3, contact_tangent_length_m: float
) -> wp.vec3:
    """Return source yaw at the target robot's calibrated contact tangent length."""
    tangent = wp.vec3(source_point[0] - source_base[0], source_point[1] - source_base[1], 0.0)
    tangent_length = wp.length(tangent)
    target = source_point
    if (
        wp.isfinite(tangent_length)
        and tangent_length > 1.0e-8
        and wp.isfinite(contact_tangent_length_m)
        and contact_tangent_length_m > 0.0
    ):
        target = source_base + contact_tangent_length_m * tangent / tangent_length
    return target


@wp.func
def _source_position_is_coincident_distal(
    position_row: wp.int64,
    contact_channel: wp.int64,
    position_body_indices: wp.array(dtype=wp.int64),
    direction_body_indices: wp.array(dtype=wp.int64),
    contact_direction_rows: wp.array(dtype=wp.int64),
    direction_point_body_m: wp.array2d(dtype=wp.float32),
) -> int:
    """Return whether one position body origin is the distal point for its contact channel."""
    if contact_channel < 0:
        return 0
    direction_row = contact_direction_rows[contact_channel]
    point = wp.vec3(
        direction_point_body_m[direction_row, 0],
        direction_point_body_m[direction_row, 1],
        direction_point_body_m[direction_row, 2],
    )
    return int(
        position_body_indices[position_row] == direction_body_indices[direction_row] and wp.dot(point, point) <= 1.0e-16
    )


@wp.kernel
def _source_phase_position_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    target_positions: wp.array(dtype=wp.vec3),
    source_base_positions: wp.array(dtype=wp.vec3),
    source_channel_confidence: wp.array2d(dtype=wp.float32),
    source_channel_normal_owned: wp.array2d(dtype=wp.float32),
    link_index: int,
    link_offset: wp.vec3,
    contact_channel: int,
    contact_point_owned: bool,
    contact_tangent_length_m: float,
    start: int,
    weight: float,
    problem_indices: wp.array(dtype=wp.int32),
    residuals: wp.array2d(dtype=wp.float32),
):
    """Write true-swing 3-D and contact-active planar distal-point residuals."""
    frame = wp.tid()
    target_frame = problem_indices[frame]
    if contact_point_owned and contact_channel >= 0 and source_channel_confidence[target_frame, contact_channel] > 0.0:
        residuals[frame, start] = 0.0
        residuals[frame, start + 1] = 0.0
        residuals[frame, start + 2] = 0.0
        return
    target = target_positions[target_frame]
    if contact_channel >= 0 and source_channel_confidence[target_frame, contact_channel] > 0.0:
        target = _source_contact_planar_distal_target(
            source_base_positions[target_frame], target, contact_tangent_length_m
        )
    actual = wp.transform_point(body_q[frame, link_index], link_offset)
    error = target - actual
    if contact_channel >= 0 and source_channel_normal_owned[target_frame, contact_channel] > 0.0:
        error = wp.vec3(error[0], error[1], 0.0)
    residuals[frame, start] = weight * error[0]
    residuals[frame, start + 1] = weight * error[1]
    residuals[frame, start + 2] = weight * error[2]


@wp.kernel
def _source_phase_position_jacobian(
    body_q: wp.array2d(dtype=wp.transform),
    joint_screw: wp.array2d(dtype=wp.spatial_vector),
    source_channel_confidence: wp.array2d(dtype=wp.float32),
    source_channel_normal_owned: wp.array2d(dtype=wp.float32),
    affects_dof: wp.array(dtype=wp.uint8),
    link_index: int,
    link_offset: wp.vec3,
    contact_channel: int,
    contact_point_owned: bool,
    start: int,
    weight: float,
    jacobian: wp.array3d(dtype=wp.float32),
):
    """Write the analytic Jacobian for contact-conditioned distal positions."""
    frame, dof = wp.tid()
    if contact_point_owned and contact_channel >= 0 and source_channel_confidence[frame, contact_channel] > 0.0:
        jacobian[frame, start, dof] = 0.0
        jacobian[frame, start + 1, dof] = 0.0
        jacobian[frame, start + 2, dof] = 0.0
        return
    if affects_dof[dof] == wp.uint8(0):
        return
    actual = wp.transform_point(body_q[frame, link_index], link_offset)
    screw = joint_screw[frame, dof]
    linear = wp.vec3(screw[0], screw[1], screw[2])
    angular = wp.vec3(screw[3], screw[4], screw[5])
    velocity = linear + wp.cross(angular, actual)
    jacobian[frame, start, dof] = -weight * velocity[0]
    jacobian[frame, start + 1, dof] = -weight * velocity[1]
    if contact_channel >= 0 and source_channel_normal_owned[frame, contact_channel] > 0.0:
        jacobian[frame, start + 2, dof] = 0.0
    else:
        jacobian[frame, start + 2, dof] = -weight * velocity[2]


class _IKObjectiveSourcePhasePosition(ik.IKObjectivePosition):
    """Match source points in 3-D during true swing and in-plane during contact phases."""

    def __init__(
        self,
        link_index: int,
        link_offset: wp.vec3,
        target_positions: wp.array,
        source_base_positions: wp.array,
        source_channel_confidence: wp.array,
        source_channel_normal_owned: wp.array,
        contact_tangent_length_m: float,
        contact_channel: int,
        contact_point_owned: bool,
        weight: float,
    ) -> None:
        super().__init__(link_index, link_offset, target_positions, weight)
        self.source_base_positions = source_base_positions
        self.source_channel_confidence = source_channel_confidence
        self.source_channel_normal_owned = source_channel_normal_owned
        self.contact_tangent_length_m = contact_tangent_length_m
        self.contact_channel = contact_channel
        self.contact_point_owned = contact_point_owned

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        """Write contact-conditioned distal-point residuals."""
        del joint_q, model
        wp.launch(
            _source_phase_position_residuals,
            dim=body_q.shape[0],
            inputs=[
                body_q,
                self.target_positions,
                self.source_base_positions,
                self.source_channel_confidence,
                self.source_channel_normal_owned,
                self.link_index,
                self.link_offset,
                self.contact_channel,
                self.contact_point_owned,
                self.contact_tangent_length_m,
                start_idx,
                self.weight,
                problem_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_screw, start_idx) -> None:
        """Write the exact contact-conditioned distal-point Jacobian."""
        del joint_q, model
        wp.launch(
            _source_phase_position_jacobian,
            dim=(body_q.shape[0], joint_screw.shape[1]),
            inputs=[
                body_q,
                joint_screw,
                self.source_channel_confidence,
                self.source_channel_normal_owned,
                self.affects_dof,
                self.link_index,
                self.link_offset,
                self.contact_channel,
                self.contact_point_owned,
                start_idx,
                self.weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )


def motion_objective_source_direction_point(
    cfg: MotionSourceDirectionPointObjectiveCfg,
    targets: MotionTrajectoryTargets,
    source_channel_normal_owned: torch.Tensor | None = None,
    source_channel_confidence: torch.Tensor | None = None,
) -> list[object]:
    """Match distal points while keeping contact geometry and clearance ownership independent."""
    if (source_channel_normal_owned is None) != (source_channel_confidence is None):
        raise ValueError("Source normal ownership and contact confidence must be provided together.")
    expected_shape = (targets.source_landmark_position_m.shape[1], len(targets.contact_direction_rows))
    for name, tensor in (
        ("normal ownership", source_channel_normal_owned),
        ("contact confidence", source_channel_confidence),
    ):
        if tensor is not None and (
            tensor.shape != expected_shape
            or tensor.dtype is not torch.float32
            or tensor.device != targets.source_landmark_position_m.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"Source {name} must be contiguous float32 [frame, contact channel].")
    contact_channels = {row: channel for channel, row in enumerate(targets.contact_direction_rows)}
    normal_owned_wp = None if source_channel_normal_owned is None else wp.from_torch(source_channel_normal_owned)
    confidence_wp = None if source_channel_confidence is None else wp.from_torch(source_channel_confidence)
    objectives = []
    for row, body_index in enumerate(targets.direction_body_indices):
        arguments = dict(
            link_index=body_index,
            link_offset=wp.vec3(*targets.direction_point_body_m[row].tolist()),
            target_positions=wp.from_torch(targets.source_direction_point_position_m[row], dtype=wp.vec3),
            weight=cfg.weight * targets.direction_weights[row] / targets.direction_length_values_m[row],
        )
        if normal_owned_wp is None:
            objectives.append(ik.IKObjectivePosition(**arguments))
        else:
            objectives.append(
                _IKObjectiveSourcePhasePosition(
                    **arguments,
                    source_channel_confidence=confidence_wp,
                    source_channel_normal_owned=normal_owned_wp,
                    source_base_positions=wp.from_torch(
                        targets.source_landmark_position_m[targets.direction_position_rows[row]], dtype=wp.vec3
                    ),
                    contact_tangent_length_m=targets.direction_length_values_m[row],
                    contact_channel=contact_channels.get(row, -1),
                    contact_point_owned=False,
                )
            )
    return objectives


@wp.kernel
def _source_fidelity_guard_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    source_landmark_position_m: wp.array3d(dtype=wp.float32),
    source_direction_point_position_m: wp.array3d(dtype=wp.float32),
    source_landmark_rotation_xyzw: wp.array3d(dtype=wp.float32),
    position_body_indices: wp.array(dtype=wp.int64),
    direction_body_indices: wp.array(dtype=wp.int64),
    direction_position_rows: wp.array(dtype=wp.int64),
    required_position_rows: wp.array(dtype=wp.int64),
    position_normal_channel_slots: wp.array(dtype=wp.int64),
    contact_direction_rows: wp.array(dtype=wp.int64),
    direction_contact_channel_slots: wp.array(dtype=wp.int64),
    required_direction_rows: wp.array(dtype=wp.int64),
    source_channel_confidence: wp.array2d(dtype=wp.float32),
    source_channel_normal_owned: wp.array2d(dtype=wp.float32),
    direction_point_body_m: wp.array2d(dtype=wp.float32),
    contact_distal_point_body_m: wp.array2d(dtype=wp.float32),
    required_position_count: int,
    required_distal_count: int,
    root_body: int,
    start: int,
    problem_indices: wp.array(dtype=wp.int32),
    residuals: wp.array2d(dtype=wp.float32),
):
    """Write raw source-fidelity distances in publication-metric order."""
    frame, guard = wp.tid()
    target_frame = problem_indices[frame]
    distal_stop = required_position_count + required_distal_count
    direction_stop = distal_stop + required_distal_count
    value = float(wp.inf)
    if guard < required_position_count:
        position_row = required_position_rows[guard]
        contact_channel = position_normal_channel_slots[position_row]
        contact_owned = (
            contact_channel >= 0
            and source_channel_confidence[target_frame, contact_channel] > 0.0
            and _source_position_is_coincident_distal(
                position_row,
                contact_channel,
                position_body_indices,
                direction_body_indices,
                contact_direction_rows,
                direction_point_body_m,
            )
            != 0
        )
        if contact_owned:
            value = 0.0
        else:
            body = position_body_indices[position_row]
            actual = wp.transform_get_translation(body_q[frame, body])
            target = wp.vec3(
                source_landmark_position_m[position_row, target_frame, 0],
                source_landmark_position_m[position_row, target_frame, 1],
                source_landmark_position_m[position_row, target_frame, 2],
            )
            error = actual - target
            if contact_channel >= 0 and source_channel_normal_owned[target_frame, contact_channel] > 0.0:
                error = wp.vec3(error[0], error[1], 0.0)
            distance = wp.length(error)
            if wp.isfinite(distance):
                value = distance
    elif guard < distal_stop:
        required = guard - required_position_count
        direction_row = required_direction_rows[required]
        contact_channel = direction_contact_channel_slots[direction_row]
        body = direction_body_indices[direction_row]
        position_row = direction_position_rows[direction_row]
        point_offset = wp.vec3(
            direction_point_body_m[direction_row, 0],
            direction_point_body_m[direction_row, 1],
            direction_point_body_m[direction_row, 2],
        )
        actual = wp.transform_point(body_q[frame, body], point_offset)
        target = wp.vec3(
            source_direction_point_position_m[direction_row, target_frame, 0],
            source_direction_point_position_m[direction_row, target_frame, 1],
            source_direction_point_position_m[direction_row, target_frame, 2],
        )
        if contact_channel >= 0 and source_channel_confidence[target_frame, contact_channel] > 0.0:
            source_base = wp.vec3(
                source_landmark_position_m[position_row, target_frame, 0],
                source_landmark_position_m[position_row, target_frame, 1],
                source_landmark_position_m[position_row, target_frame, 2],
            )
            contact_tangent = wp.vec3(
                contact_distal_point_body_m[contact_channel, 0],
                contact_distal_point_body_m[contact_channel, 1],
                contact_distal_point_body_m[contact_channel, 2],
            )
            target = _source_contact_planar_distal_target(source_base, target, wp.length(contact_tangent))
        error = actual - target
        if contact_channel >= 0 and source_channel_normal_owned[target_frame, contact_channel] > 0.0:
            error = wp.vec3(error[0], error[1], 0.0)
        distance = wp.length(error)
        if wp.isfinite(distance):
            value = distance
    elif guard < direction_stop:
        required = guard - distal_stop
        direction_row = required_direction_rows[required]
        contact_channel = direction_contact_channel_slots[direction_row]
        position_row = direction_position_rows[direction_row]
        point_body = direction_body_indices[direction_row]
        base_body = position_body_indices[position_row]
        point_offset = wp.vec3(
            direction_point_body_m[direction_row, 0],
            direction_point_body_m[direction_row, 1],
            direction_point_body_m[direction_row, 2],
        )
        actual_point = wp.transform_point(body_q[frame, point_body], point_offset)
        actual_base = wp.transform_get_translation(body_q[frame, base_body])
        source_point = wp.vec3(
            source_direction_point_position_m[direction_row, target_frame, 0],
            source_direction_point_position_m[direction_row, target_frame, 1],
            source_direction_point_position_m[direction_row, target_frame, 2],
        )
        source_base = wp.vec3(
            source_landmark_position_m[position_row, target_frame, 0],
            source_landmark_position_m[position_row, target_frame, 1],
            source_landmark_position_m[position_row, target_frame, 2],
        )
        actual_direction = actual_point - actual_base
        source_direction = source_point - source_base
        if contact_channel >= 0 and source_channel_confidence[target_frame, contact_channel] > 0.0:
            actual_direction = wp.vec3(actual_direction[0], actual_direction[1], 0.0)
            source_direction = wp.vec3(source_direction[0], source_direction[1], 0.0)
        actual_length = wp.length(actual_direction)
        source_length = wp.length(source_direction)
        value = float(wp.inf)
        if (
            wp.isfinite(actual_length)
            and wp.isfinite(source_length)
            and actual_length > 1.0e-8
            and source_length > 1.0e-8
        ):
            unit_actual = actual_direction / actual_length
            unit_source = source_direction / source_length
            chord_distance = wp.length(unit_actual - unit_source)
            if wp.isfinite(chord_distance):
                value = chord_distance
    else:
        root_transform = body_q[frame, root_body]
        actual_rotation = wp.quat(root_transform[3], root_transform[4], root_transform[5], root_transform[6])
        target_rotation = wp.quat(
            source_landmark_rotation_xyzw[0, target_frame, 0],
            source_landmark_rotation_xyzw[0, target_frame, 1],
            source_landmark_rotation_xyzw[0, target_frame, 2],
            source_landmark_rotation_xyzw[0, target_frame, 3],
        )
        rotation_error = actual_rotation * wp.quat_inverse(target_rotation)
        if wp.dot(actual_rotation, target_rotation) < 0.0:
            rotation_error = -rotation_error
        vector_length = wp.length(wp.vec3(rotation_error[0], rotation_error[1], rotation_error[2]))
        value = float(wp.inf)
        if wp.isfinite(vector_length) and wp.isfinite(rotation_error[3]):
            chord_distance = 2.0 * vector_length
            if wp.isfinite(chord_distance):
                value = chord_distance
    residuals[frame, start + guard] = value


@wp.kernel
def _source_fidelity_guard_jacobian(
    body_q: wp.array2d(dtype=wp.transform),
    joint_screw: wp.array2d(dtype=wp.spatial_vector),
    source_landmark_position_m: wp.array3d(dtype=wp.float32),
    source_direction_point_position_m: wp.array3d(dtype=wp.float32),
    source_landmark_rotation_xyzw: wp.array3d(dtype=wp.float32),
    position_body_indices: wp.array(dtype=wp.int64),
    direction_body_indices: wp.array(dtype=wp.int64),
    direction_position_rows: wp.array(dtype=wp.int64),
    required_position_rows: wp.array(dtype=wp.int64),
    position_normal_channel_slots: wp.array(dtype=wp.int64),
    contact_direction_rows: wp.array(dtype=wp.int64),
    direction_contact_channel_slots: wp.array(dtype=wp.int64),
    required_direction_rows: wp.array(dtype=wp.int64),
    source_channel_confidence: wp.array2d(dtype=wp.float32),
    source_channel_normal_owned: wp.array2d(dtype=wp.float32),
    direction_point_body_m: wp.array2d(dtype=wp.float32),
    contact_distal_point_body_m: wp.array2d(dtype=wp.float32),
    required_position_count: int,
    required_distal_count: int,
    root_body: int,
    ancestry_state: wp.array2d(dtype=wp.uint8),
    start: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    """Write fused analytic world-screw derivatives for active guard rows."""
    frame, guard, dof = wp.tid()
    state = ancestry_state[guard, dof]
    if state == wp.uint8(0):
        return
    screw = joint_screw[frame, dof]
    linear = wp.vec3(screw[0], screw[1], screw[2])
    angular = wp.vec3(screw[3], screw[4], screw[5])
    distal_stop = required_position_count + required_distal_count
    direction_stop = distal_stop + required_distal_count
    derivative = 0.0
    if guard < required_position_count:
        position_row = required_position_rows[guard]
        contact_channel = position_normal_channel_slots[position_row]
        if (
            contact_channel >= 0
            and source_channel_confidence[frame, contact_channel] > 0.0
            and _source_position_is_coincident_distal(
                position_row,
                contact_channel,
                position_body_indices,
                direction_body_indices,
                contact_direction_rows,
                direction_point_body_m,
            )
            != 0
        ):
            jacobian[frame, start + guard, dof] = 0.0
            return
        body = position_body_indices[position_row]
        actual = wp.transform_get_translation(body_q[frame, body])
        target = wp.vec3(
            source_landmark_position_m[position_row, frame, 0],
            source_landmark_position_m[position_row, frame, 1],
            source_landmark_position_m[position_row, frame, 2],
        )
        error = actual - target
        if contact_channel >= 0 and source_channel_normal_owned[frame, contact_channel] > 0.0:
            error = wp.vec3(error[0], error[1], 0.0)
        distance = wp.length(error)
        if not wp.isfinite(distance) or distance <= 1.0e-8:
            return
        velocity = linear + wp.cross(angular, actual)
        if contact_channel >= 0 and source_channel_normal_owned[frame, contact_channel] > 0.0:
            velocity = wp.vec3(velocity[0], velocity[1], 0.0)
        derivative = wp.dot(error / distance, velocity)
    elif guard < distal_stop:
        required = guard - required_position_count
        direction_row = required_direction_rows[required]
        contact_channel = direction_contact_channel_slots[direction_row]
        body = direction_body_indices[direction_row]
        position_row = direction_position_rows[direction_row]
        point_offset = wp.vec3(
            direction_point_body_m[direction_row, 0],
            direction_point_body_m[direction_row, 1],
            direction_point_body_m[direction_row, 2],
        )
        actual = wp.transform_point(body_q[frame, body], point_offset)
        target = wp.vec3(
            source_direction_point_position_m[direction_row, frame, 0],
            source_direction_point_position_m[direction_row, frame, 1],
            source_direction_point_position_m[direction_row, frame, 2],
        )
        if contact_channel >= 0 and source_channel_confidence[frame, contact_channel] > 0.0:
            source_base = wp.vec3(
                source_landmark_position_m[position_row, frame, 0],
                source_landmark_position_m[position_row, frame, 1],
                source_landmark_position_m[position_row, frame, 2],
            )
            contact_tangent = wp.vec3(
                contact_distal_point_body_m[contact_channel, 0],
                contact_distal_point_body_m[contact_channel, 1],
                contact_distal_point_body_m[contact_channel, 2],
            )
            target = _source_contact_planar_distal_target(source_base, target, wp.length(contact_tangent))
        error = actual - target
        if contact_channel >= 0 and source_channel_normal_owned[frame, contact_channel] > 0.0:
            error = wp.vec3(error[0], error[1], 0.0)
        distance = wp.length(error)
        if not wp.isfinite(distance) or distance <= 1.0e-8:
            return
        velocity = linear + wp.cross(angular, actual)
        if contact_channel >= 0 and source_channel_normal_owned[frame, contact_channel] > 0.0:
            velocity = wp.vec3(velocity[0], velocity[1], 0.0)
        derivative = wp.dot(error / distance, velocity)
    elif guard < direction_stop:
        required = guard - distal_stop
        direction_row = required_direction_rows[required]
        contact_channel = direction_contact_channel_slots[direction_row]
        position_row = direction_position_rows[direction_row]
        point_body = direction_body_indices[direction_row]
        base_body = position_body_indices[position_row]
        point_offset = wp.vec3(
            direction_point_body_m[direction_row, 0],
            direction_point_body_m[direction_row, 1],
            direction_point_body_m[direction_row, 2],
        )
        actual_point = wp.transform_point(body_q[frame, point_body], point_offset)
        actual_base = wp.transform_get_translation(body_q[frame, base_body])
        source_point = wp.vec3(
            source_direction_point_position_m[direction_row, frame, 0],
            source_direction_point_position_m[direction_row, frame, 1],
            source_direction_point_position_m[direction_row, frame, 2],
        )
        source_base = wp.vec3(
            source_landmark_position_m[position_row, frame, 0],
            source_landmark_position_m[position_row, frame, 1],
            source_landmark_position_m[position_row, frame, 2],
        )
        actual_direction = actual_point - actual_base
        source_direction = source_point - source_base
        if contact_channel >= 0 and source_channel_confidence[frame, contact_channel] > 0.0:
            actual_direction = wp.vec3(actual_direction[0], actual_direction[1], 0.0)
            source_direction = wp.vec3(source_direction[0], source_direction[1], 0.0)
        actual_length = wp.length(actual_direction)
        source_length = wp.length(source_direction)
        if (
            not wp.isfinite(actual_length)
            or not wp.isfinite(source_length)
            or actual_length <= 1.0e-8
            or source_length <= 1.0e-8
        ):
            return
        unit_actual = actual_direction / actual_length
        unit_source = source_direction / source_length
        chord_distance = wp.length(unit_actual - unit_source)
        if not wp.isfinite(chord_distance) or chord_distance <= 1.0e-8:
            return
        cosine = wp.clamp(wp.dot(unit_actual, unit_source), -1.0, 1.0)
        gradient = (cosine * unit_actual - unit_source) / (chord_distance * actual_length)
        point_velocity = wp.vec3(0.0, 0.0, 0.0)
        base_velocity = wp.vec3(0.0, 0.0, 0.0)
        if (state & wp.uint8(1)) != wp.uint8(0):
            point_velocity = linear + wp.cross(angular, actual_point)
        if (state & wp.uint8(2)) != wp.uint8(0):
            base_velocity = linear + wp.cross(angular, actual_base)
        derivative = wp.dot(gradient, point_velocity - base_velocity)
    else:
        root_transform = body_q[frame, root_body]
        actual_rotation = wp.quat(root_transform[3], root_transform[4], root_transform[5], root_transform[6])
        target_rotation = wp.quat(
            source_landmark_rotation_xyzw[0, frame, 0],
            source_landmark_rotation_xyzw[0, frame, 1],
            source_landmark_rotation_xyzw[0, frame, 2],
            source_landmark_rotation_xyzw[0, frame, 3],
        )
        rotation_error = actual_rotation * wp.quat_inverse(target_rotation)
        if wp.dot(actual_rotation, target_rotation) < 0.0:
            rotation_error = -rotation_error
        vector = wp.vec3(rotation_error[0], rotation_error[1], rotation_error[2])
        vector_length = wp.length(vector)
        chord_distance = 2.0 * vector_length
        if not wp.isfinite(vector_length) or not wp.isfinite(rotation_error[3]) or vector_length <= 1.0e-8:
            return
        derivative = rotation_error[3] * wp.dot(vector / vector_length, angular)
    if wp.isfinite(derivative):
        jacobian[frame, start + guard, dof] = derivative


class _IKObjectiveSourceFidelityGuard(ik.IKObjective):
    """Own the packed source-fidelity guard for identity-mapped trajectory batches."""

    def __init__(
        self,
        targets: MotionTrajectoryTargets,
        source_channel_normal_owned: torch.Tensor,
        source_channel_confidence: torch.Tensor,
        body_dof_ancestry: np.ndarray,
    ) -> None:
        super().__init__()
        required_position_count = len(targets.required_position_rows)
        required_distal_count = len(targets.required_direction_rows)
        contact_count = len(targets.contact_direction_rows)
        ancestry = np.asarray(body_dof_ancestry, dtype=np.uint8)
        if required_position_count < 1 or required_distal_count < 1 or ancestry.ndim != 2:
            raise ValueError("Source-fidelity bounds require nonempty rows and ancestry.")
        guard_count = required_position_count + 2 * required_distal_count + 1
        ancestry_state = np.empty((guard_count, ancestry.shape[1]), dtype=np.uint8)
        for guard, position_row in enumerate(targets.required_position_rows):
            ancestry_state[guard] = ancestry[targets.position_body_indices[position_row]]
        distal_start = required_position_count
        for required, direction_row in enumerate(targets.required_direction_rows):
            position_row = targets.direction_position_rows[direction_row]
            point_body = targets.direction_body_indices[direction_row]
            base_body = targets.position_body_indices[position_row]
            ancestry_state[distal_start + required] = ancestry[point_body]
            ancestry_state[distal_start + required_distal_count + required] = ancestry[point_body] | (
                ancestry[base_body] << np.uint8(1)
            )
        ancestry_state[-1] = ancestry[targets.root_body_index]
        tensors = (
            targets.source_landmark_position_m,
            targets.source_direction_point_position_m,
            source_channel_normal_owned,
            source_channel_confidence,
            targets.source_landmark_rotation_xyzw,
            targets.position_body_index_tensor,
            targets.direction_body_index_tensor,
            targets.direction_position_row_tensor,
            targets.required_position_row_tensor,
            targets.position_normal_channel_slots,
            targets.contact_direction_row_tensor,
            targets.direction_contact_channel_slots,
            targets.required_direction_row_tensor,
            targets.direction_point_body_m,
            targets.contact_distal_point_body_m,
        )
        if (
            targets.source_landmark_position_m.ndim != 3
            or targets.source_direction_point_position_m.ndim != 3
            or targets.source_landmark_rotation_xyzw.ndim != 3
            or source_channel_confidence.shape != (targets.source_landmark_position_m.shape[1], contact_count)
            or source_channel_normal_owned.shape != source_channel_confidence.shape
            or source_channel_normal_owned.dtype != torch.float32
            or source_channel_confidence.dtype != torch.float32
            or targets.source_landmark_rotation_xyzw.shape[0] < 1
            or targets.direction_point_body_m.shape != (len(targets.direction_body_indices), 3)
            or targets.contact_distal_point_body_m.shape != (contact_count, 3)
            or targets.required_position_row_tensor.shape != (required_position_count,)
            or targets.position_normal_channel_slots.shape != (len(targets.position_body_indices),)
            or targets.contact_direction_row_tensor.shape != (contact_count,)
            or targets.direction_contact_channel_slots.shape != (len(targets.direction_body_indices),)
            or targets.required_direction_row_tensor.shape != (required_distal_count,)
            or any(
                tensor.dtype is not torch.int64
                for tensor in (
                    targets.required_position_row_tensor,
                    targets.position_normal_channel_slots,
                    targets.contact_direction_row_tensor,
                    targets.direction_contact_channel_slots,
                    targets.required_direction_row_tensor,
                )
            )
            or any(not tensor.is_contiguous() for tensor in tensors)
            or any(tensor.device != targets.source_landmark_position_m.device for tensor in tensors)
        ):
            raise ValueError("Source-fidelity guard geometry must use contiguous same-device target tensors.")
        self._source_landmark_position_t = targets.source_landmark_position_m
        self._source_direction_point_position_t = targets.source_direction_point_position_m
        self._source_channel_normal_owned_t = source_channel_normal_owned
        self._source_channel_confidence_t = source_channel_confidence
        self._source_landmark_rotation_t = targets.source_landmark_rotation_xyzw
        self._position_body_indices_t = targets.position_body_index_tensor
        self._direction_body_indices_t = targets.direction_body_index_tensor
        self._direction_position_rows_t = targets.direction_position_row_tensor
        self._required_position_rows_t = targets.required_position_row_tensor
        self._position_normal_channel_slots_t = targets.position_normal_channel_slots
        self._contact_direction_rows_t = targets.contact_direction_row_tensor
        self._direction_contact_channel_slots_t = targets.direction_contact_channel_slots
        self._required_direction_rows_t = targets.required_direction_row_tensor
        self._direction_point_body_t = targets.direction_point_body_m
        self._contact_distal_point_body_t = targets.contact_distal_point_body_m
        self._required_position_count = required_position_count
        self._required_distal_count = required_distal_count
        self._root_body = targets.root_body_index
        self._ancestry_state_np = ancestry_state

    def supports_analytic(self) -> bool:
        """Return true because the packed guard supplies its exact Jacobian."""
        return True

    def residual_dim(self) -> int:
        """Return required global, distal, direction, and root guard rows."""
        return self._ancestry_state_np.shape[0]

    def init_buffers(self, model, jacobian_mode: ik.IKJacobianType) -> None:
        """Bind caller-owned source evidence and packed uint8 ancestry state."""
        self._require_batch_layout()
        if (
            self._source_landmark_position_t.shape[1] != self.n_batch
            or self._source_direction_point_position_t.shape[1] != self.n_batch
            or self._source_channel_normal_owned_t.shape[0] != self.n_batch
            or self._source_channel_confidence_t.shape[0] != self.n_batch
            or self._source_landmark_rotation_t.shape[1] != self.n_batch
            or self._ancestry_state_np.shape != (self.residual_dim(), model.joint_dof_count)
        ):
            raise ValueError("Source-fidelity guard evidence or ancestry differs from the IK batch.")
        if jacobian_mode not in (ik.IKJacobianType.ANALYTIC, ik.IKJacobianType.MIXED):
            raise ValueError("Source-fidelity guards require an analytic or mixed Jacobian.")
        self._source_landmark_position = wp.from_torch(self._source_landmark_position_t)
        self._source_direction_point_position = wp.from_torch(self._source_direction_point_position_t)
        self._source_channel_normal_owned = wp.from_torch(self._source_channel_normal_owned_t, dtype=wp.float32)
        self._source_channel_confidence = wp.from_torch(self._source_channel_confidence_t, dtype=wp.float32)
        self._source_landmark_rotation = wp.from_torch(self._source_landmark_rotation_t)
        self._position_body_indices = wp.from_torch(self._position_body_indices_t, dtype=wp.int64)
        self._direction_body_indices = wp.from_torch(self._direction_body_indices_t, dtype=wp.int64)
        self._direction_position_rows = wp.from_torch(self._direction_position_rows_t, dtype=wp.int64)
        self._required_position_rows = wp.from_torch(self._required_position_rows_t, dtype=wp.int64)
        self._position_normal_channel_slots = wp.from_torch(self._position_normal_channel_slots_t, dtype=wp.int64)
        self._contact_direction_rows = wp.from_torch(self._contact_direction_rows_t, dtype=wp.int64)
        self._direction_contact_channel_slots = wp.from_torch(self._direction_contact_channel_slots_t, dtype=wp.int64)
        self._required_direction_rows = wp.from_torch(self._required_direction_rows_t, dtype=wp.int64)
        self._direction_point_body = wp.from_torch(self._direction_point_body_t)
        self._contact_distal_point_body = wp.from_torch(self._contact_distal_point_body_t)
        self._ancestry_state = wp.from_numpy(self._ancestry_state_np, dtype=wp.uint8, device=self.device)

    def estimate_memory(self, model, jacobian_mode, n_problems, n_batch, total_residuals) -> int:
        """Estimate the objective-owned two-bit ancestry state [byte]."""
        del model, n_problems, n_batch, total_residuals
        if jacobian_mode not in (ik.IKJacobianType.ANALYTIC, ik.IKJacobianType.MIXED):
            raise ValueError("Source-fidelity guards require an analytic or mixed Jacobian.")
        return self._ancestry_state_np.nbytes

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        """Write packed bounded residuals using the identity-mapped source evidence."""
        del joint_q, model
        wp.launch(
            _source_fidelity_guard_residuals,
            dim=(body_q.shape[0], self.residual_dim()),
            inputs=[
                body_q,
                self._source_landmark_position,
                self._source_direction_point_position,
                self._source_landmark_rotation,
                self._position_body_indices,
                self._direction_body_indices,
                self._direction_position_rows,
                self._required_position_rows,
                self._position_normal_channel_slots,
                self._contact_direction_rows,
                self._direction_contact_channel_slots,
                self._required_direction_rows,
                self._source_channel_confidence,
                self._source_channel_normal_owned,
                self._direction_point_body,
                self._contact_distal_point_body,
                self._required_position_count,
                self._required_distal_count,
                self._root_body,
                start_idx,
                problem_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_screw, start_idx) -> None:
        """Write exact active guard derivatives for identity-mapped frames."""
        del joint_q
        wp.launch(
            _source_fidelity_guard_jacobian,
            dim=(body_q.shape[0], self.residual_dim(), model.joint_dof_count),
            inputs=[
                body_q,
                joint_screw,
                self._source_landmark_position,
                self._source_direction_point_position,
                self._source_landmark_rotation,
                self._position_body_indices,
                self._direction_body_indices,
                self._direction_position_rows,
                self._required_position_rows,
                self._position_normal_channel_slots,
                self._contact_direction_rows,
                self._direction_contact_channel_slots,
                self._required_direction_rows,
                self._source_channel_confidence,
                self._source_channel_normal_owned,
                self._direction_point_body,
                self._contact_distal_point_body,
                self._required_position_count,
                self._required_distal_count,
                self._root_body,
                self._ancestry_state,
                start_idx,
            ],
            outputs=[jacobian],
            device=self.device,
        )


def _motion_frame_seed_objectives(
    cfg: MotionTrajectorySolveCfg,
    targets: MotionTrajectoryTargets,
) -> list[object]:
    """Build the frame-local morphology fit that seeds the whole-trajectory solve."""
    expected = (
        MotionSourceGlobalPositionObjectiveCfg,
        MotionSourceRotationObjectiveCfg,
        MotionSourceDirectionPointObjectiveCfg,
    )
    matched_cfgs = tuple(objective for objective in cfg.objectives if type(objective) in expected)
    if len(matched_cfgs) != len(expected) or {type(objective) for objective in matched_cfgs} != set(expected):
        raise ValueError("Motion frame initialization requires one position, rotation, and distal term.")
    objective_cfgs = {type(objective): objective for objective in matched_cfgs}
    objectives = motion_objective_source_global_position(
        objective_cfgs[MotionSourceGlobalPositionObjectiveCfg], targets
    )
    objectives.extend(motion_objective_source_rotation(objective_cfgs[MotionSourceRotationObjectiveCfg], targets))
    objectives.extend(
        motion_objective_source_direction_point(objective_cfgs[MotionSourceDirectionPointObjectiveCfg], targets)
    )
    return objectives


def motion_objective_contact(
    _cfg: MotionContactObjectiveCfg,
    targets: MotionTrajectoryTargets,
    reference,
    support_pose: torch.Tensor,
    contact_cfg: MotionTrajectorySolveCfg.ContactCfg,
    acceptance: MotionTrajectorySolveCfg.AcceptanceCfg.ContactCfg,
) -> list[object]:
    """Build soft support gap, upright, and planted-point errors."""
    ancestry = reference.topology.body_dof_ancestry
    objectives = []
    for channel, (start, stop) in enumerate(
        zip(targets.support_patch_offsets[:-1], targets.support_patch_offsets[1:], strict=True)
    ):
        body = int(targets.contact_body_indices[channel])
        objectives.append(
            IKObjectiveSupportPatch(
                body=body,
                points_body=targets.support_point_body_m[start:stop],
                target_points_world=targets.target_support_position_m[start:stop],
                normal_body=targets.contact_normal_body[channel],
                support_pose=support_pose,
                affects_dof=ancestry[body],
                gap_tolerance_m=acceptance.gap_upper_m,
                tilt_tolerance_rad=acceptance.tilt_upper_rad,
                point_tolerance_m=contact_cfg.point_tolerance_m,
            )
        )
    return objectives


def _motion_clip_batch_index(clip_index: MotionClipIndex, start: int, stop: int) -> MotionClipIndex:
    """Slice complete clips and remap only the skeleton identities used by the batch."""
    clips = []
    skeleton_ids: dict[int, int] = {}
    skeleton_identities = []
    for clip in clip_index.clips[start:stop]:
        if clip.skeleton_id not in skeleton_ids:
            skeleton_ids[clip.skeleton_id] = len(skeleton_ids)
            skeleton_identities.append(clip_index.skeleton_identity_sha256s[clip.skeleton_id])
        clips.append(replace(clip, skeleton_id=skeleton_ids[clip.skeleton_id]))
    return MotionClipIndex(
        source_content_sha256=clip_index.source_content_sha256,
        skeleton_identity_sha256s=tuple(skeleton_identities),
        clips=tuple(clips),
    )


def _motion_trajectory_residual_layout(
    targets: MotionTrajectoryTargets,
    collision_probe_count: int,
    term_cfgs: tuple[StateCommandCfg.TaskTableCfg.ObjectiveCfg | IKObjectiveBaseCfg, ...],
) -> _MotionTrajectoryResidualLayout:
    """Return row ownership derived from objective identities and declaration order."""
    position_count = len(targets.position_body_indices)
    cursor = 0
    owned: dict[type, slice] = {}
    widths = {
        MotionSourceGlobalPositionObjectiveCfg: 3 * position_count,
        MotionSourceRotationObjectiveCfg: 3 * len(targets.rotation_body_indices),
        MotionSourceDirectionPointObjectiveCfg: 3 * len(targets.direction_body_indices),
        MotionContactObjectiveCfg: sum(
            4 + 3 * (stop - start)
            for start, stop in zip(targets.support_patch_offsets[:-1], targets.support_patch_offsets[1:], strict=True)
        ),
        IKObjectiveJointDefaultCfg: targets.coordinate_indices.shape[0],
        IKObjectiveJointPinCfg: targets.coordinate_indices.shape[0],
        IKObjectiveMeshCollisionCfg: collision_probe_count,
        IKObjectiveMeshNonpenetrationCfg: collision_probe_count,
    }
    for term_cfg in term_cfgs:
        term_type = type(term_cfg)
        if term_type not in widths or term_type in owned:
            raise ValueError(f"Motion trajectory term {term_type.__name__} is unsupported or duplicated.")
        owned[term_type] = slice(cursor, cursor + widths[term_type])
        cursor += widths[term_type]
    missing = tuple(term_type.__name__ for term_type in widths if term_type not in owned)
    if missing:
        raise ValueError(f"Motion trajectory solve is missing required terms: {missing}.")
    contact_channel_count = len(targets.support_patch_offsets) - 1
    guard_count = len(targets.required_position_rows) + 2 * len(targets.required_direction_rows) + 1
    source_fidelity_guard = slice(cursor, cursor + guard_count)
    cursor = source_fidelity_guard.stop
    source_global = owned[MotionSourceGlobalPositionObjectiveCfg]
    source_rotation = owned[MotionSourceRotationObjectiveCfg]
    source_direction_point = owned[MotionSourceDirectionPointObjectiveCfg]
    contact = owned[MotionContactObjectiveCfg]
    activity_group_by_residual = torch.full(
        (cursor,), -1, dtype=torch.int32, device=targets.source_landmark_position_m.device
    )
    first_difference_group_by_residual = torch.full_like(activity_group_by_residual, -1)
    contact_cursor = contact.start
    for channel, (start, stop) in enumerate(
        zip(targets.support_patch_offsets[:-1], targets.support_patch_offsets[1:], strict=True)
    ):
        support_width = 4 + 3 * (stop - start)
        activity_group_by_residual[contact_cursor : contact_cursor + support_width] = channel
        first_difference_group_by_residual[contact_cursor + 4 : contact_cursor + support_width] = (
            contact_channel_count + channel
        )
        contact_cursor += support_width
    if contact_cursor != contact.stop:
        raise RuntimeError("Motion contact rows differ from the declared support patches.")

    return _MotionTrajectoryResidualLayout(
        source_global_position=source_global,
        source_rotation=source_rotation,
        source_direction_point=source_direction_point,
        source_fidelity_guard=source_fidelity_guard,
        contact=contact,
        activity_group_by_residual=activity_group_by_residual,
        first_difference_group_by_residual=first_difference_group_by_residual,
        joint_default=owned[IKObjectiveJointDefaultCfg],
        joint_reference=owned[IKObjectiveJointPinCfg],
        collision_objective=owned[IKObjectiveMeshCollisionCfg],
        nonpenetration_objective=owned[IKObjectiveMeshNonpenetrationCfg],
    )


def _motion_source_fidelity_inequalities(
    layout: _MotionTrajectoryResidualLayout,
    targets: MotionTrajectoryTargets,
    policy: MotionTrajectorySolveCfg.AcceptanceCfg.SourceCfg,
    kkt_relative_tolerance: float,
) -> IKTrajectorySolver.ResidualInequalities:
    """Bind source rows inside publication limits by two relative KKT tolerances."""
    position_count = len(targets.required_position_rows)
    distal_count = len(targets.required_direction_rows)
    expected_count = position_count + 2 * distal_count + 1
    if (
        layout.source_fidelity_guard.stop - layout.source_fidelity_guard.start != expected_count
        or position_count < 1
        or distal_count < 1
        or not math.isfinite(kkt_relative_tolerance)
        or not 0.0 < kkt_relative_tolerance < 1.0
    ):
        raise ValueError("Packed source-fidelity rows differ from the target-required metrics.")
    device = targets.source_landmark_position_m.device
    interior_scale = 1.0 / (1.0 + 2.0 * kkt_relative_tolerance)
    indices = torch.arange(
        layout.source_fidelity_guard.start, layout.source_fidelity_guard.stop, dtype=torch.int32, device=device
    )
    upper = torch.empty(expected_count, dtype=torch.float32, device=device)
    position_stop = position_count
    distal_stop = position_stop + distal_count
    direction_stop = distal_stop + distal_count
    upper[:position_stop].fill_(policy.required_position_upper_m)
    upper[position_stop:distal_stop].fill_(policy.required_distal_position_upper_m)
    upper[distal_stop:direction_stop].fill_(2.0 * math.sin(0.5 * policy.required_distal_direction_upper_rad))
    upper[direction_stop:].fill_(2.0 * math.sin(0.5 * policy.root_rotation_upper_rad))
    upper.mul_(interior_scale)
    return IKTrajectorySolver.ResidualInequalities(residual_indices=indices, upper=upper)


def _motion_phase_update(
    segment_active: torch.Tensor,
    constraint_geometry_feasible: torch.Tensor,
    inner_solve_converged: torch.Tensor,
    globalization_succeeded: torch.Tensor,
    iteration_geometry_feasible: torch.Tensor,
    iteration_inner_converged: torch.Tensor,
    iteration_globalization_succeeded: torch.Tensor,
) -> None:
    """Accumulate independent iteration outcomes and stop failed segments."""
    torch.logical_and(constraint_geometry_feasible, iteration_geometry_feasible, out=constraint_geometry_feasible)
    torch.logical_and(inner_solve_converged, iteration_inner_converged, out=inner_solve_converged)
    torch.logical_and(globalization_succeeded, iteration_globalization_succeeded, out=globalization_succeeded)
    torch.logical_and(iteration_geometry_feasible, iteration_inner_converged, out=iteration_geometry_feasible)
    torch.logical_and(
        iteration_geometry_feasible,
        iteration_globalization_succeeded,
        out=iteration_geometry_feasible,
    )
    segment_active.mul_(iteration_geometry_feasible)


def _motion_phase_finish(
    segment_active: torch.Tensor,
    constraint_geometry_feasible: torch.Tensor,
    inner_solve_converged: torch.Tensor,
    globalization_succeeded: torch.Tensor,
    nonlinear_phases_converged: torch.Tensor,
    phase_converged: torch.Tensor,
) -> None:
    """Accumulate nonlinear convergence without conflating failure causes."""
    torch.eq(segment_active, 0, out=phase_converged)
    torch.logical_and(phase_converged, constraint_geometry_feasible, out=phase_converged)
    torch.logical_and(phase_converged, inner_solve_converged, out=phase_converged)
    torch.logical_and(phase_converged, globalization_succeeded, out=phase_converged)
    torch.logical_and(nonlinear_phases_converged, phase_converged, out=nonlinear_phases_converged)


def _motion_source_weights(
    layout: _MotionTrajectoryResidualLayout,
    cfg: MotionTrajectorySolveCfg,
    targets: MotionTrajectoryTargets,
    base_weights: torch.Tensor,
    temporal_weights: torch.Tensor,
) -> None:
    """Materialize source tracking, acceptance-normalized guard margins, and joint precision."""
    if base_weights.shape != (layout.residual_count,) or temporal_weights.shape != (3, layout.residual_count):
        raise ValueError("Motion projection weights differ from the explicit residual layout.")
    base_weights.zero_()
    temporal_weights.zero_()
    policy = cfg.acceptance.source
    position_count = len(targets.required_position_rows)
    distal_count = len(targets.required_direction_rows)
    guard = layout.source_fidelity_guard
    if guard.stop - guard.start != position_count + 2 * distal_count + 1:
        raise ValueError("Source guard precision differs from the publication-metric layout.")
    position_stop = guard.start + position_count
    distal_stop = position_stop + distal_count
    direction_stop = distal_stop + distal_count
    direction_chord = 2.0 * math.sin(0.5 * policy.required_distal_direction_upper_rad)
    root_chord = 2.0 * math.sin(0.5 * policy.root_rotation_upper_rad)
    base_weights[guard.start : position_stop].fill_(1.0 / policy.required_position_upper_m**2)
    base_weights[position_stop:distal_stop].fill_(1.0 / policy.required_distal_position_upper_m**2)
    base_weights[distal_stop:direction_stop].fill_(1.0 / direction_chord**2)
    base_weights[direction_stop : guard.stop].fill_(1.0 / root_chord**2)

    base_weights[layout.source_global_position] = 1.0
    base_weights[layout.source_rotation] = 1.0
    base_weights[layout.source_direction_point] = 1.0
    base_weights[layout.joint_default] = cfg.joint_default_position_weight
    temporal_weights[0, layout.source_global_position] = cfg.source_position_velocity_weight
    temporal_weights[1, layout.source_global_position] = cfg.source_position_acceleration_weight
    temporal_weights[0, layout.source_direction_point] = cfg.source_position_velocity_weight
    temporal_weights[1, layout.source_direction_point] = cfg.source_position_acceleration_weight
    temporal_weights[0, layout.source_rotation] = cfg.source_rotation_velocity_weight
    temporal_weights[1, layout.source_rotation] = cfg.source_rotation_acceleration_weight
    temporal_weights[0, layout.joint_default] = cfg.joint_temporal_velocity_weight
    temporal_weights[1, layout.joint_default] = cfg.joint_temporal_acceleration_weight
    temporal_weights[2, layout.joint_default] = cfg.joint_temporal_jerk_weight


def _motion_physical_projection_weights(
    layout: _MotionTrajectoryResidualLayout,
    cfg: MotionTrajectorySolveCfg,
    targets: MotionTrajectoryTargets,
    base_weights: torch.Tensor,
    temporal_weights: torch.Tensor,
) -> None:
    """Project physical constraints while retaining source tracking and certified-q trust."""
    _motion_source_weights(layout, cfg, targets, base_weights, temporal_weights)
    base_weights[layout.joint_reference] = 1.0
    temporal_weights[0, layout.joint_reference] = cfg.joint_temporal_velocity_weight
    temporal_weights[1, layout.joint_reference] = cfg.joint_temporal_acceleration_weight
    temporal_weights[2, layout.joint_reference] = cfg.joint_temporal_jerk_weight
    base_weights[layout.collision_objective] = 1.0
    base_weights[layout.nonpenetration_objective] = 1.0


def _motion_contact_weights(
    layout: _MotionTrajectoryResidualLayout,
    cfg: MotionTrajectorySolveCfg,
    targets: MotionTrajectoryTargets,
    base_weights: torch.Tensor,
    temporal_weights: torch.Tensor,
) -> None:
    """Add confidence-normalized support-plane, upright, and no-slip precision."""
    _motion_physical_projection_weights(layout, cfg, targets, base_weights, temporal_weights)
    relative_point_precision = (cfg.contact.point_tolerance_m / cfg.acceptance.contact.slip_speed_upper_mps) ** 2
    cursor = layout.contact.start
    for start, stop in zip(targets.support_patch_offsets[:-1], targets.support_patch_offsets[1:], strict=True):
        width = 4 + 3 * (stop - start)
        base_weights[cursor : cursor + 4] = 1.0
        base_weights[cursor + 6 : cursor + width : 3] = 1.0
        temporal_weights[0, cursor + 4 : cursor + width] = relative_point_precision
        cursor += width
    if cursor != layout.contact.stop:
        raise RuntimeError("Motion contact weights differ from the declared support patches.")


def _motion_monolithic_weights(
    layout: _MotionTrajectoryResidualLayout,
    cfg: MotionTrajectorySolveCfg,
    targets: MotionTrajectoryTargets,
    base_weights: torch.Tensor,
    temporal_weights: torch.Tensor,
) -> None:
    """Activate one joint source, physical, and contact objective without phase guards."""
    _motion_contact_weights(layout, cfg, targets, base_weights, temporal_weights)
    base_weights[layout.source_fidelity_guard] = 0.0
    base_weights[layout.joint_reference] = 0.0
    temporal_weights[:, layout.joint_reference] = 0.0


@wp.kernel
def _trajectory_support_points(
    body_q: wp.array3d(dtype=wp.float32),
    support_body_indices: wp.array(dtype=wp.int64),
    support_point_body_m: wp.array2d(dtype=wp.float32),
    frame_count: int,
    support_count: int,
    output: wp.array3d(dtype=wp.float32),
):
    support, frame = wp.tid()
    if support >= support_count or frame >= frame_count:
        return
    body = support_body_indices[support]
    rotation = wp.quat(body_q[frame, body, 3], body_q[frame, body, 4], body_q[frame, body, 5], body_q[frame, body, 6])
    offset = wp.vec3(
        support_point_body_m[support, 0],
        support_point_body_m[support, 1],
        support_point_body_m[support, 2],
    )
    point = wp.vec3(body_q[frame, body, 0], body_q[frame, body, 1], body_q[frame, body, 2])
    point += wp.quat_rotate(rotation, offset)
    output[support, frame, 0] = point[0]
    output[support, frame, 1] = point[1]
    output[support, frame, 2] = point[2]


@wp.func
def _trajectory_rotation_error(actual_rotation: wp.quat, source_rotation: wp.quat) -> float:
    actual_norm = wp.sqrt(wp.dot(actual_rotation, actual_rotation))
    source_norm = wp.sqrt(wp.dot(source_rotation, source_rotation))
    error = float(wp.inf)
    if wp.isfinite(actual_norm) and wp.isfinite(source_norm) and actual_norm > 1.0e-8 and source_norm > 1.0e-8:
        rotation_dot = wp.abs(wp.dot(actual_rotation, source_rotation) / (actual_norm * source_norm))
        error = 2.0 * wp.acos(wp.clamp(rotation_dot, 0.0, 1.0))
        if not wp.isfinite(error):
            error = float(wp.inf)
    return error


@wp.kernel
def _trajectory_quality_frames(
    body_q: wp.array3d(dtype=wp.float32),
    source_landmark_positions: wp.array3d(dtype=wp.float32),
    position_body_indices: wp.array(dtype=wp.int64),
    required_position_rows: wp.array(dtype=wp.int64),
    position_normal_channel_slots: wp.array(dtype=wp.int64),
    parent_rows: wp.array(dtype=wp.int64),
    source_landmark_rotations: wp.array3d(dtype=wp.float32),
    rotation_body_indices: wp.array(dtype=wp.int64),
    source_direction_points: wp.array3d(dtype=wp.float32),
    direction_body_indices: wp.array(dtype=wp.int64),
    direction_position_rows: wp.array(dtype=wp.int64),
    contact_direction_rows: wp.array(dtype=wp.int64),
    direction_contact_channel_slots: wp.array(dtype=wp.int64),
    required_direction_rows: wp.array(dtype=wp.int64),
    source_channel_confidence: wp.array2d(dtype=wp.float32),
    source_channel_normal_owned: wp.array2d(dtype=wp.float32),
    direction_point_body_m: wp.array2d(dtype=wp.float32),
    contact_distal_point_body_m: wp.array2d(dtype=wp.float32),
    frame_count: int,
    position_count: int,
    rotation_count: int,
    direction_count: int,
    required_position_count: int,
    required_direction_count: int,
    metric_count: int,
    metric_source_required_position: int,
    metric_source_required_distal_position: int,
    metric_source_required_distal_direction: int,
    metric_source_root_rotation: int,
    metric_source_all_position: int,
    metric_source_all_distal_position: int,
    metric_source_all_landmark_direction: int,
    metric_source_all_distal_direction: int,
    metric_source_nonroot_rotation: int,
    quality: wp.array2d(dtype=wp.float32),
):
    """Measure frame-local semantic source fidelity with strict finite directions."""
    frame = wp.tid()
    if frame >= frame_count:
        return
    for metric in range(metric_count):
        quality[frame, metric] = 0.0

    source_required_position_error = float(0.0)
    source_all_position_error = float(0.0)
    for row in range(position_count):
        body = position_body_indices[row]
        delta = wp.vec3(
            body_q[frame, body, 0] - source_landmark_positions[row, frame, 0],
            body_q[frame, body, 1] - source_landmark_positions[row, frame, 1],
            body_q[frame, body, 2] - source_landmark_positions[row, frame, 2],
        )
        error = wp.length(delta)
        if not wp.isfinite(error):
            error = float(wp.inf)
        source_all_position_error = wp.max(source_all_position_error, error)
        for required_index in range(required_position_count):
            if required_position_rows[required_index] == row:
                contact_channel = position_normal_channel_slots[row]
                contact_owned = (
                    contact_channel >= 0
                    and source_channel_confidence[frame, contact_channel] > 0.0
                    and _source_position_is_coincident_distal(
                        wp.int64(row),
                        contact_channel,
                        position_body_indices,
                        direction_body_indices,
                        contact_direction_rows,
                        direction_point_body_m,
                    )
                    != 0
                )
                if not contact_owned:
                    required_delta = delta
                    if contact_channel >= 0 and source_channel_normal_owned[frame, contact_channel] > 0.0:
                        required_delta = wp.vec3(required_delta[0], required_delta[1], 0.0)
                    required_error = wp.length(required_delta)
                    if not wp.isfinite(required_error):
                        required_error = float(wp.inf)
                    source_required_position_error = wp.max(source_required_position_error, required_error)

    source_all_landmark_direction_error = float(0.0)
    for row in range(1, position_count):
        parent_row = parent_rows[row]
        body = position_body_indices[row]
        parent_body = position_body_indices[parent_row]
        actual_direction = wp.vec3(
            body_q[frame, body, 0] - body_q[frame, parent_body, 0],
            body_q[frame, body, 1] - body_q[frame, parent_body, 1],
            body_q[frame, body, 2] - body_q[frame, parent_body, 2],
        )
        source_direction = wp.vec3(
            source_landmark_positions[row, frame, 0] - source_landmark_positions[parent_row, frame, 0],
            source_landmark_positions[row, frame, 1] - source_landmark_positions[parent_row, frame, 1],
            source_landmark_positions[row, frame, 2] - source_landmark_positions[parent_row, frame, 2],
        )
        actual_length = wp.length(actual_direction)
        source_length = wp.length(source_direction)
        if (
            not wp.isfinite(actual_length)
            or not wp.isfinite(source_length)
            or actual_length <= 1.0e-8
            or source_length <= 1.0e-8
        ):
            source_all_landmark_direction_error = float(wp.inf)
        else:
            cosine = wp.dot(actual_direction, source_direction) / (actual_length * source_length)
            angle = wp.acos(wp.clamp(cosine, -1.0, 1.0))
            if not wp.isfinite(angle):
                source_all_landmark_direction_error = float(wp.inf)
            else:
                source_all_landmark_direction_error = wp.max(source_all_landmark_direction_error, angle)

    source_required_distal_position_error = float(0.0)
    source_required_distal_direction_error = float(0.0)
    source_all_distal_position_error = float(0.0)
    source_all_distal_direction_error = float(0.0)
    for row in range(direction_count):
        body = direction_body_indices[row]
        position_row = direction_position_rows[row]
        base_body = position_body_indices[position_row]
        rotation = wp.quat(
            body_q[frame, body, 3],
            body_q[frame, body, 4],
            body_q[frame, body, 5],
            body_q[frame, body, 6],
        )
        point_body = wp.vec3(
            direction_point_body_m[row, 0],
            direction_point_body_m[row, 1],
            direction_point_body_m[row, 2],
        )
        actual_point = wp.vec3(body_q[frame, body, 0], body_q[frame, body, 1], body_q[frame, body, 2])
        actual_point += wp.quat_rotate(rotation, point_body)
        source_point = wp.vec3(
            source_direction_points[row, frame, 0],
            source_direction_points[row, frame, 1],
            source_direction_points[row, frame, 2],
        )
        required_row = int(-1)
        for required_index in range(required_direction_count):
            if required_direction_rows[required_index] == row:
                required_row = required_index
        contact_channel = direction_contact_channel_slots[row]
        position_delta = actual_point - source_point
        position_error = wp.length(position_delta)
        if not wp.isfinite(position_error):
            position_error = float(wp.inf)
        source_all_distal_position_error = wp.max(source_all_distal_position_error, position_error)
        if required_row >= 0:
            required_source_point = source_point
            if contact_channel >= 0 and source_channel_confidence[frame, contact_channel] > 0.0:
                source_base = wp.vec3(
                    source_landmark_positions[position_row, frame, 0],
                    source_landmark_positions[position_row, frame, 1],
                    source_landmark_positions[position_row, frame, 2],
                )
                contact_tangent = wp.vec3(
                    contact_distal_point_body_m[contact_channel, 0],
                    contact_distal_point_body_m[contact_channel, 1],
                    contact_distal_point_body_m[contact_channel, 2],
                )
                required_source_point = _source_contact_planar_distal_target(
                    source_base, source_point, wp.length(contact_tangent)
                )
            required_position_delta = actual_point - required_source_point
            if contact_channel >= 0 and source_channel_normal_owned[frame, contact_channel] > 0.0:
                required_position_delta = wp.vec3(required_position_delta[0], required_position_delta[1], 0.0)
            required_position_error = wp.length(required_position_delta)
            if not wp.isfinite(required_position_error):
                required_position_error = float(wp.inf)
            source_required_distal_position_error = wp.max(
                source_required_distal_position_error, required_position_error
            )
        actual_direction = actual_point - wp.vec3(
            body_q[frame, base_body, 0], body_q[frame, base_body, 1], body_q[frame, base_body, 2]
        )
        source_direction = source_point - wp.vec3(
            source_landmark_positions[position_row, frame, 0],
            source_landmark_positions[position_row, frame, 1],
            source_landmark_positions[position_row, frame, 2],
        )
        actual_length = wp.length(actual_direction)
        source_length = wp.length(source_direction)
        direction_error = float(0.0)
        if (
            not wp.isfinite(actual_length)
            or not wp.isfinite(source_length)
            or actual_length <= 1.0e-8
            or source_length <= 1.0e-8
        ):
            direction_error = float(wp.inf)
        else:
            cosine = wp.dot(actual_direction, source_direction) / (actual_length * source_length)
            angle = wp.acos(wp.clamp(cosine, -1.0, 1.0))
            if not wp.isfinite(angle):
                direction_error = float(wp.inf)
            else:
                direction_error = angle
        source_all_distal_direction_error = wp.max(source_all_distal_direction_error, direction_error)
        if required_row >= 0:
            required_actual_direction = actual_direction
            required_source_direction = source_direction
            if contact_channel >= 0 and source_channel_confidence[frame, contact_channel] > 0.0:
                required_actual_direction = wp.vec3(required_actual_direction[0], required_actual_direction[1], 0.0)
                required_source_direction = wp.vec3(required_source_direction[0], required_source_direction[1], 0.0)
            required_actual_length = wp.length(required_actual_direction)
            required_source_length = wp.length(required_source_direction)
            required_direction_error = float(0.0)
            if (
                not wp.isfinite(required_actual_length)
                or not wp.isfinite(required_source_length)
                or required_actual_length <= 1.0e-8
                or required_source_length <= 1.0e-8
            ):
                required_direction_error = float(wp.inf)
            else:
                required_cosine = wp.dot(required_actual_direction, required_source_direction) / (
                    required_actual_length * required_source_length
                )
                required_angle = wp.acos(wp.clamp(required_cosine, -1.0, 1.0))
                if not wp.isfinite(required_angle):
                    required_direction_error = float(wp.inf)
                else:
                    required_direction_error = required_angle
            source_required_distal_direction_error = wp.max(
                source_required_distal_direction_error, required_direction_error
            )

    source_root_rotation_error = float(wp.inf)
    source_nonroot_rotation_error = float(wp.nan)
    source_nonroot_rotation_count = int(0)
    for row in range(rotation_count):
        body = rotation_body_indices[row]
        actual_rotation = wp.quat(
            body_q[frame, body, 3],
            body_q[frame, body, 4],
            body_q[frame, body, 5],
            body_q[frame, body, 6],
        )
        source_rotation = wp.quat(
            source_landmark_rotations[row, frame, 0],
            source_landmark_rotations[row, frame, 1],
            source_landmark_rotations[row, frame, 2],
            source_landmark_rotations[row, frame, 3],
        )
        rotation_error = _trajectory_rotation_error(actual_rotation, source_rotation)
        if row == 0:
            source_root_rotation_error = rotation_error
        elif source_nonroot_rotation_count == 0:
            source_nonroot_rotation_error = rotation_error
            source_nonroot_rotation_count = 1
        else:
            source_nonroot_rotation_error = wp.max(source_nonroot_rotation_error, rotation_error)

    quality[frame, metric_source_required_position] = source_required_position_error
    quality[frame, metric_source_required_distal_position] = source_required_distal_position_error
    quality[frame, metric_source_required_distal_direction] = source_required_distal_direction_error
    quality[frame, metric_source_root_rotation] = source_root_rotation_error
    quality[frame, metric_source_all_position] = source_all_position_error
    quality[frame, metric_source_all_distal_position] = source_all_distal_position_error
    quality[frame, metric_source_all_landmark_direction] = source_all_landmark_direction_error
    quality[frame, metric_source_all_distal_direction] = source_all_distal_direction_error
    quality[frame, metric_source_nonroot_rotation] = source_nonroot_rotation_error


@wp.func
def _trajectory_support_coordinate(
    body_q: wp.array3d(dtype=wp.float32),
    support_body_indices: wp.array(dtype=wp.int64),
    support_point_body_m: wp.array2d(dtype=wp.float32),
    obstacle_pose: wp.array2d(dtype=wp.float32),
    point: int,
    frame: int,
) -> wp.vec3:
    body = support_body_indices[point]
    body_rotation = wp.quat(
        body_q[frame, body, 3], body_q[frame, body, 4], body_q[frame, body, 5], body_q[frame, body, 6]
    )
    point_body = wp.vec3(support_point_body_m[point, 0], support_point_body_m[point, 1], support_point_body_m[point, 2])
    point_world = wp.vec3(body_q[frame, body, 0], body_q[frame, body, 1], body_q[frame, body, 2])
    point_world += wp.quat_rotate(body_rotation, point_body)
    support_origin = wp.vec3(obstacle_pose[frame, 0], obstacle_pose[frame, 1], obstacle_pose[frame, 2])
    support_rotation = wp.quat(
        obstacle_pose[frame, 3], obstacle_pose[frame, 4], obstacle_pose[frame, 5], obstacle_pose[frame, 6]
    )
    return wp.quat_rotate(wp.quat_inverse(support_rotation), point_world - support_origin)


@wp.kernel
def _trajectory_quality_clips(
    quality_by_frame: wp.array2d(dtype=wp.float32),
    body_q: wp.array3d(dtype=wp.float32),
    support_body_indices: wp.array(dtype=wp.int64),
    support_point_body_m: wp.array2d(dtype=wp.float32),
    support_channel_slots: wp.array(dtype=wp.int64),
    contact_body_indices: wp.array(dtype=wp.int64),
    contact_normal_body: wp.array2d(dtype=wp.float32),
    source_channel_confidence: wp.array2d(dtype=wp.float32),
    source_channel_stable: wp.array2d(dtype=wp.uint8),
    source_channel_edge_stable: wp.array2d(dtype=wp.uint8),
    obstacle_pose: wp.array2d(dtype=wp.float32),
    clip_offsets: wp.array(dtype=wp.int32),
    step_seconds: wp.array(dtype=wp.float32),
    clip_count: int,
    channel_count: int,
    point_count: int,
    metric_source_required_position: int,
    metric_source_required_distal_position: int,
    metric_source_required_distal_direction: int,
    metric_source_root_rotation: int,
    metric_source_all_position: int,
    metric_source_all_distal_position: int,
    metric_source_all_landmark_direction: int,
    metric_source_all_distal_direction: int,
    metric_source_nonroot_rotation: int,
    metric_contact_gap: int,
    metric_contact_tilt: int,
    metric_contact_slip_speed: int,
    metric_contact_cumulative_drift: int,
    metric_contact_applicable: int,
    metric_contact_stable_count: int,
    metric_source_contact_confidence: int,
    quality_by_clip: wp.array2d(dtype=wp.float32),
):
    """Aggregate semantic fidelity and stable-contact evidence per complete clip."""
    clip = wp.tid()
    if clip >= clip_count:
        return
    start = clip_offsets[clip]
    stop = clip_offsets[clip + 1]
    source_required_position_error = float(0.0)
    source_required_distal_position_error = float(0.0)
    source_required_distal_direction_error = float(0.0)
    source_root_rotation_error = float(0.0)
    source_all_position_error = float(0.0)
    source_all_distal_position_error = float(0.0)
    source_all_landmark_direction_error = float(0.0)
    source_all_distal_direction_error = float(0.0)
    source_nonroot_rotation_error = quality_by_frame[start, metric_source_nonroot_rotation]
    contact_gap = float(0.0)
    contact_tilt = float(0.0)
    contact_slip_speed = float(0.0)
    contact_cumulative_drift = float(0.0)
    confidence_sum = float(0.0)
    stable_frame_channel_count = int(0)
    dt = step_seconds[clip]

    for frame in range(start, stop):
        source_required_position_error = wp.max(
            source_required_position_error, quality_by_frame[frame, metric_source_required_position]
        )
        source_required_distal_position_error = wp.max(
            source_required_distal_position_error,
            quality_by_frame[frame, metric_source_required_distal_position],
        )
        source_required_distal_direction_error = wp.max(
            source_required_distal_direction_error,
            quality_by_frame[frame, metric_source_required_distal_direction],
        )
        source_root_rotation_error = wp.max(
            source_root_rotation_error, quality_by_frame[frame, metric_source_root_rotation]
        )
        source_all_position_error = wp.max(
            source_all_position_error, quality_by_frame[frame, metric_source_all_position]
        )
        source_all_distal_position_error = wp.max(
            source_all_distal_position_error, quality_by_frame[frame, metric_source_all_distal_position]
        )
        source_all_landmark_direction_error = wp.max(
            source_all_landmark_direction_error,
            quality_by_frame[frame, metric_source_all_landmark_direction],
        )
        source_all_distal_direction_error = wp.max(
            source_all_distal_direction_error, quality_by_frame[frame, metric_source_all_distal_direction]
        )
        source_nonroot_rotation_error = wp.max(
            source_nonroot_rotation_error, quality_by_frame[frame, metric_source_nonroot_rotation]
        )
        support_rotation = wp.quat(
            obstacle_pose[frame, 3], obstacle_pose[frame, 4], obstacle_pose[frame, 5], obstacle_pose[frame, 6]
        )
        support_normal = wp.quat_rotate(support_rotation, wp.vec3(0.0, 0.0, 1.0))
        support_normal_length = wp.length(support_normal)
        for channel in range(channel_count):
            confidence_sum += source_channel_confidence[frame, channel]
            if source_channel_stable[frame, channel] != wp.uint8(0):
                stable_frame_channel_count += 1
                gap = float(0.0)
                patch_point_count = int(0)
                gap_finite = wp.uint8(1)
                for point in range(point_count):
                    if support_channel_slots[point] == channel:
                        coordinate = _trajectory_support_coordinate(
                            body_q, support_body_indices, support_point_body_m, obstacle_pose, point, frame
                        )
                        if not wp.isfinite(coordinate[2]):
                            gap_finite = wp.uint8(0)
                        else:
                            gap += coordinate[2]
                        patch_point_count += 1
                if patch_point_count < 1 or gap_finite == wp.uint8(0):
                    contact_gap = float(wp.inf)
                else:
                    contact_gap = wp.max(contact_gap, wp.max(gap / float(patch_point_count), 0.0))

                body = contact_body_indices[channel]
                body_rotation = wp.quat(
                    body_q[frame, body, 3],
                    body_q[frame, body, 4],
                    body_q[frame, body, 5],
                    body_q[frame, body, 6],
                )
                normal_body = wp.vec3(
                    contact_normal_body[channel, 0],
                    contact_normal_body[channel, 1],
                    contact_normal_body[channel, 2],
                )
                patch_normal = wp.quat_rotate(body_rotation, normal_body)
                patch_normal_length = wp.length(patch_normal)
                if (
                    not wp.isfinite(support_normal_length)
                    or not wp.isfinite(patch_normal_length)
                    or support_normal_length <= 1.0e-8
                    or patch_normal_length <= 1.0e-8
                ):
                    contact_tilt = float(wp.inf)
                else:
                    cosine = wp.dot(support_normal, patch_normal) / (support_normal_length * patch_normal_length)
                    tilt = wp.acos(wp.clamp(cosine, -1.0, 1.0))
                    if not wp.isfinite(tilt):
                        contact_tilt = float(wp.inf)
                    else:
                        contact_tilt = wp.max(contact_tilt, tilt)

    for channel in range(channel_count):
        for point in range(point_count):
            if support_channel_slots[point] == channel:
                interval_origin = wp.vec3(0.0, 0.0, 0.0)
                previous = wp.vec3(0.0, 0.0, 0.0)
                for frame in range(start, stop):
                    stable = source_channel_stable[frame, channel]
                    edge_stable = source_channel_edge_stable[frame, channel]
                    current = _trajectory_support_coordinate(
                        body_q, support_body_indices, support_point_body_m, obstacle_pose, point, frame
                    )
                    current_finite = wp.isfinite(current[0]) and wp.isfinite(current[1]) and wp.isfinite(current[2])
                    if stable != wp.uint8(0):
                        if not current_finite:
                            contact_slip_speed = float(wp.inf)
                            contact_cumulative_drift = float(wp.inf)
                        elif edge_stable != wp.uint8(0):
                            speed = wp.length(current - previous) / dt
                            drift = wp.length(current - interval_origin)
                            if not wp.isfinite(speed) or not wp.isfinite(drift):
                                contact_slip_speed = float(wp.inf)
                                contact_cumulative_drift = float(wp.inf)
                            else:
                                contact_slip_speed = wp.max(contact_slip_speed, speed)
                                contact_cumulative_drift = wp.max(contact_cumulative_drift, drift)
                        else:
                            interval_origin = current
                    previous = current

    contact_applicable = stable_frame_channel_count > 0
    quality_by_clip[clip, metric_source_required_position] = source_required_position_error
    quality_by_clip[clip, metric_source_required_distal_position] = source_required_distal_position_error
    quality_by_clip[clip, metric_source_required_distal_direction] = source_required_distal_direction_error
    quality_by_clip[clip, metric_source_root_rotation] = source_root_rotation_error
    quality_by_clip[clip, metric_source_all_position] = source_all_position_error
    quality_by_clip[clip, metric_source_all_distal_position] = source_all_distal_position_error
    quality_by_clip[clip, metric_source_all_landmark_direction] = source_all_landmark_direction_error
    quality_by_clip[clip, metric_source_all_distal_direction] = source_all_distal_direction_error
    quality_by_clip[clip, metric_source_nonroot_rotation] = source_nonroot_rotation_error
    quality_by_clip[clip, metric_contact_gap] = contact_gap if contact_applicable else wp.nan
    quality_by_clip[clip, metric_contact_tilt] = contact_tilt if contact_applicable else wp.nan
    quality_by_clip[clip, metric_contact_slip_speed] = contact_slip_speed if contact_applicable else wp.nan
    quality_by_clip[clip, metric_contact_cumulative_drift] = contact_cumulative_drift if contact_applicable else wp.nan
    quality_by_clip[clip, metric_contact_applicable] = 1.0 if contact_applicable else 0.0
    quality_by_clip[clip, metric_contact_stable_count] = float(stable_frame_channel_count)
    quality_by_clip[clip, metric_source_contact_confidence] = confidence_sum / float((stop - start) * channel_count)


@wp.kernel
def _trajectory_inspection_contacts(
    body_q: wp.array3d(dtype=wp.float32),
    support_body_indices: wp.array(dtype=wp.int64),
    support_point_body_m: wp.array2d(dtype=wp.float32),
    support_channel_slots: wp.array(dtype=wp.int64),
    source_channel_stable: wp.array2d(dtype=wp.uint8),
    frame_count: int,
    point_count: int,
    points: wp.array3d(dtype=wp.float32),
    valid: wp.array2d(dtype=wp.uint8),
):
    """Expose solved contact-patch points for generic inspection."""
    point, frame = wp.tid()
    if point >= point_count or frame >= frame_count:
        return
    active = source_channel_stable[frame, support_channel_slots[point]] != wp.uint8(0)
    valid[frame, point] = wp.uint8(1) if active else wp.uint8(0)
    if active:
        body = support_body_indices[point]
        rotation = wp.quat(
            body_q[frame, body, 3], body_q[frame, body, 4], body_q[frame, body, 5], body_q[frame, body, 6]
        )
        point_body = wp.vec3(
            support_point_body_m[point, 0], support_point_body_m[point, 1], support_point_body_m[point, 2]
        )
        value = wp.vec3(body_q[frame, body, 0], body_q[frame, body, 1], body_q[frame, body, 2])
        value += wp.quat_rotate(rotation, point_body)
        points[frame, point, 0] = value[0]
        points[frame, point, 1] = value[1]
        points[frame, point, 2] = value[2]


def _trajectory_clip_quality(
    body_q: torch.Tensor,
    rotation_body_indices: torch.Tensor,
    frame_quality: torch.Tensor,
    source_channel_confidence: torch.Tensor,
    source_channel_normal_owned: torch.Tensor,
    source_channel_stable: torch.Tensor,
    source_channel_edge_stable: torch.Tensor,
    obstacle_pose: torch.Tensor,
    targets: MotionTrajectoryTargets,
    clip_offsets: torch.Tensor,
    step_seconds: torch.Tensor,
    quality: torch.Tensor,
    inspection_points: torch.Tensor | None = None,
    inspection_valid: torch.Tensor | None = None,
) -> None:
    """Measure complete-clip source fidelity and physical contact quality."""
    frame_count = body_q.shape[0]
    clip_count = clip_offsets.shape[0] - 1
    device = str(body_q.device)
    wp.launch(
        _trajectory_quality_frames,
        dim=frame_count,
        inputs=[
            wp.from_torch(body_q),
            wp.from_torch(targets.source_landmark_position_m),
            wp.from_torch(targets.position_body_index_tensor),
            wp.from_torch(targets.required_position_row_tensor),
            wp.from_torch(targets.position_normal_channel_slots),
            wp.from_torch(targets.parent_row_tensor),
            wp.from_torch(targets.source_landmark_rotation_xyzw),
            wp.from_torch(rotation_body_indices),
            wp.from_torch(targets.source_direction_point_position_m),
            wp.from_torch(targets.direction_body_index_tensor),
            wp.from_torch(targets.direction_position_row_tensor),
            wp.from_torch(targets.contact_direction_row_tensor),
            wp.from_torch(targets.direction_contact_channel_slots),
            wp.from_torch(targets.required_direction_row_tensor),
            wp.from_torch(source_channel_confidence, dtype=wp.float32),
            wp.from_torch(source_channel_normal_owned, dtype=wp.float32),
            wp.from_torch(targets.direction_point_body_m),
            wp.from_torch(targets.contact_distal_point_body_m),
            frame_count,
            len(targets.position_body_indices),
            len(targets.rotation_body_indices),
            len(targets.direction_body_indices),
            len(targets.required_position_rows),
            len(targets.required_direction_rows),
            _TRAJECTORY_METRIC_COUNT,
            _METRIC_SOURCE_REQUIRED_POSITION,
            _METRIC_SOURCE_REQUIRED_DISTAL_POSITION,
            _METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION,
            _METRIC_SOURCE_ROOT_ROTATION,
            _METRIC_SOURCE_ALL_POSITION,
            _METRIC_SOURCE_ALL_DISTAL_POSITION,
            _METRIC_SOURCE_ALL_LANDMARK_DIRECTION,
            _METRIC_SOURCE_ALL_DISTAL_DIRECTION,
            _METRIC_SOURCE_NONROOT_ROTATION,
        ],
        outputs=[wp.from_torch(frame_quality)],
        device=device,
    )
    wp.launch(
        _trajectory_quality_clips,
        dim=clip_count,
        inputs=[
            wp.from_torch(frame_quality),
            wp.from_torch(body_q),
            wp.from_torch(targets.support_body_indices),
            wp.from_torch(targets.support_point_body_m),
            wp.from_torch(targets.support_channel_slots),
            wp.from_torch(targets.contact_body_indices),
            wp.from_torch(targets.contact_normal_body),
            wp.from_torch(source_channel_confidence),
            wp.from_torch(source_channel_stable),
            wp.from_torch(source_channel_edge_stable),
            wp.from_torch(obstacle_pose),
            wp.from_torch(clip_offsets),
            wp.from_torch(step_seconds),
            clip_count,
            targets.contact_channel_probe_offsets.shape[0] - 1,
            targets.support_body_indices.shape[0],
            _METRIC_SOURCE_REQUIRED_POSITION,
            _METRIC_SOURCE_REQUIRED_DISTAL_POSITION,
            _METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION,
            _METRIC_SOURCE_ROOT_ROTATION,
            _METRIC_SOURCE_ALL_POSITION,
            _METRIC_SOURCE_ALL_DISTAL_POSITION,
            _METRIC_SOURCE_ALL_LANDMARK_DIRECTION,
            _METRIC_SOURCE_ALL_DISTAL_DIRECTION,
            _METRIC_SOURCE_NONROOT_ROTATION,
            _METRIC_CONTACT_GAP,
            _METRIC_CONTACT_TILT,
            _METRIC_CONTACT_SLIP_SPEED,
            _METRIC_CONTACT_CUMULATIVE_DRIFT,
            _METRIC_CONTACT_APPLICABLE,
            _METRIC_CONTACT_STABLE_COUNT,
            _METRIC_SOURCE_CONTACT_CONFIDENCE,
        ],
        outputs=[wp.from_torch(quality)],
        device=device,
    )
    if (inspection_points is None) != (inspection_valid is None):
        raise ValueError("Motion inspection contact points and validity must be requested together.")
    if inspection_points is not None and inspection_valid is not None:
        if inspection_points.shape != (frame_count, targets.support_body_indices.shape[0], 3) or (
            inspection_valid.shape != inspection_points.shape[:2]
        ):
            raise ValueError("Motion inspection contact evidence differs from the active batch shape.")
        inspection_points.zero_()
        inspection_valid.zero_()
        wp.launch(
            _trajectory_inspection_contacts,
            dim=(targets.support_body_indices.shape[0], frame_count),
            inputs=[
                wp.from_torch(body_q),
                wp.from_torch(targets.support_body_indices),
                wp.from_torch(targets.support_point_body_m),
                wp.from_torch(targets.support_channel_slots),
                wp.from_torch(source_channel_stable),
                frame_count,
                targets.support_body_indices.shape[0],
            ],
            outputs=[wp.from_torch(inspection_points), wp.from_torch(inspection_valid, dtype=wp.uint8)],
            device=device,
        )


def _trajectory_recompute_clip_quality(
    target: MotionFrameTarget,
    workspace: _MotionTrajectoryWorkspace,
    targets: MotionTrajectoryTargets,
    joint_q: torch.Tensor,
    clip_offsets: torch.Tensor,
    step_seconds: torch.Tensor,
    quality: torch.Tensor,
    inspection_points: torch.Tensor | None = None,
    inspection_valid: torch.Tensor | None = None,
) -> None:
    """Recompute canonical clip quality and optional inspection evidence from retained coordinates."""
    frame_count = joint_q.shape[0]
    write_velocity_canonical(target, joint_q, clip_offsets, step_seconds, workspace.joint_qd[:frame_count])
    target.kinematics.eval_fk_batched_torch(
        joint_q,
        workspace.joint_qd[:frame_count],
        workspace.body_q[:frame_count],
        workspace.body_qd[:frame_count],
    )
    _trajectory_clip_quality(
        workspace.body_q[:frame_count],
        workspace.rotation_body_indices,
        workspace.frame_quality[:frame_count],
        workspace.source_channel_confidence[:frame_count],
        workspace.source_channel_normal_owned[:frame_count],
        workspace.source_channel_stable[:frame_count],
        workspace.source_channel_edge_stable[:frame_count],
        workspace.obstacle_pose[:frame_count],
        targets,
        clip_offsets,
        step_seconds,
        quality,
        inspection_points,
        inspection_valid,
    )


def _trajectory_max_batch_clips(clip_lengths: tuple[int, ...], frame_capacity: int) -> int:
    """Return the largest greedy whole-clip batch for one frame capacity."""
    if not clip_lengths or any(length < 1 for length in clip_lengths) or frame_capacity < 1:
        raise ValueError("Trajectory batch sizing requires positive clip lengths and frame capacity.")
    maximum = 1
    frames = 0
    clips = 0
    for length in clip_lengths:
        if length > frame_capacity:
            frames = 0
            clips = 0
            continue
        if frames and frames + length > frame_capacity:
            maximum = max(maximum, clips)
            frames = 0
            clips = 0
        frames += length
        clips += 1
    return max(maximum, clips)


def _trajectory_coexisting_peak_bytes(persistent_bytes: int, transient_stage_bytes: tuple[int, ...]) -> int:
    """Return persistent storage plus the largest mutually exclusive stage [byte]."""
    if persistent_bytes < 0 or not transient_stage_bytes or any(value < 0 for value in transient_stage_bytes):
        raise ValueError("Trajectory memory components must be nonnegative and contain at least one transient stage.")
    return persistent_bytes + max(transient_stage_bytes)


def _trajectory_tensor_storage_bytes(
    *,
    frame_capacity: int,
    batch_clip_count: int,
    frame_seed_batch_count: int,
    coordinate_count: int,
    dof_count: int,
    body_count: int,
    residual_count: int,
    position_count: int,
    rotation_count: int,
    direction_count: int,
    source_probe_count: int,
    contact_channel_count: int,
    target_support_count: int,
    physical_inequality_count: int,
    joint_reference_count: int,
) -> dict[str, int]:
    """Return named target, workspace, and batch-tensor allocations [byte]."""
    if frame_seed_batch_count not in (0, batch_clip_count):
        raise ValueError("Trajectory frame-seed storage is inconsistent.")
    f32 = torch.float32.itemsize
    u8 = torch.uint8.itemsize
    i32 = torch.int32.itemsize
    i64 = torch.int64.itemsize
    return {
        "targets.source_landmark_position_m": f32 * frame_capacity * position_count * 3,
        "targets.source_landmark_rotation_xyzw": f32 * frame_capacity * rotation_count * 4,
        "targets.source_direction_point_position_m": f32 * frame_capacity * direction_count * 3,
        "targets.initial_joint_q": f32 * frame_capacity * coordinate_count,
        "targets.source_contact_probe_position_m": f32 * frame_capacity * source_probe_count * 3,
        "workspace.joint_q": f32 * frame_capacity * coordinate_count,
        "workspace.certified_joint_q": f32 * frame_capacity * coordinate_count,
        "workspace.segment_iteration_attempted": i32 * batch_clip_count,
        "workspace.segment_damping": f32 * batch_clip_count,
        "workspace.segment_recovery_count": i32 * batch_clip_count,
        "workspace.joint_qd": f32 * frame_capacity * dof_count,
        "targets.target_support_position_m": f32 * frame_capacity * target_support_count * 3,
        "workspace.achieved_direction_position_m": f32 * frame_capacity * direction_count * 3,
        "workspace.joint_reference": f32 * frame_capacity * joint_reference_count,
        "workspace.body_q": f32 * frame_capacity * body_count * 7,
        "workspace.body_qd": f32 * frame_capacity * body_count * 6,
        "workspace.frame_quality": f32 * frame_capacity * _TRAJECTORY_METRIC_COUNT,
        "workspace.velocity_reachable_lower": f32 * frame_capacity * joint_reference_count,
        "workspace.velocity_reachable_upper": f32 * frame_capacity * joint_reference_count,
        "workspace.source_plane_height_m": f32 * batch_clip_count,
        "workspace.source_probe_active": u8 * frame_capacity * source_probe_count,
        "workspace.source_probe_stable": u8 * frame_capacity * source_probe_count,
        "workspace.source_channel_normal_owned": f32 * frame_capacity * contact_channel_count,
        "workspace.source_channel_clearance_lift_m": f32 * frame_capacity * contact_channel_count,
        "workspace.source_channel_confidence": f32 * frame_capacity * contact_channel_count,
        "workspace.source_channel_activity": f32 * frame_capacity * contact_channel_count * 2,
        "workspace.source_channel_stable": u8 * frame_capacity * contact_channel_count,
        "workspace.source_channel_edge_stable": u8 * frame_capacity * contact_channel_count,
        "targets.contact_distal_point_body_m": f32 * contact_channel_count * 3,
        "root.frozen_dof_indices": i32 * 6,
        "workspace.obstacle_pose": f32 * frame_capacity * 7,
        "workspace.segment_phase_attempted": u8 * batch_clip_count,
        "workspace.rotation_body_indices": i64 * rotation_count,
        "workspace.segment_active": i32 * batch_clip_count,
        "workspace.segment_iteration_geometry_feasible": u8 * batch_clip_count,
        "workspace.segment_iteration_inner_converged": u8 * batch_clip_count,
        "workspace.segment_iteration_globalization_succeeded": u8 * batch_clip_count,
        "workspace.segment_iteration_residual_constraints_satisfied": u8 * batch_clip_count,
        "workspace.segment_phase_globalization_succeeded": u8 * batch_clip_count,
        "workspace.segment_phase_converged": u8 * batch_clip_count,
        "workspace.segment_contact_refinement_required": u8 * batch_clip_count,
        "layout.activity_group_by_residual": i32 * residual_count,
        "layout.first_difference_group_by_residual": i32 * residual_count,
        "physical.inequality_indices": i32 * physical_inequality_count,
        "physical.inequality_upper": f32 * physical_inequality_count,
        "workspace.base_weights": f32 * residual_count,
        "workspace.temporal_weights": f32 * residual_count * 3,
        "workspace.velocity_lower": f32 * dof_count,
        "workspace.velocity_upper": f32 * dof_count,
        "workspace.source_velocity_lower": f32 * dof_count,
        "workspace.source_velocity_upper": f32 * dof_count,
        "frame_seed.global.source_landmark_position_m": f32 * frame_seed_batch_count * position_count * 3,
        "frame_seed.global.source_landmark_rotation_xyzw": f32 * frame_seed_batch_count * rotation_count * 4,
        "frame_seed.global.source_direction_point_position_m": f32 * frame_seed_batch_count * direction_count * 3,
        "frame_seed.global.joint_q": f32 * frame_seed_batch_count * coordinate_count,
        "frame_seed.local.source_landmark_position_m": f32 * frame_seed_batch_count * position_count * 3,
        "frame_seed.local.source_landmark_rotation_xyzw": f32 * frame_seed_batch_count * rotation_count * 4,
        "frame_seed.local.source_direction_point_position_m": f32 * frame_seed_batch_count * direction_count * 3,
        "frame_seed.local.joint_q": f32 * frame_seed_batch_count * coordinate_count,
        "frame_seed.local.candidate_joint_q": (
            f32 * _FRAME_SEED_LOCAL_CANDIDATES * frame_seed_batch_count * coordinate_count
        ),
        "frame_seed.local.problem_indices": i32 * _FRAME_SEED_LOCAL_CANDIDATES * frame_seed_batch_count,
        "frame_seed.frame_indices": i32 * frame_seed_batch_count,
        "frame_seed.frame_active": i32 * frame_seed_batch_count,
        "batch.clip_offsets": i32 * (batch_clip_count + 1),
        "batch.step_seconds": f32 * batch_clip_count,
    }


def _trajectory_projection_peak_bytes(
    candidate: _builder._MotionTrajectoryTargetCandidate,
    targets: MotionTrajectoryTargets,
    reference,
) -> int:
    """Bound the largest lazy source-decode and morphology-projection stage [byte]."""
    landmark_count = len(targets.position_body_indices)
    direction_count = len(targets.direction_body_indices)
    target_support_count = targets.target_support_position_m.shape[0]
    source_probe_count = targets.source_contact_probe_position_m.shape[0]
    coordinate_count = reference.model.joint_coord_count
    dof_count = reference.model.joint_dof_count
    target_body_count = reference.model.body_count
    peaks = []
    for clip, source_body_count in zip(candidate.clip_index.clips, candidate.source_body_counts, strict=True):
        # Sum all lazy decode, source-FK, calibration, seed-fit, support, and
        # terminal-point tensors even when Python releases some before later stages.
        source_decode_values = 3 + 4 * source_body_count + dof_count
        source_fk_values = 7 * source_body_count
        rotation_calibration_values = 16 + 16 * (landmark_count + 1)
        position_calibration_values = 6 * landmark_count
        seed_fit_values = coordinate_count + 7 * targets.coordinate_indices.numel() + 13 * target_body_count
        support_values = 3 * source_probe_count + 10 * target_support_count + 1
        direction_values = 10 * direction_count
        values_per_frame = (
            source_decode_values
            + source_fk_values
            + rotation_calibration_values
            + position_calibration_values
            + seed_fit_values
            + support_values
            + direction_values
        )
        peaks.append(clip.frame_count * values_per_frame * 4)
    return max(peaks)


def _trajectory_coordinate_projection_peak_bytes(frame_capacity: int, coordinate_count: int, dof_count: int) -> int:
    """Bound target-specific solved-coordinate conversion temporaries [byte]."""
    return frame_capacity * (4 * coordinate_count + 4 * dof_count + 32)


def motion_solve_trajectory(  # noqa: C901
    cfg: MotionTrajectorySolveCfg, candidate: _builder._MotionTrajectoryTargetCandidate
) -> _builder._MotionTrajectorySolvedCandidate:
    """Solve streamed source semantics as memory-bounded complete trajectories."""
    from .motion_task_table_builder import _certify_target_coordinates, _MotionTrajectorySolvedCandidate

    try:
        first_targets = next(candidate.pending)
    except StopIteration as error:
        raise ValueError("Motion trajectory solve requires at least one source clip.") from error
    prototype = _motion_workspace_targets(first_targets, 1)
    stream = _MotionSourceEvidenceStream(candidate.pending, candidate.clip_index, prototype, first_targets)
    stream._validate(first_targets)
    if prototype.source_root_policy not in ("fixed", "optimized"):
        raise ValueError("Motion source-root policy must be 'fixed' or 'optimized'.")
    source_root_fixed = prototype.source_root_policy == "fixed"
    if prototype.initializer_policy not in ("direct", "batched_frame_ik"):
        raise ValueError("Motion initializer policy must be 'direct' or 'batched_frame_ik'.")
    use_frame_ik = prototype.initializer_policy == "batched_frame_ik"

    reference = candidate.target.kinematics
    model = reference.model
    tree = KinematicTree.from_newton(reference)
    coordinate_q_indices = np.asarray(tree.coordinate_q_indices, dtype=np.int32)
    coordinate_qd_indices = np.asarray(tree.coordinate_qd_indices, dtype=np.int32)
    coordinate_count = model.joint_coord_count
    dof_count = model.joint_dof_count
    body_count = model.body_count
    rotation_count = len(prototype.rotation_body_indices)
    clip_count = len(candidate.clip_index.clips)
    source_probe_count = prototype.source_contact_probe_position_m.shape[0]
    contact_channel_count = prototype.contact_channel_probe_offsets.shape[0] - 1
    target_point_count = prototype.target_support_position_m.shape[0]
    if (
        prototype.contact_body_indices.shape != (contact_channel_count,)
        or prototype.contact_forward_body.shape != (contact_channel_count, 3)
        or prototype.contact_distal_point_body_m.shape != (contact_channel_count, 3)
        or prototype.contact_direction_row_tensor.shape != (contact_channel_count,)
        or prototype.direction_contact_channel_slots.shape != (len(prototype.direction_body_indices),)
        or prototype.required_direction_row_tensor.shape != (len(prototype.required_direction_rows),)
        or prototype.direction_position_row_tensor.shape != (len(prototype.direction_body_indices),)
        or len(prototype.support_patch_offsets) != contact_channel_count + 1
        or prototype.support_patch_offsets[0] != 0
        or prototype.support_patch_offsets[-1] != target_point_count
    ):
        raise ValueError("Motion target contact poses must match the canonical source channels.")
    if (
        rotation_count < 1
        or len(prototype.rotation_weights) != rotation_count
        or prototype.rotation_body_indices[0] != prototype.root_body_index
    ):
        raise ValueError("Motion rotation roles require root row zero and one weight per body.")
    view_landmark_count = len(prototype.position_body_indices) + len(prototype.direction_body_indices)
    if (
        coordinate_count != dof_count + 1
        or source_probe_count < 1
        or contact_channel_count < 1
        or target_point_count < 1
    ):
        raise ValueError("Motion trajectory solve requires one free root and nonempty contact probes and patches.")
    if tuple(int(value) for value in reference.topology.joint_q_start[:2]) != (0, 7) or tuple(
        int(value) for value in reference.topology.joint_qd_start[:2]
    ) != (0, 6):
        raise ValueError("Motion contact refinement requires the canonical leading free-root coordinates and DOFs.")
    if tuple(int(index) for index in prototype.coordinate_indices.cpu().tolist()) != tree.coordinate_q_indices:
        raise ValueError("Motion target coordinates differ from the canonical target-tree q-to-qd mapping.")
    output_coordinates = candidate.target.allocate_coordinates(
        candidate.clip_index.total_frames, device=candidate.device
    )
    trajectory_quality = torch.empty(
        (clip_count, _TRAJECTORY_METRIC_COUNT), dtype=torch.float32, device=candidate.device
    )
    constraint_geometry_feasible = torch.empty(clip_count, dtype=torch.bool, device=candidate.device)
    inner_solve_converged = torch.empty(clip_count, dtype=torch.bool, device=candidate.device)
    nonlinear_refinement_required = torch.empty(clip_count, dtype=torch.bool, device=candidate.device)
    nonlinear_phases_converged = torch.empty(clip_count, dtype=torch.bool, device=candidate.device)
    contact_evidence = (
        _MotionContactEvidence(
            source_stable=torch.empty(
                (candidate.clip_index.total_frames, contact_channel_count), dtype=torch.bool, device=candidate.device
            ),
            support_body_indices=tuple(int(index) for index in prototype.support_body_indices.detach().cpu().tolist()),
            support_point_body_m=prototype.support_point_body_m,
            support_channel_slots=prototype.support_channel_slots,
            policy=cfg.dynamics,
        )
        if candidate.inspection
        else None
    )
    view_evidence = None
    if candidate.inspection:
        view_evidence = _MotionTrajectoryViewEvidence(
            target_landmarks=torch.empty(
                (candidate.clip_index.total_frames, view_landmark_count, 3),
                dtype=torch.float32,
                device=candidate.device,
            ),
            solved_robot_landmarks=torch.empty(
                (candidate.clip_index.total_frames, view_landmark_count, 3),
                dtype=torch.float32,
                device=candidate.device,
            ),
            target_support=torch.empty(
                (candidate.clip_index.total_frames, target_point_count, 3),
                dtype=torch.float32,
                device=candidate.device,
            ),
            contact_points=torch.empty(
                (candidate.clip_index.total_frames, target_point_count, 3),
                dtype=torch.float32,
                device=candidate.device,
            ),
            contact_valid=torch.empty(
                (candidate.clip_index.total_frames, target_point_count), dtype=torch.bool, device=candidate.device
            ),
            stage_quality=torch.full(
                (clip_count, len(_TRAJECTORY_INSPECTION_CAPTURE_STAGE_NAMES), _TRAJECTORY_METRIC_COUNT),
                float("nan"),
                dtype=torch.float32,
                device=candidate.device,
            ),
        )

    if any(
        not isinstance(term, (StateCommandCfg.TaskTableCfg.ObjectiveCfg, IKObjectiveBaseCfg)) for term in cfg.objectives
    ):
        raise TypeError("Motion trajectory solve terms must be numerical objective configurations.")
    collision_cfgs = tuple(term for term in cfg.objectives if isinstance(term, IKObjectiveMeshCollisionCfg))
    nonpenetration_cfgs = tuple(term for term in cfg.objectives if isinstance(term, IKObjectiveMeshNonpenetrationCfg))
    if len(collision_cfgs) != 1 or len(nonpenetration_cfgs) != 1:
        raise ValueError("Motion trajectory solve requires one soft collision and one nonpenetration objective.")
    collision_cfg = collision_cfgs[0]
    nonpenetration_cfg = nonpenetration_cfgs[0]
    if (
        collision_cfg.n_samples != _MOTION_COLLISION_PROBES_PER_BODY
        or nonpenetration_cfg.n_samples != _MOTION_COLLISION_PROBES_PER_BODY
    ):
        raise ValueError(
            f"Collision terms must declare the target-owned {_MOTION_COLLISION_PROBES_PER_BODY}-probe geometry."
        )
    probe_bodies, probe_offsets, probe_contact_slots, probe_normal_slots = validate_collision_probe_geometry(
        candidate.target,
        device=candidate.device,
        contact_channel_count=contact_channel_count,
    )
    probe_count = probe_bodies.shape[0]
    probe_bodies_np = probe_bodies.detach().cpu().numpy()
    probe_offsets_np = probe_offsets.detach().cpu().numpy()
    probe_contact_slots_np = probe_contact_slots.detach().cpu().numpy()
    collision_mesh = _trajectory_ground_mesh(reference.device)

    def build_system(
        batch_size: int,
        targets: MotionTrajectoryTargets,
        obstacle_pose,
        contact_normal_owned: torch.Tensor,
        joint_reference: torch.Tensor,
        contact_confidence: torch.Tensor,
    ):
        collision_context = IKObjectiveMeshCollisionBuildContext(
            kinematics=reference,
            collision_mesh=collision_mesh,
            asset_name="robot",
            batch_size=batch_size,
            obstacle_pose=obstacle_pose,
            probe_offsets=probe_offsets_np,
            probe_bodies=probe_bodies_np,
            probe_contact_slots=probe_contact_slots_np,
            contact_confidence=contact_confidence,
        )
        joint_pin_context = IKJointPinObjectiveBuildContext(
            kinematics=reference,
            asset_name="robot",
            batch_size=batch_size,
            coordinate_indices=coordinate_q_indices,
            dof_indices=coordinate_qd_indices,
            targets=joint_reference,
        )
        base_context = IKObjectiveBuildContext(kinematics=reference, asset_name="robot", batch_size=batch_size)
        features = []
        for term_cfg in cfg.objectives:
            factory = term_cfg.class_type
            if not callable(factory):
                factory = string_to_callable(str(factory))
            if isinstance(term_cfg, IKObjectiveMeshCollisionCfg | IKObjectiveMeshNonpenetrationCfg):
                build = factory(term_cfg, collision_context)
            elif isinstance(term_cfg, IKObjectiveJointPinCfg):
                build = factory(term_cfg, joint_pin_context)
            elif isinstance(term_cfg, IKObjectiveJointDefaultCfg):
                build = factory(term_cfg, base_context)
            elif isinstance(term_cfg, MotionSourceGlobalPositionObjectiveCfg | MotionSourceDirectionPointObjectiveCfg):
                features.extend(factory(term_cfg, targets, contact_normal_owned, contact_confidence))
                continue
            elif isinstance(term_cfg, MotionContactObjectiveCfg):
                features.extend(
                    factory(term_cfg, targets, reference, obstacle_pose, cfg.contact, cfg.acceptance.contact)
                )
                continue
            else:
                features.extend(factory(term_cfg, targets))
                continue
            if not isinstance(build, IKObjectiveBuild):
                raise TypeError(f"IK objective builder returned {type(build).__name__}, expected IKObjectiveBuild.")
            features.extend(build.objectives)
        features.append(
            _IKObjectiveSourceFidelityGuard(
                targets,
                contact_normal_owned,
                contact_confidence,
                reference.topology.body_dof_ancestry,
            )
        )
        return features

    residual_layout = _motion_trajectory_residual_layout(prototype, probe_count, cfg.objectives)
    representative_pose = torch.zeros((1, 7), dtype=torch.float32, device=candidate.device)
    representative_pose[:, 6] = 1.0
    representative_joint_reference = prototype.initial_joint_q[:1].index_select(1, prototype.coordinate_indices)
    representative_contact_confidence = torch.zeros(
        (1, contact_channel_count), dtype=torch.float32, device=candidate.device
    )
    representative_contact_normal_owned = torch.zeros_like(representative_contact_confidence)

    representative_features = build_system(
        1,
        prototype,
        representative_pose,
        representative_contact_normal_owned,
        representative_joint_reference,
        representative_contact_confidence,
    )
    representative_frame_seed_objectives = _motion_frame_seed_objectives(cfg, prototype) if use_frame_ik else ()
    jacobian_mode = (
        ik.IKJacobianType.MIXED
        if any(not objective.supports_analytic() for objective in representative_features)
        else ik.IKJacobianType.ANALYTIC
    )
    residual_count = sum(objective.residual_dim() for objective in representative_features)
    if residual_count != residual_layout.residual_count:
        raise RuntimeError("Motion residual rows differ from the explicit residual layout.")
    clip_lengths = tuple(clip.frame_count for clip in candidate.clip_index.clips)
    minimum_frames = max(2, candidate.target.materialization_minimum_frames)
    if minimum_frames < 2 or any(length < minimum_frames for length in clip_lengths):
        raise ValueError(
            f"Motion trajectory clips require at least {minimum_frames} frames for target derivative laws."
        )
    projection_peak_bytes = _trajectory_projection_peak_bytes(candidate, prototype, reference)

    def estimate_memory(frame_capacity: int) -> int:
        batch_clip_count = _trajectory_max_batch_clips(clip_lengths, frame_capacity)
        trajectory_optimizer_bytes = ik.IKSolver.estimate_memory(
            model,
            frame_capacity,
            representative_features,
            optimizer=ik.IKOptimizer.LM,
            jacobian_mode=jacobian_mode,
            sampler=ik.IKSampler.NONE,
            n_seeds=1,
        ).total_bytes
        if use_frame_ik:
            frame_seed_global_solver_bytes = ik.IKSolver.estimate_memory(
                model,
                batch_clip_count,
                representative_frame_seed_objectives,
                optimizer=ik.IKOptimizer.LM,
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
                sampler=ik.IKSampler.GAUSS,
                n_seeds=_FRAME_SEED_GLOBAL_SEEDS,
            ).total_bytes
            frame_seed_local_estimate = ik.IKSolver.estimate_memory(
                model,
                batch_clip_count,
                representative_frame_seed_objectives,
                optimizer=ik.IKOptimizer.LM,
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
                sampler=ik.IKSampler.GAUSS,
                n_seeds=_FRAME_SEED_LOCAL_CANDIDATES,
            )
            frame_seed_local_solver_bytes = (
                frame_seed_local_estimate.optimizer_bytes + frame_seed_local_estimate.objective_bytes
            )
        else:
            frame_seed_global_solver_bytes = 0
            frame_seed_local_solver_bytes = 0
        trajectory_bytes = IKTrajectorySolver.estimate_workspace_bytes(
            frame_capacity,
            coordinate_count,
            dof_count,
            residual_count,
            batch_clip_count,
            max_equality_residuals_per_frame=0,
        )
        tensor_bytes = _trajectory_tensor_storage_bytes(
            frame_capacity=frame_capacity,
            batch_clip_count=batch_clip_count,
            frame_seed_batch_count=batch_clip_count if use_frame_ik else 0,
            coordinate_count=coordinate_count,
            dof_count=dof_count,
            body_count=body_count,
            residual_count=residual_count,
            position_count=len(prototype.position_body_indices),
            direction_count=len(prototype.direction_body_indices),
            source_probe_count=source_probe_count,
            rotation_count=len(prototype.rotation_body_indices),
            contact_channel_count=contact_channel_count,
            target_support_count=target_point_count,
            physical_inequality_count=0,
            joint_reference_count=len(coordinate_q_indices),
        )
        batch_offset_transient_bytes = torch.int64.itemsize * (batch_clip_count + 1)
        persistent_bytes = (
            trajectory_optimizer_bytes
            + frame_seed_global_solver_bytes
            + frame_seed_local_solver_bytes
            + trajectory_bytes
            + sum(tensor_bytes.values())
            + batch_offset_transient_bytes
        )
        return _trajectory_coexisting_peak_bytes(
            persistent_bytes,
            (
                projection_peak_bytes,
                _trajectory_coordinate_projection_peak_bytes(frame_capacity, coordinate_count, dof_count),
            ),
        )

    memory_plan = plan_trajectory_memory(clip_lengths, candidate.device, estimate_memory)
    capacity = memory_plan.workspace_frame_capacity
    max_batch_clips = _trajectory_max_batch_clips(clip_lengths, capacity)
    planned_max_batch_clips = max(
        stop - start
        for start, stop in zip(
            memory_plan.batch_segment_offsets[:-1], memory_plan.batch_segment_offsets[1:], strict=True
        )
    )
    if max_batch_clips != planned_max_batch_clips:
        raise RuntimeError("Trajectory workspace and memory plan disagree on maximum batch clip count.")
    targets = _motion_workspace_targets(prototype, capacity)
    obstacle_pose = torch.zeros((capacity, 7), dtype=torch.float32, device=candidate.device)
    obstacle_pose[:, 6] = 1.0
    joint_reference = torch.empty((capacity, len(coordinate_q_indices)), dtype=torch.float32, device=candidate.device)
    source_channel_confidence = torch.empty(
        (capacity, contact_channel_count), dtype=torch.float32, device=candidate.device
    )
    source_channel_normal_owned = torch.empty_like(source_channel_confidence)
    source_channel_clearance_lift_m = torch.zeros_like(source_channel_confidence)
    source_channel_activity = torch.empty(
        (capacity, 2 * contact_channel_count), dtype=torch.float32, device=candidate.device
    )
    source_channel_stable = torch.empty((capacity, contact_channel_count), dtype=torch.uint8, device=candidate.device)
    if use_frame_ik:
        frame_seed_global_targets = _motion_frame_seed_targets(prototype, max_batch_clips)
        frame_seed_local_targets = _motion_frame_seed_targets(prototype, max_batch_clips)
        frame_seed_local_targets.source_landmark_position_m.zero_()
        frame_seed_local_targets.source_landmark_rotation_xyzw.zero_()
        frame_seed_local_targets.source_landmark_rotation_xyzw[..., 3].fill_(1.0)
        frame_seed_local_targets.source_direction_point_position_m.zero_()
        frame_seed_local_targets.initial_joint_q.zero_()
        frame_seed_local_targets.initial_joint_q[:, 6].fill_(1.0)
        frame_seed_local_candidate_joint_q = torch.empty(
            (_FRAME_SEED_LOCAL_CANDIDATES * max_batch_clips, coordinate_count),
            dtype=torch.float32,
            device=candidate.device,
        )
        frame_seed_local_problem_indices = torch.arange(
            max_batch_clips, dtype=torch.int32, device=candidate.device
        ).repeat_interleave(_FRAME_SEED_LOCAL_CANDIDATES)
        frame_seed_global_solver = reference.create_ik_solver(
            _motion_frame_seed_objectives(cfg, frame_seed_global_targets),
            max_batch_clips,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            sampler=ik.IKSampler.GAUSS,
            n_seeds=_FRAME_SEED_GLOBAL_SEEDS,
            noise_std=_FRAME_SEED_GLOBAL_NOISE_STD,
            rng_seed=_FRAME_SEED_GLOBAL_RNG_SEED,
        )
        frame_seed_local_optimizer = ik.IKOptimizerLM(
            model,
            _FRAME_SEED_LOCAL_CANDIDATES * max_batch_clips,
            _motion_frame_seed_objectives(cfg, frame_seed_local_targets),
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            problem_idx=wp.from_torch(frame_seed_local_problem_indices),
        )
    features = build_system(
        capacity,
        targets,
        obstacle_pose,
        source_channel_normal_owned,
        joint_reference,
        source_channel_confidence,
    )
    optimizer = ik.IKOptimizerLM(model, capacity, features, jacobian_mode=jacobian_mode)
    solver = IKTrajectorySolver(
        optimizer,
        max_segments=max_batch_clips,
        max_equality_residuals_per_frame=0,
        damping=cfg.damping,
        krylov_max_iterations=cfg.krylov_max_iterations,
        krylov_relative_tolerance=cfg.krylov_relative_tolerance,
        kkt_relative_tolerance=cfg.kkt_relative_tolerance,
        line_search_steps=tuple(0.5**index for index in range(11)),
    )
    base_weights = torch.zeros(residual_count, dtype=torch.float32, device=candidate.device)
    temporal_weights = torch.zeros((3, residual_count), dtype=torch.float32, device=candidate.device)
    solver_coordinate_bounds = IKTrajectorySolver.CoordinateBounds(
        coordinate_indices=torch.as_tensor(coordinate_q_indices, dtype=torch.int32, device=candidate.device),
        dof_indices=torch.as_tensor(coordinate_qd_indices, dtype=torch.int32, device=candidate.device),
        lower=targets.coordinate_lower_limits_rad,
        upper=targets.coordinate_upper_limits_rad,
    )
    root_dof_indices = torch.arange(6, dtype=torch.int32, device=candidate.device)
    velocity_lower = torch.tensor(reference.topology.joint_velocity_lower, dtype=torch.float32, device=candidate.device)
    velocity_upper = torch.tensor(reference.topology.joint_velocity_upper, dtype=torch.float32, device=candidate.device)
    source_velocity_lower = torch.full_like(velocity_lower, -torch.inf)
    source_velocity_upper = torch.full_like(velocity_upper, torch.inf)
    workspace = _MotionTrajectoryWorkspace(
        joint_q=torch.empty((capacity, coordinate_count), dtype=torch.float32, device=candidate.device),
        certified_joint_q=torch.empty((capacity, coordinate_count), dtype=torch.float32, device=candidate.device),
        segment_iteration_attempted=torch.empty(max_batch_clips, dtype=torch.int32, device=candidate.device),
        segment_damping=torch.empty(max_batch_clips, dtype=torch.float32, device=candidate.device),
        segment_recovery_count=torch.empty(max_batch_clips, dtype=torch.int32, device=candidate.device),
        joint_qd=torch.empty((capacity, dof_count), dtype=torch.float32, device=candidate.device),
        achieved_direction_position_m=torch.empty(
            (len(prototype.direction_body_indices), capacity, 3), dtype=torch.float32, device=candidate.device
        ),
        body_q=torch.empty((capacity, body_count, 7), dtype=torch.float32, device=candidate.device),
        joint_reference=joint_reference,
        body_qd=torch.empty((capacity, body_count, 6), dtype=torch.float32, device=candidate.device),
        segment_phase_attempted=torch.empty(max_batch_clips, dtype=torch.bool, device=candidate.device),
        frame_quality=torch.empty((capacity, _TRAJECTORY_METRIC_COUNT), dtype=torch.float32, device=candidate.device),
        velocity_reachable_lower=torch.empty(
            (capacity, len(coordinate_q_indices)), dtype=torch.float32, device=candidate.device
        ),
        velocity_reachable_upper=torch.empty(
            (capacity, len(coordinate_q_indices)), dtype=torch.float32, device=candidate.device
        ),
        segment_active=torch.empty(max_batch_clips, dtype=torch.int32, device=candidate.device),
        source_plane_height_m=torch.empty(max_batch_clips, dtype=torch.float32, device=candidate.device),
        segment_iteration_geometry_feasible=torch.empty(max_batch_clips, dtype=torch.bool, device=candidate.device),
        segment_iteration_inner_converged=torch.empty(max_batch_clips, dtype=torch.bool, device=candidate.device),
        segment_iteration_globalization_succeeded=torch.empty(
            max_batch_clips, dtype=torch.bool, device=candidate.device
        ),
        segment_iteration_residual_constraints_satisfied=torch.empty(
            max_batch_clips, dtype=torch.bool, device=candidate.device
        ),
        segment_phase_globalization_succeeded=torch.empty(max_batch_clips, dtype=torch.bool, device=candidate.device),
        segment_phase_converged=torch.empty(max_batch_clips, dtype=torch.bool, device=candidate.device),
        segment_contact_refinement_required=torch.empty(max_batch_clips, dtype=torch.bool, device=candidate.device),
        source_probe_active=torch.empty((capacity, source_probe_count), dtype=torch.uint8, device=candidate.device),
        source_channel_normal_owned=source_channel_normal_owned,
        source_channel_clearance_lift_m=source_channel_clearance_lift_m,
        source_probe_stable=torch.empty((capacity, source_probe_count), dtype=torch.uint8, device=candidate.device),
        source_channel_confidence=source_channel_confidence,
        source_channel_activity=source_channel_activity,
        source_channel_stable=source_channel_stable,
        source_channel_edge_stable=torch.empty(
            (capacity, contact_channel_count), dtype=torch.uint8, device=candidate.device
        ),
        obstacle_pose=obstacle_pose,
        rotation_body_indices=torch.tensor(prototype.rotation_body_indices, dtype=torch.int64, device=candidate.device),
        residual_layout=residual_layout,
        base_weights=base_weights,
        temporal_weights=temporal_weights,
        velocity_lower=velocity_lower,
        velocity_upper=velocity_upper,
        source_velocity_lower=source_velocity_lower,
        source_velocity_upper=source_velocity_upper,
    )
    initializer_joint_q_wp = wp.from_torch(workspace.joint_q)
    initializer_baseline_wp = wp.from_torch(targets.initial_joint_q)
    initializer_coordinate_indices_wp = wp.from_torch(targets.coordinate_indices)
    initializer_coordinate_lower_wp = wp.from_torch(targets.coordinate_lower_limits_rad)
    initializer_coordinate_upper_wp = wp.from_torch(targets.coordinate_upper_limits_rad)
    if use_frame_ik:
        frame_seed_frame_indices = torch.empty(max_batch_clips, dtype=torch.int32, device=candidate.device)
        frame_seed_frame_active = torch.empty(max_batch_clips, dtype=torch.int32, device=candidate.device)
        frame_seed_frame_indices_wp = wp.from_torch(frame_seed_frame_indices)
        frame_seed_frame_active_wp = wp.from_torch(frame_seed_frame_active)
        frame_seed_global_joint_q_wp = wp.from_torch(frame_seed_global_targets.initial_joint_q)
        frame_seed_local_joint_q_wp = wp.from_torch(frame_seed_local_targets.initial_joint_q)
        frame_seed_local_candidate_joint_q_wp = wp.from_torch(frame_seed_local_candidate_joint_q)
        frame_seed_local_cost_wp = frame_seed_local_optimizer.costs
        frame_seed_source_landmark_position_wp = wp.from_torch(targets.source_landmark_position_m)
        frame_seed_source_landmark_rotation_wp = wp.from_torch(targets.source_landmark_rotation_xyzw)
        frame_seed_source_direction_position_wp = wp.from_torch(targets.source_direction_point_position_m)
        frame_seed_global_landmark_position_wp = wp.from_torch(frame_seed_global_targets.source_landmark_position_m)
        frame_seed_global_landmark_rotation_wp = wp.from_torch(frame_seed_global_targets.source_landmark_rotation_xyzw)
        frame_seed_global_direction_position_wp = wp.from_torch(
            frame_seed_global_targets.source_direction_point_position_m
        )
        frame_seed_local_landmark_position_wp = wp.from_torch(frame_seed_local_targets.source_landmark_position_m)
        frame_seed_local_landmark_rotation_wp = wp.from_torch(frame_seed_local_targets.source_landmark_rotation_xyzw)
        frame_seed_local_direction_position_wp = wp.from_torch(
            frame_seed_local_targets.source_direction_point_position_m
        )

    def update_phase_acceptance(
        quality: torch.Tensor,
        attempted: torch.Tensor,
        constraint_geometry_feasible: torch.Tensor,
        residual_constraints_satisfied: torch.Tensor,
        accepted: torch.Tensor,
        active: torch.Tensor,
        *,
        acceptance: Literal["constraints", "source", "source_contact"],
    ) -> None:
        """Freeze the first phase iterate satisfying the requested acceptance scope."""
        if acceptance == "constraints":
            acceptance_mode = _TERMINAL_ACCEPT_CONSTRAINTS
        elif acceptance == "source":
            acceptance_mode = _TERMINAL_ACCEPT_SOURCE
        elif acceptance == "source_contact":
            acceptance_mode = _TERMINAL_ACCEPT_SOURCE_CONTACT
        else:
            raise ValueError(f"Unsupported terminal acceptance scope: {acceptance}.")
        wp.launch(
            _motion_terminal_acceptance_update,
            dim=active.shape[0],
            inputs=[
                wp.from_torch(quality),
                wp.from_torch(attempted),
                wp.from_torch(constraint_geometry_feasible),
                wp.from_torch(residual_constraints_satisfied),
                wp.uint8(acceptance_mode),
                active.shape[0],
                _METRIC_SOURCE_REQUIRED_POSITION,
                _METRIC_SOURCE_REQUIRED_DISTAL_POSITION,
                _METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION,
                _METRIC_SOURCE_ROOT_ROTATION,
                _METRIC_CONTACT_GAP,
                _METRIC_CONTACT_TILT,
                _METRIC_CONTACT_SLIP_SPEED,
                _METRIC_CONTACT_CUMULATIVE_DRIFT,
                _METRIC_CONTACT_APPLICABLE,
                _METRIC_CONTACT_STABLE_COUNT,
                _METRIC_SOURCE_CONTACT_CONFIDENCE,
                cfg.acceptance.source.required_position_upper_m,
                cfg.acceptance.source.required_distal_position_upper_m,
                cfg.acceptance.source.required_distal_direction_upper_rad,
                cfg.acceptance.source.root_rotation_upper_rad,
                cfg.acceptance.contact.gap_upper_m,
                cfg.acceptance.contact.tilt_upper_rad,
                cfg.acceptance.contact.slip_speed_upper_mps,
                cfg.acceptance.contact.cumulative_drift_upper_m,
            ],
            outputs=[wp.from_torch(accepted), wp.from_torch(active)],
            device=str(candidate.device),
        )

    def solve_phase(
        frame_count: int,
        clip_offsets: torch.Tensor,
        step_seconds: torch.Tensor,
        phase_active: torch.Tensor,
        phase_constraint_geometry_feasible: torch.Tensor,
        phase_inner_solve_converged: torch.Tensor,
        phase_globalization_succeeded: torch.Tensor,
        *,
        frozen_dof_indices: torch.Tensor | None,
        inequalities: IKTrajectorySolver.ResidualInequalities | None,
        restore_initial_root: bool = False,
        residual_activity: IKTrajectorySolver.ResidualActivity | None = None,
        iteration_limit: int | None = None,
        velocity_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
        terminal_quality: torch.Tensor | None = None,
        terminal_attempted: torch.Tensor | None = None,
        terminal_accepted: torch.Tensor | None = None,
        terminal_acceptance: Literal["constraints", "source", "source_contact"] = "source_contact",
        feasibility_only: bool = False,
        adaptive_recovery: bool = False,
    ) -> None:
        """Run one hard-constrained phase until every eligible segment converges or reaches its cap."""
        if terminal_acceptance not in ("constraints", "source", "source_contact"):
            raise ValueError(f"Unsupported terminal acceptance scope: {terminal_acceptance}.")
        if type(feasibility_only) is not bool:
            raise TypeError("feasibility_only must be a bool.")
        if type(restore_initial_root) is not bool:
            raise TypeError("restore_initial_root must be a bool.")
        if restore_initial_root and frozen_dof_indices is None:
            raise ValueError("Initial-root restoration requires frozen root degrees of freedom.")
        if type(adaptive_recovery) is not bool:
            raise TypeError("adaptive_recovery must be a bool.")
        if (terminal_quality is None) != (terminal_attempted is None) or (terminal_quality is None) != (
            terminal_accepted is None
        ):
            raise ValueError("Terminal quality, attempted, and accepted tensors must be provided together.")
        terminal_enabled = terminal_quality is not None
        if terminal_enabled and (
            terminal_quality.shape != (phase_active.shape[0], _TRAJECTORY_METRIC_COUNT)
            or terminal_attempted.shape != phase_active.shape
            or terminal_accepted.shape != phase_active.shape
        ):
            raise ValueError("Terminal acceptance tensors differ from the active phase shape.")
        if adaptive_recovery and not terminal_enabled:
            raise ValueError("Adaptive recovery requires terminal constraint evidence.")
        if not bool(torch.any(phase_active)):
            return
        limit = cfg.max_iterations if iteration_limit is None else iteration_limit
        if type(limit) is not int or limit < 1 or limit > cfg.max_iterations:
            raise ValueError("Trajectory phase iteration limit must be positive and within the configured cap.")
        phase_velocity_lower, phase_velocity_upper = (
            (velocity_lower, velocity_upper) if velocity_bounds is None else velocity_bounds
        )
        iteration_geometry_feasible = workspace.segment_iteration_geometry_feasible[: phase_active.shape[0]]
        iteration_inner_converged = workspace.segment_iteration_inner_converged[: phase_active.shape[0]]
        iteration_globalization_succeeded = workspace.segment_iteration_globalization_succeeded[: phase_active.shape[0]]
        iteration_residual_constraints_satisfied = workspace.segment_iteration_residual_constraints_satisfied[
            : phase_active.shape[0]
        ]
        iteration_attempted = workspace.segment_iteration_attempted[: phase_active.shape[0]]
        segment_damping = workspace.segment_damping[: phase_active.shape[0]]
        recovery_count = workspace.segment_recovery_count[: phase_active.shape[0]]
        segment_damping.fill_(cfg.damping)
        recovery_count.zero_()
        for iteration in range(limit):
            check_convergence = (iteration + 1) % cfg.convergence_check_interval == 0 or iteration + 1 == limit
            if adaptive_recovery:
                iteration_attempted.copy_(phase_active)
            write_velocity_canonical(
                candidate.target, joint_q, clip_offsets, step_seconds, workspace.joint_qd[:frame_count]
            )
            solver.solve(
                joint_q,
                joint_q,
                clip_offsets,
                step_seconds,
                workspace.base_weights,
                workspace.temporal_weights,
                residual_activity=residual_activity,
                inequalities=inequalities,
                frozen_dof_indices=frozen_dof_indices,
                coordinate_bounds=solver_coordinate_bounds,
                joint_velocity=workspace.joint_qd[:frame_count],
                velocity_lower=phase_velocity_lower,
                velocity_upper=phase_velocity_upper,
                segment_feasible=iteration_geometry_feasible,
                segment_active=phase_active,
                segment_damping=segment_damping,
                segment_direction_valid=iteration_inner_converged,
                segment_globalization_succeeded=iteration_globalization_succeeded,
                segment_residual_constraints_satisfied=(
                    iteration_residual_constraints_satisfied if terminal_enabled else None
                ),
                feasibility_only=feasibility_only,
                convergence_tolerance=cfg.convergence_tolerance if check_convergence else None,
            )
            if restore_initial_root:
                joint_q[:, :7].copy_(targets.initial_joint_q[:frame_count, :7])
            if adaptive_recovery:
                wp.launch(
                    _motion_solver_recovery_update,
                    dim=phase_active.shape[0],
                    inputs=[
                        wp.from_torch(iteration_attempted),
                        wp.from_torch(iteration_geometry_feasible),
                        wp.from_torch(iteration_inner_converged),
                        wp.from_torch(iteration_globalization_succeeded),
                        wp.from_torch(iteration_residual_constraints_satisfied),
                        _SOLVER_RECOVERY_LIMIT,
                    ],
                    outputs=[
                        wp.from_torch(phase_active),
                        wp.from_torch(recovery_count),
                        wp.from_torch(segment_damping),
                    ],
                    device=str(candidate.device),
                )
            _motion_phase_update(
                phase_active,
                phase_constraint_geometry_feasible,
                phase_inner_solve_converged,
                phase_globalization_succeeded,
                iteration_geometry_feasible,
                iteration_inner_converged,
                iteration_globalization_succeeded,
            )
            if terminal_enabled:
                if terminal_acceptance != "constraints":
                    _trajectory_recompute_clip_quality(
                        candidate.target,
                        workspace,
                        targets,
                        joint_q,
                        clip_offsets,
                        step_seconds,
                        terminal_quality,
                    )
                update_phase_acceptance(
                    terminal_quality,
                    terminal_attempted,
                    phase_constraint_geometry_feasible,
                    iteration_residual_constraints_satisfied,
                    terminal_accepted,
                    phase_active,
                    acceptance=terminal_acceptance,
                )
            if (check_convergence or terminal_enabled) and not bool(torch.any(phase_active)):
                break

    def prepare_contact_targets(frame_count: int, clip_offsets: torch.Tensor, phase_active: torch.Tensor) -> None:
        workspace.joint_qd.zero_()
        reference.eval_fk_batched_torch(workspace.joint_q, workspace.joint_qd, workspace.body_q, workspace.body_qd)
        wp.launch(
            _trajectory_support_points,
            dim=(target_point_count, frame_count),
            inputs=[
                wp.from_torch(workspace.body_q),
                wp.from_torch(targets.support_body_indices),
                wp.from_torch(targets.support_point_body_m),
                frame_count,
                target_point_count,
            ],
            outputs=[wp.from_torch(targets.target_support_position_m)],
            device=str(candidate.device),
        )
        wp.launch(
            _motion_contact_interval_targets,
            dim=(phase_active.shape[0], contact_channel_count),
            inputs=[
                wp.from_torch(workspace.obstacle_pose),
                wp.from_torch(targets.source_landmark_position_m),
                wp.from_torch(targets.source_direction_point_position_m),
                wp.from_torch(targets.contact_direction_row_tensor),
                wp.from_torch(targets.direction_position_row_tensor),
                wp.from_torch(targets.contact_forward_body),
                wp.from_torch(targets.contact_normal_body),
                wp.from_torch(targets.support_point_body_m),
                wp.from_torch(targets.support_channel_slots),
                wp.from_torch(workspace.source_channel_confidence),
                wp.from_torch(workspace.source_channel_stable),
                wp.from_torch(workspace.source_channel_edge_stable),
                wp.from_torch(clip_offsets),
                wp.from_torch(phase_active),
                phase_active.shape[0],
                contact_channel_count,
                target_point_count,
            ],
            outputs=[wp.from_torch(targets.target_support_position_m)],
            device=str(candidate.device),
        )

    for clip_start, clip_stop in zip(
        memory_plan.batch_segment_offsets[:-1], memory_plan.batch_segment_offsets[1:], strict=True
    ):
        frame_start = candidate.clip_index.offsets[clip_start]
        frame_stop = candidate.clip_index.offsets[clip_stop]
        frame_count = frame_stop - frame_start
        batch_index = _motion_clip_batch_index(candidate.clip_index, clip_start, clip_stop)
        batch_clip_count = clip_stop - clip_start
        local_offsets = tuple(
            offset - frame_start for offset in candidate.clip_index.offsets[clip_start : clip_stop + 1]
        )
        step_values = tuple(1.0 / clip.source_fps for clip in batch_index.clips)
        stream.fill(targets, frame_start, frame_stop)
        if frame_count < capacity:
            targets.initial_joint_q[frame_count:].copy_(targets.initial_joint_q[frame_count - 1 : frame_count])
            targets.source_contact_probe_position_m[:, frame_count:].copy_(
                targets.source_contact_probe_position_m[:, frame_count - 1 : frame_count]
            )
            targets.source_landmark_position_m[:, frame_count:].copy_(
                targets.source_landmark_position_m[:, frame_count - 1 : frame_count]
            )
            targets.source_landmark_rotation_xyzw[:, frame_count:].copy_(
                targets.source_landmark_rotation_xyzw[:, frame_count - 1 : frame_count]
            )
            targets.source_direction_point_position_m[:, frame_count:].copy_(
                targets.source_direction_point_position_m[:, frame_count - 1 : frame_count]
            )
            targets.target_support_position_m[:, frame_count:].copy_(
                targets.target_support_position_m[:, frame_count - 1 : frame_count]
            )
        clip_offsets = torch.tensor(local_offsets, dtype=torch.int32, device=candidate.device)
        step_seconds = torch.tensor(step_values, dtype=torch.float32, device=candidate.device)
        _motion_infer_contact_evidence(
            cfg.contact,
            targets,
            clip_offsets,
            step_seconds,
            workspace.source_plane_height_m,
            workspace.source_probe_active,
            workspace.source_probe_stable,
            workspace.source_channel_confidence,
            workspace.source_channel_stable,
            workspace.source_channel_edge_stable,
        )
        workspace.source_channel_clearance_lift_m.zero_()
        wp.launch(
            _motion_contact_activity,
            dim=(frame_count, contact_channel_count),
            inputs=[
                wp.from_torch(workspace.source_channel_stable),
                wp.from_torch(workspace.source_channel_edge_stable),
                wp.from_torch(workspace.source_channel_confidence),
                wp.from_torch(workspace.source_channel_clearance_lift_m),
                frame_count,
                contact_channel_count,
            ],
            outputs=[
                wp.from_torch(workspace.source_channel_normal_owned),
                wp.from_torch(workspace.source_channel_activity),
            ],
            device=str(candidate.device),
        )
        if contact_evidence is not None:
            torch.ne(
                workspace.source_channel_stable[:frame_count],
                0,
                out=contact_evidence.source_stable[frame_start:frame_stop],
            )
        workspace.joint_q.copy_(targets.initial_joint_q)
        workspace.segment_active[:batch_clip_count].fill_(1)
        if use_frame_ik:
            clip_offsets_wp = wp.from_torch(clip_offsets)
            wp.launch(
                _motion_frame_seed_global_gather,
                dim=batch_clip_count,
                inputs=[
                    frame_seed_source_landmark_position_wp,
                    frame_seed_source_landmark_rotation_wp,
                    frame_seed_source_direction_position_wp,
                    initializer_baseline_wp,
                    clip_offsets_wp,
                    batch_clip_count,
                    len(prototype.position_body_indices),
                    len(prototype.rotation_body_indices),
                    len(prototype.direction_body_indices),
                    coordinate_count,
                ],
                outputs=[
                    frame_seed_global_landmark_position_wp,
                    frame_seed_global_landmark_rotation_wp,
                    frame_seed_global_direction_position_wp,
                    frame_seed_global_joint_q_wp,
                ],
                device=str(candidate.device),
            )

            def project_global_frame_seeds(values: wp.array) -> None:
                wp.launch(
                    _motion_frame_seed_project,
                    dim=(batch_clip_count * _FRAME_SEED_GLOBAL_SEEDS, max(1, len(coordinate_q_indices))),
                    inputs=[
                        values,
                        frame_seed_global_joint_q_wp,
                        initializer_coordinate_indices_wp,
                        initializer_coordinate_lower_wp,
                        initializer_coordinate_upper_wp,
                        batch_clip_count * _FRAME_SEED_GLOBAL_SEEDS,
                        len(coordinate_q_indices),
                        _FRAME_SEED_GLOBAL_SEEDS,
                        wp.uint8(1 if source_root_fixed else 0),
                    ],
                    device=str(candidate.device),
                )

            frame_seed_global_solver.solve(
                frame_seed_global_joint_q_wp,
                frame_seed_global_joint_q_wp,
                max_iterations=_FRAME_SEED_GLOBAL_ITERATIONS,
                active_problem_count=batch_clip_count,
                convergence_tolerance=None,
                projection=project_global_frame_seeds,
                projection_interval=1,
            )
            wp.launch(
                _motion_frame_seed_global_scatter,
                dim=(batch_clip_count, coordinate_count),
                inputs=[
                    frame_seed_global_joint_q_wp,
                    clip_offsets_wp,
                    batch_clip_count,
                    coordinate_count,
                ],
                outputs=[initializer_joint_q_wp],
                device=str(candidate.device),
            )

            def project_local_frame_seeds(values: wp.array) -> None:
                wp.launch(
                    _motion_frame_seed_project,
                    dim=(
                        _FRAME_SEED_LOCAL_CANDIDATES * batch_clip_count,
                        max(1, len(coordinate_q_indices)),
                    ),
                    inputs=[
                        values,
                        frame_seed_local_joint_q_wp,
                        initializer_coordinate_indices_wp,
                        initializer_coordinate_lower_wp,
                        initializer_coordinate_upper_wp,
                        _FRAME_SEED_LOCAL_CANDIDATES * batch_clip_count,
                        len(coordinate_q_indices),
                        _FRAME_SEED_LOCAL_CANDIDATES,
                        wp.uint8(1 if source_root_fixed else 0),
                    ],
                    device=str(candidate.device),
                )

            maximum_local_frames = max(
                local_offsets[clip + 1] - local_offsets[clip] for clip in range(batch_clip_count)
            )
            for relative_frame in range(1, maximum_local_frames):
                wp.launch(
                    _motion_frame_seed_local_gather,
                    dim=batch_clip_count,
                    inputs=[
                        frame_seed_source_landmark_position_wp,
                        frame_seed_source_landmark_rotation_wp,
                        frame_seed_source_direction_position_wp,
                        initializer_joint_q_wp,
                        clip_offsets_wp,
                        initializer_baseline_wp,
                        relative_frame,
                        batch_clip_count,
                        len(prototype.position_body_indices),
                        len(prototype.rotation_body_indices),
                        len(prototype.direction_body_indices),
                        coordinate_count,
                    ],
                    outputs=[
                        frame_seed_local_landmark_position_wp,
                        frame_seed_local_landmark_rotation_wp,
                        frame_seed_local_direction_position_wp,
                        frame_seed_local_joint_q_wp,
                        frame_seed_local_candidate_joint_q_wp,
                        frame_seed_frame_indices_wp,
                        frame_seed_frame_active_wp,
                    ],
                    device=str(candidate.device),
                )
                project_local_frame_seeds(frame_seed_local_candidate_joint_q_wp)
                frame_seed_local_optimizer.reset()
                frame_seed_local_optimizer.step(
                    frame_seed_local_candidate_joint_q_wp,
                    frame_seed_local_candidate_joint_q_wp,
                    iterations=_FRAME_SEED_LOCAL_ITERATIONS,
                    projection=project_local_frame_seeds,
                    projection_interval=1,
                )
                frame_seed_local_optimizer.compute_costs(frame_seed_local_candidate_joint_q_wp)
                wp.launch(
                    _motion_frame_seed_local_scatter,
                    dim=batch_clip_count,
                    inputs=[
                        frame_seed_local_candidate_joint_q_wp,
                        frame_seed_local_cost_wp,
                        initializer_baseline_wp,
                        frame_seed_frame_indices_wp,
                        frame_seed_frame_active_wp,
                        initializer_coordinate_indices_wp,
                        initializer_coordinate_lower_wp,
                        initializer_coordinate_upper_wp,
                        batch_clip_count,
                        coordinate_count,
                        len(coordinate_q_indices),
                        wp.uint8(1 if source_root_fixed else 0),
                    ],
                    outputs=[initializer_joint_q_wp],
                    device=str(candidate.device),
                )
        joint_q = workspace.joint_q[:frame_count]
        write_velocity_canonical(
            candidate.target,
            joint_q,
            clip_offsets,
            step_seconds,
            workspace.joint_qd[:frame_count],
        )
        wp.launch(
            _motion_initializer_validate_or_restore,
            dim=batch_clip_count,
            inputs=[
                initializer_joint_q_wp,
                initializer_baseline_wp,
                wp.from_torch(workspace.joint_qd),
                initializer_coordinate_indices_wp,
                initializer_coordinate_lower_wp,
                initializer_coordinate_upper_wp,
                wp.from_torch(clip_offsets),
                batch_clip_count,
                coordinate_count,
                dof_count,
                len(coordinate_q_indices),
            ],
            outputs=[wp.from_torch(workspace.segment_active)],
            device=str(candidate.device),
        )
        if source_root_fixed:
            torch._assert_async(
                torch.all(workspace.joint_q[:frame_count, :7] == targets.initial_joint_q[:frame_count, :7]),
                "Fixed-root frame initialization must preserve every mapped source root.",
            )
        phase_active = workspace.segment_active[:batch_clip_count]
        targets.initial_joint_q[:frame_count].copy_(joint_q)
        workspace.certified_joint_q[:frame_count].copy_(joint_q)
        batch_quality = trajectory_quality[clip_start:clip_stop]
        _trajectory_recompute_clip_quality(
            candidate.target,
            workspace,
            targets,
            joint_q,
            clip_offsets,
            step_seconds,
            batch_quality,
        )
        if view_evidence is not None:
            view_evidence.stage_quality[clip_start:clip_stop, 0].copy_(batch_quality)
        batch_constraint_geometry_feasible = constraint_geometry_feasible[clip_start:clip_stop]
        batch_inner_solve_converged = inner_solve_converged[clip_start:clip_stop]
        batch_nonlinear_refinement_required = nonlinear_refinement_required[clip_start:clip_stop]
        batch_nonlinear_phases_converged = nonlinear_phases_converged[clip_start:clip_stop]
        phase_attempted = workspace.segment_phase_attempted[:batch_clip_count]
        phase_globalization_succeeded = workspace.segment_phase_globalization_succeeded[:batch_clip_count]
        phase_converged = workspace.segment_phase_converged[:batch_clip_count]
        torch.ne(phase_active, 0, out=batch_constraint_geometry_feasible)
        batch_inner_solve_converged.copy_(batch_constraint_geometry_feasible)
        batch_nonlinear_phases_converged.copy_(batch_constraint_geometry_feasible)
        batch_nonlinear_refinement_required.zero_()
        phase_globalization_succeeded.copy_(batch_constraint_geometry_feasible)
        phase_attempted.copy_(batch_constraint_geometry_feasible)
        batch_nonlinear_refinement_required.copy_(phase_attempted)
        phase_active.copy_(phase_attempted)
        torch.index_select(workspace.joint_q, 1, targets.coordinate_indices, out=workspace.joint_reference)
        residual_activity = IKTrajectorySolver.ResidualActivity(
            values=workspace.source_channel_activity[:frame_count],
            group_by_residual=workspace.residual_layout.activity_group_by_residual,
            first_difference_group_by_residual=workspace.residual_layout.first_difference_group_by_residual,
        )
        phase_active.copy_(phase_attempted)
        workspace.joint_qd.zero_()
        reference.eval_fk_batched_torch(workspace.joint_q, workspace.joint_qd, workspace.body_q, workspace.body_qd)
        wp.launch(
            _motion_target_ground_gauge,
            dim=batch_clip_count,
            inputs=[
                wp.from_torch(workspace.obstacle_pose),
                wp.from_torch(workspace.body_q),
                wp.from_torch(targets.contact_body_indices),
                wp.from_torch(targets.support_point_body_m),
                wp.from_torch(targets.support_channel_slots),
                wp.from_torch(workspace.source_channel_confidence),
                wp.from_torch(workspace.source_channel_stable),
                wp.from_torch(clip_offsets),
                wp.from_torch(phase_active),
                batch_clip_count,
                contact_channel_count,
                len(targets.position_body_indices),
                len(targets.direction_body_indices),
                target_point_count,
                0.0,
            ],
            outputs=[
                wp.from_torch(workspace.joint_q),
                wp.from_torch(targets.source_landmark_position_m),
                wp.from_torch(targets.source_direction_point_position_m),
            ],
            device=str(candidate.device),
        )
        targets.initial_joint_q[:frame_count, :7].copy_(joint_q[:, :7])
        wp.launch(
            _motion_scalar_velocity_box_witness,
            dim=(batch_clip_count, len(coordinate_q_indices)),
            inputs=[
                wp.from_torch(workspace.joint_q),
                wp.from_torch(solver_coordinate_bounds.coordinate_indices),
                wp.from_torch(solver_coordinate_bounds.dof_indices),
                wp.from_torch(solver_coordinate_bounds.lower),
                wp.from_torch(solver_coordinate_bounds.upper),
                wp.from_torch(workspace.velocity_lower),
                wp.from_torch(workspace.velocity_upper),
                wp.from_torch(step_seconds),
                wp.from_torch(clip_offsets),
                wp.from_torch(phase_active),
                batch_clip_count,
                len(coordinate_q_indices),
            ],
            outputs=[
                wp.from_torch(workspace.velocity_reachable_lower),
                wp.from_torch(workspace.velocity_reachable_upper),
            ],
            device=str(candidate.device),
        )
        torch.ne(phase_active, 0, out=phase_attempted)
        batch_constraint_geometry_feasible.logical_and_(phase_attempted)
        prepare_contact_targets(frame_count, clip_offsets, phase_active)
        torch.ne(phase_active, 0, out=phase_attempted)
        batch_constraint_geometry_feasible.logical_and_(phase_attempted)
        _motion_monolithic_weights(
            workspace.residual_layout,
            cfg,
            targets,
            workspace.base_weights,
            workspace.temporal_weights,
        )
        phase_converged.zero_()
        phase_active.copy_(phase_attempted)
        solve_phase(
            frame_count,
            clip_offsets,
            step_seconds,
            phase_active,
            batch_constraint_geometry_feasible,
            batch_inner_solve_converged,
            phase_globalization_succeeded,
            residual_activity=residual_activity,
            inequalities=None,
            frozen_dof_indices=root_dof_indices if source_root_fixed else None,
            restore_initial_root=source_root_fixed,
            velocity_bounds=(workspace.source_velocity_lower, workspace.source_velocity_upper),
        )
        _motion_phase_finish(
            phase_active,
            batch_constraint_geometry_feasible,
            batch_inner_solve_converged,
            phase_globalization_succeeded,
            batch_nonlinear_phases_converged,
            phase_converged,
        )
        _trajectory_recompute_clip_quality(
            candidate.target,
            workspace,
            targets,
            joint_q,
            clip_offsets,
            step_seconds,
            batch_quality,
            None if view_evidence is None else view_evidence.contact_points[frame_start:frame_stop],
            None if view_evidence is None else view_evidence.contact_valid[frame_start:frame_stop],
        )
        batch_coordinates = candidate.target.coordinates_from_newton(joint_q, batch_index)
        if view_evidence is not None:
            position_count = len(targets.position_body_indices)
            direction_count = len(targets.direction_body_indices)
            target_landmarks = view_evidence.target_landmarks[frame_start:frame_stop]
            solved_landmarks = view_evidence.solved_robot_landmarks[frame_start:frame_stop]
            target_landmarks[:, :position_count].copy_(
                targets.source_landmark_position_m[:, :frame_count].transpose(0, 1)
            )
            target_landmarks[:, position_count:].copy_(
                targets.source_direction_point_position_m[:, :frame_count].transpose(0, 1)
            )
            torch.index_select(
                workspace.body_q[:frame_count, :, :3],
                1,
                targets.position_body_index_tensor,
                out=solved_landmarks[:, :position_count],
            )
            wp.launch(
                _trajectory_support_points,
                dim=(direction_count, frame_count),
                inputs=[
                    wp.from_torch(workspace.body_q),
                    wp.from_torch(targets.direction_body_index_tensor),
                    wp.from_torch(targets.direction_point_body_m),
                    frame_count,
                    direction_count,
                ],
                outputs=[wp.from_torch(workspace.achieved_direction_position_m)],
                device=str(joint_q.device),
            )
            solved_landmarks[:, position_count:].copy_(
                workspace.achieved_direction_position_m[:, :frame_count].transpose(0, 1)
            )
            view_evidence.target_support[frame_start:frame_stop].copy_(
                targets.target_support_position_m[:, :frame_count].transpose(0, 1)
            )
        output_coordinates._copy_clip_(frame_start, frame_stop, batch_coordinates)
    stream.finish()
    target_coordinate_evidence = _certify_target_coordinates(candidate.target, output_coordinates, candidate.clip_index)
    return _MotionTrajectorySolvedCandidate(
        target=candidate.target,
        clip_index=candidate.clip_index,
        coordinates=output_coordinates,
        trajectory_quality=trajectory_quality,
        target_coordinate_evidence=target_coordinate_evidence,
        constraint_geometry_feasible=constraint_geometry_feasible,
        inner_solve_converged=inner_solve_converged,
        nonlinear_refinement_required=nonlinear_refinement_required,
        nonlinear_phases_converged=nonlinear_phases_converged,
        acceptance=cfg.acceptance,
        device=candidate.device,
        contact_evidence=contact_evidence,
        view_evidence=view_evidence,
    )


def motion_criterion_constraint_geometry_feasible(
    cfg, candidate: _builder._MotionTrajectorySolvedCandidate, rows: torch.Tensor
) -> torch.Tensor:
    """Accept solved sequences only when every constrained iteration was geometrically feasible."""
    del cfg
    return candidate.constraint_geometry_feasible[rows]


def motion_criterion_inner_solve_converged(
    cfg, candidate: _builder._MotionTrajectorySolvedCandidate, rows: torch.Tensor
) -> torch.Tensor:
    """Accept solved sequences only when every inner trajectory solve converged."""
    del cfg
    return candidate.inner_solve_converged[rows]


def motion_criterion_required_refinement_converged(
    cfg, candidate: _builder._MotionTrajectorySolvedCandidate, rows: torch.Tensor
) -> torch.Tensor:
    """Accept solved sequences when skipped refinement or all required nonlinear phases converged."""
    del cfg
    return ~candidate.nonlinear_refinement_required[rows] | candidate.nonlinear_phases_converged[rows]


def _motion_source_fidelity_accepted(
    policy: MotionTrajectorySolveCfg.AcceptanceCfg.SourceCfg, quality: torch.Tensor
) -> torch.Tensor:
    """Return whether every target-required source-fidelity maximum is finite and bounded."""
    required_position = quality[:, _METRIC_SOURCE_REQUIRED_POSITION]
    required_distal_position = quality[:, _METRIC_SOURCE_REQUIRED_DISTAL_POSITION]
    required_distal_direction = quality[:, _METRIC_SOURCE_REQUIRED_DISTAL_DIRECTION]
    root_rotation = quality[:, _METRIC_SOURCE_ROOT_ROTATION]
    return (
        torch.isfinite(required_position)
        & (required_position >= 0.0)
        & (required_position <= policy.required_position_upper_m)
        & torch.isfinite(required_distal_position)
        & (required_distal_position >= 0.0)
        & (required_distal_position <= policy.required_distal_position_upper_m)
        & torch.isfinite(required_distal_direction)
        & (required_distal_direction >= 0.0)
        & (required_distal_direction <= policy.required_distal_direction_upper_rad)
        & torch.isfinite(root_rotation)
        & (root_rotation >= 0.0)
        & (root_rotation <= policy.root_rotation_upper_rad)
    )


def motion_criterion_source_fidelity(
    cfg: MotionSourceFidelityCriterionCfg,
    candidate: _builder._MotionTrajectorySolvedCandidate,
    rows: torch.Tensor,
) -> torch.Tensor:
    """Accept clips only when every source-fidelity bound in the solve policy is met."""
    del cfg
    return _motion_source_fidelity_accepted(candidate.acceptance.source, candidate.trajectory_quality[rows])


def _motion_contact_rows_accepted(
    policy: MotionTrajectorySolveCfg.AcceptanceCfg.ContactCfg, quality: torch.Tensor
) -> torch.Tensor:
    """Return bounded applicable contact or honest NaN rows when contact is not applicable."""
    gap = quality[:, _METRIC_CONTACT_GAP]
    tilt = quality[:, _METRIC_CONTACT_TILT]
    slip_speed = quality[:, _METRIC_CONTACT_SLIP_SPEED]
    cumulative_drift = quality[:, _METRIC_CONTACT_CUMULATIVE_DRIFT]
    applicable = quality[:, _METRIC_CONTACT_APPLICABLE]
    stable_count = quality[:, _METRIC_CONTACT_STABLE_COUNT]
    confidence = quality[:, _METRIC_SOURCE_CONTACT_CONFIDENCE]

    metadata_valid = (
        torch.isfinite(applicable)
        & ((applicable == 0.0) | (applicable == 1.0))
        & torch.isfinite(stable_count)
        & (stable_count >= 0.0)
        & (stable_count == torch.floor(stable_count))
        & ((applicable > 0.5) == (stable_count > 0.0))
        & torch.isfinite(confidence)
        & (confidence >= 0.0)
        & (confidence <= 1.0)
    )
    applicable_valid = (
        torch.isfinite(gap)
        & (gap >= 0.0)
        & (gap <= policy.gap_upper_m)
        & torch.isfinite(tilt)
        & (tilt >= 0.0)
        & (tilt <= policy.tilt_upper_rad)
        & torch.isfinite(slip_speed)
        & (slip_speed >= 0.0)
        & (slip_speed <= policy.slip_speed_upper_mps)
        & torch.isfinite(cumulative_drift)
        & (cumulative_drift >= 0.0)
        & (cumulative_drift <= policy.cumulative_drift_upper_m)
    )
    not_applicable_valid = (
        torch.isnan(gap) & torch.isnan(tilt) & torch.isnan(slip_speed) & torch.isnan(cumulative_drift)
    )
    return metadata_valid & torch.where(applicable > 0.5, applicable_valid, not_applicable_valid)


def motion_criterion_contact(
    cfg: MotionContactCriterionCfg,
    candidate: _builder._MotionTrajectorySolvedCandidate,
    rows: torch.Tensor,
) -> torch.Tensor:
    """Accept clips only when every contact bound in the solve policy is met."""
    del cfg
    policy = candidate.acceptance.contact
    accepted = _motion_contact_rows_accepted(policy, candidate.trajectory_quality[rows])
    if policy.require_any_stable_contact:
        accepted &= torch.any(candidate.trajectory_quality[:, _METRIC_CONTACT_STABLE_COUNT] > 0.0)
    return accepted


def _trajectory_corpus_quality(
    candidate: _builder._MotionTrajectorySolvedCandidate, accepted: torch.Tensor
) -> torch.Tensor:
    """Materialize one quality row per solved trajectory candidate."""
    clip_count = len(candidate.clip_index.clips)
    if (
        candidate.trajectory_quality.shape != (clip_count, _TRAJECTORY_METRIC_COUNT)
        or candidate.target_coordinate_evidence.shape != (clip_count, len(_TARGET_COORDINATE_QUALITY_NAMES))
        or candidate.constraint_geometry_feasible.shape != (clip_count,)
        or candidate.inner_solve_converged.shape != (clip_count,)
        or candidate.nonlinear_refinement_required.shape != (clip_count,)
        or candidate.nonlinear_phases_converged.shape != (clip_count,)
        or accepted.shape != (clip_count,)
        or accepted.dtype is not torch.bool
    ):
        raise ValueError("Trajectory quality requires one solved evidence row per complete source clip.")
    quality = torch.zeros(clip_count, len(_QUALITY_NAMES), dtype=torch.float32, device=candidate.coordinates.device)
    quality[:, _QUALITY_TRAJECTORY_ROUTE].fill_(1.0)
    quality[:, _QUALITY_ACCEPTED].copy_(accepted)
    quality[:, _TRAJECTORY_METRIC_START:_TRAJECTORY_METRIC_STOP].copy_(candidate.trajectory_quality)
    quality[:, _QUALITY_CONSTRAINT_GEOMETRY_FEASIBLE].copy_(candidate.constraint_geometry_feasible)
    quality[:, _QUALITY_INNER_SOLVE_CONVERGED].copy_(candidate.inner_solve_converged)
    quality[:, _QUALITY_NONLINEAR_REFINEMENT_REQUIRED].copy_(candidate.nonlinear_refinement_required)
    quality[:, _QUALITY_NONLINEAR_PHASES_CONVERGED].copy_(candidate.nonlinear_phases_converged)
    quality[:, _TARGET_COORDINATE_QUALITY_START:_TARGET_COORDINATE_QUALITY_STOP].copy_(
        candidate.target_coordinate_evidence
    )
    quality[:, _DYNAMICS_QUALITY_START:_DYNAMICS_QUALITY_STOP].fill_(torch.nan)
    quality[:, -1].fill_(1.0)
    return quality


def _populate_motion_contact_quality(
    target: MotionFrameTarget,
    clip_index: MotionClipIndex,
    coordinates: MotionGeneralizedCoordinates,
    quality: torch.Tensor,
    evidence: _MotionAcceptedContactEvidence | None,
) -> None:
    """Evaluate retained trajectory clips once and write only clip diagnostics."""
    if evidence is None:
        return
    sequence_indices = evidence.sequence_indices
    support_count = len(evidence.support_body_indices)
    contact_channel_count = evidence.source_stable.shape[1] if evidence.source_stable.ndim == 2 else -1
    expected_frames = sum(clip_index.clips[index].frame_count for index in sequence_indices)
    if (
        not sequence_indices
        or quality.shape != (len(clip_index.clips), len(_QUALITY_NAMES))
        or evidence.source_stable.shape[0] != expected_frames
        or support_count < 1
        or evidence.support_point_body_m.shape != (support_count, 3)
        or evidence.support_channel_slots.shape != (support_count,)
        or contact_channel_count < 1
    ):
        raise ValueError("Selected trajectory contact evidence differs from the final corpus schema.")
    torch._assert_async(
        torch.all((evidence.support_channel_slots >= 0) & (evidence.support_channel_slots < contact_channel_count)),
        "Every robot support point must map to one source-contact slot.",
    )

    reference = target.kinematics
    model = reference.model
    clip_lengths = tuple(clip_index.clips[index].frame_count for index in sequence_indices)

    def estimate_memory(frame_capacity: int) -> int:
        derivative_bytes = 2 * frame_capacity * model.joint_dof_count * 4
        return ContactFeasibilityWorkspace.estimate_memory(reference, frame_capacity, support_count) + derivative_bytes

    memory_plan = plan_trajectory_memory(clip_lengths, coordinates.device, estimate_memory)
    workspace = ContactFeasibilityWorkspace(
        reference, memory_plan.workspace_frame_capacity, evidence.support_body_indices
    )
    evidence_offsets = [0]
    for length in clip_lengths:
        evidence_offsets.append(evidence_offsets[-1] + length)

    for clip_start, clip_stop in zip(
        memory_plan.batch_segment_offsets[:-1], memory_plan.batch_segment_offsets[1:], strict=True
    ):
        local_offsets = [0]
        for trajectory_index in range(clip_start, clip_stop):
            sequence_index = sequence_indices[trajectory_index]
            source_start, source_stop = clip_index.offsets[sequence_index : sequence_index + 2]
            destination_start = local_offsets[-1]
            destination_stop = destination_start + source_stop - source_start
            target.write_joint_position_newton(
                MotionGeneralizedCoordinates(
                    coordinates.joint_q[source_start:source_stop],
                    None if coordinates.joint_qd is None else coordinates.joint_qd[source_start:source_stop],
                ),
                workspace.joint_q[destination_start:destination_stop],
            )
            local_offsets.append(destination_stop)

        frame_count = local_offsets[-1]
        offsets = torch.tensor(local_offsets, dtype=torch.int32, device=coordinates.device)
        offsets_i64 = offsets.to(torch.int64)
        step_seconds = torch.tensor(
            tuple(1.0 / clip_index.clips[sequence_indices[index]].source_fps for index in range(clip_start, clip_stop)),
            dtype=torch.float32,
            device=coordinates.device,
        )
        joint_q = workspace.joint_q[:frame_count]
        joint_qd = workspace.joint_qd[:frame_count]
        write_velocity_canonical(target, joint_q, offsets, step_seconds, joint_qd)
        workspace.joint_qdd[:frame_count].copy_(time_gradient_segmented(joint_qd, offsets_i64, step_seconds))
        source_stable = evidence.source_stable[evidence_offsets[clip_start] : evidence_offsets[clip_stop]]
        torch.index_select(source_stable, 1, evidence.support_channel_slots, out=workspace.support_active[:frame_count])
        workspace.support_normal_world[:frame_count].zero_()
        workspace.support_normal_world[:frame_count, :, 2].fill_(1.0)
        workspace.friction_coefficient[:frame_count].fill_(evidence.policy.friction_coefficient)
        result = workspace.evaluate(
            joint_q,
            joint_qd,
            workspace.joint_qdd[:frame_count],
            support_point_body_m=evidence.support_point_body_m,
            support_active=workspace.support_active[:frame_count],
            support_normal_world=workspace.support_normal_world[:frame_count],
            friction_coefficient=workspace.friction_coefficient[:frame_count],
            segment_offsets=tuple(local_offsets),
            iterations=evidence.policy.iterations,
            effort_weight=evidence.policy.effort_weight,
            force_regularization=evidence.policy.force_regularization,
        )
        rows = torch.tensor(sequence_indices[clip_start:clip_stop], dtype=torch.int64, device=quality.device)
        values = (
            result.segment_balance_force_residual_n,
            result.segment_balance_torque_residual_nm,
            result.segment_effort_margin_ratio,
            result.segment_normal_force_min_n,
            result.segment_friction_margin_n,
            result.segment_contact_transition_count.to(quality.dtype),
        )
        for column, value in zip(range(_DYNAMICS_QUALITY_START, _DYNAMICS_QUALITY_STOP), values, strict=True):
            quality[:, column].index_copy_(0, rows, value)
        del result, rows, values
