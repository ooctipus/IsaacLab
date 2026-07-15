# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure Warp kernels for matrix-free whole-trajectory refinement."""

from __future__ import annotations

import warp as wp


@wp.func
def _difference_coefficient(order: int, index: int) -> float:
    if order == 1:
        return -1.0 if index == 0 else 1.0
    if order == 2:
        if index == 1:
            return -2.0
        return 1.0
    if index == 0:
        return -1.0
    if index == 1:
        return 3.0
    if index == 2:
        return -3.0
    return 1.0


@wp.func
def _temporal_stencil_confidence(
    residual_activity: wp.array2d(dtype=wp.float32),
    activity_group: int,
    first_difference_group: int,
    start: int,
    order: int,
) -> float:
    """Return explicit edge confidence or the legacy vertex-stencil minimum."""
    if order == 1 and first_difference_group >= 0:
        return residual_activity[start + 1, first_difference_group]
    confidence = float(1.0)
    if activity_group >= 0:
        for index in range(order + 1):
            confidence = wp.min(confidence, residual_activity[start + index, activity_group])
    return confidence


@wp.kernel
def _frame_segment_indices(
    segment_offsets: wp.array(dtype=wp.int32),
    segment_count: int,
    active_frames: int,
    frame_segment: wp.array(dtype=wp.int32),
):
    frame = wp.tid()
    if frame >= active_frames:
        return
    low = wp.int32(0)
    high = wp.int32(segment_count)
    while low + 1 < high:
        middle = (low + high) // 2
        if frame < segment_offsets[middle]:
            high = middle
        else:
            low = middle
    frame_segment[frame] = low


@wp.kernel
def _segment_damping_validate(
    values: wp.array(dtype=wp.float32),
    segment_count: int,
    contract_error: wp.array(dtype=wp.int32),
):
    """Validate the positive finite LM diagonal owned by each active segment."""
    segment = wp.tid()
    if segment < segment_count:
        value = values[segment]
        if not wp.isfinite(value) or value <= 0.0:
            contract_error[0] = 1


@wp.kernel
def _frozen_dof_indices_validate(
    dof_indices: wp.array(dtype=wp.int32),
    coordinate_bound_dof_indices: wp.array(dtype=wp.int32),
    velocity_lower: wp.array(dtype=wp.float32),
    velocity_upper: wp.array(dtype=wp.float32),
    frozen_count: int,
    coordinate_bound_count: int,
    dof_count: int,
    contract_error: wp.array(dtype=wp.int32),
):
    """Validate an ordered, unconstrained frozen tangent subspace."""
    frozen = wp.tid()
    if frozen >= frozen_count:
        return
    dof = dof_indices[frozen]
    malformed = dof < 0 or dof >= dof_count or (frozen > 0 and dof <= dof_indices[frozen - 1])
    if not malformed:
        malformed = wp.isfinite(velocity_lower[dof]) or wp.isfinite(velocity_upper[dof])
        for bound in range(coordinate_bound_count):
            if coordinate_bound_dof_indices[bound] == dof:
                malformed = True
    if malformed:
        wp.atomic_max(contract_error, 0, 10)


@wp.kernel
def _frozen_dof_jacobian_zero(
    jacobian: wp.array3d(dtype=wp.float32),
    dof_indices: wp.array(dtype=wp.int32),
    frozen_count: int,
    active_frames: int,
):
    """Remove frozen tangent columns from one frame-local Jacobian."""
    frame, residual, frozen = wp.tid()
    if frame >= active_frames or frozen >= frozen_count:
        return
    dof = dof_indices[frozen]
    if dof >= 0 and dof < jacobian.shape[2]:
        jacobian[frame, residual, dof] = 0.0


@wp.kernel
def _frozen_dof_values_zero(
    values: wp.array2d(dtype=wp.float32),
    dof_indices: wp.array(dtype=wp.int32),
    frozen_count: int,
    active_frames: int,
):
    """Keep a solved tangent direction inside the declared active subspace."""
    frame, frozen = wp.tid()
    if frame >= active_frames or frozen >= frozen_count:
        return
    dof = dof_indices[frozen]
    if dof >= 0 and dof < values.shape[1]:
        values[frame, dof] = 0.0


@wp.kernel
def _residual_activity_validate(
    values: wp.array2d(dtype=wp.float32),
    group_by_residual: wp.array(dtype=wp.int32),
    first_difference_group_by_residual: wp.array(dtype=wp.int32),
    frame_count: int,
    group_count: int,
    contract_error: wp.array(dtype=wp.int32),
):
    """Validate compact confidence values and residual-group indices."""
    index = wp.tid()
    if index < group_by_residual.shape[0]:
        group = group_by_residual[index]
        first_difference_group = first_difference_group_by_residual[index]
        if group < -1 or group >= group_count or first_difference_group < -1 or first_difference_group >= group_count:
            contract_error[0] = 1
    if index < frame_count * group_count:
        frame = index // group_count
        group = index - frame * group_count
        confidence = values[frame, group]
        if not wp.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            contract_error[0] = 1


@wp.kernel
def _coordinate_bounds_validate(
    joint_q: wp.array2d(dtype=wp.float32),
    coordinate_indices: wp.array(dtype=wp.int32),
    dof_indices: wp.array(dtype=wp.int32),
    lower: wp.array(dtype=wp.float32),
    upper: wp.array(dtype=wp.float32),
    bound_count: int,
    active_frames: int,
    coordinate_count: int,
    dof_count: int,
    contract_error: wp.array(dtype=wp.int32),
):
    """Validate explicit bounded-scalar ownership and the initial feasible point."""
    index = wp.tid()
    if index < bound_count:
        coordinate = coordinate_indices[index]
        dof = dof_indices[index]
        lower_value = lower[index]
        upper_value = upper[index]
        malformed = (
            coordinate < 0
            or coordinate >= coordinate_count
            or dof < 0
            or dof >= dof_count
            or lower_value != lower_value
            or upper_value != upper_value
            or lower_value > upper_value
            or (not wp.isfinite(lower_value) and lower_value > 0.0)
            or (not wp.isfinite(upper_value) and upper_value < 0.0)
            or (not wp.isfinite(lower_value) and not wp.isfinite(upper_value))
        )
        for previous in range(index):
            if coordinate_indices[previous] == coordinate or dof_indices[previous] == dof:
                malformed = True
        if malformed:
            wp.atomic_max(contract_error, 0, 8)
    if index < active_frames * bound_count:
        frame = index // bound_count
        bound = index - frame * bound_count
        coordinate = coordinate_indices[bound]
        if coordinate >= 0 and coordinate < coordinate_count:
            value = joint_q[frame, coordinate]
            lower_value = lower[bound]
            upper_value = upper[bound]
            if not wp.isfinite(value) or value < lower_value or value > upper_value:
                wp.atomic_max(contract_error, 0, 9)


@wp.kernel
def _coordinate_bounds_project_candidate(
    joint_q: wp.array2d(dtype=wp.float32),
    joint_q_candidate: wp.array2d(dtype=wp.float32),
    delta_candidate: wp.array2d(dtype=wp.float32),
    coordinate_indices: wp.array(dtype=wp.int32),
    dof_indices: wp.array(dtype=wp.int32),
    lower: wp.array(dtype=wp.float32),
    upper: wp.array(dtype=wp.float32),
    bound_count: int,
    active_frames: int,
):
    """Project a scalar-coordinate trial and expose its actual tangent step."""
    frame, bound = wp.tid()
    if frame >= active_frames or bound >= bound_count:
        return
    coordinate = coordinate_indices[bound]
    dof = dof_indices[bound]
    value = wp.clamp(joint_q_candidate[frame, coordinate], lower[bound], upper[bound])
    joint_q_candidate[frame, coordinate] = value
    delta_candidate[frame, dof] = value - joint_q[frame, coordinate]


@wp.kernel
def _coordinate_bounds_mask_binding_dofs(
    joint_q: wp.array2d(dtype=wp.float32),
    right_hand_side: wp.array2d(dtype=wp.float32),
    coordinate_indices: wp.array(dtype=wp.int32),
    dof_indices: wp.array(dtype=wp.int32),
    lower: wp.array(dtype=wp.float32),
    upper: wp.array(dtype=wp.float32),
    bound_count: int,
    active_frames: int,
    residual: wp.array2d(dtype=wp.float32),
    preconditioned: wp.array2d(dtype=wp.float32),
    direction: wp.array2d(dtype=wp.float32),
    operator_direction: wp.array2d(dtype=wp.float32),
):
    """Restrict PCG vectors to the current box-constraint face."""
    frame, bound = wp.tid()
    if frame >= active_frames or bound >= bound_count:
        return
    coordinate = coordinate_indices[bound]
    dof = dof_indices[bound]
    value = joint_q[frame, coordinate]
    rhs = right_hand_side[frame, dof]
    binding = (value <= lower[bound] and rhs <= 0.0) or (value >= upper[bound] and rhs >= 0.0)
    if binding:
        residual[frame, dof] = 0.0
        preconditioned[frame, dof] = 0.0
        direction[frame, dof] = 0.0
        operator_direction[frame, dof] = 0.0


@wp.kernel
def _coordinate_bound_violation_max(
    joint_q: wp.array2d(dtype=wp.float32),
    coordinate_indices: wp.array(dtype=wp.int32),
    lower: wp.array(dtype=wp.float32),
    upper: wp.array(dtype=wp.float32),
    bound_count: int,
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array(dtype=wp.float32),
):
    """Accumulate exact scalar-coordinate box violation by segment."""
    frame, bound = wp.tid()
    if frame >= active_frames or bound >= bound_count or enabled[frame_segment[frame]] == 0:
        return
    value = joint_q[frame, coordinate_indices[bound]]
    if not wp.isfinite(value):
        violation = 3.402823466e38
    else:
        violation = wp.max(lower[bound] - value, value - upper[bound])
    wp.atomic_max(output, frame_segment[frame], wp.max(violation, 0.0))


@wp.kernel
def _dot_segments(
    frame_values: wp.array(dtype=wp.float32),
    segment_offsets: wp.array(dtype=wp.int32),
    segment_count: int,
    enabled: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    segment = wp.tid()
    if segment >= segment_count:
        return
    if enabled[segment] == 0:
        output[segment] = 0.0
        return
    value = float(0.0)
    for frame in range(segment_offsets[segment], segment_offsets[segment + 1]):
        value += frame_values[frame]
    output[segment] = value


@wp.kernel
def _constraints_initialize(
    pose_weights: wp.array(dtype=wp.float32),
    active_frames: int,
    constraint_kind: wp.array2d(dtype=wp.uint8),
    objective_weights: wp.array2d(dtype=wp.float32),
):
    frame, residual = wp.tid()
    if frame < active_frames:
        constraint_kind[frame, residual] = wp.uint8(0)
        objective_weights[frame, residual] = pose_weights[residual]


@wp.kernel
def _inequalities_mark(
    residual_indices: wp.array(dtype=wp.int32),
    inequality_upper: wp.array(dtype=wp.float32),
    inequality_count: int,
    active_frames: int,
    residual_count: int,
    constraint_kind: wp.array2d(dtype=wp.uint8),
    objective_weights: wp.array2d(dtype=wp.float32),
    residual_upper: wp.array(dtype=wp.float32),
    contract_error: wp.array(dtype=wp.int32),
):
    frame, slot = wp.tid()
    if frame >= active_frames or slot >= inequality_count:
        return
    residual = residual_indices[slot]
    upper = inequality_upper[slot]
    if (
        residual < 0
        or residual >= residual_count
        or (slot > 0 and residual <= residual_indices[slot - 1])
        or not wp.isfinite(upper)
    ):
        contract_error[0] = 1
        return
    residual_upper[residual] = upper
    constraint_kind[frame, residual] = wp.uint8(2)
    objective_weights[frame, residual] = 0.0


@wp.kernel
def _equalities_mark(
    active: wp.array2d(dtype=wp.uint8),
    residual_starts_by_target: wp.array(dtype=wp.int32),
    target_count: int,
    active_frames: int,
    residual_count: int,
    constraint_kind: wp.array2d(dtype=wp.uint8),
    objective_weights: wp.array2d(dtype=wp.float32),
    contract_error: wp.array(dtype=wp.int32),
):
    frame, target, axis = wp.tid()
    if frame >= active_frames or target >= target_count:
        return
    state = active[frame, target]
    residual_start = residual_starts_by_target[target]
    if state > wp.uint8(1) or residual_start < 0 or residual_start + 3 > residual_count:
        wp.atomic_max(contract_error, 0, 7)
        return
    if frame == 0 and axis == 0:
        for other in range(target):
            other_start = residual_starts_by_target[other]
            if residual_start < other_start + 3 and other_start < residual_start + 3:
                wp.atomic_max(contract_error, 0, 7)
                return
    if state == wp.uint8(0):
        return
    residual = residual_start + axis
    if constraint_kind[frame, residual] != wp.uint8(0):
        wp.atomic_max(contract_error, 0, 2)
        return
    constraint_kind[frame, residual] = wp.uint8(1)
    objective_weights[frame, residual] = 0.0


@wp.func
def _qp_row_family(code: int, frame_capacity: int, residual_count: int, bound_count: int, dof_count: int) -> int:
    """Return the family of a residual, coordinate, or velocity QP row code."""
    residual_span = frame_capacity * residual_count
    bound_span = frame_capacity * bound_count
    dof_span = frame_capacity * dof_count
    if code < residual_span:
        return 0
    if code < 2 * residual_span:
        return 1
    if code < 2 * residual_span + bound_span:
        return 2
    if code < 2 * residual_span + 2 * bound_span:
        return 3
    if code < 2 * residual_span + 2 * bound_span + dof_span:
        return 4
    return 5


@wp.func
def _qp_row_frame(code: int, frame_capacity: int, residual_count: int, bound_count: int, dof_count: int) -> int:
    """Decode the owning frame of a residual, coordinate, or velocity QP row."""
    residual_span = frame_capacity * residual_count
    bound_span = frame_capacity * bound_count
    dof_span = frame_capacity * dof_count
    if code < residual_span:
        return code // residual_count
    if code < 2 * residual_span:
        return (code - residual_span) // residual_count
    if code < 2 * residual_span + bound_span:
        return (code - 2 * residual_span) // bound_count
    if code < 2 * residual_span + 2 * bound_span:
        return (code - 2 * residual_span - bound_span) // bound_count
    if code < 2 * residual_span + 2 * bound_span + dof_span:
        return (code - 2 * residual_span - 2 * bound_span) // dof_count
    return (code - 2 * residual_span - 2 * bound_span - dof_span) // dof_count


@wp.func
def _qp_row_item(code: int, frame_capacity: int, residual_count: int, bound_count: int, dof_count: int) -> int:
    """Decode the residual, coordinate-bound, or velocity-DOF index of a QP row."""
    residual_span = frame_capacity * residual_count
    bound_span = frame_capacity * bound_count
    dof_span = frame_capacity * dof_count
    if code < residual_span:
        return code % residual_count
    if code < 2 * residual_span:
        return (code - residual_span) % residual_count
    if code < 2 * residual_span + bound_span:
        return (code - 2 * residual_span) % bound_count
    if code < 2 * residual_span + 2 * bound_span:
        return (code - 2 * residual_span - bound_span) % bound_count
    if code < 2 * residual_span + 2 * bound_span + dof_span:
        return (code - 2 * residual_span - 2 * bound_span) % dof_count
    return (code - 2 * residual_span - 2 * bound_span - dof_span) % dof_count


@wp.func
def _qp_slot_active(
    slot: int,
    segment: int,
    segment_offsets: wp.array(dtype=wp.int32),
    row_code: wp.array2d(dtype=wp.int32),
    equality_count: wp.array(dtype=wp.int32),
) -> bool:
    width = row_code.shape[1]
    begin = segment_offsets[segment]
    relative = slot - begin * width
    return relative >= 0 and relative < equality_count[segment]


@wp.func
def _qp_row_coefficient(
    code: int,
    frame: int,
    dof: int,
    jacobian: wp.array3d(dtype=wp.float32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    coordinate_lower: wp.array(dtype=wp.float32),
    coordinate_upper: wp.array(dtype=wp.float32),
    coordinate_bound_count: int,
    dof_count: int,
) -> float:
    """Return one coefficient of a residual, coordinate, or velocity QP row."""
    family = _qp_row_family(code, jacobian.shape[0], jacobian.shape[1], coordinate_bound_count, dof_count)
    row_frame = _qp_row_frame(code, jacobian.shape[0], jacobian.shape[1], coordinate_bound_count, dof_count)
    item = _qp_row_item(code, jacobian.shape[0], jacobian.shape[1], coordinate_bound_count, dof_count)
    if family < 2:
        return jacobian[frame, item, dof] if row_frame == frame else 0.0
    if family < 4:
        if row_frame != frame or coordinate_dof_indices[item] != dof:
            return 0.0
        if family == 2 and coordinate_lower[item] != coordinate_upper[item]:
            return -1.0
        return 1.0
    if item != dof:
        return 0.0
    if frame == row_frame - 1:
        return 1.0 if family == 4 else -1.0
    if frame == row_frame:
        return -1.0 if family == 4 else 1.0
    return 0.0


@wp.func
def _qp_row_rhs_value(
    code: int,
    residuals: wp.array2d(dtype=wp.float32),
    residual_upper: wp.array(dtype=wp.float32),
    joint_q: wp.array2d(dtype=wp.float32),
    coordinate_indices: wp.array(dtype=wp.int32),
    coordinate_lower: wp.array(dtype=wp.float32),
    coordinate_upper: wp.array(dtype=wp.float32),
    coordinate_bound_count: int,
    velocity_current: wp.array2d(dtype=wp.float32),
    velocity_lower: wp.array(dtype=wp.float32),
    velocity_upper: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    step_seconds: wp.array(dtype=wp.float32),
    dof_count: int,
) -> float:
    """Return the unscaled right-hand side of a QP row."""
    family = _qp_row_family(code, residuals.shape[0], residuals.shape[1], coordinate_bound_count, dof_count)
    frame = _qp_row_frame(code, residuals.shape[0], residuals.shape[1], coordinate_bound_count, dof_count)
    item = _qp_row_item(code, residuals.shape[0], residuals.shape[1], coordinate_bound_count, dof_count)
    if family == 0:
        return -residuals[frame, item]
    if family == 1:
        return residual_upper[item] - residuals[frame, item]
    if family < 4:
        value = joint_q[frame, coordinate_indices[item]]
        if family == 2 and coordinate_lower[item] == coordinate_upper[item]:
            return coordinate_lower[item] - value
        if family == 2:
            return value - coordinate_lower[item]
        return coordinate_upper[item] - value
    dt = step_seconds[frame_segment[frame]]
    if family == 4:
        return dt * (velocity_current[frame, item] - velocity_lower[item])
    return dt * (velocity_upper[item] - velocity_current[frame, item])


@wp.kernel
def _qp_rows_initialize(
    constraint_kind: wp.array2d(dtype=wp.uint8),
    coordinate_lower: wp.array(dtype=wp.float32),
    coordinate_upper: wp.array(dtype=wp.float32),
    coordinate_bound_count: int,
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    segment_count: int,
    enabled: wp.array(dtype=wp.int32),
    contract_error: wp.array(dtype=wp.int32),
    row_code: wp.array2d(dtype=wp.int32),
    row_scale: wp.array2d(dtype=wp.float32),
    row_rhs: wp.array2d(dtype=wp.float32),
    equality_count: wp.array(dtype=wp.int32),
):
    """Initialize compact mandatory equality rows."""
    segment = wp.tid()
    if segment >= segment_count:
        return
    width = row_code.shape[1]
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    base = begin * width
    slot_count = (end - begin) * width
    for relative in range(slot_count):
        slot = base + relative
        row = slot // width
        column = slot - row * width
        row_code[row, column] = -1
        row_scale[row, column] = 0.0
        row_rhs[row, column] = 0.0
    equality_count[segment] = 0
    if enabled[segment] == 0:
        return
    equality_capacity = slot_count
    count = int(0)
    residual_count = constraint_kind.shape[1]
    for frame in range(begin, end):
        for residual in range(residual_count):
            if constraint_kind[frame, residual] == wp.uint8(1):
                if count >= equality_capacity:
                    wp.atomic_max(contract_error, 0, 3)
                    equality_count[segment] = equality_capacity
                    return
                slot = base + count
                row = slot // width
                row_code[row, slot - row * width] = frame * residual_count + residual
                count += 1
    residual_span = constraint_kind.shape[0] * residual_count
    for frame in range(begin, end):
        for bound in range(coordinate_bound_count):
            if coordinate_lower[bound] == coordinate_upper[bound]:
                if count >= equality_capacity:
                    wp.atomic_max(contract_error, 0, 3)
                    equality_count[segment] = equality_capacity
                    return
                slot = base + count
                row = slot // width
                code = 2 * residual_span + frame * coordinate_bound_count + bound
                row_code[row, slot - row * width] = code
                count += 1
    equality_count[segment] = count


@wp.kernel
def _qp_rows_refresh(
    jacobian: wp.array3d(dtype=wp.float32),
    residuals: wp.array2d(dtype=wp.float32),
    residual_upper: wp.array(dtype=wp.float32),
    joint_q: wp.array2d(dtype=wp.float32),
    coordinate_indices: wp.array(dtype=wp.int32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    coordinate_lower: wp.array(dtype=wp.float32),
    coordinate_upper: wp.array(dtype=wp.float32),
    coordinate_bound_count: int,
    velocity_current: wp.array2d(dtype=wp.float32),
    velocity_lower: wp.array(dtype=wp.float32),
    velocity_upper: wp.array(dtype=wp.float32),
    step_seconds: wp.array(dtype=wp.float32),
    dof_count: int,
    normal_diagonal: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    epsilon: float,
    row_code: wp.array2d(dtype=wp.int32),
    equality_count: wp.array(dtype=wp.int32),
    row_scale: wp.array2d(dtype=wp.float32),
    row_rhs: wp.array2d(dtype=wp.float32),
):
    """Rebuild row scaling and the scaled QP right-hand side."""
    row, column = wp.tid()
    if row >= active_frames:
        return
    width = row_code.shape[1]
    slot = row * width + column
    segment = frame_segment[row]
    if enabled[segment] == 0 or not _qp_slot_active(slot, segment, segment_offsets, row_code, equality_count):
        row_scale[row, column] = 0.0
        row_rhs[row, column] = 0.0
        return
    code = row_code[row, column]
    family = _qp_row_family(code, jacobian.shape[0], jacobian.shape[1], coordinate_bound_count, dof_count)
    row_frame = _qp_row_frame(code, jacobian.shape[0], jacobian.shape[1], coordinate_bound_count, dof_count)
    item = _qp_row_item(code, jacobian.shape[0], jacobian.shape[1], coordinate_bound_count, dof_count)
    schur_diagonal = float(0.0)
    if family < 2:
        for dof in range(jacobian.shape[2]):
            value = jacobian[row_frame, item, dof]
            schur_diagonal += value * value / normal_diagonal[row_frame, dof]
    elif family < 4:
        dof = coordinate_dof_indices[item]
        schur_diagonal = 1.0 / normal_diagonal[row_frame, dof]
    else:
        schur_diagonal = 1.0 / normal_diagonal[row_frame - 1, item]
        schur_diagonal += 1.0 / normal_diagonal[row_frame, item]
    scale = 1.0 / wp.sqrt(wp.max(schur_diagonal, epsilon))
    row_scale[row, column] = scale
    row_rhs[row, column] = scale * _qp_row_rhs_value(
        code,
        residuals,
        residual_upper,
        joint_q,
        coordinate_indices,
        coordinate_lower,
        coordinate_upper,
        coordinate_bound_count,
        velocity_current,
        velocity_lower,
        velocity_upper,
        frame_segment,
        step_seconds,
        dof_count,
    )


@wp.kernel
def _qp_rows_operator_dual(
    jacobian_primal: wp.array2d(dtype=wp.float32),
    primal: wp.array2d(dtype=wp.float32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    coordinate_lower: wp.array(dtype=wp.float32),
    coordinate_upper: wp.array(dtype=wp.float32),
    coordinate_bound_count: int,
    dof_count: int,
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    row_code: wp.array2d(dtype=wp.int32),
    row_scale: wp.array2d(dtype=wp.float32),
    equality_count: wp.array(dtype=wp.int32),
    dual: wp.array2d(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float32),
):
    """Apply scaled equality rows, retaining identity blocks in unused dual slots."""
    row, column = wp.tid()
    if row >= active_frames:
        return
    width = row_code.shape[1]
    slot = row * width + column
    segment = frame_segment[row]
    if enabled[segment] == 0:
        output[row, column] = 0.0
        return
    if not _qp_slot_active(slot, segment, segment_offsets, row_code, equality_count):
        output[row, column] = dual[row, column]
        return
    code = row_code[row, column]
    family = _qp_row_family(code, jacobian_primal.shape[0], jacobian_primal.shape[1], coordinate_bound_count, dof_count)
    row_frame = _qp_row_frame(
        code, jacobian_primal.shape[0], jacobian_primal.shape[1], coordinate_bound_count, dof_count
    )
    item = _qp_row_item(code, jacobian_primal.shape[0], jacobian_primal.shape[1], coordinate_bound_count, dof_count)
    value = float(0.0)
    if family < 2:
        value = jacobian_primal[row_frame, item]
    elif family < 4:
        value = primal[row_frame, coordinate_dof_indices[item]]
        if family == 2 and coordinate_lower[item] != coordinate_upper[item]:
            value = -value
    else:
        value = primal[row_frame - 1, item] - primal[row_frame, item]
        if family == 5:
            value = -value
    output[row, column] = row_scale[row, column] * value


@wp.kernel
def _qp_rows_residual_unscale(
    values: wp.array2d(dtype=wp.float32),
    row_scale: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array2d(dtype=wp.float32),
):
    """Convert compact equality residuals back to their raw units."""
    frame, column = wp.tid()
    if frame >= active_frames:
        return
    scale = row_scale[frame, column]
    output[frame, column] = (
        values[frame, column] / scale if enabled[frame_segment[frame]] != 0 and scale != 0.0 else 0.0
    )


@wp.kernel
def _qp_rows_transpose(
    jacobian: wp.array3d(dtype=wp.float32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    coordinate_lower: wp.array(dtype=wp.float32),
    coordinate_upper: wp.array(dtype=wp.float32),
    coordinate_bound_count: int,
    dof_count: int,
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    row_code: wp.array2d(dtype=wp.int32),
    row_scale: wp.array2d(dtype=wp.float32),
    equality_count: wp.array(dtype=wp.int32),
    dual: wp.array2d(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float32),
):
    """Apply the scaled equality-row transpose without nondeterministic atomics."""
    frame, dof = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0:
        output[frame, dof] = 0.0
        return
    width = row_code.shape[1]
    begin = segment_offsets[segment]
    equality_base = begin * width
    result = float(0.0)
    # Direct scans keep accumulation deterministic across all equality families.
    for index in range(equality_count[segment]):
        slot = equality_base + index
        row = slot // width
        column = slot - row * width
        code = row_code[row, column]
        coefficient = _qp_row_coefficient(
            code,
            frame,
            dof,
            jacobian,
            coordinate_dof_indices,
            coordinate_lower,
            coordinate_upper,
            coordinate_bound_count,
            dof_count,
        )
        result += coefficient * row_scale[row, column] * dual[row, column]
    output[frame, dof] = result


@wp.kernel
def _constraint_violation_max(
    residuals: wp.array2d(dtype=wp.float32),
    residual_upper: wp.array(dtype=wp.float32),
    constraint_kind: wp.array2d(dtype=wp.uint8),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array(dtype=wp.float32),
):
    frame, residual = wp.tid()
    if frame >= active_frames or enabled[frame_segment[frame]] == 0:
        return
    kind = constraint_kind[frame, residual]
    residual_value = residuals[frame, residual]
    value = float(0.0)
    if kind != wp.uint8(0) and not wp.isfinite(residual_value):
        value = 3.402823466e38
    elif kind == wp.uint8(1):
        value = wp.abs(residual_value)
    elif kind == wp.uint8(2):
        value = wp.max(residual_value - residual_upper[residual], 0.0)
    wp.atomic_max(output, frame_segment[frame], value)


@wp.kernel
def _restoration_current_merit_max(
    constraint_rhs: wp.array2d(dtype=wp.float32),
    constraint_scale: wp.array2d(dtype=wp.float32),
    equality_rhs: wp.array2d(dtype=wp.float32),
    equality_code: wp.array2d(dtype=wp.int32),
    equality_count: wp.array(dtype=wp.int32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    inequality_width: int,
    active_frames: int,
    output: wp.array(dtype=wp.float32),
):
    """Accumulate the current D-scaled hard-constraint restoration merit."""
    frame, column = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0:
        return
    merit = float(0.0)
    if column < inequality_width and constraint_scale[frame, column] != 0.0:
        merit = wp.max(-constraint_rhs[frame, column], 0.0)
    if column < equality_code.shape[1]:
        slot = frame * equality_code.shape[1] + column
        if _qp_slot_active(slot, segment, segment_offsets, equality_code, equality_count):
            merit = wp.max(merit, wp.abs(equality_rhs[frame, column]))
    wp.atomic_max(output, segment, merit)


@wp.kernel
def _restoration_candidate_merit_max(
    residuals: wp.array2d(dtype=wp.float32),
    residual_upper: wp.array(dtype=wp.float32),
    joint_q: wp.array2d(dtype=wp.float32),
    coordinate_indices: wp.array(dtype=wp.int32),
    coordinate_lower: wp.array(dtype=wp.float32),
    coordinate_upper: wp.array(dtype=wp.float32),
    coordinate_bound_count: int,
    joint_velocity: wp.array2d(dtype=wp.float32),
    velocity_lower: wp.array(dtype=wp.float32),
    velocity_upper: wp.array(dtype=wp.float32),
    step_seconds: wp.array(dtype=wp.float32),
    constraint_scale: wp.array2d(dtype=wp.float32),
    equality_code: wp.array2d(dtype=wp.int32),
    equality_scale: wp.array2d(dtype=wp.float32),
    equality_count: wp.array(dtype=wp.int32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    inequality_width: int,
    active_frames: int,
    output: wp.array(dtype=wp.float32),
):
    """Accumulate nonlinear candidate violation in the current D-scaled row metric."""
    frame, column = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0:
        return
    residual_count = residuals.shape[1]
    coordinate_capacity = coordinate_indices.shape[0]
    dof_count = joint_velocity.shape[1]
    coordinate_lower_base = residual_count
    coordinate_upper_base = coordinate_lower_base + coordinate_capacity
    velocity_lower_base = coordinate_upper_base + coordinate_capacity
    velocity_upper_base = velocity_lower_base + dof_count
    merit = float(0.0)
    if column < inequality_width:
        scale = constraint_scale[frame, column]
        if scale != 0.0:
            row_rhs = float(0.0)
            if column < residual_count:
                row_rhs = residual_upper[column] - residuals[frame, column]
            elif column < velocity_lower_base:
                upper = column >= coordinate_upper_base
                bound = column - (coordinate_upper_base if upper else coordinate_lower_base)
                if bound < coordinate_bound_count:
                    value = joint_q[frame, coordinate_indices[bound]]
                    row_rhs = coordinate_upper[bound] - value if upper else value - coordinate_lower[bound]
            else:
                upper = column >= velocity_upper_base
                dof = column - (velocity_upper_base if upper else velocity_lower_base)
                if dof < dof_count and frame > segment_offsets[segment]:
                    value = joint_velocity[frame, dof]
                    limit = velocity_upper[dof] if upper else velocity_lower[dof]
                    dt = step_seconds[segment]
                    row_rhs = dt * (limit - value) if upper else dt * (value - limit)
            merit = wp.max(-scale * row_rhs, 0.0)
    if column < equality_code.shape[1]:
        slot = frame * equality_code.shape[1] + column
        if _qp_slot_active(slot, segment, segment_offsets, equality_code, equality_count):
            code = equality_code[frame, column]
            row_rhs = _qp_row_rhs_value(
                code,
                residuals,
                residual_upper,
                joint_q,
                coordinate_indices,
                coordinate_lower,
                coordinate_upper,
                coordinate_bound_count,
                joint_velocity,
                velocity_lower,
                velocity_upper,
                frame_segment,
                step_seconds,
                dof_count,
            )
            merit = wp.max(merit, wp.abs(equality_scale[frame, column] * row_rhs))
    if not wp.isfinite(merit):
        merit = 3.402823466e38
    wp.atomic_max(output, segment, merit)


@wp.kernel
def _constraint_feasible_rows_mark(
    residuals: wp.array2d(dtype=wp.float32),
    residual_upper: wp.array(dtype=wp.float32),
    constraint_kind: wp.array2d(dtype=wp.uint8),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    numerical_tolerance: float,
    protected_kind: wp.array2d(dtype=wp.uint8),
):
    """Protect each residual constraint that is feasible at the current iterate."""
    frame, residual = wp.tid()
    protected_kind[frame, residual] = wp.uint8(0)
    if frame >= active_frames or enabled[frame_segment[frame]] == 0:
        return
    kind = constraint_kind[frame, residual]
    residual_value = residuals[frame, residual]
    value = 3.402823466e38
    if wp.isfinite(residual_value):
        if kind == wp.uint8(1):
            value = wp.abs(residual_value)
        elif kind == wp.uint8(2):
            value = wp.max(residual_value - residual_upper[residual], 0.0)
    if kind != wp.uint8(0) and value <= numerical_tolerance:
        protected_kind[frame, residual] = kind


@wp.kernel
def _constraint_protected_violation_max(
    residuals: wp.array2d(dtype=wp.float32),
    residual_upper: wp.array(dtype=wp.float32),
    protected_kind: wp.array2d(dtype=wp.uint8),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array(dtype=wp.float32),
):
    """Measure candidate violation over rows feasible at the current iterate."""
    frame, residual = wp.tid()
    if frame >= active_frames or enabled[frame_segment[frame]] == 0:
        return
    kind = protected_kind[frame, residual]
    residual_value = residuals[frame, residual]
    value = float(0.0)
    if kind != wp.uint8(0) and not wp.isfinite(residual_value):
        value = 3.402823466e38
    elif kind == wp.uint8(1):
        value = wp.abs(residual_value)
    elif kind == wp.uint8(2):
        value = wp.max(residual_value - residual_upper[residual], 0.0)
    wp.atomic_max(output, frame_segment[frame], value)


@wp.kernel
def _precondition_primal(
    values: wp.array2d(dtype=wp.float32),
    diagonal: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array2d(dtype=wp.float32),
):
    frame, dof = wp.tid()
    if frame < active_frames:
        output[frame, dof] = values[frame, dof] / diagonal[frame, dof] if enabled[frame_segment[frame]] != 0 else 0.0


@wp.kernel
def _minres_vectors_initialize(
    right_hand_side: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_rows: int,
    solution: wp.array2d(dtype=wp.float32),
    r1: wp.array2d(dtype=wp.float32),
    r2: wp.array2d(dtype=wp.float32),
    direction_older: wp.array2d(dtype=wp.float32),
    direction_old: wp.array2d(dtype=wp.float32),
    direction: wp.array2d(dtype=wp.float32),
):
    row, column = wp.tid()
    if row >= active_rows:
        return
    active = enabled[frame_segment[row]] != 0
    value = right_hand_side[row, column] if active else 0.0
    solution[row, column] = 0.0
    r1[row, column] = value
    r2[row, column] = value
    direction_older[row, column] = 0.0
    direction_old[row, column] = 0.0
    direction[row, column] = 0.0


@wp.kernel
def _minres_initialize(
    primal_dot: wp.array(dtype=wp.float32),
    dual_dot: wp.array(dtype=wp.float32),
    kkt_enabled: wp.array(dtype=wp.int32),
    numerical_epsilon: float,
    state: wp.array2d(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    failed: wp.array(dtype=wp.int32),
):
    segment = wp.tid()
    beta_squared = primal_dot[segment] + dual_dot[segment]
    invalid = not wp.isfinite(beta_squared) or beta_squared < -numerical_epsilon
    beta = 1.0 if invalid else wp.sqrt(wp.max(beta_squared, 0.0))
    state[segment, 0] = -1.0 if invalid else beta
    state[segment, 1] = 0.0
    state[segment, 2] = beta
    state[segment, 3] = 0.0
    state[segment, 4] = 0.0
    state[segment, 5] = beta
    state[segment, 6] = -1.0
    state[segment, 7] = 0.0
    state[segment, 8] = 0.0
    state[segment, 9] = 0.0
    state[segment, 10] = 0.0
    state[segment, 11] = 1.0
    state[segment, 12] = 0.0
    state[segment, 13] = 0.0
    requested = kkt_enabled[segment] != 0
    enabled[segment] = 1 if requested and not invalid and beta > numerical_epsilon else 0
    failed[segment] = 1 if requested and invalid else 0


@wp.kernel
def _minres_basis(
    values: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    state: wp.array2d(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    active_rows: int,
    output: wp.array2d(dtype=wp.float32),
):
    row, column = wp.tid()
    if row >= active_rows:
        return
    segment = frame_segment[row]
    output[row, column] = values[row, column] / state[segment, 2] if enabled[segment] != 0 else 0.0


@wp.kernel
def _minres_previous_lanczos(
    values: wp.array2d(dtype=wp.float32),
    previous: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    state: wp.array2d(dtype=wp.float32),
    iteration: int,
    enabled: wp.array(dtype=wp.int32),
    active_rows: int,
):
    row, column = wp.tid()
    if row >= active_rows:
        return
    segment = frame_segment[row]
    if enabled[segment] != 0 and iteration > 0:
        values[row, column] -= state[segment, 2] / state[segment, 1] * previous[row, column]


@wp.kernel
def _minres_store_alfa(
    primal_dot: wp.array(dtype=wp.float32),
    dual_dot: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    state: wp.array2d(dtype=wp.float32),
):
    segment = wp.tid()
    if enabled[segment] != 0:
        state[segment, 8] = primal_dot[segment] + dual_dot[segment]


@wp.kernel
def _minres_current_lanczos(
    values: wp.array2d(dtype=wp.float32),
    current: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    state: wp.array2d(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    active_rows: int,
):
    row, column = wp.tid()
    if row >= active_rows:
        return
    segment = frame_segment[row]
    if enabled[segment] != 0:
        values[row, column] -= state[segment, 8] / state[segment, 2] * current[row, column]


@wp.kernel
def _minres_recurrence(
    primal_dot: wp.array(dtype=wp.float32),
    dual_dot: wp.array(dtype=wp.float32),
    relative_tolerance: float,
    numerical_epsilon: float,
    state: wp.array2d(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    failed: wp.array(dtype=wp.int32),
):
    segment = wp.tid()
    if enabled[segment] == 0:
        return
    beta_squared = primal_dot[segment] + dual_dot[segment]
    if state[segment, 0] < 0.0 or not wp.isfinite(beta_squared) or beta_squared < -numerical_epsilon:
        state[segment, 0] = -1.0
        state[segment, 12] = 0.0
        state[segment, 13] = 0.0
        failed[segment] = 1
        enabled[segment] = 0
        return
    beta = wp.sqrt(wp.max(beta_squared, 0.0))
    oldeps = state[segment, 4]
    delta = state[segment, 6] * state[segment, 3] + state[segment, 7] * state[segment, 8]
    gbar = state[segment, 7] * state[segment, 3] - state[segment, 6] * state[segment, 8]
    epsln = state[segment, 7] * beta
    dbar = -state[segment, 6] * beta
    gamma_squared = gbar * gbar + beta * beta
    if not wp.isfinite(gamma_squared):
        state[segment, 0] = -1.0
        state[segment, 12] = 0.0
        state[segment, 13] = 0.0
        failed[segment] = 1
        enabled[segment] = 0
        return
    gamma = wp.max(wp.sqrt(gamma_squared), numerical_epsilon)
    cs = gbar / gamma
    sn = beta / gamma
    phi = cs * state[segment, 5]
    phibar = sn * state[segment, 5]
    state[segment, 1] = state[segment, 2]
    state[segment, 2] = beta
    state[segment, 3] = dbar
    state[segment, 4] = epsln
    state[segment, 5] = phibar
    state[segment, 6] = cs
    state[segment, 7] = sn
    state[segment, 9] = oldeps
    state[segment, 10] = delta
    state[segment, 11] = gamma
    state[segment, 12] = phi
    threshold = wp.max(numerical_epsilon, relative_tolerance * state[segment, 0])
    state[segment, 13] = 1.0 if wp.isfinite(phibar) and wp.abs(phibar) <= threshold else 0.0
    if not wp.isfinite(phi) or not wp.isfinite(phibar) or (beta <= numerical_epsilon and state[segment, 13] == 0.0):
        state[segment, 0] = -1.0
        state[segment, 12] = 0.0
        state[segment, 13] = 0.0
        failed[segment] = 1
        enabled[segment] = 0


@wp.kernel
def _minres_solution_update(
    basis: wp.array2d(dtype=wp.float32),
    direction_older: wp.array2d(dtype=wp.float32),
    direction_old: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    state: wp.array2d(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    active_rows: int,
    solution: wp.array2d(dtype=wp.float32),
    direction: wp.array2d(dtype=wp.float32),
):
    row, column = wp.tid()
    if row >= active_rows:
        return
    segment = frame_segment[row]
    if enabled[segment] == 0:
        return
    value = (
        basis[row, column]
        - state[segment, 9] * direction_older[row, column]
        - state[segment, 10] * direction_old[row, column]
    ) / state[segment, 11]
    direction[row, column] = value
    solution[row, column] += state[segment, 12] * value


@wp.kernel
def _minres_finalize_iteration(
    state: wp.array2d(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
):
    segment = wp.tid()
    if state[segment, 13] != 0.0:
        enabled[segment] = 0


@wp.kernel
def _krylov_convergence_accumulate(
    requested: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    converged: wp.array(dtype=wp.int32),
):
    segment = wp.tid()
    if segment < segment_count and requested[segment] != 0 and enabled[segment] != 0:
        converged[segment] = 0


@wp.kernel
def _precision_apply(
    values: wp.array2d(dtype=wp.float32),
    base_weights: wp.array2d(dtype=wp.float32),
    temporal_weights: wp.array2d(dtype=wp.float32),
    residual_activity: wp.array2d(dtype=wp.float32),
    activity_group_by_residual: wp.array(dtype=wp.int32),
    first_difference_group_by_residual: wp.array(dtype=wp.int32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    step_seconds: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array2d(dtype=wp.float32),
):
    frame, residual = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0:
        output[frame, residual] = 0.0
        return
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    dt = step_seconds[segment]
    activity_group = activity_group_by_residual[residual]
    first_difference_group = first_difference_group_by_residual[residual]
    frame_confidence = float(1.0) if activity_group < 0 else residual_activity[frame, activity_group]
    result = frame_confidence * base_weights[frame, residual] * values[frame, residual]
    for order in range(1, 4):
        weight = temporal_weights[order - 1, residual]
        if weight != 0.0:
            inverse_dt = 1.0 / dt
            scale = weight
            for _ in range(order):
                scale = scale * inverse_dt * inverse_dt
            first_start = wp.max(begin, frame - order)
            last_start = wp.min(frame, end - order - 1)
            for start in range(first_start, last_start + 1):
                stencil_confidence = _temporal_stencil_confidence(
                    residual_activity, activity_group, first_difference_group, start, order
                )
                if stencil_confidence <= 0.0:
                    continue
                difference = float(0.0)
                for index in range(order + 1):
                    difference += _difference_coefficient(order, index) * values[start + index, residual]
                result += stencil_confidence * scale * _difference_coefficient(order, frame - start) * difference
    output[frame, residual] = result


@wp.kernel
def _precision_diagonal(
    base_weights: wp.array2d(dtype=wp.float32),
    temporal_weights: wp.array2d(dtype=wp.float32),
    residual_activity: wp.array2d(dtype=wp.float32),
    activity_group_by_residual: wp.array(dtype=wp.int32),
    first_difference_group_by_residual: wp.array(dtype=wp.int32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    step_seconds: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array2d(dtype=wp.float32),
):
    frame, residual = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0:
        output[frame, residual] = 0.0
        return
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    dt = step_seconds[segment]
    activity_group = activity_group_by_residual[residual]
    first_difference_group = first_difference_group_by_residual[residual]
    frame_confidence = float(1.0) if activity_group < 0 else residual_activity[frame, activity_group]
    result = frame_confidence * base_weights[frame, residual]
    for order in range(1, 4):
        weight = temporal_weights[order - 1, residual]
        if weight != 0.0:
            inverse_dt = 1.0 / dt
            scale = weight
            for _ in range(order):
                scale = scale * inverse_dt * inverse_dt
            first_start = wp.max(begin, frame - order)
            last_start = wp.min(frame, end - order - 1)
            for start in range(first_start, last_start + 1):
                stencil_confidence = _temporal_stencil_confidence(
                    residual_activity, activity_group, first_difference_group, start, order
                )
                if stencil_confidence <= 0.0:
                    continue
                coefficient = _difference_coefficient(order, frame - start)
                result += stencil_confidence * scale * coefficient * coefficient
    output[frame, residual] = result


@wp.kernel
def _normal_diagonal(
    jacobian: wp.array3d(dtype=wp.float32),
    precision_diagonal: wp.array2d(dtype=wp.float32),
    segment_damping: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array2d(dtype=wp.float32),
):
    frame, dof = wp.tid()
    if frame >= active_frames:
        return
    if enabled[frame_segment[frame]] == 0:
        output[frame, dof] = 1.0
        return
    result = segment_damping[frame_segment[frame]]
    for residual in range(jacobian.shape[1]):
        value = jacobian[frame, residual, dof]
        result += precision_diagonal[frame, residual] * value * value
    output[frame, dof] = wp.max(result, 1.0e-12)


@wp.kernel
def _segment_diagonal_initialize(
    segment_damping: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array2d(dtype=wp.float32),
):
    """Initialize a frame-local diagonal from its owning segment damping."""
    frame, dof = wp.tid()
    if frame < active_frames:
        output[frame, dof] = segment_damping[frame_segment[frame]]


@wp.kernel
def _segment_scaled_add(
    values: wp.array2d(dtype=wp.float32),
    segment_damping: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array2d(dtype=wp.float32),
):
    """Add a segment-scaled frame-local vector without allocating a broadcast view."""
    frame, dof = wp.tid()
    if frame < active_frames:
        output[frame, dof] += segment_damping[frame_segment[frame]] * values[frame, dof]


@wp.kernel
def _segment_scalar_scaled_add(
    values: wp.array(dtype=wp.float32),
    segment_damping: wp.array(dtype=wp.float32),
    active_segments: int,
    output: wp.array(dtype=wp.float32),
):
    """Add a segment-scaled scalar vector without allocating temporary storage."""
    segment = wp.tid()
    if segment < active_segments:
        output[segment] += segment_damping[segment] * values[segment]


@wp.kernel
def _normal_temporal_band_build(
    jacobian: wp.array3d(dtype=wp.float32),
    normal_diagonal: wp.array2d(dtype=wp.float32),
    temporal_weights: wp.array2d(dtype=wp.float32),
    residual_activity: wp.array2d(dtype=wp.float32),
    activity_group_by_residual: wp.array(dtype=wp.int32),
    first_difference_group_by_residual: wp.array(dtype=wp.int32),
    step_seconds: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    band: wp.array3d(dtype=wp.float64),
):
    """Build the exact same-DOF bandwidth-three part of a trajectory normal matrix."""
    frame, dof, lag = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    begin = segment_offsets[segment]
    if enabled[segment] == 0:
        band[frame, dof, lag] = wp.float64(1.0) if lag == 0 else wp.float64(0.0)
        return
    if lag == 0:
        band[frame, dof, 0] = wp.float64(normal_diagonal[frame, dof])
        return
    previous = frame - lag
    if previous < begin:
        band[frame, dof, lag] = wp.float64(0.0)
        return

    end = segment_offsets[segment + 1]
    dt = wp.float64(step_seconds[segment])
    result = wp.float64(0.0)
    for residual in range(jacobian.shape[1]):
        precision = wp.float64(0.0)
        activity_group = activity_group_by_residual[residual]
        first_difference_group = first_difference_group_by_residual[residual]
        for order in range(lag, 4):
            weight = temporal_weights[order - 1, residual]
            if weight == 0.0:
                continue
            temporal_scale = wp.float64(weight)
            inverse_dt = wp.float64(1.0) / dt
            for _ in range(order):
                temporal_scale *= inverse_dt * inverse_dt
            first_start = wp.max(begin, frame - order)
            last_start = wp.min(previous, end - order - 1)
            for start in range(first_start, last_start + 1):
                stencil_confidence = wp.float64(
                    _temporal_stencil_confidence(
                        residual_activity, activity_group, first_difference_group, start, order
                    )
                )
                precision += (
                    stencil_confidence
                    * temporal_scale
                    * wp.float64(_difference_coefficient(order, frame - start))
                    * wp.float64(_difference_coefficient(order, previous - start))
                )
        result += wp.float64(jacobian[frame, residual, dof]) * precision * wp.float64(jacobian[previous, residual, dof])
    band[frame, dof, lag] = result


@wp.kernel
def _pcg_initialize(
    right_hand_side: wp.array2d(dtype=wp.float32),
    diagonal: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    delta: wp.array2d(dtype=wp.float32),
    residual: wp.array2d(dtype=wp.float32),
    preconditioned: wp.array2d(dtype=wp.float32),
    direction: wp.array2d(dtype=wp.float32),
):
    frame, dof = wp.tid()
    if frame >= active_frames:
        return
    active = enabled[frame_segment[frame]] != 0
    value = right_hand_side[frame, dof] if active else 0.0
    z = value / diagonal[frame, dof] if active else 0.0
    delta[frame, dof] = 0.0
    residual[frame, dof] = value
    preconditioned[frame, dof] = z
    direction[frame, dof] = z


@wp.kernel
def _pcg_temporal_forward_f32(
    values: wp.array2d(dtype=wp.float32),
    diagonal: wp.array2d(dtype=wp.float32),
    band: wp.array3d(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    factor_failed: wp.array(dtype=wp.int32),
    segment_count: int,
    output: wp.array2d(dtype=wp.float32),
):
    """Apply a lower temporal inverse, falling back to diagonal scaling after factor failure."""
    segment, dof = wp.tid()
    if segment >= segment_count:
        return
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    active = enabled[segment] != 0
    if factor_failed[segment] != 0:
        for frame in range(begin, end):
            value = wp.float64(values[frame, dof]) if active else wp.float64(0.0)
            scale = wp.sqrt(wp.max(wp.float64(diagonal[frame, dof]), wp.float64(1.0e-24)))
            output[frame, dof] = wp.float32(value / scale)
        return
    for frame in range(begin, end):
        value = wp.float64(values[frame, dof]) if active else wp.float64(0.0)
        for lag in range(1, wp.min(3, frame - begin) + 1):
            value -= band[frame, dof, lag] * wp.float64(output[frame - lag, dof])
        output[frame, dof] = wp.float32(value / band[frame, dof, 0])


@wp.kernel
def _pcg_temporal_backward_f32(
    values: wp.array2d(dtype=wp.float32),
    diagonal: wp.array2d(dtype=wp.float32),
    band: wp.array3d(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    factor_failed: wp.array(dtype=wp.int32),
    segment_count: int,
    output: wp.array2d(dtype=wp.float32),
):
    """Apply a transposed temporal inverse, falling back to diagonal scaling after factor failure."""
    segment, dof = wp.tid()
    if segment >= segment_count:
        return
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    active = enabled[segment] != 0
    if factor_failed[segment] != 0:
        for frame in range(begin, end):
            value = wp.float64(values[frame, dof]) if active else wp.float64(0.0)
            scale = wp.sqrt(wp.max(wp.float64(diagonal[frame, dof]), wp.float64(1.0e-24)))
            output[frame, dof] = wp.float32(value / scale)
        return
    for reverse in range(end - begin):
        frame = end - reverse - 1
        value = wp.float64(values[frame, dof]) if active else wp.float64(0.0)
        for lag in range(1, wp.min(3, end - frame - 1) + 1):
            value -= band[frame + lag, dof, lag] * wp.float64(output[frame + lag, dof])
        output[frame, dof] = wp.float32(value / band[frame, dof, 0])


@wp.kernel
def _pcg_convergence_initialize(
    residual_dot: wp.array(dtype=wp.float32),
    requested: wp.array(dtype=wp.int32),
    segment_count: int,
    numerical_epsilon: float,
    state: wp.array2d(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
):
    segment = wp.tid()
    if segment >= segment_count:
        return
    residual_squared = residual_dot[segment]
    invalid = not wp.isfinite(residual_squared) or residual_squared < -numerical_epsilon
    residual_norm = 0.0 if invalid else wp.sqrt(wp.max(residual_squared, 0.0))
    state[segment, 0] = -1.0 if invalid else residual_norm
    enabled[segment] = 1 if requested[segment] != 0 and (invalid or residual_norm > numerical_epsilon) else 0


@wp.kernel
def _pcg_convergence_update(
    residual_dot: wp.array(dtype=wp.float32),
    state: wp.array2d(dtype=wp.float32),
    relative_tolerance: float,
    numerical_epsilon: float,
    enabled: wp.array(dtype=wp.int32),
):
    segment = wp.tid()
    if enabled[segment] == 0:
        return
    residual_squared = residual_dot[segment]
    if state[segment, 0] < 0.0 or not wp.isfinite(residual_squared) or residual_squared < -numerical_epsilon:
        state[segment, 0] = -1.0
        return
    residual_norm = wp.sqrt(wp.max(residual_squared, 0.0))
    threshold = wp.max(numerical_epsilon, relative_tolerance * state[segment, 0])
    if wp.isfinite(residual_norm) and residual_norm <= threshold:
        enabled[segment] = 0


@wp.kernel
def _ipm_pcg_convergence_update(
    residual_dot: wp.array(dtype=wp.float32),
    state: wp.array2d(dtype=wp.float32),
    relative_tolerance: float,
    numerical_epsilon: float,
    enabled: wp.array(dtype=wp.int32),
    recursive_converged: wp.array(dtype=wp.int32),
    failed: wp.array(dtype=wp.int32),
):
    """Measure tentative recursive convergence without stopping the physical solve."""
    segment = wp.tid()
    if enabled[segment] == 0:
        recursive_converged[segment] = 0
        return
    residual_squared = residual_dot[segment]
    if state[segment, 0] < 0.0 or not wp.isfinite(residual_squared) or residual_squared < -numerical_epsilon:
        state[segment, 0] = -1.0
        enabled[segment] = 0
        recursive_converged[segment] = 0
        failed[segment] = 1
        return
    residual_norm = wp.sqrt(wp.max(residual_squared, 0.0))
    threshold = wp.max(numerical_epsilon, relative_tolerance * state[segment, 0])
    if wp.isfinite(residual_norm) and residual_norm <= threshold:
        recursive_converged[segment] = 1


@wp.kernel
def _pcg_identity_initialize(
    right_hand_side: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    delta: wp.array2d(dtype=wp.float32),
    residual: wp.array2d(dtype=wp.float32),
    preconditioned: wp.array2d(dtype=wp.float32),
    direction: wp.array2d(dtype=wp.float32),
):
    """Initialize an unpreconditioned correction solve in transformed coordinates."""
    frame, dof = wp.tid()
    if frame >= active_frames:
        return
    value = right_hand_side[frame, dof] if enabled[frame_segment[frame]] != 0 else 0.0
    delta[frame, dof] = 0.0
    residual[frame, dof] = value
    preconditioned[frame, dof] = value
    direction[frame, dof] = value


@wp.kernel
def _line_search_stationarity_initialize(
    current_cost: wp.array(dtype=wp.float32),
    gradient_dot_step: wp.array(dtype=wp.float32),
    objective_term_count: wp.array(dtype=wp.int32),
    convergence_tolerance: float,
    constraint_violation: wp.array(dtype=wp.float32),
    numerical_tolerance: float,
    constrained: wp.uint8,
    linear_converged: wp.array(dtype=wp.bool),
    enabled: wp.array(dtype=wp.int32),
    outcome: wp.array(dtype=wp.int32),
    take_candidate: wp.array(dtype=wp.int32),
):
    """Resolve finite model-stationary segments before evaluating trial states."""
    segment = wp.tid()
    outcome[segment] = 0
    take_candidate[segment] = 0
    if enabled[segment] == 0 or not linear_converged[segment]:
        return
    current = current_cost[segment]
    model_descent = -2.0 * gradient_dot_step[segment]
    descent_floor = 0.0
    objective_count = float(wp.max(objective_term_count[segment], 1))
    stationary_threshold = 0.0
    if convergence_tolerance >= 0.0:
        descent_floor = numerical_tolerance * wp.max(wp.abs(current), 1.0)
        stationary_threshold = wp.max(stationary_threshold, convergence_tolerance * objective_count)
        stationary_threshold = wp.max(stationary_threshold, descent_floor)
    constraint_finite = constrained == wp.uint8(0) or wp.isfinite(constraint_violation[segment])
    feasible = constrained == wp.uint8(0) or constraint_violation[segment] <= numerical_tolerance
    if (
        wp.isfinite(current)
        and wp.isfinite(model_descent)
        and constraint_finite
        and feasible
        and model_descent >= -descent_floor
        and wp.max(model_descent, 0.0) <= stationary_threshold
    ):
        outcome[segment] = 1


@wp.kernel
def _line_search_decide(
    current_cost: wp.array(dtype=wp.float32),
    candidate_cost: wp.array(dtype=wp.float32),
    trial_scale: float,
    gradient_dot_step: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    outcome: wp.array(dtype=wp.int32),
    take_candidate: wp.array(dtype=wp.int32),
):
    """Accept a finite trial with sufficient objective descent."""
    segment = wp.tid()
    take_candidate[segment] = 0
    current = current_cost[segment]
    candidate = candidate_cost[segment]
    if enabled[segment] == 0 or outcome[segment] != 0:
        return
    if not wp.isfinite(current) or not wp.isfinite(candidate):
        return
    model_descent = -2.0 * gradient_dot_step[segment]
    if not wp.isfinite(model_descent):
        return
    actual_decrease = current - candidate
    required_decrease = 1.0e-4 * trial_scale * model_descent
    take = model_descent > 0.0 and actual_decrease >= required_decrease
    if take:
        outcome[segment] = 2
        take_candidate[segment] = 1


@wp.kernel
def _projected_fallback_initialize(
    right_hand_side: wp.array2d(dtype=wp.float32),
    diagonal: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    unresolved: wp.array(dtype=wp.int32),
    active_frames: int,
    residual: wp.array2d(dtype=wp.float32),
    preconditioned: wp.array2d(dtype=wp.float32),
    direction: wp.array2d(dtype=wp.float32),
    operator_direction: wp.array2d(dtype=wp.float32),
):
    """Prepare a projected-gradient fallback only for unresolved PCG segments."""
    frame, dof = wp.tid()
    if frame >= active_frames:
        return
    enabled = unresolved[frame_segment[frame]] != 0
    value = right_hand_side[frame, dof] if enabled else 0.0
    residual[frame, dof] = value
    preconditioned[frame, dof] = value / diagonal[frame, dof] if enabled else 0.0
    direction[frame, dof] = preconditioned[frame, dof]
    operator_direction[frame, dof] = 0.0


@wp.kernel
def _projected_direction_validate(
    numerator: wp.array(dtype=wp.float32),
    denominator: wp.array(dtype=wp.float32),
    unresolved: wp.array(dtype=wp.int32),
    requested: wp.array(dtype=wp.int32),
    segment_count: int,
    numerical_tolerance: float,
    valid: wp.array(dtype=wp.int32),
):
    """Accept converged PCG or a finite curvature-scaled projected-gradient fallback."""
    segment = wp.tid()
    if segment >= segment_count:
        return
    if requested[segment] == 0:
        valid[segment] = 1
        return
    if unresolved[segment] == 0:
        valid[segment] = 1
        return
    top = numerator[segment]
    bottom = denominator[segment]
    valid[segment] = (
        1 if (wp.isfinite(top) and top > numerical_tolerance and wp.isfinite(bottom) and bottom > 1.0e-20) else 0
    )


@wp.kernel
def _projected_fallback_apply(
    direction: wp.array2d(dtype=wp.float32),
    numerator: wp.array(dtype=wp.float32),
    denominator: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    unresolved: wp.array(dtype=wp.int32),
    valid: wp.array(dtype=wp.int32),
    active_frames: int,
    delta: wp.array2d(dtype=wp.float32),
):
    """Replace unresolved PCG steps with their projected-gradient fallback."""
    frame, dof = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if unresolved[segment] != 0:
        scale = numerator[segment] / denominator[segment] if valid[segment] != 0 else 0.0
        delta[frame, dof] = scale * direction[frame, dof]


@wp.kernel
def _projected_line_search_decide(
    current_cost: wp.array(dtype=wp.float32),
    candidate_cost: wp.array(dtype=wp.float32),
    gradient_dot_candidate_step: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    outcome: wp.array(dtype=wp.int32),
    take_candidate: wp.array(dtype=wp.int32),
):
    """Accept a projected trial using its actual bounded tangent step."""
    segment = wp.tid()
    take_candidate[segment] = 0
    current = current_cost[segment]
    candidate = candidate_cost[segment]
    if enabled[segment] == 0 or outcome[segment] != 0:
        return
    if not wp.isfinite(current) or not wp.isfinite(candidate):
        return
    model_descent = -2.0 * gradient_dot_candidate_step[segment]
    if not wp.isfinite(model_descent):
        return
    actual_decrease = current - candidate
    required_decrease = 1.0e-4 * model_descent
    if model_descent > 0.0 and actual_decrease >= required_decrease:
        outcome[segment] = 2
        take_candidate[segment] = 1


@wp.kernel
def _constraint_line_search_decide(
    current_cost: wp.array(dtype=wp.float32),
    candidate_cost: wp.array(dtype=wp.float32),
    current_violation: wp.array(dtype=wp.float32),
    candidate_violation: wp.array(dtype=wp.float32),
    candidate_bound_violation: wp.array(dtype=wp.float32),
    candidate_protected_violation: wp.array(dtype=wp.float32),
    current_restoration_merit: wp.array(dtype=wp.float32),
    candidate_restoration_merit: wp.array(dtype=wp.float32),
    numerical_tolerance: float,
    trial_scale: float,
    gradient_dot_step: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    outcome: wp.array(dtype=wp.int32),
    take_candidate: wp.array(dtype=wp.int32),
):
    segment = wp.tid()
    take_candidate[segment] = 0
    if enabled[segment] == 0 or outcome[segment] != 0:
        return
    current = current_cost[segment]
    candidate = candidate_cost[segment]
    if not wp.isfinite(current) or not wp.isfinite(candidate):
        return
    model_descent = -2.0 * gradient_dot_step[segment]
    current_constraint = current_violation[segment]
    candidate_bound = candidate_bound_violation[segment]
    candidate_constraint = wp.max(candidate_violation[segment], candidate_bound)
    candidate_protected = candidate_protected_violation[segment]
    current_restoration = current_restoration_merit[segment]
    candidate_restoration = candidate_restoration_merit[segment]
    if (
        not wp.isfinite(model_descent)
        or not wp.isfinite(current_constraint)
        or not wp.isfinite(candidate_constraint)
        or not wp.isfinite(candidate_bound)
        or not wp.isfinite(candidate_protected)
        or not wp.isfinite(current_restoration)
        or not wp.isfinite(candidate_restoration)
    ):
        return
    current_feasible = current_constraint <= numerical_tolerance
    candidate_protected_feasible = candidate_protected <= numerical_tolerance
    candidate_feasible = candidate_constraint <= numerical_tolerance and candidate_protected_feasible
    take = bool(False)
    if not current_feasible:
        required_restoration = 1.0e-4 * trial_scale * wp.max(current_restoration, numerical_tolerance)
        take = current_restoration - candidate_restoration >= required_restoration
    elif candidate_feasible:
        actual_decrease = current - candidate
        required_decrease = 1.0e-4 * trial_scale * model_descent
        take = model_descent > 0.0 and actual_decrease >= required_decrease
    if take:
        outcome[segment] = 2
        take_candidate[segment] = 1


@wp.kernel
def _second_order_correction_request(
    current_cost: wp.array(dtype=wp.float32),
    candidate_cost: wp.array(dtype=wp.float32),
    current_violation: wp.array(dtype=wp.float32),
    candidate_violation: wp.array(dtype=wp.float32),
    candidate_bound_violation: wp.array(dtype=wp.float32),
    candidate_protected_violation: wp.array(dtype=wp.float32),
    current_restoration_merit: wp.array(dtype=wp.float32),
    candidate_restoration_merit: wp.array(dtype=wp.float32),
    trial_scale: float,
    gradient_dot_step: wp.array(dtype=wp.float32),
    numerical_tolerance: float,
    linear_converged: wp.array(dtype=wp.bool),
    segment_feasible: wp.array(dtype=wp.uint8),
    enabled: wp.array(dtype=wp.int32),
    outcome: wp.array(dtype=wp.int32),
    requested: wp.array(dtype=wp.int32),
):
    """Request curvature correction for blocked feasibility or restoration."""
    segment = wp.tid()
    requested[segment] = 0
    if (
        enabled[segment] == 0
        or outcome[segment] != 0
        or not linear_converged[segment]
        or segment_feasible[segment] == wp.uint8(0)
    ):
        return
    current = current_cost[segment]
    candidate = candidate_cost[segment]
    current_constraint = current_violation[segment]
    candidate_bound = candidate_bound_violation[segment]
    candidate_constraint = wp.max(candidate_violation[segment], candidate_bound)
    candidate_protected = candidate_protected_violation[segment]
    current_restoration = current_restoration_merit[segment]
    candidate_restoration = candidate_restoration_merit[segment]
    model_descent = -2.0 * gradient_dot_step[segment]
    if (
        not wp.isfinite(current)
        or not wp.isfinite(candidate)
        or not wp.isfinite(current_constraint)
        or not wp.isfinite(candidate_constraint)
        or not wp.isfinite(candidate_bound)
        or not wp.isfinite(candidate_protected)
        or not wp.isfinite(current_restoration)
        or not wp.isfinite(candidate_restoration)
        or not wp.isfinite(model_descent)
    ):
        return
    objective_descent = model_descent > 0.0 and current - candidate >= 1.0e-4 * trial_scale * model_descent
    current_feasible = current_constraint <= numerical_tolerance
    required_restoration = 1.0e-4 * trial_scale * wp.max(current_restoration, numerical_tolerance)
    restoration_missing = current_restoration - candidate_restoration < required_restoration
    feasible_correction = current_feasible and candidate_constraint > numerical_tolerance and objective_descent
    restoration_correction = not current_feasible and restoration_missing
    protected_correction = current_feasible and candidate_protected > numerical_tolerance and objective_descent
    if feasible_correction or restoration_correction or protected_correction:
        requested[segment] = 1


@wp.kernel
def _second_order_correction_decide(
    current_cost: wp.array(dtype=wp.float32),
    candidate_cost: wp.array(dtype=wp.float32),
    candidate_violation: wp.array(dtype=wp.float32),
    candidate_bound_violation: wp.array(dtype=wp.float32),
    candidate_protected_violation: wp.array(dtype=wp.float32),
    candidate_restoration_merit: wp.array(dtype=wp.float32),
    trial_scale: float,
    gradient_dot_step: wp.array(dtype=wp.float32),
    numerical_tolerance: float,
    correction_linear_converged: wp.array(dtype=wp.int32),
    correction_feasible: wp.array(dtype=wp.uint8),
    requested: wp.array(dtype=wp.int32),
    previous_candidate_state: wp.array2d(dtype=wp.float32),
    outcome: wp.array(dtype=wp.int32),
    take_candidate: wp.array(dtype=wp.int32),
    continue_correction: wp.array(dtype=wp.int32),
):
    """Accept corrected feasibility or restoration, or request one more correction."""
    segment = wp.tid()
    take_candidate[segment] = 0
    continue_correction[segment] = 0
    if requested[segment] == 0:
        return
    correction_solved = correction_linear_converged[segment] != 0 and correction_feasible[segment] != wp.uint8(0)
    correction_feasible[segment] = wp.uint8(1)
    if not correction_solved:
        return
    current = current_cost[segment]
    candidate = candidate_cost[segment]
    candidate_bound = candidate_bound_violation[segment]
    candidate_constraint = wp.max(candidate_violation[segment], candidate_bound)
    candidate_protected = candidate_protected_violation[segment]
    current_constraint = previous_candidate_state[segment, 1]
    previous_constraint = previous_candidate_state[segment, 2]
    previous_protected = previous_candidate_state[segment, 3]
    current_restoration = previous_candidate_state[segment, 4]
    previous_restoration = previous_candidate_state[segment, 5]
    candidate_restoration = candidate_restoration_merit[segment]
    model_descent = -2.0 * gradient_dot_step[segment]
    if (
        not wp.isfinite(current)
        or not wp.isfinite(candidate)
        or not wp.isfinite(candidate_constraint)
        or not wp.isfinite(candidate_bound)
        or not wp.isfinite(candidate_protected)
        or not wp.isfinite(current_constraint)
        or not wp.isfinite(previous_constraint)
        or not wp.isfinite(previous_protected)
        or not wp.isfinite(current_restoration)
        or not wp.isfinite(previous_restoration)
        or not wp.isfinite(candidate_restoration)
        or not wp.isfinite(model_descent)
    ):
        return
    actual_decrease = current - candidate
    required_decrease = 1.0e-4 * trial_scale * model_descent
    objective_descent = model_descent > 0.0 and actual_decrease >= required_decrease
    current_feasible = current_constraint <= numerical_tolerance
    protected_feasible = candidate_protected <= numerical_tolerance
    required_restoration = 1.0e-4 * trial_scale * wp.max(current_restoration, numerical_tolerance)
    restored = current_restoration - candidate_restoration >= required_restoration
    accept = bool(False)
    if current_feasible:
        accept = objective_descent and candidate_constraint <= numerical_tolerance and protected_feasible
    else:
        accept = restored
    if accept:
        outcome[segment] = 3
        take_candidate[segment] = 1
    elif (
        (candidate_constraint > numerical_tolerance or (current_feasible and not protected_feasible))
        and (
            (not current_feasible and candidate_restoration < previous_restoration)
            or (current_feasible and candidate_constraint < previous_constraint)
            or (current_feasible and candidate_protected < previous_protected)
        )
        and (objective_descent or not current_feasible)
    ):
        continue_correction[segment] = 1


@wp.kernel
def _line_search_pending(
    enabled: wp.array(dtype=wp.int32),
    outcome: wp.array(dtype=wp.int32),
    segment_count: int,
    pending: wp.array(dtype=wp.int32),
):
    """Mark whether any enabled segment still needs a line-search trial."""
    segment = wp.tid()
    if segment < segment_count and enabled[segment] != 0 and outcome[segment] == 0:
        wp.atomic_max(pending, 0, 1)


@wp.kernel
def _segments_pending(
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    pending: wp.array(dtype=wp.int32),
):
    """Mark whether any segment remains enabled."""
    segment = wp.tid()
    if segment < segment_count and enabled[segment] != 0:
        wp.atomic_max(pending, 0, 1)


@wp.kernel
def _accepted_violation_update(
    candidate_violation: wp.array(dtype=wp.float32),
    take_candidate: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    accepted_violation: wp.array(dtype=wp.float32),
):
    segment = wp.tid()
    if enabled[segment] != 0 and take_candidate[segment] != 0:
        accepted_violation[segment] = candidate_violation[segment]


@wp.kernel
def _segment_convergence_update(
    current_cost: wp.array(dtype=wp.float32),
    objective_term_count: wp.array(dtype=wp.int32),
    segment_count: int,
    tolerance: float,
    constraint_violation: wp.array(dtype=wp.float32),
    constraint_tolerance: float,
    constrained: wp.uint8,
    line_search_outcome: wp.array(dtype=wp.int32),
    gradient_dot_step: wp.array(dtype=wp.float32),
    restoration_merit: wp.array(dtype=wp.float32),
    accepted_restoration_merit: wp.array(dtype=wp.float32),
    linear_converged: wp.array(dtype=wp.bool),
    globalization_succeeded: wp.array(dtype=wp.bool),
    restoration_stalled: wp.array(dtype=wp.bool),
    enabled: wp.array(dtype=wp.int32),
):
    """Resolve objective stationarity, restoration stall, or globalization failure."""
    segment = wp.tid()
    if segment >= segment_count:
        return
    restoration_stalled[segment] = False
    if enabled[segment] == 0:
        return
    current = current_cost[segment]
    model_descent = -2.0 * gradient_dot_step[segment]
    descent_floor = 0.0
    objective_count = float(wp.max(objective_term_count[segment], 1))
    stationary_threshold = 0.0
    if tolerance >= 0.0:
        descent_floor = constraint_tolerance * wp.max(wp.abs(current), 1.0)
        stationary_threshold = wp.max(stationary_threshold, tolerance * objective_count)
        stationary_threshold = wp.max(stationary_threshold, descent_floor)
    constraint_finite = constrained == wp.uint8(0) or wp.isfinite(constraint_violation[segment])
    feasible = constrained == wp.uint8(0) or constraint_violation[segment] <= constraint_tolerance
    finite_model = wp.isfinite(current) and wp.isfinite(model_descent) and constraint_finite
    stationary = (
        finite_model
        and feasible
        and model_descent >= -descent_floor
        and wp.max(model_descent, 0.0) <= stationary_threshold
    )
    outcome = line_search_outcome[segment]
    current_restoration = restoration_merit[segment]
    accepted_restoration = accepted_restoration_merit[segment]
    restoration_progress = current_restoration - accepted_restoration
    restoration_threshold = wp.max(tolerance, constraint_tolerance) * wp.max(current_restoration, constraint_tolerance)
    stalled = (
        tolerance >= 0.0
        and constrained != wp.uint8(0)
        and not feasible
        and (outcome == 2 or outcome == 3)
        and wp.isfinite(current_restoration)
        and wp.isfinite(accepted_restoration)
        and restoration_progress >= 0.0
        and restoration_progress <= restoration_threshold
    )
    if stalled:
        restoration_stalled[segment] = True
        globalization_succeeded[segment] = False
        enabled[segment] = 0
        return
    if not linear_converged[segment]:
        if tolerance >= 0.0:
            enabled[segment] = 0
    elif not finite_model or outcome == 0 or (outcome == 1 and not stationary):
        globalization_succeeded[segment] = False
        if tolerance >= 0.0:
            enabled[segment] = 0
    elif outcome == 1 and tolerance >= 0.0:
        enabled[segment] = 0


@wp.kernel
def _segment_step_max(
    delta: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array(dtype=wp.float32),
):
    frame, dof = wp.tid()
    if frame < active_frames and enabled[frame_segment[frame]] != 0:
        value = wp.abs(delta[frame, dof])
        wp.atomic_max(output, frame_segment[frame], value if wp.isfinite(value) else wp.float32(wp.inf))


@wp.kernel
def _objective_term_count(
    base_weights: wp.array2d(dtype=wp.float32),
    temporal_weights: wp.array2d(dtype=wp.float32),
    residual_activity: wp.array2d(dtype=wp.float32),
    activity_group_by_residual: wp.array(dtype=wp.int32),
    first_difference_group_by_residual: wp.array(dtype=wp.int32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array(dtype=wp.int32),
):
    """Count scalar objective terms without counting zero-weight helper rows."""
    frame, residual = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0:
        return
    group = activity_group_by_residual[residual]
    first_difference_group = first_difference_group_by_residual[residual]
    confidence = float(1.0) if group < 0 else residual_activity[frame, group]
    count = 1 if base_weights[frame, residual] > 0.0 and confidence > 0.0 else 0
    end = segment_offsets[segment + 1]
    for order in range(1, 4):
        if temporal_weights[order - 1, residual] <= 0.0 or frame + order >= end:
            continue
        stencil_confidence = _temporal_stencil_confidence(
            residual_activity, group, first_difference_group, frame, order
        )
        if stencil_confidence > 0.0:
            count += 1
    if count > 0:
        wp.atomic_add(output, segment, count)


@wp.kernel
def _conditional_configuration_copy(
    candidate: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    take_candidate: wp.array(dtype=wp.int32),
    active_frames: int,
    current: wp.array2d(dtype=wp.float32),
):
    frame, coordinate = wp.tid()
    if frame < active_frames and enabled[frame_segment[frame]] != 0 and take_candidate[frame_segment[frame]] == 1:
        current[frame, coordinate] = candidate[frame, coordinate]


@wp.kernel
def _delta_scale(
    delta: wp.array2d(dtype=wp.float32),
    scale: float,
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array2d(dtype=wp.float32),
):
    frame, dof = wp.tid()
    if frame < active_frames:
        output[frame, dof] = scale * delta[frame, dof] if enabled[frame_segment[frame]] != 0 else 0.0


@wp.kernel
def _velocity_candidate(
    delta: wp.array2d(dtype=wp.float32),
    velocity_current: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    step_seconds: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array2d(dtype=wp.float32),
):
    frame, dof = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0:
        output[frame, dof] = velocity_current[frame, dof]
        return
    if frame == segment_offsets[segment]:
        output[frame, dof] = velocity_current[frame, dof]
    else:
        output[frame, dof] = (
            velocity_current[frame, dof] + (delta[frame, dof] - delta[frame - 1, dof]) / step_seconds[segment]
        )


@wp.kernel
def _ipm_constraints_initialize(
    jacobian: wp.array3d(dtype=wp.float32),
    residuals: wp.array2d(dtype=wp.float32),
    residual_upper: wp.array(dtype=wp.float32),
    constraint_kind: wp.array2d(dtype=wp.uint8),
    joint_q: wp.array2d(dtype=wp.float32),
    coordinate_indices: wp.array(dtype=wp.int32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    coordinate_lower: wp.array(dtype=wp.float32),
    coordinate_upper: wp.array(dtype=wp.float32),
    coordinate_bound_count: int,
    joint_velocity: wp.array2d(dtype=wp.float32),
    velocity_lower: wp.array(dtype=wp.float32),
    velocity_upper: wp.array(dtype=wp.float32),
    step_seconds: wp.array(dtype=wp.float32),
    normal_diagonal: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    numerical_epsilon: float,
    scale: wp.array2d(dtype=wp.float32),
    rhs: wp.array2d(dtype=wp.float32),
    inequality_count: wp.array(dtype=wp.int32),
):
    """Initialize fixed family slots for every declared inequality."""
    frame, column = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    residual_count = residuals.shape[1]
    coordinate_capacity = coordinate_indices.shape[0]
    dof_count = joint_velocity.shape[1]
    coordinate_lower_base = residual_count
    coordinate_upper_base = coordinate_lower_base + coordinate_capacity
    velocity_lower_base = coordinate_upper_base + coordinate_capacity
    velocity_upper_base = velocity_lower_base + dof_count
    if column >= velocity_upper_base + dof_count:
        scale[frame, column] = 0.0
        rhs[frame, column] = 0.0
        return
    active = enabled[segment] != 0
    row_rhs = float(0.0)
    schur_diagonal = float(0.0)
    if column < residual_count:
        active = active and constraint_kind[frame, column] == wp.uint8(2)
        if active:
            row_rhs = residual_upper[column] - residuals[frame, column]
            for dof in range(dof_count):
                value = jacobian[frame, column, dof]
                schur_diagonal += value * value / normal_diagonal[frame, dof]
    elif column < velocity_lower_base:
        upper = column >= coordinate_upper_base
        bound = column - (coordinate_upper_base if upper else coordinate_lower_base)
        active = active and bound < coordinate_bound_count
        if active:
            lower_value = coordinate_lower[bound]
            upper_value = coordinate_upper[bound]
            active = wp.isfinite(upper_value if upper else lower_value)
        if active:
            value = joint_q[frame, coordinate_indices[bound]]
            row_rhs = upper_value - value if upper else value - lower_value
            schur_diagonal = 1.0 / normal_diagonal[frame, coordinate_dof_indices[bound]]
    else:
        upper = column >= velocity_upper_base
        dof = column - (velocity_upper_base if upper else velocity_lower_base)
        active = active and dof < dof_count and frame > segment_offsets[segment]
        if active:
            limit = velocity_upper[dof] if upper else velocity_lower[dof]
            active = wp.isfinite(limit)
        if active:
            dt = step_seconds[segment]
            current = joint_velocity[frame, dof]
            row_rhs = dt * (limit - current) if upper else dt * (current - limit)
            schur_diagonal = 1.0 / normal_diagonal[frame - 1, dof]
            schur_diagonal += 1.0 / normal_diagonal[frame, dof]
    if active:
        row_scale = 1.0 / wp.sqrt(wp.max(schur_diagonal, numerical_epsilon))
        scale[frame, column] = row_scale
        rhs[frame, column] = row_scale * row_rhs
        wp.atomic_add(inequality_count, segment, 1)
    else:
        scale[frame, column] = 0.0
        rhs[frame, column] = 0.0


@wp.kernel
def _ipm_constraints_apply(
    jacobian_primal: wp.array2d(dtype=wp.float32),
    primal: wp.array2d(dtype=wp.float32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    scale: wp.array2d(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float32),
):
    """Apply the fixed residual, coordinate, and velocity inequality families."""
    frame, column = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    row_scale = scale[frame, column]
    if enabled[segment] == 0 or row_scale == 0.0:
        output[frame, column] = 0.0
        return
    residual_count = jacobian_primal.shape[1]
    coordinate_capacity = coordinate_dof_indices.shape[0]
    dof_count = primal.shape[1]
    coordinate_lower_base = residual_count
    coordinate_upper_base = coordinate_lower_base + coordinate_capacity
    velocity_lower_base = coordinate_upper_base + coordinate_capacity
    velocity_upper_base = velocity_lower_base + dof_count
    if column >= velocity_upper_base + dof_count:
        output[frame, column] = 0.0
        return
    value = float(0.0)
    if column < residual_count:
        value = jacobian_primal[frame, column]
    elif column < velocity_lower_base:
        upper = column >= coordinate_upper_base
        bound = column - (coordinate_upper_base if upper else coordinate_lower_base)
        value = primal[frame, coordinate_dof_indices[bound]]
        if not upper:
            value = -value
    else:
        upper = column >= velocity_upper_base
        dof = column - (velocity_upper_base if upper else velocity_lower_base)
        value = primal[frame, dof] - primal[frame - 1, dof]
        if not upper:
            value = -value
    output[frame, column] = row_scale * value


@wp.kernel
def _ipm_constraints_apply_f64(
    jacobian_primal: wp.array2d(dtype=wp.float64),
    primal: wp.array2d(dtype=wp.float64),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    scale: wp.array2d(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float64),
):
    """Apply the Phase-II inequality Jacobian to a float64 direction."""
    frame, column = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    row_scale = scale[frame, column]
    if enabled[segment] == 0 or row_scale == 0.0:
        output[frame, column] = wp.float64(0.0)
        return
    residual_count = jacobian_primal.shape[1]
    coordinate_capacity = coordinate_dof_indices.shape[0]
    dof_count = primal.shape[1]
    coordinate_lower_base = residual_count
    coordinate_upper_base = coordinate_lower_base + coordinate_capacity
    velocity_lower_base = coordinate_upper_base + coordinate_capacity
    velocity_upper_base = velocity_lower_base + dof_count
    if column >= velocity_upper_base + dof_count:
        output[frame, column] = wp.float64(0.0)
        return
    value = wp.float64(0.0)
    if column < residual_count:
        value = jacobian_primal[frame, column]
    elif column < velocity_lower_base:
        upper = column >= coordinate_upper_base
        bound = column - (coordinate_upper_base if upper else coordinate_lower_base)
        value = primal[frame, coordinate_dof_indices[bound]]
        if not upper:
            value = -value
    else:
        upper = column >= velocity_upper_base
        dof = column - (velocity_upper_base if upper else velocity_lower_base)
        value = primal[frame, dof] - primal[frame - 1, dof]
        if not upper:
            value = -value
    output[frame, column] = wp.float64(row_scale) * value


@wp.kernel
def _ipm_rows_narrow_f32(
    values: wp.array2d(dtype=wp.float64),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array2d(dtype=wp.float32),
):
    """Narrow active float64 inequality rows to their float32 residual representation."""
    frame, column = wp.tid()
    if frame < active_frames:
        output[frame, column] = wp.float32(values[frame, column]) if enabled[frame_segment[frame]] != 0 else 0.0


@wp.kernel
def _ipm_constraints_transpose(
    jacobian: wp.array3d(dtype=wp.float32),
    values: wp.array2d(dtype=wp.float32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    coordinate_bound_count: int,
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    scale: wp.array2d(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float32),
):
    """Apply the transpose of every fixed inequality family deterministically."""
    frame, dof = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0:
        output[frame, dof] = 0.0
        return
    residual_count = jacobian.shape[1]
    coordinate_capacity = coordinate_dof_indices.shape[0]
    dof_count = jacobian.shape[2]
    coordinate_lower_base = residual_count
    coordinate_upper_base = coordinate_lower_base + coordinate_capacity
    velocity_lower_base = coordinate_upper_base + coordinate_capacity
    velocity_upper_base = velocity_lower_base + dof_count
    result = float(0.0)
    for residual in range(residual_count):
        result += jacobian[frame, residual, dof] * scale[frame, residual] * values[frame, residual]
    for bound in range(coordinate_bound_count):
        if coordinate_dof_indices[bound] == dof:
            lower_column = coordinate_lower_base + bound
            upper_column = coordinate_upper_base + bound
            result -= scale[frame, lower_column] * values[frame, lower_column]
            result += scale[frame, upper_column] * values[frame, upper_column]
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    if frame > begin:
        lower_column = velocity_lower_base + dof
        upper_column = velocity_upper_base + dof
        result -= scale[frame, lower_column] * values[frame, lower_column]
        result += scale[frame, upper_column] * values[frame, upper_column]
    if frame + 1 < end:
        lower_column = velocity_lower_base + dof
        upper_column = velocity_upper_base + dof
        result += scale[frame + 1, lower_column] * values[frame + 1, lower_column]
        result -= scale[frame + 1, upper_column] * values[frame + 1, upper_column]
    output[frame, dof] = result


@wp.kernel
def _ipm_constraints_transpose_f64(
    jacobian: wp.array3d(dtype=wp.float32),
    values: wp.array2d(dtype=wp.float32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    coordinate_bound_count: int,
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    scale: wp.array2d(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float64),
):
    """Apply the inequality transpose with float64 accumulation."""
    frame, dof = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0:
        output[frame, dof] = wp.float64(0.0)
        return
    residual_count = jacobian.shape[1]
    coordinate_capacity = coordinate_dof_indices.shape[0]
    dof_count = jacobian.shape[2]
    coordinate_lower_base = residual_count
    coordinate_upper_base = coordinate_lower_base + coordinate_capacity
    velocity_lower_base = coordinate_upper_base + coordinate_capacity
    velocity_upper_base = velocity_lower_base + dof_count
    result = wp.float64(0.0)
    for residual in range(residual_count):
        result += (
            wp.float64(jacobian[frame, residual, dof])
            * wp.float64(scale[frame, residual])
            * wp.float64(values[frame, residual])
        )
    for bound in range(coordinate_bound_count):
        if coordinate_dof_indices[bound] == dof:
            lower_column = coordinate_lower_base + bound
            upper_column = coordinate_upper_base + bound
            result -= wp.float64(scale[frame, lower_column]) * wp.float64(values[frame, lower_column])
            result += wp.float64(scale[frame, upper_column]) * wp.float64(values[frame, upper_column])
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    if frame > begin:
        lower_column = velocity_lower_base + dof
        upper_column = velocity_upper_base + dof
        result -= wp.float64(scale[frame, lower_column]) * wp.float64(values[frame, lower_column])
        result += wp.float64(scale[frame, upper_column]) * wp.float64(values[frame, upper_column])
    if frame + 1 < end:
        lower_column = velocity_lower_base + dof
        upper_column = velocity_upper_base + dof
        result += wp.float64(scale[frame + 1, lower_column]) * wp.float64(values[frame + 1, lower_column])
        result -= wp.float64(scale[frame + 1, upper_column]) * wp.float64(values[frame + 1, upper_column])
    output[frame, dof] = result


@wp.kernel
def _ipm_normal_diagonal_add(
    jacobian: wp.array3d(dtype=wp.float32),
    weights: wp.array2d(dtype=wp.float32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    coordinate_bound_count: int,
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    scale: wp.array2d(dtype=wp.float32),
    diagonal: wp.array2d(dtype=wp.float32),
):
    """Add the diagonal of C-transpose W C to the primal preconditioner."""
    frame, dof = wp.tid()
    if frame >= active_frames or enabled[frame_segment[frame]] == 0:
        return
    segment = frame_segment[frame]
    residual_count = jacobian.shape[1]
    coordinate_capacity = coordinate_dof_indices.shape[0]
    dof_count = jacobian.shape[2]
    coordinate_lower_base = residual_count
    coordinate_upper_base = coordinate_lower_base + coordinate_capacity
    velocity_lower_base = coordinate_upper_base + coordinate_capacity
    velocity_upper_base = velocity_lower_base + dof_count
    result = diagonal[frame, dof]
    for residual in range(residual_count):
        coefficient = scale[frame, residual] * jacobian[frame, residual, dof]
        result += weights[frame, residual] * coefficient * coefficient
    for bound in range(coordinate_bound_count):
        if coordinate_dof_indices[bound] == dof:
            lower_column = coordinate_lower_base + bound
            upper_column = coordinate_upper_base + bound
            result += weights[frame, lower_column] * scale[frame, lower_column] * scale[frame, lower_column]
            result += weights[frame, upper_column] * scale[frame, upper_column] * scale[frame, upper_column]
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    if frame > begin:
        lower_column = velocity_lower_base + dof
        upper_column = velocity_upper_base + dof
        result += weights[frame, lower_column] * scale[frame, lower_column] * scale[frame, lower_column]
        result += weights[frame, upper_column] * scale[frame, upper_column] * scale[frame, upper_column]
    if frame + 1 < end:
        lower_column = velocity_lower_base + dof
        upper_column = velocity_upper_base + dof
        result += weights[frame + 1, lower_column] * scale[frame + 1, lower_column] * scale[frame + 1, lower_column]
        result += weights[frame + 1, upper_column] * scale[frame + 1, upper_column] * scale[frame + 1, upper_column]
    diagonal[frame, dof] = wp.max(result, 1.0e-12)


@wp.kernel
def _ipm_temporal_band_build(
    jacobian: wp.array3d(dtype=wp.float32),
    precision_diagonal: wp.array2d(dtype=wp.float32),
    temporal_weights: wp.array2d(dtype=wp.float32),
    residual_activity: wp.array2d(dtype=wp.float32),
    activity_group_by_residual: wp.array(dtype=wp.int32),
    first_difference_group_by_residual: wp.array(dtype=wp.int32),
    step_seconds: wp.array(dtype=wp.float32),
    weights: wp.array2d(dtype=wp.float32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    coordinate_bound_count: int,
    constraint_scale: wp.array2d(dtype=wp.float32),
    segment_damping: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    band: wp.array3d(dtype=wp.float64),
):
    """Build exact same-DOF temporal bands of the condensed Phase-II matrix."""
    frame, dof, lag = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    if enabled[segment] == 0:
        band[frame, dof, lag] = wp.float64(1.0) if lag == 0 else wp.float64(0.0)
        return
    residual_count = jacobian.shape[1]
    coordinate_capacity = coordinate_dof_indices.shape[0]
    dof_count = jacobian.shape[2]
    coordinate_lower_base = residual_count
    coordinate_upper_base = coordinate_lower_base + coordinate_capacity
    velocity_lower_base = coordinate_upper_base + coordinate_capacity
    velocity_upper_base = velocity_lower_base + dof_count
    if lag == 0:
        result = wp.float64(segment_damping[segment])
        for residual in range(residual_count):
            value = wp.float64(jacobian[frame, residual, dof])
            result += wp.float64(precision_diagonal[frame, residual]) * value * value
            coefficient = wp.float64(constraint_scale[frame, residual]) * value
            result += wp.float64(weights[frame, residual]) * coefficient * coefficient
        for bound in range(coordinate_bound_count):
            if coordinate_dof_indices[bound] == dof:
                lower_column = coordinate_lower_base + bound
                upper_column = coordinate_upper_base + bound
                lower_scale = wp.float64(constraint_scale[frame, lower_column])
                upper_scale = wp.float64(constraint_scale[frame, upper_column])
                result += wp.float64(weights[frame, lower_column]) * lower_scale * lower_scale
                result += wp.float64(weights[frame, upper_column]) * upper_scale * upper_scale
        if frame > begin:
            lower_scale = wp.float64(constraint_scale[frame, velocity_lower_base + dof])
            upper_scale = wp.float64(constraint_scale[frame, velocity_upper_base + dof])
            result += wp.float64(weights[frame, velocity_lower_base + dof]) * lower_scale * lower_scale
            result += wp.float64(weights[frame, velocity_upper_base + dof]) * upper_scale * upper_scale
        if frame + 1 < end:
            lower_scale = wp.float64(constraint_scale[frame + 1, velocity_lower_base + dof])
            upper_scale = wp.float64(constraint_scale[frame + 1, velocity_upper_base + dof])
            result += wp.float64(weights[frame + 1, velocity_lower_base + dof]) * lower_scale * lower_scale
            result += wp.float64(weights[frame + 1, velocity_upper_base + dof]) * upper_scale * upper_scale
        band[frame, dof, 0] = wp.max(result, wp.float64(1.0e-24))
        return
    previous = frame - lag
    if previous < begin:
        band[frame, dof, lag] = wp.float64(0.0)
        return
    dt = wp.float64(step_seconds[segment])
    result = wp.float64(0.0)
    for residual in range(residual_count):
        precision = wp.float64(0.0)
        activity_group = activity_group_by_residual[residual]
        first_difference_group = first_difference_group_by_residual[residual]
        for order in range(lag, 4):
            weight = temporal_weights[order - 1, residual]
            if weight == 0.0:
                continue
            temporal_scale = wp.float64(weight)
            inverse_dt = wp.float64(1.0) / dt
            for _ in range(order):
                temporal_scale *= inverse_dt * inverse_dt
            first_start = wp.max(begin, frame - order)
            last_start = wp.min(previous, end - order - 1)
            for start in range(first_start, last_start + 1):
                stencil_confidence = wp.float64(
                    _temporal_stencil_confidence(
                        residual_activity, activity_group, first_difference_group, start, order
                    )
                )
                precision += (
                    stencil_confidence
                    * temporal_scale
                    * wp.float64(_difference_coefficient(order, frame - start))
                    * wp.float64(_difference_coefficient(order, previous - start))
                )
        result += wp.float64(jacobian[frame, residual, dof]) * precision * wp.float64(jacobian[previous, residual, dof])
    if lag == 1:
        lower_scale = wp.float64(constraint_scale[frame, velocity_lower_base + dof])
        upper_scale = wp.float64(constraint_scale[frame, velocity_upper_base + dof])
        result -= wp.float64(weights[frame, velocity_lower_base + dof]) * lower_scale * lower_scale
        result -= wp.float64(weights[frame, velocity_upper_base + dof]) * upper_scale * upper_scale
    band[frame, dof, lag] = result


@wp.kernel
def _ipm_temporal_band_factor(
    band: wp.array3d(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    failed: wp.array(dtype=wp.int32),
):
    """Factor independent bandwidth-three temporal blocks in place."""
    segment, dof = wp.tid()
    if segment >= segment_count or enabled[segment] == 0:
        return
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    for frame in range(begin, end):
        max_lag = wp.min(3, frame - begin)
        for reverse in range(max_lag):
            lag = max_lag - reverse
            previous = frame - lag
            value = band[frame, dof, lag]
            first_shared = wp.max(begin, frame - 3)
            first_shared = wp.max(first_shared, previous - 3)
            for shared in range(first_shared, previous):
                value -= band[frame, dof, frame - shared] * band[previous, dof, previous - shared]
            value /= band[previous, dof, 0]
            band[frame, dof, lag] = value
        diagonal = band[frame, dof, 0]
        for lag in range(1, max_lag + 1):
            value = band[frame, dof, lag]
            diagonal -= value * value
        if not wp.isfinite(diagonal) or diagonal <= wp.float64(1.0e-24):
            wp.atomic_max(failed, segment, 1)
            diagonal = wp.float64(1.0e-24)
        band[frame, dof, 0] = wp.sqrt(diagonal)


@wp.func
def _ipm_block_band_forward_segment_f64(
    values: wp.array2d(dtype=wp.float64),
    factor: wp.array4d(dtype=wp.float64),
    begin: int,
    end: int,
):
    """Apply the inverse lower block-band factor serially in place."""
    width = factor.shape[2]
    for frame in range(begin, end):
        for row in range(width):
            value = values[frame, row]
            for lag in range(1, wp.min(3, frame - begin) + 1):
                previous = frame - lag
                for column in range(width):
                    value -= factor[frame, lag, row, column] * values[previous, column]
            for column in range(row):
                value -= factor[frame, 0, row, column] * values[frame, column]
            values[frame, row] = value / factor[frame, 0, row, row]


@wp.func
def _ipm_block_band_backward_segment_f64(
    values: wp.array2d(dtype=wp.float64),
    factor: wp.array4d(dtype=wp.float64),
    begin: int,
    end: int,
):
    """Apply the inverse transposed block-band factor serially in place."""
    width = factor.shape[2]
    for reverse_frame in range(end - begin):
        frame = end - reverse_frame - 1
        for reverse_row in range(width):
            row = width - reverse_row - 1
            value = values[frame, row]
            for column in range(row + 1, width):
                value -= factor[frame, 0, column, row] * values[frame, column]
            for lag in range(1, wp.min(3, end - frame - 1) + 1):
                following = frame + lag
                for column in range(width):
                    value -= factor[following, lag, column, row] * values[following, column]
            values[frame, row] = value / factor[frame, 0, row, row]


@wp.func
def _ipm_augmented_full_row_value_f64(
    values: wp.array2d(dtype=wp.float64),
    jacobian: wp.array3d(dtype=wp.float32),
    weights: wp.array2d(dtype=wp.float32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    coordinate_bound_count: int,
    constraint_scale: wp.array2d(dtype=wp.float32),
    begin: int,
    frame: int,
    column: int,
) -> wp.float64:
    """Apply one full canonical Phase-II row ``R = sqrt(W) C``."""
    scale = wp.float64(constraint_scale[frame, column])
    if scale == wp.float64(0.0):
        return wp.float64(0.0)
    residual_count = jacobian.shape[1]
    coordinate_capacity = coordinate_dof_indices.shape[0]
    dof_count = jacobian.shape[2]
    coordinate_lower_base = residual_count
    coordinate_upper_base = coordinate_lower_base + coordinate_capacity
    velocity_lower_base = coordinate_upper_base + coordinate_capacity
    velocity_upper_base = velocity_lower_base + dof_count
    value = wp.float64(0.0)
    if column < residual_count:
        for dof in range(dof_count):
            value += wp.float64(jacobian[frame, column, dof]) * values[frame, dof]
    elif column < velocity_lower_base:
        upper = column >= coordinate_upper_base
        bound = column - (coordinate_upper_base if upper else coordinate_lower_base)
        if bound < coordinate_bound_count:
            value = values[frame, coordinate_dof_indices[bound]]
            if not upper:
                value = -value
    elif frame > begin:
        upper = column >= velocity_upper_base
        dof = column - (velocity_upper_base if upper else velocity_lower_base)
        value = values[frame, dof] - values[frame - 1, dof]
        if not upper:
            value = -value
    return wp.sqrt(wp.float64(weights[frame, column])) * scale * value


@wp.func
def _ipm_augmented_full_transpose_value_f64(
    values: wp.array2d(dtype=wp.float64),
    jacobian: wp.array3d(dtype=wp.float32),
    weights: wp.array2d(dtype=wp.float32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    coordinate_bound_count: int,
    constraint_scale: wp.array2d(dtype=wp.float32),
    begin: int,
    end: int,
    frame: int,
    dof: int,
) -> wp.float64:
    """Apply the transpose of every full canonical Phase-II row."""
    residual_count = jacobian.shape[1]
    coordinate_capacity = coordinate_dof_indices.shape[0]
    dof_count = jacobian.shape[2]
    coordinate_lower_base = residual_count
    coordinate_upper_base = coordinate_lower_base + coordinate_capacity
    velocity_lower_base = coordinate_upper_base + coordinate_capacity
    velocity_upper_base = velocity_lower_base + dof_count
    result = wp.float64(0.0)
    for residual in range(residual_count):
        scale = wp.float64(constraint_scale[frame, residual])
        if scale != wp.float64(0.0):
            coefficient = wp.sqrt(wp.float64(weights[frame, residual])) * scale
            result += coefficient * wp.float64(jacobian[frame, residual, dof]) * values[frame, residual]
    for bound in range(coordinate_bound_count):
        if coordinate_dof_indices[bound] == dof:
            lower_column = coordinate_lower_base + bound
            upper_column = coordinate_upper_base + bound
            lower_scale = wp.float64(constraint_scale[frame, lower_column])
            upper_scale = wp.float64(constraint_scale[frame, upper_column])
            if lower_scale != wp.float64(0.0):
                result -= wp.sqrt(wp.float64(weights[frame, lower_column])) * lower_scale * values[frame, lower_column]
            if upper_scale != wp.float64(0.0):
                result += wp.sqrt(wp.float64(weights[frame, upper_column])) * upper_scale * values[frame, upper_column]
    if frame > begin:
        lower_column = velocity_lower_base + dof
        upper_column = velocity_upper_base + dof
        lower_scale = wp.float64(constraint_scale[frame, lower_column])
        upper_scale = wp.float64(constraint_scale[frame, upper_column])
        if lower_scale != wp.float64(0.0):
            result -= wp.sqrt(wp.float64(weights[frame, lower_column])) * lower_scale * values[frame, lower_column]
        if upper_scale != wp.float64(0.0):
            result += wp.sqrt(wp.float64(weights[frame, upper_column])) * upper_scale * values[frame, upper_column]
    if frame + 1 < end:
        lower_column = velocity_lower_base + dof
        upper_column = velocity_upper_base + dof
        lower_scale = wp.float64(constraint_scale[frame + 1, lower_column])
        upper_scale = wp.float64(constraint_scale[frame + 1, upper_column])
        if lower_scale != wp.float64(0.0):
            result += (
                wp.sqrt(wp.float64(weights[frame + 1, lower_column])) * lower_scale * values[frame + 1, lower_column]
            )
        if upper_scale != wp.float64(0.0):
            result -= (
                wp.sqrt(wp.float64(weights[frame + 1, upper_column])) * upper_scale * values[frame + 1, upper_column]
            )
    return result


@wp.func
def _ipm_augmented_true_residual_squared_f64(
    factor: wp.array4d(dtype=wp.float64),
    jacobian: wp.array3d(dtype=wp.float32),
    weights: wp.array2d(dtype=wp.float32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    coordinate_bound_count: int,
    constraint_scale: wp.array2d(dtype=wp.float32),
    right_hand_side: wp.array2d(dtype=wp.float64),
    begin: int,
    end: int,
    primal_solution: wp.array2d(dtype=wp.float64),
    row_solution: wp.array2d(dtype=wp.float64),
    primal_basis: wp.array2d(dtype=wp.float64),
    primal_work: wp.array2d(dtype=wp.float64),
    row_basis: wp.array2d(dtype=wp.float64),
) -> wp.float64:
    """Recompute the transformed augmented residual without disturbing Lanczos state."""
    dof_count = jacobian.shape[2]
    row_width = row_solution.shape[1]
    for frame in range(begin, end):
        for dof in range(dof_count):
            primal_work[frame, dof] = primal_solution[frame, dof]
    _ipm_block_band_backward_segment_f64(primal_work, factor, begin, end)
    result = wp.float64(0.0)
    for frame in range(begin, end):
        for column in range(row_width):
            value = row_solution[frame, column] - _ipm_augmented_full_row_value_f64(
                primal_work,
                jacobian,
                weights,
                coordinate_dof_indices,
                coordinate_bound_count,
                constraint_scale,
                begin,
                frame,
                column,
            )
            row_basis[frame, column] = value
            result += value * value
        for dof in range(dof_count):
            primal_basis[frame, dof] = _ipm_augmented_full_transpose_value_f64(
                row_solution,
                jacobian,
                weights,
                coordinate_dof_indices,
                coordinate_bound_count,
                constraint_scale,
                begin,
                end,
                frame,
                dof,
            )
    _ipm_block_band_forward_segment_f64(primal_basis, factor, begin, end)
    for frame in range(begin, end):
        for dof in range(dof_count):
            primal_work[frame, dof] = right_hand_side[frame, dof]
    _ipm_block_band_forward_segment_f64(primal_work, factor, begin, end)
    for frame in range(begin, end):
        for dof in range(dof_count):
            value = primal_work[frame, dof] - primal_solution[frame, dof] - primal_basis[frame, dof]
            primal_basis[frame, dof] = value
            result += value * value
    return result


@wp.kernel
def _ipm_augmented_minres_solve_f64(  # noqa: C901
    factor: wp.array4d(dtype=wp.float64),
    factor_failed: wp.array(dtype=wp.int32),
    jacobian: wp.array3d(dtype=wp.float32),
    weights: wp.array2d(dtype=wp.float32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    coordinate_bound_count: int,
    constraint_scale: wp.array2d(dtype=wp.float32),
    segment_offsets: wp.array(dtype=wp.int32),
    right_hand_side: wp.array2d(dtype=wp.float64),
    relative_tolerance: float,
    numerical_epsilon: float,
    max_iterations: int,
    segment_count: int,
    primal_solution: wp.array2d(dtype=wp.float64),
    primal_r1: wp.array2d(dtype=wp.float64),
    primal_r2: wp.array2d(dtype=wp.float64),
    primal_basis: wp.array2d(dtype=wp.float64),
    primal_work: wp.array2d(dtype=wp.float64),
    primal_direction_older: wp.array2d(dtype=wp.float64),
    primal_direction_old: wp.array2d(dtype=wp.float64),
    row_solution: wp.array2d(dtype=wp.float64),
    row_r1: wp.array2d(dtype=wp.float64),
    row_r2: wp.array2d(dtype=wp.float64),
    row_basis: wp.array2d(dtype=wp.float64),
    row_direction_older: wp.array2d(dtype=wp.float64),
    row_direction_old: wp.array2d(dtype=wp.float64),
    solve_enabled: wp.array(dtype=wp.int32),
    failed: wp.array(dtype=wp.int32),
):
    """Solve the two-sided-whitened full Phase-II augmented system in float64."""
    segment = wp.tid()
    if segment >= segment_count:
        return
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    dof_count = jacobian.shape[2]
    row_width = row_solution.shape[1]
    if solve_enabled[segment] == 0:
        return
    if factor_failed[segment] != 0:
        failed[segment] = 1
        return
    if failed[segment] != 0:
        return
    for frame in range(begin, end):
        for dof in range(dof_count):
            value = right_hand_side[frame, dof]
            primal_solution[frame, dof] = wp.float64(0.0)
            primal_r1[frame, dof] = value
            primal_r2[frame, dof] = value
            primal_basis[frame, dof] = wp.float64(0.0)
            primal_work[frame, dof] = wp.float64(0.0)
            primal_direction_older[frame, dof] = wp.float64(0.0)
            primal_direction_old[frame, dof] = wp.float64(0.0)
        for column in range(row_width):
            row_solution[frame, column] = wp.float64(0.0)
            row_r1[frame, column] = wp.float64(0.0)
            row_r2[frame, column] = wp.float64(0.0)
            row_basis[frame, column] = wp.float64(0.0)
            row_direction_older[frame, column] = wp.float64(0.0)
            row_direction_old[frame, column] = wp.float64(0.0)
    _ipm_block_band_forward_segment_f64(primal_r1, factor, begin, end)
    beta_squared = wp.float64(0.0)
    for frame in range(begin, end):
        for dof in range(dof_count):
            value = primal_r1[frame, dof]
            primal_r2[frame, dof] = value
            beta_squared += value * value
    failed_local = bool(not wp.isfinite(beta_squared) or beta_squared < -wp.float64(numerical_epsilon))
    beta = wp.float64(0.0) if failed_local else wp.sqrt(wp.max(beta_squared, wp.float64(0.0)))
    beta_initial = beta
    threshold = wp.max(wp.float64(numerical_epsilon), wp.float64(relative_tolerance) * beta_initial)
    converged = bool(not failed_local and beta <= wp.float64(numerical_epsilon))
    old_beta = wp.float64(0.0)
    dbar = wp.float64(0.0)
    epsln = wp.float64(0.0)
    phibar = beta_initial
    cs = wp.float64(-1.0)
    sn = wp.float64(0.0)

    for iteration in range(max_iterations):
        if not converged and not failed_local:
            for frame in range(begin, end):
                for dof in range(dof_count):
                    primal_basis[frame, dof] = primal_r2[frame, dof] / beta
                    primal_work[frame, dof] = primal_basis[frame, dof]
                for column in range(row_width):
                    row_basis[frame, column] = row_r2[frame, column] / beta
            _ipm_block_band_backward_segment_f64(primal_work, factor, begin, end)
            previous_coefficient = wp.float64(0.0) if iteration == 0 else beta / old_beta
            for frame in range(begin, end):
                for column in range(row_width):
                    operator_value = (
                        _ipm_augmented_full_row_value_f64(
                            primal_work,
                            jacobian,
                            weights,
                            coordinate_dof_indices,
                            coordinate_bound_count,
                            constraint_scale,
                            begin,
                            frame,
                            column,
                        )
                        - row_basis[frame, column]
                    )
                    row_r1[frame, column] = operator_value - previous_coefficient * row_r1[frame, column]
                for dof in range(dof_count):
                    primal_work[frame, dof] = _ipm_augmented_full_transpose_value_f64(
                        row_basis,
                        jacobian,
                        weights,
                        coordinate_dof_indices,
                        coordinate_bound_count,
                        constraint_scale,
                        begin,
                        end,
                        frame,
                        dof,
                    )
            _ipm_block_band_forward_segment_f64(primal_work, factor, begin, end)
            alpha = wp.float64(0.0)
            for frame in range(begin, end):
                for dof in range(dof_count):
                    previous_value = primal_r1[frame, dof]
                    operator_value = primal_basis[frame, dof] + primal_work[frame, dof]
                    operator_value -= previous_coefficient * previous_value
                    primal_r1[frame, dof] = operator_value
                    alpha += primal_basis[frame, dof] * operator_value
                for column in range(row_width):
                    alpha += row_basis[frame, column] * row_r1[frame, column]
            failed_local = bool(not wp.isfinite(alpha))
            current_coefficient = wp.float64(0.0) if failed_local else alpha / beta
            next_beta_squared = wp.float64(0.0)
            if not failed_local:
                for frame in range(begin, end):
                    for dof in range(dof_count):
                        operator_value = primal_r1[frame, dof] - current_coefficient * primal_r2[frame, dof]
                        primal_r1[frame, dof] = primal_r2[frame, dof]
                        primal_r2[frame, dof] = operator_value
                        next_beta_squared += operator_value * operator_value
                    for column in range(row_width):
                        operator_value = row_r1[frame, column] - current_coefficient * row_r2[frame, column]
                        row_r1[frame, column] = row_r2[frame, column]
                        row_r2[frame, column] = operator_value
                        next_beta_squared += operator_value * operator_value
                failed_local = bool(
                    not wp.isfinite(next_beta_squared) or next_beta_squared < -wp.float64(numerical_epsilon)
                )
            beta_next = wp.float64(0.0) if failed_local else wp.sqrt(wp.max(next_beta_squared, wp.float64(0.0)))
            oldeps = epsln
            delta = cs * dbar + sn * alpha
            gbar = sn * dbar - cs * alpha
            epsln = sn * beta_next
            dbar = -cs * beta_next
            gamma_squared = gbar * gbar + beta_next * beta_next
            failed_local = bool(failed_local or not wp.isfinite(gamma_squared))
            gamma = wp.float64(0.0) if failed_local else wp.sqrt(wp.max(gamma_squared, wp.float64(0.0)))
            failed_local = bool(failed_local or gamma <= wp.float64(numerical_epsilon))
            if not failed_local:
                cs = gbar / gamma
                sn = beta_next / gamma
                phi = cs * phibar
                phibar = sn * phibar
                failed_local = bool(not wp.isfinite(phi) or not wp.isfinite(phibar))
                if not failed_local:
                    for frame in range(begin, end):
                        for dof in range(dof_count):
                            direction = (
                                primal_basis[frame, dof]
                                - oldeps * primal_direction_older[frame, dof]
                                - delta * primal_direction_old[frame, dof]
                            ) / gamma
                            primal_solution[frame, dof] += phi * direction
                            primal_direction_older[frame, dof] = primal_direction_old[frame, dof]
                            primal_direction_old[frame, dof] = direction
                        for column in range(row_width):
                            direction = (
                                row_basis[frame, column]
                                - oldeps * row_direction_older[frame, column]
                                - delta * row_direction_old[frame, column]
                            ) / gamma
                            row_solution[frame, column] += phi * direction
                            row_direction_older[frame, column] = row_direction_old[frame, column]
                            row_direction_old[frame, column] = direction
                    estimated_converged = bool(wp.abs(phibar) <= threshold)
                    if estimated_converged:
                        true_residual_squared = _ipm_augmented_true_residual_squared_f64(
                            factor,
                            jacobian,
                            weights,
                            coordinate_dof_indices,
                            coordinate_bound_count,
                            constraint_scale,
                            right_hand_side,
                            begin,
                            end,
                            primal_solution,
                            row_solution,
                            primal_basis,
                            primal_work,
                            row_basis,
                        )
                        failed_local = bool(
                            not wp.isfinite(true_residual_squared)
                            or true_residual_squared < -wp.float64(numerical_epsilon)
                        )
                        true_residual = (
                            wp.float64(0.0) if failed_local else wp.sqrt(wp.max(true_residual_squared, wp.float64(0.0)))
                        )
                        converged = bool(not failed_local and true_residual <= threshold)
                    failed_local = bool(failed_local or (not converged and beta_next <= wp.float64(numerical_epsilon)))
            old_beta = beta
            beta = beta_next

    if failed_local or not converged:
        failed[segment] = 1
        solve_enabled[segment] = 0
        return

    true_residual_squared = _ipm_augmented_true_residual_squared_f64(
        factor,
        jacobian,
        weights,
        coordinate_dof_indices,
        coordinate_bound_count,
        constraint_scale,
        right_hand_side,
        begin,
        end,
        primal_solution,
        row_solution,
        primal_basis,
        primal_work,
        row_basis,
    )
    failed_local = bool(
        not wp.isfinite(true_residual_squared) or true_residual_squared < -wp.float64(numerical_epsilon)
    )
    true_residual = wp.float64(0.0) if failed_local else wp.sqrt(wp.max(true_residual_squared, wp.float64(0.0)))
    if failed_local or true_residual > threshold:
        failed[segment] = 1
        solve_enabled[segment] = 0
        return
    for frame in range(begin, end):
        for dof in range(dof_count):
            primal_work[frame, dof] = primal_solution[frame, dof]
    _ipm_block_band_backward_segment_f64(primal_work, factor, begin, end)
    for frame in range(begin, end):
        for dof in range(dof_count):
            primal_solution[frame, dof] = primal_work[frame, dof]


@wp.kernel
def _ipm_block_band_matrix_build(
    jacobian: wp.array3d(dtype=wp.float32),
    base_weights: wp.array2d(dtype=wp.float32),
    temporal_weights: wp.array2d(dtype=wp.float32),
    residual_activity: wp.array2d(dtype=wp.float32),
    activity_group_by_residual: wp.array(dtype=wp.int32),
    first_difference_group_by_residual: wp.array(dtype=wp.int32),
    step_seconds: wp.array(dtype=wp.float32),
    segment_damping: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    matrix: wp.array4d(dtype=wp.float64),
):
    """Build the objective-only bandwidth-three blocks used to whiten Phase II."""
    frame, lag, row, column = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    begin = segment_offsets[segment]
    previous = frame - lag
    if enabled[segment] == 0:
        return
    if lag > 3 or previous < begin:
        matrix[frame, lag, row, column] = wp.float64(1.0) if lag == 0 and row == column else wp.float64(0.0)
        return

    residual_count = jacobian.shape[1]
    end = segment_offsets[segment + 1]
    dt = wp.float64(step_seconds[segment])
    result = wp.float64(segment_damping[segment]) if lag == 0 and row == column else wp.float64(0.0)

    for residual in range(residual_count):
        activity_group = activity_group_by_residual[residual]
        first_difference_group = first_difference_group_by_residual[residual]
        precision = wp.float64(0.0)
        if lag == 0:
            frame_confidence = (
                wp.float64(1.0) if activity_group < 0 else wp.float64(residual_activity[frame, activity_group])
            )
            precision = frame_confidence * wp.float64(base_weights[frame, residual])
        for order in range(1, 4):
            weight = temporal_weights[order - 1, residual]
            if order < lag or weight == 0.0:
                continue
            inverse_dt = wp.float64(1.0) / dt
            temporal_scale = wp.float64(weight)
            for _ in range(order):
                temporal_scale *= inverse_dt * inverse_dt
            first_start = wp.max(begin, frame - order)
            last_start = wp.min(previous, end - order - 1)
            for start in range(first_start, last_start + 1):
                stencil_confidence = wp.float64(
                    _temporal_stencil_confidence(
                        residual_activity, activity_group, first_difference_group, start, order
                    )
                )
                precision += (
                    stencil_confidence
                    * temporal_scale
                    * wp.float64(_difference_coefficient(order, frame - start))
                    * wp.float64(_difference_coefficient(order, previous - start))
                )
        result += (
            wp.float64(jacobian[frame, residual, row]) * precision * wp.float64(jacobian[previous, residual, column])
        )
    matrix[frame, lag, row, column] = result


@wp.kernel
def _ipm_block_band_barrier_add(
    jacobian: wp.array3d(dtype=wp.float32),
    weights: wp.array2d(dtype=wp.float32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    coordinate_bound_count: int,
    constraint_scale: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    matrix: wp.array4d(dtype=wp.float64),
):
    """Add the complete Phase-II barrier normal matrix to objective blocks."""
    frame, lag, row, column = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    begin = segment_offsets[segment]
    if enabled[segment] == 0 or lag > 1 or frame - lag < begin:
        return
    residual_count = jacobian.shape[1]
    coordinate_capacity = coordinate_dof_indices.shape[0]
    dof_count = jacobian.shape[2]
    coordinate_lower_base = residual_count
    coordinate_upper_base = coordinate_lower_base + coordinate_capacity
    velocity_lower_base = coordinate_upper_base + coordinate_capacity
    velocity_upper_base = velocity_lower_base + dof_count
    end = segment_offsets[segment + 1]
    result = matrix[frame, lag, row, column]
    if lag == 0:
        for residual in range(residual_count):
            scale = wp.float64(constraint_scale[frame, residual])
            precision = wp.float64(weights[frame, residual]) * scale * scale
            result += (
                wp.float64(jacobian[frame, residual, row]) * precision * wp.float64(jacobian[frame, residual, column])
            )
        if row == column:
            for bound in range(coordinate_bound_count):
                if coordinate_dof_indices[bound] == row:
                    lower_column = coordinate_lower_base + bound
                    upper_column = coordinate_upper_base + bound
                    lower_scale = wp.float64(constraint_scale[frame, lower_column])
                    upper_scale = wp.float64(constraint_scale[frame, upper_column])
                    result += wp.float64(weights[frame, lower_column]) * lower_scale * lower_scale
                    result += wp.float64(weights[frame, upper_column]) * upper_scale * upper_scale
            if frame > begin:
                lower_scale = wp.float64(constraint_scale[frame, velocity_lower_base + row])
                upper_scale = wp.float64(constraint_scale[frame, velocity_upper_base + row])
                result += wp.float64(weights[frame, velocity_lower_base + row]) * lower_scale * lower_scale
                result += wp.float64(weights[frame, velocity_upper_base + row]) * upper_scale * upper_scale
            if frame + 1 < end:
                lower_scale = wp.float64(constraint_scale[frame + 1, velocity_lower_base + row])
                upper_scale = wp.float64(constraint_scale[frame + 1, velocity_upper_base + row])
                result += wp.float64(weights[frame + 1, velocity_lower_base + row]) * lower_scale * lower_scale
                result += wp.float64(weights[frame + 1, velocity_upper_base + row]) * upper_scale * upper_scale
    elif row == column:
        lower_scale = wp.float64(constraint_scale[frame, velocity_lower_base + row])
        upper_scale = wp.float64(constraint_scale[frame, velocity_upper_base + row])
        result -= wp.float64(weights[frame, velocity_lower_base + row]) * lower_scale * lower_scale
        result -= wp.float64(weights[frame, velocity_upper_base + row]) * upper_scale * upper_scale
    matrix[frame, lag, row, column] = result


@wp.kernel
def _ipm_block_band_scale_build_f64(
    matrix: wp.array4d(dtype=wp.float64),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    scale: wp.array2d(dtype=wp.float64),
):
    """Build the positive diagonal scale for full-normal equilibration."""
    frame, dof = wp.tid()
    if frame >= active_frames:
        return
    diagonal = matrix[frame, 0, dof, dof]
    scale[frame, dof] = (
        wp.sqrt(diagonal)
        if enabled[frame_segment[frame]] != 0 and wp.isfinite(diagonal) and diagonal > wp.float64(0.0)
        else wp.float64(1.0)
    )


@wp.kernel
def _ipm_block_band_equilibrate_f64(
    scale: wp.array2d(dtype=wp.float64),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    machine_epsilon: float,
    active_frames: int,
    matrix: wp.array4d(dtype=wp.float64),
):
    """Equilibrate a full normal matrix and add an ``n * eps`` PSD guard."""
    frame, lag, row, column = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    begin = segment_offsets[segment]
    previous = frame - lag
    if enabled[segment] == 0 or lag > 3 or previous < begin:
        return
    value = matrix[frame, lag, row, column] / (scale[frame, row] * scale[previous, column])
    if lag == 0 and row == column:
        dimension = (segment_offsets[segment + 1] - begin) * matrix.shape[2]
        value += wp.float64(dimension) * wp.float64(machine_epsilon)
    matrix[frame, lag, row, column] = value


@wp.kernel
def _ipm_solution_equilibrate_f64(
    scale: wp.array2d(dtype=wp.float64),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    values: wp.array2d(dtype=wp.float64),
):
    """Map a right-hand side or solution through the diagonal equilibration."""
    frame, dof = wp.tid()
    if frame < active_frames and enabled[frame_segment[frame]] != 0:
        values[frame, dof] /= scale[frame, dof]


@wp.kernel
def _phase_one_block_band_matrix_build(
    jacobian: wp.array3d(dtype=wp.float32),
    weights: wp.array2d(dtype=wp.float32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    coordinate_bound_count: int,
    constraint_scale: wp.array2d(dtype=wp.float32),
    segment_damping: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    matrix: wp.array4d(dtype=wp.float64),
):
    """Build the physical primal block of the condensed Phase-I arrowhead."""
    frame, lag, row, column = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    begin = segment_offsets[segment]
    previous = frame - lag
    if enabled[segment] == 0 or lag > 1 or previous < begin:
        matrix[frame, lag, row, column] = wp.float64(1.0) if lag == 0 and row == column else wp.float64(0.0)
        return

    residual_count = jacobian.shape[1]
    coordinate_capacity = coordinate_dof_indices.shape[0]
    dof_count = jacobian.shape[2]
    coordinate_lower_base = residual_count
    coordinate_upper_base = coordinate_lower_base + coordinate_capacity
    velocity_lower_base = coordinate_upper_base + coordinate_capacity
    velocity_upper_base = velocity_lower_base + dof_count
    end = segment_offsets[segment + 1]
    result = wp.float64(segment_damping[segment]) if lag == 0 and row == column else wp.float64(0.0)

    if lag == 0:
        for residual in range(residual_count):
            scale = wp.float64(constraint_scale[frame, residual])
            precision = wp.float64(weights[frame, residual]) * scale * scale
            result += (
                wp.float64(jacobian[frame, residual, row]) * precision * wp.float64(jacobian[frame, residual, column])
            )
        if row == column:
            for bound in range(coordinate_bound_count):
                if coordinate_dof_indices[bound] == row:
                    lower_column = coordinate_lower_base + bound
                    upper_column = coordinate_upper_base + bound
                    lower_scale = wp.float64(constraint_scale[frame, lower_column])
                    upper_scale = wp.float64(constraint_scale[frame, upper_column])
                    result += wp.float64(weights[frame, lower_column]) * lower_scale * lower_scale
                    result += wp.float64(weights[frame, upper_column]) * upper_scale * upper_scale
            if frame > begin:
                lower_scale = wp.float64(constraint_scale[frame, velocity_lower_base + row])
                upper_scale = wp.float64(constraint_scale[frame, velocity_upper_base + row])
                result += wp.float64(weights[frame, velocity_lower_base + row]) * lower_scale * lower_scale
                result += wp.float64(weights[frame, velocity_upper_base + row]) * upper_scale * upper_scale
            if frame + 1 < end:
                lower_scale = wp.float64(constraint_scale[frame + 1, velocity_lower_base + row])
                upper_scale = wp.float64(constraint_scale[frame + 1, velocity_upper_base + row])
                result += wp.float64(weights[frame + 1, velocity_lower_base + row]) * lower_scale * lower_scale
                result += wp.float64(weights[frame + 1, velocity_upper_base + row]) * upper_scale * upper_scale
    elif row == column:
        lower_scale = wp.float64(constraint_scale[frame, velocity_lower_base + row])
        upper_scale = wp.float64(constraint_scale[frame, velocity_upper_base + row])
        result -= wp.float64(weights[frame, velocity_lower_base + row]) * lower_scale * lower_scale
        result -= wp.float64(weights[frame, velocity_upper_base + row]) * upper_scale * upper_scale
    matrix[frame, lag, row, column] = result


@wp.kernel
def _ipm_block_band_matrix_factor(
    matrix: wp.array4d(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    failed: wp.array(dtype=wp.int32),
):
    """Factor exact bandwidth-three dense frame blocks in place."""
    segment = wp.tid()
    if segment >= segment_count or enabled[segment] == 0:
        return
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    width = matrix.shape[2]
    for frame in range(begin, end):
        first_previous = wp.max(begin, frame - 3)
        for previous in range(first_previous, frame):
            lag = frame - previous
            first_shared = wp.max(first_previous, previous - 3)
            for row in range(width):
                for column in range(width):
                    value = matrix[frame, lag, row, column]
                    for shared in range(first_shared, previous):
                        frame_lag = frame - shared
                        previous_lag = previous - shared
                        for inner in range(width):
                            value -= (
                                matrix[frame, frame_lag, row, inner] * matrix[previous, previous_lag, column, inner]
                            )
                    for inner in range(column):
                        value -= matrix[frame, lag, row, inner] * matrix[previous, 0, column, inner]
                    diagonal = matrix[previous, 0, column, column]
                    if not wp.isfinite(value) or not wp.isfinite(diagonal) or diagonal <= wp.float64(0.0):
                        wp.atomic_max(failed, segment, 1)
                        value = wp.float64(0.0)
                        diagonal = wp.float64(1.0e-12)
                    matrix[frame, lag, row, column] = value / diagonal
        for row in range(width):
            for column in range(row + 1):
                value = matrix[frame, 0, row, column]
                for previous in range(first_previous, frame):
                    lag = frame - previous
                    for inner in range(width):
                        value -= matrix[frame, lag, row, inner] * matrix[frame, lag, column, inner]
                for inner in range(column):
                    value -= matrix[frame, 0, row, inner] * matrix[frame, 0, column, inner]
                if row == column:
                    if not wp.isfinite(value) or value <= wp.float64(1.0e-24):
                        wp.atomic_max(failed, segment, 1)
                        value = wp.float64(1.0e-24)
                    matrix[frame, 0, row, column] = wp.sqrt(value)
                else:
                    diagonal = matrix[frame, 0, column, column]
                    if not wp.isfinite(value) or not wp.isfinite(diagonal) or diagonal <= wp.float64(0.0):
                        wp.atomic_max(failed, segment, 1)
                        value = wp.float64(0.0)
                        diagonal = wp.float64(1.0e-12)
                    matrix[frame, 0, row, column] = value / diagonal


@wp.kernel
def _ipm_block_band_forward_f64(
    values: wp.array2d(dtype=wp.float64),
    factor: wp.array4d(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    output: wp.array2d(dtype=wp.float64),
):
    """Apply the inverse lower block-band factor in double precision."""
    segment = wp.tid()
    if segment >= segment_count:
        return
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    width = factor.shape[2]
    for frame in range(begin, end):
        for row in range(width):
            value = values[frame, row] if enabled[segment] != 0 else wp.float64(0.0)
            for lag in range(1, wp.min(3, frame - begin) + 1):
                previous = frame - lag
                for column in range(width):
                    value -= factor[frame, lag, row, column] * output[previous, column]
            for column in range(row):
                value -= factor[frame, 0, row, column] * output[frame, column]
            output[frame, row] = value / factor[frame, 0, row, row]


@wp.kernel
def _ipm_block_band_backward_f64(
    values: wp.array2d(dtype=wp.float64),
    factor: wp.array4d(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    work: wp.array2d(dtype=wp.float64),
    output: wp.array2d(dtype=wp.float64),
):
    """Apply the inverse transposed block-band factor in double precision."""
    segment = wp.tid()
    if segment >= segment_count:
        return
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    width = factor.shape[2]
    for reverse_frame in range(end - begin):
        frame = end - reverse_frame - 1
        for reverse_row in range(width):
            row = width - reverse_row - 1
            value = values[frame, row] if enabled[segment] != 0 else wp.float64(0.0)
            for column in range(row + 1, width):
                value -= factor[frame, 0, column, row] * work[frame, column]
            for lag in range(1, wp.min(3, end - frame - 1) + 1):
                following = frame + lag
                for column in range(width):
                    value -= factor[following, lag, column, row] * work[following, column]
            value /= factor[frame, 0, row, row]
            work[frame, row] = value
            output[frame, row] = value


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    __syncthreads();
#endif
"""
)
def _ipm_block_band_sync():
    """Synchronize threads in an exploratory cooperative block-band kernel."""
    return


@wp.kernel
def _ipm_block_band_matrix_factor_parallel(
    matrix: wp.array4d(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    failed: wp.array(dtype=wp.int32),
):
    """Factor block-band matrices cooperatively with one CUDA block per segment."""
    thread = wp.tid()
    block_size = wp.block_dim()
    segment = thread // block_size
    lane = thread - segment * block_size
    if segment >= segment_count or enabled[segment] == 0:
        return
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    width = matrix.shape[2]
    element_count = width * width
    for frame in range(begin, end):
        first_previous = wp.max(begin, frame - 3)
        for previous in range(first_previous, frame):
            lag = frame - previous
            first_shared = wp.max(first_previous, previous - 3)
            element = lane
            while element < element_count:
                row = element // width
                column = element - row * width
                value = matrix[frame, lag, row, column]
                for shared in range(first_shared, previous):
                    frame_lag = frame - shared
                    previous_lag = previous - shared
                    for inner in range(width):
                        value -= matrix[frame, frame_lag, row, inner] * matrix[previous, previous_lag, column, inner]
                matrix[frame, lag, row, column] = value
                element += block_size
            _ipm_block_band_sync()

            row = lane
            while row < width:
                for column in range(width):
                    value = matrix[frame, lag, row, column]
                    for inner in range(column):
                        value -= matrix[frame, lag, row, inner] * matrix[previous, 0, column, inner]
                    diagonal = matrix[previous, 0, column, column]
                    if not wp.isfinite(value) or not wp.isfinite(diagonal) or diagonal <= wp.float64(0.0):
                        wp.atomic_max(failed, segment, 1)
                        value = wp.float64(0.0)
                        diagonal = wp.float64(1.0e-12)
                    matrix[frame, lag, row, column] = value / diagonal
                row += block_size
            _ipm_block_band_sync()

        element = lane
        while element < element_count:
            row = element // width
            column = element - row * width
            if column <= row:
                value = matrix[frame, 0, row, column]
                for previous in range(first_previous, frame):
                    lag = frame - previous
                    for inner in range(width):
                        value -= matrix[frame, lag, row, inner] * matrix[frame, lag, column, inner]
                matrix[frame, 0, row, column] = value
            element += block_size
        _ipm_block_band_sync()

        for column in range(width):
            if lane == 0:
                value = matrix[frame, 0, column, column]
                for inner in range(column):
                    entry = matrix[frame, 0, column, inner]
                    value -= entry * entry
                if not wp.isfinite(value) or value <= wp.float64(1.0e-24):
                    wp.atomic_max(failed, segment, 1)
                    value = wp.float64(1.0e-24)
                matrix[frame, 0, column, column] = wp.sqrt(value)
            _ipm_block_band_sync()

            row = column + lane + 1
            while row < width:
                value = matrix[frame, 0, row, column]
                for inner in range(column):
                    value -= matrix[frame, 0, row, inner] * matrix[frame, 0, column, inner]
                diagonal = matrix[frame, 0, column, column]
                if not wp.isfinite(value) or not wp.isfinite(diagonal) or diagonal <= wp.float64(0.0):
                    wp.atomic_max(failed, segment, 1)
                    value = wp.float64(0.0)
                    diagonal = wp.float64(1.0e-12)
                matrix[frame, 0, row, column] = value / diagonal
                row += block_size
            _ipm_block_band_sync()


@wp.kernel
def _ipm_block_band_forward_f64_parallel(
    values: wp.array2d(dtype=wp.float64),
    factor: wp.array4d(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    output: wp.array2d(dtype=wp.float64),
):
    """Apply the lower block-band inverse cooperatively with one CUDA block per segment."""
    thread = wp.tid()
    block_size = wp.block_dim()
    segment = thread // block_size
    lane = thread - segment * block_size
    if segment >= segment_count:
        return
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    width = factor.shape[2]
    for frame in range(begin, end):
        row = lane
        while row < width:
            value = values[frame, row] if enabled[segment] != 0 else wp.float64(0.0)
            for lag in range(1, wp.min(3, frame - begin) + 1):
                previous = frame - lag
                for column in range(width):
                    value -= factor[frame, lag, row, column] * output[previous, column]
            output[frame, row] = value
            row += block_size
        _ipm_block_band_sync()

        if lane == 0:
            for row in range(width):
                value = output[frame, row]
                for column in range(row):
                    value -= factor[frame, 0, row, column] * output[frame, column]
                output[frame, row] = value / factor[frame, 0, row, row]
        _ipm_block_band_sync()


@wp.kernel
def _ipm_block_band_backward_f64_parallel(
    values: wp.array2d(dtype=wp.float64),
    factor: wp.array4d(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    work: wp.array2d(dtype=wp.float64),
    output: wp.array2d(dtype=wp.float64),
):
    """Apply the transposed block-band inverse cooperatively with one CUDA block per segment."""
    thread = wp.tid()
    block_size = wp.block_dim()
    segment = thread // block_size
    lane = thread - segment * block_size
    if segment >= segment_count:
        return
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    width = factor.shape[2]
    for reverse_frame in range(end - begin):
        frame = end - reverse_frame - 1
        row = lane
        while row < width:
            value = values[frame, row] if enabled[segment] != 0 else wp.float64(0.0)
            for lag in range(1, wp.min(3, end - frame - 1) + 1):
                following = frame + lag
                for column in range(width):
                    value -= factor[following, lag, column, row] * work[following, column]
            work[frame, row] = value
            row += block_size
        _ipm_block_band_sync()

        if lane == 0:
            for reverse_row in range(width):
                row = width - reverse_row - 1
                value = work[frame, row]
                for column in range(row + 1, width):
                    value -= factor[frame, 0, column, row] * work[frame, column]
                value /= factor[frame, 0, row, row]
                work[frame, row] = value
                output[frame, row] = value
        _ipm_block_band_sync()


@wp.kernel
def _ipm_frame_matrix_build(
    jacobian: wp.array3d(dtype=wp.float32),
    precision_diagonal: wp.array2d(dtype=wp.float32),
    weights: wp.array2d(dtype=wp.float32),
    constraint_scale: wp.array2d(dtype=wp.float32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    coordinate_bound_count: int,
    segment_damping: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    matrix: wp.array3d(dtype=wp.float64),
    segment_coupled: wp.array(dtype=wp.int32),
):
    """Build exact frame-diagonal blocks of the condensed Phase-II matrix."""
    frame, row, column = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    singleton = segment_offsets[segment + 1] - segment_offsets[segment] == 1
    if enabled[segment] == 0 or singleton or column > row:
        matrix[frame, row, column] = wp.float64(0.0)
        return
    residual_count = jacobian.shape[1]
    coordinate_capacity = coordinate_dof_indices.shape[0]
    dof_count = jacobian.shape[2]
    coordinate_lower_base = residual_count
    coordinate_upper_base = coordinate_lower_base + coordinate_capacity
    velocity_lower_base = coordinate_upper_base + coordinate_capacity
    velocity_upper_base = velocity_lower_base + dof_count
    result = wp.float64(segment_damping[segment]) if row == column else wp.float64(0.0)
    for residual in range(residual_count):
        row_scale = wp.float64(constraint_scale[frame, residual])
        precision = wp.float64(precision_diagonal[frame, residual])
        precision += wp.float64(weights[frame, residual]) * row_scale * row_scale
        result += wp.float64(jacobian[frame, residual, row]) * precision * wp.float64(jacobian[frame, residual, column])
    if row == column:
        for bound in range(coordinate_bound_count):
            if coordinate_dof_indices[bound] == row:
                lower_scale = wp.float64(constraint_scale[frame, coordinate_lower_base + bound])
                upper_scale = wp.float64(constraint_scale[frame, coordinate_upper_base + bound])
                result += wp.float64(weights[frame, coordinate_lower_base + bound]) * lower_scale * lower_scale
                result += wp.float64(weights[frame, coordinate_upper_base + bound]) * upper_scale * upper_scale
        begin = segment_offsets[segment]
        end = segment_offsets[segment + 1]
        if frame > begin:
            lower_scale = wp.float64(constraint_scale[frame, velocity_lower_base + row])
            upper_scale = wp.float64(constraint_scale[frame, velocity_upper_base + row])
            result += wp.float64(weights[frame, velocity_lower_base + row]) * lower_scale * lower_scale
            result += wp.float64(weights[frame, velocity_upper_base + row]) * upper_scale * upper_scale
        if frame + 1 < end:
            lower_scale = wp.float64(constraint_scale[frame + 1, velocity_lower_base + row])
            upper_scale = wp.float64(constraint_scale[frame + 1, velocity_upper_base + row])
            result += wp.float64(weights[frame + 1, velocity_lower_base + row]) * lower_scale * lower_scale
            result += wp.float64(weights[frame + 1, velocity_upper_base + row]) * upper_scale * upper_scale
    matrix[frame, row, column] = result
    if row != column and result != wp.float64(0.0):
        wp.atomic_max(segment_coupled, segment, 1)


@wp.kernel
def _ipm_frame_matrix_factor(
    matrix: wp.array3d(dtype=wp.float64),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    failed: wp.array(dtype=wp.int32),
):
    """Factor independent dense frame blocks in place."""
    frame = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0 or segment_offsets[segment + 1] - segment_offsets[segment] == 1:
        return
    width = matrix.shape[1]
    for row in range(width):
        for column in range(row + 1):
            value = matrix[frame, row, column]
            for inner in range(column):
                value -= matrix[frame, row, inner] * matrix[frame, column, inner]
            if row == column:
                if not wp.isfinite(value) or value <= wp.float64(1.0e-24):
                    wp.atomic_max(failed, segment, 1)
                    value = wp.float64(1.0e-24)
                matrix[frame, row, column] = wp.sqrt(value)
            else:
                matrix[frame, row, column] = value / matrix[frame, column, column]


@wp.kernel
def _ipm_temporal_lower_multiply_f32(
    values: wp.array2d(dtype=wp.float32),
    factor: wp.array3d(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    output: wp.array2d(dtype=wp.float32),
):
    """Multiply by the lower same-DOF temporal factor."""
    segment, dof = wp.tid()
    if segment >= segment_count:
        return
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    if enabled[segment] == 0 or end - begin == 1:
        for frame in range(begin, end):
            output[frame, dof] = 0.0
        return
    for frame in range(begin, end):
        result = wp.float64(0.0)
        for lag in range(wp.min(3, frame - begin) + 1):
            result += factor[frame, dof, lag] * wp.float64(values[frame - lag, dof])
        output[frame, dof] = wp.float32(result)


@wp.kernel
def _ipm_temporal_upper_multiply_f32(
    values: wp.array2d(dtype=wp.float32),
    factor: wp.array3d(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    output: wp.array2d(dtype=wp.float32),
):
    """Multiply by the transposed same-DOF temporal factor."""
    segment, dof = wp.tid()
    if segment >= segment_count:
        return
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    if enabled[segment] == 0 or end - begin == 1:
        for frame in range(begin, end):
            output[frame, dof] = 0.0
        return
    for frame in range(begin, end):
        result = wp.float64(0.0)
        for lag in range(wp.min(3, end - frame - 1) + 1):
            result += factor[frame + lag, dof, lag] * wp.float64(values[frame + lag, dof])
        output[frame, dof] = wp.float32(result)


@wp.kernel
def _ipm_frame_solve_f32(
    values: wp.array2d(dtype=wp.float32),
    factor: wp.array3d(dtype=wp.float64),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_coupled: wp.array(dtype=wp.int32),
    active_frames: int,
    work: wp.array2d(dtype=wp.float64),
    output: wp.array2d(dtype=wp.float32),
):
    """Apply independent dense frame-block Cholesky inverses."""
    frame = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    width = values.shape[1]
    if (
        enabled[segment] == 0
        or segment_offsets[segment + 1] - segment_offsets[segment] == 1
        or segment_coupled[segment] == 0
    ):
        for row in range(width):
            work[frame, row] = wp.float64(0.0)
            output[frame, row] = 0.0
        return
    for row in range(width):
        value = wp.float64(values[frame, row])
        for column in range(row):
            value -= factor[frame, row, column] * work[frame, column]
        work[frame, row] = value / factor[frame, row, row]
    for reverse in range(width):
        row = width - reverse - 1
        value = work[frame, row]
        for column in range(row + 1, width):
            value -= factor[frame, column, row] * work[frame, column]
        work[frame, row] = value / factor[frame, row, row]
        output[frame, row] = wp.float32(work[frame, row])


@wp.kernel
def _pcg_additive_precondition_combine(
    residual: wp.array2d(dtype=wp.float32),
    correction: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array2d(dtype=wp.float32),
):
    """Apply the identity-plus-frame correction in transformed coordinates."""
    frame, dof = wp.tid()
    if frame < active_frames:
        output[frame, dof] = (
            residual[frame, dof] + correction[frame, dof] if enabled[frame_segment[frame]] != 0 else 0.0
        )


@wp.kernel
def _pcg_direction_restart(
    preconditioned: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    direction: wp.array2d(dtype=wp.float32),
):
    """Initialize only requested PCG directions from preconditioned residuals."""
    frame, dof = wp.tid()
    if frame < active_frames and enabled[frame_segment[frame]] != 0:
        direction[frame, dof] = preconditioned[frame, dof]


@wp.kernel
def _ipm_singleton_matrix_build(
    jacobian: wp.array3d(dtype=wp.float32),
    base_weights: wp.array2d(dtype=wp.float32),
    residual_activity: wp.array2d(dtype=wp.float32),
    activity_group_by_residual: wp.array(dtype=wp.int32),
    weights: wp.array2d(dtype=wp.float32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    coordinate_bound_count: int,
    constraint_scale: wp.array2d(dtype=wp.float32),
    segment_damping: wp.array(dtype=wp.float32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    matrix: wp.array3d(dtype=wp.float64),
):
    """Build exact frame-local condensed matrices for singleton segments."""
    segment, row, column = wp.tid()
    if segment >= segment_count:
        return
    begin = segment_offsets[segment]
    singleton = segment_offsets[segment + 1] - begin == 1
    if enabled[segment] == 0 or not singleton or column > row:
        matrix[segment, row, column] = wp.float64(1.0) if row == column else wp.float64(0.0)
        return
    residual_count = jacobian.shape[1]
    coordinate_capacity = coordinate_dof_indices.shape[0]
    coordinate_lower_base = residual_count
    coordinate_upper_base = coordinate_lower_base + coordinate_capacity
    result = wp.float64(segment_damping[segment]) if row == column else wp.float64(0.0)
    for residual in range(residual_count):
        activity_group = activity_group_by_residual[residual]
        activity = wp.float64(1.0) if activity_group < 0 else wp.float64(residual_activity[begin, activity_group])
        row_value = wp.float64(jacobian[begin, residual, row])
        column_value = wp.float64(jacobian[begin, residual, column])
        objective_weight = activity * wp.float64(base_weights[begin, residual])
        constraint_row_scale = wp.float64(constraint_scale[begin, residual])
        barrier_weight = wp.float64(weights[begin, residual]) * constraint_row_scale * constraint_row_scale
        result += (objective_weight + barrier_weight) * row_value * column_value
    if row == column:
        for bound in range(coordinate_bound_count):
            if coordinate_dof_indices[bound] == row:
                lower_column = coordinate_lower_base + bound
                upper_column = coordinate_upper_base + bound
                lower_scale = wp.float64(constraint_scale[begin, lower_column])
                upper_scale = wp.float64(constraint_scale[begin, upper_column])
                result += wp.float64(weights[begin, lower_column]) * lower_scale * lower_scale
                result += wp.float64(weights[begin, upper_column]) * upper_scale * upper_scale
    matrix[segment, row, column] = result


@wp.kernel
def _ipm_singleton_matrix_factor(
    matrix: wp.array3d(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    failed: wp.array(dtype=wp.int32),
):
    """Factor exact singleton condensed matrices in place."""
    segment = wp.tid()
    if (
        segment >= segment_count
        or enabled[segment] == 0
        or segment_offsets[segment + 1] - segment_offsets[segment] != 1
    ):
        return
    width = matrix.shape[1]
    for row in range(width):
        for column in range(row + 1):
            value = matrix[segment, row, column]
            for inner in range(column):
                value -= matrix[segment, row, inner] * matrix[segment, column, inner]
            if row == column:
                if not wp.isfinite(value) or value <= wp.float64(1.0e-24):
                    wp.atomic_max(failed, segment, 1)
                    value = wp.float64(1.0e-24)
                matrix[segment, row, column] = wp.sqrt(value)
            else:
                matrix[segment, row, column] = value / matrix[segment, column, column]


@wp.kernel
def _ipm_singleton_forward_f64(
    values: wp.array2d(dtype=wp.float64),
    matrix: wp.array3d(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    output: wp.array2d(dtype=wp.float64),
):
    """Apply an exact singleton lower-factor inverse in float64."""
    segment = wp.tid()
    if segment >= segment_count or segment_offsets[segment + 1] - segment_offsets[segment] != 1:
        return
    frame = segment_offsets[segment]
    for row in range(matrix.shape[1]):
        value = values[frame, row] if enabled[segment] != 0 else wp.float64(0.0)
        for column in range(row):
            value -= matrix[segment, row, column] * output[frame, column]
        output[frame, row] = value / matrix[segment, row, row]


@wp.kernel
def _ipm_singleton_forward_f32(
    values: wp.array2d(dtype=wp.float32),
    matrix: wp.array3d(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    output: wp.array2d(dtype=wp.float32),
):
    """Apply an exact singleton lower-factor inverse in transformed float32 CG."""
    segment = wp.tid()
    if segment >= segment_count or segment_offsets[segment + 1] - segment_offsets[segment] != 1:
        return
    frame = segment_offsets[segment]
    for row in range(matrix.shape[1]):
        value = wp.float64(values[frame, row]) if enabled[segment] != 0 else wp.float64(0.0)
        for column in range(row):
            value -= matrix[segment, row, column] * wp.float64(output[frame, column])
        output[frame, row] = wp.float32(value / matrix[segment, row, row])


@wp.kernel
def _ipm_singleton_backward_f32(
    values: wp.array2d(dtype=wp.float32),
    matrix: wp.array3d(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    output: wp.array2d(dtype=wp.float32),
):
    """Apply an exact singleton transposed-factor inverse in float32 CG."""
    segment = wp.tid()
    if segment >= segment_count or segment_offsets[segment + 1] - segment_offsets[segment] != 1:
        return
    frame = segment_offsets[segment]
    for reverse in range(matrix.shape[1]):
        row = matrix.shape[1] - reverse - 1
        value = wp.float64(values[frame, row]) if enabled[segment] != 0 else wp.float64(0.0)
        for column in range(row + 1, matrix.shape[1]):
            value -= matrix[segment, column, row] * wp.float64(output[frame, column])
        output[frame, row] = wp.float32(value / matrix[segment, row, row])


@wp.kernel
def _ipm_temporal_forward_f64(
    values: wp.array2d(dtype=wp.float64),
    band: wp.array3d(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    output: wp.array2d(dtype=wp.float64),
):
    """Apply the inverse lower temporal factor to double-precision values."""
    segment, dof = wp.tid()
    if segment >= segment_count:
        return
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    if end - begin == 1:
        return
    for frame in range(begin, end):
        value = values[frame, dof] if enabled[segment] != 0 else wp.float64(0.0)
        for lag in range(1, wp.min(3, frame - begin) + 1):
            value -= band[frame, dof, lag] * output[frame - lag, dof]
        output[frame, dof] = value / band[frame, dof, 0]


@wp.kernel
def _ipm_temporal_forward_f32(
    values: wp.array2d(dtype=wp.float32),
    band: wp.array3d(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    output: wp.array2d(dtype=wp.float32),
):
    """Apply the inverse lower temporal factor in transformed float32 CG."""
    segment, dof = wp.tid()
    if segment >= segment_count:
        return
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    if end - begin == 1:
        return
    for frame in range(begin, end):
        value = wp.float64(values[frame, dof]) if enabled[segment] != 0 else wp.float64(0.0)
        for lag in range(1, wp.min(3, frame - begin) + 1):
            value -= band[frame, dof, lag] * wp.float64(output[frame - lag, dof])
        output[frame, dof] = wp.float32(value / band[frame, dof, 0])


@wp.kernel
def _ipm_temporal_backward_f32(
    values: wp.array2d(dtype=wp.float32),
    band: wp.array3d(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    output: wp.array2d(dtype=wp.float32),
):
    """Apply the inverse transposed temporal factor in transformed float32 CG."""
    segment, dof = wp.tid()
    if segment >= segment_count:
        return
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    if end - begin == 1:
        return
    for reverse in range(end - begin):
        frame = end - reverse - 1
        value = wp.float64(values[frame, dof]) if enabled[segment] != 0 else wp.float64(0.0)
        for lag in range(1, wp.min(3, end - frame - 1) + 1):
            value -= band[frame + lag, dof, lag] * wp.float64(output[frame + lag, dof])
        output[frame, dof] = wp.float32(value / band[frame, dof, 0])


@wp.kernel
def _ipm_feature_f64(
    jacobian: wp.array3d(dtype=wp.float32),
    values: wp.array2d(dtype=wp.float64),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array2d(dtype=wp.float64),
):
    """Apply the float32 Jacobian with float64 accumulation."""
    frame, residual = wp.tid()
    if frame >= active_frames:
        return
    if enabled[frame_segment[frame]] == 0:
        output[frame, residual] = wp.float64(0.0)
        return
    result = wp.float64(0.0)
    for dof in range(jacobian.shape[2]):
        result += wp.float64(jacobian[frame, residual, dof]) * values[frame, dof]
    output[frame, residual] = result


@wp.kernel
def _precision_apply_f64(
    values: wp.array2d(dtype=wp.float64),
    base_weights: wp.array2d(dtype=wp.float32),
    temporal_weights: wp.array2d(dtype=wp.float32),
    residual_activity: wp.array2d(dtype=wp.float32),
    activity_group_by_residual: wp.array(dtype=wp.int32),
    first_difference_group_by_residual: wp.array(dtype=wp.int32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    step_seconds: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array2d(dtype=wp.float64),
):
    """Apply residual precision with float64 accumulation."""
    frame, residual = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0:
        output[frame, residual] = wp.float64(0.0)
        return
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    dt = wp.float64(step_seconds[segment])
    activity_group = activity_group_by_residual[residual]
    first_difference_group = first_difference_group_by_residual[residual]
    frame_confidence = wp.float64(1.0) if activity_group < 0 else wp.float64(residual_activity[frame, activity_group])
    result = frame_confidence * wp.float64(base_weights[frame, residual]) * values[frame, residual]
    for order in range(1, 4):
        weight = temporal_weights[order - 1, residual]
        if weight != 0.0:
            inverse_dt = wp.float64(1.0) / dt
            scale = wp.float64(weight)
            for _ in range(order):
                scale *= inverse_dt * inverse_dt
            first_start = wp.max(begin, frame - order)
            last_start = wp.min(frame, end - order - 1)
            for start in range(first_start, last_start + 1):
                stencil_confidence = wp.float64(
                    _temporal_stencil_confidence(
                        residual_activity, activity_group, first_difference_group, start, order
                    )
                )
                if stencil_confidence <= wp.float64(0.0):
                    continue
                difference = wp.float64(0.0)
                for index in range(order + 1):
                    difference += wp.float64(_difference_coefficient(order, index)) * values[start + index, residual]
                result += (
                    stencil_confidence * scale * wp.float64(_difference_coefficient(order, frame - start)) * difference
                )
    output[frame, residual] = result


@wp.kernel
def _normal_apply_f64(
    jacobian: wp.array3d(dtype=wp.float32),
    precision_values: wp.array2d(dtype=wp.float64),
    values: wp.array2d(dtype=wp.float64),
    segment_damping: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array2d(dtype=wp.float64),
):
    """Apply J-transpose P J plus damping with float64 accumulation."""
    frame, dof = wp.tid()
    if frame >= active_frames:
        return
    if enabled[frame_segment[frame]] == 0:
        output[frame, dof] = wp.float64(0.0)
        return
    result = wp.float64(segment_damping[frame_segment[frame]]) * values[frame, dof]
    for residual in range(jacobian.shape[1]):
        result += wp.float64(jacobian[frame, residual, dof]) * precision_values[frame, residual]
    output[frame, dof] = result


@wp.kernel
def _ipm_barrier_add_f64(
    jacobian: wp.array3d(dtype=wp.float32),
    feature: wp.array2d(dtype=wp.float64),
    values: wp.array2d(dtype=wp.float64),
    weights: wp.array2d(dtype=wp.float32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    coordinate_bound_count: int,
    constraint_scale: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array2d(dtype=wp.float64),
):
    """Add C-transpose W C values with float64 accumulation."""
    frame, dof = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0:
        output[frame, dof] = wp.float64(0.0)
        return
    residual_count = jacobian.shape[1]
    coordinate_capacity = coordinate_dof_indices.shape[0]
    dof_count = jacobian.shape[2]
    coordinate_lower_base = residual_count
    coordinate_upper_base = coordinate_lower_base + coordinate_capacity
    velocity_lower_base = coordinate_upper_base + coordinate_capacity
    velocity_upper_base = velocity_lower_base + dof_count
    result = output[frame, dof]
    for residual in range(residual_count):
        row_scale = wp.float64(constraint_scale[frame, residual])
        coefficient = row_scale * wp.float64(jacobian[frame, residual, dof])
        result += coefficient * wp.float64(weights[frame, residual]) * row_scale * feature[frame, residual]
    for bound in range(coordinate_bound_count):
        if coordinate_dof_indices[bound] == dof:
            lower_column = coordinate_lower_base + bound
            upper_column = coordinate_upper_base + bound
            lower_scale = wp.float64(constraint_scale[frame, lower_column])
            upper_scale = wp.float64(constraint_scale[frame, upper_column])
            result += wp.float64(weights[frame, lower_column]) * lower_scale * lower_scale * values[frame, dof]
            result += wp.float64(weights[frame, upper_column]) * upper_scale * upper_scale * values[frame, dof]
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    if frame > begin:
        difference = values[frame, dof] - values[frame - 1, dof]
        lower_scale = wp.float64(constraint_scale[frame, velocity_lower_base + dof])
        upper_scale = wp.float64(constraint_scale[frame, velocity_upper_base + dof])
        result += wp.float64(weights[frame, velocity_lower_base + dof]) * lower_scale * lower_scale * difference
        result += wp.float64(weights[frame, velocity_upper_base + dof]) * upper_scale * upper_scale * difference
    if frame + 1 < end:
        difference = values[frame, dof] - values[frame + 1, dof]
        lower_scale = wp.float64(constraint_scale[frame + 1, velocity_lower_base + dof])
        upper_scale = wp.float64(constraint_scale[frame + 1, velocity_upper_base + dof])
        result += wp.float64(weights[frame + 1, velocity_lower_base + dof]) * lower_scale * lower_scale * difference
        result += wp.float64(weights[frame + 1, velocity_upper_base + dof]) * upper_scale * upper_scale * difference
    output[frame, dof] = result


@wp.kernel
def _ipm_physical_residual_f64(
    right_hand_side: wp.array2d(dtype=wp.float32),
    operator_value: wp.array2d(dtype=wp.float64),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    residual: wp.array2d(dtype=wp.float64),
):
    """Reconstruct the physical condensed residual in float64."""
    frame, dof = wp.tid()
    if frame < active_frames:
        residual[frame, dof] = (
            wp.float64(right_hand_side[frame, dof]) - operator_value[frame, dof]
            if enabled[frame_segment[frame]] != 0
            else wp.float64(0.0)
        )


@wp.kernel
def _ipm_physical_residual_from_f64(
    right_hand_side: wp.array2d(dtype=wp.float64),
    operator_value: wp.array2d(dtype=wp.float64),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    residual: wp.array2d(dtype=wp.float64),
):
    """Reconstruct a physical condensed residual from a float64 right-hand side."""
    frame, dof = wp.tid()
    if frame < active_frames:
        residual[frame, dof] = (
            right_hand_side[frame, dof] - operator_value[frame, dof]
            if enabled[frame_segment[frame]] != 0
            else wp.float64(0.0)
        )


@wp.kernel
def _ipm_outer_forcing_update(
    physical_residual: wp.array2d(dtype=wp.float64),
    stationarity: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    requested: wp.array(dtype=wp.int32),
    numerical_failed: wp.array(dtype=wp.int32),
    relative_tolerance: float,
    absolute_tolerance: float,
    active_frames: int,
    enabled: wp.array(dtype=wp.int32),
):
    """Reject a physical certificate whose residual can spoil outer stationarity."""
    frame, dof = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if requested[segment] == 0 or numerical_failed[segment] != 0:
        return
    residual = physical_residual[frame, dof]
    dual_residual = wp.float64(stationarity[frame, dof])
    threshold = wp.max(
        wp.float64(absolute_tolerance),
        wp.float64(relative_tolerance) * wp.abs(dual_residual),
    )
    if not wp.isfinite(residual) or not wp.isfinite(dual_residual) or wp.abs(residual) > threshold:
        wp.atomic_max(enabled, segment, 1)


@wp.kernel
def _ipm_rhs_stationarity_f64(
    normal: wp.array2d(dtype=wp.float64),
    gradient: wp.array2d(dtype=wp.float32),
    inequality_transpose: wp.array2d(dtype=wp.float64),
    equality_transpose: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    stationarity: wp.array2d(dtype=wp.float64),
    stationarity_f32: wp.array2d(dtype=wp.float32),
):
    """Assemble the Phase-II dual residual with float64 accumulation."""
    frame, dof = wp.tid()
    if frame < active_frames:
        value = wp.float64(0.0)
        if enabled[frame_segment[frame]] != 0:
            value = normal[frame, dof] + wp.float64(gradient[frame, dof])
            value += inequality_transpose[frame, dof] + wp.float64(equality_transpose[frame, dof])
        stationarity[frame, dof] = value
        stationarity_f32[frame, dof] = wp.float32(value)


@wp.kernel
def _ipm_rhs_condense_f64(
    stationarity: wp.array2d(dtype=wp.float64),
    condensed_transpose: wp.array2d(dtype=wp.float64),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    right_hand_side: wp.array2d(dtype=wp.float64),
    right_hand_side_f32: wp.array2d(dtype=wp.float32),
):
    """Assemble the condensed Phase-II RHS without a float32 publication boundary."""
    frame, dof = wp.tid()
    if frame < active_frames:
        value = wp.float64(0.0)
        if enabled[frame_segment[frame]] != 0:
            value = condensed_transpose[frame, dof] - stationarity[frame, dof]
        right_hand_side[frame, dof] = value
        right_hand_side_f32[frame, dof] = wp.float32(value)


@wp.kernel
def _ipm_solution_accumulate_f64(
    correction: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    solution: wp.array2d(dtype=wp.float64),
):
    """Accumulate one float32 correction into the float64 physical solution."""
    frame, dof = wp.tid()
    if frame < active_frames and enabled[frame_segment[frame]] != 0:
        solution[frame, dof] += wp.float64(correction[frame, dof])


@wp.kernel
def _ipm_solution_accumulate_from_f64(
    correction: wp.array2d(dtype=wp.float64),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    solution: wp.array2d(dtype=wp.float64),
):
    """Accumulate one double-precision correction into the physical solution."""
    frame, dof = wp.tid()
    if frame < active_frames and enabled[frame_segment[frame]] != 0:
        solution[frame, dof] += correction[frame, dof]


@wp.kernel
def _ipm_solution_canonicalize_f32(
    solution: wp.array2d(dtype=wp.float64),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
):
    """Round active physical solutions to their published float32 values."""
    frame, dof = wp.tid()
    if frame < active_frames and enabled[frame_segment[frame]] != 0:
        solution[frame, dof] = wp.float64(wp.float32(solution[frame, dof]))


@wp.kernel
def _ipm_solution_copy_f32(
    solution: wp.array2d(dtype=wp.float64),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array2d(dtype=wp.float32),
):
    """Narrow the certified physical solution for the IPM iterate update."""
    frame, dof = wp.tid()
    if frame < active_frames:
        output[frame, dof] = wp.float32(solution[frame, dof]) if enabled[frame_segment[frame]] != 0 else wp.float32(0.0)


@wp.kernel
def _square_norm_f64(
    values: wp.array2d(dtype=wp.float64),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array(dtype=wp.float64),
):
    """Write one deterministic double-precision squared norm per frame."""
    frame = wp.tid()
    if frame >= active_frames:
        return
    result = wp.float64(0.0)
    if enabled[frame_segment[frame]] != 0:
        for column in range(values.shape[1]):
            value = values[frame, column]
            result += value * value
    output[frame] = result


@wp.kernel
def _dot_segments_f64(
    frame_values: wp.array(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    segment_count: int,
    output: wp.array(dtype=wp.float64),
):
    """Reduce deterministic double-precision frame values by segment."""
    segment = wp.tid()
    if segment >= segment_count:
        return
    result = wp.float64(0.0)
    for frame in range(segment_offsets[segment], segment_offsets[segment + 1]):
        result += frame_values[frame]
    output[segment] = result


@wp.kernel
def _ipm_physical_convergence_initialize(
    residual_dot: wp.array(dtype=wp.float64),
    requested: wp.array(dtype=wp.int32),
    segment_count: int,
    numerical_epsilon: float,
    initial_norm: wp.array(dtype=wp.float64),
    enabled: wp.array(dtype=wp.int32),
    failed: wp.array(dtype=wp.int32),
):
    """Initialize physical mixed-precision residual certification."""
    segment = wp.tid()
    if segment >= segment_count:
        return
    residual_squared = residual_dot[segment]
    invalid = not wp.isfinite(residual_squared) or residual_squared < -wp.float64(numerical_epsilon)
    residual_norm = wp.float64(0.0) if invalid else wp.sqrt(wp.max(residual_squared, wp.float64(0.0)))
    initial_norm[segment] = residual_norm
    factor_failed = failed[segment] != 0
    failed[segment] = 1 if factor_failed or (requested[segment] != 0 and invalid) else 0
    enabled[segment] = (
        1
        if requested[segment] != 0
        and not factor_failed
        and not invalid
        and residual_norm > wp.float64(numerical_epsilon)
        else 0
    )


@wp.kernel
def _ipm_physical_reference_norm_f64(
    operator_squared: wp.array(dtype=wp.float64),
    right_hand_side: wp.array2d(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    reference_norm: wp.array(dtype=wp.float64),
):
    """Build the symmetric physical residual scale ``||b|| + ||Ax||``."""
    segment = wp.tid()
    if segment >= segment_count:
        return
    if enabled[segment] == 0:
        reference_norm[segment] = wp.float64(0.0)
        return
    right_hand_side_squared = wp.float64(0.0)
    for frame in range(segment_offsets[segment], segment_offsets[segment + 1]):
        for dof in range(right_hand_side.shape[1]):
            value = right_hand_side[frame, dof]
            right_hand_side_squared += value * value
    operator_norm_squared = operator_squared[segment]
    invalid = (
        not wp.isfinite(operator_norm_squared)
        or operator_norm_squared < wp.float64(0.0)
        or not wp.isfinite(right_hand_side_squared)
        or right_hand_side_squared < wp.float64(0.0)
    )
    reference_norm[segment] = (
        wp.float64(-1.0) if invalid else wp.sqrt(operator_norm_squared) + wp.sqrt(right_hand_side_squared)
    )


@wp.kernel
def _ipm_physical_convergence_update(
    residual_dot: wp.array(dtype=wp.float64),
    reference_norm: wp.array(dtype=wp.float64),
    relative_tolerance: float,
    numerical_epsilon: float,
    enabled: wp.array(dtype=wp.int32),
    failed: wp.array(dtype=wp.int32),
):
    """Certify a correction only from its fresh physical residual."""
    segment = wp.tid()
    if enabled[segment] == 0:
        return
    residual_squared = residual_dot[segment]
    scale = reference_norm[segment]
    if (
        not wp.isfinite(residual_squared)
        or residual_squared < -wp.float64(numerical_epsilon)
        or not wp.isfinite(scale)
        or scale < wp.float64(0.0)
    ):
        failed[segment] = 1
        enabled[segment] = 0
        return
    residual_norm = wp.sqrt(wp.max(residual_squared, wp.float64(0.0)))
    threshold = wp.max(
        wp.float64(numerical_epsilon),
        wp.float64(relative_tolerance) * scale,
    )
    if residual_norm <= threshold:
        enabled[segment] = 0


@wp.kernel
def _ipm_fallback_promote_factor_failure(
    requested: wp.array(dtype=wp.int32),
    failed: wp.array(dtype=wp.int32),
    segment_count: int,
    fallback: wp.array(dtype=wp.int32),
    promoted: wp.array(dtype=wp.int32),
):
    """Promote provisional full-normal factor failures to the stable route."""
    segment = wp.tid()
    if segment >= segment_count:
        return
    use_fallback = requested[segment] != 0 and failed[segment] != 0
    promoted[segment] = 1 if use_fallback else 0
    if use_fallback:
        fallback[segment] = 1
        failed[segment] = 0


@wp.kernel
def _ipm_route_select(
    active: wp.array(dtype=wp.int32),
    fallback: wp.array(dtype=wp.int32),
    failed: wp.array(dtype=wp.int32),
    select_fallback: wp.uint8,
    segment_count: int,
    selected: wp.array(dtype=wp.int32),
):
    """Select active direct or stable-route segments into caller-owned work."""
    segment = wp.tid()
    if segment < segment_count:
        matches = (fallback[segment] != 0) == (select_fallback != wp.uint8(0))
        selected[segment] = 1 if active[segment] != 0 and failed[segment] == 0 and matches else 0


@wp.kernel
def _ipm_fallback_promote_certificate_failure(
    requested: wp.array(dtype=wp.int32),
    fallback: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    failed: wp.array(dtype=wp.int32),
    segment_count: int,
    promoted: wp.array(dtype=wp.int32),
):
    """Promote only uncertified direct solves and preserve stable-route failures."""
    segment = wp.tid()
    if segment >= segment_count:
        return
    use_fallback = (
        requested[segment] != 0 and fallback[segment] == 0 and (enabled[segment] != 0 or failed[segment] != 0)
    )
    promoted[segment] = 1 if use_fallback else 0
    if use_fallback:
        fallback[segment] = 1
        enabled[segment] = 1
        failed[segment] = 0


@wp.kernel
def _ipm_certification_select(
    requested: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    recursive_converged: wp.array(dtype=wp.int32),
    failed: wp.array(dtype=wp.int32),
    segment_count: int,
    certify: wp.array(dtype=wp.int32),
):
    """Select recursively converged segments for physical certification."""
    segment = wp.tid()
    if segment < segment_count:
        certify[segment] = (
            1
            if requested[segment] != 0
            and enabled[segment] != 0
            and recursive_converged[segment] != 0
            and failed[segment] == 0
            else 0
        )


@wp.kernel
def _ipm_certification_resolve(
    certify: wp.array(dtype=wp.int32),
    recursive_converged: wp.array(dtype=wp.int32),
    segment_count: int,
    requested: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
):
    """Accept a physical certificate or request a reliable residual restart."""
    segment = wp.tid()
    if segment >= segment_count or requested[segment] == 0 or recursive_converged[segment] == 0:
        return
    recursive_converged[segment] = 0
    if certify[segment] == 0:
        requested[segment] = 0
        enabled[segment] = 0


@wp.kernel
def _pcg_reliable_restart(
    physical_residual: wp.array2d(dtype=wp.float64),
    frame_segment: wp.array(dtype=wp.int32),
    restart: wp.array(dtype=wp.int32),
    active_frames: int,
    delta: wp.array2d(dtype=wp.float32),
    residual: wp.array2d(dtype=wp.float32),
    preconditioned: wp.array2d(dtype=wp.float32),
    direction: wp.array2d(dtype=wp.float32),
):
    """Restart only a published solution that failed its physical certificate."""
    frame, dof = wp.tid()
    if frame >= active_frames or restart[frame_segment[frame]] == 0:
        return
    value = wp.float32(physical_residual[frame, dof])
    delta[frame, dof] = 0.0
    residual[frame, dof] = value
    preconditioned[frame, dof] = value
    direction[frame, dof] = value


@wp.kernel
def _pcg_reliable_restart_state(
    residual_dot: wp.array(dtype=wp.float32),
    restart: wp.array(dtype=wp.int32),
    segment_count: int,
    state: wp.array2d(dtype=wp.float32),
):
    """Reset recursive CG norms only for a reliable residual restart."""
    segment = wp.tid()
    if segment >= segment_count or restart[segment] == 0:
        return
    residual_squared = residual_dot[segment]
    state[segment, 0] = wp.sqrt(wp.max(residual_squared, 0.0))


@wp.kernel
def _ipm_iterate_initialize(
    constraint_value: wp.array2d(dtype=wp.float32),
    constraint_rhs: wp.array2d(dtype=wp.float32),
    constraint_scale: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    slack: wp.array2d(dtype=wp.float32),
    multiplier: wp.array2d(dtype=wp.float32),
):
    """Initialize a strictly positive infeasible-start primal-dual point."""
    frame, column = wp.tid()
    if frame >= active_frames:
        return
    active = enabled[frame_segment[frame]] != 0 and constraint_scale[frame, column] != 0.0
    if active:
        initial_slack = wp.max(1.0, wp.abs(constraint_rhs[frame, column] - constraint_value[frame, column]))
        slack[frame, column] = initial_slack
        multiplier[frame, column] = 1.0 / initial_slack
    else:
        slack[frame, column] = 1.0
        multiplier[frame, column] = 0.0


@wp.kernel
def _phase_one_constraints_initialize(
    row_code: wp.array2d(dtype=wp.int32),
    row_scale: wp.array2d(dtype=wp.float32),
    equality_rhs: wp.array2d(dtype=wp.float32),
    equality_count: wp.array(dtype=wp.int32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    inequality_width: int,
    constraint_scale: wp.array2d(dtype=wp.float32),
    constraint_rhs: wp.array2d(dtype=wp.float32),
    constraint_count: wp.array(dtype=wp.int32),
):
    """Append signed equality elastics and one nonnegative elastic scalar row."""
    frame, column = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    equality_width = row_code.shape[1]
    full_column = inequality_width + column
    if enabled[segment] == 0:
        constraint_scale[frame, full_column] = 0.0
        constraint_rhs[frame, full_column] = 0.0
        return
    if column < 2 * equality_width:
        local = column if column < equality_width else column - equality_width
        slot = frame * equality_width + local
        active = _qp_slot_active(slot, segment, segment_offsets, row_code, equality_count)
        if active:
            sign = 1.0 if column < equality_width else -1.0
            constraint_scale[frame, full_column] = row_scale[frame, local]
            constraint_rhs[frame, full_column] = sign * equality_rhs[frame, local]
            wp.atomic_add(constraint_count, segment, 1)
        else:
            constraint_scale[frame, full_column] = 0.0
            constraint_rhs[frame, full_column] = 0.0
    else:
        active = frame == segment_offsets[segment]
        constraint_scale[frame, full_column] = 1.0 if active else 0.0
        constraint_rhs[frame, full_column] = 0.0
        if active:
            wp.atomic_add(constraint_count, segment, 1)


@wp.kernel
def _phase_one_constraints_apply(
    equality_value: wp.array2d(dtype=wp.float32),
    elastic: wp.array(dtype=wp.float32),
    constraint_scale: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    inequality_width: int,
    output: wp.array2d(dtype=wp.float32),
):
    """Add the elastic scalar to original inequalities and signed equalities."""
    frame, column = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0 or constraint_scale[frame, column] == 0.0:
        output[frame, column] = 0.0
        return
    equality_width = equality_value.shape[1]
    value = -elastic[segment]
    if column < inequality_width:
        value += output[frame, column]
    elif column < inequality_width + equality_width:
        value += equality_value[frame, column - inequality_width]
    elif column < inequality_width + 2 * equality_width:
        value -= equality_value[frame, column - inequality_width - equality_width]
    output[frame, column] = value


@wp.kernel
def _phase_one_equality_transpose_values(
    values: wp.array2d(dtype=wp.float32),
    inequality_width: int,
    row_code: wp.array2d(dtype=wp.int32),
    equality_count: wp.array(dtype=wp.int32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array2d(dtype=wp.float32),
):
    """Map signed Phase-I equality multipliers back to compact equality slots."""
    frame, column = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    width = row_code.shape[1]
    slot = frame * width + column
    if enabled[segment] != 0 and _qp_slot_active(slot, segment, segment_offsets, row_code, equality_count):
        output[frame, column] = (
            values[frame, inequality_width + column] - values[frame, inequality_width + width + column]
        )
    else:
        output[frame, column] = 0.0


@wp.kernel
def _phase_one_scalar_transpose(
    values: wp.array2d(dtype=wp.float32),
    constraint_scale: wp.array2d(dtype=wp.float32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    output: wp.array(dtype=wp.float32),
):
    """Apply the transpose of the shared negative elastic coefficient."""
    segment = wp.tid()
    if segment >= segment_count or enabled[segment] == 0:
        output[segment] = 0.0
        return
    result = float(0.0)
    for frame in range(segment_offsets[segment], segment_offsets[segment + 1]):
        for column in range(values.shape[1]):
            if constraint_scale[frame, column] != 0.0:
                result -= values[frame, column]
    output[segment] = result


@wp.kernel
def _phase_one_scalar_rhs(
    multiplier_transpose: wp.array(dtype=wp.float32),
    condensed_transpose: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    dual_residual: wp.array(dtype=wp.float32),
    right_hand_side: wp.array(dtype=wp.float32),
):
    """Build elastic stationarity without overwriting a frozen certificate."""
    segment = wp.tid()
    if segment >= segment_count:
        return
    if enabled[segment] != 0:
        rd = 1.0 + multiplier_transpose[segment]
        dual_residual[segment] = rd
        right_hand_side[segment] = -rd + condensed_transpose[segment]
    else:
        right_hand_side[segment] = 0.0


@wp.kernel
def _phase_one_scalar_pcg_initialize(
    right_hand_side: wp.array(dtype=wp.float32),
    diagonal: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    solution: wp.array(dtype=wp.float32),
    residual: wp.array(dtype=wp.float32),
    preconditioned: wp.array(dtype=wp.float32),
    direction: wp.array(dtype=wp.float32),
):
    """Initialize the scalar block of a combined Phase-I PCG solve."""
    segment = wp.tid()
    if segment >= segment_count:
        return
    active = enabled[segment] != 0
    value = right_hand_side[segment] if active else 0.0
    z = value / diagonal[segment] if active else 0.0
    solution[segment] = 0.0
    residual[segment] = value
    preconditioned[segment] = z
    direction[segment] = z


@wp.kernel
def _phase_one_scalar_pcg_update(
    operator_direction: wp.array(dtype=wp.float32),
    residual_dot: wp.array(dtype=wp.float32),
    operator_dot: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    direction: wp.array(dtype=wp.float32),
    solution: wp.array(dtype=wp.float32),
    residual: wp.array(dtype=wp.float32),
):
    """Apply the PCG alpha step to the Phase-I scalar block."""
    segment = wp.tid()
    if segment >= segment_count or enabled[segment] == 0:
        return
    denominator = operator_dot[segment]
    alpha = residual_dot[segment] / denominator if wp.abs(denominator) > 1.0e-20 else 0.0
    solution[segment] += alpha * direction[segment]
    residual[segment] -= alpha * operator_direction[segment]


@wp.kernel
def _phase_one_scalar_precondition(
    residual: wp.array(dtype=wp.float32),
    diagonal: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    output: wp.array(dtype=wp.float32),
):
    segment = wp.tid()
    if segment < segment_count:
        output[segment] = residual[segment] / diagonal[segment] if enabled[segment] != 0 else 0.0


@wp.kernel
def _phase_one_scalar_direction_update(
    preconditioned: wp.array(dtype=wp.float32),
    residual_dot: wp.array(dtype=wp.float32),
    residual_dot_next: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    direction: wp.array(dtype=wp.float32),
):
    segment = wp.tid()
    if segment >= segment_count:
        return
    if enabled[segment] == 0:
        direction[segment] = 0.0
        return
    denominator = residual_dot[segment]
    beta = residual_dot_next[segment] / denominator if wp.abs(denominator) > 1.0e-20 else 0.0
    direction[segment] = preconditioned[segment] + beta * direction[segment]


@wp.kernel
def _phase_one_scalar_diagonal(
    weights: wp.array2d(dtype=wp.float32),
    constraint_scale: wp.array2d(dtype=wp.float32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_damping: wp.array(dtype=wp.float32),
    segment_count: int,
    output: wp.array(dtype=wp.float32),
):
    """Build the elastic scalar diagonal of the condensed Phase-I system."""
    segment = wp.tid()
    if segment >= segment_count or enabled[segment] == 0:
        output[segment] = 1.0
        return
    result = segment_damping[segment]
    for frame in range(segment_offsets[segment], segment_offsets[segment + 1]):
        for column in range(weights.shape[1]):
            if constraint_scale[frame, column] != 0.0:
                result += weights[frame, column]
    output[segment] = wp.max(result, 1.0e-12)


@wp.kernel
def _phase_one_arrowhead_cross_build(
    jacobian: wp.array3d(dtype=wp.float32),
    weights: wp.array2d(dtype=wp.float32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    coordinate_bound_count: int,
    constraint_scale: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    cross: wp.array2d(dtype=wp.float64),
    output: wp.array2d(dtype=wp.float64),
):
    """Build and preserve the primal-to-elastic coupling ``-C-transpose W 1``."""
    frame, dof = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0:
        cross[frame, dof] = wp.float64(0.0)
        output[frame, dof] = wp.float64(0.0)
        return
    residual_count = jacobian.shape[1]
    coordinate_capacity = coordinate_dof_indices.shape[0]
    dof_count = jacobian.shape[2]
    coordinate_lower_base = residual_count
    coordinate_upper_base = coordinate_lower_base + coordinate_capacity
    velocity_lower_base = coordinate_upper_base + coordinate_capacity
    velocity_upper_base = velocity_lower_base + dof_count
    result = wp.float64(0.0)
    for residual in range(residual_count):
        result -= (
            wp.float64(jacobian[frame, residual, dof])
            * wp.float64(constraint_scale[frame, residual])
            * wp.float64(weights[frame, residual])
        )
    for bound in range(coordinate_bound_count):
        if coordinate_dof_indices[bound] == dof:
            lower_column = coordinate_lower_base + bound
            upper_column = coordinate_upper_base + bound
            result += wp.float64(constraint_scale[frame, lower_column]) * wp.float64(weights[frame, lower_column])
            result -= wp.float64(constraint_scale[frame, upper_column]) * wp.float64(weights[frame, upper_column])
    begin = segment_offsets[segment]
    end = segment_offsets[segment + 1]
    if frame > begin:
        result += wp.float64(constraint_scale[frame, velocity_lower_base + dof]) * wp.float64(
            weights[frame, velocity_lower_base + dof]
        )
        result -= wp.float64(constraint_scale[frame, velocity_upper_base + dof]) * wp.float64(
            weights[frame, velocity_upper_base + dof]
        )
    if frame + 1 < end:
        result -= wp.float64(constraint_scale[frame + 1, velocity_lower_base + dof]) * wp.float64(
            weights[frame + 1, velocity_lower_base + dof]
        )
        result += wp.float64(constraint_scale[frame + 1, velocity_upper_base + dof]) * wp.float64(
            weights[frame + 1, velocity_upper_base + dof]
        )
    cross[frame, dof] = result
    output[frame, dof] = result


@wp.kernel
def _phase_one_arrowhead_pair_dot(
    cross: wp.array2d(dtype=wp.float64),
    primal_coupling: wp.array2d(dtype=wp.float64),
    primal_particular: wp.array2d(dtype=wp.float64),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array2d(dtype=wp.float64),
):
    """Write deterministic ``(c-transpose u, c-transpose v)`` pairs per frame."""
    frame = wp.tid()
    if frame >= active_frames:
        return
    coupling_coupling = wp.float64(0.0)
    coupling_particular = wp.float64(0.0)
    if enabled[frame_segment[frame]] != 0:
        for dof in range(cross.shape[1]):
            coefficient = cross[frame, dof]
            coupling_coupling += coefficient * primal_coupling[frame, dof]
            coupling_particular += coefficient * primal_particular[frame, dof]
    output[frame, 0] = coupling_coupling
    output[frame, 1] = coupling_particular


@wp.kernel
def _phase_one_arrowhead_pair_reduce(
    frame_pairs: wp.array2d(dtype=wp.float64),
    segment_offsets: wp.array(dtype=wp.int32),
    segment_count: int,
    output: wp.array2d(dtype=wp.float64),
):
    """Reduce arrowhead dot-product pairs deterministically by segment."""
    segment = wp.tid()
    if segment >= segment_count:
        return
    coupling_coupling = wp.float64(0.0)
    coupling_particular = wp.float64(0.0)
    for frame in range(segment_offsets[segment], segment_offsets[segment + 1]):
        coupling_coupling += frame_pairs[frame, 0]
        coupling_particular += frame_pairs[frame, 1]
    output[segment, 0] = coupling_coupling
    output[segment, 1] = coupling_particular


@wp.kernel
def _phase_one_arrowhead_scalar_solve(
    dot_pairs: wp.array2d(dtype=wp.float64),
    scalar_diagonal: wp.array(dtype=wp.float32),
    scalar_rhs: wp.array(dtype=wp.float32),
    segment_count: int,
    failed: wp.array(dtype=wp.int32),
    scalar_dual_residual: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    direction_f64: wp.array(dtype=wp.float64),
    direction: wp.array(dtype=wp.float32),
):
    """Solve the elastic scalar from the precomputed arrowhead dot products."""
    segment = wp.tid()
    if segment >= segment_count:
        return
    if enabled[segment] == 0:
        direction_f64[segment] = wp.float64(0.0)
        direction[segment] = 0.0
        return
    diagonal = wp.float64(scalar_diagonal[segment])
    coupling_coupling = dot_pairs[segment, 0]
    coupling_particular = dot_pairs[segment, 1]
    schur = diagonal - coupling_coupling
    numerator = wp.float64(scalar_rhs[segment]) - coupling_particular
    invalid = (
        failed[segment] != 0
        or not wp.isfinite(diagonal)
        or not wp.isfinite(coupling_particular)
        or not wp.isfinite(coupling_coupling)
        or not wp.isfinite(schur)
        or not wp.isfinite(numerator)
        or schur <= wp.float64(0.0)
    )
    elastic_direction = wp.float64(0.0) if invalid else numerator / schur
    published_elastic = wp.float32(elastic_direction)
    invalid = invalid or not wp.isfinite(elastic_direction) or not wp.isfinite(published_elastic)
    if invalid:
        failed[segment] = 1
        scalar_dual_residual[segment] = 3.402823466e38
        enabled[segment] = 0
        elastic_direction = wp.float64(0.0)
        published_elastic = 0.0
    direction_f64[segment] = elastic_direction
    direction[segment] = published_elastic


@wp.kernel
def _phase_one_arrowhead_failure_finalize(
    failed: wp.array(dtype=wp.int32),
    segment_count: int,
    scalar_dual_residual: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    direction_f64: wp.array(dtype=wp.float64),
    direction: wp.array(dtype=wp.float32),
):
    """Fail closed after the parallel published-primal validation pass."""
    segment = wp.tid()
    if segment < segment_count and failed[segment] != 0:
        scalar_dual_residual[segment] = 3.402823466e38
        enabled[segment] = 0
        direction_f64[segment] = wp.float64(0.0)
        direction[segment] = 0.0


@wp.kernel
def _phase_one_arrowhead_primal_recover(
    primal_particular: wp.array2d(dtype=wp.float64),
    primal_coupling: wp.array2d(dtype=wp.float64),
    elastic_direction: wp.array(dtype=wp.float64),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    failed: wp.array(dtype=wp.int32),
    direction: wp.array2d(dtype=wp.float32),
):
    """Recover and validate the published primal direction in parallel."""
    frame, dof = wp.tid()
    if frame < active_frames:
        segment = frame_segment[frame]
        value = wp.float64(0.0)
        published = wp.float32(0.0)
        if enabled[segment] != 0:
            value = primal_particular[frame, dof] - primal_coupling[frame, dof] * elastic_direction[segment]
            published = wp.float32(value)
            if not wp.isfinite(value) or not wp.isfinite(published):
                wp.atomic_max(failed, segment, 1)
                published = wp.float32(0.0)
        direction[frame, dof] = published


@wp.kernel
def _phase_one_combined_dot_add(
    a: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    output: wp.array(dtype=wp.float32),
):
    segment = wp.tid()
    if segment < segment_count and enabled[segment] != 0:
        output[segment] += a[segment] * b[segment]


@wp.kernel
def _phase_one_scalar_max_add(
    values: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    output: wp.array(dtype=wp.float32),
):
    segment = wp.tid()
    if segment < segment_count and enabled[segment] != 0:
        wp.atomic_max(output, segment, wp.abs(values[segment]))


@wp.kernel
def _phase_one_elastic_update(
    direction: wp.array(dtype=wp.float32),
    step: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    elastic: wp.array(dtype=wp.float32),
):
    segment = wp.tid()
    if segment < segment_count and enabled[segment] != 0:
        elastic[segment] += step[segment] * direction[segment]


@wp.kernel
def _phase_one_finalize(
    feasibility_violation: wp.array(dtype=wp.float32),
    dual_residual: wp.array(dtype=wp.float32),
    scalar_dual_residual: wp.array(dtype=wp.float32),
    primal_residual: wp.array(dtype=wp.float32),
    separating_margin: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    segment_active: wp.array(dtype=wp.int32),
    tolerance: float,
    segment_count: int,
    feasible: wp.array(dtype=wp.uint8),
    linear_converged: wp.array(dtype=wp.int32),
    phase_two_enabled: wp.array(dtype=wp.int32),
):
    """Classify Phase-I feasibility, certified infeasibility, or numerical unknown."""
    segment = wp.tid()
    if segment >= segment_count:
        return
    stopped = enabled[segment] == 0
    phase_two_enabled[segment] = 0
    if segment_active[segment] == 0:
        linear_converged[segment] = 1
        return
    primal_feasible = wp.isfinite(feasibility_violation[segment]) and feasibility_violation[segment] <= tolerance
    if primal_feasible:
        linear_converged[segment] = 1
        phase_two_enabled[segment] = 1
        return
    converged = (
        stopped
        and dual_residual[segment] <= tolerance
        and wp.abs(scalar_dual_residual[segment]) <= tolerance
        and primal_residual[segment] <= tolerance
    )
    farkas = converged and separating_margin[segment] > tolerance
    if farkas:
        feasible[segment] = wp.uint8(0)
        linear_converged[segment] = 0
        return
    # Phase II is an infeasible-start primal-dual method.  An unfinished
    # min-elastic search is not evidence that the original QP is infeasible;
    # only the Farkas branch above may suppress the optimization solve.
    linear_converged[segment] = 1
    phase_two_enabled[segment] = 1


@wp.kernel
def _phase_one_witness_select(
    feasibility_violation: wp.array(dtype=wp.float32),
    phase_two_enabled: wp.array(dtype=wp.int32),
    segment_active: wp.array(dtype=wp.int32),
    tolerance: float,
    segment_count: int,
    witness_selected: wp.array(dtype=wp.int32),
):
    """Keep a feasible Phase-I primal and suppress its Phase-II solve."""
    segment = wp.tid()
    if segment >= segment_count:
        return
    if witness_selected[segment] != 0:
        phase_two_enabled[segment] = 0
        return
    selected = (
        segment_active[segment] != 0
        and phase_two_enabled[segment] != 0
        and wp.isfinite(feasibility_violation[segment])
        and feasibility_violation[segment] <= tolerance
    )
    if selected:
        witness_selected[segment] = 1
        phase_two_enabled[segment] = 0


@wp.kernel
def _phase_two_primal_handoff(
    witness_selected: wp.array(dtype=wp.int32),
    frame_segment: wp.array(dtype=wp.int32),
    active_frames: int,
    primal: wp.array2d(dtype=wp.float32),
    primal_f64: wp.array2d(dtype=wp.float64),
):
    """Preserve certified Phase-I primals and reset the remaining Phase-II frames."""
    frame, dof = wp.tid()
    if frame >= active_frames:
        return
    if witness_selected[frame_segment[frame]] != 0:
        primal_f64[frame, dof] = wp.float64(primal[frame, dof])
    else:
        primal[frame, dof] = 0.0
        primal_f64[frame, dof] = wp.float64(0.0)


@wp.kernel
def _phase_one_original_violation_max(
    phase_one_value: wp.array2d(dtype=wp.float32),
    constraint_rhs: wp.array2d(dtype=wp.float32),
    constraint_scale: wp.array2d(dtype=wp.float32),
    elastic: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array(dtype=wp.float32),
):
    """Measure original-QP violation from a Phase-I row before residual work overwrites it."""
    frame, column = wp.tid()
    if frame >= active_frames or constraint_scale[frame, column] == 0.0:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0:
        return
    original_value = phase_one_value[frame, column] + elastic[segment]
    violation = (original_value - constraint_rhs[frame, column]) / constraint_scale[frame, column]
    if not wp.isfinite(violation):
        violation = 3.402823466e38
    wp.atomic_max(output, segment, wp.max(violation, 0.0))


@wp.kernel
def _ipm_residual(
    constraint_value: wp.array2d(dtype=wp.float32),
    constraint_rhs: wp.array2d(dtype=wp.float32),
    constraint_scale: wp.array2d(dtype=wp.float32),
    slack: wp.array2d(dtype=wp.float32),
    multiplier: wp.array2d(dtype=wp.float32),
    affine_slack: wp.array2d(dtype=wp.float32),
    affine_multiplier: wp.array2d(dtype=wp.float32),
    centering: wp.array(dtype=wp.float32),
    complementarity: wp.array(dtype=wp.float32),
    corrector: wp.uint8,
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    primal_residual: wp.array2d(dtype=wp.float32),
    complementarity_residual: wp.array2d(dtype=wp.float32),
    weights: wp.array2d(dtype=wp.float32),
    condensed_rhs: wp.array2d(dtype=wp.float32),
):
    """Build infeasible-start residuals and the condensed primal RHS term."""
    frame, column = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    active = enabled[segment] != 0 and constraint_scale[frame, column] != 0.0
    if not active:
        primal_residual[frame, column] = 0.0
        complementarity_residual[frame, column] = 0.0
        weights[frame, column] = 0.0
        condensed_rhs[frame, column] = 0.0
        return
    s = slack[frame, column]
    z = multiplier[frame, column]
    rp = constraint_value[frame, column] + s - constraint_rhs[frame, column]
    correction = (
        affine_slack[frame, column] * affine_multiplier[frame, column] - centering[segment] * complementarity[segment]
        if corrector != wp.uint8(0)
        else 0.0
    )
    rc = s * z + correction
    primal_residual[frame, column] = rp
    complementarity_residual[frame, column] = rc
    weights[frame, column] = z / s
    condensed_rhs[frame, column] = (rc - z * rp) / s


@wp.kernel
def _ipm_residual_corrector_f64(
    constraint_value: wp.array2d(dtype=wp.float32),
    constraint_rhs: wp.array2d(dtype=wp.float32),
    constraint_scale: wp.array2d(dtype=wp.float32),
    slack: wp.array2d(dtype=wp.float32),
    multiplier: wp.array2d(dtype=wp.float32),
    affine_slack: wp.array2d(dtype=wp.float64),
    affine_multiplier: wp.array2d(dtype=wp.float64),
    centering: wp.array(dtype=wp.float32),
    complementarity: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    primal_residual: wp.array2d(dtype=wp.float32),
    complementarity_residual: wp.array2d(dtype=wp.float32),
    weights: wp.array2d(dtype=wp.float32),
    condensed_rhs: wp.array2d(dtype=wp.float32),
):
    """Build the corrector residual from float64 affine directions."""
    frame, column = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    active = enabled[segment] != 0 and constraint_scale[frame, column] != 0.0
    if not active:
        primal_residual[frame, column] = 0.0
        complementarity_residual[frame, column] = 0.0
        weights[frame, column] = 0.0
        condensed_rhs[frame, column] = 0.0
        return
    s = wp.float64(slack[frame, column])
    z = wp.float64(multiplier[frame, column])
    rp = wp.float64(constraint_value[frame, column]) + s - wp.float64(constraint_rhs[frame, column])
    correction = affine_slack[frame, column] * affine_multiplier[frame, column]
    correction -= wp.float64(centering[segment]) * wp.float64(complementarity[segment])
    rc = s * z + correction
    primal_residual[frame, column] = wp.float32(rp)
    complementarity_residual[frame, column] = wp.float32(rc)
    weights[frame, column] = wp.float32(z / s)
    condensed_rhs[frame, column] = wp.float32((rc - z * rp) / s)


@wp.kernel
def _ipm_primal_feasibility_violation_max(
    constraint_value: wp.array2d(dtype=wp.float32),
    constraint_rhs: wp.array2d(dtype=wp.float32),
    constraint_scale: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array(dtype=wp.float32),
):
    """Measure direct physical-row violation without auxiliary slack error."""
    frame, column = wp.tid()
    if frame >= active_frames or constraint_scale[frame, column] == 0.0:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0:
        return
    violation = (constraint_value[frame, column] - constraint_rhs[frame, column]) / constraint_scale[frame, column]
    if not wp.isfinite(violation):
        violation = 3.402823466e38
    wp.atomic_max(output, segment, wp.max(violation, 0.0))


@wp.kernel
def _ipm_direction_recover(
    constraint_direction: wp.array2d(dtype=wp.float32),
    constraint_scale: wp.array2d(dtype=wp.float32),
    slack: wp.array2d(dtype=wp.float32),
    multiplier: wp.array2d(dtype=wp.float32),
    primal_residual: wp.array2d(dtype=wp.float32),
    complementarity_residual: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    slack_direction: wp.array2d(dtype=wp.float32),
    multiplier_direction: wp.array2d(dtype=wp.float32),
):
    """Recover slack and multiplier directions after a condensed solve."""
    frame, column = wp.tid()
    if frame >= active_frames:
        return
    active = enabled[frame_segment[frame]] != 0 and constraint_scale[frame, column] != 0.0
    if active:
        ds = -primal_residual[frame, column] - constraint_direction[frame, column]
        slack_direction[frame, column] = ds
        multiplier_direction[frame, column] = (
            -complementarity_residual[frame, column] - multiplier[frame, column] * ds
        ) / slack[frame, column]
    else:
        slack_direction[frame, column] = 0.0
        multiplier_direction[frame, column] = 0.0


@wp.kernel
def _ipm_direction_recover_f64(
    constraint_direction: wp.array2d(dtype=wp.float64),
    constraint_scale: wp.array2d(dtype=wp.float32),
    primal_residual: wp.array2d(dtype=wp.float32),
    condensed_rhs: wp.array2d(dtype=wp.float32),
    weights: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    slack_direction: wp.array2d(dtype=wp.float64),
    multiplier_direction: wp.array2d(dtype=wp.float64),
):
    """Recover float64 slack and multiplier directions."""
    frame, column = wp.tid()
    if frame >= active_frames:
        return
    active = enabled[frame_segment[frame]] != 0 and constraint_scale[frame, column] != 0.0
    if active:
        ds = -wp.float64(primal_residual[frame, column]) - constraint_direction[frame, column]
        slack_direction[frame, column] = ds
        multiplier_direction[frame, column] = -wp.float64(condensed_rhs[frame, column])
        multiplier_direction[frame, column] += wp.float64(weights[frame, column]) * constraint_direction[frame, column]
    else:
        slack_direction[frame, column] = wp.float64(0.0)
        multiplier_direction[frame, column] = wp.float64(0.0)


@wp.kernel
def _ipm_step_bound(
    constraint_scale: wp.array2d(dtype=wp.float32),
    slack: wp.array2d(dtype=wp.float32),
    multiplier: wp.array2d(dtype=wp.float32),
    slack_direction: wp.array2d(dtype=wp.float32),
    multiplier_direction: wp.array2d(dtype=wp.float32),
    fraction: float,
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    primal_step: wp.array(dtype=wp.float32),
    dual_step: wp.array(dtype=wp.float32),
):
    """Reduce fraction-to-boundary step lengths per segment."""
    frame, column = wp.tid()
    if frame >= active_frames or constraint_scale[frame, column] == 0.0:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0:
        return
    ds = slack_direction[frame, column]
    dz = multiplier_direction[frame, column]
    if ds < 0.0:
        wp.atomic_min(primal_step, segment, -fraction * slack[frame, column] / ds)
    if dz < 0.0:
        wp.atomic_min(dual_step, segment, -fraction * multiplier[frame, column] / dz)


@wp.kernel
def _ipm_step_bound_f64(
    constraint_scale: wp.array2d(dtype=wp.float32),
    slack: wp.array2d(dtype=wp.float32),
    multiplier: wp.array2d(dtype=wp.float32),
    slack_direction: wp.array2d(dtype=wp.float64),
    multiplier_direction: wp.array2d(dtype=wp.float64),
    fraction: float,
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    primal_step: wp.array(dtype=wp.float32),
    dual_step: wp.array(dtype=wp.float32),
):
    """Reduce fraction-to-boundary steps from float64 directions."""
    frame, column = wp.tid()
    if frame >= active_frames or constraint_scale[frame, column] == 0.0:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0:
        return
    ds = slack_direction[frame, column]
    dz = multiplier_direction[frame, column]
    if ds < wp.float64(0.0):
        step = -wp.float64(fraction) * wp.float64(slack[frame, column]) / ds
        wp.atomic_min(primal_step, segment, wp.float32(step))
    if dz < wp.float64(0.0):
        step = -wp.float64(fraction) * wp.float64(multiplier[frame, column]) / dz
        wp.atomic_min(dual_step, segment, wp.float32(step))


@wp.kernel
def _ipm_step_couple(
    primal_step: wp.array(dtype=wp.float32),
    dual_step: wp.array(dtype=wp.float32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
):
    """Use one common corrector step so linear KKT residuals contract together."""
    segment = wp.tid()
    if segment < segment_count and enabled[segment] != 0:
        step = wp.min(primal_step[segment], dual_step[segment])
        primal_step[segment] = step
        dual_step[segment] = step


@wp.kernel
def _ipm_affine_complementarity_frame(
    constraint_scale: wp.array2d(dtype=wp.float32),
    slack: wp.array2d(dtype=wp.float32),
    multiplier: wp.array2d(dtype=wp.float32),
    slack_direction: wp.array2d(dtype=wp.float32),
    multiplier_direction: wp.array2d(dtype=wp.float32),
    primal_step: wp.array(dtype=wp.float32),
    dual_step: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array(dtype=wp.float32),
):
    """Reduce affine complementarity deterministically within one frame."""
    frame = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    result = wp.float64(0.0)
    if enabled[segment] != 0:
        for column in range(constraint_scale.shape[1]):
            if constraint_scale[frame, column] != 0.0:
                s = wp.float64(slack[frame, column]) + wp.float64(primal_step[segment]) * wp.float64(
                    slack_direction[frame, column]
                )
                z = wp.float64(multiplier[frame, column]) + wp.float64(dual_step[segment]) * wp.float64(
                    multiplier_direction[frame, column]
                )
                result += s * z
    output[frame] = wp.float32(result)


@wp.kernel
def _ipm_affine_complementarity_frame_f64(
    constraint_scale: wp.array2d(dtype=wp.float32),
    slack: wp.array2d(dtype=wp.float32),
    multiplier: wp.array2d(dtype=wp.float32),
    slack_direction: wp.array2d(dtype=wp.float64),
    multiplier_direction: wp.array2d(dtype=wp.float64),
    primal_step: wp.array(dtype=wp.float32),
    dual_step: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array(dtype=wp.float32),
):
    """Reduce affine complementarity from float64 directions."""
    frame = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    result = wp.float64(0.0)
    if enabled[segment] != 0:
        for column in range(slack_direction.shape[1]):
            if constraint_scale[frame, column] != 0.0:
                s = wp.float64(slack[frame, column]) + wp.float64(primal_step[segment]) * slack_direction[frame, column]
                z = (
                    wp.float64(multiplier[frame, column])
                    + wp.float64(dual_step[segment]) * multiplier_direction[frame, column]
                )
                result += s * z
    output[frame] = wp.float32(result)


@wp.kernel
def _ipm_centering_update(
    complementarity_sum: wp.array(dtype=wp.float32),
    affine_complementarity_sum: wp.array(dtype=wp.float32),
    inequality_count: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    segment_count: int,
    complementarity: wp.array(dtype=wp.float32),
    centering: wp.array(dtype=wp.float32),
):
    """Compute per-segment complementarity and Mehrotra centering."""
    segment = wp.tid()
    if segment >= segment_count:
        return
    count = inequality_count[segment]
    if enabled[segment] == 0 or count == 0:
        complementarity[segment] = 0.0
        centering[segment] = 0.0
        return
    mu = complementarity_sum[segment] / float(count)
    mu_affine = affine_complementarity_sum[segment] / float(count)
    ratio = wp.max(mu_affine, 0.0) / wp.max(mu, 1.0e-20)
    complementarity[segment] = mu
    centering[segment] = wp.min(1.0, ratio * ratio * ratio)


@wp.kernel
def _ipm_relative_correction_max(
    direction: wp.array2d(dtype=wp.float32),
    current: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array(dtype=wp.float32),
):
    """Measure the largest affine correction relative to its current coordinate."""
    frame, dof = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0:
        return
    scale = wp.max(1.0, wp.abs(current[frame, dof]))
    correction = wp.abs(direction[frame, dof]) / scale
    if not wp.isfinite(correction):
        correction = 3.402823466e38
    wp.atomic_max(output, segment, correction)


@wp.kernel
def _ipm_relative_correction_max_f64(
    direction: wp.array2d(dtype=wp.float64),
    current: wp.array2d(dtype=wp.float64),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array(dtype=wp.float32),
):
    """Measure the largest float64 correction relative to the current coordinate."""
    frame, dof = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0:
        return
    scale = wp.max(wp.float64(1.0), wp.abs(current[frame, dof]))
    correction = wp.abs(direction[frame, dof]) / scale
    if not wp.isfinite(correction):
        correction = wp.float64(3.402823466e38)
    wp.atomic_max(output, segment, wp.float32(correction))


@wp.kernel
def _ipm_complementarity_max(
    slack: wp.array2d(dtype=wp.float32),
    multiplier: wp.array2d(dtype=wp.float32),
    constraint_scale: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array(dtype=wp.float32),
):
    """Measure the largest physical primal-dual gap in each segment."""
    frame, column = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if enabled[segment] != 0 and constraint_scale[frame, column] != 0.0:
        gap = slack[frame, column] * multiplier[frame, column]
        if not wp.isfinite(gap) or gap < 0.0:
            gap = 3.402823466e38
        wp.atomic_max(output, segment, gap)


@wp.kernel
def _ipm_phase_two_convergence_mask(
    primal_correction: wp.array(dtype=wp.float32),
    complementarity: wp.array(dtype=wp.float32),
    dual_residual: wp.array(dtype=wp.float32),
    equality_residual: wp.array(dtype=wp.float32),
    inequality_residual: wp.array(dtype=wp.float32),
    correction_tolerance: float,
    residual_tolerance: float,
    equality_tolerance: float,
    segment_count: int,
    enabled: wp.array(dtype=wp.int32),
):
    """Freeze Phase II when its affine primal correction and KKT residuals are negligible."""
    segment = wp.tid()
    if (
        segment < segment_count
        and enabled[segment] != 0
        and primal_correction[segment] <= correction_tolerance
        and complementarity[segment] <= correction_tolerance
        and dual_residual[segment] <= residual_tolerance
        and equality_residual[segment] <= equality_tolerance
        and inequality_residual[segment] <= residual_tolerance
    ):
        enabled[segment] = 0


@wp.kernel
def _ipm_barrier_convergence_mask(
    complementarity: wp.array(dtype=wp.float32),
    inequality_count: wp.array(dtype=wp.int32),
    dual_residual: wp.array(dtype=wp.float32),
    equality_residual: wp.array(dtype=wp.float32),
    inequality_residual: wp.array(dtype=wp.float32),
    complementarity_tolerance: float,
    residual_tolerance: float,
    equality_tolerance: float,
    segment_count: int,
    enabled: wp.array(dtype=wp.int32),
):
    """Freeze converged barrier iterates while preserving fixed launch work."""
    segment = wp.tid()
    if (
        segment < segment_count
        and enabled[segment] != 0
        and (inequality_count[segment] == 0 or complementarity[segment] <= complementarity_tolerance)
        and dual_residual[segment] <= residual_tolerance
        and equality_residual[segment] <= equality_tolerance
        and inequality_residual[segment] <= residual_tolerance
    ):
        enabled[segment] = 0


@wp.kernel
def _ipm_iterate_update(
    constraint_scale: wp.array2d(dtype=wp.float32),
    slack_direction: wp.array2d(dtype=wp.float32),
    multiplier_direction: wp.array2d(dtype=wp.float32),
    primal_step: wp.array(dtype=wp.float32),
    dual_step: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    slack: wp.array2d(dtype=wp.float32),
    multiplier: wp.array2d(dtype=wp.float32),
):
    """Apply one strictly interior corrector update."""
    frame, column = wp.tid()
    if frame >= active_frames or constraint_scale[frame, column] == 0.0:
        return
    segment = frame_segment[frame]
    if enabled[segment] != 0:
        slack[frame, column] += primal_step[segment] * slack_direction[frame, column]
        multiplier[frame, column] += dual_step[segment] * multiplier_direction[frame, column]


@wp.kernel
def _ipm_iterate_update_f64(
    constraint_scale: wp.array2d(dtype=wp.float32),
    slack_direction: wp.array2d(dtype=wp.float64),
    multiplier_direction: wp.array2d(dtype=wp.float64),
    primal_step: wp.array(dtype=wp.float32),
    dual_step: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    slack: wp.array2d(dtype=wp.float32),
    multiplier: wp.array2d(dtype=wp.float32),
):
    """Apply one float64 corrector direction to float32 iterate state."""
    frame, column = wp.tid()
    if frame >= active_frames or constraint_scale[frame, column] == 0.0:
        return
    segment = frame_segment[frame]
    if enabled[segment] != 0:
        next_slack = (
            wp.float64(slack[frame, column]) + wp.float64(primal_step[segment]) * slack_direction[frame, column]
        )
        next_multiplier = (
            wp.float64(multiplier[frame, column]) + wp.float64(dual_step[segment]) * multiplier_direction[frame, column]
        )
        slack[frame, column] = wp.float32(next_slack)
        multiplier[frame, column] = wp.float32(next_multiplier)


@wp.kernel
def _ipm_primal_update(
    direction: wp.array2d(dtype=wp.float32),
    step: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    primal: wp.array2d(dtype=wp.float32),
):
    """Apply a per-segment primal or equality-dual step."""
    frame, column = wp.tid()
    if frame < active_frames and enabled[frame_segment[frame]] != 0:
        primal[frame, column] += step[frame_segment[frame]] * direction[frame, column]


@wp.kernel
def _ipm_primal_update_state_f64(
    direction: wp.array2d(dtype=wp.float64),
    step: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    primal: wp.array2d(dtype=wp.float64),
    primal_f32: wp.array2d(dtype=wp.float32),
):
    """Update float64 primal state and its float32 external mirror together."""
    frame, column = wp.tid()
    if frame < active_frames and enabled[frame_segment[frame]] != 0:
        primal[frame, column] += wp.float64(step[frame_segment[frame]]) * direction[frame, column]
        primal_f32[frame, column] = wp.float32(primal[frame, column])


@wp.kernel
def _ipm_solve_status(
    requested: wp.array(dtype=wp.int32),
    krylov_enabled: wp.array(dtype=wp.int32),
    krylov_failed: wp.array(dtype=wp.int32),
    segment_count: int,
    linear_converged: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
):
    """Disable segments only when a condensed direction is numerically invalid."""
    segment = wp.tid()
    if segment >= segment_count or requested[segment] == 0:
        return
    if krylov_enabled[segment] != 0 or krylov_failed[segment] != 0:
        linear_converged[segment] = 0
        enabled[segment] = 0


@wp.kernel
def _ipm_step_enable(
    segment_active: wp.array(dtype=wp.int32),
    feasible: wp.array(dtype=wp.uint8),
    linear_converged: wp.array(dtype=wp.int32),
    segment_count: int,
    output: wp.array(dtype=wp.int32),
):
    """Enable line search only for certified constrained directions."""
    segment = wp.tid()
    if segment < segment_count:
        valid = segment_active[segment] != 0 and feasible[segment] != wp.uint8(0)
        output[segment] = 1 if valid and linear_converged[segment] != 0 else 0


@wp.kernel
def _ipm_locked_coordinate_canonicalize(
    joint_q: wp.array2d(dtype=wp.float32),
    coordinate_indices: wp.array(dtype=wp.int32),
    coordinate_dof_indices: wp.array(dtype=wp.int32),
    lower: wp.array(dtype=wp.float32),
    upper: wp.array(dtype=wp.float32),
    bound_count: int,
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    delta: wp.array2d(dtype=wp.float32),
):
    """Set locked scalar-coordinate directions to their exact unique value."""
    frame, bound = wp.tid()
    if frame >= active_frames or bound >= bound_count or enabled[frame_segment[frame]] == 0:
        return
    if lower[bound] == upper[bound]:
        delta[frame, coordinate_dof_indices[bound]] = lower[bound] - joint_q[frame, coordinate_indices[bound]]


@wp.kernel
def _ipm_velocity_bound_violation_max(
    velocity: wp.array2d(dtype=wp.float32),
    lower: wp.array(dtype=wp.float32),
    upper: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    segment_offsets: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    output: wp.array(dtype=wp.float32),
):
    """Accumulate exact non-head generalized-velocity box violation."""
    frame, dof = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0 or frame == segment_offsets[segment]:
        return
    value = velocity[frame, dof]
    if not wp.isfinite(value):
        violation = 3.402823466e38
    else:
        violation = wp.max(lower[dof] - value, value - upper[dof])
    wp.atomic_max(output, segment, wp.max(violation, 0.0))


@wp.kernel
def _pcg_alpha_update(
    direction: wp.array2d(dtype=wp.float32),
    operator_direction: wp.array2d(dtype=wp.float32),
    residual_dot: wp.array(dtype=wp.float32),
    operator_dot: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    delta: wp.array2d(dtype=wp.float32),
    residual: wp.array2d(dtype=wp.float32),
):
    frame, dof = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0:
        return
    denominator = operator_dot[segment]
    alpha = float(0.0)
    if wp.abs(denominator) > 1.0e-20:
        alpha = residual_dot[segment] / denominator
    delta[frame, dof] += alpha * direction[frame, dof]
    residual[frame, dof] -= alpha * operator_direction[frame, dof]


@wp.kernel
def _pcg_precondition(
    residual: wp.array2d(dtype=wp.float32),
    diagonal: wp.array2d(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    preconditioned: wp.array2d(dtype=wp.float32),
):
    frame, dof = wp.tid()
    if frame < active_frames:
        preconditioned[frame, dof] = (
            residual[frame, dof] / diagonal[frame, dof] if enabled[frame_segment[frame]] != 0 else 0.0
        )


@wp.kernel
def _pcg_beta_update(
    preconditioned: wp.array2d(dtype=wp.float32),
    residual_dot: wp.array(dtype=wp.float32),
    residual_dot_next: wp.array(dtype=wp.float32),
    frame_segment: wp.array(dtype=wp.int32),
    enabled: wp.array(dtype=wp.int32),
    active_frames: int,
    direction: wp.array2d(dtype=wp.float32),
):
    frame, dof = wp.tid()
    if frame >= active_frames:
        return
    segment = frame_segment[frame]
    if enabled[segment] == 0:
        direction[frame, dof] = 0.0
        return
    denominator = residual_dot[segment]
    beta = float(0.0)
    if wp.abs(denominator) > 1.0e-20:
        beta = residual_dot_next[segment] / denominator
    direction[frame, dof] = preconditioned[frame, dof] + beta * direction[frame, dof]
