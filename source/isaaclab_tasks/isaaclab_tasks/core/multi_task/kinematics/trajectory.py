# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Matrix-free whole-segment refinement over Newton IK residuals."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import newton.ik as ik
import torch
import warp as wp

from .ik_execution import plan_ik_memory
from .impl.trajectory_warp import (
    _accepted_violation_update,
    _conditional_configuration_copy,
    _constraint_feasible_rows_mark,
    _constraint_line_search_decide,
    _constraint_protected_violation_max,
    _constraint_violation_max,
    _constraints_initialize,
    _coordinate_bound_violation_max,
    _coordinate_bounds_mask_binding_dofs,
    _coordinate_bounds_project_candidate,
    _coordinate_bounds_validate,
    _delta_scale,
    _dot_segments,
    _dot_segments_f64,
    _equalities_mark,
    _frame_segment_indices,
    _frozen_dof_indices_validate,
    _frozen_dof_jacobian_zero,
    _frozen_dof_values_zero,
    _inequalities_mark,
    _ipm_affine_complementarity_frame,
    _ipm_affine_complementarity_frame_f64,
    _ipm_augmented_minres_solve_f64,
    _ipm_barrier_add_f64,
    _ipm_barrier_convergence_mask,
    _ipm_block_band_backward_f64,
    _ipm_block_band_backward_f64_parallel,
    _ipm_block_band_barrier_add,
    _ipm_block_band_equilibrate_f64,
    _ipm_block_band_forward_f64,
    _ipm_block_band_forward_f64_parallel,
    _ipm_block_band_matrix_build,
    _ipm_block_band_matrix_factor,
    _ipm_block_band_matrix_factor_parallel,
    _ipm_block_band_scale_build_f64,
    _ipm_centering_update,
    _ipm_certification_resolve,
    _ipm_certification_select,
    _ipm_complementarity_max,
    _ipm_constraints_apply,
    _ipm_constraints_apply_f64,
    _ipm_constraints_initialize,
    _ipm_constraints_transpose,
    _ipm_constraints_transpose_f64,
    _ipm_direction_recover,
    _ipm_direction_recover_f64,
    _ipm_fallback_promote_certificate_failure,
    _ipm_fallback_promote_factor_failure,
    _ipm_feature_f64,
    _ipm_frame_matrix_build,
    _ipm_frame_matrix_factor,
    _ipm_frame_solve_f32,
    _ipm_iterate_initialize,
    _ipm_iterate_update,
    _ipm_iterate_update_f64,
    _ipm_locked_coordinate_canonicalize,
    _ipm_normal_diagonal_add,
    _ipm_outer_forcing_update,
    _ipm_pcg_convergence_update,
    _ipm_phase_two_convergence_mask,
    _ipm_physical_convergence_initialize,
    _ipm_physical_convergence_update,
    _ipm_physical_reference_norm_f64,
    _ipm_physical_residual_f64,
    _ipm_physical_residual_from_f64,
    _ipm_primal_feasibility_violation_max,
    _ipm_primal_update,
    _ipm_primal_update_state_f64,
    _ipm_relative_correction_max,
    _ipm_relative_correction_max_f64,
    _ipm_residual,
    _ipm_residual_corrector_f64,
    _ipm_rhs_condense_f64,
    _ipm_rhs_stationarity_f64,
    _ipm_route_select,
    _ipm_rows_narrow_f32,
    _ipm_singleton_backward_f32,
    _ipm_singleton_forward_f32,
    _ipm_singleton_forward_f64,
    _ipm_singleton_matrix_build,
    _ipm_singleton_matrix_factor,
    _ipm_solution_accumulate_f64,
    _ipm_solution_accumulate_from_f64,
    _ipm_solution_canonicalize_f32,
    _ipm_solution_copy_f32,
    _ipm_solution_equilibrate_f64,
    _ipm_solve_status,
    _ipm_step_bound,
    _ipm_step_bound_f64,
    _ipm_step_couple,
    _ipm_step_enable,
    _ipm_temporal_backward_f32,
    _ipm_temporal_band_build,
    _ipm_temporal_band_factor,
    _ipm_temporal_forward_f32,
    _ipm_temporal_forward_f64,
    _ipm_temporal_lower_multiply_f32,
    _ipm_temporal_upper_multiply_f32,
    _ipm_velocity_bound_violation_max,
    _krylov_convergence_accumulate,
    _line_search_decide,
    _line_search_pending,
    _line_search_stationarity_initialize,
    _minres_basis,
    _minres_current_lanczos,
    _minres_finalize_iteration,
    _minres_initialize,
    _minres_previous_lanczos,
    _minres_recurrence,
    _minres_solution_update,
    _minres_store_alfa,
    _minres_vectors_initialize,
    _normal_apply_f64,
    _normal_diagonal,
    _normal_temporal_band_build,
    _objective_term_count,
    _pcg_additive_precondition_combine,
    _pcg_alpha_update,
    _pcg_beta_update,
    _pcg_convergence_initialize,
    _pcg_convergence_update,
    _pcg_direction_restart,
    _pcg_identity_initialize,
    _pcg_initialize,
    _pcg_precondition,
    _pcg_reliable_restart,
    _pcg_reliable_restart_state,
    _pcg_temporal_backward_f32,
    _pcg_temporal_forward_f32,
    _phase_one_arrowhead_cross_build,
    _phase_one_arrowhead_failure_finalize,
    _phase_one_arrowhead_pair_dot,
    _phase_one_arrowhead_pair_reduce,
    _phase_one_arrowhead_primal_recover,
    _phase_one_arrowhead_scalar_solve,
    _phase_one_block_band_matrix_build,
    _phase_one_combined_dot_add,
    _phase_one_constraints_apply,
    _phase_one_constraints_initialize,
    _phase_one_elastic_update,
    _phase_one_equality_transpose_values,
    _phase_one_finalize,
    _phase_one_original_violation_max,
    _phase_one_scalar_diagonal,
    _phase_one_scalar_direction_update,
    _phase_one_scalar_max_add,
    _phase_one_scalar_pcg_initialize,
    _phase_one_scalar_pcg_update,
    _phase_one_scalar_precondition,
    _phase_one_scalar_rhs,
    _phase_one_scalar_transpose,
    _phase_one_witness_select,
    _phase_two_primal_handoff,
    _precision_apply,
    _precision_apply_f64,
    _precision_diagonal,
    _precondition_primal,
    _projected_direction_validate,
    _projected_fallback_apply,
    _projected_fallback_initialize,
    _projected_line_search_decide,
    _qp_rows_initialize,
    _qp_rows_operator_dual,
    _qp_rows_refresh,
    _qp_rows_residual_unscale,
    _qp_rows_transpose,
    _residual_activity_validate,
    _restoration_candidate_merit_max,
    _restoration_current_merit_max,
    _second_order_correction_decide,
    _second_order_correction_request,
    _segment_convergence_update,
    _segment_damping_validate,
    _segment_diagonal_initialize,
    _segment_scalar_scaled_add,
    _segment_scaled_add,
    _segment_step_max,
    _segments_pending,
    _square_norm_f64,
    _velocity_candidate,
)

_IPM_ITERATIONS = 20
_IPM_BLOCK_BAND_BLOCK_DIM = 256
_IPM_FRACTION_TO_BOUNDARY = 0.995
_SECOND_ORDER_CORRECTION_ITERATIONS = 3


@dataclass(frozen=True, slots=True)
class IKTrajectoryMemoryPlan:
    """Contiguous whole-segment batches sharing one maximum workspace."""

    segment_count: int
    frame_count: int
    batch_segment_offsets: tuple[int, ...]
    batch_frame_counts: tuple[int, ...]
    max_segment_frames: int
    max_safe_frames: int
    workspace_frame_capacity: int
    fixed_bytes: int
    bytes_per_frame: int
    device_free_bytes: int | None
    safety_reserve_bytes: int
    memory_budget_bytes: int | None
    peak_additional_workspace_bytes: int


def plan_trajectory_memory(
    segment_lengths: Sequence[int],
    device: str | torch.device,
    estimate_memory: Callable[[int], int],
    *,
    device_free_bytes: int | None = None,
) -> IKTrajectoryMemoryPlan:
    """Pack balanced contiguous segment batches without splitting a segment.

    Args:
        segment_lengths: Frame count of every segment in source order.
        device: Device used by the reusable trajectory workspace.
        estimate_memory: Exact workspace bytes for a frame capacity.
        device_free_bytes: Optional CUDA free-memory value for deterministic tests.

    Returns:
        A contiguous batch partition and one reusable maximum frame capacity.

    Raises:
        ValueError: If segment lengths are empty or invalid.
        MemoryError: If the largest complete segment cannot fit the live budget.
    """
    lengths = tuple(segment_lengths)
    if not lengths or any(type(length) is not int or length < 1 for length in lengths):
        raise ValueError("Trajectory segment lengths must be positive integers.")
    total_frames = sum(lengths)
    flat_plan = plan_ik_memory(
        total_frames,
        device,
        estimate_memory,
        device_free_bytes=device_free_bytes,
    )
    max_segment_frames = max(lengths)
    if max_segment_frames > flat_plan.max_safe_capacity:
        required = estimate_memory(max_segment_frames)
        raise MemoryError(
            f"Largest trajectory segment requires {required} bytes for {max_segment_frames} frames, "
            f"but the live budget fits only {flat_plan.max_safe_capacity} frames."
        )

    def batch_count(capacity: int) -> int:
        count = 1
        frames = 0
        for length in lengths:
            if frames and frames + length > capacity:
                count += 1
                frames = 0
            frames += length
        return count

    minimum_batches = batch_count(flat_plan.max_safe_capacity)
    low = max_segment_frames
    high = flat_plan.max_safe_capacity
    while low < high:
        middle = (low + high) // 2
        if batch_count(middle) <= minimum_batches:
            high = middle
        else:
            low = middle + 1
    balanced_capacity = low

    batch_segment_offsets = [0]
    batch_frame_counts: list[int] = []
    frames = 0
    for segment_index, length in enumerate(lengths):
        if frames and frames + length > balanced_capacity:
            batch_frame_counts.append(frames)
            batch_segment_offsets.append(segment_index)
            frames = 0
        frames += length
    batch_frame_counts.append(frames)
    batch_segment_offsets.append(len(lengths))
    workspace_frame_capacity = max(batch_frame_counts)
    return IKTrajectoryMemoryPlan(
        segment_count=len(lengths),
        frame_count=total_frames,
        batch_segment_offsets=tuple(batch_segment_offsets),
        batch_frame_counts=tuple(batch_frame_counts),
        max_segment_frames=max_segment_frames,
        max_safe_frames=flat_plan.max_safe_capacity,
        workspace_frame_capacity=workspace_frame_capacity,
        fixed_bytes=flat_plan.fixed_bytes,
        bytes_per_frame=flat_plan.bytes_per_problem,
        device_free_bytes=flat_plan.device_free_bytes,
        safety_reserve_bytes=flat_plan.safety_reserve_bytes,
        memory_budget_bytes=flat_plan.memory_budget_bytes,
        peak_additional_workspace_bytes=estimate_memory(workspace_frame_capacity),
    )


class IKTrajectorySolver:
    """Refine complete segments with Newton residuals and matrix-free temporal coupling.

    The Newton optimizer owns frame-local objective residuals and Jacobians. This
    class adds first-, second-, and third-difference precision within explicit
    segment offsets, exact residual equalities, and residual inequalities. Its
    persistent memory is linear in frame count apart from the frame-local
    Jacobian already owned by Newton.

    Args:
        optimizer: Public Newton LM optimizer whose capacity is the maximum
            number of frames in one segment batch.
        max_segments: Maximum segments represented by one solve.
        max_equality_residuals_per_frame: Maximum exact-equality residual scalars
            represented in one frame. This capacity must be nonnegative and
            divisible by three. Zero disables exact equalities.
        damping: Positive Gauss-Newton diagonal damping.
        krylov_max_iterations: Maximum matrix-free PCG or MINRES iterations per
            outer iteration.
        krylov_relative_tolerance: Scale-free preconditioned residual tolerance
            for each independent segment.
        kkt_relative_tolerance: Maximum relative affine KKT-root correction and
            physical complementarity for a constrained segment.
        krylov_check_interval: Non-capturing CUDA iterations between bounded
            host checks for all-segment convergence. CPU checks every iteration;
            native Warp capture retains the static maximum loop.
        line_search_steps: Strictly descending trial scales in (0, 1]. With
            every solve, feasible trials must satisfy Armijo objective descent
            and infeasible trials must make scale-aware constraint restoration
            progress. Model-stationary segments do not materialize a trial.
    """

    @dataclass(frozen=True, slots=True)
    class ResidualEqualities:
        """Dense frame-local exact residual equalities."""

        active: torch.Tensor
        """Equality activity, shape [frame_count, target_count], uint8."""

        residual_starts_by_target: torch.Tensor
        """Inclusive Newton residual-row start per target, shape [target_count], int32."""

        residual_width: int = 3
        """Consecutive XYZ residual rows owned by every target."""

    @dataclass(frozen=True, slots=True)
    class ResidualActivity:
        """Compact per-frame confidence shared by residual groups."""

        values: torch.Tensor
        """Confidence in [0, 1], shape [frame_count, group_count], float32."""

        group_by_residual: torch.Tensor
        """Frame-activity group per Newton residual; -1 means always active."""

        first_difference_group_by_residual: torch.Tensor | None = None
        """Optional end-indexed edge-activity group per residual; -1 keeps vertex-min gating."""

    @dataclass(frozen=True, slots=True)
    class ResidualInequalities:
        """Static residual rows constrained by upper bounds."""

        residual_indices: torch.Tensor
        """Strictly increasing residual rows, shape [inequality_count], int32."""

        upper: torch.Tensor
        """Upper bound per indexed residual row, shape [inequality_count], float32."""

    @dataclass(frozen=True, slots=True)
    class CoordinateBounds:
        """Scalar generalized-coordinate bounds with explicit tangent ownership.

        Coordinate and tangent-DOF indices must each be unique and form an
        explicit one-to-one mapping. Every box must have at least one finite
        side. Initial coordinates must be feasible; a coordinate whose lower
        and upper bounds are equal must initially equal that locked value
        exactly.
        """

        coordinate_indices: torch.Tensor
        """Bounded generalized-coordinate indices, shape [bound_count], int32."""

        dof_indices: torch.Tensor
        """Unique matching tangent-DOF indices, shape [bound_count], int32."""

        lower: torch.Tensor
        """Lower bounds [m or rad, depending on joint type], shape [bound_count], float32."""

        upper: torch.Tensor
        """Upper bounds [m or rad, depending on joint type], shape [bound_count], float32."""

    @dataclass(frozen=True, slots=True)
    class Statistics:
        """Fixed work and storage facts for one trajectory solve."""

        frame_count: int
        segment_count: int
        equality_target_count: int
        krylov_max_iterations: int
        workspace_bytes: int

    @staticmethod
    def estimate_workspace_bytes(
        frame_capacity: int,
        coordinate_count: int,
        dof_count: int,
        residual_count: int,
        max_segments: int,
        max_equality_residuals_per_frame: int,
    ) -> int:
        """Return exact persistent trajectory workspace storage [byte]."""
        if (
            min(frame_capacity, coordinate_count, dof_count, residual_count, max_segments) < 1
            or max_equality_residuals_per_frame < 0
        ):
            raise ValueError("Trajectory dimensions must be positive and equality capacity must be nonnegative.")
        frame_values = frame_capacity * (
            23 * coordinate_count + 20 * residual_count + 81 * dof_count + 32 * max_equality_residuals_per_frame + 15
        )
        fixed_values = 8 * residual_count + 4 * coordinate_count + 2 * dof_count + 66 * max_segments + 4
        byte_values = 2 * frame_capacity * residual_count + frame_capacity + 2 * max_segments
        singleton_factor_bytes = torch.float64.itemsize * max_segments * dof_count * dof_count
        frame_factor_bytes = torch.float64.itemsize * frame_capacity * dof_count * dof_count
        block_band_factor_bytes = torch.float64.itemsize * 4 * frame_capacity * dof_count * dof_count
        inequality_width = residual_count + 2 * coordinate_count + 2 * dof_count
        active_width = dof_count + max_equality_residuals_per_frame
        augmented_bytes = torch.float64.itemsize * frame_capacity * (6 * inequality_width + 8 * dof_count)
        restoration_bytes = torch.float32.itemsize * (
            frame_capacity * (inequality_width + 2 * active_width) + 6 * max_segments
        )
        phase_one_arrowhead_bytes = torch.float64.itemsize * (frame_capacity * (dof_count + 2) + 2 * max_segments)
        return (
            torch.float32.itemsize * (frame_values + fixed_values)
            + byte_values
            + max_segments
            + singleton_factor_bytes
            + frame_factor_bytes
            + 3 * block_band_factor_bytes
            + augmented_bytes
            + restoration_bytes
            + phase_one_arrowhead_bytes
        )

    def __init__(
        self,
        optimizer: ik.IKOptimizerLM,
        *,
        max_segments: int,
        max_equality_residuals_per_frame: int,
        damping: float = 1.0e-4,
        krylov_max_iterations: int = 128,
        krylov_relative_tolerance: float = 1.0e-4,
        kkt_relative_tolerance: float = 1.0e-4,
        krylov_check_interval: int = 8,
        line_search_steps: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125),
    ) -> None:
        if min(max_segments, krylov_max_iterations) < 1:
            raise ValueError("Segment and Krylov capacities must be positive.")
        if max_equality_residuals_per_frame < 0:
            raise ValueError("Per-frame equality residual capacity must be nonnegative.")
        if max_equality_residuals_per_frame % 3 != 0:
            raise ValueError("Per-frame equality residual capacity must be divisible by three.")
        if not math.isfinite(damping) or damping <= 0.0:
            raise ValueError("Trajectory damping must be positive and finite.")
        if not math.isfinite(krylov_relative_tolerance) or not 0.0 < krylov_relative_tolerance < 1.0:
            raise ValueError("Krylov relative tolerance must be finite and in (0, 1).")
        if not math.isfinite(kkt_relative_tolerance) or not 0.0 < kkt_relative_tolerance < 1.0:
            raise ValueError("KKT relative tolerance must be finite and in (0, 1).")
        if type(krylov_check_interval) is not int or krylov_check_interval < 1:
            raise ValueError("Krylov check interval must be a positive integer.")
        if (
            not line_search_steps
            or any(not math.isfinite(step) or not 0.0 < step <= 1.0 for step in line_search_steps)
            or any(
                current <= following
                for current, following in zip(line_search_steps, line_search_steps[1:], strict=False)
            )
        ):
            raise ValueError("Trajectory line-search steps must be finite, positive, at most one, and descending.")
        self.optimizer = optimizer
        self.capacity = optimizer.n_batch
        self.coordinate_count = optimizer.n_coords
        self.dof_count = optimizer.n_dofs
        self.residual_count = optimizer.n_residuals
        self.max_segments = max_segments
        self.max_equality_residuals_per_frame = max_equality_residuals_per_frame
        self.active_width = self.dof_count + self.max_equality_residuals_per_frame
        self.inequality_width = self.residual_count + 2 * self.coordinate_count + 2 * self.dof_count
        self.constraint_width = self.inequality_width + 2 * self.active_width + 1
        self.damping = damping
        self.krylov_max_iterations = krylov_max_iterations
        self.krylov_relative_tolerance = krylov_relative_tolerance
        self.kkt_relative_tolerance = kkt_relative_tolerance
        self.krylov_check_interval = krylov_check_interval
        self.line_search_steps = line_search_steps
        self.device = torch.device(str(optimizer.device))
        self._coordinate_bound_count = 0
        self._ipm_operator_active = False
        self._phase_one_has_equalities = False

        jacobian_buffer = optimizer.jacobian
        if isinstance(jacobian_buffer, torch.Tensor):
            self._jacobian = jacobian_buffer
            self._wp_jacobian = wp.from_torch(jacobian_buffer)
        else:
            self._wp_jacobian = jacobian_buffer
            self._jacobian = wp.to_torch(jacobian_buffer)
        if self._jacobian.shape != (self.capacity, self.residual_count, self.dof_count):
            raise ValueError("The fixed-capacity optimizer exposes an incompatible Jacobian buffer.")

        float_options = {"device": self.device, "dtype": torch.float32}
        double_options = {"device": self.device, "dtype": torch.float64}
        int_options = {"device": self.device, "dtype": torch.int32}
        frame_residual_shape = (self.capacity, self.residual_count)
        frame_dof_shape = (self.capacity, self.dof_count)
        frame_active_shape = (self.capacity, self.active_width)
        frame_inequality_shape = (self.capacity, self.constraint_width)
        frame_augmented_shape = (self.capacity, self.inequality_width)
        temporal_band_shape = (self.capacity, self.dof_count, 4)
        self._joint_q = torch.empty((self.capacity, self.coordinate_count), **float_options)
        self._joint_q_next = torch.empty_like(self._joint_q)
        self._base_weights = torch.empty(frame_residual_shape, **float_options)
        self._feature = torch.empty(frame_residual_shape, **float_options)
        self._objective_weights = torch.empty(frame_residual_shape, **float_options)
        self._linearized_feature = torch.empty(frame_residual_shape, **float_options)
        self._constraint_kind = torch.empty(frame_residual_shape, dtype=torch.uint8, device=self.device)
        self._constraint_protected_kind = torch.empty(frame_residual_shape, dtype=torch.uint8, device=self.device)
        self._precision_feature = torch.empty(frame_residual_shape, **float_options)
        self._precision_diagonal = torch.empty(frame_residual_shape, **float_options)
        self._dual_solution = torch.empty(frame_active_shape, **float_options)
        self._dual_r1 = torch.empty(frame_active_shape, **float_options)
        self._dual_r2 = torch.empty(frame_active_shape, **float_options)
        self._dual_z = torch.empty(frame_active_shape, **float_options)
        self._dual_basis = torch.empty(frame_active_shape, **float_options)
        self._dual_direction_older = torch.empty(frame_active_shape, **float_options)
        self._dual_direction_old = torch.empty(frame_active_shape, **float_options)
        self._dual_direction = torch.empty(frame_active_shape, **float_options)
        self._active_row_codes = torch.empty(frame_active_shape, **int_options)
        self._active_row_scales = torch.empty(frame_active_shape, **float_options)
        self._active_equality_count = torch.empty(self.max_segments, **int_options)
        self._right_hand_side = torch.empty(frame_dof_shape, **float_options)
        self._active_row_transpose = torch.empty(frame_dof_shape, **float_options)
        self._delta = torch.empty(frame_dof_shape, **float_options)
        self._delta_correction = torch.empty(frame_dof_shape, **float_options)
        self._delta_candidate = torch.empty(frame_dof_shape, **float_options)
        self._minres_direction = torch.empty(frame_dof_shape, **float_options)
        self._joint_velocity = torch.empty(frame_dof_shape, **float_options)
        self._joint_velocity_next = torch.empty(frame_dof_shape, **float_options)
        self._velocity_lower = torch.empty(self.dof_count, **float_options)
        self._velocity_upper = torch.empty(self.dof_count, **float_options)
        self._pcg_residual = torch.empty(frame_dof_shape, **float_options)
        self._pcg_preconditioned = torch.empty(frame_dof_shape, **float_options)
        self._pcg_direction = torch.empty(frame_dof_shape, **float_options)
        self._pcg_operator_direction = torch.empty(frame_dof_shape, **float_options)
        self._normal_diagonal = torch.empty(frame_dof_shape, **float_options)
        self._ipm_temporal_factor = torch.empty(temporal_band_shape, **double_options)
        self._ipm_singleton_factor = torch.empty((self.max_segments, self.dof_count, self.dof_count), **double_options)
        self._ipm_frame_factor = torch.empty((self.capacity, self.dof_count, self.dof_count), **double_options)
        self._ipm_segment_coupled = torch.empty(self.max_segments, **int_options)
        self._ipm_block_band_factor = torch.empty((self.capacity, 4, self.dof_count, self.dof_count), **double_options)
        self._ipm_objective_block_band = torch.empty_like(self._ipm_block_band_factor)
        self._ipm_objective_block_band_factor = torch.empty_like(self._ipm_block_band_factor)
        self._ipm_objective_factor_failed = torch.empty(self.max_segments, **int_options)
        self._ipm_objective_factor_prepared = False
        self._ipm_block_band_scale_f64 = torch.empty(frame_dof_shape, **double_options)
        self._ipm_recursive_converged = torch.empty(self.max_segments, **int_options)
        self._ipm_augmented_fallback = torch.empty(self.max_segments, **int_options)
        self._ipm_solution_f64 = torch.empty(frame_dof_shape, **double_options)
        self._ipm_right_hand_side_f64 = torch.empty(frame_dof_shape, **double_options)
        self._ipm_primal_f64 = torch.empty(frame_dof_shape, **double_options)
        self._ipm_feature_f64 = torch.empty(frame_residual_shape, **double_options)
        self._ipm_precision_f64 = torch.empty(frame_residual_shape, **double_options)
        self._ipm_operator_f64 = torch.empty(frame_dof_shape, **double_options)
        self._ipm_dot_frame_f64 = torch.empty(self.capacity, **double_options)
        self._ipm_residual_dot_f64 = torch.empty(self.max_segments, **double_options)
        self._phase_one_cross_f64 = torch.empty(frame_dof_shape, **double_options)
        self._phase_one_dot_frame_f64 = torch.empty((self.capacity, 2), **double_options)
        self._phase_one_dot_segment_f64 = torch.empty((self.max_segments, 2), **double_options)
        self._ipm_initial_norm_f64 = torch.empty(self.max_segments, **double_options)
        self._ipm_augmented_primal_r2_f64 = torch.empty(frame_dof_shape, **double_options)
        self._ipm_augmented_primal_basis_f64 = torch.empty(frame_dof_shape, **double_options)
        self._ipm_augmented_primal_work_f64 = torch.empty(frame_dof_shape, **double_options)
        self._ipm_augmented_primal_direction_older_f64 = torch.empty(frame_dof_shape, **double_options)
        self._ipm_augmented_primal_direction_old_f64 = torch.empty(frame_dof_shape, **double_options)
        self._ipm_augmented_row_solution_f64 = torch.empty(frame_augmented_shape, **double_options)
        self._ipm_augmented_row_r1_f64 = torch.empty(frame_augmented_shape, **double_options)
        self._ipm_augmented_row_r2_f64 = torch.empty(frame_augmented_shape, **double_options)
        self._ipm_augmented_row_basis_f64 = torch.empty(frame_augmented_shape, **double_options)
        self._ipm_augmented_row_direction_older_f64 = torch.empty(frame_augmented_shape, **double_options)
        self._ipm_augmented_row_direction_old_f64 = torch.empty(frame_augmented_shape, **double_options)
        self._pose_weights = torch.empty(self.residual_count, **float_options)
        self._temporal_weights = torch.empty((3, self.residual_count), **float_options)
        self._inactive_residual_activity = torch.ones((self.capacity, 1), **float_options)
        self._inactive_activity_group_by_residual = torch.full((self.residual_count,), -1, **int_options)
        self._residual_upper = torch.empty(self.residual_count, **float_options)
        self._inequality_upper = torch.empty(self.residual_count, **float_options)
        self._inequality_indices = torch.empty(self.residual_count, **int_options)
        self._coordinate_indices = torch.empty(self.coordinate_count, **int_options)
        self._coordinate_dof_indices = torch.empty(self.coordinate_count, **int_options)
        self._coordinate_lower = torch.empty(self.coordinate_count, **float_options)
        self._coordinate_upper = torch.empty(self.coordinate_count, **float_options)
        self._segment_offsets = torch.empty(self.max_segments + 1, **int_options)
        self._frame_segment = torch.empty(self.capacity, **int_options)
        self._step_seconds = torch.empty(self.max_segments, **float_options)
        self._segment_damping = torch.empty(self.max_segments, **float_options)
        self._inactive_equality_active = torch.zeros((self.capacity, 1), dtype=torch.uint8, device=self.device)
        self._inactive_equality_residual_starts = torch.zeros(1, **int_options)
        self._dot_frame = torch.empty(self.capacity, **float_options)
        self._residual_dot = torch.empty(self.max_segments, **float_options)
        self._residual_dot_next = torch.empty(self.max_segments, **float_options)
        self._operator_dot = torch.empty(self.max_segments, **float_options)
        self._objective_cost = torch.empty(self.max_segments, **float_options)
        self._candidate_cost = torch.empty(self.max_segments, **float_options)
        self._accepted_cost = torch.empty(self.max_segments, **float_options)
        self._objective_term_count = torch.empty(self.max_segments, **int_options)
        self._segment_active = torch.empty(self.max_segments, **int_options)
        self._constraint_violation = torch.empty(self.max_segments, **float_options)
        self._candidate_violation = torch.empty(self.max_segments, **float_options)
        self._candidate_bound_violation = torch.empty(self.max_segments, **float_options)
        self._candidate_protected_violation = torch.empty(self.max_segments, **float_options)
        self._restoration_merit = torch.empty(self.max_segments, **float_options)
        self._candidate_restoration_merit = torch.empty(self.max_segments, **float_options)
        self._accepted_restoration_merit = torch.empty(self.max_segments, **float_options)
        self._restoration_constraint_scale = torch.empty(frame_augmented_shape, **float_options)
        self._restoration_active_row_codes = torch.empty(frame_active_shape, **int_options)
        self._restoration_active_row_scales = torch.empty(frame_active_shape, **float_options)
        self._restoration_active_equality_count = torch.empty(self.max_segments, **int_options)
        self._minres_state = torch.empty((self.max_segments, 14), **float_options)
        self._line_search_outcome = torch.empty(self.max_segments, **int_options)
        self._line_search_take = torch.empty(self.max_segments, **int_options)
        self._second_order_correction_direction = torch.empty(frame_dof_shape, **float_options)
        self._second_order_correction_joint_q = torch.empty_like(self._joint_q)
        self._second_order_correction_state = torch.empty((self.max_segments, 6), **float_options)
        self._second_order_correction_active = torch.empty(self.max_segments, **int_options)
        self._second_order_correction_line_search_enabled = torch.empty(self.max_segments, **int_options)
        self._control_pending = torch.empty(1, **int_options)
        self._constraint_contract_error = torch.empty(1, **int_options)
        self._segment_feasible = torch.empty(self.max_segments, dtype=torch.uint8, device=self.device)
        self._globalization_succeeded = torch.empty(self.max_segments, dtype=torch.bool, device=self.device)
        self._restoration_stalled = torch.empty(self.max_segments, dtype=torch.bool, device=self.device)
        self._kkt_enabled = torch.empty(self.max_segments, **int_options)
        self._ipm_certify = torch.empty(self.max_segments, **int_options)
        self._minres_enabled = torch.empty(self.max_segments, **int_options)
        self._minres_failed = torch.empty(self.max_segments, **int_options)
        self._ipm_constraint_scale = torch.empty(frame_inequality_shape, **float_options)
        self._ipm_constraint_rhs = torch.empty(frame_inequality_shape, **float_options)
        self._ipm_slack = torch.empty(frame_inequality_shape, **float_options)
        self._ipm_multiplier = torch.empty(frame_inequality_shape, **float_options)
        self._ipm_primal_residual = torch.empty(frame_inequality_shape, **float_options)
        self._ipm_complementarity_residual = torch.empty(frame_inequality_shape, **float_options)
        self._ipm_weights = torch.empty(frame_inequality_shape, **float_options)
        self._ipm_constraint_work = torch.empty(frame_inequality_shape, **float_options)
        self._ipm_affine_slack = torch.empty(frame_inequality_shape, **float_options)
        self._ipm_affine_multiplier = torch.empty(frame_inequality_shape, **float_options)
        self._ipm_primal = torch.empty(frame_dof_shape, **float_options)
        self._ipm_gradient = torch.empty(frame_dof_shape, **float_options)
        self._ipm_transpose = torch.empty(frame_dof_shape, **float_options)
        self._ipm_equality_dual = torch.empty(frame_active_shape, **float_options)
        self._ipm_equality_rhs = torch.empty(frame_active_shape, **float_options)
        self._ipm_inequality_count = torch.empty(self.max_segments, **int_options)
        self._ipm_complementarity = torch.empty(self.max_segments, **float_options)
        self._ipm_centering = torch.empty(self.max_segments, **float_options)
        self._ipm_primal_step = torch.empty(self.max_segments, **float_options)
        self._ipm_dual_step = torch.empty(self.max_segments, **float_options)
        self._ipm_enabled = torch.empty(self.max_segments, **int_options)
        self._ipm_linear_converged = torch.empty(self.max_segments, **int_options)
        self._phase_one_witness_selected = torch.empty(self.max_segments, **int_options)
        self._phase_one_elastic = torch.empty(self.max_segments, **float_options)
        self._phase_one_rhs = torch.empty(self.max_segments, **float_options)
        self._phase_one_delta = torch.empty(self.max_segments, **float_options)
        self._phase_one_residual = torch.empty(self.max_segments, **float_options)
        self._phase_one_preconditioned = torch.empty(self.max_segments, **float_options)
        self._phase_one_direction = torch.empty(self.max_segments, **float_options)
        self._phase_one_operator = torch.empty(self.max_segments, **float_options)
        self._phase_one_diagonal = torch.empty(self.max_segments, **float_options)

        self._wp_joint_q = wp.from_torch(self._joint_q)
        self._wp_joint_q_next = wp.from_torch(self._joint_q_next)
        self._wp_base_weights = wp.from_torch(self._base_weights)
        self._wp_feature = wp.from_torch(self._feature)
        self._wp_objective_weights = wp.from_torch(self._objective_weights)
        self._wp_linearized_feature = wp.from_torch(self._linearized_feature)
        self._wp_constraint_kind = wp.from_torch(self._constraint_kind)
        self._wp_constraint_protected_kind = wp.from_torch(self._constraint_protected_kind)
        self._wp_precision_feature = wp.from_torch(self._precision_feature)
        self._wp_precision_diagonal = wp.from_torch(self._precision_diagonal)
        self._wp_dual_solution = wp.from_torch(self._dual_solution)
        self._wp_dual_r1 = wp.from_torch(self._dual_r1)
        self._wp_dual_r2 = wp.from_torch(self._dual_r2)
        self._wp_dual_z = wp.from_torch(self._dual_z)
        self._wp_dual_basis = wp.from_torch(self._dual_basis)
        self._wp_dual_direction_older = wp.from_torch(self._dual_direction_older)
        self._wp_dual_direction_old = wp.from_torch(self._dual_direction_old)
        self._wp_dual_direction = wp.from_torch(self._dual_direction)
        self._wp_active_row_codes = wp.from_torch(self._active_row_codes)
        self._wp_active_row_scales = wp.from_torch(self._active_row_scales)
        self._wp_active_equality_count = wp.from_torch(self._active_equality_count)
        self._wp_right_hand_side = wp.from_torch(self._right_hand_side)
        self._wp_active_row_transpose = wp.from_torch(self._active_row_transpose)
        self._wp_delta = wp.from_torch(self._delta)
        self._wp_delta_correction = wp.from_torch(self._delta_correction)
        self._wp_delta_candidate = wp.from_torch(self._delta_candidate)
        self._wp_minres_direction = wp.from_torch(self._minres_direction)
        self._wp_joint_velocity = wp.from_torch(self._joint_velocity)
        self._wp_joint_velocity_next = wp.from_torch(self._joint_velocity_next)
        self._wp_velocity_lower = wp.from_torch(self._velocity_lower)
        self._wp_velocity_upper = wp.from_torch(self._velocity_upper)
        self._wp_pcg_residual = wp.from_torch(self._pcg_residual)
        self._wp_pcg_preconditioned = wp.from_torch(self._pcg_preconditioned)
        self._wp_pcg_direction = wp.from_torch(self._pcg_direction)
        self._wp_pcg_operator_direction = wp.from_torch(self._pcg_operator_direction)
        self._wp_normal_diagonal = wp.from_torch(self._normal_diagonal)
        self._wp_ipm_temporal_factor = wp.from_torch(self._ipm_temporal_factor)
        self._wp_ipm_singleton_factor = wp.from_torch(self._ipm_singleton_factor)
        self._wp_ipm_frame_factor = wp.from_torch(self._ipm_frame_factor)
        self._wp_ipm_segment_coupled = wp.from_torch(self._ipm_segment_coupled)
        self._wp_ipm_block_band_factor = wp.from_torch(self._ipm_block_band_factor)
        self._wp_ipm_objective_block_band = wp.from_torch(self._ipm_objective_block_band)
        self._wp_ipm_objective_block_band_factor = wp.from_torch(self._ipm_objective_block_band_factor)
        self._wp_ipm_objective_factor_failed = wp.from_torch(self._ipm_objective_factor_failed)
        self._wp_ipm_block_band_scale_f64 = wp.from_torch(self._ipm_block_band_scale_f64)
        self._wp_ipm_recursive_converged = wp.from_torch(self._ipm_recursive_converged)
        self._wp_ipm_augmented_fallback = wp.from_torch(self._ipm_augmented_fallback)
        self._wp_ipm_solution_f64 = wp.from_torch(self._ipm_solution_f64)
        self._wp_ipm_right_hand_side_f64 = wp.from_torch(self._ipm_right_hand_side_f64)
        self._wp_ipm_primal_f64 = wp.from_torch(self._ipm_primal_f64)
        self._wp_ipm_feature_f64 = wp.from_torch(self._ipm_feature_f64)
        self._wp_ipm_precision_f64 = wp.from_torch(self._ipm_precision_f64)
        self._wp_ipm_operator_f64 = wp.from_torch(self._ipm_operator_f64)
        self._wp_ipm_dot_frame_f64 = wp.from_torch(self._ipm_dot_frame_f64)
        self._wp_ipm_residual_dot_f64 = wp.from_torch(self._ipm_residual_dot_f64)
        self._wp_phase_one_cross_f64 = wp.from_torch(self._phase_one_cross_f64)
        self._wp_phase_one_dot_frame_f64 = wp.from_torch(self._phase_one_dot_frame_f64)
        self._wp_phase_one_dot_segment_f64 = wp.from_torch(self._phase_one_dot_segment_f64)
        self._wp_ipm_initial_norm_f64 = wp.from_torch(self._ipm_initial_norm_f64)
        self._wp_ipm_augmented_primal_r2_f64 = wp.from_torch(self._ipm_augmented_primal_r2_f64)
        self._wp_ipm_augmented_primal_basis_f64 = wp.from_torch(self._ipm_augmented_primal_basis_f64)
        self._wp_ipm_augmented_primal_work_f64 = wp.from_torch(self._ipm_augmented_primal_work_f64)
        self._wp_ipm_augmented_primal_direction_older_f64 = wp.from_torch(
            self._ipm_augmented_primal_direction_older_f64
        )
        self._wp_ipm_augmented_primal_direction_old_f64 = wp.from_torch(self._ipm_augmented_primal_direction_old_f64)
        self._wp_ipm_augmented_row_solution_f64 = wp.from_torch(self._ipm_augmented_row_solution_f64)
        self._wp_ipm_augmented_row_r1_f64 = wp.from_torch(self._ipm_augmented_row_r1_f64)
        self._wp_ipm_augmented_row_r2_f64 = wp.from_torch(self._ipm_augmented_row_r2_f64)
        self._wp_ipm_augmented_row_basis_f64 = wp.from_torch(self._ipm_augmented_row_basis_f64)
        self._wp_ipm_augmented_row_direction_older_f64 = wp.from_torch(self._ipm_augmented_row_direction_older_f64)
        self._wp_ipm_augmented_row_direction_old_f64 = wp.from_torch(self._ipm_augmented_row_direction_old_f64)
        self._wp_pose_weights = wp.from_torch(self._pose_weights)
        self._wp_temporal_weights = wp.from_torch(self._temporal_weights)
        self._wp_inactive_residual_activity = wp.from_torch(self._inactive_residual_activity)
        self._wp_inactive_activity_group_by_residual = wp.from_torch(self._inactive_activity_group_by_residual)
        self._wp_residual_activity = self._wp_inactive_residual_activity
        self._wp_activity_group_by_residual = self._wp_inactive_activity_group_by_residual
        self._wp_first_difference_group_by_residual = self._wp_inactive_activity_group_by_residual
        self._wp_residual_upper = wp.from_torch(self._residual_upper)
        self._wp_inequality_upper = wp.from_torch(self._inequality_upper)
        self._wp_inequality_indices = wp.from_torch(self._inequality_indices)
        self._wp_coordinate_indices = wp.from_torch(self._coordinate_indices)
        self._wp_coordinate_dof_indices = wp.from_torch(self._coordinate_dof_indices)
        self._wp_coordinate_lower = wp.from_torch(self._coordinate_lower)
        self._wp_coordinate_upper = wp.from_torch(self._coordinate_upper)
        self._wp_segment_offsets = wp.from_torch(self._segment_offsets)
        self._wp_frame_segment = wp.from_torch(self._frame_segment)
        self._wp_step_seconds = wp.from_torch(self._step_seconds)
        self._wp_segment_damping = wp.from_torch(self._segment_damping)
        self._wp_inactive_equality_active = wp.from_torch(self._inactive_equality_active)
        self._wp_inactive_equality_residual_starts = wp.from_torch(self._inactive_equality_residual_starts)
        self._wp_dot_frame = wp.from_torch(self._dot_frame)
        self._wp_residual_dot = wp.from_torch(self._residual_dot)
        self._wp_residual_dot_next = wp.from_torch(self._residual_dot_next)
        self._wp_operator_dot = wp.from_torch(self._operator_dot)
        self._wp_objective_cost = wp.from_torch(self._objective_cost)
        self._wp_candidate_cost = wp.from_torch(self._candidate_cost)
        self._wp_accepted_cost = wp.from_torch(self._accepted_cost)
        self._wp_objective_term_count = wp.from_torch(self._objective_term_count)
        self._wp_segment_active = wp.from_torch(self._segment_active)
        self._wp_constraint_violation = wp.from_torch(self._constraint_violation)
        self._wp_candidate_violation = wp.from_torch(self._candidate_violation)
        self._wp_candidate_bound_violation = wp.from_torch(self._candidate_bound_violation)
        self._wp_candidate_protected_violation = wp.from_torch(self._candidate_protected_violation)
        self._wp_restoration_merit = wp.from_torch(self._restoration_merit)
        self._wp_candidate_restoration_merit = wp.from_torch(self._candidate_restoration_merit)
        self._wp_accepted_restoration_merit = wp.from_torch(self._accepted_restoration_merit)
        self._wp_restoration_constraint_scale = wp.from_torch(self._restoration_constraint_scale)
        self._wp_restoration_active_row_codes = wp.from_torch(self._restoration_active_row_codes)
        self._wp_restoration_active_row_scales = wp.from_torch(self._restoration_active_row_scales)
        self._wp_restoration_active_equality_count = wp.from_torch(self._restoration_active_equality_count)
        self._wp_minres_state = wp.from_torch(self._minres_state)
        self._wp_line_search_outcome = wp.from_torch(self._line_search_outcome)
        self._wp_line_search_take = wp.from_torch(self._line_search_take)
        self._wp_second_order_correction_direction = wp.from_torch(self._second_order_correction_direction)
        self._wp_second_order_correction_joint_q = wp.from_torch(self._second_order_correction_joint_q)
        self._wp_second_order_correction_state = wp.from_torch(self._second_order_correction_state)
        self._wp_second_order_correction_active = wp.from_torch(self._second_order_correction_active)
        self._wp_control_pending = wp.from_torch(self._control_pending)
        self._wp_constraint_contract_error = wp.from_torch(self._constraint_contract_error)
        self._wp_kkt_enabled = wp.from_torch(self._kkt_enabled)
        self._wp_ipm_certify = wp.from_torch(self._ipm_certify)
        self._wp_minres_enabled = wp.from_torch(self._minres_enabled)
        self._wp_minres_failed = wp.from_torch(self._minres_failed)
        self._wp_segment_feasible = wp.from_torch(self._segment_feasible)
        self._wp_globalization_succeeded = wp.from_torch(self._globalization_succeeded)
        self._wp_restoration_stalled = wp.from_torch(self._restoration_stalled)
        self._wp_ipm_constraint_scale = wp.from_torch(self._ipm_constraint_scale)
        self._wp_ipm_constraint_rhs = wp.from_torch(self._ipm_constraint_rhs)
        self._wp_ipm_slack = wp.from_torch(self._ipm_slack)
        self._wp_ipm_multiplier = wp.from_torch(self._ipm_multiplier)
        self._wp_ipm_primal_residual = wp.from_torch(self._ipm_primal_residual)
        self._wp_ipm_complementarity_residual = wp.from_torch(self._ipm_complementarity_residual)
        self._wp_ipm_weights = wp.from_torch(self._ipm_weights)
        self._wp_ipm_constraint_work = wp.from_torch(self._ipm_constraint_work)
        self._wp_ipm_affine_slack = wp.from_torch(self._ipm_affine_slack)
        self._wp_ipm_affine_multiplier = wp.from_torch(self._ipm_affine_multiplier)
        self._wp_ipm_primal = wp.from_torch(self._ipm_primal)
        self._wp_ipm_gradient = wp.from_torch(self._ipm_gradient)
        self._wp_ipm_transpose = wp.from_torch(self._ipm_transpose)
        self._wp_ipm_equality_dual = wp.from_torch(self._ipm_equality_dual)
        self._wp_ipm_equality_rhs = wp.from_torch(self._ipm_equality_rhs)
        self._wp_ipm_inequality_count = wp.from_torch(self._ipm_inequality_count)
        self._wp_ipm_complementarity = wp.from_torch(self._ipm_complementarity)
        self._wp_ipm_centering = wp.from_torch(self._ipm_centering)
        self._wp_ipm_primal_step = wp.from_torch(self._ipm_primal_step)
        self._wp_ipm_dual_step = wp.from_torch(self._ipm_dual_step)
        self._wp_ipm_enabled = wp.from_torch(self._ipm_enabled)
        self._wp_ipm_linear_converged = wp.from_torch(self._ipm_linear_converged)
        self._wp_phase_one_witness_selected = wp.from_torch(self._phase_one_witness_selected)
        self._wp_phase_one_elastic = wp.from_torch(self._phase_one_elastic)
        self._wp_phase_one_rhs = wp.from_torch(self._phase_one_rhs)
        self._wp_phase_one_delta = wp.from_torch(self._phase_one_delta)
        self._wp_phase_one_residual = wp.from_torch(self._phase_one_residual)
        self._wp_phase_one_preconditioned = wp.from_torch(self._phase_one_preconditioned)
        self._wp_phase_one_direction = wp.from_torch(self._phase_one_direction)
        self._wp_phase_one_operator = wp.from_torch(self._phase_one_operator)
        self._wp_phase_one_diagonal = wp.from_torch(self._phase_one_diagonal)
        self.workspace_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self._joint_q,
                self._joint_q_next,
                self._base_weights,
                self._feature,
                self._objective_weights,
                self._linearized_feature,
                self._constraint_kind,
                self._constraint_protected_kind,
                self._precision_feature,
                self._precision_diagonal,
                self._dual_solution,
                self._dual_r1,
                self._dual_r2,
                self._dual_z,
                self._dual_basis,
                self._dual_direction_older,
                self._dual_direction_old,
                self._dual_direction,
                self._active_row_codes,
                self._active_row_scales,
                self._active_equality_count,
                self._right_hand_side,
                self._active_row_transpose,
                self._delta,
                self._delta_correction,
                self._delta_candidate,
                self._minres_direction,
                self._joint_velocity,
                self._joint_velocity_next,
                self._velocity_lower,
                self._velocity_upper,
                self._pcg_residual,
                self._pcg_preconditioned,
                self._pcg_direction,
                self._pcg_operator_direction,
                self._normal_diagonal,
                self._ipm_temporal_factor,
                self._ipm_singleton_factor,
                self._ipm_block_band_factor,
                self._ipm_objective_block_band,
                self._ipm_objective_block_band_factor,
                self._ipm_objective_factor_failed,
                self._ipm_block_band_scale_f64,
                self._ipm_frame_factor,
                self._ipm_segment_coupled,
                self._ipm_recursive_converged,
                self._ipm_augmented_fallback,
                self._ipm_solution_f64,
                self._ipm_right_hand_side_f64,
                self._ipm_primal_f64,
                self._ipm_feature_f64,
                self._ipm_precision_f64,
                self._ipm_operator_f64,
                self._ipm_dot_frame_f64,
                self._ipm_residual_dot_f64,
                self._phase_one_cross_f64,
                self._phase_one_dot_frame_f64,
                self._phase_one_dot_segment_f64,
                self._ipm_initial_norm_f64,
                self._ipm_augmented_primal_r2_f64,
                self._ipm_augmented_primal_basis_f64,
                self._ipm_augmented_primal_work_f64,
                self._ipm_augmented_primal_direction_older_f64,
                self._ipm_augmented_primal_direction_old_f64,
                self._ipm_augmented_row_solution_f64,
                self._ipm_augmented_row_r1_f64,
                self._ipm_augmented_row_r2_f64,
                self._ipm_augmented_row_basis_f64,
                self._ipm_augmented_row_direction_older_f64,
                self._ipm_augmented_row_direction_old_f64,
                self._pose_weights,
                self._temporal_weights,
                self._inactive_residual_activity,
                self._inactive_activity_group_by_residual,
                self._residual_upper,
                self._inequality_upper,
                self._inequality_indices,
                self._coordinate_indices,
                self._coordinate_dof_indices,
                self._coordinate_lower,
                self._coordinate_upper,
                self._segment_offsets,
                self._frame_segment,
                self._step_seconds,
                self._segment_damping,
                self._inactive_equality_active,
                self._inactive_equality_residual_starts,
                self._dot_frame,
                self._residual_dot,
                self._residual_dot_next,
                self._operator_dot,
                self._objective_cost,
                self._candidate_cost,
                self._accepted_cost,
                self._objective_term_count,
                self._segment_active,
                self._constraint_violation,
                self._candidate_violation,
                self._candidate_bound_violation,
                self._candidate_protected_violation,
                self._restoration_merit,
                self._candidate_restoration_merit,
                self._accepted_restoration_merit,
                self._restoration_constraint_scale,
                self._restoration_active_row_codes,
                self._restoration_active_row_scales,
                self._restoration_active_equality_count,
                self._minres_state,
                self._line_search_outcome,
                self._line_search_take,
                self._second_order_correction_joint_q,
                self._second_order_correction_direction,
                self._second_order_correction_state,
                self._second_order_correction_active,
                self._second_order_correction_line_search_enabled,
                self._control_pending,
                self._constraint_contract_error,
                self._segment_feasible,
                self._globalization_succeeded,
                self._restoration_stalled,
                self._kkt_enabled,
                self._ipm_certify,
                self._minres_enabled,
                self._minres_failed,
                self._ipm_constraint_scale,
                self._ipm_constraint_rhs,
                self._ipm_slack,
                self._ipm_multiplier,
                self._ipm_primal_residual,
                self._ipm_complementarity_residual,
                self._ipm_weights,
                self._ipm_constraint_work,
                self._ipm_affine_slack,
                self._ipm_affine_multiplier,
                self._ipm_primal,
                self._ipm_gradient,
                self._ipm_transpose,
                self._ipm_equality_dual,
                self._ipm_equality_rhs,
                self._ipm_inequality_count,
                self._ipm_complementarity,
                self._ipm_centering,
                self._ipm_primal_step,
                self._ipm_dual_step,
                self._ipm_enabled,
                self._ipm_linear_converged,
                self._phase_one_witness_selected,
                self._phase_one_elastic,
                self._phase_one_rhs,
                self._phase_one_delta,
                self._phase_one_residual,
                self._phase_one_preconditioned,
                self._phase_one_direction,
                self._phase_one_operator,
                self._phase_one_diagonal,
            )
        )

    def solve(  # noqa: C901
        self,
        joint_q: torch.Tensor,
        joint_q_out: torch.Tensor,
        segment_offsets: torch.Tensor,
        step_seconds: torch.Tensor,
        pose_weights: torch.Tensor,
        temporal_weights: torch.Tensor,
        *,
        coordinate_bounds: CoordinateBounds,
        joint_velocity: torch.Tensor,
        velocity_lower: torch.Tensor,
        velocity_upper: torch.Tensor,
        segment_active: torch.Tensor,
        segment_damping: torch.Tensor | None = None,
        frozen_dof_indices: torch.Tensor | None = None,
        residual_activity: ResidualActivity | None = None,
        equalities: ResidualEqualities | None = None,
        inequalities: ResidualInequalities | None = None,
        segment_feasible: torch.Tensor | None = None,
        segment_direction_valid: torch.Tensor,
        segment_globalization_succeeded: torch.Tensor | None = None,
        segment_restoration_stalled: torch.Tensor | None = None,
        segment_residual_constraints_satisfied: torch.Tensor | None = None,
        convergence_tolerance: float | None = None,
        feasibility_only: bool = False,
    ) -> Statistics:
        """Refine complete segments in caller order.

        Args:
            joint_q: Initial generalized coordinates [m or rad, depending on
                joint type], shape [frame_count, coordinate_count].
            joint_q_out: Refined generalized coordinates with the same shape and units.
            segment_offsets: Prefix frame offsets, shape [segment_count + 1], int32.
            step_seconds: Sample period [s] per segment, shape [segment_count].
            pose_weights: Frame-local residual precision, shape [residual_count].
            temporal_weights: Difference precision, shape [3, residual_count].
            residual_activity: Optional compact confidence for base and temporal residual precision.
            equalities: Optional dense frame-local exact residual equalities.
            inequalities: Optional static residual upper bounds.
            coordinate_bounds: Scalar generalized-coordinate box bounds with
                explicit one-to-one tangent-DOF ownership. Every initial point
                must be feasible, and locked coordinates must equal their bound
                exactly.
            joint_velocity: Initial generalized velocities [m/s or rad/s,
                depending on joint type], shape [frame_count, dof_count].
                Values must be finite. Phase I restores finite initial edge
                rates outside :paramref:`velocity_lower` and :paramref:`velocity_upper`.
            velocity_lower: Generalized-velocity lower bounds
                [m/s or rad/s, depending on joint type], shape [dof_count].
            velocity_upper: Generalized-velocity upper bounds
                [m/s or rad/s, depending on joint type], shape [dof_count].
            frozen_dof_indices: Optional strictly increasing tangent-DOF
                indices excluded from this solve, shape [frozen_count], int32.
                Frozen DOFs must have unbounded velocity and must not own a
                scalar coordinate bound.
            segment_feasible: Optional per-segment geometric-feasibility output,
                shape [segment_count], bool. Numerical and input-contract failures
                still raise.
            segment_direction_valid: Caller-owned per-segment inner-direction
                validity output, shape [segment_count], bool. Unconstrained
                rows certify the PCG residual, constrained rows certify the
                final primal-dual Newton solve, and box-only rows certify
                either reduced-space PCG convergence or a finite
                curvature-scaled projected-gradient fallback. Geometric
                feasibility is reported by :paramref:`segment_feasible`.
            segment_globalization_succeeded: Optional caller-owned line-search
                outcome, shape [segment_count], bool. False identifies
                non-finite evidence, trial-ladder failure, or insufficient
                normalized restoration independently of the inner-direction certificate.
            segment_restoration_stalled: Optional caller-owned normalized-restoration
                stationarity output, shape [segment_count], bool. True is an explicit
                subtype of globalization failure, independent of inner-direction validity.
            segment_residual_constraints_satisfied: Optional caller-owned certificate for residual equalities and
                inequalities at the returned coordinates, shape [segment_count], bool. This is independent of
                geometric feasibility, inner-direction validity, and globalization.
            segment_active: Caller-owned active mask, shape
                [segment_count], int32. Zero rows remain frozen. With a convergence
                tolerance, model-stationary and failed rows become inactive while
                accepted-progress rows remain active for relinearization.
            segment_damping: Optional positive finite Gauss-Newton diagonal per
                segment, shape [segment_count], float32. The solver's configured
                scalar damping is used when omitted.
            convergence_tolerance: Maximum predicted descent per active objective scalar for model stationarity
                and minimum material relative normalized-restoration progress.
                Constrained segments must also be numerically feasible. Set to None to disable configured stationarity;
                Armijo/restoration and failure reporting remain active while the input active mask is preserved.
            feasibility_only: Retain a linearly feasible Phase-I witness when available. Objective-guided Phase II
                remains the fallback when Phase I is numerically inconclusive.


        Returns:
            Fixed solve and workspace statistics.
        """
        frame_count = joint_q.shape[0]
        segment_count = segment_offsets.shape[0] - 1
        if frame_count < 1 or frame_count > self.capacity or segment_count < 1 or segment_count > self.max_segments:
            raise ValueError("Trajectory frames or segments exceed the solver capacity.")
        if type(feasibility_only) is not bool:
            raise TypeError("feasibility_only must be a bool.")
        if convergence_tolerance is not None and (
            not math.isfinite(convergence_tolerance) or convergence_tolerance < 0.0
        ):
            raise ValueError("Trajectory convergence tolerance must be finite and nonnegative or None.")
        self._check_tensor(joint_q, (frame_count, self.coordinate_count), torch.float32, "joint_q")
        self._check_tensor(joint_q_out, joint_q.shape, torch.float32, "joint_q_out")
        self._check_tensor(segment_offsets, (segment_count + 1,), torch.int32, "segment_offsets")
        self._check_tensor(step_seconds, (segment_count,), torch.float32, "step_seconds")
        self._check_tensor(pose_weights, (self.residual_count,), torch.float32, "pose_weights")
        self._check_tensor(temporal_weights, (3, self.residual_count), torch.float32, "temporal_weights")
        if segment_feasible is not None:
            self._check_tensor(segment_feasible, (segment_count,), torch.bool, "segment_feasible")
        self._check_tensor(
            segment_direction_valid,
            (segment_count,),
            torch.bool,
            "segment_direction_valid",
        )
        if segment_residual_constraints_satisfied is not None:
            self._check_tensor(
                segment_residual_constraints_satisfied,
                (segment_count,),
                torch.bool,
                "segment_residual_constraints_satisfied",
            )
        segment_direction_valid_wp = wp.from_torch(segment_direction_valid)
        if segment_globalization_succeeded is None:
            globalization_succeeded = self._globalization_succeeded[:segment_count]
            segment_globalization_succeeded_wp = self._wp_globalization_succeeded
        else:
            self._check_tensor(
                segment_globalization_succeeded,
                (segment_count,),
                torch.bool,
                "segment_globalization_succeeded",
            )
            globalization_succeeded = segment_globalization_succeeded
            segment_globalization_succeeded_wp = wp.from_torch(segment_globalization_succeeded)
        globalization_succeeded.fill_(True)
        if segment_restoration_stalled is None:
            restoration_stalled = self._restoration_stalled[:segment_count]
            segment_restoration_stalled_wp = self._wp_restoration_stalled
        else:
            self._check_tensor(
                segment_restoration_stalled,
                (segment_count,),
                torch.bool,
                "segment_restoration_stalled",
            )
            restoration_stalled = segment_restoration_stalled
            segment_restoration_stalled_wp = wp.from_torch(segment_restoration_stalled)
        restoration_stalled.fill_(False)
        self._check_tensor(segment_active, (segment_count,), torch.int32, "segment_active")
        self._segment_active[:segment_count].copy_(segment_active)
        self._segment_damping[:segment_count].fill_(self.damping)
        if segment_damping is not None:
            self._check_tensor(segment_damping, (segment_count,), torch.float32, "segment_damping")
            self._segment_damping[:segment_count].copy_(segment_damping)
            self._constraint_contract_error.zero_()
            wp.launch(
                _segment_damping_validate,
                dim=self.max_segments,
                inputs=[self._wp_segment_damping, segment_count],
                outputs=[self._wp_constraint_contract_error],
                device=self.optimizer.device,
            )
            torch._assert_async(
                self._constraint_contract_error[0] == 0,
                "segment_damping must contain positive finite values.",
            )

        self._joint_q[:frame_count].copy_(joint_q)
        self._segment_offsets[: segment_count + 1].copy_(segment_offsets)
        self._step_seconds[:segment_count].copy_(step_seconds)
        self._pose_weights.copy_(pose_weights)
        self._temporal_weights.copy_(temporal_weights)
        bound_count = (
            coordinate_bounds.coordinate_indices.shape[0] if coordinate_bounds.coordinate_indices.ndim == 1 else -1
        )
        if bound_count < 0 or bound_count > min(self.coordinate_count, self.dof_count):
            raise ValueError("Coordinate-bound count must fit the scalar coordinate and tangent-DOF capacities.")
        self._check_tensor(
            coordinate_bounds.coordinate_indices,
            (bound_count,),
            torch.int32,
            "coordinate_bounds.coordinate_indices",
        )
        self._check_tensor(
            coordinate_bounds.dof_indices,
            (bound_count,),
            torch.int32,
            "coordinate_bounds.dof_indices",
        )
        self._check_tensor(coordinate_bounds.lower, (bound_count,), torch.float32, "coordinate_bounds.lower")
        self._check_tensor(coordinate_bounds.upper, (bound_count,), torch.float32, "coordinate_bounds.upper")
        self._coordinate_indices[:bound_count].copy_(coordinate_bounds.coordinate_indices)
        self._coordinate_dof_indices[:bound_count].copy_(coordinate_bounds.dof_indices)
        self._coordinate_lower[:bound_count].copy_(coordinate_bounds.lower)
        self._coordinate_upper[:bound_count].copy_(coordinate_bounds.upper)
        self._coordinate_bound_count = bound_count
        self._constraint_contract_error.zero_()
        wp.launch(
            _coordinate_bounds_validate,
            dim=max(1, frame_count * bound_count),
            inputs=[
                joint_q,
                self._wp_coordinate_indices,
                self._wp_coordinate_dof_indices,
                self._wp_coordinate_lower,
                self._wp_coordinate_upper,
                bound_count,
                frame_count,
                self.coordinate_count,
                self.dof_count,
            ],
            outputs=[self._wp_constraint_contract_error],
            device=self.optimizer.device,
        )
        torch._assert_async(
            self._constraint_contract_error[0] == 0,
            (
                "Coordinate bounds require unique in-range scalar coordinate/DOF pairs, valid finite-sided boxes, "
                "and an initially feasible point with locked coordinates exactly at their bound."
            ),
        )
        self._wp_residual_activity = self._wp_inactive_residual_activity
        self._wp_activity_group_by_residual = self._wp_inactive_activity_group_by_residual
        self._wp_first_difference_group_by_residual = self._wp_inactive_activity_group_by_residual
        if residual_activity is not None:
            group_count = residual_activity.values.shape[1] if residual_activity.values.ndim == 2 else -1
            if group_count < 1:
                raise ValueError("Residual activity requires at least one confidence group.")
            self._check_tensor(
                residual_activity.values,
                (frame_count, group_count),
                torch.float32,
                "residual_activity.values",
            )
            self._check_tensor(
                residual_activity.group_by_residual,
                (self.residual_count,),
                torch.int32,
                "residual_activity.group_by_residual",
            )
            if residual_activity.first_difference_group_by_residual is not None:
                self._check_tensor(
                    residual_activity.first_difference_group_by_residual,
                    (self.residual_count,),
                    torch.int32,
                    "residual_activity.first_difference_group_by_residual",
                )
            self._wp_residual_activity = wp.from_torch(residual_activity.values)
            self._wp_activity_group_by_residual = wp.from_torch(residual_activity.group_by_residual)
            if residual_activity.first_difference_group_by_residual is not None:
                self._wp_first_difference_group_by_residual = wp.from_torch(
                    residual_activity.first_difference_group_by_residual
                )
            self._constraint_contract_error.zero_()
            wp.launch(
                _residual_activity_validate,
                dim=max(self.residual_count, frame_count * group_count),
                inputs=[
                    self._wp_residual_activity,
                    self._wp_activity_group_by_residual,
                    self._wp_first_difference_group_by_residual,
                    frame_count,
                    group_count,
                ],
                outputs=[self._wp_constraint_contract_error],
                device=self.optimizer.device,
            )
            torch._assert_async(
                self._constraint_contract_error[0] == 0,
                "Residual confidence must be finite in [0, 1] and every group index must be -1 or in range.",
            )
        self._objective_weights[:frame_count].copy_(self._pose_weights)
        wp.launch(
            _frame_segment_indices,
            dim=self.capacity,
            inputs=[self._wp_segment_offsets, segment_count, frame_count],
            outputs=[self._wp_frame_segment],
            device=self.optimizer.device,
        )

        equality_target_count = 0
        equality_active_wp = self._wp_inactive_equality_active
        equality_residual_starts_wp = self._wp_inactive_equality_residual_starts
        if equalities is not None:
            equality_target_count = equalities.residual_starts_by_target.shape[0]
            if equality_target_count < 1:
                raise ValueError("Residual equalities must contain at least one target.")
            if equalities.residual_width != 3:
                raise ValueError("Residual equality width must be exactly three.")
            if 3 * equality_target_count > self.max_equality_residuals_per_frame:
                raise ValueError("Per-frame equality residual count exceeds the solver capacity.")
            self._check_tensor(
                equalities.active,
                (frame_count, equality_target_count),
                torch.uint8,
                "equalities.active",
            )
            self._check_tensor(
                equalities.residual_starts_by_target,
                (equality_target_count,),
                torch.int32,
                "equalities.residual_starts_by_target",
            )
            equality_active_wp = wp.from_torch(equalities.active)
            equality_residual_starts_wp = wp.from_torch(equalities.residual_starts_by_target)
        self._check_tensor(joint_velocity, (frame_count, self.dof_count), torch.float32, "joint_velocity")
        self._check_tensor(velocity_lower, (self.dof_count,), torch.float32, "velocity_lower")
        self._check_tensor(velocity_upper, (self.dof_count,), torch.float32, "velocity_upper")
        torch._assert_async(
            torch.all(torch.isfinite(joint_velocity)) & torch.all(velocity_lower <= velocity_upper),
            "Velocity bounds must be ordered and initial generalized velocities must be finite.",
        )
        self._joint_velocity[:frame_count].copy_(joint_velocity)
        self._velocity_lower.copy_(velocity_lower)
        self._velocity_upper.copy_(velocity_upper)
        frozen_dof_count = 0
        frozen_dof_indices_wp = self._wp_coordinate_dof_indices
        if frozen_dof_indices is not None:
            frozen_dof_count = frozen_dof_indices.shape[0] if frozen_dof_indices.ndim == 1 else -1
            if frozen_dof_count < 1 or frozen_dof_count > self.dof_count:
                raise ValueError("Frozen tangent-DOF count must fit the optimizer tangent width.")
            self._check_tensor(
                frozen_dof_indices,
                (frozen_dof_count,),
                torch.int32,
                "frozen_dof_indices",
            )
            frozen_dof_indices_wp = wp.from_torch(frozen_dof_indices)
            self._constraint_contract_error.zero_()
            wp.launch(
                _frozen_dof_indices_validate,
                dim=frozen_dof_count,
                inputs=[
                    frozen_dof_indices_wp,
                    self._wp_coordinate_dof_indices,
                    self._wp_velocity_lower,
                    self._wp_velocity_upper,
                    frozen_dof_count,
                    bound_count,
                    self.dof_count,
                ],
                outputs=[self._wp_constraint_contract_error],
                device=self.optimizer.device,
            )
            torch._assert_async(
                self._constraint_contract_error[0] == 0,
                (
                    "Frozen tangent DOFs must be sorted, unique, in range, velocity-unbounded, "
                    "and disjoint from scalar coordinate bounds."
                ),
            )
        velocity_constrained = (self.device.type == "cuda" and self.optimizer.device.is_capturing) or bool(
            torch.any(torch.isfinite(velocity_lower)) or torch.any(torch.isfinite(velocity_upper))
        )
        projected_bounds = bound_count > 0 and equalities is None and inequalities is None and not velocity_constrained
        constrained = (
            equalities is not None
            or inequalities is not None
            or (bound_count > 0 and not projected_bounds)
            or velocity_constrained
        )
        inequality_count = 0
        if constrained:
            if inequalities is not None:
                inequality_count = inequalities.residual_indices.shape[0]
                if inequality_count > self.residual_count:
                    raise ValueError("Inequality count exceeds the residual width.")
                self._check_tensor(
                    inequalities.residual_indices,
                    (inequality_count,),
                    torch.int32,
                    "inequalities.residual_indices",
                )
                self._check_tensor(inequalities.upper, (inequality_count,), torch.float32, "inequalities.upper")
                self._inequality_indices[:inequality_count].copy_(inequalities.residual_indices)
                self._inequality_upper[:inequality_count].copy_(inequalities.upper)
            self._residual_upper.fill_(torch.inf)
            self._constraint_contract_error.zero_()
            wp.launch(
                _constraints_initialize,
                dim=(self.capacity, self.residual_count),
                inputs=[self._wp_pose_weights, frame_count],
                outputs=[self._wp_constraint_kind, self._wp_objective_weights],
                device=self.optimizer.device,
            )
            wp.launch(
                _inequalities_mark,
                dim=(self.capacity, max(1, inequality_count)),
                inputs=[
                    self._wp_inequality_indices,
                    self._wp_inequality_upper,
                    inequality_count,
                    frame_count,
                    self.residual_count,
                ],
                outputs=[
                    self._wp_constraint_kind,
                    self._wp_objective_weights,
                    self._wp_residual_upper,
                    self._wp_constraint_contract_error,
                ],
                device=self.optimizer.device,
            )
            if equalities is not None:
                wp.launch(
                    _equalities_mark,
                    dim=(self.capacity, equality_target_count, 3),
                    inputs=[
                        equality_active_wp,
                        equality_residual_starts_wp,
                        equality_target_count,
                        frame_count,
                        self.residual_count,
                    ],
                    outputs=[
                        self._wp_constraint_kind,
                        self._wp_objective_weights,
                        self._wp_constraint_contract_error,
                    ],
                    device=self.optimizer.device,
                )
            torch._assert_async(
                self._constraint_contract_error[0] == 0,
                (
                    "Inequality rows must be sorted, unique, finite, in range, and disjoint from residual equalities; "
                    "equality activity and ranges must be valid."
                ),
            )
        self._base_weights[:frame_count].copy_(self._objective_weights[:frame_count])
        self._segment_feasible[:segment_count].fill_(1)
        self._objective_term_count[:segment_count].zero_()
        wp.launch(
            _objective_term_count,
            dim=(self.capacity, self.residual_count),
            inputs=[
                self._wp_base_weights,
                self._wp_temporal_weights,
                self._wp_residual_activity,
                self._wp_activity_group_by_residual,
                self._wp_first_difference_group_by_residual,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_segment_active,
                frame_count,
            ],
            outputs=[self._wp_objective_term_count],
            device=self.optimizer.device,
        )

        self.optimizer.linearize(
            self._wp_joint_q[:frame_count],
            residuals=self._wp_linearized_feature[:frame_count],
        )
        if frozen_dof_count > 0:
            wp.launch(
                _frozen_dof_jacobian_zero,
                dim=(self.capacity, self.residual_count, frozen_dof_count),
                inputs=[
                    self._wp_jacobian,
                    frozen_dof_indices_wp,
                    frozen_dof_count,
                    frame_count,
                ],
                device=self.optimizer.device,
            )
        self._trajectory_cost(
            self._linearized_feature,
            self._wp_linearized_feature,
            self._wp_objective_weights,
            frame_count,
            segment_count,
            self._objective_cost,
        )
        self._line_search_take[:segment_count].fill_(1)
        self._restoration_merit[:segment_count].zero_()
        self._accepted_restoration_merit[:segment_count].zero_()
        line_search_enabled_wp = self._wp_segment_active
        if constrained:
            self._solve_ipm_linearized(
                frame_count,
                segment_count,
                inequality_count,
                bound_count,
                self._wp_line_search_take,
                has_equalities=equalities is not None,
                joint_q=self._wp_joint_q,
                joint_velocity=self._wp_joint_velocity,
                include_linear_term=True,
                feasibility_only=feasibility_only,
            )
            self._ipm_inequality_count[:segment_count].zero_()
            wp.launch(
                _ipm_constraints_initialize,
                dim=(self.capacity, self.inequality_width),
                inputs=[
                    self._wp_jacobian,
                    self._wp_linearized_feature,
                    self._wp_residual_upper,
                    self._wp_constraint_kind,
                    self._wp_joint_q,
                    self._wp_coordinate_indices,
                    self._wp_coordinate_dof_indices,
                    self._wp_coordinate_lower,
                    self._wp_coordinate_upper,
                    bound_count,
                    self._wp_joint_velocity,
                    self._wp_velocity_lower,
                    self._wp_velocity_upper,
                    self._wp_step_seconds,
                    self._wp_normal_diagonal,
                    self._wp_frame_segment,
                    self._wp_segment_offsets,
                    self._wp_segment_active,
                    frame_count,
                    float(torch.finfo(torch.float32).eps),
                ],
                outputs=[
                    self._wp_restoration_constraint_scale,
                    self._wp_ipm_constraint_rhs,
                    self._wp_ipm_inequality_count,
                ],
                device=self.optimizer.device,
            )
            self._restoration_active_row_codes[:frame_count].copy_(self._active_row_codes[:frame_count])
            self._restoration_active_row_scales[:frame_count].copy_(self._active_row_scales[:frame_count])
            self._restoration_active_equality_count[:segment_count].copy_(self._active_equality_count[:segment_count])
            self._restoration_merit[:segment_count].zero_()
            wp.launch(
                _restoration_current_merit_max,
                dim=(self.capacity, max(self.inequality_width, self.active_width)),
                inputs=[
                    self._wp_ipm_constraint_rhs,
                    self._wp_restoration_constraint_scale,
                    self._wp_ipm_equality_rhs,
                    self._wp_restoration_active_row_codes,
                    self._wp_restoration_active_equality_count,
                    self._wp_frame_segment,
                    self._wp_segment_offsets,
                    self._wp_segment_active,
                    self.inequality_width,
                    frame_count,
                ],
                outputs=[self._wp_restoration_merit],
                device=self.optimizer.device,
            )
            self._accepted_restoration_merit[:segment_count].copy_(self._restoration_merit[:segment_count])
            line_search_enabled_wp = self._wp_kkt_enabled
            self._constraint_measure(
                self._wp_linearized_feature,
                frame_count,
                self._constraint_violation,
            )
            self._hard_bound_measure(
                self._wp_joint_q,
                self._wp_joint_velocity,
                bound_count,
                frame_count,
            )
            torch.maximum(self._constraint_violation, self._candidate_bound_violation, out=self._constraint_violation)
            wp.launch(
                _constraint_feasible_rows_mark,
                dim=(self.capacity, self.residual_count),
                inputs=[
                    self._wp_linearized_feature,
                    self._wp_residual_upper,
                    self._wp_constraint_kind,
                    self._wp_frame_segment,
                    self._wp_segment_active,
                    frame_count,
                    64.0 * float(torch.finfo(torch.float32).eps),
                ],
                outputs=[self._wp_constraint_protected_kind],
                device=self.optimizer.device,
            )
        else:
            self._solve_linearized(
                self._linearized_feature,
                self._wp_linearized_feature,
                self._jacobian[:frame_count],
                self._wp_jacobian,
                frame_count,
                segment_count,
                self._wp_line_search_take,
                projected=projected_bounds,
            )
        if frozen_dof_count > 0:
            wp.launch(
                _frozen_dof_values_zero,
                dim=(self.capacity, frozen_dof_count),
                inputs=[
                    self._wp_delta,
                    frozen_dof_indices_wp,
                    frozen_dof_count,
                    frame_count,
                ],
                device=self.optimizer.device,
            )
        segment_direction_valid.copy_(self._line_search_take[:segment_count])
        if not constrained:
            self._kkt_enabled[:segment_count].copy_(self._line_search_take[:segment_count])
            self._kkt_enabled[:segment_count].mul_(self._segment_active[:segment_count])
            line_search_enabled_wp = self._wp_kkt_enabled
        wp.launch(
            _precision_apply,
            dim=(self.capacity, self.residual_count),
            inputs=[
                self._wp_linearized_feature,
                self._wp_base_weights,
                self._wp_temporal_weights,
                self._wp_residual_activity,
                self._wp_activity_group_by_residual,
                self._wp_first_difference_group_by_residual,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_step_seconds,
                self._wp_segment_active,
                frame_count,
            ],
            outputs=[self._wp_precision_feature],
            device=self.optimizer.device,
        )
        torch.bmm(
            self._jacobian[:frame_count].transpose(1, 2),
            self._precision_feature[:frame_count].unsqueeze(-1),
            out=self._right_hand_side[:frame_count].unsqueeze(-1),
        )
        self._dot(
            self._right_hand_side,
            self._delta,
            self._residual_dot,
            frame_count,
            segment_count,
        )
        convergence_gradient_dot_wp = self._wp_operator_dot if projected_bounds else self._wp_residual_dot
        wp.launch(
            _line_search_stationarity_initialize,
            dim=segment_count,
            inputs=[
                self._wp_objective_cost,
                self._wp_residual_dot,
                self._wp_objective_term_count,
                convergence_tolerance if convergence_tolerance is not None else -1.0,
                self._wp_constraint_violation,
                64.0 * float(torch.finfo(torch.float32).eps),
                wp.uint8(1 if constrained else 0),
                segment_direction_valid_wp,
                line_search_enabled_wp,
            ],
            outputs=[self._wp_line_search_outcome, self._wp_line_search_take],
            device=self.optimizer.device,
        )
        for line_search_step in self.line_search_steps:
            wp.launch(
                _delta_scale,
                dim=(self.capacity, self.dof_count),
                inputs=[
                    self._wp_delta,
                    line_search_step,
                    self._wp_frame_segment,
                    line_search_enabled_wp,
                    frame_count,
                ],
                outputs=[self._wp_delta_candidate],
                device=self.optimizer.device,
            )
            wp.launch(
                _velocity_candidate,
                dim=(self.capacity, self.dof_count),
                inputs=[
                    self._wp_delta_candidate,
                    self._wp_joint_velocity,
                    self._wp_frame_segment,
                    self._wp_segment_offsets,
                    self._wp_step_seconds,
                    line_search_enabled_wp,
                    frame_count,
                ],
                outputs=[self._wp_joint_velocity_next],
                device=self.optimizer.device,
            )
            self.optimizer.integrate(
                self._wp_joint_q[:frame_count],
                self._wp_delta_candidate[:frame_count],
                self._wp_joint_q_next[:frame_count],
                step_size=1.0,
            )
            if bound_count > 0:
                wp.launch(
                    _coordinate_bounds_project_candidate,
                    dim=(self.capacity, bound_count),
                    inputs=[
                        self._wp_joint_q,
                        self._wp_joint_q_next,
                        self._wp_delta_candidate,
                        self._wp_coordinate_indices,
                        self._wp_coordinate_dof_indices,
                        self._wp_coordinate_lower,
                        self._wp_coordinate_upper,
                        bound_count,
                        frame_count,
                    ],
                    device=self.optimizer.device,
                )
                if constrained:
                    wp.launch(
                        _velocity_candidate,
                        dim=(self.capacity, self.dof_count),
                        inputs=[
                            self._wp_delta_candidate,
                            self._wp_joint_velocity,
                            self._wp_frame_segment,
                            self._wp_segment_offsets,
                            self._wp_step_seconds,
                            line_search_enabled_wp,
                            frame_count,
                        ],
                        outputs=[self._wp_joint_velocity_next],
                        device=self.optimizer.device,
                    )
            if projected_bounds:
                self._dot(
                    self._right_hand_side,
                    self._delta_candidate,
                    self._operator_dot,
                    frame_count,
                    segment_count,
                )
                if line_search_step == self.line_search_steps[0]:
                    wp.launch(
                        _line_search_stationarity_initialize,
                        dim=segment_count,
                        inputs=[
                            self._wp_objective_cost,
                            self._wp_operator_dot,
                            self._wp_objective_term_count,
                            convergence_tolerance if convergence_tolerance is not None else -1.0,
                            self._wp_constraint_violation,
                            64.0 * float(torch.finfo(torch.float32).eps),
                            wp.uint8(0),
                            segment_direction_valid_wp,
                            line_search_enabled_wp,
                        ],
                        outputs=[self._wp_line_search_outcome, self._wp_line_search_take],
                        device=self.optimizer.device,
                    )
            self.optimizer.compute_residuals(
                self._wp_joint_q_next[:frame_count],
                residuals=self._wp_linearized_feature[:frame_count],
            )
            self._trajectory_cost(
                self._linearized_feature,
                self._wp_linearized_feature,
                self._wp_base_weights,
                frame_count,
                segment_count,
                self._candidate_cost,
            )
            if constrained:
                self._constraint_measure(
                    self._wp_linearized_feature,
                    frame_count,
                    self._candidate_violation,
                )
                self._hard_bound_measure(
                    self._wp_joint_q_next,
                    self._wp_joint_velocity_next,
                    bound_count,
                    frame_count,
                )
                self._constraint_protected_measure(self._wp_linearized_feature, frame_count)
                torch.maximum(
                    self._candidate_violation[:segment_count],
                    self._candidate_bound_violation[:segment_count],
                    out=self._candidate_violation[:segment_count],
                )
                self._candidate_restoration_merit[:segment_count].zero_()
                wp.launch(
                    _restoration_candidate_merit_max,
                    dim=(self.capacity, max(self.inequality_width, self.active_width)),
                    inputs=[
                        self._wp_linearized_feature,
                        self._wp_residual_upper,
                        self._wp_joint_q_next,
                        self._wp_coordinate_indices,
                        self._wp_coordinate_lower,
                        self._wp_coordinate_upper,
                        bound_count,
                        self._wp_joint_velocity_next,
                        self._wp_velocity_lower,
                        self._wp_velocity_upper,
                        self._wp_step_seconds,
                        self._wp_restoration_constraint_scale,
                        self._wp_restoration_active_row_codes,
                        self._wp_restoration_active_row_scales,
                        self._wp_restoration_active_equality_count,
                        self._wp_frame_segment,
                        self._wp_segment_offsets,
                        line_search_enabled_wp,
                        self.inequality_width,
                        frame_count,
                    ],
                    outputs=[self._wp_candidate_restoration_merit],
                    device=self.optimizer.device,
                )
                wp.launch(
                    _constraint_line_search_decide,
                    dim=segment_count,
                    inputs=[
                        self._wp_objective_cost,
                        self._wp_candidate_cost,
                        self._wp_constraint_violation,
                        self._wp_candidate_violation,
                        self._wp_candidate_bound_violation,
                        self._wp_candidate_protected_violation,
                        self._wp_restoration_merit,
                        self._wp_candidate_restoration_merit,
                        64.0 * float(torch.finfo(torch.float32).eps),
                        line_search_step,
                        self._wp_residual_dot,
                        line_search_enabled_wp,
                        self._wp_line_search_outcome,
                    ],
                    outputs=[self._wp_line_search_take],
                    device=self.optimizer.device,
                )
                wp.launch(
                    _accepted_violation_update,
                    dim=segment_count,
                    inputs=[
                        self._wp_candidate_violation,
                        self._wp_line_search_take,
                        line_search_enabled_wp,
                    ],
                    outputs=[self._wp_constraint_violation],
                    device=self.optimizer.device,
                )
                wp.launch(
                    _accepted_violation_update,
                    dim=segment_count,
                    inputs=[
                        self._wp_candidate_restoration_merit,
                        self._wp_line_search_take,
                        line_search_enabled_wp,
                    ],
                    outputs=[self._wp_accepted_restoration_merit],
                    device=self.optimizer.device,
                )
            elif projected_bounds:
                wp.launch(
                    _projected_line_search_decide,
                    dim=segment_count,
                    inputs=[
                        self._wp_objective_cost,
                        self._wp_candidate_cost,
                        self._wp_operator_dot,
                        line_search_enabled_wp,
                        self._wp_line_search_outcome,
                    ],
                    outputs=[self._wp_line_search_take],
                    device=self.optimizer.device,
                )
            else:
                wp.launch(
                    _line_search_decide,
                    dim=segment_count,
                    inputs=[
                        self._wp_objective_cost,
                        self._wp_candidate_cost,
                        line_search_step,
                        self._wp_residual_dot,
                        line_search_enabled_wp,
                        self._wp_line_search_outcome,
                    ],
                    outputs=[self._wp_line_search_take],
                    device=self.optimizer.device,
                )
            wp.launch(
                _conditional_configuration_copy,
                dim=(self.capacity, self.coordinate_count),
                inputs=[
                    self._wp_joint_q_next,
                    self._wp_frame_segment,
                    line_search_enabled_wp,
                    self._wp_line_search_take,
                    frame_count,
                ],
                outputs=[self._wp_joint_q],
                device=self.optimizer.device,
            )
            wp.launch(
                _conditional_configuration_copy,
                dim=(self.capacity, self.dof_count),
                inputs=[
                    self._wp_joint_velocity_next,
                    self._wp_frame_segment,
                    line_search_enabled_wp,
                    self._wp_line_search_take,
                    frame_count,
                ],
                outputs=[self._wp_joint_velocity],
                device=self.optimizer.device,
            )
            if constrained and line_search_step == self.line_search_steps[0]:
                self._second_order_correct(
                    frame_count,
                    segment_count,
                    inequality_count,
                    bound_count,
                    segment_direction_valid_wp,
                    line_search_step,
                    has_equalities=equalities is not None,
                    frozen_dof_indices=frozen_dof_indices_wp,
                    frozen_dof_count=frozen_dof_count,
                    feasibility_only=feasibility_only,
                )
            if self.device.type != "cuda" or not self.optimizer.device.is_capturing:
                self._control_pending.zero_()
                wp.launch(
                    _line_search_pending,
                    dim=segment_count,
                    inputs=[
                        line_search_enabled_wp,
                        self._wp_line_search_outcome,
                        segment_count,
                    ],
                    outputs=[self._wp_control_pending],
                    device=self.optimizer.device,
                )
                if not bool(self._control_pending[0]):
                    break
        if segment_residual_constraints_satisfied is not None:
            if equalities is None and inequalities is None:
                segment_residual_constraints_satisfied.fill_(True)
            else:
                self.optimizer.compute_residuals(
                    self._wp_joint_q[:frame_count],
                    residuals=self._wp_linearized_feature[:frame_count],
                )
                self._constraint_measure(
                    self._wp_linearized_feature,
                    frame_count,
                    self._candidate_violation,
                )
                torch.le(
                    self._candidate_violation[:segment_count],
                    64.0 * float(torch.finfo(torch.float32).eps),
                    out=segment_residual_constraints_satisfied,
                )
        wp.launch(
            _segment_convergence_update,
            dim=self.max_segments,
            inputs=[
                self._wp_objective_cost,
                self._wp_objective_term_count,
                segment_count,
                convergence_tolerance if convergence_tolerance is not None else -1.0,
                self._wp_constraint_violation,
                64.0 * float(torch.finfo(torch.float32).eps),
                wp.uint8(1 if constrained else 0),
                self._wp_line_search_outcome,
                convergence_gradient_dot_wp,
                self._wp_restoration_merit,
                self._wp_accepted_restoration_merit,
            ],
            outputs=[
                segment_direction_valid_wp,
                segment_globalization_succeeded_wp,
                segment_restoration_stalled_wp,
                self._wp_segment_active,
            ],
            device=self.optimizer.device,
        )
        joint_q_out.copy_(self._joint_q[:frame_count])
        if segment_feasible is not None:
            segment_feasible.copy_(self._segment_feasible[:segment_count])
        segment_active.copy_(self._segment_active[:segment_count])
        return self.Statistics(
            frame_count=frame_count,
            segment_count=segment_count,
            equality_target_count=equality_target_count,
            krylov_max_iterations=self.krylov_max_iterations,
            workspace_bytes=self.workspace_bytes,
        )

    def _second_order_correct(
        self,
        frame_count: int,
        segment_count: int,
        inequality_count: int,
        coordinate_bound_count: int,
        outer_linear_converged: wp.array,
        trial_scale: float,
        *,
        has_equalities: bool,
        frozen_dof_indices: wp.array,
        frozen_dof_count: int,
        feasibility_only: bool,
    ) -> None:
        """Correct a tangent trial blocked by nonlinear constraint curvature."""
        numerical_tolerance = 64.0 * float(torch.finfo(torch.float32).eps)
        wp.launch(
            _second_order_correction_request,
            dim=segment_count,
            inputs=[
                self._wp_objective_cost,
                self._wp_candidate_cost,
                self._wp_constraint_violation,
                self._wp_candidate_violation,
                self._wp_candidate_bound_violation,
                self._wp_candidate_protected_violation,
                self._wp_restoration_merit,
                self._wp_candidate_restoration_merit,
                trial_scale,
                self._wp_residual_dot,
                numerical_tolerance,
                outer_linear_converged,
                self._wp_segment_feasible,
                self._wp_segment_active,
                self._wp_line_search_outcome,
            ],
            outputs=[self._wp_line_search_take],
            device=self.optimizer.device,
        )
        if self.device.type != "cuda" or not self.optimizer.device.is_capturing:
            self._control_pending.zero_()
            wp.launch(
                _segments_pending,
                dim=segment_count,
                inputs=[self._wp_line_search_take, segment_count],
                outputs=[self._wp_control_pending],
                device=self.optimizer.device,
            )
            if not bool(self._control_pending[0]):
                return

        self._second_order_correction_state[:segment_count, 0].copy_(self._residual_dot[:segment_count])
        self._second_order_correction_state[:segment_count, 1].copy_(self._constraint_violation[:segment_count])
        self._second_order_correction_state[:segment_count, 2].copy_(self._candidate_violation[:segment_count])
        self._second_order_correction_state[:segment_count, 3].copy_(
            self._candidate_protected_violation[:segment_count]
        )
        self._second_order_correction_state[:segment_count, 4].copy_(self._restoration_merit[:segment_count])
        self._second_order_correction_state[:segment_count, 5].copy_(self._candidate_restoration_merit[:segment_count])
        self._second_order_correction_direction[:frame_count].copy_(self._delta[:frame_count])
        self._second_order_correction_active[:segment_count].copy_(self._segment_active[:segment_count])
        self._second_order_correction_line_search_enabled[:segment_count].copy_(self._kkt_enabled[:segment_count])
        self._segment_active[:segment_count].copy_(self._line_search_take[:segment_count])
        for _ in range(_SECOND_ORDER_CORRECTION_ITERATIONS):
            self._second_order_correction_joint_q[:frame_count].copy_(self._joint_q_next[:frame_count])
            self._solve_ipm_linearized(
                frame_count,
                segment_count,
                inequality_count,
                coordinate_bound_count,
                self._wp_ipm_certify,
                has_equalities=has_equalities,
                joint_q=self._wp_joint_q_next,
                joint_velocity=self._wp_joint_velocity_next,
                include_linear_term=False,
                feasibility_only=feasibility_only,
            )
            if frozen_dof_count > 0:
                wp.launch(
                    _frozen_dof_values_zero,
                    dim=(self.capacity, frozen_dof_count),
                    inputs=[
                        self._wp_delta,
                        frozen_dof_indices,
                        frozen_dof_count,
                        frame_count,
                    ],
                    device=self.optimizer.device,
                )
            self.optimizer.integrate(
                self._wp_joint_q_next[:frame_count],
                self._wp_delta[:frame_count],
                self._wp_joint_q_next[:frame_count],
                step_size=1.0,
            )
            if coordinate_bound_count > 0:
                wp.launch(
                    _coordinate_bounds_project_candidate,
                    dim=(self.capacity, coordinate_bound_count),
                    inputs=[
                        self._wp_second_order_correction_joint_q,
                        self._wp_joint_q_next,
                        self._wp_delta,
                        self._wp_coordinate_indices,
                        self._wp_coordinate_dof_indices,
                        self._wp_coordinate_lower,
                        self._wp_coordinate_upper,
                        coordinate_bound_count,
                        frame_count,
                    ],
                    device=self.optimizer.device,
                )
            wp.launch(
                _velocity_candidate,
                dim=(self.capacity, self.dof_count),
                inputs=[
                    self._wp_delta,
                    self._wp_joint_velocity_next,
                    self._wp_frame_segment,
                    self._wp_segment_offsets,
                    self._wp_step_seconds,
                    self._wp_segment_active,
                    frame_count,
                ],
                outputs=[self._wp_joint_velocity_next],
                device=self.optimizer.device,
            )
            self.optimizer.compute_residuals(
                self._wp_joint_q_next[:frame_count],
                residuals=self._wp_linearized_feature[:frame_count],
            )
            self._trajectory_cost(
                self._linearized_feature,
                self._wp_linearized_feature,
                self._wp_base_weights,
                frame_count,
                segment_count,
                self._candidate_cost,
            )
            self._constraint_measure(self._wp_linearized_feature, frame_count, self._candidate_violation)
            self._hard_bound_measure(
                self._wp_joint_q_next,
                self._wp_joint_velocity_next,
                coordinate_bound_count,
                frame_count,
            )
            self._constraint_protected_measure(self._wp_linearized_feature, frame_count)
            torch.maximum(
                self._candidate_violation[:segment_count],
                self._candidate_bound_violation[:segment_count],
                out=self._candidate_violation[:segment_count],
            )
            self._candidate_restoration_merit[:segment_count].zero_()
            wp.launch(
                _restoration_candidate_merit_max,
                dim=(self.capacity, max(self.inequality_width, self.active_width)),
                inputs=[
                    self._wp_linearized_feature,
                    self._wp_residual_upper,
                    self._wp_joint_q_next,
                    self._wp_coordinate_indices,
                    self._wp_coordinate_lower,
                    self._wp_coordinate_upper,
                    coordinate_bound_count,
                    self._wp_joint_velocity_next,
                    self._wp_velocity_lower,
                    self._wp_velocity_upper,
                    self._wp_step_seconds,
                    self._wp_restoration_constraint_scale,
                    self._wp_restoration_active_row_codes,
                    self._wp_restoration_active_row_scales,
                    self._wp_restoration_active_equality_count,
                    self._wp_frame_segment,
                    self._wp_segment_offsets,
                    self._wp_segment_active,
                    self.inequality_width,
                    frame_count,
                ],
                outputs=[self._wp_candidate_restoration_merit],
                device=self.optimizer.device,
            )
            self._residual_dot[:segment_count].copy_(self._second_order_correction_state[:segment_count, 0])
            self._constraint_violation[:segment_count].copy_(self._second_order_correction_state[:segment_count, 1])
            wp.launch(
                _second_order_correction_decide,
                dim=segment_count,
                inputs=[
                    self._wp_objective_cost,
                    self._wp_candidate_cost,
                    self._wp_candidate_violation,
                    self._wp_candidate_bound_violation,
                    self._wp_candidate_protected_violation,
                    self._wp_candidate_restoration_merit,
                    trial_scale,
                    self._wp_residual_dot,
                    numerical_tolerance,
                    self._wp_ipm_certify,
                    self._wp_segment_feasible,
                    self._wp_segment_active,
                    self._wp_second_order_correction_state,
                    self._wp_line_search_outcome,
                ],
                outputs=[self._wp_ipm_enabled, self._wp_kkt_enabled],
                device=self.optimizer.device,
            )
            for candidate, current, width in (
                (self._wp_joint_q_next, self._wp_joint_q, self.coordinate_count),
                (self._wp_joint_velocity_next, self._wp_joint_velocity, self.dof_count),
            ):
                wp.launch(
                    _conditional_configuration_copy,
                    dim=(self.capacity, width),
                    inputs=[
                        candidate,
                        self._wp_frame_segment,
                        self._wp_line_search_take,
                        self._wp_ipm_enabled,
                        frame_count,
                    ],
                    outputs=[current],
                    device=self.optimizer.device,
                )
            wp.launch(
                _accepted_violation_update,
                dim=segment_count,
                inputs=[
                    self._wp_candidate_violation,
                    self._wp_ipm_enabled,
                    self._wp_line_search_take,
                ],
                outputs=[self._wp_constraint_violation],
                device=self.optimizer.device,
            )
            wp.launch(
                _accepted_violation_update,
                dim=segment_count,
                inputs=[
                    self._wp_candidate_restoration_merit,
                    self._wp_ipm_enabled,
                    self._wp_line_search_take,
                ],
                outputs=[self._wp_accepted_restoration_merit],
                device=self.optimizer.device,
            )
            self._second_order_correction_state[:segment_count, 2].copy_(self._candidate_violation[:segment_count])
            self._second_order_correction_state[:segment_count, 3].copy_(
                self._candidate_protected_violation[:segment_count]
            )
            self._second_order_correction_state[:segment_count, 5].copy_(
                self._candidate_restoration_merit[:segment_count]
            )
            self._segment_active[:segment_count].copy_(self._kkt_enabled[:segment_count])
            if self.device.type != "cuda" or not self.optimizer.device.is_capturing:
                self._control_pending.zero_()
                wp.launch(
                    _segments_pending,
                    dim=segment_count,
                    inputs=[self._wp_segment_active, segment_count],
                    outputs=[self._wp_control_pending],
                    device=self.optimizer.device,
                )
                if not bool(self._control_pending[0]):
                    break
        self._segment_active[:segment_count].copy_(self._second_order_correction_active[:segment_count])
        self._kkt_enabled[:segment_count].copy_(self._second_order_correction_line_search_enabled[:segment_count])
        self._delta[:frame_count].copy_(self._second_order_correction_direction[:frame_count])

    def _trajectory_cost(
        self,
        residuals: torch.Tensor,
        residuals_wp: wp.array,
        base_weights: wp.array,
        frame_count: int,
        segment_count: int,
        output: torch.Tensor,
    ) -> None:
        wp.launch(
            _precision_apply,
            dim=(self.capacity, self.residual_count),
            inputs=[
                residuals_wp,
                base_weights,
                self._wp_temporal_weights,
                self._wp_residual_activity,
                self._wp_activity_group_by_residual,
                self._wp_first_difference_group_by_residual,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_step_seconds,
                self._wp_segment_active,
                frame_count,
            ],
            outputs=[self._wp_precision_feature],
            device=self.optimizer.device,
        )
        self._dot(residuals, self._precision_feature, output, frame_count, segment_count)

    def _constraint_measure(
        self,
        residuals: wp.array,
        frame_count: int,
        output: torch.Tensor,
    ) -> None:
        output.zero_()
        output_wp = (
            self._wp_constraint_violation if output is self._constraint_violation else self._wp_candidate_violation
        )
        wp.launch(
            _constraint_violation_max,
            dim=(self.capacity, self.residual_count),
            inputs=[
                residuals,
                self._wp_residual_upper,
                self._wp_constraint_kind,
                self._wp_frame_segment,
                self._wp_segment_active,
                frame_count,
            ],
            outputs=[output_wp],
            device=self.optimizer.device,
        )

    def _constraint_protected_measure(self, residuals: wp.array, frame_count: int) -> None:
        """Measure candidate violation over constraints feasible at the current iterate."""
        self._candidate_protected_violation.zero_()
        wp.launch(
            _constraint_protected_violation_max,
            dim=(self.capacity, self.residual_count),
            inputs=[
                residuals,
                self._wp_residual_upper,
                self._wp_constraint_protected_kind,
                self._wp_frame_segment,
                self._wp_segment_active,
                frame_count,
            ],
            outputs=[self._wp_candidate_protected_violation],
            device=self.optimizer.device,
        )

    def _hard_bound_measure(
        self,
        joint_q: wp.array,
        joint_velocity: wp.array,
        coordinate_bound_count: int,
        frame_count: int,
    ) -> None:
        """Write exact scalar-coordinate and generalized-velocity violation."""
        self._candidate_bound_violation.zero_()
        wp.launch(
            _coordinate_bound_violation_max,
            dim=(self.capacity, max(1, coordinate_bound_count)),
            inputs=[
                joint_q,
                self._wp_coordinate_indices,
                self._wp_coordinate_lower,
                self._wp_coordinate_upper,
                coordinate_bound_count,
                self._wp_frame_segment,
                self._wp_segment_active,
                frame_count,
            ],
            outputs=[self._wp_candidate_bound_violation],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_velocity_bound_violation_max,
            dim=(self.capacity, self.dof_count),
            inputs=[
                joint_velocity,
                self._wp_velocity_lower,
                self._wp_velocity_upper,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_segment_active,
                frame_count,
            ],
            outputs=[self._wp_candidate_bound_violation],
            device=self.optimizer.device,
        )

    def _solve_ipm_linearized(
        self,
        frame_count: int,
        segment_count: int,
        inequality_count: int,
        coordinate_bound_count: int,
        convergence_accumulator: wp.array,
        *,
        has_equalities: bool,
        joint_q: wp.array,
        joint_velocity: wp.array,
        include_linear_term: bool,
        feasibility_only: bool,
    ) -> None:
        """Solve the all-constraint QP with fixed-work Mehrotra iterations."""
        root_tolerance = max(
            32.0 * float(torch.finfo(torch.float32).eps),
            self.kkt_relative_tolerance,
        )
        wp.launch(
            _inequalities_mark,
            dim=(self.capacity, max(1, inequality_count)),
            inputs=[
                self._wp_inequality_indices,
                self._wp_inequality_upper,
                inequality_count,
                frame_count,
                self.residual_count,
            ],
            outputs=[
                self._wp_constraint_kind,
                self._wp_objective_weights,
                self._wp_residual_upper,
                self._wp_constraint_contract_error,
            ],
            device=self.optimizer.device,
        )
        wp.launch(
            _precision_diagonal,
            dim=(self.capacity, self.residual_count),
            inputs=[
                self._wp_objective_weights,
                self._wp_temporal_weights,
                self._wp_residual_activity,
                self._wp_activity_group_by_residual,
                self._wp_first_difference_group_by_residual,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_step_seconds,
                self._wp_segment_active,
                frame_count,
            ],
            outputs=[self._wp_precision_diagonal],
            device=self.optimizer.device,
        )
        wp.launch(
            _normal_diagonal,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_jacobian,
                self._wp_precision_diagonal,
                self._wp_segment_damping,
                self._wp_frame_segment,
                self._wp_segment_active,
                frame_count,
            ],
            outputs=[self._wp_normal_diagonal],
            device=self.optimizer.device,
        )
        wp.launch(
            _qp_rows_initialize,
            dim=segment_count,
            inputs=[
                self._wp_constraint_kind,
                self._wp_coordinate_lower,
                self._wp_coordinate_upper,
                0,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                segment_count,
                self._wp_segment_active,
            ],
            outputs=[
                self._wp_constraint_contract_error,
                self._wp_active_row_codes,
                self._wp_active_row_scales,
                self._wp_ipm_equality_rhs,
                self._wp_active_equality_count,
            ],
            device=self.optimizer.device,
        )
        wp.launch(
            _qp_rows_refresh,
            dim=(self.capacity, self.active_width),
            inputs=[
                self._wp_jacobian,
                self._wp_linearized_feature,
                self._wp_residual_upper,
                joint_q,
                self._wp_coordinate_indices,
                self._wp_coordinate_dof_indices,
                self._wp_coordinate_lower,
                self._wp_coordinate_upper,
                0,
                joint_velocity,
                self._wp_velocity_lower,
                self._wp_velocity_upper,
                self._wp_step_seconds,
                self.dof_count,
                self._wp_normal_diagonal,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_segment_active,
                frame_count,
                float(torch.finfo(torch.float32).eps),
            ],
            outputs=[
                self._wp_active_row_codes,
                self._wp_active_equality_count,
                self._wp_active_row_scales,
                self._wp_ipm_equality_rhs,
            ],
            device=self.optimizer.device,
        )
        if include_linear_term:
            wp.launch(
                _precision_apply,
                dim=(self.capacity, self.residual_count),
                inputs=[
                    self._wp_linearized_feature,
                    self._wp_objective_weights,
                    self._wp_temporal_weights,
                    self._wp_residual_activity,
                    self._wp_activity_group_by_residual,
                    self._wp_first_difference_group_by_residual,
                    self._wp_frame_segment,
                    self._wp_segment_offsets,
                    self._wp_step_seconds,
                    self._wp_segment_active,
                    frame_count,
                ],
                outputs=[self._wp_precision_feature],
                device=self.optimizer.device,
            )
            torch.bmm(
                self._jacobian[:frame_count].transpose(1, 2),
                self._precision_feature[:frame_count].unsqueeze(-1),
                out=self._ipm_gradient[:frame_count].unsqueeze(-1),
            )
        else:
            self._ipm_gradient[:frame_count].zero_()
        self._ipm_primal[:frame_count].zero_()
        self._ipm_primal_f64[:frame_count].zero_()
        self._ipm_equality_dual[:frame_count].zero_()
        self._ipm_linear_converged[:segment_count].fill_(1)
        wp.copy(self._wp_ipm_enabled, self._wp_segment_active, count=segment_count)
        self._ipm_inequality_count[:segment_count].zero_()
        wp.launch(
            _ipm_constraints_initialize,
            dim=(self.capacity, self.constraint_width),
            inputs=[
                self._wp_jacobian,
                self._wp_linearized_feature,
                self._wp_residual_upper,
                self._wp_constraint_kind,
                joint_q,
                self._wp_coordinate_indices,
                self._wp_coordinate_dof_indices,
                self._wp_coordinate_lower,
                self._wp_coordinate_upper,
                coordinate_bound_count,
                joint_velocity,
                self._wp_velocity_lower,
                self._wp_velocity_upper,
                self._wp_step_seconds,
                self._wp_normal_diagonal,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_segment_active,
                frame_count,
                float(torch.finfo(torch.float32).eps),
            ],
            outputs=[
                self._wp_ipm_constraint_scale,
                self._wp_ipm_constraint_rhs,
                self._wp_ipm_inequality_count,
            ],
            device=self.optimizer.device,
        )
        self._phase_one_has_equalities = has_equalities
        self._solve_ipm_phase_one(frame_count, segment_count)
        if feasibility_only:
            self._phase_one_witness_selected[:segment_count].zero_()
            wp.launch(
                _phase_one_witness_select,
                dim=segment_count,
                inputs=[
                    self._wp_candidate_bound_violation,
                    self._wp_ipm_enabled,
                    self._wp_segment_active,
                    64.0 * float(torch.finfo(torch.float32).eps),
                    segment_count,
                ],
                outputs=[self._wp_phase_one_witness_selected],
                device=self.optimizer.device,
            )
        self._prepare_ipm_objective_block_band(frame_count, segment_count)
        # Restore the objective diagonal that defines Phase-II row normalization.
        # Feasibility-only segments retain a certified Phase-I primal; only
        # inconclusive segments fall back to the objective solve.
        wp.launch(
            _normal_diagonal,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_jacobian,
                self._wp_precision_diagonal,
                self._wp_segment_damping,
                self._wp_frame_segment,
                self._wp_segment_active,
                frame_count,
            ],
            outputs=[self._wp_normal_diagonal],
            device=self.optimizer.device,
        )
        wp.launch(
            _phase_two_primal_handoff,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_phase_one_witness_selected,
                self._wp_frame_segment,
                frame_count,
            ],
            outputs=[self._wp_ipm_primal, self._wp_ipm_primal_f64],
            device=self.optimizer.device,
        )
        self._ipm_inequality_count[:segment_count].zero_()
        wp.launch(
            _ipm_constraints_initialize,
            dim=(self.capacity, self.constraint_width),
            inputs=[
                self._wp_jacobian,
                self._wp_linearized_feature,
                self._wp_residual_upper,
                self._wp_constraint_kind,
                joint_q,
                self._wp_coordinate_indices,
                self._wp_coordinate_dof_indices,
                self._wp_coordinate_lower,
                self._wp_coordinate_upper,
                coordinate_bound_count,
                joint_velocity,
                self._wp_velocity_lower,
                self._wp_velocity_upper,
                self._wp_step_seconds,
                self._wp_normal_diagonal,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_ipm_enabled,
                frame_count,
                float(torch.finfo(torch.float32).eps),
            ],
            outputs=[
                self._wp_ipm_constraint_scale,
                self._wp_ipm_constraint_rhs,
                self._wp_ipm_inequality_count,
            ],
            device=self.optimizer.device,
        )
        self._ipm_primal_constraints_apply_f64(frame_count)
        wp.launch(
            _ipm_iterate_initialize,
            dim=(self.capacity, self.constraint_width),
            inputs=[
                self._wp_ipm_constraint_work,
                self._wp_ipm_constraint_rhs,
                self._wp_ipm_constraint_scale,
                self._wp_frame_segment,
                self._wp_segment_active,
                frame_count,
            ],
            outputs=[self._wp_ipm_slack, self._wp_ipm_multiplier],
            device=self.optimizer.device,
        )
        self._ipm_operator_active = True
        phase_two_complete = False
        for _ in range(_IPM_ITERATIONS):
            self._dot(
                self._ipm_slack,
                self._ipm_multiplier,
                self._candidate_cost,
                frame_count,
                segment_count,
            )
            self._residual_dot_next[:segment_count].zero_()
            wp.launch(
                _ipm_centering_update,
                dim=segment_count,
                inputs=[
                    self._wp_candidate_cost,
                    self._wp_residual_dot_next,
                    self._wp_ipm_inequality_count,
                    self._wp_ipm_enabled,
                    segment_count,
                ],
                outputs=[self._wp_ipm_complementarity, self._wp_ipm_centering],
                device=self.optimizer.device,
            )
            self._ipm_newton_right_hand_side(frame_count, corrector=False)
            self._constraint_violation[:segment_count].zero_()
            self._candidate_violation[:segment_count].zero_()
            self._candidate_bound_violation[:segment_count].zero_()
            for values, output in (
                (self._wp_delta_correction, self._wp_constraint_violation),
                (self._wp_ipm_primal_residual, self._wp_candidate_violation),
                (self._wp_dual_r1, self._wp_candidate_bound_violation),
            ):
                wp.launch(
                    _segment_step_max,
                    dim=(self.capacity, values.shape[1]),
                    inputs=[
                        values,
                        self._wp_frame_segment,
                        self._wp_ipm_enabled,
                        frame_count,
                    ],
                    outputs=[output],
                    device=self.optimizer.device,
                )
            self._solve_ipm_direction(frame_count, segment_count, has_equalities)
            self._ipm_constraints_apply_values_f64(
                self._wp_ipm_solution_f64, self._wp_ipm_augmented_row_r2_f64, frame_count
            )
            wp.launch(
                _ipm_direction_recover_f64,
                dim=(self.capacity, self.inequality_width),
                inputs=[
                    self._wp_ipm_augmented_row_r2_f64,
                    self._wp_ipm_constraint_scale,
                    self._wp_ipm_primal_residual,
                    self._wp_ipm_constraint_work,
                    self._wp_ipm_weights,
                    self._wp_frame_segment,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[self._wp_ipm_augmented_row_solution_f64, self._wp_ipm_augmented_row_r1_f64],
                device=self.optimizer.device,
            )
            self._ipm_relative_correction_measure_f64(frame_count, segment_count)
            self._ipm_complementarity_max_measure(frame_count, segment_count)
            wp.launch(
                _ipm_phase_two_convergence_mask,
                dim=segment_count,
                inputs=[
                    self._wp_residual_dot_next,
                    self._wp_operator_dot,
                    self._wp_constraint_violation,
                    self._wp_candidate_bound_violation,
                    self._wp_candidate_violation,
                    root_tolerance,
                    root_tolerance,
                    float(torch.finfo(torch.float32).eps),
                    segment_count,
                ],
                outputs=[self._wp_ipm_enabled],
                device=self.optimizer.device,
            )
            if self.device.type != "cuda" or not self.optimizer.device.is_capturing:
                self._control_pending.zero_()
                wp.launch(
                    _segments_pending,
                    dim=segment_count,
                    inputs=[self._wp_ipm_enabled, segment_count],
                    outputs=[self._wp_control_pending],
                    device=self.optimizer.device,
                )
                if not bool(self._control_pending[0]):
                    phase_two_complete = True
                    break
            self._ipm_primal_step[:segment_count].fill_(1.0)
            self._ipm_dual_step[:segment_count].fill_(1.0)
            wp.launch(
                _ipm_step_bound_f64,
                dim=(self.capacity, self.inequality_width),
                inputs=[
                    self._wp_ipm_constraint_scale,
                    self._wp_ipm_slack,
                    self._wp_ipm_multiplier,
                    self._wp_ipm_augmented_row_solution_f64,
                    self._wp_ipm_augmented_row_r1_f64,
                    1.0,
                    self._wp_frame_segment,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[self._wp_ipm_primal_step, self._wp_ipm_dual_step],
                device=self.optimizer.device,
            )
            wp.launch(
                _ipm_affine_complementarity_frame_f64,
                dim=self.capacity,
                inputs=[
                    self._wp_ipm_constraint_scale,
                    self._wp_ipm_slack,
                    self._wp_ipm_multiplier,
                    self._wp_ipm_augmented_row_solution_f64,
                    self._wp_ipm_augmented_row_r1_f64,
                    self._wp_ipm_primal_step,
                    self._wp_ipm_dual_step,
                    self._wp_frame_segment,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[self._wp_dot_frame],
                device=self.optimizer.device,
            )
            wp.launch(
                _dot_segments,
                dim=self.max_segments,
                inputs=[
                    self._wp_dot_frame,
                    self._wp_segment_offsets,
                    segment_count,
                    self._wp_ipm_enabled,
                ],
                outputs=[self._wp_residual_dot_next],
                device=self.optimizer.device,
            )
            wp.launch(
                _ipm_centering_update,
                dim=segment_count,
                inputs=[
                    self._wp_candidate_cost,
                    self._wp_residual_dot_next,
                    self._wp_ipm_inequality_count,
                    self._wp_ipm_enabled,
                    segment_count,
                ],
                outputs=[self._wp_ipm_complementarity, self._wp_ipm_centering],
                device=self.optimizer.device,
            )
            self._ipm_newton_right_hand_side(frame_count, corrector=True)
            self._solve_ipm_direction(
                frame_count,
                segment_count,
                has_equalities,
                rebuild_factor=False,
            )
            self._ipm_constraints_apply_values_f64(
                self._wp_ipm_solution_f64, self._wp_ipm_augmented_row_r2_f64, frame_count
            )
            wp.launch(
                _ipm_direction_recover_f64,
                dim=(self.capacity, self.inequality_width),
                inputs=[
                    self._wp_ipm_augmented_row_r2_f64,
                    self._wp_ipm_constraint_scale,
                    self._wp_ipm_primal_residual,
                    self._wp_ipm_constraint_work,
                    self._wp_ipm_weights,
                    self._wp_frame_segment,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[self._wp_ipm_augmented_row_solution_f64, self._wp_ipm_augmented_row_r1_f64],
                device=self.optimizer.device,
            )
            self._ipm_primal_step[:segment_count].fill_(1.0)
            self._ipm_dual_step[:segment_count].fill_(1.0)
            wp.launch(
                _ipm_step_bound_f64,
                dim=(self.capacity, self.inequality_width),
                inputs=[
                    self._wp_ipm_constraint_scale,
                    self._wp_ipm_slack,
                    self._wp_ipm_multiplier,
                    self._wp_ipm_augmented_row_solution_f64,
                    self._wp_ipm_augmented_row_r1_f64,
                    _IPM_FRACTION_TO_BOUNDARY,
                    self._wp_frame_segment,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[self._wp_ipm_primal_step, self._wp_ipm_dual_step],
                device=self.optimizer.device,
            )
            wp.launch(
                _ipm_step_couple,
                dim=segment_count,
                inputs=[
                    self._wp_ipm_primal_step,
                    self._wp_ipm_dual_step,
                    self._wp_ipm_enabled,
                    segment_count,
                ],
                device=self.optimizer.device,
            )
            wp.launch(
                _ipm_iterate_update_f64,
                dim=(self.capacity, self.inequality_width),
                inputs=[
                    self._wp_ipm_constraint_scale,
                    self._wp_ipm_augmented_row_solution_f64,
                    self._wp_ipm_augmented_row_r1_f64,
                    self._wp_ipm_primal_step,
                    self._wp_ipm_dual_step,
                    self._wp_frame_segment,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[self._wp_ipm_slack, self._wp_ipm_multiplier],
                device=self.optimizer.device,
            )
            wp.launch(
                _ipm_primal_update_state_f64,
                dim=(self.capacity, self.dof_count),
                inputs=[
                    self._wp_ipm_solution_f64,
                    self._wp_ipm_primal_step,
                    self._wp_frame_segment,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[self._wp_ipm_primal_f64, self._wp_ipm_primal],
                device=self.optimizer.device,
            )
            wp.launch(
                _ipm_primal_update,
                dim=(self.capacity, self.active_width),
                inputs=[
                    self._wp_dual_solution,
                    self._wp_ipm_dual_step,
                    self._wp_frame_segment,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[self._wp_ipm_equality_dual],
                device=self.optimizer.device,
            )
        if not phase_two_complete:
            self._dot(
                self._ipm_slack,
                self._ipm_multiplier,
                self._candidate_cost,
                frame_count,
                segment_count,
            )
            self._residual_dot_next[:segment_count].zero_()
            wp.launch(
                _ipm_centering_update,
                dim=segment_count,
                inputs=[
                    self._wp_candidate_cost,
                    self._wp_residual_dot_next,
                    self._wp_ipm_inequality_count,
                    self._wp_ipm_enabled,
                    segment_count,
                ],
                outputs=[self._wp_ipm_complementarity, self._wp_ipm_centering],
                device=self.optimizer.device,
            )
            self._ipm_newton_right_hand_side(frame_count, corrector=False)
            self._ipm_convergence_measure(frame_count, segment_count, include_phase_one_scalar=False)
            self._solve_ipm_direction(frame_count, segment_count, has_equalities)
            self._ipm_relative_correction_measure_f64(frame_count, segment_count)
            self._ipm_complementarity_max_measure(frame_count, segment_count)
            wp.launch(
                _ipm_phase_two_convergence_mask,
                dim=segment_count,
                inputs=[
                    self._wp_residual_dot_next,
                    self._wp_operator_dot,
                    self._wp_constraint_violation,
                    self._wp_candidate_bound_violation,
                    self._wp_candidate_violation,
                    root_tolerance,
                    root_tolerance,
                    float(torch.finfo(torch.float32).eps),
                    segment_count,
                ],
                outputs=[self._wp_ipm_enabled],
                device=self.optimizer.device,
            )
        self._ipm_operator_active = False
        wp.launch(
            _krylov_convergence_accumulate,
            dim=segment_count,
            inputs=[
                self._wp_segment_active,
                self._wp_ipm_enabled,
                segment_count,
            ],
            outputs=[self._wp_ipm_linear_converged],
            device=self.optimizer.device,
        )
        self._delta[:frame_count].copy_(self._ipm_primal[:frame_count])
        wp.launch(
            _ipm_locked_coordinate_canonicalize,
            dim=(self.capacity, max(1, coordinate_bound_count)),
            inputs=[
                joint_q,
                self._wp_coordinate_indices,
                self._wp_coordinate_dof_indices,
                self._wp_coordinate_lower,
                self._wp_coordinate_upper,
                coordinate_bound_count,
                self._wp_frame_segment,
                self._wp_ipm_linear_converged,
                frame_count,
            ],
            outputs=[self._wp_delta],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_step_enable,
            dim=segment_count,
            inputs=[
                self._wp_segment_active,
                self._wp_segment_feasible,
                self._wp_ipm_linear_converged,
                segment_count,
            ],
            outputs=[self._wp_kkt_enabled],
            device=self.optimizer.device,
        )
        wp.copy(convergence_accumulator, self._wp_ipm_linear_converged)

    def _solve_ipm_phase_one(self, frame_count: int, segment_count: int) -> None:
        """Find a feasible QP point or a checked Farkas separator."""
        self._phase_one_witness_selected[:segment_count].zero_()
        wp.launch(
            _phase_one_constraints_initialize,
            dim=(self.capacity, 2 * self.active_width + 1),
            inputs=[
                self._wp_active_row_codes,
                self._wp_active_row_scales,
                self._wp_ipm_equality_rhs,
                self._wp_active_equality_count,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_ipm_enabled,
                frame_count,
                self.inequality_width,
            ],
            outputs=[
                self._wp_ipm_constraint_scale,
                self._wp_ipm_constraint_rhs,
                self._wp_ipm_inequality_count,
            ],
            device=self.optimizer.device,
        )
        self._phase_one_elastic[:segment_count].zero_()
        self._phase_one_constraints_apply(
            self._ipm_primal,
            self._wp_ipm_primal,
            self._wp_phase_one_elastic,
            frame_count,
        )
        self._phase_one_witness_freeze(frame_count, segment_count)
        if self.device.type != "cuda" or not self.optimizer.device.is_capturing:
            self._control_pending.zero_()
            wp.launch(
                _segments_pending,
                dim=segment_count,
                inputs=[self._wp_ipm_enabled, segment_count],
                outputs=[self._wp_control_pending],
                device=self.optimizer.device,
            )
            if not bool(self._control_pending[0]):
                self._ipm_enabled[:segment_count].copy_(self._segment_active[:segment_count])
                return
        wp.launch(
            _ipm_iterate_initialize,
            dim=(self.capacity, self.constraint_width),
            inputs=[
                self._wp_ipm_constraint_work,
                self._wp_ipm_constraint_rhs,
                self._wp_ipm_constraint_scale,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_ipm_slack, self._wp_ipm_multiplier],
            device=self.optimizer.device,
        )
        for _ in range(_IPM_ITERATIONS):
            self._dot(
                self._ipm_slack,
                self._ipm_multiplier,
                self._candidate_cost,
                frame_count,
                segment_count,
            )
            self._residual_dot_next[:segment_count].zero_()
            wp.launch(
                _ipm_centering_update,
                dim=segment_count,
                inputs=[
                    self._wp_candidate_cost,
                    self._wp_residual_dot_next,
                    self._wp_ipm_inequality_count,
                    self._wp_ipm_enabled,
                    segment_count,
                ],
                outputs=[self._wp_ipm_complementarity, self._wp_ipm_centering],
                device=self.optimizer.device,
            )
            self._phase_one_newton_right_hand_side(frame_count, segment_count, corrector=False)
            self._ipm_convergence_mask(frame_count, segment_count, include_phase_one_scalar=True)
            if self.device.type != "cuda" or not self.optimizer.device.is_capturing:
                self._control_pending.zero_()
                wp.launch(
                    _segments_pending,
                    dim=segment_count,
                    inputs=[self._wp_ipm_enabled, segment_count],
                    outputs=[self._wp_control_pending],
                    device=self.optimizer.device,
                )
                if not bool(self._control_pending[0]):
                    break
            self._solve_phase_one_direction(frame_count, segment_count)
            self._phase_one_constraints_apply(
                self._delta,
                self._wp_delta,
                self._wp_phase_one_delta,
                frame_count,
            )
            wp.launch(
                _ipm_direction_recover,
                dim=(self.capacity, self.constraint_width),
                inputs=[
                    self._wp_ipm_constraint_work,
                    self._wp_ipm_constraint_scale,
                    self._wp_ipm_slack,
                    self._wp_ipm_multiplier,
                    self._wp_ipm_primal_residual,
                    self._wp_ipm_complementarity_residual,
                    self._wp_frame_segment,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[self._wp_ipm_affine_slack, self._wp_ipm_affine_multiplier],
                device=self.optimizer.device,
            )
            self._ipm_primal_step[:segment_count].fill_(1.0)
            self._ipm_dual_step[:segment_count].fill_(1.0)
            wp.launch(
                _ipm_step_bound,
                dim=(self.capacity, self.constraint_width),
                inputs=[
                    self._wp_ipm_constraint_scale,
                    self._wp_ipm_slack,
                    self._wp_ipm_multiplier,
                    self._wp_ipm_affine_slack,
                    self._wp_ipm_affine_multiplier,
                    1.0,
                    self._wp_frame_segment,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[self._wp_ipm_primal_step, self._wp_ipm_dual_step],
                device=self.optimizer.device,
            )
            wp.launch(
                _ipm_affine_complementarity_frame,
                dim=self.capacity,
                inputs=[
                    self._wp_ipm_constraint_scale,
                    self._wp_ipm_slack,
                    self._wp_ipm_multiplier,
                    self._wp_ipm_affine_slack,
                    self._wp_ipm_affine_multiplier,
                    self._wp_ipm_primal_step,
                    self._wp_ipm_dual_step,
                    self._wp_frame_segment,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[self._wp_dot_frame],
                device=self.optimizer.device,
            )
            wp.launch(
                _dot_segments,
                dim=self.max_segments,
                inputs=[
                    self._wp_dot_frame,
                    self._wp_segment_offsets,
                    segment_count,
                    self._wp_ipm_enabled,
                ],
                outputs=[self._wp_residual_dot_next],
                device=self.optimizer.device,
            )
            wp.launch(
                _ipm_centering_update,
                dim=segment_count,
                inputs=[
                    self._wp_candidate_cost,
                    self._wp_residual_dot_next,
                    self._wp_ipm_inequality_count,
                    self._wp_ipm_enabled,
                    segment_count,
                ],
                outputs=[self._wp_ipm_complementarity, self._wp_ipm_centering],
                device=self.optimizer.device,
            )
            self._phase_one_newton_right_hand_side(frame_count, segment_count, corrector=True)
            self._solve_phase_one_direction(frame_count, segment_count, rebuild_factor=False)
            self._phase_one_constraints_apply(
                self._delta,
                self._wp_delta,
                self._wp_phase_one_delta,
                frame_count,
            )
            wp.launch(
                _ipm_direction_recover,
                dim=(self.capacity, self.constraint_width),
                inputs=[
                    self._wp_ipm_constraint_work,
                    self._wp_ipm_constraint_scale,
                    self._wp_ipm_slack,
                    self._wp_ipm_multiplier,
                    self._wp_ipm_primal_residual,
                    self._wp_ipm_complementarity_residual,
                    self._wp_frame_segment,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[self._wp_ipm_primal_residual, self._wp_ipm_complementarity_residual],
                device=self.optimizer.device,
            )
            self._ipm_primal_step[:segment_count].fill_(1.0)
            self._ipm_dual_step[:segment_count].fill_(1.0)
            wp.launch(
                _ipm_step_bound,
                dim=(self.capacity, self.constraint_width),
                inputs=[
                    self._wp_ipm_constraint_scale,
                    self._wp_ipm_slack,
                    self._wp_ipm_multiplier,
                    self._wp_ipm_primal_residual,
                    self._wp_ipm_complementarity_residual,
                    _IPM_FRACTION_TO_BOUNDARY,
                    self._wp_frame_segment,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[self._wp_ipm_primal_step, self._wp_ipm_dual_step],
                device=self.optimizer.device,
            )
            wp.launch(
                _ipm_iterate_update,
                dim=(self.capacity, self.constraint_width),
                inputs=[
                    self._wp_ipm_constraint_scale,
                    self._wp_ipm_primal_residual,
                    self._wp_ipm_complementarity_residual,
                    self._wp_ipm_primal_step,
                    self._wp_ipm_dual_step,
                    self._wp_frame_segment,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[self._wp_ipm_slack, self._wp_ipm_multiplier],
                device=self.optimizer.device,
            )
            wp.launch(
                _ipm_primal_update,
                dim=(self.capacity, self.dof_count),
                inputs=[
                    self._wp_delta,
                    self._wp_ipm_primal_step,
                    self._wp_frame_segment,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[self._wp_ipm_primal],
                device=self.optimizer.device,
            )
            wp.launch(
                _phase_one_elastic_update,
                dim=segment_count,
                inputs=[
                    self._wp_phase_one_delta,
                    self._wp_ipm_primal_step,
                    self._wp_ipm_enabled,
                    segment_count,
                ],
                outputs=[self._wp_phase_one_elastic],
                device=self.optimizer.device,
            )
        self._dot(
            self._ipm_slack,
            self._ipm_multiplier,
            self._candidate_cost,
            frame_count,
            segment_count,
        )
        self._residual_dot_next[:segment_count].zero_()
        wp.launch(
            _ipm_centering_update,
            dim=segment_count,
            inputs=[
                self._wp_candidate_cost,
                self._wp_residual_dot_next,
                self._wp_ipm_inequality_count,
                self._wp_ipm_enabled,
                segment_count,
            ],
            outputs=[self._wp_ipm_complementarity, self._wp_ipm_centering],
            device=self.optimizer.device,
        )
        self._phase_one_newton_right_hand_side(frame_count, segment_count, corrector=False)
        self._ipm_convergence_mask(frame_count, segment_count, include_phase_one_scalar=True)
        self._kkt_enabled[:segment_count].copy_(self._ipm_enabled[:segment_count])
        self._ipm_enabled[:segment_count].copy_(self._segment_active[:segment_count])
        self._phase_one_delta[:segment_count].zero_()
        self._phase_one_constraints_apply(
            self._ipm_primal,
            self._wp_ipm_primal,
            self._wp_phase_one_delta,
            frame_count,
        )
        self._candidate_bound_violation[:segment_count].zero_()
        wp.launch(
            _ipm_primal_feasibility_violation_max,
            dim=(self.capacity, self.constraint_width),
            inputs=[
                self._wp_ipm_constraint_work,
                self._wp_ipm_constraint_rhs,
                self._wp_ipm_constraint_scale,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_candidate_bound_violation],
            device=self.optimizer.device,
        )
        self._ipm_enabled[:segment_count].copy_(self._kkt_enabled[:segment_count])
        self._dot(
            self._ipm_constraint_rhs,
            self._ipm_multiplier,
            self._candidate_cost,
            frame_count,
            segment_count,
        )
        self._candidate_cost[:segment_count].neg_()
        wp.launch(
            _phase_one_finalize,
            dim=segment_count,
            inputs=[
                self._wp_candidate_bound_violation,
                self._wp_constraint_violation,
                self._wp_accepted_cost,
                self._wp_candidate_violation,
                self._wp_candidate_cost,
                self._wp_ipm_enabled,
                self._wp_segment_active,
                64.0 * float(torch.finfo(torch.float32).eps),
                segment_count,
            ],
            outputs=[
                self._wp_segment_feasible,
                self._wp_ipm_linear_converged,
                self._wp_ipm_enabled,
            ],
            device=self.optimizer.device,
        )

    def _phase_one_newton_right_hand_side(
        self,
        frame_count: int,
        segment_count: int,
        *,
        corrector: bool,
    ) -> None:
        """Build a condensed Newton RHS for the exact min-elastic model."""
        self._phase_one_constraints_apply(
            self._ipm_primal,
            self._wp_ipm_primal,
            self._wp_phase_one_elastic,
            frame_count,
        )
        if not corrector:
            self._phase_one_witness_freeze(frame_count, segment_count)
        wp.launch(
            _ipm_residual,
            dim=(self.capacity, self.constraint_width),
            inputs=[
                self._wp_ipm_constraint_work,
                self._wp_ipm_constraint_rhs,
                self._wp_ipm_constraint_scale,
                self._wp_ipm_slack,
                self._wp_ipm_multiplier,
                self._wp_ipm_affine_slack,
                self._wp_ipm_affine_multiplier,
                self._wp_ipm_centering,
                self._wp_ipm_complementarity,
                wp.uint8(1 if corrector else 0),
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[
                self._wp_ipm_primal_residual,
                self._wp_ipm_complementarity_residual,
                self._wp_ipm_weights,
                self._wp_ipm_constraint_work,
            ],
            device=self.optimizer.device,
        )
        self._phase_one_constraints_transpose(
            self._wp_ipm_multiplier,
            self._wp_phase_one_operator,
            frame_count,
            segment_count,
        )
        self._delta_correction[:frame_count].copy_(self._ipm_transpose[:frame_count])
        self._right_hand_side[:frame_count].copy_(self._ipm_transpose[:frame_count]).neg_()
        self._phase_one_constraints_transpose(
            self._wp_ipm_constraint_work,
            self._wp_phase_one_residual,
            frame_count,
            segment_count,
        )
        self._right_hand_side[:frame_count].add_(self._ipm_transpose[:frame_count])
        wp.launch(
            _phase_one_scalar_rhs,
            dim=segment_count,
            inputs=[
                self._wp_phase_one_operator,
                self._wp_phase_one_residual,
                self._wp_ipm_enabled,
                segment_count,
            ],
            outputs=[self._wp_accepted_cost, self._wp_phase_one_rhs],
            device=self.optimizer.device,
        )
        wp.launch(
            _segment_diagonal_initialize,
            dim=(self.capacity, self.dof_count),
            inputs=[self._wp_segment_damping, self._wp_frame_segment, frame_count],
            outputs=[self._wp_normal_diagonal],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_normal_diagonal_add,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_jacobian,
                self._wp_ipm_weights,
                self._wp_coordinate_dof_indices,
                self._coordinate_bound_count,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_ipm_enabled,
                frame_count,
                self._wp_ipm_constraint_scale,
            ],
            outputs=[self._wp_normal_diagonal],
            device=self.optimizer.device,
        )
        wp.launch(
            _phase_one_scalar_diagonal,
            dim=segment_count,
            inputs=[
                self._wp_ipm_weights,
                self._wp_ipm_constraint_scale,
                self._wp_segment_offsets,
                self._wp_ipm_enabled,
                self._wp_segment_damping,
                segment_count,
            ],
            outputs=[self._wp_phase_one_diagonal],
            device=self.optimizer.device,
        )

    def _phase_one_witness_freeze(self, frame_count: int, segment_count: int) -> None:
        """Freeze each segment at its first physically feasible Phase-I iterate."""
        self._candidate_bound_violation[:segment_count].zero_()
        wp.launch(
            _phase_one_original_violation_max,
            dim=(self.capacity, self.constraint_width),
            inputs=[
                self._wp_ipm_constraint_work,
                self._wp_ipm_constraint_rhs,
                self._wp_ipm_constraint_scale,
                self._wp_phase_one_elastic,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_candidate_bound_violation],
            device=self.optimizer.device,
        )
        wp.launch(
            _phase_one_witness_select,
            dim=segment_count,
            inputs=[
                self._wp_candidate_bound_violation,
                self._wp_ipm_enabled,
                self._wp_segment_active,
                64.0 * float(torch.finfo(torch.float32).eps),
                segment_count,
            ],
            outputs=[self._wp_phase_one_witness_selected],
            device=self.optimizer.device,
        )

    def _phase_one_constraints_apply(
        self,
        primal: torch.Tensor,
        primal_wp: wp.array,
        elastic_wp: wp.array,
        frame_count: int,
    ) -> None:
        self._ipm_constraints_apply(primal, primal_wp, self._wp_ipm_constraint_work, frame_count)
        self._dual_r2[:frame_count].zero_()
        wp.launch(
            _qp_rows_operator_dual,
            dim=(self.capacity, self.active_width),
            inputs=[
                self._wp_feature,
                primal_wp,
                self._wp_coordinate_dof_indices,
                self._wp_coordinate_lower,
                self._wp_coordinate_upper,
                0,
                self.dof_count,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_ipm_enabled,
                frame_count,
                self._wp_active_row_codes,
                self._wp_active_row_scales,
                self._wp_active_equality_count,
                self._wp_dual_r2,
            ],
            outputs=[self._wp_dual_r2],
            device=self.optimizer.device,
        )
        wp.launch(
            _phase_one_constraints_apply,
            dim=(self.capacity, self.constraint_width),
            inputs=[
                self._wp_dual_r2,
                elastic_wp,
                self._wp_ipm_constraint_scale,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
                self.inequality_width,
            ],
            outputs=[self._wp_ipm_constraint_work],
            device=self.optimizer.device,
        )

    def _phase_one_constraints_transpose(
        self,
        values_wp: wp.array,
        scalar_output_wp: wp.array,
        frame_count: int,
        segment_count: int,
    ) -> None:
        self._ipm_constraints_transpose(values_wp, frame_count)
        wp.launch(
            _phase_one_equality_transpose_values,
            dim=(self.capacity, self.active_width),
            inputs=[
                values_wp,
                self.inequality_width,
                self._wp_active_row_codes,
                self._wp_active_equality_count,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_dual_r2],
            device=self.optimizer.device,
        )
        wp.launch(
            _qp_rows_transpose,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_jacobian,
                self._wp_coordinate_dof_indices,
                self._wp_coordinate_lower,
                self._wp_coordinate_upper,
                0,
                self.dof_count,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_ipm_enabled,
                frame_count,
                self._wp_active_row_codes,
                self._wp_active_row_scales,
                self._wp_active_equality_count,
                self._wp_dual_r2,
            ],
            outputs=[self._wp_active_row_transpose],
            device=self.optimizer.device,
        )
        self._ipm_transpose[:frame_count].add_(self._active_row_transpose[:frame_count])
        wp.launch(
            _phase_one_scalar_transpose,
            dim=segment_count,
            inputs=[
                values_wp,
                self._wp_ipm_constraint_scale,
                self._wp_segment_offsets,
                self._wp_ipm_enabled,
                segment_count,
            ],
            outputs=[scalar_output_wp],
            device=self.optimizer.device,
        )

    def _solve_phase_one_direction(
        self,
        frame_count: int,
        segment_count: int,
        *,
        rebuild_factor: bool = True,
    ) -> None:
        """Solve one combined primal-elastic Phase-I direction."""
        if self._phase_one_has_equalities:
            self._solve_phase_one_direction_pcg(frame_count, segment_count)
            return
        self._solve_phase_one_direction_arrowhead(
            frame_count,
            segment_count,
            rebuild_factor=rebuild_factor,
        )

    def _solve_phase_one_direction_arrowhead(
        self,
        frame_count: int,
        segment_count: int,
        *,
        rebuild_factor: bool,
    ) -> None:
        """Solve the no-equality Phase-I arrowhead by exact block elimination."""
        self._minres_enabled[:segment_count].zero_()
        if rebuild_factor:
            self._minres_failed[:segment_count].zero_()
            wp.launch(
                _phase_one_block_band_matrix_build,
                dim=(self.capacity, 4, self.dof_count, self.dof_count),
                inputs=[
                    self._wp_jacobian,
                    self._wp_ipm_weights,
                    self._wp_coordinate_dof_indices,
                    self._coordinate_bound_count,
                    self._wp_ipm_constraint_scale,
                    self._wp_segment_damping,
                    self._wp_frame_segment,
                    self._wp_segment_offsets,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[self._wp_ipm_block_band_factor],
                device=self.optimizer.device,
            )
            if self.device.type == "cuda":
                wp.launch(
                    _ipm_block_band_matrix_factor_parallel,
                    dim=self.max_segments * _IPM_BLOCK_BAND_BLOCK_DIM,
                    block_dim=_IPM_BLOCK_BAND_BLOCK_DIM,
                    inputs=[
                        self._wp_ipm_block_band_factor,
                        self._wp_segment_offsets,
                        self._wp_ipm_enabled,
                        segment_count,
                    ],
                    outputs=[self._wp_minres_failed],
                    device=self.optimizer.device,
                )
            else:
                wp.launch(
                    _ipm_block_band_matrix_factor,
                    dim=self.max_segments,
                    inputs=[
                        self._wp_ipm_block_band_factor,
                        self._wp_segment_offsets,
                        self._wp_ipm_enabled,
                        segment_count,
                    ],
                    outputs=[self._wp_minres_failed],
                    device=self.optimizer.device,
                )

        wp.launch(
            _phase_one_arrowhead_cross_build,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_jacobian,
                self._wp_ipm_weights,
                self._wp_coordinate_dof_indices,
                self._coordinate_bound_count,
                self._wp_ipm_constraint_scale,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_phase_one_cross_f64, self._wp_ipm_solution_f64],
            device=self.optimizer.device,
        )
        self._solve_block_band_in_place(self._wp_ipm_solution_f64, self._wp_ipm_enabled, segment_count)
        self._ipm_operator_f64[:frame_count].zero_()
        wp.launch(
            _ipm_solution_accumulate_f64,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_right_hand_side,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_ipm_operator_f64],
            device=self.optimizer.device,
        )
        self._solve_block_band_in_place(self._wp_ipm_operator_f64, self._wp_ipm_enabled, segment_count)
        wp.launch(
            _phase_one_arrowhead_pair_dot,
            dim=self.capacity,
            inputs=[
                self._wp_phase_one_cross_f64,
                self._wp_ipm_solution_f64,
                self._wp_ipm_operator_f64,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_phase_one_dot_frame_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _phase_one_arrowhead_pair_reduce,
            dim=segment_count,
            inputs=[
                self._wp_phase_one_dot_frame_f64,
                self._wp_segment_offsets,
                segment_count,
            ],
            outputs=[self._wp_phase_one_dot_segment_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _phase_one_arrowhead_scalar_solve,
            dim=segment_count,
            inputs=[
                self._wp_phase_one_dot_segment_f64,
                self._wp_phase_one_diagonal,
                self._wp_phase_one_rhs,
                segment_count,
            ],
            outputs=[
                self._wp_minres_failed,
                self._wp_accepted_cost,
                self._wp_ipm_enabled,
                self._wp_ipm_residual_dot_f64,
                self._wp_phase_one_delta,
            ],
            device=self.optimizer.device,
        )
        wp.launch(
            _phase_one_arrowhead_primal_recover,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_ipm_operator_f64,
                self._wp_ipm_solution_f64,
                self._wp_ipm_residual_dot_f64,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_minres_failed, self._wp_delta],
            device=self.optimizer.device,
        )
        wp.launch(
            _phase_one_arrowhead_failure_finalize,
            dim=segment_count,
            inputs=[self._wp_minres_failed, segment_count],
            outputs=[
                self._wp_accepted_cost,
                self._wp_ipm_enabled,
                self._wp_ipm_residual_dot_f64,
                self._wp_phase_one_delta,
            ],
            device=self.optimizer.device,
        )
        wp.launch(
            _phase_one_arrowhead_primal_recover,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_ipm_operator_f64,
                self._wp_ipm_solution_f64,
                self._wp_ipm_residual_dot_f64,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_minres_failed, self._wp_delta],
            device=self.optimizer.device,
        )

    def _solve_block_band_in_place(
        self,
        values: wp.array,
        enabled: wp.array,
        segment_count: int,
    ) -> None:
        """Apply the current block-band inverse to one float64 buffer in place."""
        if self.device.type == "cuda":
            wp.launch(
                _ipm_block_band_forward_f64_parallel,
                dim=self.max_segments * _IPM_BLOCK_BAND_BLOCK_DIM,
                block_dim=_IPM_BLOCK_BAND_BLOCK_DIM,
                inputs=[
                    values,
                    self._wp_ipm_block_band_factor,
                    self._wp_segment_offsets,
                    enabled,
                    segment_count,
                ],
                outputs=[values],
                device=self.optimizer.device,
            )
            wp.launch(
                _ipm_block_band_backward_f64_parallel,
                dim=self.max_segments * _IPM_BLOCK_BAND_BLOCK_DIM,
                block_dim=_IPM_BLOCK_BAND_BLOCK_DIM,
                inputs=[
                    values,
                    self._wp_ipm_block_band_factor,
                    self._wp_segment_offsets,
                    enabled,
                    segment_count,
                ],
                outputs=[values, values],
                device=self.optimizer.device,
            )
        else:
            wp.launch(
                _ipm_block_band_forward_f64,
                dim=self.max_segments,
                inputs=[
                    values,
                    self._wp_ipm_block_band_factor,
                    self._wp_segment_offsets,
                    enabled,
                    segment_count,
                ],
                outputs=[values],
                device=self.optimizer.device,
            )
            wp.launch(
                _ipm_block_band_backward_f64,
                dim=self.max_segments,
                inputs=[
                    values,
                    self._wp_ipm_block_band_factor,
                    self._wp_segment_offsets,
                    enabled,
                    segment_count,
                ],
                outputs=[values, values],
                device=self.optimizer.device,
            )

    def _solve_phase_one_direction_pcg(self, frame_count: int, segment_count: int) -> None:
        """Preserve the combined PCG solve for equality-bearing Phase I."""
        self._minres_failed[:segment_count].zero_()
        wp.launch(
            _pcg_initialize,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_right_hand_side,
                self._wp_normal_diagonal,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[
                self._wp_delta,
                self._wp_pcg_residual,
                self._wp_pcg_preconditioned,
                self._wp_pcg_direction,
            ],
            device=self.optimizer.device,
        )
        wp.launch(
            _phase_one_scalar_pcg_initialize,
            dim=segment_count,
            inputs=[
                self._wp_phase_one_rhs,
                self._wp_phase_one_diagonal,
                self._wp_ipm_enabled,
                segment_count,
            ],
            outputs=[
                self._wp_phase_one_delta,
                self._wp_phase_one_residual,
                self._wp_phase_one_preconditioned,
                self._wp_phase_one_direction,
            ],
            device=self.optimizer.device,
        )
        self._phase_one_combined_dot(
            self._pcg_residual,
            self._pcg_preconditioned,
            self._phase_one_residual,
            self._phase_one_preconditioned,
            self._residual_dot,
            frame_count,
            segment_count,
        )
        wp.launch(
            _pcg_convergence_initialize,
            dim=segment_count,
            inputs=[
                self._wp_residual_dot,
                self._wp_ipm_enabled,
                segment_count,
                float(torch.finfo(torch.float32).eps),
            ],
            outputs=[self._wp_minres_state, self._wp_minres_enabled],
            device=self.optimizer.device,
        )
        for iteration in range(self.krylov_max_iterations):
            self._phase_one_k_apply(
                self._pcg_direction,
                self._wp_pcg_direction,
                self._wp_phase_one_direction,
                frame_count,
                segment_count,
            )
            self._phase_one_combined_dot(
                self._pcg_direction,
                self._pcg_operator_direction,
                self._phase_one_direction,
                self._phase_one_operator,
                self._operator_dot,
                frame_count,
                segment_count,
            )
            wp.launch(
                _pcg_alpha_update,
                dim=(self.capacity, self.dof_count),
                inputs=[
                    self._wp_pcg_direction,
                    self._wp_pcg_operator_direction,
                    self._wp_residual_dot,
                    self._wp_operator_dot,
                    self._wp_frame_segment,
                    self._wp_minres_enabled,
                    frame_count,
                ],
                outputs=[self._wp_delta, self._wp_pcg_residual],
                device=self.optimizer.device,
            )
            wp.launch(
                _phase_one_scalar_pcg_update,
                dim=segment_count,
                inputs=[
                    self._wp_phase_one_operator,
                    self._wp_residual_dot,
                    self._wp_operator_dot,
                    self._wp_minres_enabled,
                    segment_count,
                    self._wp_phase_one_direction,
                ],
                outputs=[self._wp_phase_one_delta, self._wp_phase_one_residual],
                device=self.optimizer.device,
            )
            wp.launch(
                _pcg_precondition,
                dim=(self.capacity, self.dof_count),
                inputs=[
                    self._wp_pcg_residual,
                    self._wp_normal_diagonal,
                    self._wp_frame_segment,
                    self._wp_minres_enabled,
                    frame_count,
                ],
                outputs=[self._wp_pcg_preconditioned],
                device=self.optimizer.device,
            )
            wp.launch(
                _phase_one_scalar_precondition,
                dim=segment_count,
                inputs=[
                    self._wp_phase_one_residual,
                    self._wp_phase_one_diagonal,
                    self._wp_minres_enabled,
                    segment_count,
                ],
                outputs=[self._wp_phase_one_preconditioned],
                device=self.optimizer.device,
            )
            self._phase_one_combined_dot(
                self._pcg_residual,
                self._pcg_preconditioned,
                self._phase_one_residual,
                self._phase_one_preconditioned,
                self._residual_dot_next,
                frame_count,
                segment_count,
            )
            wp.launch(
                _pcg_convergence_update,
                dim=segment_count,
                inputs=[
                    self._wp_residual_dot_next,
                    self._wp_minres_state,
                    0.0,
                    float(torch.finfo(torch.float32).eps),
                ],
                outputs=[self._wp_minres_enabled],
                device=self.optimizer.device,
            )
            check_convergence = self.device.type != "cuda" or (iteration + 1) % self.krylov_check_interval == 0
            if check_convergence and (self.device.type != "cuda" or not self.optimizer.device.is_capturing):
                self._control_pending.zero_()
                wp.launch(
                    _segments_pending,
                    dim=segment_count,
                    inputs=[self._wp_minres_enabled, segment_count],
                    outputs=[self._wp_control_pending],
                    device=self.optimizer.device,
                )
                if not bool(self._control_pending[0]):
                    break
            wp.launch(
                _pcg_beta_update,
                dim=(self.capacity, self.dof_count),
                inputs=[
                    self._wp_pcg_preconditioned,
                    self._wp_residual_dot,
                    self._wp_residual_dot_next,
                    self._wp_frame_segment,
                    self._wp_minres_enabled,
                    frame_count,
                ],
                outputs=[self._wp_pcg_direction],
                device=self.optimizer.device,
            )
            wp.launch(
                _phase_one_scalar_direction_update,
                dim=segment_count,
                inputs=[
                    self._wp_phase_one_preconditioned,
                    self._wp_residual_dot,
                    self._wp_residual_dot_next,
                    self._wp_minres_enabled,
                    segment_count,
                ],
                outputs=[self._wp_phase_one_direction],
                device=self.optimizer.device,
            )
            self._residual_dot[:segment_count].copy_(self._residual_dot_next[:segment_count])

    def _phase_one_k_apply(
        self,
        values: torch.Tensor,
        values_wp: wp.array,
        elastic_values_wp: wp.array,
        frame_count: int,
        segment_count: int,
    ) -> None:
        self._phase_one_constraints_apply(values, values_wp, elastic_values_wp, frame_count)
        self._ipm_constraint_work[:frame_count].mul_(self._ipm_weights[:frame_count])
        self._phase_one_constraints_transpose(
            self._wp_ipm_constraint_work,
            self._wp_phase_one_operator,
            frame_count,
            segment_count,
        )
        self._pcg_operator_direction[:frame_count].copy_(self._ipm_transpose[:frame_count])
        wp.launch(
            _segment_scaled_add,
            dim=(self.capacity, self.dof_count),
            inputs=[values_wp, self._wp_segment_damping, self._wp_frame_segment, frame_count],
            outputs=[self._wp_pcg_operator_direction],
            device=self.optimizer.device,
        )
        wp.launch(
            _segment_scalar_scaled_add,
            dim=self.max_segments,
            inputs=[self._wp_phase_one_direction, self._wp_segment_damping, segment_count],
            outputs=[self._wp_phase_one_operator],
            device=self.optimizer.device,
        )

    def _phase_one_combined_dot(
        self,
        primal_a: torch.Tensor,
        primal_b: torch.Tensor,
        scalar_a: torch.Tensor,
        scalar_b: torch.Tensor,
        output: torch.Tensor,
        frame_count: int,
        segment_count: int,
    ) -> None:
        self._dot(primal_a, primal_b, output, frame_count, segment_count)
        if output is self._residual_dot:
            output_wp = self._wp_residual_dot
        elif output is self._residual_dot_next:
            output_wp = self._wp_residual_dot_next
        else:
            output_wp = self._wp_operator_dot
        if scalar_a is self._phase_one_residual:
            scalar_a_wp = self._wp_phase_one_residual
        else:
            scalar_a_wp = self._wp_phase_one_direction
        if scalar_b is self._phase_one_preconditioned:
            scalar_b_wp = self._wp_phase_one_preconditioned
        else:
            scalar_b_wp = self._wp_phase_one_operator
        wp.launch(
            _phase_one_combined_dot_add,
            dim=segment_count,
            inputs=[
                scalar_a_wp,
                scalar_b_wp,
                self._wp_ipm_enabled,
                segment_count,
            ],
            outputs=[output_wp],
            device=self.optimizer.device,
        )

    def _ipm_convergence_measure(
        self,
        frame_count: int,
        segment_count: int,
        *,
        include_phase_one_scalar: bool,
    ) -> None:
        self._constraint_violation[:segment_count].zero_()
        self._candidate_violation[:segment_count].zero_()
        self._candidate_bound_violation[:segment_count].zero_()
        wp.launch(
            _segment_step_max,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_delta_correction,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_constraint_violation],
            device=self.optimizer.device,
        )
        if include_phase_one_scalar:
            wp.launch(
                _segment_step_max,
                dim=(self.capacity, self.constraint_width),
                inputs=[
                    self._wp_ipm_primal_residual,
                    self._wp_frame_segment,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[self._wp_candidate_violation],
                device=self.optimizer.device,
            )
        else:
            self._ipm_primal_constraints_apply_f64(frame_count)
            wp.launch(
                _ipm_primal_feasibility_violation_max,
                dim=(self.capacity, self.constraint_width),
                inputs=[
                    self._wp_ipm_constraint_work,
                    self._wp_ipm_constraint_rhs,
                    self._wp_ipm_constraint_scale,
                    self._wp_frame_segment,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[self._wp_candidate_violation],
                device=self.optimizer.device,
            )
            wp.launch(
                _segment_step_max,
                dim=(self.capacity, self.active_width),
                inputs=[
                    self._wp_dual_z,
                    self._wp_frame_segment,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[self._wp_candidate_bound_violation],
                device=self.optimizer.device,
            )
        if include_phase_one_scalar:
            wp.launch(
                _phase_one_scalar_max_add,
                dim=segment_count,
                inputs=[
                    self._wp_accepted_cost,
                    self._wp_ipm_enabled,
                    segment_count,
                ],
                outputs=[self._wp_constraint_violation],
                device=self.optimizer.device,
            )

    def _ipm_relative_correction_measure(self, frame_count: int, segment_count: int) -> None:
        """Measure the largest relative affine correction by segment."""
        self._residual_dot_next[:segment_count].zero_()
        wp.launch(
            _ipm_relative_correction_max,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_delta,
                self._wp_ipm_primal,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_residual_dot_next],
            device=self.optimizer.device,
        )

    def _ipm_relative_correction_measure_f64(self, frame_count: int, segment_count: int) -> None:
        """Measure the largest float64 affine correction by segment."""
        self._residual_dot_next[:segment_count].zero_()
        wp.launch(
            _ipm_relative_correction_max_f64,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_ipm_solution_f64,
                self._wp_ipm_primal_f64,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_residual_dot_next],
            device=self.optimizer.device,
        )

    def _ipm_complementarity_max_measure(self, frame_count: int, segment_count: int) -> None:
        """Measure the largest physical primal-dual gap by segment."""
        self._operator_dot[:segment_count].zero_()
        wp.launch(
            _ipm_complementarity_max,
            dim=(self.capacity, self.constraint_width),
            inputs=[
                self._wp_ipm_slack,
                self._wp_ipm_multiplier,
                self._wp_ipm_constraint_scale,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_operator_dot],
            device=self.optimizer.device,
        )

    def _ipm_convergence_mask(
        self,
        frame_count: int,
        segment_count: int,
        *,
        include_phase_one_scalar: bool,
    ) -> None:
        self._ipm_convergence_measure(
            frame_count,
            segment_count,
            include_phase_one_scalar=include_phase_one_scalar,
        )
        wp.launch(
            _ipm_barrier_convergence_mask,
            dim=segment_count,
            inputs=[
                self._wp_ipm_complementarity,
                self._wp_ipm_inequality_count,
                self._wp_constraint_violation,
                self._wp_candidate_bound_violation,
                self._wp_candidate_violation,
                float(torch.finfo(torch.float32).eps),
                32.0 * float(torch.finfo(torch.float32).eps),
                32.0 * float(torch.finfo(torch.float32).eps),
                segment_count,
            ],
            outputs=[self._wp_ipm_enabled],
            device=self.optimizer.device,
        )

    def _ipm_newton_right_hand_side(self, frame_count: int, *, corrector: bool) -> None:
        """Build one condensed predictor or corrector right-hand side."""
        self._ipm_primal_constraints_apply_f64(frame_count)
        if corrector:
            wp.launch(
                _ipm_residual_corrector_f64,
                dim=(self.capacity, self.inequality_width),
                inputs=[
                    self._wp_ipm_constraint_work,
                    self._wp_ipm_constraint_rhs,
                    self._wp_ipm_constraint_scale,
                    self._wp_ipm_slack,
                    self._wp_ipm_multiplier,
                    self._wp_ipm_augmented_row_solution_f64,
                    self._wp_ipm_augmented_row_r1_f64,
                    self._wp_ipm_centering,
                    self._wp_ipm_complementarity,
                    self._wp_frame_segment,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[
                    self._wp_ipm_primal_residual,
                    self._wp_ipm_complementarity_residual,
                    self._wp_ipm_weights,
                    self._wp_ipm_constraint_work,
                ],
                device=self.optimizer.device,
            )
        else:
            wp.launch(
                _ipm_residual,
                dim=(self.capacity, self.constraint_width),
                inputs=[
                    self._wp_ipm_constraint_work,
                    self._wp_ipm_constraint_rhs,
                    self._wp_ipm_constraint_scale,
                    self._wp_ipm_slack,
                    self._wp_ipm_multiplier,
                    self._wp_ipm_affine_slack,
                    self._wp_ipm_affine_multiplier,
                    self._wp_ipm_centering,
                    self._wp_ipm_complementarity,
                    wp.uint8(0),
                    self._wp_frame_segment,
                    self._wp_ipm_enabled,
                    frame_count,
                ],
                outputs=[
                    self._wp_ipm_primal_residual,
                    self._wp_ipm_complementarity_residual,
                    self._wp_ipm_weights,
                    self._wp_ipm_constraint_work,
                ],
                device=self.optimizer.device,
            )
        wp.launch(
            _ipm_feature_f64,
            dim=(self.capacity, self.residual_count),
            inputs=[
                self._wp_jacobian,
                self._wp_ipm_primal_f64,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_ipm_feature_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _precision_apply_f64,
            dim=(self.capacity, self.residual_count),
            inputs=[
                self._wp_ipm_feature_f64,
                self._wp_base_weights,
                self._wp_temporal_weights,
                self._wp_residual_activity,
                self._wp_activity_group_by_residual,
                self._wp_first_difference_group_by_residual,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_step_seconds,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_ipm_precision_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _normal_apply_f64,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_jacobian,
                self._wp_ipm_precision_f64,
                self._wp_ipm_primal_f64,
                self._wp_segment_damping,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_ipm_operator_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_constraints_transpose_f64,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_jacobian,
                self._wp_ipm_multiplier,
                self._wp_coordinate_dof_indices,
                self._coordinate_bound_count,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_ipm_enabled,
                frame_count,
                self._wp_ipm_constraint_scale,
            ],
            outputs=[self._wp_ipm_augmented_primal_work_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _qp_rows_transpose,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_jacobian,
                self._wp_coordinate_dof_indices,
                self._wp_coordinate_lower,
                self._wp_coordinate_upper,
                0,
                self.dof_count,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_ipm_enabled,
                frame_count,
                self._wp_active_row_codes,
                self._wp_active_row_scales,
                self._wp_active_equality_count,
                self._wp_ipm_equality_dual,
            ],
            outputs=[self._wp_active_row_transpose],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_rhs_stationarity_f64,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_ipm_operator_f64,
                self._wp_ipm_gradient,
                self._wp_ipm_augmented_primal_work_f64,
                self._wp_active_row_transpose,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_ipm_augmented_primal_r2_f64, self._wp_delta_correction],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_constraints_transpose_f64,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_jacobian,
                self._wp_ipm_constraint_work,
                self._wp_coordinate_dof_indices,
                self._coordinate_bound_count,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_ipm_enabled,
                frame_count,
                self._wp_ipm_constraint_scale,
            ],
            outputs=[self._wp_ipm_augmented_primal_work_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_rhs_condense_f64,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_ipm_augmented_primal_r2_f64,
                self._wp_ipm_augmented_primal_work_f64,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_ipm_right_hand_side_f64, self._wp_right_hand_side],
            device=self.optimizer.device,
        )
        self._dual_r1[:frame_count].copy_(self._ipm_equality_rhs[:frame_count])
        self._dual_r2[:frame_count].zero_()
        torch.bmm(
            self._jacobian[:frame_count],
            self._ipm_primal[:frame_count].unsqueeze(-1),
            out=self._feature[:frame_count].unsqueeze(-1),
        )
        wp.launch(
            _qp_rows_operator_dual,
            dim=(self.capacity, self.active_width),
            inputs=[
                self._wp_feature,
                self._wp_ipm_primal,
                self._wp_coordinate_dof_indices,
                self._wp_coordinate_lower,
                self._wp_coordinate_upper,
                0,
                self.dof_count,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_ipm_enabled,
                frame_count,
                self._wp_active_row_codes,
                self._wp_active_row_scales,
                self._wp_active_equality_count,
                self._wp_dual_r2,
            ],
            outputs=[self._wp_dual_r2],
            device=self.optimizer.device,
        )
        self._dual_r1[:frame_count].sub_(self._dual_r2[:frame_count])
        wp.launch(
            _qp_rows_residual_unscale,
            dim=(self.capacity, self.active_width),
            inputs=[
                self._wp_dual_r1,
                self._wp_active_row_scales,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_dual_z],
            device=self.optimizer.device,
        )
        wp.launch(
            _normal_diagonal,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_jacobian,
                self._wp_precision_diagonal,
                self._wp_segment_damping,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_normal_diagonal],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_normal_diagonal_add,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_jacobian,
                self._wp_ipm_weights,
                self._wp_coordinate_dof_indices,
                self._coordinate_bound_count,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_ipm_enabled,
                frame_count,
                self._wp_ipm_constraint_scale,
            ],
            outputs=[self._wp_normal_diagonal],
            device=self.optimizer.device,
        )

    def _solve_ipm_direction(
        self,
        frame_count: int,
        segment_count: int,
        has_equalities: bool,
        *,
        rebuild_factor: bool = True,
    ) -> None:
        """Solve one fixed-work condensed Newton direction."""
        self._kkt_enabled[:segment_count].copy_(self._ipm_enabled[:segment_count])
        if has_equalities:
            self._solve_minres(frame_count, segment_count)
            self._ipm_solution_f64[:frame_count].zero_()
            wp.launch(
                _ipm_solution_accumulate_f64,
                dim=(self.capacity, self.dof_count),
                inputs=[self._wp_delta, self._wp_frame_segment, self._wp_kkt_enabled, frame_count],
                outputs=[self._wp_ipm_solution_f64],
                device=self.optimizer.device,
            )
        else:
            self._solve_ipm_block_band(frame_count, segment_count, rebuild_factor=rebuild_factor)
        wp.launch(
            _ipm_solve_status,
            dim=segment_count,
            inputs=[
                self._wp_ipm_enabled,
                self._wp_minres_enabled,
                self._wp_minres_failed,
                segment_count,
            ],
            outputs=[self._wp_ipm_linear_converged, self._wp_ipm_enabled],
            device=self.optimizer.device,
        )

    def _prepare_ipm_objective_block_band(self, frame_count: int, segment_count: int) -> None:
        """Build the invariant objective block band and eagerly factor it under capture."""
        self._ipm_objective_factor_prepared = False
        wp.launch(
            _ipm_block_band_matrix_build,
            dim=(self.capacity, 4, self.dof_count, self.dof_count),
            inputs=[
                self._wp_jacobian,
                self._wp_base_weights,
                self._wp_temporal_weights,
                self._wp_residual_activity,
                self._wp_activity_group_by_residual,
                self._wp_first_difference_group_by_residual,
                self._wp_step_seconds,
                self._wp_segment_damping,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_ipm_objective_block_band],
            device=self.optimizer.device,
        )
        if self.device.type == "cuda" and self.optimizer.device.is_capturing:
            self._factor_ipm_objective_block_band(frame_count, segment_count)

    def _factor_ipm_objective_block_band(self, frame_count: int, segment_count: int) -> None:
        """Cache the invariant objective factor for the augmented fallback."""
        wp.copy(
            self._wp_ipm_objective_block_band_factor,
            self._wp_ipm_objective_block_band,
            count=frame_count * 4 * self.dof_count * self.dof_count,
        )
        self._ipm_objective_factor_failed[:segment_count].zero_()
        if self.device.type == "cuda":
            wp.launch(
                _ipm_block_band_matrix_factor_parallel,
                dim=self.max_segments * _IPM_BLOCK_BAND_BLOCK_DIM,
                block_dim=_IPM_BLOCK_BAND_BLOCK_DIM,
                inputs=[
                    self._wp_ipm_objective_block_band_factor,
                    self._wp_segment_offsets,
                    self._wp_ipm_enabled,
                    segment_count,
                ],
                outputs=[self._wp_ipm_objective_factor_failed],
                device=self.optimizer.device,
            )
        else:
            wp.launch(
                _ipm_block_band_matrix_factor,
                dim=self.max_segments,
                inputs=[
                    self._wp_ipm_objective_block_band_factor,
                    self._wp_segment_offsets,
                    self._wp_ipm_enabled,
                    segment_count,
                ],
                outputs=[self._wp_ipm_objective_factor_failed],
                device=self.optimizer.device,
            )
        self._ipm_objective_factor_prepared = True

    def _factor_ipm_block_band(
        self,
        frame_count: int,
        segment_count: int,
        enabled_wp: wp.array,
    ) -> None:
        """Restore the objective blocks, add the current barrier, and factor."""
        wp.copy(
            self._wp_ipm_block_band_factor,
            self._wp_ipm_objective_block_band,
            count=frame_count * 4 * self.dof_count * self.dof_count,
        )
        wp.launch(
            _ipm_block_band_barrier_add,
            dim=(self.capacity, 4, self.dof_count, self.dof_count),
            inputs=[
                self._wp_jacobian,
                self._wp_ipm_weights,
                self._wp_coordinate_dof_indices,
                self._coordinate_bound_count,
                self._wp_ipm_constraint_scale,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                enabled_wp,
                frame_count,
            ],
            outputs=[self._wp_ipm_block_band_factor],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_block_band_scale_build_f64,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_ipm_block_band_factor,
                self._wp_frame_segment,
                enabled_wp,
                frame_count,
            ],
            outputs=[self._wp_ipm_block_band_scale_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_block_band_equilibrate_f64,
            dim=(self.capacity, 4, self.dof_count, self.dof_count),
            inputs=[
                self._wp_ipm_block_band_scale_f64,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                enabled_wp,
                float(torch.finfo(torch.float64).eps),
                frame_count,
            ],
            outputs=[self._wp_ipm_block_band_factor],
            device=self.optimizer.device,
        )
        self._factor_ipm_block_band_in_place(enabled_wp, segment_count)

    def _factor_ipm_block_band_in_place(self, enabled_wp: wp.array, segment_count: int) -> None:
        """Factor selected assembled and scaled block-band segments in place."""
        if self.device.type == "cuda":
            wp.launch(
                _ipm_block_band_matrix_factor_parallel,
                dim=self.max_segments * _IPM_BLOCK_BAND_BLOCK_DIM,
                block_dim=_IPM_BLOCK_BAND_BLOCK_DIM,
                inputs=[
                    self._wp_ipm_block_band_factor,
                    self._wp_segment_offsets,
                    enabled_wp,
                    segment_count,
                ],
                outputs=[self._wp_minres_failed],
                device=self.optimizer.device,
            )
        else:
            wp.launch(
                _ipm_block_band_matrix_factor,
                dim=self.max_segments,
                inputs=[self._wp_ipm_block_band_factor, self._wp_segment_offsets, enabled_wp, segment_count],
                outputs=[self._wp_minres_failed],
                device=self.optimizer.device,
            )

    def _solve_ipm_block_band(
        self,
        frame_count: int,
        segment_count: int,
        *,
        rebuild_factor: bool = True,
    ) -> None:
        """Solve directly when certified, with a full augmented fallback."""
        self._minres_failed[:segment_count].zero_()
        self._minres_enabled[:segment_count].zero_()
        if rebuild_factor:
            self._ipm_augmented_fallback[:segment_count].zero_()
            self._factor_ipm_block_band(frame_count, segment_count, self._wp_kkt_enabled)
            wp.launch(
                _ipm_fallback_promote_factor_failure,
                dim=segment_count,
                inputs=[self._wp_kkt_enabled, self._wp_minres_failed, segment_count],
                outputs=[self._wp_ipm_augmented_fallback, self._wp_ipm_certify],
                device=self.optimizer.device,
            )

        self._ipm_solution_f64[:frame_count].zero_()
        wp.launch(
            _ipm_physical_residual_from_f64,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_ipm_right_hand_side_f64,
                self._wp_ipm_solution_f64,
                self._wp_frame_segment,
                self._wp_kkt_enabled,
                frame_count,
            ],
            outputs=[self._wp_ipm_operator_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _square_norm_f64,
            dim=self.capacity,
            inputs=[
                self._wp_ipm_operator_f64,
                self._wp_frame_segment,
                self._wp_kkt_enabled,
                frame_count,
            ],
            outputs=[self._wp_ipm_dot_frame_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _dot_segments_f64,
            dim=self.max_segments,
            inputs=[
                self._wp_ipm_dot_frame_f64,
                self._wp_segment_offsets,
                segment_count,
            ],
            outputs=[self._wp_ipm_residual_dot_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_physical_convergence_initialize,
            dim=segment_count,
            inputs=[
                self._wp_ipm_residual_dot_f64,
                self._wp_kkt_enabled,
                segment_count,
                float(torch.finfo(torch.float32).eps),
            ],
            outputs=[self._wp_ipm_initial_norm_f64, self._wp_minres_enabled, self._wp_minres_failed],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_route_select,
            dim=segment_count,
            inputs=[
                self._wp_minres_enabled,
                self._wp_ipm_augmented_fallback,
                self._wp_minres_failed,
                wp.uint8(0),
                segment_count,
            ],
            outputs=[self._wp_ipm_certify],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_solution_accumulate_from_f64,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_ipm_right_hand_side_f64,
                self._wp_frame_segment,
                self._wp_ipm_certify,
                frame_count,
            ],
            outputs=[self._wp_ipm_solution_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_solution_equilibrate_f64,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_ipm_block_band_scale_f64,
                self._wp_frame_segment,
                self._wp_ipm_certify,
                frame_count,
            ],
            outputs=[self._wp_ipm_solution_f64],
            device=self.optimizer.device,
        )
        self._solve_block_band_in_place(self._wp_ipm_solution_f64, self._wp_ipm_certify, segment_count)
        wp.launch(
            _ipm_solution_equilibrate_f64,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_ipm_block_band_scale_f64,
                self._wp_frame_segment,
                self._wp_ipm_certify,
                frame_count,
            ],
            outputs=[self._wp_ipm_solution_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_route_select,
            dim=segment_count,
            inputs=[
                self._wp_minres_enabled,
                self._wp_ipm_augmented_fallback,
                self._wp_minres_failed,
                wp.uint8(1),
                segment_count,
            ],
            outputs=[self._wp_ipm_certify],
            device=self.optimizer.device,
        )
        self._solve_ipm_augmented(self._wp_ipm_certify, frame_count, segment_count)
        self._ipm_physical_certificate(self._wp_minres_enabled, frame_count, segment_count)
        wp.launch(
            _ipm_fallback_promote_certificate_failure,
            dim=segment_count,
            inputs=[
                self._wp_kkt_enabled,
                self._wp_ipm_augmented_fallback,
                self._wp_minres_enabled,
                self._wp_minres_failed,
                segment_count,
            ],
            outputs=[self._wp_ipm_certify],
            device=self.optimizer.device,
        )
        self._solve_ipm_augmented(self._wp_ipm_certify, frame_count, segment_count)
        self._ipm_physical_certificate(self._wp_minres_enabled, frame_count, segment_count)

    def _ipm_physical_certificate(
        self,
        enabled_wp: wp.array,
        frame_count: int,
        segment_count: int,
    ) -> None:
        """Certify a float64 direction against the unchanged physical operator."""
        self._ipm_strict_physical_certificate(self._wp_ipm_solution_f64, enabled_wp, frame_count, segment_count)

    def _ipm_strict_physical_certificate(
        self,
        solution_wp: wp.array,
        enabled_wp: wp.array,
        frame_count: int,
        segment_count: int,
    ) -> None:
        """Certify one physical solution with the unchanged symmetric residual tolerance."""
        self._ipm_k_apply_f64(solution_wp, enabled_wp, frame_count)
        wp.launch(
            _square_norm_f64,
            dim=self.capacity,
            inputs=[self._wp_ipm_operator_f64, self._wp_frame_segment, enabled_wp, frame_count],
            outputs=[self._wp_ipm_dot_frame_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _dot_segments_f64,
            dim=self.max_segments,
            inputs=[self._wp_ipm_dot_frame_f64, self._wp_segment_offsets, segment_count],
            outputs=[self._wp_ipm_residual_dot_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_physical_reference_norm_f64,
            dim=segment_count,
            inputs=[
                self._wp_ipm_residual_dot_f64,
                self._wp_ipm_right_hand_side_f64,
                self._wp_segment_offsets,
                enabled_wp,
                segment_count,
            ],
            outputs=[self._wp_ipm_initial_norm_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_physical_residual_from_f64,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_ipm_right_hand_side_f64,
                self._wp_ipm_operator_f64,
                self._wp_frame_segment,
                enabled_wp,
                frame_count,
            ],
            outputs=[self._wp_ipm_operator_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _square_norm_f64,
            dim=self.capacity,
            inputs=[self._wp_ipm_operator_f64, self._wp_frame_segment, enabled_wp, frame_count],
            outputs=[self._wp_ipm_dot_frame_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _dot_segments_f64,
            dim=self.max_segments,
            inputs=[self._wp_ipm_dot_frame_f64, self._wp_segment_offsets, segment_count],
            outputs=[self._wp_ipm_residual_dot_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_physical_convergence_update,
            dim=segment_count,
            inputs=[
                self._wp_ipm_residual_dot_f64,
                self._wp_ipm_initial_norm_f64,
                self.krylov_relative_tolerance,
                float(torch.finfo(torch.float32).eps),
            ],
            outputs=[enabled_wp, self._wp_minres_failed],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_outer_forcing_update,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_ipm_operator_f64,
                self._wp_delta_correction,
                self._wp_frame_segment,
                self._wp_kkt_enabled,
                self._wp_minres_failed,
                self.krylov_relative_tolerance,
                max(32.0 * float(torch.finfo(torch.float32).eps), self.kkt_relative_tolerance),
                frame_count,
            ],
            outputs=[enabled_wp],
            device=self.optimizer.device,
        )

    def _solve_ipm_augmented(self, enabled_wp: wp.array, frame_count: int, segment_count: int) -> None:
        """Solve the full two-sided-whitened augmented system in float64."""
        if self.device.type != "cuda" or not self.optimizer.device.is_capturing:
            self._control_pending.zero_()
            wp.launch(
                _segments_pending,
                dim=segment_count,
                inputs=[enabled_wp, segment_count],
                outputs=[self._wp_control_pending],
                device=self.optimizer.device,
            )
            if not bool(self._control_pending[0]):
                return
            if not self._ipm_objective_factor_prepared:
                self._factor_ipm_objective_block_band(frame_count, segment_count)
        wp.launch(
            _ipm_augmented_minres_solve_f64,
            dim=self.max_segments,
            inputs=[
                self._wp_ipm_objective_block_band_factor,
                self._wp_ipm_objective_factor_failed,
                self._wp_jacobian,
                self._wp_ipm_weights,
                self._wp_coordinate_dof_indices,
                self._coordinate_bound_count,
                self._wp_ipm_constraint_scale,
                self._wp_segment_offsets,
                self._wp_ipm_right_hand_side_f64,
                self.krylov_relative_tolerance,
                float(torch.finfo(torch.float64).eps),
                self.krylov_max_iterations,
                segment_count,
            ],
            outputs=[
                self._wp_ipm_solution_f64,
                self._wp_ipm_operator_f64,
                self._wp_ipm_augmented_primal_r2_f64,
                self._wp_ipm_augmented_primal_basis_f64,
                self._wp_ipm_augmented_primal_work_f64,
                self._wp_ipm_augmented_primal_direction_older_f64,
                self._wp_ipm_augmented_primal_direction_old_f64,
                self._wp_ipm_augmented_row_solution_f64,
                self._wp_ipm_augmented_row_r1_f64,
                self._wp_ipm_augmented_row_r2_f64,
                self._wp_ipm_augmented_row_basis_f64,
                self._wp_ipm_augmented_row_direction_older_f64,
                self._wp_ipm_augmented_row_direction_old_f64,
                enabled_wp,
                self._wp_minres_failed,
            ],
            device=self.optimizer.device,
        )

    def _ipm_temporal_factorize(self, frame_count: int, segment_count: int) -> None:
        """Factor the exact same-DOF bandwidth-three temporal blocks."""
        wp.launch(
            _ipm_temporal_band_build,
            dim=(self.capacity, self.dof_count, 4),
            inputs=[
                self._wp_jacobian,
                self._wp_precision_diagonal,
                self._wp_temporal_weights,
                self._wp_residual_activity,
                self._wp_activity_group_by_residual,
                self._wp_first_difference_group_by_residual,
                self._wp_step_seconds,
                self._wp_ipm_weights,
                self._wp_coordinate_dof_indices,
                self._coordinate_bound_count,
                self._wp_ipm_constraint_scale,
                self._wp_segment_damping,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_kkt_enabled,
                frame_count,
            ],
            outputs=[self._wp_ipm_temporal_factor],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_temporal_band_factor,
            dim=(self.max_segments, self.dof_count),
            inputs=[
                self._wp_ipm_temporal_factor,
                self._wp_segment_offsets,
                self._wp_kkt_enabled,
                segment_count,
            ],
            outputs=[self._wp_minres_failed],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_singleton_matrix_build,
            dim=(self.max_segments, self.dof_count, self.dof_count),
            inputs=[
                self._wp_jacobian,
                self._wp_base_weights,
                self._wp_residual_activity,
                self._wp_activity_group_by_residual,
                self._wp_ipm_weights,
                self._wp_coordinate_dof_indices,
                self._coordinate_bound_count,
                self._wp_ipm_constraint_scale,
                self._wp_segment_damping,
                self._wp_segment_offsets,
                self._wp_kkt_enabled,
                segment_count,
            ],
            outputs=[self._wp_ipm_singleton_factor],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_singleton_matrix_factor,
            dim=self.max_segments,
            inputs=[
                self._wp_ipm_singleton_factor,
                self._wp_segment_offsets,
                self._wp_kkt_enabled,
                segment_count,
            ],
            outputs=[self._wp_minres_failed],
            device=self.optimizer.device,
        )
        self._ipm_segment_coupled[:segment_count].zero_()
        wp.launch(
            _ipm_frame_matrix_build,
            dim=(self.capacity, self.dof_count, self.dof_count),
            inputs=[
                self._wp_jacobian,
                self._wp_precision_diagonal,
                self._wp_ipm_weights,
                self._wp_ipm_constraint_scale,
                self._wp_coordinate_dof_indices,
                self._coordinate_bound_count,
                self._wp_segment_damping,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_kkt_enabled,
                frame_count,
            ],
            outputs=[self._wp_ipm_frame_factor, self._wp_ipm_segment_coupled],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_frame_matrix_factor,
            dim=self.capacity,
            inputs=[
                self._wp_ipm_frame_factor,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_kkt_enabled,
                frame_count,
            ],
            outputs=[self._wp_minres_failed],
            device=self.optimizer.device,
        )

    def _ipm_k_apply_f64(self, values_wp: wp.array, enabled_wp: wp.array, frame_count: int) -> None:
        """Apply the physical condensed Phase-II matrix with float64 accumulation."""
        wp.launch(
            _ipm_feature_f64,
            dim=(self.capacity, self.residual_count),
            inputs=[
                self._wp_jacobian,
                values_wp,
                self._wp_frame_segment,
                enabled_wp,
                frame_count,
            ],
            outputs=[self._wp_ipm_feature_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _precision_apply_f64,
            dim=(self.capacity, self.residual_count),
            inputs=[
                self._wp_ipm_feature_f64,
                self._wp_base_weights,
                self._wp_temporal_weights,
                self._wp_residual_activity,
                self._wp_activity_group_by_residual,
                self._wp_first_difference_group_by_residual,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_step_seconds,
                enabled_wp,
                frame_count,
            ],
            outputs=[self._wp_ipm_precision_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _normal_apply_f64,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_jacobian,
                self._wp_ipm_precision_f64,
                values_wp,
                self._wp_segment_damping,
                self._wp_frame_segment,
                enabled_wp,
                frame_count,
            ],
            outputs=[self._wp_ipm_operator_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_barrier_add_f64,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_jacobian,
                self._wp_ipm_feature_f64,
                values_wp,
                self._wp_ipm_weights,
                self._wp_coordinate_dof_indices,
                self._coordinate_bound_count,
                self._wp_ipm_constraint_scale,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                enabled_wp,
                frame_count,
            ],
            outputs=[self._wp_ipm_operator_f64],
            device=self.optimizer.device,
        )

    def _ipm_physical_residual_update(
        self,
        enabled_wp: wp.array,
        frame_count: int,
        segment_count: int,
    ) -> None:
        """Recompute and temporally whiten the physical residual in float64."""
        self._ipm_k_apply_f64(self._wp_ipm_solution_f64, enabled_wp, frame_count)
        wp.launch(
            _ipm_physical_residual_f64,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_right_hand_side,
                self._wp_ipm_operator_f64,
                self._wp_frame_segment,
                enabled_wp,
                frame_count,
            ],
            outputs=[self._wp_ipm_operator_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_temporal_forward_f64,
            dim=(self.max_segments, self.dof_count),
            inputs=[
                self._wp_ipm_operator_f64,
                self._wp_ipm_temporal_factor,
                self._wp_segment_offsets,
                enabled_wp,
                segment_count,
            ],
            outputs=[self._wp_ipm_operator_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_singleton_forward_f64,
            dim=self.max_segments,
            inputs=[
                self._wp_ipm_operator_f64,
                self._wp_ipm_singleton_factor,
                self._wp_segment_offsets,
                enabled_wp,
                segment_count,
            ],
            outputs=[self._wp_ipm_operator_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _square_norm_f64,
            dim=self.capacity,
            inputs=[
                self._wp_ipm_operator_f64,
                self._wp_frame_segment,
                enabled_wp,
                frame_count,
            ],
            outputs=[self._wp_ipm_dot_frame_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _dot_segments_f64,
            dim=self.max_segments,
            inputs=[
                self._wp_ipm_dot_frame_f64,
                self._wp_segment_offsets,
                segment_count,
            ],
            outputs=[self._wp_ipm_residual_dot_f64],
            device=self.optimizer.device,
        )

    def _ipm_transformed_k_apply(self, frame_count: int, segment_count: int) -> None:
        """Apply L-inverse K L-transpose-inverse to one float32 CG direction."""
        wp.launch(
            _ipm_temporal_backward_f32,
            dim=(self.max_segments, self.dof_count),
            inputs=[
                self._wp_pcg_direction,
                self._wp_ipm_temporal_factor,
                self._wp_segment_offsets,
                self._wp_minres_enabled,
                segment_count,
            ],
            outputs=[self._wp_pcg_preconditioned],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_singleton_backward_f32,
            dim=self.max_segments,
            inputs=[
                self._wp_pcg_direction,
                self._wp_ipm_singleton_factor,
                self._wp_segment_offsets,
                self._wp_minres_enabled,
                segment_count,
            ],
            outputs=[self._wp_pcg_preconditioned],
            device=self.optimizer.device,
        )
        self._ipm_k_apply(self._pcg_preconditioned, self._wp_pcg_preconditioned, frame_count)
        wp.launch(
            _ipm_temporal_forward_f32,
            dim=(self.max_segments, self.dof_count),
            inputs=[
                self._wp_pcg_operator_direction,
                self._wp_ipm_temporal_factor,
                self._wp_segment_offsets,
                self._wp_minres_enabled,
                segment_count,
            ],
            outputs=[self._wp_delta_candidate],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_singleton_forward_f32,
            dim=self.max_segments,
            inputs=[
                self._wp_pcg_operator_direction,
                self._wp_ipm_singleton_factor,
                self._wp_segment_offsets,
                self._wp_minres_enabled,
                segment_count,
            ],
            outputs=[self._wp_delta_candidate],
            device=self.optimizer.device,
        )

    def _ipm_additive_precondition(
        self,
        values: torch.Tensor,
        values_wp: wp.array,
        enabled_wp: wp.array,
        frame_count: int,
        segment_count: int,
    ) -> None:
        """Apply identity plus the transformed exact frame-block inverse."""
        wp.launch(
            _ipm_temporal_lower_multiply_f32,
            dim=(self.max_segments, self.dof_count),
            inputs=[
                values_wp,
                self._wp_ipm_temporal_factor,
                self._wp_segment_offsets,
                enabled_wp,
                segment_count,
            ],
            outputs=[self._wp_delta_candidate],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_frame_solve_f32,
            dim=self.capacity,
            inputs=[
                self._wp_delta_candidate,
                self._wp_ipm_frame_factor,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                enabled_wp,
                self._wp_ipm_segment_coupled,
                frame_count,
            ],
            outputs=[self._wp_ipm_operator_f64, self._wp_pcg_preconditioned],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_temporal_upper_multiply_f32,
            dim=(self.max_segments, self.dof_count),
            inputs=[
                self._wp_pcg_preconditioned,
                self._wp_ipm_temporal_factor,
                self._wp_segment_offsets,
                enabled_wp,
                segment_count,
            ],
            outputs=[self._wp_delta_candidate],
            device=self.optimizer.device,
        )
        wp.launch(
            _pcg_additive_precondition_combine,
            dim=(self.capacity, self.dof_count),
            inputs=[
                values_wp,
                self._wp_delta_candidate,
                self._wp_frame_segment,
                enabled_wp,
                frame_count,
            ],
            outputs=[self._wp_pcg_preconditioned],
            device=self.optimizer.device,
        )

    def _solve_ipm_pcg(
        self,
        frame_count: int,
        segment_count: int,
        *,
        rebuild_factor: bool = True,
    ) -> None:
        """Solve one SPD condensed system by split PCG with physical refinement."""
        self._minres_failed[:segment_count].zero_()
        self._ipm_recursive_converged[:segment_count].zero_()
        if rebuild_factor:
            self._ipm_temporal_factorize(frame_count, segment_count)

        self._ipm_solution_f64[:frame_count].zero_()
        self._ipm_physical_residual_update(self._wp_kkt_enabled, frame_count, segment_count)
        wp.launch(
            _ipm_physical_convergence_initialize,
            dim=segment_count,
            inputs=[
                self._wp_ipm_residual_dot_f64,
                self._wp_ipm_enabled,
                segment_count,
                float(torch.finfo(torch.float32).eps),
            ],
            outputs=[
                self._wp_ipm_initial_norm_f64,
                self._wp_kkt_enabled,
                self._wp_minres_failed,
            ],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_solution_copy_f32,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_ipm_operator_f64,
                self._wp_frame_segment,
                self._wp_kkt_enabled,
                frame_count,
            ],
            outputs=[self._wp_delta_correction],
            device=self.optimizer.device,
        )

        wp.launch(
            _pcg_identity_initialize,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_delta_correction,
                self._wp_frame_segment,
                self._wp_kkt_enabled,
                frame_count,
            ],
            outputs=[
                self._wp_delta,
                self._wp_pcg_residual,
                self._wp_pcg_preconditioned,
                self._wp_pcg_direction,
            ],
            device=self.optimizer.device,
        )
        self._ipm_additive_precondition(
            self._pcg_residual,
            self._wp_pcg_residual,
            self._wp_kkt_enabled,
            frame_count,
            segment_count,
        )
        self._pcg_direction[:frame_count].copy_(self._pcg_preconditioned[:frame_count])
        self._dot(
            self._pcg_residual,
            self._pcg_preconditioned,
            self._residual_dot,
            frame_count,
            segment_count,
        )
        wp.launch(
            _pcg_convergence_initialize,
            dim=segment_count,
            inputs=[
                self._wp_residual_dot,
                self._wp_kkt_enabled,
                segment_count,
                float(torch.finfo(torch.float32).eps),
            ],
            outputs=[self._wp_minres_state, self._wp_minres_enabled],
            device=self.optimizer.device,
        )
        completed_iterations = 0
        while completed_iterations < self.krylov_max_iterations:
            chunk_iterations = min(self.krylov_check_interval, self.krylov_max_iterations - completed_iterations)
            for _ in range(chunk_iterations):
                self._ipm_transformed_k_apply(frame_count, segment_count)
                self._dot(
                    self._pcg_direction,
                    self._delta_candidate,
                    self._operator_dot,
                    frame_count,
                    segment_count,
                )
                wp.launch(
                    _pcg_alpha_update,
                    dim=(self.capacity, self.dof_count),
                    inputs=[
                        self._wp_pcg_direction,
                        self._wp_delta_candidate,
                        self._wp_residual_dot,
                        self._wp_operator_dot,
                        self._wp_frame_segment,
                        self._wp_minres_enabled,
                        frame_count,
                    ],
                    outputs=[self._wp_delta, self._wp_pcg_residual],
                    device=self.optimizer.device,
                )
                self._ipm_additive_precondition(
                    self._pcg_residual,
                    self._wp_pcg_residual,
                    self._wp_minres_enabled,
                    frame_count,
                    segment_count,
                )
                self._dot(
                    self._pcg_residual,
                    self._pcg_preconditioned,
                    self._residual_dot_next,
                    frame_count,
                    segment_count,
                )
                wp.launch(
                    _ipm_pcg_convergence_update,
                    dim=segment_count,
                    inputs=[
                        self._wp_residual_dot_next,
                        self._wp_minres_state,
                        self.krylov_relative_tolerance,
                        float(torch.finfo(torch.float32).eps),
                    ],
                    outputs=[
                        self._wp_minres_enabled,
                        self._wp_ipm_recursive_converged,
                        self._wp_minres_failed,
                    ],
                    device=self.optimizer.device,
                )
                wp.launch(
                    _pcg_beta_update,
                    dim=(self.capacity, self.dof_count),
                    inputs=[
                        self._wp_pcg_preconditioned,
                        self._wp_residual_dot,
                        self._wp_residual_dot_next,
                        self._wp_frame_segment,
                        self._wp_minres_enabled,
                        frame_count,
                    ],
                    outputs=[self._wp_pcg_direction],
                    device=self.optimizer.device,
                )
                self._residual_dot[:segment_count].copy_(self._residual_dot_next[:segment_count])
            completed_iterations += chunk_iterations
            if completed_iterations == self.krylov_max_iterations:
                break

            wp.launch(
                _ipm_certification_select,
                dim=segment_count,
                inputs=[
                    self._wp_kkt_enabled,
                    self._wp_minres_enabled,
                    self._wp_ipm_recursive_converged,
                    self._wp_minres_failed,
                    segment_count,
                ],
                outputs=[self._wp_ipm_certify],
                device=self.optimizer.device,
            )
            wp.launch(
                _ipm_temporal_backward_f32,
                dim=(self.max_segments, self.dof_count),
                inputs=[
                    self._wp_delta,
                    self._wp_ipm_temporal_factor,
                    self._wp_segment_offsets,
                    self._wp_ipm_certify,
                    segment_count,
                ],
                outputs=[self._wp_pcg_preconditioned],
                device=self.optimizer.device,
            )
            wp.launch(
                _ipm_singleton_backward_f32,
                dim=self.max_segments,
                inputs=[
                    self._wp_delta,
                    self._wp_ipm_singleton_factor,
                    self._wp_segment_offsets,
                    self._wp_ipm_certify,
                    segment_count,
                ],
                outputs=[self._wp_pcg_preconditioned],
                device=self.optimizer.device,
            )
            wp.launch(
                _ipm_solution_accumulate_f64,
                dim=(self.capacity, self.dof_count),
                inputs=[
                    self._wp_pcg_preconditioned,
                    self._wp_frame_segment,
                    self._wp_ipm_certify,
                    frame_count,
                ],
                outputs=[self._wp_ipm_solution_f64],
                device=self.optimizer.device,
            )
            wp.launch(
                _ipm_solution_canonicalize_f32,
                dim=(self.capacity, self.dof_count),
                inputs=[
                    self._wp_ipm_solution_f64,
                    self._wp_frame_segment,
                    self._wp_ipm_certify,
                    frame_count,
                ],
                device=self.optimizer.device,
            )
            self._ipm_physical_residual_update(self._wp_ipm_certify, frame_count, segment_count)
            wp.launch(
                _ipm_physical_convergence_update,
                dim=segment_count,
                inputs=[
                    self._wp_ipm_residual_dot_f64,
                    self._wp_ipm_initial_norm_f64,
                    self.krylov_relative_tolerance,
                    float(torch.finfo(torch.float32).eps),
                ],
                outputs=[self._wp_ipm_certify, self._wp_minres_failed],
                device=self.optimizer.device,
            )
            wp.launch(
                _ipm_certification_resolve,
                dim=segment_count,
                inputs=[self._wp_ipm_certify, self._wp_ipm_recursive_converged, segment_count],
                outputs=[self._wp_kkt_enabled, self._wp_minres_enabled],
                device=self.optimizer.device,
            )
            wp.launch(
                _pcg_reliable_restart,
                dim=(self.capacity, self.dof_count),
                inputs=[
                    self._wp_ipm_operator_f64,
                    self._wp_frame_segment,
                    self._wp_ipm_certify,
                    frame_count,
                ],
                outputs=[
                    self._wp_delta,
                    self._wp_pcg_residual,
                    self._wp_pcg_preconditioned,
                    self._wp_pcg_direction,
                ],
                device=self.optimizer.device,
            )
            self._ipm_additive_precondition(
                self._pcg_residual,
                self._wp_pcg_residual,
                self._wp_minres_enabled,
                frame_count,
                segment_count,
            )
            self._dot(
                self._pcg_residual,
                self._pcg_preconditioned,
                self._residual_dot,
                frame_count,
                segment_count,
            )
            wp.launch(
                _pcg_reliable_restart_state,
                dim=segment_count,
                inputs=[self._wp_residual_dot, self._wp_ipm_certify, segment_count],
                outputs=[self._wp_minres_state],
                device=self.optimizer.device,
            )
            wp.launch(
                _pcg_direction_restart,
                dim=(self.capacity, self.dof_count),
                inputs=[
                    self._wp_pcg_preconditioned,
                    self._wp_frame_segment,
                    self._wp_ipm_certify,
                    frame_count,
                ],
                outputs=[self._wp_pcg_direction],
                device=self.optimizer.device,
            )

        wp.launch(
            _ipm_temporal_backward_f32,
            dim=(self.max_segments, self.dof_count),
            inputs=[
                self._wp_delta,
                self._wp_ipm_temporal_factor,
                self._wp_segment_offsets,
                self._wp_kkt_enabled,
                segment_count,
            ],
            outputs=[self._wp_pcg_preconditioned],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_singleton_backward_f32,
            dim=self.max_segments,
            inputs=[
                self._wp_delta,
                self._wp_ipm_singleton_factor,
                self._wp_segment_offsets,
                self._wp_kkt_enabled,
                segment_count,
            ],
            outputs=[self._wp_pcg_preconditioned],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_solution_accumulate_f64,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_pcg_preconditioned,
                self._wp_frame_segment,
                self._wp_kkt_enabled,
                frame_count,
            ],
            outputs=[self._wp_ipm_solution_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_solution_canonicalize_f32,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_ipm_solution_f64,
                self._wp_frame_segment,
                self._wp_kkt_enabled,
                frame_count,
            ],
            device=self.optimizer.device,
        )
        self._ipm_physical_residual_update(self._wp_kkt_enabled, frame_count, segment_count)
        self._minres_enabled[:segment_count].copy_(self._kkt_enabled[:segment_count])
        wp.launch(
            _ipm_physical_convergence_update,
            dim=segment_count,
            inputs=[
                self._wp_ipm_residual_dot_f64,
                self._wp_ipm_initial_norm_f64,
                self.krylov_relative_tolerance,
                float(torch.finfo(torch.float32).eps),
            ],
            outputs=[self._wp_minres_enabled, self._wp_minres_failed],
            device=self.optimizer.device,
        )

        wp.launch(
            _ipm_solution_copy_f32,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_ipm_solution_f64,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_delta],
            device=self.optimizer.device,
        )
        self._ipm_solution_f64[:frame_count].zero_()
        wp.launch(
            _ipm_solution_accumulate_f64,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_delta,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_ipm_solution_f64],
            device=self.optimizer.device,
        )
        self._ipm_physical_residual_update(self._wp_ipm_enabled, frame_count, segment_count)
        self._minres_enabled[:segment_count].copy_(self._ipm_enabled[:segment_count])
        wp.launch(
            _ipm_physical_convergence_update,
            dim=segment_count,
            inputs=[
                self._wp_ipm_residual_dot_f64,
                self._wp_ipm_initial_norm_f64,
                self.krylov_relative_tolerance,
                float(torch.finfo(torch.float32).eps),
            ],
            outputs=[self._wp_minres_enabled, self._wp_minres_failed],
            device=self.optimizer.device,
        )

    def _ipm_constraints_apply(
        self,
        primal: torch.Tensor,
        primal_wp: wp.array,
        output_wp: wp.array,
        frame_count: int,
    ) -> None:
        torch.bmm(
            self._jacobian[:frame_count],
            primal[:frame_count].unsqueeze(-1),
            out=self._feature[:frame_count].unsqueeze(-1),
        )
        wp.launch(
            _ipm_constraints_apply,
            dim=(self.capacity, self.constraint_width),
            inputs=[
                self._wp_feature,
                primal_wp,
                self._wp_coordinate_dof_indices,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_ipm_enabled,
                frame_count,
                self._wp_ipm_constraint_scale,
            ],
            outputs=[output_wp],
            device=self.optimizer.device,
        )

    def _ipm_constraints_apply_values_f64(self, values_wp: wp.array, output_wp: wp.array, frame_count: int) -> None:
        """Apply the Phase-II inequality Jacobian to float64 coordinate values."""
        wp.launch(
            _ipm_feature_f64,
            dim=(self.capacity, self.residual_count),
            inputs=[
                self._wp_jacobian,
                values_wp,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_ipm_feature_f64],
            device=self.optimizer.device,
        )
        wp.launch(
            _ipm_constraints_apply_f64,
            dim=(self.capacity, self.inequality_width),
            inputs=[
                self._wp_ipm_feature_f64,
                values_wp,
                self._wp_coordinate_dof_indices,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_ipm_enabled,
                frame_count,
                self._wp_ipm_constraint_scale,
            ],
            outputs=[output_wp],
            device=self.optimizer.device,
        )

    def _ipm_primal_constraints_apply_f64(self, frame_count: int) -> None:
        """Evaluate the float64 Phase-II primal in the coherent float32 row representation."""
        self._ipm_constraints_apply_values_f64(
            self._wp_ipm_primal_f64, self._wp_ipm_augmented_row_basis_f64, frame_count
        )
        wp.launch(
            _ipm_rows_narrow_f32,
            dim=(self.capacity, self.inequality_width),
            inputs=[
                self._wp_ipm_augmented_row_basis_f64,
                self._wp_frame_segment,
                self._wp_ipm_enabled,
                frame_count,
            ],
            outputs=[self._wp_ipm_constraint_work],
            device=self.optimizer.device,
        )

    def _ipm_constraints_transpose(self, values_wp: wp.array, frame_count: int) -> None:
        wp.launch(
            _ipm_constraints_transpose,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_jacobian,
                values_wp,
                self._wp_coordinate_dof_indices,
                self._coordinate_bound_count,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_ipm_enabled,
                frame_count,
                self._wp_ipm_constraint_scale,
            ],
            outputs=[self._wp_ipm_transpose],
            device=self.optimizer.device,
        )

    def _ipm_k_apply(self, values: torch.Tensor, values_wp: wp.array, frame_count: int) -> None:
        self._normal_apply(self._jacobian[:frame_count], self._wp_jacobian, values, values_wp, frame_count)
        self._ipm_constraints_apply(values, values_wp, self._wp_ipm_complementarity_residual, frame_count)
        self._ipm_complementarity_residual[:frame_count].mul_(self._ipm_weights[:frame_count])
        self._ipm_constraints_transpose(self._wp_ipm_complementarity_residual, frame_count)
        self._pcg_operator_direction[:frame_count].add_(self._ipm_transpose[:frame_count])

    def _solve_minres(self, frame_count: int, segment_count: int) -> None:
        solve_enabled = self._wp_kkt_enabled
        machine_epsilon = float(torch.finfo(torch.float32).eps) ** 2
        wp.launch(
            _minres_vectors_initialize,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_right_hand_side,
                self._wp_frame_segment,
                solve_enabled,
                frame_count,
            ],
            outputs=[
                self._wp_delta,
                self._wp_right_hand_side,
                self._wp_pcg_residual,
                self._wp_pcg_operator_direction,
                self._wp_delta_candidate,
                self._wp_minres_direction,
            ],
            device=self.optimizer.device,
        )
        wp.launch(
            _minres_vectors_initialize,
            dim=(self.capacity, self.active_width),
            inputs=[
                self._wp_dual_r1,
                self._wp_frame_segment,
                solve_enabled,
                frame_count,
            ],
            outputs=[
                self._wp_dual_solution,
                self._wp_dual_r1,
                self._wp_dual_r2,
                self._wp_dual_direction_older,
                self._wp_dual_direction_old,
                self._wp_dual_direction,
            ],
            device=self.optimizer.device,
        )
        wp.launch(
            _precondition_primal,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_right_hand_side,
                self._wp_normal_diagonal,
                self._wp_frame_segment,
                solve_enabled,
                frame_count,
            ],
            outputs=[self._wp_pcg_preconditioned],
            device=self.optimizer.device,
        )
        self._dual_z[:frame_count].copy_(self._dual_r1[:frame_count])
        self._combined_dot(
            self._right_hand_side,
            self._pcg_preconditioned,
            self._dual_r1,
            self._dual_z,
            frame_count,
            segment_count,
        )
        wp.launch(
            _minres_initialize,
            dim=segment_count,
            inputs=[
                self._wp_residual_dot,
                self._wp_residual_dot_next,
                solve_enabled,
                machine_epsilon,
            ],
            outputs=[self._wp_minres_state, self._wp_minres_enabled, self._wp_minres_failed],
            device=self.optimizer.device,
        )

        primal_r1 = (self._right_hand_side, self._wp_right_hand_side)
        primal_r2 = (self._pcg_residual, self._wp_pcg_residual)
        primal_z = (self._pcg_preconditioned, self._wp_pcg_preconditioned)
        dual_r1 = (self._dual_r1, self._wp_dual_r1)
        dual_r2 = (self._dual_r2, self._wp_dual_r2)
        dual_z = (self._dual_z, self._wp_dual_z)
        primal_w_older = (self._pcg_operator_direction, self._wp_pcg_operator_direction)
        primal_w_old = (self._delta_candidate, self._wp_delta_candidate)
        primal_w = (self._minres_direction, self._wp_minres_direction)
        dual_w_older = (self._dual_direction_older, self._wp_dual_direction_older)
        dual_w_old = (self._dual_direction_old, self._wp_dual_direction_old)
        dual_w = (self._dual_direction, self._wp_dual_direction)

        for iteration in range(self.krylov_max_iterations):
            wp.launch(
                _minres_basis,
                dim=(self.capacity, self.dof_count),
                inputs=[
                    primal_z[1],
                    self._wp_frame_segment,
                    self._wp_minres_state,
                    self._wp_minres_enabled,
                    frame_count,
                ],
                outputs=[self._wp_pcg_direction],
                device=self.optimizer.device,
            )
            wp.launch(
                _minres_basis,
                dim=(self.capacity, self.active_width),
                inputs=[
                    dual_z[1],
                    self._wp_frame_segment,
                    self._wp_minres_state,
                    self._wp_minres_enabled,
                    frame_count,
                ],
                outputs=[self._wp_dual_basis],
                device=self.optimizer.device,
            )
            self._kkt_apply(
                self._pcg_direction,
                self._wp_pcg_direction,
                self._dual_basis,
                self._wp_dual_basis,
                primal_z[0],
                primal_z[1],
                dual_z[0],
                dual_z[1],
                self._wp_minres_enabled,
                frame_count,
            )
            for values, previous, width in (
                (primal_z[1], primal_r1[1], self.dof_count),
                (dual_z[1], dual_r1[1], self.active_width),
            ):
                wp.launch(
                    _minres_previous_lanczos,
                    dim=(self.capacity, width),
                    inputs=[
                        values,
                        previous,
                        self._wp_frame_segment,
                        self._wp_minres_state,
                        iteration,
                        self._wp_minres_enabled,
                        frame_count,
                    ],
                    device=self.optimizer.device,
                )
            self._combined_dot(
                self._pcg_direction,
                primal_z[0],
                self._dual_basis,
                dual_z[0],
                frame_count,
                segment_count,
            )
            wp.launch(
                _minres_store_alfa,
                dim=segment_count,
                inputs=[
                    self._wp_residual_dot,
                    self._wp_residual_dot_next,
                    self._wp_minres_enabled,
                ],
                outputs=[self._wp_minres_state],
                device=self.optimizer.device,
            )
            for values, current, width in (
                (primal_z[1], primal_r2[1], self.dof_count),
                (dual_z[1], dual_r2[1], self.active_width),
            ):
                wp.launch(
                    _minres_current_lanczos,
                    dim=(self.capacity, width),
                    inputs=[
                        values,
                        current,
                        self._wp_frame_segment,
                        self._wp_minres_state,
                        self._wp_minres_enabled,
                        frame_count,
                    ],
                    device=self.optimizer.device,
                )

            primal_r1, primal_r2, primal_z = primal_r2, primal_z, primal_r1
            dual_r1, dual_r2, dual_z = dual_r2, dual_z, dual_r1
            wp.launch(
                _precondition_primal,
                dim=(self.capacity, self.dof_count),
                inputs=[
                    primal_r2[1],
                    self._wp_normal_diagonal,
                    self._wp_frame_segment,
                    self._wp_minres_enabled,
                    frame_count,
                ],
                outputs=[primal_z[1]],
                device=self.optimizer.device,
            )
            dual_z[0][:frame_count].copy_(dual_r2[0][:frame_count])
            self._combined_dot(
                primal_r2[0],
                primal_z[0],
                dual_r2[0],
                dual_z[0],
                frame_count,
                segment_count,
            )
            wp.launch(
                _minres_recurrence,
                dim=segment_count,
                inputs=[
                    self._wp_residual_dot,
                    self._wp_residual_dot_next,
                    self.krylov_relative_tolerance,
                    machine_epsilon,
                ],
                outputs=[self._wp_minres_state, self._wp_minres_enabled, self._wp_minres_failed],
                device=self.optimizer.device,
            )
            wp.launch(
                _minres_solution_update,
                dim=(self.capacity, self.dof_count),
                inputs=[
                    self._wp_pcg_direction,
                    primal_w_older[1],
                    primal_w_old[1],
                    self._wp_frame_segment,
                    self._wp_minres_state,
                    self._wp_minres_enabled,
                    frame_count,
                ],
                outputs=[self._wp_delta, primal_w[1]],
                device=self.optimizer.device,
            )
            wp.launch(
                _minres_solution_update,
                dim=(self.capacity, self.active_width),
                inputs=[
                    self._wp_dual_basis,
                    dual_w_older[1],
                    dual_w_old[1],
                    self._wp_frame_segment,
                    self._wp_minres_state,
                    self._wp_minres_enabled,
                    frame_count,
                ],
                outputs=[self._wp_dual_solution, dual_w[1]],
                device=self.optimizer.device,
            )
            primal_w_older, primal_w_old, primal_w = primal_w_old, primal_w, primal_w_older
            dual_w_older, dual_w_old, dual_w = dual_w_old, dual_w, dual_w_older
            wp.launch(
                _minres_finalize_iteration,
                dim=segment_count,
                inputs=[self._wp_minres_state],
                outputs=[self._wp_minres_enabled],
                device=self.optimizer.device,
            )
            if self._krylov_all_converged(iteration, segment_count):
                break

    def _combined_dot(
        self,
        primal_a: torch.Tensor,
        primal_b: torch.Tensor,
        dual_a: torch.Tensor,
        dual_b: torch.Tensor,
        frame_count: int,
        segment_count: int,
    ) -> None:
        self._dot(primal_a, primal_b, self._residual_dot, frame_count, segment_count)
        self._dot(dual_a, dual_b, self._residual_dot_next, frame_count, segment_count)

    def _kkt_apply(
        self,
        primal: torch.Tensor,
        primal_wp: wp.array,
        dual: torch.Tensor,
        dual_wp: wp.array,
        primal_output: torch.Tensor,
        primal_output_wp: wp.array,
        dual_output: torch.Tensor,
        dual_output_wp: wp.array,
        enabled_wp: wp.array,
        frame_count: int,
    ) -> None:
        torch.bmm(
            self._jacobian[:frame_count],
            primal[:frame_count].unsqueeze(-1),
            out=self._feature[:frame_count].unsqueeze(-1),
        )
        wp.launch(
            _qp_rows_operator_dual,
            dim=(self.capacity, self.active_width),
            inputs=[
                self._wp_feature,
                primal_wp,
                self._wp_coordinate_dof_indices,
                self._wp_coordinate_lower,
                self._wp_coordinate_upper,
                self._coordinate_bound_count,
                self.dof_count,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                enabled_wp,
                frame_count,
                self._wp_active_row_codes,
                self._wp_active_row_scales,
                self._wp_active_equality_count,
                dual_wp,
            ],
            outputs=[dual_output_wp],
            device=self.optimizer.device,
        )
        wp.launch(
            _precision_apply,
            dim=(self.capacity, self.residual_count),
            inputs=[
                self._wp_feature,
                self._wp_objective_weights,
                self._wp_temporal_weights,
                self._wp_residual_activity,
                self._wp_activity_group_by_residual,
                self._wp_first_difference_group_by_residual,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_step_seconds,
                self._wp_segment_active,
                frame_count,
            ],
            outputs=[self._wp_precision_feature],
            device=self.optimizer.device,
        )
        torch.bmm(
            self._jacobian[:frame_count].transpose(1, 2),
            self._precision_feature[:frame_count].unsqueeze(-1),
            out=primal_output[:frame_count].unsqueeze(-1),
        )
        wp.launch(
            _segment_scaled_add,
            dim=(self.capacity, self.dof_count),
            inputs=[primal_wp, self._wp_segment_damping, self._wp_frame_segment, frame_count],
            outputs=[primal_output_wp],
            device=self.optimizer.device,
        )
        wp.launch(
            _qp_rows_transpose,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_jacobian,
                self._wp_coordinate_dof_indices,
                self._wp_coordinate_lower,
                self._wp_coordinate_upper,
                self._coordinate_bound_count,
                self.dof_count,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                enabled_wp,
                frame_count,
                self._wp_active_row_codes,
                self._wp_active_row_scales,
                self._wp_active_equality_count,
                dual_wp,
            ],
            outputs=[self._wp_active_row_transpose],
            device=self.optimizer.device,
        )
        primal_output[:frame_count].add_(self._active_row_transpose[:frame_count])
        if self._ipm_operator_active:
            self._ipm_constraints_apply(primal, primal_wp, self._wp_ipm_complementarity_residual, frame_count)
            self._ipm_complementarity_residual[:frame_count].mul_(self._ipm_weights[:frame_count])
            self._ipm_constraints_transpose(self._wp_ipm_complementarity_residual, frame_count)
            primal_output[:frame_count].add_(self._ipm_transpose[:frame_count])

    def _projected_mask_binding_dofs(self, frame_count: int) -> None:
        """Restrict all projected-PCG work vectors to the current box face."""
        wp.launch(
            _coordinate_bounds_mask_binding_dofs,
            dim=(self.capacity, self._coordinate_bound_count),
            inputs=[
                self._wp_joint_q,
                self._wp_right_hand_side,
                self._wp_coordinate_indices,
                self._wp_coordinate_dof_indices,
                self._wp_coordinate_lower,
                self._wp_coordinate_upper,
                self._coordinate_bound_count,
                frame_count,
            ],
            outputs=[
                self._wp_pcg_residual,
                self._wp_pcg_preconditioned,
                self._wp_pcg_direction,
                self._wp_pcg_operator_direction,
            ],
            device=self.optimizer.device,
        )

    def _projected_temporal_precondition(self, enabled: wp.array, segment_count: int) -> None:
        """Apply the factored same-DOF temporal preconditioner to the PCG residual."""
        wp.launch(
            _pcg_temporal_forward_f32,
            dim=(self.max_segments, self.dof_count),
            inputs=[
                self._wp_pcg_residual,
                self._wp_normal_diagonal,
                self._wp_ipm_temporal_factor,
                self._wp_segment_offsets,
                enabled,
                self._wp_minres_failed,
                segment_count,
            ],
            outputs=[self._wp_pcg_operator_direction],
            device=self.optimizer.device,
        )
        wp.launch(
            _pcg_temporal_backward_f32,
            dim=(self.max_segments, self.dof_count),
            inputs=[
                self._wp_pcg_operator_direction,
                self._wp_normal_diagonal,
                self._wp_ipm_temporal_factor,
                self._wp_segment_offsets,
                enabled,
                self._wp_minres_failed,
                segment_count,
            ],
            outputs=[self._wp_pcg_preconditioned],
            device=self.optimizer.device,
        )

    def _solve_linearized(
        self,
        residuals: torch.Tensor,
        residuals_wp: wp.array,
        jacobian: torch.Tensor,
        jacobian_wp: wp.array,
        frame_count: int,
        segment_count: int,
        convergence_accumulator: wp.array,
        *,
        projected: bool = False,
    ) -> None:
        wp.launch(
            _precision_apply,
            dim=(self.capacity, self.residual_count),
            inputs=[
                residuals_wp,
                self._wp_base_weights,
                self._wp_temporal_weights,
                self._wp_residual_activity,
                self._wp_activity_group_by_residual,
                self._wp_first_difference_group_by_residual,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_step_seconds,
                self._wp_segment_active,
                frame_count,
            ],
            outputs=[self._wp_precision_feature],
            device=self.optimizer.device,
        )
        torch.bmm(
            jacobian[:frame_count].transpose(1, 2),
            self._precision_feature[:frame_count].unsqueeze(-1),
            out=self._right_hand_side[:frame_count].unsqueeze(-1),
        )
        self._right_hand_side[:frame_count].neg_()
        wp.launch(
            _precision_diagonal,
            dim=(self.capacity, self.residual_count),
            inputs=[
                self._wp_base_weights,
                self._wp_temporal_weights,
                self._wp_residual_activity,
                self._wp_activity_group_by_residual,
                self._wp_first_difference_group_by_residual,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_step_seconds,
                self._wp_segment_active,
                frame_count,
            ],
            outputs=[self._wp_precision_diagonal],
            device=self.optimizer.device,
        )
        wp.launch(
            _normal_diagonal,
            dim=(self.capacity, self.dof_count),
            inputs=[
                jacobian_wp,
                self._wp_precision_diagonal,
                self._wp_segment_damping,
                self._wp_frame_segment,
                self._wp_segment_active,
                frame_count,
            ],
            outputs=[self._wp_normal_diagonal],
            device=self.optimizer.device,
        )
        if projected:
            wp.launch(
                _normal_temporal_band_build,
                dim=(self.capacity, self.dof_count, 4),
                inputs=[
                    jacobian_wp,
                    self._wp_normal_diagonal,
                    self._wp_temporal_weights,
                    self._wp_residual_activity,
                    self._wp_activity_group_by_residual,
                    self._wp_first_difference_group_by_residual,
                    self._wp_step_seconds,
                    self._wp_frame_segment,
                    self._wp_segment_offsets,
                    self._wp_segment_active,
                    frame_count,
                ],
                outputs=[self._wp_ipm_temporal_factor],
                device=self.optimizer.device,
            )
            self._minres_failed[:segment_count].zero_()
            wp.launch(
                _ipm_temporal_band_factor,
                dim=(self.max_segments, self.dof_count),
                inputs=[
                    self._wp_ipm_temporal_factor,
                    self._wp_segment_offsets,
                    self._wp_segment_active,
                    segment_count,
                ],
                outputs=[self._wp_minres_failed],
                device=self.optimizer.device,
            )
        wp.launch(
            _pcg_initialize,
            dim=(self.capacity, self.dof_count),
            inputs=[
                self._wp_right_hand_side,
                self._wp_normal_diagonal,
                self._wp_frame_segment,
                self._wp_segment_active,
                frame_count,
            ],
            outputs=[
                self._wp_delta,
                self._wp_pcg_residual,
                self._wp_pcg_preconditioned,
                self._wp_pcg_direction,
            ],
            device=self.optimizer.device,
        )
        if projected:
            self._projected_mask_binding_dofs(frame_count)
            self._projected_temporal_precondition(self._wp_segment_active, segment_count)
            self._projected_mask_binding_dofs(frame_count)
            self._pcg_direction[:frame_count].copy_(self._pcg_preconditioned[:frame_count])
        self._dot(
            self._pcg_residual,
            self._pcg_preconditioned,
            self._residual_dot,
            frame_count,
            segment_count,
        )
        machine_epsilon = float(torch.finfo(torch.float32).eps)
        wp.launch(
            _pcg_convergence_initialize,
            dim=segment_count,
            inputs=[
                self._wp_residual_dot,
                self._wp_segment_active,
                segment_count,
                machine_epsilon,
            ],
            outputs=[self._wp_minres_state, self._wp_minres_enabled],
            device=self.optimizer.device,
        )
        for iteration in range(self.krylov_max_iterations):
            self._normal_apply(jacobian, jacobian_wp, self._pcg_direction, self._wp_pcg_direction, frame_count)
            if projected:
                self._projected_mask_binding_dofs(frame_count)
            self._dot(
                self._pcg_direction,
                self._pcg_operator_direction,
                self._operator_dot,
                frame_count,
                segment_count,
            )
            wp.launch(
                _pcg_alpha_update,
                dim=(self.capacity, self.dof_count),
                inputs=[
                    self._wp_pcg_direction,
                    self._wp_pcg_operator_direction,
                    self._wp_residual_dot,
                    self._wp_operator_dot,
                    self._wp_frame_segment,
                    self._wp_minres_enabled,
                    frame_count,
                ],
                outputs=[self._wp_delta, self._wp_pcg_residual],
                device=self.optimizer.device,
            )
            if projected:
                self._projected_temporal_precondition(self._wp_minres_enabled, segment_count)
                self._projected_mask_binding_dofs(frame_count)
            else:
                wp.launch(
                    _pcg_precondition,
                    dim=(self.capacity, self.dof_count),
                    inputs=[
                        self._wp_pcg_residual,
                        self._wp_normal_diagonal,
                        self._wp_frame_segment,
                        self._wp_minres_enabled,
                        frame_count,
                    ],
                    outputs=[self._wp_pcg_preconditioned],
                    device=self.optimizer.device,
                )
            self._dot(
                self._pcg_residual,
                self._pcg_preconditioned,
                self._residual_dot_next,
                frame_count,
                segment_count,
            )
            wp.launch(
                _pcg_convergence_update,
                dim=segment_count,
                inputs=[
                    self._wp_residual_dot_next,
                    self._wp_minres_state,
                    self.krylov_relative_tolerance,
                    machine_epsilon,
                ],
                outputs=[self._wp_minres_enabled],
                device=self.optimizer.device,
            )
            wp.launch(
                _pcg_beta_update,
                dim=(self.capacity, self.dof_count),
                inputs=[
                    self._wp_pcg_preconditioned,
                    self._wp_residual_dot,
                    self._wp_residual_dot_next,
                    self._wp_frame_segment,
                    self._wp_minres_enabled,
                    frame_count,
                ],
                outputs=[self._wp_pcg_direction],
                device=self.optimizer.device,
            )
            self._residual_dot[:segment_count].copy_(self._residual_dot_next[:segment_count])
            if self._krylov_all_converged(iteration, segment_count):
                break
        if projected and bool(torch.any(self._minres_enabled[:segment_count])):
            wp.launch(
                _projected_fallback_initialize,
                dim=(self.capacity, self.dof_count),
                inputs=[
                    self._wp_right_hand_side,
                    self._wp_normal_diagonal,
                    self._wp_frame_segment,
                    self._wp_minres_enabled,
                    frame_count,
                ],
                outputs=[
                    self._wp_pcg_residual,
                    self._wp_pcg_preconditioned,
                    self._wp_pcg_direction,
                    self._wp_pcg_operator_direction,
                ],
                device=self.optimizer.device,
            )
            self._projected_mask_binding_dofs(frame_count)
            self._dot(
                self._pcg_residual,
                self._pcg_preconditioned,
                self._residual_dot,
                frame_count,
                segment_count,
            )
            self._normal_apply(jacobian, jacobian_wp, self._pcg_direction, self._wp_pcg_direction, frame_count)
            self._projected_mask_binding_dofs(frame_count)
            self._dot(
                self._pcg_direction,
                self._pcg_operator_direction,
                self._operator_dot,
                frame_count,
                segment_count,
            )
            wp.launch(
                _projected_direction_validate,
                dim=segment_count,
                inputs=[
                    self._wp_residual_dot,
                    self._wp_operator_dot,
                    self._wp_minres_enabled,
                    self._wp_segment_active,
                    segment_count,
                    float(torch.finfo(torch.float32).eps),
                ],
                outputs=[convergence_accumulator],
                device=self.optimizer.device,
            )
            wp.launch(
                _projected_fallback_apply,
                dim=(self.capacity, self.dof_count),
                inputs=[
                    self._wp_pcg_direction,
                    self._wp_residual_dot,
                    self._wp_operator_dot,
                    self._wp_frame_segment,
                    self._wp_minres_enabled,
                    convergence_accumulator,
                    frame_count,
                ],
                outputs=[self._wp_delta],
                device=self.optimizer.device,
            )
        elif not projected:
            wp.launch(
                _krylov_convergence_accumulate,
                dim=segment_count,
                inputs=[self._wp_segment_active, self._wp_minres_enabled, segment_count],
                outputs=[convergence_accumulator],
                device=self.optimizer.device,
            )

    def _krylov_all_converged(self, iteration: int, segment_count: int) -> bool:
        """Return whether a bounded non-capturing inner-solve check can stop."""
        interval = 1 if self.device.type == "cpu" else self.krylov_check_interval
        if (iteration + 1) % interval != 0:
            return False
        if self.device.type == "cuda" and self.optimizer.device.is_capturing:
            return False
        return not bool(torch.any(self._minres_enabled[:segment_count]))

    def _normal_apply(
        self,
        jacobian: torch.Tensor,
        jacobian_wp: wp.array,
        values: torch.Tensor,
        values_wp: wp.array,
        frame_count: int,
    ) -> None:
        torch.bmm(
            jacobian,
            values[:frame_count].unsqueeze(-1),
            out=self._feature[:frame_count].unsqueeze(-1),
        )
        wp.launch(
            _precision_apply,
            dim=(self.capacity, self.residual_count),
            inputs=[
                self._wp_feature,
                self._wp_base_weights,
                self._wp_temporal_weights,
                self._wp_residual_activity,
                self._wp_activity_group_by_residual,
                self._wp_first_difference_group_by_residual,
                self._wp_frame_segment,
                self._wp_segment_offsets,
                self._wp_step_seconds,
                self._wp_segment_active,
                frame_count,
            ],
            outputs=[self._wp_precision_feature],
            device=self.optimizer.device,
        )
        torch.bmm(
            jacobian.transpose(1, 2),
            self._precision_feature[:frame_count].unsqueeze(-1),
            out=self._pcg_operator_direction[:frame_count].unsqueeze(-1),
        )
        wp.launch(
            _segment_scaled_add,
            dim=(self.capacity, self.dof_count),
            inputs=[values_wp, self._wp_segment_damping, self._wp_frame_segment, frame_count],
            outputs=[self._wp_pcg_operator_direction],
            device=self.optimizer.device,
        )

    def _dot(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        output: torch.Tensor,
        frame_count: int,
        segment_count: int,
    ) -> None:
        """Write deterministic allocation-free dot products for each independent segment."""
        torch.bmm(
            a[:frame_count].unsqueeze(1),
            b[:frame_count].unsqueeze(-1),
            out=self._dot_frame[:frame_count].view(-1, 1, 1),
        )
        if output is self._residual_dot:
            output_wp = self._wp_residual_dot
        elif output is self._residual_dot_next:
            output_wp = self._wp_residual_dot_next
        elif output is self._operator_dot:
            output_wp = self._wp_operator_dot
        elif output is self._objective_cost:
            output_wp = self._wp_objective_cost
        elif output is self._candidate_cost:
            output_wp = self._wp_candidate_cost
        else:
            raise ValueError("Unsupported segmented reduction output buffer.")
        wp.launch(
            _dot_segments,
            dim=self.max_segments,
            inputs=[
                self._wp_dot_frame,
                self._wp_segment_offsets,
                segment_count,
                self._wp_segment_active,
            ],
            outputs=[output_wp],
            device=self.optimizer.device,
        )

    def _check_tensor(
        self,
        tensor: torch.Tensor,
        shape: tuple[int, ...] | torch.Size,
        dtype: torch.dtype,
        name: str,
    ) -> None:
        if tensor.shape != shape or tensor.dtype != dtype or tensor.device != self.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous {dtype} on {self.device} with shape {tuple(shape)}.")
