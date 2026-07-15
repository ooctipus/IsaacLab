# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for whole-segment matrix-free Newton trajectory refinement."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import newton
import newton.ik as ik
import pytest
import torch
import warp as wp

from isaaclab_tasks.core.multi_task.kinematics import trajectory as trajectory_module
from isaaclab_tasks.core.multi_task.kinematics.ik_objectives.joint_pin import IKObjectiveJointPin
from isaaclab_tasks.core.multi_task.kinematics.impl.trajectory_warp import (
    _constraint_line_search_decide,
    _constraint_violation_max,
    _coordinate_bound_violation_max,
    _ipm_primal_feasibility_violation_max,
    _ipm_solve_status,
    _ipm_velocity_bound_violation_max,
    _line_search_decide,
    _line_search_stationarity_initialize,
    _minres_basis,
    _minres_initialize,
    _minres_recurrence,
    _pcg_convergence_initialize,
    _phase_one_constraints_apply,
    _phase_one_constraints_initialize,
    _phase_one_finalize,
    _phase_one_original_violation_max,
    _phase_one_witness_select,
    _restoration_candidate_merit_max,
    _restoration_current_merit_max,
    _second_order_correction_decide,
    _second_order_correction_request,
    _segment_convergence_update,
    _segment_step_max,
)
from isaaclab_tasks.core.multi_task.kinematics.trajectory import (
    IKTrajectorySolver,
    plan_trajectory_memory,
)


def test_trajectory_warp_backend_owns_all_device_kernels() -> None:
    """The wrapper owns Torch storage while one pure-Warp module owns device code."""
    kinematics_root = Path(__file__).parents[1] / "kinematics"
    wrapper_text = (kinematics_root / "trajectory.py").read_text(encoding="utf-8")
    wrapper_tree = ast.parse(wrapper_text)
    backend_text = (kinematics_root / "impl" / "trajectory_warp.py").read_text(encoding="utf-8")
    backend_tree = ast.parse(backend_text)

    backend_imports = {
        alias.name.partition(".")[0]
        for node in backend_tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    backend_imports.update(
        node.module.partition(".")[0]
        for node in backend_tree.body
        if isinstance(node, ast.ImportFrom) and node.module not in (None, "__future__")
    )
    assert backend_imports == {"warp"}
    assert "_constraint_projection" not in backend_text
    assert "_velocity_project" not in backend_text
    assert "_qp_rows_add" not in backend_text
    assert "_qp_rows_remove_negative" not in backend_text
    assert "_qp_rows_violation_scan" not in backend_text
    assert "_phase_one_arrowhead_scalar_recover" not in backend_text
    assert "_phase_one_arrowhead_pair_dot" in backend_text
    assert "_phase_one_arrowhead_pair_reduce" in backend_text
    assert "_phase_one_arrowhead_scalar_solve" in backend_text
    assert "active_iteration_limit" not in ast.unparse(wrapper_tree)
    assert "wp.types.matrix" not in backend_text
    assert "wp.types.vector" not in backend_text
    assert not any(isinstance(node, ast.ClassDef) for node in backend_tree.body)
    assert (kinematics_root / "impl" / "__init__.py").is_file()
    assert not any(
        isinstance(decorator, (ast.Attribute, ast.Call)) and ast.unparse(decorator).startswith(("wp.func", "wp.kernel"))
        for node in ast.walk(wrapper_tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        for decorator in node.decorator_list
    )
    assert "segment_damping: torch.Tensor | None = None" in wrapper_text
    assert wrapper_text.count("self.damping") == 2
    assert "torch.all(torch.isfinite(segment_damping)" not in wrapper_text
    assert "self._wp_segment_damping" in wrapper_text
    assert "damping: float" not in backend_text
    assert "regularization: float" not in backend_text
    assert "segment_damping[" in backend_text
    assert "_segment_damping_validate" in backend_text


class _LinearOptimizer:
    """Minimal public Newton-optimizer contract with exact linear residuals."""

    def __init__(self, jacobian: torch.Tensor, target: torch.Tensor) -> None:
        self.device = wp.get_device(str(jacobian.device))
        self.n_batch, self.n_residuals, self.n_dofs = jacobian.shape
        self.n_coords = self.n_dofs
        self._model_jacobian = jacobian.contiguous()
        self.jacobian = self._model_jacobian.clone()
        self.target = target.contiguous()
        self.residuals = torch.empty_like(target)
        self._wp_jacobian = wp.from_torch(self.jacobian)
        self._wp_residuals = wp.from_torch(self.residuals)

    def compute_residuals(self, joint_q, residuals=None):
        active = joint_q.shape[0]
        joint_q_torch = wp.to_torch(joint_q)
        torch.bmm(
            self._model_jacobian[:active],
            joint_q_torch.unsqueeze(-1),
            out=self.residuals[:active].unsqueeze(-1),
        )
        self.residuals[:active].sub_(self.target[:active])
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
            return residuals
        return self._wp_residuals[:active]

    def linearize(self, joint_q, residuals=None, jacobian=None):
        active = joint_q.shape[0]
        joint_q_torch = wp.to_torch(joint_q)
        self.jacobian.copy_(self._model_jacobian)
        torch.bmm(
            self._model_jacobian[:active],
            joint_q_torch.unsqueeze(-1),
            out=self.residuals[:active].unsqueeze(-1),
        )
        self.residuals[:active].sub_(self.target[:active])
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
            residual_view = residuals
        else:
            residual_view = self._wp_residuals[:active]
        if jacobian is not None:
            wp.copy(jacobian, self._wp_jacobian[:active])
            jacobian_view = jacobian
        else:
            jacobian_view = self._wp_jacobian[:active]
        return residual_view, jacobian_view

    def integrate(self, joint_q, delta, joint_q_out, *, step_size=1.0):
        source = wp.to_torch(joint_q)
        output = wp.to_torch(joint_q_out)
        output.copy_(source)
        output.add_(wp.to_torch(delta), alpha=step_size)


class _QuadraticOptimizer(_LinearOptimizer):
    """One-dimensional nonlinear residual q squared minus target."""

    def __init__(self, capacity: int, target: float, device: torch.device) -> None:
        jacobian = torch.empty((capacity, 1, 1), dtype=torch.float32, device=device)
        targets = torch.full((capacity, 1), target, dtype=torch.float32, device=device)
        super().__init__(jacobian, targets)
        self.linearize_calls = 0
        self.residual_calls = 0

    def compute_residuals(self, joint_q, residuals=None):
        self.residual_calls += 1
        active = joint_q.shape[0]
        values = wp.to_torch(joint_q)
        self.residuals[:active, 0].copy_(values[:, 0]).square_().sub_(self.target[:active, 0])
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
            return residuals
        return self._wp_residuals[:active]

    def linearize(self, joint_q, residuals=None, jacobian=None):
        self.linearize_calls += 1
        active = joint_q.shape[0]
        values = wp.to_torch(joint_q)
        self.residuals[:active, 0].copy_(values[:, 0]).square_().sub_(self.target[:active, 0])
        self.jacobian[:active, 0, 0].copy_(values[:, 0]).mul_(2.0)
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
            residual_view = residuals
        else:
            residual_view = self._wp_residuals[:active]
        if jacobian is not None:
            wp.copy(jacobian, self._wp_jacobian[:active])
            jacobian_view = jacobian
        else:
            jacobian_view = self._wp_jacobian[:active]
        return residual_view, jacobian_view


class _NonlinearPoseEqualityOptimizer(_LinearOptimizer):
    """Independent nonlinear translation and rotation residuals with no soft objective."""

    def __init__(self, capacity: int, device: torch.device) -> None:
        jacobian = torch.zeros((capacity, 6, 6), dtype=torch.float32, device=device)
        target = torch.zeros((capacity, 6), dtype=torch.float32, device=device)
        super().__init__(jacobian, target)

    def compute_residuals(self, joint_q, residuals=None):
        active = joint_q.shape[0]
        values = wp.to_torch(joint_q)[:active]
        self.residuals[:active, :3].copy_(values[:, :3]).square_().sub_(1.0)
        self.residuals[:active, 3:].copy_(torch.sin(values[:, 3:])).sub_(math.sin(0.5))
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
            return residuals
        return self._wp_residuals[:active]

    def linearize(self, joint_q, residuals=None, jacobian=None):
        active = joint_q.shape[0]
        values = wp.to_torch(joint_q)[:active]
        self.residuals[:active, :3].copy_(values[:, :3]).square_().sub_(1.0)
        self.residuals[:active, 3:].copy_(torch.sin(values[:, 3:])).sub_(math.sin(0.5))
        self.jacobian[:active].zero_()
        for axis in range(3):
            self.jacobian[:active, axis, axis].copy_(2.0 * values[:, axis])
            self.jacobian[:active, axis + 3, axis + 3].copy_(torch.cos(values[:, axis + 3]))
        residual_view = self._wp_residuals[:active] if residuals is None else residuals
        jacobian_view = self._wp_jacobian[:active] if jacobian is None else jacobian
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
        if jacobian is not None:
            wp.copy(jacobian, self._wp_jacobian[:active])
        return residual_view, jacobian_view


class _HiddenNonlinearConstraintOptimizer(_LinearOptimizer):
    """Linear objective with a constraint whose gradient vanishes at the feasible point."""

    def __init__(self, capacity: int, device: torch.device) -> None:
        jacobian = torch.empty((capacity, 2, 1), dtype=torch.float32, device=device)
        target = torch.zeros((capacity, 2), dtype=torch.float32, device=device)
        super().__init__(jacobian, target)

    def compute_residuals(self, joint_q, residuals=None):
        active = joint_q.shape[0]
        values = wp.to_torch(joint_q)[:active, 0]
        self.residuals[:active, 0].copy_(values).sub_(1.0)
        self.residuals[:active, 1].copy_(values).square_()
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
            return residuals
        return self._wp_residuals[:active]

    def linearize(self, joint_q, residuals=None, jacobian=None):
        active = joint_q.shape[0]
        values = wp.to_torch(joint_q)[:active, 0]
        self.residuals[:active, 0].copy_(values).sub_(1.0)
        self.residuals[:active, 1].copy_(values).square_()
        self.jacobian[:active, 0, 0].fill_(1.0)
        self.jacobian[:active, 1, 0].copy_(values).mul_(2.0)
        residual_view = self._wp_residuals[:active] if residuals is None else residuals
        jacobian_view = self._wp_jacobian[:active] if jacobian is None else jacobian
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
        if jacobian is not None:
            wp.copy(jacobian, self._wp_jacobian[:active])
        return residual_view, jacobian_view


class _ParabolicConstraintOptimizer(_LinearOptimizer):
    """Linear objective tangent to the curved feasible set ``x + y² <= 1``."""

    def __init__(self, capacity: int, device: torch.device) -> None:
        super().__init__(
            torch.empty((capacity, 2, 2), dtype=torch.float32, device=device),
            torch.empty((capacity, 2), dtype=torch.float32, device=device),
        )
        self.linearize_calls = 0
        self.residual_calls = 0

    def compute_residuals(self, joint_q, residuals=None):
        self.residual_calls += 1
        active = joint_q.shape[0]
        values = wp.to_torch(joint_q)[:active]
        self.residuals[:active, 0].copy_(values[:, 1]).sub_(1.0)
        self.residuals[:active, 1].copy_(values[:, 1]).square_().add_(values[:, 0]).sub_(1.0)
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
            return residuals
        return self._wp_residuals[:active]

    def linearize(self, joint_q, residuals=None, jacobian=None):
        self.linearize_calls += 1
        active = joint_q.shape[0]
        values = wp.to_torch(joint_q)[:active]
        self.residuals[:active, 0].copy_(values[:, 1]).sub_(1.0)
        self.residuals[:active, 1].copy_(values[:, 1]).square_().add_(values[:, 0]).sub_(1.0)
        self.jacobian[:active].zero_()
        self.jacobian[:active, 0, 1].fill_(1.0)
        self.jacobian[:active, 1, 0].fill_(1.0)
        self.jacobian[:active, 1, 1].copy_(values[:, 1]).mul_(2.0)
        residual_view = self._wp_residuals[:active] if residuals is None else residuals
        jacobian_view = self._wp_jacobian[:active] if jacobian is None else jacobian
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
        if jacobian is not None:
            wp.copy(jacobian, self._wp_jacobian[:active])
        return residual_view, jacobian_view


class _ExponentialNormalConstraintOptimizer(_LinearOptimizer):
    """Tangential objective over ``exp(x) - 1 + y² <= 0`` with a fixed normal linearization."""

    def __init__(self, capacity: int, device: torch.device) -> None:
        super().__init__(
            torch.empty((capacity, 4, 2), dtype=torch.float32, device=device),
            torch.empty((capacity, 4), dtype=torch.float32, device=device),
        )
        self.linearize_calls = 0
        self.residual_calls = 0

    def compute_residuals(self, joint_q, residuals=None):
        self.residual_calls += 1
        active = joint_q.shape[0]
        values = wp.to_torch(joint_q)[:active]
        self.residuals[:active, 0].copy_(values[:, 1]).sub_(0.3)
        self.residuals[:active, 1].copy_(values[:, 0]).exp_().sub_(1.0).add_(values[:, 1].square())
        self.residuals[:active, 2:].zero_()
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
            return residuals
        return self._wp_residuals[:active]

    def linearize(self, joint_q, residuals=None, jacobian=None):
        self.linearize_calls += 1
        active = joint_q.shape[0]
        values = wp.to_torch(joint_q)[:active]
        self.residuals[:active, 0].copy_(values[:, 1]).sub_(0.3)
        self.residuals[:active, 1].copy_(values[:, 0]).exp_().sub_(1.0).add_(values[:, 1].square())
        self.residuals[:active, 2:].zero_()
        self.jacobian[:active].zero_()
        self.jacobian[:active, 0, 1].fill_(1.0)
        self.jacobian[:active, 1, 0].copy_(values[:, 0]).exp_()
        self.jacobian[:active, 1, 1].copy_(values[:, 1]).mul_(2.0)
        residual_view = self._wp_residuals[:active] if residuals is None else residuals
        jacobian_view = self._wp_jacobian[:active] if jacobian is None else jacobian
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
        if jacobian is not None:
            wp.copy(jacobian, self._wp_jacobian[:active])
        return residual_view, jacobian_view


class _FlatCurvatureConstraintOptimizer(_LinearOptimizer):
    """Objective tangent to ``q² <= 0.25`` where the current constraint Jacobian vanishes."""

    def __init__(self, capacity: int, device: torch.device) -> None:
        super().__init__(
            torch.empty((capacity, 2, 1), dtype=torch.float32, device=device),
            torch.empty((capacity, 2), dtype=torch.float32, device=device),
        )

    def compute_residuals(self, joint_q, residuals=None):
        active = joint_q.shape[0]
        values = wp.to_torch(joint_q)[:active, 0]
        self.residuals[:active, 0].copy_(values).sub_(1.0)
        self.residuals[:active, 1].copy_(values).square_().sub_(0.25)
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
            return residuals
        return self._wp_residuals[:active]

    def linearize(self, joint_q, residuals=None, jacobian=None):
        active = joint_q.shape[0]
        values = wp.to_torch(joint_q)[:active, 0]
        self.residuals[:active, 0].copy_(values).sub_(1.0)
        self.residuals[:active, 1].copy_(values).square_().sub_(0.25)
        self.jacobian[:active, 0, 0].fill_(1.0)
        self.jacobian[:active, 1, 0].copy_(values).mul_(2.0)
        residual_view = self._wp_residuals[:active] if residuals is None else residuals
        jacobian_view = self._wp_jacobian[:active] if jacobian is None else jacobian
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
        if jacobian is not None:
            wp.copy(jacobian, self._wp_jacobian[:active])
        return residual_view, jacobian_view


class _UpperBoundCurvatureConstraintOptimizer(_LinearOptimizer):
    """Tangential objective whose curved constraint meets a scalar upper-bound face."""

    def __init__(self, capacity: int, device: torch.device) -> None:
        super().__init__(
            torch.empty((capacity, 2, 2), dtype=torch.float32, device=device),
            torch.empty((capacity, 2), dtype=torch.float32, device=device),
        )

    def compute_residuals(self, joint_q, residuals=None):
        active = joint_q.shape[0]
        values = wp.to_torch(joint_q)[:active]
        self.residuals[:active, 0].copy_(values[:, 1]).sub_(1.0)
        self.residuals[:active, 1].copy_(values[:, 1]).square_().sub_(values[:, 0])
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
            return residuals
        return self._wp_residuals[:active]

    def linearize(self, joint_q, residuals=None, jacobian=None):
        active = joint_q.shape[0]
        values = wp.to_torch(joint_q)[:active]
        self.residuals[:active, 0].copy_(values[:, 1]).sub_(1.0)
        self.residuals[:active, 1].copy_(values[:, 1]).square_().sub_(values[:, 0])
        self.jacobian[:active].zero_()
        self.jacobian[:active, 0, 1].fill_(1.0)
        self.jacobian[:active, 1, 0].fill_(-1.0)
        self.jacobian[:active, 1, 1].copy_(values[:, 1]).mul_(2.0)
        residual_view = self._wp_residuals[:active] if residuals is None else residuals
        jacobian_view = self._wp_jacobian[:active] if jacobian is None else jacobian
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
        if jacobian is not None:
            wp.copy(jacobian, self._wp_jacobian[:active])
        return residual_view, jacobian_view

    def integrate(self, joint_q, delta, joint_q_out, *, step_size=1.0):
        super().integrate(joint_q, delta, joint_q_out, step_size=step_size)
        wp.to_torch(joint_q_out)[:, 0].add_(1.0e-4)


class _NaNTrialConstraintOptimizer(_LinearOptimizer):
    """Finite linear objective whose constraint becomes NaN away from the current state."""

    def __init__(self, capacity: int, device: torch.device) -> None:
        super().__init__(
            torch.zeros((capacity, 2, 1), dtype=torch.float32, device=device),
            torch.zeros((capacity, 2), dtype=torch.float32, device=device),
        )

    def compute_residuals(self, joint_q, residuals=None):
        active = joint_q.shape[0]
        values = wp.to_torch(joint_q)[:active, 0]
        self.residuals[:active, 0].copy_(values).sub_(1.0)
        self.residuals[:active, 1].copy_(torch.where(values == 0.0, 0.0, torch.nan))
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
            return residuals
        return self._wp_residuals[:active]

    def linearize(self, joint_q, residuals=None, jacobian=None):
        active = joint_q.shape[0]
        values = wp.to_torch(joint_q)[:active, 0]
        self.residuals[:active, 0].copy_(values).sub_(1.0)
        self.residuals[:active, 1].zero_()
        self.jacobian[:active].zero_()
        self.jacobian[:active, 0, 0].fill_(1.0)
        residual_view = self._wp_residuals[:active] if residuals is None else residuals
        jacobian_view = self._wp_jacobian[:active] if jacobian is None else jacobian
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
        if jacobian is not None:
            wp.copy(jacobian, self._wp_jacobian[:active])
        return residual_view, jacobian_view


def _solver(
    optimizer,
    *,
    max_segments: int = 4,
    max_equality_residuals_per_frame: int = 6,
    damping: float = 0.1,
    krylov_max_iterations: int = 64,
    krylov_relative_tolerance: float = 1.0e-4,
    kkt_relative_tolerance: float = 1.0e-4,
    krylov_check_interval: int = 8,
) -> IKTrajectorySolver:
    return IKTrajectorySolver(
        optimizer,
        max_segments=max_segments,
        max_equality_residuals_per_frame=max_equality_residuals_per_frame,
        damping=damping,
        krylov_max_iterations=krylov_max_iterations,
        krylov_relative_tolerance=krylov_relative_tolerance,
        kkt_relative_tolerance=kkt_relative_tolerance,
        krylov_check_interval=krylov_check_interval,
    )


def _solve(
    solver: IKTrajectorySolver,
    joint_q: torch.Tensor,
    joint_q_out: torch.Tensor,
    segment_offsets: torch.Tensor,
    step_seconds: torch.Tensor,
    pose_weights: torch.Tensor,
    temporal_weights: torch.Tensor,
    **kwargs,
) -> IKTrajectorySolver.Statistics:
    """Call the strict solver contract with unconstrained test defaults."""
    frame_count = joint_q.shape[0]
    segment_count = segment_offsets.shape[0] - 1
    if "coordinate_bounds" not in kwargs:
        kwargs["coordinate_bounds"] = IKTrajectorySolver.CoordinateBounds(
            coordinate_indices=torch.empty(0, dtype=torch.int32, device=joint_q.device),
            dof_indices=torch.empty(0, dtype=torch.int32, device=joint_q.device),
            lower=torch.empty(0, device=joint_q.device),
            upper=torch.empty(0, device=joint_q.device),
        )
    if "joint_velocity" not in kwargs:
        kwargs["joint_velocity"] = torch.zeros((frame_count, solver.dof_count), device=joint_q.device)
    if "velocity_lower" not in kwargs:
        kwargs["velocity_lower"] = torch.full((solver.dof_count,), -torch.inf, device=joint_q.device)
    if "velocity_upper" not in kwargs:
        kwargs["velocity_upper"] = torch.full((solver.dof_count,), torch.inf, device=joint_q.device)
    if "segment_active" not in kwargs:
        kwargs["segment_active"] = torch.ones(segment_count, dtype=torch.int32, device=joint_q.device)
    if "segment_direction_valid" not in kwargs:
        kwargs["segment_direction_valid"] = torch.empty(
            segment_count,
            dtype=torch.bool,
            device=joint_q.device,
        )
    if "segment_globalization_succeeded" not in kwargs:
        kwargs["segment_globalization_succeeded"] = torch.empty(
            segment_count,
            dtype=torch.bool,
            device=joint_q.device,
        )
    return solver.solve(
        joint_q,
        joint_q_out,
        segment_offsets,
        step_seconds,
        pose_weights,
        temporal_weights,
        **kwargs,
    )


def _residual_multipliers(solver: IKTrajectorySolver, frame: int) -> torch.Tensor:
    """Return physical residual-row multipliers from equality and IPM storage."""
    multipliers = (
        solver._ipm_constraint_scale[frame, : solver.residual_count]
        * solver._ipm_multiplier[frame, : solver.residual_count]
    ).clone()
    segment = solver._frame_segment[frame].item()
    base = solver._segment_offsets[segment].item() * solver.active_width
    count = solver._active_equality_count[segment].item()
    row_codes = solver._active_row_codes.flatten()
    row_scales = solver._active_row_scales.flatten()
    dual = solver._ipm_equality_dual.flatten()
    residual_span = solver.capacity * solver.residual_count
    for slot in range(base, base + count):
        code = row_codes[slot].item()
        if code < residual_span:
            row_frame = code // solver.residual_count
            residual = code % solver.residual_count
        elif code < 2 * residual_span:
            row_frame = (code - residual_span) // solver.residual_count
            residual = (code - residual_span) % solver.residual_count
        else:
            continue
        if row_frame == frame:
            multipliers[residual] = row_scales[slot] * dual[slot]
    return multipliers


def _dense_solution(
    jacobian: torch.Tensor,
    target: torch.Tensor,
    segment_offsets: tuple[int, ...],
    step_seconds: tuple[float, ...],
    base_weights: torch.Tensor,
    temporal_weights: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    frame_count, residual_count, dof_count = jacobian.shape
    size = frame_count * dof_count
    matrix = torch.eye(size, dtype=torch.float64) * damping
    right_hand_side = torch.zeros(size, dtype=torch.float64)
    jacobian = jacobian.to(torch.float64)
    target = target.to(torch.float64)
    base_weights = base_weights.to(torch.float64)
    temporal_weights = temporal_weights.to(torch.float64)
    for residual in range(residual_count):
        precision = torch.diag(base_weights[:, residual])
        for segment, (begin, end) in enumerate(zip(segment_offsets[:-1], segment_offsets[1:], strict=True)):
            length = end - begin
            for order in range(1, 4):
                weight = temporal_weights[order - 1, residual]
                if weight == 0.0 or length <= order:
                    continue
                coefficients = torch.tensor(
                    ((-1.0, 1.0), (1.0, -2.0, 1.0), (-1.0, 3.0, -3.0, 1.0))[order - 1],
                    dtype=torch.float64,
                )
                difference = torch.zeros((length - order, length), dtype=torch.float64)
                for row in range(length - order):
                    difference[row, row : row + order + 1] = coefficients
                precision[begin:end, begin:end] += (
                    weight / step_seconds[segment] ** (2 * order) * difference.T @ difference
                )
        residual_jacobian = torch.zeros((frame_count, size), dtype=torch.float64)
        for frame in range(frame_count):
            residual_jacobian[frame, frame * dof_count : (frame + 1) * dof_count] = jacobian[frame, residual]
        matrix += residual_jacobian.T @ precision @ residual_jacobian
        right_hand_side += residual_jacobian.T @ precision @ target[:, residual]
    return torch.linalg.solve(matrix, right_hand_side).reshape(frame_count, dof_count).to(torch.float32)


def _run_linear(
    jacobian: torch.Tensor,
    target: torch.Tensor,
    segment_offsets: torch.Tensor,
    pose_weights: torch.Tensor,
    temporal_weights: torch.Tensor,
    *,
    damping: float = 0.1,
    equalities: IKTrajectorySolver.ResidualEqualities | None = None,
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
    residual_sign: float = 1.0,
) -> tuple[torch.Tensor, IKTrajectorySolver]:
    optimizer = _LinearOptimizer(residual_sign * jacobian, residual_sign * target)
    solver = _solver(
        optimizer,
        max_segments=segment_offsets.numel() - 1,
        damping=damping,
    )
    initial = torch.zeros((jacobian.shape[0], jacobian.shape[2]), dtype=torch.float32, device=jacobian.device)
    output = torch.empty_like(initial)
    _solve(
        solver,
        initial,
        output,
        segment_offsets,
        torch.ones(segment_offsets.numel() - 1, dtype=torch.float32, device=jacobian.device),
        pose_weights,
        temporal_weights,
        equalities=equalities,
        coordinate_bounds=(
            IKTrajectorySolver.CoordinateBounds(
                coordinate_indices=torch.arange(jacobian.shape[2], dtype=torch.int32, device=jacobian.device),
                dof_indices=torch.arange(jacobian.shape[2], dtype=torch.int32, device=jacobian.device),
                lower=lower,
                upper=upper,
            )
            if lower is not None and upper is not None
            else IKTrajectorySolver.CoordinateBounds(
                coordinate_indices=torch.empty(0, dtype=torch.int32, device=jacobian.device),
                dof_indices=torch.empty(0, dtype=torch.int32, device=jacobian.device),
                lower=torch.empty(0, device=jacobian.device),
                upper=torch.empty(0, device=jacobian.device),
            )
        ),
    )
    return output, solver


def test_known_linear_solution_matches_dense_block_system() -> None:
    generator = torch.Generator().manual_seed(7)
    jacobian = torch.randn((7, 3, 2), generator=generator)
    target = torch.randn((7, 3), generator=generator)
    segment_offsets = torch.tensor([0, 4, 7], dtype=torch.int32)
    pose_weights = torch.tensor([1.0, 0.7, 1.3])
    temporal_weights = torch.tensor([[0.8, 0.0, 0.3], [0.2, 0.1, 0.0], [0.05, 0.0, 0.0]])
    output, _ = _run_linear(
        jacobian,
        target,
        segment_offsets,
        pose_weights,
        temporal_weights,
        damping=0.2,
    )
    expected = _dense_solution(
        jacobian,
        target,
        (0, 4, 7),
        (1.0, 1.0),
        pose_weights.expand(7, -1),
        temporal_weights,
        0.2,
    )
    torch.testing.assert_close(output, expected, atol=2.0e-4, rtol=2.0e-4)


def test_frozen_dofs_match_reduced_oracle_and_preserve_varying_values() -> None:
    """A masked full solve equals active-coordinate elimination without moving frozen values."""
    generator = torch.Generator().manual_seed(17)
    frame_count = 6
    jacobian = torch.randn((frame_count, 4, 4), generator=generator)
    target = torch.randn((frame_count, 4), generator=generator)
    initial = torch.zeros((frame_count, 4))
    initial[:, 0] = torch.linspace(-0.3, 0.4, frame_count)
    initial[:, 1] = torch.linspace(0.5, -0.2, frame_count)
    output = torch.empty_like(initial)
    offsets = torch.tensor((0, frame_count), dtype=torch.int32)
    pose_weights = torch.tensor((1.0, 0.7, 1.3, 0.5))
    temporal_weights = torch.tensor(((0.4, 0.0, 0.2, 0.1), (0.1, 0.05, 0.0, 0.0), (0.02, 0.0, 0.0, 0.0)))
    damping = 0.2
    optimizer = _LinearOptimizer(jacobian, target)
    solver = _solver(optimizer, damping=damping)
    _solve(
        solver,
        initial,
        output,
        offsets,
        torch.ones(1),
        pose_weights,
        temporal_weights,
        frozen_dof_indices=torch.tensor((0, 1), dtype=torch.int32),
    )

    frozen_contribution = torch.einsum("fri,fi->fr", jacobian[:, :, :2], initial[:, :2])
    expected_active = _dense_solution(
        jacobian[:, :, 2:],
        target - frozen_contribution,
        (0, frame_count),
        (1.0,),
        pose_weights.expand(frame_count, -1),
        temporal_weights,
        damping,
    )
    assert torch.equal(output[:, :2], initial[:, :2])
    assert torch.count_nonzero(output[:, 2:]) > 0
    assert torch.count_nonzero(solver._delta[:frame_count, :2]) == 0
    torch.testing.assert_close(output[:, 2:], expected_active, atol=2.0e-4, rtol=2.0e-4)


def test_segment_offsets_isolate_neighboring_trajectories() -> None:
    jacobian = torch.ones((8, 1, 1))
    first_target = torch.arange(4, dtype=torch.float32)
    target_a = torch.cat((first_target, torch.arange(4, dtype=torch.float32))).unsqueeze(-1)
    target_b = target_a.clone()
    target_b[4:] += 1000.0
    segment_offsets = torch.tensor([0, 4, 8], dtype=torch.int32)
    pose = torch.ones(1)
    temporal = torch.tensor([[10.0], [2.0], [0.5]])
    output_a, _ = _run_linear(jacobian, target_a, segment_offsets, pose, temporal, damping=1.0)
    output_b, _ = _run_linear(jacobian, target_b, segment_offsets, pose, temporal, damping=1.0)
    torch.testing.assert_close(output_a[:4], output_b[:4], atol=2.0e-4, rtol=2.0e-4)


def test_first_difference_preserves_target_velocity() -> None:
    frame_count = 8
    jacobian = torch.ones((frame_count, 2, 1))
    target = torch.stack((torch.arange(frame_count, dtype=torch.float32), torch.zeros(frame_count)), dim=-1)
    offsets = torch.tensor([0, frame_count], dtype=torch.int32)
    pose = torch.ones(2)
    no_temporal = torch.zeros((3, 2))
    velocity = no_temporal.clone()
    velocity[0, 0] = 100.0
    pose_only, _ = _run_linear(jacobian, target, offsets, pose, no_temporal)
    preserved, _ = _run_linear(jacobian, target, offsets, pose, velocity)
    target_velocity = torch.diff(target[:, 0])
    pose_error = torch.linalg.vector_norm(torch.diff(pose_only[:, 0]) - target_velocity)
    preserved_error = torch.linalg.vector_norm(torch.diff(preserved[:, 0]) - target_velocity)
    assert preserved_error < pose_error * 0.25


def test_segment_damping_matches_independent_dense_singleton_systems() -> None:
    """Each segment owns the LM diagonal used by its normal equation."""
    jacobian = torch.full((2, 1, 1), 2.0)
    target = torch.ones((2, 1))
    optimizer = _LinearOptimizer(jacobian, target)
    solver = _solver(optimizer, max_segments=2, damping=0.5)
    initial = torch.zeros((2, 1))
    output = torch.empty_like(initial)
    segment_damping = torch.tensor((0.25, 4.0))
    direction_valid = torch.empty(2, dtype=torch.bool)
    globalization_succeeded = torch.empty(2, dtype=torch.bool)

    _solve(
        solver,
        initial,
        output,
        torch.tensor((0, 1, 2), dtype=torch.int32),
        torch.ones(2),
        torch.ones(1),
        torch.zeros((3, 1)),
        segment_damping=segment_damping,
        segment_direction_valid=direction_valid,
        segment_globalization_succeeded=globalization_succeeded,
    )

    expected = 2.0 / (4.0 + segment_damping)
    torch.testing.assert_close(output[:, 0], expected, atol=2.0e-5, rtol=2.0e-5)
    assert torch.all(direction_valid)
    assert torch.all(globalization_succeeded)
    uniform_damping = torch.full((2,), solver.damping)
    omitted_output = torch.empty_like(initial)
    explicit_output = torch.empty_like(initial)
    solve_args = (torch.tensor((0, 1, 2), dtype=torch.int32), torch.ones(2), torch.ones(1), torch.zeros((3, 1)))
    _solve(solver, initial, omitted_output, *solve_args)
    _solve(solver, initial, explicit_output, *solve_args, segment_damping=uniform_damping)
    torch.testing.assert_close(omitted_output, explicit_output, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("bad_damping", [torch.tensor((0.1, 0.0)), torch.tensor((0.1, torch.nan))])
def test_segment_damping_rejects_nonpositive_or_nonfinite_values(bad_damping: torch.Tensor) -> None:
    optimizer = _LinearOptimizer(torch.ones((2, 1, 1)), torch.ones((2, 1)))
    solver = _solver(optimizer, max_segments=2)
    with pytest.raises(RuntimeError, match="segment_damping must contain positive finite values"):
        _solve(
            solver,
            torch.zeros((2, 1)),
            torch.empty((2, 1)),
            torch.tensor((0, 1, 2), dtype=torch.int32),
            torch.ones(2),
            torch.ones(1),
            torch.zeros((3, 1)),
            segment_damping=bad_damping,
        )


@pytest.mark.parametrize("damping", (torch.nan, torch.inf))
def test_trajectory_constructor_rejects_nonfinite_damping(damping: float) -> None:
    optimizer = _LinearOptimizer(torch.ones((1, 1, 1)), torch.ones((1, 1)))
    with pytest.raises(ValueError, match="Trajectory damping must be positive and finite"):
        _solver(optimizer, max_segments=1, damping=damping)


def test_segment_damping_requires_exact_float32_segment_boundary() -> None:
    optimizer = _LinearOptimizer(torch.ones((2, 1, 1)), torch.ones((2, 1)))
    solver = _solver(optimizer, max_segments=2)
    with pytest.raises(ValueError, match="segment_damping must be contiguous"):
        _solve(
            solver,
            torch.zeros((2, 1)),
            torch.empty((2, 1)),
            torch.tensor((0, 1, 2), dtype=torch.int32),
            torch.ones(2),
            torch.ones(1),
            torch.zeros((3, 1)),
            segment_damping=torch.ones(2, dtype=torch.float64),
        )


def test_hierarchical_step_minimizes_pose_subject_to_equality_and_collision() -> None:
    """Constraint rows define feasibility instead of weighted objective tradeoffs."""
    frame_count = 3
    jacobian = torch.zeros((frame_count, 6, 2))
    jacobian[:, 0, 0] = 1.0
    jacobian[:, 1, 1] = 1.0
    jacobian[:, 2] = 1.0
    jacobian[:, 3, 1] = 1.0
    target = torch.tensor(
        [
            [2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    optimizer = _LinearOptimizer(jacobian, target)
    solver = _solver(optimizer, damping=1.0e-6)
    equalities = IKTrajectorySolver.ResidualEqualities(
        active=torch.tensor([[0], [1], [1]], dtype=torch.uint8),
        residual_starts_by_target=torch.tensor([3], dtype=torch.int32),
    )
    output = torch.empty((frame_count, 2))
    statistics = _solve(
        solver,
        torch.zeros_like(output),
        output,
        torch.tensor([0, frame_count], dtype=torch.int32),
        torch.ones(1),
        torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        torch.zeros((3, 6)),
        equalities=equalities,
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor([2], dtype=torch.int32),
            upper=torch.tensor([1.0]),
        ),
    )

    expected = torch.tensor([[0.5, 0.5], [1.0, 0.0], [1.0, 0.0]])
    torch.testing.assert_close(output, expected, atol=2.0e-3, rtol=2.0e-3)
    assert torch.all(output.sum(dim=-1) <= 1.0 + 2.0e-3)
    assert statistics.equality_target_count == 1


def test_hierarchical_step_matches_dense_kkt_oracle() -> None:
    """Matrix-free MINRES matches the dense equality-constrained Gauss-Newton step."""
    jacobian = torch.tensor([[[1.0, 0.5], [0.25, 1.0], [0.0, 0.0], [0.0, 0.0], [-0.5, 1.5]]])
    target = torch.tensor([[2.0, 0.0, 0.0, 0.0, -1.0]])
    damping = 0.1
    pose_weights = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.5])
    optimizer = _LinearOptimizer(jacobian, target)
    solver = _solver(optimizer, damping=damping)
    equalities = IKTrajectorySolver.ResidualEqualities(
        active=torch.ones((1, 1), dtype=torch.uint8),
        residual_starts_by_target=torch.tensor([1], dtype=torch.int32),
    )
    output = torch.empty((1, 2))
    _solve(
        solver,
        torch.zeros_like(output),
        output,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.ones(1),
        pose_weights,
        torch.zeros((3, 5)),
        equalities=equalities,
    )

    residual = -target[0].to(torch.float64)
    jacobian_dense = jacobian[0].to(torch.float64)
    precision = torch.diag(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.5], dtype=torch.float64))
    hessian = jacobian_dense.T @ precision @ jacobian_dense + damping * torch.eye(2, dtype=torch.float64)
    gradient = jacobian_dense.T @ precision @ residual
    constraint = jacobian_dense[1:2]
    kkt = torch.cat(
        (
            torch.cat((hessian, constraint.T), dim=1),
            torch.cat((constraint, torch.zeros((1, 1), dtype=torch.float64)), dim=1),
        ),
        dim=0,
    )
    expected = torch.linalg.solve(kkt, torch.cat((-gradient, -residual[1:2])))[:2].to(torch.float32)
    torch.testing.assert_close(output[0], expected, atol=2.0e-5, rtol=2.0e-5)


def test_residual_equalities_require_dense_uint8_xyz_view() -> None:
    optimizer = _LinearOptimizer(torch.zeros((1, 6, 1)), torch.zeros((1, 6)))
    solver = _solver(optimizer, max_equality_residuals_per_frame=3)
    output = torch.empty((1, 1))
    arguments = (
        torch.zeros_like(output),
        output,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.ones(1),
        torch.zeros(6),
        torch.zeros((3, 6)),
    )

    with pytest.raises(ValueError, match="equalities.active"):
        _solve(
            solver,
            *arguments,
            equalities=IKTrajectorySolver.ResidualEqualities(
                active=torch.ones((1, 1), dtype=torch.bool),
                residual_starts_by_target=torch.tensor([0], dtype=torch.int32),
            ),
        )
    with pytest.raises(ValueError, match="exactly three"):
        _solve(
            solver,
            *arguments,
            equalities=IKTrajectorySolver.ResidualEqualities(
                active=torch.ones((1, 1), dtype=torch.uint8),
                residual_starts_by_target=torch.tensor([0], dtype=torch.int32),
                residual_width=2,
            ),
        )
    with pytest.raises(ValueError, match="exceeds the solver capacity"):
        _solve(
            solver,
            *arguments,
            equalities=IKTrajectorySolver.ResidualEqualities(
                active=torch.ones((1, 2), dtype=torch.uint8),
                residual_starts_by_target=torch.tensor([0, 3], dtype=torch.int32),
            ),
        )


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_hierarchical_solve_is_invariant_to_segment_batch_partition_and_permutation(device_type: str) -> None:
    """Independent KKT blocks must not share Krylov or acceptance scalars."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_type)
    segment_count = 8
    generator = torch.Generator().manual_seed(123)
    jacobian = torch.randn((segment_count, 5, 2), generator=generator).to(device)
    target = torch.randn((segment_count, 5), generator=generator).to(device)
    jacobian[:, 3:].zero_()
    target[:, 2:].zero_()

    def solve(selected: torch.Tensor) -> torch.Tensor:
        selected_jacobian = jacobian[selected]
        selected_target = target[selected]
        count = selected.numel()
        optimizer = _LinearOptimizer(selected_jacobian, selected_target)
        solver = _solver(
            optimizer,
            max_segments=count,
            max_equality_residuals_per_frame=3,
            damping=1.0e-4,
            krylov_max_iterations=4,
        )
        equalities = IKTrajectorySolver.ResidualEqualities(
            active=torch.ones((count, 1), dtype=torch.uint8, device=device),
            residual_starts_by_target=torch.tensor([2], dtype=torch.int32, device=device),
        )
        output = torch.empty((count, 2), device=device)
        _solve(
            solver,
            torch.zeros_like(output),
            output,
            torch.arange(count + 1, dtype=torch.int32, device=device),
            torch.ones(count, device=device),
            torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0], device=device),
            torch.zeros((3, 5), device=device),
            equalities=equalities,
        )
        return output

    individual = torch.cat([solve(torch.tensor([index], device=device)) for index in range(segment_count)])
    packed = solve(torch.arange(segment_count, device=device))
    permutation = torch.tensor([5, 0, 7, 2, 6, 1, 4, 3], device=device)
    permuted = solve(permutation)
    inverse = torch.argsort(permutation)

    torch.testing.assert_close(packed, individual, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(permuted[inverse], individual, atol=2.0e-5, rtol=2.0e-5)


def test_hierarchical_row_scaling_does_not_change_primal_step() -> None:
    """Equivalent inequality units alter multiplier coordinates, not the primal solution."""
    outputs = []
    for scale in (1.0, 100.0):
        optimizer = _LinearOptimizer(
            torch.tensor([[[1.0], [scale]]]),
            torch.tensor([[2.0, 0.0]]),
        )
        solver = _solver(optimizer, damping=0.1)
        output = torch.empty((1, 1))
        _solve(
            solver,
            torch.zeros_like(output),
            output,
            torch.tensor([0, 1], dtype=torch.int32),
            torch.ones(1),
            torch.tensor([1.0, 0.0]),
            torch.zeros((3, 2)),
            inequalities=IKTrajectorySolver.ResidualInequalities(
                residual_indices=torch.tensor([1], dtype=torch.int32),
                upper=torch.tensor([scale]),
            ),
        )
        outputs.append(output.clone())

    torch.testing.assert_close(outputs[0], torch.ones_like(outputs[0]), atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(outputs[1], outputs[0], atol=2.0e-5, rtol=2.0e-5)


def test_hierarchical_interior_point_rejects_objective_opposed_inequality() -> None:
    """An objective-opposed inequality remains inactive at the optimum."""
    optimizer = _LinearOptimizer(torch.ones((1, 2, 1)), torch.zeros((1, 2)))
    solver = _solver(optimizer, damping=0.1)
    output = torch.empty((1, 1))
    _solve(
        solver,
        torch.tensor([[2.0]]),
        output,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.ones(1),
        torch.tensor([1.0, 0.0]),
        torch.zeros((3, 2)),
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor([1], dtype=torch.int32),
            upper=torch.tensor([1.0]),
        ),
    )

    expected = torch.tensor([[2.0 * 0.1 / 1.1]])
    torch.testing.assert_close(output, expected, atol=2.0e-5, rtol=2.0e-5)
    assert output[0, 0] < 1.0
    torch.testing.assert_close(_residual_multipliers(solver, 0)[1], torch.tensor(0.0), atol=2.0e-6, rtol=0.0)


def test_hierarchical_interior_point_solves_cycling_qp_oracle() -> None:
    """A dependent entering row replaces its blocker without an add-remove cycle."""
    damping = 1.0e-6
    hessian = torch.tensor(
        ((0.59857169, -0.01572850), (-0.01572850, 0.44724936)),
        dtype=torch.float64,
    )
    gradient = torch.tensor((0.96127121, -0.10166808), dtype=torch.float64)
    inequalities = torch.tensor(
        (
            (1.90452133, 0.21425327),
            (-2.97439862, -0.43955370),
            (1.66503411, 0.00634583),
            (-0.43783368, -0.87365974),
            (1.32379783, 1.00282453),
            (0.67203485, 0.42810194),
            (1.44090625, 0.82120170),
            (-1.22980040, 0.76844354),
        ),
        dtype=torch.float64,
    )
    bounds = torch.tensor(
        (0.37428542, 1.04834212, 1.77394889, 0.26912404, 1.35905882, 0.58189583, 1.94722202, -0.02500532),
        dtype=torch.float32,
    )
    objective_jacobian = torch.linalg.cholesky(hessian - damping * torch.eye(2, dtype=torch.float64)).T
    objective_target = torch.linalg.solve(objective_jacobian.T, -gradient)
    jacobian = torch.cat((objective_jacobian, inequalities)).to(torch.float32).unsqueeze(0)
    target = torch.cat((objective_target, torch.zeros(8, dtype=torch.float64))).to(torch.float32).unsqueeze(0)
    optimizer = _LinearOptimizer(jacobian, target)
    solver = _solver(
        optimizer,
        max_equality_residuals_per_frame=3,
        damping=damping,
        kkt_relative_tolerance=1.0e-6,
    )
    output = torch.empty((1, 2))
    feasible = torch.empty(1, dtype=torch.bool)
    linear_converged = torch.empty(1, dtype=torch.bool)

    _solve(
        solver,
        torch.zeros_like(output),
        output,
        torch.tensor((0, 1), dtype=torch.int32),
        torch.ones(1),
        torch.tensor((1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        torch.zeros((3, 10)),
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.arange(2, 10, dtype=torch.int32),
            upper=bounds,
        ),
        segment_feasible=feasible,
        segment_direction_valid=linear_converged,
    )

    multipliers = _residual_multipliers(solver, 0)[2:]
    assert feasible.item()
    assert linear_converged.item()
    torch.testing.assert_close(output, torch.tensor(((-0.13109606, -0.24234351),)), atol=3.0e-5, rtol=0.0)
    torch.testing.assert_close(multipliers[[3, 7]], torch.tensor((0.30159982, 0.61356458)), atol=5.0e-5, rtol=0.0)
    assert multipliers[1] <= 1.0e-5
    assert torch.max(inequalities.to(torch.float32) @ output[0] - bounds) <= 64.0 * torch.finfo(torch.float32).eps


def test_hierarchical_correlated_inequalities_satisfy_primal_dual_kkt() -> None:
    """Correlated feasible rows stop at primal, dual, complementarity, and stationarity KKT conditions."""
    damping = 1.0e-6
    inequality_jacobian = torch.tensor([[1.0, 1.0], [1.0, 0.9]])
    bounds = torch.tensor([1.0, 0.95])
    target = torch.tensor([[2.5, 2.4, 0.0, 0.0]])
    optimizer = _LinearOptimizer(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.9]]]),
        target,
    )
    solver = _solver(optimizer, damping=damping)
    output = torch.empty((1, 2))
    feasible = torch.empty(1, dtype=torch.bool)
    linear_converged = torch.empty(1, dtype=torch.bool)
    _solve(
        solver,
        torch.zeros_like(output),
        output,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.ones(1),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        torch.zeros((3, 4)),
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor([2, 3], dtype=torch.int32),
            upper=bounds,
        ),
        segment_feasible=feasible,
        segment_direction_valid=linear_converged,
    )

    multipliers = _residual_multipliers(solver, 0)[2:4]
    primal = inequality_jacobian @ output[0] - bounds
    unconstrained = target[0, :2] / (1.0 + damping)
    stationarity = output[0] - unconstrained + inequality_jacobian.T @ multipliers
    assert feasible.item()
    assert linear_converged.item()
    torch.testing.assert_close(output, torch.tensor([[0.5, 0.5]]), atol=2.0e-4, rtol=2.0e-4)
    assert torch.all(primal <= 64.0 * torch.finfo(torch.float32).eps)
    assert torch.all(multipliers >= 0.0)
    torch.testing.assert_close(multipliers * primal, torch.zeros(2), atol=2.0e-5, rtol=0.0)
    torch.testing.assert_close(stationarity, torch.zeros(2), atol=2.0e-4, rtol=0.0)


def test_hierarchical_independent_nearly_parallel_rows_use_direct_rank() -> None:
    """Independent rows remain full rank when their Gram spectrum falls below a row-space tolerance."""
    epsilon = 5.0e-4
    machine_epsilon = torch.finfo(torch.float32).eps
    inequality_jacobian = torch.tensor(((1.0, 0.0), (1.0, epsilon)))
    unconstrained = inequality_jacobian.T @ torch.ones(2)
    direct_minimum = torch.linalg.svdvals(inequality_jacobian)[-1]
    gram_minimum = torch.linalg.eigvalsh(inequality_jacobian @ inequality_jacobian.T)[0]
    assert direct_minimum > machine_epsilon * inequality_jacobian.shape[1]
    assert gram_minimum < machine_epsilon * (inequality_jacobian.shape[0] + 1)
    optimizer = _LinearOptimizer(
        torch.tensor(
            (
                (
                    (1.0, 0.0),
                    (0.0, 1.0),
                    (1.0, 0.0),
                    (1.0, epsilon),
                ),
            )
        ),
        torch.tensor(((2.0, epsilon, 0.0, 0.0),)),
    )
    solver = _solver(optimizer, damping=1.0e-6)
    output = torch.empty((1, 2))
    feasible = torch.empty(1, dtype=torch.bool)
    _solve(
        solver,
        torch.zeros_like(output),
        output,
        torch.tensor((0, 1), dtype=torch.int32),
        torch.ones(1),
        torch.tensor((1.0, 1.0, 0.0, 0.0)),
        torch.zeros((3, 4)),
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor((2, 3), dtype=torch.int32),
            upper=torch.zeros(2),
        ),
        segment_feasible=feasible,
    )

    multipliers = _residual_multipliers(solver, 0)[2:4]
    primal = inequality_jacobian @ output[0]
    stationarity = output[0] - unconstrained / (1.0 + solver.damping) + inequality_jacobian.T @ multipliers
    assert feasible.item()
    torch.testing.assert_close(output, torch.zeros_like(output), atol=2.0e-5, rtol=0.0)
    assert torch.all(primal <= 64.0 * machine_epsilon)
    assert torch.all(multipliers > 0.0)
    torch.testing.assert_close(multipliers * primal, torch.zeros(2), atol=2.0e-5, rtol=0.0)
    torch.testing.assert_close(stationarity, torch.zeros(2), atol=2.0e-4, rtol=0.0)


def test_hierarchical_redundant_inequalities_preserve_kkt_solution() -> None:
    """Duplicate feasible half-spaces preserve the primal solution and a nonnegative dual certificate."""
    optimizer = _LinearOptimizer(torch.ones((1, 3, 1)), torch.tensor([[2.0, 0.0, 0.0]]))
    solver = _solver(optimizer, damping=1.0e-6)
    output = torch.empty((1, 1))
    feasible = torch.empty(1, dtype=torch.bool)
    _solve(
        solver,
        torch.zeros_like(output),
        output,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.ones(1),
        torch.tensor([1.0, 0.0, 0.0]),
        torch.zeros((3, 3)),
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor([1, 2], dtype=torch.int32),
            upper=torch.ones(2),
        ),
        segment_feasible=feasible,
    )

    multipliers = _residual_multipliers(solver, 0)[1:3]
    assert feasible.item()
    torch.testing.assert_close(output, torch.ones_like(output), atol=2.0e-5, rtol=2.0e-5)
    assert torch.all(multipliers >= 0.0)
    torch.testing.assert_close(multipliers.sum(), torch.tensor(1.0), atol=2.0e-5, rtol=2.0e-5)


def test_hierarchical_saturated_rank_solves_all_constraints_together() -> None:
    """A rank-saturated constraint family still reaches the supplied QP optimum."""
    optimizer = _LinearOptimizer(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]]),
        torch.tensor([[3.0, 2.0, 0.0, 0.0, -0.5]]),
    )
    solver = _solver(optimizer, max_equality_residuals_per_frame=0, damping=1.0e-6)
    output = torch.empty((1, 2))
    feasible = torch.empty(1, dtype=torch.bool)
    _solve(
        solver,
        torch.zeros_like(output),
        output,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.ones(1),
        torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]),
        torch.zeros((3, 5)),
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor([2, 3, 4], dtype=torch.int32),
            upper=torch.zeros(3),
        ),
        segment_feasible=feasible,
    )

    assert feasible.item()
    torch.testing.assert_close(output, torch.tensor([[-0.5, 0.0]]), atol=2.0e-5, rtol=2.0e-5)
    multipliers = _residual_multipliers(solver, 0)
    assert multipliers[2] <= 2.0e-5
    assert torch.all(multipliers[3:5] > 0.0)


def test_hierarchical_geometric_infeasibility_is_separate_from_contract_failure() -> None:
    """Contradictory geometry returns a false segment mask while malformed declarations raise."""
    jacobian = torch.tensor([[[1.0], [-1.0]]])
    target = torch.tensor([[0.0, -1.0]])
    optimizer = _LinearOptimizer(jacobian, target)
    solver = _solver(optimizer, damping=1.0e-6)
    output = torch.empty((1, 1))
    feasible = torch.empty(1, dtype=torch.bool)
    _solve(
        solver,
        torch.zeros_like(output),
        output,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.ones(1),
        torch.zeros(2),
        torch.zeros((3, 2)),
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor([0, 1], dtype=torch.int32),
            upper=torch.zeros(2),
        ),
        segment_feasible=feasible,
    )
    assert not feasible.item()

    malformed_solver = _solver(_LinearOptimizer(jacobian, target), damping=1.0e-6)
    with pytest.raises(RuntimeError, match="sorted, unique, finite, in range, and disjoint"):
        _solve(
            malformed_solver,
            torch.zeros_like(output),
            output,
            torch.tensor([0, 1], dtype=torch.int32),
            torch.ones(1),
            torch.zeros(2),
            torch.zeros((3, 2)),
            inequalities=IKTrajectorySolver.ResidualInequalities(
                residual_indices=torch.tensor([1, 0], dtype=torch.int32),
                upper=torch.zeros(2),
            ),
        )


def test_hierarchical_redundant_consistent_equalities_remain_solvable() -> None:
    """MINRES returns a feasible primal step for a singular but consistent KKT system."""
    optimizer = _LinearOptimizer(torch.ones((1, 3, 1)), torch.ones((1, 3)))
    solver = _solver(optimizer, max_equality_residuals_per_frame=3, damping=0.1)
    equalities = IKTrajectorySolver.ResidualEqualities(
        active=torch.ones((1, 1), dtype=torch.uint8),
        residual_starts_by_target=torch.tensor([0], dtype=torch.int32),
    )
    output = torch.empty((1, 1))
    _solve(
        solver,
        torch.zeros_like(output),
        output,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.ones(1),
        torch.ones(3),
        torch.zeros((3, 3)),
        equalities=equalities,
    )

    torch.testing.assert_close(output, torch.ones_like(output), atol=2.0e-5, rtol=2.0e-5)


def test_hierarchical_contradictory_equalities_require_phase_one_certificate() -> None:
    """Contradictory mandatory rows report geometry without publishing an unknown step."""
    optimizer = _LinearOptimizer(torch.ones((1, 3, 1)), torch.tensor(((1.0, 2.0, 1.0),)))
    solver = _solver(optimizer, max_equality_residuals_per_frame=3, damping=0.1)
    equalities = IKTrajectorySolver.ResidualEqualities(
        active=torch.ones((1, 1), dtype=torch.uint8),
        residual_starts_by_target=torch.tensor((0,), dtype=torch.int32),
    )
    initial = torch.zeros((1, 1))
    output = torch.empty_like(initial)
    feasible = torch.empty(1, dtype=torch.bool)
    linear_converged = torch.empty(1, dtype=torch.bool)

    _solve(
        solver,
        initial,
        output,
        torch.tensor((0, 1), dtype=torch.int32),
        torch.ones(1),
        torch.zeros(3),
        torch.zeros((3, 3)),
        equalities=equalities,
        segment_feasible=feasible,
        segment_direction_valid=linear_converged,
    )

    assert not feasible.item()
    assert not linear_converged.item()
    torch.testing.assert_close(output, initial)


@pytest.mark.parametrize("initial", (0.0, 2.0), ids=("safe_step_crosses", "initially_violated"))
def test_hierarchical_inequality_remains_or_becomes_feasible(initial: float) -> None:
    """Inequality feasibility dominates an improving ordinary objective."""
    optimizer = _LinearOptimizer(torch.tensor([[[1.0], [1.0]]]), torch.tensor([[2.0, 0.0]]))
    solver = _solver(optimizer, damping=1.0e-6)
    joint_q = torch.tensor([[initial]])
    output = torch.empty_like(joint_q)
    _solve(
        solver,
        joint_q,
        output,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.ones(1),
        torch.tensor([1.0, 0.0]),
        torch.zeros((3, 2)),
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor([1], dtype=torch.int32),
            upper=torch.tensor([1.0]),
        ),
    )

    torch.testing.assert_close(output, torch.ones_like(output), atol=1.0e-3, rtol=1.0e-3)


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_zero_equality_capacity_runs_inequality_only_ipm(device_type: str) -> None:
    """Zero declared equality capacity supports an inequality-only IPM solve."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_type)
    optimizer = _LinearOptimizer(
        torch.tensor([[[1.0], [1.0]]], device=device),
        torch.tensor([[2.0, 0.0]], device=device),
    )
    solver = _solver(optimizer, max_equality_residuals_per_frame=0, damping=1.0e-6)
    output = torch.empty((1, 1), device=device)
    statistics = _solve(
        solver,
        torch.zeros_like(output),
        output,
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.ones(1, device=device),
        torch.tensor([1.0, 0.0], device=device),
        torch.zeros((3, 2), device=device),
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor([1], dtype=torch.int32, device=device),
            upper=torch.tensor([1.0], device=device),
        ),
    )

    torch.testing.assert_close(output, torch.ones_like(output), atol=1.0e-3, rtol=1.0e-3)
    assert statistics.equality_target_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture requires CUDA.")
def test_constrained_ipm_cuda_graph_capture_replays() -> None:
    """The fixed-work constrained IPM captures and replays on a shared Torch/Warp stream."""
    device = torch.device("cuda")
    optimizer = _LinearOptimizer(
        torch.tensor([[[1.0], [1.0]]], device=device),
        torch.tensor([[1.0, 0.0]], device=device),
    )
    solver = _solver(
        optimizer,
        max_segments=1,
        max_equality_residuals_per_frame=0,
        damping=1.0e-4,
        krylov_max_iterations=8,
    )
    joint_q = torch.zeros((1, 1), device=device)
    output = torch.empty_like(joint_q)
    segment_offsets = torch.tensor([0, 1], dtype=torch.int32, device=device)
    step_seconds = torch.ones(1, device=device)
    pose_weights = torch.tensor([1.0, 0.0], device=device)
    temporal_weights = torch.zeros((3, 2), device=device)
    empty_indices = torch.empty(0, dtype=torch.int32, device=device)
    empty_values = torch.empty(0, device=device)
    coordinate_bounds = IKTrajectorySolver.CoordinateBounds(
        coordinate_indices=empty_indices,
        dof_indices=empty_indices,
        lower=empty_values,
        upper=empty_values,
    )
    joint_velocity = torch.zeros_like(joint_q)
    velocity_lower = torch.full((1,), -torch.inf, device=device)
    velocity_upper = torch.full((1,), torch.inf, device=device)
    segment_active = torch.ones(1, dtype=torch.int32, device=device)
    segment_feasible = torch.empty(1, dtype=torch.bool, device=device)
    segment_direction_valid = torch.empty(1, dtype=torch.bool, device=device)
    segment_globalization_succeeded = torch.empty(1, dtype=torch.bool, device=device)
    inequalities = IKTrajectorySolver.ResidualInequalities(
        residual_indices=torch.tensor([1], dtype=torch.int32, device=device),
        upper=torch.zeros(1, device=device),
    )

    def solve_once() -> None:
        solver.solve(
            joint_q,
            output,
            segment_offsets,
            step_seconds,
            pose_weights,
            temporal_weights,
            coordinate_bounds=coordinate_bounds,
            joint_velocity=joint_velocity,
            velocity_lower=velocity_lower,
            velocity_upper=velocity_upper,
            segment_active=segment_active,
            inequalities=inequalities,
            segment_feasible=segment_feasible,
            segment_direction_valid=segment_direction_valid,
            segment_globalization_succeeded=segment_globalization_succeeded,
            convergence_tolerance=1.0e-6,
        )

    solve_once()
    torch.cuda.synchronize()
    segment_active.fill_(1)
    torch_stream = torch.cuda.Stream()
    warp_stream = wp.stream_from_torch(torch_stream)
    wp.capture_begin(stream=warp_stream)
    with torch.cuda.stream(torch_stream), wp.ScopedStream(warp_stream, sync_enter=False):
        solve_once()
    graph = wp.capture_end(stream=warp_stream)
    wp.capture_launch(graph, stream=warp_stream)
    wp.synchronize_stream(warp_stream)

    torch.testing.assert_close(output, torch.zeros_like(output))
    assert segment_feasible.item()
    assert segment_direction_valid.item()
    assert segment_globalization_succeeded.item()


def test_hierarchical_line_search_rejects_objective_gain_that_breaks_feasibility() -> None:
    """Exact nonlinear violation is rejected when its linearization is locally hidden."""
    optimizer = _HiddenNonlinearConstraintOptimizer(1, torch.device("cpu"))
    solver = _solver(optimizer, damping=1.0e-6)
    initial = torch.zeros((1, 1))
    output = torch.empty_like(initial)
    _solve(
        solver,
        initial,
        output,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.ones(1),
        torch.tensor([1.0, 0.0]),
        torch.zeros((3, 2)),
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor([1], dtype=torch.int32),
            upper=torch.tensor([0.0]),
        ),
    )

    torch.testing.assert_close(output, initial)


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_second_order_correction_follows_curved_feasible_set(device_type: str) -> None:
    """A tangent descent receives one normal correction before materialization."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable.")
    device = torch.device(device_type)
    optimizer = _ParabolicConstraintOptimizer(1, device)
    solver = _solver(optimizer, max_segments=1, damping=1.0e-6)
    initial = torch.tensor(((1.0, 0.0),), device=device)
    output = torch.empty_like(initial)
    segment_active = torch.ones(1, dtype=torch.int32, device=device)
    segment_feasible = torch.empty(1, dtype=torch.bool, device=device)
    linear_converged = torch.empty(1, dtype=torch.bool, device=device)
    globalization_succeeded = torch.empty(1, dtype=torch.bool, device=device)

    _solve(
        solver,
        initial,
        output,
        torch.tensor((0, 1), dtype=torch.int32, device=device),
        torch.ones(1, device=device),
        torch.tensor((1.0, 0.0), device=device),
        torch.zeros((3, 2), device=device),
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor((1,), dtype=torch.int32, device=device),
            upper=torch.zeros(1, device=device),
        ),
        segment_active=segment_active,
        segment_feasible=segment_feasible,
        segment_direction_valid=linear_converged,
        segment_globalization_succeeded=globalization_succeeded,
        convergence_tolerance=1.0e-8,
    )

    assert output[0, 1] > 0.0
    assert output[0, 0] + output[0, 1].square() - 1.0 <= 64.0 * torch.finfo(torch.float32).eps
    assert segment_active[0] == 1
    assert segment_feasible[0]
    assert linear_converged[0]
    assert globalization_succeeded[0]
    assert solver._line_search_outcome[0] == 3
    assert optimizer.linearize_calls == 1
    assert optimizer.residual_calls == 2


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_failed_second_order_correction_preserves_smaller_backtracking_trials(device_type: str) -> None:
    """A failed full-step correction leaves the ordinary smaller trial ladder enabled."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable.")
    device = torch.device(device_type)
    optimizer = _FlatCurvatureConstraintOptimizer(1, device)
    solver = _solver(optimizer, max_segments=1, damping=1.0e-6)
    initial = torch.zeros((1, 1), device=device)
    output = torch.empty_like(initial)
    segment_active = torch.ones(1, dtype=torch.int32, device=device)
    segment_feasible = torch.empty(1, dtype=torch.bool, device=device)
    direction_valid = torch.empty(1, dtype=torch.bool, device=device)
    globalization_succeeded = torch.empty(1, dtype=torch.bool, device=device)

    _solve(
        solver,
        initial,
        output,
        torch.tensor((0, 1), dtype=torch.int32, device=device),
        torch.ones(1, device=device),
        torch.tensor((1.0, 0.0), device=device),
        torch.zeros((3, 2), device=device),
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor((1,), dtype=torch.int32, device=device),
            upper=torch.zeros(1, device=device),
        ),
        segment_active=segment_active,
        segment_feasible=segment_feasible,
        segment_direction_valid=direction_valid,
        segment_globalization_succeeded=globalization_succeeded,
        convergence_tolerance=1.0e-8,
    )

    torch.testing.assert_close(output, torch.full_like(output, 0.5), atol=2.0e-4, rtol=2.0e-4)
    assert segment_active[0] == 1
    assert segment_feasible[0]
    assert direction_valid[0]
    assert globalization_succeeded[0]


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_second_order_correction_canonicalizes_coordinate_bound_face(device_type: str) -> None:
    """A curved correction materializes on its scalar coordinate face without roundoff rejection."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable.")
    device = torch.device(device_type)
    optimizer = _UpperBoundCurvatureConstraintOptimizer(1, device)
    solver = _solver(optimizer, max_segments=1, damping=1.0e-6)
    initial = torch.zeros((1, 2), device=device)
    output = torch.empty_like(initial)
    segment_active = torch.ones(1, dtype=torch.int32, device=device)
    segment_feasible = torch.empty(1, dtype=torch.bool, device=device)
    direction_valid = torch.empty(1, dtype=torch.bool, device=device)
    globalization_succeeded = torch.empty(1, dtype=torch.bool, device=device)

    _solve(
        solver,
        initial,
        output,
        torch.tensor((0, 1), dtype=torch.int32, device=device),
        torch.ones(1, device=device),
        torch.tensor((1.0, 0.0), device=device),
        torch.zeros((3, 2), device=device),
        coordinate_bounds=IKTrajectorySolver.CoordinateBounds(
            coordinate_indices=torch.tensor((0,), dtype=torch.int32, device=device),
            dof_indices=torch.tensor((0,), dtype=torch.int32, device=device),
            lower=torch.zeros(1, device=device),
            upper=torch.ones(1, device=device),
        ),
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor((1,), dtype=torch.int32, device=device),
            upper=torch.zeros(1, device=device),
        ),
        segment_active=segment_active,
        segment_feasible=segment_feasible,
        segment_direction_valid=direction_valid,
        segment_globalization_succeeded=globalization_succeeded,
        convergence_tolerance=1.0e-8,
    )

    constraint = output[0, 1].square() - output[0, 0]
    assert output[0, 1] > 0.0
    assert 0.0 <= output[0, 0] <= 1.0
    assert constraint <= 64.0 * torch.finfo(torch.float32).eps
    assert segment_active[0] == 1
    assert segment_feasible[0]
    assert direction_valid[0]
    assert globalization_succeeded[0]


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_second_order_correction_restores_smooth_constraint_after_three_steps(device_type: str) -> None:
    """Three bounded smooth normal corrections restore an infeasible curved step."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable.")
    device = torch.device(device_type)
    optimizer = _ExponentialNormalConstraintOptimizer(1, device)
    solver = _solver(optimizer, max_segments=1, damping=1.0e-6)
    initial = torch.tensor(((1.0e-4, 0.0),), device=device)
    output = torch.empty_like(initial)
    segment_active = torch.ones(1, dtype=torch.int32, device=device)
    segment_feasible = torch.empty(1, dtype=torch.bool, device=device)
    direction_valid = torch.empty(1, dtype=torch.bool, device=device)
    globalization_succeeded = torch.empty(1, dtype=torch.bool, device=device)

    _solve(
        solver,
        initial,
        output,
        torch.tensor((0, 1), dtype=torch.int32, device=device),
        torch.ones(1, device=device),
        torch.tensor((1.0, 0.0, 0.0, 0.0), device=device),
        torch.zeros((3, 4), device=device),
        equalities=IKTrajectorySolver.ResidualEqualities(
            active=torch.ones((1, 1), dtype=torch.uint8, device=device),
            residual_starts_by_target=torch.tensor((1,), dtype=torch.int32, device=device),
        ),
        segment_active=segment_active,
        segment_feasible=segment_feasible,
        segment_direction_valid=direction_valid,
        segment_globalization_succeeded=globalization_succeeded,
        convergence_tolerance=1.0e-8,
    )

    initial_constraint = torch.exp(initial[0, 0]) - 1.0 + initial[0, 1].square()
    constraint = torch.exp(output[0, 0]) - 1.0 + output[0, 1].square()
    assert output[0, 1] > 0.0
    assert constraint.abs() < initial_constraint.abs()
    assert initial_constraint.abs() - constraint.abs() >= 1.0e-4 * initial_constraint.abs()
    assert segment_active[0] == 1
    assert segment_feasible[0]
    assert direction_valid[0]
    assert globalization_succeeded[0]
    assert solver._line_search_outcome[0] == 3
    assert optimizer.linearize_calls == 1
    assert optimizer.residual_calls == 4


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_second_order_correction_restores_infeasible_curved_constraint(device_type: str) -> None:
    """A curved step that worsens an infeasible point receives a normal restoration correction."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable.")
    device = torch.device(device_type)
    optimizer = _ParabolicConstraintOptimizer(1, device)
    solver = _solver(optimizer, max_segments=1, damping=1.0e-6)
    initial = torch.tensor(((1.0005, 0.0),), device=device)
    output = torch.empty_like(initial)
    segment_active = torch.ones(1, dtype=torch.int32, device=device)
    segment_feasible = torch.empty(1, dtype=torch.bool, device=device)
    direction_valid = torch.empty(1, dtype=torch.bool, device=device)
    globalization_succeeded = torch.empty(1, dtype=torch.bool, device=device)

    _solve(
        solver,
        initial,
        output,
        torch.tensor((0, 1), dtype=torch.int32, device=device),
        torch.ones(1, device=device),
        torch.tensor((1.0, 0.0), device=device),
        torch.zeros((3, 2), device=device),
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor((1,), dtype=torch.int32, device=device),
            upper=torch.zeros(1, device=device),
        ),
        segment_active=segment_active,
        segment_feasible=segment_feasible,
        segment_direction_valid=direction_valid,
        segment_globalization_succeeded=globalization_succeeded,
        convergence_tolerance=1.0e-8,
    )

    initial_violation = initial[0, 0] + initial[0, 1].square() - 1.0
    output_violation = output[0, 0] + output[0, 1].square() - 1.0
    assert output[0, 1] > 0.0
    assert output_violation < initial_violation
    assert initial_violation - output_violation >= 1.0e-4 * initial_violation
    assert segment_active[0] == 1
    assert segment_feasible[0]
    assert direction_valid[0]
    assert globalization_succeeded[0]
    assert solver._line_search_outcome[0] == 3


def test_second_order_restoration_allows_objective_ascent() -> None:
    """An infeasible correction follows restoration merit even when objective cost rises."""
    current_cost = torch.ones(1)
    candidate_cost = torch.full((1,), 2.0)
    current_violation = torch.ones(1)
    failed_violation = torch.full((1,), 1.1)
    corrected_violation = torch.full((1,), 0.5)
    zero = torch.zeros(1)
    gradient_dot_step = torch.full((1,), 0.5)
    tolerance = 64.0 * torch.finfo(torch.float32).eps
    enabled = torch.ones(1, dtype=torch.int32)
    outcome = torch.zeros_like(enabled)
    requested = torch.empty_like(enabled)

    wp.launch(
        _second_order_correction_request,
        dim=1,
        inputs=[
            current_cost,
            candidate_cost,
            current_violation,
            failed_violation,
            zero,
            zero,
            current_violation,
            failed_violation,
            1.0,
            gradient_dot_step,
            tolerance,
            torch.ones(1, dtype=torch.bool),
            torch.ones(1, dtype=torch.uint8),
            enabled,
            outcome,
        ],
        outputs=[requested],
        device="cpu",
    )

    assert requested[0]
    assert candidate_cost[0] > current_cost[0]

    take = torch.empty_like(enabled)
    continue_correction = torch.empty_like(enabled)
    wp.launch(
        _second_order_correction_decide,
        dim=1,
        inputs=[
            current_cost,
            candidate_cost,
            corrected_violation,
            zero,
            zero,
            corrected_violation,
            1.0,
            gradient_dot_step,
            tolerance,
            enabled,
            torch.ones(1, dtype=torch.uint8),
            requested,
            torch.tensor(((0.0, 1.0, 1.1, 0.0, 1.0, 1.1),)),
            outcome,
        ],
        outputs=[take, continue_correction],
        device="cpu",
    )

    assert take[0]
    assert not continue_correction[0]
    assert outcome[0] == 3


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_second_order_correction_preserves_nonrequested_segments(device_type: str) -> None:
    """A correction transaction leaves an independently stationary segment unchanged."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable.")
    device = torch.device(device_type)
    optimizer = _ParabolicConstraintOptimizer(2, device)
    solver = _solver(optimizer, max_segments=2, damping=1.0e-6)
    initial = torch.tensor(((1.0, 0.0), (0.0, 1.0)), device=device)
    output = torch.empty_like(initial)
    segment_active = torch.ones(2, dtype=torch.int32, device=device)
    segment_feasible = torch.empty(2, dtype=torch.bool, device=device)
    linear_converged = torch.empty(2, dtype=torch.bool, device=device)
    globalization_succeeded = torch.empty(2, dtype=torch.bool, device=device)

    _solve(
        solver,
        initial,
        output,
        torch.tensor((0, 1, 2), dtype=torch.int32, device=device),
        torch.ones(2, device=device),
        torch.tensor((1.0, 0.0), device=device),
        torch.zeros((3, 2), device=device),
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor((1,), dtype=torch.int32, device=device),
            upper=torch.zeros(1, device=device),
        ),
        segment_active=segment_active,
        segment_feasible=segment_feasible,
        segment_direction_valid=linear_converged,
        segment_globalization_succeeded=globalization_succeeded,
        convergence_tolerance=1.0e-8,
    )

    assert output[0, 1] > 0.0
    assert output[0, 0] + output[0, 1].square() - 1.0 <= 64.0 * torch.finfo(torch.float32).eps
    torch.testing.assert_close(output[1], initial[1])
    torch.testing.assert_close(segment_active, torch.tensor((1, 0), dtype=torch.int32, device=device))
    assert torch.all(segment_feasible)
    assert torch.all(linear_converged)
    assert torch.all(globalization_succeeded)
    torch.testing.assert_close(solver._line_search_outcome[:2], torch.tensor((3, 1), dtype=torch.int32, device=device))


def test_constraint_line_search_backtracks_large_numerically_insignificant_step() -> None:
    """A small decrease backtracks until it is sufficient for the trial scale."""
    current_cost = torch.ones(1)
    candidate_cost = torch.tensor([0.99995])
    zero = torch.zeros(1)
    enabled = torch.ones(1, dtype=torch.int32)
    outcome = torch.zeros_like(enabled)
    take = torch.empty_like(enabled)

    wp.launch(
        _constraint_line_search_decide,
        dim=1,
        inputs=[
            current_cost,
            candidate_cost,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            64.0 * torch.finfo(torch.float32).eps,
            1.0,
            torch.tensor([-0.5]),
            enabled,
            outcome,
        ],
        outputs=[take],
        device="cpu",
    )

    assert not take[0]

    outcome.zero_()
    wp.launch(
        _constraint_line_search_decide,
        dim=1,
        inputs=[
            current_cost,
            candidate_cost,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            64.0 * torch.finfo(torch.float32).eps,
            0.25,
            torch.tensor([-0.5]),
            enabled,
            outcome,
        ],
        outputs=[take],
        device="cpu",
    )

    assert take[0]
    assert outcome[0] == 2


def test_restoration_merit_is_invariant_to_constraint_row_scale() -> None:
    """Positive rescaling of a hard row leaves its normalized restoration merit unchanged."""
    wp.init()
    frame_segment = torch.zeros(1, dtype=torch.int32)
    segment_offsets = torch.tensor((0, 1), dtype=torch.int32)
    enabled = torch.ones(1, dtype=torch.int32)
    coordinate_indices = torch.zeros(1, dtype=torch.int32)
    coordinate_lower = torch.zeros(1)
    coordinate_upper = torch.zeros(1)
    joint_q = torch.zeros((1, 1))
    joint_velocity = torch.zeros((1, 1))
    velocity_lower = torch.full((1,), -torch.inf)
    velocity_upper = torch.full((1,), torch.inf)
    step_seconds = torch.ones(1)
    active_row_code = torch.full((1, 1), -1, dtype=torch.int32)
    active_row_scale = torch.zeros((1, 1))
    active_row_rhs = torch.zeros((1, 1))
    active_equality_count = torch.zeros(1, dtype=torch.int32)

    def merit(row_multiplier: float) -> tuple[torch.Tensor, torch.Tensor]:
        residual = torch.tensor(((2.0 * row_multiplier,),))
        upper = torch.tensor((1.0 * row_multiplier,))
        constraint_scale = torch.zeros((1, 5))
        constraint_scale[0, 0] = 0.5 / row_multiplier
        constraint_rhs = torch.zeros_like(constraint_scale)
        constraint_rhs[0, 0] = constraint_scale[0, 0] * (upper[0] - residual[0, 0])
        current = torch.zeros(1)
        candidate = torch.zeros(1)
        wp.launch(
            _restoration_current_merit_max,
            dim=(1, 5),
            inputs=[
                wp.from_torch(constraint_rhs),
                wp.from_torch(constraint_scale),
                wp.from_torch(active_row_rhs),
                wp.from_torch(active_row_code),
                wp.from_torch(active_equality_count),
                wp.from_torch(frame_segment),
                wp.from_torch(segment_offsets),
                wp.from_torch(enabled),
                5,
                1,
            ],
            outputs=[wp.from_torch(current)],
            device="cpu",
        )
        wp.launch(
            _restoration_candidate_merit_max,
            dim=(1, 5),
            inputs=[
                wp.from_torch(residual),
                wp.from_torch(upper),
                wp.from_torch(joint_q),
                wp.from_torch(coordinate_indices),
                wp.from_torch(coordinate_lower),
                wp.from_torch(coordinate_upper),
                0,
                wp.from_torch(joint_velocity),
                wp.from_torch(velocity_lower),
                wp.from_torch(velocity_upper),
                wp.from_torch(step_seconds),
                wp.from_torch(constraint_scale),
                wp.from_torch(active_row_code),
                wp.from_torch(active_row_scale),
                wp.from_torch(active_equality_count),
                wp.from_torch(frame_segment),
                wp.from_torch(segment_offsets),
                wp.from_torch(enabled),
                5,
                1,
            ],
            outputs=[wp.from_torch(candidate)],
            device="cpu",
        )
        return current, candidate

    base_current, base_candidate = merit(1.0)
    scaled_current, scaled_candidate = merit(1.0e3)

    torch.testing.assert_close(base_current, torch.tensor((0.5,)))
    torch.testing.assert_close(base_candidate, base_current)
    torch.testing.assert_close(scaled_current, base_current)
    torch.testing.assert_close(scaled_candidate, base_candidate)


@pytest.mark.parametrize(
    ("current_merit", "accepted_merit", "expected_stalled"),
    ((1.0, 1.0 - 5.0e-7, True), (1.0e-4, 1.0e-4 - 1.0e-8, False)),
)
def test_infeasible_restoration_requires_material_relative_progress(
    current_merit: float, accepted_merit: float, expected_stalled: bool
) -> None:
    """Tiny relative drift stalls while material normalized restoration relinearizes."""
    linear_converged = torch.ones(1, dtype=torch.bool)
    globalization_succeeded = torch.ones(1, dtype=torch.bool)
    restoration_stalled = torch.empty(1, dtype=torch.bool)
    segment_active = torch.ones(1, dtype=torch.int32)
    wp.launch(
        _segment_convergence_update,
        dim=1,
        inputs=[
            torch.ones(1),
            torch.ones(1, dtype=torch.int32),
            1,
            1.0e-6,
            torch.ones(1),
            64.0 * torch.finfo(torch.float32).eps,
            wp.uint8(1),
            torch.full((1,), 2, dtype=torch.int32),
            torch.tensor((-0.5,)),
            torch.tensor((current_merit,)),
            torch.tensor((accepted_merit,)),
        ],
        outputs=[linear_converged, globalization_succeeded, restoration_stalled, segment_active],
        device="cpu",
    )

    assert linear_converged[0]
    assert bool(globalization_succeeded[0]) is not expected_stalled
    assert bool(restoration_stalled[0]) is expected_stalled
    assert bool(segment_active[0]) is not expected_stalled


def test_constraint_restoration_requires_scale_aware_violation_decrease() -> None:
    """One-ULP restoration noise cannot select a full material trial."""
    current_cost = torch.ones(1)
    candidate_cost = torch.full((1,), 2.0)
    current_violation = torch.ones(1)
    candidate_violation = torch.nextafter(current_violation, torch.zeros(1))
    zero = torch.zeros(1)
    enabled = torch.ones(1, dtype=torch.int32)
    outcome = torch.zeros_like(enabled)
    take = torch.empty_like(enabled)

    wp.launch(
        _constraint_line_search_decide,
        dim=1,
        inputs=[
            current_cost,
            candidate_cost,
            current_violation,
            candidate_violation,
            zero,
            zero,
            current_violation,
            candidate_violation,
            64.0 * torch.finfo(torch.float32).eps,
            1.0,
            torch.tensor([0.5]),
            enabled,
            outcome,
        ],
        outputs=[take],
        device="cpu",
    )

    assert not take[0]

    candidate_violation.fill_(0.9)
    wp.launch(
        _constraint_line_search_decide,
        dim=1,
        inputs=[
            current_cost,
            candidate_cost,
            current_violation,
            candidate_violation,
            zero,
            zero,
            current_violation,
            candidate_violation,
            64.0 * torch.finfo(torch.float32).eps,
            0.5,
            torch.tensor([0.5]),
            enabled,
            outcome,
        ],
        outputs=[take],
        device="cpu",
    )


def test_constraint_restoration_uses_total_hard_violation() -> None:
    """An infeasible iterate may trade active rows and bounds while reducing total violation."""
    current_cost = torch.ones(1)
    candidate_cost = torch.full((1,), 2.0)
    current_violation = torch.full((1,), 10.0)
    candidate_violation = torch.full((1,), 0.4)
    candidate_bound_violation = torch.zeros(1)
    candidate_protected_violation = torch.full((1,), 0.4)
    enabled = torch.ones(1, dtype=torch.int32)
    outcome = torch.zeros_like(enabled)
    take = torch.empty_like(enabled)

    wp.launch(
        _constraint_line_search_decide,
        dim=1,
        inputs=[
            current_cost,
            candidate_cost,
            current_violation,
            candidate_violation,
            candidate_bound_violation,
            candidate_protected_violation,
            current_violation,
            torch.maximum(candidate_violation, candidate_bound_violation),
            64.0 * torch.finfo(torch.float32).eps,
            1.0,
            torch.tensor([0.5]),
            enabled,
            outcome,
        ],
        outputs=[take],
        device="cpu",
    )

    assert take[0]
    assert outcome[0] == 2

    outcome.zero_()
    candidate_bound_violation.fill_(6.0)
    candidate_protected_violation.zero_()
    wp.launch(
        _constraint_line_search_decide,
        dim=1,
        inputs=[
            current_cost,
            candidate_cost,
            current_violation,
            candidate_violation,
            candidate_bound_violation,
            candidate_protected_violation,
            current_violation,
            torch.maximum(candidate_violation, candidate_bound_violation),
            64.0 * torch.finfo(torch.float32).eps,
            0.5,
            torch.tensor([0.5]),
            enabled,
            outcome,
        ],
        outputs=[take],
        device="cpu",
    )

    assert take[0]
    assert outcome[0] == 2


def test_constraint_feasibility_uses_numerical_bound_tolerance() -> None:
    """A roundoff-sized bound residual does not reject an otherwise feasible descent trial."""
    tolerance = 64.0 * torch.finfo(torch.float32).eps
    enabled = torch.ones(1, dtype=torch.int32)
    outcome = torch.zeros_like(enabled)
    take = torch.empty_like(enabled)
    wp.launch(
        _constraint_line_search_decide,
        dim=1,
        inputs=[
            torch.ones(1),
            torch.full((1,), 0.5),
            torch.zeros(1),
            torch.zeros(1),
            torch.full((1,), 0.5 * tolerance),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            tolerance,
            1.0,
            torch.tensor([-0.5]),
            enabled,
            outcome,
        ],
        outputs=[take],
        device="cpu",
    )

    assert take[0]
    assert outcome[0] == 2


def test_constraint_restoration_balances_scaled_velocity_and_nonlinear_residual() -> None:
    """D-scaled restoration can satisfy residual and velocity rows in one accepted step."""
    optimizer = _HiddenNonlinearConstraintOptimizer(2, torch.device("cpu"))
    solver = _solver(optimizer, max_segments=1, damping=1.0e-6)
    initial = torch.zeros((2, 1))
    output = torch.empty_like(initial)
    joint_velocity = torch.tensor(((0.0,), (-1.0,)))
    active = torch.ones(1, dtype=torch.int32)
    feasible = torch.empty(1, dtype=torch.bool)
    direction_valid = torch.empty(1, dtype=torch.bool)
    globalization_succeeded = torch.empty(1, dtype=torch.bool)
    residual_satisfied = torch.empty(1, dtype=torch.bool)

    _solve(
        solver,
        initial,
        output,
        torch.tensor((0, 2), dtype=torch.int32),
        torch.ones(1),
        torch.tensor((1.0, 0.0)),
        torch.zeros((3, 2)),
        residual_activity=IKTrajectorySolver.ResidualActivity(
            values=torch.tensor(((0.0,), (1.0,))),
            group_by_residual=torch.tensor((0, -1), dtype=torch.int32),
        ),
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor((1,), dtype=torch.int32),
            upper=torch.tensor((0.7,)),
        ),
        joint_velocity=joint_velocity,
        velocity_lower=torch.tensor((-0.5,)),
        velocity_upper=torch.tensor((0.5,)),
        segment_active=active,
        segment_feasible=feasible,
        segment_direction_valid=direction_valid,
        segment_globalization_succeeded=globalization_succeeded,
        segment_residual_constraints_satisfied=residual_satisfied,
        convergence_tolerance=1.0e-8,
    )

    candidate_velocity = joint_velocity[1, 0] + torch.diff(output[:, 0] - initial[:, 0])[0]
    velocity_violation = torch.clamp(torch.maximum(-0.5 - candidate_velocity, candidate_velocity - 0.5), min=0.0)
    numerical_tolerance = 64.0 * torch.finfo(torch.float32).eps
    residual_violation = torch.clamp(output[:, 0].square() - 0.7, min=0.0).max()

    assert residual_violation <= numerical_tolerance
    assert velocity_violation <= numerical_tolerance
    assert solver._line_search_outcome[0] == 2
    assert active[0] == 1
    assert feasible[0] and direction_valid[0] and globalization_succeeded[0]
    assert residual_satisfied[0]
    torch.testing.assert_close(solver._constraint_violation[0], residual_violation)


def test_accepted_restoration_relinearizes_before_stationarity() -> None:
    """A feasible restoration candidate cannot reuse stationarity from its infeasible predecessor."""
    linear_converged = torch.ones(1, dtype=torch.bool)
    globalization_succeeded = torch.ones(1, dtype=torch.bool)
    restoration_stalled = torch.empty(1, dtype=torch.bool)
    segment_active = torch.ones(1, dtype=torch.int32)
    wp.launch(
        _segment_convergence_update,
        dim=1,
        inputs=[
            torch.ones(1),
            torch.ones(1, dtype=torch.int32),
            1,
            1.0e-6,
            torch.zeros(1),
            64.0 * torch.finfo(torch.float32).eps,
            wp.uint8(1),
            torch.full((1,), 2, dtype=torch.int32),
            torch.tensor([-2.5e-7]),
            torch.zeros(1),
            torch.zeros(1),
        ],
        outputs=[linear_converged, globalization_succeeded, restoration_stalled, segment_active],
        device="cpu",
    )

    assert linear_converged[0]
    assert globalization_succeeded[0]
    assert segment_active[0]
    assert not restoration_stalled[0]


@pytest.mark.parametrize(
    ("current", "candidate", "gradient_dot_step"),
    (
        (math.nan, 0.0, -0.5),
        (0.0, math.nan, -0.5),
        (0.0, 0.0, math.nan),
    ),
)
def test_line_search_never_accepts_nonfinite_evidence(
    current: float,
    candidate: float,
    gradient_dot_step: float,
) -> None:
    """Non-finite costs or directions cannot become an accepted configuration."""
    outcome = torch.zeros(1, dtype=torch.int32)
    take = torch.empty_like(outcome)
    wp.launch(
        _line_search_decide,
        dim=1,
        inputs=[
            torch.tensor([current]),
            torch.tensor([candidate]),
            1.0,
            torch.tensor([gradient_dot_step]),
            torch.ones(1, dtype=torch.int32),
            outcome,
        ],
        outputs=[take],
        device="cpu",
    )

    assert not outcome[0]
    assert not take[0]


def test_line_search_does_not_round_required_decrease_into_an_equal_cost_acceptance() -> None:
    """An Armijo threshold below one cost ULP still rejects an unchanged objective."""
    outcome = torch.zeros(1, dtype=torch.int32)
    take = torch.empty_like(outcome)
    wp.launch(
        _line_search_decide,
        dim=1,
        inputs=[
            torch.ones(1),
            torch.ones(1),
            1.0,
            torch.tensor([-5.0e-6]),
            torch.ones(1, dtype=torch.int32),
            outcome,
        ],
        outputs=[take],
        device="cpu",
    )

    assert not outcome[0]
    assert not take[0]


def test_line_search_backtracks_from_nonfinite_trial_to_finite_descent() -> None:
    """A bad full trial does not poison a later finite descending trial."""
    current_cost = torch.ones(1)
    candidate_cost = torch.tensor([math.nan])
    enabled = torch.ones(1, dtype=torch.int32)
    outcome = torch.zeros_like(enabled)
    take = torch.empty_like(enabled)
    common = [
        1.0,
        torch.tensor([-0.5]),
        enabled,
        outcome,
    ]

    wp.launch(
        _line_search_decide,
        dim=1,
        inputs=[current_cost, candidate_cost, *common],
        outputs=[take],
        device="cpu",
    )
    assert not take[0]
    assert not outcome[0]

    candidate_cost.fill_(0.5)
    common[0] = 0.5
    wp.launch(
        _line_search_decide,
        dim=1,
        inputs=[current_cost, candidate_cost, *common],
        outputs=[take],
        device="cpu",
    )

    assert outcome[0] == 2
    assert take[0]


@pytest.mark.parametrize(("current", "gradient_dot_step"), ((math.nan, -0.5), (1.0, math.nan)))
def test_nonfinite_model_evidence_is_an_explicit_globalization_failure(
    current: float,
    gradient_dot_step: float,
) -> None:
    """Non-finite model evidence stops the segment and clears its success status."""
    linear_converged = torch.ones(1, dtype=torch.bool)
    globalization_succeeded = torch.ones(1, dtype=torch.bool)
    restoration_stalled = torch.empty(1, dtype=torch.bool)
    segment_active = torch.ones(1, dtype=torch.int32)
    wp.launch(
        _segment_convergence_update,
        dim=1,
        inputs=[
            torch.tensor([current]),
            torch.ones(1, dtype=torch.int32),
            1,
            1.0e-6,
            torch.zeros(1),
            64.0 * torch.finfo(torch.float32).eps,
            wp.uint8(0),
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([gradient_dot_step]),
            torch.zeros(1),
            torch.zeros(1),
        ],
        outputs=[linear_converged, globalization_succeeded, restoration_stalled, segment_active],
        device="cpu",
    )

    assert linear_converged[0]
    assert not globalization_succeeded[0]
    assert not segment_active[0]
    assert not restoration_stalled[0]


def test_model_stationarity_converges_without_copying_a_trial() -> None:
    """A finite model-stationary segment resolves before trial-state acceptance."""
    current_cost = torch.ones(1)
    gradient_dot_step = torch.tensor([-2.5e-7])
    linear_converged = torch.ones(1, dtype=torch.bool)
    globalization_succeeded = torch.ones(1, dtype=torch.bool)
    restoration_stalled = torch.empty(1, dtype=torch.bool)
    segment_active = torch.ones(1, dtype=torch.int32)
    outcome = torch.empty(1, dtype=torch.int32)
    take = torch.empty_like(outcome)
    wp.launch(
        _line_search_stationarity_initialize,
        dim=1,
        inputs=[
            current_cost,
            gradient_dot_step,
            torch.ones(1, dtype=torch.int32),
            1.0e-6,
            torch.zeros(1),
            64.0 * torch.finfo(torch.float32).eps,
            wp.uint8(0),
            linear_converged,
            segment_active,
        ],
        outputs=[outcome, take],
        device="cpu",
    )

    assert outcome[0] == 1
    assert not take[0]
    wp.launch(
        _segment_convergence_update,
        dim=1,
        inputs=[
            current_cost,
            torch.ones(1, dtype=torch.int32),
            1,
            1.0e-6,
            torch.zeros(1),
            64.0 * torch.finfo(torch.float32).eps,
            wp.uint8(0),
            outcome,
            gradient_dot_step,
            torch.zeros(1),
            torch.zeros(1),
        ],
        outputs=[linear_converged, globalization_succeeded, restoration_stalled, segment_active],
        device="cpu",
    )

    assert linear_converged[0]
    assert globalization_succeeded[0]
    assert not segment_active[0]
    assert not restoration_stalled[0]


def test_none_tolerance_does_not_hide_finite_model_descent() -> None:
    """Disabling convergence leaves every positive finite model descent pending."""
    outcome = torch.empty(1, dtype=torch.int32)
    take = torch.empty_like(outcome)
    wp.launch(
        _line_search_stationarity_initialize,
        dim=1,
        inputs=[
            torch.tensor([1.0e-12]),
            torch.tensor([-5.0e-9]),
            torch.ones(1, dtype=torch.int32),
            -1.0,
            torch.zeros(1),
            64.0 * torch.finfo(torch.float32).eps,
            wp.uint8(0),
            torch.ones(1, dtype=torch.bool),
            torch.ones(1, dtype=torch.int32),
        ],
        outputs=[outcome, take],
        device="cpu",
    )

    assert not outcome[0]
    assert not take[0]


def test_segment_step_max_marks_nonfinite_direction() -> None:
    """A non-finite tangent direction cannot masquerade as a stationary step."""
    output = torch.zeros(1)
    wp.launch(
        _segment_step_max,
        dim=(1, 1),
        inputs=[
            torch.tensor([[math.nan]]),
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            1,
        ],
        outputs=[output],
        device="cpu",
    )

    assert torch.isinf(output[0])


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_nonlinear_feasibility_measures_fail_closed_on_nonfinite_values(device_type: str) -> None:
    """Equality, inequality, coordinate, and velocity NaNs become maximal violation."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_type)
    segment = torch.zeros(2, dtype=torch.int32, device=device)
    enabled = torch.ones(1, dtype=torch.int32, device=device)
    output = torch.zeros(1, device=device)
    for kind in (1, 2):
        output.zero_()
        wp.launch(
            _constraint_violation_max,
            dim=(1, 1),
            inputs=[
                torch.full((1, 1), torch.nan, device=device),
                torch.zeros(1, device=device),
                torch.full((1, 1), kind, dtype=torch.uint8, device=device),
                segment[:1],
                enabled,
                1,
            ],
            outputs=[output],
            device=device_type,
        )
        assert output[0] > 1.0e30

    output.zero_()
    wp.launch(
        _coordinate_bound_violation_max,
        dim=(1, 1),
        inputs=[
            torch.full((1, 1), torch.nan, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.full((1,), -1.0, device=device),
            torch.ones(1, device=device),
            1,
            segment[:1],
            enabled,
            1,
        ],
        outputs=[output],
        device=device_type,
    )
    assert output[0] > 1.0e30

    output.zero_()
    wp.launch(
        _ipm_velocity_bound_violation_max,
        dim=(2, 1),
        inputs=[
            torch.tensor([[0.0], [math.nan]], device=device),
            torch.full((1,), -1.0, device=device),
            torch.ones(1, device=device),
            segment,
            torch.tensor([0, 2], dtype=torch.int32, device=device),
            enabled,
            2,
        ],
        outputs=[output],
        device=device_type,
    )
    assert output[0] > 1.0e30


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_constraint_only_nan_trial_is_never_materialized(device_type: str) -> None:
    """A finite objective cannot hide NaN evidence in a constraint-only residual row."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_type)
    solver = _solver(_NaNTrialConstraintOptimizer(1, device), max_segments=1, damping=1.0e-6)
    initial = torch.zeros((1, 1), device=device)
    output = torch.empty_like(initial)
    active = torch.ones(1, dtype=torch.int32, device=device)
    linear_converged = torch.empty(1, dtype=torch.bool, device=device)
    globalization_succeeded = torch.empty(1, dtype=torch.bool, device=device)
    _solve(
        solver,
        initial,
        output,
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.ones(1, device=device),
        torch.tensor([1.0, 0.0], device=device),
        torch.zeros((3, 2), device=device),
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor([1], dtype=torch.int32, device=device),
            upper=torch.zeros(1, device=device),
        ),
        segment_active=active,
        segment_direction_valid=linear_converged,
        segment_globalization_succeeded=globalization_succeeded,
        convergence_tolerance=1.0e-6,
    )

    torch.testing.assert_close(output, initial)
    assert linear_converged[0]
    assert not globalization_succeeded[0]
    assert not active[0]


def test_line_search_and_convergence_ignore_zero_weight_residual_padding() -> None:
    """Unused helper rows cannot change an otherwise identical trajectory step."""

    def solve(residual_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        jacobian = torch.zeros((1, residual_count, 1))
        target = torch.zeros((1, residual_count))
        weights = torch.zeros(residual_count)
        jacobian[0, 0, 0] = 1.0
        target[0, 0] = 0.01
        weights[0] = 1.0
        solver = _solver(_LinearOptimizer(jacobian, target), max_segments=1, damping=0.1)
        output = torch.empty((1, 1))
        active = torch.ones(1, dtype=torch.int32)
        _solve(
            solver,
            torch.zeros_like(output),
            output,
            torch.tensor((0, 1), dtype=torch.int32),
            torch.ones(1),
            weights,
            torch.zeros((3, residual_count)),
            segment_active=active,
            convergence_tolerance=1.0e-6,
        )
        return output, active

    compact_output, compact_active = solve(1)
    padded_output, padded_active = solve(256)

    torch.testing.assert_close(padded_output, compact_output)
    torch.testing.assert_close(padded_active, compact_active)


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_coordinate_bounds_project_the_accepted_candidate(device_type: str) -> None:
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_type)
    output, _ = _run_linear(
        torch.ones((4, 1, 1), device=device),
        torch.full((4, 1), 3.0, device=device),
        torch.tensor([0, 4], dtype=torch.int32, device=device),
        torch.ones(1, device=device),
        torch.zeros((3, 1), device=device),
        lower=torch.tensor([-0.25], device=device),
        upper=torch.tensor([0.5], device=device),
    )
    torch.testing.assert_close(output, torch.full_like(output, 0.5))


def test_box_only_bounds_use_projected_solver_and_stop_at_the_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    optimizer = _LinearOptimizer(torch.ones((1, 1, 1)), torch.full((1, 1), 3.0))
    solver = _solver(optimizer, max_segments=1)
    monkeypatch.setattr(
        solver,
        "_solve_ipm_linearized",
        lambda *_args, **_kwargs: pytest.fail("Box-only coordinate bounds entered the mixed-constraint IPM path."),
    )
    bounds = IKTrajectorySolver.CoordinateBounds(
        coordinate_indices=torch.zeros(1, dtype=torch.int32),
        dof_indices=torch.zeros(1, dtype=torch.int32),
        lower=torch.full((1,), -0.25),
        upper=torch.full((1,), 0.5),
    )
    offsets = torch.tensor((0, 1), dtype=torch.int32)
    step_seconds = torch.ones(1)
    weights = torch.ones(1)
    temporal_weights = torch.zeros((3, 1))
    active = torch.ones(1, dtype=torch.int32)
    linear_converged = torch.empty(1, dtype=torch.bool)
    globalization_succeeded = torch.empty(1, dtype=torch.bool)
    initial = torch.zeros((1, 1))
    boundary = torch.empty_like(initial)

    _solve(
        solver,
        initial,
        boundary,
        offsets,
        step_seconds,
        weights,
        temporal_weights,
        coordinate_bounds=bounds,
        segment_active=active,
        segment_direction_valid=linear_converged,
        segment_globalization_succeeded=globalization_succeeded,
        convergence_tolerance=1.0e-6,
    )
    torch.testing.assert_close(boundary, torch.full_like(boundary, 0.5))
    assert active[0]
    assert linear_converged[0]
    assert globalization_succeeded[0]

    stationary = torch.empty_like(initial)
    _solve(
        solver,
        boundary,
        stationary,
        offsets,
        step_seconds,
        weights,
        temporal_weights,
        coordinate_bounds=bounds,
        segment_active=active,
        segment_direction_valid=linear_converged,
        segment_globalization_succeeded=globalization_succeeded,
        convergence_tolerance=1.0e-6,
    )
    torch.testing.assert_close(stationary, boundary)
    assert not active[0]
    assert linear_converged[0]
    assert globalization_succeeded[0]


def test_projected_bounds_scale_coupled_descent_before_backtracking() -> None:
    """A coupled bounded objective chooses a curvature-scaled trial below the fixed backtracking ladder."""
    dof_count = 17
    damping = 1.0e-6
    optimizer = _LinearOptimizer(torch.ones((1, 1, dof_count)), torch.ones((1, 1)))
    solver = _solver(optimizer, max_segments=1, damping=damping)
    output = torch.empty((1, dof_count))
    globalization_succeeded = torch.empty(1, dtype=torch.bool)

    _solve(
        solver,
        torch.zeros_like(output),
        output,
        torch.tensor((0, 1), dtype=torch.int32),
        torch.ones(1),
        torch.ones(1),
        torch.zeros((3, 1)),
        coordinate_bounds=IKTrajectorySolver.CoordinateBounds(
            coordinate_indices=torch.arange(dof_count, dtype=torch.int32),
            dof_indices=torch.arange(dof_count, dtype=torch.int32),
            lower=torch.full((dof_count,), -10.0),
            upper=torch.full((dof_count,), 10.0),
        ),
        segment_globalization_succeeded=globalization_succeeded,
    )

    torch.testing.assert_close(
        output,
        torch.full_like(output, 1.0 / (dof_count + damping)),
        atol=2.0e-6,
        rtol=0.0,
    )
    assert globalization_succeeded.item()


def test_coordinate_bound_uses_the_objective_metric() -> None:
    """Projected box iterations must minimize the objective metric, not clamp a coupled Newton step."""
    damping = 1.0e-6
    hessian = torch.tensor(((1.0, -2.0), (-2.0, 5.0)))
    jacobian = torch.linalg.cholesky(hessian - damping * torch.eye(2)).T.unsqueeze(0)
    gradient = torch.tensor((1.0, -3.0))
    target = torch.linalg.solve(jacobian[0].T, -gradient).unsqueeze(0)
    solver = _solver(_LinearOptimizer(jacobian, target), damping=damping)
    current = torch.zeros((1, 2))
    output = torch.empty_like(current)
    active = torch.ones(1, dtype=torch.int32)
    bounds = IKTrajectorySolver.CoordinateBounds(
        coordinate_indices=torch.tensor((1,), dtype=torch.int32),
        dof_indices=torch.tensor((1,), dtype=torch.int32),
        lower=torch.tensor((-torch.inf,)),
        upper=torch.zeros(1),
    )
    for _ in range(64):
        _solve(
            solver,
            current,
            output,
            torch.tensor((0, 1), dtype=torch.int32),
            torch.ones(1),
            torch.ones(2),
            torch.zeros((3, 2)),
            coordinate_bounds=bounds,
            segment_active=active,
            convergence_tolerance=1.0e-7,
        )
        current, output = output, current
        if not active[0]:
            break

    assert not active[0]
    torch.testing.assert_close(
        current,
        torch.tensor(((-1.0, 0.0),)),
        atol=solver.kkt_relative_tolerance,
        rtol=0.0,
    )


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
@pytest.mark.parametrize("lower_x", (-torch.inf, 0.0), ids=("one_sided", "locked"))
def test_coordinate_bound_and_residual_inequality_share_one_ipm_family(device_type: str, lower_x: float) -> None:
    """A bound-active objective direction must use the remaining clearance tangent."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_type)
    optimizer = _LinearOptimizer(
        torch.tensor([[[1.0, 0.0], [-1.0, -1.0]]], device=device),
        torch.tensor([[2.0, -1.0]], device=device),
    )
    solver = _solver(optimizer, damping=1.0e-6, kkt_relative_tolerance=1.0e-6)
    initial = torch.zeros((1, 2), device=device)
    output = torch.empty_like(initial)
    feasible = torch.empty(1, dtype=torch.bool, device=device)
    _solve(
        solver,
        initial,
        output,
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.ones(1, device=device),
        torch.tensor([1.0, 0.0], device=device),
        torch.zeros((3, 2), device=device),
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor([1], dtype=torch.int32, device=device),
            upper=torch.zeros(1, device=device),
        ),
        coordinate_bounds=IKTrajectorySolver.CoordinateBounds(
            coordinate_indices=torch.tensor([0], dtype=torch.int32, device=device),
            dof_indices=torch.tensor([0], dtype=torch.int32, device=device),
            lower=torch.tensor([lower_x], device=device),
            upper=torch.tensor([0.0], device=device),
        ),
        segment_feasible=feasible,
    )
    residuals, _ = optimizer.linearize(wp.from_torch(output))

    assert feasible.item()
    torch.testing.assert_close(output, torch.tensor([[0.0, 1.0]], device=device), atol=2.0e-5, rtol=0.0)
    assert wp.to_torch(residuals)[0, 1] <= torch.finfo(torch.float32).eps


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_locked_coordinate_bounds_report_residual_inequality_infeasibility(device_type: str) -> None:
    """A residual constraint with no remaining tangent must report geometric infeasibility."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_type)
    optimizer = _LinearOptimizer(
        torch.tensor([[[1.0, 0.0], [-1.0, -1.0]]], device=device),
        torch.tensor([[2.0, -1.0]], device=device),
    )
    solver = _solver(optimizer, damping=1.0e-6)
    initial = torch.zeros((1, 2), device=device)
    output = torch.empty_like(initial)
    feasible = torch.empty(1, dtype=torch.bool, device=device)
    _solve(
        solver,
        initial,
        output,
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.ones(1, device=device),
        torch.tensor([1.0, 0.0], device=device),
        torch.zeros((3, 2), device=device),
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor([1], dtype=torch.int32, device=device),
            upper=torch.zeros(1, device=device),
        ),
        coordinate_bounds=IKTrajectorySolver.CoordinateBounds(
            coordinate_indices=torch.tensor([0, 1], dtype=torch.int32, device=device),
            dof_indices=torch.tensor([0, 1], dtype=torch.int32, device=device),
            lower=torch.zeros(2, device=device),
            upper=torch.zeros(2, device=device),
        ),
        segment_feasible=feasible,
    )

    assert not feasible.item()
    torch.testing.assert_close(output, initial)


@pytest.mark.parametrize(
    ("coordinate_indices", "dof_indices", "lower", "upper", "initial"),
    (
        ([0, 0], [0, 1], [-torch.inf, -torch.inf], [0.0, 0.0], [0.0, 0.0]),
        ([0, 1], [0, 0], [-torch.inf, -torch.inf], [0.0, 0.0], [0.0, 0.0]),
        ([2], [0], [-torch.inf], [0.0], [0.0, 0.0]),
        ([0], [2], [-torch.inf], [0.0], [0.0, 0.0]),
        ([0], [0], [-torch.inf], [torch.inf], [0.0, 0.0]),
        ([0], [0], [torch.nan], [0.0], [0.0, 0.0]),
        ([0], [0], [1.0], [0.0], [0.0, 0.0]),
        ([0], [0], [-1.0], [0.0], [1.0, 0.0]),
        ([0], [0], [0.0], [0.0], [torch.finfo(torch.float32).eps, 0.0]),
    ),
    ids=(
        "duplicate_coordinate",
        "duplicate_dof",
        "coordinate_out_of_range",
        "dof_out_of_range",
        "unbounded",
        "nan",
        "reversed",
        "initially_outside",
        "locked_not_exact",
    ),
)
def test_coordinate_bound_contract_rejects_invalid_input(
    coordinate_indices: list[int],
    dof_indices: list[int],
    lower: list[float],
    upper: list[float],
    initial: list[float],
) -> None:
    optimizer = _LinearOptimizer(torch.zeros((1, 1, 2)), torch.zeros((1, 1)))
    solver = _solver(optimizer)
    joint_q = torch.tensor([initial])
    with pytest.raises(RuntimeError, match="Coordinate bounds require unique in-range scalar coordinate/DOF pairs"):
        _solve(
            solver,
            joint_q,
            torch.empty_like(joint_q),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.ones(1),
            torch.ones(1),
            torch.zeros((3, 1)),
            coordinate_bounds=IKTrajectorySolver.CoordinateBounds(
                coordinate_indices=torch.tensor(coordinate_indices, dtype=torch.int32),
                dof_indices=torch.tensor(dof_indices, dtype=torch.int32),
                lower=torch.tensor(lower),
                upper=torch.tensor(upper),
            ),
        )


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
@pytest.mark.parametrize("face", ("lower", "upper"))
def test_coordinate_bound_saturated_output_is_exactly_reusable(device_type: str, face: str) -> None:
    """A solver output on a box face must satisfy its next solve's strict input contract."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_type)
    direction = -1.0 if face == "lower" else 1.0
    bound = torch.tensor(direction * 0.8726699948310852, device=device)
    target_coordinate = torch.nextafter(bound, torch.tensor(direction * torch.inf, device=device))
    optimizer = _LinearOptimizer(
        torch.full((1, 1, 1), 100.0, device=device),
        (100.0 * target_coordinate).reshape(1, 1),
    )
    solver = _solver(optimizer, damping=1.0e-6)
    initial = bound.reshape(1, 1)
    first = torch.empty_like(initial)
    second = torch.empty_like(initial)
    coordinate_bounds = IKTrajectorySolver.CoordinateBounds(
        coordinate_indices=torch.tensor([0], dtype=torch.int32, device=device),
        dof_indices=torch.tensor([0], dtype=torch.int32, device=device),
        lower=bound.reshape(1) if face == "lower" else torch.tensor([-1.0], device=device),
        upper=torch.tensor([1.0], device=device) if face == "lower" else bound.reshape(1),
    )
    solve_args = (
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.ones(1, device=device),
        torch.ones(1, device=device),
        torch.zeros((3, 1), device=device),
    )

    _solve(solver, initial, first, *solve_args, coordinate_bounds=coordinate_bounds)
    _solve(solver, first, second, *solve_args, coordinate_bounds=coordinate_bounds)

    assert torch.equal(first, initial)
    assert torch.equal(second, initial)


def test_constrained_coordinate_trials_are_projected_before_certification() -> None:
    """All coordinate-box trials canonicalize roundoff before hard-bound certification."""
    import inspect

    solve_source = inspect.getsource(IKTrajectorySolver.solve)
    projection = solve_source.index("_coordinate_bounds_project_candidate")
    projection_gate = solve_source.rfind("if ", 0, projection)
    projected_only = solve_source.index("if projected_bounds:", projection)
    hard_bound_measure = solve_source.index("self._hard_bound_measure(", projected_only)
    projection_block = solve_source[projection_gate:projected_only]

    assert projection_block.startswith("if bound_count > 0:")
    assert "_velocity_candidate" in projection_block
    assert projection < projected_only < hard_bound_measure


def test_generalized_velocity_qp_enforces_hard_edge_bounds() -> None:
    frame_count = 6
    target = torch.tensor([[0.0], [8.0], [-8.0], [8.0], [-8.0], [8.0]])
    optimizer = _LinearOptimizer(torch.ones((frame_count, 1, 1)), target)
    solver = _solver(optimizer, damping=1.0e-5)
    initial = torch.zeros((frame_count, 1))
    initial_velocity = torch.zeros_like(initial)
    output = torch.empty_like(initial)
    _solve(
        solver,
        initial,
        output,
        torch.tensor([0, frame_count], dtype=torch.int32),
        torch.ones(1),
        torch.ones(1),
        torch.zeros((3, 1)),
        joint_velocity=initial_velocity,
        velocity_lower=torch.tensor([-0.5]),
        velocity_upper=torch.tensor([0.5]),
    )
    edge_velocity = torch.diff(output[:, 0])
    assert torch.all(edge_velocity >= -0.5 - 1.0e-6)
    assert torch.all(edge_velocity <= 0.5 + 1.0e-6)


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_generalized_velocity_qp_restores_infeasible_initial_edges(device_type: str) -> None:
    """Phase I restores finite initial edge rates that exceed the declared bounds."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_type)
    initial = torch.tensor(((0.0,), (2.0,), (4.0,)), device=device)
    optimizer = _LinearOptimizer(torch.ones((3, 1, 1), device=device), initial.clone())
    solver = _solver(optimizer, max_segments=1, damping=1.0e-5)
    output = torch.empty_like(initial)
    segment_feasible = torch.empty(1, dtype=torch.bool, device=device)
    segment_direction_valid = torch.empty(1, dtype=torch.bool, device=device)

    _solve(
        solver,
        initial,
        output,
        torch.tensor((0, 3), dtype=torch.int32, device=device),
        torch.ones(1, device=device),
        torch.ones(1, device=device),
        torch.zeros((3, 1), device=device),
        joint_velocity=torch.tensor(((0.0,), (2.0,), (2.0,)), device=device),
        velocity_lower=torch.tensor((-0.5,), device=device),
        velocity_upper=torch.tensor((0.5,), device=device),
        segment_feasible=segment_feasible,
        segment_direction_valid=segment_direction_valid,
    )

    edge_velocity = torch.diff(output[:, 0])
    assert segment_feasible.item()
    assert segment_direction_valid.item()
    assert torch.all(edge_velocity >= -0.5 - 1.0e-6)
    assert torch.all(edge_velocity <= 0.5 + 1.0e-6)


def test_line_search_rejects_extrapolation_that_can_break_velocity_bounds() -> None:
    """Line-search factors cannot extrapolate beyond the exact bound-constrained QP step."""
    optimizer = _LinearOptimizer(torch.ones((2, 1, 1)), torch.tensor(((0.0,), (2.0,))))

    with pytest.raises(ValueError, match="at most one"):
        IKTrajectorySolver(
            optimizer,
            max_segments=1,
            max_equality_residuals_per_frame=0,
            damping=1.0e-6,
            line_search_steps=(2.0, 1.0),
        )

    with pytest.raises(ValueError, match="descending"):
        IKTrajectorySolver(
            optimizer,
            max_segments=1,
            max_equality_residuals_per_frame=0,
            damping=1.0e-6,
            line_search_steps=(1.0, 0.25, 0.5),
        )


def test_velocity_bound_uses_the_objective_metric() -> None:
    """A velocity-bound LM step must minimize the joint model, not causally clip the free step."""
    optimizer = _LinearOptimizer(
        torch.tensor((([1.5], [1.0], [0.0]), ([1.0], [0.0], [1.0]))),
        torch.tensor(((1.5, 9.0 / 7.0, 0.0), (2.0, 0.0, 4.0))),
    )
    solver = _solver(optimizer, damping=1.0 / 300.0)
    initial = torch.zeros((2, 1))
    output = torch.empty_like(initial)
    temporal_weights = torch.zeros((3, 3))
    temporal_weights[0, 0] = 149.0 / 150.0

    _solve(
        solver,
        initial,
        output,
        torch.tensor((0, 2), dtype=torch.int32),
        torch.ones(1),
        torch.tensor((0.0, 7.0 / 600.0, 1.0 / 300.0)),
        temporal_weights,
        joint_velocity=torch.zeros_like(initial),
        velocity_lower=torch.full((1,), -torch.inf),
        velocity_upper=torch.zeros(1),
    )

    torch.testing.assert_close(output, torch.full_like(output, -22.0 / 27.0), atol=3.0e-5, rtol=0.0)


def test_active_prefix_does_not_write_workspace_tail() -> None:
    optimizer = _LinearOptimizer(torch.ones((12, 1, 1)), torch.ones((12, 1)))
    solver = _solver(optimizer)
    solver._joint_q[5:].fill_(123.0)
    output = torch.empty((5, 1))
    _solve(
        solver,
        torch.zeros_like(output),
        output,
        torch.tensor([0, 5], dtype=torch.int32),
        torch.ones(1),
        torch.ones(1),
        torch.zeros((3, 1)),
    )
    torch.testing.assert_close(solver._joint_q[5:], torch.full_like(solver._joint_q[5:], 123.0))


def test_nonlinear_line_search_never_increases_full_objective() -> None:
    optimizer = _QuadraticOptimizer(4, target=4.0, device=torch.device("cpu"))
    solver = _solver(optimizer, damping=1.0e-6)
    initial = torch.full((4, 1), 0.1)
    output = torch.empty_like(initial)
    initial_cost = torch.sum((initial.square() - 4.0).square())
    _solve(
        solver,
        initial,
        output,
        torch.tensor([0, 4], dtype=torch.int32),
        torch.ones(1),
        torch.ones(1),
        torch.zeros((3, 1)),
    )
    final_cost = torch.sum((output.square() - 4.0).square())
    assert final_cost <= initial_cost


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_line_search_accepted_trial_stops_without_rebuilding_jacobian(device_type: str) -> None:
    """An accepted trial computes residuals once and preserves the outer Jacobian."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_type)
    optimizer = _QuadraticOptimizer(4, target=0.04, device=device)
    solver = _solver(optimizer, damping=1.0e-6)
    initial = torch.full((4, 1), 0.1, device=device)
    output = torch.empty_like(initial)
    _solve(
        solver,
        initial,
        output,
        torch.tensor([0, 4], dtype=torch.int32, device=device),
        torch.ones(1, device=device),
        torch.ones(1, device=device),
        torch.zeros((3, 1), device=device),
    )

    assert optimizer.linearize_calls == 1
    assert optimizer.residual_calls == 1
    torch.testing.assert_close(optimizer.jacobian, 2.0 * initial.unsqueeze(-1))


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_constrained_convergence_waits_for_independent_pose_equalities(device_type: str) -> None:
    """Flat soft cost cannot deactivate a segment before its six pose rows are feasible."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_type)
    optimizer = _NonlinearPoseEqualityOptimizer(1, device)
    solver = _solver(optimizer, max_segments=1, max_equality_residuals_per_frame=6, damping=1.0e-6)
    equalities = IKTrajectorySolver.ResidualEqualities(
        active=torch.ones((1, 2), dtype=torch.uint8, device=device),
        residual_starts_by_target=torch.tensor((0, 3), dtype=torch.int32, device=device),
    )
    joint_q = torch.tensor(((0.5, 0.5, 0.5, 0.0, 0.0, 0.0),), dtype=torch.float32, device=device)
    output = torch.empty_like(joint_q)
    segment_active = torch.ones(1, dtype=torch.int32, device=device)
    for _ in range(8):
        _solve(
            solver,
            joint_q,
            output,
            torch.tensor((0, 1), dtype=torch.int32, device=device),
            torch.ones(1, device=device),
            torch.zeros(6, device=device),
            torch.zeros((3, 6), device=device),
            equalities=equalities,
            segment_active=segment_active,
            convergence_tolerance=1.0e-8,
        )
        joint_q.copy_(output)
        if segment_active[0] == 0:
            break

    residuals, _ = optimizer.linearize(wp.from_torch(joint_q))
    assert torch.max(torch.abs(wp.to_torch(residuals))) <= 32.0 * torch.finfo(torch.float32).eps
    assert segment_active[0] == 0


def test_rejected_material_direction_reports_globalization_failure() -> None:
    """Exhausting all trial scales stops instead of repeating an unchanged material direction."""
    optimizer = _QuadraticOptimizer(1, target=4.0, device=torch.device("cpu"))
    solver = _solver(optimizer, max_segments=1, damping=1.0e-6)
    joint_q = torch.tensor(((0.01,),), dtype=torch.float32)
    output = torch.empty_like(joint_q)
    segment_active = torch.ones(1, dtype=torch.int32)
    linear_converged = torch.empty(1, dtype=torch.bool)
    globalization_succeeded = torch.empty(1, dtype=torch.bool)

    _solve(
        solver,
        joint_q,
        output,
        torch.tensor((0, 1), dtype=torch.int32),
        torch.ones(1),
        torch.ones(1),
        torch.zeros((3, 1)),
        segment_active=segment_active,
        segment_direction_valid=linear_converged,
        segment_globalization_succeeded=globalization_succeeded,
        convergence_tolerance=1.0e-6,
    )

    torch.testing.assert_close(output, joint_q)
    assert torch.max(torch.abs(solver._delta)) > 1.0
    assert linear_converged[0]
    assert not globalization_succeeded[0]
    assert not segment_active[0]


def test_small_weighted_objective_is_model_stationary_without_moving() -> None:
    """A ray improvement below the configured objective tolerance does not move."""
    optimizer = _LinearOptimizer(torch.tensor([[[1.0e-3]]]), torch.tensor([[1.0e-4]]))
    solver = _solver(optimizer, max_segments=1, damping=1.0e-12)
    joint_q = torch.zeros((1, 1))
    output = torch.empty_like(joint_q)
    segment_active = torch.ones(1, dtype=torch.int32)

    _solve(
        solver,
        joint_q,
        output,
        torch.tensor((0, 1), dtype=torch.int32),
        torch.ones(1),
        torch.ones(1),
        torch.zeros((3, 1)),
        segment_active=segment_active,
        convergence_tolerance=1.0e-6,
    )

    torch.testing.assert_close(output, joint_q)
    assert not segment_active[0]


def test_model_stationary_small_step_does_not_materialize_a_candidate() -> None:
    """A stationary model direction converges at the current state."""
    optimizer = _LinearOptimizer(torch.tensor([[[0.0], [1.0]]]), torch.tensor([[1.0, 5.0e-4]]))
    solver = _solver(optimizer, max_segments=1, damping=0.1)
    joint_q = torch.zeros((1, 1))
    output = torch.empty_like(joint_q)
    segment_active = torch.ones(1, dtype=torch.int32)

    _solve(
        solver,
        joint_q,
        output,
        torch.tensor((0, 1), dtype=torch.int32),
        torch.ones(1),
        torch.ones(2),
        torch.zeros((3, 2)),
        segment_active=segment_active,
        convergence_tolerance=1.0e-6,
    )

    torch.testing.assert_close(output, joint_q)
    assert not segment_active[0]


@pytest.mark.parametrize("constrained", (False, True))
def test_linear_convergence_matches_final_preconditioned_residual(constrained: bool) -> None:
    """The caller mask certifies the final solve residual, not individual Krylov iteration caps."""
    jacobian = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.01, 0.0], [0.0, 1.0, 0.01], [0.0, 0.0, 1.0]]])
    optimizer = _LinearOptimizer(jacobian, torch.ones((1, 4)))
    solver = IKTrajectorySolver(
        optimizer,
        max_segments=1,
        max_equality_residuals_per_frame=0,
        damping=1.0e-6,
        krylov_max_iterations=1,
        krylov_relative_tolerance=1.0e-6,
        kkt_relative_tolerance=1.0e-6,
    )
    joint_q = torch.zeros((1, 3))
    output = torch.empty_like(joint_q)
    segment_active = torch.ones(1, dtype=torch.int32)
    linear_converged = torch.empty(1, dtype=torch.bool)
    coordinate_bounds = (
        IKTrajectorySolver.CoordinateBounds(
            coordinate_indices=torch.tensor((0,), dtype=torch.int32),
            dof_indices=torch.tensor((0,), dtype=torch.int32),
            lower=torch.tensor((-100.0,)),
            upper=torch.tensor((100.0,)),
        )
        if constrained
        else IKTrajectorySolver.CoordinateBounds(
            coordinate_indices=torch.empty(0, dtype=torch.int32),
            dof_indices=torch.empty(0, dtype=torch.int32),
            lower=torch.empty(0),
            upper=torch.empty(0),
        )
    )
    velocity_lower = torch.full((3,), -100.0 if constrained else -torch.inf)
    velocity_upper = torch.full((3,), 100.0 if constrained else torch.inf)

    _solve(
        solver,
        joint_q,
        output,
        torch.tensor((0, 1), dtype=torch.int32),
        torch.ones(1),
        torch.ones(4),
        torch.zeros((3, 4)),
        coordinate_bounds=coordinate_bounds,
        segment_active=segment_active,
        velocity_lower=velocity_lower,
        velocity_upper=velocity_upper,
        segment_direction_valid=linear_converged,
        convergence_tolerance=1.0e6,
    )

    normal = jacobian[0].T @ jacobian[0] + 1.0e-6 * torch.eye(3)
    right_hand_side = jacobian[0].T @ torch.ones(4)
    residual = right_hand_side - normal @ solver._delta[0]
    initial_norm = torch.sqrt(torch.sum(right_hand_side.square() / torch.diagonal(normal)))
    final_norm = torch.sqrt(torch.sum(residual.square() / torch.diagonal(normal)))
    expected = bool(final_norm <= max(torch.finfo(torch.float32).eps, 1.0e-6 * initial_norm))

    assert expected is constrained
    assert linear_converged.item() is expected
    assert not segment_active.item()
    if not expected:
        torch.testing.assert_close(output, joint_q)


@pytest.mark.parametrize("feasibility_only", (False, True))
def test_ipm_phase_two_handoff_preserves_only_certified_primals(
    monkeypatch: pytest.MonkeyPatch, feasibility_only: bool
) -> None:
    """Phase II preserves certified Phase-I witnesses and resets inconclusive segments."""
    optimizer = _LinearOptimizer(torch.ones((2, 2, 1)), torch.zeros((2, 2)))
    solver = _solver(optimizer, max_segments=2, max_equality_residuals_per_frame=0)
    output = torch.empty((2, 1))
    observed = []
    observed_f64 = []
    observed_enabled = []
    observed_diagonal = []
    phase_two_diagonal = []
    phase_one_rows = []
    phase_two_rows = []
    phase_one_finished = False

    def seed_phase_one(frame_count: int, segment_count: int) -> None:
        nonlocal phase_one_finished
        phase_two_diagonal.append(solver._normal_diagonal[:frame_count].clone())
        phase_one_rows.append(
            (
                solver._ipm_constraint_scale[:frame_count, : solver.inequality_width].clone(),
                solver._ipm_constraint_rhs[:frame_count, : solver.inequality_width].clone(),
            )
        )
        solver._ipm_primal[:frame_count].copy_(torch.tensor(((0.25,), (0.75,))))
        solver._candidate_bound_violation[:segment_count].copy_(torch.tensor((0.0, 1.0)))
        solver._normal_diagonal[:frame_count].fill_(123.0)
        solver._ipm_enabled[:segment_count].copy_(solver._segment_active[:segment_count])
        solver._phase_one_witness_selected[:segment_count].copy_(torch.tensor((1, 0), dtype=torch.int32))
        phase_one_finished = True

    original_apply = solver._ipm_primal_constraints_apply_f64

    def observe_phase_two(frame_count: int) -> None:
        if phase_one_finished:
            observed.append(solver._ipm_primal[:frame_count].clone())
            observed_f64.append(solver._ipm_primal_f64[:frame_count].clone())
            observed_enabled.append(solver._ipm_enabled.clone())
            observed_diagonal.append(solver._normal_diagonal[:frame_count].clone())
            phase_two_rows.append(
                (
                    solver._ipm_constraint_scale[:frame_count, : solver.inequality_width].clone(),
                    solver._ipm_constraint_rhs[:frame_count, : solver.inequality_width].clone(),
                )
            )
            raise RuntimeError("observed Phase-II initial primal")
        original_apply(frame_count)

    monkeypatch.setattr(solver, "_solve_ipm_phase_one", seed_phase_one)
    monkeypatch.setattr(solver, "_ipm_primal_constraints_apply_f64", observe_phase_two)
    with pytest.raises(RuntimeError, match="observed Phase-II initial primal"):
        _solve(
            solver,
            torch.zeros_like(output),
            output,
            torch.tensor((0, 1, 2), dtype=torch.int32),
            torch.ones(2),
            torch.tensor((1.0, 0.0)),
            torch.zeros((3, 2)),
            inequalities=IKTrajectorySolver.ResidualInequalities(
                residual_indices=torch.tensor((1,), dtype=torch.int32),
                upper=torch.ones(1),
            ),
            feasibility_only=feasibility_only,
        )

    expected = torch.tensor(((0.25,), (0.0,)))
    torch.testing.assert_close(observed[0], expected)
    torch.testing.assert_close(observed_f64[0], expected.to(torch.float64))
    expected_enabled = torch.tensor((0, 1) if feasibility_only else (1, 1), dtype=torch.int32)
    torch.testing.assert_close(observed_enabled[0], expected_enabled)
    torch.testing.assert_close(observed_diagonal[0], phase_two_diagonal[0])
    if feasibility_only:
        torch.testing.assert_close(phase_two_rows[0][0][1], phase_one_rows[0][0][1])
        torch.testing.assert_close(phase_two_rows[0][1][1], phase_one_rows[0][1][1])
    else:
        torch.testing.assert_close(phase_two_rows[0][0], phase_one_rows[0][0])
        torch.testing.assert_close(phase_two_rows[0][1], phase_one_rows[0][1])


def test_ipm_phase_one_skips_directions_when_zero_is_feasible(monkeypatch: pytest.MonkeyPatch) -> None:
    """A certified feasible zero step enters Phase II without solving the auxiliary LP."""
    optimizer = _LinearOptimizer(torch.ones((1, 1, 1)), torch.tensor([[2.0]]))
    solver = _solver(optimizer, damping=1.0e-6)
    output = torch.empty((1, 1))
    feasible = torch.empty(1, dtype=torch.bool)
    linear_converged = torch.empty(1, dtype=torch.bool)
    direction_calls = 0
    solve_direction = solver._solve_phase_one_direction

    def count_direction(frame_count: int, segment_count: int, *, rebuild_factor: bool = True) -> None:
        nonlocal direction_calls
        direction_calls += 1
        solve_direction(frame_count, segment_count, rebuild_factor=rebuild_factor)

    monkeypatch.setattr(solver, "_solve_phase_one_direction", count_direction)
    _solve(
        solver,
        torch.zeros_like(output),
        output,
        torch.tensor((0, 1), dtype=torch.int32),
        torch.ones(1),
        torch.ones(1),
        torch.zeros((3, 1)),
        coordinate_bounds=IKTrajectorySolver.CoordinateBounds(
            coordinate_indices=torch.tensor((0,), dtype=torch.int32),
            dof_indices=torch.tensor((0,), dtype=torch.int32),
            lower=torch.tensor((-1.0,)),
            upper=torch.tensor((1.0,)),
        ),
        segment_feasible=feasible,
        velocity_lower=torch.tensor((-100.0,)),
        velocity_upper=torch.tensor((100.0,)),
        segment_direction_valid=linear_converged,
    )

    assert direction_calls == 0
    assert feasible.item()
    assert linear_converged.item()
    assert 0.99 <= output.item() <= 1.0

    mixed_optimizer = _LinearOptimizer(
        torch.ones((2, 2, 1)),
        torch.tensor(((0.0, 0.0), (-1.0, -2.0))),
    )
    mixed_solver = _solver(mixed_optimizer, max_segments=2, damping=1.0e-6)
    mixed_output = torch.empty((2, 1))
    mixed_feasible = torch.empty(2, dtype=torch.bool)
    mixed_linear_converged = torch.empty(2, dtype=torch.bool)
    mixed_direction_calls = 0
    mixed_solve_direction = mixed_solver._solve_phase_one_direction

    def count_mixed_direction(frame_count: int, segment_count: int, *, rebuild_factor: bool = True) -> None:
        nonlocal mixed_direction_calls
        mixed_direction_calls += 1
        mixed_solve_direction(frame_count, segment_count, rebuild_factor=rebuild_factor)

    monkeypatch.setattr(mixed_solver, "_solve_phase_one_direction", count_mixed_direction)
    _solve(
        mixed_solver,
        torch.zeros_like(mixed_output),
        mixed_output,
        torch.tensor((0, 1, 2), dtype=torch.int32),
        torch.ones(2),
        torch.tensor((1.0, 0.0)),
        torch.zeros((3, 2)),
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor((1,), dtype=torch.int32),
            upper=torch.ones(1),
        ),
        segment_feasible=mixed_feasible,
        segment_direction_valid=mixed_linear_converged,
    )

    assert mixed_direction_calls > 0
    assert torch.all(mixed_feasible)
    assert torch.all(mixed_linear_converged)
    torch.testing.assert_close(mixed_output, torch.tensor(((0.0,), (-1.0,))), atol=2.0e-4, rtol=0.0)


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_ipm_phase_one_stops_when_every_segment_is_converged(
    monkeypatch: pytest.MonkeyPatch,
    device_type: str,
) -> None:
    """A converged feasibility search does not execute the fixed no-op IPM tail."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_type)
    optimizer = _LinearOptimizer(
        torch.ones((1, 2, 1), device=device),
        torch.tensor(((-1.0, -2.0),), device=device),
    )
    solver = _solver(optimizer, damping=1.0e-6)
    direction_calls = 0
    solve_direction = solver._solve_phase_one_direction

    def count_direction(frame_count: int, segment_count: int, *, rebuild_factor: bool = True) -> None:
        nonlocal direction_calls
        direction_calls += 1
        solve_direction(frame_count, segment_count, rebuild_factor=rebuild_factor)

    convergence_mask = solver._ipm_convergence_mask

    def complete_convergence(
        frame_count: int,
        segment_count: int,
        *,
        include_phase_one_scalar: bool,
    ) -> None:
        convergence_mask(
            frame_count,
            segment_count,
            include_phase_one_scalar=include_phase_one_scalar,
        )
        solver._ipm_enabled[:segment_count].zero_()

    class PhaseOneComplete(RuntimeError):
        pass

    solve_phase_one = solver._solve_ipm_phase_one

    def stop_after_phase_one(frame_count: int, segment_count: int) -> None:
        solve_phase_one(frame_count, segment_count)
        raise PhaseOneComplete

    monkeypatch.setattr(solver, "_solve_phase_one_direction", count_direction)
    monkeypatch.setattr(solver, "_ipm_convergence_mask", complete_convergence)
    monkeypatch.setattr(solver, "_solve_ipm_phase_one", stop_after_phase_one)
    output = torch.empty((1, 1), device=device)
    with pytest.raises(PhaseOneComplete):
        _solve(
            solver,
            torch.zeros_like(output),
            output,
            torch.tensor((0, 1), dtype=torch.int32, device=device),
            torch.ones(1, device=device),
            torch.tensor((1.0, 0.0), device=device),
            torch.zeros((3, 2), device=device),
            inequalities=IKTrajectorySolver.ResidualInequalities(
                residual_indices=torch.tensor((1,), dtype=torch.int32, device=device),
                upper=torch.ones(1, device=device),
            ),
        )

    assert direction_calls == 0


def test_ipm_phase_one_certifies_the_final_allowed_iterate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase I measures the iterate produced by its last predictor-corrector step."""
    optimizer = _LinearOptimizer(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.9]]]),
        torch.tensor([[2.5, 2.4, 0.0, 0.0]]),
    )
    solver = _solver(optimizer, damping=1.0e-6)
    original_phase_one = solver._solve_ipm_phase_one

    class PhaseOneComplete(RuntimeError):
        pass

    def stop_after_phase_one(frame_count: int, segment_count: int) -> None:
        original_phase_one(frame_count, segment_count)
        raise PhaseOneComplete

    monkeypatch.setattr(trajectory_module, "_IPM_ITERATIONS", 4)
    monkeypatch.setattr(solver, "_solve_ipm_phase_one", stop_after_phase_one)
    with pytest.raises(PhaseOneComplete):
        _solve(
            solver,
            torch.zeros((1, 2)),
            torch.empty((1, 2)),
            torch.tensor((0, 1), dtype=torch.int32),
            torch.ones(1),
            torch.tensor((1.0, 1.0, 0.0, 0.0)),
            torch.zeros((3, 4)),
            inequalities=IKTrajectorySolver.ResidualInequalities(
                residual_indices=torch.tensor((2, 3), dtype=torch.int32),
                upper=torch.tensor((-1.0, -0.95)),
            ),
        )

    assert solver._segment_feasible[0] == 1
    assert solver._ipm_linear_converged[0] == 1
    assert solver._ipm_enabled[0] == 1


def test_ipm_phase_one_feasible_primal_does_not_require_auxiliary_kkt_optimum() -> None:
    """A certified feasible primal advances even when the min-elastic KKT solve is unfinished."""
    wp.init()
    feasible = torch.ones(1, dtype=torch.uint8)
    linear_converged = torch.ones(1, dtype=torch.int32)
    phase_two_enabled = torch.ones(1, dtype=torch.int32)
    wp.launch(
        _phase_one_finalize,
        dim=1,
        inputs=[
            wp.from_torch(torch.zeros(1)),
            wp.from_torch(torch.zeros(1)),
            wp.from_torch(torch.zeros(1)),
            wp.from_torch(torch.tensor((0.015625,))),
            wp.from_torch(torch.zeros(1)),
            wp.from_torch(torch.ones(1, dtype=torch.int32)),
            wp.from_torch(torch.ones(1, dtype=torch.int32)),
            32.0 * torch.finfo(torch.float32).eps,
            1,
        ],
        outputs=[wp.from_torch(feasible), wp.from_torch(linear_converged), wp.from_torch(phase_two_enabled)],
        device="cpu",
    )

    assert feasible.item() == 1
    assert linear_converged.item() == 1
    assert phase_two_enabled.item() == 1


def test_ipm_phase_one_inconclusive_search_does_not_block_infeasible_start_phase_two() -> None:
    """Only a Farkas certificate may prevent the infeasible-start optimization solve."""
    wp.init()
    feasible = torch.ones(1, dtype=torch.uint8)
    linear_converged = torch.zeros(1, dtype=torch.int32)
    phase_two_enabled = torch.zeros(1, dtype=torch.int32)
    wp.launch(
        _phase_one_finalize,
        dim=1,
        inputs=[
            wp.from_torch(torch.tensor((0.1,))),
            wp.from_torch(torch.ones(1)),
            wp.from_torch(torch.ones(1)),
            wp.from_torch(torch.ones(1)),
            wp.from_torch(torch.zeros(1)),
            wp.from_torch(torch.ones(1, dtype=torch.int32)),
            wp.from_torch(torch.ones(1, dtype=torch.int32)),
            32.0 * torch.finfo(torch.float32).eps,
            1,
        ],
        outputs=[wp.from_torch(feasible), wp.from_torch(linear_converged), wp.from_torch(phase_two_enabled)],
        device="cpu",
    )

    assert feasible.item() == 1
    assert linear_converged.item() == 1
    assert phase_two_enabled.item() == 1


def test_ipm_phase_one_feasibility_materializes_original_rows_with_zero_elastic() -> None:
    """A large auxiliary elastic cannot erase an original-row violation during certification."""
    wp.init()
    stored_elastic = torch.tensor((1.0e8,))
    certification_elastic = torch.zeros_like(stored_elastic)
    constraint_value = torch.ones((1, 1))
    constraint_scale = torch.ones_like(constraint_value)
    frame_segment = torch.zeros(1, dtype=torch.int32)
    enabled = torch.ones(1, dtype=torch.int32)
    wp.launch(
        _phase_one_constraints_apply,
        dim=(1, 1),
        inputs=[
            wp.from_torch(torch.zeros((1, 1))),
            wp.from_torch(certification_elastic),
            wp.from_torch(constraint_scale),
            wp.from_torch(frame_segment),
            wp.from_torch(enabled),
            1,
            1,
        ],
        outputs=[wp.from_torch(constraint_value)],
        device="cpu",
    )
    violation = torch.zeros(1)
    wp.launch(
        _ipm_primal_feasibility_violation_max,
        dim=(1, 1),
        inputs=[
            wp.from_torch(constraint_value),
            wp.from_torch(torch.zeros_like(constraint_value)),
            wp.from_torch(constraint_scale),
            wp.from_torch(frame_segment),
            wp.from_torch(enabled),
            1,
        ],
        outputs=[wp.from_torch(violation)],
        device="cpu",
    )

    assert ((torch.ones(1) - stored_elastic) + stored_elastic).item() == 0.0
    torch.testing.assert_close(violation, torch.ones(1))


def test_ipm_phase_one_feasibility_is_invariant_to_row_scaling() -> None:
    """The Phase-I certificate reports raw constraint units after preconditioning."""
    wp.init()
    scale = torch.tensor(((1.0e-3,),))
    violation = torch.zeros(1)
    wp.launch(
        _ipm_primal_feasibility_violation_max,
        dim=(1, 1),
        inputs=[
            wp.from_torch(scale.clone()),
            wp.from_torch(torch.zeros_like(scale)),
            wp.from_torch(scale),
            wp.from_torch(torch.zeros(1, dtype=torch.int32)),
            wp.from_torch(torch.ones(1, dtype=torch.int32)),
            1,
        ],
        outputs=[wp.from_torch(violation)],
        device="cpu",
    )

    torch.testing.assert_close(violation, torch.ones(1))


def test_ipm_phase_one_equality_witness_uses_raw_row_units() -> None:
    """A scaled equality violation cannot be mistaken for a feasible raw equality."""
    wp.init()
    row_scale = torch.tensor(((1.0e-3,),))
    constraint_scale = torch.zeros((1, 3))
    constraint_rhs = torch.zeros_like(constraint_scale)
    constraint_count = torch.zeros(1, dtype=torch.int32)
    segment_offsets = torch.tensor((0, 1), dtype=torch.int32)
    enabled = torch.ones(1, dtype=torch.int32)
    wp.launch(
        _phase_one_constraints_initialize,
        dim=(1, 3),
        inputs=[
            wp.from_torch(torch.zeros((1, 1), dtype=torch.int32)),
            wp.from_torch(row_scale),
            wp.from_torch(torch.zeros((1, 1))),
            wp.from_torch(torch.ones(1, dtype=torch.int32)),
            wp.from_torch(torch.zeros(1, dtype=torch.int32)),
            wp.from_torch(segment_offsets),
            wp.from_torch(enabled),
            1,
            0,
        ],
        outputs=[
            wp.from_torch(constraint_scale),
            wp.from_torch(constraint_rhs),
            wp.from_torch(constraint_count),
        ],
        device="cpu",
    )
    torch.testing.assert_close(constraint_scale[0, :2], row_scale.expand(1, 2).flatten())

    raw_violation = torch.zeros(1)
    wp.launch(
        _phase_one_original_violation_max,
        dim=(1, 3),
        inputs=[
            wp.from_torch(torch.tensor(((4.0e-6, -4.0e-6, 0.0),))),
            wp.from_torch(constraint_rhs),
            wp.from_torch(constraint_scale),
            wp.from_torch(torch.zeros(1)),
            wp.from_torch(torch.zeros(1, dtype=torch.int32)),
            wp.from_torch(enabled),
            1,
        ],
        outputs=[wp.from_torch(raw_violation)],
        device="cpu",
    )
    phase_two_enabled = torch.ones(1, dtype=torch.int32)
    witness_selected = torch.zeros(1, dtype=torch.int32)
    wp.launch(
        _phase_one_witness_select,
        dim=1,
        inputs=[
            wp.from_torch(raw_violation),
            wp.from_torch(phase_two_enabled),
            wp.from_torch(torch.ones(1, dtype=torch.int32)),
            1.0e-5,
            1,
        ],
        outputs=[wp.from_torch(witness_selected)],
        device="cpu",
    )

    torch.testing.assert_close(raw_violation, torch.tensor((4.0e-3,)))
    assert witness_selected.item() == 0
    assert phase_two_enabled.item() == 1


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_ipm_phase_two_stops_when_every_segment_is_converged(
    monkeypatch: pytest.MonkeyPatch,
    device_type: str,
) -> None:
    """A converged equality QP does not execute the fixed no-op IPM tail."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_type)
    optimizer = _LinearOptimizer(torch.ones((1, 3, 1), device=device), torch.zeros((1, 3), device=device))
    solver = _solver(optimizer, max_segments=1, max_equality_residuals_per_frame=3)
    equalities = IKTrajectorySolver.ResidualEqualities(
        active=torch.ones((1, 1), dtype=torch.uint8, device=device),
        residual_starts_by_target=torch.tensor((0,), dtype=torch.int32, device=device),
    )
    solve_direction = solver._solve_ipm_direction
    direction_calls = 0

    def count_direction(
        frame_count: int,
        segment_count: int,
        has_equalities: bool,
        *,
        rebuild_factor: bool = True,
    ) -> None:
        nonlocal direction_calls
        direction_calls += 1
        solve_direction(frame_count, segment_count, has_equalities, rebuild_factor=rebuild_factor)

    monkeypatch.setattr(solver, "_solve_ipm_direction", count_direction)
    output = torch.empty((1, 1), device=device)
    _solve(
        solver,
        torch.zeros_like(output),
        output,
        torch.tensor((0, 1), dtype=torch.int32, device=device),
        torch.ones(1, device=device),
        torch.zeros(3, device=device),
        torch.zeros((3, 3), device=device),
        equalities=equalities,
    )

    assert direction_calls == 1
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_ipm_phase_two_convergence_uses_direct_primal_feasibility() -> None:
    """An inactive far inequality is certified from the primal, not a rounded slack identity."""
    solver = _solver(
        _LinearOptimizer(torch.zeros((1, 1, 1)), torch.zeros((1, 1))),
        max_segments=1,
        max_equality_residuals_per_frame=0,
    )
    solver._frame_segment[0] = 0
    solver._segment_offsets[:2].copy_(torch.tensor((0, 1), dtype=torch.int32))
    solver._segment_active[0] = 1
    solver._ipm_enabled[0] = 1
    solver._ipm_primal[0].zero_()
    solver._jacobian[0].zero_()
    solver._ipm_constraint_scale[0].zero_()
    solver._ipm_constraint_scale[0, 0] = 1.0
    solver._ipm_constraint_rhs[0].zero_()
    solver._ipm_constraint_rhs[0, 0] = 100_000.0
    solver._ipm_primal_residual[0].zero_()
    solver._ipm_primal_residual[0, 0] = 0.015625
    solver._delta_correction[0].zero_()
    solver._dual_z[0].zero_()

    solver._ipm_convergence_measure(frame_count=1, segment_count=1, include_phase_one_scalar=False)

    assert solver._candidate_violation[0] == 0.0


def test_ipm_temporal_split_certifies_third_difference_system() -> None:
    """Temporal split PCG certifies a coupled third-difference system within four iterations."""
    frame_count = 9
    damping = 1.0e-4
    optimizer = _LinearOptimizer(torch.ones((frame_count, 1, 1)), torch.zeros((frame_count, 1)))
    solver = _solver(
        optimizer,
        max_segments=1,
        max_equality_residuals_per_frame=0,
        damping=damping,
        krylov_max_iterations=4,
        krylov_relative_tolerance=1.0e-4,
    )
    difference = torch.zeros((frame_count - 3, frame_count))
    for row in range(frame_count - 3):
        difference[row, row : row + 4] = torch.tensor((-1.0, 3.0, -3.0, 1.0))
    normal = difference.T @ difference + damping * torch.eye(frame_count)
    right_hand_side = torch.tensor((1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0))

    solver._frame_segment[:frame_count].zero_()
    solver._segment_offsets[:2].copy_(torch.tensor((0, frame_count), dtype=torch.int32))
    solver._step_seconds[0] = 1.0
    solver._segment_active[0] = 1
    solver._ipm_enabled[0] = 1
    solver._kkt_enabled[0] = 1
    solver._right_hand_side[:frame_count, 0].copy_(right_hand_side)
    solver._base_weights[:frame_count].zero_()
    solver._temporal_weights.zero_()
    solver._temporal_weights[2, 0] = 1.0
    solver._precision_diagonal[:frame_count, 0].copy_(torch.diagonal(normal) - damping)
    solver._ipm_weights[:frame_count].zero_()
    solver._ipm_constraint_scale[:frame_count].zero_()

    solver._solve_ipm_pcg(frame_count=frame_count, segment_count=1)

    final_norm = torch.sqrt(solver._ipm_residual_dot_f64[0])
    threshold = max(
        torch.finfo(torch.float32).eps,
        solver.krylov_relative_tolerance * solver._ipm_initial_norm_f64[0],
    )
    assert final_norm <= threshold
    assert solver._minres_enabled[0] == 0
    assert solver._minres_failed[0] == 0


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_ipm_block_band_direction_matches_dense_lag_three_system(
    monkeypatch: pytest.MonkeyPatch, device_type: str
) -> None:
    """The no-equality direction is the exact dense lag-three solution, independent of the Krylov cap."""
    device = torch.device(device_type)
    frame_count = 7
    dof_count = 2
    residual_count = 3
    damping = 0.07
    step_seconds = 0.8
    jacobian = torch.tensor(
        [
            [[1.0, -0.5], [0.25, 1.5], [-1.0, 0.75]],
            [[0.5, 1.0], [-0.75, 0.5], [1.25, -0.25]],
            [[-1.5, 0.25], [0.5, -1.0], [0.75, 1.25]],
            [[0.75, -1.25], [1.0, 0.5], [-0.5, 1.5]],
            [[1.25, 0.5], [-1.5, 0.75], [0.25, -1.0]],
            [[-0.25, 1.5], [0.75, -1.25], [1.0, 0.5]],
            [[1.5, -0.75], [-0.5, 1.0], [-1.25, 0.25]],
        ],
        dtype=torch.float32,
        device=device,
    )
    base_weights = torch.tensor(
        [
            [0.5, 1.0, 1.5],
            [1.25, 0.75, 0.5],
            [0.8, 1.2, 0.6],
            [1.5, 0.4, 1.0],
            [0.7, 1.4, 0.9],
            [1.1, 0.6, 1.3],
            [0.9, 1.25, 0.75],
        ],
        dtype=torch.float32,
        device=device,
    )
    temporal_weights = torch.tensor([[0.3, 0.2, 0.4], [0.15, 0.25, 0.1], [0.08, 0.05, 0.12]], device=device)

    linear_tolerances: list[float] = []
    objective_builds = 0
    launch = wp.launch

    def record_linear_tolerance(kernel, *args, **kwargs):
        nonlocal objective_builds
        if kernel is trajectory_module._ipm_block_band_matrix_build:
            objective_builds += 1
        if kernel is trajectory_module._ipm_physical_convergence_update:
            linear_tolerances.append(kwargs["inputs"][2])
        return launch(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", record_linear_tolerance)
    solver = _solver(
        _LinearOptimizer(jacobian, torch.zeros((frame_count, residual_count))),
        max_segments=1,
        max_equality_residuals_per_frame=0,
        damping=damping,
        krylov_max_iterations=1,
        krylov_relative_tolerance=2.0e-5,
        kkt_relative_tolerance=0.25,
    )
    solver._frame_segment[:frame_count].zero_()
    solver._segment_offsets[:2].copy_(torch.tensor((0, frame_count), dtype=torch.int32, device=device))
    solver._step_seconds[0] = step_seconds
    solver._base_weights[:frame_count].copy_(base_weights)
    solver._temporal_weights.copy_(temporal_weights)
    solver._ipm_weights[:frame_count].zero_()
    solver._ipm_constraint_scale[:frame_count].zero_()
    solver._ipm_enabled[0] = 1
    solver._prepare_ipm_objective_block_band(frame_count, 1)
    objective_blocks = solver._ipm_objective_block_band[:frame_count].clone()
    objective_factor = solver._ipm_objective_block_band_factor[:frame_count].clone()

    size = frame_count * dof_count
    matrix = damping * torch.eye(size, dtype=torch.float64, device=device)
    for residual in range(residual_count):
        precision = torch.diag(base_weights[:, residual].double())
        for order in range(1, 4):
            coefficients = torch.tensor(
                ((-1.0, 1.0), (1.0, -2.0, 1.0), (-1.0, 3.0, -3.0, 1.0))[order - 1],
                dtype=torch.float64,
                device=device,
            )
            difference = torch.zeros((frame_count - order, frame_count), dtype=torch.float64, device=device)
            for row in range(frame_count - order):
                difference[row, row : row + order + 1] = coefficients
            precision += (
                temporal_weights[order - 1, residual].double()
                / step_seconds ** (2 * order)
                * (difference.T @ difference)
            )
        residual_jacobian = torch.zeros((frame_count, size), dtype=torch.float64, device=device)
        for frame in range(frame_count):
            residual_jacobian[frame, frame * dof_count : (frame + 1) * dof_count] = jacobian[frame, residual].double()
        matrix += residual_jacobian.T @ precision @ residual_jacobian

    for rebuild_factor, right_hand_side in (
        (True, torch.linspace(-1.5, 1.25, size, device=device).reshape(frame_count, dof_count)),
        (False, torch.linspace(0.75, -1.0, size, device=device).reshape(frame_count, dof_count)),
    ):
        solver._right_hand_side[:frame_count].copy_(right_hand_side)
        solver._ipm_right_hand_side_f64[:frame_count].copy_(right_hand_side.double())
        solver._delta_correction[:frame_count].copy_(right_hand_side)
        solver._solve_ipm_direction(
            frame_count=frame_count,
            segment_count=1,
            has_equalities=False,
            rebuild_factor=rebuild_factor,
        )
        expected = torch.linalg.solve(matrix, right_hand_side.flatten().double()).reshape(frame_count, dof_count)
        torch.testing.assert_close(solver._ipm_solution_f64[:frame_count], expected, atol=2.0e-6, rtol=2.0e-6)
        assert solver._minres_enabled[0] == 0
        assert solver._minres_failed[0] == 0
        assert solver._ipm_augmented_fallback[0] == 0
    assert objective_builds == 1
    assert torch.equal(solver._ipm_objective_block_band[:frame_count], objective_blocks)
    assert torch.equal(solver._ipm_objective_block_band_factor[:frame_count], objective_factor)
    assert linear_tolerances
    assert all(tolerance == solver.krylov_relative_tolerance for tolerance in linear_tolerances)


def _cached_scalar_ipm_solver() -> IKTrajectorySolver:
    """Return one prepared scalar Phase-II system with an exact objective cache."""
    optimizer = _LinearOptimizer(torch.zeros((1, 1, 1)), torch.zeros((1, 1)))
    solver = _solver(
        optimizer,
        max_segments=1,
        max_equality_residuals_per_frame=0,
        damping=1.0,
        krylov_max_iterations=8,
    )
    solver._frame_segment[0] = 0
    solver._segment_offsets[:2].copy_(torch.tensor((0, 1), dtype=torch.int32))
    solver._step_seconds[0] = 1.0
    solver._coordinate_bound_count = 0
    solver._base_weights.zero_()
    solver._temporal_weights.zero_()
    solver._ipm_weights.zero_()
    solver._ipm_constraint_scale.zero_()
    solver._right_hand_side[0, 0] = 2.0
    solver._ipm_right_hand_side_f64[0, 0] = 2.0
    solver._delta_correction[0, 0] = 2.0
    solver._ipm_enabled[0] = 1
    solver._prepare_ipm_objective_block_band(frame_count=1, segment_count=1)
    return solver


def test_ipm_cached_objective_factor_handles_full_factor_failure() -> None:
    """A failed changing barrier factor falls back through the immutable objective factor."""
    solver = _cached_scalar_ipm_solver()
    solver._ipm_objective_block_band[0, 0, 0, 0] = -1.0

    solver._solve_ipm_direction(frame_count=1, segment_count=1, has_equalities=False)

    torch.testing.assert_close(solver._ipm_solution_f64[0, 0], torch.tensor(2.0, dtype=torch.float64))
    assert solver._ipm_augmented_fallback[0] == 1
    assert solver._minres_enabled[0] == 0
    assert solver._minres_failed[0] == 0


def test_ipm_cached_objective_factor_handles_certificate_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """An uncertified direct solve retries through the same cached objective factor."""
    solver = _cached_scalar_ipm_solver()
    physical_certificate = solver._ipm_physical_certificate
    certificate_calls = 0

    def reject_first(enabled_wp: wp.array, frame_count: int, segment_count: int) -> None:
        nonlocal certificate_calls
        certificate_calls += 1
        if certificate_calls > 1:
            physical_certificate(enabled_wp, frame_count, segment_count)

    monkeypatch.setattr(solver, "_ipm_physical_certificate", reject_first)

    solver._solve_ipm_direction(frame_count=1, segment_count=1, has_equalities=False)

    torch.testing.assert_close(solver._ipm_solution_f64[0, 0], torch.tensor(2.0, dtype=torch.float64))
    assert certificate_calls == 2
    assert solver._ipm_augmented_fallback[0] == 1
    assert solver._minres_enabled[0] == 0
    assert solver._minres_failed[0] == 0


def _strict_physical_certificate_witness(
    operator_value: float, right_hand_side: float, stationarity: float
) -> IKTrajectorySolver:
    """Run the strict physical certificate for one scalar identity operator."""
    optimizer = _LinearOptimizer(torch.zeros((1, 1, 1)), torch.zeros((1, 1)))
    solver = _solver(
        optimizer,
        max_segments=1,
        max_equality_residuals_per_frame=0,
        damping=1.0,
        krylov_relative_tolerance=1.0e-4,
        kkt_relative_tolerance=1.0e-4,
    )
    solver._frame_segment[0] = 0
    solver._segment_offsets[:2].copy_(torch.tensor((0, 1), dtype=torch.int32))
    solver._step_seconds[0] = 1.0
    solver._base_weights.zero_()
    solver._temporal_weights.zero_()
    solver._ipm_weights.zero_()
    solver._ipm_constraint_scale.zero_()
    solver._ipm_solution_f64[0, 0] = operator_value
    solver._ipm_right_hand_side_f64[0, 0] = right_hand_side
    solver._delta_correction[0, 0] = stationarity
    solver._kkt_enabled[0] = 1
    solver._minres_enabled[0] = 1
    solver._minres_failed[0] = 0

    solver._ipm_strict_physical_certificate(
        solver._wp_ipm_solution_f64, solver._wp_minres_enabled, frame_count=1, segment_count=1
    )
    return solver


def test_ipm_physical_certificate_rejects_cancellation_scaled_outer_residual() -> None:
    """A large symmetric reference cannot hide an order-one outer stationarity residual."""
    operator_value = 1.0e8
    right_hand_side = operator_value + 1.0
    solver = _strict_physical_certificate_witness(operator_value, right_hand_side, stationarity=1.0)

    assert solver.krylov_relative_tolerance * (abs(operator_value) + abs(right_hand_side)) >= 1.0
    assert solver._minres_enabled[0] == 1
    assert solver._minres_failed[0] == 0


def test_ipm_physical_certificate_accepts_direction_40_outer_residual() -> None:
    """The observed late-barrier direction remains below the outer KKT tolerance."""
    reference_norm = 51_717_061.9031
    residual = 2.7966e-5
    operator_value = 0.5 * (reference_norm - residual)
    right_hand_side = operator_value + residual
    solver = _strict_physical_certificate_witness(operator_value, right_hand_side, stationarity=1.0)

    assert abs(operator_value) + abs(right_hand_side) == pytest.approx(reference_norm)
    assert solver._minres_enabled[0] == 0
    assert solver._minres_failed[0] == 0


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_ipm_augmented_direction_matches_dense_high_range_system(device_type: str) -> None:
    """The full augmented solve handles dependent high-range rows without forming their normal matrix."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device(device_type)
    frame_count = 1
    dof_count = 2
    residual_count = 18
    damping = 1.0
    step_seconds = 1.0
    jacobian = torch.ones((frame_count, residual_count, dof_count), device=device)
    solver = _solver(
        _LinearOptimizer(jacobian, torch.zeros((frame_count, residual_count), device=device)),
        max_segments=1,
        max_equality_residuals_per_frame=0,
        damping=damping,
        krylov_max_iterations=8,
        krylov_relative_tolerance=1.0e-4,
    )
    solver._frame_segment[:frame_count].zero_()
    solver._segment_offsets[:2].copy_(torch.tensor((0, frame_count), dtype=torch.int32, device=device))
    solver._step_seconds[0] = step_seconds
    solver._coordinate_bound_count = 0
    coordinate_dofs = torch.arange(dof_count, dtype=torch.int32, device=device)
    solver._coordinate_dof_indices[:dof_count].copy_(coordinate_dofs)
    solver._base_weights[:frame_count].zero_()
    solver._temporal_weights.zero_()
    solver._ipm_constraint_scale[:frame_count].zero_()
    solver._ipm_weights[:frame_count].zero_()
    solver._ipm_constraint_scale[:frame_count, :residual_count].fill_(1.0)
    row_weights = torch.zeros(residual_count, device=device)
    row_weights[0] = 1.0e22
    solver._ipm_weights[0, :residual_count].copy_(row_weights)
    velocity_lower_base = residual_count + 2 * solver.coordinate_count
    solver._ipm_enabled[0] = 1

    size = frame_count * dof_count
    hessian = damping * torch.eye(size, dtype=torch.float64, device=device)
    for residual in range(residual_count):
        precision = torch.diag(solver._base_weights[:frame_count, residual].double())
        for order in range(1, 4):
            if order >= frame_count:
                continue
            coefficients = torch.tensor(
                ((-1.0, 1.0), (1.0, -2.0, 1.0), (-1.0, 3.0, -3.0, 1.0))[order - 1],
                dtype=torch.float64,
                device=device,
            )
            difference = torch.zeros((frame_count - order, frame_count), dtype=torch.float64, device=device)
            for row in range(frame_count - order):
                difference[row, row : row + order + 1] = coefficients
            precision += (
                solver._temporal_weights[order - 1, residual].double()
                / step_seconds ** (2 * order)
                * (difference.T @ difference)
            )
        residual_jacobian = torch.zeros((frame_count, size), dtype=torch.float64, device=device)
        for frame in range(frame_count):
            residual_jacobian[frame, frame * dof_count : (frame + 1) * dof_count] = jacobian[frame, residual].double()
        hessian += residual_jacobian.T @ precision @ residual_jacobian

    rows = []
    for frame in range(frame_count):
        frame_slice = slice(frame * dof_count, (frame + 1) * dof_count)
        for residual in range(residual_count):
            row = torch.zeros(size, dtype=torch.float64, device=device)
            row[frame_slice] = jacobian[frame, residual].double()
            rows.append((frame, residual, row))
        for bound in range(dof_count):
            dof = int(coordinate_dofs[bound])
            for column, sign in ((residual_count + bound, -1.0), (residual_count + dof_count + bound, 1.0)):
                row = torch.zeros(size, dtype=torch.float64, device=device)
                row[frame * dof_count + dof] = sign
                rows.append((frame, column, row))
        if frame > 0:
            for dof in range(dof_count):
                for column, sign in (
                    (velocity_lower_base + dof, -1.0),
                    (velocity_lower_base + dof_count + dof, 1.0),
                ):
                    row = torch.zeros(size, dtype=torch.float64, device=device)
                    row[frame * dof_count + dof] = sign
                    row[(frame - 1) * dof_count + dof] = -sign
                    rows.append((frame, column, row))
    augmented_rows = torch.stack(
        [
            torch.sqrt(solver._ipm_weights[frame, column].double())
            * solver._ipm_constraint_scale[frame, column].double()
            * row
            for frame, column, row in rows
        ]
    )
    active_row_count = torch.count_nonzero(solver._ipm_constraint_scale[:frame_count, : solver.inequality_width])
    assert active_row_count >= 15
    assert augmented_rows.shape[0] > size
    normal = hessian + augmented_rows.T @ augmented_rows
    first_pivot = torch.sqrt(normal[0, 0])
    assert normal[1, 1] - (normal[1, 0] / first_pivot) ** 2 <= 0.0

    right_hand_side = torch.full((size,), 1.0e15, device=device)
    solver._right_hand_side[:frame_count].copy_(right_hand_side.reshape(frame_count, dof_count))
    solver._ipm_right_hand_side_f64[:frame_count].copy_(right_hand_side.reshape(frame_count, dof_count).double())
    solver._delta_correction[:frame_count].copy_(right_hand_side.reshape(frame_count, dof_count))
    expected = right_hand_side.double() / (damping + 2.0 * row_weights[0].double())

    solver._kkt_enabled[0] = 1
    solver._prepare_ipm_objective_block_band(frame_count, 1)
    solver._ipm_augmented_fallback[0] = 1
    solver._solve_ipm_direction(frame_count=frame_count, segment_count=1, has_equalities=False, rebuild_factor=False)

    actual = solver._ipm_solution_f64[:frame_count].flatten()
    assert torch.all(torch.isfinite(actual))
    torch.testing.assert_close(actual.sum(), expected.sum(), atol=1.0e-12, rtol=5.0e-4)
    root_tolerance = max(
        32.0 * torch.finfo(torch.float32).eps,
        solver.kkt_relative_tolerance,
    )
    assert damping * torch.abs(actual[0] - actual[1]) <= root_tolerance
    physical_residual = hessian @ actual + augmented_rows.T @ (augmented_rows @ actual) - right_hand_side.double()
    assert torch.linalg.vector_norm(physical_residual) <= solver.krylov_relative_tolerance * torch.linalg.vector_norm(
        right_hand_side
    )
    assert solver._minres_enabled[0] == 0
    assert solver._minres_failed[0] == 0
    assert solver._ipm_augmented_fallback[0] == 1


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_ipm_phase_one_arrowhead_direction_matches_dense_system(device_type: str) -> None:
    """The no-equality Phase-I direction is the exact dense arrowhead solution."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device(device_type)
    frame_count = 4
    dof_count = 2
    residual_count = 2
    damping = 0.125
    jacobian = torch.tensor(
        [
            [[1.0, -0.5], [0.25, 1.5]],
            [[-0.75, 0.5], [1.25, -0.25]],
            [[0.5, 1.0], [-1.5, 0.25]],
            [[1.5, -0.75], [-0.5, 1.0]],
        ],
        device=device,
    )
    solver = _solver(
        _LinearOptimizer(jacobian, torch.zeros((frame_count, residual_count), device=device)),
        max_segments=1,
        max_equality_residuals_per_frame=0,
        damping=damping,
        krylov_max_iterations=1,
    )
    solver._frame_segment[:frame_count].zero_()
    solver._segment_offsets[:2].copy_(torch.tensor((0, frame_count), dtype=torch.int32, device=device))
    solver._coordinate_bound_count = dof_count
    solver._coordinate_dof_indices[:dof_count].copy_(torch.arange(dof_count, dtype=torch.int32, device=device))
    solver._ipm_enabled[0] = 1
    solver._active_equality_count[0] = 0
    solver._ipm_constraint_scale[:frame_count].zero_()
    solver._ipm_weights[:frame_count].zero_()

    coordinate_lower_base = residual_count
    coordinate_upper_base = coordinate_lower_base + solver.coordinate_count
    velocity_lower_base = coordinate_upper_base + solver.coordinate_count
    velocity_upper_base = velocity_lower_base + dof_count
    scalar_column = solver.constraint_width - 1
    physical_width = solver.inequality_width
    scales = torch.linspace(0.25, 1.75, frame_count * physical_width, device=device).reshape(
        frame_count, physical_width
    )
    weights = torch.logspace(-2, 8, frame_count * physical_width, device=device).reshape(frame_count, physical_width)
    solver._ipm_constraint_scale[:frame_count, :physical_width].copy_(scales)
    solver._ipm_weights[:frame_count, :physical_width].copy_(weights)
    solver._ipm_constraint_scale[0, scalar_column] = 1.0
    solver._ipm_weights[0, scalar_column] = 3.0e7
    solver._ipm_constraint_scale[0, velocity_lower_base : velocity_upper_base + dof_count].zero_()
    solver._ipm_weights[0, velocity_lower_base : velocity_upper_base + dof_count].zero_()

    size = frame_count * dof_count
    rows = []
    row_weights = []
    for frame in range(frame_count):
        frame_slice = slice(frame * dof_count, (frame + 1) * dof_count)
        for residual in range(residual_count):
            row = torch.zeros(size, dtype=torch.float64, device=device)
            scale = solver._ipm_constraint_scale[frame, residual].double()
            row[frame_slice] = scale * jacobian[frame, residual].double()
            rows.append(row)
            row_weights.append(solver._ipm_weights[frame, residual].double())
        for dof in range(dof_count):
            for column, sign in (
                (coordinate_lower_base + dof, -1.0),
                (coordinate_upper_base + dof, 1.0),
            ):
                row = torch.zeros(size, dtype=torch.float64, device=device)
                row[frame * dof_count + dof] = sign * solver._ipm_constraint_scale[frame, column].double()
                rows.append(row)
                row_weights.append(solver._ipm_weights[frame, column].double())
            if frame > 0:
                for column, sign in (
                    (velocity_lower_base + dof, -1.0),
                    (velocity_upper_base + dof, 1.0),
                ):
                    row = torch.zeros(size, dtype=torch.float64, device=device)
                    scale = sign * solver._ipm_constraint_scale[frame, column].double()
                    row[frame * dof_count + dof] = scale
                    row[(frame - 1) * dof_count + dof] = -scale
                    rows.append(row)
                    row_weights.append(solver._ipm_weights[frame, column].double())
    rows.append(torch.zeros(size, dtype=torch.float64, device=device))
    row_weights.append(solver._ipm_weights[0, scalar_column].double())
    constraint = torch.stack(rows)
    diagonal_weights = torch.diag(torch.stack(row_weights))
    elastic_column = -torch.ones(len(rows), dtype=torch.float64, device=device)
    full_operator = torch.cat((constraint, elastic_column[:, None]), dim=1)
    matrix = (
        damping * torch.eye(size + 1, dtype=torch.float64, device=device)
        + full_operator.T @ diagonal_weights @ full_operator
    )
    right_hand_side = torch.linspace(-1.25, 1.5, size, device=device)
    scalar_right_hand_side = torch.tensor(-0.75, device=device)
    solver._right_hand_side[:frame_count].copy_(right_hand_side.reshape(frame_count, dof_count))
    solver._phase_one_rhs[0] = scalar_right_hand_side
    solver._normal_diagonal[:frame_count].copy_(torch.diagonal(matrix)[:-1].reshape(frame_count, dof_count))
    solver._phase_one_diagonal[0] = matrix[-1, -1]

    solver._solve_phase_one_direction(frame_count=frame_count, segment_count=1)

    expected = torch.linalg.solve(matrix, torch.cat((right_hand_side.double(), scalar_right_hand_side[None].double())))
    primal_matrix = matrix[:-1, :-1]
    cross = matrix[:-1, -1]
    particular = torch.linalg.solve(primal_matrix, right_hand_side.double())
    coupling = torch.linalg.solve(primal_matrix, cross)
    torch.testing.assert_close(solver._delta[:frame_count].flatten().double(), expected[:-1], atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(solver._phase_one_delta[:1].double(), expected[-1:], atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(solver._phase_one_cross_f64[:frame_count].flatten(), cross, atol=1.0e-8, rtol=1.0e-8)
    torch.testing.assert_close(
        solver._phase_one_dot_segment_f64[0],
        torch.stack((torch.dot(cross, coupling), torch.dot(cross, particular))),
        atol=1.0e-7,
        rtol=1.0e-7,
    )
    assert solver._minres_failed[0] == 0


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_ipm_phase_one_arrowhead_rejects_nonfinite_scalar_diagonal(device_type: str) -> None:
    """The algebraic scalar solve preserves the existing fail-closed publication contract."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device(device_type)
    solver = _solver(
        _LinearOptimizer(torch.zeros((1, 1, 1), device=device), torch.zeros((1, 1), device=device)),
        max_segments=1,
        max_equality_residuals_per_frame=0,
        damping=0.125,
    )
    solver._frame_segment[0] = 0
    solver._segment_offsets[:2].copy_(torch.tensor((0, 1), dtype=torch.int32, device=device))
    solver._coordinate_bound_count = 0
    solver._ipm_enabled[0] = 1
    solver._ipm_constraint_scale[0].zero_()
    solver._ipm_weights[0].zero_()
    solver._right_hand_side[0, 0] = 1.0
    solver._phase_one_rhs[0] = 1.0
    solver._phase_one_diagonal[0] = torch.nan

    solver._solve_phase_one_direction(frame_count=1, segment_count=1)

    assert solver._minres_failed[0] == 1
    assert solver._ipm_enabled[0] == 0
    assert solver._accepted_cost[0] == torch.finfo(torch.float32).max
    torch.testing.assert_close(solver._phase_one_delta[:1], torch.zeros(1, device=device))
    torch.testing.assert_close(solver._delta[:1], torch.zeros((1, 1), device=device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture requires CUDA.")
def test_ipm_phase_one_arrowhead_cuda_graph_capture_replays() -> None:
    """The fixed-storage arrowhead solve captures and deterministically replays."""
    device = torch.device("cuda")
    solver = _solver(
        _LinearOptimizer(torch.zeros((2, 1, 1), device=device), torch.zeros((2, 1), device=device)),
        max_segments=1,
        max_equality_residuals_per_frame=0,
        damping=0.125,
    )
    solver._segment_damping[0] = solver.damping
    solver._frame_segment[:2].zero_()
    solver._segment_offsets[:2].copy_(torch.tensor((0, 2), dtype=torch.int32, device=device))
    solver._coordinate_bound_count = 0
    solver._ipm_enabled[0] = 1
    solver._ipm_constraint_scale[:2].zero_()
    solver._ipm_weights[:2].zero_()
    solver._right_hand_side[:2].copy_(torch.tensor(((0.5,), (-0.25,)), device=device))
    solver._phase_one_rhs[0] = 0.75
    solver._phase_one_diagonal[0] = 0.5

    def solve_once() -> None:
        solver._solve_phase_one_direction(frame_count=2, segment_count=1)

    solve_once()
    torch.cuda.synchronize()
    expected_primal = solver._delta[:2].clone()
    expected_elastic = solver._phase_one_delta[:1].clone()
    torch_stream = torch.cuda.Stream()
    warp_stream = wp.stream_from_torch(torch_stream)
    wp.capture_begin(stream=warp_stream)
    with torch.cuda.stream(torch_stream), wp.ScopedStream(warp_stream, sync_enter=False):
        solve_once()
    graph = wp.capture_end(stream=warp_stream)
    wp.capture_launch(graph, stream=warp_stream)
    wp.capture_launch(graph, stream=warp_stream)
    wp.synchronize_stream(warp_stream)

    torch.testing.assert_close(solver._delta[:2], expected_primal)
    torch.testing.assert_close(solver._phase_one_delta[:1], expected_elastic)
    assert solver._minres_failed[0] == 0


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_ipm_phase_one_equality_pcg_stops_on_backend_cadence(
    monkeypatch: pytest.MonkeyPatch,
    device_type: str,
) -> None:
    """Equality Phase-I PCG stops at one CPU iteration or one CUDA check interval."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_type)
    solver = _solver(
        _LinearOptimizer(torch.zeros((1, 1, 1), device=device), torch.zeros((1, 1), device=device)),
        max_segments=1,
        max_equality_residuals_per_frame=3,
        krylov_max_iterations=17,
    )
    solver._frame_segment[0] = 0
    solver._segment_offsets[:2].copy_(torch.tensor((0, 1), dtype=torch.int32, device=device))
    solver._segment_active[0] = 1
    solver._ipm_enabled[0] = 1
    solver._right_hand_side[0, 0] = 1.0
    solver._normal_diagonal[0, 0] = 1.0
    solver._phase_one_rhs[0] = 1.0
    solver._phase_one_diagonal[0] = 1.0
    apply_calls = 0

    def apply_identity(values, values_wp, elastic_values_wp, frame_count: int, segment_count: int) -> None:
        del values_wp, elastic_values_wp
        nonlocal apply_calls
        apply_calls += 1
        solver._pcg_operator_direction[:frame_count].copy_(values[:frame_count])
        solver._phase_one_operator[:segment_count].copy_(solver._phase_one_direction[:segment_count])

    monkeypatch.setattr(solver, "_phase_one_k_apply", apply_identity)
    solver._solve_phase_one_direction_pcg(frame_count=1, segment_count=1)

    assert apply_calls == (1 if device_type == "cpu" else solver.krylov_check_interval)
    torch.testing.assert_close(solver._delta[:1], torch.ones((1, 1), device=device))
    torch.testing.assert_close(solver._phase_one_delta[:1], torch.ones(1, device=device))
    assert solver._minres_enabled[0] == 0


def test_ipm_phase_one_equality_pcg_preserves_unconverged_segment_at_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One converged segment cannot hide a second segment that exhausts the PCG cap."""
    solver = _solver(
        _LinearOptimizer(torch.zeros((2, 1, 1)), torch.zeros((2, 1))),
        max_segments=2,
        max_equality_residuals_per_frame=3,
        krylov_max_iterations=3,
    )
    solver._frame_segment[:2].copy_(torch.tensor((0, 1), dtype=torch.int32))
    solver._segment_offsets[:3].copy_(torch.tensor((0, 1, 2), dtype=torch.int32))
    solver._segment_active[:2].fill_(1)
    solver._ipm_enabled[:2].fill_(1)
    solver._right_hand_side[:2].fill_(1.0)
    solver._normal_diagonal[:2].fill_(1.0)
    solver._phase_one_rhs[:2].zero_()
    solver._phase_one_diagonal[:2].fill_(1.0)
    apply_calls = 0

    def apply_first_segment(values, values_wp, elastic_values_wp, frame_count: int, segment_count: int) -> None:
        del values_wp, elastic_values_wp, segment_count
        nonlocal apply_calls
        apply_calls += 1
        solver._pcg_operator_direction[:frame_count].zero_()
        solver._pcg_operator_direction[0].copy_(values[0])
        solver._phase_one_operator[:2].zero_()

    monkeypatch.setattr(solver, "_phase_one_k_apply", apply_first_segment)
    solver._solve_phase_one_direction_pcg(frame_count=2, segment_count=2)

    assert apply_calls == solver.krylov_max_iterations
    torch.testing.assert_close(solver._delta[:, 0], torch.tensor((1.0, 0.0)))
    torch.testing.assert_close(solver._minres_enabled[:2], torch.tensor((0, 1), dtype=torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture requires CUDA.")
def test_ipm_phase_one_equality_pcg_cuda_capture_keeps_static_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture emits the complete PCG cap while masks make converged iterations no-ops."""
    device = torch.device("cuda")
    solver = _solver(
        _LinearOptimizer(torch.zeros((1, 1, 1), device=device), torch.zeros((1, 1), device=device)),
        max_segments=1,
        max_equality_residuals_per_frame=3,
        krylov_max_iterations=9,
    )
    solver._frame_segment[0] = 0
    solver._segment_offsets[:2].copy_(torch.tensor((0, 1), dtype=torch.int32, device=device))
    solver._segment_active[0] = 1
    solver._ipm_enabled[0] = 1
    solver._right_hand_side[0, 0] = 1.0
    solver._normal_diagonal[0, 0] = 1.0
    solver._phase_one_rhs[0] = 1.0
    solver._phase_one_diagonal[0] = 1.0
    apply_calls = 0

    def apply_identity(values, values_wp, elastic_values_wp, frame_count: int, segment_count: int) -> None:
        del values_wp, elastic_values_wp
        nonlocal apply_calls
        apply_calls += 1
        solver._pcg_operator_direction[:frame_count].copy_(values[:frame_count])
        solver._phase_one_operator[:segment_count].copy_(solver._phase_one_direction[:segment_count])

    monkeypatch.setattr(solver, "_phase_one_k_apply", apply_identity)
    solver._solve_phase_one_direction_pcg(frame_count=1, segment_count=1)
    torch.cuda.synchronize()
    expected_primal = solver._delta[:1].clone()
    expected_elastic = solver._phase_one_delta[:1].clone()
    apply_calls = 0
    torch_stream = torch.cuda.Stream()
    warp_stream = wp.stream_from_torch(torch_stream)
    wp.capture_begin(stream=warp_stream)
    with torch.cuda.stream(torch_stream), wp.ScopedStream(warp_stream, sync_enter=False):
        solver._solve_phase_one_direction_pcg(frame_count=1, segment_count=1)
    graph = wp.capture_end(stream=warp_stream)
    assert apply_calls == solver.krylov_max_iterations
    wp.capture_launch(graph, stream=warp_stream)
    wp.synchronize_stream(warp_stream)

    torch.testing.assert_close(solver._delta[:1], expected_primal)
    torch.testing.assert_close(solver._phase_one_delta[:1], expected_elastic)


def test_ipm_frame_factor_matches_dense_condensed_blocks() -> None:
    """Frame factors include exact residual, coordinate, and adjacent velocity contributions."""
    frame_count = 3
    damping = 0.125
    jacobian = torch.tensor(
        [
            [[1.0, 2.0], [-0.5, 1.5]],
            [[0.75, -1.25], [2.0, 0.5]],
            [[-1.5, 0.25], [0.5, -2.0]],
        ]
    )
    solver = _solver(
        _LinearOptimizer(jacobian, torch.zeros((frame_count, 2))),
        max_segments=1,
        max_equality_residuals_per_frame=0,
        damping=damping,
    )
    solver._frame_segment[:frame_count].zero_()
    solver._segment_offsets[:2].copy_(torch.tensor((0, frame_count), dtype=torch.int32))
    solver._step_seconds[0] = 1.0
    solver._kkt_enabled[0] = 1
    solver._minres_failed[0] = 0
    solver._coordinate_bound_count = 2
    coordinate_dofs = (1, 0)
    solver._coordinate_dof_indices[:2].copy_(torch.tensor(coordinate_dofs, dtype=torch.int32))
    solver._base_weights[:frame_count].fill_(1.0)
    solver._temporal_weights.zero_()
    solver._precision_diagonal[:frame_count].copy_(torch.tensor([[1.25, 0.75], [0.5, 2.0], [1.5, 1.0]]))
    solver._ipm_weights[:frame_count].zero_()
    solver._ipm_constraint_scale[:frame_count].zero_()

    residual_scale = torch.tensor([[0.5, 1.5], [1.25, 0.75], [2.0, 0.25]])
    residual_weight = torch.tensor([[0.75, 0.5], [1.5, 0.25], [0.125, 2.0]])
    coordinate_lower_scale = torch.tensor([[0.5, 1.0], [1.5, 0.25], [0.75, 2.0]])
    coordinate_upper_scale = torch.tensor([[1.25, 0.5], [0.5, 1.75], [1.0, 0.25]])
    coordinate_lower_weight = torch.tensor([[0.25, 1.5], [0.75, 0.5], [2.0, 0.125]])
    coordinate_upper_weight = torch.tensor([[1.0, 0.5], [0.25, 1.25], [0.75, 2.0]])
    velocity_lower_scale = torch.tensor([[4.0, 3.0], [0.5, 1.5], [2.0, 0.25]])
    velocity_upper_scale = torch.tensor([[3.0, 4.0], [1.25, 0.75], [0.5, 2.0]])
    velocity_lower_weight = torch.tensor([[2.0, 2.0], [0.25, 1.0], [1.5, 0.5]])
    velocity_upper_weight = torch.tensor([[2.0, 2.0], [0.75, 0.5], [0.25, 1.25]])

    coordinate_lower_base = solver.residual_count
    coordinate_upper_base = coordinate_lower_base + solver.coordinate_count
    velocity_lower_base = coordinate_upper_base + solver.coordinate_count
    velocity_upper_base = velocity_lower_base + solver.dof_count
    solver._ipm_constraint_scale[:frame_count, : solver.residual_count].copy_(residual_scale)
    solver._ipm_weights[:frame_count, : solver.residual_count].copy_(residual_weight)
    solver._ipm_constraint_scale[:frame_count, coordinate_lower_base:coordinate_upper_base].copy_(
        coordinate_lower_scale
    )
    solver._ipm_constraint_scale[:frame_count, coordinate_upper_base:velocity_lower_base].copy_(coordinate_upper_scale)
    solver._ipm_weights[:frame_count, coordinate_lower_base:coordinate_upper_base].copy_(coordinate_lower_weight)
    solver._ipm_weights[:frame_count, coordinate_upper_base:velocity_lower_base].copy_(coordinate_upper_weight)
    solver._ipm_constraint_scale[:frame_count, velocity_lower_base:velocity_upper_base].copy_(velocity_lower_scale)
    solver._ipm_constraint_scale[:frame_count, velocity_upper_base : velocity_upper_base + solver.dof_count].copy_(
        velocity_upper_scale
    )
    solver._ipm_weights[:frame_count, velocity_lower_base:velocity_upper_base].copy_(velocity_lower_weight)
    solver._ipm_weights[:frame_count, velocity_upper_base : velocity_upper_base + solver.dof_count].copy_(
        velocity_upper_weight
    )

    solver._ipm_temporal_factorize(frame_count=frame_count, segment_count=1)

    for frame in range(frame_count):
        expected = damping * torch.eye(solver.dof_count, dtype=torch.float64)
        frame_jacobian = solver._jacobian[frame].double()
        residual_precision = solver._precision_diagonal[frame].double()
        residual_precision += (
            solver._ipm_weights[frame, : solver.residual_count].double()
            * solver._ipm_constraint_scale[frame, : solver.residual_count].double().square()
        )
        expected += frame_jacobian.T @ (residual_precision[:, None] * frame_jacobian)
        for bound, dof in enumerate(coordinate_dofs):
            for base in (coordinate_lower_base, coordinate_upper_base):
                scale = solver._ipm_constraint_scale[frame, base + bound].double()
                expected[dof, dof] += solver._ipm_weights[frame, base + bound].double() * scale.square()
        for edge_frame in range(max(1, frame), min(frame + 2, frame_count)):
            for dof in range(solver.dof_count):
                for base in (velocity_lower_base, velocity_upper_base):
                    scale = solver._ipm_constraint_scale[edge_frame, base + dof].double()
                    expected[dof, dof] += solver._ipm_weights[edge_frame, base + dof].double() * scale.square()
        actual = torch.tril(solver._ipm_frame_factor[frame])
        torch.testing.assert_close(actual, torch.linalg.cholesky(expected), rtol=1.0e-12, atol=1.0e-12)

    torch.testing.assert_close(solver._ipm_segment_coupled[:1], torch.ones(1, dtype=torch.int32))
    assert solver._minres_failed[0] == 0


def test_ipm_additive_selection_preserves_diagonal_and_singleton_segments() -> None:
    """Structural segment selection preserves exact solves and corrects every frame of a coupled segment."""
    jacobian = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[2.0, 0.0], [0.0, 0.5]],
            [[1.0, 1.0], [0.5, -1.0]],
            [[1.0, 1.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 1.0]],
        ]
    )
    solver = _solver(
        _LinearOptimizer(jacobian, torch.zeros((5, 2))),
        max_segments=3,
        max_equality_residuals_per_frame=0,
        damping=0.5,
    )
    solver._frame_segment[:5].copy_(torch.tensor((0, 0, 1, 2, 2), dtype=torch.int32))
    solver._segment_offsets[:4].copy_(torch.tensor((0, 2, 3, 5), dtype=torch.int32))
    solver._step_seconds[:3].fill_(1.0)
    solver._kkt_enabled[:3].fill_(1)
    solver._minres_failed[:3].zero_()
    solver._coordinate_bound_count = 0
    solver._base_weights[:5].fill_(1.0)
    solver._precision_diagonal[:5].fill_(1.0)
    solver._temporal_weights.zero_()
    solver._ipm_weights[:5].zero_()
    solver._ipm_constraint_scale[:5].zero_()

    solver._ipm_temporal_factorize(frame_count=5, segment_count=3)

    torch.testing.assert_close(
        solver._ipm_segment_coupled[:3],
        torch.tensor((0, 0, 1), dtype=torch.int32),
    )
    singleton_matrix = 0.5 * torch.eye(2, dtype=torch.float64)
    singleton_jacobian = solver._jacobian[2].double()
    singleton_matrix += singleton_jacobian.T @ singleton_jacobian
    singleton_factor = torch.tril(solver._ipm_singleton_factor[1])
    torch.testing.assert_close(
        singleton_factor,
        torch.linalg.cholesky(singleton_matrix),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert solver._ipm_frame_factor[4, 1, 0] == 0.0

    values = torch.tensor([[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25], [2.0, -0.5], [1.5, 0.75]])
    solver._pcg_residual[:5].copy_(values)
    solver._ipm_additive_precondition(
        solver._pcg_residual,
        solver._wp_pcg_residual,
        solver._wp_kkt_enabled,
        frame_count=5,
        segment_count=3,
    )

    torch.testing.assert_close(solver._pcg_preconditioned[:3], values[:3], rtol=0.0, atol=0.0)
    temporal_lower = torch.zeros((4, 4), dtype=torch.float64)
    frame_inverse = torch.zeros_like(temporal_lower)
    for local_frame, frame in enumerate(range(3, 5)):
        for dof in range(2):
            for lag in range(local_frame + 1):
                temporal_lower[2 * local_frame + dof, 2 * (local_frame - lag) + dof] = solver._ipm_temporal_factor[
                    frame, dof, lag
                ]
        frame_factor = torch.tril(solver._ipm_frame_factor[frame])
        frame_inverse[2 * local_frame : 2 * local_frame + 2, 2 * local_frame : 2 * local_frame + 2] = (
            torch.cholesky_inverse(frame_factor)
        )
    coupled_values = values[3:5].double().flatten()
    expected = coupled_values + temporal_lower.T @ frame_inverse @ temporal_lower @ coupled_values
    torch.testing.assert_close(
        solver._pcg_preconditioned[3:5].flatten(),
        expected.float(),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert torch.linalg.vector_norm(solver._pcg_preconditioned[4] - values[4]) > 0.1


def test_ipm_additive_preconditioner_certifies_cross_dof_system_at_fixed_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mixed diagonal/coupled segment needs the additive correction to meet a six-step budget."""
    frame_count = 4
    damping = 1.0e-4
    correlations = (0.0, 0.4, 0.8, 0.98)
    blocks = torch.stack([torch.tensor([[1.0, value], [value, 1.0]]) for value in correlations])
    jacobian = torch.linalg.cholesky(blocks - damping * torch.eye(2)).transpose(1, 2).float()
    right_hand_side = torch.tensor(
        [
            [0.2722937762737274, -0.5983819365501404],
            [-0.8794832825660706, -0.8120310306549072],
            [-0.9729922413825989, -0.9565433859825134],
            [1.0334581136703491, 1.2872191667556763],
        ]
    )

    def solve(*, additive: bool) -> IKTrajectorySolver:
        solver = _solver(
            _LinearOptimizer(jacobian.clone(), torch.zeros((frame_count, 2))),
            max_segments=1,
            max_equality_residuals_per_frame=0,
            damping=damping,
            krylov_max_iterations=6,
            krylov_relative_tolerance=1.0e-4,
        )
        solver._frame_segment[:frame_count].zero_()
        solver._segment_offsets[:2].copy_(torch.tensor((0, frame_count), dtype=torch.int32))
        solver._step_seconds[0] = 1.0
        solver._segment_active[0] = 1
        solver._ipm_enabled[0] = 1
        solver._kkt_enabled[0] = 1
        solver._right_hand_side[:frame_count].copy_(right_hand_side)
        solver._base_weights[:frame_count].fill_(1.0)
        solver._temporal_weights.zero_()
        solver._precision_diagonal[:frame_count].fill_(1.0)
        solver._ipm_weights[:frame_count].zero_()
        solver._ipm_constraint_scale[:frame_count].zero_()
        if not additive:

            def apply_temporal_only(values, _values_wp, _enabled_wp, active_frames, _segment_count):
                solver._pcg_preconditioned[:active_frames].copy_(values[:active_frames])

            monkeypatch.setattr(solver, "_ipm_additive_precondition", apply_temporal_only)
        solver._solve_ipm_pcg(frame_count=frame_count, segment_count=1)
        return solver

    additive_solver = solve(additive=True)
    temporal_solver = solve(additive=False)
    additive_relative_residual = float(
        torch.sqrt(additive_solver._ipm_residual_dot_f64[0]) / additive_solver._ipm_initial_norm_f64[0]
    )
    temporal_relative_residual = float(
        torch.sqrt(temporal_solver._ipm_residual_dot_f64[0]) / temporal_solver._ipm_initial_norm_f64[0]
    )

    assert additive_solver._ipm_segment_coupled[0] == 1
    assert additive_solver._ipm_frame_factor[0, 1, 0] == 0.0
    assert additive_relative_residual < 5.0e-5
    assert additive_solver._minres_enabled[0] == 0
    assert additive_solver._minres_failed[0] == 0
    assert temporal_relative_residual > 1.0e-2
    assert temporal_solver._minres_enabled[0] == 1


def test_ipm_f64_recurrence_certifies_ill_conditioned_temporal_system() -> None:
    """Phase-II PCG retains its SPD recurrence across the required dynamic range."""
    jacobian = torch.tensor(
        [
            [[2.0, -3.0], [-1.0, 1.0]],
            [[80.0, -40.0], [-130.0, 100.0]],
            [[5.0, 13.0], [-33.0, -85.0]],
        ]
    )
    base_weights = torch.tensor((72.0, 6.0))
    temporal_weights = torch.tensor(((1.0e-3, 3.0e-4), (0.65, 15000.0), (5.0, 0.6)))
    right_hand_side = torch.tensor(((0.25, -1.0), (0.0, 0.25), (0.0, -1.0)))
    solver = _solver(
        _LinearOptimizer(jacobian, torch.zeros((3, 2))),
        max_segments=1,
        max_equality_residuals_per_frame=0,
        damping=1.0e-4,
        krylov_max_iterations=16,
        krylov_relative_tolerance=1.0e-4,
    )
    first_difference = torch.tensor(((-1.0, 1.0, 0.0), (0.0, -1.0, 1.0)))
    second_difference = torch.tensor(((1.0, -2.0, 1.0),))
    precision_diagonal = base_weights + (
        torch.diagonal(first_difference.T @ first_difference)[:, None] * temporal_weights[0]
        + torch.diagonal(second_difference.T @ second_difference)[:, None] * temporal_weights[1]
    )

    solver._frame_segment[:3].zero_()
    solver._segment_offsets[:2].copy_(torch.tensor((0, 3), dtype=torch.int32))
    solver._step_seconds[0] = 1.0
    solver._segment_active[0] = 1
    solver._ipm_enabled[0] = 1
    solver._kkt_enabled[0] = 1
    solver._ipm_linear_converged[0] = 1
    solver._right_hand_side[:3].copy_(right_hand_side)
    solver._ipm_right_hand_side_f64[:3].copy_(right_hand_side.double())
    solver._delta_correction[:3].copy_(right_hand_side)
    solver._base_weights[:3].copy_(base_weights.expand(3, -1))
    solver._temporal_weights.copy_(temporal_weights)
    solver._precision_diagonal[:3].copy_(precision_diagonal)
    solver._ipm_weights[:3].zero_()
    solver._ipm_constraint_scale[:3].zero_()
    solver._prepare_ipm_objective_block_band(frame_count=3, segment_count=1)

    solver._solve_ipm_direction(frame_count=3, segment_count=1, has_equalities=False)

    solver_norm = torch.sqrt(solver._ipm_residual_dot_f64[0])
    threshold = max(
        torch.finfo(torch.float32).eps,
        solver.krylov_relative_tolerance * solver._ipm_initial_norm_f64[0],
    )
    assert solver_norm <= threshold
    assert solver._ipm_linear_converged[0] == 1
    assert solver._minres_failed[0] == 0
    published_direction = solver._ipm_solution_f64[:3].float()
    solver._ipm_solution_f64[:3].copy_(published_direction.double())
    solver._ipm_enabled[0] = 1
    solver._ipm_strict_physical_certificate(
        solver._wp_ipm_solution_f64, solver._wp_ipm_enabled, frame_count=3, segment_count=1
    )
    publication_norm = torch.sqrt(solver._ipm_residual_dot_f64[0])
    assert publication_norm > threshold


@pytest.mark.parametrize(("check_interval", "expected_converged"), ((2, True), (6, False)))
def test_ipm_configured_check_interval_controls_reliable_refinement(
    check_interval: int, expected_converged: bool
) -> None:
    """The configured checkpoint cadence controls physical float32 refinement within a fixed budget."""
    jacobian = torch.tensor([[[-0.5533722639083862, 0.8329341411590576], [2.6339690685272217, 1.749916434288025]]])
    solver = _solver(
        _LinearOptimizer(jacobian, torch.zeros((1, 2))),
        max_segments=1,
        max_equality_residuals_per_frame=0,
        damping=1.0e-6,
        krylov_check_interval=check_interval,
        krylov_max_iterations=6,
        krylov_relative_tolerance=1.0e-8,
    )
    solver._jacobian[0].copy_(jacobian[0])
    solver._frame_segment[0] = 0
    solver._segment_offsets[:2].copy_(torch.tensor((0, 1), dtype=torch.int32))
    solver._step_seconds[0] = 1.0
    solver._segment_active[0] = 1
    solver._ipm_enabled[0] = 1
    solver._kkt_enabled[0] = 1
    solver._right_hand_side[0].copy_(torch.tensor((0.03136507421731949, 1.1248315572738647)))
    solver._base_weights[0].fill_(1.0)
    solver._precision_diagonal[0].fill_(1.0)
    solver._temporal_weights.zero_()
    solver._ipm_weights[0].zero_()
    solver._ipm_constraint_scale[0].zero_()
    solver._ipm_temporal_factor[0].zero_()
    solver._ipm_temporal_factor[0, :, 0] = 1.0
    solver._ipm_singleton_factor[0].copy_(torch.eye(2, dtype=torch.float64))

    solver._solve_ipm_pcg(frame_count=1, segment_count=1, rebuild_factor=False)

    final_norm = torch.sqrt(solver._ipm_residual_dot_f64[0])
    threshold = max(
        torch.finfo(torch.float32).eps,
        solver.krylov_relative_tolerance * solver._ipm_initial_norm_f64[0],
    )
    assert bool(final_norm <= threshold) is expected_converged
    assert bool(solver._minres_enabled[0] == 0) is expected_converged
    assert solver._minres_failed[0] == 0


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_ipm_direction_recovery_uses_stored_condensed_rows(device_type: str) -> None:
    """Slack and multiplier recovery must use the same rounded q and W solved by the condensed system."""
    wp.init()
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device(device_type)
    constraint_direction = torch.tensor([[16_777_216.5, -9.0]], dtype=torch.float64, device=device)
    constraint_scale = torch.tensor([[1.0, 0.0]], device=device)
    primal_residual = torch.tensor([[0.25, 5.0]], device=device)
    condensed_rhs = torch.tensor([[16_777_216.0, 8.0]], device=device)
    weights = torch.tensor([[1.0000001192092896, 2.0]], device=device)
    frame_segment = torch.zeros(1, dtype=torch.int32, device=device)
    enabled = torch.ones(1, dtype=torch.int32, device=device)
    slack_direction = torch.full_like(constraint_direction, torch.nan)
    multiplier_direction = torch.full_like(constraint_direction, torch.nan)

    wp.launch(
        trajectory_module._ipm_direction_recover_f64,
        dim=(1, 2),
        inputs=[
            wp.from_torch(constraint_direction),
            wp.from_torch(constraint_scale),
            wp.from_torch(primal_residual),
            wp.from_torch(condensed_rhs),
            wp.from_torch(weights),
            wp.from_torch(frame_segment),
            wp.from_torch(enabled),
            1,
        ],
        outputs=[wp.from_torch(slack_direction), wp.from_torch(multiplier_direction)],
        device=device_type,
    )

    expected_slack = -primal_residual[:, :1].double() - constraint_direction[:, :1]
    expected_multiplier = -condensed_rhs[:, :1].double() + weights[:, :1].double() * constraint_direction[:, :1]
    torch.testing.assert_close(slack_direction[:, :1], expected_slack, rtol=0.0, atol=0.0)
    torch.testing.assert_close(multiplier_direction[:, :1], expected_multiplier, rtol=0.0, atol=0.0)
    torch.testing.assert_close(slack_direction[:, 1:], torch.zeros_like(slack_direction[:, 1:]))
    torch.testing.assert_close(multiplier_direction[:, 1:], torch.zeros_like(multiplier_direction[:, 1:]))


def test_ipm_barrier_operator_preserves_condensed_recovery_rows() -> None:
    """Krylov operator scratch must not overwrite q before direction recovery."""
    solver = _solver(
        _LinearOptimizer(torch.ones((1, 1, 1)), torch.zeros((1, 1))),
        max_segments=1,
        max_equality_residuals_per_frame=0,
    )
    solver._jacobian[0].fill_(1.0)
    solver._frame_segment[0] = 0
    solver._segment_offsets[:2].copy_(torch.tensor((0, 1), dtype=torch.int32))
    solver._step_seconds[0] = 1.0
    solver._segment_active[0] = 1
    solver._ipm_enabled[0] = 1
    solver._base_weights[0].fill_(1.0)
    solver._temporal_weights.zero_()
    solver._ipm_constraint_scale[0].zero_()
    solver._ipm_constraint_scale[0, 0] = 2.0
    solver._ipm_weights[0].zero_()
    solver._ipm_weights[0, 0] = 3.0
    solver._pcg_preconditioned[0] = 2.0
    solver._ipm_constraint_work[0].copy_(torch.arange(solver.constraint_width, dtype=torch.float32).add_(7.25))
    condensed_rows = solver._ipm_constraint_work[0].clone()

    solver._ipm_k_apply(solver._pcg_preconditioned, solver._wp_pcg_preconditioned, frame_count=1)

    torch.testing.assert_close(solver._ipm_constraint_work[0], condensed_rows, rtol=0.0, atol=0.0)
    assert solver._ipm_complementarity_residual[0, 0] == 12.0


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_ipm_common_step_contracts_linear_stationarity_residual(device_type: str) -> None:
    """The accepted primal and dual steps must share one alpha so linear stationarity contracts."""
    wp.init()
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device(device_type)
    primal_step = torch.tensor((0.25, 0.9), device=device)
    dual_step = torch.tensor((0.75, 0.4), device=device)
    enabled = torch.tensor((1, 0), dtype=torch.int32, device=device)

    wp.launch(
        trajectory_module._ipm_step_couple,
        dim=2,
        inputs=[wp.from_torch(primal_step), wp.from_torch(dual_step), wp.from_torch(enabled), 2],
        device=device_type,
    )

    torch.testing.assert_close(primal_step, torch.tensor((0.25, 0.9), device=device))
    torch.testing.assert_close(dual_step, torch.tensor((0.25, 0.4), device=device))
    residual = 1.0
    primal_newton_term = -0.25
    dual_newton_term = -0.75
    updated = residual + float(primal_step[0]) * primal_newton_term + float(dual_step[0]) * dual_newton_term
    assert updated == (1.0 - float(primal_step[0])) * residual


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
def test_ipm_primal_f64_owns_sub_ulp_updates_and_preserves_frozen_segments(device_type: str) -> None:
    """Sub-ULP updates accumulate in the float64 owner while disabled segments remain unchanged."""
    wp.init()
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device(device_type)
    increment = 2.0**-25
    direction = torch.tensor([[increment], [7.0]], dtype=torch.float64, device=device)
    step = torch.ones(2, device=device)
    frame_segment = torch.tensor((0, 1), dtype=torch.int32, device=device)
    enabled = torch.tensor((1, 0), dtype=torch.int32, device=device)
    primal = torch.tensor([[1.0], [2.0]], dtype=torch.float64, device=device)
    primal_f32 = primal.float()

    for _ in range(4):
        wp.launch(
            trajectory_module._ipm_primal_update_state_f64,
            dim=(2, 1),
            inputs=[
                wp.from_torch(direction),
                wp.from_torch(step),
                wp.from_torch(frame_segment),
                wp.from_torch(enabled),
                2,
            ],
            outputs=[wp.from_torch(primal), wp.from_torch(primal_f32)],
            device=device_type,
        )

    expected = torch.tensor([[1.0 + 4.0 * increment], [2.0]], dtype=torch.float64, device=device)
    torch.testing.assert_close(primal, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(primal_f32, expected.float(), rtol=0.0, atol=0.0)


def test_ipm_checkpoint_preserves_unfinished_cg_recurrence() -> None:
    """A physical checkpoint cannot restart an unfinished conjugate-gradient solve."""
    jacobian = torch.diag(torch.sqrt(torch.tensor((0.9, 1.9, 3.9)))).unsqueeze(0)
    solver = _solver(
        _LinearOptimizer(jacobian, torch.zeros((1, 3))),
        max_segments=1,
        max_equality_residuals_per_frame=0,
        damping=0.1,
        krylov_check_interval=2,
        krylov_max_iterations=3,
        krylov_relative_tolerance=1.0e-5,
    )
    solver._jacobian[0].copy_(jacobian[0])
    solver._frame_segment[0] = 0
    solver._segment_offsets[:2].copy_(torch.tensor((0, 1), dtype=torch.int32))
    solver._step_seconds[0] = 1.0
    solver._segment_active[0] = 1
    solver._ipm_enabled[0] = 1
    solver._kkt_enabled[0] = 1
    solver._right_hand_side[0].fill_(1.0)
    solver._base_weights[0].fill_(1.0)
    solver._precision_diagonal[0].copy_(torch.tensor((0.9, 1.9, 3.9)))
    solver._temporal_weights.zero_()
    solver._ipm_weights[0].zero_()
    solver._ipm_constraint_scale[0].zero_()
    solver._ipm_temporal_factor[0].zero_()
    solver._ipm_temporal_factor[0, :, 0] = 1.0
    solver._ipm_singleton_factor[0].copy_(torch.eye(3, dtype=torch.float64))

    solver._solve_ipm_pcg(frame_count=1, segment_count=1, rebuild_factor=False)

    final_norm = torch.sqrt(solver._ipm_residual_dot_f64[0])
    threshold = max(
        torch.finfo(torch.float32).eps,
        solver.krylov_relative_tolerance * solver._ipm_initial_norm_f64[0],
    )
    assert final_norm <= threshold
    torch.testing.assert_close(solver._delta[0], torch.tensor((1.0, 0.5, 0.25)), atol=1.0e-6, rtol=0.0)
    assert solver._minres_enabled[0] == 0
    assert solver._minres_failed[0] == 0


def test_ipm_recursive_convergence_waits_for_physical_certificate() -> None:
    """Tentative recursive convergence keeps executing until a physical certificate passes."""
    wp.init()
    residual_dot = torch.tensor((1.0e-10,), dtype=torch.float32)
    state = torch.zeros((1, 14), dtype=torch.float32)
    state[0, 0] = 1.0
    enabled = torch.ones(1, dtype=torch.int32)
    recursive_converged = torch.zeros(1, dtype=torch.int32)
    failed = torch.zeros(1, dtype=torch.int32)

    wp.launch(
        trajectory_module._ipm_pcg_convergence_update,
        dim=1,
        inputs=[wp.from_torch(residual_dot), wp.from_torch(state), 1.0e-4, torch.finfo(torch.float32).eps],
        outputs=[wp.from_torch(enabled), wp.from_torch(recursive_converged), wp.from_torch(failed)],
        device="cpu",
    )

    torch.testing.assert_close(enabled, torch.ones_like(enabled))
    torch.testing.assert_close(recursive_converged, torch.ones_like(recursive_converged))
    torch.testing.assert_close(failed, torch.zeros_like(failed))

    requested = torch.ones(1, dtype=torch.int32)
    certify = torch.ones(1, dtype=torch.int32)
    wp.launch(
        trajectory_module._ipm_certification_resolve,
        dim=1,
        inputs=[wp.from_torch(certify), wp.from_torch(recursive_converged), 1],
        outputs=[wp.from_torch(requested), wp.from_torch(enabled)],
        device="cpu",
    )
    torch.testing.assert_close(requested, torch.ones_like(requested))
    torch.testing.assert_close(enabled, torch.ones_like(enabled))
    torch.testing.assert_close(recursive_converged, torch.zeros_like(recursive_converged))

    recursive_converged.fill_(1)
    certify.zero_()
    wp.launch(
        trajectory_module._ipm_certification_resolve,
        dim=1,
        inputs=[wp.from_torch(certify), wp.from_torch(recursive_converged), 1],
        outputs=[wp.from_torch(requested), wp.from_torch(enabled)],
        device="cpu",
    )
    torch.testing.assert_close(requested, torch.zeros_like(requested))
    torch.testing.assert_close(enabled, torch.zeros_like(enabled))
    torch.testing.assert_close(recursive_converged, torch.zeros_like(recursive_converged))


def test_ipm_rejects_unfinished_pcg_at_iteration_cap() -> None:
    """An unfinished PCG mask cannot publish its capped direction as converged."""
    wp.init()
    requested = torch.ones(1, dtype=torch.int32)
    krylov_enabled = torch.ones(1, dtype=torch.int32)
    krylov_failed = torch.zeros(1, dtype=torch.int32)
    linear_converged = torch.ones(1, dtype=torch.int32)
    enabled = torch.ones(1, dtype=torch.int32)

    wp.launch(
        _ipm_solve_status,
        dim=1,
        inputs=[wp.from_torch(requested), wp.from_torch(krylov_enabled), wp.from_torch(krylov_failed), 1],
        outputs=[wp.from_torch(linear_converged), wp.from_torch(enabled)],
        device="cpu",
    )

    assert linear_converged.item() == 0
    assert enabled.item() == 0


def test_krylov_negative_or_nonfinite_residual_measure_cannot_converge() -> None:
    """Only tiny negative roundoff may clamp to zero in PCG/MINRES residual norms."""
    wp.init()
    epsilon = torch.finfo(torch.float32).eps
    residual_dot = torch.tensor((-1.0e-3, -0.5 * epsilon, torch.nan), dtype=torch.float32)
    requested = torch.ones(3, dtype=torch.int32)
    expected = torch.tensor((1, 0, 1), dtype=torch.int32)

    pcg_state = torch.empty((3, 14), dtype=torch.float32)
    pcg_enabled = torch.empty(3, dtype=torch.int32)
    wp.launch(
        _pcg_convergence_initialize,
        dim=3,
        inputs=[wp.from_torch(residual_dot), wp.from_torch(requested), 3, epsilon],
        outputs=[wp.from_torch(pcg_state), wp.from_torch(pcg_enabled)],
        device="cpu",
    )

    minres_state = torch.empty((3, 14), dtype=torch.float32)
    minres_enabled = torch.empty(3, dtype=torch.int32)
    minres_failed = torch.empty(3, dtype=torch.int32)
    wp.launch(
        _minres_initialize,
        dim=3,
        inputs=[
            wp.from_torch(residual_dot),
            wp.from_torch(torch.zeros_like(residual_dot)),
            wp.from_torch(requested),
            epsilon,
        ],
        outputs=[wp.from_torch(minres_state), wp.from_torch(minres_enabled), wp.from_torch(minres_failed)],
        device="cpu",
    )

    torch.testing.assert_close(pcg_enabled, expected)
    torch.testing.assert_close(minres_enabled, torch.zeros_like(minres_enabled))
    torch.testing.assert_close(minres_failed, expected)
    assert pcg_state[0, 0] < 0.0 and pcg_state[2, 0] < 0.0
    assert minres_state[0, 0] < 0.0 and minres_state[2, 0] < 0.0


def test_minres_tiny_lanczos_beta_stops_before_next_basis_division() -> None:
    """An under-resolved Lanczos norm cannot remain enabled for a numerically singular next basis."""
    wp.init()
    epsilon = torch.finfo(torch.float32).eps
    state = torch.zeros((1, 14), dtype=torch.float32)
    state[0, 0] = 1.0
    state[0, 2] = 1.0
    state[0, 5] = 1.0
    state[0, 6] = -1.0
    enabled = torch.ones(1, dtype=torch.int32)
    failed = torch.zeros(1, dtype=torch.int32)
    tiny_beta = 1.0e-10
    wp.launch(
        _minres_recurrence,
        dim=1,
        inputs=[
            wp.from_torch(torch.tensor((tiny_beta * tiny_beta,), dtype=torch.float32)),
            wp.from_torch(torch.zeros(1)),
            1.0e-4,
            epsilon,
        ],
        outputs=[wp.from_torch(state), wp.from_torch(enabled), wp.from_torch(failed)],
        device="cpu",
    )

    assert state[0, 5] > 1.0e-4
    assert enabled.item() == 0
    assert failed.item() == 1
    values = torch.ones((1, 1))
    basis = torch.full_like(values, torch.nan)
    wp.launch(
        _minres_basis,
        dim=(1, 1),
        inputs=[
            wp.from_torch(values),
            wp.from_torch(torch.zeros(1, dtype=torch.int32)),
            wp.from_torch(state),
            wp.from_torch(enabled),
            1,
        ],
        outputs=[wp.from_torch(basis)],
        device="cpu",
    )
    torch.testing.assert_close(basis, torch.zeros_like(basis))


@pytest.mark.parametrize(("device_type", "expected_applications"), (("cpu", 1), ("cuda", 8)))
def test_krylov_loop_stops_when_all_segment_residuals_converge(device_type: str, expected_applications: int) -> None:
    """Non-capturing execution must poll zeroed PCG masks at its bounded interval."""
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_type)
    optimizer = _LinearOptimizer(
        torch.ones((1, 1, 1), device=device),
        torch.ones((1, 1), device=device),
    )
    solver = _solver(optimizer, max_segments=1, krylov_max_iterations=128)
    normal_apply = solver._normal_apply
    applications = 0

    def counted_normal_apply(*args, **kwargs) -> None:
        nonlocal applications
        applications += 1
        normal_apply(*args, **kwargs)

    solver._normal_apply = counted_normal_apply
    output = torch.empty((1, 1), device=device)
    _solve(
        solver,
        torch.zeros_like(output),
        output,
        torch.tensor((0, 1), dtype=torch.int32, device=device),
        torch.ones(1, device=device),
        torch.ones(1, device=device),
        torch.zeros((3, 1), device=device),
    )

    assert applications == expected_applications
    torch.testing.assert_close(
        output,
        torch.tensor(((1.0 / 1.1,),), device=device),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_segment_convergence_freezes_only_converged_rows() -> None:
    """A mixed batch must preserve converged segments while unfinished peers continue."""
    target = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    optimizer = _LinearOptimizer(torch.ones((4, 1, 1)), target)
    solver = _solver(optimizer, max_segments=2, damping=0.1)
    offsets = torch.tensor([0, 2, 4], dtype=torch.int32)
    step_seconds = torch.ones(2)
    pose_weights = torch.ones(1)
    temporal_weights = torch.zeros((3, 1))
    active = torch.ones(2, dtype=torch.int32)
    initial = torch.zeros((4, 1))
    first = torch.empty_like(initial)

    _solve(
        solver,
        initial,
        first,
        offsets,
        step_seconds,
        pose_weights,
        temporal_weights,
        segment_active=active,
        convergence_tolerance=1.0e-6,
    )

    torch.testing.assert_close(active, torch.tensor([0, 1], dtype=torch.int32))
    frozen = first[:2].clone()
    optimizer.target[:2].fill_(5.0)
    second = torch.empty_like(first)
    _solve(
        solver,
        first,
        second,
        offsets,
        step_seconds,
        pose_weights,
        temporal_weights,
        segment_active=active,
        convergence_tolerance=None,
    )

    torch.testing.assert_close(second[:2], frozen)
    assert torch.linalg.vector_norm(second[2:] - first[2:]) > 0.0
    torch.testing.assert_close(active, torch.tensor([0, 1], dtype=torch.int32))


def test_whole_segment_memory_plan_balances_without_splitting() -> None:
    mib = 1024 * 1024

    def estimate(frames: int) -> int:
        return 10 * mib + frames * 10 * mib

    plan = plan_trajectory_memory(
        (20, 15, 18, 12),
        "cuda",
        estimate,
        device_free_bytes=600 * mib,
    )
    assert plan.max_safe_frames == 47
    assert plan.batch_segment_offsets == (0, 2, 4)
    assert plan.batch_frame_counts == (35, 30)
    assert plan.workspace_frame_capacity == 35
    assert plan.bytes_per_frame == 10 * mib
    assert plan.peak_additional_workspace_bytes == estimate(35)
    with pytest.raises(MemoryError, match="Largest trajectory segment"):
        plan_trajectory_memory((50, 5), "cuda", estimate, device_free_bytes=600 * mib)


def test_ipm_workspace_estimate_is_exact() -> None:
    capacity = 64
    dof_count = 75
    optimizer = _LinearOptimizer(torch.ones((capacity, dof_count, dof_count)), torch.zeros((capacity, dof_count)))
    solver = IKTrajectorySolver(
        optimizer,
        max_segments=1,
        max_equality_residuals_per_frame=0,
    )
    expected_bytes = IKTrajectorySolver.estimate_workspace_bytes(capacity, dof_count, dof_count, dof_count, 1, 0)
    assert expected_bytes == 41_516_851
    assert solver.workspace_bytes == expected_bytes

    free_root_optimizer = _LinearOptimizer(torch.ones((2, 2, 3)), torch.zeros((2, 2)))
    free_root_optimizer.n_coords = 4
    free_root_solver = IKTrajectorySolver(
        free_root_optimizer,
        max_segments=1,
        max_equality_residuals_per_frame=0,
    )
    assert free_root_solver.workspace_bytes == IKTrajectorySolver.estimate_workspace_bytes(2, 4, 3, 2, 1, 0)

    assert (
        IKTrajectorySolver.estimate_workspace_bytes(1, 75, 75, 75, 1, 6)
        - (IKTrajectorySolver.estimate_workspace_bytes(1, 75, 75, 75, 1, 0))
        == 34 * 6 * torch.float32.itemsize
    )
    with pytest.raises(ValueError, match="positive"):
        IKTrajectorySolver.estimate_workspace_bytes(0, 75, 75, 75, 1, 0)


def test_workspace_is_linear_and_reused() -> None:
    capacity = 128
    optimizer = _LinearOptimizer(torch.ones((capacity, 3, 8)), torch.zeros((capacity, 3)))
    solver = _solver(optimizer)
    tensors = [value for value in vars(solver).values() if isinstance(value, torch.Tensor)]
    pointers = tuple(tensor.data_ptr() for tensor in tensors)
    expected_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors if tensor is not solver._jacobian)
    assert solver.workspace_bytes == expected_bytes
    assert solver.workspace_bytes == IKTrajectorySolver.estimate_workspace_bytes(capacity, 8, 8, 3, 4, 6)
    assert solver.workspace_bytes < (capacity * optimizer.n_dofs) ** 2 * 4
    output = torch.empty((16, 8))
    arguments = (
        torch.zeros_like(output),
        output,
        torch.tensor([0, 16], dtype=torch.int32),
        torch.ones(1),
        torch.ones(3),
        torch.zeros((3, 3)),
    )
    _solve(solver, *arguments)
    _solve(solver, *arguments)
    assert tuple(tensor.data_ptr() for tensor in tensors) == pointers


def test_lm_projection_participates_in_candidate_acceptance() -> None:
    """LM rejects a projected candidate whose retained cost exceeds the current feasible state."""
    builder = newton.ModelBuilder()
    root = builder.add_link(mass=1.0)
    child = builder.add_link(mass=1.0)
    root_joint = builder.add_joint_fixed(parent=-1, child=root)
    hinge_joint = builder.add_joint_revolute(parent=root, child=child, axis=newton.Axis.Z)
    builder.add_articulation([root_joint, hinge_joint])
    model = builder.finalize(device="cpu", requires_grad=True)
    objective = IKObjectiveJointPin(
        coordinate_indices=torch.tensor((0,), dtype=torch.int32).numpy(),
        dof_indices=torch.tensor((0,), dtype=torch.int32).numpy(),
        targets=torch.ones((1, 1), dtype=torch.float32),
        weight=1.0,
    )
    optimizer = ik.IKOptimizerLM(model, 1, [objective], jacobian_mode=ik.IKJacobianType.ANALYTIC)

    def project_away_from_target(values: wp.array) -> None:
        wp.to_torch(values).fill_(-1.0)

    for method in ("step", "solve"):
        joint_q = torch.zeros((1, 1), dtype=torch.float32)
        joint_q_wp = wp.from_torch(joint_q)
        if method == "step":
            optimizer.step(joint_q_wp, joint_q_wp, iterations=1, projection=project_away_from_target)
        else:
            optimizer.solve(
                joint_q_wp,
                joint_q_wp,
                max_iterations=1,
                convergence_tolerance=None,
                projection=project_away_from_target,
            )
        torch.testing.assert_close(joint_q, torch.zeros_like(joint_q))
        torch.testing.assert_close(wp.to_torch(optimizer.compute_costs(joint_q_wp)), torch.ones(1))


def test_joint_pin_analytic_jacobian_matches_scalar_coordinate_finite_difference() -> None:
    builder = newton.ModelBuilder()
    root = builder.add_link(mass=1.0)
    child = builder.add_link(mass=1.0)
    root_joint = builder.add_joint_fixed(parent=-1, child=root)
    hinge_joint = builder.add_joint_revolute(parent=root, child=child, axis=newton.Axis.Z)
    builder.add_articulation([root_joint, hinge_joint])
    model = builder.finalize(device="cpu", requires_grad=True)
    targets = torch.tensor(((0.2,), (-0.3,)), dtype=torch.float32)
    objective = IKObjectiveJointPin(
        coordinate_indices=torch.tensor((0,), dtype=torch.int32).numpy(),
        dof_indices=torch.tensor((0,), dtype=torch.int32).numpy(),
        targets=targets,
        weight=1.0,
    )
    optimizer = ik.IKOptimizerLM(model, 2, [objective], jacobian_mode=ik.IKJacobianType.ANALYTIC)
    joint_q = torch.tensor(((0.4,), (-0.1,)), dtype=torch.float32)
    residuals, jacobian = optimizer.linearize(wp.from_torch(joint_q))
    residuals = wp.to_torch(residuals).clone()
    jacobian = wp.to_torch(jacobian).clone()

    torch.testing.assert_close(residuals, joint_q - targets)
    torch.testing.assert_close(jacobian[..., 0], torch.ones_like(residuals))

    epsilon = 1.0e-4
    delta = torch.full((2, 1), epsilon)
    plus = torch.empty_like(joint_q)
    minus = torch.empty_like(joint_q)
    optimizer.integrate(wp.from_torch(joint_q), wp.from_torch(delta), wp.from_torch(plus))
    optimizer.integrate(wp.from_torch(joint_q), wp.from_torch(delta), wp.from_torch(minus), step_size=-1.0)
    residual_plus = wp.to_torch(optimizer.linearize(wp.from_torch(plus))[0]).clone()
    residual_minus = wp.to_torch(optimizer.linearize(wp.from_torch(minus))[0]).clone()
    finite_difference = (residual_plus - residual_minus) / (2.0 * epsilon)
    torch.testing.assert_close(jacobian[..., 0], finite_difference, atol=2.0e-4, rtol=2.0e-4)


def test_newton_rotation_linearization_is_quaternion_sign_invariant() -> None:
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0)
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
    model = builder.finalize(device="cpu", requires_grad=True)
    target = wp.array([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=wp.vec4, device="cpu")
    objective = ik.IKObjectiveRotation(body, wp.quat_identity(), target)
    optimizer = ik.IKOptimizerLM(model, 2, [objective], jacobian_mode=ik.IKJacobianType.ANALYTIC)
    angle = 0.4
    quaternion = torch.tensor([0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)])
    joint_q = torch.zeros((2, 7), dtype=torch.float32)
    joint_q[0, 3:7] = quaternion
    joint_q[1, 3:7] = -quaternion
    residuals, jacobian = optimizer.linearize(wp.from_torch(joint_q))
    residuals = wp.to_torch(residuals).clone()
    jacobian = wp.to_torch(jacobian).clone()
    torch.testing.assert_close(residuals[0], residuals[1], atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(jacobian[0], jacobian[1], atol=1.0e-6, rtol=1.0e-6)

    epsilon = 1.0e-3
    delta = torch.zeros((2, 6), dtype=torch.float32)
    delta[:, 5] = epsilon
    plus = torch.empty_like(joint_q)
    minus = torch.empty_like(joint_q)
    optimizer.integrate(wp.from_torch(joint_q), wp.from_torch(delta), wp.from_torch(plus))
    optimizer.integrate(
        wp.from_torch(joint_q),
        wp.from_torch(delta),
        wp.from_torch(minus),
        step_size=-1.0,
    )
    residual_plus = wp.to_torch(optimizer.linearize(wp.from_torch(plus))[0]).clone()
    residual_minus = wp.to_torch(optimizer.linearize(wp.from_torch(minus))[0]).clone()
    finite_difference = (residual_plus - residual_minus) / (2.0 * epsilon)
    torch.testing.assert_close(jacobian[:, :, 5], finite_difference, atol=2.0e-3, rtol=2.0e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_known_solution_matches_on_cuda() -> None:
    device = torch.device("cuda:0")
    jacobian = torch.tensor([[[1.0]], [[2.0]], [[-1.0]]], device=device)
    target = torch.tensor([[1.0], [0.5], [-0.25]], device=device)
    output, _ = _run_linear(
        jacobian,
        target,
        torch.tensor([0, 3], dtype=torch.int32, device=device),
        torch.ones(1, device=device),
        torch.tensor([[1.0], [0.2], [0.0]], device=device),
    )
    expected = _dense_solution(
        jacobian.cpu(),
        target.cpu(),
        (0, 3),
        (1.0,),
        torch.ones((3, 1)),
        torch.tensor([[1.0], [0.2], [0.0]]),
        0.1,
    )
    torch.testing.assert_close(output.cpu(), expected, atol=2.0e-4, rtol=2.0e-4)


def test_residual_activity_masks_base_rows_and_temporal_edges() -> None:
    """Compact confidence gates base rows and only fully active temporal stencils."""
    jacobian = torch.ones((4, 1, 1), dtype=torch.float32)
    target = torch.zeros((4, 1), dtype=torch.float32)
    solver = _solver(_LinearOptimizer(jacobian, target), max_segments=1, damping=0.1)
    joint_q = torch.tensor(((10.0,), (0.0,), (2.0,), (-10.0,)))
    output = torch.empty_like(joint_q)
    activity = IKTrajectorySolver.ResidualActivity(
        values=torch.tensor(((0.0,), (1.0,), (1.0,), (0.0,))),
        group_by_residual=torch.tensor((0,), dtype=torch.int32),
    )

    _solve(
        solver,
        joint_q,
        output,
        torch.tensor((0, 4), dtype=torch.int32),
        torch.ones(1),
        torch.ones(1),
        torch.tensor(((1.0,), (0.0,), (0.0,))),
        residual_activity=activity,
    )

    torch.testing.assert_close(output[[0, 3]], joint_q[[0, 3]])
    assert 0.0 < output[1, 0] < output[2, 0] < 2.0
