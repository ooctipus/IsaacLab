# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent dense conformance tests for constrained trajectory refinement."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import torch
import warp as wp

from isaaclab_tasks.core.multi_task.kinematics.trajectory import IKTrajectorySolver


class _Optimizer:
    """Minimal fixed-capacity optimizer used only through the public solver path."""

    def __init__(
        self,
        capacity: int,
        residual_count: int,
        dof_count: int,
        residual_fn: Callable[[torch.Tensor], torch.Tensor],
        jacobian_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.device = wp.get_device("cpu")
        self.n_batch = capacity
        self.n_residuals = residual_count
        self.n_dofs = dof_count
        self.n_coords = dof_count
        self._residual_fn = residual_fn
        self._jacobian_fn = jacobian_fn
        self.residuals = torch.empty((capacity, residual_count), dtype=torch.float32)
        self.jacobian = torch.empty((capacity, residual_count, dof_count), dtype=torch.float32)
        self._wp_residuals = wp.from_torch(self.residuals)
        self._wp_jacobian = wp.from_torch(self.jacobian)

    def compute_residuals(self, joint_q, residuals=None):
        active = joint_q.shape[0]
        self.residuals[:active].copy_(self._residual_fn(wp.to_torch(joint_q)[:active]))
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
            return residuals
        return self._wp_residuals[:active]

    def linearize(self, joint_q, residuals=None, jacobian=None):
        active = joint_q.shape[0]
        values = wp.to_torch(joint_q)[:active]
        self.residuals[:active].copy_(self._residual_fn(values))
        self.jacobian[:active].copy_(self._jacobian_fn(values))
        residual_view = self._wp_residuals[:active] if residuals is None else residuals
        jacobian_view = self._wp_jacobian[:active] if jacobian is None else jacobian
        if residuals is not None:
            wp.copy(residuals, self._wp_residuals[:active])
        if jacobian is not None:
            wp.copy(jacobian, self._wp_jacobian[:active])
        return residual_view, jacobian_view

    def integrate(self, joint_q, delta, joint_q_out, *, step_size=1.0):
        output = wp.to_torch(joint_q_out)
        output.copy_(wp.to_torch(joint_q))
        output.add_(wp.to_torch(delta), alpha=step_size)


def _finite_difference(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    joint_q: torch.Tensor,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Return the full float64 residual Jacobian without assuming frame locality."""
    frame_count, dof_count = joint_q.shape
    residual_count = residual_fn(joint_q).shape[1]
    jacobian = torch.empty((frame_count, residual_count, frame_count, dof_count), dtype=torch.float64)
    for source_frame in range(frame_count):
        for dof in range(dof_count):
            perturbation = torch.zeros_like(joint_q)
            perturbation[source_frame, dof] = epsilon
            jacobian[:, :, source_frame, dof] = (
                residual_fn(joint_q + perturbation) - residual_fn(joint_q - perturbation)
            ) / (2.0 * epsilon)
    return jacobian


def _assert_optimizer_derivatives(
    optimizer: _Optimizer,
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    joint_q: torch.Tensor,
) -> None:
    """Compare the adapter's analytic contract with an independent finite difference."""
    residuals_wp, jacobian_wp = optimizer.linearize(wp.from_torch(joint_q.to(torch.float32)))
    residuals = wp.to_torch(residuals_wp).to(torch.float64)
    analytic = wp.to_torch(jacobian_wp).to(torch.float64)
    finite_difference = _finite_difference(residual_fn, joint_q)
    torch.testing.assert_close(residuals, residual_fn(joint_q), atol=2.0e-7, rtol=2.0e-7)
    for residual_frame in range(joint_q.shape[0]):
        for source_frame in range(joint_q.shape[0]):
            expected = (
                analytic[residual_frame]
                if residual_frame == source_frame
                else torch.zeros_like(analytic[residual_frame])
            )
            torch.testing.assert_close(
                finite_difference[residual_frame, :, source_frame],
                expected,
                atol=2.0e-6,
                rtol=2.0e-6,
            )


def _dense_model(
    joint_q: torch.Tensor,
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    pose_weights: torch.Tensor,
    temporal_weights: torch.Tensor,
    damping: float,
    step_seconds: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble the dense float64 Gauss-Newton QP independently of production."""
    residuals = residual_fn(joint_q)
    full_jacobian = _finite_difference(residual_fn, joint_q)
    jacobian = torch.stack([full_jacobian[frame, :, frame] for frame in range(joint_q.shape[0])])
    frame_count, residual_count, dof_count = jacobian.shape
    variable_count = frame_count * dof_count
    hessian = damping * torch.eye(variable_count, dtype=torch.float64)
    gradient = torch.zeros(variable_count, dtype=torch.float64)
    coefficients = ((-1.0, 1.0), (1.0, -2.0, 1.0), (-1.0, 3.0, -3.0, 1.0))
    for residual_index in range(residual_count):
        precision = pose_weights[residual_index] * torch.eye(frame_count, dtype=torch.float64)
        for order, stencil in enumerate(coefficients, start=1):
            weight = temporal_weights[order - 1, residual_index]
            if weight == 0.0 or frame_count <= order:
                continue
            difference = torch.zeros((frame_count - order, frame_count), dtype=torch.float64)
            for row in range(frame_count - order):
                difference[row, row : row + order + 1] = torch.tensor(stencil, dtype=torch.float64)
            difference /= step_seconds**order
            precision += weight * difference.T @ difference
        residual_jacobian = torch.zeros((frame_count, variable_count), dtype=torch.float64)
        for frame in range(frame_count):
            begin = frame * dof_count
            residual_jacobian[frame, begin : begin + dof_count] = jacobian[frame, residual_index]
        hessian += residual_jacobian.T @ precision @ residual_jacobian
        gradient += residual_jacobian.T @ precision @ residuals[:, residual_index]
    return hessian, gradient, residuals, jacobian


def _hard_rows(
    joint_q: torch.Tensor,
    residuals: torch.Tensor,
    jacobian: torch.Tensor,
    *,
    residual_indices: tuple[int, ...],
    residual_upper: tuple[float, ...],
    coordinate_bounds: tuple[tuple[int, int, float, float], ...],
    joint_velocity: torch.Tensor,
    velocity_lower: torch.Tensor,
    velocity_upper: torch.Tensor,
    step_seconds: float,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Build A step <= b for every finite public hard-bound family."""
    frame_count, _, dof_count = jacobian.shape
    variable_count = frame_count * dof_count
    rows: list[torch.Tensor] = []
    bounds: list[torch.Tensor] = []
    labels: list[str] = []
    for residual_index, upper in zip(residual_indices, residual_upper, strict=True):
        for frame in range(frame_count):
            row = torch.zeros(variable_count, dtype=torch.float64)
            begin = frame * dof_count
            row[begin : begin + dof_count] = jacobian[frame, residual_index]
            rows.append(row)
            bounds.append(torch.tensor(upper, dtype=torch.float64) - residuals[frame, residual_index])
            labels.append(f"residual[{frame},{residual_index}]")
    for coordinate, dof, lower, upper in coordinate_bounds:
        for frame in range(frame_count):
            variable = frame * dof_count + dof
            if math.isfinite(lower):
                row = torch.zeros(variable_count, dtype=torch.float64)
                row[variable] = -1.0
                rows.append(row)
                bounds.append(joint_q[frame, coordinate] - lower)
                labels.append(f"coordinate_lower[{frame},{coordinate}]")
            if math.isfinite(upper):
                row = torch.zeros(variable_count, dtype=torch.float64)
                row[variable] = 1.0
                rows.append(row)
                bounds.append(torch.tensor(upper, dtype=torch.float64) - joint_q[frame, coordinate])
                labels.append(f"coordinate_upper[{frame},{coordinate}]")
    for frame in range(1, frame_count):
        for dof in range(dof_count):
            current_velocity = joint_velocity[frame, dof]
            if torch.isfinite(velocity_lower[dof]):
                row = torch.zeros(variable_count, dtype=torch.float64)
                row[(frame - 1) * dof_count + dof] = 1.0
                row[frame * dof_count + dof] = -1.0
                rows.append(row)
                bounds.append(step_seconds * (current_velocity - velocity_lower[dof]))
                labels.append(f"velocity_lower[{frame},{dof}]")
            if torch.isfinite(velocity_upper[dof]):
                row = torch.zeros(variable_count, dtype=torch.float64)
                row[(frame - 1) * dof_count + dof] = -1.0
                row[frame * dof_count + dof] = 1.0
                rows.append(row)
                bounds.append(step_seconds * (velocity_upper[dof] - current_velocity))
                labels.append(f"velocity_upper[{frame},{dof}]")
    return torch.stack(rows), torch.stack(bounds), labels


def _active_set_qp(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    rows: torch.Tensor,
    bounds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve a tiny convex QP by exhaustively enumerating independent active sets."""
    variable_count = gradient.numel()
    best: tuple[float, torch.Tensor, torch.Tensor, tuple[int, ...]] | None = None
    tolerance = 2.0e-8
    for active_count in range(min(variable_count, rows.shape[0]) + 1):
        for active in itertools.combinations(range(rows.shape[0]), active_count):
            active_rows = rows[list(active)] if active else rows.new_empty((0, variable_count))
            if active and torch.linalg.matrix_rank(active_rows, tol=1.0e-10) != active_count:
                continue
            size = variable_count + active_count
            kkt = torch.zeros((size, size), dtype=torch.float64)
            kkt[:variable_count, :variable_count] = hessian
            if active:
                kkt[:variable_count, variable_count:] = active_rows.T
                kkt[variable_count:, :variable_count] = active_rows
            active_bounds = bounds[list(active)] if active else bounds.new_empty(0)
            right_hand_side = torch.cat((-gradient, active_bounds))
            try:
                solution = torch.linalg.solve(kkt, right_hand_side)
            except RuntimeError:
                continue
            step = solution[:variable_count]
            multipliers = solution[variable_count:]
            if torch.max(rows @ step - bounds) > tolerance:
                continue
            if active and torch.min(multipliers) < -tolerance:
                continue
            objective = float(0.5 * step @ hessian @ step + gradient @ step)
            if best is None or objective < best[0]:
                best = (objective, step, multipliers, active)
    if best is None:
        raise AssertionError("The dense active-set oracle found no feasible QP point.")
    _, step, active_multipliers, active = best
    multipliers = torch.zeros(rows.shape[0], dtype=torch.float64)
    if active:
        multipliers[list(active)] = active_multipliers
    return step, multipliers


def _assert_kkt(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    rows: torch.Tensor,
    bounds: torch.Tensor,
    step: torch.Tensor,
    multipliers: torch.Tensor,
) -> None:
    slack = rows @ step - bounds
    torch.testing.assert_close(
        hessian @ step + gradient + rows.T @ multipliers,
        torch.zeros_like(step),
        atol=2.0e-8,
        rtol=0.0,
    )
    assert torch.max(slack) <= 2.0e-8
    assert torch.min(multipliers) >= -2.0e-8
    assert torch.max(torch.abs(multipliers * slack)) <= 2.0e-8


def _solve_public(
    optimizer: _Optimizer,
    joint_q: torch.Tensor,
    pose_weights: torch.Tensor,
    temporal_weights: torch.Tensor,
    damping: float,
    *,
    inequalities: IKTrajectorySolver.ResidualInequalities,
    coordinate_bounds: IKTrajectorySolver.CoordinateBounds,
    joint_velocity: torch.Tensor,
    velocity_lower: torch.Tensor,
    velocity_upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    solver = IKTrajectorySolver(
        optimizer,
        max_segments=1,
        max_equality_residuals_per_frame=0,
        damping=damping,
        krylov_max_iterations=256,
        krylov_relative_tolerance=1.0e-6,
        kkt_relative_tolerance=1.0e-6,
        krylov_check_interval=1,
    )
    output = torch.empty_like(joint_q)
    feasible = torch.empty(1, dtype=torch.bool)
    direction_valid = torch.empty(1, dtype=torch.bool)
    globalization_succeeded = torch.empty(1, dtype=torch.bool)
    solver.solve(
        joint_q,
        output,
        torch.tensor((0, joint_q.shape[0]), dtype=torch.int32),
        torch.ones(1),
        pose_weights,
        temporal_weights,
        coordinate_bounds=coordinate_bounds,
        joint_velocity=joint_velocity,
        velocity_lower=velocity_lower,
        velocity_upper=velocity_upper,
        segment_active=torch.ones(1, dtype=torch.int32),
        inequalities=inequalities,
        segment_feasible=feasible,
        segment_direction_valid=direction_valid,
        segment_globalization_succeeded=globalization_succeeded,
        convergence_tolerance=1.0e-8,
    )
    return output, feasible, direction_valid, globalization_succeeded


def test_dense_qp_conforms_for_residual_coordinate_and_velocity_constraints() -> None:
    """The public solve matches an exhaustive dense QP and its KKT certificate."""
    scale64 = torch.tensor((10.0, 1.0, 1.0), dtype=torch.float64)
    target64 = torch.tensor((0.0, 2.0, 4.0), dtype=torch.float64)
    scale32 = scale64.to(torch.float32)
    target32 = target64.to(torch.float32)

    def residual64(joint_q: torch.Tensor) -> torch.Tensor:
        x, y = joint_q.unbind(dim=1)
        return torch.stack((scale64 * x - target64, y - 1.0, x + y), dim=1)

    def residual32(joint_q: torch.Tensor) -> torch.Tensor:
        x, y = joint_q.unbind(dim=1)
        return torch.stack((scale32 * x - target32, y - 1.0, x + y), dim=1)

    def jacobian32(joint_q: torch.Tensor) -> torch.Tensor:
        jacobian = joint_q.new_zeros((joint_q.shape[0], 3, 2))
        jacobian[:, 0, 0] = scale32[: joint_q.shape[0]]
        jacobian[:, 1, 1] = 1.0
        jacobian[:, 2] = 1.0
        return jacobian

    optimizer = _Optimizer(3, 3, 2, residual32, jacobian32)
    probe = torch.tensor(((0.1, -0.2), (0.3, 0.4), (-0.2, 0.7)), dtype=torch.float64)
    _assert_optimizer_derivatives(optimizer, residual64, probe)

    initial = torch.zeros((3, 2), dtype=torch.float64)
    pose_weights = torch.tensor((1.0, 1.0, 0.0), dtype=torch.float64)
    temporal_weights = torch.tensor(
        ((0.02, 0.03, 0.0), (0.005, 0.007, 0.0), (0.0, 0.0, 0.0)),
        dtype=torch.float64,
    )
    damping = 0.05
    hessian, gradient, residuals, jacobian = _dense_model(
        initial,
        residual64,
        pose_weights,
        temporal_weights,
        damping,
        1.0,
    )
    joint_velocity = torch.zeros_like(initial)
    velocity_lower = torch.full((2,), -torch.inf, dtype=torch.float64)
    velocity_upper = torch.tensor((0.25, torch.inf), dtype=torch.float64)
    rows, bounds, labels = _hard_rows(
        initial,
        residuals,
        jacobian,
        residual_indices=(2,),
        residual_upper=(0.9,),
        coordinate_bounds=((1, 1, -math.inf, 0.4),),
        joint_velocity=joint_velocity,
        velocity_lower=velocity_lower,
        velocity_upper=velocity_upper,
        step_seconds=1.0,
    )
    expected_step, multipliers = _active_set_qp(hessian, gradient, rows, bounds)
    _assert_kkt(hessian, gradient, rows, bounds, expected_step, multipliers)
    slack = rows @ expected_step - bounds
    for prefix in ("residual", "coordinate_upper", "velocity_upper"):
        family = [abs(float(slack[index])) for index, label in enumerate(labels) if label.startswith(prefix)]
        assert min(family) <= 2.0e-8

    output, feasible, direction_valid, globalization_succeeded = _solve_public(
        optimizer,
        initial.to(torch.float32),
        pose_weights.to(torch.float32),
        temporal_weights.to(torch.float32),
        damping,
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor((2,), dtype=torch.int32),
            upper=torch.full((1,), 0.9),
        ),
        coordinate_bounds=IKTrajectorySolver.CoordinateBounds(
            coordinate_indices=torch.tensor((1,), dtype=torch.int32),
            dof_indices=torch.tensor((1,), dtype=torch.int32),
            lower=torch.tensor((-torch.inf,)),
            upper=torch.tensor((0.4,)),
        ),
        joint_velocity=joint_velocity.to(torch.float32),
        velocity_lower=velocity_lower.to(torch.float32),
        velocity_upper=velocity_upper.to(torch.float32),
    )
    torch.testing.assert_close(
        output.to(torch.float64).flatten(),
        expected_step,
        atol=4.0e-4,
        rtol=4.0e-4,
    )
    assert feasible.item()
    assert direction_valid.item()
    assert globalization_succeeded.item()
    output64 = output.to(torch.float64)
    assert torch.max(residual64(output64)[:, 2]) <= 0.9 + 2.0e-5
    assert torch.max(output64[:, 1]) <= 0.4 + 2.0e-5
    assert torch.max(torch.diff(output64[:, 0])) <= 0.25 + 2.0e-5


def test_dense_nlp_conforms_for_curved_second_order_correction() -> None:
    """The public result matches dense SOC after every ordinary trial is infeasible."""

    def residual64(joint_q: torch.Tensor) -> torch.Tensor:
        x, y = joint_q.unbind(dim=1)
        return torch.stack((y - 1.0, x + y.square() - 1.0), dim=1)

    def residual32(joint_q: torch.Tensor) -> torch.Tensor:
        x, y = joint_q.unbind(dim=1)
        return torch.stack((y - 1.0, x + y.square() - 1.0), dim=1)

    def jacobian32(joint_q: torch.Tensor) -> torch.Tensor:
        jacobian = joint_q.new_zeros((joint_q.shape[0], 2, 2))
        jacobian[:, 0, 1] = 1.0
        jacobian[:, 1, 0] = 1.0
        jacobian[:, 1, 1] = 2.0 * joint_q[:, 1]
        return jacobian

    optimizer = _Optimizer(1, 2, 2, residual32, jacobian32)
    probe = torch.tensor(((0.2, 0.3),), dtype=torch.float64)
    _assert_optimizer_derivatives(optimizer, residual64, probe)

    initial = torch.tensor(((1.0, 0.0),), dtype=torch.float64)
    pose_weights = torch.tensor((1.0, 0.0), dtype=torch.float64)
    temporal_weights = torch.zeros((3, 2), dtype=torch.float64)
    damping = 1.0e-6
    hessian, gradient, residuals, jacobian = _dense_model(
        initial,
        residual64,
        pose_weights,
        temporal_weights,
        damping,
        1.0,
    )
    no_velocity = torch.zeros_like(initial)
    infinite_lower = torch.full((2,), -torch.inf, dtype=torch.float64)
    infinite_upper = torch.full((2,), torch.inf, dtype=torch.float64)
    rows, bounds, _ = _hard_rows(
        initial,
        residuals,
        jacobian,
        residual_indices=(1,),
        residual_upper=(0.0,),
        coordinate_bounds=(),
        joint_velocity=no_velocity,
        velocity_lower=infinite_lower,
        velocity_upper=infinite_upper,
        step_seconds=1.0,
    )
    tangent_step, tangent_multipliers = _active_set_qp(hessian, gradient, rows, bounds)
    _assert_kkt(hessian, gradient, rows, bounds, tangent_step, tangent_multipliers)
    ordinary_candidates = [initial + scale * tangent_step.reshape_as(initial) for scale in (1.0, 0.5, 0.25, 0.125)]
    numerical_tolerance = 64.0 * torch.finfo(torch.float32).eps
    assert all(residual64(candidate)[0, 1] > numerical_tolerance for candidate in ordinary_candidates)

    ordinary = ordinary_candidates[0]
    correction_rows, correction_bounds, _ = _hard_rows(
        ordinary,
        residual64(ordinary),
        jacobian,
        residual_indices=(1,),
        residual_upper=(0.0,),
        coordinate_bounds=(),
        joint_velocity=no_velocity,
        velocity_lower=infinite_lower,
        velocity_upper=infinite_upper,
        step_seconds=1.0,
    )
    correction, correction_multipliers = _active_set_qp(
        hessian,
        torch.zeros_like(gradient),
        correction_rows,
        correction_bounds,
    )
    _assert_kkt(
        hessian,
        torch.zeros_like(gradient),
        correction_rows,
        correction_bounds,
        correction,
        correction_multipliers,
    )
    corrected = ordinary + correction.reshape_as(initial)
    assert residual64(corrected)[0, 1] <= 2.0e-8
    current_cost = 0.5 * residual64(initial)[0, 0].square()
    corrected_cost = 0.5 * residual64(corrected)[0, 0].square()
    model_descent = -(gradient @ tangent_step)
    assert current_cost - corrected_cost >= 1.0e-4 * model_descent

    output, feasible, direction_valid, globalization_succeeded = _solve_public(
        optimizer,
        initial.to(torch.float32),
        pose_weights.to(torch.float32),
        temporal_weights.to(torch.float32),
        damping,
        inequalities=IKTrajectorySolver.ResidualInequalities(
            residual_indices=torch.tensor((1,), dtype=torch.int32),
            upper=torch.zeros(1),
        ),
        coordinate_bounds=IKTrajectorySolver.CoordinateBounds(
            coordinate_indices=torch.empty(0, dtype=torch.int32),
            dof_indices=torch.empty(0, dtype=torch.int32),
            lower=torch.empty(0),
            upper=torch.empty(0),
        ),
        joint_velocity=no_velocity.to(torch.float32),
        velocity_lower=infinite_lower.to(torch.float32),
        velocity_upper=infinite_upper.to(torch.float32),
    )
    torch.testing.assert_close(
        output.to(torch.float64),
        corrected,
        atol=4.0e-4,
        rtol=4.0e-4,
    )
    assert (
        min(torch.linalg.vector_norm(output.to(torch.float64) - candidate) for candidate in ordinary_candidates)
        > 1.0e-3
    )
    assert residual64(output.to(torch.float64))[0, 1] <= 2.0e-5
    assert feasible.item()
    assert direction_valid.item()
    assert globalization_succeeded.item()
