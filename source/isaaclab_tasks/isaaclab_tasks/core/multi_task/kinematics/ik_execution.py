# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Memory-bounded execution of independent Newton IK problems."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import newton.ik as ik
import torch

_MIB = 1024 * 1024
_MINIMUM_CUDA_RESERVE_BYTES = 256 * _MIB
_CUDA_RESERVE_FRACTION = 0.05
_MAXIMUM_CUDA_RESERVE_FRACTION = 0.20

_BatchResources = TypeVar("_BatchResources")


@dataclass(frozen=True, slots=True)
class IKExecutionStatistics:
    """Immutable execution and memory facts for one independent IK workload."""

    problem_count: int
    max_safe_capacity: int
    batch_capacity: int
    batch_count: int
    iterations_total: int
    iterations_min: int
    iterations_max: int
    converged_batches: int
    fixed_bytes: int
    bytes_per_problem: int
    device_free_bytes: int | None
    safety_reserve_bytes: int
    memory_budget_bytes: int | None
    peak_additional_workspace_bytes: int


@dataclass(frozen=True, slots=True)
class IKMemoryPlan:
    """Balanced minimum-launch batch that fits one live device-memory budget."""

    problem_count: int
    max_safe_capacity: int
    batch_capacity: int
    fixed_bytes: int
    bytes_per_problem: int
    device_free_bytes: int | None
    safety_reserve_bytes: int
    memory_budget_bytes: int | None
    peak_additional_workspace_bytes: int


def plan_ik_memory(
    problem_count: int,
    device: str | torch.device,
    estimate_memory: Callable[[int], int],
    *,
    device_free_bytes: int | None = None,
) -> IKMemoryPlan:
    """Select the smallest balanced IK batch preserving the minimum launch count.

    CPU workloads retain the full problem count. CUDA workloads reserve five
    percent of currently free memory, with a 256 MiB floor and a twenty-percent
    cap for small devices. The remaining budget is searched against Newton's
    exact workload estimate. The selected maximum determines the minimum launch
    count; the allocated capacity is then balanced down to avoid wasted tail rows.
    No domain-specific batch-size setting participates.

    Args:
        problem_count: Number of independent IK problems.
        device: Torch device on which the solve executes.
        estimate_memory: Exact bytes allocated by a batch of the requested size.
        device_free_bytes: Optional CUDA free-memory value used by deterministic tests.

    Returns:
        Immutable selected capacity and memory evidence.

    Raises:
        ValueError: If the problem count or an estimate is invalid.
        MemoryError: If one IK problem does not fit the live CUDA budget.
    """
    if type(problem_count) is not int or problem_count < 1:
        raise ValueError("IK problem_count must be a positive integer.")
    torch_device = torch.device(device)
    estimate_one = _estimate_bytes(estimate_memory, 1)
    estimate_two = _estimate_bytes(estimate_memory, 2)
    bytes_per_problem = max(0, estimate_two - estimate_one)
    fixed_bytes = max(0, estimate_one - bytes_per_problem)

    if torch_device.type != "cuda":
        peak = _estimate_bytes(estimate_memory, problem_count)
        return IKMemoryPlan(
            problem_count=problem_count,
            max_safe_capacity=problem_count,
            batch_capacity=problem_count,
            fixed_bytes=fixed_bytes,
            bytes_per_problem=bytes_per_problem,
            device_free_bytes=None,
            safety_reserve_bytes=0,
            memory_budget_bytes=None,
            peak_additional_workspace_bytes=peak,
        )

    if device_free_bytes is None:
        device_free_bytes = _cuda_free_memory(torch_device)
    if type(device_free_bytes) is not int or device_free_bytes < 1:
        raise ValueError("CUDA free memory must be a positive integer.")
    reserve = max(_MINIMUM_CUDA_RESERVE_BYTES, round(device_free_bytes * _CUDA_RESERVE_FRACTION))
    reserve = min(reserve, round(device_free_bytes * _MAXIMUM_CUDA_RESERVE_FRACTION))
    budget = device_free_bytes - reserve
    if estimate_one > budget:
        raise MemoryError(
            f"One IK problem requires {estimate_one} bytes, but only {budget} bytes remain after the "
            f"{reserve}-byte CUDA safety reserve."
        )

    low = 1
    high = problem_count
    while low < high:
        middle = (low + high + 1) // 2
        if _estimate_bytes(estimate_memory, middle) <= budget:
            low = middle
        else:
            high = middle - 1
    max_safe_capacity = low
    batch_count = (problem_count + max_safe_capacity - 1) // max_safe_capacity
    batch_capacity = (problem_count + batch_count - 1) // batch_count
    peak = _estimate_bytes(estimate_memory, batch_capacity)
    return IKMemoryPlan(
        problem_count=problem_count,
        max_safe_capacity=max_safe_capacity,
        batch_capacity=batch_capacity,
        fixed_bytes=fixed_bytes,
        bytes_per_problem=bytes_per_problem,
        device_free_bytes=device_free_bytes,
        safety_reserve_bytes=reserve,
        memory_budget_bytes=budget,
        peak_additional_workspace_bytes=peak,
    )


def execute_ik_batches(
    problem_count: int,
    device: str | torch.device,
    estimate_memory: Callable[[int], int],
    build_batch: Callable[[int], _BatchResources],
    solve_batch: Callable[[_BatchResources, int, int, int, float | None, int], ik.IKSolveResult],
    *,
    max_iterations: int,
    convergence_tolerance: float | None,
    convergence_check_interval: int,
    device_free_bytes: int | None = None,
) -> IKExecutionStatistics:
    """Allocate one balanced largest-safe IK workspace and reuse it for every batch.

    The batch resource factory runs exactly once at the selected capacity. The
    solve callback binds rows [start, stop) into the active workspace prefix and
    passes the exact active count to Newton. Inactive tail storage is never sampled,
    optimized, reduced, or copied.

    Args:
        problem_count: Number of independent IK problems.
        device: Torch device on which the solve executes.
        estimate_memory: Exact bytes allocated by a batch of the requested size.
        build_batch: Factory for one capacity-sized objective and solver workspace.
        solve_batch: Binder and solver for one source row interval.
        max_iterations: Maximum continuous optimizer iterations.
        convergence_tolerance: Maximum mean-cost change used for early convergence.
        convergence_check_interval: Optimizer iterations between convergence checks.
        device_free_bytes: Optional CUDA free-memory value used by deterministic tests.

    Returns:
        Immutable batch, convergence, and memory statistics.
    """
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("IK max_iterations must be a positive integer.")
    if convergence_tolerance is not None and convergence_tolerance < 0.0:
        raise ValueError("IK convergence_tolerance cannot be negative.")
    if type(convergence_check_interval) is not int or convergence_check_interval < 1:
        raise ValueError("IK convergence_check_interval must be a positive integer.")
    plan = plan_ik_memory(problem_count, device, estimate_memory, device_free_bytes=device_free_bytes)
    resources = build_batch(plan.batch_capacity)
    batch_count = 0
    iterations_total = 0
    iterations_min = max_iterations
    iterations_max = 0
    converged_batches = 0
    for start in range(0, problem_count, plan.batch_capacity):
        stop = min(start + plan.batch_capacity, problem_count)
        result = solve_batch(
            resources,
            start,
            stop,
            max_iterations,
            convergence_tolerance,
            convergence_check_interval,
        )
        batch_count += 1
        iterations_total += result.iterations
        iterations_min = min(iterations_min, result.iterations)
        iterations_max = max(iterations_max, result.iterations)
        converged_batches += result.converged
    return IKExecutionStatistics(
        problem_count=problem_count,
        max_safe_capacity=plan.max_safe_capacity,
        batch_capacity=plan.batch_capacity,
        batch_count=batch_count,
        iterations_total=iterations_total,
        iterations_min=iterations_min,
        iterations_max=iterations_max,
        converged_batches=converged_batches,
        fixed_bytes=plan.fixed_bytes,
        bytes_per_problem=plan.bytes_per_problem,
        device_free_bytes=plan.device_free_bytes,
        safety_reserve_bytes=plan.safety_reserve_bytes,
        memory_budget_bytes=plan.memory_budget_bytes,
        peak_additional_workspace_bytes=plan.peak_additional_workspace_bytes,
    )


def _cuda_free_memory(device: torch.device) -> int:
    """Release inactive Torch cache and return allocator-visible CUDA memory [byte]."""
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free_bytes, _ = torch.cuda.mem_get_info(device)
    return free_bytes


def _estimate_bytes(estimate_memory: Callable[[int], int], batch_size: int) -> int:
    estimate = estimate_memory(batch_size)
    if type(estimate) is not int or estimate < 1:
        raise ValueError("IK memory estimates must be positive integers.")
    return estimate
