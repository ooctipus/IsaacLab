# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared memory-bounded IK execution."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.core.multi_task.kinematics import execute_ik_batches, plan_ik_memory


def _linear_estimate(batch_size: int) -> int:
    return 100 + 10 * batch_size


@pytest.mark.parametrize(
    ("free_bytes", "expected_capacity", "expected_reserve"),
    (
        (175, 4, 35),
        (250, 10, 50),
        (500, 20, 100),
    ),
)
def test_memory_plan_uses_largest_batch_below_live_cuda_budget(
    free_bytes: int,
    expected_capacity: int,
    expected_reserve: int,
) -> None:
    """CUDA capacity must expand to the exact largest batch allowed by current memory."""
    plan = plan_ik_memory(20, "cuda:0", _linear_estimate, device_free_bytes=free_bytes)

    assert plan.max_safe_capacity == expected_capacity
    assert plan.batch_capacity == expected_capacity
    assert plan.safety_reserve_bytes == expected_reserve
    assert plan.memory_budget_bytes == free_bytes - expected_reserve
    assert plan.peak_additional_workspace_bytes == _linear_estimate(expected_capacity)
    assert plan.fixed_bytes == 100
    assert plan.bytes_per_problem == 10


def test_memory_plan_balances_capacity_without_adding_a_launch() -> None:
    """Capacity shrinks to the minimum size preserving the fastest batch count."""
    plan = plan_ik_memory(5, "cuda:0", _linear_estimate, device_free_bytes=175)

    assert plan.max_safe_capacity == 4
    assert plan.batch_capacity == 3
    assert (plan.problem_count + plan.batch_capacity - 1) // plan.batch_capacity == 2
    assert plan.peak_additional_workspace_bytes == _linear_estimate(3)


def test_memory_plan_keeps_one_full_cpu_batch() -> None:
    """CPU execution must avoid arbitrary chunking when no CUDA memory budget applies."""
    plan = plan_ik_memory(37, "cpu", _linear_estimate)

    assert plan.batch_capacity == 37
    assert plan.device_free_bytes is None
    assert plan.memory_budget_bytes is None
    assert plan.peak_additional_workspace_bytes == _linear_estimate(37)


def test_executor_builds_one_workspace_and_preserves_every_row_across_tail() -> None:
    """One capacity workspace must cover full batches and a padded tail without changing row semantics."""
    source = torch.arange(10)
    output = torch.full_like(source, -1)
    build_capacities: list[int] = []
    intervals: list[tuple[int, int]] = []

    def build_batch(capacity: int) -> torch.Tensor:
        build_capacities.append(capacity)
        return torch.empty(capacity, dtype=source.dtype)

    def solve_batch(resources, start, stop, max_iterations, tolerance, check_interval):
        valid = stop - start
        resources[:valid].copy_(source[start:stop])
        if valid < resources.shape[0]:
            resources[valid:].copy_(resources[0])
        output[start:stop].copy_(resources[:valid])
        intervals.append((start, stop))
        return SimpleNamespace(
            iterations=min(max_iterations, check_interval * 2),
            converged=tolerance > 0.0,
            initial_mean_cost=1.0,
            final_mean_cost=0.0,
        )

    stats = execute_ik_batches(
        problem_count=10,
        device="cuda:0",
        estimate_memory=_linear_estimate,
        build_batch=build_batch,
        solve_batch=solve_batch,
        max_iterations=200,
        convergence_tolerance=1.0e-6,
        convergence_check_interval=3,
        device_free_bytes=175,
    )

    torch.testing.assert_close(output, source)
    assert build_capacities == [4]
    assert intervals == [(0, 4), (4, 8), (8, 10)]
    assert stats.problem_count == 10
    assert stats.batch_capacity == 4
    assert stats.batch_count == 3
    assert stats.iterations_min == 6
    assert stats.iterations_total == 18
    assert stats.iterations_max == 6
    assert stats.converged_batches == 3


def test_cuda_memory_query_synchronizes_and_releases_inactive_torch_cache(monkeypatch) -> None:
    """Warp sizing must see memory that Torch can release at this one-time boundary."""
    from isaaclab_tasks.core.multi_task.kinematics import ik_execution

    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(f"sync:{device}"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1234, 5678))

    assert ik_execution._cuda_free_memory(torch.device("cuda:1")) == 1234
    assert calls == ["sync:cuda:1", "empty_cache"]


def test_memory_plan_rejects_a_workload_that_cannot_fit_one_problem() -> None:
    """A live budget below one exact Newton estimate must fail before allocating a solver."""
    with pytest.raises(MemoryError, match="One IK problem requires"):
        plan_ik_memory(10, "cuda:0", _linear_estimate, device_free_bytes=130)
