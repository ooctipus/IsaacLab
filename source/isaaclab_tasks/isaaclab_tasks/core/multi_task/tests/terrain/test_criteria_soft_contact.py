# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for cached Position final-measure acceptance."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab_tasks.core.multi_task.terrain.retarget.buffer import RetargetBuffer
from isaaclab_tasks.core.multi_task.terrain.retarget.criteria import (
    JointWithinLimit,
    SolverCostOutlier,
    evaluate_foot_position_error,
    evaluate_support_polygon_stability,
)
from isaaclab_tasks.core.multi_task.terrain.retarget.criteria_cfg import (
    FootPositionErrorCfg,
    JointWithinLimitCfg,
    SolverCostOutlierCfg,
    SupportPolygonStabilityCfg,
)

DEVICE = "cuda:0"


@pytest.fixture(scope="module", autouse=True)
def _init_warp():
    wp.init()


def _mock_joint_limit_pipeline(
    lower: list[float],
    upper: list[float],
    device: str,
    default_q: list[float] | None = None,
    fk_joint_range: float = 10.0,
) -> object:
    """Minimal pipeline stand-in exposing Newton joint limits."""
    if default_q is None:
        default_q = [0.0] * (7 + len(lower) - 6)
    return SimpleNamespace(
        kinematics=SimpleNamespace(
            default_joint_q=np.asarray(default_q, dtype=np.float32),
            device=device,
            find_joint_scalar_coordinates=lambda _pattern: (
                list(range(7, 7 + len(lower) - 6)),
                list(range(6, len(lower))),
                [f"joint_{index}" for index in range(len(lower) - 6)],
            ),
            topology=SimpleNamespace(
                joint_limit_lower=np.asarray(lower, dtype=np.float32),
                joint_limit_upper=np.asarray(upper, dtype=np.float32),
            ),
        ),
        sampler_cfg=SimpleNamespace(
            fk_joint_range=fk_joint_range,
            fk_joint_range_overrides={},
        ),
    )


@pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
def test_joint_within_limit_uses_scaled_interval():
    """JointWithinLimit should reject every non-root joint outside 90% of its limits."""
    device = torch.device(DEVICE)
    # Six root-DOF limits are ignored. The three non-root joints have
    # symmetric limits, so ``limit_ratio = 0.9`` gives exact bounds:
    # [-0.9, 0.9], [-1.8, 1.8], [-0.45, 0.45].
    lower = [-10.0] * 6 + [-1.0, -2.0, -0.5]
    upper = [10.0] * 6 + [1.0, 2.0, 0.5]
    pipeline = _mock_joint_limit_pipeline(lower, upper, DEVICE)

    buf = RetargetBuffer(4, joint_coord_count=10, num_bodies=1, num_contacts=1, device=DEVICE)
    buf.joint_q_result_t[:, 0:7] = 100.0  # root coordinates must not affect this criterion.
    buf.joint_q_result_t[0, 7:10] = torch.tensor([0.0, 0.0, 0.0], device=device)
    buf.joint_q_result_t[1, 7:10] = torch.tensor([0.9, -1.8, 0.45], device=device)
    buf.joint_q_result_t[2, 7:10] = torch.tensor([0.91, 0.0, 0.0], device=device)
    buf.joint_q_result_t[3, 7:10] = torch.tensor([0.0, -1.81, 0.0], device=device)

    criterion = JointWithinLimit(JointWithinLimitCfg(limit_ratio=0.9), pipeline)
    result = criterion(buf, torch.arange(4, device=device))

    assert bool(result[0])
    assert bool(result[1])
    assert not bool(result[2])
    assert not bool(result[3])

    unbounded_pipeline = _mock_joint_limit_pipeline(
        [-1.0e10] * 9,
        [1.0e10] * 9,
        DEVICE,
        fk_joint_range=1.0,
    )
    unbounded_buf = RetargetBuffer(2, joint_coord_count=10, num_bodies=1, num_contacts=1, device=DEVICE)
    unbounded_buf.joint_q_result_t[0, 7:10] = torch.tensor([0.9, -0.9, 0.0], device=device)
    unbounded_buf.joint_q_result_t[1, 7:10] = torch.tensor([0.91, 0.0, 0.0], device=device)

    unbounded_criterion = JointWithinLimit(JointWithinLimitCfg(limit_ratio=0.9), unbounded_pipeline)
    unbounded_result = unbounded_criterion(unbounded_buf, torch.arange(2, device=device))
    assert bool(unbounded_result[0])
    assert not bool(unbounded_result[1])


@pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
def test_foot_position_error_reads_cached_measure_and_masks_air_slots() -> None:
    """Acceptance masks cached per-foot errors without recomputing final FK."""
    device = torch.device(DEVICE)
    buffer = RetargetBuffer(2, joint_coord_count=16, num_bodies=4, num_contacts=4, device=DEVICE)
    is_contact = torch.ones(2, 4, dtype=torch.bool, device=device)
    is_contact[0, 3] = False
    buffer.is_contact_t[:8] = is_contact.reshape(-1)
    candidates = SimpleNamespace(
        buffer=buffer,
        num_rows=2,
        foot_position_error=torch.tensor(
            ((0.0, 0.0, 0.0, 0.5), (0.0, 0.0, 0.0, 0.5)),
            device=device,
        ),
    )
    rows = torch.arange(2, device=device)
    max_result = evaluate_foot_position_error(FootPositionErrorCfg(max_err=0.02, aggregate="max"), candidates, rows)
    assert max_result.tolist() == [True, False]
    sum_result = evaluate_foot_position_error(FootPositionErrorCfg(max_err=0.02, aggregate="sum"), candidates, rows)
    assert sum_result.tolist() == [True, False]


def test_support_polygon_stability_reads_objective_margin_and_contact_count() -> None:
    """Acceptance applies bounds to the objective's exact cached signed margin."""
    candidates = SimpleNamespace(
        stability_margin=torch.tensor((0.1, 0.1, -0.1)),
        active_contact_count=torch.tensor((4, 2, 3), dtype=torch.int32),
    )
    cfg = SupportPolygonStabilityCfg(minimum_contacts=3, minimum_margin=0.0)
    rows = torch.arange(3)
    result = evaluate_support_polygon_stability(cfg, candidates, rows)
    assert result.tolist() == [True, False, False]
    strict = evaluate_support_polygon_stability(
        SupportPolygonStabilityCfg(minimum_contacts=4, minimum_margin=0.15), candidates, rows
    )
    assert strict.tolist() == [False, False, False]


def test_solver_cost_outlier_keeps_the_complete_solved_population_threshold() -> None:
    """Active-row filtering must not recompute a population statistic on prior survivors."""
    costs = torch.tensor((1.0, 2.0, 100.0, 101.0))
    criterion = SolverCostOutlier(
        SolverCostOutlierCfg(threshold_multiplier=10.0),
        SimpleNamespace(solver_costs=costs),
    )

    result = criterion(None, torch.tensor((2, 3)))

    assert result.tolist() == [False, False]
