# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for is_contact-aware criteria (Phase B soft polygon).

:class:`~isaaclab_tasks.core.multi_task.terrain.retarget.criteria.FootPositionError`
and :class:`~isaaclab_tasks.core.multi_task.terrain.retarget.criteria.SupportPolygonStability`
must consult ``buffer.is_contact_t`` so that air slots (``is_contact = False``)
are skipped by foot-position error aggregation and support-region tests.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab_tasks.core.multi_task.terrain.retarget.buffer import RetargetBuffer
from isaaclab_tasks.core.multi_task.terrain.retarget.criteria import (
    FootPositionError,
    JointWithinLimit,
    SupportPolygonStability,
)
from isaaclab_tasks.core.multi_task.terrain.retarget.criteria_cfg import (
    FootPositionErrorCfg,
    JointWithinLimitCfg,
    SupportPolygonStabilityCfg,
)

DEVICE = "cuda:0"


@pytest.fixture(scope="module", autouse=True)
def _init_warp():
    wp.init()


def _mock_pipeline(num_bodies: int, foot_ids: list[int]) -> object:
    """Minimal pipeline stand-in exposing only ``kin.model.body_count`` and ``foot_body_ids``."""
    return SimpleNamespace(
        kin=SimpleNamespace(model=SimpleNamespace(body_count=num_bodies)),
        foot_body_ids=foot_ids,
    )


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
        kin=SimpleNamespace(
            default_joint_q=np.asarray(default_q, dtype=np.float32),
            device=device,
            find_joint_dof_indices=lambda _pattern: [],
            model=SimpleNamespace(
                joint_limit_lower=wp.array(lower, dtype=wp.float32, device=device),
                joint_limit_upper=wp.array(upper, dtype=wp.float32, device=device),
            ),
        ),
        cfg=SimpleNamespace(
            sampler=SimpleNamespace(
                fk_joint_range=fk_joint_range,
                fk_joint_range_overrides={},
            ),
        ),
    )


def _populate_buffer(buf: RetargetBuffer, body_q_per: torch.Tensor, ct: torch.Tensor, is_contact: torch.Tensor):
    N, nb, _ = body_q_per.shape
    nc = ct.shape[1]
    buf.body_q_t[: N * nb] = body_q_per.reshape(-1, 7)
    buf.contact_targets_t[: N * nc] = ct.reshape(-1, 3)
    buf.is_contact_t[: N * nc] = is_contact.reshape(-1)
    buf.num_written = N


@pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
def test_foot_position_error_ignores_air_slots():
    """Air-slot foot drift must not trip FootPositionError (max / sum)."""
    nc = 4
    nb = 4
    # Two candidates. Both have the same foot targets. Candidate 0 has a
    # ~0.5m air-slot drift on slot 3 (far above max_err = 0.02); it should
    # still pass because slot 3 is marked ``is_contact = False``. Candidate
    # 1 has the same drift but with all-contact ``is_contact`` and should
    # be rejected.
    device = torch.device(DEVICE)
    buf = RetargetBuffer(2, joint_coord_count=16, num_bodies=nb, num_contacts=nc, device=DEVICE)

    targets = torch.tensor(
        [
            [0.30, 0.20, 0.0],
            [-0.30, 0.20, 0.0],
            [-0.30, -0.20, 0.0],
            [0.30, -0.20, 0.0],
        ],
        device=device,
    )
    ct = targets.unsqueeze(0).expand(2, nc, 3).contiguous()

    body_q = torch.zeros(2, nb, 7, device=device)
    body_q[..., 6] = 1.0  # identity quat w
    body_q[:, :, :3] = targets.unsqueeze(0)
    body_q[:, 3, :3] = targets[3] + torch.tensor([0.5, 0.0, 0.0], device=device)

    is_contact = torch.ones(2, nc, dtype=torch.bool, device=device)
    is_contact[0, 3] = False  # air slot -- drift must be ignored

    _populate_buffer(buf, body_q, ct, is_contact)

    pipeline = _mock_pipeline(nb, list(range(nc)))
    criterion = FootPositionError(FootPositionErrorCfg(max_err=0.02, aggregate="max"), pipeline)
    result = criterion(buf, N=2)
    assert bool(result[0]), "Air-slot drift must not fail FootPositionError"
    assert not bool(result[1]), "Contact-slot drift must fail FootPositionError"

    criterion_sum = FootPositionError(FootPositionErrorCfg(max_err=0.02, aggregate="sum"), pipeline)
    result_sum = criterion_sum(buf, N=2)
    assert bool(result_sum[0]), "Air-slot drift must not contribute to sum"
    assert not bool(result_sum[1]), "Contact-slot drift must fail sum aggregate"


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
    result = criterion(buf, N=4)

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
    unbounded_result = unbounded_criterion(unbounded_buf, N=2)
    assert bool(unbounded_result[0])
    assert not bool(unbounded_result[1])


@pytest.mark.skipif(not wp.is_device_available("cuda:0"), reason="GPU required")
def test_support_polygon_stability_variable_active_contacts():
    """SupportPolygonStability must use only active slots per candidate."""
    nc = 4
    nb = 4
    device = torch.device(DEVICE)
    buf = RetargetBuffer(3, joint_coord_count=16, num_bodies=nb, num_contacts=nc, device=DEVICE)

    # Three candidates, all with the same four foot xy targets:
    #   slot 0 = (+0.30, +0.20) FR
    #   slot 1 = (-0.30, +0.20) FL
    #   slot 2 = (-0.30, -0.20) HL
    #   slot 3 = (+0.30, -0.20) HR
    #
    # Candidate 0: all active, base at centroid → inside hull → pass.
    # Candidate 1: slots 0,1 active (front-only segment at y = +0.20), base
    #   at (0, +0.20) → on the segment → pass.  Base at (0, 0) (centroid)
    #   is lateral to the segment, perpendicular distance 0.20 > tol, so
    #   flipping base to (0,0) with the same mask must fail.
    # Candidate 2: slots 0,1,2 active (triangle), base at the triangle
    #   centroid → pass; moving base outside triangle must fail.
    targets = torch.tensor(
        [
            [0.30, 0.20, 0.0],
            [-0.30, 0.20, 0.0],
            [-0.30, -0.20, 0.0],
            [0.30, -0.20, 0.0],
        ],
        device=device,
    )
    ct = targets.unsqueeze(0).expand(3, nc, 3).contiguous()
    body_q = torch.zeros(3, nb, 7, device=device)
    body_q[..., 6] = 1.0
    joint_q = torch.zeros(3, buf.joint_coord_count, device=device)

    # Base xy is stored in joint_q_result_t[:, 0:2].
    joint_q[0, 0:2] = torch.tensor([0.0, 0.0], device=device)  # centroid
    joint_q[1, 0:2] = torch.tensor([0.0, 0.20], device=device)  # on front segment
    tri_centroid = targets[:3, :2].mean(dim=0)
    joint_q[2, 0:2] = tri_centroid  # inside triangle slots 0/1/2

    is_contact = torch.zeros(3, nc, dtype=torch.bool, device=device)
    is_contact[0] = True
    is_contact[1, 0] = True
    is_contact[1, 1] = True
    is_contact[2, 0] = True
    is_contact[2, 1] = True
    is_contact[2, 2] = True

    buf.body_q_t[: 3 * nb] = body_q.reshape(-1, 7)
    buf.contact_targets_t[: 3 * nc] = ct.reshape(-1, 3)
    buf.is_contact_t[: 3 * nc] = is_contact.reshape(-1)
    buf.joint_q_result_t[:3] = joint_q

    crit = SupportPolygonStability(SupportPolygonStabilityCfg())
    ok = crit(buf, N=3)
    assert bool(ok[0]), "Full 4-foot polygon at centroid should be stable"
    assert bool(ok[1]), "2-foot segment with base on segment should be stable"
    assert bool(ok[2]), "3-foot triangle with base at triangle centroid should be stable"

    # Negative: move each base off-support and re-check.
    joint_q_bad = joint_q.clone()
    joint_q_bad[0, 0:2] = torch.tensor([5.0, 5.0], device=device)  # outside hull
    joint_q_bad[1, 0:2] = torch.tensor([0.0, -0.10], device=device)  # far from segment
    joint_q_bad[2, 0:2] = torch.tensor([5.0, 5.0], device=device)  # outside triangle
    buf.joint_q_result_t[:3] = joint_q_bad
    bad = crit(buf, N=3)
    assert not bool(bad[0]), "Base far outside 4-foot hull should fail"
    assert not bool(bad[1]), "Base far from 2-foot segment should fail"
    assert not bool(bad[2]), "Base outside 3-foot triangle should fail"
