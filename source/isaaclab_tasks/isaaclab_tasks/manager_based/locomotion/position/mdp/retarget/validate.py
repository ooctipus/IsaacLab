# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 4: Post-IK validation via user-defined acceptance criteria."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .buffer import RetargetBuffer

CriterionFn = Callable[[RetargetBuffer, int], torch.Tensor]
"""Signature for a validation criterion.

Args:
    buffer: The retarget buffer with IK results populated.
    n_active: Number of geometry-valid candidates (first ``n_active`` rows).

Returns:
    Boolean tensor of shape ``[n_active]`` -- ``True`` = passes this criterion.
"""


def validate_results(
    buffer: RetargetBuffer,
    criteria: dict[str, CriterionFn],
) -> dict[str, int]:
    """Run user-defined acceptance criteria and update ``buffer.ik_valid``.

    Each criterion is a callable ``(buffer, N) -> bool[N]``.  Criteria
    are evaluated in insertion order; the rejection breakdown reports
    the *first* failing criterion per candidate (waterfall).

    Args:
        buffer: Retarget buffer with ``joint_q_result`` populated.
        criteria: Ordered mapping from criterion name to callable.

    Returns:
        Rejection breakdown mapping each criterion name to the number
        of candidates it rejected (first-failure attribution), plus
        an ``"ok"`` key for candidates that passed everything.
    """
    N = buffer.num_geometry_valid
    if N == 0:
        return {}

    device = buffer.device
    masks: dict[str, torch.Tensor] = {}
    for name, fn in criteria.items():
        masks[name] = fn(buffer, N)

    # Waterfall rejection: attribute each failure to the first criterion it fails
    reject: dict[str, int] = {}
    passed = torch.ones(N, device=device, dtype=torch.bool)
    for name, mask in masks.items():
        failed_here = passed & ~mask
        reject[name] = int(failed_here.sum())
        passed = passed & mask

    reject["ok"] = int(passed.sum())

    buffer._ik_valid[:N] = passed

    buffer.num_ik_valid = reject["ok"]
    buffer.num_final_valid = reject["ok"]

    return reject
