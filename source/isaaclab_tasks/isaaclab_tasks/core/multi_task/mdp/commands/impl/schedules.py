# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fused-schedule lowering for Warp command backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from .kernel_ids import STATE_KERNEL_ID

if TYPE_CHECKING:
    import torch

SCHEDULE_DIRECT_VEC3_DELTA = 0
SCHEDULE_DIRECT_SCALAR_DELTA = 1
SCHEDULE_DIRECT_QUAT_DELTA = 2
SCHEDULE_VEC3_THRESHOLD_VECTOR_DELTA = 3
SCHEDULE_VEC3_THRESHOLD_SUM_DELTA = 4
SCHEDULE_VEC3_THRESHOLD_PAIR_DIFF_DELTA = 5
SCHEDULE_SCALAR_SUM_DELTA = 6

SCHEDULE_STATE_KERNELS = (
    # SCHEDULE_DIRECT_VEC3_DELTA
    (
        int(STATE_KERNEL_ID.BODY_POS),
        int(STATE_KERNEL_ID.BODY_LIN_VEL),
        int(STATE_KERNEL_ID.BODY_ANG_VEL),
    ),
    # SCHEDULE_DIRECT_SCALAR_DELTA
    (
        int(STATE_KERNEL_ID.JOINT_POS),
        int(STATE_KERNEL_ID.JOINT_VEL),
        int(STATE_KERNEL_ID.BODY_POS_Z),
    ),
    # SCHEDULE_DIRECT_QUAT_DELTA
    (int(STATE_KERNEL_ID.BODY_QUAT),),
    # SCHEDULE_VEC3_THRESHOLD_VECTOR_DELTA
    (int(STATE_KERNEL_ID.BODY_CONTACT),),
    # SCHEDULE_VEC3_THRESHOLD_SUM_DELTA
    (int(STATE_KERNEL_ID.BODY_CONTACT_COUNT),),
    # SCHEDULE_VEC3_THRESHOLD_PAIR_DIFF_DELTA
    (int(STATE_KERNEL_ID.BODY_CONTACT_COUNT_DIFF),),
    # SCHEDULE_SCALAR_SUM_DELTA
    (int(STATE_KERNEL_ID.JOINT_MECH_POWER),),
)
NUM_SCHEDULES = len(SCHEDULE_STATE_KERNELS)


def validate_schedule_support(state_kernel_ids: torch.Tensor, backend_name: str) -> None:
    """Fail if ``state_kernel_ids`` contains a semantic kernel without schedule lowering."""
    supported = {state_kernel_id for group in SCHEDULE_STATE_KERNELS for state_kernel_id in group}
    requested = {int(state_kernel_id) for state_kernel_id in state_kernel_ids.detach().cpu().tolist()}
    unsupported = requested - supported
    if unsupported:
        raise ValueError(
            f"{backend_name} does not support state kernel ids {sorted(unsupported)}. "
            "Add an explicit fused-schedule lowering first."
        )


def build_subtask_schedule_ids(state_kernel_ids: torch.Tensor, backend_name: str) -> wp.array:
    """Return schedule id per subtask from semantic state-kernel ids, as a Warp array."""
    validate_schedule_support(state_kernel_ids, backend_name)
    state_kernel_to_schedule: dict[int, int] = {
        state_kernel_id: schedule_id
        for schedule_id, group in enumerate(SCHEDULE_STATE_KERNELS)
        for state_kernel_id in group
    }
    schedule_ids = [state_kernel_to_schedule[int(k)] for k in state_kernel_ids.detach().cpu().tolist()]
    return wp.array(schedule_ids, dtype=wp.int32, device=str(state_kernel_ids.device))
