# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Execute phase for primitive-queued local outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..kernels_wp import (
    dispatch_primitive_local_direct_quat,
    dispatch_primitive_local_direct_scalar,
    dispatch_primitive_local_direct_vec3,
    dispatch_primitive_local_scalar_sum,
    dispatch_primitive_local_vec3_threshold_pair_diff,
    dispatch_primitive_local_vec3_threshold_sum,
    dispatch_primitive_local_vec3_threshold_vector,
)

if TYPE_CHECKING:
    from ..multi_task_command_warp import MultiTaskCommandWarp
    from .bindings import PrimitiveQueueLocalPlan


_PRIMITIVE_KERNELS = (
    dispatch_primitive_local_direct_vec3,
    dispatch_primitive_local_direct_scalar,
    dispatch_primitive_local_direct_quat,
    dispatch_primitive_local_vec3_threshold_vector,
    dispatch_primitive_local_vec3_threshold_sum,
    dispatch_primitive_local_vec3_threshold_pair_diff,
    dispatch_primitive_local_scalar_sum,
)


def dispatch_primitive_queue_local_warp(command: MultiTaskCommandWarp, plan: PrimitiveQueueLocalPlan) -> None:
    """Run one branch-free launch per non-empty primitive schedule."""
    if plan.total_work == 0:
        return
    device = str(command.device)
    for kernel, count in zip(_PRIMITIVE_KERNELS, plan.schedule_counts_py):
        if count != 0:
            wp.launch(kernel, dim=count, inputs=[plan.queue, plan.spec, plan.state, plan.outputs], device=device)
