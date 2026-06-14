# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Execute phase for a packed fused-pipeline queue with legacy scatter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..kernels_wp import dispatch_packed_scatter_flat

if TYPE_CHECKING:
    from ..multi_task_command_warp import MultiTaskCommandWarp
    from .bindings import PackedScatterPlan


def dispatch_packed_scatter_warp(command: MultiTaskCommandWarp, plan: PackedScatterPlan) -> None:
    """Run a branch-light fused-pipeline dispatch over packed work."""
    if plan.total_work == 0:
        return
    device = str(command.device)
    wp.launch(
        dispatch_packed_scatter_flat,
        dim=plan.total_work,
        inputs=[plan.flat_queue, plan.spec, plan.state, plan.outputs],
        device=device,
    )
