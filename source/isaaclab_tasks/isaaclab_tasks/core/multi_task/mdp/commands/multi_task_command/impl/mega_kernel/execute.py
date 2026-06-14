# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Execute phase for the current branchy mega-kernel backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..kernels_wp import dispatch_mega

if TYPE_CHECKING:
    from ..multi_task_command_warp import MultiTaskCommandWarp
    from .bindings import MegaKernelPlan


def dispatch_mega_warp(command: MultiTaskCommandWarp, plan: MegaKernelPlan) -> None:
    """Launch the current branchy mega-kernel over ``(env, slot)``."""
    wp.launch(
        dispatch_mega,
        dim=(command.num_envs, command.k_max),
        inputs=[plan.env_slots, plan.spec, plan.state, plan.outputs],
        device=str(command.device),
    )
