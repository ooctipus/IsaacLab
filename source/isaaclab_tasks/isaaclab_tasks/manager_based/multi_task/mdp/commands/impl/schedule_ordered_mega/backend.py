# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend object for schedule-ordered mega-kernel execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..mega_kernel.compose import compose_warp
from ..mega_kernel.execute import dispatch_mega_warp
from ..mega_kernel.read import fill_unified_buffer_warp
from ..mega_kernel.rotation import rotate_canonical_slots_to_body_frame_warp
from .bindings import ScheduleOrderedMegaPlan, build_schedule_ordered_mega_plan, refresh_schedule_ordered_mega_plan

if TYPE_CHECKING:
    from ...multi_task_command_warp import MultiTaskCommandWarp


class ScheduleOrderedMegaBackend:
    """Execute one mega-kernel launch over backend-owned schedule-ordered slots."""

    name = "schedule_ordered_mega"

    def __init__(self, command: MultiTaskCommandWarp):
        self.plan: ScheduleOrderedMegaPlan = build_schedule_ordered_mega_plan(command)

    def on_resample(self, command: MultiTaskCommandWarp, env_ids: torch.Tensor) -> None:
        """Refresh schedule-ordered slot ranks after task assignment changes."""
        refresh_schedule_ordered_mega_plan(command, self.plan, env_ids)

    def dispatch(self, command: MultiTaskCommandWarp, valid_slots: torch.Tensor) -> None:
        """Run read, schedule-ordered execute, and body-frame rotation phases."""
        del valid_slots
        fill_unified_buffer_warp(command, self.plan.mega)
        dispatch_mega_warp(command, self.plan.mega)
        rotate_canonical_slots_to_body_frame_warp(command, self.plan.mega)

    def compose(self, command: MultiTaskCommandWarp, valid_slots: torch.Tensor) -> None:
        """Advance composer state and write reward outputs."""
        del valid_slots
        compose_warp(command, self.plan.mega)
