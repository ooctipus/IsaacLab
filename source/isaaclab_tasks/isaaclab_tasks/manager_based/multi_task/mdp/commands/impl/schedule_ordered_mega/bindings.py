# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-owned slot-order plan for schedule-ordered mega dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..mega_kernel.bindings import MegaKernelPlan, build_mega_kernel_plan
from ..schedules import NUM_SCHEDULES, build_subtask_schedule_ids

if TYPE_CHECKING:
    from ..multi_task_command_warp import MultiTaskCommandWarp


@dataclass
class ScheduleOrderedMegaPlan:
    """Long-lived Warp plan whose env slot tables are sorted by fused schedule."""

    mega: MegaKernelPlan
    subtask_schedule_ids_i32: torch.Tensor


def build_schedule_ordered_mega_plan(command: MultiTaskCommandWarp) -> ScheduleOrderedMegaPlan:
    """Construct the schedule-ordered mega-kernel execution plan."""
    mega = build_mega_kernel_plan(command)
    subtask_schedule_ids_i32 = build_subtask_schedule_ids(
        command.spec.state_kernel_id,
        backend_name="schedule_ordered_mega",
    )
    plan = ScheduleOrderedMegaPlan(
        mega=mega,
        subtask_schedule_ids_i32=subtask_schedule_ids_i32,
    )
    env_ids = torch.arange(command.num_envs, device=command.device, dtype=torch.long)
    refresh_schedule_ordered_mega_plan(command, plan, env_ids)
    return plan


def refresh_schedule_ordered_mega_plan(
    command: MultiTaskCommandWarp,
    plan: ScheduleOrderedMegaPlan,
    env_ids: torch.Tensor,
) -> None:
    """Sort resampled env slot tables by fused schedule."""
    if env_ids.numel() == 0:
        return

    slot_ids = torch.arange(command.k_max, device=command.device, dtype=torch.long).unsqueeze(0)
    slot_ids = slot_ids.expand(env_ids.numel(), -1)
    active = slot_ids < command._env_slot_count[env_ids].long().unsqueeze(1)
    subtask_ids = command._env_subtask_ids[env_ids].long().clamp_min(0)
    schedule_ids = plan.subtask_schedule_ids_i32[subtask_ids]
    schedule_ids = torch.where(active, schedule_ids, torch.full_like(schedule_ids, NUM_SCHEDULES))
    slot_order = torch.argsort(schedule_ids, dim=1, stable=True)

    command._env_subtask_ids[env_ids] = torch.gather(command._env_subtask_ids[env_ids], 1, slot_order)
    command._env_slot_offsets[env_ids] = torch.gather(command._env_slot_offsets[env_ids], 1, slot_order)
    command._env_slot_strides[env_ids] = torch.gather(command._env_slot_strides[env_ids], 1, slot_order)
