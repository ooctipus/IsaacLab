# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-owned execution plan for primitive-queued local outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import warp as wp

from ...kernels_wp import (
    ComposerState,
    EnvSlots,
    Outputs,
    PrimitiveLocalQueue,
    StateAccess,
    SubtaskSpec,
)
from ..mega_kernel.bindings import RotationBinding, build_rotation_bindings
from ..schedules import NUM_SCHEDULES, SCHEDULE_STATE_KERNELS, validate_schedule_support

if TYPE_CHECKING:
    from ...multi_task_command_warp import MultiTaskCommandWarp


@dataclass
class PrimitiveQueueLocalPlan:
    """Long-lived Warp plan for primitive queues with local composer rows."""

    state_kernel_id_i32: torch.Tensor
    metric_kernel_id_i32: torch.Tensor
    activation_kernel_id_i32: torch.Tensor
    state_stride_i32: torch.Tensor
    canonical_offset_i32: torch.Tensor
    is_instant_i32: torch.Tensor
    is_tracking_i32: torch.Tensor
    subtask_gather_offset_i32: torch.Tensor
    subtask_gather_count_i32: torch.Tensor
    gather_indices_flat_i32: torch.Tensor
    flat_env_ids_i32: torch.Tensor
    flat_slot_ids_i32: torch.Tensor
    flat_subtask_ids_i32: torch.Tensor
    flat_target_offsets_i32: torch.Tensor
    schedule_offsets_i32: torch.Tensor
    schedule_counts_i32: torch.Tensor
    flat_count_i32: torch.Tensor
    max_work: int
    total_work: int
    schedule_counts_py: list[int]
    env_slots: EnvSlots
    queue: PrimitiveLocalQueue
    spec: SubtaskSpec
    state: StateAccess
    composer_state: ComposerState
    outputs: Outputs
    local_delta_wp: wp.array2d(dtype=float)
    local_error_wp: wp.array(dtype=float)
    local_activation_wp: wp.array(dtype=float)
    rotations: tuple[RotationBinding, ...]


def build_primitive_queue_local_plan(command: MultiTaskCommandWarp) -> PrimitiveQueueLocalPlan:
    """Construct the backend-owned primitive-queue execution plan."""
    wp.init()
    s = command.spec
    validate_schedule_support(s.state_kernel_id, backend_name="primitive_queue_local")

    state_kernel_id_i32 = s.state_kernel_id.to(torch.int32)
    metric_kernel_id_i32 = s.metric_kernel_id.to(torch.int32)
    activation_kernel_id_i32 = s.activation_kernel_id.to(torch.int32)
    state_stride_i32 = s.state_stride.to(torch.int32)
    canonical_offset_i32 = s.canonical_offset.to(torch.int32)
    is_instant_i32 = s.is_instant.to(torch.int32)
    is_tracking_i32 = s.is_tracking.to(torch.int32)
    subtask_gather_offset_i32 = s.subtask_gather_offset.to(torch.int32)
    subtask_gather_count_i32 = s.subtask_gather_count.to(torch.int32)
    gather_indices_flat_i32 = s.gather_indices_flat.to(torch.int32)

    env_slots = EnvSlots()
    env_slots.subtask_ids = wp.from_torch(command._env_subtask_ids)
    env_slots.slot_count = wp.from_torch(command._env_slot_count)
    env_slots.slot_offsets = wp.from_torch(command._env_slot_offsets)

    spec_struct = SubtaskSpec()
    spec_struct.state_kernel_id = wp.from_torch(state_kernel_id_i32)
    spec_struct.metric_kernel_id = wp.from_torch(metric_kernel_id_i32)
    spec_struct.activation_kernel_id = wp.from_torch(activation_kernel_id_i32)
    spec_struct.activation_kernel_param = wp.from_torch(s.activation_kernel_param)
    spec_struct.state_stride = wp.from_torch(state_stride_i32)
    spec_struct.canonical_offset = wp.from_torch(canonical_offset_i32)
    spec_struct.is_instant_flag = wp.from_torch(is_instant_i32)
    spec_struct.is_tracking_flag = wp.from_torch(is_tracking_i32)
    spec_struct.gather_offset = wp.from_torch(subtask_gather_offset_i32)
    spec_struct.gather_count = wp.from_torch(subtask_gather_count_i32)
    spec_struct.gather_indices_flat = wp.from_torch(gather_indices_flat_i32)

    state = StateAccess()
    state.unified = wp.from_torch(command._unified_buffer)
    state.targets_flat = wp.from_torch(command._targets_flat)

    composer_state = ComposerState()
    composer_state.sum_activation = wp.from_torch(command._sum_activation)
    composer_state.transit_steps = wp.from_torch(command._transit_steps)
    composer_state.instant_achieved = wp.from_torch(command._instant_achieved)

    outputs = Outputs()
    outputs.buf_error = wp.from_torch(command._buf_error)
    outputs.buf_activation = wp.from_torch(command._buf_activation)
    outputs.command_reach = wp.from_torch(command._command_reach)
    outputs.command_track = wp.from_torch(command._command_track)
    outputs.task_reward = wp.from_torch(command._task_reward)
    outputs.task_done_success = wp.from_torch(command._task_done_success)
    outputs.progress = wp.from_torch(command._progress)

    local_outputs = command._outputs
    max_work = command.num_envs * command.k_max
    flat_env_ids_i32 = torch.empty(max_work, device=command.device, dtype=torch.int32)
    flat_slot_ids_i32 = torch.empty(max_work, device=command.device, dtype=torch.int32)
    flat_subtask_ids_i32 = torch.empty(max_work, device=command.device, dtype=torch.int32)
    flat_target_offsets_i32 = torch.empty(max_work, device=command.device, dtype=torch.int32)
    schedule_offsets_i32 = torch.zeros(NUM_SCHEDULES, device=command.device, dtype=torch.int32)
    schedule_counts_i32 = torch.zeros(NUM_SCHEDULES, device=command.device, dtype=torch.int32)
    flat_count_i32 = torch.zeros(1, device=command.device, dtype=torch.int32)

    queue = PrimitiveLocalQueue()
    queue.env_ids = wp.from_torch(flat_env_ids_i32)
    queue.slot_ids = wp.from_torch(flat_slot_ids_i32)
    queue.subtask_ids = wp.from_torch(flat_subtask_ids_i32)
    queue.target_offsets = wp.from_torch(flat_target_offsets_i32)
    queue.slot_local_index = wp.from_torch(local_outputs.slot_local_index)
    queue.schedule_offsets = wp.from_torch(schedule_offsets_i32)
    queue.schedule_counts = wp.from_torch(schedule_counts_i32)
    queue.count = wp.from_torch(flat_count_i32)

    plan = PrimitiveQueueLocalPlan(
        state_kernel_id_i32=state_kernel_id_i32,
        metric_kernel_id_i32=metric_kernel_id_i32,
        activation_kernel_id_i32=activation_kernel_id_i32,
        state_stride_i32=state_stride_i32,
        canonical_offset_i32=canonical_offset_i32,
        is_instant_i32=is_instant_i32,
        is_tracking_i32=is_tracking_i32,
        subtask_gather_offset_i32=subtask_gather_offset_i32,
        subtask_gather_count_i32=subtask_gather_count_i32,
        gather_indices_flat_i32=gather_indices_flat_i32,
        flat_env_ids_i32=flat_env_ids_i32,
        flat_slot_ids_i32=flat_slot_ids_i32,
        flat_subtask_ids_i32=flat_subtask_ids_i32,
        flat_target_offsets_i32=flat_target_offsets_i32,
        schedule_offsets_i32=schedule_offsets_i32,
        schedule_counts_i32=schedule_counts_i32,
        flat_count_i32=flat_count_i32,
        max_work=max_work,
        total_work=0,
        schedule_counts_py=[0] * NUM_SCHEDULES,
        env_slots=env_slots,
        queue=queue,
        spec=spec_struct,
        state=state,
        composer_state=composer_state,
        outputs=outputs,
        local_delta_wp=wp.from_torch(local_outputs.local_delta),
        local_error_wp=wp.from_torch(local_outputs.local_error),
        local_activation_wp=wp.from_torch(local_outputs.local_activation),
        rotations=build_rotation_bindings(command),
    )
    refresh_primitive_queue_local_plan(command, plan)
    return plan


def refresh_primitive_queue_local_plan(command: MultiTaskCommandWarp, plan: PrimitiveQueueLocalPlan) -> None:
    """Refresh primitive queues from the command's current task assignment."""
    plan.schedule_offsets_i32.zero_()
    plan.schedule_counts_i32.zero_()
    plan.flat_count_i32.zero_()
    command._outputs.slot_local_index.zero_()

    cursor = 0
    slot_idx = torch.arange(command.k_max, device=command.device, dtype=torch.int32).unsqueeze(0)
    valid = slot_idx < command._env_slot_count.unsqueeze(1)
    if not bool(valid.any()):
        plan.total_work = 0
        plan.schedule_counts_py = [0] * NUM_SCHEDULES
        return

    env_ids, slot_ids = valid.nonzero(as_tuple=True)
    subtask_ids = command._env_subtask_ids[env_ids, slot_ids].long()
    state_kernel_ids = command.spec.state_kernel_id[subtask_ids].long()
    target_offsets = command._env_slot_offsets[env_ids, slot_ids]
    env_ids_i32 = env_ids.to(torch.int32)
    slot_ids_i32 = slot_ids.to(torch.int32)
    subtask_ids_i32 = subtask_ids.to(torch.int32)
    local_indices = torch.arange(env_ids.numel(), device=command.device, dtype=torch.int32)

    schedule_counts_py: list[int] = []
    for schedule_id, state_kernel_group in enumerate(SCHEDULE_STATE_KERNELS):
        group_mask = state_kernel_ids == state_kernel_group[0]
        for state_kernel_id in state_kernel_group[1:]:
            group_mask |= state_kernel_ids == state_kernel_id
        count = int(group_mask.sum().item())
        schedule_counts_py.append(count)
        plan.schedule_offsets_i32[schedule_id] = cursor
        plan.schedule_counts_i32[schedule_id] = count
        if count == 0:
            continue
        stop = cursor + count
        plan.flat_env_ids_i32[cursor:stop] = env_ids_i32[group_mask]
        plan.flat_slot_ids_i32[cursor:stop] = slot_ids_i32[group_mask]
        plan.flat_subtask_ids_i32[cursor:stop] = subtask_ids_i32[group_mask]
        plan.flat_target_offsets_i32[cursor:stop] = target_offsets[group_mask]
        command._outputs.slot_local_index[env_ids[group_mask], slot_ids[group_mask]] = local_indices[cursor:stop]
        cursor = stop

    if cursor != int(env_ids.numel()):
        raise ValueError("primitive_queue_local failed to lower every active subtask into a primitive schedule.")
    plan.total_work = cursor
    plan.schedule_counts_py = schedule_counts_py
    plan.flat_count_i32[0] = cursor
