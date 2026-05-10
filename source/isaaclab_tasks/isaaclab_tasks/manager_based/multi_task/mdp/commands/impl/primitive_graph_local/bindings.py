# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-owned primitive graph plan.

This backend keeps the primitive-family queue layout, but makes reusable
target-independent computations explicit shared IR nodes:

``current/projection/reduction -> target-specific consumer -> local output``

The public command wrapper stays unchanged; the backend owns the graph layout
and decides which schedules materialize producer rows.
"""

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
    PrimitiveProducerQueue,
    StateAccess,
    SubtaskSpec,
)
from ..mega_kernel.bindings import (
    BodyPosSlabBinding,
    CopySlabBinding,
    DynamicSlabBinding,
    RotationBinding,
    build_combined_copy_slab_metadata,
    build_rotation_bindings,
    build_slab_bindings,
)
from ..schedules import (
    NUM_SCHEDULES,
    SCHEDULE_DIRECT_QUAT_DELTA,
    SCHEDULE_DIRECT_SCALAR_DELTA,
    SCHEDULE_DIRECT_VEC3_DELTA,
    SCHEDULE_SCALAR_SUM_DELTA,
    SCHEDULE_STATE_KERNELS,
    SCHEDULE_VEC3_THRESHOLD_PAIR_DIFF_DELTA,
    SCHEDULE_VEC3_THRESHOLD_SUM_DELTA,
    SCHEDULE_VEC3_THRESHOLD_VECTOR_DELTA,
    build_subtask_schedule_ids,
    validate_schedule_support,
)

if TYPE_CHECKING:
    from ...multi_task_command_warp import MultiTaskCommandWarp

_CONTACT_SCHEDULES = (
    SCHEDULE_VEC3_THRESHOLD_VECTOR_DELTA,
    SCHEDULE_VEC3_THRESHOLD_SUM_DELTA,
    SCHEDULE_VEC3_THRESHOLD_PAIR_DIFF_DELTA,
)
# Per-kind materialization breakeven. Light producers (1-4 array reads) need
# many consumers to amortize the global-memory roundtrip cost; heavy producers
# (multi-element reductions, multi-schedule contact predicate) earn it back at
# much lower fanout. Determined empirically via bench_tile_fusion_testbed: heavy
# producers see ~3.5x at fanout 4, light producers break even closer to 16.
_MIN_FANOUT_LIGHT = 16  # direct vec3 / scalar / quat
_MIN_FANOUT_HEAVY = 4  # scalar_sum (joint reduction), contact (multi-schedule reuse)


def _build_signature_tables(
    command: MultiTaskCommandWarp,
    schedule_ids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group subtasks by target-independent unified-buffer gather block."""
    scheduled_state_ids = {
        state_kernel_id for schedule_id in schedule_ids for state_kernel_id in SCHEDULE_STATE_KERNELS[schedule_id]
    }
    s = command.spec
    state_kernel_id_cpu = s.state_kernel_id.detach().cpu()
    gather_indices_cpu = s.gather_indices_flat.detach().cpu()
    gather_offset_cpu = s.subtask_gather_offset.detach().cpu()
    gather_count_cpu = s.subtask_gather_count.detach().cpu()

    signature_id_by_gather: dict[tuple[int, ...], int] = {}
    representative_subtasks: list[int] = []
    subtask_signature = torch.full((int(s.state_kernel_id.numel()),), -1, device=command.device, dtype=torch.int32)
    for sid, state_kernel_id in enumerate(state_kernel_id_cpu.tolist()):
        if int(state_kernel_id) not in scheduled_state_ids:
            continue
        start = int(gather_offset_cpu[sid])
        count = int(gather_count_cpu[sid])
        key = tuple(int(v) for v in gather_indices_cpu[start : start + count].tolist())
        signature_id = signature_id_by_gather.get(key)
        if signature_id is None:
            signature_id = len(representative_subtasks)
            signature_id_by_gather[key] = signature_id
            representative_subtasks.append(sid)
        subtask_signature[sid] = signature_id

    signature_subtask = torch.tensor(representative_subtasks or [-1], device=command.device, dtype=torch.int32)
    return subtask_signature, signature_subtask


def _build_schedule_mask(state_kernel_ids: torch.Tensor, schedule_id: int) -> torch.Tensor:
    """Return active-item mask for one fused primitive schedule."""
    state_kernel_group = SCHEDULE_STATE_KERNELS[schedule_id]
    mask = state_kernel_ids == state_kernel_group[0]
    for state_kernel_id in state_kernel_group[1:]:
        mask |= state_kernel_ids == state_kernel_id
    return mask


@dataclass
class PrimitiveProducerPlan:
    """Long-lived node queue for one target-independent producer kind."""

    subtask_signature_i32: torch.Tensor
    signature_subtask_i32: torch.Tensor
    env_ids_i32: torch.Tensor
    subtask_ids_i32: torch.Tensor
    consumer_node_ids_i32: torch.Tensor
    consumer_indices_i32: torch.Tensor
    consumer_offsets_i32: torch.Tensor
    consumer_counts_i32: torch.Tensor
    count_i32: torch.Tensor
    count: int
    queue: PrimitiveProducerQueue


def _make_producer_plan(
    command: MultiTaskCommandWarp,
    max_work: int,
    schedule_ids: tuple[int, ...],
) -> PrimitiveProducerPlan:
    """Allocate one producer queue and its subtask signature tables."""
    subtask_signature_i32, signature_subtask_i32 = _build_signature_tables(command, schedule_ids)
    env_ids_i32 = torch.empty(max_work, device=command.device, dtype=torch.int32)
    subtask_ids_i32 = torch.empty(max_work, device=command.device, dtype=torch.int32)
    consumer_node_ids_i32 = torch.full((max_work,), -1, device=command.device, dtype=torch.int32)
    consumer_indices_i32 = torch.empty(max_work, device=command.device, dtype=torch.int32)
    consumer_offsets_i32 = torch.zeros(max_work, device=command.device, dtype=torch.int32)
    consumer_counts_i32 = torch.zeros(max_work, device=command.device, dtype=torch.int32)
    count_i32 = torch.zeros(1, device=command.device, dtype=torch.int32)

    queue = PrimitiveProducerQueue()
    queue.subtask_signature = wp.from_torch(subtask_signature_i32)
    queue.signature_subtask = wp.from_torch(signature_subtask_i32)
    queue.env_ids = wp.from_torch(env_ids_i32)
    queue.subtask_ids = wp.from_torch(subtask_ids_i32)
    queue.consumer_node_ids = wp.from_torch(consumer_node_ids_i32)
    queue.consumer_indices = wp.from_torch(consumer_indices_i32)
    queue.consumer_offsets = wp.from_torch(consumer_offsets_i32)
    queue.consumer_counts = wp.from_torch(consumer_counts_i32)
    queue.count = wp.from_torch(count_i32)

    return PrimitiveProducerPlan(
        subtask_signature_i32=subtask_signature_i32,
        signature_subtask_i32=signature_subtask_i32,
        env_ids_i32=env_ids_i32,
        subtask_ids_i32=subtask_ids_i32,
        consumer_node_ids_i32=consumer_node_ids_i32,
        consumer_indices_i32=consumer_indices_i32,
        consumer_offsets_i32=consumer_offsets_i32,
        consumer_counts_i32=consumer_counts_i32,
        count_i32=count_i32,
        count=0,
        queue=queue,
    )


def _reset_producer_plan(plan: PrimitiveProducerPlan) -> None:
    """Clear dynamic producer state before rebuilding the graph plan."""
    plan.consumer_node_ids_i32.fill_(-1)
    plan.consumer_offsets_i32.zero_()
    plan.consumer_counts_i32.zero_()
    plan.count_i32.zero_()
    plan.count = 0


def _refresh_producer_nodes(
    plan: PrimitiveProducerPlan,
    env_ids: torch.Tensor,
    subtask_ids: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Build unique ``(env, producer_signature)`` nodes and return per-item ids."""
    node_by_item = torch.full((env_ids.numel(),), -1, device=env_ids.device, dtype=torch.int32)
    if not bool(active_mask.any()):
        return node_by_item

    signature_ids = plan.subtask_signature_i32[subtask_ids].long()
    pairs = torch.stack([env_ids[active_mask].long(), signature_ids[active_mask]], dim=1)
    unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    count = int(unique_pairs.shape[0])
    plan.env_ids_i32[:count] = unique_pairs[:, 0].to(torch.int32)
    plan.subtask_ids_i32[:count] = plan.signature_subtask_i32[unique_pairs[:, 1].long()]
    plan.count_i32[0] = count
    plan.count = count
    node_by_item[active_mask] = inverse.to(torch.int32)
    return node_by_item


def _set_grouped_consumers(
    plan: PrimitiveProducerPlan,
    row_start: int,
    row_stop: int,
    node_ids: torch.Tensor,
) -> None:
    """Group local output rows by producer node for one producer kind."""
    if row_stop == row_start:
        return
    local_rows = torch.arange(row_start, row_stop, device=node_ids.device, dtype=torch.int32)
    order = torch.argsort(node_ids)
    sorted_nodes = node_ids[order]
    sorted_rows = local_rows[order]
    unique_nodes, counts = torch.unique_consecutive(sorted_nodes, return_counts=True)
    offsets = counts.cumsum(0) - counts
    count = int(sorted_rows.numel())
    plan.consumer_indices_i32[:count] = sorted_rows
    plan.consumer_offsets_i32[unique_nodes.long()] = offsets.to(torch.int32)
    plan.consumer_counts_i32[unique_nodes.long()] = counts.to(torch.int32)


def _should_materialize_producer(work_count: int, node_count: int, threshold: int) -> bool:
    """Return whether a producer kind has enough fanout to pay for materialization."""
    return node_count > 0 and work_count >= node_count * threshold


def _sort_command_slots_by_schedule(command: MultiTaskCommandWarp, schedule_ids: torch.Tensor) -> None:
    """Sort each env's active slots by primitive schedule for dense output locality."""
    slot_ids = torch.arange(command.k_max, device=command.device, dtype=torch.long).unsqueeze(0)
    slot_ids = slot_ids.expand(command.num_envs, -1)
    active = slot_ids < command._env_slot_count.long().unsqueeze(1)
    subtask_ids = command._env_subtask_ids.long().clamp_min(0)
    slot_schedule_ids = schedule_ids[subtask_ids]
    slot_schedule_ids = torch.where(active, slot_schedule_ids, torch.full_like(slot_schedule_ids, NUM_SCHEDULES))
    slot_order = torch.argsort(slot_schedule_ids, dim=1, stable=True)

    command._env_subtask_ids[:] = torch.gather(command._env_subtask_ids, 1, slot_order)
    command._env_slot_offsets[:] = torch.gather(command._env_slot_offsets, 1, slot_order)
    command._env_slot_strides[:] = torch.gather(command._env_slot_strides, 1, slot_order)


@dataclass
class PrimitiveGraphLocalPlan:
    """Long-lived Warp plan for primitive graph execution."""

    subtask_schedule_ids_i32: torch.Tensor
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
    vec3_nodes: PrimitiveProducerPlan
    scalar_nodes: PrimitiveProducerPlan
    quat_nodes: PrimitiveProducerPlan
    scalar_sum_nodes: PrimitiveProducerPlan
    contact_nodes: PrimitiveProducerPlan
    direct_vec3: torch.Tensor
    direct_scalar: torch.Tensor
    direct_quat: torch.Tensor
    scalar_sum: torch.Tensor
    contact_mask: torch.Tensor
    max_work: int
    total_work: int
    vec3_signature_count: int
    scalar_signature_count: int
    quat_signature_count: int
    scalar_sum_signature_count: int
    contact_signature_count: int
    vec3_count: int
    scalar_count: int
    quat_count: int
    scalar_sum_count: int
    contact_count: int
    use_vec3_graph: bool
    use_scalar_graph: bool
    use_quat_graph: bool
    use_scalar_sum_graph: bool
    use_dense_graph_consumer: bool
    schedule_counts_py: list[int]
    env_slots: EnvSlots
    queue: PrimitiveLocalQueue
    spec: SubtaskSpec
    state: StateAccess
    composer_state: ComposerState
    outputs: Outputs
    subtask_schedule_ids_wp: wp.array(dtype=int)
    direct_vec3_wp: wp.array2d(dtype=float)
    direct_scalar_wp: wp.array(dtype=float)
    direct_quat_wp: wp.array2d(dtype=float)
    local_delta_wp: wp.array2d(dtype=float)
    local_error_wp: wp.array(dtype=float)
    local_activation_wp: wp.array(dtype=float)
    scalar_sum_wp: wp.array(dtype=float)
    contact_mask_wp: wp.array2d(dtype=float)
    rotations: tuple[RotationBinding, ...]
    episode_length_buf_wp: wp.array
    effective_max_episode_length_wp: wp.array
    copy_slabs: tuple[CopySlabBinding, ...] = ()
    body_pos_slabs: tuple[BodyPosSlabBinding, ...] = ()
    dynamic_slabs: tuple[DynamicSlabBinding, ...] = ()
    combined_slab_sources_wp: tuple[wp.array, ...] = ()
    combined_slab_cumsizes_wp: wp.array | None = None
    combined_slab_offsets_wp: wp.array | None = None
    combined_slab_total_size: int = 0
    combined_slab_num_slabs: int = 0
    combined_slab_cumsizes_torch: torch.Tensor | None = None
    combined_slab_offsets_torch: torch.Tensor | None = None


def build_primitive_graph_local_plan(command: MultiTaskCommandWarp) -> PrimitiveGraphLocalPlan:
    """Construct the backend-owned primitive graph plan."""
    wp.init()
    s = command.spec
    validate_schedule_support(s.state_kernel_id, backend_name="primitive_graph_local")

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
    subtask_schedule_ids_i32 = build_subtask_schedule_ids(
        s.state_kernel_id,
        backend_name="primitive_graph_local",
    )

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
    vec3_nodes = _make_producer_plan(command, max_work, (SCHEDULE_DIRECT_VEC3_DELTA,))
    scalar_nodes = _make_producer_plan(command, max_work, (SCHEDULE_DIRECT_SCALAR_DELTA,))
    quat_nodes = _make_producer_plan(command, max_work, (SCHEDULE_DIRECT_QUAT_DELTA,))
    scalar_sum_nodes = _make_producer_plan(command, max_work, (SCHEDULE_SCALAR_SUM_DELTA,))
    contact_nodes = _make_producer_plan(command, max_work, _CONTACT_SCHEDULES)

    flat_env_ids_i32 = torch.empty(max_work, device=command.device, dtype=torch.int32)
    flat_slot_ids_i32 = torch.empty(max_work, device=command.device, dtype=torch.int32)
    flat_subtask_ids_i32 = torch.empty(max_work, device=command.device, dtype=torch.int32)
    flat_target_offsets_i32 = torch.empty(max_work, device=command.device, dtype=torch.int32)
    schedule_offsets_i32 = torch.zeros(NUM_SCHEDULES, device=command.device, dtype=torch.int32)
    schedule_counts_i32 = torch.zeros(NUM_SCHEDULES, device=command.device, dtype=torch.int32)
    flat_count_i32 = torch.zeros(1, device=command.device, dtype=torch.int32)

    direct_vec3 = torch.empty((max_work, 3), device=command.device, dtype=command._unified_buffer.dtype)
    direct_scalar = torch.empty(max_work, device=command.device, dtype=command._unified_buffer.dtype)
    direct_quat = torch.empty((max_work, 4), device=command.device, dtype=command._unified_buffer.dtype)
    scalar_sum = torch.empty(max_work, device=command.device, dtype=command._unified_buffer.dtype)
    contact_mask = torch.zeros((max_work, 4), device=command.device, dtype=command._unified_buffer.dtype)

    queue = PrimitiveLocalQueue()
    queue.env_ids = wp.from_torch(flat_env_ids_i32)
    queue.slot_ids = wp.from_torch(flat_slot_ids_i32)
    queue.subtask_ids = wp.from_torch(flat_subtask_ids_i32)
    queue.target_offsets = wp.from_torch(flat_target_offsets_i32)
    queue.slot_local_index = wp.from_torch(local_outputs.slot_local_index)
    queue.schedule_offsets = wp.from_torch(schedule_offsets_i32)
    queue.schedule_counts = wp.from_torch(schedule_counts_i32)
    queue.count = wp.from_torch(flat_count_i32)

    copy_slabs, body_pos_slabs, dynamic_slabs = build_slab_bindings(command)
    combined_slabs = build_combined_copy_slab_metadata(command, copy_slabs)

    plan = PrimitiveGraphLocalPlan(
        subtask_schedule_ids_i32=subtask_schedule_ids_i32,
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
        vec3_nodes=vec3_nodes,
        scalar_nodes=scalar_nodes,
        quat_nodes=quat_nodes,
        scalar_sum_nodes=scalar_sum_nodes,
        contact_nodes=contact_nodes,
        direct_vec3=direct_vec3,
        direct_scalar=direct_scalar,
        direct_quat=direct_quat,
        scalar_sum=scalar_sum,
        contact_mask=contact_mask,
        max_work=max_work,
        total_work=0,
        vec3_signature_count=max(0, int(vec3_nodes.signature_subtask_i32.numel())),
        scalar_signature_count=max(0, int(scalar_nodes.signature_subtask_i32.numel())),
        quat_signature_count=max(0, int(quat_nodes.signature_subtask_i32.numel())),
        scalar_sum_signature_count=max(0, int(scalar_sum_nodes.signature_subtask_i32.numel())),
        contact_signature_count=max(0, int(contact_nodes.signature_subtask_i32.numel())),
        vec3_count=0,
        scalar_count=0,
        quat_count=0,
        scalar_sum_count=0,
        contact_count=0,
        use_vec3_graph=False,
        use_scalar_graph=False,
        use_quat_graph=False,
        use_scalar_sum_graph=False,
        use_dense_graph_consumer=False,
        schedule_counts_py=[0] * NUM_SCHEDULES,
        env_slots=env_slots,
        queue=queue,
        spec=spec_struct,
        state=state,
        composer_state=composer_state,
        outputs=outputs,
        subtask_schedule_ids_wp=wp.from_torch(subtask_schedule_ids_i32),
        direct_vec3_wp=wp.from_torch(direct_vec3),
        direct_scalar_wp=wp.from_torch(direct_scalar),
        direct_quat_wp=wp.from_torch(direct_quat),
        local_delta_wp=wp.from_torch(local_outputs.local_delta),
        local_error_wp=wp.from_torch(local_outputs.local_error),
        local_activation_wp=wp.from_torch(local_outputs.local_activation),
        scalar_sum_wp=wp.from_torch(scalar_sum),
        contact_mask_wp=wp.from_torch(contact_mask),
        rotations=build_rotation_bindings(command),
        episode_length_buf_wp=wp.from_torch(command._env.episode_length_buf),
        effective_max_episode_length_wp=wp.from_torch(command._effective_max_episode_length),
        copy_slabs=copy_slabs,
        body_pos_slabs=body_pos_slabs,
        dynamic_slabs=dynamic_slabs,
        combined_slab_sources_wp=combined_slabs["sources_wp"],
        combined_slab_cumsizes_wp=combined_slabs["cumsizes_wp"],
        combined_slab_offsets_wp=combined_slabs["offsets_wp"],
        combined_slab_total_size=combined_slabs["total_size"],
        combined_slab_num_slabs=combined_slabs["num_slabs"],
        combined_slab_cumsizes_torch=combined_slabs["cumsizes_torch"],
        combined_slab_offsets_torch=combined_slabs["offsets_torch"],
    )
    refresh_primitive_graph_local_plan(command, plan)
    return plan


def refresh_primitive_graph_local_plan(command: MultiTaskCommandWarp, plan: PrimitiveGraphLocalPlan) -> None:
    """Refresh primitive graph queues from the command's current assignment."""
    _sort_command_slots_by_schedule(command, plan.subtask_schedule_ids_i32)
    plan.schedule_offsets_i32.zero_()
    plan.schedule_counts_i32.zero_()
    plan.flat_count_i32.zero_()
    _reset_producer_plan(plan.vec3_nodes)
    _reset_producer_plan(plan.scalar_nodes)
    _reset_producer_plan(plan.quat_nodes)
    _reset_producer_plan(plan.scalar_sum_nodes)
    _reset_producer_plan(plan.contact_nodes)
    command._outputs.slot_local_index.zero_()

    cursor = 0
    slot_idx = torch.arange(command.k_max, device=command.device, dtype=torch.int32).unsqueeze(0)
    valid = slot_idx < command._env_slot_count.unsqueeze(1)
    if not bool(valid.any()):
        plan.total_work = 0
        plan.vec3_count = 0
        plan.scalar_count = 0
        plan.quat_count = 0
        plan.scalar_sum_count = 0
        plan.contact_count = 0
        plan.use_vec3_graph = False
        plan.use_scalar_graph = False
        plan.use_quat_graph = False
        plan.use_scalar_sum_graph = False
        plan.use_dense_graph_consumer = False
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

    vec3_mask = _build_schedule_mask(state_kernel_ids, SCHEDULE_DIRECT_VEC3_DELTA)
    scalar_mask = _build_schedule_mask(state_kernel_ids, SCHEDULE_DIRECT_SCALAR_DELTA)
    quat_mask = _build_schedule_mask(state_kernel_ids, SCHEDULE_DIRECT_QUAT_DELTA)
    scalar_sum_mask = _build_schedule_mask(state_kernel_ids, SCHEDULE_SCALAR_SUM_DELTA)
    contact_schedule_mask = torch.zeros_like(state_kernel_ids, dtype=torch.bool)
    for schedule_id in _CONTACT_SCHEDULES:
        contact_schedule_mask |= _build_schedule_mask(state_kernel_ids, schedule_id)

    vec3_node_by_item = _refresh_producer_nodes(plan.vec3_nodes, env_ids, subtask_ids, vec3_mask)
    scalar_node_by_item = _refresh_producer_nodes(plan.scalar_nodes, env_ids, subtask_ids, scalar_mask)
    quat_node_by_item = _refresh_producer_nodes(plan.quat_nodes, env_ids, subtask_ids, quat_mask)
    scalar_sum_node_by_item = _refresh_producer_nodes(plan.scalar_sum_nodes, env_ids, subtask_ids, scalar_sum_mask)
    contact_node_by_item = _refresh_producer_nodes(plan.contact_nodes, env_ids, subtask_ids, contact_schedule_mask)

    schedule_counts_py: list[int] = []
    for schedule_id, state_kernel_group in enumerate(SCHEDULE_STATE_KERNELS):
        group_mask = _build_schedule_mask(state_kernel_ids, schedule_id)
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
        if schedule_id == SCHEDULE_DIRECT_VEC3_DELTA:
            node_ids = vec3_node_by_item[group_mask]
            plan.vec3_nodes.consumer_node_ids_i32[cursor:stop] = node_ids
            _set_grouped_consumers(plan.vec3_nodes, cursor, stop, node_ids)
        elif schedule_id == SCHEDULE_DIRECT_SCALAR_DELTA:
            node_ids = scalar_node_by_item[group_mask]
            plan.scalar_nodes.consumer_node_ids_i32[cursor:stop] = node_ids
            _set_grouped_consumers(plan.scalar_nodes, cursor, stop, node_ids)
        elif schedule_id == SCHEDULE_DIRECT_QUAT_DELTA:
            node_ids = quat_node_by_item[group_mask]
            plan.quat_nodes.consumer_node_ids_i32[cursor:stop] = node_ids
            _set_grouped_consumers(plan.quat_nodes, cursor, stop, node_ids)
        elif schedule_id == SCHEDULE_SCALAR_SUM_DELTA:
            node_ids = scalar_sum_node_by_item[group_mask]
            plan.scalar_sum_nodes.consumer_node_ids_i32[cursor:stop] = node_ids
            _set_grouped_consumers(plan.scalar_sum_nodes, cursor, stop, node_ids)
        elif schedule_id in _CONTACT_SCHEDULES:
            plan.contact_nodes.consumer_node_ids_i32[cursor:stop] = contact_node_by_item[group_mask]
        command._outputs.slot_local_index[env_ids[group_mask], slot_ids[group_mask]] = local_indices[cursor:stop]
        cursor = stop

    if cursor != int(env_ids.numel()):
        raise ValueError("primitive_graph_local failed to lower every active subtask into a primitive schedule.")
    plan.total_work = cursor
    plan.vec3_count = plan.vec3_nodes.count
    plan.scalar_count = plan.scalar_nodes.count
    plan.quat_count = plan.quat_nodes.count
    plan.scalar_sum_count = plan.scalar_sum_nodes.count
    plan.contact_count = plan.contact_nodes.count
    plan.schedule_counts_py = schedule_counts_py
    plan.use_vec3_graph = _should_materialize_producer(
        schedule_counts_py[SCHEDULE_DIRECT_VEC3_DELTA], plan.vec3_count, _MIN_FANOUT_LIGHT
    )
    plan.use_scalar_graph = _should_materialize_producer(
        schedule_counts_py[SCHEDULE_DIRECT_SCALAR_DELTA], plan.scalar_count, _MIN_FANOUT_LIGHT
    )
    plan.use_quat_graph = _should_materialize_producer(
        schedule_counts_py[SCHEDULE_DIRECT_QUAT_DELTA], plan.quat_count, _MIN_FANOUT_LIGHT
    )
    plan.use_scalar_sum_graph = _should_materialize_producer(
        schedule_counts_py[SCHEDULE_SCALAR_SUM_DELTA], plan.scalar_sum_count, _MIN_FANOUT_HEAVY
    )
    plan.use_dense_graph_consumer = (
        (schedule_counts_py[SCHEDULE_DIRECT_VEC3_DELTA] == 0 or plan.use_vec3_graph)
        and (schedule_counts_py[SCHEDULE_DIRECT_SCALAR_DELTA] == 0 or plan.use_scalar_graph)
        and (schedule_counts_py[SCHEDULE_DIRECT_QUAT_DELTA] == 0 or plan.use_quat_graph)
        and (schedule_counts_py[SCHEDULE_SCALAR_SUM_DELTA] == 0 or plan.use_scalar_sum_graph)
    )
    plan.flat_count_i32[0] = cursor
