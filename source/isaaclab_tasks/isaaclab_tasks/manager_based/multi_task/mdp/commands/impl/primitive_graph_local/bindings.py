# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-owned primitive graph plan.

Mathematical structure:

* **Producer nodes** — one per unique ``(env, signature)`` pair. A signature
  is a target-independent gather block (e.g. "the world-frame velocity of
  body B on this env"). Producers compute their value once per resample and
  write it into a per-kind buffer (``direct_vec3``, ``scalar_sum``,
  ``contact_mask``, etc.). Stored in :class:`ProducerNodeTable`, one table
  per producer kind.
* **Consumer items** — one per active subtask. Each consumer reads from
  exactly one producer node (its source signature) and writes a target-
  specific result to its local output row. Stored in the plan's
  ``consumer_*_i32`` arrays (one row per consumer), partitioned by kernel
  via ``schedule_offsets`` / ``schedule_counts``.
* **Adjacency** (producer → its consumers) — stored per producer table as
  CSR ``(consumer_offsets, consumer_counts)`` over ``consumer_indices``.
* **Reverse adjacency** (consumer → its producer) — stored per consumer as
  ``consumer_node_ids[item]`` inside the producer table.

All producer kinds always materialize; the only dispatch-time choice is
whether to use the fused-compose dense kernel (set at cfg time via
``use_parallel_compose``). Callers who want direct (non-materialized)
compute should select the ``primitive_queue_local`` backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import warp as wp

from ..kernel_ids import BUFFER_KIND
from ..kernels_wp import (
    ComposerState,
    EnvSlots,
    Outputs,
    PrimitiveLocalQueue,
    PrimitiveProducerQueue,
    StateAccess,
    SubtaskSpec,
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
    from ..multi_task_command_warp import MultiTaskCommandWarp

_CONTACT_SCHEDULES = (
    SCHEDULE_VEC3_THRESHOLD_VECTOR_DELTA,
    SCHEDULE_VEC3_THRESHOLD_SUM_DELTA,
    SCHEDULE_VEC3_THRESHOLD_PAIR_DIFF_DELTA,
)


@dataclass
class RotationBinding:
    """Warp views for rotating one asset's canonical vec3 command slots."""

    root_quat_w: torch.Tensor
    reach_offsets_i32: torch.Tensor
    track_offsets_i32: torch.Tensor
    root_quat_w_wp: wp.array
    reach_offsets_wp: wp.array
    track_offsets_wp: wp.array
    num_reach_offsets: int
    num_offsets: int


# -- Per-kind slab bindings (this backend's own) ----------------------------


@dataclass
class FloatSlabBinding:
    """Float-typed scene slab — JOINT_POS, JOINT_VEL."""

    source_wp: wp.array
    offset: int
    size: int


@dataclass
class Vec3SlabBinding:
    """vec3-typed slab without frame transform — BODY_LIN_VEL_W, BODY_ANG_VEL_W,
    CONTACT_NET_FORCES_W."""

    source_wp: wp.array
    offset: int
    size: int


@dataclass
class Vec3EnvLocalSlabBinding:
    """vec3-typed body slab with env-origin subtraction — BODY_POS_W."""

    source_wp: wp.array
    env_origins_wp: wp.array
    offset: int
    size: int


@dataclass
class QuatSlabBinding:
    """quat-typed body slab — BODY_QUAT_W."""

    source_wp: wp.array
    offset: int
    size: int


@dataclass
class JointMechPowerSlabBinding:
    """Computed slab ``|τ · q̇|`` — JOINT_MECH_POWER_ABS."""

    applied_torque_wp: wp.array
    joint_vel_wp: wp.array
    offset: int
    size: int


def _resolve_slabs(
    command: MultiTaskCommandWarp,
) -> tuple[
    tuple[FloatSlabBinding, ...],
    tuple[Vec3SlabBinding, ...],
    tuple[Vec3EnvLocalSlabBinding, ...],
    tuple[QuatSlabBinding, ...],
    tuple[JointMechPowerSlabBinding, ...],
]:
    """Resolve each spec slab to a stable ``wp.array`` view of scene data."""
    spec = command.spec
    kinds = spec.slab_buffer_kinds
    assets = spec.slab_asset_names
    offsets = spec.slab_offsets_py
    sizes = spec.slab_sizes_py

    env_origins_torch = command._env.scene.env_origins
    env_origins_vec3_wp = wp.array(
        ptr=env_origins_torch.data_ptr(),
        dtype=wp.vec3,
        shape=(command.num_envs,),
        device=str(command.device),
    )

    float_slabs: list[FloatSlabBinding] = []
    vec3_slabs: list[Vec3SlabBinding] = []
    vec3_env_local_slabs: list[Vec3EnvLocalSlabBinding] = []
    quat_slabs: list[QuatSlabBinding] = []
    joint_mech_power_slabs: list[JointMechPowerSlabBinding] = []

    for slab_id in range(len(kinds)):
        kind = int(kinds[slab_id])
        asset_name = assets[slab_id]
        offset = int(offsets[slab_id])
        size = int(sizes[slab_id])

        if kind == BUFFER_KIND.JOINT_POS:
            float_slabs.append(FloatSlabBinding(command._env.scene[asset_name].data.joint_pos.warp, offset, size))
        elif kind == BUFFER_KIND.JOINT_VEL:
            float_slabs.append(FloatSlabBinding(command._env.scene[asset_name].data.joint_vel.warp, offset, size))
        elif kind == BUFFER_KIND.BODY_POS_W:
            vec3_env_local_slabs.append(
                Vec3EnvLocalSlabBinding(
                    command._env.scene[asset_name].data.body_pos_w.warp, env_origins_vec3_wp, offset, size
                )
            )
        elif kind == BUFFER_KIND.BODY_LIN_VEL_W:
            vec3_slabs.append(Vec3SlabBinding(command._env.scene[asset_name].data.body_lin_vel_w.warp, offset, size))
        elif kind == BUFFER_KIND.BODY_ANG_VEL_W:
            vec3_slabs.append(Vec3SlabBinding(command._env.scene[asset_name].data.body_ang_vel_w.warp, offset, size))
        elif kind == BUFFER_KIND.BODY_QUAT_W:
            quat_slabs.append(QuatSlabBinding(command._env.scene[asset_name].data.body_quat_w.warp, offset, size))
        elif kind == BUFFER_KIND.CONTACT_NET_FORCES_W:
            vec3_slabs.append(
                Vec3SlabBinding(command._env.scene.sensors[asset_name].data.net_forces_w.warp, offset, size)
            )
        elif kind == BUFFER_KIND.JOINT_MECH_POWER_ABS:
            art = command._env.scene[asset_name]
            joint_mech_power_slabs.append(
                JointMechPowerSlabBinding(art.data.applied_torque.warp, art.data.joint_vel.warp, offset, size)
            )
        else:
            raise ValueError(
                f"Unsupported BUFFER_KIND {kind!r} for slab (asset={asset_name!r}, "
                f"offset={offset}, size={size}). primitive_graph_local requires every reader to "
                "expose a stable ``wp.array`` via ``ProxyArray.warp``."
            )

    return (
        tuple(float_slabs),
        tuple(vec3_slabs),
        tuple(vec3_env_local_slabs),
        tuple(quat_slabs),
        tuple(joint_mech_power_slabs),
    )


def _root_quat_torch(command: MultiTaskCommandWarp, asset_name: str) -> torch.Tensor:
    quat = command._env.scene[asset_name].data.root_quat_w
    if isinstance(quat, torch.Tensor):
        return quat
    if hasattr(quat, "torch"):
        return quat.torch
    return wp.to_torch(quat)


def _build_rotation_bindings(command: MultiTaskCommandWarp) -> tuple[RotationBinding, ...]:
    s = command.spec
    bindings: list[RotationBinding] = []
    asset_names = sorted(set(s.reach_rotatable_vec3_by_asset.keys()) | set(s.track_rotatable_vec3_by_asset.keys()))
    for asset_name in asset_names:
        reach_offsets = s.reach_rotatable_vec3_by_asset.get(asset_name, ())
        track_offsets = s.track_rotatable_vec3_by_asset.get(asset_name, ())
        num_offsets = len(reach_offsets) + len(track_offsets)
        if num_offsets == 0:
            continue
        root_quat_w = _root_quat_torch(command, asset_name)
        reach_offsets_i32 = torch.tensor(reach_offsets, device=command.device, dtype=torch.int32)
        track_offsets_i32 = torch.tensor(track_offsets, device=command.device, dtype=torch.int32)
        bindings.append(
            RotationBinding(
                root_quat_w=root_quat_w,
                reach_offsets_i32=reach_offsets_i32,
                track_offsets_i32=track_offsets_i32,
                root_quat_w_wp=wp.from_torch(root_quat_w),
                reach_offsets_wp=wp.from_torch(reach_offsets_i32),
                track_offsets_wp=wp.from_torch(track_offsets_i32),
                num_reach_offsets=len(reach_offsets),
                num_offsets=num_offsets,
            )
        )
    return tuple(bindings)


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

    signature_subtask = torch.tensor(representative_subtasks, device=command.device, dtype=torch.int32)
    return subtask_signature, signature_subtask


def _build_schedule_mask(state_kernel_ids: torch.Tensor, schedule_id: int) -> torch.Tensor:
    """Return active-item mask for one fused primitive schedule."""
    state_kernel_group = SCHEDULE_STATE_KERNELS[schedule_id]
    mask = state_kernel_ids == state_kernel_group[0]
    for state_kernel_id in state_kernel_group[1:]:
        mask |= state_kernel_ids == state_kernel_id
    return mask


@dataclass
class ProducerNodeTable:
    """Producer-node state for one target-independent producer kind.

    Each *node* is a unique ``(env, signature)`` pair; consumer items reach
    their producing node via ``consumer_node_ids[item]`` and producers iterate
    their consumers via the CSR pair ``(consumer_offsets, consumer_counts)``.

    NOTE: ``nodes_view`` is a :class:`PrimitiveProducerQueue` — the kernel-side
    Warp struct is still named "Queue" in :mod:`kernels_wp` because it's
    shared kernel-API glue. The role here is a producer-node table; we keep
    the Python-side naming honest.
    """

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
    nodes_view: PrimitiveProducerQueue


def _make_producer_node_table(
    command: MultiTaskCommandWarp,
    max_consumers: int,
    schedule_ids: tuple[int, ...],
) -> ProducerNodeTable:
    """Allocate one kind's producer-node table + its subtask-signature lookup."""
    subtask_signature_i32, signature_subtask_i32 = _build_signature_tables(command, schedule_ids)
    env_ids_i32 = torch.empty(max_consumers, device=command.device, dtype=torch.int32)
    subtask_ids_i32 = torch.empty(max_consumers, device=command.device, dtype=torch.int32)
    consumer_node_ids_i32 = torch.full((max_consumers,), -1, device=command.device, dtype=torch.int32)
    consumer_indices_i32 = torch.empty(max_consumers, device=command.device, dtype=torch.int32)
    consumer_offsets_i32 = torch.zeros(max_consumers, device=command.device, dtype=torch.int32)
    consumer_counts_i32 = torch.zeros(max_consumers, device=command.device, dtype=torch.int32)
    count_i32 = torch.zeros(1, device=command.device, dtype=torch.int32)

    nodes_view = PrimitiveProducerQueue()
    nodes_view.subtask_signature = wp.from_torch(subtask_signature_i32)
    nodes_view.signature_subtask = wp.from_torch(signature_subtask_i32)
    nodes_view.env_ids = wp.from_torch(env_ids_i32)
    nodes_view.subtask_ids = wp.from_torch(subtask_ids_i32)
    nodes_view.consumer_node_ids = wp.from_torch(consumer_node_ids_i32)
    nodes_view.consumer_indices = wp.from_torch(consumer_indices_i32)
    nodes_view.consumer_offsets = wp.from_torch(consumer_offsets_i32)
    nodes_view.consumer_counts = wp.from_torch(consumer_counts_i32)
    nodes_view.count = wp.from_torch(count_i32)

    return ProducerNodeTable(
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
        nodes_view=nodes_view,
    )


def _reset_producer_node_table(plan: ProducerNodeTable) -> None:
    """Clear dynamic producer state before rebuilding the graph plan."""
    plan.consumer_node_ids_i32.fill_(-1)
    plan.consumer_offsets_i32.zero_()
    plan.consumer_counts_i32.zero_()
    plan.count_i32.zero_()
    plan.count = 0


def _refresh_producer_nodes(
    plan: ProducerNodeTable,
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
    plan: ProducerNodeTable,
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
    consumer_env_ids_i32: torch.Tensor
    consumer_slot_ids_i32: torch.Tensor
    consumer_subtask_ids_i32: torch.Tensor
    consumer_target_offsets_i32: torch.Tensor
    schedule_offsets_i32: torch.Tensor
    schedule_counts_i32: torch.Tensor
    consumer_count_i32: torch.Tensor
    vec3_nodes: ProducerNodeTable
    scalar_nodes: ProducerNodeTable
    quat_nodes: ProducerNodeTable
    scalar_sum_nodes: ProducerNodeTable
    contact_nodes: ProducerNodeTable
    direct_vec3: torch.Tensor
    direct_scalar: torch.Tensor
    direct_quat: torch.Tensor
    scalar_sum: torch.Tensor
    contact_mask: torch.Tensor
    max_consumers: int
    total_consumers: int
    vec3_signature_count: int
    scalar_signature_count: int
    quat_signature_count: int
    scalar_sum_signature_count: int
    contact_signature_count: int
    vec3_node_count: int
    scalar_node_count: int
    quat_node_count: int
    scalar_sum_node_count: int
    contact_node_count: int
    schedule_counts_py: list[int]
    env_slots: EnvSlots
    # NOTE: ``consumer_view`` is a :class:`PrimitiveLocalQueue` — the kernel-side
    # Warp struct is shared with ``primitive_queue_local`` (kernel-API glue).
    # The role here is the *consumer table*: one entry per active subtask
    # (env, slot, subtask, target_offset), partitioned by kernel via
    # ``schedule_offsets`` / ``schedule_counts``.
    consumer_view: PrimitiveLocalQueue
    spec: SubtaskSpec
    state: StateAccess
    composer_state: ComposerState
    outputs: Outputs
    subtask_schedule_ids_wp: wp.array(dtype=int)
    direct_vec3_wp: wp.array2d(dtype=float)
    direct_scalar_wp: wp.array(dtype=float)
    direct_quat_wp: wp.array2d(dtype=float)
    # Backend-owned local-output buffers (was: command._outputs.local_*).
    # Torch tensors kept on the plan so the wp.from_torch views stay alive.
    local_delta_torch: torch.Tensor
    local_error_torch: torch.Tensor
    local_activation_torch: torch.Tensor
    slot_local_index_torch: torch.Tensor
    local_delta_wp: wp.array2d(dtype=float)
    local_error_wp: wp.array(dtype=float)
    local_activation_wp: wp.array(dtype=float)
    scalar_sum_wp: wp.array(dtype=float)
    contact_mask_wp: wp.array2d(dtype=float)
    rotations: tuple[RotationBinding, ...]
    episode_length_buf_wp: wp.array
    effective_max_episode_length_wp: wp.array
    float_slabs: tuple[FloatSlabBinding, ...] = ()
    vec3_slabs: tuple[Vec3SlabBinding, ...] = ()
    vec3_env_local_slabs: tuple[Vec3EnvLocalSlabBinding, ...] = ()
    quat_slabs: tuple[QuatSlabBinding, ...] = ()
    joint_mech_power_slabs: tuple[JointMechPowerSlabBinding, ...] = ()


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

    max_consumers = command.num_envs * command.k_max
    vec3_nodes = _make_producer_node_table(command, max_consumers, (SCHEDULE_DIRECT_VEC3_DELTA,))
    scalar_nodes = _make_producer_node_table(command, max_consumers, (SCHEDULE_DIRECT_SCALAR_DELTA,))
    quat_nodes = _make_producer_node_table(command, max_consumers, (SCHEDULE_DIRECT_QUAT_DELTA,))
    scalar_sum_nodes = _make_producer_node_table(command, max_consumers, (SCHEDULE_SCALAR_SUM_DELTA,))
    contact_nodes = _make_producer_node_table(command, max_consumers, _CONTACT_SCHEDULES)

    consumer_env_ids_i32 = torch.empty(max_consumers, device=command.device, dtype=torch.int32)
    consumer_slot_ids_i32 = torch.empty(max_consumers, device=command.device, dtype=torch.int32)
    consumer_subtask_ids_i32 = torch.empty(max_consumers, device=command.device, dtype=torch.int32)
    consumer_target_offsets_i32 = torch.empty(max_consumers, device=command.device, dtype=torch.int32)
    schedule_offsets_i32 = torch.zeros(NUM_SCHEDULES, device=command.device, dtype=torch.int32)
    schedule_counts_i32 = torch.zeros(NUM_SCHEDULES, device=command.device, dtype=torch.int32)
    consumer_count_i32 = torch.zeros(1, device=command.device, dtype=torch.int32)

    direct_vec3 = torch.empty((max_consumers, 3), device=command.device, dtype=command._unified_buffer.dtype)
    direct_scalar = torch.empty(max_consumers, device=command.device, dtype=command._unified_buffer.dtype)
    direct_quat = torch.empty((max_consumers, 4), device=command.device, dtype=command._unified_buffer.dtype)
    scalar_sum = torch.empty(max_consumers, device=command.device, dtype=command._unified_buffer.dtype)
    contact_mask = torch.zeros((max_consumers, 4), device=command.device, dtype=command._unified_buffer.dtype)

    # Backend-owned local outputs.
    local_delta_torch = torch.zeros((max_consumers, 4), device=command.device)
    local_error_torch = torch.zeros(max_consumers, device=command.device)
    local_activation_torch = torch.zeros(max_consumers, device=command.device)
    slot_local_index_torch = torch.zeros((command.num_envs, command.k_max), device=command.device, dtype=torch.int32)

    consumer_view = PrimitiveLocalQueue()
    consumer_view.env_ids = wp.from_torch(consumer_env_ids_i32)
    consumer_view.slot_ids = wp.from_torch(consumer_slot_ids_i32)
    consumer_view.subtask_ids = wp.from_torch(consumer_subtask_ids_i32)
    consumer_view.target_offsets = wp.from_torch(consumer_target_offsets_i32)
    consumer_view.slot_local_index = wp.from_torch(slot_local_index_torch)
    consumer_view.schedule_offsets = wp.from_torch(schedule_offsets_i32)
    consumer_view.schedule_counts = wp.from_torch(schedule_counts_i32)
    consumer_view.count = wp.from_torch(consumer_count_i32)

    (
        float_slabs,
        vec3_slabs,
        vec3_env_local_slabs,
        quat_slabs,
        joint_mech_power_slabs,
    ) = _resolve_slabs(command)

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
        consumer_env_ids_i32=consumer_env_ids_i32,
        consumer_slot_ids_i32=consumer_slot_ids_i32,
        consumer_subtask_ids_i32=consumer_subtask_ids_i32,
        consumer_target_offsets_i32=consumer_target_offsets_i32,
        schedule_offsets_i32=schedule_offsets_i32,
        schedule_counts_i32=schedule_counts_i32,
        consumer_count_i32=consumer_count_i32,
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
        max_consumers=max_consumers,
        total_consumers=0,
        vec3_signature_count=max(0, int(vec3_nodes.signature_subtask_i32.numel())),
        scalar_signature_count=max(0, int(scalar_nodes.signature_subtask_i32.numel())),
        quat_signature_count=max(0, int(quat_nodes.signature_subtask_i32.numel())),
        scalar_sum_signature_count=max(0, int(scalar_sum_nodes.signature_subtask_i32.numel())),
        contact_signature_count=max(0, int(contact_nodes.signature_subtask_i32.numel())),
        vec3_node_count=0,
        scalar_node_count=0,
        quat_node_count=0,
        scalar_sum_node_count=0,
        contact_node_count=0,
        schedule_counts_py=[0] * NUM_SCHEDULES,
        env_slots=env_slots,
        consumer_view=consumer_view,
        spec=spec_struct,
        state=state,
        composer_state=composer_state,
        outputs=outputs,
        subtask_schedule_ids_wp=wp.from_torch(subtask_schedule_ids_i32),
        direct_vec3_wp=wp.from_torch(direct_vec3),
        direct_scalar_wp=wp.from_torch(direct_scalar),
        direct_quat_wp=wp.from_torch(direct_quat),
        local_delta_torch=local_delta_torch,
        local_error_torch=local_error_torch,
        local_activation_torch=local_activation_torch,
        slot_local_index_torch=slot_local_index_torch,
        local_delta_wp=wp.from_torch(local_delta_torch),
        local_error_wp=wp.from_torch(local_error_torch),
        local_activation_wp=wp.from_torch(local_activation_torch),
        scalar_sum_wp=wp.from_torch(scalar_sum),
        contact_mask_wp=wp.from_torch(contact_mask),
        rotations=_build_rotation_bindings(command),
        episode_length_buf_wp=wp.from_torch(command._env.episode_length_buf),
        effective_max_episode_length_wp=wp.from_torch(command._effective_max_episode_length),
        float_slabs=float_slabs,
        vec3_slabs=vec3_slabs,
        vec3_env_local_slabs=vec3_env_local_slabs,
        quat_slabs=quat_slabs,
        joint_mech_power_slabs=joint_mech_power_slabs,
    )
    refresh_primitive_graph_local_plan(command, plan)
    return plan


def refresh_primitive_graph_local_plan(command: MultiTaskCommandWarp, plan: PrimitiveGraphLocalPlan) -> None:
    """Refresh primitive graph queues from the command's current assignment."""
    _sort_command_slots_by_schedule(command, plan.subtask_schedule_ids_i32)
    plan.schedule_offsets_i32.zero_()
    plan.schedule_counts_i32.zero_()
    plan.consumer_count_i32.zero_()
    _reset_producer_node_table(plan.vec3_nodes)
    _reset_producer_node_table(plan.scalar_nodes)
    _reset_producer_node_table(plan.quat_nodes)
    _reset_producer_node_table(plan.scalar_sum_nodes)
    _reset_producer_node_table(plan.contact_nodes)
    plan.slot_local_index_torch.zero_()

    cursor = 0
    slot_idx = torch.arange(command.k_max, device=command.device, dtype=torch.int32).unsqueeze(0)
    valid = slot_idx < command._env_slot_count.unsqueeze(1)
    if not bool(valid.any()):
        plan.total_consumers = 0
        plan.vec3_node_count = 0
        plan.scalar_node_count = 0
        plan.quat_node_count = 0
        plan.scalar_sum_node_count = 0
        plan.contact_node_count = 0
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
        plan.consumer_env_ids_i32[cursor:stop] = env_ids_i32[group_mask]
        plan.consumer_slot_ids_i32[cursor:stop] = slot_ids_i32[group_mask]
        plan.consumer_subtask_ids_i32[cursor:stop] = subtask_ids_i32[group_mask]
        plan.consumer_target_offsets_i32[cursor:stop] = target_offsets[group_mask]
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
        plan.slot_local_index_torch[env_ids[group_mask], slot_ids[group_mask]] = local_indices[cursor:stop]
        cursor = stop

    if cursor != int(env_ids.numel()):
        raise ValueError("primitive_graph_local failed to lower every active subtask into a primitive schedule.")
    plan.total_consumers = cursor
    plan.vec3_node_count = plan.vec3_nodes.count
    plan.scalar_node_count = plan.scalar_nodes.count
    plan.quat_node_count = plan.quat_nodes.count
    plan.scalar_sum_node_count = plan.scalar_sum_nodes.count
    plan.contact_node_count = plan.contact_nodes.count
    plan.schedule_counts_py = schedule_counts_py
    plan.consumer_count_i32[0] = cursor
