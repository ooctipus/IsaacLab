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

Spec/env-slot/output Warp views come from :class:`MultiTaskCommandWarp`
(``command.spec_wp`` and friends); the cold refresh path operates on
``wp.to_torch`` views of the plan's Warp-owned arrays via tensor methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import warp as wp

from ...kernel_ids import BUFFER_KIND
from ..kernels_wp import (
    ComposerState,
    EnvSlots,
    Outputs,
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
)
from .csr_graph import CSRGraph

if TYPE_CHECKING:
    import torch

    from ...multi_task_command_warp import MultiTaskCommandWarp

_CONTACT_SCHEDULES = (
    SCHEDULE_VEC3_THRESHOLD_VECTOR_DELTA,
    SCHEDULE_VEC3_THRESHOLD_SUM_DELTA,
    SCHEDULE_VEC3_THRESHOLD_PAIR_DIFF_DELTA,
)


@dataclass
class RotationBinding:
    """Warp views for rotating one asset's canonical vec3 command slots.

    All arrays are Warp-owned (``wp.array``) or asset-anchored ProxyArray
    accessors; no torch tensors on the binding.
    """

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

    applied_effort_wp: wp.array
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
    device_str = str(command.device)

    env_origins_torch = command._env.scene.env_origins
    env_origins_vec3_wp = wp.array(
        ptr=env_origins_torch.data_ptr(),
        dtype=wp.vec3,
        shape=(command.num_envs,),
        device=device_str,
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
                JointMechPowerSlabBinding(art.actuators.applied_effort.warp, art.data.joint_vel.warp, offset, size)
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


def _build_rotation_bindings(command: MultiTaskCommandWarp) -> tuple[RotationBinding, ...]:
    s = command.spec
    device_str = str(command.device)
    bindings: list[RotationBinding] = []
    asset_names = sorted(set(s.reach_rotatable_vec3_by_asset.keys()) | set(s.track_rotatable_vec3_by_asset.keys()))
    for asset_name in asset_names:
        reach_offsets = s.reach_rotatable_vec3_by_asset.get(asset_name, ())
        track_offsets = s.track_rotatable_vec3_by_asset.get(asset_name, ())
        num_offsets = len(reach_offsets) + len(track_offsets)
        if num_offsets == 0:
            continue
        root_quat_w_wp = command._env.scene[asset_name].data.root_quat_w.warp
        reach_offsets_wp = wp.array(list(reach_offsets), dtype=wp.int32, device=device_str)
        track_offsets_wp = wp.array(list(track_offsets), dtype=wp.int32, device=device_str)
        bindings.append(
            RotationBinding(
                root_quat_w_wp=root_quat_w_wp,
                reach_offsets_wp=reach_offsets_wp,
                track_offsets_wp=track_offsets_wp,
                num_reach_offsets=len(reach_offsets),
                num_offsets=num_offsets,
            )
        )
    return tuple(bindings)


def _build_consumer_csr_graph(
    command: MultiTaskCommandWarp,
    schedule_ids: tuple[int, ...],
) -> CSRGraph:
    """Build the producer-consumer fan-out graph for one fused schedule kind.

    Consumers are subtask slots; producers are unique gather signatures
    (target-independent unified-buffer gather blocks). Subtasks whose state
    kernel is not in this schedule kind get a ``None`` key and contribute no
    edge — they appear as ``-1`` in the resulting
    :attr:`CSRGraph.consumer_to_producer`.
    """
    scheduled_state_ids = {
        state_kernel_id for schedule_id in schedule_ids for state_kernel_id in SCHEDULE_STATE_KERNELS[schedule_id]
    }
    s = command.spec
    state_kernel_id_cpu = s.state_kernel_id.detach().cpu().tolist()
    gather_indices_cpu = s.gather_indices_flat.detach().cpu().tolist()
    gather_offset_cpu = s.subtask_gather_offset.detach().cpu().tolist()
    gather_count_cpu = s.subtask_gather_count.detach().cpu().tolist()

    consumer_keys: list[tuple[int, ...] | None] = []
    for sid in range(len(state_kernel_id_cpu)):
        if int(state_kernel_id_cpu[sid]) not in scheduled_state_ids:
            consumer_keys.append(None)
            continue
        start = int(gather_offset_cpu[sid])
        count = int(gather_count_cpu[sid])
        consumer_keys.append(tuple(int(v) for v in gather_indices_cpu[start : start + count]))

    return CSRGraph.build_from_consumer_keys(consumer_keys, device=command.device)


def _build_schedule_mask(state_kernel_ids, schedule_id: int):
    """Return active-item mask for one fused primitive schedule.

    ``state_kernel_ids`` is a torch tensor passed in by the caller; we only
    use tensor comparison / OR methods here, no ``torch.X`` symbols.
    """
    state_kernel_group = SCHEDULE_STATE_KERNELS[schedule_id]
    mask = state_kernel_ids == state_kernel_group[0]
    for state_kernel_id in state_kernel_group[1:]:
        mask |= state_kernel_ids == state_kernel_id
    return mask


@dataclass
class ProducerNodeTable:
    """Static signature-lookup tables for one producer kind.

    Both tables are spec-derived and constant across resamples; the dense
    kernels index them by ``signature`` (producer side) and by ``sid``
    (consumer side) without any per-resample bookkeeping.

    NOTE: ``nodes_view`` is a :class:`PrimitiveProducerQueue` — the kernel-side
    Warp struct is still named "Queue" because it's shared kernel-API glue.
    The role here is a producer-node table; we keep the Python-side naming
    honest.

    :attr:`csr_graph` holds the full CSR view of the same producer-consumer
    fan-out — kept reachable so later phases can introspect it (fanout,
    locality, etc.) without rebuilding.
    """

    subtask_signature_wp: wp.array
    signature_subtask_wp: wp.array
    nodes_view: PrimitiveProducerQueue
    csr_graph: CSRGraph


def _make_producer_node_table(
    command: MultiTaskCommandWarp,
    schedule_ids: tuple[int, ...],
) -> ProducerNodeTable:
    """Allocate one kind's static subtask-signature lookup tables."""
    g = _build_consumer_csr_graph(command, schedule_ids)
    nodes_view = PrimitiveProducerQueue()
    nodes_view.subtask_signature = g.consumer_to_producer
    nodes_view.signature_subtask = g.producer_to_representative_consumer
    return ProducerNodeTable(
        subtask_signature_wp=g.consumer_to_producer,
        signature_subtask_wp=g.producer_to_representative_consumer,
        nodes_view=nodes_view,
        csr_graph=g,
    )


def _sort_command_slots_by_schedule_and_producer(
    command: MultiTaskCommandWarp,
    schedule_ids_torch,
    producer_within_kind_torch,
    producer_id_stride: int,
) -> None:
    """Sort each env's active slots by ``(schedule_id, producer_within_kind)``.

    Outer key (``schedule_id``) groups slots by primitive kind — gives the
    producer kernel warp-coherent state-kernel regions. Inner key
    (``producer_within_kind``) groups slots that read from the same
    producer-buf entry adjacently, so a warp of consecutive consumer
    threads sees a broadcast load instead of a scatter.

    The two keys are packed into a single integer via the precomputed
    ``producer_id_stride`` so we can do one stable :func:`argsort` rather
    than chaining two passes. Inactive slots (``slot >= slot_count``) get
    a sentinel key ``NUM_SCHEDULES * stride`` that sorts strictly after
    every valid composite.

    ``schedule_ids_torch`` and ``producer_within_kind_torch`` are torch
    views (the latter is stored on the plan; the former is a per-refresh
    :func:`wp.to_torch` view). We use tensor methods only — no ``torch.X``
    symbols.
    """
    slot_ids = command._slot_arange.expand(command.num_envs, -1)
    active = slot_ids < command._env_slot_count.unsqueeze(1)
    subtask_ids = command._env_subtask_ids.long().clamp_min(0)
    slot_schedule_ids = schedule_ids_torch[subtask_ids]
    # Inactive subtasks contribute -1 in producer_within_kind; clamp to 0 so the
    # composite arithmetic is well-defined even for slots we're about to mask.
    slot_producer = producer_within_kind_torch[subtask_ids].clamp_min(0)
    composite = slot_schedule_ids * producer_id_stride + slot_producer
    fallback = composite.new_full(composite.shape, NUM_SCHEDULES * producer_id_stride)
    composite = composite.where(active, fallback)
    slot_order = composite.argsort(dim=1, stable=True)
    command._env_subtask_ids[:] = command._env_subtask_ids.gather(1, slot_order)
    command._env_slot_offsets[:] = command._env_slot_offsets.gather(1, slot_order)
    command._env_slot_strides[:] = command._env_slot_strides.gather(1, slot_order)


@dataclass
class PrimitiveGraphLocalPlan:
    """Long-lived Warp plan for primitive graph execution.

    The dense graph kernels iterate over ``(env, slot)`` via ``env_slots`` and
    branch on ``subtask_schedule_ids`` — no consumer-table partitioning is
    needed at dispatch time. Per-resample state is just the sorted slot order
    (mutated in place on ``command``) plus diagnostic schedule counts.
    """

    vec3_nodes: ProducerNodeTable
    scalar_nodes: ProducerNodeTable
    quat_nodes: ProducerNodeTable
    scalar_sum_nodes: ProducerNodeTable
    contact_nodes: ProducerNodeTable
    total_consumers: int
    vec3_signature_count: int
    scalar_signature_count: int
    quat_signature_count: int
    scalar_sum_signature_count: int
    contact_signature_count: int
    schedule_counts_py: list[int]
    env_slots: EnvSlots
    spec: SubtaskSpec
    state: StateAccess
    composer_state: ComposerState
    outputs: Outputs
    subtask_schedule_ids_wp: wp.array
    # Per-subtask producer id WITHIN its own kind (vec3/scalar/quat/scalar_sum/contact).
    # Used as the secondary sort key in :func:`_sort_command_slots_by_schedule_and_producer`
    # so adjacent slots within a kind read the same producer-buf entry — turning
    # warp-scatter producer-buf loads into broadcast loads. Held as a torch
    # tensor because the sort path is torch-tensor-method based; built once
    # at plan time by ORing the 5 per-kind CSRGraph ``consumer_to_producer``
    # arrays (each subtask is active in exactly one kind).
    producer_within_kind_torch: torch.Tensor
    # Stride used when packing ``(schedule_id, producer_within_kind)`` into a single
    # int sort key: ``key = schedule_id * producer_id_stride + producer_within_kind``.
    # Set to ``max_num_producers_across_kinds + 1`` at plan build.
    producer_id_stride: int
    direct_vec3_wp: wp.array
    direct_scalar_wp: wp.array
    direct_quat_wp: wp.array
    scalar_sum_wp: wp.array
    contact_mask_wp: wp.array
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
    device_str = str(command.device)

    subtask_schedule_ids_wp = build_subtask_schedule_ids(
        s.state_kernel_id,
        backend_name="primitive_graph_local",
    )

    vec3_nodes = _make_producer_node_table(command, (SCHEDULE_DIRECT_VEC3_DELTA,))
    scalar_nodes = _make_producer_node_table(command, (SCHEDULE_DIRECT_SCALAR_DELTA,))
    quat_nodes = _make_producer_node_table(command, (SCHEDULE_DIRECT_QUAT_DELTA,))
    scalar_sum_nodes = _make_producer_node_table(command, (SCHEDULE_SCALAR_SUM_DELTA,))
    contact_nodes = _make_producer_node_table(command, _CONTACT_SCHEDULES)
    vec3_signature_count = int(vec3_nodes.signature_subtask_wp.shape[0])
    scalar_signature_count = int(scalar_nodes.signature_subtask_wp.shape[0])
    quat_signature_count = int(quat_nodes.signature_subtask_wp.shape[0])
    scalar_sum_signature_count = int(scalar_sum_nodes.signature_subtask_wp.shape[0])
    contact_signature_count = int(contact_nodes.signature_subtask_wp.shape[0])

    # Combine the 5 per-kind ``consumer_to_producer`` arrays into one per-subtask
    # producer-within-kind tensor. Each subtask is active in exactly one kind, so
    # element-wise max over (-1 for not-this-kind, ≥ 0 for this-kind) recovers the
    # subtask's local producer id. The accompanying stride lets the refresh sort
    # pack (schedule_id, producer_within_kind) into a single argsort key.
    kind_tables = (vec3_nodes, scalar_nodes, quat_nodes, scalar_sum_nodes, contact_nodes)
    producer_within_kind_torch = wp.to_torch(kind_tables[0].csr_graph.consumer_to_producer).clone()
    for nodes in kind_tables[1:]:
        producer_within_kind_torch = producer_within_kind_torch.maximum(
            wp.to_torch(nodes.csr_graph.consumer_to_producer)
        )
    producer_id_stride = max(nodes.csr_graph.num_producers for nodes in kind_tables) + 1

    # Producers materialize every unique signature for every environment. The
    # signature count can exceed ``k_max`` when many one-slot tasks reference
    # different state sources, so consumer capacity is not a valid bound.
    direct_vec3_wp = wp.zeros(shape=(command.num_envs * vec3_signature_count, 3), dtype=wp.float32, device=device_str)
    direct_scalar_wp = wp.zeros(shape=command.num_envs * scalar_signature_count, dtype=wp.float32, device=device_str)
    direct_quat_wp = wp.zeros(shape=(command.num_envs * quat_signature_count, 4), dtype=wp.float32, device=device_str)
    scalar_sum_wp = wp.zeros(shape=command.num_envs * scalar_sum_signature_count, dtype=wp.float32, device=device_str)
    contact_mask_wp = wp.zeros(
        shape=(command.num_envs * contact_signature_count, 4), dtype=wp.float32, device=device_str
    )

    (
        float_slabs,
        vec3_slabs,
        vec3_env_local_slabs,
        quat_slabs,
        joint_mech_power_slabs,
    ) = _resolve_slabs(command)

    plan = PrimitiveGraphLocalPlan(
        vec3_nodes=vec3_nodes,
        scalar_nodes=scalar_nodes,
        quat_nodes=quat_nodes,
        scalar_sum_nodes=scalar_sum_nodes,
        contact_nodes=contact_nodes,
        total_consumers=0,
        vec3_signature_count=vec3_signature_count,
        scalar_signature_count=scalar_signature_count,
        quat_signature_count=quat_signature_count,
        scalar_sum_signature_count=scalar_sum_signature_count,
        contact_signature_count=contact_signature_count,
        schedule_counts_py=[0] * NUM_SCHEDULES,
        env_slots=command.env_slots_wp,
        spec=command.spec_wp,
        state=command.state_wp,
        composer_state=command.composer_state_wp,
        outputs=command.outputs_wp,
        subtask_schedule_ids_wp=subtask_schedule_ids_wp,
        producer_within_kind_torch=producer_within_kind_torch,
        producer_id_stride=producer_id_stride,
        direct_vec3_wp=direct_vec3_wp,
        direct_scalar_wp=direct_scalar_wp,
        direct_quat_wp=direct_quat_wp,
        scalar_sum_wp=scalar_sum_wp,
        contact_mask_wp=contact_mask_wp,
        rotations=_build_rotation_bindings(command),
        episode_length_buf_wp=command.episode_length_buf_wp,
        effective_max_episode_length_wp=command.effective_max_episode_length_wp,
        float_slabs=float_slabs,
        vec3_slabs=vec3_slabs,
        vec3_env_local_slabs=vec3_env_local_slabs,
        quat_slabs=quat_slabs,
        joint_mech_power_slabs=joint_mech_power_slabs,
    )
    refresh_primitive_graph_local_plan(command, plan)
    return plan


def refresh_primitive_graph_local_plan(command: MultiTaskCommandWarp, plan: PrimitiveGraphLocalPlan) -> None:
    """Re-sort slots by schedule and refresh diagnostic counts.

    Producer state is fully static (signature lookup tables only) and the
    dense kernel iterates ``(env, slot)`` directly — so the only per-resample
    work here is the in-place slot sort (warp-coherent state-kernel regions)
    plus updating ``total_consumers`` and ``schedule_counts_py``.
    """
    schedule_ids_torch = wp.to_torch(plan.subtask_schedule_ids_wp)
    _sort_command_slots_by_schedule_and_producer(
        command,
        schedule_ids_torch,
        plan.producer_within_kind_torch,
        plan.producer_id_stride,
    )

    slot_idx = command._slot_arange
    valid = slot_idx < command._env_slot_count.unsqueeze(1)
    if not bool(valid.any()):
        plan.total_consumers = 0
        plan.schedule_counts_py = [0] * NUM_SCHEDULES
        return

    subtask_ids = command._env_subtask_ids[valid].long()
    state_kernel_ids = command.spec.state_kernel_id[subtask_ids].long()

    schedule_counts_py: list[int] = [
        int(_build_schedule_mask(state_kernel_ids, schedule_id).sum().item())
        for schedule_id in range(len(SCHEDULE_STATE_KERNELS))
    ]
    total = sum(schedule_counts_py)
    if total != int(subtask_ids.numel()):
        raise ValueError("primitive_graph_local failed to classify every active subtask into a primitive schedule.")
    plan.total_consumers = total
    plan.schedule_counts_py = schedule_counts_py
