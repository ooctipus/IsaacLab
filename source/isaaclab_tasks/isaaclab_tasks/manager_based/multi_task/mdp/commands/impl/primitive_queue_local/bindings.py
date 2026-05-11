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

from ..kernel_ids import BUFFER_KIND
from ..kernels_wp import (
    ComposerState,
    EnvSlots,
    Outputs,
    PrimitiveLocalQueue,
    StateAccess,
    SubtaskSpec,
)
from ..schedules import NUM_SCHEDULES, SCHEDULE_STATE_KERNELS, validate_schedule_support

if TYPE_CHECKING:
    from ..multi_task_command_warp import MultiTaskCommandWarp


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
    # Backend-owned local-output buffers. Kept as torch tensors on the plan
    # so the ``wp.from_torch`` views below stay alive for the captured graph.
    local_delta_torch: torch.Tensor
    local_error_torch: torch.Tensor
    local_activation_torch: torch.Tensor
    slot_local_index_torch: torch.Tensor
    local_delta_wp: wp.array2d(dtype=float)
    local_error_wp: wp.array(dtype=float)
    local_activation_wp: wp.array(dtype=float)
    rotations: tuple[RotationBinding, ...]
    episode_length_buf_wp: wp.array
    effective_max_episode_length_wp: wp.array
    float_slabs: tuple[FloatSlabBinding, ...] = ()
    vec3_slabs: tuple[Vec3SlabBinding, ...] = ()
    vec3_env_local_slabs: tuple[Vec3EnvLocalSlabBinding, ...] = ()
    quat_slabs: tuple[QuatSlabBinding, ...] = ()
    joint_mech_power_slabs: tuple[JointMechPowerSlabBinding, ...] = ()


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

    max_work = command.num_envs * command.k_max
    flat_env_ids_i32 = torch.empty(max_work, device=command.device, dtype=torch.int32)
    flat_slot_ids_i32 = torch.empty(max_work, device=command.device, dtype=torch.int32)
    flat_subtask_ids_i32 = torch.empty(max_work, device=command.device, dtype=torch.int32)
    flat_target_offsets_i32 = torch.empty(max_work, device=command.device, dtype=torch.int32)
    schedule_offsets_i32 = torch.zeros(NUM_SCHEDULES, device=command.device, dtype=torch.int32)
    schedule_counts_i32 = torch.zeros(NUM_SCHEDULES, device=command.device, dtype=torch.int32)
    flat_count_i32 = torch.zeros(1, device=command.device, dtype=torch.int32)

    # Backend-owned local outputs (was: command._outputs.local_*). Kept on
    # the plan so the wp.from_torch views below outlive any captured graph.
    local_delta_torch = torch.zeros((max_work, 4), device=command.device)
    local_error_torch = torch.zeros(max_work, device=command.device)
    local_activation_torch = torch.zeros(max_work, device=command.device)
    slot_local_index_torch = torch.zeros((command.num_envs, command.k_max), device=command.device, dtype=torch.int32)

    queue = PrimitiveLocalQueue()
    queue.env_ids = wp.from_torch(flat_env_ids_i32)
    queue.slot_ids = wp.from_torch(flat_slot_ids_i32)
    queue.subtask_ids = wp.from_torch(flat_subtask_ids_i32)
    queue.target_offsets = wp.from_torch(flat_target_offsets_i32)
    queue.slot_local_index = wp.from_torch(slot_local_index_torch)
    queue.schedule_offsets = wp.from_torch(schedule_offsets_i32)
    queue.schedule_counts = wp.from_torch(schedule_counts_i32)
    queue.count = wp.from_torch(flat_count_i32)

    (
        float_slabs,
        vec3_slabs,
        vec3_env_local_slabs,
        quat_slabs,
        joint_mech_power_slabs,
    ) = _resolve_slabs(command)

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
        local_delta_torch=local_delta_torch,
        local_error_torch=local_error_torch,
        local_activation_torch=local_activation_torch,
        slot_local_index_torch=slot_local_index_torch,
        local_delta_wp=wp.from_torch(local_delta_torch),
        local_error_wp=wp.from_torch(local_error_torch),
        local_activation_wp=wp.from_torch(local_activation_torch),
        rotations=_build_rotation_bindings(command),
        episode_length_buf_wp=wp.from_torch(command._env.episode_length_buf),
        effective_max_episode_length_wp=wp.from_torch(command._effective_max_episode_length),
        float_slabs=float_slabs,
        vec3_slabs=vec3_slabs,
        vec3_env_local_slabs=vec3_env_local_slabs,
        quat_slabs=quat_slabs,
        joint_mech_power_slabs=joint_mech_power_slabs,
    )
    refresh_primitive_queue_local_plan(command, plan)
    return plan


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
                f"offset={offset}, size={size}). primitive_queue_local requires every reader to "
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


def refresh_primitive_queue_local_plan(command: MultiTaskCommandWarp, plan: PrimitiveQueueLocalPlan) -> None:
    """Refresh primitive queues from the command's current task assignment."""
    plan.schedule_offsets_i32.zero_()
    plan.schedule_counts_i32.zero_()
    plan.flat_count_i32.zero_()
    plan.slot_local_index_torch.zero_()

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
        plan.slot_local_index_torch[env_ids[group_mask], slot_ids[group_mask]] = local_indices[cursor:stop]
        cursor = stop

    if cursor != int(env_ids.numel()):
        raise ValueError("primitive_queue_local failed to lower every active subtask into a primitive schedule.")
    plan.total_work = cursor
    plan.schedule_counts_py = schedule_counts_py
    plan.flat_count_i32[0] = cursor
