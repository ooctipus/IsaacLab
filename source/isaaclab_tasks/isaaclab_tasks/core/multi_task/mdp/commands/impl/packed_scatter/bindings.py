# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-owned execution plan for fused-pipeline packed-scatter dispatch.

Pure Warp — no ``import torch``. Slab resolution is local to this backend.
Each ``BUFFER_KIND`` resolves to a stable ``wp.array`` exposed via
``ProxyArray.warp`` on the scene assets. Per-resample queue mutations
operate on ``wp.to_torch`` views of the Warp-owned plan storage so the
indexing logic stays vectorised; the per-step kernels read the same
storage through the ``PackedScatterQueue`` wp-struct accessors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import warp as wp

from ..kernel_ids import BUFFER_KIND
from ..kernels_wp import (
    ComposerState,
    EnvSlots,
    Outputs,
    PackedScatterQueue,
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
    validate_schedule_support,
)

if TYPE_CHECKING:
    from ..multi_task_command_warp import MultiTaskCommandWarp

PIPELINE_DIRECT_VEC3_DELTA = SCHEDULE_DIRECT_VEC3_DELTA
PIPELINE_DIRECT_SCALAR_DELTA = SCHEDULE_DIRECT_SCALAR_DELTA
PIPELINE_DIRECT_QUAT_DELTA = SCHEDULE_DIRECT_QUAT_DELTA
PIPELINE_VEC3_THRESHOLD_VECTOR_DELTA = SCHEDULE_VEC3_THRESHOLD_VECTOR_DELTA
PIPELINE_VEC3_THRESHOLD_SUM_DELTA = SCHEDULE_VEC3_THRESHOLD_SUM_DELTA
PIPELINE_VEC3_THRESHOLD_PAIR_DIFF_DELTA = SCHEDULE_VEC3_THRESHOLD_PAIR_DIFF_DELTA
PIPELINE_SCALAR_SUM_DELTA = SCHEDULE_SCALAR_SUM_DELTA

_PIPELINE_STATE_KERNELS = SCHEDULE_STATE_KERNELS
NUM_PACKED_PIPELINES = NUM_SCHEDULES


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
    """Float-typed scene slab — used by JOINT_POS, JOINT_VEL."""

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
class PackedScatterPlan:
    """Long-lived Warp plan for fused-pipeline execution with legacy scatter."""

    flat_env_ids_wp: wp.array
    flat_slot_ids_wp: wp.array
    flat_subtask_ids_wp: wp.array
    flat_target_offsets_wp: wp.array
    flat_pipeline_ids_wp: wp.array
    flat_count_wp: wp.array
    max_work: int
    total_work: int
    env_slots: EnvSlots
    flat_queue: PackedScatterQueue
    spec: SubtaskSpec
    state: StateAccess
    composer_state: ComposerState
    outputs: Outputs
    rotations: tuple[RotationBinding, ...]
    episode_length_buf_wp: wp.array
    effective_max_episode_length_wp: wp.array
    float_slabs: tuple[FloatSlabBinding, ...] = field(default_factory=tuple)
    vec3_slabs: tuple[Vec3SlabBinding, ...] = field(default_factory=tuple)
    vec3_env_local_slabs: tuple[Vec3EnvLocalSlabBinding, ...] = field(default_factory=tuple)
    quat_slabs: tuple[QuatSlabBinding, ...] = field(default_factory=tuple)
    joint_mech_power_slabs: tuple[JointMechPowerSlabBinding, ...] = field(default_factory=tuple)


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
                JointMechPowerSlabBinding(art.data.applied_torque.warp, art.data.joint_vel.warp, offset, size)
            )
        else:
            raise ValueError(
                f"Unsupported BUFFER_KIND {kind!r} for slab (asset={asset_name!r}, "
                f"offset={offset}, size={size}). packed_scatter requires every reader to "
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


def build_packed_scatter_plan(command: MultiTaskCommandWarp) -> PackedScatterPlan:
    """Construct the backend-owned packed-scatter execution plan."""
    wp.init()
    s = command.spec
    validate_schedule_support(s.state_kernel_id, backend_name="packed_scatter")
    device_str = str(command.device)

    max_work = command.num_envs * command.k_max
    flat_env_ids_wp = wp.zeros(shape=max_work, dtype=wp.int32, device=device_str)
    flat_slot_ids_wp = wp.zeros(shape=max_work, dtype=wp.int32, device=device_str)
    flat_subtask_ids_wp = wp.zeros(shape=max_work, dtype=wp.int32, device=device_str)
    flat_target_offsets_wp = wp.zeros(shape=max_work, dtype=wp.int32, device=device_str)
    flat_pipeline_ids_wp = wp.zeros(shape=max_work, dtype=wp.int32, device=device_str)
    flat_count_wp = wp.zeros(shape=1, dtype=wp.int32, device=device_str)

    flat_queue = PackedScatterQueue()
    flat_queue.env_ids = flat_env_ids_wp
    flat_queue.slot_ids = flat_slot_ids_wp
    flat_queue.subtask_ids = flat_subtask_ids_wp
    flat_queue.target_offsets = flat_target_offsets_wp
    flat_queue.pipeline_ids = flat_pipeline_ids_wp
    flat_queue.count = flat_count_wp

    (
        float_slabs,
        vec3_slabs,
        vec3_env_local_slabs,
        quat_slabs,
        joint_mech_power_slabs,
    ) = _resolve_slabs(command)

    plan = PackedScatterPlan(
        flat_env_ids_wp=flat_env_ids_wp,
        flat_slot_ids_wp=flat_slot_ids_wp,
        flat_subtask_ids_wp=flat_subtask_ids_wp,
        flat_target_offsets_wp=flat_target_offsets_wp,
        flat_pipeline_ids_wp=flat_pipeline_ids_wp,
        flat_count_wp=flat_count_wp,
        max_work=max_work,
        total_work=0,
        env_slots=command.env_slots_wp,
        flat_queue=flat_queue,
        spec=command.spec_wp,
        state=command.state_wp,
        composer_state=command.composer_state_wp,
        outputs=command.outputs_wp,
        rotations=_build_rotation_bindings(command),
        episode_length_buf_wp=command.episode_length_buf_wp,
        effective_max_episode_length_wp=command.effective_max_episode_length_wp,
        float_slabs=float_slabs,
        vec3_slabs=vec3_slabs,
        vec3_env_local_slabs=vec3_env_local_slabs,
        quat_slabs=quat_slabs,
        joint_mech_power_slabs=joint_mech_power_slabs,
    )
    refresh_packed_scatter_plan(command, plan)
    return plan


def refresh_packed_scatter_plan(command: MultiTaskCommandWarp, plan: PackedScatterPlan) -> None:
    """Refresh packed queues from the command's current per-env task assignment.

    The flat queue arrays are Warp-owned; we mutate the storage through
    ``wp.to_torch`` views so the per-pipeline indexing stays vectorised.
    Same storage is read by ``dispatch_packed_scatter_flat`` at step time.
    """
    flat_count_torch = wp.to_torch(plan.flat_count_wp)
    flat_env_ids_torch = wp.to_torch(plan.flat_env_ids_wp)
    flat_slot_ids_torch = wp.to_torch(plan.flat_slot_ids_wp)
    flat_subtask_ids_torch = wp.to_torch(plan.flat_subtask_ids_wp)
    flat_target_offsets_torch = wp.to_torch(plan.flat_target_offsets_wp)
    flat_pipeline_ids_torch = wp.to_torch(plan.flat_pipeline_ids_wp)

    flat_count_torch.zero_()
    cursor = 0
    slot_idx = command._slot_arange
    valid = slot_idx < command._env_slot_count.unsqueeze(1)
    if not bool(valid.any()):
        plan.total_work = 0
        return

    env_ids, slot_ids = valid.nonzero(as_tuple=True)
    subtask_ids = command._env_subtask_ids[env_ids, slot_ids].long()
    state_kernel_ids = command.spec.state_kernel_id[subtask_ids].long()
    target_offsets = command._env_slot_offsets[env_ids, slot_ids]
    env_ids_i32 = env_ids.int()
    slot_ids_i32 = slot_ids.int()
    subtask_ids_i32 = subtask_ids.int()

    for pipeline_id, state_kernel_group in enumerate(_PIPELINE_STATE_KERNELS):
        group_mask = state_kernel_ids == state_kernel_group[0]
        for state_kernel_id in state_kernel_group[1:]:
            group_mask |= state_kernel_ids == state_kernel_id
        count = int(group_mask.sum().item())
        if count == 0:
            continue
        stop = cursor + count
        flat_env_ids_torch[cursor:stop] = env_ids_i32[group_mask]
        flat_slot_ids_torch[cursor:stop] = slot_ids_i32[group_mask]
        flat_subtask_ids_torch[cursor:stop] = subtask_ids_i32[group_mask]
        flat_target_offsets_torch[cursor:stop] = target_offsets[group_mask]
        flat_pipeline_ids_torch[cursor:stop] = pipeline_id
        cursor = stop
    if cursor != int(env_ids.numel()):
        raise ValueError("packed_scatter failed to lower every active subtask into a fused pipeline.")
    plan.total_work = cursor
    flat_count_torch[0] = cursor
