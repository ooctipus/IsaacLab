# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-owned execution plan for the current mega-kernel layout.

Slab resolution is direct: each ``BUFFER_KIND`` maps to a stable ``wp.array``
exposed by the IsaacLab scene (via ``ProxyArray.warp``). No Torch laundering,
no reshape allocations, no stability fallback. Each slab knows the kernel
that fills it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import warp as wp

from ..kernel_ids import BUFFER_KIND
from ..kernels_wp import ComposerState, EnvSlots, Outputs, StateAccess, SubtaskSpec

if TYPE_CHECKING:
    from ..multi_task_command_warp import MultiTaskCommandWarp


@dataclass
class RotationBinding:
    """Warp views for rotating one asset's canonical vec3 command slots.

    ``root_quat_w_wp`` comes from the asset's :class:`ProxyArray` directly, so
    the underlying storage is anchored by the scene itself — no torch view
    needed on this binding. The ``*_offsets_i32`` torch tensors *are* kept
    because we construct them ourselves (via :func:`torch.tensor`) and the
    ``*_offsets_wp`` views need them alive.
    """

    reach_offsets_i32: torch.Tensor
    track_offsets_i32: torch.Tensor
    root_quat_w_wp: wp.array
    reach_offsets_wp: wp.array
    track_offsets_wp: wp.array
    num_reach_offsets: int
    num_offsets: int


# -- Per-kind slab bindings -------------------------------------------------
# Each slab dataclass holds only the fields its fill kernel signature needs.
# The kind is implied by which tuple the slab lives in on the plan.


@dataclass
class FloatSlabBinding:
    """Float-typed scene slab — used by JOINT_POS, JOINT_VEL."""

    source_wp: wp.array  # wp.array(dtype=float, shape=(num_envs, size))
    offset: int
    size: int


@dataclass
class Vec3SlabBinding:
    """vec3-typed scene slab with no frame transform — used by BODY_LIN_VEL_W,
    BODY_ANG_VEL_W, CONTACT_NET_FORCES_W."""

    source_wp: wp.array  # wp.array(dtype=vec3, shape=(num_envs, num_elements))
    offset: int
    size: int  # = num_elements * 3


@dataclass
class Vec3EnvLocalSlabBinding:
    """vec3-typed body slab with per-env origin subtraction — used by BODY_POS_W."""

    source_wp: wp.array  # wp.array(dtype=vec3, shape=(num_envs, num_bodies))
    env_origins_wp: wp.array  # wp.array(dtype=vec3, shape=(num_envs,))
    offset: int
    size: int  # = num_bodies * 3


@dataclass
class QuatSlabBinding:
    """quat-typed body slab — used by BODY_QUAT_W."""

    source_wp: wp.array  # wp.array(dtype=quat, shape=(num_envs, num_bodies))
    offset: int
    size: int  # = num_bodies * 4


@dataclass
class JointMechPowerSlabBinding:
    """Computed slab ``|τ · q̇|`` — used by JOINT_MECH_POWER_ABS.

    Both inputs are float arrays exposed by ``Articulation.data``; the fill
    kernel produces the element-wise absolute product directly into the
    unified buffer.
    """

    applied_torque_wp: wp.array
    joint_vel_wp: wp.array
    offset: int
    size: int  # = num_joints


@dataclass
class MegaKernelPlan:
    """Long-lived Warp plan for the mega-kernel ``(env, slot)`` layout."""

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
    env_slots: EnvSlots
    spec: SubtaskSpec
    state: StateAccess
    composer_state: ComposerState
    outputs: Outputs
    rotations: tuple[RotationBinding, ...]
    episode_length_buf_wp: wp.array
    effective_max_episode_length_wp: wp.array
    # Per-kind slab tuples. Each tuple feeds one fill kernel.
    float_slabs: tuple[FloatSlabBinding, ...] = field(default_factory=tuple)
    vec3_slabs: tuple[Vec3SlabBinding, ...] = field(default_factory=tuple)
    vec3_env_local_slabs: tuple[Vec3EnvLocalSlabBinding, ...] = field(default_factory=tuple)
    quat_slabs: tuple[QuatSlabBinding, ...] = field(default_factory=tuple)
    joint_mech_power_slabs: tuple[JointMechPowerSlabBinding, ...] = field(default_factory=tuple)
    # Inline rotation — when ``use_inline_rotation`` is set, the fused
    # dispatch+compose kernel rotates vec3 deltas in-register and the
    # standalone ``rotate_canonical_vec3_pair`` launch can be skipped. Only
    # enabled when there is exactly 1 rotation asset (multi-asset rotation
    # falls back to the separate rotate kernel).
    inline_rotation_quat_torch: torch.Tensor | None = None
    inline_rotation_quat_wp: wp.array | None = None
    subtask_is_rotatable_i32: torch.Tensor | None = None
    subtask_is_rotatable_wp: wp.array | None = None
    use_inline_rotation: int = 0


def build_mega_kernel_plan(command: MultiTaskCommandWarp) -> MegaKernelPlan:
    """Construct the backend-owned mega-kernel execution plan."""
    wp.init()
    s = command.spec

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

    (
        float_slabs,
        vec3_slabs,
        vec3_env_local_slabs,
        quat_slabs,
        joint_mech_power_slabs,
    ) = _resolve_slabs(command)
    rotations = build_rotation_bindings(command)
    inline_rot = _build_inline_rotation_metadata(command, rotations)

    return MegaKernelPlan(
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
        env_slots=env_slots,
        spec=spec_struct,
        state=state,
        composer_state=composer_state,
        outputs=outputs,
        rotations=rotations,
        episode_length_buf_wp=wp.from_torch(command._env.episode_length_buf),
        effective_max_episode_length_wp=wp.from_torch(command._effective_max_episode_length),
        float_slabs=float_slabs,
        vec3_slabs=vec3_slabs,
        vec3_env_local_slabs=vec3_env_local_slabs,
        quat_slabs=quat_slabs,
        joint_mech_power_slabs=joint_mech_power_slabs,
        inline_rotation_quat_torch=inline_rot["quat_torch"],
        inline_rotation_quat_wp=inline_rot["quat_wp"],
        subtask_is_rotatable_i32=inline_rot["flags_torch"],
        subtask_is_rotatable_wp=inline_rot["flags_wp"],
        use_inline_rotation=inline_rot["enabled"],
    )


def _resolve_slabs(
    command: MultiTaskCommandWarp,
) -> tuple[
    tuple[FloatSlabBinding, ...],
    tuple[Vec3SlabBinding, ...],
    tuple[Vec3EnvLocalSlabBinding, ...],
    tuple[QuatSlabBinding, ...],
    tuple[JointMechPowerSlabBinding, ...],
]:
    """Resolve each spec slab to a stable ``wp.array`` view of the scene data.

    Reads ``ProxyArray.warp`` directly — no Torch reshape, no ``wp.from_torch``,
    no laundering. Padded body buffers (vec3/quat with PhysX alignment) are
    handled natively by Warp's typed indexing in the fill kernels.

    Raises ``ValueError`` at construction time for unsupported buffer kinds —
    no silent dynamic fallback.
    """
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
            source_wp = command._env.scene[asset_name].data.joint_pos.warp
            float_slabs.append(FloatSlabBinding(source_wp=source_wp, offset=offset, size=size))
        elif kind == BUFFER_KIND.JOINT_VEL:
            source_wp = command._env.scene[asset_name].data.joint_vel.warp
            float_slabs.append(FloatSlabBinding(source_wp=source_wp, offset=offset, size=size))
        elif kind == BUFFER_KIND.BODY_POS_W:
            source_wp = command._env.scene[asset_name].data.body_pos_w.warp
            vec3_env_local_slabs.append(
                Vec3EnvLocalSlabBinding(
                    source_wp=source_wp,
                    env_origins_wp=env_origins_vec3_wp,
                    offset=offset,
                    size=size,
                )
            )
        elif kind == BUFFER_KIND.BODY_LIN_VEL_W:
            source_wp = command._env.scene[asset_name].data.body_lin_vel_w.warp
            vec3_slabs.append(Vec3SlabBinding(source_wp=source_wp, offset=offset, size=size))
        elif kind == BUFFER_KIND.BODY_ANG_VEL_W:
            source_wp = command._env.scene[asset_name].data.body_ang_vel_w.warp
            vec3_slabs.append(Vec3SlabBinding(source_wp=source_wp, offset=offset, size=size))
        elif kind == BUFFER_KIND.BODY_QUAT_W:
            source_wp = command._env.scene[asset_name].data.body_quat_w.warp
            quat_slabs.append(QuatSlabBinding(source_wp=source_wp, offset=offset, size=size))
        elif kind == BUFFER_KIND.CONTACT_NET_FORCES_W:
            source_wp = command._env.scene.sensors[asset_name].data.net_forces_w.warp
            vec3_slabs.append(Vec3SlabBinding(source_wp=source_wp, offset=offset, size=size))
        elif kind == BUFFER_KIND.JOINT_MECH_POWER_ABS:
            articulation = command._env.scene[asset_name]
            joint_mech_power_slabs.append(
                JointMechPowerSlabBinding(
                    applied_torque_wp=articulation.data.applied_torque.warp,
                    joint_vel_wp=articulation.data.joint_vel.warp,
                    offset=offset,
                    size=size,
                )
            )
        else:
            raise ValueError(
                f"Unsupported BUFFER_KIND {kind!r} for slab (asset={asset_name!r}, "
                f"offset={offset}, size={size}). Warp backends require every reader to "
                "expose a stable ``wp.array`` via ``ProxyArray.warp``."
            )

    return (
        tuple(float_slabs),
        tuple(vec3_slabs),
        tuple(vec3_env_local_slabs),
        tuple(quat_slabs),
        tuple(joint_mech_power_slabs),
    )


def _build_inline_rotation_metadata(
    command: MultiTaskCommandWarp,
    rotations: tuple[RotationBinding, ...],
) -> dict:
    """Build inline-rotation metadata + dummy fallbacks for the fused kernel.

    Always returns valid wp.array handles so the fused kernel signature is
    fixed. ``enabled`` is 1 only when there is exactly one rotation binding
    (single-asset rotation); 0 otherwise. With multiple bindings, the
    standalone rotate kernel still runs and the fused kernel skips rotation.

    For the single-binding path the quat comes straight from the asset's
    :class:`ProxyArray` (anchored by the scene) so no torch holder is needed;
    the multi-binding fallback still owns a zero tensor that must stay alive
    for its ``wp.from_torch`` view.
    """
    spec = command.spec
    num_subtasks = max(1, int(spec.state_kernel_id.numel()))
    num_envs = command.num_envs

    if len(rotations) == 1:
        binding = rotations[0]
        quat_torch_anchor: torch.Tensor | None = None
        quat_wp = binding.root_quat_w_wp

        reach_offsets = {int(o) for o in binding.reach_offsets_i32.cpu().tolist()}
        track_offsets = {int(o) for o in binding.track_offsets_i32.cpu().tolist()}

        canonical_offset = spec.canonical_offset.cpu().tolist()
        is_instant = spec.is_instant.cpu().tolist()

        flags = [0] * num_subtasks
        for sid in range(int(spec.state_kernel_id.numel())):
            co = int(canonical_offset[sid])
            if co < 0:
                continue
            in_set = reach_offsets if bool(is_instant[sid]) else track_offsets
            if co in in_set:
                flags[sid] = 1

        flags_torch = torch.tensor(flags, dtype=torch.int32, device=command.device)
        flags_wp = wp.from_torch(flags_torch)
        enabled = 1
    else:
        # Dummy zero arrays so the kernel signature stays valid; flag = 0 makes
        # the rotation branch a no-op. The zero tensor is owned here, so we
        # do need the torch holder for ``wp.from_torch`` lifetime.
        quat_torch_anchor = torch.zeros((num_envs, 4), device=command.device, dtype=torch.float32)
        quat_wp = wp.from_torch(quat_torch_anchor, dtype=wp.quat)
        flags_torch = torch.zeros(num_subtasks, dtype=torch.int32, device=command.device)
        flags_wp = wp.from_torch(flags_torch)
        enabled = 0

    return {
        "quat_torch": quat_torch_anchor,
        "quat_wp": quat_wp,
        "flags_torch": flags_torch,
        "flags_wp": flags_wp,
        "enabled": enabled,
    }


def build_rotation_bindings(command: MultiTaskCommandWarp) -> tuple[RotationBinding, ...]:
    """Build body-frame rotation bindings for policy-facing vec3 command slots."""
    s = command.spec
    bindings: list[RotationBinding] = []
    asset_names = sorted(set(s.reach_rotatable_vec3_by_asset.keys()) | set(s.track_rotatable_vec3_by_asset.keys()))
    for asset_name in asset_names:
        reach_offsets = s.reach_rotatable_vec3_by_asset.get(asset_name, ())
        track_offsets = s.track_rotatable_vec3_by_asset.get(asset_name, ())
        num_offsets = len(reach_offsets) + len(track_offsets)
        if num_offsets == 0:
            continue
        root_quat_w_wp = command._env.scene[asset_name].data.root_quat_w.warp
        reach_offsets_i32 = torch.tensor(reach_offsets, device=command.device, dtype=torch.int32)
        track_offsets_i32 = torch.tensor(track_offsets, device=command.device, dtype=torch.int32)
        bindings.append(
            RotationBinding(
                reach_offsets_i32=reach_offsets_i32,
                track_offsets_i32=track_offsets_i32,
                root_quat_w_wp=root_quat_w_wp,
                reach_offsets_wp=wp.from_torch(reach_offsets_i32),
                track_offsets_wp=wp.from_torch(track_offsets_i32),
                num_reach_offsets=len(reach_offsets),
                num_offsets=num_offsets,
            )
        )
    return tuple(bindings)
