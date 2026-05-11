# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-owned execution plan for the current mega-kernel layout.

Pure Warp — no ``import torch``. Slab resolution is direct: each
``BUFFER_KIND`` maps to a stable ``wp.array`` exposed by the IsaacLab scene
(via ``ProxyArray.warp``). Spec/env-slot/output Warp views are owned by
:class:`MultiTaskCommandWarp` and consumed here through ``command.spec_wp``,
``command.env_slots_wp``, ``command.state_wp``, ``command.outputs_wp``, and
``command.composer_state_wp``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import warp as wp

from ..kernel_ids import BUFFER_KIND
from ..kernels_wp import ComposerState, EnvSlots, Outputs, StateAccess, SubtaskSpec

if TYPE_CHECKING:
    from ..multi_task_command_warp import MultiTaskCommandWarp


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
    inline_rotation_quat_wp: wp.array | None = None
    subtask_is_rotatable_wp: wp.array | None = None
    use_inline_rotation: int = 0


def build_mega_kernel_plan(command: MultiTaskCommandWarp) -> MegaKernelPlan:
    """Construct the backend-owned mega-kernel execution plan."""
    wp.init()

    (
        float_slabs,
        vec3_slabs,
        vec3_env_local_slabs,
        quat_slabs,
        joint_mech_power_slabs,
    ) = _resolve_slabs(command)
    rotations = build_rotation_bindings(command)
    inline_quat_wp, subtask_is_rotatable_wp, use_inline_rotation = _build_inline_rotation_metadata(command, rotations)

    return MegaKernelPlan(
        env_slots=command.env_slots_wp,
        spec=command.spec_wp,
        state=command.state_wp,
        composer_state=command.composer_state_wp,
        outputs=command.outputs_wp,
        rotations=rotations,
        episode_length_buf_wp=command.episode_length_buf_wp,
        effective_max_episode_length_wp=command.effective_max_episode_length_wp,
        float_slabs=float_slabs,
        vec3_slabs=vec3_slabs,
        vec3_env_local_slabs=vec3_env_local_slabs,
        quat_slabs=quat_slabs,
        joint_mech_power_slabs=joint_mech_power_slabs,
        inline_rotation_quat_wp=inline_quat_wp,
        subtask_is_rotatable_wp=subtask_is_rotatable_wp,
        use_inline_rotation=use_inline_rotation,
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
) -> tuple[wp.array, wp.array, int]:
    """Build inline-rotation metadata + dummy fallbacks for the fused kernel.

    Always returns valid wp.array handles so the fused kernel signature is
    fixed. ``enabled`` is 1 only when there is exactly one rotation binding
    (single-asset rotation); 0 otherwise. With multiple bindings, the
    standalone rotate kernel still runs and the fused kernel skips rotation.
    """
    spec = command.spec
    s = command.spec
    num_subtasks = max(1, int(spec.state_kernel_id.numel()))
    num_envs = command.num_envs
    device_str = str(command.device)

    if len(rotations) == 1:
        # Single rotation asset — pull the quat directly from the asset's
        # ProxyArray (anchored by the scene) and build the per-subtask
        # rotatable flag from the spec's Python offset lists.
        asset_names = sorted(set(s.reach_rotatable_vec3_by_asset.keys()) | set(s.track_rotatable_vec3_by_asset.keys()))
        # ``rotations`` only emits one binding when exactly one asset has
        # rotatable offsets; that asset is the only one in ``asset_names``.
        asset_name = next(iter(asset_names))
        reach_offsets = set(s.reach_rotatable_vec3_by_asset.get(asset_name, ()))
        track_offsets = set(s.track_rotatable_vec3_by_asset.get(asset_name, ()))

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

        return (
            rotations[0].root_quat_w_wp,
            wp.array(flags, dtype=wp.int32, device=device_str),
            1,
        )

    # Dummy zero arrays so the kernel signature stays valid; flag = 0 makes
    # the rotation branch a no-op.
    return (
        wp.zeros(shape=(num_envs,), dtype=wp.quat, device=device_str),
        wp.zeros(shape=num_subtasks, dtype=wp.int32, device=device_str),
        0,
    )


def build_rotation_bindings(command: MultiTaskCommandWarp) -> tuple[RotationBinding, ...]:
    """Build body-frame rotation bindings for policy-facing vec3 command slots."""
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
