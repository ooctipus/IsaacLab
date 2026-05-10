# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-owned execution plan for the current mega-kernel layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import warp as wp

from ... import multi_task_command as _base_module
from ..kernel_ids import BUFFER_KIND
from ..kernels_wp import ComposerState, EnvSlots, Outputs, StateAccess, SubtaskSpec

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


@dataclass
class CopySlabBinding:
    """Stable scene-backed slab handle for ``fill_slab_copy``."""

    source_torch: torch.Tensor
    source_wp: wp.array
    offset: int
    size: int


@dataclass
class BodyPosSlabBinding:
    """Stable body-pos slab handle for ``fill_slab_body_pos_env_local``."""

    source_torch: torch.Tensor
    source_wp: wp.array
    env_origins_wp: wp.array
    offset: int
    size: int


@dataclass
class DynamicSlabBinding:
    """Slab whose reader allocates fresh tensors per call (cannot prebind)."""

    kind: int
    asset_name: str
    offset: int
    size: int
    is_body_pos: bool
    env_origins_wp: wp.array | None


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
    copy_slabs: tuple[CopySlabBinding, ...] = field(default_factory=tuple)
    body_pos_slabs: tuple[BodyPosSlabBinding, ...] = field(default_factory=tuple)
    dynamic_slabs: tuple[DynamicSlabBinding, ...] = field(default_factory=tuple)
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
    # Combined slab-copy launch — when num_copy_slabs <= MAX_COPY_SLABS,
    # all copy slabs run in a single ``fill_slabs_combined_8`` launch.
    combined_slab_sources_wp: tuple[wp.array, ...] = ()
    combined_slab_cumsizes_torch: torch.Tensor | None = None
    combined_slab_cumsizes_wp: wp.array | None = None
    combined_slab_offsets_torch: torch.Tensor | None = None
    combined_slab_offsets_wp: wp.array | None = None
    combined_slab_total_size: int = 0
    combined_slab_num_slabs: int = 0


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

    copy_slabs, body_pos_slabs, dynamic_slabs = build_slab_bindings(command)
    rotations = build_rotation_bindings(command)
    inline_rot = _build_inline_rotation_metadata(command, rotations)
    combined_slabs = build_combined_copy_slab_metadata(command, copy_slabs)

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
        copy_slabs=copy_slabs,
        body_pos_slabs=body_pos_slabs,
        dynamic_slabs=dynamic_slabs,
        inline_rotation_quat_torch=inline_rot["quat_torch"],
        inline_rotation_quat_wp=inline_rot["quat_wp"],
        subtask_is_rotatable_i32=inline_rot["flags_torch"],
        subtask_is_rotatable_wp=inline_rot["flags_wp"],
        use_inline_rotation=inline_rot["enabled"],
        combined_slab_sources_wp=combined_slabs["sources_wp"],
        combined_slab_cumsizes_torch=combined_slabs["cumsizes_torch"],
        combined_slab_cumsizes_wp=combined_slabs["cumsizes_wp"],
        combined_slab_offsets_torch=combined_slabs["offsets_torch"],
        combined_slab_offsets_wp=combined_slabs["offsets_wp"],
        combined_slab_total_size=combined_slabs["total_size"],
        combined_slab_num_slabs=combined_slabs["num_slabs"],
    )


def _read_slab_source(command: MultiTaskCommandWarp, kind: int, asset_name: str, size: int) -> torch.Tensor:
    """Call the reader and reshape to ``[num_envs, size]``."""
    raw = _base_module.BUFFER_KIND_READERS[kind](command._env, asset_name)
    raw_per_env = raw.numel() // command.num_envs
    if raw_per_env != size:
        raise RuntimeError(
            f"State kernel output dim mismatch for slab (kind={kind}, asset={asset_name}): "
            f"reader returned {raw_per_env} floats per env, but slab was sized for {size}."
        )
    return raw.reshape(command.num_envs, size)


def build_slab_bindings(
    command: MultiTaskCommandWarp,
) -> tuple[tuple[CopySlabBinding, ...], tuple[BodyPosSlabBinding, ...], tuple[DynamicSlabBinding, ...]]:
    """Prebind slab source handles. Detects unstable readers and falls back."""
    spec = command.spec
    kinds = spec.slab_buffer_kinds
    assets = spec.slab_asset_names
    offsets = spec.slab_offsets_py
    sizes = spec.slab_sizes_py
    body_pos_kind = int(BUFFER_KIND.BODY_POS_W)

    env_origins_torch = command._env.scene.env_origins
    env_origins_wp = wp.from_torch(env_origins_torch)

    copy_slabs: list[CopySlabBinding] = []
    body_pos_slabs: list[BodyPosSlabBinding] = []
    dynamic_slabs: list[DynamicSlabBinding] = []

    for slab_id in range(len(kinds)):
        kind = kinds[slab_id]
        asset_name = assets[slab_id]
        offset = offsets[slab_id]
        size = sizes[slab_id]

        first = _read_slab_source(command, kind, asset_name, size)
        second = _read_slab_source(command, kind, asset_name, size)
        is_stable = first.data_ptr() == second.data_ptr()
        is_body_pos = kind == body_pos_kind

        if not is_stable:
            dynamic_slabs.append(
                DynamicSlabBinding(
                    kind=kind,
                    asset_name=asset_name,
                    offset=offset,
                    size=size,
                    is_body_pos=is_body_pos,
                    env_origins_wp=env_origins_wp if is_body_pos else None,
                )
            )
            continue

        if is_body_pos:
            body_pos_slabs.append(
                BodyPosSlabBinding(
                    source_torch=first,
                    source_wp=wp.from_torch(first),
                    env_origins_wp=env_origins_wp,
                    offset=offset,
                    size=size,
                )
            )
        else:
            copy_slabs.append(
                CopySlabBinding(
                    source_torch=first,
                    source_wp=wp.from_torch(first),
                    offset=offset,
                    size=size,
                )
            )

    return tuple(copy_slabs), tuple(body_pos_slabs), tuple(dynamic_slabs)


_MAX_COMBINED_COPY_SLABS = 8
# Combined-launch only wins when per-launch overhead is a noticeable fraction
# of the read phase, which happens at large env counts. At 16k envs the
# per-thread routing (linear scan + 8-way branch) costs more than the saved
# launches; at 131k envs it's a ~25% win on the full step. Threshold picked
# empirically — re-run bench_multi_task_command_backends with locomotion
# preset across env counts to retune.
_COMBINED_SLAB_NUM_ENVS_MIN = 32768


def build_combined_copy_slab_metadata(
    command: MultiTaskCommandWarp,
    copy_slabs: tuple[CopySlabBinding, ...],
) -> dict:
    """Build per-thread routing for the combined slab-copy kernel."""
    num_slabs = len(copy_slabs)
    if num_slabs == 0 or num_slabs > _MAX_COMBINED_COPY_SLABS or command.num_envs < _COMBINED_SLAB_NUM_ENVS_MIN:
        # Disabled — backend falls back to per-slab launches.
        dummy = torch.zeros((command.num_envs, 1), device=command.device, dtype=torch.float32)
        dummy_wp = wp.from_torch(dummy)
        return {
            "sources_wp": tuple(dummy_wp for _ in range(_MAX_COMBINED_COPY_SLABS)),
            "cumsizes_torch": None,
            "cumsizes_wp": None,
            "offsets_torch": None,
            "offsets_wp": None,
            "total_size": 0,
            "num_slabs": 0,
        }

    sizes = [int(s.size) for s in copy_slabs]
    offsets = [int(s.offset) for s in copy_slabs]
    cumsizes = [0]
    for s in sizes:
        cumsizes.append(cumsizes[-1] + s)

    cumsizes_torch = torch.tensor(cumsizes, dtype=torch.int32, device=command.device)
    offsets_torch = torch.tensor(offsets, dtype=torch.int32, device=command.device)

    # Pad source array references up to MAX with a zero-sized dummy.
    dummy = torch.zeros((command.num_envs, 1), device=command.device, dtype=torch.float32)
    dummy_wp = wp.from_torch(dummy)
    sources_wp: list[wp.array] = [s.source_wp for s in copy_slabs]
    while len(sources_wp) < _MAX_COMBINED_COPY_SLABS:
        sources_wp.append(dummy_wp)

    return {
        "sources_wp": tuple(sources_wp),
        "cumsizes_torch": cumsizes_torch,
        "cumsizes_wp": wp.from_torch(cumsizes_torch),
        "offsets_torch": offsets_torch,
        "offsets_wp": wp.from_torch(offsets_torch),
        "total_size": cumsizes[-1],
        "num_slabs": num_slabs,
    }


def _build_inline_rotation_metadata(
    command: MultiTaskCommandWarp,
    rotations: tuple[RotationBinding, ...],
) -> dict:
    """Build inline-rotation metadata + dummy fallbacks for the fused kernel.

    Always returns valid wp.array handles so the fused kernel signature is
    fixed. ``enabled`` is 1 only when there is exactly one rotation binding
    (single-asset rotation); 0 otherwise. With multiple bindings, the
    standalone rotate kernel still runs and the fused kernel skips rotation.
    """
    spec = command.spec
    num_subtasks = max(1, int(spec.state_kernel_id.numel()))
    num_envs = command.num_envs

    if len(rotations) == 1:
        binding = rotations[0]
        quat_torch = binding.root_quat_w
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
        # the rotation branch a no-op.
        quat_torch = torch.zeros((num_envs, 4), device=command.device, dtype=torch.float32)
        quat_wp = wp.from_torch(quat_torch)
        flags_torch = torch.zeros(num_subtasks, dtype=torch.int32, device=command.device)
        flags_wp = wp.from_torch(flags_torch)
        enabled = 0

    return {
        "quat_torch": quat_torch,
        "quat_wp": quat_wp,
        "flags_torch": flags_torch,
        "flags_wp": flags_wp,
        "enabled": enabled,
    }


def _root_quat_torch(command: MultiTaskCommandWarp, asset_name: str) -> torch.Tensor:
    quat = command._env.scene[asset_name].data.root_quat_w
    if isinstance(quat, torch.Tensor):
        return quat
    if hasattr(quat, "torch"):
        return quat.torch
    return wp.to_torch(quat)


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
