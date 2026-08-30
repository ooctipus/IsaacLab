# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable spec tables built once from a :class:`MultiTaskCfg` at command-term init.

Separation of concerns:

- This module holds **build-time** logic: cfg → flat, indexable tables (:class:`TaskSpec`).
  All computation here happens exactly once; nothing is read in the per-step hot path.
- :mod:`multi_task_command` holds **run-time** logic: resample + per-step dispatch over
  those tables.

The spec handles subtask deduplication (identical signatures collapse to one row),
per-task layout computation (slot offsets / strides / total stride for the flat targets
buffer), and the correctness gate that enforces stride consistency within each
``(state_kernel, entity)`` equivalence class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene

    from .multi_task_cfg import MultiTaskCfg


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def pad_index_rows(index_rows: list[list[int]], device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ragged ``[[id, ...], ...]`` → rectangular ``(index_table, valid_table)``.

    Short rows are right-padded with ``-1`` in ``index_table``; ``valid_table`` marks
    the real entries. Callers typically ``clamp_min(0)`` the index table for gathers
    and use ``valid_table`` to suppress the garbage at padded positions.
    """
    max_len = max((len(row) for row in index_rows), default=0)
    if max_len == 0:
        index_table = torch.full((len(index_rows), 1), -1, dtype=torch.long, device=device)
        valid_table = torch.zeros((len(index_rows), 1), dtype=torch.bool, device=device)
        return index_table, valid_table

    index_table = torch.full((len(index_rows), max_len), -1, dtype=torch.long, device=device)
    valid_table = torch.zeros((len(index_rows), max_len), dtype=torch.bool, device=device)
    for row_index, row in enumerate(index_rows):
        if not row:
            continue
        count = len(row)
        index_table[row_index, :count] = torch.tensor(row, dtype=torch.long, device=device)
        valid_table[row_index, :count] = True
    return index_table, valid_table


def _ids_sig(ids) -> tuple:
    """Stable hashable signature for ids (list/tensor/slice/None) used in subtask dedup."""
    if ids is None:
        return ()
    if isinstance(ids, slice):
        return ("ALL",)
    if torch.is_tensor(ids):
        return tuple(int(x) for x in ids.tolist())
    return tuple(int(x) for x in ids)


def _resolve_asset_once(asset_cfg: SceneEntityCfg, scene: InteractiveScene, _seen: set) -> None:
    """Idempotent ``SceneEntityCfg.resolve`` — guards against double-resolve.

    ``SceneEntityCfg.resolve`` is *not* idempotent for regex-style ``body_names``:
    the first call leaves ``body_names`` as the regex string and sets
    ``body_ids`` to the resolved list; the second call sees both populated
    and rejects ``regex_string != resolved_name``. Sharing a single
    ``SceneEntityCfg`` instance across multiple subtasks (which is what
    makes flat preset authoring viable — every safety subtask references
    the same template) thus blows up at spec build time.

    Tracking resolved asset_cfg objects by ``id()`` and skipping subsequent
    resolves on the same instance solves this without any cfg-level
    workaround. Different instances with the same content are still each
    resolved (and produce identical signatures via :func:`_ids_sig`, so
    subtask dedup still collapses them into one row).
    """
    obj_id = id(asset_cfg)
    if obj_id in _seen:
        return
    asset_cfg.resolve(scene)
    _seen.add(obj_id)


def _subtask_signature(subtask_cfg, scene: InteractiveScene, _seen: set) -> tuple:
    """Dedup signature for a subtask cfg — every field that changes behavior is here."""
    _resolve_asset_once(subtask_cfg.asset_cfg, scene, _seen)
    asset = subtask_cfg.asset_cfg
    sampler_sig = (
        int(subtask_cfg.sampler.kernel),
        tuple(map(float, subtask_cfg.sampler.minimum)),
        tuple(map(float, subtask_cfg.sampler.maximum)),
        int(subtask_cfg.sampler.out_dim) if subtask_cfg.sampler.out_dim is not None else -1,
    )
    return (
        type(subtask_cfg).__name__,
        asset.name,
        _ids_sig(asset.body_ids),
        _ids_sig(asset.joint_ids),
        int(subtask_cfg.state_kernel),
        int(subtask_cfg.metric_kernel),
        sampler_sig,
        int(subtask_cfg.activation_kernel),
        float(subtask_cfg.activation_kernel_param),
    )


# -----------------------------------------------------------------------------
# Spec dataclass
# -----------------------------------------------------------------------------


@dataclass
class TaskSpec:
    """Flattened, indexable representation of a :class:`MultiTaskCfg`.

    Immutable once built. ``M`` = number of unique subtasks after dedup, ``T`` = number
    of tasks, ``k_max`` = max active slots across tasks, ``E`` = number of distinct entities.
    """

    task_names: list[str]
    task_subtask_ids: torch.Tensor
    """``[T, k_max]`` long — ragged, ``-1`` pads."""
    task_subtask_valid: torch.Tensor
    """``[T, k_max]`` bool — ``True`` where the slot holds a real subtask id."""
    task_slot_count: torch.Tensor
    """``[T]`` int — number of active slots per task."""

    # Stride / offset tables for the flat targets buffer.
    state_stride: torch.Tensor
    """``[M]`` int — number of floats per subtask's state/target slice."""
    task_slot_strides: torch.Tensor
    """``[T, k_max]`` int — per-slot stride; ``0`` on padded slots."""
    task_slot_offsets: torch.Tensor
    """``[T, k_max]`` int — cumulative stride offset per slot; ``0`` on padded slots."""
    task_total_stride: torch.Tensor
    """``[T]`` int — ``sum(state_stride[task_subtask_ids[t, :]])``."""

    # Per-subtask kernel selections (all ``[M]``).
    state_kernel_id: torch.Tensor
    metric_kernel_id: torch.Tensor
    sampler_kernel_id: torch.Tensor
    sampler_kernel_param: torch.Tensor
    """``[M, 2 · D_max_sampler_dim]`` float — interleaved ``[min, range]`` per subtask."""
    activation_kernel_id: torch.Tensor
    activation_kernel_param: torch.Tensor

    is_tracking: torch.Tensor
    """``[M]`` bool — quality-side subtasks (the unified "tracking" kind that
    includes both ordinary tracking goals and soft-safety constraints).
    Tracking and instant are mutually exclusive."""
    is_instant: torch.Tensor
    """``[M]`` bool — milestone subtasks that drive ``instant_gate`` through
    a one-way latch. Mutually exclusive with ``is_tracking``."""
    expose_in_obs: torch.Tensor
    """``[M]`` bool — whether this subtask's delta channel + active-mask bit
    appears in the policy obs (``command_track`` / ``command_reach`` /
    ``command_active``). Always ``True`` for instant subtasks. ``True`` for
    tracking subtasks with ``expose_in_obs=True`` (the default — ordinary
    tracking goals); ``False`` for tracking subtasks declared with
    ``expose_in_obs=False`` (soft-safety constraints — internal-only)."""

    subtask_asset_cfgs: list[SceneEntityCfg]
    subtask_entity_id: torch.Tensor
    """``[M]`` int — dedup key for ``(asset.name, body_ids, joint_ids)``."""

    # Read-group dispatch — fuses the state-kernel compute across subtasks that
    # share both the kernel id AND the per-subtask state stride (proxying for
    # source-tensor shape). Fusion crosses asset boundaries: two subtasks on
    # different sensors using the same kernel with matching stride land in one
    # group and share a single batched compute call.
    read_group_id: torch.Tensor
    """``[M]`` int — which read group each subtask belongs to."""
    read_group_state_kernel_id: torch.Tensor
    """``[G]`` int — the state kernel shared by every subtask in the group."""
    read_group_member_sids: list[list[int]]
    """``[G]`` list-of-lists — subtask ids per group in insertion order."""
    read_group_member_asset_cfgs: list[list[SceneEntityCfg]]
    """``[G]`` list-of-lists — per-member asset_cfgs, parallel to
    :attr:`read_group_member_sids`. Each member keeps its own asset_cfg (no
    merging); the dispatch calls ``source_fn`` per member and stacks the results."""
    subtask_member_index: torch.Tensor
    """``[M]`` int — each subtask's position within its group's member list.
    Used to slice the stacked compute output ``x_stacked[m, env, ...]`` at dispatch
    time."""

    # Unified per-step state buffer.
    # ``unified_width`` is the total number of floats across every
    # ``(buffer_kind, asset_name)`` slab referenced by the cfg. The command term
    # allocates ``[N, unified_width]`` once; each step, the read dispatch fills
    # every slab by calling the kernel's registered reader. From then on
    # Phase 2 only does ``(offset, stride)`` gathers into this tensor —
    # no per-asset lookup, no dict of views.
    unified_width: int
    """Total floats across all slabs — size of the per-step unified state tensor."""
    slab_buffer_kinds: list[int]
    """``[S]`` int — buffer kind for each slab."""
    slab_asset_names: list[str]
    """``[S]`` str — asset name for each slab."""
    slab_offsets: torch.Tensor
    """``[S]`` int — starting float offset of each slab in the unified buffer."""
    slab_sizes: torch.Tensor
    """``[S]`` int — total float count per env for each slab."""

    # Read-group gather indices — precomputed absolute indices into the unified
    # buffer that Phase 2 uses to pull each group's member inputs in one shot.
    read_group_gather_indices: list[torch.Tensor]
    """``[G]`` tensors of shape ``[M_g, slice_size_g]`` — absolute unified indices
    the group's members read. ``unified[:, indices]`` → ``[N, M_g, slice_size_g]``."""

    # Per-subtask gather CSR — flat layout consumed by the Warp mega-dispatch
    # kernel. ``subtask_gather_offset[sid]`` points into ``gather_indices_flat``;
    # ``subtask_gather_count[sid]`` says how many floats to read. Unlike the
    # per-group layout above, this tells ONE thread exactly what to gather for
    # its subtask without needing to know about the group.
    gather_indices_flat: torch.Tensor
    """``[sum(gather_count)]`` int — concatenated absolute unified indices, one
    block per subtask in ``sid`` order."""
    subtask_gather_offset: torch.Tensor
    """``[M]`` int — start of each subtask's block in :attr:`gather_indices_flat`."""
    subtask_gather_count: torch.Tensor
    """``[M]`` int — length of each subtask's block; equals slice_size of its
    read group. Typically 1, 3, 4, or K·3 (for K-body contact subtasks)."""

    # Python-list companions to the build-time integer tables. The hot path uses
    # these to drive its outer ``for`` loops without touching device tensors —
    # converting tensor scalars in a loop forces a per-iteration CPU sync.
    slab_offsets_py: list[int]
    """``[S]`` int — Python copy of :attr:`slab_offsets` for hot-path iteration."""
    slab_sizes_py: list[int]
    """``[S]`` int — Python copy of :attr:`slab_sizes` for hot-path iteration."""
    read_group_state_kernel_id_py: list[int]
    """``[G]`` int — Python copy of :attr:`read_group_state_kernel_id`."""
    read_group_metric_kids_py: list[list[int]]
    """``[G]`` list-of-lists — sorted unique metric_kernel_id values per group.
    Eliminates the per-step ``torch.unique(metric_kids).tolist()`` CPU sync."""
    unique_activation_kids_py: list[int]
    """List of every activation_kernel_id appearing in any subtask, sorted.
    Eliminates the per-step ``torch.unique(activation_kids).tolist()`` CPU sync."""

    # Canonical observation layout — split into REACH and TRACK tensors so the
    # policy reads them as semantically distinct obs blocks, not as one stream
    # that needs a per-slot "instant vs tracking" flag to interpret.
    #
    # Two independent canonical widths are built at spec time, one per subtask
    # type. Each ``(entity, kernel)`` pair that appears among instant subtasks
    # gets disjoint channels in the reach layout; same for tracking subtasks in
    # the track layout. A pair referenced by BOTH subtask kinds lands in both
    # tensors with separate offsets — no conflict. POS and POS_Z on the same
    # entity also get disjoint channels (no z-slot aliasing), so "reach standing
    # z" + "track crouch z" coexist cleanly.
    #
    # Joint kernels (no canonical projection) leave ``canonical_offset = -1``.
    reach_canonical_width: int
    """Total channels in the policy's ``goal_reach_delta`` — only instant subtasks."""
    track_canonical_width: int
    """Total channels in the policy's ``goal_track_delta`` — only tracking subtasks."""
    canonical_offset: torch.Tensor
    """``[M]`` int — the subtask's scatter offset INTO ITS OWN layout (reach layout
    if the subtask is instant, track layout if tracking). ``-1`` if no canonical
    projection."""
    canonical_stride: torch.Tensor
    """``[M]`` int — canonical projection width; matches ``state_stride`` when
    projected, else ``-1``."""

    # Per-task active-channel mask — answers "which channels of the flat
    # ``command`` obs (``cat([command_reach, command_track])``) are populated
    # by a live subtask of this task?". Used by :class:`MultiTaskCommand` to
    # refresh its ``_command_active`` tensor on resample via a single indexed
    # copy: ``_command_active[env_ids] = task_active_mask[task_idx]``. Entirely
    # a function of the spec — no runtime state needed.
    task_active_mask: torch.Tensor
    """``[T, reach_canonical_width + track_canonical_width]`` float — ``1.0``
    where the channel is populated by a subtask of task ``t``, ``0.0``
    otherwise. Layout matches ``cat([command_reach, command_track], dim=-1)``:
    ``[0, reach_canonical_width)`` is reach, remainder is track. Joint
    kernels (no canonical projection) contribute no ``1.0``."""

    # Per-layout rotatable-vec3 slot offsets, grouped by asset. The composer
    # reads state in world frame, computes world-frame delta, then
    # :meth:`MultiTaskCommandTorch._compute_state_delta_error_reference` rotates
    # every POS / LIN_VEL / ANG_VEL slot into the originating asset's root
    # frame so the emitted ``command_reach`` / ``command_track`` tensors are
    # body-aligned for policy consumption. Reward/error (``_buf_error``)
    # stays frame-agnostic since it's an L2 norm — we only rotate the obs
    # tensor, not the error.
    #
    # Map: ``{asset_name: (off0, off1, ...)}`` — each offset points at a
    # ``[off, off+3)`` slice in the respective canonical layout. Multiple
    # subtasks sharing ``(entity, kernel)`` collapse to one slot.
    reach_rotatable_vec3_by_asset: dict[str, tuple[int, ...]] = field(default_factory=dict)
    """Per-asset rotatable offsets in ``command_reach``. See class-level note."""
    track_rotatable_vec3_by_asset: dict[str, tuple[int, ...]] = field(default_factory=dict)
    """Per-asset rotatable offsets in ``command_track``. See class-level note."""

    # Per-task "has at least one instant subtask" flag. Selects the MDP-level
    # timeout semantic per env at runtime:
    #
    #   - Reach tasks (``has_instant=True``, with or without tracking) are
    #     infinite-horizon in spirit: the episode cap is an artificial budget,
    #     and "policy ran out of time trying to reach" is a truncation. rsl_rl
    #     should bootstrap ``γ·V(s_T)`` so value propagates through partial
    #     progress even when reach fails.
    #   - Pure-tracking tasks (``has_instant=False``) are finite-horizon: the
    #     episode cap IS the task's end. Reward ``G = transit_mean`` is the
    #     complete episodic return. Bootstrap would double-count.
    #
    # Two ``DoneTerm`` functions gate on this field — see
    # :func:`~..mdp.terminations.time_out_reach_truncate` /
    # :func:`~..mdp.terminations.time_out_track_terminate`.
    task_has_instant: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.bool))
    """``[T]`` bool — ``True`` iff task ``t`` contains ≥1 instant subtask."""

    # Per-task per-kernel target offset for debug visualization. ``[T, K]``
    # where ``K = len(STATE_KERNELS)``. Each entry is the offset into
    # ``_targets_flat`` for that ``(task, kernel)`` pair, or ``-1`` if the
    # task doesn't use that kernel. The base command term reads this row at
    # ``debug_vis_callback`` time to gather per-env targets and dispatch to
    # each registered viz ``update_fn`` in :data:`~.impl.kernels_viz.VIZ_REGISTRY`.
    #
    # Multiple subtasks of the same kernel within one task collapse to the
    # first slot's offset (multi-instance kernels per task aren't currently
    # used; the assumption is captured by taking the first match per task).
    task_kernel_target_offset: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 0), dtype=torch.long))
    """``[T, num_state_kernels]`` int — per-task offset of each kernel's
    target slice in ``_targets_flat``. ``-1`` if the task doesn't use the
    kernel. Read at debug-vis time only."""


# -----------------------------------------------------------------------------
# Spec factory
# -----------------------------------------------------------------------------


def build_spec(cfg: MultiTaskCfg, scene: InteractiveScene, device: torch.device | str) -> TaskSpec:
    """Build a :class:`TaskSpec` from a cfg. One-shot; called from command-term init.

    Pipeline:

    1. Walk every subtask cfg, dedup by signature, collect per-subtask kernel / sampler
       / type metadata.
    2. Pad the ragged ``task → subtask ids`` table to ``[T, k_max]``.
    3. Enforce ``state_stride`` consistency within every ``(state_kernel, entity)``
       equivalence class — the per-step dispatch picks one stride per class, so any
       mismatch would silently mis-slice targets. Raised as ``ValueError`` with a
       descriptor naming the offending class.
    4. Compute per-task ``slot_strides``, ``slot_offsets``, ``total_stride`` for the
       flat targets buffer layout.
    5. Pad sampler params to rectangular ``[M, Pmax]``.
    """
    from .multi_task_cfg import MultiTaskCfg as _MultiTaskCfg

    task_names = list(cfg.tasks.keys())
    sig_to_sid: dict[tuple, int] = {}

    # Per-subtask lists (built in dedup order).
    state_kernel_id: list[int] = []
    metric_kernel_id: list[int] = []
    sampler_kernel_id: list[int] = []
    sampler_kernel_param_rows: list[torch.Tensor] = []
    activation_kernel_id: list[int] = []
    activation_kernel_param: list[float] = []
    is_tracking: list[bool] = []
    is_instant: list[bool] = []
    expose_in_obs: list[bool] = []
    subtask_asset_cfgs: list[SceneEntityCfg] = []
    state_stride_list: list[int] = []

    entity_sig_to_id: dict[tuple, int] = {}
    subtask_entity_id: list[int] = []
    task_to_subtask_ids: list[list[int]] = []
    p_max = 0

    # Tracks ``id(asset_cfg)`` for instances that have already had
    # ``resolve()`` called. Shared :class:`SceneEntityCfg` instances across
    # tasks (e.g. the safety templates) thus get resolved exactly once,
    # avoiding the regex/id consistency check on second-resolve.
    resolved_asset_cfgs: set[int] = set()

    for task_name in task_names:
        row: list[int] = []
        for subtask_cfg in cfg.tasks[task_name]:
            sig = _subtask_signature(subtask_cfg, scene, resolved_asset_cfgs)
            sid = sig_to_sid.get(sig)

            if sid is None:
                sid = len(state_kernel_id)
                sig_to_sid[sig] = sid

                asset_cfg = subtask_cfg.asset_cfg
                subtask_asset_cfgs.append(asset_cfg)

                ent_sig = (asset_cfg.name, _ids_sig(asset_cfg.body_ids), _ids_sig(asset_cfg.joint_ids))
                ent_id = entity_sig_to_id.setdefault(ent_sig, len(entity_sig_to_id))
                subtask_entity_id.append(ent_id)

                state_kernel_id.append(int(subtask_cfg.state_kernel))
                metric_kernel_id.append(int(subtask_cfg.metric_kernel))
                sampler_kernel_id.append(int(subtask_cfg.sampler.kernel))

                prow = subtask_cfg.sampler.get_kernel_input(device=device)
                sampler_kernel_param_rows.append(prow)
                p_max = max(p_max, int(prow.numel()))
                # Sampler output dim (half the interleaved min/range count); out_dim
                # override already folds into prow's length.
                state_stride_list.append(int(prow.numel()) // 2)

                activation_kernel_id.append(int(subtask_cfg.activation_kernel))
                activation_kernel_param.append(float(subtask_cfg.activation_kernel_param))

                is_tracking_subtask = isinstance(subtask_cfg, _MultiTaskCfg.TrackingTaskCfg)
                is_instant_subtask = isinstance(subtask_cfg, _MultiTaskCfg.InstantaneousTaskCfg)
                is_tracking.append(is_tracking_subtask)
                is_instant.append(is_instant_subtask)
                # Instant subtasks always show in obs (the policy needs the delta
                # to act on the milestone). Tracking subtasks expose by default
                # but can opt out via ``expose_in_obs=False`` for soft-safety
                # constraints — quality factors the policy should learn to
                # respect implicitly via the reward gradient, not by observing
                # their internal violation directly.
                if is_tracking_subtask:
                    expose_in_obs.append(bool(getattr(subtask_cfg, "expose_in_obs", True)))
                else:
                    expose_in_obs.append(True)

            row.append(sid)

        if not row:
            raise ValueError(f"Task '{task_name}' has no subtasks.")
        task_to_subtask_ids.append(row)

    # Correctness gate: within each (state_kid, entity) class, all subtasks must share
    # the same stride. Per-step dispatch picks one stride per class; disagreement would
    # silently mis-slice targets.
    _assert_stride_consistent_within_class(state_kernel_id, subtask_entity_id, state_stride_list, subtask_asset_cfgs)

    # Pad task → subtask table to [T, k_max].
    task_subtask_ids, task_subtask_valid = pad_index_rows(task_to_subtask_ids, device=device)
    T = len(task_to_subtask_ids)
    k_max = int(task_subtask_ids.shape[1])
    task_slot_count = torch.tensor([len(r) for r in task_to_subtask_ids], dtype=torch.long, device=device)
    # ``True`` iff task ``t`` has ≥1 instant subtask. Drives per-env timeout
    # semantic at runtime: reach tasks → truncation + bootstrap; pure-tracking
    # → finite-horizon termination. See :attr:`TaskSpec.task_has_instant`.
    task_has_instant = torch.tensor(
        [any(bool(is_instant[sid]) for sid in row) for row in task_to_subtask_ids],
        dtype=torch.bool,
        device=device,
    )

    # Per-task slot offsets/strides + total_stride (cumulative sum along slot dim).
    state_stride_tensor = torch.tensor(state_stride_list, dtype=torch.long, device=device)
    task_slot_offsets = torch.zeros((T, max(k_max, 1)), dtype=torch.long, device=device)
    task_slot_strides = torch.zeros((T, max(k_max, 1)), dtype=torch.long, device=device)
    task_total_stride = torch.zeros(T, dtype=torch.long, device=device)
    # Walk subtasks per task, computing slot offsets and the per-task
    # per-kernel target offset table for debug viz.
    from .impl.kernels_torch import STATE_KERNELS as _STATE_KERNELS  # noqa: PLC0415

    num_state_kernels = len(_STATE_KERNELS)
    task_kernel_target_offset = torch.full((T, num_state_kernels), -1, dtype=torch.long, device=device)
    for t, row in enumerate(task_to_subtask_ids):
        offset = 0
        for slot, sid in enumerate(row):
            stride = int(state_stride_tensor[sid].item())
            task_slot_offsets[t, slot] = offset
            task_slot_strides[t, slot] = stride
            kid = int(state_kernel_id[sid])
            # First-match wins per (task, kernel). Multi-instance kernels
            # within a task aren't currently used; spec-build asserts this
            # implicitly via the canonical-layout merge.
            if task_kernel_target_offset[t, kid] < 0:
                task_kernel_target_offset[t, kid] = offset
            offset += stride
        task_total_stride[t] = offset

    # Pad sampler params to rectangular [M, Pmax] (even length for [min, range] pairs).
    if p_max % 2 == 1:
        p_max += 1
    sampler_param_t = torch.zeros((len(sampler_kernel_param_rows), max(p_max, 2)), dtype=torch.float32, device=device)
    for j, prow in enumerate(sampler_kernel_param_rows):
        sampler_param_t[j, : prow.numel()] = prow

    # Per-subtask gather size — number of floats read from the unified buffer.
    # Used to key read groups so kernels with stride-1 output but variable K
    # (BODY_CONTACT_COUNT, BODY_CONTACT_COUNT_DIFF, JOINT_MECH_POWER)
    # don't get pooled across mismatched K values.
    from .impl.kernels_torch import (  # noqa: PLC0415
        STATE_KERNEL_BUFFER_KIND,
        buffer_kind_is_body_indexed,
        buffer_kind_per_element_stride,
        state_kernel_intra_body_stride,
    )

    def _resolve_id_count(asset_cfg: SceneEntityCfg, body_indexed: bool) -> int:
        """Number of body / joint elements gathered for this subtask."""
        ids = asset_cfg.body_ids if body_indexed else asset_cfg.joint_ids
        if isinstance(ids, slice):
            entity = scene[asset_cfg.name]
            num_elem = entity.num_bodies if body_indexed else entity.num_joints
            return len(list(range(num_elem))[ids])
        return len(list(ids))

    subtask_gather_size: list[int] = []
    for sid in range(len(state_kernel_id)):
        skid = int(state_kernel_id[sid])
        bk = int(STATE_KERNEL_BUFFER_KIND[skid])
        body_indexed = buffer_kind_is_body_indexed(bk)
        per_stride = buffer_kind_per_element_stride(bk)
        intra_stride = state_kernel_intra_body_stride(skid)
        # Body-indexed buffers read ``per_stride`` floats per body and the
        # state kernel's ``intra_body_stride`` selects a sub-slice. For full
        # block reads ``intra_stride == per_stride`` so gather = K · per_stride.
        # For sliced reads (BODY_POS_Z) gather = K · intra_stride.
        # Joint-indexed buffers always have per_stride == intra_stride == 1.
        slice_per_elem = intra_stride if body_indexed else per_stride
        K = _resolve_id_count(subtask_asset_cfgs[sid], body_indexed=body_indexed)
        # CONTACT-family kernels reduce K bodies but read the full per_stride
        # block, so override to full block-read size regardless of intra_stride.
        from .kernel_ids import STATE_KERNEL_ID as _SKID  # noqa: PLC0415

        if skid in (
            int(_SKID.BODY_CONTACT),
            int(_SKID.BODY_CONTACT_COUNT),
            int(_SKID.BODY_CONTACT_COUNT_DIFF),
        ):
            slice_per_elem = per_stride
        subtask_gather_size.append(K * slice_per_elem)

    # Read groups — dispatch-time kernel fusion across subtasks with matching
    # (state_kid, gather_size). Each member keeps its own asset_cfg; the dispatch
    # stages per-member sources and runs ONE batched compute per group, even
    # when members live on different scene assets.
    (
        read_group_id_list,
        read_group_member_sids,
        read_group_member_asset_cfgs,
        read_group_state_kernel_id_list,
        subtask_member_index_list,
    ) = _compute_read_groups(
        state_kernel_id=state_kernel_id,
        subtask_asset_cfgs=subtask_asset_cfgs,
        state_stride_list=state_stride_list,
        subtask_gather_size=subtask_gather_size,
    )

    # Unified buffer layout + per-group gather indices.
    (
        unified_width,
        slab_buffer_kinds,
        slab_asset_names,
        slab_offsets_list,
        slab_sizes_list,
        read_group_gather_indices_list,
    ) = _compute_unified_layout(
        scene=scene,
        state_kernel_id=state_kernel_id,
        subtask_asset_cfgs=subtask_asset_cfgs,
        read_group_state_kernel_id_list=read_group_state_kernel_id_list,
        read_group_member_sids=read_group_member_sids,
        read_group_member_asset_cfgs=read_group_member_asset_cfgs,
    )

    # Canonical obs layout — split into two independent tensors by subtask type.
    # Instant subtasks write their deltas into the reach tensor; tracking subtasks
    # into the track tensor. Same (entity, kernel) used by both types gets channels
    # in both; no aliasing. POS / POS_Z on the same entity also get disjoint
    # channels so "reach standing" + "track crouch z" can coexist.
    (
        reach_canonical_width,
        track_canonical_width,
        canonical_offset_list,
        canonical_stride_list,
    ) = _compute_canonical_layout(
        num_entities=len(entity_sig_to_id),
        subtask_entity_id=subtask_entity_id,
        state_kernel_id=state_kernel_id,
        state_stride_list=state_stride_list,
        subtask_asset_cfgs=subtask_asset_cfgs,
        is_instant=is_instant,
        expose_in_obs=expose_in_obs,
    )

    # Rotatable-vec3 slot offsets per layout, grouped by originating asset.
    # POS / LIN_VEL / ANG_VEL state kernels produce world-frame 3-vec deltas
    # that MultiTaskCommand's dispatch rotates into the asset's root frame
    # after scatter — so downstream obs terms see body-frame deltas without
    # any frame logic. Multiple subtasks sharing (entity, kernel) collapse to
    # one canonical slot; dedup via a set before sorting.
    from .kernel_ids import STATE_KERNEL_ID as _SKID  # noqa: PLC0415

    _rotatable_kids = {int(_SKID.BODY_POS), int(_SKID.BODY_LIN_VEL), int(_SKID.BODY_ANG_VEL)}
    _reach_by_asset: dict[str, set[int]] = {}
    _track_by_asset: dict[str, set[int]] = {}
    for sid in range(len(state_kernel_id)):
        if state_kernel_id[sid] not in _rotatable_kids:
            continue
        off = canonical_offset_list[sid]
        if off < 0:
            continue
        asset_name = subtask_asset_cfgs[sid].name
        bucket = _reach_by_asset if is_instant[sid] else _track_by_asset
        bucket.setdefault(asset_name, set()).add(off)
    reach_rotatable_vec3_by_asset = {name: tuple(sorted(offs)) for name, offs in _reach_by_asset.items()}
    track_rotatable_vec3_by_asset = {name: tuple(sorted(offs)) for name, offs in _track_by_asset.items()}

    # Per-task active-channel mask for the flat ``command`` obs layout.
    # Built once, indexed by ``task_idx`` at resample to refresh
    # ``MultiTaskCommand._command_active``. A subtask writes ``1.0`` across
    # its ``[canonical_offset, canonical_offset + canonical_stride)`` slice,
    # offset by ``reach_canonical_width`` if it's a tracking subtask (so the
    # layout matches ``cat([command_reach, command_track], dim=-1)``). Joint
    # kernels (``canonical_offset == -1``) contribute nothing.
    flat_mask_width = max(1, reach_canonical_width + track_canonical_width)
    task_active_mask_t = torch.zeros(T, flat_mask_width, dtype=torch.float32, device=device)
    for t, row in enumerate(task_to_subtask_ids):
        for sid in row:
            canon_off = canonical_offset_list[sid]
            if canon_off < 0:
                continue
            canon_stride = canonical_stride_list[sid]
            if is_instant[sid]:
                flat_off = canon_off
            else:
                flat_off = canon_off + reach_canonical_width
            task_active_mask_t[t, flat_off : flat_off + canon_stride] = 1.0

    # Build Python-list companions. These are tiny (one int per group / per slab)
    # and avoid forcing a CPU sync inside the per-step dispatch loop.
    read_group_metric_kids_py: list[list[int]] = []
    for member_sids in read_group_member_sids:
        seen: set[int] = set()
        kids: list[int] = []
        for sid in member_sids:
            mkid = int(metric_kernel_id[sid])
            if mkid not in seen:
                seen.add(mkid)
                kids.append(mkid)
        read_group_metric_kids_py.append(sorted(kids))
    unique_activation_kids_py: list[int] = sorted({int(akid) for akid in activation_kernel_id})

    # Per-subtask gather CSR for the Warp mega-dispatch. The read-group layout
    # is group-oriented (``[M_g, slice_size]``); the mega-kernel wants one flat
    # block per subtask since each thread is a ``(env, slot)`` pair resolving
    # to one subtask.
    num_subtasks = len(state_kernel_id)
    sid_gather_indices: list[list[int]] = [[] for _ in range(num_subtasks)]
    for gid, member_sids in enumerate(read_group_member_sids):
        group_gather = read_group_gather_indices_list[gid]  # [M_g, slice_size]
        for member_pos, sid in enumerate(member_sids):
            sid_gather_indices[sid] = [int(x) for x in group_gather[member_pos].tolist()]
    gather_indices_flat_list: list[int] = []
    subtask_gather_offset_list: list[int] = []
    subtask_gather_count_list: list[int] = []
    cursor_g = 0
    for sid in range(num_subtasks):
        block = sid_gather_indices[sid]
        subtask_gather_offset_list.append(cursor_g)
        subtask_gather_count_list.append(len(block))
        gather_indices_flat_list.extend(block)
        cursor_g += len(block)

    return TaskSpec(
        task_names=task_names,
        task_subtask_ids=task_subtask_ids,
        task_subtask_valid=task_subtask_valid,
        task_slot_count=task_slot_count,
        state_stride=state_stride_tensor,
        task_slot_strides=task_slot_strides,
        task_slot_offsets=task_slot_offsets,
        task_total_stride=task_total_stride,
        state_kernel_id=torch.tensor(state_kernel_id, dtype=torch.long, device=device),
        metric_kernel_id=torch.tensor(metric_kernel_id, dtype=torch.long, device=device),
        sampler_kernel_id=torch.tensor(sampler_kernel_id, dtype=torch.long, device=device),
        sampler_kernel_param=sampler_param_t,
        activation_kernel_id=torch.tensor(activation_kernel_id, dtype=torch.long, device=device),
        activation_kernel_param=torch.tensor(activation_kernel_param, dtype=torch.float32, device=device),
        is_tracking=torch.tensor(is_tracking, dtype=torch.bool, device=device),
        is_instant=torch.tensor(is_instant, dtype=torch.bool, device=device),
        expose_in_obs=torch.tensor(expose_in_obs, dtype=torch.bool, device=device),
        subtask_asset_cfgs=subtask_asset_cfgs,
        subtask_entity_id=torch.tensor(subtask_entity_id, dtype=torch.long, device=device),
        read_group_id=torch.tensor(read_group_id_list, dtype=torch.long, device=device),
        read_group_state_kernel_id=torch.tensor(read_group_state_kernel_id_list, dtype=torch.long, device=device),
        read_group_member_sids=read_group_member_sids,
        read_group_member_asset_cfgs=read_group_member_asset_cfgs,
        subtask_member_index=torch.tensor(subtask_member_index_list, dtype=torch.long, device=device),
        unified_width=unified_width,
        slab_buffer_kinds=slab_buffer_kinds,
        slab_asset_names=slab_asset_names,
        slab_offsets=torch.tensor(slab_offsets_list, dtype=torch.long, device=device),
        slab_sizes=torch.tensor(slab_sizes_list, dtype=torch.long, device=device),
        read_group_gather_indices=[t.to(device) for t in read_group_gather_indices_list],
        gather_indices_flat=torch.tensor(gather_indices_flat_list, dtype=torch.long, device=device),
        subtask_gather_offset=torch.tensor(subtask_gather_offset_list, dtype=torch.long, device=device),
        subtask_gather_count=torch.tensor(subtask_gather_count_list, dtype=torch.long, device=device),
        reach_canonical_width=reach_canonical_width,
        track_canonical_width=track_canonical_width,
        canonical_offset=torch.tensor(canonical_offset_list, dtype=torch.long, device=device),
        canonical_stride=torch.tensor(canonical_stride_list, dtype=torch.long, device=device),
        task_active_mask=task_active_mask_t,
        reach_rotatable_vec3_by_asset=reach_rotatable_vec3_by_asset,
        track_rotatable_vec3_by_asset=track_rotatable_vec3_by_asset,
        task_has_instant=task_has_instant,
        task_kernel_target_offset=task_kernel_target_offset,
        slab_offsets_py=list(slab_offsets_list),
        slab_sizes_py=list(slab_sizes_list),
        read_group_state_kernel_id_py=list(read_group_state_kernel_id_list),
        read_group_metric_kids_py=read_group_metric_kids_py,
        unique_activation_kids_py=unique_activation_kids_py,
    )


def _compute_read_groups(
    state_kernel_id: list[int],
    subtask_asset_cfgs: list[SceneEntityCfg],
    state_stride_list: list[int],
    subtask_gather_size: list[int],
) -> tuple[list[int], list[list[int]], list[list[SceneEntityCfg]], list[int], list[int]]:
    """Assign subtasks to kernel-fused read groups.

    A read group is a bucket of subtasks that share the same state kernel AND
    the same per-subtask gather size (the actual number of floats read from
    the unified buffer). The output dimension (``state_stride``) alone is
    *not* a sufficient discriminator for kernels that reduce across K elements
    — :data:`STATE_KERNEL_ID.BODY_CONTACT_COUNT` always emits a stride-1
    scalar but its gather is ``K · 3`` floats with K varying per subtask
    (e.g. 4 feet for tripod_walk vs 1 chassis body for the undesired-contact
    safety subtask). Keying on ``(state_kid, gather_size)`` ensures every
    member in a group reads exactly the same number of floats so the batched
    ``compute_fn`` can run on a rectangular stack.

    Within a group, the dispatch:

      1. Calls the kernel's ``source_fn`` once per member (cheap tensor view)
         to read per-subtask raw data of shape ``[N, *subtask_shape]``.
      2. Stacks members into ``[M, N, *subtask_shape]``.
      3. Calls the kernel's batched ``compute_fn`` once on the stack.

    Fusion crosses assets — a COUNT_DIFF subtask on ``contact_forces`` and
    another on a different sensor with matching K land in the same group.
    The only constraint is uniform shape within the group; subtasks with
    mismatched K get split into separate groups for the same kernel.

    Returns:
      ``(read_group_id, read_group_member_sids, read_group_member_asset_cfgs,
      read_group_state_kid, subtask_member_index)``.

      - ``read_group_id[M]`` — which group each subtask belongs to.
      - ``read_group_member_sids[G]`` — list of subtask ids per group.
      - ``read_group_member_asset_cfgs[G]`` — parallel list of the members' asset_cfgs.
      - ``read_group_state_kid[G]`` — the shared state kernel id per group.
      - ``subtask_member_index[M]`` — each subtask's position (``m``) in its group's
        member list. Used at dispatch time to slice ``x_stacked[m, env, ...]``.
    """
    group_key_to_id: dict[tuple[int, int], int] = {}
    group_state_kid: list[int] = []
    group_member_sids: list[list[int]] = []
    group_member_asset_cfgs: list[list[SceneEntityCfg]] = []

    read_group_id: list[int] = []
    subtask_member_index: list[int] = []

    for sid, asset in enumerate(subtask_asset_cfgs):
        state_kid = int(state_kernel_id[sid])
        gather_size = int(subtask_gather_size[sid])
        # Group key: (kernel, gather_size). Same-kernel subtasks with matching
        # gather size can be stacked into one batched compute call regardless
        # of asset; mismatched K gets split.
        key = (state_kid, gather_size)
        gid = group_key_to_id.get(key)
        if gid is None:
            gid = len(group_state_kid)
            group_key_to_id[key] = gid
            group_state_kid.append(state_kid)
            group_member_sids.append([])
            group_member_asset_cfgs.append([])
        m = len(group_member_sids[gid])
        group_member_sids[gid].append(sid)
        group_member_asset_cfgs[gid].append(asset)
        read_group_id.append(gid)
        subtask_member_index.append(m)

    return (
        read_group_id,
        group_member_sids,
        group_member_asset_cfgs,
        group_state_kid,
        subtask_member_index,
    )


def _compute_unified_layout(
    scene: InteractiveScene,
    state_kernel_id: list[int],
    subtask_asset_cfgs: list[SceneEntityCfg],
    read_group_state_kernel_id_list: list[int],
    read_group_member_sids: list[list[int]],
    read_group_member_asset_cfgs: list[list[SceneEntityCfg]],
) -> tuple[int, list[int], list[str], list[int], list[int], list[torch.Tensor]]:
    """Compute the unified state buffer layout and per-group gather indices.

    Each unique ``(buffer_kind, asset_name)`` pair referenced by the cfg becomes
    a slab in the unified buffer. Slab size is determined from the scene's
    asset (e.g. articulation body count × per-body stride). Every subtask's
    required floats are given by an absolute index list into unified; read-
    group gather indices stack these across members for one advanced-index
    gather per group at step time.
    """
    from .impl.kernels_torch import (
        STATE_KERNEL_BUFFER_KIND,
        buffer_kind_is_body_indexed,
        buffer_kind_per_element_stride,
        state_kernel_intra_body_offset,
        state_kernel_intra_body_stride,
    )

    # -- Step 1: discover all (buffer_kind, asset_name) slabs the cfg uses.
    slab_keys: list[tuple[int, str]] = []
    slab_key_set: set[tuple[int, str]] = set()
    for sid in range(len(state_kernel_id)):
        state_kid = int(state_kernel_id[sid])
        bk = int(STATE_KERNEL_BUFFER_KIND[state_kid])
        asset_name = subtask_asset_cfgs[sid].name
        key = (bk, asset_name)
        if key not in slab_key_set:
            slab_key_set.add(key)
            slab_keys.append(key)

    # -- Step 2: determine slab sizes by looking up each asset's body/joint count.
    # For body-indexed buffers: size = num_bodies_on_asset × per_body_stride.
    # For joint-indexed buffers: size = num_joints_on_asset × 1.
    slab_offsets_list: list[int] = []
    slab_sizes_list: list[int] = []
    slab_buffer_kinds: list[int] = []
    slab_asset_names: list[str] = []
    # Also remember per-slab "num_elements" for later offset arithmetic.
    slab_num_elements: dict[tuple[int, str], int] = {}
    cursor = 0
    for bk, asset_name in slab_keys:
        # ``scene[asset_name]`` already searches articulations, rigid objects,
        # sensors and extras in one pass; no ``in scene`` guard needed (and
        # ``InteractiveScene`` doesn't implement ``__contains__`` — the legacy
        # ``in`` fallback via ``__getitem__(0)`` raises KeyError, not IndexError).
        asset = scene[asset_name]
        per_stride = buffer_kind_per_element_stride(bk)
        if buffer_kind_is_body_indexed(bk):
            num_elements = asset.num_bodies
        else:
            num_elements = asset.num_joints
        slab_size = num_elements * per_stride
        slab_buffer_kinds.append(bk)
        slab_asset_names.append(asset_name)
        slab_offsets_list.append(cursor)
        slab_sizes_list.append(slab_size)
        slab_num_elements[(bk, asset_name)] = num_elements
        cursor += slab_size
    unified_width = cursor
    slab_key_to_idx: dict[tuple[int, str], int] = {k: i for i, k in enumerate(slab_keys)}

    # -- Step 3: per subtask, compute absolute read indices into unified.
    def subtask_read_indices(sid: int) -> list[int]:
        state_kid = int(state_kernel_id[sid])
        bk = int(STATE_KERNEL_BUFFER_KIND[state_kid])
        asset_cfg = subtask_asset_cfgs[sid]
        key = (bk, asset_cfg.name)
        slab_start = slab_offsets_list[slab_key_to_idx[key]]
        per_stride = buffer_kind_per_element_stride(bk)
        intra_off = state_kernel_intra_body_offset(state_kid)
        intra_w = state_kernel_intra_body_stride(state_kid)
        # Resolve body_ids / joint_ids to a concrete list.
        if buffer_kind_is_body_indexed(bk):
            ids = asset_cfg.body_ids
        else:
            ids = asset_cfg.joint_ids
        if isinstance(ids, slice):
            num_elem = slab_num_elements[key]
            id_list = list(range(num_elem))[ids]
        else:
            id_list = list(ids)
        indices: list[int] = []
        for eid in id_list:
            base = slab_start + int(eid) * per_stride + intra_off
            indices.extend(range(base, base + intra_w))
        return indices

    # -- Step 4: per read group, stack members' indices.
    read_group_gather_indices_list: list[torch.Tensor] = []
    for gid in range(len(read_group_state_kernel_id_list)):
        member_index_lists = [subtask_read_indices(sid) for sid in read_group_member_sids[gid]]
        # Within a group (same state_kid + state_stride), slice_size is uniform.
        slice_sizes = {len(x) for x in member_index_lists}
        if len(slice_sizes) > 1:
            raise ValueError(
                f"Read group {gid} members have inconsistent slice sizes {slice_sizes} — "
                "this should not happen because state_stride is the group key."
            )
        read_group_gather_indices_list.append(torch.tensor(member_index_lists, dtype=torch.long))

    return (
        unified_width,
        slab_buffer_kinds,
        slab_asset_names,
        slab_offsets_list,
        slab_sizes_list,
        read_group_gather_indices_list,
    )


def _compute_canonical_layout(
    num_entities: int,
    subtask_entity_id: list[int],
    state_kernel_id: list[int],
    state_stride_list: list[int],
    subtask_asset_cfgs: list[SceneEntityCfg],
    is_instant: list[bool],
    expose_in_obs: list[bool],
) -> tuple[int, int, list[int], list[int]]:
    """Assemble two independent canonical obs layouts — one for reach (instant)
    subtasks, one for track (tracking) subtasks.

    Each layout builds a per-entity block of only the state-kernel slices its
    subtasks (of the relevant type) actually reference, in canonical order
    (POS → POS_Z → QUAT → LIN_VEL → ANG_VEL → CONTACT → CONTACT_COUNT →
    CONTACT_COUNT_DIFF). POS and POS_Z now get **disjoint** channels (no z-slot
    aliasing) so a reach POS subtask and a track POS_Z subtask on the same entity
    can coexist without overwriting each other.

    Each subtask gets a single ``canonical_offset`` into **its own layout**
    (reach if instant, track if tracking). Downstream dispatch picks the right
    output buffer based on ``is_instant[sid]``.

    Joint kernels (no canonical projection) emit ``(-1, -1)``. Tracking
    subtasks declared with ``expose_in_obs=False`` (soft-safety constraints)
    also emit ``(-1, -1)``: they contribute to ``G``'s quality factor but
    never appear in the policy obs — the policy learns to satisfy them
    implicitly via the reward gradient.

    Returns ``(reach_width, track_width, per_subtask_offset, per_subtask_stride)``.
    """
    from .kernel_ids import STATE_KERNEL_ID as SKID

    # Canonical order: walk kernel ids in a fixed enumeration so every entity
    # lays them out in the same order. Stride comes from the subtasks
    # themselves (``state_stride_list``), not from a hardcoded per-kernel
    # width — this lets variable-K kernels (notably :data:`BODY_CONTACT`,
    # whose width = number of bodies in the subtask's ``asset_cfg``) coexist
    # in the layout without special-casing. Stride consistency across
    # subtasks sharing the same ``(state_kid, entity)`` is enforced upstream
    # by :func:`_assert_stride_consistent_within_class`, so any per-entity
    # (kid, stride) pair is well-defined regardless of which subtask we read.
    _CANONICAL_ORDER = (
        int(SKID.BODY_POS),
        int(SKID.BODY_POS_Z),
        int(SKID.BODY_QUAT),
        int(SKID.BODY_LIN_VEL),
        int(SKID.BODY_ANG_VEL),
        int(SKID.BODY_CONTACT),
        int(SKID.BODY_CONTACT_COUNT),
        int(SKID.BODY_CONTACT_COUNT_DIFF),
    )

    def build_entity_layout(kernels_with_stride: dict[int, int]) -> tuple[int, dict[int, tuple[int, int]]]:
        """Compute ``(block_width, per_kernel_(relative_offset, stride))`` for one entity.

        ``kernels_with_stride`` maps each kernel id present on this entity to
        its canonical stride. Strides are taken from the subtask's
        ``state_stride`` — uniform (e.g. POS=3, QUAT=4) for most kernels,
        variable (= K) for BODY_CONTACT.
        """
        layout: dict[int, tuple[int, int]] = {}
        cursor = 0
        for kid in _CANONICAL_ORDER:
            if kid in kernels_with_stride:
                stride = kernels_with_stride[kid]
                layout[kid] = (cursor, stride)
                cursor += stride
        return cursor, layout

    def build_split_layout(use_instant: bool) -> tuple[int, list[int], list[dict[int, tuple[int, int]]]]:
        """Assemble the canonical layout for one subtask-type slice."""
        # ``{entity_id: {kernel_id: stride}}`` — stride comes from whichever
        # subtask on (entity, kernel) we see first; stride consistency is
        # already gated upstream so any subtask on the same pair agrees.
        entity_to_kernels: dict[int, dict[int, int]] = {}
        for sid in range(len(state_kernel_id)):
            # Tracking subtasks declared expose_in_obs=False (soft-safety
            # constraints) never enter the policy obs — see docstring.
            if not bool(expose_in_obs[sid]):
                continue
            if bool(is_instant[sid]) != use_instant:
                continue
            ent = int(subtask_entity_id[sid])
            kid = int(state_kernel_id[sid])
            stride = int(state_stride_list[sid])
            entity_to_kernels.setdefault(ent, {})[kid] = stride

        entity_base_offset: list[int] = []
        entity_layouts: list[dict[int, tuple[int, int]]] = []
        cursor = 0
        for ent_id in range(num_entities):
            width, layout = build_entity_layout(entity_to_kernels.get(ent_id, {}))
            entity_base_offset.append(cursor)
            entity_layouts.append(layout)
            cursor += width
        return cursor, entity_base_offset, entity_layouts

    reach_width, reach_base_offsets, reach_layouts = build_split_layout(use_instant=True)
    track_width, track_base_offsets, track_layouts = build_split_layout(use_instant=False)

    canonical_offset_list: list[int] = []
    canonical_stride_list: list[int] = []
    for sid in range(len(state_kernel_id)):
        ent = int(subtask_entity_id[sid])
        kid = int(state_kernel_id[sid])
        if not bool(expose_in_obs[sid]):
            # Soft-safety subtasks have no canonical projection (no policy obs).
            canonical_offset_list.append(-1)
            canonical_stride_list.append(-1)
            continue
        if bool(is_instant[sid]):
            base_offsets, layouts = reach_base_offsets, reach_layouts
        else:
            base_offsets, layouts = track_base_offsets, track_layouts
        if kid in layouts[ent]:
            rel_off, stride = layouts[ent][kid]
            if stride != state_stride_list[sid]:
                raise ValueError(
                    f"MultiTaskCfg: canonical layout stride mismatch for subtask {sid} "
                    f"(state_kernel={kid}, asset={subtask_asset_cfgs[sid].name}): "
                    f"canonical stride {stride} != state_stride {state_stride_list[sid]}. "
                    f"The sampler's output dim must match the state kernel's projection."
                )
            canonical_offset_list.append(base_offsets[ent] + rel_off)
            canonical_stride_list.append(stride)
        else:
            canonical_offset_list.append(-1)
            canonical_stride_list.append(-1)

    return reach_width, track_width, canonical_offset_list, canonical_stride_list


def _assert_stride_consistent_within_class(
    state_kernel_id: list[int],
    subtask_entity_id: list[int],
    state_stride_list: list[int],
    subtask_asset_cfgs: list[SceneEntityCfg],
) -> None:
    """Enforce: within each ``(state_kid, entity)`` class, all subtasks share one stride.

    Raises ``ValueError`` with the offending class identifier and the conflicting
    strides. See :func:`build_spec` for why this matters.
    """
    class_stride: dict[tuple[int, int], tuple[int, str]] = {}
    for sid, (skid, ent, stride) in enumerate(zip(state_kernel_id, subtask_entity_id, state_stride_list)):
        key = (int(skid), int(ent))
        prev = class_stride.get(key)
        if prev is None:
            descriptor = f"state_kernel={int(skid)}, entity={int(ent)}, asset={subtask_asset_cfgs[sid].name}"
            class_stride[key] = (int(stride), descriptor)
        elif prev[0] != int(stride):
            raise ValueError(
                f"MultiTaskCfg: state_stride inconsistency within equivalence class "
                f"{prev[1]}: existing subtasks have stride {prev[0]}, new subtask "
                f"{sid} has stride {int(stride)}. All subtasks sharing a state kernel "
                f"and entity must emit state of the same dimension — the per-step "
                f"dispatch uses one stride for the whole group. Fix the cfg so the "
                f"sampler's output dim matches the state kernel's."
            )
