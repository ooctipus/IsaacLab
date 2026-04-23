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

from dataclasses import dataclass
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


def _subtask_signature(subtask_cfg, scene: InteractiveScene) -> tuple:
    """Dedup signature for a subtask cfg — every field that changes behavior is here."""
    subtask_cfg.asset_cfg.resolve(scene)
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
    of tasks, ``k_max`` = max active slots across tasks.
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
    """``[M]`` bool."""
    is_instant: torch.Tensor
    """``[M]`` bool."""

    subtask_asset_cfgs: list[SceneEntityCfg]
    subtask_entity_id: torch.Tensor
    """``[M]`` int — dedup key for ``(asset.name, body_ids, joint_ids)``."""


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
    subtask_asset_cfgs: list[SceneEntityCfg] = []
    state_stride_list: list[int] = []

    entity_sig_to_id: dict[tuple, int] = {}
    subtask_entity_id: list[int] = []
    task_to_subtask_ids: list[list[int]] = []
    p_max = 0

    for task_name in task_names:
        row: list[int] = []
        for subtask_cfg in cfg.tasks[task_name]:
            sig = _subtask_signature(subtask_cfg, scene)
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

                is_tracking.append(isinstance(subtask_cfg, _MultiTaskCfg.TrackingTaskCfg))
                is_instant.append(isinstance(subtask_cfg, _MultiTaskCfg.InstantaneousTaskCfg))

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

    # Per-task slot offsets/strides + total_stride (cumulative sum along slot dim).
    state_stride_tensor = torch.tensor(state_stride_list, dtype=torch.long, device=device)
    task_slot_offsets = torch.zeros((T, max(k_max, 1)), dtype=torch.long, device=device)
    task_slot_strides = torch.zeros((T, max(k_max, 1)), dtype=torch.long, device=device)
    task_total_stride = torch.zeros(T, dtype=torch.long, device=device)
    for t, row in enumerate(task_to_subtask_ids):
        offset = 0
        for slot, sid in enumerate(row):
            stride = int(state_stride_tensor[sid].item())
            task_slot_offsets[t, slot] = offset
            task_slot_strides[t, slot] = stride
            offset += stride
        task_total_stride[t] = offset

    # Pad sampler params to rectangular [M, Pmax] (even length for [min, range] pairs).
    if p_max % 2 == 1:
        p_max += 1
    sampler_param_t = torch.zeros((len(sampler_kernel_param_rows), max(p_max, 2)), dtype=torch.float32, device=device)
    for j, prow in enumerate(sampler_kernel_param_rows):
        sampler_param_t[j, : prow.numel()] = prow

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
        subtask_asset_cfgs=subtask_asset_cfgs,
        subtask_entity_id=torch.tensor(subtask_entity_id, dtype=torch.long, device=device),
    )


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
