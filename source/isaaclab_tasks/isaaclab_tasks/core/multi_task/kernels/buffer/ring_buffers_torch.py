# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


def ring_append_bool_count_rate(
    data: torch.Tensor,
    stream_ids: torch.Tensor,
    values: torch.Tensor,
    ptr: torch.Tensor,
    size: torch.Tensor,
    true_count: torch.Tensor,
    rate: torch.Tensor,
) -> None:
    """Append bool values and update true-count and rate tensors."""
    if stream_ids.numel() == 0:
        return

    capacity = data.shape[1]
    if bool(stream_ids[0] == stream_ids[-1]) and bool((stream_ids == stream_ids[0]).all()):
        stream_id = stream_ids[0]
        num_updates = values.numel()
        write_count = min(num_updates, capacity)
        write_offsets = torch.arange(write_count, dtype=torch.long, device=data.device)
        write_start = (ptr[stream_id].long() + num_updates - write_count) % capacity
        write_cols = (write_start + write_offsets) % capacity
        write_values = values[-write_count:]
        if num_updates >= capacity:
            new_true_count = write_values.sum(dtype=true_count.dtype)
            new_size = torch.full((), capacity, device=data.device, dtype=torch.long)
        else:
            offsets = torch.arange(num_updates, dtype=torch.long, device=data.device)
            cols = (ptr[stream_id].long() + offsets) % capacity
            overwrite_mask = offsets >= capacity - size[stream_id].long()
            overwritten = data[stream_id, cols[overwrite_mask]].sum(dtype=true_count.dtype)
            new_true_count = true_count[stream_id] - overwritten + values.sum(dtype=true_count.dtype)
            new_size = (size[stream_id].long() + num_updates).clamp(max=capacity)

        data[stream_id, write_cols] = write_values
        ptr[stream_id] = ((ptr[stream_id].long() + num_updates) % capacity).to(dtype=ptr.dtype)
        size[stream_id] = new_size.to(dtype=size.dtype)
        true_count[stream_id] = new_true_count
        rate[stream_id] = new_true_count.to(rate.dtype) / new_size.clamp(min=1).to(rate.dtype)
        return

    unique_ids, inv, counts = torch.unique(stream_ids, return_inverse=True, return_counts=True)
    if unique_ids.numel() == stream_ids.numel():
        ptrs = ptr[stream_ids].long()
        full = size[stream_ids] == capacity
        overwritten = torch.where(
            full,
            data[stream_ids, ptrs].to(dtype=true_count.dtype),
            torch.zeros_like(true_count[stream_ids]),
        )
        new_true_counts = true_count[stream_ids] - overwritten + values.to(dtype=true_count.dtype)
        data[stream_ids, ptrs] = values
        ptr[stream_ids] = ((ptrs + 1) % capacity).to(dtype=ptr.dtype)
        size[stream_ids] = (size[stream_ids] + 1).clamp(max=capacity)
        true_count[stream_ids] = new_true_counts
        rate[stream_ids] = new_true_counts.to(rate.dtype) / size[stream_ids].clamp(min=1)
        return

    order = torch.argsort(inv, stable=True)
    sorted_ids = stream_ids[order]
    sorted_values = values[order]
    group_starts = counts.cumsum(0) - counts
    local_rank = torch.arange(stream_ids.numel(), device=data.device) - torch.repeat_interleave(group_starts, counts)
    inv_sorted = inv[order]
    counts_sorted = counts[inv_sorted]
    true_added = torch.zeros(unique_ids.shape, device=data.device, dtype=true_count.dtype)
    true_added.scatter_add_(0, inv, values.to(dtype=true_count.dtype))

    if bool((counts < capacity).all()):
        overwrite_start = capacity - size[sorted_ids].long()
        overwrite_mask = local_rank >= overwrite_start
        overwritten = torch.zeros_like(true_added)
        overwrite_ids = sorted_ids[overwrite_mask]
        overwrite_cols = (ptr[overwrite_ids].long() + local_rank[overwrite_mask]) % capacity
        overwritten.scatter_add_(
            0,
            inv_sorted[overwrite_mask],
            data[overwrite_ids, overwrite_cols].to(dtype=true_count.dtype),
        )

        cols = (ptr[sorted_ids].long() + local_rank) % capacity
        data[sorted_ids, cols] = sorted_values

        new_size = (size[unique_ids].long() + counts).clamp(max=capacity)
        new_true_counts = true_count[unique_ids] - overwritten + true_added
        ptr[unique_ids] = ((ptr[unique_ids].long() + counts) % capacity).to(dtype=ptr.dtype)
        size[unique_ids] = new_size.to(dtype=size.dtype)
        true_count[unique_ids] = new_true_counts
        rate[unique_ids] = new_true_counts.to(rate.dtype) / new_size.clamp(min=1).to(rate.dtype)
        return

    keep_start = (counts - capacity).clamp(min=0)
    keep = local_rank >= torch.repeat_interleave(keep_start, counts)

    true_kept = torch.zeros_like(true_added)
    true_kept.scatter_add_(0, inv_sorted[keep], sorted_values[keep].to(dtype=true_count.dtype))

    old_size = size[unique_ids].long()
    overwrite_start = capacity - size[sorted_ids].long()
    overwrite_mask = (counts_sorted < capacity) & (local_rank >= overwrite_start)
    overwritten = torch.zeros_like(true_added)
    overwrite_ids = sorted_ids[overwrite_mask]
    overwrite_cols = (ptr[overwrite_ids].long() + local_rank[overwrite_mask]) % capacity
    overwritten.scatter_add_(
        0,
        inv_sorted[overwrite_mask],
        data[overwrite_ids, overwrite_cols].to(dtype=true_count.dtype),
    )

    kept_ids = sorted_ids[keep]
    kept_cols = (ptr[kept_ids].long() + local_rank[keep]) % capacity
    data[kept_ids, kept_cols] = sorted_values[keep]

    replace = counts >= capacity
    new_true_counts = torch.where(replace, true_kept, true_count[unique_ids] - overwritten + true_added)
    new_size = (old_size + counts).clamp(max=capacity)
    ptr[unique_ids] = ((ptr[unique_ids].long() + counts) % capacity).to(dtype=ptr.dtype)
    size[unique_ids] = new_size.to(dtype=size.dtype)
    true_count[unique_ids] = new_true_counts
    rate[unique_ids] = new_true_counts.to(rate.dtype) / new_size.clamp(min=1).to(rate.dtype)


def ring_append_values_changed(
    data: torch.Tensor,
    stream_ids: torch.Tensor,
    values: torch.Tensor,
    ptr: torch.Tensor,
    size: torch.Tensor,
    changed_ids: torch.Tensor,
    num_changed: torch.Tensor,
    num_updates: int,
    capacity: int,
) -> None:
    """Append generic values into ring buffers and report touched stream ids."""
    if num_updates == 0:
        num_changed.zero_()
        return

    if bool(stream_ids.min() == stream_ids.max()):
        stream_id = stream_ids[0]
        write_count = min(num_updates, capacity)
        start = int((ptr[stream_id].item() + num_updates - write_count) % capacity)
        cols = torch.arange(start, start + write_count, dtype=torch.long, device=data.device) % capacity
        data[stream_id, cols] = values[-write_count:].to(device=data.device, dtype=data.dtype)
        ptr[stream_id] = (int(ptr[stream_id].item()) + num_updates) % capacity
        size[stream_id] = min(int(size[stream_id].item()) + num_updates, capacity)
        changed_ids[:1] = stream_id.unsqueeze(0).to(device=changed_ids.device, dtype=changed_ids.dtype)
        num_changed[0] = 1
        return

    unique_ids, inv, counts = torch.unique(stream_ids, return_inverse=True, return_counts=True)
    if bool((counts == 1).all()):
        ptrs = ptr[stream_ids].long()
        data[stream_ids, ptrs] = values.to(device=data.device, dtype=data.dtype)
        ptr[stream_ids] = ((ptrs + 1) % capacity).to(dtype=ptr.dtype)
        size[stream_ids] = (size[stream_ids] + 1).clamp(max=capacity)
        changed_ids[: stream_ids.numel()] = stream_ids.to(device=changed_ids.device, dtype=changed_ids.dtype)
        num_changed[0] = stream_ids.numel()
        return

    order = torch.argsort(inv, stable=True)
    sorted_ids = stream_ids[order]
    sorted_values = values[order].to(device=data.device, dtype=data.dtype)
    group_starts = counts.cumsum(0) - counts
    local_rank = torch.arange(num_updates, device=data.device) - torch.repeat_interleave(group_starts, counts)
    keep_start = (counts - capacity).clamp(min=0)
    keep = local_rank >= torch.repeat_interleave(keep_start, counts)
    kept_ids = sorted_ids[keep]
    kept_rank = local_rank[keep]
    cols = (ptr[kept_ids].long() + kept_rank) % capacity
    data.index_put_((kept_ids, cols), sorted_values[keep])
    ptr[unique_ids] = ((ptr[unique_ids].to(torch.int64) + counts) % capacity).to(dtype=ptr.dtype)
    size[unique_ids] = (size[unique_ids].to(torch.int64) + counts).clamp(max=capacity).to(dtype=size.dtype)
    changed_ids[: unique_ids.numel()] = unique_ids.to(device=changed_ids.device, dtype=changed_ids.dtype)
    num_changed[0] = unique_ids.numel()
