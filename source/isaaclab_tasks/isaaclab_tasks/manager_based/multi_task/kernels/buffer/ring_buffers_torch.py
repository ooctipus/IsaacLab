# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


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
