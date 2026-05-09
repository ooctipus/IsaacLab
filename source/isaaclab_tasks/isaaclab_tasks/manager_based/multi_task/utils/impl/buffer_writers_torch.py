# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


class FIFOBufferWriterTorch:
    """Manage circular writes into caller-owned per-slot buffers."""

    def __init__(
        self,
        num_streams: int | None = None,
        device: str | torch.device = "cpu",
        start_ptr: torch.Tensor | None = None,
        size: torch.Tensor | None = None,
        max_updates: int | None = None,
    ):
        if start_ptr is None:
            if num_streams is None:
                raise ValueError("num_streams is required when start_ptr is not provided.")
            start_ptr = torch.zeros(num_streams, device=device, dtype=torch.int32)
        elif num_streams is None:
            num_streams = int(start_ptr.shape[0])
        if size is None:
            size = torch.zeros_like(start_ptr, dtype=torch.int32)
        self.start_ptr = start_ptr
        self.size = size
        max_update_capacity = int(max_updates) if max_updates is not None else int(num_streams)
        self.changed_ids = torch.empty(max_update_capacity, device=start_ptr.device, dtype=torch.int64)
        self.num_changed = torch.zeros(1, device=start_ptr.device, dtype=torch.int32)

    def _record_changed(self, stream_ids: torch.Tensor) -> None:
        n = int(stream_ids.numel())
        self.changed_ids[:n] = stream_ids.to(device=self.changed_ids.device, dtype=self.changed_ids.dtype)
        self.num_changed[0] = n

    def add(self, data: torch.Tensor, stream_ids: torch.Tensor, new_data: torch.Tensor) -> None:
        """Append values into ``data`` and update changed-stream state."""
        if stream_ids.numel() == 0:
            self.num_changed.zero_()
            return

        capacity = data.shape[1]
        if bool(stream_ids.min() == stream_ids.max()):
            stream_id = stream_ids[0]
            count = stream_ids.numel()
            write_count = min(count, capacity)
            start = int((self.start_ptr[stream_id].item() + count - write_count) % capacity)
            cols = torch.arange(start, start + write_count, dtype=torch.long, device=data.device) % capacity
            data[stream_id, cols] = new_data[-write_count:].to(device=data.device, dtype=data.dtype)
            self.start_ptr[stream_id] = (int(self.start_ptr[stream_id].item()) + count) % capacity
            self.size[stream_id] = min(int(self.size[stream_id].item()) + count, capacity)
            self._record_changed(stream_id.unsqueeze(0))
            return

        unique_ids, inv, counts = torch.unique(stream_ids, return_inverse=True, return_counts=True)
        if bool((counts == 1).all()):
            ptrs = self.start_ptr[stream_ids].long()
            data[stream_ids, ptrs] = new_data.to(device=data.device, dtype=data.dtype)
            self.start_ptr[stream_ids] = ((ptrs + 1) % capacity).to(dtype=self.start_ptr.dtype)
            self.size[stream_ids] = (self.size[stream_ids] + 1).clamp(max=capacity)
            self._record_changed(stream_ids)
            return

        order = torch.argsort(inv, stable=True)
        sorted_ids = stream_ids[order]
        sorted_values = new_data[order].to(device=data.device, dtype=data.dtype)
        group_starts = counts.cumsum(0) - counts
        local_rank = torch.arange(stream_ids.numel(), device=data.device) - torch.repeat_interleave(
            group_starts, counts
        )
        keep_start = (counts - capacity).clamp(min=0)
        keep = local_rank >= torch.repeat_interleave(keep_start, counts)
        kept_ids = sorted_ids[keep]
        kept_rank = local_rank[keep]
        cols = (self.start_ptr[kept_ids].long() + kept_rank) % capacity
        data.index_put_((kept_ids, cols), sorted_values[keep])
        self.start_ptr[unique_ids] = ((self.start_ptr[unique_ids].to(torch.int64) + counts) % capacity).to(
            dtype=self.start_ptr.dtype
        )
        self.size[unique_ids] = (
            (self.size[unique_ids].to(torch.int64) + counts).clamp(max=capacity).to(dtype=self.size.dtype)
        )
        self._record_changed(unique_ids)
