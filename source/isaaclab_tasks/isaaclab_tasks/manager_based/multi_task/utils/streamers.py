# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp

from .impl.streamers_torch import FIFOStreamerTorch
from .impl.streamers_warp import FIFOStreamerWarp


class Streamer:
    """Manage append-only writes into caller-owned data."""

    def __init__(self, start_ptr: int = 0):
        self.start_ptr = start_ptr

    def add(self, data: torch.Tensor, new_data: torch.Tensor) -> None:
        """Append ``new_data`` into ``data`` until the buffer is full."""
        data_end = min(data.shape[0], self.start_ptr + new_data.shape[0])
        data[self.start_ptr : data_end] = new_data[: data_end - self.start_ptr]
        self.start_ptr = data_end

    def full(self, data: torch.Tensor) -> bool:
        """Return whether ``data`` is full."""
        return self.start_ptr >= data.shape[0]


class FIFOStreamer:
    """FIFO streamer over caller-owned per-stream ring buffers.

    ``max_updates`` is the maximum number of raw rows passed to one
    :meth:`add` call. Warp mode uses it to allocate fixed sort/group scratch at
    construction time, so the hot path can be CUDA-graph captured without
    allocating. ``data`` and ``new_data`` are expected to be contiguous tensors
    with matching payload shape.
    """

    def __init__(
        self,
        num_streams: int | None = None,
        device: str | torch.device = "cpu",
        start_ptr: torch.Tensor | None = None,
        size: torch.Tensor | None = None,
        max_updates: int | None = None,
        warp: bool = False,
    ):
        self._warp = warp
        if warp:
            wp.init()
            if start_ptr is None:
                if num_streams is None:
                    raise ValueError("num_streams is required when start_ptr is not provided.")
                start_ptr = torch.zeros(num_streams, device=device, dtype=torch.int32)
            elif num_streams is None:
                num_streams = int(start_ptr.shape[0])
            if size is None:
                size = torch.zeros_like(start_ptr, dtype=torch.int32)
            max_update_capacity = int(max_updates) if max_updates is not None else int(num_streams)
            self._start_ptr = start_ptr
            self._size = size
            self._changed_ids = torch.empty(max_update_capacity, device=start_ptr.device, dtype=torch.int64)
            self._num_changed = torch.zeros(1, device=start_ptr.device, dtype=torch.int32)
            self._impl = FIFOStreamerWarp(
                device=str(start_ptr.device),
                start_ptr=wp.from_torch(self._start_ptr, dtype=wp.int32),
                size=wp.from_torch(self._size, dtype=wp.int32),
                changed_ids=wp.from_torch(self._changed_ids, dtype=wp.int64),
                num_changed=wp.from_torch(self._num_changed, dtype=wp.int32),
                max_updates=max_update_capacity,
            )
        else:
            self._impl = FIFOStreamerTorch(
                num_streams=num_streams,
                device=device,
                start_ptr=start_ptr,
                size=size,
                max_updates=max_updates,
            )

    @property
    def start_ptr(self) -> torch.Tensor:
        return self._start_ptr if self._warp else self._impl.start_ptr

    @start_ptr.setter
    def start_ptr(self, value: torch.Tensor) -> None:
        if self._warp:
            self._start_ptr = value
            self._impl.start_ptr = wp.from_torch(value, dtype=wp.int32)
        else:
            self._impl.start_ptr = value

    @property
    def size(self) -> torch.Tensor:
        return self._size if self._warp else self._impl.size

    @size.setter
    def size(self, value: torch.Tensor) -> None:
        if self._warp:
            self._size = value
            self._impl.size = wp.from_torch(value, dtype=wp.int32)
        else:
            self._impl.size = value

    @property
    def changed_ids(self) -> torch.Tensor:
        return self._changed_ids if self._warp else self._impl.changed_ids

    @property
    def num_changed(self) -> torch.Tensor:
        return self._num_changed if self._warp else self._impl.num_changed

    def add(self, data: torch.Tensor, stream_ids: torch.Tensor, new_data: torch.Tensor) -> None:
        """Append values into ``data`` and update changed-stream state."""
        if not self._warp:
            self._impl.add(data, stream_ids, new_data)
            return
        if stream_ids.numel() == 0:
            self._num_changed.zero_()
            return
        item_bytes = (new_data.numel() // new_data.shape[0]) * new_data.element_size()
        data_bytes = (
            data.view(data.shape[0], data.shape[1], -1)
            .view(torch.uint8)
            .view(data.shape[0], data.shape[1] * item_bytes)
        )
        new_data_bytes = new_data.view(new_data.shape[0], -1).view(torch.uint8).view(new_data.shape[0], item_bytes)
        self._impl.add(
            wp.from_torch(data_bytes, dtype=wp.uint8),
            wp.from_torch(stream_ids, dtype=wp.int64),
            wp.from_torch(new_data_bytes, dtype=wp.uint8),
            stream_ids.numel(),
            data.shape[1],
            item_bytes,
        )
