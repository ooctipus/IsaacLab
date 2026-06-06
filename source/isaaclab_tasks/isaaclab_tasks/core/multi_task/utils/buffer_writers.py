# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp

from .impl.buffer_writers_torch import FIFOBufferWriterTorch
from .impl.buffer_writers_warp import FIFOBufferWriterWarp


class AppendBufferWriter:
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


class FIFOBufferWriter:
    """FIFO writer over caller-owned per-stream ring buffers.

    The caller owns all public state tensors. ``changed_ids`` fixes the maximum
    number of raw rows accepted by one :meth:`add` call and is used to size Warp
    sort/group scratch at construction time. ``data`` and ``new_data`` are
    expected to be contiguous tensors with matching payload shape.
    """

    def __init__(
        self,
        start_ptr: torch.Tensor,
        size: torch.Tensor,
        changed_ids: torch.Tensor,
        num_changed: torch.Tensor,
        warp: bool = False,
    ):
        self._validate_state(start_ptr, size, changed_ids, num_changed)
        self._warp = warp
        if warp:
            wp.init()
            self._start_ptr = start_ptr
            self._size = size
            self._changed_ids = changed_ids
            self._num_changed = num_changed
            self._impl = FIFOBufferWriterWarp(
                device=str(start_ptr.device),
                start_ptr=wp.from_torch(self._start_ptr, dtype=wp.int32),
                size=wp.from_torch(self._size, dtype=wp.int32),
                changed_ids=wp.from_torch(self._changed_ids, dtype=wp.int64),
                num_changed=wp.from_torch(self._num_changed, dtype=wp.int32),
                max_updates=int(self._changed_ids.numel()),
            )
        else:
            self._impl = FIFOBufferWriterTorch(
                start_ptr=start_ptr,
                size=size,
                changed_ids=changed_ids,
                num_changed=num_changed,
            )

    def _validate_state(
        self,
        start_ptr: torch.Tensor,
        size: torch.Tensor,
        changed_ids: torch.Tensor,
        num_changed: torch.Tensor,
    ) -> None:
        if start_ptr.dtype != torch.int32:
            raise TypeError(f"start_ptr must have dtype torch.int32, got {start_ptr.dtype}.")
        if size.dtype != torch.int32:
            raise TypeError(f"size must have dtype torch.int32, got {size.dtype}.")
        if changed_ids.dtype != torch.int64:
            raise TypeError(f"changed_ids must have dtype torch.int64, got {changed_ids.dtype}.")
        if num_changed.dtype != torch.int32:
            raise TypeError(f"num_changed must have dtype torch.int32, got {num_changed.dtype}.")
        if start_ptr.shape != size.shape:
            raise ValueError(f"start_ptr and size must have matching shape, got {start_ptr.shape} and {size.shape}.")
        if changed_ids.ndim != 1:
            raise ValueError(f"changed_ids must be a 1D tensor, got shape {changed_ids.shape}.")
        if tuple(num_changed.shape) != (1,):
            raise ValueError(f"num_changed must have shape (1,), got {tuple(num_changed.shape)}.")
        if (
            size.device != start_ptr.device
            or changed_ids.device != start_ptr.device
            or num_changed.device != start_ptr.device
        ):
            raise ValueError("start_ptr, size, changed_ids, and num_changed must be on the same device.")

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
        num_updates = stream_ids.numel()
        if num_updates == 0:
            self.num_changed.zero_()
            return

        capacity = data.shape[1]
        item_bytes = (new_data.numel() // new_data.shape[0]) * new_data.element_size()
        if self._warp:
            data = data.view(data.shape[0], capacity, -1).view(torch.uint8).view(data.shape[0], capacity * item_bytes)
            new_data = new_data.view(new_data.shape[0], -1).view(torch.uint8).view(new_data.shape[0], item_bytes)
            data = wp.from_torch(data, dtype=wp.uint8)
            stream_ids = wp.from_torch(stream_ids, dtype=wp.int64)
            new_data = wp.from_torch(new_data, dtype=wp.uint8)

        self._impl.add(
            data,
            stream_ids,
            new_data,
            num_updates,
            capacity,
            item_bytes,
        )
