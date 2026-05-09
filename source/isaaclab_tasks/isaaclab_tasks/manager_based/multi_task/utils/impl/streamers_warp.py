# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp
import warp.utils as wpu


@wp.func
def _fifo_col(start: int, offset: int, capacity: int) -> int:
    return (start + offset) % capacity


@wp.kernel
def _fifo_prepare_sort_kernel(
    stream_ids: wp.array(dtype=wp.int64),
    sort_keys: wp.array(dtype=wp.int64),
    sort_indices: wp.array(dtype=wp.int32),
    num_changed: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if i == 0:
        num_changed[0] = 0
    sort_keys[i] = stream_ids[i]
    sort_indices[i] = i


@wp.kernel
def _fifo_write_sorted_starts_kernel(
    data: wp.array2d(dtype=wp.uint8),
    sort_keys: wp.array(dtype=wp.int64),
    sort_indices: wp.array(dtype=wp.int32),
    values: wp.array2d(dtype=wp.uint8),
    start_ptr: wp.array(dtype=wp.int32),
    size: wp.array(dtype=wp.int32),
    changed_ids: wp.array(dtype=wp.int64),
    num_changed: wp.array(dtype=wp.int32),
    count: int,
    capacity: int,
    item_bytes: int,
):
    i = wp.tid()
    stream_id = int(sort_keys[i])
    if i > 0 and int(sort_keys[i - 1]) == stream_id:
        return

    lo = i + 1
    hi = count
    while lo < hi:
        mid = (lo + hi) // 2
        if int(sort_keys[mid]) == stream_id:
            lo = mid + 1
        else:
            hi = mid
    group_end = lo

    group_count = group_end - i
    write_count = group_count
    if write_count > capacity:
        write_count = capacity

    old_ptr = int(start_ptr[stream_id])
    write_start = _fifo_col(old_ptr, group_count - write_count, capacity)
    source_start = group_end - write_count
    for j in range(write_count):
        src = int(sort_indices[source_start + j])
        dst = _fifo_col(write_start, j, capacity) * item_bytes
        for b in range(item_bytes):
            data[stream_id, dst + b] = values[src, b]

    start_ptr[stream_id] = _fifo_col(old_ptr, group_count, capacity)
    new_size = int(size[stream_id]) + group_count
    if new_size > capacity:
        new_size = capacity
    size[stream_id] = new_size
    changed_slot = wp.atomic_add(num_changed, 0, 1)
    changed_ids[changed_slot] = wp.int64(stream_id)


class FIFOStreamerWarp:
    """Manage circular writes with Warp arrays and kernels.

    ``max_updates`` is the raw event capacity for one :meth:`add` call.
    ``radix_sort_pairs`` needs double-width key/value scratch. Grouping is
    fused into the post-sort write kernel to avoid scan scratch and graph-time
    allocations.
    """

    def __init__(
        self,
        device,
        start_ptr: wp.array(dtype=wp.int32),
        size: wp.array(dtype=wp.int32),
        changed_ids: wp.array(dtype=wp.int64),
        num_changed: wp.array(dtype=wp.int32),
        max_updates: int,
    ):
        wp.init()
        self.device = device
        self.start_ptr = start_ptr
        self.size = size
        self.changed_ids = changed_ids
        self.num_changed = num_changed
        self.max_updates = int(max_updates)
        self._sort_keys = wp.empty(2 * self.max_updates, dtype=wp.int64, device=device)
        self._sort_indices = wp.empty(2 * self.max_updates, dtype=wp.int32, device=device)

    def add(
        self,
        data: wp.array2d(dtype=wp.uint8),
        stream_ids: wp.array(dtype=wp.int64),
        new_data: wp.array2d(dtype=wp.uint8),
        count: int,
        capacity: int,
        item_bytes: int,
    ) -> None:
        """Append values into ``data`` and update changed-stream state."""
        wp.launch(
            _fifo_prepare_sort_kernel,
            dim=count,
            inputs=[stream_ids, self._sort_keys, self._sort_indices, self.num_changed],
            device=self.device,
        )
        wpu.radix_sort_pairs(self._sort_keys, self._sort_indices, count)
        wp.launch(
            _fifo_write_sorted_starts_kernel,
            dim=count,
            inputs=[
                data,
                self._sort_keys,
                self._sort_indices,
                new_data,
                self.start_ptr,
                self.size,
                self.changed_ids,
                self.num_changed,
                count,
                capacity,
                item_bytes,
            ],
            device=self.device,
        )
