# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp


@wp.func
def _ring_index(write_ptr: int, offset: int, capacity: int) -> int:
    return (write_ptr + offset) % capacity


@wp.kernel
def ring_stream_sort_prepare_reset_num_changed_kernel(
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
def ring_append_bytes_changed_ids_sorted_kernel(
    data: wp.array2d(dtype=wp.uint8),
    stream_ids_sorted: wp.array(dtype=wp.int64),
    sort_indices: wp.array(dtype=wp.int32),
    values: wp.array2d(dtype=wp.uint8),
    write_ptr: wp.array(dtype=wp.int32),
    size: wp.array(dtype=wp.int32),
    changed_ids: wp.array(dtype=wp.int64),
    num_changed: wp.array(dtype=wp.int32),
    num_updates: int,
    capacity: int,
    item_bytes: int,
):
    i = wp.tid()
    stream_id = int(stream_ids_sorted[i])
    if i > 0 and int(stream_ids_sorted[i - 1]) == stream_id:
        return

    lo = i + 1
    hi = num_updates
    while lo < hi:
        mid = (lo + hi) // 2
        if int(stream_ids_sorted[mid]) == stream_id:
            lo = mid + 1
        else:
            hi = mid
    group_end = lo

    group_count = group_end - i
    num_written = group_count
    if num_written > capacity:
        num_written = capacity

    old_write_ptr = int(write_ptr[stream_id])
    write_start = _ring_index(old_write_ptr, group_count - num_written, capacity)
    source_start = group_end - num_written
    for j in range(num_written):
        src = int(sort_indices[source_start + j])
        dst = _ring_index(write_start, j, capacity) * item_bytes
        for b in range(item_bytes):
            data[stream_id, dst + b] = values[src, b]

    write_ptr[stream_id] = _ring_index(old_write_ptr, group_count, capacity)
    new_size = int(size[stream_id]) + group_count
    if new_size > capacity:
        new_size = capacity
    size[stream_id] = new_size
    changed_slot = wp.atomic_add(num_changed, 0, 1)
    changed_ids[changed_slot] = wp.int64(stream_id)
