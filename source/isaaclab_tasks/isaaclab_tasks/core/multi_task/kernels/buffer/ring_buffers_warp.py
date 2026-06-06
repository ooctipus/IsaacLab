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
def ring_stream_sort_prepare_kernel(
    stream_ids: wp.array(dtype=wp.int64),
    sort_keys: wp.array(dtype=wp.int64),
    sort_indices: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    sort_keys[i] = stream_ids[i]
    sort_indices[i] = i


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


@wp.kernel
def ring_append_bool_true_count_rate_sorted_kernel(
    data: wp.array2d(dtype=wp.bool),
    stream_ids_sorted: wp.array(dtype=wp.int64),
    sort_indices: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.bool),
    write_ptr: wp.array(dtype=wp.int32),
    size: wp.array(dtype=wp.int32),
    num_true: wp.array(dtype=wp.int32),
    true_rate: wp.array(dtype=wp.float32),
    num_updates: int,
    capacity: int,
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

    old_write_ptr = int(write_ptr[stream_id])
    old_size = int(size[stream_id])
    new_num_true = int(num_true[stream_id])

    if group_count >= capacity:
        new_num_true = int(0)
        source_start = group_end - capacity
        write_start = _ring_index(old_write_ptr, group_count - capacity, capacity)
        for j in range(capacity):
            src = int(sort_indices[source_start + j])
            col = _ring_index(write_start, j, capacity)
            value = values[src]
            data[stream_id, col] = value
            if value:
                new_num_true += 1
        size[stream_id] = capacity
    else:
        overwrite_start = capacity - old_size
        for j in range(group_count):
            src = int(sort_indices[i + j])
            col = _ring_index(old_write_ptr, j, capacity)
            if j >= overwrite_start and data[stream_id, col]:
                new_num_true -= 1
            value = values[src]
            data[stream_id, col] = value
            if value:
                new_num_true += 1
        new_size = old_size + group_count
        if new_size > capacity:
            new_size = capacity
        size[stream_id] = new_size

    write_ptr[stream_id] = _ring_index(old_write_ptr, group_count, capacity)
    num_true[stream_id] = new_num_true
    denom = int(size[stream_id])
    if denom < 1:
        denom = 1
    true_rate[stream_id] = wp.float32(new_num_true) / wp.float32(denom)
