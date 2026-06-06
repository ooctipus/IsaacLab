# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp
import warp.utils as wpu

from isaaclab_tasks.core.multi_task.kernels.buffer.ring_buffers_warp import (
    ring_append_bytes_changed_ids_sorted_kernel,
    ring_stream_sort_prepare_reset_num_changed_kernel,
)


class FIFOBufferWriterWarp:
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
        if count == 0:
            self.num_changed.zero_()
            return

        wp.launch(
            ring_stream_sort_prepare_reset_num_changed_kernel,
            dim=count,
            inputs=[
                stream_ids,
                self._sort_keys,
                self._sort_indices,
                self.num_changed,
            ],
            device=self.device,
        )
        wpu.radix_sort_pairs(self._sort_keys, self._sort_indices, count)
        wp.launch(
            ring_append_bytes_changed_ids_sorted_kernel,
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
