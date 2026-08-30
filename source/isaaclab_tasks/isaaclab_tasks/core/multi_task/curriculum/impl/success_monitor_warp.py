# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp
import warp.utils as wpu

from isaaclab_tasks.core.multi_task.kernels.buffer.ring_buffers_warp import (
    ring_append_bool_true_count_rate_sorted_kernel,
    ring_stream_sort_prepare_kernel,
)


def success_update(
    success_buf: wp.array2d(dtype=wp.bool),
    ids_all: wp.array(dtype=wp.int64),
    success_mask: wp.array(dtype=wp.bool),
    success_pointer: wp.array(dtype=wp.int32),
    success_size: wp.array(dtype=wp.int32),
    success_count: wp.array(dtype=wp.int32),
    success_rate: wp.array(dtype=wp.float32),
    sort_keys: wp.array(dtype=wp.int64),
    sort_indices: wp.array(dtype=wp.int32),
    count: int,
    history_len: int,
    device: str,
) -> None:
    """Sort and record success outcomes into caller-owned Warp arrays."""
    wp.launch(
        ring_stream_sort_prepare_kernel,
        dim=count,
        inputs=[ids_all, sort_keys, sort_indices],
        device=device,
    )
    wpu.radix_sort_pairs(sort_keys, sort_indices, count)
    wp.launch(
        ring_append_bool_true_count_rate_sorted_kernel,
        dim=count,
        inputs=[
            success_buf,
            sort_keys,
            sort_indices,
            success_mask,
            success_pointer,
            success_size,
            success_count,
            success_rate,
            count,
            history_len,
        ],
        device=device,
    )
