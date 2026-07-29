# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab_tasks.core.multi_task.kernels.buffer.ring_buffers_torch import ring_append_values_changed


class FIFOBufferWriterTorch:
    """Manage circular writes into caller-owned per-slot buffers."""

    def __init__(
        self,
        start_ptr: torch.Tensor,
        size: torch.Tensor,
        changed_ids: torch.Tensor,
        num_changed: torch.Tensor,
    ):
        self.start_ptr = start_ptr
        self.size = size
        self.changed_ids = changed_ids
        self.num_changed = num_changed

    def add(
        self,
        data: torch.Tensor,
        stream_ids: torch.Tensor,
        new_data: torch.Tensor,
        num_updates: int,
        capacity: int,
        item_bytes: int,
    ) -> None:
        """Append values into ``data`` and update changed-stream state."""
        del item_bytes
        ring_append_values_changed(
            data,
            stream_ids,
            new_data,
            self.start_ptr,
            self.size,
            self.changed_ids,
            self.num_changed,
            num_updates,
            capacity,
        )
