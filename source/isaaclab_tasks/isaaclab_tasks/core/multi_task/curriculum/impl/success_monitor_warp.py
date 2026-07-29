# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
import warp.utils as wpu

from isaaclab_tasks.core.multi_task.kernels.buffer.ring_buffers_warp import (
    ring_append_bool_true_count_rate_sorted_kernel,
    ring_stream_sort_prepare_kernel,
)

if TYPE_CHECKING:
    from ..success_monitor_cfg import SuccessMonitorCfg


class SuccessMonitorWarp:
    """Warp backend for sliding-window binary success rates."""

    def __init__(self, cfg: SuccessMonitorCfg, success_rate: torch.Tensor):
        self.cfg = cfg
        self.success_rate = success_rate
        self.success_buf = torch.zeros(
            (cfg.num_monitored_data, cfg.monitored_history_len), device=cfg.device, dtype=torch.bool
        )
        if success_rate.dtype != torch.float32:
            raise TypeError("SuccessMonitor warp mode requires a float32 success_rate tensor.")
        self.success_pointer = torch.zeros(cfg.num_monitored_data, device=cfg.device, dtype=torch.int32)
        self.success_size = torch.zeros_like(self.success_pointer)
        self.success_count = torch.zeros_like(self.success_pointer)
        self._wp_success_buf = wp.from_torch(self.success_buf, dtype=wp.bool)
        self._wp_success_pointer = wp.from_torch(self.success_pointer, dtype=wp.int32)
        self._wp_success_size = wp.from_torch(self.success_size, dtype=wp.int32)
        self._wp_success_count = wp.from_torch(self.success_count, dtype=wp.int32)
        self._wp_success_rate = wp.from_torch(self.success_rate, dtype=wp.float32)
        self._max_updates = int(cfg.max_updates) if cfg.max_updates is not None else int(cfg.num_monitored_data)
        self._sort_keys = wp.empty(2 * self._max_updates, dtype=wp.int64, device=str(self.success_buf.device))
        self._sort_indices = wp.empty(2 * self._max_updates, dtype=wp.int32, device=str(self.success_buf.device))

    def _validate_inputs(self, ids_all: torch.Tensor, success_mask: torch.Tensor) -> None:
        if ids_all.dtype != torch.int64:
            raise TypeError(f"SuccessMonitor ids must have dtype torch.int64, got {ids_all.dtype}.")
        if success_mask.dtype != torch.bool:
            raise TypeError(f"SuccessMonitor success_mask must have dtype torch.bool, got {success_mask.dtype}.")
        if ids_all.device != self.success_buf.device:
            raise ValueError(f"SuccessMonitor ids must be on device {self.success_buf.device}, got {ids_all.device}.")
        if success_mask.device != self.success_buf.device:
            raise ValueError(
                f"SuccessMonitor success_mask must be on device {self.success_buf.device}, got {success_mask.device}."
            )
        if ids_all.shape != success_mask.shape:
            raise ValueError(
                f"SuccessMonitor ids and success_mask must have matching shape, got "
                f"{tuple(ids_all.shape)} and {tuple(success_mask.shape)}."
            )
        if not ids_all.is_contiguous():
            raise ValueError("SuccessMonitor warp mode requires contiguous ids.")
        if not success_mask.is_contiguous():
            raise ValueError("SuccessMonitor warp mode requires contiguous success_mask.")
        if ids_all.numel() > self._max_updates:
            raise ValueError(
                f"SuccessMonitor received {ids_all.numel()} updates, exceeding max_updates={self._max_updates}."
            )

    def success_update(self, ids_all: torch.Tensor, success_mask: torch.Tensor) -> None:
        self._validate_inputs(ids_all, success_mask)
        count = ids_all.numel()
        if count == 0:
            return
        device = str(self.success_buf.device)
        wp.launch(
            ring_stream_sort_prepare_kernel,
            dim=count,
            inputs=[
                wp.from_torch(ids_all, dtype=wp.int64),
                self._sort_keys,
                self._sort_indices,
            ],
            device=device,
        )
        wpu.radix_sort_pairs(self._sort_keys, self._sort_indices, count)
        wp.launch(
            ring_append_bool_true_count_rate_sorted_kernel,
            dim=count,
            inputs=[
                self._wp_success_buf,
                self._sort_keys,
                self._sort_indices,
                wp.from_torch(success_mask, dtype=wp.bool),
                self._wp_success_pointer,
                self._wp_success_size,
                self._wp_success_count,
                self._wp_success_rate,
                count,
                self.cfg.monitored_history_len,
            ],
            device=device,
        )
