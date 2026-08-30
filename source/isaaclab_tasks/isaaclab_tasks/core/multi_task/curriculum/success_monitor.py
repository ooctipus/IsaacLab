# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab_tasks.core.multi_task.kernels.buffer.ring_buffers_torch import ring_append_bool_count_rate

from .impl.success_monitor_warp import success_update as success_update_warp

if TYPE_CHECKING:
    from .success_monitor_cfg import SuccessMonitorCfg


class SuccessMonitor:
    """Circular-buffer tracker of per-slot binary success outcomes.

    Maintains a sliding window of boolean results for each slot and writes
    the running success rate into a **caller-provided** tensor.

    The history buffer (:attr:`success_buf`) is always allocated as
    :data:`torch.bool` — one-byte-per-cell is the documented invariant of this
    class, and any non-bool storage would multiply memory cost by 4-8× without
    changing the semantics of a binary success flag. At ``num_monitored_data ≈ 1.2 M``
    and ``monitored_history_len = 50`` (the multi-task position env), bool storage
    is ~57 MiB vs ~227 MiB for fp32. If a future caller genuinely needs non-bool
    storage (e.g. continuous-valued outcomes) they should subclass this monitor
    and override :attr:`success_buf` allocation explicitly — there is intentionally
    no cfg knob to silently change the dtype.

    The update path requires boolean success values at the monitor boundary.
    Counts are maintained incrementally in :attr:`success_count` so rate updates
    do not scan each slot's full history window. This wrapper owns the Torch
    state and, in Warp mode, its converted views and fixed sort scratch.
    """

    def __init__(self, cfg: SuccessMonitorCfg, success_rate: torch.Tensor):
        """Initialize the monitor.

        Args:
            cfg: Configuration specifying buffer dimensions and device.
            success_rate: External ``(num_slots,)`` tensor the monitor writes
                computed rates into on each :meth:`success_update` call.
        """
        self.cfg: SuccessMonitorCfg = cfg
        self.success_rate = success_rate
        self.success_buf = torch.zeros(
            (cfg.num_monitored_data, cfg.monitored_history_len), device=cfg.device, dtype=torch.bool
        )
        self.success_pointer = torch.zeros(cfg.num_monitored_data, device=cfg.device, dtype=torch.int32)
        self.success_size = torch.zeros_like(self.success_pointer)
        self.success_count = torch.zeros_like(self.success_pointer)
        self._warp = cfg.warp

        if self._warp:
            if success_rate.dtype != torch.float32:
                raise TypeError("SuccessMonitor warp mode requires a float32 success_rate tensor.")
            wp.init()
            self._device = str(self.success_buf.device)
            self._max_updates = int(cfg.max_updates) if cfg.max_updates is not None else int(cfg.num_monitored_data)
            self._wp_success_buf = wp.from_torch(self.success_buf, dtype=wp.bool)
            self._wp_success_pointer = wp.from_torch(self.success_pointer, dtype=wp.int32)
            self._wp_success_size = wp.from_torch(self.success_size, dtype=wp.int32)
            self._wp_success_count = wp.from_torch(self.success_count, dtype=wp.int32)
            self._wp_success_rate = wp.from_torch(self.success_rate, dtype=wp.float32)
            self._sort_keys = wp.empty(2 * self._max_updates, dtype=wp.int64, device=self._device)
            self._sort_indices = wp.empty(2 * self._max_updates, dtype=wp.int32, device=self._device)

    def success_update(self, ids_all: torch.Tensor, success_mask: torch.Tensor) -> None:
        """Record boolean success outcomes for the provided slot ids."""
        self._validate_inputs(ids_all, success_mask)
        count = ids_all.numel()
        if count == 0:
            return

        if self._warp:
            success_update_warp(
                self._wp_success_buf,
                wp.from_torch(ids_all, dtype=wp.int64),
                wp.from_torch(success_mask, dtype=wp.bool),
                self._wp_success_pointer,
                self._wp_success_size,
                self._wp_success_count,
                self._wp_success_rate,
                self._sort_keys,
                self._sort_indices,
                count,
                self.cfg.monitored_history_len,
                self._device,
            )
        else:
            ring_append_bool_count_rate(
                self.success_buf,
                ids_all,
                success_mask,
                self.success_pointer,
                self.success_size,
                self.success_count,
                self.success_rate,
            )

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
        if not self._warp:
            return
        if not ids_all.is_contiguous():
            raise ValueError("SuccessMonitor warp mode requires contiguous ids.")
        if not success_mask.is_contiguous():
            raise ValueError("SuccessMonitor warp mode requires contiguous success_mask.")
        if ids_all.numel() > self._max_updates:
            raise ValueError(
                f"SuccessMonitor received {ids_all.numel()} updates, exceeding max_updates={self._max_updates}."
            )
