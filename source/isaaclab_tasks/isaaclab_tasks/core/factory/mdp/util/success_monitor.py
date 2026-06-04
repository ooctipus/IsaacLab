# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .success_monitor_cfg import SuccessMonitorCfg


class SuccessMonitor:
    """Circular-buffer tracker of per-slot binary success outcomes.

    Maintains a sliding window of boolean results for each slot and writes
    the running success rate into a **caller-provided** tensor.
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
        self.success_buf = torch.zeros((cfg.num_monitored_data, cfg.monitored_history_len), device=cfg.device)
        self.success_pointer = torch.zeros(cfg.num_monitored_data, device=cfg.device, dtype=torch.int32)
        self.success_size = torch.zeros(cfg.num_monitored_data, device=cfg.device, dtype=torch.int32)

    def success_update(self, ids_all: torch.Tensor, success_mask: torch.Tensor) -> None:
        unique_indices, inv, counts = torch.unique(ids_all, return_inverse=True, return_counts=True)
        counts_clamped = counts.clamp(max=self.cfg.monitored_history_len).to(dtype=self.success_pointer.dtype)

        ptrs = self.success_pointer[unique_indices]
        values = (success_mask[torch.argsort(inv)]).to(device=self.cfg.device, dtype=self.success_buf.dtype)
        values_splits = torch.split(values, counts.tolist())
        clamped_values = torch.cat([grp[-n:] for grp, n in zip(values_splits, counts_clamped.tolist())])
        state_indices = torch.repeat_interleave(unique_indices, counts_clamped)
        buf_indices = torch.cat([
            torch.arange(start, start + n, dtype=torch.int64, device=self.cfg.device) % self.cfg.monitored_history_len
            for start, n in zip(ptrs.tolist(), counts_clamped.tolist())
        ])

        self.success_buf.index_put_((state_indices, buf_indices), clamped_values)

        self.success_pointer.index_add_(0, unique_indices, counts_clamped)
        self.success_pointer.remainder_(self.cfg.monitored_history_len)

        self.success_size.index_add_(0, unique_indices, counts_clamped)
        self.success_size.clamp_(max=self.cfg.monitored_history_len)
        self.success_rate[:] = self.success_buf.sum(dim=1) / self.success_size.clamp(min=1)
