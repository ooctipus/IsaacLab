# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab_tasks.core.multi_task.kernels.buffer.ring_buffers_torch import ring_append_bool_count_rate

if TYPE_CHECKING:
    from ..success_monitor_cfg import SuccessMonitorCfg


class SuccessMonitorTorch:
    """Torch backend for sliding-window binary success rates."""

    def __init__(self, cfg: SuccessMonitorCfg, success_rate: torch.Tensor):
        self.cfg = cfg
        self.success_rate = success_rate
        self.success_buf = torch.zeros(
            (cfg.num_monitored_data, cfg.monitored_history_len), device=cfg.device, dtype=torch.bool
        )
        self.success_pointer = torch.zeros(cfg.num_monitored_data, device=cfg.device, dtype=torch.int32)
        self.success_size = torch.zeros_like(self.success_pointer)
        self.success_count = torch.zeros_like(self.success_pointer)

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

    def success_update(self, ids_all: torch.Tensor, success_mask: torch.Tensor) -> None:
        self._validate_inputs(ids_all, success_mask)
        ring_append_bool_count_rate(
            self.success_buf,
            ids_all,
            success_mask,
            self.success_pointer,
            self.success_size,
            self.success_count,
            self.success_rate,
        )
