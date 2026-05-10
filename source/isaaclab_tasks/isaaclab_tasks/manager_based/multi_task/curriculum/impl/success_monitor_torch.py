# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab_tasks.manager_based.multi_task.utils.buffer_writers import FIFOBufferWriter

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
        max_update_capacity = int(cfg.max_updates) if cfg.max_updates is not None else int(cfg.num_monitored_data)
        changed_ids = torch.empty(max_update_capacity, device=cfg.device, dtype=torch.int64)
        num_changed = torch.zeros(1, device=cfg.device, dtype=torch.int32)
        self.buffer_writer = FIFOBufferWriter(
            self.success_pointer,
            self.success_size,
            changed_ids,
            num_changed,
            warp=False,
        )

    def success_update(self, ids_all: torch.Tensor, success_mask: torch.Tensor) -> None:
        self.buffer_writer.add(self.success_buf, ids_all, success_mask)
        stream_ids = self.buffer_writer.changed_ids[: int(self.buffer_writer.num_changed.item())]
        self.success_rate[stream_ids] = self.success_buf[stream_ids].sum(dim=1) / self.success_size[stream_ids].clamp(
            min=1
        )
