# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab_tasks.manager_based.multi_task.utils.streamers import FIFOStreamer

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
        self.streamer = FIFOStreamer(
            cfg.num_monitored_data,
            cfg.device,
            max_updates=cfg.max_updates,
            warp=False,
        )
        self.success_pointer = self.streamer.start_ptr
        self.success_size = self.streamer.size

    def success_update(self, ids_all: torch.Tensor, success_mask: torch.Tensor) -> None:
        self.streamer.add(self.success_buf, ids_all, success_mask)
        stream_ids = self.streamer.changed_ids[: int(self.streamer.num_changed.item())]
        self.success_rate[stream_ids] = self.success_buf[stream_ids].sum(dim=1) / self.success_size[stream_ids].clamp(
            min=1
        )
