# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab_tasks.manager_based.multi_task.utils.streamers import FIFOStreamer

if TYPE_CHECKING:
    from ..success_monitor_cfg import SuccessMonitorCfg


@wp.kernel
def _success_rate_update_kernel(
    success_buf: wp.array2d(dtype=wp.bool),
    success_size: wp.array(dtype=wp.int32),
    success_rate: wp.array(dtype=wp.float32),
    changed_ids: wp.array(dtype=wp.int64),
    num_changed: wp.array(dtype=wp.int32),
    history_len: int,
):
    i = wp.tid()
    if i >= int(num_changed[0]):
        return

    slot = int(changed_ids[i])
    count = int(0)
    for j in range(history_len):
        if success_buf[slot, j]:
            count += 1
    denom = int(success_size[slot])
    if denom < 1:
        denom = 1
    success_rate[slot] = wp.float32(count) / wp.float32(denom)


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
        self.streamer = FIFOStreamer(
            cfg.num_monitored_data,
            cfg.device,
            max_updates=cfg.max_updates,
            warp=True,
        )
        self.success_pointer = self.streamer.start_ptr
        self.success_size = self.streamer.size
        self._wp_success_buf = wp.from_torch(self.success_buf, dtype=wp.bool)
        self._wp_success_size = wp.from_torch(self.success_size, dtype=wp.int32)
        self._wp_success_rate = wp.from_torch(self.success_rate, dtype=wp.float32)
        self._wp_changed_ids = wp.from_torch(self.streamer.changed_ids, dtype=wp.int64)
        self._wp_num_changed = wp.from_torch(self.streamer.num_changed, dtype=wp.int32)
        self._max_updates = self.streamer.changed_ids.numel()

    def success_update(self, ids_all: torch.Tensor, success_mask: torch.Tensor) -> None:
        self.streamer.add(self.success_buf, ids_all, success_mask)
        wp.launch(
            _success_rate_update_kernel,
            dim=self._max_updates,
            inputs=[
                self._wp_success_buf,
                self._wp_success_size,
                self._wp_success_rate,
                self._wp_changed_ids,
                self._wp_num_changed,
                self.cfg.monitored_history_len,
            ],
            device=str(self.success_buf.device),
        )
