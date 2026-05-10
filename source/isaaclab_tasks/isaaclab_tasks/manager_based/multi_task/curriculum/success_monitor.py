# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .impl.success_monitor_torch import SuccessMonitorTorch
from .impl.success_monitor_warp import SuccessMonitorWarp

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
    do not scan each slot's full history window.
    """

    def __init__(self, cfg: SuccessMonitorCfg, success_rate: torch.Tensor):
        """Initialize the monitor.

        Args:
            cfg: Configuration specifying buffer dimensions and device.
            success_rate: External ``(num_slots,)`` tensor the monitor writes
                computed rates into on each :meth:`success_update` call.
        """
        self.cfg: SuccessMonitorCfg = cfg
        backend = SuccessMonitorWarp if cfg.warp else SuccessMonitorTorch
        self._impl = backend(cfg, success_rate)
        self.success_rate = self._impl.success_rate
        self.success_buf = self._impl.success_buf
        self.success_pointer = self._impl.success_pointer
        self.success_size = self._impl.success_size
        self.success_count = self._impl.success_count

    def success_update(self, ids_all: torch.Tensor, success_mask: torch.Tensor) -> None:
        """Record boolean success outcomes for the provided slot ids."""
        self._impl.success_update(ids_all, success_mask)
