# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from .success_monitor import SuccessMonitor


@configclass
class SuccessMonitorCfg:
    class_type: type[SuccessMonitor] = SuccessMonitor

    monitored_history_len: int = 100
    """Sliding window length per slot."""

    num_monitored_data: int = 0
    """Number of slots to track. Set to 0 as placeholder; the consumer
    (e.g. ``reset_accumulator`` or ``TermChoice``) overrides this at init
    based on buffer size or partition count."""

    device: str = "cpu"
    """Device for internal buffers."""

    max_updates: int | None = None
    """Maximum number of raw rows per :meth:`SuccessMonitor.success_update` call."""

    warp: bool = False
    """Whether to use the graph-friendly Warp buffer writer and rate-update path."""
