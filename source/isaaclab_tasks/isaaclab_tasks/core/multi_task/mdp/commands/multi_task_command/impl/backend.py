# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend protocol for Warp-backed multi-task command dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch

    from ..multi_task_command_warp import MultiTaskCommandWarp


class CommandBackend(Protocol):
    """Backend interface consumed by :class:`MultiTaskCommandWarp`."""

    name: str

    def on_resample(self, command: MultiTaskCommandWarp, env_ids: torch.Tensor) -> None:
        """Refresh backend-owned execution layout after task assignment changes."""
        ...

    def dispatch(self, command: MultiTaskCommandWarp) -> None:
        """Compute per-step command deltas and activation buffers."""
        ...

    def compose(self, command: MultiTaskCommandWarp) -> None:
        """Advance composer state and write per-env reward outputs."""
        ...
