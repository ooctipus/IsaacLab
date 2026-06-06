# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend selection for Warp-backed multi-task command dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .mega_kernel.backend import MegaKernelBackend
from .packed_scatter.backend import PackedScatterBackend
from .primitive_graph_local.backend import PrimitiveGraphLocalBackend
from .primitive_queue_local.backend import PrimitiveQueueLocalBackend

if TYPE_CHECKING:
    import torch

    from .multi_task_command_warp import MultiTaskCommandWarp


# ``schedule_ordered_mega`` is not a separate backend — its data management
# is identical to ``mega_kernel``; the only delta is sorting slot tables by
# fused-schedule id on resample. Selecting this string constructs
# ``MegaKernelBackend(slot_order="schedule")`` instead.
_SCHEDULE_ORDERED_MEGA_NAME = "schedule_ordered_mega"


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


def build_command_backend(command: MultiTaskCommandWarp, name: str) -> CommandBackend:
    """Construct the requested Warp command backend."""
    if name == MegaKernelBackend.name:
        return MegaKernelBackend(command, slot_order="natural")
    if name == _SCHEDULE_ORDERED_MEGA_NAME:
        return MegaKernelBackend(command, slot_order="schedule")
    if name == PackedScatterBackend.name:
        return PackedScatterBackend(command)
    if name == PrimitiveQueueLocalBackend.name:
        return PrimitiveQueueLocalBackend(command)
    if name == PrimitiveGraphLocalBackend.name:
        return PrimitiveGraphLocalBackend(command)
    raise ValueError(f"Unsupported MultiTaskCommand dispatch backend: {name!r}.")
