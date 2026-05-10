# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend selection for Warp-backed multi-task command dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

from .mega_kernel.backend import MegaKernelBackend
from .outputs import DenseCommandOutputs, PrimitiveLocalCommandOutputs
from .packed_scatter.backend import PackedScatterBackend
from .primitive_graph_local.backend import PrimitiveGraphLocalBackend
from .primitive_queue_local.backend import PrimitiveQueueLocalBackend
from .schedule_ordered_mega.backend import ScheduleOrderedMegaBackend

if TYPE_CHECKING:
    from ..multi_task_command_warp import MultiTaskCommandWarp


class CommandBackend(Protocol):
    """Backend interface consumed by :class:`MultiTaskCommandWarp`."""

    name: str

    def on_resample(self, command: MultiTaskCommandWarp, env_ids: torch.Tensor) -> None:
        """Refresh backend-owned execution layout after task assignment changes."""
        ...

    def dispatch(self, command: MultiTaskCommandWarp, valid_slots: torch.Tensor) -> None:
        """Compute per-step command deltas and activation buffers."""
        ...

    def compose(self, command: MultiTaskCommandWarp, valid_slots: torch.Tensor) -> None:
        """Advance composer state and write per-env reward outputs."""
        ...


def build_command_backend(command: MultiTaskCommandWarp, name: str) -> CommandBackend:
    """Construct the requested Warp command backend."""
    if name == MegaKernelBackend.name:
        return MegaKernelBackend(command)
    if name == ScheduleOrderedMegaBackend.name:
        return ScheduleOrderedMegaBackend(command)
    if name == PackedScatterBackend.name:
        return PackedScatterBackend(command)
    if name == PrimitiveQueueLocalBackend.name:
        return PrimitiveQueueLocalBackend(command)
    if name == PrimitiveGraphLocalBackend.name:
        return PrimitiveGraphLocalBackend(command)
    raise ValueError(f"Unsupported MultiTaskCommand dispatch backend: {name!r}.")


def build_command_output_store(command: MultiTaskCommandWarp, name: str):
    """Construct the output storage layout for a Warp command backend."""
    if name in (MegaKernelBackend.name, ScheduleOrderedMegaBackend.name, PackedScatterBackend.name):
        return DenseCommandOutputs(command)
    if name == PrimitiveQueueLocalBackend.name:
        return PrimitiveLocalCommandOutputs(command)
    if name == PrimitiveGraphLocalBackend.name:
        return PrimitiveLocalCommandOutputs(command)
    raise ValueError(f"Unsupported MultiTaskCommand dispatch backend: {name!r}.")
