# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend object for primitive graph local-output execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from ..mega_kernel.read import fill_unified_buffer_warp
from ..mega_kernel.rotation import rotate_canonical_slots_to_body_frame_warp
from .bindings import (
    PrimitiveGraphLocalPlan,
    build_primitive_graph_local_plan,
    refresh_primitive_graph_local_plan,
)
from .compose import compose_primitive_graph_local_warp
from .execute import dispatch_primitive_graph_local_warp

if TYPE_CHECKING:
    from ...multi_task_command_warp import MultiTaskCommandWarp


class PrimitiveGraphLocalBackend:
    """Execute through a primitive graph with shared producer nodes."""

    name = "primitive_graph_local"

    def __init__(self, command: MultiTaskCommandWarp):
        self.plan: PrimitiveGraphLocalPlan = build_primitive_graph_local_plan(command)
        self._dispatch_graph = None

    def on_resample(self, command: MultiTaskCommandWarp, env_ids: torch.Tensor) -> None:
        """Refresh primitive graph queues after task assignment changes."""
        del env_ids
        refresh_primitive_graph_local_plan(command, self.plan)
        self._dispatch_graph = None

    def dispatch(self, command: MultiTaskCommandWarp, valid_slots: torch.Tensor) -> None:
        """Run graph-replayed read, primitive graph execute, and rotation."""
        del valid_slots
        if wp.get_device(str(command.device)).is_capturing:
            self._dispatch_uncaptured(command)
            return
        if self._dispatch_graph is None:
            self._dispatch_uncaptured(command)
            with wp.ScopedCapture(device=str(command.device)) as capture:
                self._dispatch_uncaptured(command)
            self._dispatch_graph = capture.graph
            return
        wp.capture_launch(self._dispatch_graph)

    def _dispatch_uncaptured(self, command: MultiTaskCommandWarp) -> None:
        """Launch the dispatch phase once; used for warmup and graph capture."""
        fill_unified_buffer_warp(command, self.plan)
        dispatch_primitive_graph_local_warp(command, self.plan)
        rotate_canonical_slots_to_body_frame_warp(command, self.plan)

    def compose(self, command: MultiTaskCommandWarp, valid_slots: torch.Tensor) -> None:
        """Advance composer state from local activation rows."""
        del valid_slots
        compose_primitive_graph_local_warp(command, self.plan)
