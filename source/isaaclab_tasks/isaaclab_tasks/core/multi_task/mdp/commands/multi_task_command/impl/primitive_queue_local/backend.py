# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend object for primitive-queued local-output execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from .bindings import (
    PrimitiveQueueLocalPlan,
    build_primitive_queue_local_plan,
    refresh_primitive_queue_local_plan,
)
from .compose import compose_primitive_queue_local_warp
from .execute import dispatch_primitive_queue_local_warp
from .read import fill_unified_buffer_warp
from .rotation import rotate_canonical_slots_to_body_frame_warp

if TYPE_CHECKING:
    from ...multi_task_command_warp import MultiTaskCommandWarp


class PrimitiveQueueLocalBackend:
    """Execute through primitive queues and compose from local output rows."""

    name = "primitive_queue_local"

    def __init__(self, command: MultiTaskCommandWarp):
        self.plan: PrimitiveQueueLocalPlan = build_primitive_queue_local_plan(command)
        self._dispatch_graph = None

    def on_resample(self, command: MultiTaskCommandWarp, env_ids) -> None:
        """Refresh primitive queues after task assignment changes."""
        del env_ids
        refresh_primitive_queue_local_plan(command, self.plan)
        self._dispatch_graph = None

    def dispatch(self, command: MultiTaskCommandWarp) -> None:
        """Run the full per-step pipeline through a captured graph (compose included)."""
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
        """Launch the full per-step pipeline eagerly; used for warmup and graph capture."""
        fill_unified_buffer_warp(command, self.plan)
        dispatch_primitive_queue_local_warp(command, self.plan)
        rotate_canonical_slots_to_body_frame_warp(command, self.plan)
        compose_primitive_queue_local_warp(command, self.plan)

    def compose(self, command: MultiTaskCommandWarp) -> None:
        """No-op — compose was captured as part of the dispatch graph."""
        del command
