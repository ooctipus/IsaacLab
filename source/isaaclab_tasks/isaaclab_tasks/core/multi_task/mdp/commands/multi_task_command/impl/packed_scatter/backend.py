# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend object for a packed fused-pipeline queue with legacy output scatter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from .bindings import PackedScatterPlan, build_packed_scatter_plan, refresh_packed_scatter_plan
from .compose import compose_warp
from .execute import dispatch_packed_scatter_warp
from .read import fill_unified_buffer_warp
from .rotation import rotate_canonical_slots_to_body_frame_warp

if TYPE_CHECKING:
    from ...multi_task_command_warp import MultiTaskCommandWarp


class PackedScatterBackend:
    """Execute through a packed fused-pipeline queue and scatter to legacy outputs."""

    name = "packed_scatter"

    def __init__(self, command: MultiTaskCommandWarp):
        self.plan: PackedScatterPlan = build_packed_scatter_plan(command)
        self._dispatch_graph: wp.Graph | None = None

    def on_resample(self, command: MultiTaskCommandWarp, env_ids) -> None:
        """Refresh packed queues after task assignment changes."""
        del env_ids
        refresh_packed_scatter_plan(command, self.plan)
        # Launch dim depends on plan.total_work, which can change after resample.
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
        dispatch_packed_scatter_warp(command, self.plan)
        rotate_canonical_slots_to_body_frame_warp(command, self.plan)
        compose_warp(command, self.plan)

    def compose(self, command: MultiTaskCommandWarp) -> None:
        """No-op — compose was captured as part of the dispatch graph."""
        del command
