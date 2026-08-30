# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend object for primitive graph local-output execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..compose_select import use_parallel_compose
from .bindings import (
    PrimitiveGraphLocalPlan,
    build_primitive_graph_local_plan,
    refresh_primitive_graph_local_plan,
)
from .compose import compose_primitive_graph_local_warp
from .execute import (
    dispatch_primitive_graph_local_warp,
    launch_dense_consumer_fused,
    launch_dense_producers,
)
from .read import fill_unified_buffer_warp
from .rotation import rotate_canonical_slots_to_body_frame_warp

if TYPE_CHECKING:
    from ...multi_task_command_warp import MultiTaskCommandWarp


class PrimitiveGraphLocalBackend:
    """Execute through a primitive graph with shared producer nodes."""

    name = "primitive_graph_local"

    def __init__(self, command: MultiTaskCommandWarp):
        self.plan: PrimitiveGraphLocalPlan = build_primitive_graph_local_plan(command)
        self._dispatch_graph = None
        # Fuse the dense-graph consumer with compose when ``k_max`` is large
        # enough for the parallel composer to win. cfg-time decision; resampling
        # does not flip the mode.
        self._use_fused_compose = use_parallel_compose(command.k_max)

    def on_resample(self, command: MultiTaskCommandWarp, env_ids) -> None:
        """Refresh primitive graph queues after task assignment changes."""
        del env_ids
        refresh_primitive_graph_local_plan(command, self.plan)
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
        if self._use_fused_compose and self.plan.total_consumers > 0:
            launch_dense_producers(command, self.plan)
            launch_dense_consumer_fused(command, self.plan)
            rotate_canonical_slots_to_body_frame_warp(command, self.plan)
        else:
            dispatch_primitive_graph_local_warp(command, self.plan)
            rotate_canonical_slots_to_body_frame_warp(command, self.plan)
            compose_primitive_graph_local_warp(command, self.plan)

    def compose(self, command: MultiTaskCommandWarp) -> None:
        """No-op — compose was captured as part of the dispatch graph."""
        del command
