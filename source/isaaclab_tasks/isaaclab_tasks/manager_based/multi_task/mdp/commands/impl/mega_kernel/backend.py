# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend object for the current mega-kernel execution layout."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from .bindings import MegaKernelPlan, build_mega_kernel_plan
from .compose import compose_warp
from .execute import dispatch_mega_warp
from .read import fill_unified_buffer_warp
from .rotation import rotate_canonical_slots_to_body_frame_warp

if TYPE_CHECKING:
    from ...multi_task_command_warp import MultiTaskCommandWarp


class MegaKernelBackend:
    """Execute ``MultiTaskCommand`` through the mega-kernel ``(env, slot)`` plan."""

    name = "mega_kernel"

    def __init__(self, command: MultiTaskCommandWarp):
        self.plan: MegaKernelPlan = build_mega_kernel_plan(command)
        self._dispatch_graph: wp.Graph | None = None

    def on_resample(self, command: MultiTaskCommandWarp, env_ids: torch.Tensor) -> None:
        """No-op: env-slot tensors are wrapped directly and mutate in place."""
        del command, env_ids

    def dispatch(self, command: MultiTaskCommandWarp, valid_slots: torch.Tensor) -> None:
        """Run the full per-step pipeline (read + dispatch + rotate + compose) through a captured graph.

        The captured graph includes ``compose`` so the public ``compose()`` hook
        becomes a no-op — saves one launch + one host-side stream synchronization
        per step relative to launching compose separately.
        """
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
        """Launch the full per-step pipeline eagerly; used for warmup and graph capture."""
        fill_unified_buffer_warp(command, self.plan)
        dispatch_mega_warp(command, self.plan)
        rotate_canonical_slots_to_body_frame_warp(command, self.plan)
        compose_warp(command, self.plan)

    def compose(self, command: MultiTaskCommandWarp, valid_slots: torch.Tensor) -> None:
        """No-op — compose was captured as part of the dispatch graph."""
        del command, valid_slots
