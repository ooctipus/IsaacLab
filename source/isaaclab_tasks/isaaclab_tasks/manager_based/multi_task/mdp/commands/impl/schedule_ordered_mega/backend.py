# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend object for schedule-ordered mega-kernel execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from ...kernels_wp import dispatch_compose_fused
from ..compose_select import use_parallel_compose
from ..mega_kernel.compose import compose_warp
from ..mega_kernel.execute import dispatch_mega_warp
from ..mega_kernel.read import fill_unified_buffer_warp
from ..mega_kernel.rotation import rotate_canonical_slots_to_body_frame_warp
from .bindings import ScheduleOrderedMegaPlan, build_schedule_ordered_mega_plan, refresh_schedule_ordered_mega_plan

if TYPE_CHECKING:
    from ...multi_task_command_warp import MultiTaskCommandWarp


class ScheduleOrderedMegaBackend:
    """Execute one mega-kernel launch over backend-owned schedule-ordered slots."""

    name = "schedule_ordered_mega"

    def __init__(self, command: MultiTaskCommandWarp):
        self.plan: ScheduleOrderedMegaPlan = build_schedule_ordered_mega_plan(command)
        self._dispatch_graph: wp.Graph | None = None
        self._use_fused = use_parallel_compose(command.k_max)

    def on_resample(self, command: MultiTaskCommandWarp, env_ids: torch.Tensor) -> None:
        """Refresh schedule-ordered slot ranks after task assignment changes."""
        refresh_schedule_ordered_mega_plan(command, self.plan, env_ids)

    def dispatch(self, command: MultiTaskCommandWarp, valid_slots: torch.Tensor) -> None:
        """Run the full per-step pipeline through a captured graph (compose included)."""
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
        fill_unified_buffer_warp(command, self.plan.mega)
        if self._use_fused:
            wp.launch_tiled(
                dispatch_compose_fused,
                dim=[command.num_envs],
                inputs=[
                    self.plan.mega.env_slots,
                    self.plan.mega.spec,
                    self.plan.mega.state,
                    self.plan.mega.outputs,
                    self.plan.mega.composer_state,
                    self.plan.mega.episode_length_buf_wp,
                    self.plan.mega.effective_max_episode_length_wp,
                    0.5,
                    float(command.cfg.quality_easing),
                ],
                block_dim=max(command.k_max, 32),
                device=str(command.device),
            )
            rotate_canonical_slots_to_body_frame_warp(command, self.plan.mega)
        else:
            dispatch_mega_warp(command, self.plan.mega)
            rotate_canonical_slots_to_body_frame_warp(command, self.plan.mega)
            compose_warp(command, self.plan.mega)

    def compose(self, command: MultiTaskCommandWarp, valid_slots: torch.Tensor) -> None:
        """No-op — compose was captured as part of the dispatch graph."""
        del command, valid_slots
