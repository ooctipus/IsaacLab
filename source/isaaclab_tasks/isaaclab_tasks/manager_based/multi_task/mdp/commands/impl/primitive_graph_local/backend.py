# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend object for primitive graph local-output execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from ..compose_select import use_parallel_compose
from ..kernels_wp import compute_dense_graph_producers, dispatch_graph_dense_compose_fused
from .bindings import (
    PrimitiveGraphLocalPlan,
    build_primitive_graph_local_plan,
    refresh_primitive_graph_local_plan,
)
from .compose import compose_primitive_graph_local_warp
from .execute import dispatch_primitive_graph_local_warp
from .read import fill_unified_buffer_warp
from .rotation import rotate_canonical_slots_to_body_frame_warp

if TYPE_CHECKING:
    from ..multi_task_command_warp import MultiTaskCommandWarp


class PrimitiveGraphLocalBackend:
    """Execute through a primitive graph with shared producer nodes."""

    name = "primitive_graph_local"

    def __init__(self, command: MultiTaskCommandWarp):
        self.plan: PrimitiveGraphLocalPlan = build_primitive_graph_local_plan(command)
        self._dispatch_graph = None
        # Fuse the dense-graph consumer with compose when both conditions hold:
        # the producer/consumer split is in dense-graph mode AND k_max is large
        # enough for the parallel composer to win.
        self._use_fused_compose = use_parallel_compose(command.k_max)

    def on_resample(self, command: MultiTaskCommandWarp, env_ids: torch.Tensor) -> None:
        """Refresh primitive graph queues after task assignment changes."""
        del env_ids
        refresh_primitive_graph_local_plan(command, self.plan)
        self._dispatch_graph = None

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
        fill_unified_buffer_warp(command, self.plan)

        if self._use_fused_compose and self.plan.use_dense_graph_consumer and self.plan.total_consumers > 0:
            self._launch_dense_graph_producers(command)
            self._launch_dense_graph_consumer_fused(command)
            rotate_canonical_slots_to_body_frame_warp(command, self.plan)
        else:
            dispatch_primitive_graph_local_warp(command, self.plan)
            rotate_canonical_slots_to_body_frame_warp(command, self.plan)
            compose_primitive_graph_local_warp(command, self.plan)

    def _launch_dense_graph_producers(self, command: MultiTaskCommandWarp) -> None:
        plan = self.plan
        total_signature_count = (
            plan.vec3_signature_count
            + plan.scalar_signature_count
            + plan.quat_signature_count
            + plan.scalar_sum_signature_count
            + plan.contact_signature_count
        )
        if total_signature_count == 0:
            return
        wp.launch(
            compute_dense_graph_producers,
            dim=(command.num_envs, total_signature_count),
            inputs=[
                plan.vec3_nodes.nodes_view,
                plan.scalar_nodes.nodes_view,
                plan.quat_nodes.nodes_view,
                plan.scalar_sum_nodes.nodes_view,
                plan.contact_nodes.nodes_view,
                plan.spec,
                plan.state,
                plan.direct_vec3_wp,
                plan.direct_scalar_wp,
                plan.direct_quat_wp,
                plan.scalar_sum_wp,
                plan.contact_mask_wp,
                plan.vec3_signature_count,
                plan.scalar_signature_count,
                plan.quat_signature_count,
                plan.scalar_sum_signature_count,
                plan.contact_signature_count,
            ],
            device=str(command.device),
        )

    def _launch_dense_graph_consumer_fused(self, command: MultiTaskCommandWarp) -> None:
        plan = self.plan
        wp.launch_tiled(
            dispatch_graph_dense_compose_fused,
            dim=[command.num_envs],
            inputs=[
                plan.env_slots,
                plan.subtask_schedule_ids_wp,
                plan.vec3_nodes.nodes_view,
                plan.scalar_nodes.nodes_view,
                plan.quat_nodes.nodes_view,
                plan.scalar_sum_nodes.nodes_view,
                plan.contact_nodes.nodes_view,
                plan.spec,
                plan.state,
                plan.outputs,
                plan.composer_state,
                plan.direct_vec3_wp,
                plan.direct_scalar_wp,
                plan.direct_quat_wp,
                plan.scalar_sum_wp,
                plan.contact_mask_wp,
                plan.episode_length_buf_wp,
                plan.effective_max_episode_length_wp,
                0.5,
                float(command.cfg.quality_easing),
                plan.vec3_signature_count,
                plan.scalar_signature_count,
                plan.quat_signature_count,
                plan.scalar_sum_signature_count,
                plan.contact_signature_count,
            ],
            block_dim=max(command.k_max, 32),
            device=str(command.device),
        )

    def compose(self, command: MultiTaskCommandWarp, valid_slots: torch.Tensor) -> None:
        """No-op — compose was captured as part of the dispatch graph."""
        del command, valid_slots
