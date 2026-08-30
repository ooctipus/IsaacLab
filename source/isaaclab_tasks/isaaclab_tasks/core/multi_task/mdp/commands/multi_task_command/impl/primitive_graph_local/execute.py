# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Execute phase for primitive graph local outputs.

Always-materialize: one producer launch over all signatures (vec3 / scalar /
quat / scalar_sum / contact), then one consumer launch that branches on each
slot's pipeline id and reads its node's value out of the per-kind buffer.
The fused-compose variant fuses the consumer launch with the composer pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..kernels_wp import compute_dense_graph_producers, dispatch_graph_dense, dispatch_graph_dense_compose_fused

if TYPE_CHECKING:
    from ...multi_task_command_warp import MultiTaskCommandWarp
    from .bindings import PrimitiveGraphLocalPlan


def launch_dense_producers(command: MultiTaskCommandWarp, plan: PrimitiveGraphLocalPlan) -> None:
    """Compute every ``(env, signature)`` producer node into its per-kind buffer."""
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


def launch_dense_consumer_fused(command: MultiTaskCommandWarp, plan: PrimitiveGraphLocalPlan) -> None:
    """Block-per-env fused dense-consumer + parallel-compose kernel."""
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


def dispatch_primitive_graph_local_warp(command: MultiTaskCommandWarp, plan: PrimitiveGraphLocalPlan) -> None:
    """Materialize all producer nodes, then run the dense consumer pass."""
    if plan.total_consumers == 0:
        return
    launch_dense_producers(command, plan)
    wp.launch(
        dispatch_graph_dense,
        dim=(command.num_envs, command.k_max),
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
