# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Execute phase for primitive graph local outputs.

Always-materialize: one producer launch over all signatures (vec3 / scalar /
quat / scalar_sum / contact), then one consumer launch that branches on each
slot's pipeline id and reads its node's value out of the per-kind buffer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..kernels_wp import compute_dense_graph_producers, dispatch_graph_dense

if TYPE_CHECKING:
    from ..multi_task_command_warp import MultiTaskCommandWarp
    from .bindings import PrimitiveGraphLocalPlan


def dispatch_primitive_graph_local_warp(command: MultiTaskCommandWarp, plan: PrimitiveGraphLocalPlan) -> None:
    """Materialize all producer nodes, then run the dense consumer pass."""
    if plan.total_consumers == 0:
        return

    total_signature_count = (
        plan.vec3_signature_count
        + plan.scalar_signature_count
        + plan.quat_signature_count
        + plan.scalar_sum_signature_count
        + plan.contact_signature_count
    )
    if total_signature_count != 0:
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
            plan.local_delta_wp,
            plan.local_error_wp,
            plan.local_activation_wp,
            plan.vec3_signature_count,
            plan.scalar_signature_count,
            plan.quat_signature_count,
            plan.scalar_sum_signature_count,
            plan.contact_signature_count,
        ],
        device=str(command.device),
    )
