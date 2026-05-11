# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Execute phase for primitive graph local outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..kernels_wp import (
    compute_contact_predicate_mask,
    compute_dense_graph_producers,
    compute_direct_quat_nodes,
    compute_direct_scalar_nodes,
    compute_direct_vec3_nodes,
    compute_scalar_sum_nodes,
    dispatch_graph_contact_pair_diff,
    dispatch_graph_contact_sum,
    dispatch_graph_contact_vector,
    dispatch_graph_dense,
    dispatch_graph_direct_quat,
    dispatch_graph_direct_scalar,
    dispatch_graph_direct_vec3,
    dispatch_graph_scalar_sum,
    dispatch_primitive_local_direct_quat,
    dispatch_primitive_local_direct_scalar,
    dispatch_primitive_local_direct_vec3,
    dispatch_primitive_local_scalar_sum,
)
from ..schedules import (
    SCHEDULE_DIRECT_QUAT_DELTA,
    SCHEDULE_DIRECT_SCALAR_DELTA,
    SCHEDULE_DIRECT_VEC3_DELTA,
    SCHEDULE_SCALAR_SUM_DELTA,
    SCHEDULE_VEC3_THRESHOLD_PAIR_DIFF_DELTA,
    SCHEDULE_VEC3_THRESHOLD_SUM_DELTA,
    SCHEDULE_VEC3_THRESHOLD_VECTOR_DELTA,
)

if TYPE_CHECKING:
    from ..multi_task_command_warp import MultiTaskCommandWarp
    from .bindings import PrimitiveGraphLocalPlan


def _launch_local_kernel(command: MultiTaskCommandWarp, plan: PrimitiveGraphLocalPlan, kernel, count: int) -> None:
    wp.launch(
        kernel,
        dim=count,
        inputs=[
            plan.consumer_view,
            plan.spec,
            plan.state,
            plan.outputs,
            plan.local_delta_wp,
            plan.local_error_wp,
            plan.local_activation_wp,
        ],
        device=str(command.device),
    )


def _launch_direct_vec3_kernel(command: MultiTaskCommandWarp, plan: PrimitiveGraphLocalPlan, count: int) -> None:
    wp.launch(
        dispatch_graph_direct_vec3,
        dim=count,
        inputs=[
            plan.consumer_view,
            plan.vec3_nodes.nodes_view,
            plan.spec,
            plan.state,
            plan.outputs,
            plan.direct_vec3_wp,
            plan.local_delta_wp,
            plan.local_error_wp,
            plan.local_activation_wp,
        ],
        device=str(command.device),
    )


def _launch_direct_scalar_kernel(command: MultiTaskCommandWarp, plan: PrimitiveGraphLocalPlan, count: int) -> None:
    wp.launch(
        dispatch_graph_direct_scalar,
        dim=count,
        inputs=[
            plan.consumer_view,
            plan.scalar_nodes.nodes_view,
            plan.spec,
            plan.state,
            plan.outputs,
            plan.direct_scalar_wp,
            plan.local_delta_wp,
            plan.local_error_wp,
            plan.local_activation_wp,
        ],
        device=str(command.device),
    )


def _launch_direct_quat_kernel(command: MultiTaskCommandWarp, plan: PrimitiveGraphLocalPlan, count: int) -> None:
    wp.launch(
        dispatch_graph_direct_quat,
        dim=count,
        inputs=[
            plan.consumer_view,
            plan.quat_nodes.nodes_view,
            plan.spec,
            plan.state,
            plan.outputs,
            plan.direct_quat_wp,
            plan.local_delta_wp,
            plan.local_error_wp,
            plan.local_activation_wp,
        ],
        device=str(command.device),
    )


def _launch_contact_kernel(command: MultiTaskCommandWarp, plan: PrimitiveGraphLocalPlan, kernel, count: int) -> None:
    wp.launch(
        kernel,
        dim=count,
        inputs=[
            plan.consumer_view,
            plan.contact_nodes.nodes_view,
            plan.spec,
            plan.state,
            plan.outputs,
            plan.contact_mask_wp,
            plan.local_delta_wp,
            plan.local_error_wp,
            plan.local_activation_wp,
        ],
        device=str(command.device),
    )


def _launch_scalar_sum_kernel(command: MultiTaskCommandWarp, plan: PrimitiveGraphLocalPlan, kernel, count: int) -> None:
    wp.launch(
        kernel,
        dim=count,
        inputs=[
            plan.consumer_view,
            plan.scalar_sum_nodes.nodes_view,
            plan.spec,
            plan.state,
            plan.outputs,
            plan.scalar_sum_wp,
            plan.local_delta_wp,
            plan.local_error_wp,
            plan.local_activation_wp,
        ],
        device=str(command.device),
    )


def _launch_dense_graph_consumer(command: MultiTaskCommandWarp, plan: PrimitiveGraphLocalPlan) -> None:
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


def dispatch_primitive_graph_local_warp(command: MultiTaskCommandWarp, plan: PrimitiveGraphLocalPlan) -> None:
    """Run the primitive graph: shared nodes first, terminal chains second."""
    if plan.total_consumers == 0:
        return

    counts = plan.schedule_counts_py
    if plan.use_dense_graph_consumer:
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
        _launch_dense_graph_consumer(command, plan)
        return

    if counts[SCHEDULE_DIRECT_VEC3_DELTA] != 0:
        if plan.use_vec3_graph:
            wp.launch(
                compute_direct_vec3_nodes,
                dim=plan.vec3_node_count,
                inputs=[plan.vec3_nodes.nodes_view, plan.spec, plan.state, plan.direct_vec3_wp],
                device=str(command.device),
            )
            _launch_direct_vec3_kernel(command, plan, counts[SCHEDULE_DIRECT_VEC3_DELTA])
        else:
            _launch_local_kernel(
                command, plan, dispatch_primitive_local_direct_vec3, counts[SCHEDULE_DIRECT_VEC3_DELTA]
            )
    if counts[SCHEDULE_DIRECT_SCALAR_DELTA] != 0:
        if plan.use_scalar_graph:
            wp.launch(
                compute_direct_scalar_nodes,
                dim=plan.scalar_node_count,
                inputs=[plan.scalar_nodes.nodes_view, plan.spec, plan.state, plan.direct_scalar_wp],
                device=str(command.device),
            )
            _launch_direct_scalar_kernel(command, plan, counts[SCHEDULE_DIRECT_SCALAR_DELTA])
        else:
            _launch_local_kernel(
                command, plan, dispatch_primitive_local_direct_scalar, counts[SCHEDULE_DIRECT_SCALAR_DELTA]
            )
    if counts[SCHEDULE_DIRECT_QUAT_DELTA] != 0:
        if plan.use_quat_graph:
            wp.launch(
                compute_direct_quat_nodes,
                dim=plan.quat_node_count,
                inputs=[plan.quat_nodes.nodes_view, plan.spec, plan.state, plan.direct_quat_wp],
                device=str(command.device),
            )
            _launch_direct_quat_kernel(command, plan, counts[SCHEDULE_DIRECT_QUAT_DELTA])
        else:
            _launch_local_kernel(
                command, plan, dispatch_primitive_local_direct_quat, counts[SCHEDULE_DIRECT_QUAT_DELTA]
            )
    if counts[SCHEDULE_SCALAR_SUM_DELTA] != 0:
        if plan.use_scalar_sum_graph:
            wp.launch(
                compute_scalar_sum_nodes,
                dim=plan.scalar_sum_node_count,
                inputs=[plan.scalar_sum_nodes.nodes_view, plan.spec, plan.state, plan.scalar_sum_wp],
                device=str(command.device),
            )
            _launch_scalar_sum_kernel(command, plan, dispatch_graph_scalar_sum, counts[SCHEDULE_SCALAR_SUM_DELTA])
        else:
            _launch_local_kernel(command, plan, dispatch_primitive_local_scalar_sum, counts[SCHEDULE_SCALAR_SUM_DELTA])

    if plan.contact_node_count == 0:
        return
    wp.launch(
        compute_contact_predicate_mask,
        dim=plan.contact_node_count,
        inputs=[plan.contact_nodes.nodes_view, plan.spec, plan.state, plan.contact_mask_wp],
        device=str(command.device),
    )
    if counts[SCHEDULE_VEC3_THRESHOLD_VECTOR_DELTA] != 0:
        _launch_contact_kernel(
            command, plan, dispatch_graph_contact_vector, counts[SCHEDULE_VEC3_THRESHOLD_VECTOR_DELTA]
        )
    if counts[SCHEDULE_VEC3_THRESHOLD_SUM_DELTA] != 0:
        _launch_contact_kernel(command, plan, dispatch_graph_contact_sum, counts[SCHEDULE_VEC3_THRESHOLD_SUM_DELTA])
    if counts[SCHEDULE_VEC3_THRESHOLD_PAIR_DIFF_DELTA] != 0:
        _launch_contact_kernel(
            command, plan, dispatch_graph_contact_pair_diff, counts[SCHEDULE_VEC3_THRESHOLD_PAIR_DIFF_DELTA]
        )
