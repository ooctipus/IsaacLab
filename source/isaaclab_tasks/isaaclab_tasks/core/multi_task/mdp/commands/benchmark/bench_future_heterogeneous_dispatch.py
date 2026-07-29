# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Synthetic future-shape benchmark for heterogeneous command dispatch.

This benchmark intentionally steps outside the current locomotion preset. It
models the expected Octi regime:

* many semantic state kernels, e.g. 128-256;
* a much smaller number of primitive families, currently fixed to 8;
* roughly one million active task/subtask instances per command update.

The benchmark compares execution layouts, not production semantics. The
``mega_loop`` variant is a parameterized stress path, but it is **not** the
primary branch-divergence proxy: Warp/LLVM can simplify the state-kernel loop
more aggressively than it can simplify a real hard branch tree. Use
``bench_dispatch_homogeneity.py`` for the hard-branch result and this script for
checking how queue/packing layouts scale when the semantic state-kernel count is
larger than that hard-coded benchmark.

* ``mega_loop``: one large heterogeneous kernel. Each thread pays a
  state-kernel dispatch loop before entering the primitive body.
* ``state_queue``: one queue/launch per semantic state kernel.
* ``primitive_queue_local_synth``: synthetic proxy for the production
  ``primitive_queue_local`` backend. It uses one queue/launch per primitive
  family.
* ``primitive_graph_local_synth``: synthetic proxy for the production
  ``primitive_graph_local`` backend. Primitive-family queues also use explicit
  shared nodes for lower-level work such as vec3 deltas, reductions, contact
  predicates, and local-frame transforms.
* ``packed_scatter``: primitive-family packed inputs with scatter back to the
  original item rows.
* ``packed_local``: primitive-family packed inputs and local outputs, i.e. the
  shape that removes the scatter write from dispatch.

Run with:

.. code-block:: bash

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/\
multi_task/mdp/commands/benchmark/bench_future_heterogeneous_dispatch.py \
--n-work 1048576 --state-kernels 192 --pattern random --graph
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
import warp as wp

from isaaclab_tasks.core.multi_task.mdp.commands.benchmark.bench_dispatch_homogeneity import (
    SRC_WIDTH,
    TGT_WIDTH,
    PackedOutput,
    VariantOutput,
    _normalize_quat,
    _write_contact_any16,
    _write_contact_diff16,
    _write_local_frame_vec3,
    _write_quat,
    _write_reduce8,
    _write_reduce32,
    _write_scalar,
    _write_vec3,
    build_contact_graph_inputs,
    build_shared_node_inputs,
    contact_predicate16_graph_kernel,
    graph_contact_any16_kernel,
    graph_contact_diff16_kernel,
    graph_scalar_reduce_kernel,
    local_frame_vec3_graph_kernel,
    packed_contact_any16_scatter_kernel,
    packed_contact_diff16_scatter_kernel,
    packed_local_frame_vec3_scatter_kernel,
    packed_quat_scatter_kernel,
    packed_reduce8_scatter_kernel,
    packed_reduce32_scatter_kernel,
    packed_scalar_scatter_kernel,
    packed_vec3_scatter_kernel,
    queue_contact_any16_kernel,
    queue_contact_diff16_kernel,
    queue_local_frame_vec3_kernel,
    queue_quat_kernel,
    queue_reduce8_kernel,
    queue_reduce32_kernel,
    queue_scalar_kernel,
    queue_vec3_kernel,
    reduce8_graph_kernel,
    reduce32_graph_kernel,
    vec3_delta_graph_kernel,
)

NUM_PRIMITIVE_FAMILIES = 8


@wp.kernel
def graph_vec3_kernel(
    work_ids: wp.array(dtype=wp.int32),
    node_ids_by_item: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    graph_vec3: wp.array2d(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    """Consume precomputed vec3 deltas from the future synthetic graph."""
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    node = int(node_ids_by_item[i])
    s = scale[int(kind[i])]
    d0 = graph_vec3[node, 0] * s
    d1 = graph_vec3[node, 1] * s
    d2 = graph_vec3[node, 2] * s
    err = wp.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    delta[i, 0] = d0
    delta[i, 1] = d1
    delta[i, 2] = d2
    delta[i, 3] = 0.0
    error[i] = err
    activation[i] = wp.tanh(act_param[i] / (err + 1.0e-6))


@wp.kernel
def future_mega_loop_kernel(
    state_kernel_id: wp.array(dtype=wp.int32),
    state_kernel_scale: wp.array(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    num_state_kernels: int,
    states_per_primitive: int,
):
    """Mega-kernel baseline with a large semantic dispatch loop."""
    i = wp.tid()
    sid = int(state_kernel_id[i])
    primitive = int(0)
    for k in range(num_state_kernels):
        if sid == k:
            primitive = k // states_per_primitive
            if primitive > 7:
                primitive = 7

    if primitive == 0:
        _write_vec3(i, state_kernel_scale[sid], src, tgt, act_param, delta, error, activation)
    elif primitive == 1:
        _write_scalar(i, state_kernel_scale[sid], src, tgt, act_param, delta, error, activation)
    elif primitive == 2:
        _write_quat(i, state_kernel_scale[sid], src, tgt, act_param, delta, error, activation)
    elif primitive == 3:
        _write_reduce8(i, state_kernel_scale[sid], src, tgt, act_param, delta, error, activation)
    elif primitive == 4:
        _write_reduce32(i, state_kernel_scale[sid], src, tgt, act_param, delta, error, activation)
    elif primitive == 5:
        _write_contact_any16(i, state_kernel_scale[sid], src, tgt, act_param, delta, error, activation)
    elif primitive == 6:
        _write_contact_diff16(i, state_kernel_scale[sid], src, tgt, act_param, delta, error, activation)
    else:
        _write_local_frame_vec3(i, state_kernel_scale[sid], src, tgt, act_param, delta, error, activation)


@dataclass
class FutureWorkload:
    """Caller-owned tensors and prebuilt queues for a future-shaped workload."""

    state_kernel_id: torch.Tensor
    src: torch.Tensor
    tgt: torch.Tensor
    act_param: torch.Tensor
    state_kernel_scale: torch.Tensor
    state_queues: list[torch.Tensor]
    primitive_queues: list[torch.Tensor]
    packed_src: list[torch.Tensor]
    packed_tgt: list[torch.Tensor]
    packed_act_param: list[torch.Tensor]
    packed_kind: list[torch.Tensor]
    packed_local_ids: list[torch.Tensor]
    vec3_node_work_ids: torch.Tensor
    vec3_node_ids_by_item: torch.Tensor
    graph_vec3: torch.Tensor
    reduce8_node_work_ids: torch.Tensor
    reduce8_node_ids_by_item: torch.Tensor
    graph_reduce8: torch.Tensor
    reduce32_node_work_ids: torch.Tensor
    reduce32_node_ids_by_item: torch.Tensor
    graph_reduce32: torch.Tensor
    contact_node_work_ids: torch.Tensor
    contact_node_ids_by_item: torch.Tensor
    contact_mask: torch.Tensor
    local_frame_node_work_ids: torch.Tensor
    local_frame_node_ids_by_item: torch.Tensor
    graph_local_frame: torch.Tensor
    states_per_primitive: int


def _make_state_ids(n_work: int, num_state_kernels: int, pattern: str, device: torch.device) -> torch.Tensor:
    """Build synthetic semantic state-kernel ids."""
    if pattern == "grouped":
        n_per_kind = (n_work + num_state_kernels - 1) // num_state_kernels
        state_id = torch.arange(num_state_kernels, device=device, dtype=torch.int32).repeat_interleave(n_per_kind)
        return state_id[:n_work].contiguous()
    if pattern == "skew":
        rank = torch.arange(1, num_state_kernels + 1, device=device, dtype=torch.float32)
        prob = (1.0 / rank) / (1.0 / rank).sum()
        return torch.multinomial(prob, n_work, replacement=True).to(torch.int32)
    return torch.randint(0, num_state_kernels, (n_work,), device=device, dtype=torch.int32)


def make_future_workload(
    n_work: int,
    num_state_kernels: int,
    pattern: str,
    seed: int,
    device: torch.device,
    graph_fanout: int,
) -> FutureWorkload:
    """Allocate one synthetic future workload and its prebuilt execution queues."""
    torch.manual_seed(seed)
    states_per_primitive = (num_state_kernels + NUM_PRIMITIVE_FAMILIES - 1) // NUM_PRIMITIVE_FAMILIES
    state_kernel_id = _make_state_ids(n_work, num_state_kernels, pattern, device)
    primitive_id = (state_kernel_id // states_per_primitive).clamp(max=NUM_PRIMITIVE_FAMILIES - 1)

    src = torch.randn(n_work, SRC_WIDTH, device=device, dtype=torch.float32)
    tgt = torch.randn(n_work, TGT_WIDTH, device=device, dtype=torch.float32)
    src[:, :4] = _normalize_quat(src[:, :4])
    src[:, 8:12] = _normalize_quat(src[:, 8:12])
    tgt[:, :4] = _normalize_quat(tgt[:, :4])
    act_param = torch.empty(n_work, device=device, dtype=torch.float32).uniform_(0.5, 2.0)
    state_kernel_scale = torch.linspace(0.40, 1.45, num_state_kernels, device=device, dtype=torch.float32)

    state_queues = [
        (state_kernel_id == k).nonzero(as_tuple=False).flatten().to(torch.int32).contiguous()
        for k in range(num_state_kernels)
    ]
    primitive_queues = [
        (primitive_id == p).nonzero(as_tuple=False).flatten().to(torch.int32).contiguous()
        for p in range(NUM_PRIMITIVE_FAMILIES)
    ]
    vec3_node_work_ids, vec3_node_ids_by_item, graph_vec3 = build_shared_node_inputs(
        n_work, primitive_queues[0], graph_fanout, ((src, 0, 3), (tgt, 0, 3)), 3
    )
    reduce8_node_work_ids, reduce8_node_ids_by_item, graph_reduce8 = build_shared_node_inputs(
        n_work, primitive_queues[3], graph_fanout, ((src, 0, 8),), 1
    )
    graph_reduce8 = graph_reduce8.flatten()
    reduce32_node_work_ids, reduce32_node_ids_by_item, graph_reduce32 = build_shared_node_inputs(
        n_work, primitive_queues[4], graph_fanout, ((src, 0, 64),), 1
    )
    graph_reduce32 = graph_reduce32.flatten()
    contact_node_work_ids, contact_node_ids_by_item, contact_mask = build_contact_graph_inputs(
        src, tgt, primitive_queues, graph_fanout
    )
    local_frame_node_work_ids, local_frame_node_ids_by_item, graph_local_frame = build_shared_node_inputs(
        n_work, primitive_queues[7], graph_fanout, ((src, 0, 3), (src, 8, 12), (tgt, 0, 3)), 3
    )
    packed_src = [src[ids.long()].contiguous() for ids in primitive_queues]
    packed_tgt = [tgt[ids.long()].contiguous() for ids in primitive_queues]
    packed_act_param = [act_param[ids.long()].contiguous() for ids in primitive_queues]
    packed_kind = [state_kernel_id[ids.long()].contiguous() for ids in primitive_queues]
    packed_local_ids = [torch.arange(ids.numel(), device=device, dtype=torch.int32) for ids in primitive_queues]
    return FutureWorkload(
        state_kernel_id=state_kernel_id,
        src=src,
        tgt=tgt,
        act_param=act_param,
        state_kernel_scale=state_kernel_scale,
        state_queues=state_queues,
        primitive_queues=primitive_queues,
        packed_src=packed_src,
        packed_tgt=packed_tgt,
        packed_act_param=packed_act_param,
        packed_kind=packed_kind,
        packed_local_ids=packed_local_ids,
        vec3_node_work_ids=vec3_node_work_ids,
        vec3_node_ids_by_item=vec3_node_ids_by_item,
        graph_vec3=graph_vec3,
        reduce8_node_work_ids=reduce8_node_work_ids,
        reduce8_node_ids_by_item=reduce8_node_ids_by_item,
        graph_reduce8=graph_reduce8,
        reduce32_node_work_ids=reduce32_node_work_ids,
        reduce32_node_ids_by_item=reduce32_node_ids_by_item,
        graph_reduce32=graph_reduce32,
        contact_node_work_ids=contact_node_work_ids,
        contact_node_ids_by_item=contact_node_ids_by_item,
        contact_mask=contact_mask,
        local_frame_node_work_ids=local_frame_node_work_ids,
        local_frame_node_ids_by_item=local_frame_node_ids_by_item,
        graph_local_frame=graph_local_frame,
        states_per_primitive=states_per_primitive,
    )


def make_output(n_work: int, device: torch.device) -> VariantOutput:
    """Allocate dense output tensors."""
    return VariantOutput(
        delta=torch.empty(n_work, 4, device=device, dtype=torch.float32),
        error=torch.empty(n_work, device=device, dtype=torch.float32),
        activation=torch.empty(n_work, device=device, dtype=torch.float32),
    )


def make_packed_output(work: FutureWorkload) -> PackedOutput:
    """Allocate packed local output tensors."""
    return PackedOutput(
        delta=[
            torch.empty(ids.numel(), 4, device=work.state_kernel_id.device, dtype=torch.float32)
            for ids in work.primitive_queues
        ],
        error=[
            torch.empty(ids.numel(), device=work.state_kernel_id.device, dtype=torch.float32)
            for ids in work.primitive_queues
        ],
        activation=[
            torch.empty(ids.numel(), device=work.state_kernel_id.device, dtype=torch.float32)
            for ids in work.primitive_queues
        ],
    )


def launch_mega_loop(work: FutureWorkload, out: VariantOutput) -> None:
    """Launch the future mega-loop baseline."""
    wp.launch(
        future_mega_loop_kernel,
        dim=work.state_kernel_id.numel(),
        inputs=[
            wp.from_torch(work.state_kernel_id, dtype=wp.int32),
            wp.from_torch(work.state_kernel_scale, dtype=wp.float32),
            wp.from_torch(work.src, dtype=wp.float32),
            wp.from_torch(work.tgt, dtype=wp.float32),
            wp.from_torch(work.act_param, dtype=wp.float32),
            wp.from_torch(out.delta, dtype=wp.float32),
            wp.from_torch(out.error, dtype=wp.float32),
            wp.from_torch(out.activation, dtype=wp.float32),
            int(work.state_kernel_scale.numel()),
            work.states_per_primitive,
        ],
        device=str(work.state_kernel_id.device),
    )


def _queue_kernels():
    return [
        queue_vec3_kernel,
        queue_scalar_kernel,
        queue_quat_kernel,
        queue_reduce8_kernel,
        queue_reduce32_kernel,
        queue_contact_any16_kernel,
        queue_contact_diff16_kernel,
        queue_local_frame_vec3_kernel,
    ]


def _packed_kernels():
    return [
        packed_vec3_scatter_kernel,
        packed_scalar_scatter_kernel,
        packed_quat_scatter_kernel,
        packed_reduce8_scatter_kernel,
        packed_reduce32_scatter_kernel,
        packed_contact_any16_scatter_kernel,
        packed_contact_diff16_scatter_kernel,
        packed_local_frame_vec3_scatter_kernel,
    ]


def _launch_queue_kernel(kernel, ids: torch.Tensor, work: FutureWorkload, out: VariantOutput) -> None:
    if ids.numel() == 0:
        return
    wp.launch(
        kernel,
        dim=ids.numel(),
        inputs=[
            wp.from_torch(ids, dtype=wp.int32),
            wp.from_torch(work.state_kernel_id, dtype=wp.int32),
            wp.from_torch(work.state_kernel_scale, dtype=wp.float32),
            wp.from_torch(work.src, dtype=wp.float32),
            wp.from_torch(work.tgt, dtype=wp.float32),
            wp.from_torch(work.act_param, dtype=wp.float32),
            wp.from_torch(out.delta, dtype=wp.float32),
            wp.from_torch(out.error, dtype=wp.float32),
            wp.from_torch(out.activation, dtype=wp.float32),
            ids.numel(),
        ],
        device=str(work.state_kernel_id.device),
    )


def launch_state_queue(work: FutureWorkload, out: VariantOutput) -> None:
    """Launch one kernel per nonempty semantic state-kernel queue."""
    kernels = _queue_kernels()
    for state_id, ids in enumerate(work.state_queues):
        primitive = min(state_id // work.states_per_primitive, NUM_PRIMITIVE_FAMILIES - 1)
        _launch_queue_kernel(kernels[primitive], ids, work, out)


def launch_primitive_queue(work: FutureWorkload, out: VariantOutput) -> None:
    """Launch one kernel per primitive family queue."""
    for kernel, ids in zip(_queue_kernels(), work.primitive_queues):
        _launch_queue_kernel(kernel, ids, work, out)


def _launch_graph_contact_kernel(kernel, ids: torch.Tensor, work: FutureWorkload, out: VariantOutput) -> None:
    if ids.numel() == 0:
        return
    wp.launch(
        kernel,
        dim=ids.numel(),
        inputs=[
            wp.from_torch(ids, dtype=wp.int32),
            wp.from_torch(work.contact_node_ids_by_item, dtype=wp.int32),
            wp.from_torch(work.state_kernel_id, dtype=wp.int32),
            wp.from_torch(work.state_kernel_scale, dtype=wp.float32),
            wp.from_torch(work.tgt, dtype=wp.float32),
            wp.from_torch(work.act_param, dtype=wp.float32),
            wp.from_torch(work.contact_mask, dtype=wp.float32),
            wp.from_torch(out.delta, dtype=wp.float32),
            wp.from_torch(out.error, dtype=wp.float32),
            wp.from_torch(out.activation, dtype=wp.float32),
            ids.numel(),
        ],
        device=str(work.state_kernel_id.device),
    )


def _launch_graph_vec3_kernel(
    kernel,
    node_ids: torch.Tensor,
    node_ids_by_item: torch.Tensor,
    graph_vec3_out: torch.Tensor,
    ids: torch.Tensor,
    work: FutureWorkload,
    out: VariantOutput,
) -> None:
    if node_ids.numel() != 0:
        wp.launch(
            kernel,
            dim=node_ids.numel(),
            inputs=[
                wp.from_torch(node_ids, dtype=wp.int32),
                wp.from_torch(work.src, dtype=wp.float32),
                wp.from_torch(work.tgt, dtype=wp.float32),
                wp.from_torch(graph_vec3_out, dtype=wp.float32),
                node_ids.numel(),
            ],
            device=str(work.state_kernel_id.device),
        )
    if ids.numel() == 0:
        return
    wp.launch(
        graph_vec3_kernel,
        dim=ids.numel(),
        inputs=[
            wp.from_torch(ids, dtype=wp.int32),
            wp.from_torch(node_ids_by_item, dtype=wp.int32),
            wp.from_torch(work.state_kernel_id, dtype=wp.int32),
            wp.from_torch(work.state_kernel_scale, dtype=wp.float32),
            wp.from_torch(work.act_param, dtype=wp.float32),
            wp.from_torch(graph_vec3_out, dtype=wp.float32),
            wp.from_torch(out.delta, dtype=wp.float32),
            wp.from_torch(out.error, dtype=wp.float32),
            wp.from_torch(out.activation, dtype=wp.float32),
            ids.numel(),
        ],
        device=str(work.state_kernel_id.device),
    )


def _launch_graph_reduce_kernel(
    kernel,
    node_ids: torch.Tensor,
    node_ids_by_item: torch.Tensor,
    graph_scalar_out: torch.Tensor,
    ids: torch.Tensor,
    work: FutureWorkload,
    out: VariantOutput,
) -> None:
    if node_ids.numel() != 0:
        wp.launch(
            kernel,
            dim=node_ids.numel(),
            inputs=[
                wp.from_torch(node_ids, dtype=wp.int32),
                wp.from_torch(work.src, dtype=wp.float32),
                wp.from_torch(graph_scalar_out, dtype=wp.float32),
                node_ids.numel(),
            ],
            device=str(work.state_kernel_id.device),
        )
    if ids.numel() == 0:
        return
    wp.launch(
        graph_scalar_reduce_kernel,
        dim=ids.numel(),
        inputs=[
            wp.from_torch(ids, dtype=wp.int32),
            wp.from_torch(node_ids_by_item, dtype=wp.int32),
            wp.from_torch(work.state_kernel_id, dtype=wp.int32),
            wp.from_torch(work.state_kernel_scale, dtype=wp.float32),
            wp.from_torch(work.tgt, dtype=wp.float32),
            wp.from_torch(work.act_param, dtype=wp.float32),
            wp.from_torch(graph_scalar_out, dtype=wp.float32),
            wp.from_torch(out.delta, dtype=wp.float32),
            wp.from_torch(out.error, dtype=wp.float32),
            wp.from_torch(out.activation, dtype=wp.float32),
            ids.numel(),
        ],
        device=str(work.state_kernel_id.device),
    )


def launch_primitive_graph(work: FutureWorkload, out: VariantOutput) -> None:
    """Launch primitive-family queues with explicit shared lower-level nodes."""
    _launch_graph_vec3_kernel(
        vec3_delta_graph_kernel,
        work.vec3_node_work_ids,
        work.vec3_node_ids_by_item,
        work.graph_vec3,
        work.primitive_queues[0],
        work,
        out,
    )
    _launch_queue_kernel(queue_scalar_kernel, work.primitive_queues[1], work, out)
    _launch_queue_kernel(queue_quat_kernel, work.primitive_queues[2], work, out)
    _launch_graph_reduce_kernel(
        reduce8_graph_kernel,
        work.reduce8_node_work_ids,
        work.reduce8_node_ids_by_item,
        work.graph_reduce8,
        work.primitive_queues[3],
        work,
        out,
    )
    _launch_graph_reduce_kernel(
        reduce32_graph_kernel,
        work.reduce32_node_work_ids,
        work.reduce32_node_ids_by_item,
        work.graph_reduce32,
        work.primitive_queues[4],
        work,
        out,
    )
    if work.contact_node_work_ids.numel() != 0:
        wp.launch(
            contact_predicate16_graph_kernel,
            dim=work.contact_node_work_ids.numel(),
            inputs=[
                wp.from_torch(work.contact_node_work_ids, dtype=wp.int32),
                wp.from_torch(work.src, dtype=wp.float32),
                wp.from_torch(work.tgt, dtype=wp.float32),
                wp.from_torch(work.contact_mask, dtype=wp.float32),
                work.contact_node_work_ids.numel(),
            ],
            device=str(work.state_kernel_id.device),
        )
    _launch_graph_contact_kernel(graph_contact_any16_kernel, work.primitive_queues[5], work, out)
    _launch_graph_contact_kernel(graph_contact_diff16_kernel, work.primitive_queues[6], work, out)
    _launch_graph_vec3_kernel(
        local_frame_vec3_graph_kernel,
        work.local_frame_node_work_ids,
        work.local_frame_node_ids_by_item,
        work.graph_local_frame,
        work.primitive_queues[7],
        work,
        out,
    )


def _launch_packed_kernel(kernel, primitive: int, work: FutureWorkload, out: VariantOutput) -> None:
    ids = work.primitive_queues[primitive]
    if ids.numel() == 0:
        return
    wp.launch(
        kernel,
        dim=ids.numel(),
        inputs=[
            wp.from_torch(ids, dtype=wp.int32),
            wp.from_torch(work.packed_kind[primitive], dtype=wp.int32),
            wp.from_torch(work.state_kernel_scale, dtype=wp.float32),
            wp.from_torch(work.packed_src[primitive], dtype=wp.float32),
            wp.from_torch(work.packed_tgt[primitive], dtype=wp.float32),
            wp.from_torch(work.packed_act_param[primitive], dtype=wp.float32),
            wp.from_torch(out.delta, dtype=wp.float32),
            wp.from_torch(out.error, dtype=wp.float32),
            wp.from_torch(out.activation, dtype=wp.float32),
            ids.numel(),
        ],
        device=str(work.state_kernel_id.device),
    )


def launch_packed_scatter(work: FutureWorkload, out: VariantOutput) -> None:
    """Launch one packed primitive-family kernel and scatter to dense rows."""
    for primitive, kernel in enumerate(_packed_kernels()):
        _launch_packed_kernel(kernel, primitive, work, out)


def _launch_packed_local_kernel(kernel, primitive: int, work: FutureWorkload, out: PackedOutput) -> None:
    ids = work.packed_local_ids[primitive]
    if ids.numel() == 0:
        return
    wp.launch(
        kernel,
        dim=ids.numel(),
        inputs=[
            wp.from_torch(ids, dtype=wp.int32),
            wp.from_torch(work.packed_kind[primitive], dtype=wp.int32),
            wp.from_torch(work.state_kernel_scale, dtype=wp.float32),
            wp.from_torch(work.packed_src[primitive], dtype=wp.float32),
            wp.from_torch(work.packed_tgt[primitive], dtype=wp.float32),
            wp.from_torch(work.packed_act_param[primitive], dtype=wp.float32),
            wp.from_torch(out.delta[primitive], dtype=wp.float32),
            wp.from_torch(out.error[primitive], dtype=wp.float32),
            wp.from_torch(out.activation[primitive], dtype=wp.float32),
            ids.numel(),
        ],
        device=str(work.state_kernel_id.device),
    )


def launch_packed_local(work: FutureWorkload, out: PackedOutput) -> None:
    """Launch one packed primitive-family kernel and write packed local rows."""
    for primitive, kernel in enumerate(_packed_kernels()):
        _launch_packed_local_kernel(kernel, primitive, work, out)


def time_variant(name: str, fn, warmup: int, runs: int, graph: bool, device: str) -> float:
    """Time one benchmark variant."""
    for _ in range(warmup):
        fn()
    wp.synchronize()
    if graph:
        with wp.ScopedCapture(device=device) as capture:
            fn()
        graph_obj = capture.graph
        for _ in range(warmup):
            wp.capture_launch(graph_obj)
        wp.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            wp.capture_launch(graph_obj)
        wp.synchronize()
    else:
        start = time.perf_counter()
        for _ in range(runs):
            fn()
        wp.synchronize()
    ms = (time.perf_counter() - start) * 1000.0 / runs
    print(f"{name:>29}: {ms:8.4f} ms")
    return ms


def verify(work: FutureWorkload) -> None:
    """Check packed-scatter output against the mega-loop reference."""
    ref = make_output(work.state_kernel_id.numel(), work.state_kernel_id.device)
    packed = make_output(work.state_kernel_id.numel(), work.state_kernel_id.device)
    graph = make_output(work.state_kernel_id.numel(), work.state_kernel_id.device)
    launch_mega_loop(work, ref)
    launch_packed_scatter(work, packed)
    launch_primitive_graph(work, graph)
    wp.synchronize()
    torch.testing.assert_close(packed.error, ref.error, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(packed.activation, ref.activation, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(graph.error, ref.error, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(graph.activation, ref.activation, atol=1.0e-6, rtol=1.0e-6)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-work", type=int, default=1_048_576)
    parser.add_argument("--state-kernels", type=int, default=192)
    parser.add_argument("--pattern", choices=["random", "grouped", "skew"], default="random")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--graph-fanout", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the future heterogeneous dispatch benchmark."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    wp.init()
    device = torch.device("cuda:0")
    work = make_future_workload(args.n_work, args.state_kernels, args.pattern, args.seed, device, args.graph_fanout)
    if not args.no_verify:
        verify(work)

    mega_out = make_output(args.n_work, device)
    state_queue_out = make_output(args.n_work, device)
    primitive_queue_out = make_output(args.n_work, device)
    primitive_graph_out = make_output(args.n_work, device)
    packed_scatter_out = make_output(args.n_work, device)
    packed_local_out = make_packed_output(work)

    state_counts = torch.bincount(work.state_kernel_id.long(), minlength=args.state_kernels).detach().cpu()
    primitive_counts = [int(ids.numel()) for ids in work.primitive_queues]
    nonempty_state_queues = int((state_counts > 0).sum().item())

    print(
        "# future heterogeneous dispatch benchmark: "
        f"n_work={args.n_work}, state_kernels={args.state_kernels}, "
        f"primitive_families={NUM_PRIMITIVE_FAMILIES}, pattern={args.pattern}, "
        f"graph_fanout={args.graph_fanout}, graph={args.graph}"
    )
    print(f"# cuda={torch.cuda.get_device_name(0)}, warp={wp.__version__}, torch={torch.__version__}")
    print(f"# nonempty_state_queues={nonempty_state_queues}, primitive_launches={NUM_PRIMITIVE_FAMILIES}")
    print(f"# state_count_range=({int(state_counts.min())}, {int(state_counts.max())})")
    print(f"# primitive_counts={primitive_counts}")
    print(
        "# synthetic proxy labels: primitive_queue_local_synth ~= production "
        "primitive_queue_local dispatch; primitive_graph_local_synth ~= production "
        "primitive_graph_local dispatch."
    )
    print("# public backend comparison: run benchmark/bench_multi_task_command_backends.py --preset shared_direct")
    print(
        "# graph_nodes="
        f"vec3:{int(work.vec3_node_work_ids.numel())}, "
        f"reduce8:{int(work.reduce8_node_work_ids.numel())}, "
        f"reduce32:{int(work.reduce32_node_work_ids.numel())}, "
        f"contact:{int(work.contact_node_work_ids.numel())}, "
        f"local_frame:{int(work.local_frame_node_work_ids.numel())}"
    )

    t_mega = time_variant(
        "mega_loop",
        lambda: launch_mega_loop(work, mega_out),
        args.warmup,
        args.runs,
        args.graph,
        "cuda:0",
    )
    t_state = time_variant(
        "state_queue", lambda: launch_state_queue(work, state_queue_out), args.warmup, args.runs, args.graph, "cuda:0"
    )
    t_primitive = time_variant(
        "primitive_queue_local_synth",
        lambda: launch_primitive_queue(work, primitive_queue_out),
        args.warmup,
        args.runs,
        args.graph,
        "cuda:0",
    )
    t_primitive_graph = time_variant(
        "primitive_graph_local_synth",
        lambda: launch_primitive_graph(work, primitive_graph_out),
        args.warmup,
        args.runs,
        args.graph,
        "cuda:0",
    )
    t_scatter = time_variant(
        "packed_scatter",
        lambda: launch_packed_scatter(work, packed_scatter_out),
        args.warmup,
        args.runs,
        args.graph,
        "cuda:0",
    )
    t_local = time_variant(
        "packed_local",
        lambda: launch_packed_local(work, packed_local_out),
        args.warmup,
        args.runs,
        args.graph,
        "cuda:0",
    )
    print(
        f"# speedup state/mega={t_mega / t_state:.3f}x, "
        f"primitive_queue_local_synth/mega={t_mega / t_primitive:.3f}x, "
        f"primitive_graph_local_synth/mega={t_mega / t_primitive_graph:.3f}x, "
        f"scatter/mega={t_mega / t_scatter:.3f}x, "
        f"local/mega={t_mega / t_local:.3f}x"
    )


if __name__ == "__main__":
    main()
