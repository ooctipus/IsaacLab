# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark queue construction strategies for heterogeneous dispatch.

The dispatch benchmark assumes prebuilt work queues. This file isolates the
cost of building those queues from a per-work-item ``kind`` vector.

Run with:

    ./isaaclab.sh -p -m isaaclab_tasks.core.multi_task.mdp.commands.benchmark.bench_dispatch_queue_build
"""

from __future__ import annotations

import argparse
import time

import torch
import warp as wp
import warp.utils as wpu

NUM_KINDS = 64
NUM_PRIMITIVES = 8
KINDS_PER_PRIMITIVE = NUM_KINDS // NUM_PRIMITIVES


@wp.kernel
def clear_counts_kernel(counts: wp.array(dtype=wp.int32), num_groups: int):
    i = wp.tid()
    if i < num_groups:
        counts[i] = 0


@wp.kernel
def primitive_count_kernel(
    kind: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    kinds_per_primitive: int,
):
    i = wp.tid()
    p = int(kind[i]) // kinds_per_primitive
    wp.atomic_add(counts, p, 1)


@wp.kernel
def primitive_queue_kernel(
    kind: wp.array(dtype=wp.int32),
    queues: wp.array2d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    kinds_per_primitive: int,
):
    i = wp.tid()
    p = int(kind[i]) // kinds_per_primitive
    q = wp.atomic_add(counts, p, 1)
    queues[p, q] = i


@wp.kernel
def kind_queue_kernel(
    kind: wp.array(dtype=wp.int32),
    queues: wp.array2d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    k = int(kind[i])
    q = wp.atomic_add(counts, k, 1)
    queues[k, q] = i


@wp.kernel
def prepare_primitive_sort_kernel(
    kind: wp.array(dtype=wp.int32),
    keys: wp.array(dtype=wp.int64),
    values: wp.array(dtype=wp.int32),
    kinds_per_primitive: int,
):
    i = wp.tid()
    keys[i] = wp.int64(int(kind[i]) // kinds_per_primitive)
    values[i] = i


@wp.kernel
def prepare_kind_sort_kernel(
    kind: wp.array(dtype=wp.int32),
    keys: wp.array(dtype=wp.int64),
    values: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    keys[i] = wp.int64(kind[i])
    values[i] = i


@wp.kernel
def mark_segment_starts_kernel(
    keys: wp.array(dtype=wp.int64),
    flags: wp.array(dtype=wp.int32),
    count: int,
):
    i = wp.tid()
    if i >= count:
        return
    flag = int(0)
    if i == 0 or keys[i] != keys[i - 1]:
        flag = 1
    flags[i] = flag


def make_kind_ids(n_work: int, num_kinds: int, pattern: str, device: torch.device) -> torch.Tensor:
    if pattern == "grouped":
        n_per_kind = (n_work + num_kinds - 1) // num_kinds
        base = torch.arange(num_kinds, device=device, dtype=torch.int32).repeat_interleave(n_per_kind)
        return base[:n_work].clone()
    if pattern == "skew":
        ranks = torch.arange(1, num_kinds + 1, device=device, dtype=torch.float32)
        probs = 1.0 / ranks
        probs = probs / probs.sum()
        return torch.multinomial(probs, n_work, replacement=True).to(torch.int32)
    return torch.randint(0, num_kinds, (n_work,), device=device, dtype=torch.int32)


def time_cuda(name: str, fn, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / runs
    print(f"{name:>26}: {ms:8.4f} ms")
    return ms


def time_warp_graph(name: str, fn, warmup: int, runs: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    wp.synchronize()
    with wp.ScopedCapture(device=str(device)) as capture:
        fn()
    graph = capture.graph
    for _ in range(warmup):
        wp.capture_launch(graph)
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        wp.capture_launch(graph)
    wp.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / runs
    print(f"{name:>26}: {ms:8.4f} ms")
    return ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-work", type=int, default=1_048_576)
    parser.add_argument("--num-kinds", type=int, default=64, choices=[8, 16, 32, 64])
    parser.add_argument("--pattern", choices=["random", "grouped", "skew"], default="random")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    wp.init()
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    kind = make_kind_ids(args.n_work, args.num_kinds, args.pattern, device)
    wp_kind = wp.from_torch(kind, dtype=wp.int32)

    primitive_count = max(1, (args.num_kinds + KINDS_PER_PRIMITIVE - 1) // KINDS_PER_PRIMITIVE)
    primitive_queues = torch.empty((primitive_count, args.n_work), device=device, dtype=torch.int32)
    primitive_counts = torch.empty(primitive_count, device=device, dtype=torch.int32)
    kind_queues = torch.empty((args.num_kinds, args.n_work), device=device, dtype=torch.int32)
    kind_counts = torch.empty(args.num_kinds, device=device, dtype=torch.int32)
    sort_keys = torch.empty(2 * args.n_work, device=device, dtype=torch.int64)
    sort_values = torch.empty(2 * args.n_work, device=device, dtype=torch.int32)
    segment_flags = torch.empty(args.n_work, device=device, dtype=torch.int32)
    segment_scan = torch.empty(args.n_work, device=device, dtype=torch.int32)

    wp_primitive_queues = wp.from_torch(primitive_queues, dtype=wp.int32)
    wp_primitive_counts = wp.from_torch(primitive_counts, dtype=wp.int32)
    wp_kind_queues = wp.from_torch(kind_queues, dtype=wp.int32)
    wp_kind_counts = wp.from_torch(kind_counts, dtype=wp.int32)
    wp_sort_keys = wp.from_torch(sort_keys, dtype=wp.int64)
    wp_sort_values = wp.from_torch(sort_values, dtype=wp.int32)
    wp_segment_flags = wp.from_torch(segment_flags, dtype=wp.int32)
    wp_segment_scan = wp.from_torch(segment_scan, dtype=wp.int32)

    print(f"# queue build benchmark: n_work={args.n_work}, num_kinds={args.num_kinds}, pattern={args.pattern}")
    print(f"# cuda={torch.cuda.get_device_name(0)}, warp={wp.__version__}, torch={torch.__version__}")

    time_cuda(
        "torch kind nonzero",
        lambda: [
            (kind == k).nonzero(as_tuple=False).flatten().to(torch.int32).contiguous() for k in range(args.num_kinds)
        ],
        args.warmup,
        max(10, args.runs // 2),
    )
    time_cuda(
        "torch primitive nonzero",
        lambda: [
            ((kind >= p * KINDS_PER_PRIMITIVE) & (kind < (p + 1) * KINDS_PER_PRIMITIVE))
            .nonzero(as_tuple=False)
            .flatten()
            .to(torch.int32)
            .contiguous()
            for p in range(primitive_count)
        ],
        args.warmup,
        args.runs,
    )
    time_cuda("torch argsort(kind)", lambda: torch.argsort(kind), args.warmup, max(10, args.runs // 2))
    time_cuda(
        "torch bincount(kind)",
        lambda: torch.bincount(kind.to(torch.int64), minlength=args.num_kinds),
        args.warmup,
        args.runs,
    )

    time_warp_graph(
        "warp primitive counts",
        lambda: (
            wp.launch(
                clear_counts_kernel,
                dim=primitive_count,
                inputs=[wp_primitive_counts, primitive_count],
                device=str(device),
            ),
            wp.launch(
                primitive_count_kernel,
                dim=args.n_work,
                inputs=[wp_kind, wp_primitive_counts, KINDS_PER_PRIMITIVE],
                device=str(device),
            ),
        ),
        args.warmup,
        args.runs,
        device,
    )
    time_warp_graph(
        "warp primitive queues",
        lambda: (
            wp.launch(
                clear_counts_kernel,
                dim=primitive_count,
                inputs=[wp_primitive_counts, primitive_count],
                device=str(device),
            ),
            wp.launch(
                primitive_queue_kernel,
                dim=args.n_work,
                inputs=[wp_kind, wp_primitive_queues, wp_primitive_counts, KINDS_PER_PRIMITIVE],
                device=str(device),
            ),
        ),
        args.warmup,
        args.runs,
        device,
    )
    time_warp_graph(
        "warp kind queues",
        lambda: (
            wp.launch(
                clear_counts_kernel,
                dim=args.num_kinds,
                inputs=[wp_kind_counts, args.num_kinds],
                device=str(device),
            ),
            wp.launch(
                kind_queue_kernel,
                dim=args.n_work,
                inputs=[wp_kind, wp_kind_queues, wp_kind_counts],
                device=str(device),
            ),
        ),
        args.warmup,
        max(10, args.runs // 2),
        device,
    )
    time_warp_graph(
        "warp primitive sort",
        lambda: (
            wp.launch(
                prepare_primitive_sort_kernel,
                dim=args.n_work,
                inputs=[wp_kind, wp_sort_keys, wp_sort_values, KINDS_PER_PRIMITIVE],
                device=str(device),
            ),
            wpu.radix_sort_pairs(wp_sort_keys, wp_sort_values, args.n_work),
        ),
        args.warmup,
        args.runs,
        device,
    )
    time_warp_graph(
        "warp primitive segment",
        lambda: (
            wp.launch(
                prepare_primitive_sort_kernel,
                dim=args.n_work,
                inputs=[wp_kind, wp_sort_keys, wp_sort_values, KINDS_PER_PRIMITIVE],
                device=str(device),
            ),
            wpu.radix_sort_pairs(wp_sort_keys, wp_sort_values, args.n_work),
            wp.launch(
                mark_segment_starts_kernel,
                dim=args.n_work,
                inputs=[wp_sort_keys, wp_segment_flags, args.n_work],
                device=str(device),
            ),
            wpu.array_scan(wp_segment_flags, wp_segment_scan, inclusive=True),
        ),
        args.warmup,
        args.runs,
        device,
    )
    time_warp_graph(
        "warp kind sort",
        lambda: (
            wp.launch(
                prepare_kind_sort_kernel,
                dim=args.n_work,
                inputs=[wp_kind, wp_sort_keys, wp_sort_values],
                device=str(device),
            ),
            wpu.radix_sort_pairs(wp_sort_keys, wp_sort_values, args.n_work),
        ),
        args.warmup,
        args.runs,
        device,
    )


if __name__ == "__main__":
    main()
