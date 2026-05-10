# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark dynamic slab source binding vs prebound Warp handles."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch
import warp as wp

from ..impl.kernels_wp import fill_slab_copy


def _time_wall(fn: Callable[[], None], repeat: int, device: str) -> float:
    wp.synchronize_device(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    wp.synchronize_device(device)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / repeat


def _time_cuda(fn: Callable[[], None], repeat: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16384)
    parser.add_argument("--slab_sizes", type=int, nargs="+", default=[36, 48, 36, 36, 16, 16])
    parser.add_argument("--repeat", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    wp.init()
    device = torch.device(args.device)
    sources = [torch.randn(args.num_envs, size, device=device) for size in args.slab_sizes]
    offsets: list[int] = []
    cursor = 0
    for size in args.slab_sizes:
        offsets.append(cursor)
        cursor += size
    unified = torch.empty(args.num_envs, cursor, device=device)

    readers = [lambda source=source: source for source in sources]
    wp_unified = wp.from_torch(unified)
    prebound = [
        (wp.from_torch(source), offset, size) for source, offset, size in zip(sources, offsets, args.slab_sizes)
    ]

    def dynamic_bind_step() -> None:
        for reader, offset, size in zip(readers, offsets, args.slab_sizes):
            raw = reader()
            source = raw.reshape(args.num_envs, size)
            wp.launch(
                fill_slab_copy,
                dim=(args.num_envs, size),
                inputs=[wp.from_torch(source), wp_unified, offset],
                device=args.device,
            )

    def prebound_step() -> None:
        for wp_source, offset, size in prebound:
            wp.launch(
                fill_slab_copy,
                dim=(args.num_envs, size),
                inputs=[wp_source, wp_unified, offset],
                device=args.device,
            )

    dynamic_bind_step()
    prebound_step()
    wp.synchronize_device(args.device)

    with wp.ScopedCapture(device=args.device) as capture:
        prebound_step()
    graph = capture.graph

    def graph_step() -> None:
        wp.capture_launch(graph)

    print("| path | wall ms/step | cuda event ms/step |")
    print("|---|---:|---:|")
    for name, fn in (
        ("dynamic reader + from_torch", dynamic_bind_step),
        ("prebound wp arrays", prebound_step),
        ("prebound graph replay", graph_step),
    ):
        wall_ms = _time_wall(fn, args.repeat, args.device)
        cuda_ms = _time_cuda(fn, args.repeat)
        print(f"| {name} | {wall_ms:.4f} | {cuda_ms:.4f} |")


if __name__ == "__main__":
    main()
