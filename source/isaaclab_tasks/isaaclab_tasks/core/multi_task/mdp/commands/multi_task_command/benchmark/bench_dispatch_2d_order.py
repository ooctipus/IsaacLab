# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Infer how 2D env-slot launch ordering affects branch divergence.

Real ``dispatch_mega`` launches ``dim=(num_envs, k_max)``. If adjacent GPU
threads mostly share the same slot across envs, arranging slot ids by primitive
can reduce divergence without an explicit sorted work list. If adjacent threads
mostly walk slots within an env, slot ordering has a different effect.

Run with:

    ./isaaclab.sh -p -m \
        isaaclab_tasks.core.multi_task.mdp.commands.multi_task_command.benchmark.bench_dispatch_2d_order
"""

from __future__ import annotations

import argparse
import time

import torch
import warp as wp

NUM_KINDS = 8


@wp.func
def _branch_score(k: int, x: float) -> float:
    if k == 0:
        return wp.sin(x) + 0.1 * x
    if k == 1:
        return wp.cos(x * 0.7) - 0.2 * x
    if k == 2:
        return wp.sqrt(wp.abs(x) + 1.0)
    if k == 3:
        return wp.tanh(x) * wp.tanh(0.5 * x)
    if k == 4:
        return wp.atan2(x, x * x + 1.0)
    if k == 5:
        return wp.log(wp.abs(x) + 1.1)
    if k == 6:
        return wp.exp(-wp.abs(x) * 0.1)
    return x * x * 0.01 - x * 0.3


@wp.kernel
def branch_2d_kernel(
    kind: wp.array2d(dtype=wp.int32),
    source: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
):
    env, slot = wp.tid()
    out[env, slot] = _branch_score(int(kind[env, slot]), source[env, slot])


@wp.kernel
def branch_1d_kernel(
    kind: wp.array(dtype=wp.int32),
    source: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    out[i] = _branch_score(int(kind[i]), source[i])


def make_kinds(num_envs: int, k_max: int, pattern: str, num_tasks: int, device: torch.device) -> torch.Tensor:
    if pattern == "slot_homogeneous":
        slot_kind = torch.arange(k_max, device=device, dtype=torch.int32) % NUM_KINDS
        return slot_kind.unsqueeze(0).expand(num_envs, -1).contiguous()
    if pattern == "env_homogeneous":
        env_kind = torch.randint(0, NUM_KINDS, (num_envs,), device=device, dtype=torch.int32)
        return env_kind.unsqueeze(1).expand(-1, k_max).contiguous()
    if pattern == "task_slots":
        task_slot_kinds = torch.randint(0, NUM_KINDS, (num_tasks, k_max), device=device, dtype=torch.int32)
        task_ids = torch.randint(0, num_tasks, (num_envs,), device=device, dtype=torch.int64)
        return task_slot_kinds[task_ids].contiguous()
    if pattern == "task_slots_sorted":
        task_slot_kinds = torch.randint(0, NUM_KINDS, (num_tasks, k_max), device=device, dtype=torch.int32)
        task_ids = torch.arange(num_envs, device=device, dtype=torch.int64) % num_tasks
        task_ids = torch.sort(task_ids).values
        return task_slot_kinds[task_ids].contiguous()
    return torch.randint(0, NUM_KINDS, (num_envs, k_max), device=device, dtype=torch.int32)


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
    print(f"{name:>18}: {ms:8.4f} ms")
    return ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=131_072)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--num-tasks", type=int, default=64)
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
    source = torch.randn(args.num_envs, args.k_max, device=device, dtype=torch.float32)
    out2d = torch.empty_like(source)
    out1d = torch.empty(args.num_envs * args.k_max, device=device, dtype=torch.float32)
    flat_source = source.flatten().contiguous()
    wp_source = wp.from_torch(source, dtype=wp.float32)
    wp_flat_source = wp.from_torch(flat_source, dtype=wp.float32)
    wp_out2d = wp.from_torch(out2d, dtype=wp.float32)
    wp_out1d = wp.from_torch(out1d, dtype=wp.float32)

    print(
        f"# 2d order benchmark: envs={args.num_envs}, k_max={args.k_max}, "
        f"tasks={args.num_tasks}, work={args.num_envs * args.k_max}"
    )
    print(f"# cuda={torch.cuda.get_device_name(0)}, warp={wp.__version__}, torch={torch.__version__}")

    for pattern in ["slot_homogeneous", "env_homogeneous", "task_slots", "task_slots_sorted", "random"]:
        kind = make_kinds(args.num_envs, args.k_max, pattern, args.num_tasks, device)
        flat_kind = kind.flatten().contiguous()
        wp_kind = wp.from_torch(kind, dtype=wp.int32)
        wp_flat_kind = wp.from_torch(flat_kind, dtype=wp.int32)
        print(f"# pattern={pattern}")
        time_warp_graph(
            "2d launch",
            lambda: wp.launch(
                branch_2d_kernel,
                dim=(args.num_envs, args.k_max),
                inputs=[wp_kind, wp_source, wp_out2d],
                device=str(device),
            ),
            args.warmup,
            args.runs,
            device,
        )
        time_warp_graph(
            "flat launch",
            lambda: wp.launch(
                branch_1d_kernel,
                dim=args.num_envs * args.k_max,
                inputs=[wp_flat_kind, wp_flat_source, wp_out1d],
                device=str(device),
            ),
            args.warmup,
            args.runs,
            device,
        )


if __name__ == "__main__":
    main()
