# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark dense command-buffer clearing, scatter, and compose layout.

The dispatch homogeneity benchmark isolates the projection/error kernel. The
real command path also clears dense buffers, scatters canonical command rows,
and runs a per-env composer. This benchmark measures those layout costs.

Run with:

    ./isaaclab.sh -p -m isaaclab_tasks.core.multi_task.mdp.commands.benchmark.bench_command_pipeline_layout
"""

from __future__ import annotations

import argparse
import time

import torch
import warp as wp


@wp.kernel
def zero2d_kernel(data: wp.array2d(dtype=wp.float32), rows: int, cols: int):
    i = wp.tid()
    total = rows * cols
    if i < total:
        row = i // cols
        col = i - row * cols
        data[row, col] = 0.0


@wp.kernel
def zero_bool2d_kernel(data: wp.array2d(dtype=wp.bool), rows: int, cols: int):
    i = wp.tid()
    total = rows * cols
    if i < total:
        row = i // cols
        col = i - row * cols
        data[row, col] = False


@wp.kernel
def zero_rows2d_kernel(
    data: wp.array2d(dtype=wp.float32),
    row_ids: wp.array(dtype=wp.int32),
    row_count: int,
    cols: int,
):
    i = wp.tid()
    total = row_count * cols
    if i < total:
        local_row = i // cols
        col = i - local_row * cols
        row = int(row_ids[local_row])
        data[row, col] = 0.0


@wp.kernel
def dense_dispatch_synthetic_kernel(
    slot_count: wp.array(dtype=wp.int32),
    source: wp.array2d(dtype=wp.float32),
    buf_error: wp.array2d(dtype=wp.float32),
    buf_activation: wp.array2d(dtype=wp.float32),
    command_reach: wp.array2d(dtype=wp.float32),
    command_track: wp.array2d(dtype=wp.float32),
    k_max: int,
    command_width: int,
):
    env, slot = wp.tid()
    if slot >= slot_count[env]:
        return

    base = source[env, slot]
    stride = slot % 4 + 1
    canon_limit = command_width - 4
    canon_off = int(0)
    if canon_limit > 0:
        canon_off = (slot * 4) % canon_limit
    instant = slot & 1

    d0 = wp.sin(base + float(slot) * 0.031)
    d1 = wp.cos(base * 0.7 + float(env & 255) * 0.001)
    d2 = d0 * d1
    d3 = d0 - d1
    err = wp.sqrt(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3)
    act = 1.0 - wp.tanh(err)
    buf_error[env, slot] = err
    buf_activation[env, slot] = act

    if instant != 0:
        command_reach[env, canon_off] = d0
        if stride >= 2:
            command_reach[env, canon_off + 1] = d1
        if stride >= 3:
            command_reach[env, canon_off + 2] = d2
        if stride >= 4:
            command_reach[env, canon_off + 3] = d3
    else:
        command_track[env, canon_off] = d0
        if stride >= 2:
            command_track[env, canon_off + 1] = d1
        if stride >= 3:
            command_track[env, canon_off + 2] = d2
        if stride >= 4:
            command_track[env, canon_off + 3] = d3


@wp.kernel
def compose_env_loop_synthetic_kernel(
    slot_count: wp.array(dtype=wp.int32),
    buf_activation: wp.array2d(dtype=wp.float32),
    sum_activation: wp.array2d(dtype=wp.float32),
    instant_achieved: wp.array2d(dtype=wp.bool),
    transit_steps: wp.array(dtype=wp.int32),
    reward: wp.array(dtype=wp.float32),
    success: wp.array(dtype=wp.bool),
    progress: wp.array(dtype=wp.float32),
    instant_threshold: float,
    quality_easing: float,
):
    env = wp.tid()
    n_slots = slot_count[env]
    transit_steps[env] = transit_steps[env] + 1
    tsteps = float(transit_steps[env])

    activation_sum = float(0.0)
    quality_product = float(1.0)
    all_instant_ok = int(1)
    has_instant = int(0)

    for slot in range(n_slots):
        act = buf_activation[env, slot]
        activation_sum = activation_sum + act
        new_sum = sum_activation[env, slot] + act
        sum_activation[env, slot] = new_sum

        if (slot & 1) != 0:
            has_instant = 1
            achieved = int(instant_achieved[env, slot])
            if act > instant_threshold:
                achieved = 1
            instant_achieved[env, slot] = achieved != 0
            if achieved == 0:
                all_instant_ok = 0
        else:
            quality_product = quality_product * (new_sum / tsteps)

    reward[env] = float(all_instant_ok) * wp.pow(quality_product, quality_easing)
    success[env] = (all_instant_ok * has_instant) != 0
    progress_val = float(0.0)
    if n_slots > 0:
        progress_val = activation_sum / float(n_slots)
    progress[env] = progress_val


@wp.kernel
def progress_slot_atomic_kernel(
    slot_count: wp.array(dtype=wp.int32),
    buf_activation: wp.array2d(dtype=wp.float32),
    progress_sum: wp.array(dtype=wp.float32),
    progress: wp.array(dtype=wp.float32),
):
    env, slot = wp.tid()
    if slot >= slot_count[env]:
        return
    wp.atomic_add(progress_sum, env, buf_activation[env, slot])
    if slot == 0:
        # This intentionally races with other slots unless launched after a
        # separate reduction pass; it is here only to keep the atomic writer hot.
        # The benchmark times the atomic pressure, not the final value.
        progress[env] = progress_sum[env] / float(slot_count[env])


def time_cuda(name: str, fn, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0 / runs
    print(f"{name:>24}: {ms:8.4f} ms")
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
    print(f"{name:>24}: {ms:8.4f} ms")
    return ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=16_384)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--command-width", type=int, default=256)
    parser.add_argument("--clear-envs", type=int, default=1024)
    parser.add_argument("--pattern", choices=["full", "half", "random"], default="full")
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

    if args.pattern == "full":
        slot_count = torch.full((args.num_envs,), args.k_max, device=device, dtype=torch.int32)
    elif args.pattern == "half":
        slot_count = torch.full((args.num_envs,), max(1, args.k_max // 2), device=device, dtype=torch.int32)
    else:
        slot_count = torch.randint(1, args.k_max + 1, (args.num_envs,), device=device, dtype=torch.int32)

    source = torch.randn(args.num_envs, args.k_max, device=device, dtype=torch.float32)
    buf_error = torch.empty(args.num_envs, args.k_max, device=device, dtype=torch.float32)
    buf_activation = torch.empty(args.num_envs, args.k_max, device=device, dtype=torch.float32)
    command_reach = torch.empty(args.num_envs, args.command_width, device=device, dtype=torch.float32)
    command_track = torch.empty(args.num_envs, args.command_width, device=device, dtype=torch.float32)
    sum_activation = torch.zeros(args.num_envs, args.k_max, device=device, dtype=torch.float32)
    instant_achieved = torch.zeros(args.num_envs, args.k_max, device=device, dtype=torch.bool)
    transit_steps = torch.zeros(args.num_envs, device=device, dtype=torch.int32)
    reward = torch.empty(args.num_envs, device=device, dtype=torch.float32)
    success = torch.empty(args.num_envs, device=device, dtype=torch.bool)
    progress = torch.empty(args.num_envs, device=device, dtype=torch.float32)
    progress_sum = torch.empty(args.num_envs, device=device, dtype=torch.float32)
    progress_sum_2d = progress_sum.view(args.num_envs, 1)
    clear_count = min(args.num_envs, max(1, args.clear_envs))
    clear_env_ids = torch.arange(clear_count, device=device, dtype=torch.int32)

    wp_slot_count = wp.from_torch(slot_count, dtype=wp.int32)
    wp_source = wp.from_torch(source, dtype=wp.float32)
    wp_buf_error = wp.from_torch(buf_error, dtype=wp.float32)
    wp_buf_activation = wp.from_torch(buf_activation, dtype=wp.float32)
    wp_command_reach = wp.from_torch(command_reach, dtype=wp.float32)
    wp_command_track = wp.from_torch(command_track, dtype=wp.float32)
    wp_sum_activation = wp.from_torch(sum_activation, dtype=wp.float32)
    wp_instant_achieved = wp.from_torch(instant_achieved, dtype=wp.bool)
    wp_transit_steps = wp.from_torch(transit_steps, dtype=wp.int32)
    wp_reward = wp.from_torch(reward, dtype=wp.float32)
    wp_success = wp.from_torch(success, dtype=wp.bool)
    wp_progress = wp.from_torch(progress, dtype=wp.float32)
    wp_progress_sum = wp.from_torch(progress_sum, dtype=wp.float32)
    wp_progress_sum_2d = wp.from_torch(progress_sum_2d, dtype=wp.float32)
    wp_clear_env_ids = wp.from_torch(clear_env_ids, dtype=wp.int32)

    print(
        f"# command pipeline layout benchmark: envs={args.num_envs}, k_max={args.k_max}, "
        f"command_width={args.command_width}, pattern={args.pattern}"
    )
    print(f"# cuda={torch.cuda.get_device_name(0)}, warp={wp.__version__}, torch={torch.__version__}")

    def zero_all_warp() -> None:
        wp.launch(
            zero2d_kernel,
            dim=buf_error.numel(),
            inputs=[wp_buf_error, args.num_envs, args.k_max],
            device=str(device),
        )
        wp.launch(
            zero2d_kernel,
            dim=buf_activation.numel(),
            inputs=[wp_buf_activation, args.num_envs, args.k_max],
            device=str(device),
        )
        wp.launch(
            zero2d_kernel,
            dim=command_reach.numel(),
            inputs=[wp_command_reach, args.num_envs, args.command_width],
            device=str(device),
        )
        wp.launch(
            zero2d_kernel,
            dim=command_track.numel(),
            inputs=[wp_command_track, args.num_envs, args.command_width],
            device=str(device),
        )

    def zero_slot_warp() -> None:
        wp.launch(
            zero2d_kernel,
            dim=buf_error.numel(),
            inputs=[wp_buf_error, args.num_envs, args.k_max],
            device=str(device),
        )
        wp.launch(
            zero2d_kernel,
            dim=buf_activation.numel(),
            inputs=[wp_buf_activation, args.num_envs, args.k_max],
            device=str(device),
        )

    def zero_command_rows_warp() -> None:
        wp.launch(
            zero_rows2d_kernel,
            dim=clear_count * args.command_width,
            inputs=[wp_command_reach, wp_clear_env_ids, clear_count, args.command_width],
            device=str(device),
        )
        wp.launch(
            zero_rows2d_kernel,
            dim=clear_count * args.command_width,
            inputs=[wp_command_track, wp_clear_env_ids, clear_count, args.command_width],
            device=str(device),
        )

    def dispatch_warp() -> None:
        wp.launch(
            dense_dispatch_synthetic_kernel,
            dim=(args.num_envs, args.k_max),
            inputs=[
                wp_slot_count,
                wp_source,
                wp_buf_error,
                wp_buf_activation,
                wp_command_reach,
                wp_command_track,
                args.k_max,
                args.command_width,
            ],
            device=str(device),
        )

    def compose_warp() -> None:
        wp.launch(
            compose_env_loop_synthetic_kernel,
            dim=args.num_envs,
            inputs=[
                wp_slot_count,
                wp_buf_activation,
                wp_sum_activation,
                wp_instant_achieved,
                wp_transit_steps,
                wp_reward,
                wp_success,
                wp_progress,
                0.5,
                1.0,
            ],
            device=str(device),
        )

    def zero_progress_warp() -> None:
        wp.launch(
            zero2d_kernel,
            dim=args.num_envs,
            inputs=[wp_progress_sum_2d, args.num_envs, 1],
            device=str(device),
        )

    def progress_atomic_warp() -> None:
        zero_progress_warp()
        wp.launch(
            progress_slot_atomic_kernel,
            dim=(args.num_envs, args.k_max),
            inputs=[wp_slot_count, wp_buf_activation, wp_progress_sum, wp_progress],
            device=str(device),
        )

    time_cuda(
        "torch zero all",
        lambda: (buf_error.zero_(), buf_activation.zero_(), command_reach.zero_(), command_track.zero_()),
        args.warmup,
        args.runs,
    )
    time_warp_graph("warp zero all", zero_all_warp, args.warmup, args.runs, device)
    time_warp_graph("warp zero slots", zero_slot_warp, args.warmup, args.runs, device)
    time_warp_graph("warp zero cmd rows", zero_command_rows_warp, args.warmup, args.runs, device)
    time_warp_graph("dispatch only", dispatch_warp, args.warmup, args.runs, device)
    time_warp_graph("compose env loop", compose_warp, args.warmup, args.runs, device)
    time_warp_graph("progress atomic", progress_atomic_warp, args.warmup, args.runs, device)
    time_warp_graph(
        "pipeline full zero",
        lambda: (zero_all_warp(), dispatch_warp(), compose_warp()),
        args.warmup,
        args.runs,
        device,
    )
    time_warp_graph(
        "pipeline slot zero",
        lambda: (zero_slot_warp(), dispatch_warp(), compose_warp()),
        args.warmup,
        args.runs,
        device,
    )
    time_warp_graph(
        "pipeline row zero",
        lambda: (zero_slot_warp(), zero_command_rows_warp(), dispatch_warp(), compose_warp()),
        args.warmup,
        args.runs,
        device,
    )
    time_warp_graph("pipeline no zero", lambda: (dispatch_warp(), compose_warp()), args.warmup, args.runs, device)


if __name__ == "__main__":
    main()
