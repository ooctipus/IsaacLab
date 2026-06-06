# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark Torch vs Warp body-frame rotation for canonical command slots."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
import warp as wp

from isaaclab.utils.math import quat_apply_inverse

from ..impl.kernels_wp import rotate_canonical_vec3_pair


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


def _make_quat(num_envs: int, device: torch.device) -> torch.Tensor:
    quat = torch.randn(num_envs, 4, device=device)
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def _bench_case(num_envs: int, num_offsets: int, repeat: int, device: torch.device) -> tuple[float, float, float]:
    offsets = torch.arange(num_offsets, device=device, dtype=torch.int32) * 3
    width = int(offsets[-1].item()) + 3
    quat = _make_quat(num_envs, device)
    base = torch.randn(num_envs, width, device=device)
    torch_out = base.clone()
    warp_out = base.clone()

    def torch_rotate() -> None:
        for off in offsets.tolist():
            torch_out[:, off : off + 3] = quat_apply_inverse(quat, torch_out[:, off : off + 3])

    wp_quat = wp.from_torch(quat, dtype=wp.quat)
    wp_offsets = wp.from_torch(offsets)
    wp_warp_out = wp.from_torch(warp_out)
    empty_offsets = torch.empty(0, device=device, dtype=torch.int32)
    wp_empty_offsets = wp.from_torch(empty_offsets)

    def warp_rotate() -> None:
        wp.launch(
            rotate_canonical_vec3_pair,
            dim=(num_envs, num_offsets),
            inputs=[wp_quat, wp_warp_out, wp_offsets, num_offsets, wp_warp_out, wp_empty_offsets],
            device=str(device),
        )

    torch_rotate()
    warp_rotate()
    torch.cuda.synchronize()
    max_err = (torch_out - warp_out).abs().max().item()

    torch_out.copy_(base)
    warp_out.copy_(base)
    torch_ms = _time_cuda(torch_rotate, repeat)
    warp_ms = _time_cuda(warp_rotate, repeat)
    return torch_ms, warp_ms, max_err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, nargs="+", default=[16384, 131072])
    parser.add_argument("--num_offsets", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    wp.init()
    device = torch.device(args.device)
    print("| envs | offsets | torch ms | warp ms | speedup | max err |")
    print("|---:|---:|---:|---:|---:|---:|")
    for num_envs in args.num_envs:
        for num_offsets in args.num_offsets:
            torch_ms, warp_ms, max_err = _bench_case(num_envs, num_offsets, args.repeat, device)
            print(
                f"| {num_envs} | {num_offsets} | {torch_ms:.4f} | {warp_ms:.4f} | "
                f"{torch_ms / warp_ms:.2f}x | {max_err:.3e} |"
            )


if __name__ == "__main__":
    main()
