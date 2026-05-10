# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark command dispatch backends through the public ``MultiTaskCommand`` API."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from unittest.mock import patch

import torch

if __package__:
    from .mock_command import build_mock_command
else:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from mock_command import build_mock_command


def _sync(device: str) -> None:
    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.synchronize()


def _run_steps(command, env, num_steps: int) -> None:
    for _ in range(num_steps):
        command._update_command()
        env.episode_length_buf += 1


def _time_backend(
    dispatch_backend: str, num_envs: int, device: str, warmup: int, runs: int, preset: str | None
) -> float:
    torch.manual_seed(0)
    command, env, readers, mtc_mod = build_mock_command(
        num_envs, device, dispatch_backend=dispatch_backend, preset=preset
    )
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", readers):
        _run_steps(command, env, warmup)
        _sync(device)

        if torch.cuda.is_available() and "cuda" in device:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _run_steps(command, env, runs)
            end.record()
            end.synchronize()
            return float(start.elapsed_time(end)) / runs

        start_t = time.perf_counter()
        _run_steps(command, env, runs)
        return (time.perf_counter() - start_t) * 1000.0 / runs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="all", help="Backend string, or 'all'.")
    parser.add_argument("--num_envs", type=int, default=16384, help="Parallel envs.")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup command updates.")
    parser.add_argument("--runs", type=int, default=1000, help="Timed command updates.")
    parser.add_argument(
        "--preset",
        default=None,
        help=(
            "Optional MultiTaskTasksPresetCfg name, 'shared_direct' for a small high-fanout mock workload, "
            "'future_synthetic' for a wide exact-backend stress workload, or "
            "'future_synthetic_heavy' for the same workload with heavier shared producers."
        ),
    )
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device for mock-mode tensors.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    backends = (
        [
            "torch",
            "mega_kernel",
            "schedule_ordered_mega",
            "packed_scatter",
            "primitive_queue_local",
            "primitive_graph_local",
        ]
        if args.backend == "all"
        else [args.backend]
    )

    print(
        "# MultiTaskCommand public backend benchmark: "
        f"num_envs={args.num_envs}, warmup={args.warmup}, runs={args.runs}, "
        f"device={args.device}, preset={args.preset or 'default'}"
    )
    print(f"{'backend':<20s} {'ms/update':>12s} {'us/env':>12s}")
    for backend in backends:
        ms = _time_backend(backend, args.num_envs, args.device, args.warmup, args.runs, args.preset)
        print(f"{backend:<20s} {ms:>12.4f} {ms * 1000.0 / args.num_envs:>12.4f}")


if __name__ == "__main__":
    main()
