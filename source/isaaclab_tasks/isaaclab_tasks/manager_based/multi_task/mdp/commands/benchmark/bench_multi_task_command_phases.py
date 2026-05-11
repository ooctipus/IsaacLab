# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase benchmark for public ``MultiTaskCommand`` backends."""

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


def _run_warmup(command, env, num_steps: int) -> None:
    for _ in range(num_steps):
        command._update_command()
        env.episode_length_buf += 1


def _time_cuda_phases(command, env, runs: int) -> dict[str, float]:
    phase_names = ("slot_mask", "reset_step", "dispatch", "compose", "episode_inc")
    phase_events = {name: [] for name in phase_names}
    for _ in range(runs):
        events = [torch.cuda.Event(enable_timing=True) for _ in range(len(phase_names) + 1)]
        events[0].record()
        torch.lt(command._slot_arange, command._env_slot_count.unsqueeze(1), out=command._slot_valid)
        events[1].record()
        command._buf_error.zero_()
        command._buf_activation.zero_()
        events[2].record()
        command._dispatch(command._slot_valid)
        events[3].record()
        command._compose(command._slot_valid)
        events[4].record()
        env.episode_length_buf += 1
        events[5].record()
        for name, start, end in zip(phase_names, events[:-1], events[1:]):
            phase_events[name].append((start, end))
    torch.cuda.synchronize()
    return {name: sum(start.elapsed_time(end) for start, end in events) / runs for name, events in phase_events.items()}


def _time_cpu_phases(command, env, runs: int) -> dict[str, float]:
    totals = {"slot_mask": 0.0, "reset_step": 0.0, "dispatch": 0.0, "compose": 0.0, "episode_inc": 0.0}
    for _ in range(runs):
        t0 = time.perf_counter()
        torch.lt(command._slot_arange, command._env_slot_count.unsqueeze(1), out=command._slot_valid)
        t1 = time.perf_counter()
        command._buf_error.zero_()
        command._buf_activation.zero_()
        t2 = time.perf_counter()
        command._dispatch(command._slot_valid)
        t3 = time.perf_counter()
        command._compose(command._slot_valid)
        t4 = time.perf_counter()
        env.episode_length_buf += 1
        t5 = time.perf_counter()
        totals["slot_mask"] += t1 - t0
        totals["reset_step"] += t2 - t1
        totals["dispatch"] += t3 - t2
        totals["compose"] += t4 - t3
        totals["episode_inc"] += t5 - t4
    return {name: value * 1000.0 / runs for name, value in totals.items()}


def _time_backend(backend: str, args: argparse.Namespace) -> dict[str, float]:
    torch.manual_seed(0)
    command, env, readers, mtc_mod = build_mock_command(
        args.num_envs,
        args.device,
        dispatch_backend=backend,
        preset=args.preset,
    )
    with patch.object(mtc_mod, "BUFFER_KIND_READERS", readers):
        _run_warmup(command, env, args.warmup)
        if torch.cuda.is_available() and "cuda" in str(args.device):
            return _time_cuda_phases(command, env, args.runs)
        return _time_cpu_phases(command, env, args.runs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="all", help="Backend string, or 'all'.")
    parser.add_argument("--num_envs", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--preset", default="future_synthetic")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    backends = (
        [
            "mega_kernel",
            "packed_scatter",
            "primitive_queue_local",
            "primitive_graph_local",
        ]
        if args.backend == "all"
        else [args.backend]
    )
    print(
        "# MultiTaskCommand phase benchmark: "
        f"num_envs={args.num_envs}, warmup={args.warmup}, runs={args.runs}, "
        f"device={args.device}, preset={args.preset}"
    )
    print(
        f"{'backend':<24s} {'slot_mask':>10s} {'reset':>10s} {'dispatch':>10s} "
        f"{'compose':>10s} {'episode':>10s} {'total':>10s}"
    )
    for backend in backends:
        phases = _time_backend(backend, args)
        total = sum(phases.values())
        print(
            f"{backend:<24s} {phases['slot_mask']:>10.4f} {phases['reset_step']:>10.4f} "
            f"{phases['dispatch']:>10.4f} {phases['compose']:>10.4f} "
            f"{phases['episode_inc']:>10.4f} {total:>10.4f}"
        )


if __name__ == "__main__":
    main()
