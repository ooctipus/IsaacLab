# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Summarise rsl_rl training runs from their tfevents file.

For each ``logs/rsl_rl/<exp>/<run>/`` directory passed as an argument,
prints reward / episode-length progression at fixed iter checkpoints plus
the final-window average. Useful for a quick report without TensorBoard.

Run with the IsaacLab venv (no IsaacLab imports, just tensorboard)::

    ./isaaclab.sh -p scripts/.../summarise_runs.py logs/rsl_rl/.../<run> ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("tensorboard not importable in this venv. Install with `pip install tensorboard`.", file=sys.stderr)
    sys.exit(1)


def _summarise(run_dir: Path) -> None:
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        print(f"[{run_dir.name}] no scalars found")
        return

    # Most rsl_rl runs log Train/mean_reward and Train/mean_episode_length.
    candidates = {
        "reward": [t for t in tags if "reward" in t.lower() and "mean" in t.lower()],
        "ep_len": [t for t in tags if "episode_length" in t.lower() and "mean" in t.lower()],
        "loss": [t for t in tags if "loss" in t.lower() and "value" in t.lower()],
    }

    print(f"\n=== {run_dir.name} ===")
    for kind, taglist in candidates.items():
        if not taglist:
            continue
        tag = sorted(taglist)[0]
        events = ea.Scalars(tag)
        if not events:
            continue
        steps = np.array([e.step for e in events])
        vals = np.array([e.value for e in events])
        # Pick a few checkpoint steps.
        n = len(events)
        checkpoints = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
        print(f"  [{tag}]")
        for c in checkpoints:
            print(f"    iter={steps[c]:>4d}  val={vals[c]:.3f}")
        # Final-window average (last 10% of run).
        tail_n = max(1, n // 10)
        tail_mean = vals[-tail_n:].mean()
        print(f"    final-{tail_n}-avg = {tail_mean:.3f}")


def _table(runs: list[Path]) -> None:
    rows = []
    for run_dir in runs:
        ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        reward_tag = next((t for t in sorted(tags) if "reward" in t.lower() and "mean" in t.lower()), None)
        if reward_tag is None:
            continue
        events = ea.Scalars(reward_tag)
        if not events:
            continue
        steps = np.array([e.step for e in events])
        vals = np.array([e.value for e in events])
        rows.append(
            {
                "run": run_dir.name,
                "iter_max": int(steps[-1]),
                "first": float(vals[0]),
                "p50": float(vals[len(vals) // 2]),
                "p99": float(np.quantile(vals, 0.99)),
                "final_30": float(vals[-max(1, len(vals) // 10) :].mean()),
            }
        )
    print()
    print(f"{'run':<30s} {'iter_max':>9s} {'first':>9s} {'p50':>9s} {'p99':>9s} {'final-30avg':>12s}")
    print("-" * 80)
    for r in rows:
        print(
            f"{r['run']:<30s} {r['iter_max']:>9d} {r['first']:>9.3f} {r['p50']:>9.3f} {r['p99']:>9.3f} {r['final_30']:>12.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="One or more rsl_rl run directories.")
    parser.add_argument("--table", action="store_true", help="Print a single-line-per-run table.")
    args = parser.parse_args()
    valid = [Path(r) for r in args.runs if Path(r).exists()]
    if args.table:
        _table(valid)
    else:
        for run_dir in valid:
            _summarise(run_dir)


if __name__ == "__main__":
    main()
