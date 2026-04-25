# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pull a multi-seed Stage-A reproduction sweep from wandb and make one clean figure.

The figure has exactly three panels — the only three that matter for "did we
reproduce scaling-crl?":

1. ``training/actor_loss``    — algorithm-level, SAC-style update
2. ``training/critic_loss``   — algorithm-level, InfoNCE
3. ``eval/episode_success_any`` — policy-level, "does it reach goals at some point"

Each panel has two lines (ours vs native), each with a shaded band for seed std.
Close overlap = reproduction confirmed; gap > ~1 std = investigate.

Usage:

.. code-block:: bash

    python scripts/reinforcement_learning/crl/make_repro_figure.py \\
        --wandb_project nvidia/crl-repro \\
        --wandb_group stage_a_ant_depth4_3seed \\
        --out logs/crl/reproduction_figure.png

Everything else in the wandb UI is noise. Feel free to ignore it.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

PANELS = [
    ("training/actor_loss", "Actor loss", "lower_is_not_meaningful"),
    ("training/critic_loss", "Critic loss (InfoNCE)", "lower_better"),
    ("eval/episode_success_any", "Success rate (reached goal any time)", "higher_better"),
]


def classify_run(run):
    """Return 'ours' or 'native' for a wandb run.

    Our pipeline tags runs with ``ours``; scaling-crl native does not, and its
    run names start with the Brax env id (e.g. ``ant_256_...``).
    """
    tags = set(run.tags or [])
    if "ours" in tags or run.name.startswith("ours_"):
        return "ours"
    return "native"


def fetch_history(run, metric, x_candidates=("training/envsteps", "env_steps", "_step")):
    """Return ``(xs, ys)`` lists for ``metric`` from a wandb run.

    Uses ``run.history()`` (DataFrame-based) which is more reliable for short
    runs than ``scan_history`` (latter can return None on partial writes).

    x-axis resolution: try each candidate key in order. Our pipeline's older runs
    log ``env_steps`` but not ``training/envsteps``; scaling-crl native logs
    ``training/envsteps``. ``_step`` is the final fallback (for ours it equals
    env_steps because we call ``wandb.log(step=env_steps, ...)``).
    """
    try:
        df = run.history(pandas=True, samples=10000, stream="default")
    except Exception as exc:
        print(f"  [warn] history() failed for {run.name}: {exc!r}")
        return [], []
    if df is None or len(df) == 0 or metric not in df.columns:
        return [], []
    for x_key in x_candidates:
        if x_key not in df.columns and x_key != "_step":
            continue
        if x_key == "_step":
            xs = df.index.tolist()
        else:
            xs = df[x_key].tolist()
        ys = df[metric].tolist()
        # drop NaNs (wandb leaves NaN when a metric wasn't logged on that step)
        pairs = [
            (x, y)
            for x, y in zip(xs, ys)
            if x is not None
            and y is not None
            and not (isinstance(x, float) and x != x)
            and not (isinstance(y, float) and y != y)
        ]
        if pairs:
            return [p[0] for p in pairs], [p[1] for p in pairs]
    return [], []


def align_and_aggregate(runs_by_group, metric):
    """Align x-axes across seeds, return dict group -> (xs, mean, std).

    Strategy: use the smallest x from each run's data as pivot, interpolate each
    seed onto a common grid (the union of rounded x-values from the first run in
    the group), compute mean + std across seeds.
    """
    import numpy as np

    out = {}
    for group, runs in runs_by_group.items():
        seeds_data = []
        for r in runs:
            xs, ys = fetch_history(r, metric)
            if not xs:
                continue
            seeds_data.append((np.asarray(xs), np.asarray(ys)))
        if not seeds_data:
            continue
        # Common grid: the first seed's x values (they're already monotonic).
        common_x = seeds_data[0][0]
        stacked = []
        for xs, ys in seeds_data:
            aligned = np.interp(common_x, xs, ys)
            stacked.append(aligned)
        stacked = np.asarray(stacked)  # [num_seeds, num_x]
        out[group] = (common_x, stacked.mean(axis=0), stacked.std(axis=0), stacked.shape[0])
    return out


def make_figure(agg_by_metric, out_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 4.5))
    colors = {"ours": "#1f77b4", "native": "#d62728"}
    labels = {"ours": "ours (our pipeline)", "native": "scaling-crl native"}

    for ax, (metric, title, _) in zip(axes, PANELS):
        agg = agg_by_metric.get(metric, {})
        for group in ("ours", "native"):
            if group not in agg:
                continue
            x, mean, std, n = agg[group]
            ax.plot(x, mean, color=colors[group], linewidth=2.0, label=f"{labels[group]} (n={n})")
            ax.fill_between(x, mean - std, mean + std, color=colors[group], alpha=0.18)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("env_steps")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Stage A: Ant, depth=4 — ours vs scaling-crl native (3 seeds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[fig] saved -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wandb_project", type=str, required=True, help="'entity/project' format")
    p.add_argument("--wandb_group", type=str, required=True)
    p.add_argument("--out", type=str, default="logs/crl/reproduction_figure.png")
    args = p.parse_args()

    try:
        import wandb
    except ImportError:
        raise SystemExit("wandb is required: pip install wandb")

    api = wandb.Api()
    runs = api.runs(args.wandb_project, filters={"group": args.wandb_group})
    runs_by_group: dict[str, list] = defaultdict(list)
    for r in runs:
        if r.state not in ("finished", "running", "crashed"):  # include crashed to see partial
            continue
        g = classify_run(r)
        runs_by_group[g].append(r)

    print(
        f"[fig] found {len(runs_by_group.get('ours', []))} ours, "
        f"{len(runs_by_group.get('native', []))} native runs "
        f"in {args.wandb_project}, group={args.wandb_group}"
    )
    if not runs_by_group:
        raise SystemExit("No runs match the filter. Did you pass the right group name?")

    agg_by_metric = {}
    for metric, _, _ in PANELS:
        agg_by_metric[metric] = align_and_aggregate(runs_by_group, metric)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    make_figure(agg_by_metric, args.out)

    # Brief textual summary on stdout
    print("\n=== final values (last point of each line) ===")
    for metric, title, _ in PANELS:
        for group in ("ours", "native"):
            if group in agg_by_metric[metric]:
                x, mean, std, n = agg_by_metric[metric][group]
                print(f"  {title:<45} {group:<7} (n={n}): {mean[-1]:+.4f} ± {std[-1]:.4f}")


if __name__ == "__main__":
    main()
