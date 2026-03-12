# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Auto-generate a 6-level performance report from benchmark sqlite files.

Produces a structured plot hierarchy::

    plots/
    ├── 1_scaling/          # How does the system scale with num_envs?
    ├── 2_step_overview/    # What happens inside one env.step?
    ├── 3_manager_breakdown/# What happens inside each compute manager?
    ├── 4_term_ops/         # What ops does each term execute?
    ├── 5_kernel_micro/     # Raw microsecond dispatch patterns
    └── 6_source_map/       # Draft CUDA dispatch → Python source mapping

Usage::

    PYTHONPATH=scripts:$PYTHONPATH env_isaaclab/bin/python \\
        scripts/octibenchmark/report.py /tmp/kuka_bench

Or via isaaclab.sh::

    ./isaaclab.sh -p scripts/octibenchmark/report.py /tmp/kuka_bench \\
        --step_index 5 --top_terms 5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from octibenchmark.analysis_utils import (
    SourceMapEntry,
    TermDecomposition,
    align_dispatches_to_source,
    analyze_scaling,
    classify_kernel,
    draft_source_template,
    find_hotpath,
    format_scaling,
    gpu_utilization,
    query_all_term_decompositions,
    query_nvtx_tree,
    query_step_timeseries,
    query_term_decomposition,
    resolve_term_source,
)

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "warp": "#4CAF50",
    "torch_reduce": "#F44336",
    "torch_index": "#9C27B0",
    "torch_cat": "#FF9800",
    "torch_cross": "#00BCD4",
    "torch_math": "#795548",
    "torch_copy": "#607D8B",
    "torch_fill": "#BDBDBD",
    "torch_compare": "#E91E63",
    "torch_binary": "#2196F3",
    "torch_unary": "#03A9F4",
    "torch_ewise": "#3F51B5",
    "torch_bool": "#E91E63",
    "other": "#9E9E9E",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def discover_dbs(bench_dir: str) -> dict[int, str]:
    """Find sqlite files and extract num_envs from filenames."""
    dbs = {}
    for f in sorted(glob.glob(os.path.join(bench_dir, "*.sqlite"))):
        for part in os.path.basename(f).split("__"):
            if part.startswith("envs"):
                try:
                    n = int(part[4:])
                    dbs[n] = f
                except ValueError:
                    pass
    return dbs


def task_label(bench_dir: str) -> str:
    """Extract a short task label from the sqlite filenames."""
    for f in glob.glob(os.path.join(bench_dir, "*.sqlite")):
        base = os.path.basename(f)
        return base.split("__")[0].replace("Isaac-", "").replace("-v0", "")
    return "benchmark"


def task_id_from_bench_dir(bench_dir: str) -> str | None:
    """Extract the full gymnasium task ID from sqlite filenames.

    E.g. ``Isaac-Dexsuite-Kuka-Allegro-Lift-v0__envs1024__...sqlite``
    → ``"Isaac-Dexsuite-Kuka-Allegro-Lift-v0"``.
    """
    for f in glob.glob(os.path.join(bench_dir, "*.sqlite")):
        return os.path.basename(f).split("__")[0]
    return None


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# Level 1: Scaling
# ---------------------------------------------------------------------------


def level1_scaling(dbs: dict[int, str], out_dir: str, label: str):
    """Generate scaling plots."""
    os.makedirs(out_dir, exist_ok=True)

    scaling = analyze_scaling(dbs)
    print(format_scaling(scaling))

    # 1a: Log-log scaling
    fig, ax = plt.subplots(figsize=(12, 7))
    entries = [e for e in scaling.entries if max(e.times_ns) > 100_000]
    entries.sort(key=lambda e: e.exponent, reverse=True)
    colors = plt.cm.tab20(range(len(entries)))
    for entry, color in zip(entries, colors):
        times_ms = [t / 1e6 for t in entry.times_ns]
        ax.loglog(
            entry.num_envs,
            times_ms,
            "o-",
            color=color,
            label=f"{entry.range_name} (exp={entry.exponent:.2f})",
            linewidth=2,
            markersize=8,
        )
    envs = sorted(dbs.keys())
    base = 0.5
    ax.loglog([envs[0], envs[-1]], [base, base], "--", color="gray", alpha=0.5, label="O(1)")
    ax.loglog([envs[0], envs[-1]], [base, base * envs[-1] / envs[0]], "--", color="gray", alpha=0.3, label="O(n)")
    ax.set_xlabel("num_envs", fontsize=13)
    ax.set_ylabel("Average wall time (ms)", fontsize=13)
    ax.set_title(f"{label} — NVTX Range Scaling", fontsize=14)
    ax.set_xticks(envs)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "scaling_loglog.png"))

    # 1b: Step timeseries
    fig, axes = plt.subplots(1, len(dbs), figsize=(5.5 * len(dbs), 4), sharey=False)
    if len(dbs) == 1:
        axes = [axes]
    for ax, (num_envs, db_path) in zip(axes, sorted(dbs.items())):
        snaps = query_step_timeseries(db_path)
        steps = [s.step_index for s in snaps]
        times_ms = [s.wall_ns / 1e6 for s in snaps]
        mean_ms = sum(times_ms) / len(times_ms) if times_ms else 0
        ax.plot(steps, times_ms, "o-", markersize=3, linewidth=1, color="steelblue")
        ax.axhline(mean_ms, color="red", linestyle="--", alpha=0.7, label=f"mean={mean_ms:.2f}ms")
        ax.set_xlabel("Step index")
        ax.set_ylabel("Wall time (ms)")
        ax.set_title(f"num_envs={num_envs}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{label} — Step Time Stability", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "step_timeseries.png"))

    # 1c: Stacked breakdown (from matrix_results.json if available)
    results_json = os.path.join(os.path.dirname(list(dbs.values())[0]), "matrix_results.json")
    if os.path.exists(results_json):
        with open(results_json) as f:
            results = json.load(f)
        top_ranges = [
            "reward.compute",
            "observation.compute",
            "termination.compute",
            "command.compute",
            "scene.write_data_to_sim",
            "scene.update",
        ]
        data = {}
        env_step_times = {}
        envs_list = sorted(dbs.keys())
        for num_envs in envs_list:
            for key, val in results.items():
                if f"envs{num_envs}" in key:
                    ranges_map = {r["name"]: r["avg_ns"] for r in val.get("nvtx_ranges", [])}
                    env_step_times[num_envs] = ranges_map.get("env.step", 0) / 1e6
                    for rng in top_ranges:
                        data.setdefault(rng, []).append(ranges_map.get(rng, 0) / 1e6)
                    break
        if data:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = range(len(envs_list))
            bottom = [0.0] * len(envs_list)
            colors = plt.cm.Set2(range(len(top_ranges)))
            for rng, color in zip(top_ranges, colors):
                vals = data.get(rng, [0] * len(envs_list))
                ax.bar(x, vals, bottom=bottom, label=rng, color=color, edgecolor="white", width=0.6)
                bottom = [b + v for b, v in zip(bottom, vals)]
            other = [env_step_times.get(n, 0) - b for n, b in zip(envs_list, bottom)]
            ax.bar(x, other, bottom=bottom, label="other", color="lightgray", edgecolor="white", width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels([str(n) for n in envs_list])
            ax.set_xlabel("num_envs", fontsize=13)
            ax.set_ylabel("Average wall time (ms)", fontsize=13)
            ax.set_title(f"{label} — env.step Breakdown", fontsize=14)
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            _save(fig, os.path.join(out_dir, "breakdown_stacked.png"))

    return scaling


# ---------------------------------------------------------------------------
# Level 2: Step overview
# ---------------------------------------------------------------------------


def level2_step_overview(db_path: str, step_index: int, out_dir: str, label: str, num_envs: int):
    """Generate single-step overview plots."""
    os.makedirs(out_dir, exist_ok=True)

    tree = query_nvtx_tree(db_path, step_index=step_index, attribute_kernels=True)
    if tree is None:
        print("  WARN: could not build tree")
        return

    # 2a: CPU/GPU breakdown
    rows = []

    def _collect(node, path=""):
        full_path = f"{path}/{node.name}" if path else node.name
        if node.depth >= 1:
            wall_ms = node.wall_ns / 1e6
            gpu_ms = node.subtree_kernel_gpu_ns / 1e6
            rows.append(
                {
                    "name": node.name,
                    "depth": node.depth,
                    "wall_ms": wall_ms,
                    "gpu_ms": gpu_ms,
                    "cpu_ms": wall_ms - gpu_ms,
                    "gpu_pct": (gpu_ms / wall_ms * 100) if wall_ms > 0 else 0,
                    "kernel_count": node.subtree_kernel_count,
                }
            )
        for child in node.children:
            _collect(child, full_path)

    _collect(tree)
    rows.sort(key=lambda r: r["wall_ms"], reverse=True)
    rows = rows[:25]
    rows.reverse()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(6, len(rows) * 0.35)), gridspec_kw={"width_ratios": [3, 1]})
    y_pos = range(len(rows))
    names = [r["name"] if r["depth"] == 1 else f"  └─ {r['name']}" for r in rows]
    cpu_vals = [r["cpu_ms"] for r in rows]
    gpu_vals = [r["gpu_ms"] for r in rows]

    ax1.barh(y_pos, cpu_vals, height=0.7, color="#4A90D9", alpha=0.5, label="CPU time")
    ax1.barh(y_pos, gpu_vals, height=0.7, left=cpu_vals, color="#D94A4A", alpha=0.8, label="GPU kernel time")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("Time (ms)", fontsize=11)
    ax1.set_title("CPU vs GPU Time per NVTX Range", fontsize=13)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, axis="x", alpha=0.3)
    for i, r in enumerate(rows):
        total = r["wall_ms"]
        if total > 0.1:
            ax1.text(total + 0.05, i, f"{r['cpu_ms']:.2f} + {r['gpu_ms']:.2f} = {total:.2f}ms", va="center", fontsize=7)

    gpu_pcts = [r["gpu_pct"] for r in rows]
    color_by_pct = ["#D94A4A" if p > 10 else "#4A90D9" for p in gpu_pcts]
    ax2.barh(y_pos, gpu_pcts, height=0.7, color=color_by_pct, alpha=0.7)
    ax2.set_xlabel("GPU %", fontsize=11)
    ax2.set_title("GPU Utilization", fontsize=13)
    ax2.set_yticks([])
    ax2.set_xlim(0, max(max(gpu_pcts) * 1.3, 20) if gpu_pcts else 20)
    ax2.grid(True, axis="x", alpha=0.3)
    for i, (pct, kc) in enumerate(zip(gpu_pcts, [r["kernel_count"] for r in rows])):
        ax2.text(pct + 0.3, i, f"{pct:.1f}% ({kc})", va="center", fontsize=7)

    fig.suptitle(f"{label} — CPU/GPU Breakdown ({num_envs} envs, step {step_index})", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "cpu_gpu_breakdown.png"))

    # 2b: GPU utilization bar
    util = gpu_utilization(tree)
    items = sorted([(n, r) for n, r in util.items() if r > 0], key=lambda x: x[1], reverse=True)
    if items:
        fig, ax = plt.subplots(figsize=(10, max(4, len(items) * 0.4)))
        u_names = [n for n, _ in items]
        ratios = [r * 100 for _, r in items]
        bars = ax.barh(range(len(u_names)), ratios, color="steelblue", edgecolor="white")
        ax.set_yticks(range(len(u_names)))
        ax.set_yticklabels(u_names, fontsize=9)
        ax.set_xlabel("GPU Utilization (%)", fontsize=12)
        ax.set_title(f"{label} — GPU Utilization ({num_envs} envs)", fontsize=13)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        for bar, val in zip(bars, ratios):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8)
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, "gpu_utilization.png"))

    # 2c: Hot path
    path = find_hotpath(tree, metric="wall_ns")
    if path:
        fig, ax = plt.subplots(figsize=(12, 3))
        colors = plt.cm.Blues([(i + 3) / (len(path) + 3) for i in range(len(path))])
        y = 0
        for i, (node, color) in enumerate(zip(path, colors)):
            wall_ms = node.wall_ns / 1e6
            gpu_ms = node.direct_kernel_gpu_ns / 1e6
            ax.barh(y, wall_ms, height=0.5, color=color, edgecolor="black", linewidth=0.5)
            ax.text(
                wall_ms / 2,
                y,
                f"{node.name}\n{wall_ms:.2f}ms wall | {gpu_ms:.3f}ms GPU",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )
            y -= 0.7
        ax.set_xlim(0, max(n.wall_ns for n in path) / 1e6 * 1.1)
        ax.set_ylim(y + 0.2, 0.5)
        ax.set_xlabel("Wall time (ms)")
        ax.set_title(f"{label} — Hot Path ({num_envs} envs, step {step_index})", fontsize=13)
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, "hotpath.png"))

    # 2d: Kernel density
    _plot_kernel_density(db_path, step_index, out_dir, label, num_envs)


def _plot_kernel_density(db_path: str, step_index: int, out_dir: str, label: str, num_envs: int):
    """Kernel launch density over time within a step."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    step = c.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE eventType=59 AND text='env.step' ORDER BY start LIMIT 1 OFFSET ?",
        (step_index,),
    ).fetchone()
    if step is None:
        conn.close()
        return
    step_start, step_end = step
    step_wall_ns = step_end - step_start

    kernels = c.execute(
        """
        SELECT (k.end - k.start), k.start
        FROM CUPTI_ACTIVITY_KIND_RUNTIME api
        JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON api.correlationId = k.correlationId
        WHERE api.start >= ? AND api.end <= ?
        ORDER BY k.start
    """,
        (step_start, step_end),
    ).fetchall()

    # NVTX depth-1 ranges for bottom panel
    all_ranges = c.execute(
        "SELECT text, start, end FROM NVTX_EVENTS "
        "WHERE eventType=59 AND text IS NOT NULL AND start >= ? AND end <= ? "
        "ORDER BY start ASC, (end - start) DESC",
        (step_start, step_end),
    ).fetchall()
    conn.close()

    if not kernels:
        return

    n_bins = 200
    bin_width_ns = step_wall_ns / n_bins
    k_count_bins = [0] * n_bins
    k_time_bins = [0.0] * n_bins
    for dur, ks in kernels:
        idx = min(int((ks - step_start) / bin_width_ns), n_bins - 1)
        k_count_bins[idx] += 1
        k_time_bins[idx] += dur
    bin_centers = [(i + 0.5) * bin_width_ns / 1e6 for i in range(n_bins)]
    util_bins = [t / bin_width_ns * 100 for t in k_time_bins]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 8), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1]})
    bw = bin_width_ns / 1e6
    ax1.bar(bin_centers, k_count_bins, width=bw, color="steelblue", alpha=0.7, edgecolor="none")
    ax1.set_ylabel("Kernel launches\nper bin", fontsize=10)
    avg_dur_us = sum(d for d, _ in kernels) / len(kernels) / 1000
    ax1.set_title(
        f"{label} — GPU Kernel Density ({num_envs} envs, step {step_index})\n"
        f"{len(kernels)} kernels across {step_wall_ns / 1e6:.1f}ms "
        f"(avg {len(kernels) / (step_wall_ns / 1e6):.0f}/ms, avg {avg_dur_us:.1f}μs each)",
        fontsize=12,
    )
    ax1.grid(True, alpha=0.3)

    ax2.bar(bin_centers, util_bins, width=bw, color="#D94A4A", alpha=0.7, edgecolor="none")
    ax2.set_ylabel("GPU util %\nper bin", fontsize=10)
    avg_util = sum(util_bins) / len(util_bins)
    ax2.axhline(avg_util, color="black", linestyle="--", alpha=0.5, label=f"avg={avg_util:.1f}%")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Depth-1 ranges as colored spans
    stack = []
    top_names = set()
    for name, s, e in all_ranges:
        while stack and not (stack[-1][1] <= s and stack[-1][2] >= e):
            stack.pop()
        if len(stack) == 1:
            top_names.add(name)
            x0 = (s - step_start) / 1e6
            xw = (e - s) / 1e6
            cmap = plt.cm.Set3(hash(name) % 12)
            ax3.axvspan(x0, x0 + xw, alpha=0.6, color=cmap, label=name)
        stack.append((name, s, e))
    handles, labels = ax3.get_legend_handles_labels()
    seen = set()
    uh, ul = [], []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            uh.append(handle)
            ul.append(label)
    ax3.legend(uh, ul, fontsize=6, ncol=4, loc="upper right")
    ax3.set_ylabel("NVTX", fontsize=10)
    ax3.set_yticks([])
    ax3.set_xlabel("Time within step (ms)", fontsize=11)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "kernel_density.png"))


# ---------------------------------------------------------------------------
# Level 3: Manager breakdown
# ---------------------------------------------------------------------------


def level3_manager_breakdown(db_path: str, step_index: int, out_dir: str, label: str, num_envs: int):
    """Generate per-manager decomposition plots."""
    os.makedirs(out_dir, exist_ok=True)

    # 3a: Term decomposition (all terms)
    decomps = query_all_term_decompositions(db_path, step_index=step_index, min_depth=1, max_depth=2)
    if not decomps:
        print("  WARN: no terms found")
        return decomps

    rows = [d for d in decomps if d.wall_ns > 10_000]
    rows = rows[:30]
    rows_rev = list(reversed(rows))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(20, max(7, len(rows_rev) * 0.38)), gridspec_kw={"width_ratios": [3, 1.2]}
    )
    y_pos = range(len(rows_rev))
    indent = {1: "", 2: "  └─ "}
    labels_list = [f"{indent.get(r.depth, '')}{r.name}" for r in rows_rev]

    c_pre, c_bet, c_api, c_post = "#2196F3", "#F44336", "#FF9800", "#9C27B0"
    pre = [r.pre_python_ns / 1e6 for r in rows_rev]
    between = [r.between_python_ns / 1e6 for r in rows_rev]
    api_oh = [r.api_overhead_ns / 1e6 for r in rows_rev]
    post = [r.post_python_ns / 1e6 for r in rows_rev]

    ax1.barh(y_pos, pre, height=0.7, color=c_pre, alpha=0.7, label="Python: before 1st dispatch")
    left = pre
    ax1.barh(y_pos, between, height=0.7, left=left, color=c_bet, alpha=0.7, label="Python: between dispatches")
    left = [a + b for a, b in zip(left, between)]
    ax1.barh(y_pos, api_oh, height=0.7, left=left, color=c_api, alpha=0.7, label="CUDA API overhead")
    left = [a + b for a, b in zip(left, api_oh)]
    ax1.barh(y_pos, post, height=0.7, left=left, color=c_post, alpha=0.7, label="Python: after last dispatch")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels_list, fontsize=8)
    ax1.set_xlabel("Time (ms)", fontsize=11)
    ax1.set_title("Where CPU Time Goes Inside Each NVTX Range", fontsize=13)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, axis="x", alpha=0.3)
    for i, r in enumerate(rows_rev):
        total_ms = r.wall_ns / 1e6
        if total_ms > 0.05:
            ax1.text(
                total_ms + 0.02,
                i,
                f"{total_ms:.2f}ms | {r.n_dispatches} calls"
                + (f" | avg gap {r.avg_gap_ns / 1000:.0f}μs" if r.n_dispatches > 1 else ""),
                va="center",
                fontsize=6.5,
                color="gray",
            )

    # Right panel: % stacked
    for i, r in enumerate(rows_rev):
        if r.wall_ns == 0:
            continue
        w = r.wall_ns
        pp = r.pre_python_ns / w * 100
        bp = r.between_python_ns / w * 100
        ap = r.api_overhead_ns / w * 100
        op = r.post_python_ns / w * 100
        ax2.barh(i, pp, height=0.7, color=c_pre, alpha=0.7)
        ax2.barh(i, bp, height=0.7, left=pp, color=c_bet, alpha=0.7)
        ax2.barh(i, ap, height=0.7, left=pp + bp, color=c_api, alpha=0.7)
        ax2.barh(i, op, height=0.7, left=pp + bp + ap, color=c_post, alpha=0.7)
        dominant = max(pp, bp, ap, op)
        if bp == dominant:
            ax2.text(101, i, f"{bp:.0f}% between", fontsize=6.5, va="center", color=c_bet)
        elif pp == dominant:
            ax2.text(101, i, f"{pp:.0f}% setup", fontsize=6.5, va="center", color=c_pre)

    ax2.set_xlim(0, 130)
    ax2.set_xlabel("% of wall time", fontsize=11)
    ax2.set_title("Breakdown %", fontsize=13)
    ax2.set_yticks([])
    ax2.grid(True, axis="x", alpha=0.3)
    fig.suptitle(f"{label} — CPU Time Decomposition ({num_envs} envs, step {step_index})", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "term_decomposition.png"))

    # 3b: Gap analysis scatter
    scatter_rows = [r for r in decomps if r.n_dispatches > 1 and r.wall_ns > 10_000]
    if scatter_rows:
        fig, ax = plt.subplots(figsize=(12, 8))
        for r in scatter_rows:
            bp = r.between_python_ns / r.wall_ns * 100 if r.wall_ns > 0 else 0
            ax.scatter(
                r.n_dispatches,
                r.avg_gap_ns / 1000,
                s=r.wall_ns / 1e6 * 200,
                c=bp,
                cmap="Reds",
                vmin=0,
                vmax=100,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )
            lbl = r.name if len(r.name) <= 35 else r.name[:32] + "..."
            ax.annotate(
                lbl, (r.n_dispatches, r.avg_gap_ns / 1000), fontsize=6.5, xytext=(5, 5), textcoords="offset points"
            )
        ax.set_xlabel("Number of dispatches", fontsize=12)
        ax.set_ylabel("Avg gap between dispatches (μs)", fontsize=12)
        ax.set_title(f"{label} — Dispatch Pattern ({num_envs} envs, step {step_index})", fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, "gap_analysis.png"))

    # 3c: Parent timelines for each top-level manager
    tree = query_nvtx_tree(db_path, step_index=step_index, attribute_kernels=True)
    if tree:
        for child in tree.children:
            if child.children:  # only managers with sub-terms
                _plot_parent_timeline(db_path, step_index, child.name, out_dir, label, num_envs)

    return decomps


def _plot_parent_timeline(db_path: str, step_index: int, parent_name: str, out_dir: str, label: str, num_envs: int):
    """Timeline of all children within a parent manager."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    step = c.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE eventType=59 AND text='env.step' ORDER BY start LIMIT 1 OFFSET ?",
        (step_index,),
    ).fetchone()
    if step is None:
        conn.close()
        return

    parent = c.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE eventType=59 AND text=? AND start >= ? AND end <= ?",
        (parent_name, step[0], step[1]),
    ).fetchone()
    if parent is None:
        conn.close()
        return
    p_start, p_end = parent
    p_wall_us = (p_end - p_start) / 1000

    # Find children
    all_ranges = c.execute(
        "SELECT text, start, end FROM NVTX_EVENTS "
        "WHERE eventType=59 AND text IS NOT NULL AND start >= ? AND end <= ? "
        "ORDER BY start ASC, (end - start) DESC",
        (p_start, p_end),
    ).fetchall()

    stack = []
    children = []
    for name, s, e in all_ranges:
        while stack and not (stack[-1][1] <= s and stack[-1][2] >= e):
            stack.pop()
        if len(stack) == 1:
            children.append((name, s, e))
        stack.append((name, s, e))

    # Get all dispatches
    dispatches = c.execute(
        """
        SELECT api.start, api.end, s.value, (k.end - k.start)
        FROM CUPTI_ACTIVITY_KIND_RUNTIME api
        JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON api.correlationId = k.correlationId
        JOIN StringIds s ON k.demangledName = s.id
        WHERE api.start >= ? AND api.end <= ?
        ORDER BY api.start
    """,
        (p_start, p_end),
    ).fetchall()
    conn.close()

    # Assign dispatches to children
    child_disps = {name: [] for name, _, _ in children}
    for api_s, api_e, kname, gpu_ns in dispatches:
        for cname, cs, ce in children:
            if cs <= api_s and ce >= api_e:
                _, cat = classify_kernel(kname)
                child_disps[cname].append((api_s - p_start, api_e - p_start, gpu_ns, cat))
                break

    visible = [(n, s, e) for n, s, e in children if (e - s) / 1000 > 5]
    n_rows = len(visible) + 1
    fig, ax = plt.subplots(figsize=(22, max(4, n_rows * 0.8 + 2)))

    row_gap = 1.0
    ylabels = []
    for ri, (cname, cs, ce) in enumerate(visible):
        y = -(ri * row_gap)
        cw = (ce - cs) / 1000
        cx = (cs - p_start) / 1000
        ax.barh(y, cw, left=cx, height=0.7, color="lightyellow", edgecolor="gray", linewidth=0.5, alpha=0.5)
        for api_s, api_e, gpu_ns, cat in child_disps.get(cname, []):
            col = CATEGORY_COLORS.get(cat, "gray")
            w = max((api_e - api_s) / 1000, 0.3)
            ax.barh(y, w, left=api_s / 1000, height=0.56, color=col, edgecolor="none", alpha=0.85)
        n_disps = len(child_disps.get(cname, []))
        gpu_us = sum(g for _, _, g, _ in child_disps.get(cname, [])) / 1000
        short = cname if len(cname) <= 40 else cname[:37] + "..."
        ylabels.append(f"{short}\n{cw:.0f}μs | {n_disps} ops | {gpu_us:.1f}μs GPU")

    # Gaps row
    ri = len(visible)
    y = -(ri * row_gap)
    ylabels.append("gaps between terms\n(Python overhead)")
    sorted_c = sorted(visible, key=lambda x: x[1])
    for i in range(len(sorted_c)):
        gs = 0 if i == 0 else (sorted_c[i - 1][2] - p_start) / 1000
        ge = (sorted_c[i][1] - p_start) / 1000
        if ge - gs > 1:
            ax.barh(y, ge - gs, left=gs, height=0.7, color="red", alpha=0.2, edgecolor="none")
            if ge - gs > 10:
                ax.text((gs + ge) / 2, y, f"{ge - gs:.0f}μs", ha="center", va="center", fontsize=6, color="red")
    if sorted_c:
        last_e = (sorted_c[-1][2] - p_start) / 1000
        if p_wall_us - last_e > 1:
            ax.barh(y, p_wall_us - last_e, left=last_e, height=0.7, color="red", alpha=0.2, edgecolor="none")

    ax.set_yticks([-(i * row_gap) for i in range(len(ylabels))])
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlim(-5, p_wall_us + 5)
    ax.set_xlabel("Time within parent (μs)", fontsize=11)
    td = sum(len(d) for d in child_disps.values())
    tg = sum(g for d in child_disps.values() for _, _, g, _ in d) / 1000
    ax.set_title(
        f"{parent_name} — Children Timeline ({num_envs} envs, step {step_index}) | "
        f"{len(visible)} children, {td} dispatches, {tg:.0f}μs GPU",
        fontsize=11,
    )
    ax.grid(True, axis="x", alpha=0.3)

    # Legend
    seen = set()
    for d in child_disps.values():
        for _, _, _, cat in d:
            seen.add(cat)
    patches = [
        mpatches.Patch(color=CATEGORY_COLORS[c], label=c.replace("_", " "))
        for c in sorted(seen)
        if c in CATEGORY_COLORS
    ]
    patches.append(mpatches.Patch(color="red", alpha=0.2, label="Python gap"))
    ax.legend(handles=patches, fontsize=7, ncol=4, loc="upper right")

    fig.tight_layout()
    safe = parent_name.replace(".", "_").replace(":", "_")
    _save(fig, os.path.join(out_dir, f"parent_{safe}.png"))


# ---------------------------------------------------------------------------
# Level 4: Term ops
# ---------------------------------------------------------------------------


def level4_term_ops(db_path: str, step_index: int, term_names: list[str], out_dir: str, label: str, num_envs: int):
    """Generate op-level timeline for selected terms."""
    os.makedirs(out_dir, exist_ok=True)

    for term_name in term_names:
        decomp = query_term_decomposition(
            db_path,
            step_index=step_index,
            range_name=term_name,
            include_dispatches=True,
        )
        if decomp is None or not decomp.dispatches:
            continue
        _plot_op_sequence(decomp, step_index, out_dir, label, num_envs)


def _plot_op_sequence(decomp: TermDecomposition, step_index: int, out_dir: str, label: str, num_envs: int):
    """Op-by-op timeline for one term."""
    wall_us = decomp.wall_ns / 1000
    disps = decomp.dispatches
    n = len(disps)

    fig, (ax_t, ax_s) = plt.subplots(2, 1, figsize=(22, 6), gridspec_kw={"height_ratios": [2, 1.2]})

    for i, d in enumerate(disps):
        x = d.api_start_ns / 1000
        w = max((d.api_end_ns - d.api_start_ns) / 1000, 0.4)
        color = CATEGORY_COLORS.get(d.category, "gray")
        rect = mpatches.FancyBboxPatch(
            (x, -0.35), w, 0.7, boxstyle="round,pad=0", facecolor=color, edgecolor="black", linewidth=0.3, alpha=0.85
        )
        ax_t.add_patch(rect)

        gap_us = d.gap_before_ns / 1000
        if gap_us > 3:
            prev_end = x - gap_us
            ax_t.axvspan(max(0, prev_end), x, alpha=0.08, color="red")

        if (i % 3 == 0) or (gap_us > 15) or i == 0 or i == n - 1:
            short = d.op_name.split(": ", 1)[-1]
            if len(short) > 18:
                short = short[:15] + "..."
            y_off = 0.55 if (i % 2 == 0) else 0.75
            ax_t.text(x + w / 2, y_off, short, ha="center", va="bottom", fontsize=5.5, rotation=60, color=color)
            if gap_us > 15:
                ax_t.text(x - gap_us / 2, -0.6, f"{gap_us:.0f}μs", ha="center", fontsize=5, color="red", alpha=0.7)

    ax_t.set_xlim(-5, wall_us + 5)
    ax_t.set_ylim(-1.0, 2.5)
    ax_t.set_yticks([])
    ax_t.set_xlabel("Time within term (μs)")
    ax_t.grid(True, axis="x", alpha=0.3)
    seen = set(d.category for d in disps)
    patches = [
        mpatches.Patch(color=CATEGORY_COLORS[c], label=c.replace("_", " "))
        for c in sorted(seen)
        if c in CATEGORY_COLORS
    ]
    patches.append(mpatches.Patch(color="red", alpha=0.1, label="Python gap"))
    ax_t.legend(handles=patches, fontsize=6.5, ncol=4, loc="upper left")
    ax_t.set_title(
        f"{decomp.name} — Op-by-Op ({num_envs} envs, step {step_index})\n"
        f"Wall: {wall_us:.0f}μs | {n} dispatches | Python between: {decomp.between_pct:.0f}%",
        fontsize=11,
    )

    # Summary: group by category
    cat_stats: dict[str, dict] = {}
    for d in disps:
        if d.category not in cat_stats:
            cat_stats[d.category] = {"count": 0, "gpu_ns": 0, "ops": set()}
        cat_stats[d.category]["count"] += 1
        cat_stats[d.category]["gpu_ns"] += d.gpu_duration_ns
        cat_stats[d.category]["ops"].add(d.op_name)

    cats = sorted(cat_stats.items(), key=lambda x: x[1]["count"], reverse=True)
    cat_labels = []
    for cat, stats in cats:
        ops_str = ", ".join(sorted(stats["ops"]))
        if len(ops_str) > 60:
            ops_str = ops_str[:57] + "..."
        cat_labels.append(f"{cat.replace('_', ' ')} ({stats['count']}x): {ops_str}")
    counts = [s["count"] for _, s in cats]
    colors = [CATEGORY_COLORS.get(c, "gray") for c, _ in cats]

    ax_s.barh(range(len(cats)), counts, height=0.7, color=colors, alpha=0.7, edgecolor="black", linewidth=0.3)
    ax_s.set_yticks(range(len(cats)))
    ax_s.set_yticklabels(cat_labels, fontsize=7)
    ax_s.set_xlabel("Dispatches")
    ax_s.set_title("Op Category Breakdown", fontsize=10)
    ax_s.invert_yaxis()
    ax_s.grid(True, axis="x", alpha=0.3)
    for i, (cat, stats) in enumerate(cats):
        ax_s.text(
            stats["count"] + 0.3, i, f"{stats['gpu_ns'] / 1000:.1f}μs GPU", va="center", fontsize=6.5, color="gray"
        )

    fig.tight_layout()
    safe = decomp.name.replace(":", "_").replace("[", "_").replace("]", "_")
    _save(fig, os.path.join(out_dir, f"ops_{safe}.png"))


# ---------------------------------------------------------------------------
# Level 5: Kernel micro
# ---------------------------------------------------------------------------


def level5_kernel_micro(db_path: str, step_index: int, term_names: list[str], out_dir: str, label: str, num_envs: int):
    """Generate microsecond-level dispatch timelines."""
    os.makedirs(out_dir, exist_ok=True)

    for term_name in term_names:
        decomp = query_term_decomposition(
            db_path,
            step_index=step_index,
            range_name=term_name,
            include_dispatches=True,
        )
        if decomp is None or not decomp.dispatches:
            continue

        wall_us = decomp.wall_ns / 1000
        disps = decomp.dispatches

        fig, ax = plt.subplots(figsize=(20, 4))
        for d in disps:
            x = d.api_start_ns / 1000
            w = max((d.api_end_ns - d.api_start_ns) / 1000, 0.3)
            color = CATEGORY_COLORS.get(d.category, "gray")
            ax.barh(0, w, left=x, height=0.6, color=color, alpha=0.8, edgecolor="none")

        # Gaps
        sorted_disps = sorted(disps, key=lambda d: d.api_start_ns)
        gap_sizes = []
        for i in range(1, len(sorted_disps)):
            gs = sorted_disps[i - 1].api_end_ns / 1000
            ge = sorted_disps[i].api_start_ns / 1000
            gap = ge - gs
            gap_sizes.append(gap)
            if gap > 2:
                ax.axvspan(gs, ge, alpha=0.15, color="red")

        pre_end = sorted_disps[0].api_start_ns / 1000
        if pre_end > 2:
            ax.axvspan(0, pre_end, alpha=0.2, color="blue")
            ax.text(pre_end / 2, 0.5, f"setup\n{pre_end:.0f}μs", ha="center", fontsize=7)

        ax.set_xlim(-5, wall_us + 5)
        ax.set_yticks([])
        ax.set_xlabel("Time within term (μs)", fontsize=11)

        cats_seen = set(d.category for d in disps)
        patches = [
            mpatches.Patch(color=CATEGORY_COLORS[c], label=c.replace("_", " "))
            for c in sorted(cats_seen)
            if c in CATEGORY_COLORS
        ]
        patches.append(mpatches.Patch(color="red", alpha=0.15, label="Python gap"))
        ax.legend(handles=patches, fontsize=8, loc="upper right", ncol=3)

        avg_g = sum(gap_sizes) / len(gap_sizes) if gap_sizes else 0
        max_g = max(gap_sizes) if gap_sizes else 0
        ax.set_title(
            f"{decomp.name} — Micro Timeline ({num_envs} envs, step {step_index})\n"
            f"Wall: {wall_us:.0f}μs | {len(disps)} dispatches | "
            f"Avg gap: {avg_g:.1f}μs | Max gap: {max_g:.1f}μs | "
            f"Python between: {decomp.between_pct:.0f}%",
            fontsize=11,
        )
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        safe = decomp.name.replace(":", "_").replace("[", "_").replace("]", "_")
        _save(fig, os.path.join(out_dir, f"micro_{safe}.png"))


# ---------------------------------------------------------------------------
# Level 6: Source map (draft — auto-generated, no agent refinement)
# ---------------------------------------------------------------------------

# Colors for source groups in the source map plot.
SOURCE_GROUP_COLORS = {
    "data_extract": "#4CAF50",
    "distance": "#2196F3",
    "contact": "#FF9800",
    "type_conv": "#9C27B0",
    "reward": "#E91E63",
    "observation": "#00BCD4",
    "default": "#607D8B",
}


def _draft_to_source_map_entries(draft: list, line_groups: dict[int, str] | None = None) -> list[SourceMapEntry]:
    """Convert auto-generated ``DraftTemplateLine`` list into ``SourceMapEntry`` list.

    Each predicted op on each source line becomes one ``SourceMapEntry``.

    Args:
        draft: Output of :func:`draft_source_template`.
        line_groups: Optional mapping of ``line_no → group`` for coloring.
            If not provided, all entries get ``"default"`` group.
    """
    entries = []
    for dtl in draft:
        for cat_pattern, note in dtl.predicted_ops:
            group = (line_groups or {}).get(dtl.line_no, "default")
            label = f"L{dtl.line_no}: {note}"
            entries.append(SourceMapEntry(cat_pattern, label, group))
    return entries


def _parse_term_range_name(range_name: str) -> tuple[str, str] | None:
    """Parse NVTX range names into ``(manager, term_name)``.

    Handles multiple formats::

        "reward.term:fingers_to_object"          → ("reward", "fingers_to_object")
        "observation.term[proprio]:joint_pos"     → ("observation", "joint_pos")
        "observation.term[perception]:object_pc"  → ("observation", "object_pc")
        "termination.term:time_out"               → ("termination", "time_out")

    Returns ``None`` if the name doesn't match any expected pattern.
    """
    import re as _re

    # Pattern: "manager.term[group]:name" or "manager.term:name"
    m = _re.match(r"^(\w+)\.term(?:\[\w+\])?[_:](.+)$", range_name)
    if m:
        return m.group(1), m.group(2)
    return None


def level6_source_map(
    db_path: str,
    step_index: int,
    term_names: list[str],
    task_id: str,
    out_dir: str,
    label: str,
    num_envs: int,
):
    """Generate draft source-to-dispatch mapping plots for top terms.

    This is a best-effort automated mapping. The draft template predicts
    dispatches from source-level pattern matching but does not expand
    function calls. Alignment will be lower than agent-refined templates
    for terms with deep call chains.

    Args:
        db_path: Path to the sqlite database.
        step_index: Which ``env.step`` to analyze.
        term_names: NVTX range names (e.g. ``"reward.term:fingers_to_object"``).
        task_id: Gymnasium task ID (e.g. ``"Isaac-Dexsuite-Kuka-Allegro-Lift-v0"``).
        out_dir: Output directory for plots.
        label: Short label for plot titles.
        num_envs: Number of environments.
    """
    os.makedirs(out_dir, exist_ok=True)

    for term_name in term_names:
        parsed = _parse_term_range_name(term_name)
        if parsed is None:
            print(f"  SKIP {term_name}: can't parse manager/term")
            continue
        manager, term = parsed

        # Phase 1: Resolve source
        try:
            src = resolve_term_source(task_id, term, manager=manager)
        except Exception as exc:
            print(f"  SKIP {term_name}: resolve_term_source failed: {exc}")
            continue

        # Phase 2: Get dispatches
        decomp = query_term_decomposition(db_path, step_index=step_index, range_name=term_name, include_dispatches=True)
        if decomp is None or not decomp.dispatches:
            print(f"  SKIP {term_name}: no dispatches")
            continue

        # Phase 3: Draft template
        draft = draft_source_template(src.source_code, src.line_number)
        template = _draft_to_source_map_entries(draft)

        # Phase 5: Align
        alignment = align_dispatches_to_source(template, decomp.dispatches)

        # Plot
        _plot_source_map(decomp, alignment, draft, src, out_dir, label, num_envs, step_index)
        print(
            f"  {term_name}: {alignment.match_pct:.0f}% aligned "
            f"({alignment.matched}/{alignment.total} dispatches, "
            f"{len(template)} predicted)"
        )


def _plot_source_map(
    decomp: TermDecomposition,
    alignment,
    draft: list,
    src,
    out_dir: str,
    label: str,
    num_envs: int,
    step_index: int,
):
    """Two-panel source map: dispatch timeline (left) + annotated source (right)."""
    disps = decomp.dispatches
    wall_us = decomp.wall_ns / 1000

    fig, (ax_timeline, ax_source) = plt.subplots(
        1,
        2,
        figsize=(24, max(8, len(draft) * 0.35)),
        gridspec_kw={"width_ratios": [2.5, 1]},
    )

    # --- Left panel: dispatch timeline colored by match status ---
    for i, d in enumerate(disps):
        x = d.api_start_ns / 1000
        w = max((d.api_end_ns - d.api_start_ns) / 1000, 0.3)

        # Check if this dispatch was matched
        entry_match = None
        for tmpl, d_idx in alignment.entries:
            if d_idx == i and tmpl is not None:
                entry_match = tmpl
                break

        if entry_match is not None:
            color = SOURCE_GROUP_COLORS.get(entry_match.group, SOURCE_GROUP_COLORS["default"])
        else:
            color = "#BDBDBD"  # gray for unmatched

        ax_timeline.barh(0, w, left=x, height=0.6, color=color, alpha=0.85, edgecolor="none")

    # Gaps
    sorted_disps = sorted(disps, key=lambda d: d.api_start_ns)
    for i in range(1, len(sorted_disps)):
        gs = sorted_disps[i - 1].api_end_ns / 1000
        ge = sorted_disps[i].api_start_ns / 1000
        if ge - gs > 2:
            ax_timeline.axvspan(gs, ge, alpha=0.1, color="red")

    ax_timeline.set_xlim(-5, wall_us + 5)
    ax_timeline.set_ylim(-1, 1.5)
    ax_timeline.set_yticks([])
    ax_timeline.set_xlabel("Time within term (μs)", fontsize=11)
    ax_timeline.set_title(
        f"Dispatch Timeline — {decomp.name}\n"
        f"{len(disps)} dispatches | {alignment.match_pct:.0f}% mapped to source "
        f"({alignment.matched}/{alignment.total})",
        fontsize=11,
    )
    ax_timeline.grid(True, axis="x", alpha=0.3)

    # Legend for timeline
    patches = [mpatches.Patch(color="#BDBDBD", label="unmatched (needs agent refinement)")]
    for group, color in SOURCE_GROUP_COLORS.items():
        if group != "default":
            patches.append(mpatches.Patch(color=color, label=group.replace("_", " ")))
    ax_timeline.legend(handles=patches, fontsize=7, loc="upper right", ncol=2)

    # --- Right panel: annotated source code ---
    source_lines = []
    for dtl in draft:
        n_ops = len(dtl.predicted_ops)
        source_lines.append((dtl.line_no, dtl.code.strip(), n_ops))

    # Draw source lines
    y = len(source_lines) - 1
    for line_no, code, n_ops in source_lines:
        display = code[:80] + "..." if len(code) > 80 else code
        text = f"L{line_no}: {display}"
        fontweight = "bold" if n_ops > 0 else "normal"
        color = "black" if n_ops > 0 else "#888888"
        ax_source.text(
            0.02,
            y,
            text,
            transform=ax_source.get_yaxis_transform(),
            fontsize=7,
            fontfamily="monospace",
            fontweight=fontweight,
            color=color,
            verticalalignment="center",
        )
        if n_ops > 0:
            ops_text = f"  ← {n_ops} dispatch(es)"
            ax_source.text(
                0.95,
                y,
                ops_text,
                transform=ax_source.get_yaxis_transform(),
                fontsize=6.5,
                color="#E91E63",
                verticalalignment="center",
                horizontalalignment="right",
            )
        y -= 1

    ax_source.set_xlim(0, 1)
    ax_source.set_ylim(-1, len(source_lines))
    ax_source.set_yticks([])
    ax_source.set_xticks([])
    ax_source.set_title(
        f"Source: {os.path.basename(src.source_file)} — {src.func_name}()",
        fontsize=10,
    )
    ax_source.spines["top"].set_visible(False)
    ax_source.spines["right"].set_visible(False)
    ax_source.spines["bottom"].set_visible(False)

    # Summary at bottom
    total_predicted = sum(len(dtl.predicted_ops) for dtl in draft)
    fig.text(
        0.5,
        -0.02,
        f"Auto-draft: {total_predicted} predicted vs {len(disps)} actual dispatches "
        f"| {alignment.match_pct:.0f}% aligned | "
        f"Unmatched dispatches likely from function calls needing expansion",
        ha="center",
        fontsize=9,
        style="italic",
        color="gray",
    )

    fig.suptitle(
        f"{label} — Source Map ({num_envs} envs, step {step_index})",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    safe = decomp.name.replace(":", "_").replace("[", "_").replace("]", "_").replace(".", "_")
    _save(fig, os.path.join(out_dir, f"source_map_{safe}.png"))


# ---------------------------------------------------------------------------
# Main: auto-generate full report
# ---------------------------------------------------------------------------


def generate_report(bench_dir: str, step_index: int = 5, top_terms: int = 5):
    """Generate the full 6-level performance report.

    Args:
        bench_dir: Directory containing ``.sqlite`` files from a benchmark run.
        step_index: Which ``env.step`` to analyze in detail.
        top_terms: Number of top terms (by wall time) to drill into at levels 4-5.
    """
    dbs = discover_dbs(bench_dir)
    if not dbs:
        print(f"ERROR: No .sqlite files found in {bench_dir}")
        return

    label = task_label(bench_dir)
    plot_dir = os.path.join(bench_dir, "plots")
    max_envs = max(dbs.keys())
    db_largest = dbs[max_envs]

    print(f"Generating report for {label}")
    print(f"  SQLite files: {len(dbs)} ({', '.join(str(k) for k in sorted(dbs.keys()))} envs)")
    print(f"  Analysis step: {step_index}")
    print(f"  Largest scale: {max_envs} envs → {db_largest}")
    print()

    # Level 1: Scaling
    print("=" * 60)
    print("Level 1: Scaling")
    print("=" * 60)
    if len(dbs) >= 2:
        level1_scaling(dbs, os.path.join(plot_dir, "1_scaling"), label)
    else:
        print("  (skipped — need ≥2 scale points)")
    print()

    # Level 2: Step overview
    print("=" * 60)
    print(f"Level 2: Step overview ({max_envs} envs, step {step_index})")
    print("=" * 60)
    level2_step_overview(db_largest, step_index, os.path.join(plot_dir, "2_step_overview"), label, max_envs)
    print()

    # Level 3: Manager breakdown
    print("=" * 60)
    print(f"Level 3: Manager breakdown ({max_envs} envs, step {step_index})")
    print("=" * 60)
    decomps = level3_manager_breakdown(
        db_largest,
        step_index,
        os.path.join(plot_dir, "3_manager_breakdown"),
        label,
        max_envs,
    )
    print()

    # Pick top terms for levels 4-5 (depth-2 terms by wall time, with dispatches)
    if decomps:
        interesting = [d.name for d in decomps if d.depth == 2 and d.n_dispatches > 0 and d.wall_ns > 50_000][
            :top_terms
        ]
    else:
        interesting = []

    # Level 4: Term ops
    print("=" * 60)
    print(f"Level 4: Term ops ({len(interesting)} terms)")
    print("=" * 60)
    if interesting:
        print(f"  Terms: {interesting}")
        level4_term_ops(db_largest, step_index, interesting, os.path.join(plot_dir, "4_term_ops"), label, max_envs)
    else:
        print("  (no interesting terms found)")
    print()

    # Level 5: Kernel micro
    print("=" * 60)
    print(f"Level 5: Kernel micro ({len(interesting)} terms)")
    print("=" * 60)
    if interesting:
        level5_kernel_micro(
            db_largest, step_index, interesting, os.path.join(plot_dir, "5_kernel_micro"), label, max_envs
        )
    else:
        print("  (no interesting terms found)")
    print()

    # Level 6: Source map (draft)
    print("=" * 60)
    print(f"Level 6: Source map — draft ({len(interesting)} terms)")
    print("=" * 60)
    tid = task_id_from_bench_dir(bench_dir)
    if interesting and tid:
        try:
            import isaaclab_tasks  # noqa: F401 — registers gym envs

            level6_source_map(
                db_largest,
                step_index,
                interesting,
                tid,
                os.path.join(plot_dir, "6_source_map"),
                label,
                max_envs,
            )
        except ImportError:
            print("  (skipped — isaaclab_tasks not importable, can't resolve source)")
    elif not tid:
        print("  (skipped — can't determine task ID from filenames)")
    else:
        print("  (no interesting terms found)")
    print()

    print("=" * 60)
    print(f"Report complete: {plot_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 5-level performance report from benchmark data.")
    parser.add_argument("bench_dir", help="Directory containing .sqlite files")
    parser.add_argument("--step_index", type=int, default=5, help="Step to analyze (default: 5)")
    parser.add_argument("--top_terms", type=int, default=5, help="Number of terms to drill into (default: 5)")
    args = parser.parse_args()

    generate_report(args.bench_dir, step_index=args.step_index, top_terms=args.top_terms)
