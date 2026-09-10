# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Summarize an `ncu --import x.ncu-rep --csv --page raw` export per kernel name.

Prints, for each distinct kernel: launches, mean duration, registers, occupancy
(theoretical vs achieved), waves, SM/DRAM/L2/L1 throughput %, L1/L2 hit rates,
DRAM bytes, L2 bytes, local (spill) bytes, issue-slot utilization, dominant stall.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def short(name: str) -> str:
    n = re.sub(r"_[0-9a-f]{8}_cuda_kernel_forward.*$", "", name)
    return n.split("(")[0][:80]


def num(v: str) -> float:
    try:
        return float(v.replace(",", ""))
    except Exception:
        return float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--json", type=Path)
    p.add_argument("--sort", default="time", choices=["time", "name"])
    args = p.parse_args()
    lines = args.csv.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
    rd = list(csv.reader(lines[start:]))
    hdr = rd[0]
    units = rd[1]
    body = rd[2:]
    col = {h: i for i, h in enumerate(hdr)}
    SCALE = {
        "byte": 1e-6,
        "Kbyte": 1e-3,
        "Mbyte": 1.0,
        "Gbyte": 1e3,
        "us": 1.0,
        "usecond": 1.0,
        "ns": 1e-3,
        "nsecond": 1e-3,
        "ms": 1e3,
        "msecond": 1e3,
    }

    def g(row, name):
        i = col.get(name)
        if i is None:
            return float("nan")
        return num(row[i]) * SCALE.get(units[i], 1.0)  # bytes -> MB, time -> us

    stall_cols = [h for h in hdr if h.startswith("smsp__average_warps_issue_stalled_") and h.endswith(".ratio")]
    agg = defaultdict(list)
    for row in body:
        agg[short(row[col["Kernel Name"]])].append(row)
    out = []
    for name, rows in agg.items():

        def m(metric):
            vals = [g(r, metric) for r in rows]
            vals = [v for v in vals if v == v]
            return statistics.fmean(vals) if vals else float("nan")

        dur_us = m("gpu__time_duration.sum")
        stalls = {
            c.replace("smsp__average_warps_issue_stalled_", "").replace("_per_issue_active.ratio", ""): m(c)
            for c in stall_cols
        }
        top_stall = sorted(stalls.items(), key=lambda kv: -kv[1] if kv[1] == kv[1] else 0)[:3]
        out.append(
            {
                "kernel": name,
                "n": len(rows),
                "dur_us": dur_us,
                "total_us": dur_us * len(rows),
                "regs": m("launch__registers_per_thread"),
                "grid": m("launch__grid_size"),
                "block": m("launch__block_size"),
                "threads": m("launch__thread_count"),
                "waves": m("launch__waves_per_multiprocessor"),
                "occ_theor_pct": m("sm__maximum_warps_per_active_cycle_pct"),
                "occ_ach_pct": m("sm__warps_active.avg.pct_of_peak_sustained_active"),
                "sm_active_pct": m("sm__cycles_active.avg.pct_of_peak_sustained_elapsed"),
                "sm_thr_pct": m("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
                "dram_thr_pct": m("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
                "l2_thr_pct": m("lts__throughput.avg.pct_of_peak_sustained_elapsed"),
                "l1_thr_pct": m("l1tex__throughput.avg.pct_of_peak_sustained_elapsed"),
                "l1_hit_pct": m("l1tex__t_sector_hit_rate.pct"),
                "l2_hit_pct": m("lts__t_sector_hit_rate.pct"),
                "dram_mb": (m("dram__bytes_read.sum") + m("dram__bytes_write.sum"))
                if "dram__bytes_read.sum" in col
                else m("dram__throughput.avg.pct_of_peak_sustained_elapsed") / 100.0 * 1792.0 * dur_us * 1e-6 * 1e3,
                "l2_mb": m("lts__t_bytes.sum"),
                "l1_mb": m("l1tex__t_bytes.sum"),
                "local_kb": (
                    m("l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum") + m("l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum")
                )
                * 1e3,
                "issue_pct": m("smsp__issue_active.avg.pct_of_peak_sustained_active"),
                "inst": m("smsp__inst_executed.sum"),
                "warp_lat": m("smsp__average_warp_latency_per_inst_issued.ratio"),
                "ffma": m("sm__sass_thread_inst_executed_op_ffma_pred_on.sum")
                + m("sm__sass_thread_inst_executed_op_fadd_pred_on.sum")
                + m("sm__sass_thread_inst_executed_op_fmul_pred_on.sum"),
                "top_stalls": top_stall,
                "smem_b": m("launch__shared_mem_per_block_static") + m("launch__shared_mem_per_block_dynamic"),
            }
        )
    out.sort(key=(lambda r: -r["total_us"]) if args.sort == "time" else (lambda r: r["kernel"]))
    print(
        f"{'tot_us':>7s} {'n':>3s} {'us':>7s} {'regs':>4s} {'thr':>7s} {'wave':>5s} {'occT%':>5s} "
        f"{'occA%':>5s} {'smAct%':>6s} {'SM%':>4s} {'DRAM%':>5s} {'L2%':>4s} {'L1hit':>5s} {'L2hit':>5s} "
        f"{'dramMB':>6s} {'l2MB':>6s} {'locKB':>5s} {'iss%':>4s} {'GF/s':>6s} stalls | kernel"
    )
    for r in out:
        gflops = r["ffma"] / (r["dur_us"] * 1e-6) / 1e9 if r["dur_us"] > 0 else 0
        st = ",".join(f"{k}={v:.1f}" for k, v in r["top_stalls"])
        print(
            f"{r['total_us']:7.0f} {r['n']:3d} {r['dur_us']:7.1f} {r['regs']:4.0f} {r['threads']:7.0f} "
            f"{r['waves']:5.2f} {r['occ_theor_pct']:5.0f} {r['occ_ach_pct']:5.1f} {r['sm_active_pct']:6.1f} "
            f"{r['sm_thr_pct']:4.0f} {r['dram_thr_pct']:5.0f} {r['l2_thr_pct']:4.0f} {r['l1_hit_pct']:5.0f} "
            f"{r['l2_hit_pct']:5.0f} {r['dram_mb']:6.1f} {r['l2_mb']:6.1f} {r['local_kb']:5.0f} "
            f"{r['issue_pct']:4.0f} {gflops:6.0f} {st} | {r['kernel'][:60]}"
        )
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
