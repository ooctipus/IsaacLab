# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scorecard across run_profiled/analyze_nsys outputs: end-to-end vs physics-graph metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
rows = []
for run in sorted(d.glob("*.json")):
    if run.name.endswith(("_analysis.json", "_roofline.json")):
        continue
    ana = d / (run.stem + "_analysis.json")
    if not ana.exists():
        continue
    r = json.loads(run.read_text())
    a = json.loads(ana.read_text())
    s = a["summary"]
    st = a["stages"]
    subs = r["model"]["num_substeps"] * r["decimation"]
    collide = st.get("collide|graph", {}).get("us", 0.0)
    sensors = st.get("sensors|graph", {}).get("us", 0.0)
    solver = s["graph_busy_us_per_step"] - collide - sensors
    rows.append(
        {
            "run": run.stem,
            "task": r["task"],
            "physics": r["physics"] + ("+" + ",".join(r.get("solver_attr", [])) if r.get("solver_attr") else ""),
            "envs": r["num_envs"],
            "fps_k": r["fps"]["mean"] / 1e3,
            "ms_step": r["ms_per_step_sync"],
            "graph_span_ms": s["graph_span_us_per_step"] / 1e3,
            "graph_busy_ms": s["graph_busy_us_per_step"] / 1e3,
            "collide_us": collide,
            "solver_us_substep": solver / subs,
            "subs": subs,
            "nodes": s["graph_nodes_per_step"],
            "eager_gpu_ms": s["non_graph_gpu_us_per_step"] / 1e3,
            "host_gap_ms": s["host_gap_us_per_step"] / 1e3,
            "wall_nsys_ms": s["wall_us_per_step"] / 1e3,
        }
    )
hdr = (
    f"{'run':20s} {'physics':26s} {'FPS(k)':>8s} {'ms/step':>8s} {'graph ms':>9s} {'busy ms':>8s} "
    f"{'collide us':>10s} {'solver us/sub':>13s} {'nodes':>6s} {'eager ms':>9s} {'hostgap ms':>10s}"
)
print(hdr)
for x in rows:
    print(
        f"{x['run']:20s} {x['physics'][:26]:26s} {x['fps_k']:8.0f} {x['ms_step']:8.2f} "
        f"{x['graph_span_ms']:9.2f} {x['graph_busy_ms']:8.2f} {x['collide_us']:10.0f} "
        f"{x['solver_us_substep']:13.0f} {x['nodes']:6.0f} {x['eager_gpu_ms']:9.2f} "
        f"{x['host_gap_ms']:10.2f}"
    )
(d / "scorecard.json").write_text(json.dumps(rows, indent=1))
