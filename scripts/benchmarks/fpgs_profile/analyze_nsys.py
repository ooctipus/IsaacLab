# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analyze an nsys SQLite export produced by run_profiled.py.

Reports, per env step: wall time, GPU busy time (union of all device work),
physics-graph span, idle gaps inside the graph, per-phase GPU/host time, and a
per-kernel table (calls, time, launch shape, registers) with stage classification.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

STAGES = [
    (
        "collide",
        r"compute_shape_aabbs|_nxn_|narrowphase|broadphase|mesh_triangle|reduce_buffered_contacts|export_reduced_contacts|_clear_active_kernel|geom_local_to_global|convert_newton_contacts|broad_phase|broadphase|narrow_phase|manifold|gjk|mpr|sap_|bvh|contact_reduc|nxn|count_contact|prepare_contact|compact_contact|collide|hash_grid|contact_pair|write_contacts|reduce_contact",
    ),
    (
        "actuator",
        r"actuator|pd_control|joint_target|apply_effort|clamp_effort|compute_effort|implicit_actuator|joint_f",
    ),
    ("s1_fk", r"eval_rigid_fk|eval_fk|fk_kinematics|body_q_from|update_kinematics|eval_articulation_fk"),
    (
        "s1_id_tau",
        r"eval_rigid_tau|rigid_tau|finalize_body_dynamics|eval_rigid_id|joint_tau|branch_tau|inverse_dynamics|body_f_s|passive_joint|augmented_joint_drive|drive_desc|prepare_augmented",
    ),
    ("s1_crba", r"composite_inertia|crba_fill|crba|scatter_armature|augmented_mass_diagonal|mass_diag"),
    ("s2_cholesky", r"cholesky|paired_inverse|inverse_cholesky|diagonal_inverse_mass"),
    ("s3_trisolve", r"trisolve|compute_v_hat|v_hat|zero_qdd"),
    (
        "s4_rows",
        r"populate_.*J|prepare_world_contact_rows|contact_rows|joint_limit|velocity_limit|row_|build_rows|contact_bias|mimic|connect|compute_world_contact|restitution|clear_grouped_jacobian|constraint_count|slot|classify|compact_rows|gather_JY|drive_rows",
    ),
    (
        "s4_response",
        r"hinv_jt|paired_hinv|delassus|diag_from_JY|matrix_free_diag|accumulate_hinv_diag|response|mf_body_Hinv|compute_mf_|mf_MiJt|preelim|eff_mass|world_diag|rhs",
    ),
    (
        "s5_pgs",
        r"pgs_solve|pgs_iter|gauss_seidel|apply_impulses|propagate_tree|propagation|prepare_impulses|prepare_world_velocity|pack_mf_meta|warmstart",
    ),
    ("s6_integrate", r"integrate|update_qdd|write_final_velocity|clamp_rigid_velocity|remove_free_root|body_qd_from"),
    ("sensors", r"contact_force|accumulate_contact|frame_transform|imu|sensor|intersect_ray|raycast"),
    ("memset_copy", r"memset|memcpy|fill_kernel|zero_|copy_kernel|array_fill|array_copy"),
    (
        "torch",
        r"at::native|at_cuda|cutlass|cub::|thrust|vectorized_elementwise|elementwise_kernel|reduce_kernel|index_elementwise|gather|scatter|cudnn|cublas|sort|radix|distribution_",
    ),
]
_STAGE_RE = [(name, re.compile(rx, re.I)) for name, rx in STAGES]


def classify(name: str) -> str:
    for stage, rx in _STAGE_RE:
        if rx.search(name):
            return stage
    return "other"


def short(name: str) -> str:
    n = re.sub(r"_[0-9a-f]{8}_cuda_kernel_forward$", "", name)
    n = re.sub(r"_cuda_kernel_forward$", "", n)
    if n.startswith("void "):
        n = n[5:]
    n = n.split("(")[0]
    return n[:110]


def union_len(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals.sort()
    total = 0
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return total + (ce - cs)


def graph_marker(connection: sqlite3.Connection, table: str) -> str:
    """Identify graph activity without assuming memory tables expose graphId."""
    columns = {row[1] for row in connection.execute(f"pragma table_info({table})")}
    markers = [name for name in ("graphId", "graphNodeId") if name in columns]
    if not markers:
        raise ValueError(f"Cannot establish graph membership for {table}: no graphId or graphNodeId")
    return "(" + " or ".join(f"coalesce({name},0)>0" for name in markers) + ")"


def validate_graph_correlations(connection: sqlite3.Connection) -> dict:
    """Require every identified graph operation to have a successful host launch."""
    launches = {}
    for corr, tid, name, result in connection.execute(
        "select r.correlationId,r.globalTid,s.value,r.returnValue "
        "from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId "
        "where s.value like 'cudaGraphLaunch%'"
    ):
        key = (tid & 0xFFFFFFFFFF000000, corr)
        if not corr or result != 0 or key in launches:
            raise ValueError("CUDA graph host launch correlation is missing, duplicated, or unsuccessful")
        launches[key] = name
    counts = {}
    scopes = set()
    seen = set()
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMSET", "CUPTI_ACTIVITY_KIND_MEMCPY"):
        if not connection.execute("select 1 from sqlite_master where type='table' and name=?", (table,)).fetchone():
            continue
        marker = graph_marker(connection, table)
        count = 0
        for pid, device, corr in connection.execute(
            f"select globalPid,deviceId,correlationId from {table} where {marker}"
        ):
            if not corr or (pid, corr) not in launches:
                raise ValueError(
                    f"Missing or unproven CUDA graph launch correlation in {table}; "
                    "use a validated whole-graph capture, not occurrence-order reconstruction"
                )
            scopes.add((pid, device))
            seen.add((pid, corr))
            count += 1
        counts[table] = count
    if len(scopes) > 1:
        raise ValueError("This per-environment-step analyzer requires one CUDA process/device scope")
    if seen != set(launches):
        raise ValueError("Host graph launches and identified device graph operations are incomplete")
    return {"method": "graphId or graphNodeId with successful process-scoped host correlation", "counts": counts}


def analyze_graph_trace(sqlite_path: Path, metadata_path: Path) -> dict:
    """Validate direct graph intervals against host launches and profiler ranges."""
    directory = sqlite_path.parent
    connection = sqlite3.connect(f"file:{sqlite_path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    schema = {
        row[0]: [column[1] for column in connection.execute(f'pragma table_info("{row[0]}")')]
        for row in connection.execute("select name from sqlite_master where type='table'")
    }
    table = "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
    if table not in schema:
        raise ValueError(f"No direct whole-graph table in {directory}; present tables: {sorted(schema)}")
    required = {
        "start",
        "end",
        "deviceId",
        "contextId",
        "streamId",
        "correlationId",
        "globalPid",
        "graphId",
        "graphExecId",
    }
    if not required <= set(schema[table]):
        raise ValueError(f"Unsupported direct graph schema: {schema[table]}")
    strings = dict(connection.execute("select id,value from StringIds"))
    graphs = [dict(row) for row in connection.execute(f"select * from {table} order by start")]
    if not graphs:
        raise ValueError("No direct whole-graph records")
    identities = {
        (
            row["globalPid"],
            row["deviceId"],
            row["contextId"],
            row["graphId"],
            row["graphExecId"],
        )
        for row in graphs
    }
    if len(identities) != 1:
        raise ValueError(f"Only a single fixed physics graph is supported: {identities}")
    launches = [
        dict(row)
        for row in connection.execute("select * from CUPTI_ACTIVITY_KIND_RUNTIME order by start")
        if strings.get(row["nameId"], "").startswith("cudaGraphLaunch")
    ]
    if len(launches) != len(graphs) or any(row["returnValue"] != 0 for row in launches):
        raise ValueError("Successful host GraphLaunch count does not match direct graph count")
    if any(not row["correlationId"] for row in graphs + launches):
        raise ValueError("Missing direct graph/runtime correlation; do not infer membership from occurrence order")
    keyed = {(row["globalTid"] & 0xFFFFFFFFFF000000, row["correlationId"]): row for row in launches}
    if len(keyed) != len(launches):
        raise ValueError("Runtime graph correlations are not unique within process")
    ranges = []
    for row in connection.execute("select * from NVTX_EVENTS where end is not null and eventType in (59,60,70)"):
        label = row["text"] if row["text"] is not None else strings.get(row["textId"], "")
        if label == "physics_graph" or label.startswith("env_step:"):
            ranges.append((row["start"], row["end"], row["globalTid"], label))
    steps = sorted(row for row in ranges if row[3].startswith("env_step:"))
    physics_ranges = [row for row in ranges if row[3] == "physics_graph"]
    meta = json.loads(metadata_path.read_text())
    if len(steps) != meta["profile_steps"] or len(physics_ranges) != len(graphs):
        raise ValueError("Profiler-window step/physics_graph ranges are incomplete")
    if not meta["cuda_graph"] or not all(meta[key].get("state_finite") is True for key in ("model", "model_after")):
        raise ValueError("Expected finite graph-enabled simulation")
    if [row[3] for row in steps] != [f"env_step:{i}" for i in range(len(steps))]:
        raise ValueError("Profiler window step labels differ from expected sequence")
    if any(a[1] > b[0] for a, b in zip(steps, steps[1:])):
        raise ValueError("Environment step host ranges overlap")
    if any(a["end"] > b["start"] for a, b in zip(graphs, graphs[1:])):
        raise ValueError("Direct physics graph executions overlap")
    per_step = {step[3]: [] for step in steps}
    seen = set()
    for graph in graphs:
        key = graph["globalPid"], graph["correlationId"]
        launch = keyed.get(key)
        if launch is None or key in seen or graph["start"] < launch["start"] or graph["end"] <= graph["start"]:
            raise ValueError("Direct graph cannot be uniquely matched to a successful preceding host launch")
        seen.add(key)
        containing_physics = [
            row
            for row in physics_ranges
            if row[2] == launch["globalTid"] and row[0] <= launch["start"] <= launch["end"] <= row[1]
        ]
        containing_steps = [
            row
            for row in steps
            if row[2] == launch["globalTid"] and row[0] <= launch["start"] <= launch["end"] <= row[1]
        ]
        if len(containing_physics) != 1 or len(containing_steps) != 1:
            raise ValueError("Graph launch is not uniquely inside physics_graph and env_step host ranges")
        per_step[containing_steps[0][3]].append({**graph, "duration_us": (graph["end"] - graph["start"]) / 1000.0})
    expected = meta["host_calls_per_step"]["physics_graph"]
    if any(len(values) != expected for values in per_step.values()):
        raise ValueError("Per-step graph launch count differs from unprofiled instrumentation")
    for work_table in (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMSET",
    ):
        if work_table not in schema:
            continue
        markers = [name for name in ("graphId", "graphNodeId") if name in schema[work_table]]
        if (
            markers
            and connection.execute(
                f"select count(*) from {work_table} where " + " or ".join(f"{name}>0" for name in markers)
            ).fetchone()[0]
        ):
            raise ValueError("Node activity is present despite requested whole-graph-only capture")
    values = [sum(row["duration_us"] for row in per_step[label]) for label in per_step]
    connection.close()
    return {
        "analysis_mode": "graph",
        "summary": {
            "timing_mode": "graph",
            "steps": len(steps),
            "graph_span_us_per_step": statistics.fmean(values),
            "graph_launches_per_step": expected,
            "wall_us_per_step": statistics.fmean((row[1] - row[0]) / 1000.0 for row in steps),
        },
        "graph_membership_audit": {"method": "direct graph records with process-scoped host correlations"},
        "directory": str(directory),
        "validated": True,
        "method": (
            "Direct CUPTI_ACTIVITY_KIND_GRAPH_TRACE start/end matched by process+correlation "
            "to successful host launches and NVTX ranges"
        ),
        "task": meta["task"],
        "num_envs": meta["num_envs"],
        "schema": {table: schema[table]},
        "graph_identity": list(next(iter(identities))),
        "graphs_per_env_step": expected,
        "mean_whole_graph_us_per_env_step": statistics.fmean(values),
        "median_whole_graph_us_per_env_step": statistics.median(values),
        "per_step_whole_graph_us": values,
        "per_step": [
            {
                "label": row[3],
                "wall_us": (row[1] - row[0]) / 1000.0,
                "graph_launches": [
                    {"corr": graph["correlationId"], "span_us": graph["duration_us"]} for graph in per_step[row[3]]
                ],
            }
            for row in steps
        ],
        "direct_graphs_by_step": per_step,
        "kernels": [],
        "stages": {},
        "phases": [],
        "unavailable_metrics": ["per-node durations", "graph busy/idle", "stage budgets"],
    }


def main() -> None:  # noqa: C901
    p = argparse.ArgumentParser()
    p.add_argument("sqlite", type=Path)
    p.add_argument("--json", type=Path)
    p.add_argument("--top", type=int, default=60)
    p.add_argument("--sequence", action="store_true", help="Print the ordered kernel sequence of one physics graph.")
    p.add_argument("--sequence-step", type=int, default=0)
    p.add_argument(
        "--require-graph-trace",
        action="store_true",
        help="Require direct whole-graph records; never fall back to nodes.",
    )
    p.add_argument(
        "--run-metadata", type=Path, help="run_profiled.py JSON (defaults to the SQLite basename with .json)."
    )
    args = p.parse_args()
    c = sqlite3.connect(f"file:{args.sqlite.resolve()}?mode=ro", uri=True)
    has_graph_trace = c.execute(
        "select 1 from sqlite_master where type='table' and name='CUPTI_ACTIVITY_KIND_GRAPH_TRACE'"
    ).fetchone()
    if args.require_graph_trace or has_graph_trace:
        c.close()
        result = analyze_graph_trace(args.sqlite, args.run_metadata or args.sqlite.with_suffix(".json"))
        if args.json:
            with args.json.open("x") as output:
                output.write(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result["summary"], indent=2))
        return
    membership_audit = validate_graph_correlations(c)
    strings = {i: v for i, v in c.execute("select id, value from StringIds")}

    # NVTX ranges (push/pop) with resolved text.
    nvtx = []
    for start, end, text, textId, tid in c.execute(
        "select start, end, text, textId, globalTid from NVTX_EVENTS where end is not null and eventType in (59,60,70)"
    ):
        label = text if text is not None else strings.get(textId, "?")
        nvtx.append((start, end, label, tid))
    nvtx.sort()
    steps = [(s, e, lbl) for s, e, lbl, _ in nvtx if lbl.startswith("env_step:")]
    phases = [(s, e, lbl, tid) for s, e, lbl, tid in nvtx if not lbl.startswith("env_step:")]

    # Runtime API rows keyed by correlation id.
    api = {}
    for start, end, corr, tid, nameId in c.execute(
        "select start, end, correlationId, globalTid, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        if corr:
            if corr in api and api[corr][2] & 0xFFFFFFFFFF000000 != tid & 0xFFFFFFFFFF000000:
                raise ValueError("Runtime correlations from multiple processes would collide")
            api[corr] = (start, end, tid, strings.get(nameId, "?"))

    def innermost_phase(t: int, tid: int) -> str:
        best = None
        for s, e, lbl, ptid in phases:
            if ptid == tid and s <= t <= e:
                if best is None or (e - s) < (best[1] - best[0]):
                    best = (s, e, lbl)
        return best[2] if best else "unattributed"

    # Device work: kernels + memset + memcpy.
    work = []  # (start, end, kind, name, attrs, corr, graphId, streamId)
    for row in c.execute(
        "select start, end, shortName, demangledName, correlationId, "
        f"{graph_marker(c, 'CUPTI_ACTIVITY_KIND_KERNEL')}, streamId, gridX, gridY, gridZ, "
        "blockX, blockY, blockZ, registersPerThread, staticSharedMemory, dynamicSharedMemory, localMemoryPerThread "
        "from CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        start, end, sn, dn, corr, gid, sid = row[:7]
        name = strings.get(sn) or strings.get(dn) or "?"
        work.append((start, end, "kernel", name, row[7:], corr, gid or 0, sid))

    for start, end, corr, gid, sid, nbytes in c.execute(
        f"select start, end, correlationId, {graph_marker(c, 'CUPTI_ACTIVITY_KIND_MEMSET')}, streamId, bytes "
        "from CUPTI_ACTIVITY_KIND_MEMSET"
    ):
        work.append((start, end, "memset", f"memset[{nbytes}B]", None, corr, gid or 0, sid))
    for start, end, corr, gid, sid, nbytes, kind in c.execute(
        f"select start, end, correlationId, {graph_marker(c, 'CUPTI_ACTIVITY_KIND_MEMCPY')}, streamId, bytes, copyKind "
        "from CUPTI_ACTIVITY_KIND_MEMCPY"
    ):
        work.append((start, end, "memcpy", f"memcpy[kind{kind},{nbytes}B]", None, corr, gid or 0, sid))
    work.sort()

    if not steps:
        print("No env_step NVTX ranges found; using whole capture as one step.")
        steps = [(work[0][0], work[-1][1], "env_step:all")]

    per_step = []
    kernel_agg = defaultdict(lambda: {"n": 0, "sum": 0, "min": 1e18, "max": 0, "attrs": None, "graph": 0})
    phase_gpu = defaultdict(int)
    phase_host = defaultdict(int)
    phase_count = defaultdict(int)
    graph_seqs = []
    for s_start, s_end, lbl in steps:
        # Attribute device work by launch-API timestamp (host side) falling in this step.
        step_work = []
        for w in work:
            a = api.get(w[5])
            t_launch = a[0] if a else w[0]
            if s_start <= t_launch <= s_end:
                step_work.append((w, a))
        if not step_work:
            continue
        gpu_union = union_len([(w[0], w[1]) for w, _ in step_work])
        graph_work = [(w, a) for w, a in step_work if w[6] > 0]
        graph_launches = defaultdict(list)
        for w, a in graph_work:
            graph_launches[w[5]].append(w)
        spans = []
        for corr, ws in graph_launches.items():
            gs = min(w[0] for w in ws)
            ge = max(w[1] for w in ws)
            busy = union_len([(w[0], w[1]) for w in ws])
            summed = sum(w[1] - w[0] for w in ws)
            spans.append(
                {
                    "span_us": (ge - gs) / 1e3,
                    "busy_us": busy / 1e3,
                    "summed_us": summed / 1e3,
                    "idle_us": (ge - gs - busy) / 1e3,
                    "nodes": len(ws),
                    "kernel_nodes": sum(w[2] == "kernel" for w in ws),
                    "memory_nodes": sum(w[2] != "kernel" for w in ws),
                    "kernel_summed_us": sum(w[1] - w[0] for w in ws if w[2] == "kernel") / 1e3,
                    "memory_summed_us": sum(w[1] - w[0] for w in ws if w[2] != "kernel") / 1e3,
                    "corr": corr,
                }
            )
        graph_seqs.append(sorted(graph_work, key=lambda wa: wa[0][0]))
        # Phase attribution.
        for w, a in step_work:
            ph = innermost_phase(a[0], a[2]) if a else "unattributed"
            phase_gpu[ph] += w[1] - w[0]
        for ps, pe, plbl, _ in phases:
            if s_start <= ps <= s_end:
                phase_host[plbl] += pe - ps
                phase_count[plbl] += 1
        # Kernel aggregation (all kernels, graph or not).
        for w, a in step_work:
            k = kernel_agg[(w[3], w[6] > 0)]
            d = w[1] - w[0]
            k["n"] += 1
            k["sum"] += d
            k["min"] = min(k["min"], d)
            k["max"] = max(k["max"], d)
            k["attrs"] = w[4]
            k["graph"] = int(w[6] > 0)
            k["kind"] = w[2]
        per_step.append(
            {
                "label": lbl,
                "wall_us": (s_end - s_start) / 1e3,
                "gpu_busy_us": gpu_union / 1e3,
                "device_ops": len(step_work),
                "graph_launches": spans,
                "non_graph_kernels": sum(1 for w, _ in step_work if w[6] == 0 and w[2] == "kernel"),
            }
        )
    nsteps = len(per_step)

    def avg(key, sub=None):
        vals = [s[key] if sub is None else sum(g[sub] for g in s["graph_launches"]) for s in per_step]
        return statistics.fmean(vals) if vals else 0.0

    summary = {
        "timing_mode": "node",
        "steps": nsteps,
        "wall_us_per_step": avg("wall_us"),
        "gpu_busy_us_per_step": avg("gpu_busy_us"),
        "graph_span_us_per_step": avg(None, "span_us"),
        "graph_busy_us_per_step": avg(None, "busy_us"),
        "graph_summed_kernel_us_per_step": avg(None, "kernel_summed_us"),
        "graph_summed_device_us_per_step": avg(None, "summed_us"),
        "graph_summed_memory_us_per_step": avg(None, "memory_summed_us"),
        "graph_idle_us_per_step": avg(None, "idle_us"),
        "graph_nodes_per_step": avg(None, "nodes"),
        "graph_kernel_nodes_per_step": avg(None, "kernel_nodes"),
        "graph_memory_nodes_per_step": avg(None, "memory_nodes"),
        "graph_launches_per_step": statistics.fmean(len(s["graph_launches"]) for s in per_step),
        "non_graph_kernels_per_step": avg("non_graph_kernels"),
    }
    summary["non_graph_gpu_us_per_step"] = summary["gpu_busy_us_per_step"] - summary["graph_busy_us_per_step"]
    summary["host_gap_us_per_step"] = summary["wall_us_per_step"] - summary["gpu_busy_us_per_step"]

    phase_rows = []
    for ph in sorted(set(phase_gpu) | set(phase_host), key=lambda k: -phase_gpu.get(k, 0)):
        phase_rows.append(
            {
                "phase": ph,
                "gpu_us": phase_gpu.get(ph, 0) / 1e3 / nsteps,
                "host_us": phase_host.get(ph, 0) / 1e3 / nsteps,
                "calls": phase_count.get(ph, 0) / nsteps,
            }
        )

    stage_agg = defaultdict(lambda: {"us": 0.0, "n": 0.0})
    krows = []
    for (name, ingraph), k in kernel_agg.items():
        st = classify(name)
        us = k["sum"] / 1e3 / nsteps
        stage_agg[(st, ingraph)]["us"] += us
        stage_agg[(st, ingraph)]["n"] += k["n"] / nsteps
        attrs = k["attrs"]
        krows.append(
            {
                "kernel": short(name),
                "stage": st,
                "graph": ingraph,
                "kind": k["kind"],
                "calls_per_step": k["n"] / nsteps,
                "us_per_step": us,
                "avg_us": k["sum"] / k["n"] / 1e3,
                "min_us": k["min"] / 1e3,
                "max_us": k["max"] / 1e3,
                "grid": list(attrs[0:3]) if attrs else None,
                "block": list(attrs[3:6]) if attrs else None,
                "regs": attrs[6] if attrs else None,
                "smem": (attrs[7] + attrs[8]) if attrs else None,
                "local_mem": attrs[9] if attrs else None,
                "full": name,
            }
        )
    krows.sort(key=lambda r: -r["us_per_step"])

    # Report.
    print("=" * 100)
    print(f"Per env step (mean of {nsteps}):")
    for k, v in summary.items():
        rendered = f"{v:12.1f}" if isinstance(v, (float, int)) else str(v)
        print(f"  {k:36s} {rendered}")
    print("\nPhase attribution (per env step, us):")
    print(f"  {'phase':28s} {'gpu_us':>10s} {'host_us':>10s} {'calls':>7s}")
    for r in phase_rows:
        print(f"  {r['phase']:28s} {r['gpu_us']:10.1f} {r['host_us']:10.1f} {r['calls']:7.1f}")
    print("\nStage totals (per env step):")
    for (st, ing), v in sorted(stage_agg.items(), key=lambda kv: -kv[1]["us"]):
        print(f"  {st:16s} {'graph' if ing else 'eager':6s} {v['us']:10.1f} us  {v['n']:7.1f} launches")
    print("\nTop kernels (per env step):")
    print(f"  {'us/step':>8s} {'calls':>6s} {'avg_us':>7s} {'regs':>4s} {'grid':>9s} {'blk':>5s} {'stage':12s} kernel")
    for r in krows[: args.top]:
        g = r["grid"][0] if r["grid"] else 0
        b = r["block"][0] if r["block"] else 0
        print(
            f"  {r['us_per_step']:8.1f} {r['calls_per_step']:6.1f} {r['avg_us']:7.2f} {str(r['regs']):>4s} "
            f"{g:9d} {b:5d} {r['stage']:12s} {r['kernel'][:70]}"
        )
    if args.sequence and graph_seqs:
        seq = graph_seqs[min(args.sequence_step, len(graph_seqs) - 1)]
        print("\nOrdered graph kernel sequence (one env step): gap_before_us, dur_us, stream, grid, block, regs, name")
        prev_end = None
        for w, _ in seq:
            gap = (w[0] - prev_end) / 1e3 if prev_end is not None else 0.0
            prev_end = max(prev_end or 0, w[1])
            a = w[4]
            g = a[0] if a else 0
            b = a[3] if a else 0
            print(
                f"  {gap:7.2f} {(w[1] - w[0]) / 1e3:8.2f} s{w[7]:<3d} {g:8d} {b:4d} "
                f"{str(a[6] if a else ''):>4s} {short(w[3])[:80]}"
            )
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "analysis_mode": "node",
                    "graph_membership_audit": membership_audit,
                    "phases": phase_rows,
                    "kernels": krows,
                    "stages": {f"{st}|{'graph' if ing else 'eager'}": v for (st, ing), v in stage_agg.items()},
                    "per_step": per_step,
                },
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
