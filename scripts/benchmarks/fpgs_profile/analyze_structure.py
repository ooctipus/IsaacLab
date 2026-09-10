# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read saved Nsight traces and produce task/hardware-separated work budgets."""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sqlite3
import statistics
from pathlib import Path


def category(kernel: dict) -> str:
    """Separate fused work from external stages without double-counting it."""
    name, stage = kernel["kernel"], kernel["stage"]
    if name.startswith("pgs_solve_parallel_"):
        return "parallel_projection_including_fused_rows_response"
    if name.startswith("pgs_solve_mf_gs_"):
        return "fallback_gauss_seidel"
    if stage == "collide":
        return "collision"
    if name.startswith("fused_dynamics_"):
        return "fused_dynamics_K1"
    if stage in ("s1_crba", "s2_cholesky") or name in ("build_mass_update_mask", "refresh_masked_body_inertia"):
        return "mass_assembly_factorization"
    if stage == "s3_trisolve" or name in ("compute_velocity_predictor", "apply_free_root_transport_to_predictor"):
        return "velocity_predictor_trisolve"
    if stage == "s1_id_tau":
        return "other_dynamics"
    if stage == "s1_fk":
        return "published_kinematics"
    if stage in ("s4_rows", "s5_pgs") or name in (
        "build_world_contact_lists",
        "prepare_world_impulses",
        "snapshot_dense_phase_bound",
    ):
        return "external_constraint_rows_setup"
    if stage == "s4_response":
        return "remaining_constraint_response"
    if stage == "sensors" or name in (
        "compute_contact_linear_force_from_impulses",
        "pack_contact_linear_force_as_spatial",
        "compute_sensing_transforms_kernel",
    ):
        return "sensors_contact_forces"
    if stage == "s6_integrate":
        return "integration"
    return "other_device_work"


def short(name: str) -> str:
    """Remove Warp's generated module suffix from a kernel name."""
    return re.sub(r"_[0-9a-f]{8}_cuda_kernel_forward$", "", name)


def sqlite_audit(directory: Path, analysis: dict) -> dict:
    """Verify graph spans and busy unions directly against read-only SQLite."""
    connection = sqlite3.connect(f"file:{directory / 'capture.sqlite'}?mode=ro", uri=True)
    ids = {row[0]: row[1] for row in connection.execute("select id, value from StringIds")}
    correlations = {graph["corr"] for step in analysis["per_step"] for graph in step["graph_launches"]}
    kernels = collections.defaultdict(list)
    graph_intervals = collections.defaultdict(list)
    for start, end, name_id, corr, graph in connection.execute(
        "select start,end,shortName,correlationId,graphId from CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        if graph and corr in correlations:
            kernels[short(ids[name_id])].append((end - start) / 1000.0)
            graph_intervals[corr].append((start, end))
    for table in ("CUPTI_ACTIVITY_KIND_MEMSET", "CUPTI_ACTIVITY_KIND_MEMCPY"):
        columns = {row[1] for row in connection.execute(f"pragma table_info({table})")}
        if "graphId" not in columns:
            continue
        for start, end, corr, graph in connection.execute(f"select start,end,correlationId,graphId from {table}"):
            if graph and corr in correlations:
                graph_intervals[corr].append((start, end))
    graph_span_ns, graph_busy_ns = 0, 0
    gaps = []
    for intervals in graph_intervals.values():
        intervals.sort()
        graph_span_ns += max(end for _, end in intervals) - intervals[0][0]
        last = intervals[0][0]
        for start, end in intervals:
            if start > last:
                gaps.append((start - last) / 1000.0)
            graph_busy_ns += max(0, end - max(last, start))
            last = max(last, end)
    steps = len(analysis["per_step"])
    for name, value in (
        ("graph_span_us_per_step", graph_span_ns / 1000 / steps),
        ("graph_busy_us_per_step", graph_busy_ns / 1000 / steps),
    ):
        assert math.isclose(value, analysis["summary"][name], abs_tol=1.0e-6), (directory, name, value)
    stats = {}
    for name, durations in kernels.items():
        ordered = sorted(durations)
        stats[name] = {
            "count": len(durations),
            "min_us": min(durations),
            "median_us": statistics.median(durations),
            "p90_us": ordered[math.ceil(0.9 * len(ordered)) - 1],
            "max_us": max(durations),
            "sum_us_per_step": sum(durations) / steps,
        }
    connection.close()
    return {
        "recomputed_span_us_per_step": graph_span_ns / 1000 / steps,
        "recomputed_busy_us_per_step": graph_busy_ns / 1000 / steps,
        "gaps_per_step": len(gaps) / steps,
        "median_gap_us": statistics.median(gaps) if gaps else 0,
        "max_gap_us": max(gaps, default=0),
        "kernel_duration_distribution": stats,
    }


def summarize_run(manifest_run: dict, study_directory: Path) -> dict:
    """Build one internally consistent work budget from a saved capture."""
    directory = study_directory / Path(manifest_run["output_dir"]).name
    analysis = json.loads((directory / "capture_analysis.json").read_text())
    meta = json.loads((directory / "capture.json").read_text())
    substeps = meta["decimation"] * meta["model"]["num_substeps"]
    if substeps <= 0:
        raise ValueError(f"Invalid solver substep count in {directory}")
    stages = collections.defaultdict(float)
    calls = collections.defaultdict(float)
    hot = []
    for kernel in analysis["kernels"]:
        if not kernel["graph"]:
            continue
        stage = category(kernel)
        stages[stage] += kernel["us_per_step"] / substeps
        calls[stage] += kernel["calls_per_step"] / substeps
        hot.append(
            {
                "name": short(kernel["full"]),
                "category": stage,
                "us_per_substep": kernel["us_per_step"] / substeps,
                "calls_per_substep": kernel["calls_per_step"] / substeps,
                "avg_call_us": kernel["avg_us"],
                "regs": kernel["regs"],
                "grid": kernel["grid"],
                "block": kernel["block"],
            }
        )
    return {
        "directory": str(directory),
        "round": manifest_run["round"],
        "task": manifest_run["task"],
        "gpu": manifest_run["gpu_index"],
        "revision": manifest_run["newton"],
        "substeps_per_env_step": substeps,
        "summary_per_env_step": analysis["summary"],
        "stages_us_per_substep": dict(stages),
        "stage_calls_per_substep": dict(calls),
        "hot_kernels": sorted(hot, key=lambda row: -row["us_per_substep"]),
        "finite": meta["model"]["state_finite"] and meta["model_after"]["state_finite"],
        "contacts_before": meta["model"]["contacts_active"],
        "contacts_after": meta["model_after"]["contacts_active"],
        "sqlite_audit": sqlite_audit(directory, analysis),
    }


def amdahl(run: dict) -> dict:
    """Estimate optimistic ceilings; summed work can include overlap."""
    stages = run["stages_us_per_substep"]
    span = run["summary_per_env_step"]["graph_span_us_per_step"] / run["substeps_per_env_step"]
    parallel = stages["parallel_projection_including_fused_rows_response"]
    constraint = parallel + sum(
        stages.get(key, 0)
        for key in (
            "fallback_gauss_seidel",
            "external_constraint_rows_setup",
            "remaining_constraint_response",
        )
    )
    dynamics = sum(
        stages.get(key, 0)
        for key in (
            "fused_dynamics_K1",
            "mass_assembly_factorization",
            "velocity_predictor_trisolve",
            "other_dynamics",
            "published_kinematics",
            "integration",
        )
    )
    groups = {
        "parallel_only": parallel,
        "all_constraint_work": constraint,
        "collision_only": stages["collision"],
        "collision_and_parallel": stages["collision"] + parallel,
        "all_constraint_and_collision": constraint + stages["collision"],
        "all_constraint_and_dynamics": constraint + dynamics,
    }
    return {
        name: {
            "summed_work_us_per_substep": value,
            "optimistic_work_fraction": value / span,
            "optimistic_infinite_speedup_ceiling": span / (span - value) if value < span else None,
            "required_group_speedup_for_2x": value / (value - span * 0.5) if value > span * 0.5 else None,
            "required_group_speedup_for_4x": value / (value - span * 0.75) if value > span * 0.75 else None,
        }
        for name, value in groups.items()
    }


def main() -> None:
    """Analyze saved compare_gpus.py outputs without starting GPU work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    studies = []
    for directory in args.directories:
        manifest = json.loads((directory / "manifest.json").read_text())
        runs = [summarize_run(run, directory) for run in manifest["runs"] if "result" in run]
        grouped = collections.defaultdict(list)
        for run in runs:
            grouped[(run["task"], run["gpu"], run["revision"])].append(run)
        representatives = []
        for key, rows in grouped.items():
            rows.sort(key=lambda row: row["summary_per_env_step"]["graph_span_us_per_step"])
            middle = rows[len(rows) // 2]
            middle["amdahl_optimistic"] = amdahl(middle)
            representatives.append(middle)
        studies.append(
            {
                "directory": str(directory),
                "gpus": manifest["gpus"],
                "status": manifest["status"],
                "runs": runs,
                "median_graph_representatives": representatives,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(studies, indent=2, allow_nan=False) + "\n")
    for study in studies:
        print("STUDY", study["directory"])
        for run in study["median_graph_representatives"]:
            print("\n", run["task"], "GPU", run["gpu"], run["revision"], "round", run["round"])
            summary = run["summary_per_env_step"]
            print(
                "graph_span,busy,sum,idle /substep:",
                *(
                    round(summary[key] / run["substeps_per_env_step"], 2)
                    for key in (
                        "graph_span_us_per_step",
                        "graph_busy_us_per_step",
                        "graph_summed_kernel_us_per_step",
                        "graph_idle_us_per_step",
                    )
                ),
            )
            print("stages /substep:", {key: round(value, 2) for key, value in run["stages_us_per_substep"].items()})
            print("top kernels:", [(row["name"], round(row["us_per_substep"], 2)) for row in run["hot_kernels"][:8]])
            print(
                "amdahl:",
                {
                    key: {k: round(v, 3) if v is not None else None for k, v in value.items()}
                    for key, value in run.get("amdahl_optimistic", {}).items()
                },
            )


if __name__ == "__main__":
    main()
