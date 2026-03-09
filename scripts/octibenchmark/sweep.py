# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sweep num_envs for a task, profile with nsys, and log results to wandb.

Runs the benchmark script under ``nsys profile`` for each ``num_envs`` value,
parses the results, and uploads charts to wandb where x-axis is ``num_envs``
and y-axis is time (ms) or percentage for each profiled component.

Usage::

    python scripts/octibenchmark/sweep.py \\
        --task Isaac-Repose-Cube-Shadow-Vision-Direct-v0 \\
        --num_envs 64 256 1024 \\
        --num_frames 100 \\
        --wandb_project isaaclab-benchmarks

    # With kernel pattern comparison (e.g. index vs mask writes)
    python scripts/octibenchmark/sweep.py \\
        --task Isaac-Repose-Cube-Shadow-Vision-Direct-v0 \\
        --num_envs 64 256 1024 \\
        --kernel_patterns "scatter|index_put" "mask|where" \\
        --wandb_project isaaclab-benchmarks

    # Skip wandb, just print results
    python scripts/octibenchmark/sweep.py \\
        --task Isaac-Repose-Cube-Shadow-Vision-Direct-v0 \\
        --num_envs 64 256 \\
        --no_wandb
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Add scripts/ to path so octibenchmark is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Resolve paths relative to the repo root
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_BENCHMARK_SCRIPT = _SCRIPT_DIR / "benchmark.py"
_ISAACLAB_SH = _REPO_ROOT / "isaaclab.sh"


def run_single_benchmark(
    task: str,
    num_envs: int,
    num_frames: int,
    warmup_frames: int,
    output_dir: str,
    extra_args: list[str],
    enable_cameras: bool = False,
) -> str:
    """Run a single benchmark under nsys profile.

    Args:
        task: Registered task name.
        num_envs: Number of environments.
        num_frames: Number of env steps to profile.
        warmup_frames: Warmup steps before nsys capture.
        output_dir: Directory for nsys output files.
        extra_args: Additional CLI arguments (e.g. hydra presets).
        enable_cameras: Whether to pass --enable_cameras.

    Returns:
        Path to the generated ``.nsys-rep`` file.
    """
    output_name = f"{task}_envs{num_envs}"
    output_path = os.path.join(output_dir, output_name)

    nsys_cmd = [
        "nsys", "profile",
        "-t", "cuda,nvtx",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "-o", output_path,
        "--force-overwrite=true",
        str(_ISAACLAB_SH), "-p",
        str(_BENCHMARK_SCRIPT),
        "--task", task,
        "--num_envs", str(num_envs),
        "--num_frames", str(num_frames),
        "--warmup_frames", str(warmup_frames),
        "--headless",
    ]

    if enable_cameras:
        nsys_cmd.append("--enable_cameras")

    nsys_cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"[sweep] Running: num_envs={num_envs}")
    print(f"[sweep] Command: {' '.join(nsys_cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(nsys_cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[WARNING] nsys exited with code {result.returncode} for num_envs={num_envs}")

    nsys_rep_path = output_path + ".nsys-rep"
    if not os.path.exists(nsys_rep_path):
        raise FileNotFoundError(f"Expected nsys output not found: {nsys_rep_path}")

    return nsys_rep_path


def collect_results(
    nsys_rep_path: str,
    kernel_patterns: list[str] | None = None,
) -> dict:
    """Analyze a single nsys-rep file and return structured results.

    Args:
        nsys_rep_path: Path to the ``.nsys-rep`` file.
        kernel_patterns: Optional regex patterns for kernel aggregation.

    Returns:
        Analysis result dict.
    """
    # Import here to avoid pulling in sqlite3 etc. at module level
    from octibenchmark.analyze import analyze

    return analyze(nsys_rep_path, kernel_patterns)


def log_to_wandb(
    all_results: dict[int, dict],
    task: str,
    wandb_project: str,
    wandb_entity: str | None,
    kernel_patterns: list[str] | None = None,
):
    """Upload sweep results to wandb.

    Creates charts with x-axis = num_envs and y-axis = time per component.

    Args:
        all_results: Mapping from num_envs to analysis results.
        task: Task name (used in run name).
        wandb_project: wandb project name.
        wandb_entity: wandb entity (team/user). None for default.
        kernel_patterns: Kernel patterns used (for labeling).
    """
    import wandb

    # Extract presets string from extra_args for labeling
    presets_str = "default"
    for key in list(all_results.values())[0].get("extra_args", []):
        if key.startswith("presets="):
            presets_str = key.split("=", 1)[1]

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=f"sweep_{task}_{presets_str}",
        config={
            "task": task,
            "num_envs_values": sorted(all_results.keys()),
            "kernel_patterns": kernel_patterns,
            "presets": presets_str,
        },
    )

    # Build per-component tables for wandb
    # Each NVTX range becomes a line on the chart
    sorted_envs = sorted(all_results.keys())

    # Collect all NVTX range names across all runs
    all_nvtx_names = set()
    for result in all_results.values():
        for r in result.get("nvtx_ranges", []):
            all_nvtx_names.add(r["name"])

    # Log NVTX range data: one chart per metric (avg_ms, pct_of_step)
    for num_envs in sorted_envs:
        result = all_results[num_envs]
        nvtx_ranges = result.get("nvtx_ranges", [])

        # Find env.step total for percentage
        step_total_ns = 0
        for r in nvtx_ranges:
            if r["name"] == "env.step":
                step_total_ns = r["total_ns"]
                break

        log_data = {"num_envs": num_envs}

        for r in nvtx_ranges:
            name = r["name"]
            avg_ms = r["avg_ns"] / 1e6
            pct = (r["total_ns"] / step_total_ns * 100) if step_total_ns else 0

            log_data[f"avg_ms/{name}"] = avg_ms
            log_data[f"pct_of_step/{name}"] = pct
            log_data[f"total_ms/{name}"] = r["total_ns"] / 1e6

        # Effective FPS: num_envs / avg_step_time_s
        for r in nvtx_ranges:
            if r["name"] == "env.step":
                avg_step_s = r["avg_ns"] / 1e9
                if avg_step_s > 0:
                    log_data["effective_fps"] = num_envs / avg_step_s
                    log_data["step_fps"] = 1.0 / avg_step_s
                break

        # Kernel pattern aggregations
        if result.get("kernel_aggregations"):
            for pattern, stats in result["kernel_aggregations"].items():
                log_data[f"kernel_pattern_ms/{pattern}"] = stats["total_ms"]
                log_data[f"kernel_pattern_calls/{pattern}"] = stats["count"]

        wandb.log(log_data)

    # Also log a summary table
    table_data = []
    for num_envs in sorted_envs:
        result = all_results[num_envs]
        for r in result.get("nvtx_ranges", []):
            table_data.append([
                num_envs,
                r["name"],
                r["count"],
                r["total_ns"] / 1e6,
                r["avg_ns"] / 1e6,
            ])
    table = wandb.Table(
        columns=["num_envs", "nvtx_range", "calls", "total_ms", "avg_ms"],
        data=table_data,
    )
    wandb.log({"nvtx_breakdown": table})

    run.finish()
    print(f"\n[sweep] wandb run finished: {run.url}")


def print_summary(all_results: dict[int, dict]):
    """Print a text summary of sweep results.

    Args:
        all_results: Mapping from num_envs to analysis results.
    """
    from octibenchmark.analyze import format_nvtx_table

    sorted_envs = sorted(all_results.keys())
    for num_envs in sorted_envs:
        result = all_results[num_envs]
        print(f"\n{'='*60}")
        print(f"  num_envs = {num_envs}")
        print(f"{'='*60}")
        print(format_nvtx_table(result.get("nvtx_ranges", [])))


def main():
    parser = argparse.ArgumentParser(description="Sweep num_envs and profile with nsys.")
    parser.add_argument("--task", type=str, required=True, help="Registered task name.")
    parser.add_argument(
        "--num_envs", type=int, nargs="+", required=True, help="List of num_envs values to sweep."
    )
    parser.add_argument("--num_frames", type=int, default=100, help="Env steps to profile per run.")
    parser.add_argument("--warmup_frames", type=int, default=10, help="Warmup steps before profiling.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for nsys outputs.")
    parser.add_argument("--enable_cameras", action="store_true", help="Enable camera rendering.")
    parser.add_argument(
        "--kernel_patterns",
        nargs="*",
        default=None,
        help="Regex patterns for GPU kernel aggregation.",
    )
    # wandb options
    parser.add_argument("--wandb_project", type=str, default="isaaclab-benchmarks", help="wandb project.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity (team/user).")
    parser.add_argument("--no_wandb", action="store_true", help="Skip wandb logging, print only.")
    # Extra args passed to benchmark.py (e.g. hydra presets like "presets=newton,rgb")
    parser.add_argument(
        "extra_args", nargs="*", default=[],
        help="Extra args for benchmark.py, including Hydra presets (e.g. 'presets=newton,rgb').",
    )

    args = parser.parse_args()

    # Create output directory
    if args.output_dir is None:
        args.output_dir = tempfile.mkdtemp(prefix="octibench_")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[sweep] Output directory: {args.output_dir}")

    # Run benchmarks
    all_results = {}
    for num_envs in args.num_envs:
        try:
            nsys_rep_path = run_single_benchmark(
                task=args.task,
                num_envs=num_envs,
                num_frames=args.num_frames,
                warmup_frames=args.warmup_frames,
                output_dir=args.output_dir,
                extra_args=args.extra_args,
                enable_cameras=args.enable_cameras,
            )
            result = collect_results(nsys_rep_path, args.kernel_patterns)
            all_results[num_envs] = result
        except Exception as e:
            print(f"[ERROR] Failed for num_envs={num_envs}: {e}")
            continue

    if not all_results:
        print("[ERROR] No successful runs.", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print_summary(all_results)

    # Log to wandb
    if not args.no_wandb:
        log_to_wandb(
            all_results=all_results,
            task=args.task,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            kernel_patterns=args.kernel_patterns,
        )

    # Save raw results as JSON
    json_path = os.path.join(args.output_dir, "sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(
            {str(k): v for k, v in all_results.items()},
            f,
            indent=2,
            default=str,
        )
    print(f"\n[sweep] Raw results saved to: {json_path}")


if __name__ == "__main__":
    main()
