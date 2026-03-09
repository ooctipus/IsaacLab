# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analyze nsys profiling results from an octibenchmark run.

Parses ``.nsys-rep`` files (exported to sqlite) and extracts per-NVTX-range
wall time and GPU kernel time. Supports regex-based kernel aggregation for
comparing low-level patterns (e.g. indexed vs masked writes).

Usage::

    # Basic NVTX breakdown
    python scripts/octibenchmark/analyze.py /tmp/bench_shadow.nsys-rep

    # Aggregate GPU kernels matching patterns
    python scripts/octibenchmark/analyze.py /tmp/bench_shadow.nsys-rep \\
        --kernel_patterns "scatter|index_put" "mask|where"

    # Output as JSON (for programmatic consumption by sweep.py)
    python scripts/octibenchmark/analyze.py /tmp/bench_shadow.nsys-rep --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path


def export_to_sqlite(nsys_rep_path: str) -> str:
    """Export a ``.nsys-rep`` file to sqlite, returning the sqlite path.

    Args:
        nsys_rep_path: Path to the ``.nsys-rep`` file.

    Returns:
        Path to the exported ``.sqlite`` file.
    """
    sqlite_path = nsys_rep_path.replace(".nsys-rep", ".sqlite")
    if not os.path.exists(sqlite_path):
        subprocess.run(
            ["nsys", "export", "--type=sqlite", "--force-overwrite=true", "-o", sqlite_path, nsys_rep_path],
            check=True,
            capture_output=True,
        )
    return sqlite_path


def query_nvtx_ranges(db_path: str) -> list[dict]:
    """Extract NVTX push/pop range statistics from the sqlite database.

    Groups by range name and computes total wall time, call count, and
    average wall time per call.

    Args:
        db_path: Path to the nsys sqlite database.

    Returns:
        List of dicts with keys: name, count, total_ns, avg_ns.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # NVTX events table: start, end, text (or textId -> StringIds)
    # The schema varies by nsys version. Try the common patterns.
    cursor = conn.cursor()

    # Check available tables
    tables = {row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    results = []

    if "NVTX_EVENTS" in tables:
        # Detect column names — nsys versions use different schemas:
        #   Newer: start/end with inline text
        #   Older: startTimestamp/endTimestamp with textId -> StringIds
        col_info = cursor.execute("PRAGMA table_info(NVTX_EVENTS)").fetchall()
        col_names = {row[1] for row in col_info}

        # Determine timestamp column names
        if "start" in col_names and "end" in col_names:
            ts_start, ts_end = "start", "end"
        elif "startTimestamp" in col_names and "endTimestamp" in col_names:
            ts_start, ts_end = "startTimestamp", "endTimestamp"
        else:
            conn.close()
            print("[WARNING] NVTX_EVENTS: cannot find timestamp columns.", file=sys.stderr)
            return results

        if "text" in col_names:
            query = f"""
                SELECT text AS name,
                       COUNT(*) AS count,
                       SUM({ts_end} - {ts_start}) AS total_ns,
                       AVG({ts_end} - {ts_start}) AS avg_ns
                FROM NVTX_EVENTS
                WHERE eventType = 59
                  AND {ts_end} IS NOT NULL
                  AND {ts_end} > {ts_start}
                  AND text IS NOT NULL
                GROUP BY text
                ORDER BY total_ns DESC
            """
        elif "textId" in col_names and "StringIds" in tables:
            query = f"""
                SELECT s.value AS name,
                       COUNT(*) AS count,
                       SUM(e.{ts_end} - e.{ts_start}) AS total_ns,
                       AVG(e.{ts_end} - e.{ts_start}) AS avg_ns
                FROM NVTX_EVENTS e
                JOIN StringIds s ON e.textId = s.id
                WHERE e.eventType = 59
                  AND e.{ts_end} IS NOT NULL
                  AND e.{ts_end} > e.{ts_start}
                GROUP BY s.value
                ORDER BY total_ns DESC
            """
        else:
            conn.close()
            print("[WARNING] NVTX_EVENTS table found but schema not recognized.", file=sys.stderr)
            return results

        try:
            for row in cursor.execute(query):
                results.append({
                    "name": row["name"],
                    "count": row["count"],
                    "total_ns": row["total_ns"],
                    "avg_ns": row["avg_ns"],
                })
        except sqlite3.OperationalError as e:
            print(f"[WARNING] Failed to query NVTX_EVENTS: {e}", file=sys.stderr)

    conn.close()
    return results


def query_gpu_kernels(db_path: str) -> list[dict]:
    """Extract GPU kernel execution statistics from the sqlite database.

    Args:
        db_path: Path to the nsys sqlite database.

    Returns:
        List of dicts with keys: name, count, total_ns, avg_ns.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    tables = {row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    results = []

    # Try CUPTI_ACTIVITY_KIND_KERNEL (most common)
    kernel_table = None
    for candidate in ["CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"]:
        if candidate in tables:
            kernel_table = candidate
            break

    if kernel_table is None:
        conn.close()
        return results

    # Check if shortName or demangledName exists
    col_info = cursor.execute(f"PRAGMA table_info({kernel_table})").fetchall()
    col_names = {row[1] for row in col_info}

    # Determine the name column and how duration is computed
    if "shortName" in col_names:
        name_col = "shortName"
    elif "demangledName" in col_names:
        name_col = "demangledName"
    elif "mangledName" in col_names:
        name_col = "mangledName"
    else:
        conn.close()
        return results

    # Check for StringIds join
    if "StringIds" in tables:
        # shortName is typically an ID referencing StringIds
        query = f"""
            SELECT s.value AS name,
                   COUNT(*) AS count,
                   SUM(k.end - k.start) AS total_ns,
                   AVG(k.end - k.start) AS avg_ns
            FROM {kernel_table} k
            JOIN StringIds s ON k.{name_col} = s.id
            GROUP BY s.value
            ORDER BY total_ns DESC
        """
    else:
        query = f"""
            SELECT {name_col} AS name,
                   COUNT(*) AS count,
                   SUM(end - start) AS total_ns,
                   AVG(end - start) AS avg_ns
            FROM {kernel_table}
            GROUP BY {name_col}
            ORDER BY total_ns DESC
        """

    try:
        for row in cursor.execute(query):
            results.append({
                "name": row["name"],
                "count": row["count"],
                "total_ns": row["total_ns"],
                "avg_ns": row["avg_ns"],
            })
    except sqlite3.OperationalError as e:
        print(f"[WARNING] Failed to query kernel table: {e}", file=sys.stderr)

    conn.close()
    return results


def aggregate_by_patterns(
    kernels: list[dict], patterns: list[str]
) -> dict[str, dict]:
    """Aggregate kernel times by regex patterns.

    Useful for comparing categories like "indexed writes" vs "masked writes".
    Each kernel is matched against all patterns; a kernel can match multiple
    patterns.

    Args:
        kernels: Kernel stats from :func:`query_gpu_kernels`.
        patterns: Regex patterns to match against kernel names.

    Returns:
        Dict mapping pattern to aggregated stats (total_ns, count, kernels).
    """
    result = {}
    for pattern in patterns:
        compiled = re.compile(pattern, re.IGNORECASE)
        matched_total_ns = 0
        matched_count = 0
        matched_kernels = []
        for k in kernels:
            if compiled.search(k["name"]):
                matched_total_ns += k["total_ns"]
                matched_count += k["count"]
                matched_kernels.append(k["name"])
        result[pattern] = {
            "total_ns": matched_total_ns,
            "total_ms": matched_total_ns / 1e6,
            "count": matched_count,
            "num_unique_kernels": len(matched_kernels),
            "kernels": matched_kernels,
        }
    return result


def format_nvtx_table(nvtx_ranges: list[dict]) -> str:
    """Format NVTX range data as a human-readable table.

    Args:
        nvtx_ranges: Output from :func:`query_nvtx_ranges`.

    Returns:
        Formatted table string.
    """
    if not nvtx_ranges:
        return "(no NVTX ranges found)"

    # Find env.step total for percentage computation
    step_total_ns = 0
    for r in nvtx_ranges:
        if r["name"] == "env.step":
            step_total_ns = r["total_ns"]
            break

    lines = []
    lines.append(f"{'NVTX Range':<45} {'Calls':>8} {'Total ms':>12} {'Avg ms':>12} {'% of step':>10}")
    lines.append("-" * 90)
    for r in nvtx_ranges:
        total_ms = r["total_ns"] / 1e6
        avg_ms = r["avg_ns"] / 1e6
        pct = (r["total_ns"] / step_total_ns * 100) if step_total_ns else 0
        lines.append(
            f"{r['name']:<45} {r['count']:>8} {total_ms:>12.3f} {avg_ms:>12.3f} {pct:>9.1f}%"
        )
    return "\n".join(lines)


def format_kernel_table(kernels: list[dict], top_n: int = 20) -> str:
    """Format GPU kernel data as a human-readable table.

    Args:
        kernels: Output from :func:`query_gpu_kernels`.
        top_n: Number of top kernels to show.

    Returns:
        Formatted table string.
    """
    if not kernels:
        return "(no GPU kernels found)"

    lines = []
    lines.append(f"{'GPU Kernel':<80} {'Calls':>8} {'Total ms':>12} {'Avg us':>12}")
    lines.append("-" * 115)
    for k in kernels[:top_n]:
        total_ms = k["total_ns"] / 1e6
        avg_us = k["avg_ns"] / 1e3
        name = k["name"][:78]
        lines.append(f"{name:<80} {k['count']:>8} {total_ms:>12.3f} {avg_us:>12.1f}")
    if len(kernels) > top_n:
        lines.append(f"  ... and {len(kernels) - top_n} more kernels")
    return "\n".join(lines)


def analyze(nsys_rep_path: str, kernel_patterns: list[str] | None = None) -> dict:
    """Run full analysis on an nsys-rep file.

    Args:
        nsys_rep_path: Path to the ``.nsys-rep`` file.
        kernel_patterns: Optional regex patterns for kernel aggregation.

    Returns:
        Dict with keys: nvtx_ranges, gpu_kernels, kernel_aggregations.
    """
    sqlite_path = export_to_sqlite(nsys_rep_path)
    nvtx_ranges = query_nvtx_ranges(sqlite_path)
    gpu_kernels = query_gpu_kernels(sqlite_path)

    result = {
        "nvtx_ranges": nvtx_ranges,
        "gpu_kernels": gpu_kernels,
    }

    if kernel_patterns:
        result["kernel_aggregations"] = aggregate_by_patterns(gpu_kernels, kernel_patterns)

    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze nsys profiling results.")
    parser.add_argument("nsys_rep", type=str, help="Path to .nsys-rep file.")
    parser.add_argument(
        "--kernel_patterns",
        nargs="*",
        default=None,
        help="Regex patterns to aggregate GPU kernels by (e.g. 'scatter|index_put' 'mask|where').",
    )
    parser.add_argument("--top_kernels", type=int, default=20, help="Number of top kernels to display.")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of tables.")
    args = parser.parse_args()

    if not os.path.exists(args.nsys_rep):
        print(f"Error: {args.nsys_rep} not found", file=sys.stderr)
        sys.exit(1)

    result = analyze(args.nsys_rep, args.kernel_patterns)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print("\n=== NVTX Range Breakdown ===\n")
        print(format_nvtx_table(result["nvtx_ranges"]))

        print(f"\n=== Top {args.top_kernels} GPU Kernels ===\n")
        print(format_kernel_table(result["gpu_kernels"], top_n=args.top_kernels))

        if result.get("kernel_aggregations"):
            print("\n=== Kernel Pattern Aggregations ===\n")
            for pattern, stats in result["kernel_aggregations"].items():
                print(
                    f"  Pattern: {pattern!r:<40} "
                    f"Total: {stats['total_ms']:>10.3f} ms  "
                    f"Calls: {stats['count']:>8}  "
                    f"Unique kernels: {stats['num_unique_kernels']}"
                )


if __name__ == "__main__":
    main()
