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

    # Step anatomy: map every GPU kernel in one step to its NVTX code section
    python scripts/octibenchmark/analyze.py /tmp/bench_shadow.nsys-rep --anatomy

    # Dissect the 3rd step instead of the first
    python scripts/octibenchmark/analyze.py /tmp/bench_shadow.nsys-rep --anatomy --step_index 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys


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
                results.append(
                    {
                        "name": row["name"],
                        "count": row["count"],
                        "total_ns": row["total_ns"],
                        "avg_ns": row["avg_ns"],
                    }
                )
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
            results.append(
                {
                    "name": row["name"],
                    "count": row["count"],
                    "total_ns": row["total_ns"],
                    "avg_ns": row["avg_ns"],
                }
            )
    except sqlite3.OperationalError as e:
        print(f"[WARNING] Failed to query kernel table: {e}", file=sys.stderr)

    conn.close()
    return results


def query_cuda_memory(db_path: str) -> dict:
    """Extract CUDA memory allocation statistics from the sqlite database.

    Queries ``CUPTI_ACTIVITY_KIND_MEMORY2`` (or ``CUPTI_ACTIVITY_KIND_MEMORY``)
    for allocation/free events.  Returns an empty dict when the table is
    absent (e.g. when captured with ``--profile_level light``).

    Args:
        db_path: Path to the nsys sqlite database.

    Returns:
        Dict with keys ``peak_allocated_bytes``, ``total_allocations``,
        ``total_frees``, or empty dict if no data.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    tables = {r[0] for r in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    mem_table = None
    for candidate in ["CUPTI_ACTIVITY_KIND_MEMORY2", "CUPTI_ACTIVITY_KIND_MEMORY"]:
        if candidate in tables:
            mem_table = candidate
            break

    if mem_table is None:
        conn.close()
        return {}

    col_info = cursor.execute(f"PRAGMA table_info({mem_table})").fetchall()
    col_names = {row[1] for row in col_info}

    result: dict = {}
    try:
        if "bytes" in col_names and "memoryOperationType" in col_names:
            # memoryOperationType: 1 = allocate, 2 = free (CUPTI convention)
            row = cursor.execute(
                f"SELECT MAX(bytes) AS peak, "
                f"SUM(CASE WHEN memoryOperationType = 1 THEN 1 ELSE 0 END) AS allocs, "
                f"SUM(CASE WHEN memoryOperationType = 2 THEN 1 ELSE 0 END) AS frees "
                f"FROM {mem_table}"
            ).fetchone()
            if row and row[0] is not None:
                result = {
                    "peak_allocated_bytes": row[0],
                    "total_allocations": row[1] or 0,
                    "total_frees": row[2] or 0,
                }
        elif "bytes" in col_names:
            row = cursor.execute(f"SELECT MAX(bytes) AS peak, COUNT(*) AS cnt FROM {mem_table}").fetchone()
            if row and row[0] is not None:
                result = {
                    "peak_allocated_bytes": row[0],
                    "total_allocations": row[1] or 0,
                    "total_frees": 0,
                }
    except sqlite3.OperationalError as e:
        print(f"[WARNING] Failed to query CUDA memory table: {e}", file=sys.stderr)

    conn.close()
    return result


def query_gpu_metrics(db_path: str) -> list[dict]:
    """Extract GPU hardware metric statistics from the sqlite database.

    Queries ``GPU_METRIC_EVENTS`` for counters like SM occupancy and
    DRAM throughput.  Returns an empty list when the table is absent
    (e.g. when captured with ``--profile_level light``).

    Args:
        db_path: Path to the nsys sqlite database.

    Returns:
        List of dicts with keys ``metric_name``, ``min_value``,
        ``avg_value``, ``max_value``, ``sample_count``.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    tables = {r[0] for r in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    if "GPU_METRIC_EVENTS" not in tables:
        conn.close()
        return []

    col_info = cursor.execute("PRAGMA table_info(GPU_METRIC_EVENTS)").fetchall()
    col_names = {row[1] for row in col_info}

    results: list[dict] = []

    # Determine name column — varies across nsys versions
    if "metricName" in col_names:
        name_col = "metricName"
    elif "typeId" in col_names and "StringIds" in tables:
        name_col = None  # join needed
    else:
        conn.close()
        return []

    value_col = "value" if "value" in col_names else None
    if value_col is None:
        conn.close()
        return []

    try:
        if name_col:
            query = f"""
                SELECT {name_col} AS metric_name,
                       MIN({value_col}) AS min_value,
                       AVG({value_col}) AS avg_value,
                       MAX({value_col}) AS max_value,
                       COUNT(*) AS sample_count
                FROM GPU_METRIC_EVENTS
                GROUP BY {name_col}
                ORDER BY metric_name
            """
        else:
            query = f"""
                SELECT s.value AS metric_name,
                       MIN(g.{value_col}) AS min_value,
                       AVG(g.{value_col}) AS avg_value,
                       MAX(g.{value_col}) AS max_value,
                       COUNT(*) AS sample_count
                FROM GPU_METRIC_EVENTS g
                JOIN StringIds s ON g.typeId = s.id
                GROUP BY s.value
                ORDER BY s.value
            """
        for row in cursor.execute(query):
            results.append(
                {
                    "metric_name": row["metric_name"],
                    "min_value": row["min_value"],
                    "avg_value": row["avg_value"],
                    "max_value": row["max_value"],
                    "sample_count": row["sample_count"],
                }
            )
    except sqlite3.OperationalError as e:
        print(f"[WARNING] Failed to query GPU_METRIC_EVENTS: {e}", file=sys.stderr)

    conn.close()
    return results


def aggregate_by_patterns(kernels: list[dict], patterns: list[str]) -> dict[str, dict]:
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
        lines.append(f"{r['name']:<45} {r['count']:>8} {total_ms:>12.3f} {avg_ms:>12.3f} {pct:>9.1f}%")
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


def _query_single_step_anatomy(db_path: str, cursor, step_start: int, step_end: int) -> list[dict]:
    """Query anatomy for a single step given its time bounds.

    Uses the CUDA correlation chain to attribute GPU kernels to NVTX
    ranges. Supports both the CUDA Runtime API
    (``CUPTI_ACTIVITY_KIND_RUNTIME``) and Driver API
    (``CUPTI_ACTIVITY_KIND_DRIVER``) dispatch paths. Also includes
    NVTX ranges that launched zero GPU kernels so the complete step
    structure is visible.

    Args:
        db_path: Path to the nsys sqlite database (unused, kept for consistency).
        cursor: Open sqlite cursor.
        step_start: Start timestamp of the env.step range.
        step_end: End timestamp of the env.step range.

    Returns:
        List of dicts in execution order with keys:
        nvtx_range, kernel_name, kernel_ns, wall_ns.
        For zero-kernel ranges, kernel_name is None, kernel_ns is 0,
        and wall_ns contains the NVTX range wall time.
    """
    # Detect which API dispatch tables are available
    tables = {r[0] for r in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    api_tables = []
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        api_tables.append("CUPTI_ACTIVITY_KIND_RUNTIME")
    if "CUPTI_ACTIVITY_KIND_DRIVER" in tables:
        api_tables.append("CUPTI_ACTIVITY_KIND_DRIVER")

    if not api_tables:
        return []

    # Build kernels_in_step CTE: UNION across all available API tables
    api_selects = []
    params = []
    for api_table in api_tables:
        api_selects.append(f"""
            SELECT api.start AS rt_start,
                   api.end AS rt_end,
                   api.correlationId,
                   s.value AS kernel_name,
                   (k.end - k.start) AS kernel_ns
            FROM {api_table} api
            JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON api.correlationId = k.correlationId
            JOIN StringIds s ON k.shortName = s.id
            WHERE api.start >= ? AND api.end <= ?
        """)
        params.extend([step_start, step_end])

    kernels_cte = "\n            UNION ALL\n".join(api_selects)

    query = f"""
        WITH kernels_in_step AS (
            {kernels_cte}
        ),
        ranges_in_step AS (
            SELECT start, end, text
            FROM NVTX_EVENTS
            WHERE eventType = 59 AND text IS NOT NULL
              AND start >= ? AND end <= ?
              AND text != 'env.step'
        ),
        attributed AS (
            SELECT k.rt_start,
                   k.correlationId,
                   k.kernel_name,
                   k.kernel_ns,
                   r.text AS nvtx_name,
                   0 AS wall_ns,
                   ROW_NUMBER() OVER (
                       PARTITION BY k.correlationId
                       ORDER BY (r.end - r.start) ASC
                   ) AS rn
            FROM kernels_in_step k
            LEFT JOIN ranges_in_step r ON (
                r.start <= k.rt_start AND r.end >= k.rt_end
            )
        ),
        empty_ranges AS (
            SELECT r.start AS rt_start,
                   r.text AS nvtx_name,
                   NULL AS kernel_name,
                   0 AS kernel_ns,
                   (r.end - r.start) AS wall_ns
            FROM ranges_in_step r
            WHERE NOT EXISTS (
                SELECT 1 FROM kernels_in_step k
                WHERE r.start <= k.rt_start AND r.end >= k.rt_end
            )
        )
        SELECT nvtx_name, kernel_name, kernel_ns, wall_ns
        FROM (
            SELECT rt_start, nvtx_name, kernel_name, kernel_ns, wall_ns
            FROM attributed WHERE rn = 1
            UNION ALL
            SELECT rt_start, nvtx_name, kernel_name, kernel_ns, wall_ns
            FROM empty_ranges
        )
        ORDER BY rt_start
    """

    params.extend([step_start, step_end])
    rows = cursor.execute(query, params).fetchall()

    return [
        {"nvtx_range": nvtx_name, "kernel_name": kernel_name, "kernel_ns": kernel_ns, "wall_ns": wall_ns}
        for nvtx_name, kernel_name, kernel_ns, wall_ns in rows
    ]


def _check_anatomy_tables(cursor) -> bool:
    """Check if the required tables for step anatomy exist.

    Requires NVTX_EVENTS, CUPTI_ACTIVITY_KIND_KERNEL, StringIds, and at
    least one of CUPTI_ACTIVITY_KIND_RUNTIME or CUPTI_ACTIVITY_KIND_DRIVER.
    """
    tables = {r[0] for r in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if not {"NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"}.issubset(tables):
        return False
    return "CUPTI_ACTIVITY_KIND_RUNTIME" in tables or "CUPTI_ACTIVITY_KIND_DRIVER" in tables


def query_step_anatomy(db_path: str, step_index: int = 0) -> tuple[list[dict], dict[str, int]]:
    """Dissect a single env.step and map every GPU kernel to its NVTX range.

    Uses the CUDA correlation chain (NVTX range -> RUNTIME/DRIVER API
    call -> KERNEL execution) to accurately attribute GPU kernels to the
    code section that launched them, even with asynchronous GPU execution.

    Args:
        db_path: Path to the nsys sqlite database.
        step_index: Which env.step occurrence to dissect (0-based).

    Returns:
        Tuple of (anatomy entries, range wall times).
        Anatomy entries are dicts in execution order with keys:
        nvtx_range, kernel_name, kernel_ns, wall_ns.
        Range wall times map range name to total NVTX wall time in ns.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if not _check_anatomy_tables(cursor):
        conn.close()
        return [], {}

    step = cursor.execute(
        """
        SELECT start, end FROM NVTX_EVENTS
        WHERE eventType = 59 AND text = 'env.step'
        ORDER BY start LIMIT 1 OFFSET ?
    """,
        (step_index,),
    ).fetchone()

    if step is None:
        conn.close()
        return [], {}

    results = _query_single_step_anatomy(db_path, cursor, step[0], step[1])
    wall_times = _query_range_wall_times(cursor, step[0], step[1])
    conn.close()
    return results, wall_times


def _query_range_wall_times(cursor, step_start: int, step_end: int) -> dict[str, int]:
    """Query total wall time per NVTX range name within a step.

    Groups all instances of the same range name (e.g. multiple
    ``sim.step`` calls from decimation) and sums their wall times.

    Args:
        cursor: Open sqlite cursor.
        step_start: Start timestamp of the env.step range.
        step_end: End timestamp of the env.step range.

    Returns:
        Dict mapping range name to total wall time in nanoseconds.
    """
    rows = cursor.execute(
        """
        SELECT text, SUM(end - start) AS total_wall_ns
        FROM NVTX_EVENTS
        WHERE eventType = 59 AND text IS NOT NULL
          AND start >= ? AND end <= ?
          AND text != 'env.step'
        GROUP BY text
    """,
        (step_start, step_end),
    ).fetchall()
    return {name: wall_ns for name, wall_ns in rows}


def _collapse_to_ranges(anatomy: list[dict], range_wall_times: dict[str, int] | None = None) -> list[dict]:
    """Collapse a per-kernel anatomy into sequential NVTX range summaries.

    Handles both kernel entries (kernel_ns > 0) and zero-kernel entries
    (wall_ns > 0, representing CPU-only NVTX ranges).

    Args:
        anatomy: Per-kernel anatomy from :func:`query_step_anatomy`.
        range_wall_times: Optional dict of range name -> total wall time
            from :func:`_query_range_wall_times`. When provided, each
            collapsed range includes the NVTX wall time so GPU vs CPU
            time can be compared.

    Returns:
        List of dicts in execution order with keys:
        nvtx_range, total_ns, kernel_count, wall_ns.
        total_ns is GPU kernel time. wall_ns is NVTX range wall time.
    """
    ranges = []
    current_range = object()
    for entry in anatomy:
        label = entry["nvtx_range"]
        is_cpu_only = entry["kernel_name"] is None
        if label != current_range:
            ranges.append({"nvtx_range": label, "total_ns": 0, "kernel_count": 0, "wall_ns": 0})
            current_range = label
        if is_cpu_only:
            ranges[-1]["wall_ns"] += entry.get("wall_ns", 0)
        else:
            ranges[-1]["total_ns"] += entry["kernel_ns"]
            ranges[-1]["kernel_count"] += 1

    # Merge NVTX wall times if provided — for ranges with kernels, this gives
    # the CPU wall time so we can show GPU/wall ratio
    if range_wall_times:
        # Distribute wall time across collapsed entries sharing the same name
        name_counts: dict[str, int] = {}
        for r in ranges:
            name = r["nvtx_range"]
            if name:
                name_counts[name] = name_counts.get(name, 0) + 1
        for r in ranges:
            name = r["nvtx_range"]
            if name and name in range_wall_times and r["kernel_count"] > 0:
                # Distribute total wall time evenly across occurrences of this range
                r["wall_ns"] = range_wall_times[name] // name_counts.get(name, 1)

    return ranges


def query_step_anatomy_averaged(db_path: str) -> tuple[list[dict], int]:
    """Average step anatomy across all captured env.step occurrences.

    Collapses each step into its sequential NVTX range structure, then
    averages times and kernel counts across steps that share the same
    structure (same sequence of NVTX range names).

    Args:
        db_path: Path to the nsys sqlite database.

    Returns:
        Tuple of (averaged range summaries, number of steps averaged).
        Each summary dict has keys: nvtx_range, avg_ns, avg_kernel_count.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if not _check_anatomy_tables(cursor):
        conn.close()
        return [], 0

    # Get all env.step ranges
    steps = cursor.execute("""
        SELECT start, end FROM NVTX_EVENTS
        WHERE eventType = 59 AND text = 'env.step'
        ORDER BY start
    """).fetchall()

    if not steps:
        conn.close()
        return [], 0

    # Collect collapsed ranges for each step
    all_collapsed = []
    for step_start, step_end in steps:
        anatomy = _query_single_step_anatomy(db_path, cursor, step_start, step_end)
        if anatomy:
            wall_times = _query_range_wall_times(cursor, step_start, step_end)
            all_collapsed.append(_collapse_to_ranges(anatomy, wall_times))

    conn.close()

    if not all_collapsed:
        return [], 0

    # Find the most common range structure (sequence of NVTX names)
    # and average only steps that match it
    from collections import Counter

    structures = Counter()
    for collapsed in all_collapsed:
        key = tuple(r["nvtx_range"] for r in collapsed)
        structures[key] += 1

    dominant_structure = structures.most_common(1)[0][0]
    matching = [c for c in all_collapsed if tuple(r["nvtx_range"] for r in c) == dominant_structure]

    num_steps = len(matching)
    num_ranges = len(dominant_structure)

    averaged = []
    for i in range(num_ranges):
        total_ns_sum = sum(m[i]["total_ns"] for m in matching)
        kernel_count_sum = sum(m[i]["kernel_count"] for m in matching)
        wall_ns_sum = sum(m[i].get("wall_ns", 0) for m in matching)
        averaged.append(
            {
                "nvtx_range": dominant_structure[i],
                "avg_ns": total_ns_sum / num_steps,
                "avg_kernel_count": kernel_count_sum / num_steps,
                "avg_wall_ns": wall_ns_sum / num_steps,
            }
        )

    return averaged, num_steps


def format_step_anatomy(
    anatomy: list[dict],
    step_index: int = 0,
    verbose: bool = True,
    range_wall_times: dict[str, int] | None = None,
) -> str:
    """Format step anatomy as a human-readable tree showing execution order.

    Args:
        anatomy: Output from :func:`query_step_anatomy`.
        step_index: Step index (for display).
        verbose: If True, show individual kernels. If False, show only
            NVTX range summaries.
        range_wall_times: Optional dict of range name -> wall time in ns.
            When provided, shows wall time alongside GPU time so
            GPU/CPU ratio is visible.

    Returns:
        Formatted string showing each NVTX range and its kernels.
    """
    if not anatomy:
        return "(no step anatomy data — requires NVTX_EVENTS, RUNTIME, and KERNEL tables)"

    total_ns = sum(k["kernel_ns"] for k in anatomy)

    if not verbose:
        # Collapsed mode: show only NVTX range summaries
        collapsed = _collapse_to_ranges(anatomy, range_wall_times)
        n_kernels = len([e for e in anatomy if e["kernel_name"]])
        lines = [f"Step #{step_index} anatomy — {n_kernels} kernel launches, {total_ns / 1e6:.3f} ms total GPU time"]
        lines.append("")
        lines.append(f"  {'NVTX Range':<45} {'Wall ms':>10} {'GPU ms':>10} {'Kernels':>10} {'%GPU':>8}")
        lines.append(f"  {'─' * 45} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 8}")
        for r in collapsed:
            display = r["nvtx_range"] if r["nvtx_range"] else "UNATTRIBUTED"
            wall_ms = r["wall_ns"] / 1e6
            if r["kernel_count"] == 0:
                lines.append(f"  {display:<45} {wall_ms:>10.3f} {'—':>10} {'(cpu)':>10} {'':>8}")
            else:
                gpu_ms = r["total_ns"] / 1e6
                pct = (r["total_ns"] / total_ns * 100) if total_ns else 0
                if wall_ms > 0:
                    lines.append(
                        f"  {display:<45} {wall_ms:>10.3f} {gpu_ms:>10.3f} {r['kernel_count']:>10} {pct:>7.1f}%"
                    )
                else:
                    lines.append(f"  {display:<45} {'—':>10} {gpu_ms:>10.3f} {r['kernel_count']:>10} {pct:>7.1f}%")
        return "\n".join(lines)

    # Verbose mode: show individual kernels
    kernel_entries = [e for e in anatomy if e["kernel_name"]]
    lines = [
        f"Step #{step_index} anatomy — {len(kernel_entries)} kernel launches, {total_ns / 1e6:.3f} ms total GPU time"
    ]
    lines.append("")

    current_range = object()  # sentinel
    range_total_ns = 0
    range_count = 0

    def _flush_range():
        nonlocal range_total_ns, range_count
        if range_count > 0 and current_range is not object():
            wall_info = ""
            if range_wall_times and current_range in range_wall_times:
                wall_ms = range_wall_times[current_range] / 1e6
                wall_info = f", wall {wall_ms:.3f} ms"
            lines.append(f"  {'':60} ────────── {range_total_ns / 1e6:>8.3f} ms ({range_count} kernels{wall_info})")
        range_total_ns = 0
        range_count = 0

    for entry in anatomy:
        label = entry["nvtx_range"]
        if label != current_range:
            _flush_range()
            current_range = label
            display = label if label else "UNATTRIBUTED"
            lines.append(f"  {display}")

        if entry["kernel_name"] is None:
            # CPU-only range — show wall time
            wall_ms = entry.get("wall_ns", 0) / 1e6
            lines.append(f"      {'(no GPU kernels — CPU only)':<58} {wall_ms:>8.3f} ms wall")
            continue

        kernel_ms = entry["kernel_ns"] / 1e6
        name = entry["kernel_name"]
        if len(name) > 58:
            name = name[:55] + "..."
        lines.append(f"      {name:<58} {kernel_ms:>8.3f} ms")
        range_total_ns += entry["kernel_ns"]
        range_count += 1

    _flush_range()
    return "\n".join(lines)


def format_step_anatomy_averaged(averaged: list[dict], num_steps: int) -> str:
    """Format averaged step anatomy as a compact summary.

    Args:
        averaged: Output from :func:`query_step_anatomy_averaged`.
        num_steps: Number of steps that were averaged.

    Returns:
        Formatted string showing averaged NVTX range breakdown with
        both wall time and GPU time for each range.
    """
    if not averaged:
        return "(no step anatomy data — requires NVTX_EVENTS, RUNTIME, and KERNEL tables)"

    total_ns = sum(r["avg_ns"] for r in averaged)
    total_kernels = sum(r["avg_kernel_count"] for r in averaged)
    lines = [
        f"Step anatomy (averaged over {num_steps} steps) — "
        f"{total_kernels:.0f} kernel launches, {total_ns / 1e6:.3f} ms avg GPU time"
    ]
    lines.append("")

    # Header
    lines.append(f"  {'NVTX Range':<45} {'Wall ms':>10} {'GPU ms':>10} {'Kernels':>10} {'%GPU':>8}")
    lines.append(f"  {'─' * 45} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 8}")

    for r in averaged:
        display = r["nvtx_range"] if r["nvtx_range"] else "UNATTRIBUTED"
        wall_ms = r.get("avg_wall_ns", 0) / 1e6
        if r["avg_kernel_count"] == 0:
            lines.append(f"  {display:<45} {wall_ms:>10.3f} {'—':>10} {'(cpu)':>10} {'':>8}")
        else:
            avg_ms = r["avg_ns"] / 1e6
            pct = (r["avg_ns"] / total_ns * 100) if total_ns else 0
            if wall_ms > 0:
                lines.append(
                    f"  {display:<45} {wall_ms:>10.3f} {avg_ms:>10.3f} {r['avg_kernel_count']:>10.0f} {pct:>7.1f}%"
                )
            else:
                lines.append(f"  {display:<45} {'—':>10} {avg_ms:>10.3f} {r['avg_kernel_count']:>10.0f} {pct:>7.1f}%")

    return "\n".join(lines)


def analyze(nsys_rep_path: str, kernel_patterns: list[str] | None = None) -> dict:
    """Run full analysis on an nsys-rep file.

    Automatically extracts CUDA memory and GPU metric data when the
    corresponding tables are present (i.e. captured with
    ``--profile_level full``).

    Args:
        nsys_rep_path: Path to the ``.nsys-rep`` file.
        kernel_patterns: Optional regex patterns for kernel aggregation.

    Returns:
        Dict with keys: nvtx_ranges, gpu_kernels, and optionally
        kernel_aggregations, cuda_memory, gpu_metrics, gpu_timing,
        anatomy_steps.
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

    cuda_mem = query_cuda_memory(sqlite_path)
    if cuda_mem:
        result["cuda_memory"] = cuda_mem

    gpu_met = query_gpu_metrics(sqlite_path)
    if gpu_met:
        result["gpu_metrics"] = gpu_met

    # GPU-attributed timing via step anatomy correlation.
    # Aggregates the sequential per-range anatomy (where ranges like
    # sim.step may appear multiple times due to decimation) into
    # per-name totals: {range_name: {gpu_ns, wall_ns, kernel_count}}.
    anatomy_avg, num_steps = query_step_anatomy_averaged(sqlite_path)
    if anatomy_avg and num_steps > 0:
        gpu_timing: dict[str, dict[str, float]] = {}
        for entry in anatomy_avg:
            name = entry["nvtx_range"]
            if name not in gpu_timing:
                gpu_timing[name] = {"gpu_ns": 0.0, "wall_ns": 0.0, "kernel_count": 0.0}
            gpu_timing[name]["gpu_ns"] += entry["avg_ns"]
            gpu_timing[name]["wall_ns"] += entry.get("avg_wall_ns", 0.0)
            gpu_timing[name]["kernel_count"] += entry["avg_kernel_count"]
        result["gpu_timing"] = gpu_timing
        result["anatomy_steps"] = num_steps

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
    parser.add_argument(
        "--anatomy",
        action="store_true",
        help="Show step anatomy: map every GPU kernel to its NVTX code section. Averages across all steps by default.",
    )
    parser.add_argument(
        "--step_index",
        type=int,
        default=None,
        help="Dissect a specific env.step (0-based). Without this flag, --anatomy averages across all steps.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show individual kernels in anatomy (default: collapsed summary).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.nsys_rep):
        print(f"Error: {args.nsys_rep} not found", file=sys.stderr)
        sys.exit(1)

    result = analyze(args.nsys_rep, args.kernel_patterns)

    if args.anatomy:
        sqlite_path = export_to_sqlite(args.nsys_rep)
        if args.step_index is not None:
            # Single step dissection
            anatomy, wall_times = query_step_anatomy(sqlite_path, step_index=args.step_index)
            if args.json:
                print(json.dumps(anatomy, indent=2, default=str))
            else:
                print(
                    format_step_anatomy(
                        anatomy, step_index=args.step_index, verbose=args.verbose, range_wall_times=wall_times
                    )
                )
        else:
            # Averaged across all steps
            averaged, num_steps = query_step_anatomy_averaged(sqlite_path)
            if args.json:
                print(json.dumps({"averaged": averaged, "num_steps": num_steps}, indent=2, default=str))
            else:
                print(format_step_anatomy_averaged(averaged, num_steps))
                if args.verbose:
                    # Also show full kernel listing for a representative step
                    mid = num_steps // 2
                    anatomy, wall_times = query_step_anatomy(sqlite_path, step_index=mid)
                    print()
                    print(format_step_anatomy(anatomy, step_index=mid, verbose=True, range_wall_times=wall_times))
        return

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    elif result.get("kernel_aggregations"):
        # Show only pattern results when --kernel_patterns is used
        total_gpu_ns = sum(k["total_ns"] for k in result["gpu_kernels"])
        total_gpu_ms = total_gpu_ns / 1e6
        print(f"\n=== Kernel Pattern Aggregations (total GPU time: {total_gpu_ms:.1f} ms) ===\n")
        print(f"  {'Pattern':<40} {'Total ms':>10} {'%':>8} {'Calls':>8} {'Kernels':>10}")
        print(f"  {'─' * 40} {'─' * 10} {'─' * 8} {'─' * 8} {'─' * 10}")
        for pattern, stats in result["kernel_aggregations"].items():
            pct = (stats["total_ns"] / total_gpu_ns * 100) if total_gpu_ns else 0
            print(
                f"  {pattern:<40} {stats['total_ms']:>10.3f} {pct:>7.1f}% "
                f"{stats['count']:>8} {stats['num_unique_kernels']:>10}"
            )
    else:
        print("\n=== NVTX Range Breakdown ===\n")
        print(format_nvtx_table(result["nvtx_ranges"]))

        print(f"\n=== Top {args.top_kernels} GPU Kernels ===\n")
        print(format_kernel_table(result["gpu_kernels"], top_n=args.top_kernels))


if __name__ == "__main__":
    main()
