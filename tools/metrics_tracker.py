#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import sys
import time
from typing import Dict, Iterable, Optional


TARGET_METRICS = (
    "Metrics/sim_time_ms",
    "Metrics/render_time_ms",
    "Metrics/sim_time_ms_total",
    "Metrics/render_time_ms_total",
)

ITERATION_RE = re.compile(r"Learning iteration\s+(\d+)")
METRIC_RE = re.compile(r"^\s*(Metrics/[^:]+):\s*([0-9.+-eE]+)\s*$")


def _iter_lines(text: str) -> Iterable[str]:
    for line in text.splitlines():
        yield line


def _parse_updates(
    lines: Iterable[str], metrics: Dict[str, float], current_iteration: Optional[int]
) -> Optional[int]:
    for line in lines:
        iteration_match = ITERATION_RE.search(line)
        if iteration_match:
            current_iteration = int(iteration_match.group(1))
        metric_match = METRIC_RE.match(line)
        if metric_match:
            key = metric_match.group(1).strip()
            if key in TARGET_METRICS:
                metrics[key] = float(metric_match.group(2))
    return current_iteration


def _emit_if_complete(
    metrics: Dict[str, float],
    current_iteration: Optional[int],
    last_emitted: Dict[str, float],
    writer: Optional[csv.DictWriter],
    emit_stdout: bool,
) -> None:
    if not all(key in metrics for key in TARGET_METRICS):
        return
    if all(metrics.get(key) == last_emitted.get(key) for key in TARGET_METRICS):
        return
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    row = {
        "timestamp": timestamp,
        "iteration": "" if current_iteration is None else str(current_iteration),
    }
    for key in TARGET_METRICS:
        row[key] = f"{metrics[key]:.6f}"
    if writer is not None:
        writer.writerow(row)
    if emit_stdout:
        print(
            f"[{row['timestamp']}] iter={row['iteration'] or '?'} "
            f"sim_ms={row['Metrics/sim_time_ms']} "
            f"render_ms={row['Metrics/render_time_ms']} "
            f"sim_total={row['Metrics/sim_time_ms_total']} "
            f"render_total={row['Metrics/render_time_ms_total']}"
        )
        sys.stdout.flush()
    last_emitted.clear()
    last_emitted.update({key: metrics[key] for key in TARGET_METRICS})


def _open_writer(path: Optional[str]) -> Optional[csv.DictWriter]:
    if not path:
        return None
    path = os.path.expanduser(path)
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        raise FileNotFoundError(f"Output directory does not exist: {directory}")
    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    csv_file = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=["timestamp", "iteration", *TARGET_METRICS],
    )
    if needs_header:
        writer.writeheader()
    return writer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Track selected training metrics from a terminal log."
    )
    parser.add_argument(
        "--log-file",
        default="~/.cursor/projects/home-horde-IsaacLab/terminals/13.txt",
        help="Path to the terminal log file to monitor.",
    )
    parser.add_argument(
        "--out-file",
        default="~/.cursor/projects/home-horde-IsaacLab/terminals/metrics_tracker.csv",
        help="CSV output path (disabled if empty).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--tail-bytes",
        type=int,
        default=20000,
        help="Number of bytes to read from the end before following.",
    )
    parser.add_argument(
        "--no-stdout",
        action="store_true",
        help="Disable stdout updates.",
    )
    args = parser.parse_args()

    log_path = os.path.expanduser(args.log_file)
    out_file = args.out_file.strip() if args.out_file else ""
    writer = _open_writer(out_file if out_file else None)
    emit_stdout = not args.no_stdout

    metrics: Dict[str, float] = {}
    last_emitted: Dict[str, float] = {}
    current_iteration: Optional[int] = None
    offset = 0
    warned_missing = False

    while True:
        if not os.path.exists(log_path):
            if not warned_missing:
                print(f"[metrics_tracker] Waiting for log file: {log_path}")
                warned_missing = True
            time.sleep(args.interval)
            continue

        warned_missing = False
        file_size = os.path.getsize(log_path)
        if offset == 0 and file_size > 0:
            offset = max(0, file_size - args.tail_bytes)

        if file_size < offset:
            offset = 0

        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            chunk = handle.read()
            offset = handle.tell()

        if chunk:
            current_iteration = _parse_updates(_iter_lines(chunk), metrics, current_iteration)
            _emit_if_complete(metrics, current_iteration, last_emitted, writer, emit_stdout)

        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
