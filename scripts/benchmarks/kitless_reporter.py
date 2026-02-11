#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import textwrap
import contextlib
import json
import os
import subprocess
import time


@dataclass
class _BaseMeasurement:
    name: str
    value: Any
    unit: str | None = None


class SingleMeasurement(_BaseMeasurement):
    pass


class ListMeasurement(_BaseMeasurement):
    pass


class DictMeasurement(_BaseMeasurement):
    pass


class KitlessBenchmark:
    """Lightweight benchmark reporter for kitless runs."""

    schema_version = "v1"

    def __init__(
        self,
        benchmark_name: str,
        workflow_metadata: dict | None = None,
        backend_type: str | None = None,
        output_dir: str | None = None,
    ) -> None:
        self.benchmark_name = benchmark_name
        self.backend_type = backend_type
        self.workflow_metadata = workflow_metadata or {}
        self.output_dir = output_dir
        self._start_time = time.time()
        self._phases: dict[str, dict[str, dict[str, Any]]] = {}
        self._current_phase = "runtime"
        self._ensure_phase(self._current_phase)
        self._run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

    def set_phase(
        self,
        phase: str,
        start_recording_frametime: bool = True,
        start_recording_runtime: bool = True,
    ) -> None:
        _ = (start_recording_frametime, start_recording_runtime)
        self._current_phase = phase
        self._ensure_phase(phase)

    def store_custom_measurement(self, phase: str, measurement: _BaseMeasurement) -> None:
        self._ensure_phase(phase)
        entry = {
            "value": measurement.value,
            "unit": measurement.unit,
            "type": measurement.__class__.__name__,
        }
        self._phases[phase][measurement.name] = entry

    def store_metadata_item(self, name: str, data: Any) -> None:
        items = self.workflow_metadata.setdefault("metadata", [])
        if not isinstance(items, list):
            return
        for item in items:
            if isinstance(item, dict) and item.get("name") == name:
                item["data"] = data
                return
        items.append({"name": name, "data": data})

    def store_measurements(self) -> None:
        # no-op for kitless; summary collected on stop()
        return None

    def stop(self) -> None:
        runtime_s = time.time() - self._start_time
        metadata = self._metadata_from_workflow()
        system_info = self._collect_system_metrics()
        canonical = self._build_canonical(metadata, system_info, runtime_s)
        payload = {
            "schema_version": self.schema_version,
            "run_id": self._run_id,
            "benchmark_name": self.benchmark_name,
            "backend_type": self.backend_type,
            "metadata": metadata,
            "system": system_info,
            "canonical": canonical,
            "phases": self._phases,
        }
        self._print_summary(payload)
        self._write_json(payload)

    def _ensure_phase(self, phase: str) -> None:
        if phase not in self._phases:
            self._phases[phase] = {}

    def _metadata_from_workflow(self) -> dict[str, Any]:
        metadata = {}
        items = self.workflow_metadata.get("metadata", [])
        if isinstance(items, Iterable):
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if name:
                    metadata[name] = item.get("data")
        return metadata

    def _collect_system_metrics(self) -> dict[str, Any]:
        system_info: dict[str, Any] = {"num_cpus": os.cpu_count()}

        try:
            import psutil

            proc = psutil.Process(os.getpid())
            system_info["cpu_usage_percent"] = psutil.cpu_percent(interval=None)
            rss = proc.memory_info().rss
            system_info["process_memory_rss_mb"] = round(rss / (1024 * 1024), 2)
            with contextlib.suppress(Exception):
                uss = proc.memory_full_info().uss
                system_info["process_memory_uss_mb"] = round(uss / (1024 * 1024), 2)
        except Exception:
            system_info.setdefault("cpu_usage_percent", None)
            system_info.setdefault("process_memory_rss_mb", None)

        try:
            import torch

            if torch.cuda.is_available():
                system_info["gpu_device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
                system_info["gpu_memory_allocated_mb"] = round(
                    torch.cuda.memory_allocated() / (1024 * 1024), 2
                )
        except Exception:
            system_info.setdefault("gpu_device_name", None)
            system_info.setdefault("gpu_memory_allocated_mb", None)

        return system_info

    def _build_canonical(
        self, metadata: dict[str, Any], system_info: dict[str, Any], runtime_s: float
    ) -> dict[str, Any]:
        commit, branch = self._get_git_info()
        throughput_value = self._pick_throughput_value()
        avg_time_ms = self._pick_avg_time_ms(throughput_value)

        return {
            "schema_version": self.schema_version,
            "task": metadata.get("task"),
            "benchmark_type": self.benchmark_name,
            "run_id": self._run_id,
            "commit": commit,
            "branch": branch,
            "num_envs": metadata.get("num_envs"),
            "total_runtime_s": round(runtime_s, 3),
            "steps_processed": metadata.get("num_frames"),
            "throughput": throughput_value,
            "avg_iteration_time_ms": avg_time_ms,
            "cpu_usage_percent": system_info.get("cpu_usage_percent"),
            "process_memory_rss_mb": system_info.get("process_memory_rss_mb"),
            "process_memory_uss_mb": system_info.get("process_memory_uss_mb"),
            "gpu_memory_allocated_mb": system_info.get("gpu_memory_allocated_mb"),
        }

    def _pick_throughput_value(self) -> float | None:
        preferred = [
            "Mean Environment step effective FPS",
            "Mean Environment only FPS",
            "Mean Collection FPS",
            "Mean Total FPS",
            "Mean Environment step FPS",
        ]
        for phase_data in self._phases.values():
            for key in preferred:
                entry = phase_data.get(key)
                if entry and isinstance(entry.get("value"), (int, float)):
                    return entry["value"]
        # fallback: first scalar metric containing 'FPS'
        for phase_data in self._phases.values():
            for name, entry in phase_data.items():
                if "FPS" in name and isinstance(entry.get("value"), (int, float)):
                    return entry["value"]
        return None

    def _pick_avg_time_ms(self, throughput_value: float | None) -> float | None:
        preferred = [
            "Mean Environment only step time",
            "Mean Environment step times",
            "Mean Env Step Time",
            "Mean Simulate Time",
        ]
        for phase_data in self._phases.values():
            for key in preferred:
                entry = phase_data.get(key)
                if entry and isinstance(entry.get("value"), (int, float)):
                    return entry["value"]
        if throughput_value and throughput_value > 0:
            return round(1000.0 / throughput_value, 3)
        return None

    def _get_git_info(self) -> tuple[str | None, str | None]:
        repo_root = Path(__file__).resolve().parents[2]
        commit = self._run_git(repo_root, ["rev-parse", "HEAD"])
        branch = self._run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
        return commit, branch

    def _run_git(self, cwd: Path, args: list[str]) -> str | None:
        try:
            result = subprocess.check_output(
                ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL
            ).decode().strip()
            return result
        except Exception:
            return None

    def _print_summary(self, payload: dict[str, Any]) -> None:
        width = 78
        metadata = payload.get("metadata", {})
        header = [
            ("workflow_name", payload.get("benchmark_name")),
            ("run_id", payload.get("run_id")),
            ("task", payload.get("canonical", {}).get("task")),
            ("seed", metadata.get("seed")),
            ("num_envs", payload.get("canonical", {}).get("num_envs")),
            ("max_iterations", metadata.get("max_iterations")),
            ("commit", payload.get("canonical", {}).get("commit")),
            ("branch", payload.get("canonical", {}).get("branch")),
            ("gpu_device_name", payload.get("system", {}).get("gpu_device_name")),
        ]

        print()
        self._print_box_separator(width)
        self._print_box_line("Summary Report (kitless mode)".center(width - 4), width)
        self._print_box_separator(width)
        for key, value in header:
            if value is not None:
                self._print_box_line(f"{key}: {value}", width)
        self._print_box_separator(width)

        notes: list[str] = []
        runtime_metrics = self._phases.get("runtime") or self._phases.get("sim_runtime") or {}
        if runtime_metrics:
            mean_fps = self._get_scalar_metric(
                runtime_metrics,
                [
                    "Mean Total FPS",
                    "Mean Collection FPS",
                    "Mean Environment step effective FPS",
                    "Mean Environment step FPS",
                ],
            )
            rendering_fps = self._get_scalar_metric(runtime_metrics, ["Mean Rendering FPS"])
            frametime_available = any(
                key in runtime_metrics for key in ("App_Update", "Physics", "Render")
            )
            mean_fps_label = self._format_value(mean_fps, unit="FPS")
            rendering_fps_label = self._format_value(rendering_fps, unit="FPS", fallback="n/a")
            self._print_box_line("Phase: sim_runtime", width)
            self._print_box_line(f"Mean FPS: {mean_fps_label}", width)
            self._print_box_line(f"Rendering FPS: {rendering_fps_label}", width)
            frametime_label = "Frametimes (ms): n/a" if not frametime_available else "Frametimes (ms): available"
            if not frametime_available or rendering_fps is None:
                frametime_label = f"{frametime_label} (see Notes)"
            self._print_box_line(frametime_label, width)
            self._print_box_separator(width)
            if not frametime_available or rendering_fps is None:
                notes.append(
                    "Kitless runs do not provide App_Update/Physics/Render breakdowns or Rendering FPS yet."
                )

        startup_metrics = self._phases.get("startup") or {}
        if startup_metrics:
            app_launch = self._get_scalar_metric(startup_metrics, ["App Launch Time"])
            python_imports = self._get_scalar_metric(startup_metrics, ["Python Imports Time"])
            total_start = self._get_scalar_metric(startup_metrics, ["Total Start Time (Launch to Train)"])
            app_launch_label = self._format_value(app_launch, unit="ms")
            python_imports_label = self._format_value(python_imports, unit="ms")
            total_start_label = self._format_value(total_start, unit="ms")
            self._print_box_line("Phase: startup", width)
            self._print_box_line(f"App Launch Time: {app_launch_label}", width)
            self._print_box_line(f"Python Imports Time: {python_imports_label}", width)
            self._print_box_line(f"Total Start Time (Launch to Train): {total_start_label}", width)
            self._print_box_separator(width)

        for phase, metrics in self._phases.items():
            if not metrics:
                continue
            if phase in {"sim_runtime", "startup"}:
                continue
            if phase == "train":
                max_rewards = self._get_scalar_metric(metrics, ["Max Rewards"])
                max_episode_len = self._get_scalar_metric(metrics, ["Max Episode Lengths"])
                self._print_box_line("Phase: train", width)
                if max_rewards is not None:
                    self._print_box_line(f"Max Rewards: {self._format_scalar(max_rewards)} float", width)
                if max_episode_len is not None:
                    self._print_box_line(
                        f"Max Episode Lengths: {self._format_scalar(max_episode_len)} float", width
                    )
                self._print_box_separator(width)
                continue
            self._print_box_line(f"Phase: {phase}", width)
            if phase == "runtime":
                runtime_rows = self._summarize_runtime_metrics(metrics)
                for row in runtime_rows:
                    self._print_box_line(row, width)
                self._print_box_separator(width)
                notes.append(
                    "Runtime section summarizes min/mean/max from training logs; kit output may show a shorter subset."
                )
                continue
            metric_items = list(metrics.items())
            metric_items.sort(key=lambda item: item[0])
            for name, entry in metric_items:
                value = entry.get("value")
                unit = self._format_unit(name, entry.get("unit"))
                if isinstance(value, (int, float, str)):
                    if isinstance(value, (int, float)):
                        value = self._format_scalar(value)
                    suffix = f" {unit}" if unit else ""
                    self._print_box_line(f"{name}: {value}{suffix}", width)
            self._print_box_separator(width)

        system = payload.get("system", {})
        if system:
            self._print_box_line("System:", width)
            for key, value in system.items():
                if value is not None:
                    if isinstance(value, (int, float)):
                        value = self._format_scalar(value)
                    self._print_box_line(f"{key}: {value}", width)
            self._print_box_separator(width)

        if notes:
            self._print_box_line("Notes:", width)
            for note in notes:
                self._print_box_wrapped_list_item(note, width)
            self._print_box_separator(width)

    def _format_unit(self, name: str, unit: str | None) -> str | None:
        if unit is None:
            if "FPS" in name:
                return "FPS"
            if "Time" in name:
                return "ms"
            return None
        if unit == "ms" and "FPS" in name:
            return "FPS"
        return unit

    def _get_scalar_metric(self, metrics: dict[str, dict[str, Any]], keys: list[str]) -> float | None:
        for key in keys:
            entry = metrics.get(key)
            if entry and isinstance(entry.get("value"), (int, float)):
                return entry["value"]
        return None

    def _format_value(self, value: float | None, unit: str, fallback: str | None = None) -> str:
        if value is None:
            return fallback or "n/a"
        return f"{self._format_scalar(value)} {unit}"

    def _format_scalar(self, value: float | int) -> str:
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def _runtime_metric_sort_key(self, item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
        name = item[0]
        if name.startswith("Min "):
            return (0, name)
        if name.startswith("Max "):
            return (1, name)
        if name.startswith("Mean "):
            return (2, name)
        return (3, name)

    def _summarize_runtime_metrics(self, metrics: dict[str, dict[str, Any]]) -> list[str]:
        series: dict[str, dict[str, float]] = {}
        units: dict[str, str | None] = {}
        for name, entry in metrics.items():
            if not isinstance(entry, dict):
                continue
            value = entry.get("value")
            if not isinstance(value, (int, float)):
                continue
            unit = self._format_unit(name, entry.get("unit"))
            if name.startswith("Min "):
                base = name[len("Min ") :]
                series.setdefault(base, {})["min"] = float(value)
                units.setdefault(base, unit)
            elif name.startswith("Max "):
                base = name[len("Max ") :]
                series.setdefault(base, {})["max"] = float(value)
                units.setdefault(base, unit)
            elif name.startswith("Mean "):
                base = name[len("Mean ") :]
                series.setdefault(base, {})["mean"] = float(value)
                units.setdefault(base, unit)

        category_order = ["Collection", "Learning", "Step Times", "Throughput", "Other"]
        categorized: dict[str, list[str]] = {key: [] for key in category_order}
        for base, stats in series.items():
            label = base
            unit = units.get(base)
            unit_suffix = f" {unit}" if unit else ""
            min_val = self._format_scalar(stats.get("min", 0.0))
            mean_val = self._format_scalar(stats.get("mean", 0.0))
            max_val = self._format_scalar(stats.get("max", 0.0))
            row = f"{label} (min/mean/max): {min_val} / {mean_val} / {max_val}{unit_suffix}"

            if "Collection" in base:
                categorized["Collection"].append(row)
            elif "Learning" in base:
                categorized["Learning"].append(row)
            elif "step time" in base:
                categorized["Step Times"].append(row)
            elif "FPS" in base or "Throughput" in base:
                categorized["Throughput"].append(row)
            else:
                categorized["Other"].append(row)

        rows: list[str] = []
        for category in category_order:
            if not categorized[category]:
                continue
            rows.append(f"{category}:")
            rows.extend(f"  {entry}" for entry in categorized[category])
        if not rows:
            rows.append("No runtime metrics available.")
        return rows

    def _print_box_separator(self, width: int) -> None:
        print("|" + "-" * (width - 2) + "|")

    def _print_box_line(self, text: str, width: int) -> None:
        inner_width = width - 4
        if not text:
            print(f"| {' ' * inner_width} |")
            return
        for line in textwrap.wrap(text, width=inner_width, break_long_words=False, break_on_hyphens=False):
            print(f"| {line.ljust(inner_width)} |")

    def _print_box_wrapped_list_item(self, text: str, width: int, bullet: str = "- ") -> None:
        inner_width = width - 4
        wrap_width = max(inner_width - len(bullet), 1)
        lines = textwrap.wrap(text, width=wrap_width, break_long_words=False, break_on_hyphens=False)
        if not lines:
            self._print_box_line(bullet.strip(), width)
            return
        self._print_box_line(f"{bullet}{lines[0]}", width)
        for line in lines[1:]:
            self._print_box_line(" " * len(bullet) + line, width)

    def _write_json(self, payload: dict[str, Any]) -> None:
        if not self.output_dir:
            return
        output_dir = Path(self.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"kitless_metrics_{self._run_id}.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


__all__ = [
    "KitlessBenchmark",
    "SingleMeasurement",
    "ListMeasurement",
    "DictMeasurement",
]
