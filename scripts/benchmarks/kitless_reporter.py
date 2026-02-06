#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
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
        header = [
            ("workflow_name", payload.get("benchmark_name")),
            ("run_id", payload.get("run_id")),
            ("task", payload.get("canonical", {}).get("task")),
            ("num_envs", payload.get("canonical", {}).get("num_envs")),
            ("commit", payload.get("canonical", {}).get("commit")),
            ("branch", payload.get("canonical", {}).get("branch")),
            ("gpu_device_name", payload.get("system", {}).get("gpu_device_name")),
        ]

        print("\n" + "-" * width)
        print("Summary Report (kitless mode)".center(width))
        print("-" * width)
        for key, value in header:
            if value is not None:
                print(f"{key}: {value}")
        print("-" * width)

        for phase, metrics in self._phases.items():
            if not metrics:
                continue
            print(f"Phase: {phase}")
            for name, entry in sorted(metrics.items()):
                value = entry.get("value")
                unit = self._format_unit(name, entry.get("unit"))
                if isinstance(value, (int, float, str)):
                    suffix = f" {unit}" if unit else ""
                    print(f"{name}: {value}{suffix}")
            print("-" * width)

        system = payload.get("system", {})
        if system:
            print("System:")
            for key, value in system.items():
                if value is not None:
                    print(f"{key}: {value}")
            print("-" * width)

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
