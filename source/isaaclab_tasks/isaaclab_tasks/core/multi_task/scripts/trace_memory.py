# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trace GPU/CPU memory usage across env startup, env steps and PPO rollout.

This script mirrors :mod:`trace_env` but records *memory deltas* rather than
durations. For each instrumented span it captures:

* ``torch.cuda.memory_allocated``  -- live torch tensors (excludes cached blocks).
* ``torch.cuda.memory_reserved``   -- bytes held by the torch caching allocator.
* ``torch.cuda.max_memory_allocated`` -- in-span peak (reset on span enter).
* ``cpu_rss``                      -- process RSS via :mod:`psutil`.
* ``gpu_used``                     -- whole-device usage from NVML, when available.
  This catches non-torch allocations such as PhysX, Warp, RTX caches and Kit.

The goal is to identify which subsystems own the most memory so that the
``num_envs=4096`` × CNN + height-scanner configuration can be slimmed down
(e.g. by reducing rollout buffer length, cropping observation tensors,
shrinking ray-cast pattern, or by sharing height-field meshes).

Example::

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/multi_task/scripts/trace_memory.py \
        --task=Isaac-Position-v0 --num_envs=4096 --trace_steps=8 --rsl_rl_iters=1 \
        --output=/tmp/trace_memory.json --tensor_top=40 \
        presets=anymal_c,res02,cnn

Pass ``--memory_history`` to also dump a CUDA allocator snapshot (``.pickle``)
that can be loaded into https://pytorch.org/memory_viz for a flame-graph view
of every allocation site.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
import gc
import importlib
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from isaaclab.utils.string import list_intersection, string_to_callable

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


parser = argparse.ArgumentParser(description="Trace memory usage across env startup, env steps and PPO rollout.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--trace_steps", type=int, default=8, help="Number of env.step calls to trace.")
parser.add_argument(
    "--rsl_rl_iters",
    type=int,
    default=1,
    help=(
        "Number of full PPO iterations to trace. ``0`` skips runner construction entirely. ``>=1`` builds the"
        " RSL-RL runner and runs that many learning iterations so the rollout buffer + update path is visible."
    ),
)
parser.add_argument(
    "--action_mode",
    choices=("random", "zero"),
    default="random",
    help="Action source for traced environment steps when running without the RSL-RL runner.",
)
parser.add_argument("--output", type=str, default=None, help="Optional JSON output path for the full memory trace.")
parser.add_argument("--top_events", type=int, default=30, help="Number of aggregate spans to print.")
parser.add_argument("--tensor_top", type=int, default=30, help="Number of largest env-owned tensors to print.")
parser.add_argument(
    "--memory_history",
    type=str,
    default=None,
    help=(
        "If set, enable ``torch.cuda.memory._record_memory_history`` and dump the snapshot pickle to this path."
        " Open it at https://pytorch.org/memory_viz to inspect every allocation site."
    ),
)
parser.add_argument(
    "--no_synchronize",
    action="store_true",
    default=False,
    help="Do not synchronize CUDA/Warp around memory snapshots (faster, but numbers may race the GPU).",
)
parser.add_argument(
    "--no_manager_trace",
    action="store_true",
    default=False,
    help="Do not monkey-patch manager/sim methods. Useful to isolate pure env-step memory cost.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
add_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
if args_cli.task is None:
    parser.error("the following arguments are required: --task")


remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args


_imports_t0 = time.perf_counter_ns()

import gymnasium as gym  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

from isaaclab_tasks.utils import launch_simulation, resolve_task_config  # noqa: E402

_imports_t1 = time.perf_counter_ns()


# ---------------------------------------------------------------------------
# Memory snapshot + recorder
# ---------------------------------------------------------------------------


_MIB = 1024 * 1024


def _bytes_to_mib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / _MIB


@dataclass
class MemorySample:
    """Snapshot of the relevant memory counters at one point in time."""

    timestamp_ns: int
    torch_allocated: int  # bytes
    torch_reserved: int  # bytes
    torch_peak_allocated: int  # bytes (cumulative since last reset)
    cpu_rss: int  # bytes
    nvml_used: int | None  # bytes; None if NVML unavailable
    nvml_total: int | None  # bytes; None if NVML unavailable

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "torch_allocated_mib": _bytes_to_mib(self.torch_allocated),
            "torch_reserved_mib": _bytes_to_mib(self.torch_reserved),
            "torch_peak_allocated_mib": _bytes_to_mib(self.torch_peak_allocated),
            "cpu_rss_mib": _bytes_to_mib(self.cpu_rss),
            "nvml_used_mib": _bytes_to_mib(self.nvml_used),
            "nvml_total_mib": _bytes_to_mib(self.nvml_total),
        }


@dataclass
class MemoryEvent:
    """One traced span with start/end memory samples."""

    event_id: int
    name: str
    parent_id: int | None
    start: MemorySample
    end: MemorySample
    in_span_peak_allocated: int  # bytes; ``end.torch_peak_allocated`` reset at span enter
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def torch_allocated_delta(self) -> int:
        return self.end.torch_allocated - self.start.torch_allocated

    @property
    def torch_reserved_delta(self) -> int:
        return self.end.torch_reserved - self.start.torch_reserved

    @property
    def cpu_rss_delta(self) -> int:
        return self.end.cpu_rss - self.start.cpu_rss

    @property
    def nvml_used_delta(self) -> int | None:
        if self.start.nvml_used is None or self.end.nvml_used is None:
            return None
        return self.end.nvml_used - self.start.nvml_used

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.event_id,
            "name": self.name,
            "parent_id": self.parent_id,
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "deltas_mib": {
                "torch_allocated": _bytes_to_mib(self.torch_allocated_delta),
                "torch_reserved": _bytes_to_mib(self.torch_reserved_delta),
                "cpu_rss": _bytes_to_mib(self.cpu_rss_delta),
                "nvml_used": _bytes_to_mib(self.nvml_used_delta),
            },
            "in_span_peak_allocated_mib": _bytes_to_mib(self.in_span_peak_allocated),
            "metadata": self.metadata,
        }


class _NvmlProbe:
    """Light wrapper around NVML; gracefully degrades if pynvml is unavailable."""

    def __init__(self) -> None:
        self._handle = None
        self._total: int | None = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            device_index = 0
            if torch.cuda.is_available():
                device_index = torch.cuda.current_device()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._total = int(mem.total)
        except Exception:  # noqa: BLE001 -- NVML is opportunistic
            self._pynvml = None
            self._handle = None

    def sample(self) -> tuple[int | None, int | None]:
        if self._handle is None:
            return None, None
        try:
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return int(mem.used), int(mem.total)
        except Exception:  # noqa: BLE001
            return None, self._total


class MemoryRecorder:
    """Hierarchical memory trace recorder.

    The structure mirrors :class:`isaaclab_tasks.core.multi_task.utils.trace.TraceRecorder`,
    but stores memory snapshots instead of timing.
    """

    def __init__(self, metadata: dict[str, Any] | None = None, synchronize: bool = True) -> None:
        self.metadata = metadata or {}
        self.synchronize = synchronize
        self.events: list[MemoryEvent] = []
        self._stack: list[int] = []
        self._next_event_id = 0
        self._process = psutil.Process(os.getpid())
        self._nvml = _NvmlProbe()
        self.metadata["nvml_available"] = self._nvml._handle is not None

    def span(self, name: str, **metadata) -> _MemorySpan:
        return _MemorySpan(self, name, metadata)

    def sample(self) -> MemorySample:
        if self.synchronize:
            wp.synchronize()
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
        nvml_used, nvml_total = self._nvml.sample()
        if torch.cuda.is_available():
            torch_alloc = int(torch.cuda.memory_allocated())
            torch_reserved = int(torch.cuda.memory_reserved())
            torch_peak = int(torch.cuda.max_memory_allocated())
        else:
            torch_alloc = torch_reserved = torch_peak = 0
        return MemorySample(
            timestamp_ns=time.perf_counter_ns(),
            torch_allocated=torch_alloc,
            torch_reserved=torch_reserved,
            torch_peak_allocated=torch_peak,
            cpu_rss=int(self._process.memory_info().rss),
            nvml_used=nvml_used,
            nvml_total=nvml_total,
        )

    def record_outside_span(self, name: str, start: MemorySample, end: MemorySample, **metadata) -> None:
        event_id = self._next_event_id
        self._next_event_id += 1
        self.events.append(
            MemoryEvent(
                event_id=event_id,
                name=name,
                parent_id=self._stack[-1] if self._stack else None,
                start=start,
                end=end,
                in_span_peak_allocated=max(end.torch_peak_allocated - start.torch_peak_allocated, 0),
                metadata=_json_safe(metadata),
            )
        )

    # -- aggregation --------------------------------------------------------

    def aggregate(self) -> list[dict[str, Any]]:
        """Aggregate spans by name with peak / mean / total deltas."""
        per_name: dict[str, list[MemoryEvent]] = defaultdict(list)
        for event in self.events:
            per_name[event.name].append(event)
        rows: list[dict[str, Any]] = []
        for name, events in per_name.items():
            torch_deltas = [e.torch_allocated_delta for e in events]
            reserved_deltas = [e.torch_reserved_delta for e in events]
            nvml_deltas = [e.nvml_used_delta for e in events if e.nvml_used_delta is not None]
            in_peaks = [e.in_span_peak_allocated for e in events]
            rows.append(
                {
                    "name": name,
                    "count": len(events),
                    "torch_alloc_total_mib": _bytes_to_mib(sum(torch_deltas)),
                    "torch_alloc_mean_mib": _bytes_to_mib(sum(torch_deltas) / len(events)),
                    "torch_alloc_max_mib": _bytes_to_mib(max(torch_deltas)),
                    "torch_reserved_total_mib": _bytes_to_mib(sum(reserved_deltas)),
                    "in_span_peak_max_mib": _bytes_to_mib(max(in_peaks)) if in_peaks else 0.0,
                    "nvml_used_total_mib": _bytes_to_mib(sum(nvml_deltas)) if nvml_deltas else None,
                }
            )
        rows.sort(key=lambda r: abs(r.get("torch_alloc_total_mib") or 0.0), reverse=True)
        return rows

    def summary_lines(self, top_n: int = 30) -> list[str]:
        rows = self.aggregate()[:top_n]
        lines = ["Memory trace summary (sorted by |torch_alloc_total| descending):"]
        if not rows:
            lines.append("  <no events>")
            return lines
        header = (
            f"  {'name':<44} {'count':>6} {'alloc_total':>12} {'alloc_mean':>12} {'alloc_max':>12}"
            f" {'reserved_total':>15} {'in_peak_max':>12} {'nvml_total':>12}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for row in rows:
            nvml_str = (
                f"{row['nvml_used_total_mib']:>12.1f}" if row["nvml_used_total_mib"] is not None else f"{'n/a':>12}"
            )
            lines.append(
                f"  {str(row['name']):<44} {int(row['count']):>6d}"
                f" {row['torch_alloc_total_mib']:>12.1f}"
                f" {row['torch_alloc_mean_mib']:>12.1f}"
                f" {row['torch_alloc_max_mib']:>12.1f}"
                f" {row['torch_reserved_total_mib']:>15.1f}"
                f" {row['in_span_peak_max_mib']:>12.1f}"
                f" {nvml_str}"
            )
        lines.append(
            "  (all values in MiB; alloc=torch.cuda.memory_allocated, reserved=cached blocks,"
            " in_peak=max(allocated) inside span, nvml=whole-device usage incl. PhysX/Kit/Warp)"
        )
        return lines

    def export_json(self, path: str | Path) -> None:
        payload = {
            "metadata": self.metadata,
            "events": [event.to_dict() for event in self.events],
            "aggregate": self.aggregate(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- internals used by the span context manager ------------------------

    def _start_event(self, name: str) -> tuple[int, MemorySample, int | None]:
        # Reset peak so ``in_span_peak_allocated`` is well-defined for nested spans.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = self.sample()
        event_id = self._next_event_id
        self._next_event_id += 1
        parent_id = self._stack[-1] if self._stack else None
        self._stack.append(event_id)
        return event_id, start, parent_id

    def _stop_event(
        self,
        event_id: int,
        name: str,
        start: MemorySample,
        parent_id: int | None,
        metadata: dict[str, Any],
    ) -> None:
        end = self.sample()
        if not self._stack or self._stack[-1] != event_id:
            raise RuntimeError(f"Memory span stack is corrupt while closing '{name}'.")
        self._stack.pop()
        in_span_peak = max(end.torch_peak_allocated - start.torch_peak_allocated, 0)
        self.events.append(
            MemoryEvent(
                event_id=event_id,
                name=name,
                parent_id=parent_id,
                start=start,
                end=end,
                in_span_peak_allocated=in_span_peak,
                metadata=_json_safe(metadata),
            )
        )


class _MemorySpan:
    """Context manager backing :meth:`MemoryRecorder.span`."""

    def __init__(self, recorder: MemoryRecorder, name: str, metadata: dict[str, Any]) -> None:
        self._recorder = recorder
        self._name = name
        self._metadata = metadata
        self._event_id: int | None = None
        self._start: MemorySample | None = None
        self._parent_id: int | None = None

    def __enter__(self) -> _MemorySpan:
        self._event_id, self._start, self._parent_id = self._recorder._start_event(self._name)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is not None:
            self._metadata = {**self._metadata, "exception": exc_type.__name__}
        if self._event_id is None or self._start is None:
            raise RuntimeError(f"Memory span '{self._name}' was closed before it was opened.")
        self._recorder._stop_event(self._event_id, self._name, self._start, self._parent_id, self._metadata)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


# ---------------------------------------------------------------------------
# Method wrapping (mirrors trace_env.py)
# ---------------------------------------------------------------------------


def _wrap_method(obj: Any, method_name: str, span_name: str, recorder: MemoryRecorder) -> None:
    method = getattr(obj, method_name, None)
    if method is None or getattr(method, "_trace_memory_wrapped", False):
        return

    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        with recorder.span(span_name):
            return method(*args, **kwargs)

    wrapped._trace_memory_wrapped = True  # type: ignore[attr-defined]
    try:
        setattr(obj, method_name, wrapped)
    except (AttributeError, TypeError):
        return


def _wrap_manager(env: Any, manager_name: str, methods: tuple[str, ...], recorder: MemoryRecorder) -> None:
    manager = getattr(env, manager_name, None)
    if manager is None:
        return
    label = manager_name.removesuffix("_manager")
    for method_name in methods:
        _wrap_method(manager, method_name, f"manager.{label}.{method_name}", recorder)


def _wrap_module_function(module_name: str, function_name: str, span_name: str, recorder: MemoryRecorder) -> None:
    with contextlib.suppress(ImportError):
        module = importlib.import_module(module_name)
        _wrap_method(module, function_name, span_name, recorder)


def _install_startup_tracing(recorder: MemoryRecorder) -> None:
    clone_spans = (
        ("isaaclab.cloner", "usd_replicate", "cloner.usd_replicate"),
        ("isaaclab.cloner.cloner_utils", "usd_replicate", "cloner.usd_replicate"),
        ("isaaclab.cloner", "filter_collisions", "cloner.filter_collisions"),
        ("isaaclab.cloner.cloner_utils", "filter_collisions", "cloner.filter_collisions"),
        ("isaaclab.cloner", "clone_from_template", "cloner.clone_from_template"),
        ("isaaclab.cloner.cloner_utils", "clone_from_template", "cloner.clone_from_template"),
        ("isaaclab_physx.cloner", "physx_replicate", "cloner.physx_replicate"),
        ("isaaclab_physx.cloner.physx_replicate", "physx_replicate", "cloner.physx_replicate"),
        ("isaaclab_ovphysx.cloner", "ovphysx_replicate", "cloner.ovphysx_replicate"),
        ("isaaclab_ovphysx.cloner.ovphysx_replicate", "ovphysx_replicate", "cloner.ovphysx_replicate"),
        ("isaaclab_newton.cloner", "newton_physics_replicate", "cloner.newton_physics_replicate"),
        ("isaaclab_newton.cloner.newton_replicate", "newton_physics_replicate", "cloner.newton_physics_replicate"),
    )
    for module_name, function_name, span_name in clone_spans:
        _wrap_module_function(module_name, function_name, span_name, recorder)


def _install_runtime_tracing(env: gym.Env, recorder: MemoryRecorder) -> None:
    unwrapped = env.unwrapped
    _wrap_method(unwrapped, "_reset_idx", "env.reset_idx", recorder)

    _wrap_manager(unwrapped, "action_manager", ("process_action", "apply_action", "reset"), recorder)
    _wrap_manager(unwrapped, "command_manager", ("compute", "reset"), recorder)
    _wrap_manager(unwrapped, "observation_manager", ("compute", "reset"), recorder)
    _wrap_manager(unwrapped, "reward_manager", ("compute", "reset"), recorder)
    _wrap_manager(unwrapped, "termination_manager", ("compute", "reset"), recorder)
    _wrap_manager(unwrapped, "curriculum_manager", ("compute", "reset"), recorder)
    _wrap_manager(unwrapped, "event_manager", ("apply", "reset"), recorder)
    _wrap_manager(unwrapped, "recorder_manager", ("reset",), recorder)

    _wrap_method(unwrapped.scene, "write_data_to_sim", "scene.write_data_to_sim", recorder)
    _wrap_method(unwrapped.scene, "update", "scene.update", recorder)
    _wrap_method(unwrapped.scene, "reset", "scene.reset", recorder)
    _wrap_method(unwrapped.sim, "step", "sim.step", recorder)
    _wrap_method(unwrapped.sim, "render", "sim.render", recorder)


# ---------------------------------------------------------------------------
# Tensor accounting -- attribute torch memory to env-owned tensors
# ---------------------------------------------------------------------------


@dataclass
class TensorRecord:
    path: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    bytes: int
    aliases: list[str] = field(default_factory=list)
    """Other attribute paths that resolve to the same GPU storage as :attr:`path`.

    Two ``torch.Tensor`` Python wrappers can share the same underlying allocation —
    most commonly when the same warp array is exposed both directly and via
    ``wp.to_torch``. We keep the first path as the canonical name and stash
    subsequent paths here so the breakdown reflects unique allocations rather than
    counting wrappers.
    """


def _walk_tensors(root: Any, max_depth: int = 6) -> list[TensorRecord]:
    """Walk ``root`` and collect every CUDA tensor reachable through ``__dict__``.

    This is a best-effort accounting: it follows attribute references but does
    not chase into arbitrary containers like dicts of dicts beyond ``max_depth``.
    Tensors reached by multiple paths are reported once. Distinct ``torch.Tensor``
    objects that map onto the same underlying CUDA allocation (e.g. a warp array
    exposed both directly and via ``wp.to_torch``) are deduped by storage identity
    — the first path wins and the rest are recorded as aliases — so the totals
    reflect unique allocations.
    """
    seen_ids: set[int] = set()
    storage_index: dict[tuple[int, int, int, int], TensorRecord] = {}
    out: list[TensorRecord] = []

    def visit(obj: Any, path: str, depth: int) -> None:
        if depth > max_depth or obj is None:
            return
        obj_id = id(obj)
        if obj_id in seen_ids:
            return
        if isinstance(obj, torch.Tensor):
            seen_ids.add(obj_id)
            # Storage identity: (device, base data_ptr, byte size, byte offset).
            # data_ptr alone is not sufficient — different slices of the same
            # buffer share base ptr but cover different ranges.
            try:
                storage_offset = obj.storage_offset() * obj.element_size() if obj.numel() else 0
                key = (
                    obj.device.index if obj.device.index is not None else -1,
                    obj.data_ptr(),
                    obj.numel() * obj.element_size(),
                    storage_offset,
                )
            except Exception:  # noqa: BLE001 -- defensive: meta tensors etc.
                key = (-2, id(obj), 0, 0)
            existing = storage_index.get(key)
            if existing is not None:
                existing.aliases.append(path)
                return
            record = TensorRecord(
                path=path,
                shape=tuple(obj.shape),
                dtype=str(obj.dtype).removeprefix("torch."),
                device=str(obj.device),
                bytes=obj.numel() * obj.element_size(),
            )
            storage_index[key] = record
            out.append(record)
            return
        if isinstance(obj, (str, bytes, int, float, bool)) or obj is None:
            return
        seen_ids.add(obj_id)
        # ``__dict__`` / ``__slots__``
        attr_iter: Iterable[tuple[str, Any]]
        if hasattr(obj, "__dict__"):
            attr_iter = list(vars(obj).items())
        elif hasattr(obj, "__slots__"):
            attr_iter = [(s, getattr(obj, s, None)) for s in obj.__slots__]  # type: ignore[attr-defined]
        else:
            attr_iter = []
        for key, value in attr_iter:
            if key.startswith("__"):
                continue
            visit(value, f"{path}.{key}", depth + 1)
        # mappings / sequences
        if isinstance(obj, dict):
            for k, v in obj.items():
                visit(v, f"{path}[{k!r}]", depth + 1)
        elif isinstance(obj, (list, tuple, set, frozenset)) and not isinstance(obj, torch.Tensor):
            for i, v in enumerate(obj):
                visit(v, f"{path}[{i}]", depth + 1)

    visit(root, "env", 0)
    return out


def _print_tensor_breakdown(label: str, root: Any, top_n: int) -> list[dict[str, Any]]:
    print(f"\nLargest CUDA tensors reachable from {label} (top {top_n}):")
    records = _walk_tensors(root)
    records = [r for r in records if r.device.startswith("cuda")]
    records.sort(key=lambda r: r.bytes, reverse=True)
    if not records:
        print("  <no CUDA tensors found>")
        return []
    total_bytes = sum(r.bytes for r in records)
    print(f"  total accounted: {_bytes_to_mib(total_bytes):.1f} MiB across {len(records)} tensors")
    header = f"  {'path':<78} {'shape':<24} {'dtype':<10} {'MiB':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    rows: list[dict[str, Any]] = []
    for r in records[:top_n]:
        shape_str = "x".join(str(x) for x in r.shape) if r.shape else "()"
        path_str = r.path if len(r.path) <= 78 else "..." + r.path[-75:]
        print(f"  {path_str:<78} {shape_str:<24} {r.dtype:<10} {_bytes_to_mib(r.bytes):>8.1f}")
        rows.append({"path": r.path, "shape": list(r.shape), "dtype": r.dtype, "bytes": r.bytes})
    return rows


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def _apply_cli_overrides(env_cfg: Any, agent_cfg: Any) -> None:
    if args_cli.num_envs is not None and hasattr(env_cfg, "scene"):
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None and hasattr(env_cfg, "sim"):
        env_cfg.sim.device = args_cli.device
    if args_cli.disable_fabric and hasattr(env_cfg, "sim"):
        env_cfg.sim.use_fabric = False

    agent_seed = getattr(agent_cfg, "seed", None)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_seed


def _make_actions(env: gym.Env) -> torch.Tensor:
    unwrapped = env.unwrapped
    if args_cli.action_mode == "zero":
        return torch.zeros(unwrapped.action_space.shape, device=unwrapped.device)
    return 2.0 * torch.rand(unwrapped.action_space.shape, device=unwrapped.device) - 1.0


def _maybe_start_history() -> None:
    if not args_cli.memory_history:
        return
    try:
        torch.cuda.memory._record_memory_history(max_entries=200_000)
        print("[TRACE] Recording CUDA allocator history (will dump on exit).")
    except Exception as exc:  # noqa: BLE001
        print(f"[TRACE] Could not enable memory history: {exc}")


def _maybe_dump_history() -> None:
    if not args_cli.memory_history:
        return
    out = Path(args_cli.memory_history)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.cuda.memory._dump_snapshot(str(out))
        print(f"[TRACE] Wrote CUDA memory snapshot pickle to: {out}")
    except Exception as exc:  # noqa: BLE001
        print(f"[TRACE] Could not dump memory snapshot: {exc}")


def _trace_env_only(env_cfg: Any, recorder: MemoryRecorder, tensor_dump: dict[str, Any]) -> None:
    env: gym.Env | None = None
    with recorder.span("env_creation", task=args_cli.task):
        env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        if not args_cli.no_manager_trace:
            _install_runtime_tracing(env, recorder)

        with recorder.span("env_first_reset"):
            env.reset()

        tensor_dump["after_first_reset"] = _print_tensor_breakdown(
            "env (after first reset)", env.unwrapped, args_cli.tensor_top
        )

        for step_id in range(max(args_cli.trace_steps, 0)):
            actions = _make_actions(env)
            with recorder.span("env_step", step=step_id, action_mode=args_cli.action_mode):
                env.step(actions)

        tensor_dump["after_steps"] = _print_tensor_breakdown("env (after stepping)", env.unwrapped, args_cli.tensor_top)
    finally:
        if env is not None:
            with recorder.span("env_close"):
                env.close()


def _trace_with_rsl_rl(env_cfg: Any, agent_cfg: Any, recorder: MemoryRecorder, tensor_dump: dict[str, Any]) -> None:
    """Build the RSL-RL runner and run ``--rsl_rl_iters`` learning iterations."""
    # Imports kept local so trace_steps-only mode does not pay for rsl_rl import cost.
    with recorder.span("rsl_rl_import"):
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    import importlib.metadata as _metadata

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _metadata.version("rsl-rl-lib"))
    env_cfg.seed = agent_cfg.seed

    env: gym.Env | None = None
    runner: Any = None
    try:
        with recorder.span("env_creation", task=args_cli.task):
            env = gym.make(args_cli.task, cfg=env_cfg)

        if not args_cli.no_manager_trace:
            _install_runtime_tracing(env, recorder)

        with recorder.span("rsl_rl_wrap"):
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        with recorder.span("rsl_rl_runner_init"):
            runner = agent_cfg.class_type(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

        tensor_dump["after_runner_init_env"] = _print_tensor_breakdown(
            "env (after runner init)", env.unwrapped, args_cli.tensor_top
        )
        tensor_dump["after_runner_init_runner"] = _print_tensor_breakdown(
            "runner (after init)", runner, args_cli.tensor_top
        )

        # learn() drives rollout collection + update. One iteration is usually
        # enough to expose the rollout buffer's steady-state footprint.
        with recorder.span("rsl_rl_learn", iterations=int(args_cli.rsl_rl_iters)):
            runner.learn(num_learning_iterations=int(args_cli.rsl_rl_iters), init_at_random_ep_len=True)

        tensor_dump["after_learn"] = _print_tensor_breakdown("runner (after learn)", runner, args_cli.tensor_top)
    finally:
        if env is not None:
            with recorder.span("env_close"):
                env.close()


def _print_outputs(recorder: MemoryRecorder, tensor_dump: dict[str, Any]) -> None:
    print()
    for line in recorder.summary_lines(top_n=args_cli.top_events):
        print(line)

    # Final absolute snapshot
    final = recorder.sample()
    print("\nFinal memory snapshot:")
    final_dict = final.to_dict()
    for key in ("torch_allocated_mib", "torch_reserved_mib", "cpu_rss_mib", "nvml_used_mib", "nvml_total_mib"):
        value = final_dict[key]
        if value is None:
            print(f"  {key:<26} n/a")
        else:
            print(f"  {key:<26} {value:>10.1f} MiB")

    if args_cli.output:
        recorder.metadata["tensor_breakdown"] = tensor_dump
        recorder.metadata["final_sample"] = final.to_dict()
        recorder.export_json(args_cli.output)
        print(f"\n[TRACE] Wrote memory trace JSON to: {args_cli.output}")


def _record_phase(recorder: MemoryRecorder, name: str, fn: Callable, *fn_args, **fn_kwargs):
    start = recorder.sample()
    result = fn(*fn_args, **fn_kwargs)
    end = recorder.sample()
    recorder.record_outside_span(name, start, end)
    return result


def main() -> None:
    recorder = MemoryRecorder(
        metadata={
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "seed": args_cli.seed,
            "trace_steps": args_cli.trace_steps,
            "rsl_rl_iters": args_cli.rsl_rl_iters,
            "action_mode": args_cli.action_mode,
            "hydra_args": remaining_args,
        },
        synchronize=not args_cli.no_synchronize,
    )

    # Imports already happened, but we still record the python-imports bracket
    # so the JSON has a measurement before the simulator starts.
    pre_import_sample = MemorySample(
        timestamp_ns=_imports_t0,
        torch_allocated=0,
        torch_reserved=0,
        torch_peak_allocated=0,
        cpu_rss=0,
        nvml_used=None,
        nvml_total=None,
    )
    post_import_sample = recorder.sample()
    post_import_sample = dataclasses.replace(post_import_sample, timestamp_ns=_imports_t1)
    recorder.record_outside_span("python_imports", pre_import_sample, post_import_sample)

    _maybe_start_history()
    tensor_dump: dict[str, Any] = {}

    env_cfg, agent_cfg = _record_phase(recorder, "task_config", resolve_task_config, args_cli.task, args_cli.agent)
    _apply_cli_overrides(env_cfg, agent_cfg)

    launch_ctx = launch_simulation(env_cfg, args_cli)
    app_launch_start = recorder.sample()
    launch_ctx.__enter__()
    app_launch_end = recorder.sample()
    recorder.record_outside_span("app_launch", app_launch_start, app_launch_end)
    try:
        _install_startup_tracing(recorder)
        if args_cli.rsl_rl_iters > 0:
            _trace_with_rsl_rl(env_cfg, agent_cfg, recorder, tensor_dump)
        else:
            _trace_env_only(env_cfg, recorder, tensor_dump)
        # Force a GC + empty_cache pass so the "after teardown" sample is comparable.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _print_outputs(recorder, tensor_dump)
    finally:
        launch_ctx.__exit__(None, None, None)
        _maybe_dump_history()


if __name__ == "__main__":
    main()
