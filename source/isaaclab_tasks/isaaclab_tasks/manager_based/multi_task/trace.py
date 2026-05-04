# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight tracing utilities for multi-task diagnostics."""

from __future__ import annotations

import contextlib
import contextvars
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import warp as wp

_ACTIVE_RECORDER: contextvars.ContextVar[TraceRecorder | None] = contextvars.ContextVar(
    "multi_task_trace_recorder", default=None
)


@dataclass
class TraceEvent:
    """Single timed trace event."""

    event_id: int
    name: str
    parent_id: int | None
    start_ns: int
    end_ns: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Event duration [ms]."""
        return (self.end_ns - self.start_ns) / 1_000_000.0

    def to_dict(self) -> dict[str, Any]:
        """Convert the event to a JSON-serializable dictionary."""
        return {
            "id": self.event_id,
            "name": self.name,
            "parent_id": self.parent_id,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


class TraceRecorder:
    """Hierarchical wall-clock trace recorder.

    Args:
        metadata: Metadata attached to the exported trace.
        synchronize: Whether to synchronize CUDA/Warp before sampling timestamps.
    """

    def __init__(self, metadata: dict[str, Any] | None = None, synchronize: bool = True):
        self.metadata = metadata or {}
        self.synchronize = synchronize
        self.events: list[TraceEvent] = []
        self._stack: list[int] = []
        self._next_event_id = 0
        self._origin_ns: int | None = None

    def span(self, name: str, **metadata) -> TraceSpan:
        """Create a timed span."""
        return TraceSpan(self, name, metadata)

    def timestamp_ns(self) -> int:
        """Return a synchronized timestamp from :func:`time.perf_counter_ns`."""
        self._synchronize()
        return time.perf_counter_ns()

    def record_duration_ns(
        self,
        name: str,
        start_ns: int,
        end_ns: int,
        parent_id: int | None = None,
        **metadata,
    ) -> None:
        """Record an externally measured duration.

        Args:
            name: Event name.
            start_ns: Start timestamp from :func:`time.perf_counter_ns`.
            end_ns: End timestamp from :func:`time.perf_counter_ns`.
            parent_id: Optional parent event id.
            **metadata: JSON-serializable event metadata.
        """
        self._append_event(name, start_ns, end_ns, parent_id, metadata)

    def export_json(self, path: str | Path) -> None:
        """Write the trace as structured JSON."""
        payload = {
            "metadata": self.metadata,
            "events": [event.to_dict() for event in self.events],
            "aggregate": self.aggregate(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def export_chrome_trace(self, path: str | Path) -> None:
        """Write a Chrome trace JSON file."""
        if not self.events:
            payload = []
        else:
            origin_ns = min(event.start_ns for event in self.events)
            payload = [
                {
                    "name": event.name,
                    "cat": "multi_task",
                    "ph": "X",
                    "ts": (event.start_ns - origin_ns) / 1000.0,
                    "dur": (event.end_ns - event.start_ns) / 1000.0,
                    "pid": 0,
                    "tid": event.parent_id if event.parent_id is not None else -1,
                    "args": event.metadata,
                }
                for event in self.events
            ]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def aggregate(self) -> list[dict[str, float | int | str]]:
        """Aggregate events by name."""
        values: dict[str, list[float]] = defaultdict(list)
        for event in self.events:
            values[event.name].append(event.duration_ms)
        rows = []
        for name, durations in values.items():
            total = sum(durations)
            rows.append(
                {
                    "name": name,
                    "count": len(durations),
                    "total_ms": total,
                    "mean_ms": total / len(durations),
                    "max_ms": max(durations),
                }
            )
        rows.sort(key=lambda row: float(row["total_ms"]), reverse=True)
        return rows

    def summary_lines(self, top_n: int = 25) -> list[str]:
        """Return a compact human-readable summary."""
        lines = ["Trace summary (aggregate by event name):"]
        rows = self.aggregate()[:top_n]
        if not rows:
            lines.append("  <no events>")
            return lines

        lines.append(f"  {'name':<44} {'count':>7} {'total ms':>10} {'mean ms':>10} {'max ms':>10}")
        lines.append(f"  {'-' * 44} {'-' * 7} {'-' * 10} {'-' * 10} {'-' * 10}")
        for row in rows:
            lines.append(
                f"  {str(row['name']):<44} {int(row['count']):>7d} "
                f"{float(row['total_ms']):>10.2f} {float(row['mean_ms']):>10.2f} "
                f"{float(row['max_ms']):>10.2f}"
            )
        return lines

    def _start_event(self, name: str) -> tuple[int, int, int | None]:
        self._synchronize()
        start_ns = time.perf_counter_ns()
        if self._origin_ns is None:
            self._origin_ns = start_ns
        event_id = self._next_event_id
        self._next_event_id += 1
        parent_id = self._stack[-1] if self._stack else None
        self._stack.append(event_id)
        return event_id, start_ns, parent_id

    def _stop_event(self, event_id: int, name: str, start_ns: int, parent_id: int | None, metadata: dict[str, Any]):
        self._synchronize()
        end_ns = time.perf_counter_ns()
        if not self._stack or self._stack[-1] != event_id:
            raise RuntimeError(f"Trace span stack is corrupt while closing '{name}'.")
        self._stack.pop()
        self._append_event(name, start_ns, end_ns, parent_id, metadata, event_id=event_id)

    def _append_event(
        self,
        name: str,
        start_ns: int,
        end_ns: int,
        parent_id: int | None,
        metadata: dict[str, Any],
        event_id: int | None = None,
    ) -> None:
        if self._origin_ns is None:
            self._origin_ns = start_ns
        if event_id is None:
            event_id = self._next_event_id
            self._next_event_id += 1
        self.events.append(
            TraceEvent(
                event_id=event_id,
                name=name,
                parent_id=parent_id,
                start_ns=start_ns,
                end_ns=end_ns,
                metadata=_json_safe(metadata),
            )
        )

    def _synchronize(self) -> None:
        if not self.synchronize:
            return
        wp.synchronize()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()


class TraceSpan:
    """Context manager backing :meth:`TraceRecorder.span`."""

    def __init__(self, recorder: TraceRecorder, name: str, metadata: dict[str, Any]):
        self._recorder = recorder
        self._name = name
        self._metadata = metadata
        self._event_id: int | None = None
        self._start_ns: int | None = None
        self._parent_id: int | None = None

    def __enter__(self) -> TraceSpan:
        self._event_id, self._start_ns, self._parent_id = self._recorder._start_event(self._name)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is not None:
            self._metadata = {**self._metadata, "exception": exc_type.__name__}
        if self._event_id is None or self._start_ns is None:
            raise RuntimeError(f"Trace span '{self._name}' was closed before it was opened.")
        self._recorder._stop_event(
            self._event_id,
            self._name,
            self._start_ns,
            self._parent_id,
            self._metadata,
        )


def set_trace_recorder(recorder: TraceRecorder | None) -> contextvars.Token:
    """Set the active recorder for the current context."""
    return _ACTIVE_RECORDER.set(recorder)


def reset_trace_recorder(token: contextvars.Token) -> None:
    """Reset the active recorder using a token from :func:`set_trace_recorder`."""
    _ACTIVE_RECORDER.reset(token)


def get_trace_recorder() -> TraceRecorder | None:
    """Return the active trace recorder, if any."""
    return _ACTIVE_RECORDER.get()


def trace_span(name: str, **metadata):
    """Return a tracing span if tracing is active, otherwise a no-op context manager."""
    recorder = get_trace_recorder()
    if recorder is None:
        return contextlib.nullcontext()
    return recorder.span(name, **metadata)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)
