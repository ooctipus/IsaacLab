# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_tasks.core.multi_task.utils.trace import (
    TraceRecorder,
    get_trace_recorder,
    reset_trace_recorder,
    set_trace_recorder,
    trace_span,
)


def test_trace_span_noops_without_active_recorder():
    with trace_span("inactive"):
        pass

    assert get_trace_recorder() is None


def test_trace_recorder_records_nested_spans():
    recorder = TraceRecorder(synchronize=False)
    token = set_trace_recorder(recorder)

    try:
        with trace_span("outer", phase="setup"):
            with trace_span("inner", count=3):
                pass
    finally:
        reset_trace_recorder(token)

    events = {event.name: event for event in recorder.events}
    assert set(events) == {"outer", "inner"}
    assert events["inner"].parent_id == events["outer"].event_id
    assert events["outer"].metadata == {"phase": "setup"}
    assert events["inner"].metadata == {"count": 3}
    assert events["outer"].duration_ms >= 0.0
    assert events["inner"].duration_ms >= 0.0

    aggregate = {row["name"]: row for row in recorder.aggregate()}
    assert aggregate["outer"]["count"] == 1
    assert aggregate["inner"]["count"] == 1
