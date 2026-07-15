# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scoped nonlinear-iteration evidence for motion-retargeting benchmarks."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps

_STAGE_NAMES = ("frame_global", "frame_local", "feasibility", "source", "contact")


@dataclass(slots=True)
class RetargetStagePerformance:
    """Benchmark-local call, iteration, and elapsed-time evidence for one stage."""

    calls: int = 0
    iterations: int = 0
    wall_seconds: float = 0.0

    def report(self, gpu_seconds: float | None) -> dict[str, float | int | None]:
        """Return one strict-JSON stage record."""
        return {
            "calls": self.calls,
            "iterations": self.iterations,
            "wall_seconds": self.wall_seconds,
            "gpu_seconds": gpu_seconds,
        }


@dataclass(slots=True)
class TrajectoryNonlinearIterations:
    """Count and time frame and trajectory solves during one benchmark build."""

    source: int = 0
    contact: int = 0
    progress_interval_seconds: float = 30.0
    stages: dict[str, RetargetStagePerformance] = field(
        default_factory=lambda: {name: RetargetStagePerformance() for name in _STAGE_NAMES}
    )
    _last_progress_ns: int = field(default_factory=time.monotonic_ns)
    _active_gpu_span: tuple[str, object, object, object] | None = None
    _gpu_spans: list[tuple[str, object, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.progress_interval_seconds < 0.0:
            raise ValueError("progress_interval_seconds must be non-negative")

    def record(
        self,
        *,
        residual_activity: object | None,
        inequalities: object | None,
        feasibility_only: bool = False,
        wall_seconds: float = 0.0,
    ) -> None:
        """Record one outer iteration from its trajectory phase inputs."""
        del inequalities
        if residual_activity is None:
            self.source += 1
            stage = "feasibility" if feasibility_only else "source"
        else:
            self.contact += 1
            stage = "contact"
        self.record_stage(stage, iterations=1, wall_seconds=wall_seconds)

    def begin_stage(self, stage: str, solver: object) -> None:
        """Start or continue one benchmark-only GPU timing span."""
        if stage not in self.stages:
            raise ValueError(f"Unknown retarget stage: {stage}")
        raw_device = getattr(solver, "device", None)
        if raw_device is None:
            raw_device = getattr(getattr(solver, "optimizer", None), "device", None)
        if raw_device is None:
            return
        import warp as wp

        device = wp.get_device(str(raw_device))
        if not device.is_cuda:
            return
        if self._active_gpu_span is not None and self._active_gpu_span[0] == stage:
            return
        self.finish_gpu_span()
        start = wp.Event(device, enable_timing=True)
        end = wp.Event(device, enable_timing=True)
        wp.get_stream(device).record_event(start)
        self._active_gpu_span = (stage, device, start, end)

    def finish_gpu_span(self) -> None:
        """Close the current asynchronous GPU span without synchronizing."""
        if self._active_gpu_span is None:
            return
        import warp as wp

        stage, device, start, end = self._active_gpu_span
        wp.get_stream(device).record_event(end)
        self._gpu_spans.append((stage, start, end))
        self._active_gpu_span = None

    def record_stage(self, stage: str, *, iterations: int, wall_seconds: float) -> None:
        """Record one completed solver call and emit rate-limited benchmark progress."""
        if iterations < 0 or wall_seconds < 0.0:
            raise ValueError("Stage iterations and wall time must be non-negative")
        values = self.stages[stage]
        values.calls += 1
        values.iterations += iterations
        values.wall_seconds += wall_seconds
        now_ns = time.monotonic_ns()
        interval_ns = int(self.progress_interval_seconds * 1.0e9)
        if interval_ns == 0 or now_ns - self._last_progress_ns >= interval_ns:
            print(
                (
                    f"[motion benchmark] stage={stage} calls={values.calls} "
                    f"iterations={values.iterations} wall_seconds={values.wall_seconds:.3f}"
                ),
                file=sys.stderr,
                flush=True,
            )
            self._last_progress_ns = now_ns

    def report(self) -> dict[str, int]:
        """Return the backward-compatible trajectory-iteration report."""
        return {"source": self.source, "contact": self.contact, "total": self.source + self.contact}

    def stage_report(self) -> dict[str, dict[str, float | int | None]]:
        """Return call, iteration, wall, and GPU timing for every observed stage."""
        self.finish_gpu_span()
        gpu_seconds: dict[str, float] = {}
        if self._gpu_spans:
            import warp as wp

            for stage, start, end in self._gpu_spans:
                gpu_seconds[stage] = gpu_seconds.get(stage, 0.0) + wp.get_event_elapsed_time(start, end) / 1000.0
        result = {name: values.report(gpu_seconds.get(name)) for name, values in self.stages.items()}
        result["total"] = {
            "calls": sum(values.calls for values in self.stages.values()),
            "iterations": sum(values.iterations for values in self.stages.values()),
            "wall_seconds": sum(values.wall_seconds for values in self.stages.values()),
            "gpu_seconds": None if not gpu_seconds else sum(gpu_seconds.values()),
        }
        return result


@contextmanager
def count_trajectory_nonlinear_iterations(
    *, progress_interval_seconds: float = 30.0
) -> Iterator[TrajectoryNonlinearIterations]:
    """Instrument benchmark-local frame and trajectory solvers and restore their methods."""
    from newton import ik

    from isaaclab_tasks.core.multi_task.kinematics import IKTrajectorySolver

    counts = TrajectoryNonlinearIterations(progress_interval_seconds=progress_interval_seconds)
    original_frame_solve = ik.IKSolver.solve
    original_frame_step = ik.IKSolver.step
    original_trajectory_solve = IKTrajectorySolver.solve

    @wraps(original_frame_solve)
    def counted_frame_solve(solver, *args, **kwargs):
        stage = "frame_global" if solver.n_seeds > 1 else "frame_local"
        counts.begin_stage(stage, solver)
        started_ns = time.monotonic_ns()
        try:
            result = original_frame_solve(solver, *args, **kwargs)
        except Exception:
            counts.record_stage(stage, iterations=0, wall_seconds=(time.monotonic_ns() - started_ns) / 1.0e9)
            raise
        counts.record_stage(
            stage,
            iterations=int(result.iterations),
            wall_seconds=(time.monotonic_ns() - started_ns) / 1.0e9,
        )
        return result

    @wraps(original_frame_step)
    def counted_frame_step(solver, *args, **kwargs):
        counts.begin_stage("frame_local", solver)
        iterations = kwargs.get("iterations", args[2] if len(args) > 2 else 50)
        started_ns = time.monotonic_ns()
        try:
            return original_frame_step(solver, *args, **kwargs)
        finally:
            counts.record_stage(
                "frame_local",
                iterations=int(iterations),
                wall_seconds=(time.monotonic_ns() - started_ns) / 1.0e9,
            )

    @wraps(original_trajectory_solve)
    def counted_trajectory_solve(solver, *args, **kwargs):
        residual_activity = kwargs.get("residual_activity")
        feasibility_only = kwargs.get("feasibility_only", False)
        stage = "contact" if residual_activity is not None else "feasibility" if feasibility_only else "source"
        counts.begin_stage(stage, solver)
        started_ns = time.monotonic_ns()
        try:
            return original_trajectory_solve(solver, *args, **kwargs)
        finally:
            counts.record(
                residual_activity=residual_activity,
                inequalities=kwargs.get("inequalities"),
                feasibility_only=feasibility_only,
                wall_seconds=(time.monotonic_ns() - started_ns) / 1.0e9,
            )

    ik.IKSolver.solve = counted_frame_solve
    ik.IKSolver.step = counted_frame_step
    IKTrajectorySolver.solve = counted_trajectory_solve
    try:
        yield counts
    finally:
        counts.finish_gpu_span()
        ik.IKSolver.solve = original_frame_solve
        ik.IKSolver.step = original_frame_step
        IKTrajectorySolver.solve = original_trajectory_solve
