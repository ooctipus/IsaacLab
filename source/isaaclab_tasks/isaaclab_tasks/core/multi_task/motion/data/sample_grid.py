# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declared sample clocks for motion references and expert transitions."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class MotionSampleGrid:
    """Map stored source frames to the sample clock consumed by a learner.

    Source-row sampling preserves every stored row exactly. Uniform sampling
    reproduces continuous-time consumers that stop before the source endpoint.
    """

    class Mode(enum.Enum):
        """Supported relations between source rows and consumed samples."""

        SOURCE_ROWS = "source_rows"
        UNIFORM_BEFORE_SOURCE_END = "uniform_before_source_end"

    mode: Mode
    step_seconds: float | None

    def __post_init__(self) -> None:
        """Validate the clock declaration."""
        if self.mode is self.Mode.SOURCE_ROWS:
            if self.step_seconds is not None:
                raise ValueError("Source-row sampling does not declare step_seconds.")
            return
        if self.step_seconds is None or not math.isfinite(self.step_seconds) or self.step_seconds <= 0.0:
            raise ValueError("Uniform sampling requires finite positive step_seconds.")

    @classmethod
    def source_rows(cls) -> MotionSampleGrid:
        """Consume every stored source row in order."""
        return cls(cls.Mode.SOURCE_ROWS, None)

    @classmethod
    def uniform_before_source_end(cls, *, step_seconds: float) -> MotionSampleGrid:
        """Consume a uniform clock whose last sample precedes the source endpoint."""
        return cls(cls.Mode.UNIFORM_BEFORE_SOURCE_END, step_seconds)

    def sample_count(self, *, frame_count: int, source_fps: float) -> int:
        """Return consumed samples for one source clip."""
        if frame_count < 1:
            raise ValueError("frame_count must be positive.")
        if not math.isfinite(source_fps) or source_fps <= 0.0:
            raise ValueError("source_fps must be finite and positive.")
        if self.mode is self.Mode.SOURCE_ROWS:
            return frame_count
        assert self.step_seconds is not None
        duration_seconds = (frame_count - 1) / source_fps
        return math.ceil(duration_seconds / self.step_seconds)

    def window_count(self, *, frame_count: int, source_fps: float, length: int) -> int:
        """Return valid current/reached window starts for one source clip."""
        if length < 1:
            raise ValueError("length must be positive.")
        return max(self.sample_count(frame_count=frame_count, source_fps=source_fps) - length, 0)

    def sample_counts(self, frame_counts: torch.Tensor, source_fps: torch.Tensor) -> torch.Tensor:
        """Return vectorized consumed-sample counts on the input tensor device."""
        if self.mode is self.Mode.SOURCE_ROWS:
            return frame_counts
        assert self.step_seconds is not None
        duration_seconds = (frame_counts - 1) / source_fps
        return torch.ceil(duration_seconds / self.step_seconds).to(torch.int64)

    def time_seconds(self, sample_indices: torch.Tensor, source_fps: torch.Tensor) -> torch.Tensor:
        """Map integer sample indices to source times [s]."""
        if self.mode is self.Mode.SOURCE_ROWS:
            return sample_indices / source_fps
        assert self.step_seconds is not None
        return sample_indices * self.step_seconds
