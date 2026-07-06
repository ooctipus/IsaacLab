# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mutable sampling policy over one immutable motion task table."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from .motion_task_table import MotionTaskTable


def _reset_source_data(
    reset_sources: tuple[tuple[str, float], ...],
    device: torch.device,
) -> tuple[tuple[str, ...], torch.Tensor]:
    """Validate reset-source policy and materialize its probability tensor."""
    if not isinstance(reset_sources, tuple) or not reset_sources:
        raise ValueError("reset_sources must be a nonempty tuple of name/probability pairs.")
    names: list[str] = []
    probabilities: list[float] = []
    for source in reset_sources:
        if not isinstance(source, tuple) or len(source) != 2:
            raise ValueError("Each reset source must be one (name, probability) tuple.")
        name, probability = source
        if not isinstance(name, str) or not name:
            raise ValueError("Reset-source names must be nonempty strings.")
        if isinstance(probability, bool) or not isinstance(probability, int | float):
            raise ValueError("Reset-source probabilities must be real scalars.")
        probability = float(probability)
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError("Reset-source probabilities must be finite and nonnegative.")
        names.append(name)
        probabilities.append(probability)
    if len(set(names)) != len(names):
        raise ValueError("Reset-source names must be unique.")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("Reset-source probabilities must sum to one.")
    return tuple(names), torch.tensor(probabilities, dtype=torch.float32, device=device)


class MotionSampler:
    """Own mutable task, reset-source, and reset-time sampling state."""

    __slots__ = (
        "_capacity",
        "_clip_slots",
        "_fractions",
        "_local_rows",
        "_reset_time_mode",
        "_row_counts",
        "_sampling_row_counts",
        "_sampling_row_starts",
        "_task_rows",
        "clip_priorities",
        "generator",
        "reset_source_names",
        "reset_source_probabilities",
        "table",
    )

    def __init__(
        self,
        table: MotionTaskTable,
        reset_sources: tuple[tuple[str, float], ...],
        *,
        capacity: int,
        seed: int,
    ) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("Motion sampler seed must be an integer.")
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Motion sampler capacity must be a positive integer.")
        reset_source_names, reset_source_probabilities = _reset_source_data(reset_sources, table.device)
        sampling_row_starts = table.clip_start_rows
        sampling_row_counts = (
            table.frame_counts if table.task_row_mode == "source_frames" else torch.ones_like(sampling_row_starts)
        )
        if sampling_row_starts.shape != (len(table.clip_ids),) or sampling_row_counts.shape != (len(table.clip_ids),):
            raise ValueError("Motion sampling rows must align with the clip axis.")
        torch._assert_async(
            torch.sum(sampling_row_counts) == table.num_tasks,
            "Motion sampling rows must cover the complete descriptor table.",
        )

        generator = torch.Generator(device=table.device)
        generator.manual_seed(seed)
        self.table = table
        self._capacity = capacity
        self._sampling_row_starts = sampling_row_starts
        self._sampling_row_counts = sampling_row_counts
        self._clip_slots = torch.empty(capacity, dtype=torch.int64, device=table.device)
        self._task_rows = torch.empty(capacity, dtype=torch.int64, device=table.device)
        self._fractions = torch.empty(capacity, device=table.device)
        self._row_counts = torch.empty(capacity, dtype=torch.int64, device=table.device)
        self._local_rows = torch.empty(capacity, dtype=torch.int64, device=table.device)
        self.reset_source_names = reset_source_names
        self.reset_source_probabilities = reset_source_probabilities
        base_priority = table.base_priorities
        if (
            base_priority.shape != (len(table.clip_ids),)
            or not bool(torch.all(torch.isfinite(base_priority)))
            or not bool(torch.all(base_priority > 0.0))
        ):
            raise ValueError("Motion base priorities must be finite positive values aligned with clips.")
        self.clip_priorities = base_priority.clone()
        self.generator = generator
        self._reset_time_mode: Literal["uniform", "range_start"] = "uniform"

    @property
    def reset_time_mode(self) -> Literal["uniform", "range_start"]:
        """Reset-time selection used by the next draw."""
        return self._reset_time_mode

    def set_reset_time_mode(self, mode: Literal["uniform", "range_start"]) -> None:
        """Select ordinary uniform resets or exact range-start resets."""
        if mode not in ("uniform", "range_start"):
            raise ValueError("Motion reset-time mode must be 'uniform' or 'range_start'.")
        self._reset_time_mode = mode

    @contextmanager
    def reset_sampling_scope(self, seed: int, reset_source_name: str | None) -> Iterator[None]:
        """Temporarily select deterministic range-start reset sampling."""
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("Evaluation seed must be an integer.")
        generator_state = self.generator.get_state().clone()
        reset_time_mode = self.reset_time_mode
        reset_probabilities = None
        if reset_source_name is not None:
            try:
                reset_source_index = self.reset_source_names.index(reset_source_name)
            except ValueError as error:
                raise ValueError(f"Unknown evaluation reset source: {reset_source_name!r}.") from error
            reset_probabilities = self.reset_source_probabilities.clone()
            self.reset_source_probabilities.zero_()
            self.reset_source_probabilities[reset_source_index] = 1.0
        self.generator.manual_seed(seed)
        self.set_reset_time_mode("range_start")
        try:
            yield
        finally:
            self.set_reset_time_mode(reset_time_mode)
            self.generator.set_state(generator_state)
            if reset_probabilities is not None:
                self.reset_source_probabilities.copy_(reset_probabilities)

    def sample_rows(self, count: int) -> torch.Tensor:
        """Sample task rows into fixed-capacity storage and return its populated prefix."""
        if count > self._capacity:
            raise ValueError(f"Motion sampler count {count} exceeds capacity {self._capacity}.")
        clip_slots = self._clip_slots[:count]
        task_rows = self._task_rows[:count]
        torch.multinomial(
            self.clip_priorities,
            count,
            replacement=True,
            generator=self.generator,
            out=clip_slots,
        )
        torch.index_select(self._sampling_row_starts, 0, clip_slots, out=task_rows)
        if self.table.task_row_mode == "source_frames":
            fractions = self._fractions[:count]
            row_counts = self._row_counts[:count]
            local_rows = self._local_rows[:count]
            torch.index_select(self._sampling_row_counts, 0, clip_slots, out=row_counts)
            torch.rand(fractions.shape, device=self.table.device, generator=self.generator, out=fractions)
            torch.mul(fractions, row_counts, out=fractions)
            torch.floor(fractions, out=fractions)
            local_rows.copy_(fractions)
            task_rows.add_(local_rows)
        return task_rows

    def sample_reset_sources(self, output: torch.Tensor) -> None:
        """Draw reset-source indices into caller-owned storage."""
        torch.multinomial(
            self.reset_source_probabilities,
            output.shape[0],
            replacement=True,
            generator=self.generator,
            out=output,
        )

    def sample_reset_times(self, reset_time_ranges_seconds: torch.Tensor, output: torch.Tensor) -> None:
        """Draw reset times [s] into caller-owned storage."""
        if self._reset_time_mode == "range_start":
            output.copy_(reset_time_ranges_seconds[:, 0])
            return
        torch.rand(output.shape, device=self.table.device, generator=self.generator, out=output)
        torch.lerp(reset_time_ranges_seconds[:, 0], reset_time_ranges_seconds[:, 1], output, out=output)
