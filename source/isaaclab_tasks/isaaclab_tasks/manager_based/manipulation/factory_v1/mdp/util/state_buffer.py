# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


class StateBuffer:
    """Ring buffer of env-origin-relative reset states with per-slot tag metadata."""

    def __init__(self, max_size: int, state_dim: int, device: torch.device):
        self.data = torch.zeros((max_size, state_dim), device=device)
        self.max_size = max_size
        self._size = 0
        self._ptr = 0
        self.tag_names: list[str] | None = None
        self.tags = torch.full((max_size,), -1, device=device, dtype=torch.int64)
        self.success_rates: torch.Tensor | None = None

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size >= self.max_size

    def add(self, states: torch.Tensor) -> tuple[int, int]:
        """Append states to the ring buffer.

        Returns:
            ``(start, count)`` — the buffer offset where writing began and how
            many states were actually written (capped to avoid wrapping mid-batch).
        """
        n = min(states.shape[0], self.max_size - self._ptr)
        start = self._ptr
        self.data[start : start + n] = states[:n]
        self._ptr = (start + n) % self.max_size
        self._size = min(self._size + n, self.max_size)
        return start, n

    def sample(self, indices: torch.Tensor) -> torch.Tensor:
        return self.data[indices]

    def set_tag_names(self, tag_names: list[str]) -> None:
        self.tag_names = tag_names

    def set_tags(self, indices: torch.Tensor, tag_ids: torch.Tensor) -> None:
        self.tags[indices] = tag_ids.long()
