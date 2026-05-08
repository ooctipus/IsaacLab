# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable

import torch

from ..grid_downsample import extract_features, grid_bucket_downsample


class StateBuffer:
    """Ring buffer of env-origin-relative reset states with per-slot tag metadata.

    When ``target_size`` is smaller than ``max_size`` the buffer enters
    *oversample-then-thin* mode: it accumulates linearly up to
    ``max_size`` states, then runs a grid-bucket FPS-style thin via
    :func:`~isaaclab_tasks.manager_based.multi_task.grid_downsample.grid_bucket_downsample`
    over an :paramref:`fps_features` extractor, keeping the most
    spatially diverse ``target_size`` survivors at the front of the
    buffer. Otherwise (``target_size == max_size``) the buffer behaves
    as a pure ring with FIFO wrap.

    Compaction notifies callbacks registered via
    :meth:`register_compact_callback` with the surviving slot indices,
    so callers can permute any parallel arrays they keep in lockstep
    with the buffer (success rates, monitor history, etc.).
    """

    def __init__(
        self,
        max_size: int,
        state_dim: int,
        device: torch.device,
        target_size: int | None = None,
        fps_features: Callable | None = None,
    ):
        self.max_size = max_size
        self._target_size = max_size if target_size is None else int(target_size)
        self.fps_features = fps_features
        self.data = torch.zeros((max_size, state_dim), device=device)
        self._size = 0
        self._ptr = 0
        self.tag_names: list[str] | None = None
        self.tags = torch.full((max_size,), -1, device=device, dtype=torch.int64)
        self.success_rates: torch.Tensor | None = None
        self._compact_callbacks: list[Callable[[torch.Tensor], None]] = []

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size >= self.max_size

    @property
    def target_size(self) -> int:
        """Post-thin target. Equal to :attr:`max_size` when oversample is disabled."""
        return self._target_size

    def register_compact_callback(self, callback: Callable[[torch.Tensor], None]) -> None:
        """Register a callback invoked after each compaction.

        Args:
            callback: Receives the surviving slot indices (shape
                ``[target_size]``, sorted, on the buffer's device,
                ``int64`` dtype). The callback should permute its
                parallel data so index ``i`` aligns with the buffer's
                new slot ``i`` post-compact.
        """
        self._compact_callbacks.append(callback)

    def add(self, states: torch.Tensor) -> tuple[int, int]:
        """Append states to the buffer.

        Returns:
            ``(start, count)`` -- the buffer offset where writing began
            and how many states were actually written. Compaction (when
            oversample is enabled and the buffer just hit capacity) runs
            after the write and shrinks ``_size`` to ``target_size``;
            the returned ``(start, count)`` still describes the just-
            written region.
        """
        n = min(states.shape[0], self.max_size - self._ptr)
        start = self._ptr
        self.data[start : start + n] = states[:n]
        self._ptr = (start + n) % self.max_size
        self._size = min(self._size + n, self.max_size)
        if self._size >= self.max_size and self._target_size < self.max_size:
            self.compact()
        return start, n

    def add_with_tags(self, states: torch.Tensor, tags: torch.Tensor) -> tuple[int, int]:
        """Append states with matching per-slot tags.

        Tags are written *before* compaction triggers so the surviving
        subset's tags are correct.
        """
        n = min(states.shape[0], self.max_size - self._ptr)
        start = self._ptr
        self.data[start : start + n] = states[:n]
        self._ptr = (start + n) % self.max_size
        self._size = min(self._size + n, self.max_size)
        if n > 0:
            indices = torch.arange(start, start + n, device=self.tags.device)
            self.set_tags(indices, tags[:n])
        if self._size >= self.max_size and self._target_size < self.max_size:
            self.compact()
        return start, n

    def sample(self, indices: torch.Tensor) -> torch.Tensor:
        return self.data[indices]

    def set_tag_names(self, tag_names: list[str]) -> None:
        self.tag_names = tag_names

    def set_tags(self, indices: torch.Tensor, tag_ids: torch.Tensor) -> None:
        self.tags[indices] = tag_ids.long()

    def compact(self) -> torch.Tensor:
        """Thin the buffer down to ``target_size`` via grid-bucket FPS.

        Idempotent: if the buffer already holds at most ``target_size``
        states, returns ``arange(_size)`` without modifying anything.
        Otherwise, survivors land in slots ``[0, target_size)`` in
        their original relative order (sorted indices), the tail is
        zeroed, and any registered compact callbacks are invoked with
        the surviving index permutation so callers can update parallel
        arrays.

        This method is invoked automatically by :meth:`add` /
        :meth:`add_with_tags` when oversample is enabled and the buffer
        hits capacity. Callers that fill the buffer in a single shot
        (e.g. the locomotion task-table builder) may invoke it
        explicitly without first overflowing.

        Returns:
            Surviving slot indices (sorted, on-device, ``int64``). Same
            tensor that registered compact callbacks receive.
        """
        if self._size <= self._target_size:
            return torch.arange(self._size, device=self.data.device, dtype=torch.int64)
        target = self._target_size
        states = self.data[: self._size]
        features = extract_features(states, self.fps_features)
        # Sorted survivors preserve the temporal ordering of slots,
        # which is convenient for any caller that interprets slot
        # position as "freshness".
        keep_sorted = grid_bucket_downsample(features, target).sort().values
        new_data = self.data[keep_sorted]
        new_tags = self.tags[keep_sorted]
        self.data[:target] = new_data
        self.data[target:] = 0
        self.tags[:target] = new_tags
        self.tags[target:] = -1
        self._size = target
        self._ptr = target
        for cb in self._compact_callbacks:
            cb(keep_sorted)
        return keep_sorted
