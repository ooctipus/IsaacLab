# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``StateLayout``: spatial structure + index mapping of a curriculum's state pool.

Decoupled from the actual reset-state buffer (e.g. terrain's ``task_table``
or factory's :class:`StateBuffer`): this object describes where the pool's
states sit in coordinate space and how items (the units the curriculum
samples among) map onto those states. Signals consume it; consumers
translate item index -> full reset state in their own way.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StateLayout:
    """Spatial layout + index mapping of a curriculum's state pool.

    Attributes:
        coords: ``[num_states, coord_dim]`` spatial coordinates of pool
            states (``coord_dim=2`` for terrain xy, ``3`` for factory xyz,
            arbitrary positive D allowed).
        spawn_index: ``[num_items]`` long tensor; for each curriculum
            item, the state index used as the spawn endpoint.
        target_index: ``[num_items]`` long tensor of target endpoints,
            or ``None`` when items have no separate target endpoint
            (factory's slot==item case).
    """

    coords: torch.Tensor
    spawn_index: torch.Tensor
    target_index: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.coords.ndim != 2:
            raise ValueError(f"coords must have shape [num_states, coord_dim]; got shape {tuple(self.coords.shape)}.")
        if self.spawn_index.ndim != 1:
            raise ValueError(
                f"spawn_index must be 1D with shape [num_items]; got shape {tuple(self.spawn_index.shape)}."
            )
        if self.target_index is not None and self.target_index.shape != self.spawn_index.shape:
            raise ValueError(
                "spawn_index and target_index must have the same shape; "
                f"got {tuple(self.spawn_index.shape)} vs {tuple(self.target_index.shape)}."
            )

    @property
    def num_states(self) -> int:
        """Size of the underlying state pool."""
        return int(self.coords.shape[0])

    @property
    def num_items(self) -> int:
        """Number of items the curriculum picks among."""
        return int(self.spawn_index.shape[0])

    @property
    def coord_dim(self) -> int:
        """Dimensionality of :attr:`coords` (2 for xy, 3 for xyz, ...)."""
        return int(self.coords.shape[1])
