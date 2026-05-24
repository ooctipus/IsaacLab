# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for :class:`~isaaclab.layout.StageLayout`.

Three free functions, no USD imports:

- :func:`make_stage_layout` constructs a :class:`StageLayout` from per-world occupancy lists.
- :func:`first_world_of` finds the first world a given source appears in.
- :func:`world_ids_of` enumerates the per-slot world ids a given source occupies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .stage_layout import StageLayout


def make_stage_layout(
    sources: Sequence[Any],
    destinations: Sequence[str],
    sources_per_world: Sequence[Sequence[int]],
    env_pose: torch.Tensor,
    *,
    destinations_per_world: Sequence[Sequence[int]] | None = None,
) -> StageLayout:
    """Build a :class:`StageLayout` from per-world occupancy lists.

    The occupancy is provided as one list of source-indices per world. World ``0`` is the
    shared scope; world ``1 + i`` is env ``i``. Total length must be ``num_envs + 1``,
    where ``num_envs == env_pose.shape[0]``.

    Args:
        sources: Unique source registry. ``sources_per_world`` entries are indices into this.
        destinations: Unique destination-template registry. ``destinations_per_world``
            entries are indices into this. Templates use ``"{}"`` as the env-id placeholder.
        sources_per_world: Per-world list of source indices.
            ``sources_per_world[0]`` is shared scope; ``sources_per_world[1 + i]`` is env ``i``.
        env_pose: Per-env pose tensor of shape ``[num_envs, 7]``; columns are
            ``[x, y, z, qx, qy, qz, qw]``.
        destinations_per_world: Optional parallel destination-template indices. When omitted,
            defaults to ``sources_per_world`` (1-to-1 source ``\u2192`` destination).

    Returns:
        :class:`StageLayout` with CSR row offsets built from the occupancy lists.

    Raises:
        ValueError: If lengths or shapes are inconsistent.
    """
    num_envs = int(env_pose.shape[0])
    if env_pose.ndim != 2 or env_pose.shape[1] != 7:
        raise ValueError(f"env_pose must have shape [num_envs, 7]; got {tuple(env_pose.shape)}.")
    if len(sources_per_world) != num_envs + 1:
        raise ValueError(
            f"sources_per_world must have length num_envs + 1 = {num_envs + 1}; got {len(sources_per_world)}."
        )

    if destinations_per_world is None:
        destinations_per_world = sources_per_world
    elif len(destinations_per_world) != len(sources_per_world):
        raise ValueError(
            "destinations_per_world must be parallel to sources_per_world "
            f"(lengths {len(destinations_per_world)} vs {len(sources_per_world)})."
        )

    flat_sources: list[int] = []
    flat_destinations: list[int] = []
    # world_start carries num_envs + 2 entries: a leading 0, plus one offset after each
    # of the num_envs + 1 worlds (shared scope at index 0, env i at index 1 + i).
    starts = [0]
    for srcs, dests in zip(sources_per_world, destinations_per_world, strict=True):
        if len(srcs) != len(dests):
            raise ValueError(
                "sources_per_world and destinations_per_world must agree slot-for-slot per world "
                f"(got {len(srcs)} source slots vs {len(dests)} destination slots)."
            )
        flat_sources.extend(srcs)
        flat_destinations.extend(dests)
        starts.append(len(flat_sources))

    n_sources = len(sources)
    n_dests = len(destinations)
    if any(s < 0 or s >= n_sources for s in flat_sources):
        raise ValueError("sources_per_world contains out-of-range source indices.")
    if any(d < 0 or d >= n_dests for d in flat_destinations):
        raise ValueError("destinations_per_world contains out-of-range destination indices.")

    return StageLayout(
        sources=tuple(sources),
        destinations=tuple(destinations),
        source_ids=torch.tensor(flat_sources, dtype=torch.long, device=env_pose.device),
        destination_ids=torch.tensor(flat_destinations, dtype=torch.long, device=env_pose.device),
        world_start=torch.tensor(starts, dtype=torch.long, device=env_pose.device),
        env_pose=env_pose,
    )


def first_world_of(layout: StageLayout, source: Any) -> int | None:
    """Return the first world id where ``source`` appears.

    Args:
        layout: Layout to query.
        source: Source to look up. Identity-keyed (``source is source``).

    Returns:
        ``None`` if ``source`` is not in :attr:`StageLayout.sources` or owns no slots.
        ``-1`` if the first appearance is in shared scope (world ``0``). Otherwise an
        env id in ``[0, num_envs - 1]`` (the first env that owns ``source``).
    """
    source_idx = _source_index(layout, source)
    if source_idx is None:
        return None
    matches = (layout.source_ids == source_idx).nonzero(as_tuple=False).flatten()
    if matches.numel() == 0:
        return None
    first_slot = int(matches[0])
    # ``world_start`` has shape ``[num_envs + 2]``. ``searchsorted(..., right=True)`` returns
    # the smallest index ``j`` such that ``world_start[j] > first_slot``; the world that owns
    # the slot is ``j - 1`` in CSR space, and the env id is ``j - 2`` (subtracting both the
    # shared-scope row at index 0 and the trailing tombstone offset).
    j = int(torch.searchsorted(layout.world_start, torch.tensor(first_slot, dtype=torch.long), right=True))
    return j - 2


def world_ids_of(layout: StageLayout, source: Any) -> torch.Tensor:
    """Return per-slot world ids for every slot occupied by ``source``, in slot order.

    Args:
        layout: Layout to query.
        source: Source to look up. Identity-keyed.

    Returns:
        Long tensor; each entry is ``-1`` (shared scope) or in ``[0, num_envs - 1]``
        (env id). Cardinality > 1 in the same world yields repeated entries.
        Empty tensor if ``source`` is not in :attr:`StageLayout.sources` or owns no slots.
    """
    source_idx = _source_index(layout, source)
    if source_idx is None:
        return torch.empty(0, dtype=torch.long, device=layout.source_ids.device)
    slots = (layout.source_ids == source_idx).nonzero(as_tuple=False).flatten()
    if slots.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=layout.source_ids.device)
    return torch.searchsorted(layout.world_start, slots, right=True) - 2


def _source_index(layout: StageLayout, source: Any) -> int | None:
    """Return the index of ``source`` in ``layout.sources`` (identity), or ``None`` if absent."""
    for i, registered in enumerate(layout.sources):
        if registered is source:
            return i
    return None
