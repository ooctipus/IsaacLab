# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local indexing of Newton model frequencies and episode participation.

Selections store indices, never simulation state. Ordinary solver sleeping does not
change membership. Dense gathers are an explicit policy boundary; compact indices
and world offsets are available to Warp terms without a host synchronization.
"""

from __future__ import annotations

import re
from dataclasses import MISSING
from typing import Literal

import numpy as np
import torch
import warp as wp
from newton import Model

from isaaclab.utils import configclass

BODY = "body"
JOINT_COORD = "joint_coord"
JOINT_DOF = "joint_dof"


@configclass
class NewtonSelectorCfg:
    """Match full model labels, preserving pattern order and model order within each world.

    Overlapping patterns select an entity only once. Global entities (world -1)
    are excluded. ``count_per_world`` validates static topology, before masking.
    Joint patterns expand into all coordinates or DOFs of each matched joint.
    """

    frequency: Literal["body", "joint_coord", "joint_dof"] = MISSING
    path: str | tuple[str, ...] | list[str] = MISSING
    count_per_world: int | None = None


@wp.kernel
def _count_active(
    starts: wp.array[int],
    bodies: wp.array[int],
    body_active: wp.array[bool],
    world_active: wp.array[bool],
    counts: wp.array[int],
):
    world = wp.tid()
    count = int(0)
    if world_active[world]:
        for i in range(starts[world], starts[world + 1]):
            if body_active[bodies[i]]:
                count += 1
    counts[world] = count


@wp.kernel
def _compact_active(
    starts: wp.array[int],
    ids: wp.array[int],
    bodies: wp.array[int],
    body_active: wp.array[bool],
    world_active: wp.array[bool],
    world_start: wp.array[int],
    freq_ids: wp.array[int],
    env_ids: wp.array[int],
    slot_ids: wp.array[int],
    active: wp.array[bool],
):
    world = wp.tid()
    dst = world_start[world]
    for i in range(starts[world], starts[world + 1]):
        enabled = world_active[world] and body_active[bodies[i]]
        active[i] = enabled
        if enabled:
            freq_ids[dst] = ids[i]
            env_ids[dst] = world
            slot_ids[dst] = i - starts[world]
            dst += 1


class NewtonSelection:
    """Static model binding with a compact, episode-filtered device representation.

    Only entries before ``world_start[-1]`` in ``freq_ids/env_ids/slot_ids`` are
    valid. Storage and pointers remain stable across membership changes. Empty
    worlds have equal adjacent offsets. ``dense`` explicitly pads excluded slots.
    """

    # Runtime binding storage stays out of the manager's configuration serializer.
    # The instance dictionary contains only the original declarative selector.
    __slots__ = (
        "__dict__",
        "owner",
        "counts",
        "capacity",
        "ids",
        "body_ids",
        "starts",
        "freq_ids",
        "env_ids",
        "slot_ids",
        "world_start",
        "_counts",
        "active",
        "joint_ids",
        "_dense_width",
        "_dense_ids",
        "_dense_active",
    )

    def __init__(self, owner: NewtonSelections, cfg: NewtonSelectorCfg, rows: list[list[int]], bodies: list[int]):
        self.frequency = cfg.frequency
        self.path = cfg.path
        self.count_per_world = cfg.count_per_world
        self.owner = owner
        self.counts = tuple(map(len, rows))
        self.capacity = sum(self.counts)
        self.ids = wp.array([i for row in rows for i in row], dtype=wp.int32, device=owner.model.device)
        self.body_ids = wp.array(bodies, dtype=wp.int32, device=owner.model.device)
        self.starts = wp.array(np.cumsum([0, *self.counts]), dtype=wp.int32, device=owner.model.device)
        self.freq_ids = wp.empty_like(self.ids)
        self.env_ids = wp.empty_like(self.ids)
        self.slot_ids = wp.empty_like(self.ids)
        self.world_start = wp.zeros(owner.model.world_count + 1, dtype=wp.int32, device=owner.model.device)
        self._counts = wp.zeros_like(self.world_start)
        self.active = wp.zeros(self.capacity, dtype=wp.bool, device=owner.model.device)
        self.joint_ids: wp.array | None = None
        self._dense_width = self.counts[0] if self.counts and len(set(self.counts)) == 1 else None
        self._dense_ids = None
        self._dense_active = None
        if self._dense_width is not None:
            self._dense_ids = wp.to_torch(self.ids).reshape(len(self.counts), self._dense_width)
            self._dense_active = wp.to_torch(self.active).reshape(len(self.counts), self._dense_width)
        self.refresh()

    def refresh(self) -> None:
        """Rebuild membership on device without changing topology or allocating storage."""
        owner = self.owner
        wp.launch(
            _count_active,
            dim=len(self.counts),
            inputs=[self.starts, self.body_ids, owner.body_active, owner.world_active],
            outputs=[self._counts],
            device=owner.model.device,
        )
        wp.utils.array_scan(self._counts, self.world_start, inclusive=False)
        wp.launch(
            _compact_active,
            dim=len(self.counts),
            inputs=[self.starts, self.ids, self.body_ids, owner.body_active, owner.world_active, self.world_start],
            outputs=[self.freq_ids, self.env_ids, self.slot_ids, self.active],
            device=owner.model.device,
        )

    def __deepcopy__(self, memo):
        # Manager configs copy their parameters; runtime bindings belong to one env.
        memo[id(self)] = self
        return self

    def dense_ids(self) -> torch.Tensor:
        """Return static IDs at a uniform policy boundary; reject ragged topology."""
        if self._dense_ids is None:
            raise ValueError(
                "Dense selection requires equal static counts per world; use compact IDs for ragged terms."
            )
        return self._dense_ids

    def dense_active(self) -> torch.Tensor:
        """Return participation in stable per-world slot order."""
        if self._dense_active is None:
            raise ValueError("Dense selection requires equal static counts per world.")
        return self._dense_active

    def dense(self, values: wp.array, fill: float = 0.0) -> torch.Tensor:
        """Gather an array in this frequency and pad excluded slots with ``fill``."""
        selected = wp.to_torch(values)[self.dense_ids()]
        active = self.dense_active()
        while active.ndim < selected.ndim:
            active = active.unsqueeze(-1)
        return torch.where(active, selected, fill)


class NewtonSelections:
    """One task's model bindings and authoritative episode participation masks."""

    def __init__(self, model: Model):
        self.model = model
        self.body_active = wp.ones(model.body_count, dtype=wp.bool, device=model.device)
        self.world_active = wp.ones(model.world_count, dtype=wp.bool, device=model.device)
        self._body_world = model.body_world.numpy()
        self._joint_world = model.joint_world.numpy()
        self._joint_child = model.joint_child.numpy()
        self._q_start = model.joint_q_start.numpy()
        self._qd_start = model.joint_qd_start.numpy()
        self.root_joint_ids = torch.full((model.body_count,), -1, dtype=torch.int64, device=str(model.device))
        roots = np.flatnonzero(model.joint_parent.numpy() == -1)
        self.root_joint_ids[torch.as_tensor(self._joint_child[roots], device=str(model.device))] = torch.as_tensor(
            roots, device=str(model.device)
        )
        self._bindings: dict[tuple, NewtonSelection] = {}

    def resolve(self, cfg: NewtonSelectorCfg) -> NewtonSelection:
        """Resolve a declarative selector once against this finalized model."""
        patterns = (cfg.path,) if isinstance(cfg.path, str) else tuple(cfg.path)
        key = (cfg.frequency, patterns, cfg.count_per_world)
        if key in self._bindings:
            return self._bindings[key]
        if cfg.frequency not in (BODY, JOINT_COORD, JOINT_DOF):
            raise ValueError(f"Unknown Newton frequency: {cfg.frequency!r}")
        labels = self.model.body_label if cfg.frequency == BODY else self.model.joint_label
        worlds = self._body_world if cfg.frequency == BODY else self._joint_world
        rows = [[] for _ in range(self.model.world_count)]
        owners = [[] for _ in rows]
        seen = set()
        for pattern in patterns:
            regex = re.compile(pattern)
            matched = False
            for entity, (label, world) in enumerate(zip(labels, worlds, strict=True)):
                if world < 0 or not regex.fullmatch(label):
                    continue
                matched = True
                if entity in seen:
                    continue
                seen.add(entity)
                if cfg.frequency == BODY:
                    indices = [entity]
                    body = entity
                else:
                    starts = self._q_start if cfg.frequency == JOINT_COORD else self._qd_start
                    indices = range(int(starts[entity]), int(starts[entity + 1]))
                    body = int(self._joint_child[entity])
                rows[world].extend(indices)
                owners[world].extend([body] * len(indices))
            if not matched:
                raise ValueError(f"Selector {pattern!r} matched no {cfg.frequency} entities.")
        if cfg.count_per_world is not None and any(len(row) != cfg.count_per_world for row in rows):
            raise ValueError(
                f"Expected {cfg.count_per_world} {cfg.frequency} entries per world; got {list(map(len, rows))}."
            )
        selection = NewtonSelection(self, cfg, rows, [b for row in owners for b in row])
        if cfg.frequency != BODY:
            starts = self._q_start if cfg.frequency == JOINT_COORD else self._qd_start
            inverse = np.repeat(np.arange(self.model.joint_count), np.diff(starts))
            flat_ids = np.array([i for row in rows for i in row], dtype=np.int32)
            selection.joint_ids = wp.array(inverse[flat_ids], dtype=wp.int32, device=self.model.device)
        self._bindings[key] = selection
        return selection

    def refresh(self) -> None:
        """Refresh all bound selections after episode masks are changed at reset."""
        for selection in self._bindings.values():
            selection.refresh()
