# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local selection over Newton rigid bodies."""

from __future__ import annotations

import re
from dataclasses import MISSING, dataclass

from isaaclab.utils.configclass import configclass


@dataclass(frozen=True, slots=True)
class NewtonBodySelection:
    """Newton body indices arranged as ``[world, requested path]``."""

    ids: tuple[tuple[int, ...], ...]

    @property
    def shape(self) -> tuple[int, int]:
        """Selection shape as ``(num_worlds, bodies_per_world)``."""
        return len(self.ids), len(self.ids[0]) if self.ids else 0


@configclass
class NewtonBodySelectorCfg:
    """Select one Newton body per world for each ordered path regex."""

    path: str | tuple[str, ...] = MISSING

    def resolve(self, model) -> NewtonBodySelection:
        """Resolve body labels against a finalized Newton model."""
        patterns = (self.path,) if isinstance(self.path, str) else self.path
        compiled = tuple(re.compile(pattern) for pattern in patterns)
        world_count = int(model.world_count)
        ids = [[-1] * len(compiled) for _ in range(world_count)]
        body_world = model.body_world.numpy().tolist() if hasattr(model.body_world, "numpy") else model.body_world

        for body_id, (label, world) in enumerate(zip(model.body_label, body_world)):
            world = int(world)
            if world < 0:
                continue
            for column, pattern in enumerate(compiled):
                if not pattern.fullmatch(label):
                    continue
                if ids[world][column] >= 0:
                    raise ValueError(f"Body selector {patterns[column]!r} matched more than one body in world {world}.")
                ids[world][column] = body_id

        for world, row in enumerate(ids):
            for column, body in enumerate(row):
                if body < 0:
                    raise ValueError(f"Body selector {patterns[column]!r} matched no body in world {world}.")
        return NewtonBodySelection(tuple(tuple(row) for row in ids))
