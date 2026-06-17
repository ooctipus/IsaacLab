# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-replicate collision pairing for the Newton model builder.

Compiles a :class:`~isaaclab_newton.physics.NewtonCollisionPairingCfg` into
Newton ``shape_collision_group`` / ``shape_collision_filter_pairs`` and an
optional SDF-on-convex resolution pass, in a single pass over the finalized
(pre-rename) model builder. See the cfg for the data contract.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from newton import ModelBuilder

# Base id for private mate groups. Newton treats a positive collision group as
# private (its members collide only within the group); env-level isolation is
# handled separately by the per-world filter, so a shared id across envs is fine.
_MATE_GROUP_BASE = 1000


def build_collision_pairing_hook(cfg) -> Callable[[ModelBuilder], None]:
    """Build a post-replicate hook that applies ``cfg`` to a model builder.

    Args:
        cfg: The :class:`~isaaclab_newton.physics.NewtonCollisionPairingCfg`.

    Returns:
        A callable taking the model builder, run once after replication and
        before per-env label renaming.
    """
    from newton._src.geometry.types import GeoType

    mate = [(re.compile(a), re.compile(b)) for a, b in cfg.mate]
    forbid = [(re.compile(a), re.compile(b)) for a, b in cfg.forbid]
    resolution = cfg.convex_sdf_resolution

    def hook(builder: ModelBuilder) -> None:
        labels = builder.shape_label
        worlds = builder.shape_world

        # mate: every a/b match for entry k joins private group _MATE_GROUP_BASE + k.
        for k, (pat_a, pat_b) in enumerate(mate):
            group = _MATE_GROUP_BASE + k
            for i, label in enumerate(labels):
                if label and (pat_a.search(label) or pat_b.search(label)):
                    builder.shape_collision_group[i] = group

        # forbid: filter every a/b pair that shares a world.
        for pat_a, pat_b in forbid:
            a_by_world: dict[int, list[int]] = {}
            b_by_world: dict[int, list[int]] = {}
            for i, label in enumerate(labels):
                if not label:
                    continue
                if pat_a.search(label):
                    a_by_world.setdefault(worlds[i], []).append(i)
                if pat_b.search(label):
                    b_by_world.setdefault(worlds[i], []).append(i)
            for world, a_ids in a_by_world.items():
                for a_id in a_ids:
                    for b_id in b_by_world.get(world, ()):
                        builder.add_shape_collision_filter_pair(a_id, b_id)

        # SDF-on-convex: route convex hulls (without an authored SDF) and boxes
        # through the planar-SDF kernel. The deferred build at finalize dedupes
        # identical meshes by source, so setting this per-shape is not per-env work.
        if resolution is not None:
            for i, shape_type in enumerate(builder.shape_type):
                if shape_type == GeoType.CONVEX_MESH:
                    source = builder.shape_source[i]
                    if source is not None and getattr(source, "sdf", None) is None:
                        builder.shape_sdf_max_resolution[i] = resolution
                elif shape_type == GeoType.BOX:
                    builder.shape_sdf_max_resolution[i] = resolution

    return hook
