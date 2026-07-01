# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from newton import ModelBuilder
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd

from isaaclab.sim.utils.transforms import resolve_prim_pose

from isaaclab_newton.cloner.newton_clone_utils import (
    add_usd_with_scoped_custom_frequencies,
    build_source_builders,
    rename_builder_labels,
    replicate_builder_mapping,
)


def build_visualization_builder_from_stage_envs(
    stage: Usd.Stage,
    env_paths: Sequence[tuple[int, str]],
    clone_plan: Any,
    *,
    up_axis: str = "Z",
) -> ModelBuilder:
    """Build the Newton shadow visualization builder from cloned USD environments."""
    env_path_by_id = dict(env_paths)

    sources = tuple(clone_plan.sources)
    destinations = tuple(clone_plan.destinations)
    env_ids = clone_plan.env_ids.detach().cpu()
    mapping = clone_plan.clone_mask.detach().cpu()

    poses = [resolve_prim_pose(stage.GetPrimAtPath(env_path_by_id[int(env_id)])) for env_id in env_ids.tolist()]
    positions = torch.tensor([pos for pos, _ in poses], dtype=torch.float32)
    quaternions = torch.tensor([quat for _, quat in poses], dtype=torch.float32)
    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]

    def import_usd(builder: ModelBuilder, stage: Usd.Stage, **kwargs: Any) -> Any:
        return add_usd_with_scoped_custom_frequencies(
            builder,
            stage,
            schema_resolvers=schema_resolvers,
            convert_mjc_equality_constraints=False,
            **kwargs,
        )

    builder = ModelBuilder(up_axis=up_axis)
    import_usd(
        builder,
        stage,
        ignore_paths=["/World/envs", *sources],
    )
    source_builders = build_source_builders(
        stage,
        sources,
        lambda: ModelBuilder(up_axis=up_axis),
        import_usd,
        simplify_meshes=False,
    )
    replicate_builder_mapping(builder, sources, mapping, positions, quaternions, source_builders)
    rename_builder_labels(builder, sources, destinations, env_ids, mapping)
    return builder
