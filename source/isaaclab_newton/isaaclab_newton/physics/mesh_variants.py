# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build Newton mesh variant sets from the clone plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp
from newton import Gaussian, Heightfield, Mesh, Model, ModelBuilder, ShapeFlags
from newton.solvers import SolverMuJoCo


@dataclass(frozen=True, slots=True)
class MeshVariantRenderShape:
    """One shape stored once in a mesh variant render bank."""

    geometry_type: int
    source: Mesh | Heightfield | Gaussian | None
    transform: tuple[float, ...]
    scale: tuple[float, float, float]
    thickness: float
    is_solid: bool
    color: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class MeshVariantRenderSet:
    """Render data coupled to one solver mesh variant set."""

    name: str
    body_indices: wp.array(dtype=wp.int32)
    model_shape_indices: tuple[int, ...]
    visual_variants: tuple[tuple[MeshVariantRenderShape, ...], ...]
    collision_variants: tuple[tuple[MeshVariantRenderShape, ...], ...]


@wp.kernel(enable_backward=False)
def _clear_shape_flags(shape_indices: wp.array(dtype=wp.int32), shape_flags: wp.array(dtype=wp.int32)):
    shape_flags[shape_indices[wp.tid()]] = 0


def _render_shapes(builder: ModelBuilder, flag: ShapeFlags) -> tuple[MeshVariantRenderShape, ...]:
    shapes = []
    for shape in builder.body_shapes[0]:
        flags = int(builder.shape_flags[shape])
        if not flags & int(flag) or flags & int(ShapeFlags.SITE):
            continue
        shapes.append(
            MeshVariantRenderShape(
                geometry_type=int(builder.shape_type[shape]),
                source=builder.shape_source[shape],
                transform=tuple(builder.shape_transform[shape]),
                scale=tuple(builder.shape_scale[shape]),
                thickness=float(builder.shape_margin[shape]),
                is_solid=bool(builder.shape_is_solid[shape]),
                color=tuple(builder.shape_color[shape]),
            )
        )
    return tuple(shapes)


def _prepare_mesh_variant_resources(
    registrations: dict[str, Any], builder: ModelBuilder, source_builders: dict[str, ModelBuilder], clone_plan: Any
) -> dict[str, tuple[tuple[int, ...], ...]]:
    """Add collision resources without adding parked rigid bodies."""
    if not registrations:
        return {}
    if clone_plan is None:
        raise ValueError("Runtime mesh variants require a heterogeneous ClonePlan.")

    resources = {}
    for name, cfg in registrations.items():
        rows = clone_plan.cfg_rows.get(id(cfg))
        if rows is None or len(rows) < 2:
            raise ValueError(f"Mesh variants for {name!r} are missing from the ClonePlan.")

        variants = []
        for row in rows:
            source_path = clone_plan.sources[row].rstrip("/") or "/"
            source = source_builders[source_path]
            if source.body_count != 1:
                raise ValueError(f"Mesh variant source {source_path!r} must contain one rigid body.")

            shapes = []
            for source_shape in source.body_shapes[0]:
                flags = int(source.shape_flags[source_shape])
                if not flags & int(ShapeFlags.COLLIDE_SHAPES):
                    continue
                shape_cfg = ModelBuilder.ShapeConfig(
                    density=0.0,
                    ke=source.shape_material_ke[source_shape],
                    kd=source.shape_material_kd[source_shape],
                    kf=source.shape_material_kf[source_shape],
                    ka=source.shape_material_ka[source_shape],
                    mu=source.shape_material_mu[source_shape],
                    restitution=source.shape_material_restitution[source_shape],
                    mu_torsional=source.shape_material_mu_torsional[source_shape],
                    mu_rolling=source.shape_material_mu_rolling[source_shape],
                    margin=source.shape_margin[source_shape],
                    gap=source.shape_gap[source_shape],
                    is_solid=source.shape_is_solid[source_shape],
                    collision_group=source.shape_collision_group[source_shape],
                    collision_filter_parent=False,
                    has_shape_collision=True,
                    has_particle_collision=bool(flags & int(ShapeFlags.COLLIDE_PARTICLES)),
                    is_visible=False,
                    is_hydroelastic=bool(flags & int(ShapeFlags.HYDROELASTIC)),
                    kh=source.shape_material_kh[source_shape],
                )
                shape = builder.add_shape(
                    body=-1,
                    type=source.shape_type[source_shape],
                    xform=source.shape_transform[source_shape],
                    cfg=shape_cfg,
                    scale=source.shape_scale[source_shape],
                    src=source.shape_source[source_shape],
                    is_static=True,
                    label=f"mesh_variant/{name}/{len(variants)}/{len(shapes)}",
                )
                for field in (
                    "shape_sdf_narrow_band_range",
                    "shape_sdf_target_voxel_size",
                    "shape_sdf_max_resolution",
                    "shape_force_sdf",
                    "shape_sdf_texture_format",
                    "shape_sdf_padding",
                ):
                    getattr(builder, field)[shape] = getattr(source, field)[source_shape]
                shapes.append(shape)
            variants.append(tuple(shapes))

        shape_counts = {len(shapes) for shapes in variants}
        if shape_counts == {0} or len(shape_counts) != 1:
            raise ValueError(f"Mesh variants for {name!r} require fixed, non-empty collision-shape topology.")
        resources[name] = tuple(variants)
    return resources


def _disable_mesh_variant_resources(model: Model, resources: dict[str, tuple[tuple[int, ...], ...]]) -> None:
    shape_indices = [shape for variants in resources.values() for shapes in variants for shape in shapes]
    if not shape_indices:
        return
    indices = wp.array(shape_indices, dtype=wp.int32, device=model.device)
    wp.launch(_clear_shape_flags, dim=len(shape_indices), inputs=[indices, model.shape_flags], device=model.device)


def build_mesh_variant_sets(
    registrations: dict[str, Any],
    model: Model,
    source_builders: dict[str, ModelBuilder],
    clone_plan: Any,
    visual_source_builders: dict[str, ModelBuilder] | None = None,
    resource_shape_indices: dict[str, tuple[tuple[int, ...], ...]] | None = None,
) -> tuple[tuple[SolverMuJoCo.MeshVariantSet, ...], tuple[MeshVariantRenderSet, ...]]:
    """Map registered rigid objects from clone-plan rows to Newton shapes."""
    if not registrations:
        return (), ()
    if clone_plan is None:
        raise ValueError("Runtime mesh variants require a heterogeneous ClonePlan.")

    clone_mask = clone_plan.clone_mask.detach().cpu().numpy()
    env_ids = (
        np.arange(clone_mask.shape[1]) if clone_plan.env_ids is None else clone_plan.env_ids.detach().cpu().numpy()
    )
    if clone_mask.shape[1] != model.world_count:
        raise ValueError("ClonePlan and Newton model world counts differ.")

    body_by_label = {label: index for index, label in enumerate(model.body_label)}
    shape_flags = model.shape_flags.numpy()
    definitions = []
    render_sets = []
    for name, cfg in registrations.items():
        rows = clone_plan.cfg_rows.get(id(cfg))
        if rows is None or len(rows) < 2:
            raise ValueError(f"Mesh variants for {name!r} are missing from the ClonePlan.")

        group_mask = clone_mask[np.asarray(rows)]
        if np.any(group_mask.sum(axis=0) != 1):
            raise ValueError(f"Mesh variants for {name!r} require one active source per world.")
        source_shapes = None if resource_shape_indices is None else resource_shape_indices.get(name)
        if source_shapes is None and np.any(group_mask.sum(axis=1) == 0):
            raise ValueError(f"Mesh variants for {name!r} require every source in at least one initial world.")
        if source_shapes is not None and len(source_shapes) != len(rows):
            raise ValueError(f"Mesh variant resources for {name!r} do not match the ClonePlan.")
        initial_variant_ids = group_mask.argmax(axis=0).astype(np.int32)

        builders = []
        body_suffixes = []
        for row in rows:
            source = clone_plan.sources[row].rstrip("/") or "/"
            builder = source_builders[source]
            if builder.body_count != 1:
                raise ValueError(f"Mesh variant source {source!r} must contain one rigid body.")
            body_label = builder.body_label[0]
            if body_label != source and not body_label.startswith(source + "/"):
                raise ValueError(f"Mesh variant body {body_label!r} is outside source {source!r}.")
            builders.append(builder)
            body_suffixes.append(body_label[len(source) :])

        body_indices = []
        shape_rows = []
        for world, variant in enumerate(initial_variant_ids):
            row = rows[variant]
            root = clone_plan.destinations[row].format(int(env_ids[world])).rstrip("/") or "/"
            body_label = root + body_suffixes[variant]
            try:
                body = body_by_label[body_label]
            except KeyError as error:
                raise ValueError(f"Mesh variant body {body_label!r} is missing from the Newton model.") from error
            body_indices.append(body)
            shapes = tuple(
                shape for shape in model.body_shapes[body] if int(shape_flags[shape]) & int(ShapeFlags.COLLIDE_SHAPES)
            )
            shape_rows.append(shapes)

        definitions.append(
            SolverMuJoCo.MeshVariantSet(
                name=name,
                shape_indices=shape_rows,
                variant_builders=builders,
                initial_variant_ids=initial_variant_ids,
                source_shape_indices=source_shapes,
                inertia_diagonal_offset=cfg.mesh_variant_inertia_diagonal_offset,
            )
        )
        if visual_source_builders:
            render_sets.append(
                MeshVariantRenderSet(
                    name=name,
                    body_indices=wp.array(body_indices, dtype=wp.int32, device=model.device),
                    model_shape_indices=tuple(shape for row in shape_rows for shape in row),
                    visual_variants=tuple(
                        _render_shapes(
                            visual_source_builders[clone_plan.sources[row].rstrip("/") or "/"], ShapeFlags.VISIBLE
                        )
                        for row in rows
                    ),
                    collision_variants=tuple(
                        _render_shapes(builder, ShapeFlags.COLLIDE_SHAPES) for builder in builders
                    ),
                )
            )

    return tuple(definitions), tuple(render_sets)
