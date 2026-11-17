# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construction of Newton model builders: Isaac Lab defaults and USD helpers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp
from newton import Axis, GeoType, Heightfield, Mesh, Model, ModelBuilder, ShapeFlags
from newton.solvers import SolverMuJoCo
from newton.usd import SchemaResolver, SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd, UsdGeom

from isaaclab.utils import checked_apply

from .newton_collision_cfg import NewtonCollisionPipelineCfg
from .newton_manager_cfg import NewtonCfg, NewtonShapeCfg, NewtonSolverCfg

logger = logging.getLogger(__name__)


def inject_terrain_heightfields(
    stage: Usd.Stage, builder: ModelBuilder, *, root_paths: Sequence[str], device: str = "cpu"
) -> list[str]:
    """Replace height-field-tagged terrain colliders with Newton heightfields.

    Scans the stage for prims carrying the ``newton:heightfield:resolution`` attribute authored by
    :class:`~isaaclab.terrains.TerrainImporter`. Each collision mesh is rasterized into a
    :class:`newton.Heightfield` and added to ``builder`` as a static shape. Heightfields compile on MuJoCo roughly
    two orders of magnitude faster than the equivalent terrain mesh while colliding identically at the same
    horizontal resolution.

    Args:
        stage: The USD stage being imported.
        builder: The Newton model builder receiving the heightfield shapes.
        root_paths: Concrete subtree roots to scan.

    Returns:
        Prim paths of converted colliders, which the caller excludes from the USD import.
    """
    ignore_paths: list[str] = []
    xform_cache = UsdGeom.XformCache()
    for prim in (prim for root_path in root_paths for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path))):
        attr = prim.GetAttribute("newton:heightfield:resolution")
        if not attr or not attr.HasAuthoredValue():
            continue
        resolution = float(attr.Get())
        mesh_prim = (
            prim if prim.IsA(UsdGeom.Mesh) else next((p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)), None)
        )
        if mesh_prim is None:
            continue
        mesh = UsdGeom.Mesh(mesh_prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        # Transform vertices into world frame (USD uses row-vector convention).
        mat = np.array(xform_cache.GetLocalToWorldTransform(mesh_prim), dtype=np.float64).reshape(4, 4)
        world = (points @ mat[:3, :3] + mat[3, :3]).astype(np.float32)
        wp_mesh = wp.Mesh(
            points=wp.array(world, dtype=wp.vec3, device=device),
            indices=wp.array(faces, dtype=wp.int32, device=device),
        )
        heightfield, xform = Heightfield.create_from_mesh(wp_mesh, resolution)
        builder.add_shape_heightfield(heightfield=heightfield, xform=xform)
        logger.info(
            "Converted terrain collider %s (%d faces) to a %dx%d heightfield.",
            prim.GetPath().pathString,
            faces.shape[0] // 3,
            heightfield.nrow,
            heightfield.ncol,
        )
        ignore_paths.append(prim.GetPath().pathString)
    return ignore_paths


def register_builder_attributes(builder: ModelBuilder, cfg: NewtonSolverCfg | None) -> None:
    """Register the custom attributes the configured solver reads on a builder that receives imported assets."""
    for solver_cls in dict.fromkeys(cfg.class_type.solver_classes(cfg) if cfg is not None else ()):
        solver_cls.register_custom_attributes(builder)


def usd_schema_resolvers(cfg: NewtonSolverCfg | None) -> list[SchemaResolver]:
    """Return ordered schema resolvers for physics-model USD imports.

    MJC is enabled when the configured solver registers ``SolverMuJoCo`` attributes. Visualization and
    articulation-ordering builders keep their fixed pair because solver attributes do not affect their outputs.
    """
    resolvers: list[SchemaResolver] = [SchemaResolverNewton(), SchemaResolverPhysx()]
    if cfg is not None and any(
        issubclass(solver_cls, SolverMuJoCo) for solver_cls in cfg.class_type.solver_classes(cfg)
    ):
        resolvers.append(SchemaResolverMjc())
    return resolvers


def create_builder(physics_cfg: NewtonCfg | None = None, up_axis: str | None = None, **kwargs) -> ModelBuilder:
    """Create a :class:`ModelBuilder` configured with Isaac Lab's defaults.

    Forwards :class:`NewtonShapeCfg` defaults onto Newton's upstream ``ModelBuilder.default_shape_cfg`` via
    :func:`~isaaclab.utils.checked_apply`, and registers the custom attributes the configured solver reads. Without a
    Newton configuration the default shape settings still apply, so rough-terrain margin and gap hold during early
    construction.

    Args:
        physics_cfg: Newton physics configuration; ``None`` uses the default shape settings and no solver attributes.
        up_axis: Override for the up-axis. Defaults to ``"Z"``.
        **kwargs: Forwarded to :class:`ModelBuilder`.

    Returns:
        New builder with up-axis and per-shape defaults (gap, margin) applied.
    """
    newton_cfg = physics_cfg if isinstance(physics_cfg, NewtonCfg) else None
    builder = ModelBuilder(up_axis=up_axis or "Z", **kwargs)
    builder.default_bvh_cfg = ModelBuilder.BvhConfig(
        mesh_constructor=newton_cfg.bvh_constructor_geometry if newton_cfg else None,
        gaussian_constructor=newton_cfg.bvh_constructor_gaussian if newton_cfg else None,
        shape_constructor=newton_cfg.bvh_constructor_scene if newton_cfg else None,
        shape_flags=ShapeFlags.VISIBLE,
    )
    register_builder_attributes(builder, newton_cfg.solver_cfg if newton_cfg else None)
    checked_apply(newton_cfg.default_shape_cfg if newton_cfg else NewtonShapeCfg(), builder.default_shape_cfg)
    return builder


def _configure_sdf_all_shapes(
    builder: ModelBuilder, cfg: NewtonCollisionPipelineCfg.SDFAllShapesCfg
) -> tuple[int, ...]:
    """Provision every finite shape collider for the mesh-SDF narrow phase.

    This runs at the final builder boundary, after USD import has completed
    authored approximations such as ``convexHull``. Mesh and convex-mesh
    sources are therefore preserved, while finite planes and analytic
    primitives are tessellated in place. Existing body mass and inertia data
    are intentionally left unchanged.

    Args:
        builder: Fully assembled Newton model builder.
        cfg: Strict all-shapes SDF configuration.

    Returns:
        Shape indices that must have valid texture SDFs and collision edges
        after model finalization.

    Raises:
        ValueError: If a colliding shape cannot satisfy the strict SDF contract.
    """
    primitive_meshes: dict[tuple[Any, ...], Mesh] = {}
    sdf_shapes: list[int] = []
    target_voxel_size = cfg.sdf_target_voxel_size
    max_resolution = None if target_voxel_size is not None else cfg.sdf_max_resolution

    for shape_index, shape_type_value in enumerate(builder.shape_type):
        if not builder.shape_flags[shape_index] & ShapeFlags.COLLIDE_SHAPES:
            continue

        shape_type = GeoType(shape_type_value)
        shape_label = builder.shape_label[shape_index] or f"shape {shape_index}"
        shape_scale = tuple(float(value) for value in builder.shape_scale[shape_index])

        if shape_type in (GeoType.MESH, GeoType.CONVEX_MESH):
            if not isinstance(builder.shape_source[shape_index], Mesh):
                raise ValueError(
                    f"Strict all-shapes SDF provisioning requires {shape_label!r} to have a valid mesh source."
                )
        elif shape_type == GeoType.PLANE:
            width, length = shape_scale[:2]
            if width <= 0.0 or length <= 0.0:
                continue
            half_thickness = 0.5 * cfg.plane_thickness
            mesh_key = (shape_type, width, length, cfg.plane_thickness)
            mesh = primitive_meshes.get(mesh_key)
            if mesh is None:
                mesh = Mesh.create_box(
                    0.5 * width,
                    0.5 * length,
                    half_thickness,
                    duplicate_vertices=False,
                    compute_normals=False,
                    compute_uvs=False,
                    compute_inertia=False,
                )
                primitive_meshes[mesh_key] = mesh
            downward_shift = wp.transform(wp.vec3(0.0, 0.0, -half_thickness), wp.quat_identity())
            builder.shape_transform[shape_index] = wp.transform_multiply(
                builder.shape_transform[shape_index], downward_shift
            )
            builder.shape_source[shape_index] = mesh
            builder.shape_type[shape_index] = GeoType.MESH
            builder.shape_scale[shape_index] = wp.vec3(1.0)
        else:
            if not all(np.isfinite(value) for value in shape_scale):
                raise ValueError(f"Strict all-shapes SDF provisioning found non-finite scale on {shape_label!r}.")
            mesh_key = (shape_type, *shape_scale, cfg.primitive_mesh_segments)
            mesh = primitive_meshes.get(mesh_key)
            if mesh is None:
                if shape_type == GeoType.BOX and all(value > 0.0 for value in shape_scale):
                    mesh = Mesh.create_box(
                        *shape_scale,
                        duplicate_vertices=False,
                        compute_normals=False,
                        compute_uvs=False,
                        compute_inertia=False,
                    )
                elif shape_type == GeoType.SPHERE and shape_scale[0] > 0.0:
                    mesh = Mesh.create_sphere(
                        shape_scale[0],
                        num_latitudes=cfg.primitive_mesh_segments,
                        num_longitudes=cfg.primitive_mesh_segments,
                        compute_uvs=False,
                        compute_inertia=False,
                    )
                elif shape_type == GeoType.CAPSULE and shape_scale[0] > 0.0 and shape_scale[1] >= 0.0:
                    mesh = Mesh.create_capsule(
                        shape_scale[0],
                        shape_scale[1],
                        up_axis=Axis.Z,
                        segments=cfg.primitive_mesh_segments,
                        compute_uvs=False,
                        compute_inertia=False,
                    )
                elif shape_type == GeoType.ELLIPSOID and all(value > 0.0 for value in shape_scale):
                    mesh = Mesh.create_ellipsoid(
                        *shape_scale,
                        num_latitudes=cfg.primitive_mesh_segments,
                        num_longitudes=cfg.primitive_mesh_segments,
                        compute_uvs=False,
                        compute_inertia=False,
                    )
                elif shape_type == GeoType.CYLINDER and shape_scale[0] > 0.0 and shape_scale[1] > 0.0:
                    mesh = Mesh.create_cylinder(
                        shape_scale[0],
                        shape_scale[1],
                        up_axis=Axis.Z,
                        segments=cfg.primitive_mesh_segments,
                        compute_uvs=False,
                        compute_inertia=False,
                    )
                elif shape_type == GeoType.CONE and shape_scale[0] > 0.0 and shape_scale[1] > 0.0:
                    mesh = Mesh.create_cone(
                        shape_scale[0],
                        shape_scale[1],
                        up_axis=Axis.Z,
                        segments=cfg.primitive_mesh_segments,
                        compute_uvs=False,
                        compute_inertia=False,
                    )
                else:
                    raise ValueError(
                        f"Strict all-shapes SDF provisioning cannot convert {shape_label!r} "
                        f"with geometry type {shape_type.name} and scale {shape_scale}."
                    )
                primitive_meshes[mesh_key] = mesh
            builder.shape_source[shape_index] = mesh
            builder.shape_type[shape_index] = GeoType.MESH
            builder.shape_scale[shape_index] = wp.vec3(1.0)

        builder.shape_sdf_narrow_band_range[shape_index] = (
            cfg.sdf_narrow_band_inner,
            cfg.sdf_narrow_band_outer,
        )
        builder.shape_sdf_target_voxel_size[shape_index] = target_voxel_size
        builder.shape_sdf_max_resolution[shape_index] = max_resolution
        builder.shape_force_sdf[shape_index] = False
        builder.shape_sdf_texture_format[shape_index] = cfg.sdf_texture_format
        builder.shape_sdf_padding[shape_index] = cfg.sdf_padding
        sdf_shapes.append(shape_index)

    return tuple(sdf_shapes)


def _validate_sdf_all_shapes(model: Model, shape_indices: Sequence[int]) -> None:
    """Verify strict SDF shapes have the resources Newton's SDF pair route requires.

    Args:
        model: Finalized Newton model.
        shape_indices: Shape indices returned by :meth:`_configure_sdf_all_shapes`.

    Raises:
        RuntimeError: If texture construction failed or a shape has no collision edges.
    """
    if not shape_indices:
        return

    sdf_indices = model._shape_sdf_index.numpy()
    edge_ranges = model._shape_edge_range.numpy()
    textures = model._texture_sdf_coarse_textures
    failures: list[str] = []
    for shape_index in shape_indices:
        sdf_index = int(sdf_indices[shape_index])
        reasons: list[str] = []
        if sdf_index < 0 or sdf_index >= len(textures) or textures[sdf_index] is None:
            reasons.append("texture SDF was not built")
        if int(edge_ranges[shape_index][1]) <= 0:
            reasons.append("no collision edges were generated")
        if reasons:
            label = model.shape_label[shape_index] or f"shape {shape_index}"
            failures.append(f"{shape_index}:{label} ({'; '.join(reasons)})")
    if failures:
        raise RuntimeError(
            "Strict all-shapes SDF provisioning failed; refusing a non-SDF collision fallback: " + ", ".join(failures)
        )
