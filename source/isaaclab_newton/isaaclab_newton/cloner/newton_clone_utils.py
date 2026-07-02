# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import torch
import warp as wp
from newton import ModelBuilder

from pxr import Sdf, Usd

from isaaclab.cloner.cloner_utils import replace_path_prefix
from isaaclab.sim.utils.newton_model_utils import replace_newton_builder_shape_colors
from isaaclab.sim.utils.transforms import resolve_prim_pose

from isaaclab_newton.sim.spawners.mjcf import (
    NEWTON_MJCF_ASSET_PATH_ATTR,
    NEWTON_MJCF_SELF_COLLISION_ATTR,
)


def add_usd_with_scoped_custom_frequencies(
    builder: ModelBuilder,
    stage: Usd.Stage,
    *,
    ignore_paths: Sequence[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Import USD while applying ``ignore_paths`` to custom-frequency filters.

    Newton's built-in USD traversal honors ``ignore_paths``, but registered
    custom-frequency filters visit the full stage. Compose each non-null filter
    with the same :func:`re.match` exclusions during import, then restore the
    original callback identity even when import fails.
    """
    ignored = tuple(ignore_paths or ())
    original_filters = []
    for frequency in getattr(builder, "custom_frequencies", {}).values():
        original = frequency.usd_prim_filter
        if original is None:
            continue

        def scoped_filter(prim, context, original=original):
            path = str(prim.GetPath())
            if any(re.match(pattern, path) for pattern in ignored):
                return False
            return original(prim, context)

        original_filters.append((frequency, original))
        frequency.usd_prim_filter = scoped_filter

    try:
        return builder.add_usd(stage, ignore_paths=ignore_paths, **kwargs)
    finally:
        for frequency, original in original_filters:
            frequency.usd_prim_filter = original


_NATIVE_MJCF_CUSTOM_LABELS = (
    "mujoco:equality_constraint_label",
    "mujoco:joint_dof_label",
    "mujoco:actuator_target_label",
    "mujoco:tendon_label",
)


def _prefix_label(root_path: str, label: str) -> str:
    """Place one non-empty Newton label below a USD marker path."""
    return f"{root_path.rstrip('/')}/{label.lstrip('/')}" if label else label


def _prefix_new_mjcf_labels(
    builder: ModelBuilder,
    root_path: str,
    builtin_starts: dict[str, int],
    custom_starts: dict[str, int | frozenset[object]],
) -> None:
    """Prefix labels appended by one native MJCF import with its marker path."""
    for kind, start in builtin_starts.items():
        labels = getattr(builder, f"{kind}_label")
        labels[start:] = [_prefix_label(root_path, label) for label in labels[start:]]

    for name in _NATIVE_MJCF_CUSTOM_LABELS:
        attribute = builder.custom_attributes.get(name)
        if attribute is None:
            continue
        values = attribute.values
        start = custom_starts.get(name, 0 if isinstance(values, list) else frozenset())
        if isinstance(values, list):
            if not isinstance(start, int):
                raise TypeError(f"Native MJCF attribute '{name}' changed storage type during import.")
            values[start:] = [_prefix_label(root_path, value) for value in values[start:]]
        elif isinstance(values, dict):
            if not isinstance(start, frozenset):
                raise TypeError(f"Native MJCF attribute '{name}' changed storage type during import.")
            for key in values.keys() - start:
                values[key] = _prefix_label(root_path, values[key])


def add_native_mjcf_from_stage(
    builder: ModelBuilder,
    stage: Usd.Stage,
    *,
    root_path: str | None = None,
    ignore_paths: Sequence[str] | None = None,
    load_visual_shapes: bool = True,
    skip_equality_constraints: bool = False,
    convert_mjc_equality_constraints: bool = True,
) -> tuple[str, ...]:
    """Import Newton-native MJCF markers from one USD stage subtree.

    Args:
        builder: Newton model builder receiving the parsed MJCF assets.
        stage: USD stage containing native MJCF marker prims.
        root_path: Optional subtree root. The full stage is searched when unset.
        ignore_paths: Optional regular expressions excluding marker paths.
        load_visual_shapes: Whether to parse MJCF visual geometry.
        skip_equality_constraints: Whether to omit MuJoCo equality metadata.
        convert_mjc_equality_constraints: Whether Newton converts MuJoCo
            equality metadata to generic constraints.

    Returns:
        Marker paths imported into ``builder``.
    """
    root = stage.GetPrimAtPath(root_path) if root_path is not None else stage.GetPseudoRoot()
    if not root.IsValid():
        return ()

    ignored = tuple(ignore_paths or ())
    imported: list[str] = []
    for prim in Usd.PrimRange(root):
        marker_path = str(prim.GetPath())
        if any(re.match(pattern, marker_path) for pattern in ignored):
            continue
        asset_attr = prim.GetAttribute(NEWTON_MJCF_ASSET_PATH_ATTR)
        if not asset_attr.IsValid() or not asset_attr.HasAuthoredValueOpinion():
            continue
        asset = asset_attr.Get()
        if not isinstance(asset, Sdf.AssetPath):
            raise TypeError(f"Native MJCF marker '{marker_path}' must store an Sdf.AssetPath.")
        asset_path = asset.resolvedPath or asset.path
        if not asset_path:
            raise ValueError(f"Native MJCF marker '{marker_path}' has an empty asset path.")
        self_collision_attr = prim.GetAttribute(NEWTON_MJCF_SELF_COLLISION_ATTR)
        if not self_collision_attr.IsValid() or not self_collision_attr.HasAuthoredValueOpinion():
            raise ValueError(f"Native MJCF marker '{marker_path}' has no self-collision policy.")

        builtin_starts = {
            kind: len(getattr(builder, f"{kind}_label"))
            for kind in _BUILTIN_LABEL_TYPES
            if kind != "equality_constraint" or "mujoco:equality_constraint_label" not in builder.custom_attributes
        }
        custom_starts: dict[str, int | frozenset[object]] = {}
        for name in _NATIVE_MJCF_CUSTOM_LABELS:
            attribute = builder.custom_attributes.get(name)
            if attribute is None:
                continue
            values = attribute.values
            if isinstance(values, list):
                custom_starts[name] = len(values)
            elif isinstance(values, dict):
                custom_starts[name] = frozenset(values)

        position, orientation = resolve_prim_pose(prim)
        builder.add_mjcf(
            asset_path,
            xform=wp.transform(position, orientation),
            up_axis=builder.up_axis,
            parse_visuals=load_visual_shapes,
            enable_self_collisions=bool(self_collision_attr.Get()),
            skip_equality_constraints=skip_equality_constraints,
            convert_mjc_equality_constraints=convert_mjc_equality_constraints,
        )
        _prefix_new_mjcf_labels(builder, marker_path, builtin_starts, custom_starts)
        imported.append(marker_path)
    return tuple(imported)


def build_source_builders(
    stage: Usd.Stage,
    sources: Sequence[str],
    create_builder: Callable[[], ModelBuilder],
    import_stage: Callable[..., Any],
    *,
    ignore_paths: Sequence[str] | None = None,
    simplify_meshes: bool = True,
) -> dict[str, ModelBuilder]:
    """Build one Newton builder for each clone source prim path."""
    builders: dict[str, ModelBuilder] = {}
    for source in sources:
        builder = create_builder()
        # When ``simplify_meshes`` is set, honor the per-shape ``physics:approximation``
        # token authored on each collider (e.g. via
        # ``CollisionPropertiesCfg(mesh_collision_property=NewtonMeshCollisionPropertiesCfg(...))``)
        # so callers can opt individual colliders into convex-hull / decomposition while
        # leaving SDF- and primitive-collider shapes untouched. ``import_stage`` skips shapes
        # carrying ``NewtonSDFCollisionAPI``, so this never clobbers SDF assets (nut/bolt).
        import_stage(
            builder,
            stage,
            root_path=source,
            load_visual_shapes=True,
            skip_mesh_approximation=not simplify_meshes,
            ignore_paths=ignore_paths,
        )
        if simplify_meshes:
            # Convex-hull only plain mesh colliders. Shapes carrying SDF/hydroelastic
            # state (e.g. an authored fine ``sdf`` thread sharing a body with a coarse
            # ``hull``) must be left untouched: ``approximate_meshes`` raises rather than
            # silently dropping their SDF when asked to remesh them.
            from newton import GeoType, ShapeFlags

            approx_shapes = [
                i
                for i, stype in enumerate(builder.shape_type)
                if stype == GeoType.MESH
                and builder.shape_flags[i] & ShapeFlags.COLLIDE_SHAPES
                and builder.shape_sdf_max_resolution[i] is None
                and builder.shape_sdf_target_voxel_size[i] is None
                and builder.shape_sdf_padding[i] is None
                and not (builder.shape_flags[i] & ShapeFlags.HYDROELASTIC)
            ]
            if approx_shapes:
                builder.approximate_meshes("convex_hull", shape_indices=approx_shapes, keep_visual_shapes=True)
        replace_newton_builder_shape_colors(builder, stage)
        builders[source] = builder
    return builders


def replicate_builder_mapping(
    builder: ModelBuilder,
    sources: Sequence[str],
    mapping: torch.Tensor,
    positions: torch.Tensor,
    quaternions: torch.Tensor,
    source_builders: dict[str, ModelBuilder],
    *,
    source_site_indices: dict[int, dict[str, list[int]]] | None = None,
    env_root_sites: dict[str, wp.transform] | None = None,
    per_world_builder_hooks: Sequence[Callable[[ModelBuilder, int, list[float], list[float]], None]] = (),
    post_replicate_hooks: Sequence[Callable[[ModelBuilder], None]] = (),
) -> tuple[dict[str, list[list[int]]], list[wp.transform]]:
    """Replicate source builders into per-env Newton worlds."""
    source_site_indices = source_site_indices or {}
    env_root_sites = env_root_sites or {}
    num_worlds = mapping.size(1)
    local_site_map: dict[str, list[list[int]]] = {}
    world_xforms: list[wp.transform] = []
    source_world_indices = mapping.to(dtype=torch.int64).argmax(dim=1)

    for col in range(num_worlds):
        builder.begin_world()
        world_xform = wp.transform(positions[col], quaternions[col])
        world_xforms.append(world_xform)

        for label, xform in env_root_sites.items():
            site_idx = builder.add_site(body=-1, xform=wp.transform_multiply(world_xform, xform), label=label)
            local_site_map.setdefault(label, [[] for _ in range(num_worlds)])[col].append(site_idx)

        for row in torch.nonzero(mapping[:, col], as_tuple=True)[0].tolist():
            source_builder = source_builders[sources[int(row)]]
            offset = builder.shape_count
            source_col = int(source_world_indices[int(row)])
            source_xform = wp.transform(positions[source_col], quaternions[source_col])
            builder.add_builder(
                source_builder, xform=wp.transform_multiply(world_xform, wp.transform_inverse(source_xform))
            )

            for label, source_shape_indices in source_site_indices.get(id(source_builder), {}).items():
                local_indices = local_site_map.setdefault(label, [[] for _ in range(num_worlds)])[col]
                local_indices.extend(offset + shape_idx for shape_idx in source_shape_indices)

        for hook in per_world_builder_hooks:
            hook(builder, col, positions[col].tolist(), quaternions[col].tolist())
        builder.end_world()

    for hook in post_replicate_hooks:
        hook(builder)
    return local_site_map, world_xforms


_BUILTIN_LABEL_TYPES: tuple[str, ...] = (
    "body",
    "joint",
    "shape",
    "articulation",
    "constraint_mimic",
    "equality_constraint",
)


def rename_builder_labels(
    builder: ModelBuilder,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
) -> list[tuple[str, int]]:
    """Rewrite source-root labels to per-env destination roots and return Fabric body bindings."""
    fabric_body_bindings: list[tuple[str, int]] = []
    bound_body_indices: set[int] = set()

    for source_index, source in enumerate(sources):
        source_root = source.rstrip("/")
        world_cols = torch.nonzero(mapping[source_index], as_tuple=True)[0].tolist()
        world_roots = {int(env_ids[col]): destinations[source_index].format(int(env_ids[col])) for col in world_cols}

        def _rename_pair(values, worlds, *, collect_body_bindings: bool = False):
            for index, (value, world) in enumerate(zip(values, worlds, strict=True)):
                if world is None:
                    continue
                world_root = world_roots.get(int(world))
                if isinstance(value, str) and world_root is not None:
                    renamed_value = replace_path_prefix(value, source_root, world_root)
                    if renamed_value != value:
                        values[index] = renamed_value
                        if collect_body_bindings:
                            fabric_body_bindings.append((renamed_value, index))
                            bound_body_indices.add(index)

        for labels, worlds, collect_body_bindings in (
            (builder.body_label, builder.body_world, True),
            (builder.joint_label, builder.joint_world, False),
            (builder.shape_label, builder.shape_world, False),
            (builder.articulation_label, builder.articulation_world, False),
            (builder.constraint_mimic_label, builder.constraint_mimic_world, False),
        ):
            _rename_pair(labels, worlds, collect_body_bindings=collect_body_bindings)

        if "mujoco:equality_constraint_label" not in builder.custom_attributes:
            _rename_pair(builder.equality_constraint_label, builder.equality_constraint_world)

        custom_attrs = builder.custom_attributes.values()
        worlds_by_freq = {attr.frequency: attr.values for attr in custom_attrs if attr.references == "world"}
        for attr in custom_attrs:
            if attr.dtype is str and attr.values and (worlds := worlds_by_freq.get(attr.frequency)):
                _rename_pair(attr.values, worlds)

    fabric_body_bindings.extend(
        (label, index) for index, label in enumerate(builder.body_label) if index not in bound_body_indices
    )
    return fabric_body_bindings
