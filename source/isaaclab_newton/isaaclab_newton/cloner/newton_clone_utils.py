# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import warp as wp
from newton import GeoType, JointType, Model, ModelBuilder, ShapeFlags

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab.cloner import path as clone_path


def _has_visible_non_collision_geometry(stage: Usd.Stage, prim_path: str) -> bool:
    """Return whether a prim hierarchy contains visible geometry without collision."""
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim:
        return False
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Gprim) or prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() != UsdGeom.Tokens.invisible and imageable.ComputePurpose() in (
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.proxy,
        ):
            return True
    return False


def _restore_visible_colliders_without_visual_shapes(
    builder: ModelBuilder,
    stage: Usd.Stage,
    path_shape_map: dict[str, int] | None,
    load_visual_shapes: bool = True,
) -> None:
    """Show viewport-visible colliders on bodies without separate visual shapes.

    Newton normally hides every collider when any visual-only shape exists in the
    imported model. Isaac Lab procedural shapes use one default-purpose USD geometry
    for both collision and visualization, so an unrelated visual asset must not hide
    them. Imported collision meshes, guide-purpose collision geometry, and
    colliders on bodies with separate visual shapes remain hidden.

    With ``load_visual_shapes=False`` the pass is skipped: Newton never hides a collider
    when the model holds no visual-only shapes, so every flag it would set is already set,
    and nothing draws them in a run that opted out of visual geometry. The skipped USD
    visibility/purpose resolution is per collider shape, so it is worth avoiding.
    """
    if not path_shape_map or not load_visual_shapes:
        return
    bodies_with_visual_shapes = {
        builder.shape_body[index]
        for index, flags in enumerate(builder.shape_flags)
        if builder.shape_body[index] >= 0 and flags & ShapeFlags.VISIBLE and not flags & ShapeFlags.COLLIDE_SHAPES
    }
    # Resolved on first use: a static parent whose colliders are all filtered out below is
    # never traversed at all.
    static_parents_with_visual_shapes: dict[str, bool] = {}
    for path, index in path_shape_map.items():
        flags = builder.shape_flags[index]
        body_index = builder.shape_body[index]
        if (
            not flags & ShapeFlags.COLLIDE_SHAPES
            or builder.shape_type[index] == GeoType.MESH
            or body_index in bodies_with_visual_shapes
        ):
            continue
        if body_index < 0:
            parent_path = path.rpartition("/")[0]
            if parent_path not in static_parents_with_visual_shapes:
                static_parents_with_visual_shapes[parent_path] = _has_visible_non_collision_geometry(stage, parent_path)
            if static_parents_with_visual_shapes[parent_path]:
                continue
        imageable = UsdGeom.Imageable(stage.GetPrimAtPath(path))
        if (
            imageable
            and imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
            and imageable.ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy)
        ):
            builder.shape_flags[index] = flags | ShapeFlags.VISIBLE


def build_source_builders(
    stage: Usd.Stage,
    sources: Sequence[str],
    create_builder: Callable[[], ModelBuilder],
    schema_resolvers: Sequence[Any],
    *,
    ignore_paths: Sequence[str] | None = None,
    load_visual_shapes: bool = True,
) -> dict[str, ModelBuilder]:
    """Build one Newton builder for each clone source prim path.

    The cloner approximates nothing. Collision geometry is whatever the asset authored:
    Newton's importer applies each shape's ``physics:approximation`` while importing, and
    USD defaults that token to ``none``, meaning "use the mesh as-is". Change it where it
    is authored -- the mesh-collision schema fragments on the spawner -- not here.

    Args:
        stage: USD stage containing the source prims.
        sources: Source prim paths to build a builder for.
        create_builder: Factory returning a fresh :class:`ModelBuilder`.
        schema_resolvers: Schema resolvers forwarded to Newton's USD importer.
        ignore_paths: Prim paths skipped during import.
        load_visual_shapes: Whether to import visual-only geometry. Importing it costs
            USD parse time and memory that only pays off when the shapes are rendered
            or ray cast.
    """
    ignored = tuple(ignore_paths or ())
    return {
        source: _build_source_builder(
            stage,
            source,
            create_builder,
            schema_resolvers,
            (
                *ignored,
                *(candidate for candidate in sources if candidate != source and candidate.startswith(source + "/")),
            ),
            load_visual_shapes,
        )
        for source in dict.fromkeys(sources)
    }


def _build_source_builder(
    stage: Usd.Stage,
    source: str,
    create_builder: Callable[[], ModelBuilder],
    schema_resolvers: Sequence[Any],
    ignore_paths: Sequence[str] | None,
    load_visual_shapes: bool = True,
) -> ModelBuilder:
    """Build one source builder."""
    builder = create_builder()
    import_result = builder.add_usd(
        stage,
        root_path=source,
        load_visual_shapes=load_visual_shapes,
        hide_collision_shapes=True,
        skip_mesh_approximation=False,
        schema_resolvers=schema_resolvers,
        ignore_paths=ignore_paths,
    )
    _restore_visible_colliders_without_visual_shapes(
        builder, stage, import_result["path_shape_map"], load_visual_shapes
    )
    if load_visual_shapes:
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="visual_material_path",
                namespace="isaaclab",
                dtype=str,
                frequency=Model.AttributeFrequency.SHAPE,
                default="",
            )
        )
        material_paths = builder.custom_attributes["isaaclab:visual_material_path"].values
        for shape_index, shape_path in enumerate(builder.shape_label):
            shape_prim = stage.GetPrimAtPath(shape_path)
            imageable = UsdGeom.Imageable(shape_prim)
            if not shape_prim.IsValid() or (imageable and imageable.ComputePurpose() == UsdGeom.Tokens.guide):
                continue
            _material, relationship = UsdShade.MaterialBindingAPI(shape_prim).ComputeBoundMaterial()
            if relationship and (targets := relationship.GetTargets()):
                material_paths[shape_index] = targets[0].pathString
    _name_root_joints_after_their_body(builder)
    return builder


def _name_root_joints_after_their_body(builder: ModelBuilder) -> None:
    """Name importer-generated free root joints after their child bodies, in place."""
    for index, label in enumerate(builder.joint_label):
        if not isinstance(label, str) or not label.startswith("joint_") or not label[6:].isdigit():
            continue
        if builder.joint_type[index] != JointType.FREE or builder.joint_parent[index] != -1:
            continue
        body_label = builder.body_label[builder.joint_child[index]]
        if isinstance(body_label, str) and body_label.startswith("/"):
            builder.joint_label[index] = f"{body_label}_free_joint"


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternion arrays, broadcast over the leading axes."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    out = np.empty(np.broadcast_shapes(a.shape, b.shape), dtype=np.float32)
    out[..., 0] = aw * bx + ax * bw + ay * bz - az * by
    out[..., 1] = aw * by - ax * bz + ay * bw + az * bx
    out[..., 2] = aw * bz + ax * by - ay * bx + az * bw
    out[..., 3] = aw * bw - ax * bx - ay * by - az * bz
    return out


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` by xyzw quaternions ``q``, broadcast over the leading axes."""
    axis, angle_w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(axis, v)
    return (v + angle_w * t + np.cross(axis, t)).astype(np.float32)


def _compose_world_xforms(world_p: np.ndarray, world_q: np.ndarray, local: Sequence[float]) -> np.ndarray:
    """``world_xform_w * local`` for every world, as one ``[num_worlds, 7]`` xyzw array."""
    local = np.asarray(local, dtype=np.float32)
    out = np.empty((world_p.shape[0], 7), dtype=np.float32)
    out[:, :3] = world_p + _quat_rotate(world_q, np.broadcast_to(local[:3], world_p.shape))
    out[:, 3:] = _quat_multiply(world_q, local[3:])
    return out


def _invert_xform(xform: Sequence[float] | np.ndarray) -> np.ndarray:
    """Inverse of a single xyzw transform, assuming a unit quaternion."""
    xform = np.asarray(xform, dtype=np.float32)
    quat_inv = np.array([-xform[0 + 3], -xform[1 + 3], -xform[2 + 3], xform[6]], dtype=np.float32)
    return np.concatenate([-_quat_rotate(quat_inv, xform[:3]), quat_inv])


def _transform_particles(builder: ModelBuilder, start: int, source: ModelBuilder, xforms: np.ndarray) -> None:
    """Apply rotations omitted by Newton's builder merge; it already applies translations."""
    if not source.particle_count or np.all(xforms[:, 3:] == (0.0, 0.0, 0.0, 1.0)):
        return
    points = np.broadcast_to(np.asarray(source.particle_q, dtype=np.float32), (len(xforms), source.particle_count, 3))
    points = _quat_rotate(xforms[:, None, 3:], points) + xforms[:, None, :3]
    builder.particle_q[start : start + points.size // 3] = points.reshape(-1, 3).tolist()


def _label_groups(builder: ModelBuilder) -> dict[str, list]:
    """Return every entity-label container owned by a Newton builder."""
    groups = {
        name: value for name, value in vars(builder).items() if name.endswith("_label") and isinstance(value, list)
    }
    groups["mujoco:equality_constraint_label"] = builder.custom_attributes["mujoco:equality_constraint_label"].values
    return groups


def _rebase_labels(builder: ModelBuilder, source: str, destination: str) -> str:
    """Make entity labels relative to the nearest templated destination ancestor."""
    source = source.rstrip("/") or "/"
    destination = destination.rstrip("/") or "/"
    prefix, _, destination_name = destination.rpartition("/")
    if "{}" in destination_name:
        prefix, destination_name = destination, ""
    for labels in _label_groups(builder).values():
        for index, label in enumerate(labels):
            if not isinstance(label, str) or not label or not label.startswith("/"):
                continue
            suffix = clone_path.relative_to(label, source)
            if suffix is None:
                suffix = label[len(source) :] if label.startswith(source + "_") else None
            if suffix is None:
                raise ValueError(f"Newton label {label!r} is outside clone source {source!r}.")
            labels[index] = (destination_name + suffix).lstrip("/")
            if not labels[index]:
                raise ValueError(f"Newton label {label!r} cannot be prefixed by destination {destination!r}.")
    return prefix


def replicate_builder_mapping(
    builder: ModelBuilder,
    sources: Sequence[str],
    mapping: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    source_builders: dict[str, ModelBuilder],
    destinations: Sequence[str] | None = None,
    env_ids: np.ndarray | None = None,
    *,
    source_site_indices: dict[int, dict[str, list[int]]] | None = None,
    env_root_sites: dict[str, wp.transform] | None = None,
    per_world_builder_hooks: Sequence[Callable[[ModelBuilder, int, np.ndarray, np.ndarray], None]] = (),
) -> tuple[dict[str, list[list[int]]], list[wp.transform], list[tuple[str, int]]]:
    """Replicate source builders, naming homogeneous copies at their destinations."""
    source_site_indices = source_site_indices or {}
    env_root_sites = env_root_sites or {}
    num_worlds = mapping.shape[1]
    local_site_map = {
        label: [indices.copy() for _ in range(num_worlds)]
        for label, indices in source_site_indices.get(id(builder), {}).items()
    }
    positions = positions.astype(np.float32, copy=False)
    quaternions = quaternions.astype(np.float32, copy=False)
    xforms_np = np.concatenate((positions, quaternions), axis=1)
    world_xforms = [wp.transform(*row) for row in xforms_np]

    can_batch = (
        len(sources) == 1
        and mapping.shape[0] == 1
        and num_worlds > 0
        and bool(mapping.all())
        and not per_world_builder_hooks
        and bool(destinations)
        and env_ids is not None
    )
    if can_batch:
        source_builder = source_builders[sources[0]]

        # Inject env-root sites into the source so replicate() copies them. Prefixed
        # by world_xforms[0] so R_w = world_xform_w * inv(world_xform_0) lands each
        # copy at world_xform_w * xform.
        site_local_indices: dict[str, list[int]] = {}
        for label, xform in env_root_sites.items():
            idx = source_builder.add_site(body=-1, xform=wp.transform_multiply(world_xforms[0], xform), label=label)
            site_local_indices.setdefault(label, []).append(idx)
        for label, indices in source_site_indices.get(id(source_builder), {}).items():
            site_local_indices.setdefault(label, []).extend(indices)

        # Site index after replicate: base_shape + world * stride + source_local_index.
        base_shape = builder.shape_count
        stride = source_builder.shape_count
        source_xform_inv = _invert_xform(xforms_np[0])
        xforms = _compose_world_xforms(positions, quaternions, source_xform_inv)

        label_groups = _label_groups(source_builder)
        original_labels = {name: list(labels) for name, labels in label_groups.items()}
        try:
            prefix = _rebase_labels(source_builder, sources[0], destinations[0])
            prefixes = [prefix.format(int(env_id)) for env_id in env_ids]
            particle_start = builder.particle_count
            builder.replicate(source_builder, num_worlds, xforms=xforms, label_prefixes=prefixes)
            _transform_particles(builder, particle_start, source_builder, xforms)
        finally:
            for name, labels in original_labels.items():
                label_groups[name][:] = labels

        for label, local_indices in site_local_indices.items():
            local_site_map[label] = [
                [base_shape + world * stride + local for local in local_indices] for world in range(num_worlds)
            ]

        bindings = rename_builder_labels(builder, sources, destinations, env_ids, mapping, skip_entity_labels=True)
        return local_site_map, world_xforms, bindings

    source_world_indices = mapping.argmax(axis=1)

    # Per-world placements for every env-root site, composed up front so the per-world loop
    # below only indexes rows.
    root_site_xforms = {
        label: _compose_world_xforms(positions, quaternions, xform) for label, xform in env_root_sites.items()
    }
    # Same for the source placements, but only for the occupied ``(row, col)`` pairs of the
    # mapping: composing a dense ``num_rows x num_worlds`` table would blow up on heterogeneous
    # plans where each row is present in a handful of worlds.
    # One scan of the transposed mapping yields the occupied pairs in world-major order, which
    # is the order both indices want.
    rows_per_world: list[list[int]] = [[] for _ in range(num_worlds)]
    worlds_per_row: dict[int, list[int]] = {}
    for col_value, row_value in np.argwhere(mapping.T):
        col, row = int(col_value), int(row_value)
        rows_per_world[col].append(row)
        worlds_per_row.setdefault(row, []).append(col)
    source_xforms: dict[tuple[int, int], np.ndarray] = {}
    for row, cols in worlds_per_row.items():
        source_col = int(source_world_indices[row])
        row_xforms = _compose_world_xforms(
            positions[cols],
            quaternions[cols],
            _invert_xform(xforms_np[source_col]),
        )
        source_xforms.update(((row, col), row_xforms[index]) for index, col in enumerate(cols))

    # A heterogeneous plan still draws from only a small set of row combinations. Assemble
    # each combination once, in plan order, so the per-world loop performs one native builder
    # merge instead of one merge per asset.
    combination_cols: dict[tuple[int, ...], list[int]] = {}
    for col, rows in enumerate(rows_per_world):
        combination_cols.setdefault(tuple(rows), []).append(col)
    native_blocks = not per_world_builder_hooks and all(
        len(cols) > 1 and cols[-1] - cols[0] + 1 == len(cols) for cols in combination_cols.values()
    )
    combination_builders: dict[tuple[int, ...], ModelBuilder] = {}
    combination_xforms: dict[tuple[int, ...], dict[int, np.ndarray]] = {}
    combination_sites: dict[tuple[int, ...], dict[str, list[int]]] = {}
    for rows, cols in combination_cols.items():
        if not native_blocks and len(rows) * len(cols) <= len(rows) + len(cols):
            continue
        representative = cols[0]
        prototype = ModelBuilder(up_axis=builder.up_axis)
        sites: dict[str, list[int]] = {}
        if native_blocks:
            for label, xform in env_root_sites.items():
                sites[label] = [
                    prototype.add_site(
                        body=-1, xform=wp.transform_multiply(world_xforms[representative], xform), label=label
                    )
                ]
        for row in rows:
            source_builder = source_builders[sources[row]]
            offset = prototype.shape_count
            particle_start = prototype.particle_count
            xform = source_xforms[row, representative]
            prototype.add_builder(source_builder, xform=xform)
            _transform_particles(prototype, particle_start, source_builder, xform[None])
            for label, source_shape_indices in source_site_indices.get(id(source_builder), {}).items():
                sites.setdefault(label, []).extend(offset + shape_idx for shape_idx in source_shape_indices)
        combination_builders[rows] = prototype
        xforms = _compose_world_xforms(positions[cols], quaternions[cols], _invert_xform(xforms_np[representative]))
        combination_xforms[rows] = dict(zip(cols, xforms, strict=True))
        combination_sites[rows] = sites

    if native_blocks:
        for rows, cols in combination_cols.items():
            prototype = combination_builders[rows]
            base_shape, stride = builder.shape_count, prototype.shape_count
            xforms = np.stack([combination_xforms[rows][col] for col in cols])
            xforms[0] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            particle_start = builder.particle_count
            builder.replicate(prototype, len(cols), xforms=xforms)
            _transform_particles(builder, particle_start, prototype, xforms)
            for label, local_indices in combination_sites[rows].items():
                sites = local_site_map.setdefault(label, [[] for _ in range(num_worlds)])
                for world, col in enumerate(cols):
                    sites[col] = [base_shape + world * stride + local for local in local_indices]
        bindings = rename_builder_labels(builder, sources, destinations, env_ids, mapping) if destinations else []
        return local_site_map, world_xforms, bindings

    for col in range(num_worlds):
        builder.begin_world()
        for label, world_site_xforms in root_site_xforms.items():
            site_idx = builder.add_site(body=-1, xform=world_site_xforms[col], label=label)
            local_site_map.setdefault(label, [[] for _ in range(num_worlds)])[col].append(site_idx)
        rows = tuple(rows_per_world[col])
        if rows in combination_builders:
            source_builder = combination_builders[rows]
            xform = combination_xforms[rows][col]
            offset = builder.shape_count
            particle_start = builder.particle_count
            builder.add_builder(source_builder, xform=xform)
            _transform_particles(builder, particle_start, source_builder, xform[None])
            for label, source_shape_indices in combination_sites[rows].items():
                local_indices = local_site_map.setdefault(label, [[] for _ in range(num_worlds)])[col]
                local_indices.extend(offset + shape_idx for shape_idx in source_shape_indices)
        else:
            for row in rows:
                source_builder = source_builders[sources[row]]
                offset = builder.shape_count
                particle_start = builder.particle_count
                builder.add_builder(source_builder, xform=source_xforms[row, col])
                _transform_particles(builder, particle_start, source_builder, source_xforms[row, col][None])
                for label, source_shape_indices in source_site_indices.get(id(source_builder), {}).items():
                    local_indices = local_site_map.setdefault(label, [[] for _ in range(num_worlds)])[col]
                    local_indices.extend(offset + shape_idx for shape_idx in source_shape_indices)
        for hook in per_world_builder_hooks:
            hook(builder, col, xforms_np[col, :3].copy(), xforms_np[col, 3:].copy())
        builder.end_world()

    bindings = rename_builder_labels(builder, sources, destinations, env_ids, mapping) if destinations else []
    return local_site_map, world_xforms, bindings


def rename_builder_labels(
    builder: ModelBuilder,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: np.ndarray,
    mapping: np.ndarray,
    *,
    skip_entity_labels: bool = False,
) -> list[tuple[str, int]]:
    """Rewrite source-root labels to per-env destination roots and return Fabric body bindings."""
    fabric_body_bindings: list[tuple[str, int]] = []
    bound_body_indices: set[int] = set()
    roots: dict[str, list[str | None]] = {}
    for source_index, (source, destination) in enumerate(zip(sources, destinations, strict=True)):
        source_root = source.rstrip("/") or "/"
        destination_root = destination
        matched = clone_path.match(source_root, destination)
        if "{}" in destination and matched is not None and not matched.suffix:
            prefix, _ = clone_path.split(destination)
            source_root = (prefix + matched.instance).rstrip("/") or "/"
            destination_root = prefix + "{}"
        world_roots = roots.setdefault(source_root, [None] * mapping.shape[1])
        for col in np.flatnonzero(mapping[source_index]):
            col = int(col)
            world_root = destination_root.format(int(env_ids[col])).rstrip("/") or "/"
            if world_roots[col] is not None and world_roots[col] != world_root:
                raise ValueError(f"Clone rows map {source_root!r} to conflicting roots in world {col}.")
            world_roots[col] = world_root

    matches: dict[str, tuple[tuple[int, list[str | None]], ...]] = {}

    def _rename_pair(values, worlds, *, collect_body_bindings=False):
        rows = (
            ((index, value, worlds[index]) for index, value in values.items())
            if isinstance(values, dict)
            else ((index, value, world) for index, (value, world) in enumerate(zip(values, worlds, strict=True)))
        )
        for index, value, world in rows:
            if not isinstance(value, str) or world is None or not 0 <= int(world) < mapping.shape[1]:
                continue
            candidates = matches.get(value)
            if candidates is None:
                path = value
                found = []
                while path:
                    if path in roots:
                        found.append((len(path), roots[path]))
                    path = "" if path == "/" else path.rpartition("/")[0]
                candidates = matches[value] = tuple(found)
            for source_root_length, world_roots in candidates:
                world_root = world_roots[int(world)]
                if world_root is None:
                    continue
                renamed_value = world_root + value[source_root_length:]
                if renamed_value != value:
                    values[index] = renamed_value
                    if collect_body_bindings:
                        fabric_body_bindings.append((renamed_value, index))
                        bound_body_indices.add(index)
                break

    if not skip_entity_labels:
        for name, labels in vars(builder).items():
            worlds = getattr(builder, f"{name[:-6]}_world", None) if name.endswith("_label") else None
            if isinstance(labels, list) and worlds is not None:
                _rename_pair(labels, worlds, collect_body_bindings=name == "body_label")

    custom_attrs = builder.custom_attributes.values()
    worlds_by_freq = {attr.frequency: attr.values for attr in custom_attrs if attr.references == "world"}
    for attr in custom_attrs:
        if attr.dtype is not str or not attr.values:
            continue
        if attr.namespace == "isaaclab" and attr.name == "visual_material_path":
            _rename_pair(attr.values, builder.shape_world)
        elif worlds := worlds_by_freq.get(attr.frequency):
            _rename_pair(attr.values, worlds)

    fabric_body_bindings.extend(
        (label, index) for index, label in enumerate(builder.body_label) if index not in bound_body_indices
    )
    return fabric_body_bindings
