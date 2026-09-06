# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-free collider geometry shared by kinematic task-table builders."""

from __future__ import annotations

import newton
import numpy as np
from newton import GeoType

from .newton_asset import resolve_newton_asset_path

_BOX_VERTICES = np.array(
    [[x, y, z] for z in (-1.0, 1.0) for y in (-1.0, 1.0) for x in (-1.0, 1.0)],
    dtype=np.float32,
)
_BOX_FACES = np.array(
    [
        [0, 2, 3],
        [0, 3, 1],
        [4, 5, 7],
        [4, 7, 6],
        [0, 1, 5],
        [0, 5, 4],
        [2, 6, 7],
        [2, 7, 3],
        [0, 4, 6],
        [0, 6, 2],
        [1, 3, 7],
        [1, 7, 5],
    ],
    dtype=np.int32,
)


def _collision_shape_flags(builder: newton.ModelBuilder) -> list[int]:
    """Return one required collision flag for every builder shape."""
    if not hasattr(builder, "shape_flags") or len(builder.shape_flags) != len(builder.shape_body):
        raise ValueError("Newton collider geometry requires one shape_flags entry per shape.")
    return builder.shape_flags


def _quaternion_rotate_xyzw(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    vector = quaternion[:3]
    cross = 2.0 * np.cross(np.broadcast_to(vector, vectors.shape), vectors)
    return vectors + quaternion[3] * cross + np.cross(np.broadcast_to(vector, vectors.shape), cross)


def points_transform_xyzw(
    points: np.ndarray,
    position: np.ndarray,
    quaternion: np.ndarray,
) -> np.ndarray:
    """Transform points [m] by one xyzw pose."""
    return _quaternion_rotate_xyzw(quaternion, points) + position


def _collider_mesh_vertices(source, scale) -> np.ndarray:
    """Return Newton mesh vertices after applying the signed shape scale [m]."""
    vertices = np.asarray(source.vertices, dtype=np.float32).reshape(-1, 3)
    scale = np.asarray(scale, dtype=np.float32)
    if not vertices.shape[0]:
        raise ValueError("Collision mesh must contain at least one vertex.")
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale == 0.0):
        raise ValueError("Collision mesh scale must contain three finite nonzero components.")
    return vertices * scale


def _collider_mesh_faces(source, scale, *, dtype: np.dtype = np.dtype(np.int32)) -> np.ndarray:
    """Return Newton mesh triangles with outward winding after signed shape scale."""
    scale = np.asarray(scale, dtype=np.float32)
    faces = np.asarray(source.indices, dtype=dtype).reshape(-1, 3)
    return faces[:, ::-1].copy() if float(np.prod(scale)) < 0.0 else faces


def _collider_z_min(shape_type: int, scale, transform, source) -> float:
    """Return one shape's lowest point in its owning body frame [m]."""
    transform = np.asarray(transform, dtype=np.float32)
    position = transform[:3]
    quaternion = transform[3:7]
    scale = np.asarray(scale, dtype=np.float32)
    if source is not None and hasattr(source, "vertices"):
        vertices = _collider_mesh_vertices(source, scale)
        return float(points_transform_xyzw(vertices, position, quaternion)[:, 2].min())
    if shape_type == int(GeoType.SPHERE):
        return float(position[2] - scale[0])
    if shape_type == int(GeoType.BOX):
        vertices = _BOX_VERTICES * scale[:3]
        return float(points_transform_xyzw(vertices, position, quaternion)[:, 2].min())

    axis = _quaternion_rotate_xyzw(
        quaternion,
        np.array(((0.0, 0.0, 1.0),), dtype=np.float32),
    )[0]
    axis_z = float(np.clip(axis[2], -1.0, 1.0))
    if shape_type == int(GeoType.CAPSULE):
        return float(position[2] - abs(axis_z) * scale[1] - scale[0])
    if shape_type == int(GeoType.CYLINDER):
        radial_z = np.sqrt(max(0.0, 1.0 - axis_z * axis_z))
        return float(position[2] - abs(axis_z) * scale[1] - radial_z * scale[0])
    raise ValueError(f"Unsupported collision shape {GeoType(shape_type).name}.")


def model_body_collider_z_min(builder: newton.ModelBuilder, body_indices: tuple[int, ...]) -> np.ndarray:
    """Return the lowest body-frame collider point for selected bodies [m].

    Args:
        builder: Parsed Newton model builder containing body-local shapes.
        body_indices: Body indices in requested output order.

    Returns:
        Lowest collider z coordinate per body, shape ``[body_count]``.
    """
    requested = {body: index for index, body in enumerate(body_indices)}
    z_min = np.full(len(body_indices), np.inf, dtype=np.float32)
    shape_flags = _collision_shape_flags(builder)
    collision_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
    for shape_index, body_index in enumerate(builder.shape_body):
        output_index = requested.get(int(body_index))
        if output_index is None:
            continue
        if not int(shape_flags[shape_index]) & collision_flag:
            continue
        z_min[output_index] = min(
            z_min[output_index],
            _collider_z_min(
                int(builder.shape_type[shape_index]),
                builder.shape_scale[shape_index],
                builder.shape_transform[shape_index],
                builder.shape_source[shape_index],
            ),
        )
    if not np.isfinite(z_min).all():
        missing = [body_indices[index] for index in np.flatnonzero(~np.isfinite(z_min))]
        raise ValueError(f"Bodies have no supported collision geometry: {missing}.")
    return z_min


def _collider_extreme_point(shape_type: int, scale, transform, source, direction_body: np.ndarray) -> np.ndarray:
    """Return one body-local collider point farthest along a body-local direction [m]."""
    transform = np.asarray(transform, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    direction_body = np.asarray(direction_body, dtype=np.float32)
    direction_body = direction_body / np.linalg.norm(direction_body)
    if source is not None and hasattr(source, "vertices"):
        points = _collider_mesh_vertices(source, scale)
        points = points_transform_xyzw(points, transform[:3], transform[3:7])
        return points[int(np.argmax(points @ direction_body))]
    if shape_type == int(GeoType.BOX):
        points = points_transform_xyzw(_BOX_VERTICES * scale[:3], transform[:3], transform[3:7])
        return points[int(np.argmax(points @ direction_body))]
    shape_conjugate = np.concatenate((-transform[3:6], transform[6:7]))
    direction_shape = _quaternion_rotate_xyzw(shape_conjugate, direction_body[None])[0]
    if shape_type == int(GeoType.SPHERE):
        point_shape = scale[0] * direction_shape
    elif shape_type == int(GeoType.CAPSULE):
        point_shape = scale[0] * direction_shape
        point_shape[2] += np.copysign(scale[1], direction_shape[2])
    elif shape_type == int(GeoType.CYLINDER):
        point_shape = np.zeros(3, dtype=np.float32)
        radial_norm = np.linalg.norm(direction_shape[:2])
        if radial_norm > 1.0e-8:
            point_shape[:2] = scale[0] * direction_shape[:2] / radial_norm
        point_shape[2] = np.copysign(scale[1], direction_shape[2])
    else:
        raise ValueError(f"Unsupported collision shape {GeoType(shape_type).name}.")
    return points_transform_xyzw(point_shape[None], transform[:3], transform[3:7])[0]


def _collider_support_points_along_up(
    builder: newton.ModelBuilder,
    body_index: int,
    body_pose: np.ndarray,
    support_up_world: np.ndarray,
    points_per_body: int,
    height_band_m: float,
) -> np.ndarray:
    """Return one planar-support candidate set for a declared world-up direction [m]."""
    support_up_world = support_up_world / np.linalg.norm(support_up_world)
    body_conjugate = np.concatenate((-body_pose[3:6], body_pose[6:7]))
    support_up_body = _quaternion_rotate_xyzw(body_conjugate, support_up_world[None])[0]
    collision_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
    shape_flags = _collision_shape_flags(builder)
    groups: list[tuple[float, float, np.ndarray]] = []
    for shape_index, shape_body in enumerate(builder.shape_body):
        if int(shape_body) != body_index or not int(shape_flags[shape_index]) & collision_flag:
            continue
        shape_type = int(builder.shape_type[shape_index])
        scale = np.asarray(builder.shape_scale[shape_index], dtype=np.float32)
        transform = np.asarray(builder.shape_transform[shape_index], dtype=np.float32)
        source = builder.shape_source[shape_index]
        if source is not None and hasattr(source, "vertices") and hasattr(source, "indices"):
            vertices_body = points_transform_xyzw(_collider_mesh_vertices(source, scale), transform[:3], transform[3:7])
            triangles = _collider_mesh_faces(source, scale, dtype=np.dtype(np.int64))
            vertices_world = points_transform_xyzw(vertices_body, body_pose[:3], body_pose[3:7])
            edges_a = vertices_world[triangles[:, 1]] - vertices_world[triangles[:, 0]]
            edges_b = vertices_world[triangles[:, 2]] - vertices_world[triangles[:, 0]]
            normals = np.cross(edges_a, edges_b)
            lengths = np.linalg.norm(normals, axis=1)
            valid = lengths > 1.0e-10
            if not np.any(valid):
                continue
            normals[valid] /= lengths[valid, None]
            alignment = normals @ -support_up_world
            alignment[~valid] = -np.inf
            for triangle_index in np.flatnonzero(valid):
                points_body = vertices_body[triangles[triangle_index]]
                support_height = float(np.min(vertices_world[triangles[triangle_index]] @ support_up_world))
                groups.append((float(alignment[triangle_index]), support_height, points_body))
            continue
        if shape_type == int(GeoType.BOX):
            shape_conjugate = np.concatenate((-transform[3:6], transform[6:7]))
            support_up_shape = _quaternion_rotate_xyzw(shape_conjugate, support_up_body[None])[0]
            axis = int(np.argmax(np.abs(support_up_shape)))
            face_sign = -1.0 if support_up_shape[axis] >= 0.0 else 1.0
            face = _BOX_VERTICES[_BOX_VERTICES[:, axis] == face_sign] * scale[:3]
            points_body = points_transform_xyzw(face, transform[:3], transform[3:7])
            normal_shape = np.zeros((1, 3), dtype=np.float32)
            normal_shape[0, axis] = face_sign
            normal_body = _quaternion_rotate_xyzw(transform[3:7], normal_shape)[0]
            normal_world = _quaternion_rotate_xyzw(body_pose[3:7], normal_body[None])[0]
            best_alignment = float(normal_world @ -support_up_world)
        elif shape_type == int(GeoType.CYLINDER):
            shape_conjugate = np.concatenate((-transform[3:6], transform[6:7]))
            support_up_shape = _quaternion_rotate_xyzw(shape_conjugate, support_up_body[None])[0]
            face_sign = -1.0 if support_up_shape[2] >= 0.0 else 1.0
            angles = 2.0 * np.pi * np.arange(max(3, points_per_body), dtype=np.float32) / max(3, points_per_body)
            face = np.stack(
                (
                    scale[0] * np.cos(angles),
                    scale[0] * np.sin(angles),
                    np.full_like(angles, face_sign * scale[1]),
                ),
                axis=-1,
            )
            points_body = points_transform_xyzw(face, transform[:3], transform[3:7])
            normal_shape = np.array(((0.0, 0.0, face_sign),), dtype=np.float32)
            normal_body = _quaternion_rotate_xyzw(transform[3:7], normal_shape)[0]
            normal_world = _quaternion_rotate_xyzw(body_pose[3:7], normal_body[None])[0]
            best_alignment = float(normal_world @ -support_up_world)
        else:
            points_body = _collider_extreme_point(shape_type, scale, transform, source, -support_up_body)[None]
            best_alignment = 1.0
        if not np.isfinite(points_body).all() or not points_body.shape[0]:
            continue
        points_world = points_transform_xyzw(points_body, body_pose[:3], body_pose[3:7])
        support_height = float(np.min(points_world @ support_up_world))
        groups.append((best_alignment, support_height, points_body))
    if not groups:
        raise ValueError(f"Body {body_index} has no supported collision geometry.")

    minimum_height = min(group[1] for group in groups)
    low_groups = [group for group in groups if group[1] <= minimum_height + height_band_m]
    best_alignment = max(group[0] for group in low_groups)
    selected_groups = [group for group in low_groups if group[0] >= best_alignment - 1.0e-5]
    points_body = np.unique(np.concatenate([group[2] for group in selected_groups]), axis=0)
    points_world = points_transform_xyzw(points_body, body_pose[:3], body_pose[3:7])
    order = np.lexsort((points_world[:, 2], points_world[:, 1], points_world[:, 0]))
    points_body = points_body[order]
    points_world = points_world[order]
    count = min(points_per_body, points_body.shape[0])
    selected = [0]
    minimum_distance = np.full(points_body.shape[0], np.inf)
    for _ in range(count - 1):
        minimum_distance = np.minimum(
            minimum_distance,
            np.linalg.norm(points_world - points_world[selected[-1]], axis=1),
        )
        selected.append(int(np.argmax(minimum_distance)))
    return points_body[selected].astype(np.float32)


def model_body_collider_extreme_points(
    builder: newton.ModelBuilder, body_indices: tuple[int, ...], directions_body: np.ndarray
) -> np.ndarray:
    """Return deterministic body-local collider extremes along body-local directions [m].

    Args:
        builder: Parsed Newton model builder containing body-local collision shapes.
        body_indices: Unique body indices in requested output order.
        directions_body: Nonzero body-local selection directions, shape [body_count, 3].

    Returns:
        Body-local extreme points [m], shape [body_count, 3].
    """
    directions_body = np.asarray(directions_body, dtype=np.float32)
    if (
        not body_indices
        or len(set(body_indices)) != len(body_indices)
        or any(body < 0 or body >= builder.body_count for body in body_indices)
        or directions_body.shape != (len(body_indices), 3)
        or not np.isfinite(directions_body).all()
        or np.any(np.linalg.norm(directions_body, axis=1) <= 1.0e-8)
    ):
        raise ValueError("Collider extremes require unique bodies and one finite nonzero direction per body.")
    requested = {body: slot for slot, body in enumerate(body_indices)}
    selected = np.full((len(body_indices), 3), np.nan, dtype=np.float32)
    selected_projection = np.full(len(body_indices), -np.inf, dtype=np.float32)
    shape_flags = _collision_shape_flags(builder)
    collision_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
    for shape_index, body_index in enumerate(builder.shape_body):
        slot = requested.get(int(body_index))
        if slot is None or not int(shape_flags[shape_index]) & collision_flag:
            continue
        point = _collider_extreme_point(
            int(builder.shape_type[shape_index]),
            builder.shape_scale[shape_index],
            builder.shape_transform[shape_index],
            builder.shape_source[shape_index],
            directions_body[slot],
        )
        projection = float(point @ directions_body[slot])
        if projection > selected_projection[slot]:
            selected[slot] = point
            selected_projection[slot] = projection
    if not np.isfinite(selected).all():
        missing = [body_indices[index] for index in np.flatnonzero(~np.isfinite(selected).all(axis=1))]
        raise ValueError(f"Bodies have no supported collision geometry: {missing}.")
    return selected


def model_body_collider_support_points(
    builder: newton.ModelBuilder,
    body_indices: tuple[int, ...],
    default_body_pose: np.ndarray,
    *,
    points_per_body: int,
    height_band_m: float,
    selection_directions_world: np.ndarray | None = None,
    support_up_world: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return default-world-low collider points in selected body frames [m].

    Args:
        builder: Parsed Newton model builder containing body-local shapes.
        body_indices: Unique body indices in source-support order.
        default_body_pose: Default world poses [m, quaternion xyzw], shape [body_count, 7].
        points_per_body: Maximum retained footprint points per body.
        height_band_m: Default-world height above the lowest point included in
            the support surface [m].
        selection_directions_world: Optional per-body world directions selecting one
            extreme low point, shape [body_count, 3].
        support_up_world: Optional semantic support-up directions used to select
            planar collider faces, shape [body_count, 3].

    Returns:
        Body-local surface points with shape [point_count, 3] and the source
        support slot owning each point with shape [point_count].
    """
    default_body_pose = np.asarray(default_body_pose, dtype=np.float32)
    if selection_directions_world is not None:
        selection_directions_world = np.asarray(selection_directions_world, dtype=np.float32)
    if support_up_world is not None:
        support_up_world = np.asarray(support_up_world, dtype=np.float32)
    if (
        not body_indices
        or len(set(body_indices)) != len(body_indices)
        or any(body < 0 or body >= builder.body_count for body in body_indices)
        or default_body_pose.shape != (builder.body_count, 7)
        or not np.isfinite(default_body_pose).all()
        or points_per_body < 1
        or (
            selection_directions_world is not None
            and (
                points_per_body != 1
                or selection_directions_world.shape != (len(body_indices), 3)
                or not np.isfinite(selection_directions_world).all()
                or np.any(np.linalg.norm(selection_directions_world, axis=1) <= 1.0e-8)
            )
        )
        or (
            support_up_world is not None
            and (
                selection_directions_world is not None
                or support_up_world.shape != (len(body_indices), 3)
                or not np.isfinite(support_up_world).all()
                or np.any(np.linalg.norm(support_up_world, axis=1) <= 1.0e-8)
            )
        )
        or not np.isfinite(height_band_m)
        or height_band_m < 0.0
    ):
        raise ValueError("Support surfaces require unique bodies, default poses, point counts, and a height band.")
    if support_up_world is not None:
        points = [
            _collider_support_points_along_up(
                builder,
                body,
                default_body_pose[body],
                support_up_world[slot],
                points_per_body,
                height_band_m,
            )
            for slot, body in enumerate(body_indices)
        ]
        return np.concatenate(points), np.concatenate(
            [np.full(len(body_points), slot, dtype=np.int64) for slot, body_points in enumerate(points)]
        )
    requested = {body: slot for slot, body in enumerate(body_indices)}
    candidates: list[list[np.ndarray]] = [[] for _ in body_indices]
    shape_flags = _collision_shape_flags(builder)
    collision_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
    world_down = np.asarray(((0.0, 0.0, -1.0),), dtype=np.float32)
    for shape_index, body_index in enumerate(builder.shape_body):
        slot = requested.get(int(body_index))
        if slot is None or not int(shape_flags[shape_index]) & collision_flag:
            continue
        shape_type = int(builder.shape_type[shape_index])
        scale = np.asarray(builder.shape_scale[shape_index], dtype=np.float32)
        transform = np.asarray(builder.shape_transform[shape_index], dtype=np.float32)
        source = builder.shape_source[shape_index]
        body_quaternion = default_body_pose[int(body_index), 3:7]
        body_conjugate = np.concatenate((-body_quaternion[:3], body_quaternion[3:]))
        down_body = _quaternion_rotate_xyzw(body_conjugate, world_down)[0]
        if source is not None and hasattr(source, "vertices"):
            points = _collider_mesh_vertices(source, scale)
            points = points_transform_xyzw(points, transform[:3], transform[3:7])
        elif shape_type == int(GeoType.BOX):
            points = points_transform_xyzw(_BOX_VERTICES * scale[:3], transform[:3], transform[3:7])
        elif shape_type == int(GeoType.SPHERE):
            points = (transform[:3] + scale[0] * down_body)[None]
        elif shape_type in (int(GeoType.CAPSULE), int(GeoType.CYLINDER)):
            shape_conjugate = np.concatenate((-transform[3:6], transform[6:7]))
            down_shape = _quaternion_rotate_xyzw(shape_conjugate, down_body[None])[0]
            point = np.zeros(3, dtype=np.float32)
            point[2] = np.copysign(scale[1], down_shape[2])
            if shape_type == int(GeoType.CAPSULE):
                point += scale[0] * down_shape
            else:
                radial = down_shape[:2]
                radial_norm = np.linalg.norm(radial)
                if radial_norm > 1.0e-8:
                    point[:2] = scale[0] * radial / radial_norm
            points = points_transform_xyzw(point[None], transform[:3], transform[3:7])
        else:
            raise ValueError(f"Unsupported support collision shape {GeoType(shape_type).name}.")
        candidates[slot].append(points)

    output_points: list[np.ndarray] = []
    output_slots: list[np.ndarray] = []
    for slot, parts in enumerate(candidates):
        if not parts:
            raise ValueError(f"Body {body_indices[slot]} has no supported collision geometry.")
        local_points = np.unique(np.concatenate(parts), axis=0)
        body_pose = default_body_pose[body_indices[slot]]
        world_points = points_transform_xyzw(local_points, body_pose[:3], body_pose[3:7])
        keep = world_points[:, 2] <= world_points[:, 2].min() + height_band_m
        local_points = local_points[keep]
        world_points = world_points[keep]
        order = np.lexsort((world_points[:, 2], world_points[:, 1], world_points[:, 0]))
        local_points = local_points[order]
        world_points = world_points[order]
        count = min(points_per_body, local_points.shape[0])
        if selection_directions_world is not None:
            direction = selection_directions_world[slot]
            direction = direction / np.linalg.norm(direction)
            selected = [int(np.argmax(world_points @ direction))]
        else:
            selected = [0]
            minimum_distance = np.full(local_points.shape[0], np.inf)
            for _ in range(count - 1):
                minimum_distance = np.minimum(
                    minimum_distance,
                    np.linalg.norm(world_points[:, :2] - world_points[selected[-1], :2], axis=1),
                )
                selected.append(int(np.argmax(minimum_distance)))
        output_points.append(local_points[selected])
        output_slots.append(np.full(count, slot, dtype=np.int64))
    return np.concatenate(output_points).astype(np.float32), np.concatenate(output_slots)


def mesh_points_farthest_sample(points: np.ndarray, count: int) -> np.ndarray:
    """Return a deterministic farthest-point subset."""
    count = min(count, points.shape[0])
    selected = [0]
    minimum_distance = np.full(points.shape[0], np.inf)
    for _ in range(count - 1):
        minimum_distance = np.minimum(
            minimum_distance,
            np.linalg.norm(points - points[selected[-1]], axis=1),
        )
        selected.append(int(np.argmax(minimum_distance)))
    return points[selected]


def mesh_edges(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unique triangle-edge endpoints [m]."""
    edges = np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return vertices[edges[:, 0]].astype(np.float32), vertices[edges[:, 1]].astype(np.float32)


def collider_mesh_load(
    usd_path: str,
    device: str,
    scale: tuple[float, float, float] | None = None,
    visual: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one rigid USD asset as a root-frame triangle mesh.

    Args:
        usd_path: Local or remote USD asset path.
        device: Newton construction device.
        scale: Optional root scale.
        visual: Whether to prefer non-collision render meshes.

    Returns:
        Root-frame vertices [m] and outward-wound triangle indices.
    """
    resolved_path = resolve_newton_asset_path(usd_path)
    builder = newton.ModelBuilder()
    builder.add_usd(resolved_path, floating=False, skip_mesh_approximation=True)
    if builder.body_count != 1:
        raise ValueError(f"{resolved_path} added {builder.body_count} bodies; expected one.")
    model = builder.finalize(device=device)

    flags = model.shape_flags.numpy()
    shape_types = model.shape_type.numpy()
    shape_scales = model.shape_scale.numpy()
    shape_transforms = model.shape_transform.numpy()
    collision_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
    is_collision = [bool(int(flags[index]) & collision_flag) for index in range(model.shape_count)]
    if visual:
        selected = [
            index
            for index in range(model.shape_count)
            if not is_collision[index] and model.shape_source[index] is not None
        ]
        if not selected:
            selected = [index for index in range(model.shape_count) if is_collision[index]]
    else:
        selected = [index for index in range(model.shape_count) if is_collision[index]]

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_offset = 0
    for shape_index in selected:
        source = model.shape_source[shape_index]
        if source is not None:
            shape_vertices = _collider_mesh_vertices(source, shape_scales[shape_index])
            shape_faces = _collider_mesh_faces(source, shape_scales[shape_index])
        elif int(shape_types[shape_index]) == int(GeoType.BOX):
            shape_vertices = _BOX_VERTICES * shape_scales[shape_index]
            shape_faces = _BOX_FACES
        else:
            shape_name = GeoType(int(shape_types[shape_index])).name
            raise ValueError(f"Unsupported collision shape {shape_name} in {resolved_path}.")
        transform = shape_transforms[shape_index]
        vertices.append(_quaternion_rotate_xyzw(transform[3:7], shape_vertices) + transform[:3])
        faces.append(shape_faces + vertex_offset)
        vertex_offset += shape_vertices.shape[0]
    if not vertices:
        raise RuntimeError(f"No selected meshes found in {resolved_path}.")

    vertices_array = np.concatenate(vertices)
    faces_array = np.concatenate(faces)
    if scale is not None:
        vertices_array *= np.asarray(scale, dtype=np.float32)
    triangle = vertices_array[faces_array]
    if float((np.cross(triangle[:, 0], triangle[:, 1]) * triangle[:, 2]).sum()) < 0.0:
        faces_array = faces_array[:, ::-1].copy()
    return vertices_array, faces_array


def model_collision_shape_indices(model: newton.Model, body_index: int | None = None) -> np.ndarray:
    """Return source-mesh collision shape indices, optionally for one body."""
    flags = model.shape_flags.numpy()
    bodies = model.shape_body.numpy()
    collision_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
    return np.asarray(
        [
            shape_index
            for shape_index in range(model.shape_count)
            if int(flags[shape_index]) & collision_flag
            and model.shape_source[shape_index] is not None
            and (body_index is None or int(bodies[shape_index]) == body_index)
        ],
        dtype=np.int32,
    )


def model_collision_mesh(model: newton.Model, shape_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Combine collision source meshes in their owning body frame."""
    shape_transforms = model.shape_transform.numpy()
    shape_scales = model.shape_scale.numpy()
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_offset = 0
    for shape_index in shape_indices:
        source = model.shape_source[int(shape_index)]
        shape_vertices = _collider_mesh_vertices(source, shape_scales[int(shape_index)])
        shape_faces = _collider_mesh_faces(source, shape_scales[int(shape_index)])
        transform = shape_transforms[int(shape_index)]
        vertices.append(points_transform_xyzw(shape_vertices, transform[:3], transform[3:7]))
        faces.append(shape_faces + vertex_offset)
        vertex_offset += shape_vertices.shape[0]
    return np.concatenate(vertices).astype(np.float32), np.concatenate(faces)


def model_shape_surface_probes(
    model: newton.Model,
    shape_index: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one collision mesh surface in its owning body frame [m]."""
    source = model.shape_source[shape_index]
    vertices = _collider_mesh_vertices(source, model.shape_scale.numpy()[shape_index])
    faces = np.asarray(source.indices, dtype=np.int32).reshape(-1, 3)
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    candidate_count = max(4 * count, 256)
    face_indices = rng.choice(faces.shape[0], size=candidate_count, p=areas / areas.sum())
    first = rng.random((candidate_count, 1), dtype=np.float32)
    second = rng.random((candidate_count, 1), dtype=np.float32)
    flip = first + second > 1.0
    first = np.where(flip, 1.0 - first, first)
    second = np.where(flip, 1.0 - second, second)
    points = (
        triangles[face_indices, 0]
        + first * (triangles[face_indices, 1] - triangles[face_indices, 0])
        + second * (triangles[face_indices, 2] - triangles[face_indices, 0])
    )
    transform = model.shape_transform.numpy()[shape_index]
    points = mesh_points_farthest_sample(points, count)
    return (_quaternion_rotate_xyzw(transform[3:7], points) + transform[:3]).astype(np.float32)


def model_collision_edges(
    model: newton.Model,
    shape_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return body-local collision-mesh edges and their body indices."""
    shape_bodies = model.shape_body.numpy()
    shape_scales = model.shape_scale.numpy()
    shape_transforms = model.shape_transform.numpy()
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    bodies: list[np.ndarray] = []
    for shape_index in shape_indices:
        source = model.shape_source[int(shape_index)]
        vertices = _collider_mesh_vertices(source, shape_scales[int(shape_index)])
        faces = np.asarray(source.indices, dtype=np.int32).reshape(-1, 3)
        start, end = mesh_edges(vertices, faces)
        transform = shape_transforms[int(shape_index)]
        starts.append(_quaternion_rotate_xyzw(transform[3:7], start) + transform[:3])
        ends.append(_quaternion_rotate_xyzw(transform[3:7], end) + transform[:3])
        bodies.append(np.full(start.shape[0], int(shape_bodies[int(shape_index)]), dtype=np.int32))
    return np.concatenate(starts).astype(np.float32), np.concatenate(ends).astype(np.float32), np.concatenate(bodies)
