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


def _collider_z_min(shape_type: int, scale, transform, source) -> float:
    """Return one shape's lowest point in its owning body frame [m]."""
    transform = np.asarray(transform, dtype=np.float32)
    position = transform[:3]
    quaternion = transform[3:7]
    scale = np.asarray(scale, dtype=np.float32)
    if source is not None and hasattr(source, "vertices"):
        vertices = np.asarray(source.vertices, dtype=np.float32).reshape(-1, 3)
        if not vertices.shape[0]:
            raise ValueError("Collision mesh must contain at least one vertex.")
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
    shape_flags = getattr(builder, "shape_flags", None)
    collision_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
    for shape_index, body_index in enumerate(builder.shape_body):
        output_index = requested.get(int(body_index))
        if output_index is None:
            continue
        if shape_flags is not None and not int(shape_flags[shape_index]) & collision_flag:
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
            shape_vertices = np.asarray(source.vertices, dtype=np.float32).reshape(-1, 3)
            shape_faces = np.asarray(source.indices, dtype=np.int32).reshape(-1, 3)
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
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_offset = 0
    for shape_index in shape_indices:
        source = model.shape_source[int(shape_index)]
        shape_vertices = np.asarray(source.vertices, dtype=np.float32).reshape(-1, 3)
        shape_faces = np.asarray(source.indices, dtype=np.int32).reshape(-1, 3)
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
    vertices = np.asarray(source.vertices, dtype=np.float32).reshape(-1, 3)
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
    shape_transforms = model.shape_transform.numpy()
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    bodies: list[np.ndarray] = []
    for shape_index in shape_indices:
        source = model.shape_source[int(shape_index)]
        vertices = np.asarray(source.vertices, dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(source.indices, dtype=np.int32).reshape(-1, 3)
        start, end = mesh_edges(vertices, faces)
        transform = shape_transforms[int(shape_index)]
        starts.append(_quaternion_rotate_xyzw(transform[3:7], start) + transform[:3])
        ends.append(_quaternion_rotate_xyzw(transform[3:7], end) + transform[:3])
        bodies.append(np.full(start.shape[0], int(shape_bodies[int(shape_index)]), dtype=np.int32))
    return np.concatenate(starts).astype(np.float32), np.concatenate(ends).astype(np.float32), np.concatenate(bodies)
