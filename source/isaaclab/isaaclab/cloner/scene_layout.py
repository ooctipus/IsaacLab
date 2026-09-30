# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declare static scene-data layout from clone-plan prototypes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from . import path as clone_path
from .clone_plan import (
    ArticulationLayout,
    CableLayout,
    ClonePlan,
    DeformableLayout,
    FrameLayout,
    GeometryLayout,
    JointLayout,
    NewtonActuatorLayout,
    PointCloudLayout,
    RigidBodyLayout,
    SurfaceGripperLayout,
)

_GEOMETRY_TYPES = frozenset(("Mesh", "Plane", "Cube", "Sphere", "Cylinder", "Capsule", "Cone"))


def _immutable_value(value: Any) -> Any:
    """Freeze nested parsed arguments before storing them on the clone plan."""
    if isinstance(value, dict):
        return tuple((key, _immutable_value(item)) for key, item in sorted(value.items()))
    if isinstance(value, list | tuple):
        return tuple(_immutable_value(item) for item in value)
    if isinstance(value, np.ndarray):
        return _immutable_value(value.tolist())
    return value


def _get_applied_schema_names(prim: Usd.Prim) -> set[str]:
    """Return applied API schema names from composed schemas and explicit metadata."""
    names = set(prim.GetAppliedSchemas())
    api_schemas = prim.GetMetadata("apiSchemas")
    if isinstance(api_schemas, Sdf.TokenListOp):
        names.update(
            str(token)
            for field in ("prependedItems", "appendedItems", "explicitItems")
            for token in getattr(api_schemas, field)
        )
    return names


def _prim_has_schema(prim: Usd.Prim, schema_substring: str) -> bool:
    """Return whether an applied API schema contains ``schema_substring``."""
    return any(schema_substring in name for name in _get_applied_schema_names(prim))


def _mesh_point_count(prim: Usd.Prim) -> int:
    """Return the authored point count of a mesh or tetrahedral mesh."""
    return len(UsdGeom.PointBased(prim).GetPointsAttr().Get() or [])


def _select_visual_mesh(candidates: list[Usd.Prim], sim_mesh: Usd.Prim) -> Usd.Prim:
    """Select the one visual mesh identified by topology or an explicit name."""
    if not candidates:
        return sim_mesh
    if len(candidates) == 1:
        return candidates[0]

    sim_count = _mesh_point_count(sim_mesh)
    named = [
        prim
        for prim in candidates
        if any(token in prim.GetName().lower() for token in ("visual", "render", "display", "proxy"))
    ]
    matching = [prim for prim in candidates if _mesh_point_count(prim) == sim_count]
    selected = named or matching
    if len(selected) != 1:
        paths = [prim.GetPath().pathString for prim in candidates]
        raise ValueError(f"Deformable body has ambiguous visual meshes: {paths}.")
    return selected[0]


def _classify_deformable(root: Usd.Prim) -> tuple[str, Usd.Prim, Usd.Prim, int]:
    """Return the type, simulation mesh, visual mesh, and unpadded node count for ``root``."""
    prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    tet_meshes = [prim for prim in prims if prim.GetTypeName() == "TetMesh"]
    meshes = [prim for prim in prims if prim.GetTypeName() == "Mesh"]

    if len(tet_meshes) > 1:
        raise ValueError(
            f"Deformable body '{root.GetPath()}' declares multiple simulation TetMesh prims: "
            f"{[prim.GetPath().pathString for prim in tet_meshes]}."
        )
    if tet_meshes:
        sim_mesh = tet_meshes[0]
        visual_mesh = _select_visual_mesh(
            [prim for prim in meshes if not _prim_has_schema(prim, "DeformableSimAPI")], sim_mesh
        )
        return "volume", sim_mesh, visual_mesh, _mesh_point_count(sim_mesh)

    sim_meshes = [prim for prim in meshes if _prim_has_schema(prim, "DeformableSimAPI")]
    if len(sim_meshes) > 1:
        raise ValueError(
            f"Deformable body '{root.GetPath()}' declares multiple simulation Mesh prims: "
            f"{[prim.GetPath().pathString for prim in sim_meshes]}."
        )
    if sim_meshes:
        sim_mesh = sim_meshes[0]
        visual_mesh = _select_visual_mesh([prim for prim in meshes if prim != sim_mesh], sim_mesh)
        return "surface", sim_mesh, visual_mesh, _mesh_point_count(sim_mesh)
    if len(meshes) == 1:
        return "surface", meshes[0], meshes[0], _mesh_point_count(meshes[0])
    raise ValueError(f"Deformable body '{root.GetPath()}' must declare exactly one simulation mesh.")


def _cable_segment_count(prim: Usd.Prim) -> int:
    """Return the segment count of one supported open linear cable."""
    curve = UsdGeom.BasisCurves(prim)
    counts = curve.GetCurveVertexCountsAttr().Get() or []
    if (
        len(counts) != 1
        or int(counts[0]) < 2
        or curve.GetTypeAttr().Get() != UsdGeom.Tokens.linear
        or curve.GetWrapAttr().Get() == UsdGeom.Tokens.periodic
    ):
        raise ValueError(f"Cable '{prim.GetPath()}' must be one open linear curve with at least two points.")
    return int(counts[0]) - 1


def _geometry_data(prim: Usd.Prim, xform_cache: UsdGeom.XformCache) -> tuple[np.ndarray, np.ndarray]:
    """Return immutable triangle geometry with the composed USD affine residual baked into its vertices."""
    from isaaclab.utils.mesh import convert_faces_to_triangles, create_trimesh_from_geom_shape  # noqa: PLC0415

    matrix = xform_cache.GetLocalToWorldTransform(prim)
    rigid = Gf.Matrix4d(matrix)
    rigid.Orthonormalize()
    residual = np.asarray(matrix * rigid.GetInverse())
    if prim.GetTypeName() == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        vertices = np.asarray(mesh.GetPointsAttr().Get()) @ residual[:3, :3] + residual[3, :3]
        faces = convert_faces_to_triangles(
            np.asarray(mesh.GetFaceVertexIndicesAttr().Get()), np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        )
    else:
        mesh = create_trimesh_from_geom_shape(prim)
        mesh.apply_transform(residual.T)
        vertices, faces = mesh.vertices, mesh.faces
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    vertices.setflags(write=False)
    faces.setflags(write=False)
    return vertices, faces


def _deformable_vertices(root: Usd.Prim, mesh: Usd.Prim) -> np.ndarray:
    """Return immutable mesh vertices baked into the deformable parent's frame."""
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    transform = (
        xform_cache.GetLocalToWorldTransform(mesh) * xform_cache.GetLocalToWorldTransform(root.GetParent()).GetInverse()
    )
    points = UsdGeom.PointBased(mesh).GetPointsAttr().Get() or []
    vertices = np.asarray([transform.Transform(Gf.Vec3d(*map(float, point))) for point in points], dtype=np.float32)
    vertices.setflags(write=False)
    return vertices


def _deformable_topology(root: Usd.Prim, mesh: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
    """Return immutable simulation topology baked into the deformable parent's frame."""
    vertices = _deformable_vertices(root, mesh)
    if mesh.GetTypeName() == "TetMesh":
        indices = np.asarray(UsdGeom.TetMesh(mesh).GetTetVertexIndicesAttr().Get() or [], dtype=np.int32).reshape(-1, 4)
    else:
        indices = np.asarray(UsdGeom.Mesh(mesh).GetFaceVertexIndicesAttr().Get() or [], dtype=np.int32).reshape(-1, 3)
    indices.setflags(write=False)
    return vertices, indices


def _deformable_point_mapping(
    deformable_type: str,
    sim_mesh: Usd.Prim,
    vis_mesh: Usd.Prim,
    sim_vertices: np.ndarray,
    sim_indices: np.ndarray,
    vis_vertices: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return immutable four-node interpolation rows from simulation to visual vertices."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    vis_count = len(vis_vertices)
    if sim_mesh == vis_mesh:
        return None, None
    elif not len(sim_indices):
        raise ValueError(f"Deformable visual mesh {vis_mesh.GetPath()} has no simulation elements to map from.")
    else:
        nodes = sim_vertices[sim_indices]
        if deformable_type == "volume":
            matrices = np.transpose(nodes[:, 1:] - nodes[:, :1], (0, 2, 1))
            valid = np.abs(np.linalg.det(matrices)) > 1.0e-12
            if not np.any(valid):
                raise ValueError(f"Deformable simulation mesh {sim_mesh.GetPath()} contains no valid tetrahedra.")
            valid_elements = sim_indices[valid]
            origins = nodes[valid, 0]
            inverses = np.linalg.inv(matrices[valid])
            tree = cKDTree(nodes[valid].mean(axis=1))
            nearest = np.asarray(tree.query(vis_vertices, k=min(16, len(valid_elements)))[1]).reshape(vis_count, -1)
            mapped_indices = []
            mapped_weights = []
            for point, initial in zip(vis_vertices, nearest, strict=True):
                candidates_ids = initial
                while True:
                    tail = np.einsum("nij,nj->ni", inverses[candidates_ids], point - origins[candidates_ids])
                    candidates = np.column_stack((1.0 - tail.sum(axis=1), tail))
                    inside = candidates.min(axis=1) >= -1.0e-5
                    if np.any(inside) or len(candidates_ids) == len(valid_elements):
                        break
                    count = min(2 * len(candidates_ids), len(valid_elements))
                    candidates_ids = np.atleast_1d(tree.query(point, k=count)[1])
                selected = int(np.argmax(candidates.min(axis=1)))
                mapped_indices.append(valid_elements[candidates_ids[selected]])
                mapped_weights.append(candidates[selected])
            indices = np.asarray(mapped_indices, dtype=np.int32)
            weights = np.asarray(mapped_weights, dtype=np.float32)
        else:
            a, ab, ac = nodes[:, 0], nodes[:, 1] - nodes[:, 0], nodes[:, 2] - nodes[:, 0]
            d00 = np.einsum("ij,ij->i", ab, ab)
            d01 = np.einsum("ij,ij->i", ab, ac)
            d11 = np.einsum("ij,ij->i", ac, ac)
            denominator = d00 * d11 - d01 * d01
            valid = np.abs(denominator) > 1.0e-12
            if not np.any(valid):
                raise ValueError(f"Deformable simulation mesh {sim_mesh.GetPath()} contains no valid triangles.")
            valid_nodes = nodes[valid]
            valid_indices = sim_indices[valid]
            tree = cKDTree(valid_nodes.mean(axis=1))
            nearest = np.asarray(tree.query(vis_vertices, k=min(16, len(valid_nodes)))[1]).reshape(vis_count, -1)
            mapped_indices = []
            mapped_weights = []
            for point, initial in zip(vis_vertices, nearest, strict=True):
                candidates_ids = initial
                while True:
                    ap = point - a[valid][candidates_ids]
                    d20 = np.einsum("ij,ij->i", ap, ab[valid][candidates_ids])
                    d21 = np.einsum("ij,ij->i", ap, ac[valid][candidates_ids])
                    w2 = (d11[valid][candidates_ids] * d20 - d01[valid][candidates_ids] * d21) / denominator[valid][
                        candidates_ids
                    ]
                    w3 = (d00[valid][candidates_ids] * d21 - d01[valid][candidates_ids] * d20) / denominator[valid][
                        candidates_ids
                    ]
                    candidates = np.column_stack((1.0 - w2 - w3, w2, w3))
                    projected = np.einsum("ni,nij->nj", candidates, valid_nodes[candidates_ids])
                    distance = np.einsum("ij,ij->i", projected - point, projected - point)
                    inside = candidates.min(axis=1) >= -1.0e-5
                    if np.any(inside) or len(candidates_ids) == len(valid_nodes):
                        break
                    count = min(2 * len(candidates_ids), len(valid_nodes))
                    candidates_ids = np.atleast_1d(tree.query(point, k=count)[1])
                selected = (
                    int(np.argmin(np.where(inside, distance, np.inf)))
                    if np.any(inside)
                    else int(np.argmax(candidates.min(axis=1)))
                )
                triangle = valid_indices[candidates_ids[selected]]
                mapped_indices.append((*triangle, triangle[0]))
                mapped_weights.append((*candidates[selected], 0.0))
            indices = np.asarray(mapped_indices, dtype=np.int32)
            weights = np.asarray(mapped_weights, dtype=np.float32)
    indices.setflags(write=False)
    weights.setflags(write=False)
    return indices, weights


def _joint_pose(joint: UsdPhysics.Joint) -> tuple[float, float, float, float, float, float, float]:
    """Return one joint's child-side local pose."""
    position = joint.GetLocalPos1Attr().Get()
    rotation = joint.GetLocalRot1Attr().Get()
    position = Gf.Vec3f() if position is None else position
    rotation = Gf.Quatf(1.0) if rotation is None else rotation
    imaginary = rotation.GetImaginary()
    return (*map(float, position), *map(float, imaginary), float(rotation.GetReal()))


def _order_articulation_bodies(
    body_paths: tuple[str, ...], joints: list[tuple[str | None, str]], root_path: str | None
) -> tuple[str, ...]:
    """Return the plan's root-first depth-first body order."""
    if not joints:
        return body_paths
    bodies = set(body_paths)
    neighbors = {body: [] for body in body_paths}
    roots = set(() if root_path is None else (root_path,))
    for body0, body1 in dict.fromkeys(joints):
        if body1 not in bodies or (body0 is not None and body0 not in bodies):
            raise ValueError("Articulation joint graph does not cover its declared rigid bodies exactly once.")
        if body0 is None:
            roots.add(body1)
        else:
            neighbors[body0].append(body1)
            neighbors[body1].append(body0)
    if len(roots) > 1:
        raise ValueError(f"Articulation joint graph has {len(roots)} roots; expected one.")

    ordered = []
    visited = set()
    stack = [(roots.pop() if roots else body_paths[0], None)]
    while stack:
        body, parent = stack.pop()
        if body in visited:
            raise ValueError("Articulation joint graph does not cover its declared rigid bodies exactly once.")
        visited.add(body)
        ordered.append(body)
        stack.extend((neighbor, body) for neighbor in reversed(neighbors[body]) if neighbor != parent)
    if visited != bodies:
        raise ValueError("Articulation joint graph does not cover its declared rigid bodies exactly once.")
    return tuple(ordered)


def _float_attribute(prim: Usd.Prim, name: str) -> float | None:
    """Return an authored scalar attribute as a float."""
    attribute = prim.GetAttribute(name)
    value = attribute.Get() if attribute else None
    return None if value is None else float(value)


def _element_name(prim: Usd.Prim) -> str:
    """Return an articulation element's authored name override or prim name."""
    for name in ("isaac:nameOverride", "isaac:NameOverride"):
        value = prim.GetAttribute(name).Get()
        if value not in (None, ""):
            return str(value)
    return prim.GetName()


def _clone_masks(plan: ClonePlan) -> tuple[np.ndarray | None, ...]:
    """Return one shared read-only clone-column mask per plan row."""
    masks = np.asarray(plan.clone_mask, dtype=np.bool_).copy()
    masks.setflags(write=False)
    rows = tuple(None if "{}" not in destination else masks[row] for row, destination in enumerate(plan.destinations))
    for row, destination in enumerate(plan.destinations):
        for previous in range(row):
            if destination != plan.destinations[previous]:
                continue
            if rows[row] is None or rows[previous] is None or np.any(rows[row] & rows[previous]):
                raise ValueError("Clone plan assigns a destination subtree to multiple rows.")
    return rows


def _exact_destinations(plan: ClonePlan, row: int, clone_mask: np.ndarray | None) -> Iterable[tuple[int | None, str]]:
    """Return exact destination roots for one prototype mask."""
    destination = plan.destinations[row]
    if clone_mask is None:
        return ((None, destination),)
    if plan.env_ids is None:
        raise ValueError("ClonePlan.env_ids is required to declare clone-plan topology.")
    env_ids = plan.env_ids
    return ((int(env_ids[column]), destination.format(int(env_ids[column]))) for column in np.flatnonzero(clone_mask))


def _add_unique(entries: dict[str, object], path: str, entry: object) -> None:
    """Add one exact path, rejecting overlapping clone-plan ownership."""
    if path in entries:
        raise ValueError(f"Clone plan declares dynamic scene path {path!r} more than once.")
    entries[path] = entry


def _prototype_mask(
    plan: ClonePlan,
    row: int,
    path: str,
    row_masks: tuple[np.ndarray | None, ...],
    mask_cache: dict[bytes, np.ndarray],
) -> tuple[bool, np.ndarray | None]:
    """Return whether a prototype path is owned and its populated clone columns."""
    destination = plan.destinations[row]
    nested = tuple(
        owner
        for owner, root in enumerate(plan.destinations)
        if owner != row
        and ("{}" in root) == ("{}" in destination)
        and root != destination
        and clone_path.under(root, destination)
        and clone_path.under(path, root)
    )
    clone_mask = row_masks[row]
    if clone_mask is None:
        return not nested, None
    if nested:
        clone_mask = clone_mask & ~np.logical_or.reduce(tuple(row_masks[owner] for owner in nested))
        if not np.any(clone_mask):
            return False, None
        clone_mask = _intern_mask(clone_mask, mask_cache)
    return bool(np.any(clone_mask)), clone_mask


def _intern_mask(clone_mask: np.ndarray, mask_cache: dict[bytes, np.ndarray]) -> np.ndarray:
    """Return one shared read-only copy of equivalent clone-column masks."""
    key = clone_mask.tobytes()
    if key not in mask_cache:
        clone_mask.setflags(write=False)
        mask_cache[key] = clone_mask
    return mask_cache[key]


def _pose(matrix: Gf.Matrix4d) -> tuple[float, float, float, float, float, float, float]:
    """Return one orthonormalized matrix as ``(tx, ty, tz, qx, qy, qz, qw)``."""
    matrix = Gf.Matrix4d(matrix)
    matrix.Orthonormalize()
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    return (*map(float, translation), *map(float, imaginary), float(rotation.GetReal()))


def _bind_frames(
    plan: ClonePlan,
    stage: Usd.Stage,
    frames: tuple[FrameLayout, ...],
    rigid_bodies: tuple[RigidBodyLayout, ...],
) -> tuple[FrameLayout, ...]:
    """Resolve each prototype frame's fixed body/world binding."""
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    source_facts: dict[tuple[int, str], tuple[Any, ...]] = {}
    bodies_by_path: dict[str, list[RigidBodyLayout]] = {}
    for body in rigid_bodies:
        bodies_by_path.setdefault(body.path, []).append(body)

    def nearest_body(path: str | None, column: int | None) -> RigidBodyLayout | None:
        while path is not None:
            for body in bodies_by_path.get(path, ()):
                if column is None:
                    if body.clone_mask is not None:
                        continue
                elif body.clone_mask is not None and body.clone_mask[column]:
                    return body
            path = path.rsplit("/", 1)[0] or None
        return None

    def representative_column(frame: FrameLayout) -> int | None:
        if frame.clone_mask is None:
            return None
        if plan.env_ids is None:
            raise ValueError("ClonePlan.env_ids is required to bind planned frames.")
        return int(np.flatnonzero(frame.clone_mask)[0])

    def body_template(body: RigidBodyLayout | None) -> str | None:
        if body is None:
            return None
        if body.source_path is None:
            raise ValueError(f"Rigid body {body.path!r} lacks clone-plan source provenance.")
        return body.path

    def prototype_pose(prim: Usd.Prim | None, body: RigidBodyLayout | None):
        if prim is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        transform = xform_cache.GetLocalToWorldTransform(prim)
        if body is None:
            return _pose(transform)
        body_prim = stage.GetPrimAtPath(body.source_path)
        if not body_prim.IsValid():
            raise ValueError(f"Rigid-body source prim is absent from the stage: {body.source_path!r}.")
        return _pose(transform * xform_cache.GetLocalToWorldTransform(body_prim).GetInverse())

    bound = []
    for frame in frames:
        key = (frame.row, frame.source_path)
        if key not in source_facts:
            prim = stage.GetPrimAtPath(frame.source_path)
            column = representative_column(frame)
            parent_path = frame.path.rsplit("/", 1)[0] or None
            prototype_body = nearest_body(frame.path, column)
            prototype_parent_body = nearest_body(parent_path, column)
            parent = prim.GetParent() if frame.parent_path is not None else None
            scale_attr = prim.GetAttribute("xformOp:scale")
            scale = (
                tuple(map(float, scale_attr.Get())) if scale_attr and scale_attr.HasAuthoredValue() else (1.0, 1.0, 1.0)
            )
            source_facts[key] = (
                prototype_body,
                prototype_parent_body,
                prototype_pose(prim, prototype_body),
                prototype_pose(parent, prototype_parent_body),
                scale,
            )
        prototype_body, prototype_parent_body, pose, parent_pose, scale = source_facts[key]
        bound.append(
            FrameLayout(
                frame.path,
                frame.source_path,
                frame.parent_path,
                frame.row,
                None,
                body_template(prototype_body),
                None if prototype_body is None else prototype_body.view_path,
                pose,
                body_template(prototype_parent_body),
                parent_pose,
                scale,
                frame.clone_mask,
            )
        )
    return tuple(bound)


def declare_scene_layout(plan: ClonePlan, stage: Usd.Stage) -> ClonePlan:  # noqa: C901
    """Return ``plan`` completed once with topology from its authored prototype rows.

    Only populated prototype subtrees are inspected. Each discovered fact keeps one destination
    template and clone-column mask; consumers materialize exact paths only when a native backend or
    targeted query requires them. No consumer inspects the replicated stage afterward.

    Args:
        plan: Clone plan whose prototype prims have been authored.
        stage: USD stage containing those prototype prims.

    Returns:
        The original plan when it is already complete, otherwise an immutable replacement carrying
        its declared topology.

    Raises:
        ValueError: If a populated source is missing, a deformable declaration is ambiguous, or
            multiple rows claim the same dynamic destination path.
    """
    if plan.is_complete:
        return plan

    row_masks = _clone_masks(plan)
    mask_cache = {mask.tobytes(): mask for mask in row_masks if mask is not None}
    frames: list[FrameLayout] = []
    requested_frames: list[FrameLayout] = []
    rigid_bodies: list[RigidBodyLayout] = []
    articulations: list[ArticulationLayout] = []
    deformables: dict[str, DeformableLayout] = {}
    cables: dict[str, CableLayout] = {}
    point_clouds: dict[str, PointCloudLayout] = {}
    surface_grippers: dict[str, SurfaceGripperLayout] = {}
    heightfields: dict[str, tuple[str, float]] = {}
    geometry_candidates: list[tuple[str, str, int, np.ndarray | None, Usd.Prim, tuple[str, float] | None]] = []
    xform_cache = UsdGeom.XformCache()
    for row, source in enumerate(plan.sources):
        if row_masks[row] is not None and not np.any(row_masks[row]):
            continue
        source_prim = stage.GetPrimAtPath(source)
        if not source_prim.IsValid():
            raise ValueError(f"Populated clone-plan source prim is absent from the stage: {source!r}.")
        view_root = plan.destinations[row].format("*") if "{}" in plan.destinations[row] else plan.destinations[row]
        prims = tuple(Usd.PrimRange(source_prim, Usd.TraverseInstanceProxies()))
        actuator_prims = tuple(prim for prim in prims if prim.GetTypeName() == "NewtonActuator")
        actuators_by_joint = {}
        if actuator_prims:
            from newton.actuators import parse_actuator_prim  # noqa: PLC0415

            for prim in actuator_prims:
                parsed = parse_actuator_prim(prim)
                if parsed is None:
                    continue
                controller_arguments = parsed.controller_class.resolve_arguments(dict(parsed.controller_kwargs))
                component_arguments = tuple(
                    (
                        component_class,
                        tuple(
                            (name, _immutable_value(value))
                            for name, value in component_class.resolve_arguments(dict(arguments)).items()
                        ),
                    )
                    for component_class, arguments in parsed.component_specs
                )
                actuators_by_joint[str(parsed.target_path)] = NewtonActuatorLayout(
                    controller_class=parsed.controller_class,
                    controller_arguments=tuple(
                        (name, _immutable_value(value)) for name, value in controller_arguments.items()
                    ),
                    component_arguments=component_arguments,
                )
        articulation_roots = tuple(
            prim
            for prim in prims
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            and prim.GetAttribute("physxArticulation:articulationEnabled").Get() is not False
        )
        for prim in prims:
            source_path = prim.GetPath().pathString
            source_suffix = source_path[len(source.rstrip("/")) :]
            if prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader) or prim.IsA(UsdPhysics.Joint):
                continue
            path = (plan.destinations[row].rstrip("/") + source_suffix) or "/"
            owned, clone_mask = _prototype_mask(plan, row, path, row_masks, mask_cache)
            if not owned:
                continue
            attr = prim.GetAttribute("newton:heightfield:resolution")
            if clone_mask is None and attr and attr.HasAuthoredValue():
                meshes = [
                    candidate
                    for candidate in prims
                    if candidate.IsA(UsdGeom.Mesh) and clone_path.under(candidate.GetPath().pathString, source_path)
                ]
                if len(meshes) != 1:
                    raise ValueError(f"Heightfield root {source_path!r} must contain exactly one mesh.")
                heightfields[meshes[0].GetPath().pathString] = (source_path, float(attr.Get()))
            if UsdGeom.Xformable(prim):
                parent_path = path.rsplit("/", 1)[0] or None
                frames.append(FrameLayout(path, source_path, parent_path, row, None, clone_mask=clone_mask))
            elif plan.geometry_requests:
                parent_path = path.rsplit("/", 1)[0] or None
                requested_frames.append(FrameLayout(path, source_path, parent_path, row, None, clone_mask=clone_mask))
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and not prim.IsA(UsdPhysics.Joint):
                view_path = clone_path.rebase(source_path, source, view_root)
                name = _element_name(prim)
                has_contact_report = _prim_has_schema(prim, "PhysxContactReportAPI")
                rigid_bodies.append(
                    RigidBodyLayout(path, view_path, row, None, name, source_path, has_contact_report, clone_mask)
                )
            heightfield = heightfields.get(source_path)
            if (plan.geometry_requests or heightfield is not None) and prim.GetTypeName() in _GEOMETRY_TYPES:
                geometry_candidates.append((path, source_path, row, clone_mask, prim, heightfield))
            if prim.IsA(UsdGeom.BasisCurves) and _prim_has_schema(prim, "PhysicsCurvesDeformableSimAPI"):
                segment_count = _cable_segment_count(prim)
                cable_view_path = clone_path.rebase(source_path, source, view_root)
                for env_id, exact_path in _exact_destinations(plan, row, clone_mask):
                    exact_path = (exact_path.rstrip("/") + source_suffix) or "/"
                    _add_unique(
                        cables, exact_path, CableLayout(exact_path, cable_view_path, segment_count, row, env_id)
                    )
            if prim.IsA(UsdGeom.Points):
                point_count = _mesh_point_count(prim)
                for env_id, exact_path in _exact_destinations(plan, row, clone_mask):
                    exact_path = (exact_path.rstrip("/") + source_suffix) or "/"
                    _add_unique(point_clouds, exact_path, PointCloudLayout(exact_path, point_count, row, env_id))
            if prim.GetTypeName() == "IsaacSurfaceGripper":
                gripper_view_path = clone_path.rebase(source_path, source, view_root)
                parameters = (
                    _float_attribute(prim, "isaac:maxGripDistance"),
                    _float_attribute(prim, "isaac:coaxialForceLimit"),
                    _float_attribute(prim, "isaac:shearForceLimit"),
                    _float_attribute(prim, "isaac:retryInterval"),
                )
                for env_id, exact_path in _exact_destinations(plan, row, clone_mask):
                    exact_path = (exact_path.rstrip("/") + source_suffix) or "/"
                    _add_unique(
                        surface_grippers,
                        exact_path,
                        SurfaceGripperLayout(exact_path, gripper_view_path, row, env_id, *parameters),
                    )
            if not _prim_has_schema(prim, "DeformableBodyAPI"):
                continue
            deformable_type, sim_mesh, vis_mesh, vertex_count = _classify_deformable(prim)
            material = None
            for material_path in UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel("physics").GetTargets():
                candidate = stage.GetPrimAtPath(material_path)
                if candidate.IsA(UsdShade.Material):
                    material = candidate
                    break
            source_material_path = None if material is None else material.GetPath().pathString
            source_sim_path = sim_mesh.GetPath().pathString
            source_vis_path = vis_mesh.GetPath().pathString
            view_path = clone_path.rebase(source_path, source, view_root)
            vertices, indices = _deformable_topology(prim, sim_mesh)
            vis_vertices = _deformable_vertices(prim, vis_mesh)
            point_indices, point_weights = _deformable_point_mapping(
                deformable_type, sim_mesh, vis_mesh, vertices, indices, vis_vertices
            )
            material_attributes = (
                ()
                if material is None
                else tuple(
                    (attribute.GetName(), value)
                    for attribute in material.GetAttributes()
                    if (value := attribute.Get()) is not None
                )
            )
            for env_id, destination in _exact_destinations(plan, row, clone_mask):
                root_path = (destination.rstrip("/") + source_suffix) or "/"
                destination = plan.destinations[row].format(env_id) if env_id is not None else plan.destinations[row]
                entry = DeformableLayout(
                    root_path=root_path,
                    sim_mesh_path=clone_path.rebase(source_sim_path, source, destination),
                    vis_mesh_path=clone_path.rebase(source_vis_path, source, destination),
                    view_path=view_path,
                    material_path=None
                    if source_material_path is None
                    else (
                        clone_path.rebase(source_material_path, source, destination)
                        if clone_path.under(source_material_path, source)
                        else source_material_path
                    ),
                    material_view_path=None
                    if source_material_path is None
                    else (
                        clone_path.rebase(source_material_path, source, view_root)
                        if clone_path.under(source_material_path, source)
                        else source_material_path
                    ),
                    deformable_type=deformable_type,
                    vertex_count=vertex_count,
                    vis_vertex_count=len(vis_vertices),
                    point_indices=point_indices,
                    point_weights=point_weights,
                    row=row,
                    env_id=env_id,
                    source_path=source_path,
                    vertices=vertices,
                    indices=indices,
                    material_attributes=material_attributes,
                )
                _add_unique(deformables, root_path, entry)

        joint_prims = tuple(
            prim
            for prim in prims
            if prim.IsA(UsdPhysics.Joint) and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is not False
        )
        joint_bodies = {
            prim.GetPath().pathString: tuple(
                target.pathString
                for relationship in (UsdPhysics.Joint(prim).GetBody0Rel(), UsdPhysics.Joint(prim).GetBody1Rel())
                for target in relationship.GetTargets()[:1]
            )
            for prim in joint_prims
        }
        claimed_joints: set[str] = set()
        for root_prim in articulation_roots:
            root_source_path = root_prim.GetPath().pathString
            if root_prim.IsA(UsdPhysics.Joint):
                reachable_bodies = set(joint_bodies.get(root_source_path, ()))
            elif root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                reachable_bodies = {root_source_path}
            else:
                reachable_bodies = {
                    body_path
                    for body_paths in joint_bodies.values()
                    for body_path in body_paths
                    if clone_path.under(body_path, root_source_path)
                }
            owned_joint_paths = {root_source_path} if root_source_path in joint_bodies else set()
            changed = True
            while changed:
                changed = False
                for joint_path, body_paths in joint_bodies.items():
                    if joint_path in owned_joint_paths or not reachable_bodies.intersection(body_paths):
                        continue
                    owned_joint_paths.add(joint_path)
                    reachable_bodies.update(body_paths)
                    changed = True
            duplicate_joints = claimed_joints.intersection(owned_joint_paths)
            if duplicate_joints:
                raise ValueError(
                    f"Articulation roots share connected joints in clone-plan row {row}: {sorted(duplicate_joints)!r}."
                )
            claimed_joints.update(owned_joint_paths)
            body_source_paths = tuple(
                prim.GetPath().pathString
                for prim in prims
                if prim.HasAPI(UsdPhysics.RigidBodyAPI)
                and not prim.IsA(UsdPhysics.Joint)
                and prim.GetPath().pathString in reachable_bodies
            )
            if not body_source_paths and not owned_joint_paths:
                body_source_paths = tuple(
                    prim.GetPath().pathString
                    for prim in prims
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    and not prim.IsA(UsdPhysics.Joint)
                    and clone_path.under(prim.GetPath().pathString, root_source_path)
                )
            owned_joints = []
            for joint_prim in sorted(joint_prims, key=lambda prim: prim.GetPath().pathString):
                joint_source_path = joint_prim.GetPath().pathString
                if joint_source_path not in owned_joint_paths:
                    continue
                joint = UsdPhysics.Joint(joint_prim)
                parent_targets = joint.GetBody0Rel().GetTargets()
                child_targets = joint.GetBody1Rel().GetTargets()
                if child_targets:
                    schemas = _get_applied_schema_names(joint_prim)
                    tendon_type = (
                        "fixed"
                        if any("PhysxTendonAxisRootAPI" in name for name in schemas)
                        else (
                            "spatial"
                            if any(
                                token in name
                                for name in schemas
                                for token in ("PhysxTendonAttachmentRootAPI", "PhysxTendonAttachmentLeafAPI")
                            )
                            else None
                        )
                    )
                    owned_joints.append(
                        (
                            joint_source_path,
                            _element_name(joint_prim),
                            None if not parent_targets else parent_targets[0].pathString,
                            child_targets[0].pathString,
                            tendon_type,
                            _joint_pose(joint),
                            joint.GetExcludeFromArticulationAttr().Get() is True,
                        )
                    )
            body_source_paths = _order_articulation_bodies(
                body_source_paths,
                [
                    (parent_path if parent_path in body_source_paths else None, child_path)
                    for _path, _name, parent_path, child_path, _tendon, _pose, excluded in owned_joints
                    if not excluded
                ],
                root_source_path if root_source_path in body_source_paths else None,
            )
            root_view_path = clone_path.rebase(root_source_path, source, view_root)
            root_template = clone_path.rebase(root_source_path, source, plan.destinations[row])
            owned, root_mask = _prototype_mask(plan, row, root_template, row_masks, mask_cache)
            if not owned:
                continue
            joints = tuple(
                JointLayout(
                    path=clone_path.rebase(joint_path, source, plan.destinations[row]),
                    name=name,
                    parent_path=None
                    if parent_path is None
                    else clone_path.rebase(parent_path, source, plan.destinations[row]),
                    child_path=clone_path.rebase(child_path, source, plan.destinations[row]),
                    tendon_type=tendon_type,
                    pose=pose,
                    newton_actuator=actuators_by_joint.get(joint_path),
                )
                for joint_path, name, parent_path, child_path, tendon_type, pose, _excluded in owned_joints
            )
            column = None if root_mask is None else int(np.flatnonzero(root_mask)[0])
            bodies = []
            for body_source_path in body_source_paths:
                body_path = clone_path.rebase(body_source_path, source, plan.destinations[row])
                candidates = tuple(
                    body
                    for body in rigid_bodies
                    if body.path == body_path
                    and (
                        (column is None and body.clone_mask is None)
                        or (column is not None and body.clone_mask is not None and body.clone_mask[column])
                    )
                )
                if len(candidates) != 1:
                    raise ValueError(
                        f"Articulation {root_template!r} resolves {len(candidates)} bodies at {body_path!r}; "
                        "expected exactly one."
                    )
                body = candidates[0]
                if root_mask is not None and np.any(root_mask & ~body.clone_mask):
                    raise ValueError(
                        f"Articulation {root_template!r} has heterogeneous body ownership at {body_path!r}."
                    )
                bodies.append(body)
            articulations.append(
                ArticulationLayout(
                    root_path=root_template,
                    view_path=root_view_path,
                    row=row,
                    joints=joints,
                    bodies=tuple(bodies),
                    clone_mask=root_mask,
                )
            )

    env_ids = () if plan.env_ids is None else tuple(map(int, plan.env_ids))
    positions = None
    if plan.positions is not None:
        positions = np.asarray(plan.positions).copy()
        positions.setflags(write=False)
    env_order = {env_id: index for index, env_id in enumerate(env_ids)}
    source_positions = []
    for source, destination in zip(plan.sources, plan.destinations, strict=True):
        matched = clone_path.match(source, destination)
        source_env = None if matched is None or not matched.instance.isdigit() else int(matched.instance)
        source_positions.append(
            None
            if positions is None or source_env not in env_order
            else tuple(map(float, positions[env_order[source_env], :3]))
        )
    source_positions = tuple(source_positions)

    if requested_frames:
        candidates = replace(
            plan,
            is_complete=True,
            frame_prototypes=tuple(requested_frames),
            _env_ids_cpu=env_ids,
        )
        selected = {
            id(frame)
            for path_expr in plan.geometry_requests
            for frame, _clone_mask in candidates._match_frame_prototypes(path_expr)
        }
        frames.extend(frame for frame in requested_frames if id(frame) in selected)
    frames.sort(key=lambda frame: (frame.row, frame.source_path))
    frames = _bind_frames(plan, stage, tuple(frames), tuple(rigid_bodies))
    prototype_plan = replace(
        plan,
        is_complete=True,
        frame_prototypes=frames,
        _env_ids_cpu=env_ids,
        _positions_cpu=positions,
        _source_positions=source_positions,
    )
    target_frames = tuple(
        (frame, None if clone_mask is None else _intern_mask(clone_mask, mask_cache))
        for path_expr in plan.geometry_requests
        for frame, clone_mask in prototype_plan._match_frame_prototypes(path_expr)
    )

    geometries = []
    frames_by_source = {(frame.row, frame.source_path): frame for frame in frames}
    geometry_data: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, bool, str, str | None]] = {}
    for path, source_path, row, clone_mask, prim, heightfield in geometry_candidates:
        selected_mask = None
        selected = heightfield is not None
        for target, target_mask in target_frames:
            if (clone_mask is None) != (target_mask is None) or not clone_path.under(path, target.path):
                continue
            if clone_mask is None:
                selected = True
                break
            overlap = clone_mask & target_mask
            if np.any(overlap):
                selected_mask = overlap if selected_mask is None else selected_mask | overlap
                selected = True
        if not selected:
            continue
        if selected_mask is not None:
            selected_mask = _intern_mask(selected_mask, mask_cache)
        data = geometry_data.get((row, source_path))
        if data is None:
            collision = _prim_has_schema(prim, "CollisionAPI")
            approximation = (
                UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                if _prim_has_schema(prim, "MeshCollisionAPI")
                else None
            )
            view_path = clone_path.rebase(
                source_path,
                plan.sources[row],
                plan.destinations[row].format("*") if "{}" in plan.destinations[row] else plan.destinations[row],
            )
            data = geometry_data[row, source_path] = (
                *_geometry_data(prim, xform_cache),
                collision,
                view_path,
                approximation,
            )
        geometries.append(
            GeometryLayout(
                path=path,
                source_path=source_path,
                row=row,
                vertices=data[0],
                faces=data[1],
                collision=data[2],
                view_path=data[3],
                collision_approximation=data[4],
                frame=frames_by_source[row, source_path],
                heightfield=heightfield,
                clone_mask=selected_mask,
            ),
        )

    def canonical(entries: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(
            sorted(
                entries.values(), key=lambda entry: (-1 if entry.env_id is None else env_order[entry.env_id], entry.row)
            )
        )

    articulation_prototypes = tuple(sorted(articulations, key=lambda entry: (entry.row, entry.root_path)))
    body_order: dict[int, int] = {}
    for articulation in articulation_prototypes:
        for body in articulation.bodies:
            identity = id(body)
            if identity in body_order:
                raise ValueError(f"Rigid body {body.path!r} belongs to multiple articulations.")
            body_order[identity] = len(body_order)
    ordered_bodies = sorted(
        rigid_bodies,
        key=lambda body: (
            body.row,
            id(body) not in body_order,
            body_order.get(id(body), 0),
        ),
    )

    return replace(
        plan,
        is_complete=True,
        frame_prototypes=frames,
        rigid_body_prototypes=tuple(ordered_bodies),
        geometry_prototypes=tuple(geometries),
        articulation_prototypes=articulation_prototypes,
        deformables=canonical(deformables),
        cables=canonical(cables),
        point_clouds=canonical(point_clouds),
        surface_grippers=canonical(surface_grippers),
        _env_ids_cpu=env_ids,
        _positions_cpu=positions,
        _source_positions=source_positions,
    )
