# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn the in-hand object clear of the hand: the exact lift that separates convex colliders.

The object (convex hull ``A``) moved by ``t`` along the up direction ``e`` overlaps a hand link (convex hull ``B``)
exactly for ``t`` in an open interval ``(t0, t1)``: ``t e`` lies inside the Minkowski difference
``D = B - A``. A plane with unit normal ``n`` separates them once ``t (n . e) >= h_D(n)``, with
``h_D(n) = h_B(n) + h_A(-n)`` the support of ``D``, so ``t1`` is the least ``h_D(n) / (n . e)`` over ``n . e > 0`` and
``t0`` the greatest over ``n . e < 0``. Every normal bounds them; the facet normals of ``D`` reach them exactly: the
face normals of ``B``, the negated face normals of ``A``, and the cross products of an edge of each whose normal arcs
meet. The candidates are those plus, for each edge of ``B``, its cross products with the edge directions of ``A`` that
lie in the edge's normal arc (``h_B`` there is the edge's, ``h_A`` comes from ``A``'s points), a superset of the
facets. The object's lift is the least ``t >= 0`` outside every link's interval.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.sim import SimulationContext

_FAR = 1.0e9
"""Bound of an empty overlap interval [m]."""

_CHUNK = 64
"""Candidate normals one thread folds before its one atomic update of the interval."""


@wp.struct
class ConvexHulls:
    """Convex hulls packed for the clearance kernels, each in its body frame [m]: the hull vertices, the face planes
    (outward unit normal, offset) and the feature edges (point, unit direction, the two adjacent face normals), with
    each hull's ``(start, count)`` of points, faces and edges and its bounding sphere (centre, radius)."""

    points: wp.array(dtype=wp.vec3f)
    normals: wp.array(dtype=wp.vec3f)
    offsets: wp.array(dtype=wp.float32)
    edges: wp.array2d(dtype=wp.vec3f)
    ranges: wp.array2d(dtype=wp.vec2i)
    centres: wp.array(dtype=wp.vec3f)
    radii: wp.array(dtype=wp.float32)


def convex_hulls(clouds: list[np.ndarray], device: str) -> tuple[ConvexHulls, np.ndarray]:
    """Pack the convex hulls of point sets.

    Args:
        clouds: Each hull's points in its body frame [m], shape ``(K, 3)``.
        device: Device of the packed arrays.

    Returns:
        The packed hulls and each hull's ``(start, count)`` of points, faces and edges, shape ``(hulls, 3, 2)``.
    """
    from scipy.spatial import ConvexHull

    points, planes, edges = [], [], []
    ranges = np.zeros((len(clouds), 3, 2), dtype=np.int32)
    for index, cloud in enumerate(clouds):
        hull = ConvexHull(np.asarray(cloud, dtype=np.float64))
        # coplanar triangles share one face plane, and the edges between them are no features
        plane_of = np.unique(np.round(hull.equations, 9), axis=0, return_inverse=True)[1].reshape(-1)
        hull_planes = np.zeros((plane_of.max() + 1, 4))
        hull_planes[plane_of] = hull.equations
        adjacent: dict[tuple[int, int], list[int]] = {}
        for triangle, simplex in enumerate(hull.simplices):
            for a, b in ((0, 1), (1, 2), (2, 0)):
                key = (min(simplex[a], simplex[b]), max(simplex[a], simplex[b]))
                adjacent.setdefault(key, []).append(int(plane_of[triangle]))
        hull_edges = []
        for (a, b), (p, q) in adjacent.items():
            if p != q:
                direction = (hull.points[b] - hull.points[a]) / np.linalg.norm(hull.points[b] - hull.points[a])
                hull_edges.append((hull.points[a], direction, hull_planes[p, :3], hull_planes[q, :3]))
        for kind, (store, items) in enumerate(
            ((points, hull.points[hull.vertices]), (planes, hull_planes), (edges, hull_edges))
        ):
            ranges[index, kind] = (len(store), len(items))
            store.extend(items)
    planes = np.asarray(planes, dtype=np.float32)
    packed = ConvexHulls()
    packed.points = wp.array(np.asarray(points, dtype=np.float32), dtype=wp.vec3f, device=device)
    packed.normals = wp.array(planes[:, :3], dtype=wp.vec3f, device=device)
    packed.offsets = wp.array(-planes[:, 3], dtype=wp.float32, device=device)
    packed.edges = wp.array(np.asarray(edges, dtype=np.float32).reshape(-1, 4, 3), dtype=wp.vec3f, device=device)
    packed.ranges = wp.array(ranges, dtype=wp.vec2i, device=device)
    clouds = [np.asarray(points[start : start + count], dtype=np.float64) for start, count in ranges[:, 0]]
    centres = np.array([0.5 * (cloud.min(0) + cloud.max(0)) for cloud in clouds])
    radii = [np.linalg.norm(cloud - centre, axis=1).max() for cloud, centre in zip(clouds, centres)]
    packed.centres = wp.array(centres.astype(np.float32), dtype=wp.vec3f, device=device)
    packed.radii = wp.array(np.asarray(radii, dtype=np.float32), dtype=wp.float32, device=device)
    return packed, ranges


@wp.func
def _in_arc(n: wp.vec3f, n1: wp.vec3f, n2: wp.vec3f) -> bool:
    """Whether ``n`` lies strictly inside the normal arc of an edge between face normals ``n1`` and ``n2``."""
    c = wp.cross(n1, n2)
    return wp.dot(wp.cross(n, n2), c) > 0.0 and wp.dot(wp.cross(n1, n), c) > 0.0


@wp.kernel(enable_backward=False)
def _overlap_interval_kernel(
    object_pose: wp.array(dtype=wp.transformf),
    link_pose: wp.array2d(dtype=wp.transformf),
    links: ConvexHulls,
    body: ConvexHulls,
    directions: wp.array(dtype=wp.vec3f),
    up: wp.vec3f,
    chunk: int,
    t0: wp.array2d(dtype=wp.float32),
    t1: wp.array2d(dtype=wp.float32),
):
    world, link, c = wp.tid()
    # everything in the link's frame: the object's rotation and origin there, and the lift direction
    qb = wp.transform_get_rotation(link_pose[world, link])
    q = wp.quat_inverse(qb) * wp.transform_get_rotation(object_pose[world])
    origin = wp.quat_rotate_inv(
        qb, wp.transform_get_translation(object_pose[world]) - wp.transform_get_translation(link_pose[world, link])
    )
    lift = wp.quat_rotate_inv(qb, up)
    # bounding spheres that pass each other along the lift line never overlap: the interval is empty
    gap = links.centres[link] - wp.quat_rotate(q, body.centres[0]) - origin
    if wp.length(gap - wp.dot(gap, lift) * lift) > links.radii[link] + body.radii[0]:
        if c == 0:
            t0[world, link] = _FAR
        return
    pa, fa = body.ranges[0, 0], body.ranges[0, 1]
    fb, eb = links.ranges[link, 1], links.ranges[link, 2]
    lower = float(-_FAR)
    upper = float(_FAR)
    cached = int(-1)  # the object's edge direction last brought into the link's frame
    da = wp.vec3f(0.0)
    for k in range(c * chunk, wp.min(fb[1] + fa[1] + directions.shape[0] * eb[1], (c + 1) * chunk)):
        # the k-th candidate normal n and the support h of the Minkowski difference (link minus object) along it
        n = wp.vec3f(0.0)
        h = float(0.0)
        if k < fb[1] + fa[1]:
            if k < fb[1]:  # a face of the link
                n = links.normals[fb[0] + k]
                h = links.offsets[fb[0] + k]
            else:  # a face of the object, negated
                face = fa[0] + k - fb[1]
                n = -wp.quat_rotate(q, body.normals[face])
                h = float(-_FAR)
                for j in range(links.ranges[link, 0][1]):
                    h = wp.max(h, wp.dot(n, links.points[links.ranges[link, 0][0] + j]))
        else:  # an edge of the link and an edge direction of the object, where the link edge's normal arc holds it
            pair = k - fb[1] - fa[1]
            j = eb[0] + pair % eb[1]
            if pair // eb[1] != cached:
                cached = pair // eb[1]
                da = wp.quat_rotate(q, directions[cached])
            axis = wp.cross(da, links.edges[j, 1])
            length = wp.length(axis)
            if length > 1.0e-6:
                axis = axis / length
                b1, b2 = links.edges[j, 2], links.edges[j, 3]
                if not _in_arc(axis, b1, b2):
                    axis = -axis
                if _in_arc(axis, b1, b2):
                    n = axis
                    h = wp.dot(n, links.edges[j, 0])
        if wp.length(n) > 0.0:
            # the object's support against n, from its own frame
            m = wp.quat_rotate_inv(q, n)
            support = float(-_FAR)
            for i in range(pa[1]):
                support = wp.max(support, -wp.dot(m, body.points[pa[0] + i]))
            h += support - wp.dot(n, origin)
        # a plane of normal n separates once t (n . lift) >= h
        s = wp.dot(n, lift)
        if s > 1.0e-6:
            upper = wp.min(upper, h / s)
        elif s < -1.0e-6:
            lower = wp.max(lower, h / s)
        elif wp.length(n) > 0.0 and h < 0.0:
            lower = _FAR  # a plane along the lift separates at every t
    if upper < _FAR:
        wp.atomic_min(t1, world, link, upper)
    if lower > -_FAR:
        wp.atomic_max(t0, world, link, lower)


@wp.kernel(enable_backward=False)
def _clear_lift_kernel(
    t0: wp.array2d(dtype=wp.float32), t1: wp.array2d(dtype=wp.float32), lift: wp.array(dtype=wp.float32)
):
    world = wp.tid()
    links = t0.shape[1]
    t = float(0.0)
    for _ in range(links):
        for link in range(links):
            if t0[world, link] < t and t < t1[world, link]:
                t = t1[world, link]
    lift[world] = t


def collider_points(prim_path_expr: str) -> dict[str, np.ndarray]:
    """The authored collider points of each rigid body under the first prim matching ``prim_path_expr``.

    Args:
        prim_path_expr: Prim path expression of an asset, e.g. ``/World/envs/env_.*/Robot``.

    Returns:
        Each rigid body's collider points in its body frame [m], scale applied, by body prim name; the convex hull
        of a body's points contains each of its colliders.

    Raises:
        NotImplementedError: For a collider other than a mesh, cube, sphere, capsule or cylinder.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    import isaaclab.sim as sim_utils

    cache = UsdGeom.XformCache()
    clouds: dict[str, list[np.ndarray]] = {}
    for prim in Usd.PrimRange(sim_utils.find_first_matching_prim(prim_path_expr), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        body = prim
        while not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        local = _enclosing_points(prim)
        to_world = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)  # row vectors
        frame = Gf.Matrix4d(cache.GetLocalToWorldTransform(body))
        rotation = np.asarray(frame.ExtractRotationMatrix(), dtype=np.float64)
        rotation /= np.linalg.norm(rotation, axis=1, keepdims=True)  # the body frame without its scale
        world = local @ to_world[:3, :3] + to_world[3, :3]
        clouds.setdefault(body.GetName(), []).append((world - np.asarray(frame.ExtractTranslation())) @ rotation.T)
    return {name: np.concatenate(parts) for name, parts in clouds.items()}


def _enclosing_points(prim) -> np.ndarray:
    """Points in a collider prim's frame [m] whose convex hull contains the collider: a mesh's own points, a cube's
    corners, and for round shapes the vertices of polytopes circumscribing their spheres and circles."""
    from pxr import UsdGeom

    if prim.IsA(UsdGeom.Mesh):
        return np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
    if prim.IsA(UsdGeom.Cube):
        return 0.5 * UsdGeom.Cube(prim).GetSizeAttr().Get() * _CORNERS
    if prim.IsA(UsdGeom.Sphere):
        return UsdGeom.Sphere(prim).GetRadiusAttr().Get() * _ball()
    for kind, round_end in ((UsdGeom.Capsule, True), (UsdGeom.Cylinder, False)):
        if prim.IsA(kind):
            shape = kind(prim)
            radius, height = shape.GetRadiusAttr().Get(), shape.GetHeightAttr().Get()
            axis = "XYZ".index(shape.GetAxisAttr().Get())
            end = np.zeros(3)
            end[axis] = 0.5 * height
            if round_end:
                cap = radius * _ball()
            else:
                # a regular polygon circumscribing the end circle, in the plane normal to the axis
                angle = np.linspace(0.0, 2.0 * np.pi, _SIDES, endpoint=False)
                cap = np.zeros((_SIDES, 3))
                cap[:, (axis + 1) % 3] = np.cos(angle)
                cap[:, (axis + 2) % 3] = np.sin(angle)
                cap *= radius / np.cos(np.pi / _SIDES)
            return np.concatenate((cap + end, cap - end))
    raise NotImplementedError(f"No clearance geometry for the {prim.GetTypeName()} collider {prim.GetPath()}.")


_CORNERS = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])
_SIDES = 64
"""Sides of the polygon circumscribing a cylinder's end circle: within 0.12% of its radius."""


@functools.cache
def _ball() -> np.ndarray:
    """Vertices of a polytope circumscribing the unit ball (a twice-subdivided icosahedron, scaled so its faces touch
    the ball): within 1.8% of the radius."""
    from scipy.spatial import ConvexHull

    phi = 0.5 * (1.0 + np.sqrt(5.0))
    vertices = np.array(
        [[0.0, a, b] for a in (-1.0, 1.0) for b in (-phi, phi)]
        + [[a, b, 0.0] for a in (-1.0, 1.0) for b in (-phi, phi)]
        + [[b, 0.0, a] for a in (-1.0, 1.0) for b in (-phi, phi)]
    )
    for _ in range(2):  # split every edge at its midpoint, back on the sphere
        edges = [(f[i], f[(i + 1) % 3]) for f in ConvexHull(vertices).simplices for i in range(3)]
        vertices = np.concatenate((vertices, [vertices[a] + vertices[b] for a, b in edges]))
        vertices = np.unique(np.round(vertices / np.linalg.norm(vertices, axis=1, keepdims=True), 12), axis=0)
    inradius = -ConvexHull(vertices).equations[:, 3].max()
    return vertices / inradius


class ConvexClearance:
    """The least lift along an up direction that brings a convex object clear of a set of convex links."""

    def __init__(self, links: list[np.ndarray], body: np.ndarray, up: tuple[float, float, float], device: str):
        """Pack the hulls.

        Args:
            links: Each link's collider points in its body frame [m], shape ``(K, 3)``.
            body: The object's collider points in its body frame [m], shape ``(K, 3)``.
            up: The lift direction (unit), world frame.
            device: Device of the queries.
        """
        self._links, link_ranges = convex_hulls(links, device)
        self._object, object_ranges = convex_hulls([body], device)
        # the object's edge directions, each once: an edge pair's normal is their cross product, and the object's
        # support along it comes from its points, so parallel edges of the object give the same candidate
        directions = np.round(self._object.edges.numpy()[:, 1], 6)
        flip = directions @ np.array([4.0, 2.0, 1.0]) < 0.0  # one sign a direction
        directions = np.unique(np.where(flip[:, None], -directions, directions), axis=0)
        self._directions = wp.array(directions.astype(np.float32), dtype=wp.vec3f, device=device)
        # each link's candidate normals: its faces, the object's faces, and its edges against the object's directions
        faces, edges = link_ranges[:, 1, 1], link_ranges[:, 2, 1]
        candidates = int((faces + object_ranges[0, 1, 1] + len(directions) * edges).max())
        self._chunks = -(-candidates // _CHUNK)
        self._up = wp.vec3f(*up)
        self._device = device

    def lift(self, object_pose: torch.Tensor, link_pose: torch.Tensor) -> torch.Tensor:
        """The least distance along the up direction that clears the object of every link.

        Args:
            object_pose: The object's drawn pose ``[pos, quat (x, y, z, w)]`` [m], shape ``(N, 7)``.
            link_pose: The links' poses, same layout, shape ``(N, B, 7)``.

        Returns:
            The lift [m], shape ``(N,)``; zero where the object is already clear.
        """
        count, links = link_pose.shape[:2]
        lift = torch.zeros(count, dtype=torch.float32, device=self._device)
        if count == 0:
            return lift
        t0 = torch.full((count, links), -_FAR, dtype=torch.float32, device=self._device)
        t1 = torch.full((count, links), _FAR, dtype=torch.float32, device=self._device)
        wp.launch(
            _overlap_interval_kernel,
            dim=(count, links, self._chunks),
            inputs=[
                wp.from_torch(object_pose.float().contiguous(), dtype=wp.transformf),
                wp.from_torch(link_pose.float().contiguous(), dtype=wp.transformf),
                self._links,
                self._object,
                self._directions,
                self._up,
                _CHUNK,
            ],
            outputs=[wp.from_torch(t0), wp.from_torch(t1)],
            device=self._device,
        )
        wp.launch(
            _clear_lift_kernel,
            dim=count,
            inputs=[wp.from_torch(t0), wp.from_torch(t1)],
            outputs=[wp.from_torch(lift)],
            device=self._device,
        )
        return lift


class ObjectClearance:
    """Lift an in-hand object's drawn pose against gravity, clear of every collider of the hand.

    The geometry is the authored colliders' convex hulls, read once from the first environment: each engine's convex
    collider is the hull of these points or of a subset of them, so an object clear of these hulls is clear of the
    engine's.
    """

    def __init__(self, hand: Articulation, obj: RigidObject, gravity: tuple[float, float, float]):
        """Read the hand's and the object's colliders.

        Args:
            hand: The hand.
            obj: The object to place.
            gravity: Gravity [m/s^2]; the object is lifted against it.

        Raises:
            ValueError: If gravity is zero, which leaves no up direction, or the object is not one rigid body.
        """
        norm = float(np.linalg.norm(gravity))
        if norm == 0.0:
            raise ValueError("The object is lifted against gravity, which is zero.")
        up = tuple(-float(g) / norm for g in gravity)
        object_points = list(collider_points(obj.cfg.prim_path).values())
        if len(object_points) != 1:
            raise ValueError(f"Expected one rigid body under {obj.cfg.prim_path}, found {len(object_points)}.")
        hand_points = collider_points(hand.cfg.prim_path)
        self._hand = hand
        self._body_ids = [index for index, name in enumerate(hand.body_names) if name in hand_points]
        links = [hand_points[hand.body_names[index]] for index in self._body_ids]
        self._clearance = ConvexClearance(links, object_points[0], up, hand.device)
        self._up = torch.tensor(up, dtype=torch.float32, device=hand.device)

    def raise_clear(self, pose: torch.Tensor, env_ids: torch.Tensor, sim: SimulationContext) -> None:
        """Lift drawn object poses, in place, clear of the hand at the joint positions just written to ``sim``.

        Args:
            pose: The object's drawn poses ``[pos, quat (x, y, z, w)]`` in the world frame [m], shape ``(N, 7)``.
            env_ids: The environments of the poses, shape ``(N,)``.
            sim: The simulation the hand's reset wrote its joint positions to.
        """
        sim.forward()  # the hand's links at the joint positions its reset wrote
        links = self._hand.data.body_link_pose_w.torch[env_ids][:, self._body_ids]
        pose[:, :3] += self._clearance.lift(pose, links)[:, None] * self._up
