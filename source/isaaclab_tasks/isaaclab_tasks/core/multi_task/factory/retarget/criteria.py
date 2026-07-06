# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-free signed-distance criteria for Factory task tables.

* :func:`collision_min_sd` -- probes on FK bodies vs a STATIC world mesh
  (gripper vs fixed asset or table).
* :func:`posed_collision_min_sd` -- probes on FK bodies vs a PER-PROBLEM-POSED
  mesh (gripper vs the held asset, whose pose is sampler data, not FK).
* :func:`points_min_sd` -- world points vs a static mesh.
* :func:`self_collision_min_sd` -- robot link-vs-link with a kinematic-adjacency
  filter.
"""

from __future__ import annotations

import torch
import warp as wp

import isaaclab.utils.math as math_utils


@wp.kernel
def _measure_grasp_targets(
    body_q: wp.array2d(dtype=wp.transformf),
    pad_bodies: wp.array1d(dtype=wp.int32),
    pad_offsets: wp.array1d(dtype=wp.vec3),
    target_plus: wp.array1d(dtype=wp.vec3),
    target_minus: wp.array1d(dtype=wp.vec3),
    ee_body: int,
    target_error_m: wp.array1d(dtype=wp.float32),
    ee_approach: wp.array1d(dtype=wp.vec3),
):
    candidate = wp.tid()
    plus = wp.transform_point(body_q[candidate, pad_bodies[0]], pad_offsets[0])
    minus = wp.transform_point(body_q[candidate, pad_bodies[1]], pad_offsets[1])
    plus_error = wp.length(plus - target_plus[candidate])
    minus_error = wp.length(minus - target_minus[candidate])
    target_error_m[candidate] = wp.max(plus_error, minus_error)
    ee_rotation = wp.transform_get_rotation(body_q[candidate, ee_body])
    ee_approach[candidate] = wp.quat_rotate(ee_rotation, wp.vec3(0.0, 0.0, 1.0))


def measure_grasp_targets(
    body_q: torch.Tensor,
    pad_bodies: wp.array,
    pad_offsets: wp.array,
    target_plus: torch.Tensor,
    target_minus: torch.Tensor,
    ee_body: int,
    target_error_m: torch.Tensor,
    ee_approach: torch.Tensor,
    device: str,
) -> None:
    """Measure fingertip target error [m] and end-effector approach without temporaries."""
    count = body_q.shape[0]
    wp.launch(
        _measure_grasp_targets,
        dim=count,
        inputs=(
            wp.from_torch(body_q, dtype=wp.transformf),
            pad_bodies,
            pad_offsets,
            wp.from_torch(target_plus, dtype=wp.vec3),
            wp.from_torch(target_minus, dtype=wp.vec3),
            ee_body,
        ),
        outputs=(wp.from_torch(target_error_m), wp.from_torch(ee_approach, dtype=wp.vec3)),
        device=device,
    )


def posed_points(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """Pose canonical-frame points per problem: ``[P, 3]`` x ``[N, 7]`` -> world ``[N, P, 3]``.

    Args:
        points: Points in the asset's canonical frame [m], shape ``[P, 3]``.
        pose: Per-problem asset poses (pos [m] + quat xyzw), shape ``[N, 7]``.
    """
    n, p = pose.shape[0], points.shape[0]
    quat = pose[:, 3:7].unsqueeze(1).expand(-1, p, 4).reshape(-1, 4)
    pts = points.unsqueeze(0).expand(n, -1, 3).reshape(-1, 3)
    return math_utils.quat_apply(quat, pts).view(n, p, 3) + pose[:, :3].unsqueeze(1)


@wp.kernel
def _probes_vs_mesh_sdf(
    body_q: wp.array2d(dtype=wp.transformf),  # [N, B] FK body transforms
    probe_body: wp.array1d(dtype=wp.int32),  # [P] body each probe is attached to
    probes: wp.array1d(dtype=wp.vec3),  # [P] offsets in their probe-body frame
    mesh_id: wp.uint64,  # static world-frame obstacle mesh
    max_dist: float,
    out_min_sd: wp.array1d(dtype=wp.float32),  # [N], pre-filled large +; atomic-min
):
    cand, p = wp.tid()
    world_p = wp.transform_point(body_q[cand, probe_body[p]], probes[p])
    q = wp.mesh_query_point(mesh_id, world_p, max_dist)
    if q.result:
        cp = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
        sd = q.sign * wp.length(world_p - cp)  # <0 inside the obstacle = penetration
        wp.atomic_min(out_min_sd, cand, sd)


def collision_min_sd(
    body_q: torch.Tensor,
    probe_body: wp.array,
    probes: wp.array,
    mesh_id: int,
    max_dist: float,
    device: str,
) -> torch.Tensor:
    """Per-candidate min signed distance from a probe set to the obstacle [N] (>=0 clear, <0 penetration).

    Args:
        body_q: FK body transforms [N, body_count, 7] (pos + xyzw).
        probe_body: Body index each probe is attached to, ``wp.array`` of ``int32`` [P].
        probes: Probe offsets in their probe-body frame, ``wp.array`` of ``wp.vec3`` [P].
        mesh_id: Static world-frame obstacle :class:`warp.Mesh` id.
        max_dist: ``wp.mesh_query_point`` search radius [m].
        device: Warp/torch device string.
    """
    n = body_q.shape[0]
    body_q_wp = wp.from_torch(body_q.contiguous(), dtype=wp.transformf)
    out = wp.full(n, 1.0e6, dtype=wp.float32, device=device)
    wp.launch(
        _probes_vs_mesh_sdf,
        dim=[n, probes.shape[0]],
        inputs=[body_q_wp, probe_body, probes, wp.uint64(mesh_id), max_dist],
        outputs=[out],
        device=device,
    )
    return wp.to_torch(out)


@wp.kernel
def _probes_vs_posed_mesh_sdf(
    body_q: wp.array2d(dtype=wp.transformf),  # [N, B] FK body transforms
    probe_body: wp.array1d(dtype=wp.int32),  # [P] body each probe is attached to
    probes: wp.array1d(dtype=wp.vec3),  # [P] offsets in their probe-body frame
    mesh_id: wp.uint64,  # obstacle mesh in ITS OWN frame
    obstacle_q: wp.array1d(dtype=wp.transformf),  # [N] per-problem obstacle pose
    max_dist: float,
    out_min_sd: wp.array1d(dtype=wp.float32),  # [N], pre-filled large +; atomic-min
):
    cand, p = wp.tid()
    world_p = wp.transform_point(body_q[cand, probe_body[p]], probes[p])
    local = wp.transform_point(wp.transform_inverse(obstacle_q[cand]), world_p)
    q = wp.mesh_query_point(mesh_id, local, max_dist)
    if q.result:
        cp = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
        sd = q.sign * wp.length(local - cp)  # <0 inside the obstacle = penetration
        wp.atomic_min(out_min_sd, cand, sd)


def posed_collision_min_sd(
    body_q: torch.Tensor,
    probe_body: wp.array,
    probes: wp.array,
    mesh_id: int,
    obstacle_pose: torch.Tensor,
    max_dist: float,
    device: str,
) -> torch.Tensor:
    """Per-candidate min signed distance from a probe set to a per-problem-posed obstacle [N].

    Args:
        body_q: FK body transforms [N, body_count, 7] (pos + xyzw).
        probe_body: Body index each probe is attached to, ``wp.array`` of ``int32`` [P].
        probes: Probe offsets in their probe-body frame, ``wp.array`` of ``wp.vec3`` [P].
        mesh_id: Obstacle :class:`warp.Mesh` id in the obstacle's own frame.
        obstacle_pose: Per-problem obstacle pose [N, 7] (pos [m] + quat xyzw).
        max_dist: ``wp.mesh_query_point`` search radius [m].
        device: Warp/torch device string.
    """
    n = body_q.shape[0]
    body_q_wp = wp.from_torch(body_q.contiguous(), dtype=wp.transformf)
    pose_wp = wp.from_torch(obstacle_pose.contiguous(), dtype=wp.transformf)
    out = wp.full(n, 1.0e6, dtype=wp.float32, device=device)
    wp.launch(
        _probes_vs_posed_mesh_sdf,
        dim=[n, probes.shape[0]],
        inputs=[body_q_wp, probe_body, probes, wp.uint64(mesh_id), pose_wp, max_dist],
        outputs=[out],
        device=device,
    )
    return wp.to_torch(out)


@wp.kernel
def _points_vs_mesh_sdf(
    points: wp.array2d(dtype=wp.vec3),  # [N, P] world points
    mesh_id: wp.uint64,
    max_dist: float,
    out_min_sd: wp.array1d(dtype=wp.float32),  # [N]
):
    cand, p = wp.tid()
    q = wp.mesh_query_point(mesh_id, points[cand, p], max_dist)
    if q.result:
        cp = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
        sd = q.sign * wp.length(points[cand, p] - cp)
        wp.atomic_min(out_min_sd, cand, sd)


def points_min_sd(points: torch.Tensor, mesh_id: int, max_dist: float, device: str) -> torch.Tensor:
    """Per-candidate min signed distance from world points to a static mesh [N].

    Args:
        points: World probe points [N, P, 3].
        mesh_id: Static world-frame obstacle :class:`warp.Mesh` id.
        max_dist: ``wp.mesh_query_point`` search radius [m].
        device: Warp/torch device string.
    """
    n = points.shape[0]
    pts_wp = wp.from_torch(points.contiguous(), dtype=wp.vec3)
    out = wp.full(n, 1.0e6, dtype=wp.float32, device=device)
    wp.launch(
        _points_vs_mesh_sdf,
        dim=[n, points.shape[1]],
        inputs=[pts_wp, wp.uint64(mesh_id), max_dist],
        outputs=[out],
        device=device,
    )
    return wp.to_torch(out)


@wp.kernel
def _posed_points_vs_mesh_sdf(
    points: wp.array1d(dtype=wp.vec3),
    point_pose: wp.array1d(dtype=wp.transformf),
    mesh_id: wp.uint64,
    max_dist: float,
    out_min_sd: wp.array1d(dtype=wp.float32),
):
    candidate, point = wp.tid()
    world_point = wp.transform_point(point_pose[candidate], points[point])
    query = wp.mesh_query_point(mesh_id, world_point, max_dist)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        distance = query.sign * wp.length(world_point - closest)
        wp.atomic_min(out_min_sd, candidate, distance)


def posed_points_min_sd(
    points: torch.Tensor,
    point_pose: torch.Tensor,
    mesh_id: int,
    max_dist: float,
    device: str,
) -> torch.Tensor:
    """Return posed canonical-point distance to one static mesh [m], shape [candidate_count]."""
    count = point_pose.shape[0]
    points_wp = wp.from_torch(points, dtype=wp.vec3)
    point_pose_wp = wp.from_torch(point_pose, dtype=wp.transformf)
    output = wp.full(count, 1.0e6, dtype=wp.float32, device=device)
    wp.launch(
        _posed_points_vs_mesh_sdf,
        dim=(count, points.shape[0]),
        inputs=(points_wp, point_pose_wp, wp.uint64(mesh_id), max_dist),
        outputs=(output,),
        device=device,
    )
    return wp.to_torch(output)


@wp.kernel
def _posed_points_vs_posed_mesh_sdf(
    points: wp.array1d(dtype=wp.vec3),
    point_pose: wp.array1d(dtype=wp.transformf),
    mesh_id: wp.uint64,
    obstacle_pose: wp.array1d(dtype=wp.transformf),
    max_dist: float,
    out_min_sd: wp.array1d(dtype=wp.float32),
):
    candidate, point = wp.tid()
    world_point = wp.transform_point(point_pose[candidate], points[point])
    local_point = wp.transform_point(wp.transform_inverse(obstacle_pose[candidate]), world_point)
    query = wp.mesh_query_point(mesh_id, local_point, max_dist)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        distance = query.sign * wp.length(local_point - closest)
        wp.atomic_min(out_min_sd, candidate, distance)


def posed_points_vs_posed_mesh_min_sd(
    points: torch.Tensor,
    point_pose: torch.Tensor,
    mesh_id: int,
    obstacle_pose: torch.Tensor,
    max_dist: float,
    device: str,
) -> torch.Tensor:
    """Return posed canonical-point distance to per-candidate posed meshes [m]."""
    count = point_pose.shape[0]
    points_wp = wp.from_torch(points, dtype=wp.vec3)
    point_pose_wp = wp.from_torch(point_pose, dtype=wp.transformf)
    obstacle_pose_wp = wp.from_torch(obstacle_pose, dtype=wp.transformf)
    output = wp.full(count, 1.0e6, dtype=wp.float32, device=device)
    wp.launch(
        _posed_points_vs_posed_mesh_sdf,
        dim=(count, points.shape[0]),
        inputs=(points_wp, point_pose_wp, wp.uint64(mesh_id), obstacle_pose_wp, max_dist),
        outputs=(output,),
        device=device,
    )
    return wp.to_torch(output)


@wp.kernel
def _points_vs_body_meshes_sdf(
    points: wp.array2d(dtype=wp.vec3),  # [N, P] world points
    body_q: wp.array2d(dtype=wp.transformf),  # [N, B] FK body transforms
    target_body: wp.array1d(dtype=wp.int32),  # [T] body each target shape is on
    target_mesh: wp.array1d(dtype=wp.uint64),  # [T] shape-local mesh ids
    target_tf: wp.array1d(dtype=wp.transformf),  # [T] shape-local transform
    max_dist: float,
    out_min_sd: wp.array1d(dtype=wp.float32),  # [N]
):
    cand, p, t = wp.tid()
    target_world = wp.transform_multiply(body_q[cand, target_body[t]], target_tf[t])
    local = wp.transform_point(wp.transform_inverse(target_world), points[cand, p])
    q = wp.mesh_query_point(target_mesh[t], local, max_dist)
    if q.result:
        cp = wp.mesh_eval_position(target_mesh[t], q.face, q.u, q.v)
        sd = q.sign * wp.length(local - cp)  # <0 inside the body's collider
        wp.atomic_min(out_min_sd, cand, sd)


def points_vs_body_meshes_min_sd(
    points: torch.Tensor,
    body_q: torch.Tensor,
    target_body: wp.array,
    target_mesh: wp.array,
    target_tf: wp.array,
    max_dist: float,
    device: str,
) -> torch.Tensor:
    """Per-candidate min signed distance from world points to FK-posed body colliders [N].

    The reverse direction of :func:`posed_collision_min_sd`: point-vs-mesh queries
    are one-directional, so checking gripper probes against the held mesh misses a
    held-asset corner poking into a gripper face between probes -- this query sees
    it from the held asset's side.

    Args:
        points: World probe points [N, P, 3].
        body_q: FK body transforms [N, body_count, 7] (pos + xyzw).
        target_body: Body index per target shape, ``wp.array`` of ``int32`` [T].
        target_mesh: Shape-local collision-mesh ids, ``wp.array`` of ``uint64`` [T].
        target_tf: Shape-local transforms, ``wp.array`` of ``wp.transformf`` [T].
        max_dist: ``wp.mesh_query_point`` search radius [m].
        device: Warp/torch device string.
    """
    n = points.shape[0]
    pts_wp = wp.from_torch(points.contiguous(), dtype=wp.vec3)
    body_q_wp = wp.from_torch(body_q.contiguous(), dtype=wp.transformf)
    out = wp.full(n, 1.0e6, dtype=wp.float32, device=device)
    wp.launch(
        _points_vs_body_meshes_sdf,
        dim=[n, points.shape[1], target_body.shape[0]],
        inputs=[pts_wp, body_q_wp, target_body, target_mesh, target_tf, max_dist],
        outputs=[out],
        device=device,
    )
    return wp.to_torch(out)


@wp.kernel
def _posed_points_vs_body_meshes_sdf(
    points: wp.array1d(dtype=wp.vec3),
    point_pose: wp.array1d(dtype=wp.transformf),
    body_q: wp.array2d(dtype=wp.transformf),
    target_body: wp.array1d(dtype=wp.int32),
    target_mesh: wp.array1d(dtype=wp.uint64),
    target_tf: wp.array1d(dtype=wp.transformf),
    max_dist: float,
    out_min_sd: wp.array1d(dtype=wp.float32),
):
    candidate, point, target = wp.tid()
    world_point = wp.transform_point(point_pose[candidate], points[point])
    target_world = wp.transform_multiply(body_q[candidate, target_body[target]], target_tf[target])
    local_point = wp.transform_point(wp.transform_inverse(target_world), world_point)
    query = wp.mesh_query_point(target_mesh[target], local_point, max_dist)
    if query.result:
        closest = wp.mesh_eval_position(target_mesh[target], query.face, query.u, query.v)
        distance = query.sign * wp.length(local_point - closest)
        wp.atomic_min(out_min_sd, candidate, distance)


def posed_points_vs_body_meshes_min_sd(
    points: torch.Tensor,
    point_pose: torch.Tensor,
    body_q: torch.Tensor,
    target_body: wp.array,
    target_mesh: wp.array,
    target_tf: wp.array,
    max_dist: float,
    device: str,
) -> torch.Tensor:
    """Return posed canonical-point distance to FK-posed body meshes [m]."""
    count = point_pose.shape[0]
    points_wp = wp.from_torch(points, dtype=wp.vec3)
    point_pose_wp = wp.from_torch(point_pose, dtype=wp.transformf)
    body_q_wp = wp.from_torch(body_q, dtype=wp.transformf)
    output = wp.full(count, 1.0e6, dtype=wp.float32, device=device)
    wp.launch(
        _posed_points_vs_body_meshes_sdf,
        dim=(count, points.shape[0], target_body.shape[0]),
        inputs=(points_wp, point_pose_wp, body_q_wp, target_body, target_mesh, target_tf, max_dist),
        outputs=(output,),
        device=device,
    )
    return wp.to_torch(output)


@wp.kernel
def _edges_vs_posed_mesh_hit(
    body_q: wp.array2d(dtype=wp.transformf),  # [N, B]
    edge_body: wp.array1d(dtype=wp.int32),  # [E] body each edge lives on
    edge_p0: wp.array1d(dtype=wp.vec3),  # [E] endpoints in the edge-body frame
    edge_p1: wp.array1d(dtype=wp.vec3),
    mesh_id: wp.uint64,  # obstacle mesh in ITS OWN frame
    obstacle_q: wp.array1d(dtype=wp.transformf),  # [N] per-problem obstacle pose
    out_hit: wp.array1d(dtype=wp.uint8),  # [N]
):
    cand, e = wp.tid()
    inv = wp.transform_inverse(obstacle_q[cand])
    a = wp.transform_point(inv, wp.transform_point(body_q[cand, edge_body[e]], edge_p0[e]))
    b = wp.transform_point(inv, wp.transform_point(body_q[cand, edge_body[e]], edge_p1[e]))
    d = b - a
    length = wp.length(d)
    if length < 1.0e-9:
        return
    q = wp.mesh_query_ray(mesh_id, a, d / length, length)
    if q.result:
        out_hit[cand] = wp.uint8(1)


def edges_vs_posed_mesh_hit(
    body_q: torch.Tensor,
    edge_body: wp.array,
    edge_p0: wp.array,
    edge_p1: wp.array,
    mesh_id: int,
    obstacle_pose: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Per-candidate surface-crossing test: collider edges vs a posed obstacle mesh [N], bool.

    Point probes miss thin obstacles (e.g. a 4 mm board) slicing between them; if
    two surfaces intersect at all, some collider edge of one crosses the surface of
    the other, so an edge raycast detects any crossing regardless of probe density.
    Containment (one mesh fully inside the other) is the point queries' job.

    Args:
        body_q: FK body transforms [N, body_count, 7] (pos + xyzw).
        edge_body: Body index per edge, ``wp.array`` of ``int32`` [E].
        edge_p0: Edge start points in their body frame, ``wp.array`` of ``wp.vec3`` [E].
        edge_p1: Edge end points in their body frame, ``wp.array`` of ``wp.vec3`` [E].
        mesh_id: Obstacle :class:`warp.Mesh` id in the obstacle's own frame.
        obstacle_pose: Per-problem obstacle pose [N, 7] (pos [m] + quat xyzw);
            identity rows for a static world-frame mesh.
        device: Warp/torch device string.
    """
    n = body_q.shape[0]
    body_q_wp = wp.from_torch(body_q.contiguous(), dtype=wp.transformf)
    pose_wp = wp.from_torch(obstacle_pose.contiguous(), dtype=wp.transformf)
    out = wp.zeros(n, dtype=wp.uint8, device=device)
    wp.launch(
        _edges_vs_posed_mesh_hit,
        dim=[n, edge_p0.shape[0]],
        inputs=[body_q_wp, edge_body, edge_p0, edge_p1, wp.uint64(mesh_id), pose_wp],
        outputs=[out],
        device=device,
    )
    return wp.to_torch(out).bool()


@wp.kernel
def _posed_edges_vs_body_meshes_hit(
    edge_p0: wp.array1d(dtype=wp.vec3),  # [E] endpoints in the obstacle frame
    edge_p1: wp.array1d(dtype=wp.vec3),
    obstacle_q: wp.array1d(dtype=wp.transformf),  # [N]
    body_q: wp.array2d(dtype=wp.transformf),  # [N, B]
    target_body: wp.array1d(dtype=wp.int32),  # [T]
    target_mesh: wp.array1d(dtype=wp.uint64),  # [T] shape-local mesh ids
    target_tf: wp.array1d(dtype=wp.transformf),  # [T]
    out_hit: wp.array1d(dtype=wp.uint8),  # [N]
):
    cand, e, t = wp.tid()
    a_w = wp.transform_point(obstacle_q[cand], edge_p0[e])
    b_w = wp.transform_point(obstacle_q[cand], edge_p1[e])
    target_world = wp.transform_multiply(body_q[cand, target_body[t]], target_tf[t])
    inv = wp.transform_inverse(target_world)
    a = wp.transform_point(inv, a_w)
    b = wp.transform_point(inv, b_w)
    d = b - a
    length = wp.length(d)
    if length < 1.0e-9:
        return
    q = wp.mesh_query_ray(target_mesh[t], a, d / length, length)
    if q.result:
        out_hit[cand] = wp.uint8(1)


def posed_edges_vs_body_meshes_hit(
    edge_p0: wp.array,
    edge_p1: wp.array,
    obstacle_pose: torch.Tensor,
    body_q: torch.Tensor,
    target_body: wp.array,
    target_mesh: wp.array,
    target_tf: wp.array,
    device: str,
) -> torch.Tensor:
    """Per-candidate crossing test: a posed obstacle's edges vs FK body colliders [N], bool.

    The reverse direction of :func:`edges_vs_posed_mesh_hit`, covering crossings
    whose intersection curve avoids the bodies' edges (e.g. a link face pierced by
    the board's edge).

    Args:
        edge_p0: Edge start points in the obstacle frame, ``wp.array`` of ``wp.vec3`` [E].
        edge_p1: Edge end points in the obstacle frame, ``wp.array`` of ``wp.vec3`` [E].
        obstacle_pose: Per-problem obstacle pose [N, 7] (pos [m] + quat xyzw).
        body_q: FK body transforms [N, body_count, 7] (pos + xyzw).
        target_body: Body index per target shape, ``wp.array`` of ``int32`` [T].
        target_mesh: Shape-local collision-mesh ids, ``wp.array`` of ``uint64`` [T].
        target_tf: Shape-local transforms, ``wp.array`` of ``wp.transformf`` [T].
        device: Warp/torch device string.
    """
    n = body_q.shape[0]
    body_q_wp = wp.from_torch(body_q.contiguous(), dtype=wp.transformf)
    pose_wp = wp.from_torch(obstacle_pose.contiguous(), dtype=wp.transformf)
    out = wp.zeros(n, dtype=wp.uint8, device=device)
    wp.launch(
        _posed_edges_vs_body_meshes_hit,
        dim=[n, edge_p0.shape[0], target_body.shape[0]],
        inputs=[edge_p0, edge_p1, pose_wp, body_q_wp, target_body, target_mesh, target_tf],
        outputs=[out],
        device=device,
    )
    return wp.to_torch(out).bool()


@wp.kernel
def _self_collision_sdf(
    body_q: wp.array2d(dtype=wp.transformf),  # [N, B]
    probe_body: wp.array1d(dtype=wp.int32),  # [P] body each probe is on
    probes: wp.array1d(dtype=wp.vec3),  # [P] offsets in their probe-body frame
    target_body: wp.array1d(dtype=wp.int32),  # [T] body each target shape is on
    target_mesh: wp.array1d(dtype=wp.uint64),  # [T] shape-local mesh ids
    target_tf: wp.array1d(dtype=wp.transformf),  # [T] shape-local transform
    adjacency: wp.array1d(dtype=wp.uint8),  # [B*B] flattened: 1 = skip pair
    n_bodies: int,
    max_dist: float,
    out_min_sd: wp.array1d(dtype=wp.float32),  # [N]
):
    cand, p, t = wp.tid()
    bp = probe_body[p]
    bt = target_body[t]
    if bp == bt:
        return
    if adjacency[bp * n_bodies + bt] != wp.uint8(0):
        return
    world_p = wp.transform_point(body_q[cand, bp], probes[p])
    # Into the target shape's frame (body pose composed with the shape-local transform).
    target_world = wp.transform_multiply(body_q[cand, bt], target_tf[t])
    local = wp.transform_point(wp.transform_inverse(target_world), world_p)
    q = wp.mesh_query_point(target_mesh[t], local, max_dist)
    if q.result:
        cp = wp.mesh_eval_position(target_mesh[t], q.face, q.u, q.v)
        sd = q.sign * wp.length(local - cp)  # <0 inside another link = self-penetration
        wp.atomic_min(out_min_sd, cand, sd)


def self_collision_min_sd(
    body_q: torch.Tensor,
    probe_body: wp.array,
    probes: wp.array,
    target_body: wp.array,
    target_mesh: wp.array,
    target_tf: wp.array,
    adjacency: wp.array,
    n_bodies: int,
    max_dist: float,
    device: str,
) -> torch.Tensor:
    """Per-candidate min robot link-vs-link signed distance [N] (>=0 clear, <0 self-penetration).

    Each probe (on a robot link) is tested against every other robot link's mesh,
    skipping pairs flagged in ``adjacency`` (same or kinematically adjacent links).

    Args:
        body_q: FK body transforms [N, body_count, 7] (pos + xyzw).
        probe_body: Body index per probe, ``wp.array`` of ``int32`` [P].
        probes: Probe offsets in their probe-body frame, ``wp.array`` of ``wp.vec3`` [P].
        target_body: Body index per target shape, ``wp.array`` of ``int32`` [T].
        target_mesh: Shape-local collision-mesh ids, ``wp.array`` of ``uint64`` [T].
        target_tf: Shape-local transforms, ``wp.array`` of ``wp.transformf`` [T].
        adjacency: Flattened ``[B*B]`` ``uint8`` skip mask (1 = same/adjacent pair).
        n_bodies: Body count ``B`` (row stride into ``adjacency``).
        max_dist: ``wp.mesh_query_point`` search radius [m].
        device: Warp/torch device string.
    """
    n = body_q.shape[0]
    body_q_wp = wp.from_torch(body_q.contiguous(), dtype=wp.transformf)
    out = wp.full(n, 1.0e6, dtype=wp.float32, device=device)
    wp.launch(
        _self_collision_sdf,
        dim=[n, probes.shape[0], target_body.shape[0]],
        inputs=[body_q_wp, probe_body, probes, target_body, target_mesh, target_tf, adjacency, n_bodies, max_dist],
        outputs=[out],
        device=device,
    )
    return wp.to_torch(out)
