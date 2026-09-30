# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.sim import SimulationContext

if TYPE_CHECKING:
    import trimesh

    from isaaclab.envs import ManagerBasedEnv

_PRIM_SAMPLE_CACHE: dict[tuple[str, tuple[float, ...], int], np.ndarray] = {}
_FINAL_SAMPLE_CACHE: dict[tuple[tuple[tuple[str, tuple[float, ...], int], ...], int], np.ndarray] = {}


def clear_pointcloud_caches():
    _PRIM_SAMPLE_CACHE.clear()
    _FINAL_SAMPLE_CACHE.clear()


def sample_object_point_cloud(num_envs: int, num_points: int, prim_path: str, device: str = "cpu") -> torch.Tensor:
    """Sample each object's plan-declared geometry into ``[num_envs, num_points, 3]``."""
    from trimesh.sample import sample_surface

    points = np.zeros((num_envs, num_points, 3), dtype=np.float32)
    plan = _clone_plan()
    env_ids = np.asarray(plan._env_ids_cpu)
    for target, geometries, clone_mask in plan.match_geometry_prototypes(prim_path):
        _body_path, geometries = _body_geometry_prototypes(target, geometries, clone_mask)
        keys = tuple((geometry.source_path, geometry.frame.pose, num_points) for geometry in geometries)
        samples_np = _FINAL_SAMPLE_CACHE.get((keys, num_points))
        if samples_np is None:
            per_geometry = []
            for geometry, key in zip(geometries, keys, strict=True):
                samples = _PRIM_SAMPLE_CACHE.get(key)
                if samples is None:
                    mesh = _geometry_mesh(geometry, geometry.frame)
                    candidates, _ = sample_surface(mesh, num_points * 2, face_weight=mesh.area_faces)
                    candidates = torch.from_numpy(candidates.astype(np.float32)).to(device)
                    samples = candidates[farthest_point_sampling(candidates, num_points)].cpu().numpy()
                    _PRIM_SAMPLE_CACHE[key] = samples
                per_geometry.append(samples)
            combined = torch.from_numpy(np.concatenate(per_geometry)).to(device)
            samples = combined if len(per_geometry) == 1 else combined[farthest_point_sampling(combined, num_points)]
            samples_np = samples.cpu().numpy()
            _FINAL_SAMPLE_CACHE[(keys, num_points)] = samples_np
        points[env_ids[clone_mask]] = samples_np

    return torch.from_numpy(points).to(device)


def _clone_plan():
    """Return the active completed clone plan."""
    sim = SimulationContext.instance()
    plan = None if sim is None else sim.get_clone_plan()
    if plan is None or not plan.is_complete:
        raise RuntimeError("Lift geometry requires a completed clone plan.")
    return plan


def _geometry_mesh(geometry, frame):
    """Return one planned geometry transformed into its owning rigid-body frame."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=geometry.vertices.copy(), faces=geometry.faces, process=False)
    x, y, z, w = frame.pose[3:]
    transform = trimesh.transformations.quaternion_matrix([w, x, y, z])
    transform[:3, 3] = frame.pose[:3]
    mesh.apply_transform(transform)
    return mesh


def _body_geometry_prototypes(target, geometries, clone_mask):
    """Return one target prototype's body path and geometry shared by every selected clone."""
    if clone_mask is None:
        raise RuntimeError(f"Object target {target.path!r} is not environment-scoped.")
    body_paths = {geometry.frame.body_path for geometry in geometries if geometry.frame.body_path is not None}
    if len(body_paths) != 1:
        raise RuntimeError(f"Planned object {target.path!r} contains {len(body_paths)} geometry-owning bodies.")
    body_path = body_paths.pop()
    geometries = tuple(geometry for geometry in geometries if geometry.frame.body_path == body_path)
    if any(geometry.clone_mask is None or np.any(clone_mask & ~geometry.clone_mask) for geometry in geometries):
        raise RuntimeError(f"Planned object {target.path!r} has clone-dependent body geometry.")
    return body_path, geometries


def farthest_point_sampling(
    points: torch.Tensor,
    n_samples: int,
    distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.cdist,
    memory_threshold: int = 2 * 1024**3,  # 2 GiB
) -> torch.Tensor:
    """Farthest point sampling (FPS): pick a maximally spread subset of the given samples.

    Greedily takes the sample whose distance to the already-picked set is largest, so the result
    covers the input instead of clustering where the input is dense. Any feature space works, not
    just 3D points, as long as :paramref:`distance_fn` measures it.

    Args:
        points: Samples to pick from, shape [num_points, feature_dim].
        n_samples: Number of samples to pick. Values at or above ``len(points)`` return all of them.
        distance_fn: Pairwise distance, called as ``(a, b) -> [len(a), len(b)]``. Defaults to the
            Euclidean :func:`torch.cdist`.
        memory_threshold: Byte budget [B] for the full pairwise distance matrix. Above it, distances
            to the picked sample are recomputed each round instead of cached.

    Returns:
        Indices of the picked samples, shape [n_samples].
    """
    num_points = len(points)
    if n_samples >= num_points:
        return torch.arange(num_points, device=points.device)

    # caching the whole matrix costs one distance_fn call instead of one per round
    matrix_bytes = num_points * num_points * points.element_size()
    cached = matrix_bytes <= memory_threshold
    if cached:
        distances = distance_fn(points, points)
    else:
        logging.warning(f"FPS fallback to iterative (needed {matrix_bytes} > {memory_threshold})")

    sampled_idx = torch.zeros(n_samples, dtype=torch.long, device=points.device)
    # distance from every sample to the picked set; picked samples sit at 0 and are never re-picked
    min_dists = torch.full((num_points,), float("inf"), device=points.device)
    # kept one-dimensional so ``points[farthest]`` stays a batch of one for distance_fn
    farthest = torch.randint(0, num_points, (1,), device=points.device)
    for j in range(n_samples):
        sampled_idx[j] = farthest
        to_farthest = distances[farthest] if cached else distance_fn(points[farthest], points)
        min_dists = torch.minimum(min_dists, to_farthest.view(-1))
        farthest = torch.argmax(min_dists, dim=0, keepdim=True)
    return sampled_idx


def _merge_body_geometries(geometries, body_path: str):
    """Merge plan-declared collision geometry in one rigid-body frame."""
    import trimesh

    meshes = [
        _geometry_mesh(geometry, geometry.frame)
        for geometry in geometries
        if geometry.collision and geometry.frame.body_path == body_path
    ]
    if not meshes:
        raise RuntimeError(f"Planned rigid body {body_path!r} has no requested collision geometry.")
    return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]


def collect_body_collision_meshes(robot, body_names: str | list[str]) -> tuple[dict[int, trimesh.Trimesh], list[str]]:
    """Return selected robot collision meshes from the immutable clone-plan geometry."""
    body_ids, names = robot.find_bodies(body_names)
    plan = _clone_plan()
    articulation = plan.match_articulation(robot.cfg.prim_path)
    bodies = {body.name: body for body in articulation.bodies}
    collision_views = {geometry.frame.body_view_path for geometry in plan.geometry_prototypes if geometry.collision}
    try:
        meshes = {
            body_id: _merge_body_geometries(plan.match_geometry_targets(bodies[name].path)[0][1], bodies[name].path)
            for body_id, name in zip(body_ids, names, strict=True)
            if bodies[name].view_path in collision_views
        }
    except KeyError as exc:
        raise RuntimeError(f"Robot body {exc.args[0]!r} is absent from the clone plan.") from exc
    if not meshes:
        raise RuntimeError(f"Selected robot bodies {names!r} have no planned collision geometry.")
    return meshes, names


def collect_rigid_object_collision_meshes(num_envs: int, prim_path: str) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Return unique planned object collision meshes and each environment's mesh index."""
    plan = _clone_plan()
    meshes = []
    mesh_by_source: dict[tuple[tuple[str, tuple[float, ...]], ...], int] = {}
    env_mesh = np.full(num_envs, -1, dtype=np.int32)
    env_ids = np.asarray(plan._env_ids_cpu)
    for target, geometries, clone_mask in plan.match_geometry_prototypes(prim_path):
        body_path, geometries = _body_geometry_prototypes(target, geometries, clone_mask)
        key = tuple((geometry.source_path, geometry.frame.pose) for geometry in geometries if geometry.collision)
        if key not in mesh_by_source:
            mesh_by_source[key] = len(meshes)
            meshes.append(_merge_body_geometries(geometries, body_path))
        env_mesh[env_ids[clone_mask]] = mesh_by_source[key]
    if (env_mesh < 0).any():
        raise RuntimeError(f"Object expression {prim_path!r} does not cover every environment in the clone plan.")
    return meshes, env_mesh


def get_reset_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    reset_assets: Sequence[str],
    is_relative: bool = False,
) -> torch.Tensor:
    """Read and concatenate the reset-state slices of the given scene assets.

    Per articulation: root pose [m, (x, y, z, w)] (7), root center-of-mass velocity
    [m/s, rad/s] (6), joint positions and joint velocities; per rigid object: root pose and
    root center-of-mass velocity. With :paramref:`is_relative`, root positions are expressed
    relative to the environment origins so states transplant across environments.
    """

    def root_state(asset) -> list[torch.Tensor]:
        pose = asset.data.root_link_pose_w.torch[env_ids]
        if is_relative:
            pose = pose.clone()
            pose[:, :3] -= env.scene.env_origins[env_ids]
        return [pose, asset.data.root_com_vel_w.torch[env_ids]]

    states: list[torch.Tensor] = []
    for name, articulation in env.scene.articulations.items():
        if name in reset_assets:
            states += root_state(articulation)
            states.append(articulation.data.joint_pos.torch[env_ids])
            states.append(articulation.data.joint_vel.torch[env_ids])
    for name, rigid_object in env.scene.rigid_objects.items():
        if name in reset_assets:
            states += root_state(rigid_object)
    return torch.cat(states, dim=-1)


def set_reset_state(
    env: ManagerBasedEnv,
    states: torch.Tensor,
    env_ids: torch.Tensor,
    reset_assets: Sequence[str],
    is_relative: bool = False,
):
    """Split :paramref:`states` by scene asset and write the reset-state slices.

    Inverse of :func:`get_reset_state`; the layout and :paramref:`is_relative` convention
    must match the call that produced :paramref:`states`.
    """
    offset = 0

    def write_root(asset):
        nonlocal offset
        pose = states[:, offset : offset + 7].clone()
        if is_relative:
            pose[:, :3] += env.scene.env_origins[env_ids]
        asset.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        asset.write_root_com_velocity_to_sim_index(
            root_velocity=states[:, offset + 7 : offset + 13].contiguous(), env_ids=env_ids
        )
        offset += 13

    for name, articulation in env.scene.articulations.items():
        if name in reset_assets:
            write_root(articulation)
            num_joints = articulation.num_joints
            articulation.write_joint_state_to_sim_index(
                position=states[:, offset : offset + num_joints].contiguous(),
                velocity=states[:, offset + num_joints : offset + 2 * num_joints].contiguous(),
                env_ids=env_ids,
            )
            offset += 2 * num_joints
    for name, rigid_object in env.scene.rigid_objects.items():
        if name in reset_assets:
            write_root(rigid_object)
