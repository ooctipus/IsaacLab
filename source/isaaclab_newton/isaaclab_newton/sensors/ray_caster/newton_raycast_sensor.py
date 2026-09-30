# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import newton
import numpy as np
import warp as wp

from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster
from isaaclab.sensors.ray_caster.kernels import ALIGNMENT_BASE, update_ray_caster_kernel
from isaaclab.sim.simulation_context import SimulationContext
from isaaclab.utils.warp import ProxyArray

from .newton_raycast_sensor_cfg import NewtonRaycastSensorCfg
from .newton_raycast_sensor_data import NewtonRaycastSensorData


@wp.kernel(enable_backward=False)
def _newton_frame_world_poses_kernel(
    body_indices: wp.array(dtype=wp.int32),
    local_poses: wp.array(dtype=wp.transformf),
    body_q: wp.array(dtype=wp.transform),
    out_pose: wp.array(dtype=wp.transformf),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.quatf),
):
    """Write world poses for exact planned frames."""
    index = wp.tid()
    body_index = body_indices[index]
    local_pose = local_poses[index]
    if body_index == -1:
        world_transform = local_pose
    else:
        world_transform = wp.transform_multiply(body_q[body_index], local_pose)
    out_pose[index] = world_transform
    out_pos[index] = wp.transform_get_translation(world_transform)
    out_quat[index] = wp.transform_get_rotation(world_transform)


@wp.kernel(enable_backward=False)
def _newton_body_world_poses_kernel(
    body_indices: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    out_pose: wp.array(dtype=wp.transformf),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.quatf),
):
    """Gather exact Newton body poses."""
    index = wp.tid()
    world_transform = body_q[body_indices[index]]
    out_pose[index] = world_transform
    out_pos[index] = wp.transform_get_translation(world_transform)
    out_quat[index] = wp.transform_get_rotation(world_transform)


@wp.kernel(enable_backward=False)
def _gather_pose_by_index_kernel(
    indices: wp.array(dtype=wp.int32),
    pos_src: wp.array(dtype=wp.vec3f),
    quat_src: wp.array(dtype=wp.quatf),
    pos_dst: wp.array(dtype=wp.vec3f),
    quat_dst: wp.array(dtype=wp.quatf),
):
    """Gather poses using an index array."""
    index = wp.tid()
    source_index = indices[index]
    pos_dst[index] = pos_src[source_index]
    quat_dst[index] = quat_src[source_index]


class _NewtonRayCasterPoseMixin:
    """Update exact planned ray-caster frames from Newton body state."""

    @property
    def count(self: Any) -> int:
        """Number of planned sensor frames."""
        return self._view_count

    def _initialize_pose_tracking(self: Any) -> None:
        """Bind exact planned frames to Newton body indices and fixed local poses."""
        plan = SimulationContext.instance().get_clone_plan()
        if plan is None or not plan.is_complete:
            raise RuntimeError(f"RayCaster at {self.cfg.prim_path!r} requires a completed clone plan.")
        frames = list(plan.match_frames(self.cfg.prim_path))
        if len(frames) == 1 and frames[0].env_id is None and self._num_envs > 1:
            frames *= self._num_envs
        model = self._physics_manager.get_model()
        body_indices = {path: index for index, path in enumerate(model.body_label)}
        try:
            indices = [-1 if frame.body_path is None else body_indices[frame.body_path] for frame in frames]
        except KeyError as exc:
            raise ValueError(f"Newton model is missing planned RayCaster body {exc.args[0]!r}.") from exc
        self._view = self
        self._view_count = len(frames)
        self._frame_body_indices = wp.array(indices, dtype=wp.int32, device=self._device)
        self._frame_local_poses = wp.array(
            [wp.transform(*frame.pose) for frame in frames], dtype=wp.transformf, device=self._device
        )
        self._newton_pose_w = wp.empty(self._view_count, dtype=wp.transformf, device=self._device)
        self._newton_pos_w = ProxyArray(wp.empty(self._view_count, dtype=wp.vec3f, device=self._device))
        self._newton_quat_w = ProxyArray(wp.empty(self._view_count, dtype=wp.quatf, device=self._device))
        self._offset_pos_wp = wp.zeros(self._view_count, dtype=wp.vec3f, device=self._device)
        identity_quat = np.zeros((self._view_count, 4), dtype=np.float32)
        identity_quat[:, 3] = 1.0
        self._offset_quat_wp = wp.array(identity_quat, dtype=wp.quatf, device=self._device)

    def _update_ray_infos(self: Any, env_mask: wp.array) -> None:
        """Update planned frame poses and transform local rays into world space."""
        self._update_newton_frame_transforms(self._newton_pose_w, self._newton_pos_w.warp, self._newton_quat_w.warp)
        pos_w = self._data.pos_w.warp
        quat_w = self._data.quat_w_world.warp if hasattr(self._data, "quat_w_world") else self._data.quat_w.warp
        ray_starts = self.ray_starts.warp if hasattr(self.ray_starts, "warp") else self._ray_starts_local
        ray_directions = (
            self.ray_directions.warp if hasattr(self.ray_directions, "warp") else self._ray_directions_local
        )
        alignment_mode = int(ALIGNMENT_BASE) if hasattr(self._data, "quat_w_world") else self._alignment_mode
        wp.launch(
            update_ray_caster_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                self._newton_pose_w,
                env_mask,
                self._offset_pos_wp,
                self._offset_quat_wp,
                self.drift.warp,
                self.ray_cast_drift.warp,
                ray_starts,
                ray_directions,
                alignment_mode,
            ],
            outputs=[pos_w, quat_w, self._ray_starts_w, self._ray_directions_w],
            device=self._device,
        )

    def get_world_poses(self: Any, indices=None) -> tuple[ProxyArray, ProxyArray]:
        """Return world poses for legacy camera helpers."""
        self._update_newton_frame_transforms(self._newton_pose_w, self._newton_pos_w.warp, self._newton_quat_w.warp)
        if indices is None:
            return self._newton_pos_w, self._newton_quat_w
        if not isinstance(indices, wp.array):
            indices = wp.array(indices, dtype=wp.int32, device=self._device)
        pos_w = wp.empty(indices.shape[0], dtype=wp.vec3f, device=self._device)
        quat_w = wp.empty(indices.shape[0], dtype=wp.quatf, device=self._device)
        wp.launch(
            _gather_pose_by_index_kernel,
            dim=indices.shape[0],
            inputs=[indices, self._newton_pos_w.warp, self._newton_quat_w.warp],
            outputs=[pos_w, quat_w],
            device=self._device,
        )
        return ProxyArray(pos_w), ProxyArray(quat_w)

    def _update_newton_frame_transforms(self: Any, pose_buf: wp.array, pos_buf: wp.array, quat_buf: wp.array) -> None:
        """Update planned frame transforms using the manager-bound state."""
        state = self._physics_manager.get_state_0()
        wp.launch(
            _newton_frame_world_poses_kernel,
            dim=self._frame_body_indices.shape[0],
            inputs=[self._frame_body_indices, self._frame_local_poses, state.body_q],
            outputs=[pose_buf, pos_buf, quat_buf],
            device=self._device,
        )

    def _update_newton_body_transforms(
        self: Any, body_indices: wp.array, pose_buf: wp.array, pos_buf: wp.array, quat_buf: wp.array
    ) -> None:
        """Gather exact Newton body transforms."""
        wp.launch(
            _newton_body_world_poses_kernel,
            dim=body_indices.shape[0],
            inputs=[body_indices, self._physics_manager.get_state_0().body_q],
            outputs=[pose_buf, pos_buf, quat_buf],
            device=self._device,
        )


@wp.kernel(enable_backward=False)
def _resolve_bvh_hits_kernel(
    # input
    env_mask: wp.array(dtype=wp.bool),
    ray_starts_w: wp.array2d(dtype=wp.vec3f),
    ray_directions_w: wp.array2d(dtype=wp.vec3f),
    hit_dist: wp.array(dtype=wp.float32),
    hit_normal: wp.array(dtype=wp.vec3f),
    max_distance: float,
    ray_cast_drift: wp.array(dtype=wp.vec3f),
    # output
    ray_hits_w: wp.array2d(dtype=wp.vec3f),
    ray_distances: wp.array2d(dtype=wp.float32),
    ray_normals_w: wp.array2d(dtype=wp.vec3f),
):
    """Turn flat BVH query results into per-env hit points, distances, and normals.

    Misses (``hit_dist < 0``) and hits beyond ``max_distance`` are written as ``inf``.
    Launch with dim=(num_envs, num_rays).
    """
    env, ray = wp.tid()
    if not env_mask[env]:
        return
    idx = env * ray_starts_w.shape[1] + ray
    t = hit_dist[idx]
    if t >= 0.0 and t <= max_distance:
        hit = ray_starts_w[env, ray] + t * ray_directions_w[env, ray]
        ray_hits_w[env, ray] = wp.vec3f(hit[0], hit[1], hit[2] + ray_cast_drift[env][2])
        ray_distances[env, ray] = t
        ray_normals_w[env, ray] = hit_normal[idx]
    else:
        inf_vec = wp.vec3f(wp.inf, wp.inf, wp.inf)
        ray_hits_w[env, ray] = inf_vec
        ray_distances[env, ray] = wp.inf
        ray_normals_w[env, ray] = inf_vec


class NewtonRaycastSensor(_NewtonRayCasterPoseMixin, BaseRayCaster):
    """Ray-cast sensor that queries the whole Newton scene through the model's shape BVH.

    Rays are cast with :func:`newton.intersect_ray` against every collision
    shape in the sensor's own world plus the global world (e.g. terrain), so
    dynamic bodies are hit without configuring target meshes. The full update
    (sensor pose, ray transform, BVH query, hit resolve) is registered as a
    task with :class:`~isaaclab_newton.physics.NewtonManager`, which owns
    the shared BVH refit and sensor execution graph.
    """

    cfg: NewtonRaycastSensorCfg
    """The configuration parameters."""

    def __init__(self, cfg: NewtonRaycastSensorCfg):
        if cfg.max_distance <= 0.0:
            raise ValueError(f"max_distance must be positive, received {cfg.max_distance}.")
        from isaaclab_newton.cloner import NewtonReplicateContext

        sim = SimulationContext.instance()
        resource = sim.get_or_create_backend(NewtonReplicateContext, sim, clone_role="scene")
        resource._sensor_bvh_shape_flags |= newton.ShapeFlags.COLLIDE_SHAPES
        super().__init__(cfg)
        self._data = NewtonRaycastSensorData()
        self._sensor_task_name: str | None = None

    @property
    def data(self) -> NewtonRaycastSensorData:
        self._update_outdated_buffers()
        return self._data

    @property
    def ray_starts_w(self) -> ProxyArray:
        """World-frame ray start positions as of the last update [m].

        Shape is (N, B), dtype ``wp.vec3f``. In torch this resolves to (N, B, 3).
        """
        return self._ray_starts_w_ta

    @property
    def ray_directions_w(self) -> ProxyArray:
        """World-frame ray directions (unit vectors) as of the last update.

        Shape is (N, B), dtype ``wp.vec3f``. In torch this resolves to (N, B, 3).
        """
        return self._ray_directions_w_ta

    def _initialize_warp_meshes(self) -> None:
        # Rays are cast against the scene BVH; no warp meshes are needed.
        return

    def _initialize_impl(self) -> None:
        super()._initialize_impl()
        if self._view_count != self._num_envs:
            raise RuntimeError(
                f"NewtonRaycastSensor '{self.cfg.prim_path}' resolved {self._view_count} planned frames"
                f" for {self._num_envs} environments; exactly one frame per environment is supported."
                " Attach the sensor to a single rigid body per environment."
            )
        ray_count = self._num_envs * self.num_rays
        self._ray_starts_w_ta = ProxyArray(self._ray_starts_w)
        self._ray_directions_w_ta = ProxyArray(self._ray_directions_w)
        # Flat views and scratch buffers for newton.intersect_ray.
        self._ray_starts_w_flat = self._ray_starts_w.reshape((ray_count,))
        self._ray_directions_w_flat = self._ray_directions_w.reshape((ray_count,))
        global_world_only = self.cfg.global_world_only
        if global_world_only:
            world_ids = np.full(ray_count, -1, dtype=np.int32)
        else:
            world_ids = np.repeat(np.arange(self._num_envs, dtype=np.int32), self.num_rays)
        self._ray_worlds = wp.array(world_ids, dtype=wp.int32, device=self._device)
        self._hit_dist = wp.empty(ray_count, dtype=wp.float32, device=self._device)
        self._hit_normal = wp.empty(ray_count, dtype=wp.vec3f, device=self._device)

        self._sensor_task_name = f"newton_raycast:{self.cfg.prim_path}:{id(self)}"
        self._physics_manager._newton._register_sensor_task(self._sensor_task_name, self._launch_raycast)

    def _launch_raycast(self) -> None:
        """Sensor pose + ray transform + BVH query + hit resolve (graph-capturable)."""
        self._update_ray_infos(self._is_outdated)
        newton.intersect_ray(
            self._physics_manager.get_model(),
            ray_origins=self._ray_starts_w_flat,
            ray_directions=self._ray_directions_w_flat,
            ray_worlds=self._ray_worlds,
            enable_global_world=not self.cfg.global_world_only,
            out_dist=self._hit_dist,
            out_normal=self._hit_normal,
        )
        wp.launch(
            _resolve_bvh_hits_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                self._is_outdated,
                self._ray_starts_w,
                self._ray_directions_w,
                self._hit_dist,
                self._hit_normal,
                float(self.cfg.max_distance),
                self.ray_cast_drift.warp,
            ],
            outputs=[
                self._data._ray_hits_w,
                self._data._ray_distances,
                self._data._ray_normals_w,
            ],
            device=self._device,
        )

    def _update_buffers_impl(self, env_mask: wp.array) -> None:
        # The captured graph is bound to ``_is_outdated``; mirror any other mask into it.
        if env_mask.ptr != self._is_outdated.ptr:
            wp.copy(self._is_outdated, env_mask)
        assert self._sensor_task_name is not None
        self._physics_manager._newton._update_sensor_tasks(self._sensor_task_name)

    def _invalidate_initialize_callback(self, event) -> None:
        if self._sensor_task_name is not None:
            self._physics_manager._newton._unregister_sensor_task(self._sensor_task_name)
        self._sensor_task_name = None
        super()._invalidate_initialize_callback(event)


class RayCaster(NewtonRaycastSensor):
    """Default Newton ray caster backed by the live scene BVH."""
