# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

# pyright: reportInvalidTypeForm=none, reportPrivateUsage=none
from typing import Any

import numpy as np
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.cloner.cloner_utils import iter_clone_plan_matches
from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster
from isaaclab.sensors.ray_caster.kernels import (
    ALIGNMENT_BASE,
    copy_mesh_poses_to_table_kernel,
    update_ray_caster_kernel,
)
from isaaclab.utils.warp import ProxyArray

from isaaclab_newton.physics import NewtonManager


@wp.kernel
def _newton_site_world_poses_kernel(
    site_indices: wp.array(dtype=wp.int32),
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    out_pose: wp.array(dtype=wp.transformf),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.quatf),
):
    i = wp.tid()
    site_idx = site_indices[i]
    body_idx = shape_body[site_idx]
    site_xform = shape_transform[site_idx]
    if body_idx == -1:
        world_xform = site_xform
    else:
        world_xform = wp.transform_multiply(body_q[body_idx], site_xform)
    out_pose[i] = world_xform
    out_pos[i] = wp.transform_get_translation(world_xform)
    out_quat[i] = wp.transform_get_rotation(world_xform)


@wp.kernel
def _gather_pose_by_index_kernel(
    indices: wp.array(dtype=wp.int32),
    pos_src: wp.array(dtype=wp.vec3f),
    quat_src: wp.array(dtype=wp.quatf),
    pos_dst: wp.array(dtype=wp.vec3f),
    quat_dst: wp.array(dtype=wp.quatf),
):
    i = wp.tid()
    src_idx = indices[i]
    pos_dst[i] = pos_src[src_idx]
    quat_dst[i] = quat_src[src_idx]


def _identity_offsets(count: int, device: str) -> tuple[wp.array, wp.array]:
    """Create identity sensor offsets for site poses that already include the offset."""
    offset_pos_wp = wp.zeros(count, dtype=wp.vec3f, device=device)
    identity_quat = np.zeros((count, 4), dtype=np.float32)
    identity_quat[:, 3] = 1.0
    return offset_pos_wp, wp.array(identity_quat, dtype=wp.quatf, device=device)


class _NewtonRayCasterMixin:
    """Newton site registration and pose tracking for ray-caster sensors.

    Sites must be registered during construction so Newton can inject them into
    prototype builders before cloning. Once physics is ready, the mixin resolves
    those labels to concrete site indices and updates the sensor-owned buffers
    directly from Newton model/state arrays.
    """

    @property
    def count(self: Any) -> int:
        """Number of resolved Newton sites tracked as sensor frames."""
        return self._view_count

    def __init__(self: Any, cfg):
        """Register sensor and dynamic target sites before cloning occurs."""
        super().__init__(cfg)  # pyright: ignore[reportCallIssue]
        identity = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0))
        attach_expr = self.cfg.prim_path
        if attach_expr.rsplit("/", 1)[-1].lower() in ("camera", "raycaster"):
            attach_expr = attach_expr.rsplit("/", 1)[0]
        plan = sim_utils.SimulationContext.instance().get_clone_plan()
        # If ``attach_expr`` resolves to the env-root source itself (no suffix), attach per-world.
        per_world = any(sp == sr for sp, sr, _, _ in iter_clone_plan_matches(plan, attach_expr))
        self._sensor_site_labels = [
            NewtonManager.cl_register_site(None, identity, per_world=True)
            if per_world
            else NewtonManager.cl_register_site(attach_expr.replace("{}", ".*"), identity)
        ]
        self._tracked_site_labels_by_target: dict[tuple[str, ...], list[str]] = {}
        for target_cfg in getattr(self, "_raycast_targets_cfg", []):
            if not target_cfg.track_mesh_transforms:
                continue
            owner_exprs: list[str] = []
            for source_path, source_root, destination, _ in iter_clone_plan_matches(plan, target_cfg.prim_expr):
                source_prims = sim_utils.find_matching_prims(source_path)
                if not source_prims:
                    raise RuntimeError(f"No ClonePlan source prims matched '{source_path}'.")
                for source_prim in source_prims:
                    sp = str(source_prim.GetPath())
                    owner = sp if sp == source_root else sp.rsplit("/", 1)[0]
                    owner_exprs.append(destination.format(".*") + owner[len(source_root) :])
            if not owner_exprs:
                raise RuntimeError(f"RayCaster target '{target_cfg.prim_expr}' is not owned by the active ClonePlan.")
            owner_exprs = list(dict.fromkeys(owner_exprs))
            labels = list(dict.fromkeys(NewtonManager.cl_register_site(e, identity) for e in owner_exprs))
            self._tracked_site_labels_by_target[tuple(owner_exprs)] = labels

    def _resolve_and_spawn(self, _sensor_name: str, **_spawn_kwargs) -> None:
        """Skip USD sensor prim spawning for Newton ray casters."""
        pass

    def _initialize_pose_tracking(self: Any) -> None:
        """Resolve registered site labels and allocate sensor-owned pose buffers."""
        site_indices = self._resolve_site_indices(self._sensor_site_labels, self.cfg.prim_path, self._num_envs)
        # The base classes still use ``self._view.count`` in a few generic
        # places. Point it at the sensor instead of constructing an adapter.
        self._view = self
        self._view_count = len(site_indices)
        self._sensor_site_indices = wp.array(site_indices, dtype=wp.int32, device=self._device)
        self._newton_pose_w = wp.empty(self._view_count, dtype=wp.transformf, device=self._device)
        self._newton_pos_w = ProxyArray(wp.empty(self._view_count, dtype=wp.vec3f, device=self._device))
        self._newton_quat_w = ProxyArray(wp.empty(self._view_count, dtype=wp.quatf, device=self._device))
        self._offset_pos_wp, self._offset_quat_wp = _identity_offsets(self._view_count, self._device)

    def _update_ray_infos(self: Any, env_mask: wp.array):
        """Update Newton site poses and transform local rays in a single ray-caster kernel."""
        self._update_newton_site_transforms(
            self._sensor_site_indices, self._newton_pose_w, self._newton_pos_w.warp, self._newton_quat_w.warp
        )
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
            outputs=[
                pos_w,
                quat_w,
                self._ray_starts_w,
                self._ray_directions_w,
            ],
            device=self._device,
        )

    def get_world_poses(self: Any, indices=None):
        """Return world poses for camera helpers that still use pose tuples."""
        self._update_newton_site_transforms(
            self._sensor_site_indices, self._newton_pose_w, self._newton_pos_w.warp, self._newton_quat_w.warp
        )
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

    def _create_tracked_target_view(self: Any, target_prim_path: str | list[str]):
        """Resolve dynamic multi-mesh target sites to raw Newton site indices."""
        target_exprs = target_prim_path if isinstance(target_prim_path, list) else [target_prim_path]
        labels = self._tracked_site_labels_by_target[tuple(target_exprs)]
        site_indices = self._resolve_site_indices(labels, str(target_prim_path), self._num_envs)
        return wp.array(site_indices, dtype=wp.int32, device=self._device)

    def _update_mesh_transforms(self: Any) -> None:
        """Refresh dynamic multi-mesh targets directly from Newton sites."""
        if not hasattr(self, "_mesh_views"):
            return
        mesh_idx = 0
        for site_indices, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]
                continue

            site_count = site_indices.shape[0]
            pos_buf = wp.empty(site_count, dtype=wp.vec3f, device=self._device)
            quat_buf = wp.empty(site_count, dtype=wp.quatf, device=self._device)
            pose_buf = wp.empty(site_count, dtype=wp.transformf, device=self._device)
            self._update_newton_site_transforms(site_indices, pose_buf, pos_buf, quat_buf)
            meshes_per_env = self._num_meshes_per_env[target_cfg.prim_expr]

            wp.launch(
                copy_mesh_poses_to_table_kernel,
                dim=(self._num_envs, meshes_per_env),
                inputs=[
                    pos_buf,
                    quat_buf,
                    int(meshes_per_env),
                    int(mesh_idx),
                    bool(site_count == 1),
                    self._mesh_positions_w,
                    self._mesh_orientations_w,
                ],
                device=self._device,
            )
            mesh_idx += meshes_per_env

    def _update_newton_site_transforms(
        self: Any,
        site_indices: wp.array,
        pose_buf: wp.array,
        pos_buf: wp.array,
        quat_buf: wp.array,
    ) -> None:
        """Launch the Newton site pose kernel into caller-provided buffers."""
        model = NewtonManager._model
        state = NewtonManager._state_0
        if model is None or state is None:
            raise RuntimeError("Newton simulation state is not initialized.")
        wp.launch(
            _newton_site_world_poses_kernel,
            dim=site_indices.shape[0],
            inputs=[site_indices, model.shape_body, model.shape_transform, state.body_q],
            outputs=[pose_buf, pos_buf, quat_buf],
            device=self._device,
        )

    @staticmethod
    def _resolve_site_indices(labels: list[str], prim_expr: str, num_envs: int) -> list[int]:
        """Expand registered site labels into per-environment Newton site indices."""
        site_map = NewtonManager._cl_site_index_map
        site_indices: list[int] = []
        for env_idx in range(num_envs):
            for label in labels:
                error_prefix = f"RayCaster target '{prim_expr}' site label '{label}'"
                if label not in site_map:
                    raise ValueError(f"{error_prefix} was not found in NewtonManager._cl_site_index_map.")
                global_idx, per_world = site_map[label]
                env_site_indices = [global_idx] if per_world is None else per_world[env_idx]
                site_indices.extend(env_site_indices)
        return site_indices


class RayCaster(_NewtonRayCasterMixin, BaseRayCaster):
    """Newton ray-caster implementation."""
