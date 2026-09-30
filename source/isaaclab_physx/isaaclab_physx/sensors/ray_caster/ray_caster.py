# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
import warp as wp

from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster
from isaaclab.sensors.ray_caster.kernels import copy_mesh_transforms_to_table_kernel
from isaaclab.sim.simulation_context import SimulationContext

if TYPE_CHECKING:
    from isaaclab.sensors.ray_caster.ray_caster_cfg import RayCasterCfg

logger = logging.getLogger(__name__)


class _PhysXRayCasterMixin:
    """PhysX pose tracking for ray-caster sensors.

    PhysX provides live transforms for plan-bound rigid bodies after physics is ready.
    Fixed frames retain the world poses declared by the same plan.
    """

    @property
    def count(self: Any) -> int:
        """Number of tracked sensor frames."""
        return self._view_count

    def _initialize_pose_tracking(self: Any) -> None:
        """Bind exact planned sensor frames to PhysX bodies or fixed world poses."""
        plan = SimulationContext.instance().get_clone_plan()
        if plan is None or not plan.is_complete:
            raise RuntimeError(f"RayCaster at {self.cfg.prim_path!r} requires a completed clone plan.")
        frames = plan.match_frames(self.cfg.prim_path)
        self._view = self
        if all(frame.body_path is None for frame in frames):
            self._initialize_static_pose_tracking([frame.pose for frame in frames])
            return
        if any(frame.body_path is None for frame in frames):
            raise ValueError(f"RayCaster expression {self.cfg.prim_path!r} mixes fixed and rigid-body frames.")
        by_body = {frame.body_path: frame for frame in frames}
        if len(by_body) != len(frames):
            raise ValueError(f"RayCaster expression {self.cfg.prim_path!r} resolves multiple frames on one body.")

        physics_sim_view = self._physics_manager.get_physics_sim_view()
        if physics_sim_view is None:
            raise RuntimeError("PhysX simulation view is not initialized.")
        self._physx_body_view = physics_sim_view.create_rigid_body_view(list(by_body))
        self._view_count = self._physx_body_view.count
        if self._view_count != len(by_body):
            raise ValueError("PhysX RayCaster body view does not match the clone plan.")
        try:
            ordered = [by_body[path] for path in self._physx_body_view.prim_paths]
        except KeyError as exc:
            raise ValueError(f"PhysX returned undeclared RayCaster body {exc.args[0]!r}.") from exc
        self._offset_pos_wp = wp.array([frame.pose[:3] for frame in ordered], dtype=wp.vec3f, device=self._device)
        self._offset_quat_contiguous = torch.tensor(
            [frame.pose[3:] for frame in ordered], dtype=torch.float32, device=self._device
        )
        self._offset_quat_wp = wp.from_torch(self._offset_quat_contiguous, dtype=wp.quatf)

    def _initialize_static_pose_tracking(self: Any, poses) -> None:
        """Cache planned world poses for non-physics sensor frames."""
        if len(poses) == 1 and self._num_envs > 1:
            poses *= self._num_envs
        self._static_view_transforms_torch = torch.tensor(poses, dtype=torch.float32, device=self._device).contiguous()
        self._static_view_transforms_wp = wp.from_torch(self._static_view_transforms_torch).view(wp.transformf)
        self._physx_body_view = None
        self._view_count = len(poses)
        self._offset_pos_wp = wp.zeros(self._view_count, dtype=wp.vec3f, device=self._device)
        identity_quat = torch.zeros(self._view_count, 4, device=self._device)
        identity_quat[:, 3] = 1.0
        self._offset_quat_contiguous = identity_quat.contiguous()
        self._offset_quat_wp = wp.from_torch(self._offset_quat_contiguous, dtype=wp.quatf)

    def _get_view_transforms_wp(self: Any) -> wp.array:
        """Return tracked sensor-frame transforms as ``wp.transformf``."""
        if self._physx_body_view is None:
            return self._static_view_transforms_wp
        transforms = self._physx_body_view.get_transforms()
        if isinstance(transforms, wp.array):
            return transforms.view(wp.transformf)
        return wp.from_torch(transforms.contiguous()).view(wp.transformf)

    def get_world_poses(self: Any, indices=None):
        """Return world poses for camera helpers that still use pose tuples."""
        transforms = self._get_view_transforms_wp()
        transforms_t = wp.to_torch(transforms).reshape(-1, 7)
        if indices is not None:
            idx = wp.to_torch(indices).to(dtype=torch.long) if isinstance(indices, wp.array) else indices
            transforms_t = transforms_t[idx]
        return SimpleNamespace(torch=transforms_t[:, 0:3]), SimpleNamespace(torch=transforms_t[:, 3:7])

    def _create_tracked_target_view(self: Any, target_prim_paths: str | list[str]):
        """Create a PhysX rigid-body view for dynamic multi-mesh targets."""
        if isinstance(target_prim_paths, str):
            target_prim_paths = [target_prim_paths]
        if not target_prim_paths:
            raise RuntimeError(f"No tracked target bodies resolved from: {target_prim_paths}")
        physics_sim_view = self._physics_manager.get_physics_sim_view()
        if physics_sim_view is None:
            raise RuntimeError("PhysX simulation view is not initialized.")
        view = physics_sim_view.create_rigid_body_view(target_prim_paths)
        if list(view.prim_paths) != target_prim_paths:
            raise ValueError("PhysX tracked ray-cast target view does not match the clone plan.")
        return view

    def _update_mesh_transforms(self: Any) -> None:
        """Refresh dynamic multi-mesh targets directly from PhysX views."""
        if not hasattr(self, "_mesh_views"):
            return
        mesh_idx = 0
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]
                continue

            transforms = view.get_transforms()
            transforms_wp = (
                transforms.view(wp.transformf)
                if isinstance(transforms, wp.array)
                else wp.from_torch(transforms.contiguous()).view(wp.transformf)
            )

            view_count = view.count
            meshes_per_env = view_count
            if view_count != 1:
                # PhysX views return a flat list across envs; the mesh table is indexed per env.
                meshes_per_env = view_count // self._num_envs

            wp.launch(
                copy_mesh_transforms_to_table_kernel,
                dim=(self._num_envs, meshes_per_env),
                inputs=[
                    transforms_wp,
                    int(meshes_per_env),
                    int(mesh_idx),
                    bool(view_count == 1),
                    self._mesh_positions_w,
                    self._mesh_orientations_w,
                ],
                device=self._device,
            )
            mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]


class RayCaster(_PhysXRayCasterMixin, BaseRayCaster):
    """PhysX ray-caster implementation."""

    def __init__(self, cfg: RayCasterCfg):
        """Initialize the PhysX ray-caster.

        Args:
            cfg: The ray-caster configuration.
        """
        super().__init__(cfg)
        self._raw_transforms: wp.array | None = None
        self._compute_graph: wp.Graph | None = None
        self._use_graph: bool = False
        self._env_mask: wp.array | None = None
        self._transforms_prefetched: bool = False

    def _initialize_impl(self) -> None:
        """Initialize the sensor and enable CUDA graph replay on CUDA devices."""
        super()._initialize_impl()
        self._use_graph = wp.get_device(self._device).is_cuda

    def _get_view_transforms_wp(self) -> wp.array:
        """Refresh the PhysX transform buffer and return its cached typed Warp view."""
        if self._transforms_prefetched:
            if self._raw_transforms is None:
                raise RuntimeError("RayCaster transforms were marked prefetched before a buffer was cached.")
            return self._raw_transforms

        if self._physx_body_view is None:
            if self._raw_transforms is None:
                self._raw_transforms = self._static_view_transforms_wp
            return self._raw_transforms

        transforms = self._physx_body_view.get_transforms()
        if self._raw_transforms is None:
            if isinstance(transforms, wp.array):
                self._raw_transforms = transforms.view(wp.transformf)
            else:
                self._raw_transforms = wp.from_torch(transforms.contiguous()).view(wp.transformf)
        return self._raw_transforms

    def _update_buffers_impl(self, env_mask: wp.array) -> None:
        """Refresh PhysX transforms and update ray data eagerly or through a CUDA graph.

        Raises:
            RuntimeError: If an outer CUDA graph capture is active. The PhysX transform read
                cannot be graph-captured, so replays of such a graph would consume stale
                transforms.
        """
        device = wp.get_device(self._device)
        if device.is_capturing:
            raise RuntimeError(
                f"Cannot update the ray caster at '{self.cfg.prim_path}' while a CUDA graph capture is"
                " active: the PhysX transform read cannot be graph-captured, so replaying the captured"
                " graph would consume stale transforms."
            )

        # PhysX refreshes this stable output buffer in place. Fetch it outside the graph, then
        # let the graph consume the cached typed view without calling back into PhysX.
        self._get_view_transforms_wp()
        self._transforms_prefetched = True
        self._env_mask = env_mask

        try:
            if not self._use_graph:
                self._compute()
                return

            if self._compute_graph is None:
                try:
                    with wp.ScopedCapture(device=device) as capture:
                        self._compute()
                except Exception as exc:
                    self._use_graph = False
                    logger.warning(
                        f"Failed to capture the update of the ray caster at '{self.cfg.prim_path}' into a"
                        f" CUDA graph. Falling back to eager kernel launches. Reason: {exc}"
                    )
                    self._compute()
                    return
                self._compute_graph = capture.graph
            wp.capture_launch(self._compute_graph)
        finally:
            self._transforms_prefetched = False

    def _compute(self) -> None:
        """Launch the Warp kernels that update the standard ray-caster data."""
        env_mask = self._env_mask
        if env_mask is None:
            raise RuntimeError("RayCaster update kernels cannot run without an environment mask.")
        super()._update_buffers_impl(env_mask)

    def _invalidate_initialize_callback(self, event) -> None:
        """Invalidate physics handles and graph state."""
        super()._invalidate_initialize_callback(event)
        self._view = None
        self._physx_body_view = None
        self._raw_transforms = None
        self._compute_graph = None
        self._env_mask = None
        self._transforms_prefetched = False
