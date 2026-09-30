# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX ray-caster sensors -- mixin + concrete RayCaster class."""

from __future__ import annotations

import contextlib
import logging
from types import SimpleNamespace
from typing import Any

import torch
import warp as wp

from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster
from isaaclab.sensors.ray_caster.kernels import copy_mesh_transforms_to_table_kernel
from isaaclab.sim.simulation_context import SimulationContext

logger = logging.getLogger(__name__)


class _OvPhysxRayCasterMixin:
    """OVPhysX pose tracking for ray-caster sensors.

    Lives as a multiple-inheritance mixin on top of the four
    :class:`~isaaclab.sensors.ray_caster.Base*` classes. Provides backend-
    specific pose tracking via the ovphysx ``RIGID_BODY_POSE`` tensor binding
    when the planned sensor frame has a rigid-body binding, or its planned
    world pose for a fixed frame.
    """

    @property
    def count(self: Any) -> int:
        """Number of tracked sensor frames (binding row count, or static count)."""
        return self._view_count

    def _initialize_pose_tracking(self: Any) -> None:
        """Bind exact planned sensor frames to OVPhysX bodies or fixed world poses."""
        from isaaclab_ov import tensor_types as TT  # noqa: PLC0415

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

        physx = self._physics_manager.get_physx_instance()
        if physx is None:
            raise RuntimeError(
                "OvPhysxManager has no PhysX instance yet -- sensor was constructed before "
                "PhysicsEvent.PHYSICS_READY. Ensure the simulation has been reset at least once."
            )

        self._ovphysx_body_view = physx.create_tensor_binding(
            prim_paths=list(by_body),
            tensor_type=TT.RIGID_BODY_POSE,
        )
        if self._ovphysx_body_view.shape[0] == 0:
            raise RuntimeError(f"OVPhysX RIGID_BODY_POSE binding for {tuple(by_body)!r} matched zero bodies.")

        self._view_count = int(self._ovphysx_body_view.shape[0])
        if self._view_count != len(by_body):
            raise ValueError("OVPhysX RayCaster body binding does not match the clone plan.")
        self._pose_buf = wp.zeros(self._ovphysx_body_view.shape, dtype=wp.float32, device=self._device)
        # Zero-copy reinterpret of the ``(N, 7)`` float32 staging buffer as
        # ``(N,)`` ``wp.transformf``. Cached so per-step
        # ``_get_view_transforms_wp`` reads don't churn allocations.
        self._pose_buf_transformf = wp.array(
            ptr=self._pose_buf.ptr,
            shape=(self._view_count,),
            dtype=wp.transformf,
            device=str(self._pose_buf.device),
            copy=False,
        )

        try:
            ordered = [by_body[path] for path in self._ovphysx_body_view.prim_paths]
        except KeyError as exc:
            raise ValueError(f"OVPhysX returned undeclared RayCaster body {exc.args[0]!r}.") from exc
        self._offset_pos_wp = wp.array([frame.pose[:3] for frame in ordered], dtype=wp.vec3f, device=self._device)
        self._offset_quat_contiguous = torch.tensor(
            [frame.pose[3:] for frame in ordered], dtype=torch.float32, device=self._device
        )
        self._offset_quat_wp = wp.from_torch(self._offset_quat_contiguous, dtype=wp.quatf)
        self._mesh_view_bufs = {}

    def _initialize_static_pose_tracking(self: Any, poses) -> None:
        """Cache planned world poses for non-physics sensor frames."""
        if len(poses) == 1 and self._num_envs > 1:
            poses *= self._num_envs
        self._static_view_transforms_torch = torch.tensor(poses, dtype=torch.float32, device=self._device).contiguous()
        self._static_view_transforms_wp = wp.from_torch(self._static_view_transforms_torch).view(wp.transformf)
        self._ovphysx_body_view = None
        self._view_count = len(poses)
        self._offset_pos_wp = wp.zeros(self._view_count, dtype=wp.vec3f, device=self._device)
        identity_quat = torch.zeros(self._view_count, 4, device=self._device)
        identity_quat[:, 3] = 1.0
        self._offset_quat_contiguous = identity_quat.contiguous()
        self._offset_quat_wp = wp.from_torch(self._offset_quat_contiguous, dtype=wp.quatf)
        self._mesh_view_bufs = {}

    def _get_view_transforms_wp(self: Any) -> wp.array:
        """Return tracked sensor-frame transforms as a ``wp.transformf`` array.

        Live path reads the ovphysx binding into the cached staging buffer
        every call; static path returns the cached snapshot directly.
        """
        if self._ovphysx_body_view is None:
            return self._static_view_transforms_wp
        self._ovphysx_body_view.read(self._pose_buf)
        return self._pose_buf_transformf

    def get_world_poses(self: Any, indices=None):
        """Return world poses as ``(positions, orientations)`` pose tuples.

        Camera-derived base classes inheriting this mixin call this method
        and read ``.torch`` on the returned objects. We mirror PhysX's
        :class:`SimpleNamespace` shape so the contract is identical.
        """
        transforms = self._get_view_transforms_wp()
        transforms_t = wp.to_torch(transforms).reshape(-1, 7)
        if indices is not None:
            idx = wp.to_torch(indices).to(dtype=torch.long) if isinstance(indices, wp.array) else indices
            transforms_t = transforms_t[idx]
        return SimpleNamespace(torch=transforms_t[:, 0:3]), SimpleNamespace(torch=transforms_t[:, 3:7])

    def _create_tracked_target_view(self: Any, target_prim_paths: str | list[str]):
        """Create an OVPhysX pose binding for exact planned target bodies."""
        from isaaclab_ov import tensor_types as TT  # noqa: PLC0415

        if isinstance(target_prim_paths, str):
            target_prim_paths = [target_prim_paths]
        if not target_prim_paths:
            raise RuntimeError(f"No tracked target bodies resolved from: {target_prim_paths}")

        physx = self._physics_manager.get_physx_instance()
        if physx is None:
            raise RuntimeError(
                "OvPhysxManager has no PhysX instance yet -- multi-mesh target view requested "
                "before PhysicsEvent.PHYSICS_READY."
            )
        view = physx.create_tensor_binding(prim_paths=target_prim_paths, tensor_type=TT.RIGID_BODY_POSE)
        if list(view.prim_paths) != target_prim_paths:
            raise ValueError("OVPhysX tracked ray-cast target binding does not match the clone plan.")
        return view

    def _update_mesh_transforms(self: Any) -> None:
        """Refresh dynamic multi-mesh target poses from their ovphysx bindings."""
        if not hasattr(self, "_mesh_views"):
            return
        mesh_idx = 0
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]
                continue

            # ``view`` here is an ovphysx TensorBinding produced by
            # :meth:`_create_tracked_target_view`. Each binding owns its own
            # staging buffer cached in ``self._mesh_view_bufs`` (initialized
            # in :meth:`_initialize_pose_tracking`).
            buf = self._mesh_view_bufs.get(id(view))
            if buf is None:
                buf = wp.zeros(view.shape, dtype=wp.float32, device=self._device)
                self._mesh_view_bufs[id(view)] = buf

            view.read(buf)
            transforms_wp = wp.array(
                ptr=buf.ptr,
                shape=(int(view.shape[0]),),
                dtype=wp.transformf,
                device=str(buf.device),
                copy=False,
            )

            view_count = int(view.shape[0])
            meshes_per_env = view_count
            if view_count != 1:
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

    def _invalidate_initialize_callback(self: Any, event) -> None:
        """Release ovphysx native handles when the simulation stops."""
        super()._invalidate_initialize_callback(event)
        view = getattr(self, "_ovphysx_body_view", None)
        if view is not None:
            with contextlib.suppress(Exception):
                view.destroy()
        self._ovphysx_body_view = None

        for buf_view in getattr(self, "_mesh_views", []) or []:
            with contextlib.suppress(Exception):
                buf_view.destroy()
        self._mesh_views = []
        self._mesh_view_bufs = {}


class RayCaster(_OvPhysxRayCasterMixin, BaseRayCaster):
    """OVPhysX RayCaster implementation."""
