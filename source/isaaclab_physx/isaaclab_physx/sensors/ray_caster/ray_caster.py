# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import warp as wp

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.cloner.cloner_utils import ascend_source_prims, source_prim
from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster
from isaaclab.sensors.ray_caster.kernels import copy_mesh_transforms_to_table_kernel
from isaaclab.utils.math import combine_frame_transforms

from isaaclab_physx.physics import PhysxManager


def _physx_body_glob(body_expr: str) -> str:
    """Convert internal env regex/template expressions to PhysX glob syntax."""
    return body_expr.replace("{}", "*").replace(".*", "*")


class _PhysXRayCasterMixin:
    """PhysX pose tracking for ray-caster sensors.

    PhysX can provide live rigid-body transforms after physics is ready. Static
    non-physics prims are cached once at initialization; they are intentionally
    not polled through USD during sensor updates.
    """

    @property
    def count(self: Any) -> int:
        """Number of tracked sensor frames."""
        return self._view_count

    def _initialize_pose_tracking(self: Any) -> None:
        """Initialize direct PhysX body tracking or a cached static pose table."""
        # ``self._view.count`` is read by the base classes; route it back to this sensor.
        self._view = self
        sensor_prim, _, destination, _ = source_prim(self._clone_plan, self.cfg.prim_path)
        bodies = ascend_source_prims(self._clone_plan, self.cfg.prim_path, lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI))
        if not bodies:
            self._initialize_static_pose_tracking()
            return
        body_prim, _, _, _ = bodies[0]

        # If the public prim path points at the rigid body itself, BaseRayCaster spawned a child
        # sensor prim under it and stashed the original body path on ``_requested_prim_path``.
        requested = getattr(self, "_requested_prim_path", self.cfg.prim_path)
        body_rel = body_prim.GetPath().pathString[len(sensor_prim.GetPath().pathString) :]
        body_glob = _physx_body_glob(
            destination if self.cfg.prim_path != requested or not body_rel else destination + body_rel
        )

        self._physx_body_view = PhysxManager.get_physics_sim_view().create_rigid_body_view(body_glob)
        self._view_count = self._physx_body_view.count
        p, q = sim_utils.resolve_prim_pose(sensor_prim, body_prim)
        self._offset_pos_wp = wp.array([p] * self._view_count, dtype=wp.vec3f, device=self._device)
        self._offset_quat_contiguous = torch.tensor([q] * self._view_count, dtype=torch.float32, device=self._device)
        self._offset_quat_wp = wp.from_torch(self._offset_quat_contiguous, dtype=wp.quatf)

    def _initialize_static_pose_tracking(self: Any) -> None:
        """Cache per-env world poses via Win A: ``env_pose ⊗ source-relative pose``."""
        plan = self._clone_plan
        sensor_prim, source_root, _, env_ids = source_prim(plan, self.cfg.prim_path)
        rel_pos, rel_quat = sim_utils.resolve_prim_pose(
            sensor_prim, sim_utils.get_current_stage().GetPrimAtPath(source_root)
        )
        env_pose = plan.env_pose[torch.as_tensor(env_ids, dtype=torch.long, device=plan.env_pose.device)]
        rel_pos_t = torch.as_tensor(rel_pos, dtype=env_pose.dtype, device=env_pose.device).expand(len(env_ids), 3)
        rel_quat_t = torch.as_tensor(rel_quat, dtype=env_pose.dtype, device=env_pose.device).expand(len(env_ids), 4)
        world_pos, world_quat = combine_frame_transforms(env_pose[:, :3], env_pose[:, 3:], rel_pos_t, rel_quat_t)
        self._static_view_transforms_torch = (
            torch.cat([world_pos, world_quat], dim=-1).to(self._device, torch.float32).contiguous()
        )
        self._static_view_transforms_wp = wp.from_torch(self._static_view_transforms_torch).view(wp.transformf)
        self._physx_body_view = None
        self._view_count = len(env_ids)
        self._offset_pos_wp = wp.zeros(self._view_count, dtype=wp.vec3f, device=self._device)
        self._offset_quat_contiguous = torch.zeros(self._view_count, 4, device=self._device)
        self._offset_quat_contiguous[:, 3] = 1.0
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
        body_paths = []
        for target_prim_path in target_prim_paths:
            prims = sim_utils.find_matching_prims(target_prim_path)
            if len(prims) == 0:
                # ClonePlan-backed targets may not have destination mesh prims.
                # In that case BaseMultiMeshRayCaster passes the destination owner-body expression.
                body_paths.append(target_prim_path)
                continue
            for prim in prims:
                body = sim_utils.get_first_matching_ancestor_prim(
                    prim.GetPath().pathString, lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI)
                )
                if body is None:
                    raise RuntimeError(
                        f"Cannot track non-physics ray-cast target '{target_prim_path}' with PhysX. "
                        "Set track_mesh_transforms=False for static targets, or apply RigidBodyAPI to dynamic targets."
                    )
                body_paths.append(body.GetPath().pathString)

        if len(body_paths) == 0:
            raise RuntimeError(f"No tracked target bodies resolved from: {target_prim_paths}")
        physics_sim_view = PhysxManager.get_physics_sim_view()
        if physics_sim_view is None:
            raise RuntimeError("PhysX simulation view is not initialized.")
        return physics_sim_view.create_rigid_body_view([_physx_body_glob(path) for path in body_paths])

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
