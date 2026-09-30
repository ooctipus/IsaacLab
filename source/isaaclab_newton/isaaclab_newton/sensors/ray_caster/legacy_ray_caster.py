# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Legacy Newton adapters for Warp-mesh ray-caster implementations."""

from __future__ import annotations

# pyright: reportInvalidTypeForm=none, reportPrivateUsage=none
import warnings
from typing import Any

import warp as wp

from isaaclab.sensors.ray_caster.base_multi_mesh_ray_caster import BaseMultiMeshRayCaster
from isaaclab.sensors.ray_caster.base_multi_mesh_ray_caster_camera import BaseMultiMeshRayCasterCamera
from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster
from isaaclab.sensors.ray_caster.base_ray_caster_camera import BaseRayCasterCamera
from isaaclab.sensors.ray_caster.kernels import copy_mesh_poses_to_table_kernel

from .newton_raycast_sensor import _NewtonRayCasterPoseMixin


class _LegacyNewtonRayCasterMixin(_NewtonRayCasterPoseMixin):
    """Add explicit mesh-target tracking required by legacy ray casters."""

    def _create_tracked_target_view(self: Any, target_prim_path: str | list[str]) -> wp.array:
        """Resolve exact planned target bodies to Newton body indices."""
        paths = target_prim_path if isinstance(target_prim_path, list) else [target_prim_path]
        indices = {path: index for index, path in enumerate(self._physics_manager.get_model().body_label)}
        try:
            return wp.array([indices[path] for path in paths], dtype=wp.int32, device=self._device)
        except KeyError as exc:
            raise ValueError(f"Newton model is missing planned ray-cast target body {exc.args[0]!r}.") from exc

    def _update_mesh_transforms(self: Any) -> None:
        """Refresh dynamic multi-mesh targets from Newton bodies."""
        if not hasattr(self, "_mesh_views"):
            return
        mesh_index = 0
        for body_indices, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                mesh_index += self._num_meshes_per_env[target_cfg.prim_expr]
                continue

            body_count = body_indices.shape[0]
            pos_buf = wp.empty(body_count, dtype=wp.vec3f, device=self._device)
            quat_buf = wp.empty(body_count, dtype=wp.quatf, device=self._device)
            pose_buf = wp.empty(body_count, dtype=wp.transformf, device=self._device)
            self._update_newton_body_transforms(body_indices, pose_buf, pos_buf, quat_buf)
            meshes_per_env = body_count if body_count == 1 else body_count // self._num_envs

            wp.launch(
                copy_mesh_poses_to_table_kernel,
                dim=(self._num_envs, meshes_per_env),
                inputs=[
                    pos_buf,
                    quat_buf,
                    int(meshes_per_env),
                    int(mesh_index),
                    bool(body_count == 1),
                    self._mesh_positions_w,
                    self._mesh_orientations_w,
                ],
                device=self._device,
            )
            mesh_index += self._num_meshes_per_env[target_cfg.prim_expr]


class LegacyRayCaster(_LegacyNewtonRayCasterMixin, BaseRayCaster):
    """Legacy Newton ray caster that queries one configured Warp mesh."""


class LegacyRayCasterCamera(_LegacyNewtonRayCasterMixin, BaseRayCasterCamera):
    """Legacy Newton ray-caster camera backed by configured Warp meshes."""


class LegacyMultiMeshRayCaster(_LegacyNewtonRayCasterMixin, BaseMultiMeshRayCaster):
    """Legacy Newton ray caster for configured static and dynamic Warp meshes."""


class LegacyMultiMeshRayCasterCamera(_LegacyNewtonRayCasterMixin, BaseMultiMeshRayCasterCamera):
    """Legacy Newton ray-caster camera for configured Warp meshes."""


def _warn_legacy_alias(old_name: str, new_name: str) -> None:
    """Warn when a pre-rename Newton backend class is constructed directly."""
    warnings.warn(
        f"isaaclab_newton.sensors.{old_name} is deprecated; use isaaclab_newton.sensors.{new_name} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class RayCasterCamera(LegacyRayCasterCamera):
    """Deprecated alias for :class:`LegacyRayCasterCamera`."""

    def __init__(self, cfg):
        _warn_legacy_alias("RayCasterCamera", "LegacyRayCasterCamera")
        super().__init__(cfg)


class MultiMeshRayCaster(LegacyMultiMeshRayCaster):
    """Deprecated alias for :class:`LegacyMultiMeshRayCaster`."""

    def __init__(self, cfg):
        _warn_legacy_alias("MultiMeshRayCaster", "LegacyMultiMeshRayCaster")
        super().__init__(cfg)


class MultiMeshRayCasterCamera(LegacyMultiMeshRayCasterCamera):
    """Deprecated alias for :class:`LegacyMultiMeshRayCasterCamera`."""

    def __init__(self, cfg):
        _warn_legacy_alias("MultiMeshRayCasterCamera", "LegacyMultiMeshRayCasterCamera")
        super().__init__(cfg)
