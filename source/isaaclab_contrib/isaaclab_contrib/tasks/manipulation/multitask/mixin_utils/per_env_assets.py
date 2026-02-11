# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

# Import the actual PhysX implementations directly to avoid Factory pattern issues
from isaaclab_physx.assets import Articulation as PhysXArticulation
from isaaclab_physx.assets import RigidObject as PhysXRigidObject
from isaaclab_physx.assets import SurfaceGripper as PhysXSurfaceGripper
from isaaclab_physx.assets import SurfaceGripperCfg

from isaaclab.assets import ArticulationCfg, RigidObjectCfg

from .per_env_mixin import PerEnvMixin


class PerEnvRigidObject(PerEnvMixin, PhysXRigidObject):
    """Rigid object that only operates on a specified subset of environments."""

    cfg: RigidObjectCfg

    def __init__(self, cfg: RigidObjectCfg):
        super().__init__(cfg)

    def reset(self, env_ids: Sequence[int] | None = None):
        resolved_env_ids = self._filter_env_ids(env_ids)
        if resolved_env_ids.numel() == 0:
            return

        super().reset(resolved_env_ids)

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        resolved_env_ids = self._filter_env_ids(env_ids)
        if resolved_env_ids.numel() == 0:
            return
        super().write_root_pose_to_sim(root_pose, resolved_env_ids)

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        resolved_env_ids = self._filter_env_ids(env_ids)
        if resolved_env_ids.numel() == 0:
            return
        super().write_root_velocity_to_sim(root_velocity, resolved_env_ids)


class PerEnvArticulation(PerEnvMixin, PhysXArticulation):
    """Articulation that only operates on a specified subset of environments."""

    cfg: ArticulationCfg

    def __init__(self, cfg: ArticulationCfg):
        super().__init__(cfg)

    def reset(self, env_ids: Sequence[int] | None = None):
        resolved_env_ids = self._filter_env_ids(env_ids)
        if resolved_env_ids.numel() == 0:
            return

        super().reset(resolved_env_ids)

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        resolved_env_ids = self._filter_env_ids(env_ids)
        if resolved_env_ids.numel() == 0:
            return
        super().write_root_pose_to_sim(root_pose, resolved_env_ids)

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        resolved_env_ids = self._filter_env_ids(env_ids)
        if resolved_env_ids.numel() == 0:
            return
        super().write_root_velocity_to_sim(root_velocity, resolved_env_ids)

    def write_joint_state_to_sim(
        self,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        resolved_env_ids = self._filter_env_ids(env_ids)
        if resolved_env_ids.numel() == 0:
            return
        super().write_joint_state_to_sim(joint_position, joint_velocity, joint_ids, resolved_env_ids)

    def set_joint_position_target(
        self,
        target: torch.Tensor,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        resolved_env_ids = self._filter_env_ids(env_ids)
        if resolved_env_ids.numel() == 0:
            return
        super().set_joint_position_target(target, joint_ids, resolved_env_ids)

    def set_joint_velocity_target(
        self,
        target: torch.Tensor,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        resolved_env_ids = self._filter_env_ids(env_ids)
        if resolved_env_ids.numel() == 0:
            return
        super().set_joint_velocity_target(target, joint_ids, resolved_env_ids)


class PerEnvSurfaceGripper(PerEnvMixin, PhysXSurfaceGripper):
    """Surface gripper that only operates on a specified subset of environments."""

    cfg: SurfaceGripperCfg

    def __init__(self, cfg: SurfaceGripperCfg):
        super().__init__(cfg)

    def reset(self, indices: torch.Tensor | None = None) -> None:
        resolved = self._filter_env_ids(indices)
        if resolved.numel() == 0:
            return
        super().reset(resolved)

    def set_grippers_command(self, states: torch.Tensor, indices: torch.Tensor | None = None) -> None:
        resolved = self._filter_env_ids(indices)
        if resolved.numel() == 0:
            return
        if indices is not None:
            env_ids_list = indices.tolist() if hasattr(indices, "tolist") else list(indices)
            keep = [i for i, g in enumerate(env_ids_list) if g in self._assigned_env_to_local]
            if not keep:
                return
            keep_t = torch.tensor(keep, dtype=torch.long, device=states.device)
            states = states[keep_t]
        super().set_grippers_command(states, resolved)

    def update_gripper_properties(
        self,
        max_grip_distance: torch.Tensor | None = None,
        coaxial_force_limit: torch.Tensor | None = None,
        shear_force_limit: torch.Tensor | None = None,
        retry_interval: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ) -> None:
        resolved = self._filter_env_ids(indices)
        if resolved.numel() == 0:
            return
        if indices is not None:
            env_ids_list = indices.tolist() if hasattr(indices, "tolist") else list(indices)
            keep = [i for i, g in enumerate(env_ids_list) if g in self._assigned_env_to_local]
            if not keep:
                return
            keep_t = torch.tensor(keep, dtype=torch.long, device=self.device)
            if max_grip_distance is not None:
                max_grip_distance = max_grip_distance[keep_t]
            if coaxial_force_limit is not None:
                coaxial_force_limit = coaxial_force_limit[keep_t]
            if shear_force_limit is not None:
                shear_force_limit = shear_force_limit[keep_t]
            if retry_interval is not None:
                retry_interval = retry_interval[keep_t]
        super().update_gripper_properties(
            max_grip_distance=max_grip_distance,
            coaxial_force_limit=coaxial_force_limit,
            shear_force_limit=shear_force_limit,
            retry_interval=retry_interval,
            indices=resolved,
        )
