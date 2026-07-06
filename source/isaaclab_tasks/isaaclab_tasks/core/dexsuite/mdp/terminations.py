# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the dexsuite task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


class out_of_bound(ManagerTermBase):
    """Termination condition for when the object falls out of bound.

    This class-based implementation caches the asset reference, ranges tensor, and env_origins
    to avoid recomputing them on every call.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self._object: RigidObject = env.scene[asset_cfg.name]

        in_bound_range: dict[str, tuple[float, float]] = cfg.params.get("in_bound_range", {})
        range_list = [in_bound_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device=env.device, dtype=torch.float32)

        # Pre-apply env_origins so we can compare directly against world-space positions.
        origins = env.scene.env_origins  # (N, 3)
        self._lower = origins + ranges[:, 0]  # (N, 3)
        self._upper = origins + ranges[:, 1]  # (N, 3)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        in_bound_range: dict[str, tuple[float, float]] = {},
    ) -> torch.Tensor:
        pos_w = self._object.data.root_pos_w.torch
        return ((pos_w < self._lower) | (pos_w > self._upper)).any(dim=1)


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate environments with non-finite robot state or violated velocity limits."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos.torch
    joint_vel = robot.data.joint_vel.torch
    joint_vel_limits = robot.data.joint_vel_limits.torch
    finite = torch.isfinite(joint_pos).all(dim=1) & torch.isfinite(joint_vel).all(dim=1)
    safe_joint_vel = torch.where(finite[:, None], joint_vel, torch.zeros_like(joint_vel))
    return (~finite) | (safe_joint_vel.abs() > (joint_vel_limits * 2)).any(dim=1)


def abnormal_object_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    max_ang_vel: float = 1000.0,
) -> torch.Tensor:
    """Terminate environments with non-finite or excessive object angular velocity.

    Args:
        env: Environment instance.
        asset_cfg: Scene entity for the manipulated object.
        max_ang_vel: Maximum allowed object angular speed [rad/s].

    Returns:
        Boolean tensor indicating environments that should terminate.
    """
    obj: RigidObject = env.scene[asset_cfg.name]
    pos = obj.data.root_pos_w.torch
    quat = obj.data.root_quat_w.torch
    lin_vel = obj.data.root_lin_vel_w.torch
    ang_vel = obj.data.root_ang_vel_w.torch
    finite = (
        torch.isfinite(pos).all(dim=1)
        & torch.isfinite(quat).all(dim=1)
        & torch.isfinite(lin_vel).all(dim=1)
        & torch.isfinite(ang_vel).all(dim=1)
    )
    safe_ang_vel = torch.where(finite[:, None], ang_vel, torch.zeros_like(ang_vel))
    ang_speed = torch.linalg.norm(safe_ang_vel, dim=1)
    return (~finite) | (ang_speed > max_ang_vel)
