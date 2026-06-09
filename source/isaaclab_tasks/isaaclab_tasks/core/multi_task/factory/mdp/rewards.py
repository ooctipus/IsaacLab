# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def reach_reward(env: ManagerBasedRLEnv, held_asset_cfg: SceneEntityCfg, ee_cfg: SceneEntityCfg, std: float = 0.1):
    robot: Articulation = env.scene[ee_cfg.name]
    held_asset: RigidObject = env.scene[held_asset_cfg.name]
    ee_pos_w = wp.to_torch(robot.data.body_pos_w)[:, ee_cfg.body_ids].view(env.num_envs, -1)
    nut_pos_w = wp.to_torch(held_asset.data.root_pos_w)
    dist = torch.norm((nut_pos_w - ee_pos_w), dim=1)
    return 1 - torch.tanh(dist / std)


def orientation_reward(env: ManagerBasedRLEnv, std: float, context: str = "progress_context") -> torch.Tensor:
    context_term: ManagerTermBase = env.termination_manager.get_term_cfg(context).func  # type: ignore
    euler_xy_diff: torch.Tensor = getattr(context_term, "euler_xy_diff")
    return 1 - torch.tanh(euler_xy_diff / std)


def concentric_reward(env: ManagerBasedRLEnv, std: float, context: str = "progress_context") -> torch.Tensor:
    context_term: ManagerTermBase = env.termination_manager.get_term_cfg(context).func  # type: ignore
    xy_distance: torch.Tensor = getattr(context_term, "xy_distance")
    orientation_aligned: torch.Tensor = getattr(context_term, "orientation_aligned")
    return torch.where(orientation_aligned, 1 - torch.tanh(xy_distance / std), 0.0)


def progress_reward(env: ManagerBasedRLEnv, std: float, context: str = "progress_context") -> torch.Tensor:
    context_term: ManagerTermBase = env.termination_manager.get_term_cfg(context).func  # type: ignore
    orientation_aligned: torch.Tensor = getattr(context_term, "orientation_aligned")
    position_centered: torch.Tensor = getattr(context_term, "position_centered")
    z_distance: torch.Tensor = getattr(context_term, "z_distance")
    return torch.where(orientation_aligned & position_centered, 1 - torch.tanh(z_distance / std), 0.0)


def success_reward(env: ManagerBasedRLEnv, context: str = "progress_context") -> torch.Tensor:
    context_term: ManagerTermBase = env.termination_manager.get_term_cfg(context).func  # type: ignore
    orientation_aligned: torch.Tensor = getattr(context_term, "orientation_aligned")
    position_centered: torch.Tensor = getattr(context_term, "position_centered")
    z_distance_reached: torch.Tensor = getattr(context_term, "z_distance_reached")
    return torch.where(orientation_aligned & position_centered & z_distance_reached, 1.0, 0.0)


class unstable_manipulation(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        self.held_asset: RigidObject = env.scene[cfg.params.get("held_asset_cfg").name]
        self.robot: Articulation = env.scene[cfg.params.get("ee_cfg").name]
        self.last_relative_position = torch.zeros((env.num_envs, 6), device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, ee_cfg: SceneEntityCfg, held_asset_cfg: SceneEntityCfg) -> torch.Tensor:
        asset_vel_w = wp.to_torch(self.held_asset.data.root_vel_w)
        asset_pos_w = wp.to_torch(self.held_asset.data.root_pos_w)
        asset_quat_w = wp.to_torch(self.held_asset.data.root_quat_w)
        ee_pos_w = wp.to_torch(self.robot.data.body_link_pos_w)[:, ee_cfg.body_ids].view(-1, 3)
        ee_quat_w = wp.to_torch(self.robot.data.body_link_quat_w)[:, ee_cfg.body_ids].view(-1, 4)

        pos_e, rot_e = math_utils.compute_pose_error(asset_pos_w, asset_quat_w, ee_pos_w, ee_quat_w)
        current_relative_pose = torch.cat((pos_e, math_utils.wrap_to_pi(rot_e) * 0.25), dim=1)
        delta_pose = current_relative_pose - self.last_relative_position
        stable_manip = (delta_pose.abs().sum(dim=1) < 0.1) | (asset_vel_w.abs().sum(dim=1) < 0.1)
        self.last_relative_position[:] = current_relative_pose
        asset_initialized = env.episode_length_buf > 10
        return ((~stable_manip) & asset_initialized).float()
