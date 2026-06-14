# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

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


def progress_reward(env: ManagerBasedRLEnv, std: float, command_name: str = "reset_state") -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    error = command_term.error
    thresholds = command_term.command_std
    orientation_aligned = error[:, 0] < thresholds[:, 0]
    position_distance = error[:, 1]
    return torch.where(orientation_aligned, 1 - torch.tanh(position_distance / std), 0.0)


def success_reward(env: ManagerBasedRLEnv, command_name: str = "reset_state") -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    return command_term.get_task_reward()
