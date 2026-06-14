# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from ...mdp.commands.state_command.state_command import StateCommand


def command_success(env: ManagerBasedRLEnv):
    command_term: StateCommand = env.command_manager.get_term("goal_point")
    return command_term.get_task_reward()


def exploration_reward(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), forward_only: bool = False
):
    # Retrieve the robot and target data
    robot: Articulation = env.scene[robot_cfg.name]
    base_velocity = robot.data.root_lin_vel_b  # Robot's current base velocity vector
    target_position = env.command_manager.get_command("goal_point")[:, :3]  # Target position relative to robot base

    # Base directional alignment (cosine similarity)
    cos_align = F.cosine_similarity(base_velocity[:, :3], target_position, dim=-1, eps=1e-6)

    if not forward_only:
        return cos_align

    # Forward preference weight: positive forward component relative to speed
    speed = torch.linalg.vector_norm(base_velocity, ord=2, dim=-1)
    forward_comp = base_velocity[:, 0].clamp_min(0)
    forward_weight = forward_comp / (speed + 1e-6)

    return cos_align * forward_weight
