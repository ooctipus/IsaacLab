# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation

"""
MDP terminations.
"""


def success(
    env: ManagerBasedRLEnv,
    std: tuple[float, float],
    command: str = "goal_point",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[robot_cfg.name]
    cmd: torch.Tensor = env.command_manager.get_command(command)
    dist = cmd[:, :3].norm(2, -1)
    head = cmd[:, 3].abs()
    speed = asset.data.body_lin_vel_w.norm(2, dim=-1).amax(dim=1)
    return ((dist < std[0]) & (head < std[1]) & (speed < 0.5))
