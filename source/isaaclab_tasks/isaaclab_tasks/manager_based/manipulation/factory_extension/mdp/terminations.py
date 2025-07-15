# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObjectCollection


def out_of_bounds(
    env: DataManagerBasedRLEnv,
    assets_cfg: SceneEntityCfg,
    pos_range: tuple[tuple[float, float, float], tuple[float, float, float]],
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Check if the robot is out of bounds."""
    assets: RigidObjectCollection = env.scene[assets_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    objects_pos_b = assets.data.object_pos_w - robot.data.root_pos_w.unsqueeze(1)  # (n_envs, n_objects, 3)
    low = torch.tensor(pos_range[0], device=objects_pos_b.device)
    high = torch.tensor(pos_range[1], device=objects_pos_b.device)
    below_min = objects_pos_b < low      # (n_envs, n_objects, 3)
    above_max = objects_pos_b > high     # (n_envs, n_objects, 3)
    object_oob = (below_min | above_max).any(dim=-1)  # (n_envs, n_objects)
    env_oob = object_oob.any(dim=-1)  # (n_envs,)

    return env_oob
