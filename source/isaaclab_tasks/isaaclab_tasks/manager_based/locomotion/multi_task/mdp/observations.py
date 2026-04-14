# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.envs import ManagerBasedRLEnv


def time_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    if hasattr(env, "episode_length_buf"):
        life_left = (1 - env.episode_length_buf.float() / env.max_episode_length) * env.max_episode_length_s
    else:
        life_left = torch.ones(env.num_envs, device=env.device, dtype=torch.float) * env.max_episode_length_s
    return life_left.view(-1, 1)
