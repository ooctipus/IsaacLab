# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def success_termination(env: ManagerBasedRLEnv, command_name: str = "reset_state") -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    return command_term.get_task_done()


def split_time_out(
    env: ManagerBasedRLEnv,
    short_episode_length_s: float = 2.0,
    split_ratio: float = 0.5,
) -> torch.Tensor:
    """Timeout with a shorter episode length for the first ``split_ratio`` fraction of envs.

    The first ``split_ratio * num_envs`` envs use ``short_episode_length_s`` as their
    timeout. The remaining envs use the environment's default ``max_episode_length``.

    Args:
        short_episode_length_s: Episode length [s] for the short-horizon group.
        split_ratio: Fraction of envs in the short-horizon group.
    """
    n_short = int(env.num_envs * split_ratio)
    short_max_length = int(short_episode_length_s / env.step_dt)
    result = env.episode_length_buf >= env.max_episode_length
    result[:n_short] = env.episode_length_buf[:n_short] >= short_max_length

    return result
