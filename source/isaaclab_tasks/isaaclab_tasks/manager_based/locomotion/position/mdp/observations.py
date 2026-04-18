# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    if hasattr(env, "episode_length_buf"):
        life_left = (1 - env.episode_length_buf.float() / env.max_episode_length) * env.max_episode_length_s
    else:
        life_left = torch.ones(env.num_envs, device=env.device, dtype=torch.float) * env.max_episode_length_s
    return life_left.view(-1, 1)


def target_pos_env(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Commanded target position expressed in the per-env local frame [m].

    Built for CRL: the commanded goal must be an *absolute* reachable-pose slice
    (not a relative-state delta) so that Hindsight Experience Replay can relabel
    with reached poses from the same trajectory.

    The returned vector is the commanded world-position minus the env's terrain-
    spawn origin, keeping the coordinate range stable across the many parallel
    envs that live at different world locations.

    Args:
        env: :class:`ManagerBasedRLEnv` instance.
        command_name: Name of the :class:`RelativeStateCommand` term. Defaults to
            ``"goal_point"`` matching the position task.

    Returns:
        Tensor of shape ``[num_envs, 3]`` with ``(x, y, z)`` targets [m] in the
        per-env local frame.
    """
    command_term = env.command_manager.get_term(command_name)
    env_origins = env.scene.terrain.env_origins  # [num_envs, 3]
    return command_term.cmd_buf[:, 0, :3] - env_origins


def achieved_pos_env(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Currently achieved root position expressed in the per-env local frame [m].

    The HER-compatible companion to :func:`target_pos_env`: at any timestep this
    returns the agent's reached pose in the same coordinate frame as the
    commanded target. Sampling a future timestep's achieved pose gives CRL an
    automatically-correct relabeled goal.

    Args:
        env: :class:`ManagerBasedRLEnv` instance.
        command_name: Name of the :class:`RelativeStateCommand` term.

    Returns:
        Tensor of shape ``[num_envs, 3]`` with the robot root position [m]
        relative to the terrain spawn origin for that env.
    """
    command_term = env.command_manager.get_term(command_name)
    env_origins = env.scene.terrain.env_origins  # [num_envs, 3]
    return command_term.cmd_buf[:, 2, :3] - env_origins
