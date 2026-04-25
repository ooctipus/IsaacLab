# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


def command_current_state(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Current state: root pose/vel (12D, env-local position) + joint positions.

    Layout: ``[x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, joint_pos...]``.

    Including joint positions ensures CRL (via HER relabeling) learns to
    match the full robot configuration, not just the root pose.

    Args:
        env: :class:`ManagerBasedRLEnv` instance.
        command_name: Name of the :class:`RelativeStateCommand` term.

    Returns:
        Tensor of shape ``[num_envs, 12 + num_joints]``.
    """
    import warp as wp

    cmd = env.command_manager.get_term(command_name)
    buf = cmd.cmd_buf[:, 2]
    pos_local = buf[:, :3] - env.scene.terrain.env_origins
    joint_pos = wp.to_torch(cmd.robot.data.joint_pos)
    return torch.cat([pos_local, buf[:, 3:12], joint_pos], dim=-1)


def command_target_state(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Target state: root pose/vel (12D, env-local position) + joint positions.

    Layout matches :func:`command_current_state`. The joint portion uses
    the robot's current joints as placeholder — HER replaces the entire
    target with a future ``current_state``, so the placeholder values
    are never used for training.

    Args:
        env: :class:`ManagerBasedRLEnv` instance.
        command_name: Name of the :class:`RelativeStateCommand` term.

    Returns:
        Tensor of shape ``[num_envs, 12 + num_joints]``.
    """
    import warp as wp

    cmd = env.command_manager.get_term(command_name)
    buf = cmd.cmd_buf[:, 0]
    pos_local = buf[:, :3] - env.scene.terrain.env_origins
    joint_pos = wp.to_torch(cmd.robot.data.joint_pos)
    return torch.cat([pos_local, buf[:, 3:12], joint_pos], dim=-1)
