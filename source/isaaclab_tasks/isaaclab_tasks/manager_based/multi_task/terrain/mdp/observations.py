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
    """Current state: root pose/vel (env-local position) + foot positions.

    Layout: ``[x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, foot_pos...]``
    where ``foot_pos`` is ``num_feet`` positions in env-local world frame
    (world minus env origin), flattened.

    Foot positions match the payload success criterion, so HER relabeling
    ``target ← future current`` produces a self-consistent goal whose error
    matches the actual reward signal.

    Args:
        env: :class:`ManagerBasedRLEnv` instance.
        command_name: Name of the :class:`RelativeStateCommand` term.

    Returns:
        Tensor of shape ``[num_envs, 12 + 3 * num_feet]``.
    """
    cmd = env.command_manager.get_term(command_name)
    env_origins = env.scene.terrain.env_origins
    return cmd.current_state_env(env_origins)


def command_std(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Per-env success thresholds of the currently-bound task as a policy observation.

    Returns the active payload's per-error-group thresholds so the policy can
    see the threshold alongside the raw command delta.

    Args:
        env: :class:`ManagerBasedRLEnv` instance.
        command_name: Name of the :class:`RelativeStateCommand` term.

    Returns:
        Tensor of shape ``[num_envs, num_error_groups]``.
    """
    return env.command_manager.get_term(command_name).command_std


def command_target_state(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Target state: root pose/vel (env-local position) + foot positions.

    Layout matches :func:`command_current_state`. The foot portion is the
    commanded target foot positions in env-local world frame.

    Args:
        env: :class:`ManagerBasedRLEnv` instance.
        command_name: Name of the :class:`RelativeStateCommand` term.

    Returns:
        Tensor of shape ``[num_envs, 12 + 3 * num_feet]``.
    """
    cmd = env.command_manager.get_term(command_name)
    env_origins = env.scene.terrain.env_origins
    return cmd.target_state_env(env_origins)
