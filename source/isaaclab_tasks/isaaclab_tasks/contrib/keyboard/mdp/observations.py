# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the SO101 keyboard letter-typing task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.utils.math import subtract_frame_transforms

from ..newton_selection import NewtonSelection

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import LetterTypingCommand


def _slots_onehot(slots: torch.Tensor, num_keys: int) -> torch.Tensor:
    """One-hot a ``(num_envs, max_len)`` slot-id buffer (``-1`` marks empty) into ``(num_envs, max_len * num_keys)``.

    Empty/padding positions (slot ``-1``) become all-zero rows, so they contribute nothing.
    """
    valid = slots >= 0
    onehot = torch.nn.functional.one_hot(slots.clamp(min=0), num_classes=num_keys).to(torch.float)
    onehot = onehot * valid.unsqueeze(-1).to(onehot.dtype)
    return onehot.reshape(slots.shape[0], -1)


def target_keys_onehot(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Target word as a one-hot sequence over key slots, shape ``(num_envs, max_len * num_keys)``.

    Each sequence position holds a one-hot of that target key (all-zero past the word's length). Pair with
    the perception key-position map (same global slot order) so a gather/attention actor can recover where
    each target key is.
    """
    command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
    active = command.key_joints.dense_active().gather(1, command.target.clamp(min=0))
    return _slots_onehot(torch.where(active, command.target, -1), command.num_keys)


def typed_keys_onehot(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Typed buffer as a one-hot sequence over key slots, shape ``(num_envs, max_len * num_keys)``.

    Each sequence position holds a one-hot of the key typed there so far (all-zero for not-yet-typed
    positions). Comparing it against :func:`target_keys_onehot` yields the typing progress and any
    mistake the agent must backspace.
    """
    command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
    active = command.key_joints.dense_active().gather(1, command.typed.clamp(min=0))
    return _slots_onehot(torch.where(active, command.typed, -1), command.num_keys)


def joint_pos(env: ManagerBasedRLEnv, joints: NewtonSelection) -> torch.Tensor:
    """Selected joint coordinates [m or rad, depending on joint type]."""
    return joints.dense(NewtonManager.get_state().joint_q)


def joint_vel(env: ManagerBasedRLEnv, joints: NewtonSelection) -> torch.Tensor:
    """Selected joint velocities [m/s or rad/s, depending on joint type]."""
    return joints.dense(NewtonManager.get_state().joint_qd)


def key_positions_b(env: ManagerBasedRLEnv, keys: NewtonSelection, root: NewtonSelection) -> torch.Tensor:
    """Key positions [m] relative to exactly one robot root per world, in stable slot order."""
    if any(count != 1 for count in root.counts):
        raise ValueError("Relative key positions require exactly one root per world.")
    state = NewtonManager.get_state()
    poses = wp.to_torch(state.body_q)
    root_pose = poses[root.dense_ids()]
    key_pose = poses[keys.dense_ids()]
    pos, _ = subtract_frame_transforms(root_pose[..., :3], root_pose[..., 3:], key_pose[..., :3])
    active = keys.dense_active() & root.dense_active()
    return torch.where(active.unsqueeze(-1), pos, 0.0).flatten(1)


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous policy actions with episode-excluded DOFs cleared."""
    active = env.action_manager.get_term("action").dofs.dense_active()
    return torch.where(active, env.action_manager.action, 0.0)
