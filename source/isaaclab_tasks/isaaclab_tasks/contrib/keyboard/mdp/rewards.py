# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the SO101 keyboard letter-typing task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import LetterTypingCommand


def mechanical_power(env: ManagerBasedRLEnv, joints, action_name: str = "action") -> torch.Tensor:
    """Absolute implicit-PD telemetry power [W], sampled at the existing physics-step boundary."""
    from isaaclab_newton.physics import NewtonManager

    action = env.action_manager.get_term(action_name)
    power = (action.applied_effort * joints.dense(NewtonManager.get_state().joint_qd)).abs().sum(dim=1)
    return torch.where(torch.isfinite(power), power, torch.zeros_like(power))


def letter_typing_progress(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    r"""High-water-mark typing-progress reward: a per-episode ratchet on the correct-prefix length.

    Fetches the :class:`LetterTypingCommand` term and returns a sparse per-step signal driven only by
    *records* in the length of the contiguous correct prefix (``prefix_len``):

    * ``+1`` on any step that sets a new episode maximum prefix length (genuine forward progress),
    * ``-1`` on any step that sets a new episode minimum prefix length (destroying correct progress),
    * ``0`` otherwise - staying put, or re-reaching a prefix length already seen this episode.

    Since one control step registers at most one keystroke, ``prefix_len`` moves by at most one, so each
    integer level is crossed cleanly. Rewarding only *new* maxima means the total positive reward per
    episode is bounded by ``target_len - prefix_0`` and the total penalty by ``prefix_0`` (with
    ``prefix_0`` the correct prefix of the reset buffer); consequently ``type-wrong -> backspace`` and
    ``backspace -> retype`` loops cannot farm reward - a re-reached level is neither a new max nor a new
    min. The marks are episode state seeded at reset, so they live on the command (see
    :meth:`~...typing_commands.LetterTypingCommand._update_metrics`); this term only reads the resulting
    per-step flags. With single-letter targets ``prefix_0`` is always ``0``, so the new-min penalty is
    inert and only the new-max bonus fires; it starts mattering for multi-letter targets.

    Args:
        env: The environment instance.
        command_name: Name of the :class:`LetterTypingCommand` term to read progress from.
    """
    command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
    return command.new_high.float() - command.new_low.float()


def typing_success(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    r"""Sparse success bonus: ``1`` when the whole target word is typed correctly.

    Fires when the :class:`LetterTypingCommand` edit distance is zero, i.e. the typed buffer exactly
    matches the target with no missing and no extra/wrong keys. It is returned on every step the env
    stays in that completed state (until the command resamples a new word).

    Args:
        env: The environment instance.
        command_name: Name of the :class:`LetterTypingCommand` term to read completion from.
    """
    command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
    return ((command.distance == 0) & (command.target_len > 0)).float()


class reach_key(ManagerTermBase):
    """Reward the closest selected jaw tip to the key, including a downward press offset [m]."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._offsets = torch.tensor(cfg.params["tip_offsets"], device=env.device).unsqueeze(0)
        self._press_offset = torch.tensor((0.0, 0.0, cfg.params.get("press_depth", 0.0)), device=env.device)

    def __call__(
        self, env, bodies, tip_offsets, command_name: str = "typing", std: float = 0.2, press_depth: float = 0.0
    ):
        from isaaclab_newton.physics import NewtonManager

        command = env.command_manager.get_term(command_name)
        pose = bodies.dense(NewtonManager.get_state().body_q)
        tips = pose[..., :3] + quat_apply(pose[..., 3:], self._offsets.expand(pose.shape[0], -1, -1))
        target = command.target_key_pos_w() - self._press_offset
        distance = torch.linalg.vector_norm(tips - target[:, None], dim=-1)
        distance = torch.where(bodies.dense_active(), distance, float("inf")).min(dim=1).values
        return 1.0 - torch.tanh(distance / std)
