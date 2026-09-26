# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the SO101 keyboard letter-typing task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.managers import ManagerTermBase

from ..newton_selection import JOINT_DOF, NewtonSelection

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import LetterTypingCommand


def typing_mistake(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate the episode when a wrong (or extra) key has been typed.

    Fires as soon as the typed buffer extends past the correct prefix (``typed_len > prefix_len``),
    i.e. the agent typed a key that does not continue the target word. Note this makes the backspace
    recovery path unreachable - a single mistyped key ends the episode.

    Args:
        env: The environment instance.
        command_name: Name of the :class:`LetterTypingCommand` term to read typing state from.
    """
    command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
    return command.typed_len > command.prefix_len


def typing_complete(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate the episode once the whole target word has been typed correctly.

    Fires when the :class:`LetterTypingCommand` edit distance reaches zero (the typed buffer exactly
    matches the target), mirroring the :func:`~...mdp.rewards.typing_success` bonus. This is a genuine
    goal-reached terminal state, so it should be registered without ``time_out=True``.

    Args:
        env: The environment instance.
        command_name: Name of the :class:`LetterTypingCommand` term to read typing state from.
    """
    command: LetterTypingCommand = env.command_manager.get_term(command_name)  # type: ignore
    return (command.distance == 0) & (command.target_len > 0)


@wp.kernel
def _velocity_limit_violation(
    ids: wp.array[int],
    worlds: wp.array[int],
    starts: wp.array[int],
    velocity: wp.array[float],
    limits: wp.array[float],
    out: wp.array[int],
):
    i = wp.tid()
    if i < starts[out.shape[0]]:
        dof = ids[i]
        if wp.abs(velocity[dof]) > limits[dof]:
            wp.atomic_max(out, worlds[i], 1)


class joint_vel_out_of_limit(ManagerTermBase):
    """Reduce compact participating DOFs into race-safe per-world velocity violations."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.params["joints"].frequency != JOINT_DOF:
            raise ValueError("Velocity limits require a JOINT_DOF selector.")
        self._out = wp.zeros(env.num_envs, dtype=wp.int32, device=env.device)

    def __call__(self, env, joints: NewtonSelection) -> torch.Tensor:
        self._out.zero_()
        model, state = NewtonManager.get_model(), NewtonManager.get_state()
        wp.launch(
            _velocity_limit_violation,
            dim=joints.capacity,
            inputs=[joints.freq_ids, joints.env_ids, joints.world_start, state.joint_qd, model.joint_velocity_limit],
            outputs=[self._out],
            device=model.device,
        )
        return wp.to_torch(self._out).bool()


class illegal_contact(ManagerTermBase):
    """Reduce existing contact-sensor forces over participating selected bodies [N]."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._sensor = env.scene.sensors[cfg.params["sensor_name"]]
        bodies = cfg.params["bodies"]
        native = self._sensor.contact_view
        if native.sensing_type != "body":
            raise ValueError("Keyboard contact termination requires a body contact sensor.")
        row_for_body = {body: row for row, body in enumerate(native.sensing_indices)}
        self._rows = torch.tensor([row_for_body[int(body)] for body in bodies.ids.numpy()], device=env.device).reshape(
            bodies.dense_ids().shape
        )

    def __call__(self, env, bodies, sensor_name: str, threshold: float):
        forces = self._sensor.data.net_normal_forces_w_history.torch
        magnitude = forces.norm(dim=-1).amax(dim=1).flatten()[self._rows]
        return ((magnitude > threshold) & bodies.dense_active()).any(dim=1)
