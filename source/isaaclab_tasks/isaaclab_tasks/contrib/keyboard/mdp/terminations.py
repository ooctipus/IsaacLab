# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the SO101 keyboard letter-typing task."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import warp as wp
from newton import Model

from isaaclab.managers import ManagerTermBase

from ..mujoco_selection import MuJoCoSelection
from ..newton_selection import NewtonSelection, scalar_field_active, scalar_field_read

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
    logical_worlds: wp.array[wp.int64],
    velocity: wp.array[float],
    limits: wp.array[float],
    out: wp.array[int],
):
    i = wp.tid()
    if i < starts[logical_worlds.shape[0]]:
        dof = ids[i]
        if wp.abs(velocity[dof]) > limits[dof]:
            wp.atomic_max(out, logical_worlds[worlds[i]], 1)


@wp.kernel
def _selected_velocity_limit_violation(
    velocity: Any,
    limits: Any,
    out: wp.array[int],
):
    world, slot = wp.tid()
    if scalar_field_active(velocity, world, slot):
        if wp.abs(scalar_field_read(velocity, world, slot)) > scalar_field_read(limits, world, slot):
            wp.atomic_max(out, world, 1)


class joint_vel_out_of_limit(ManagerTermBase):
    """Reduce selected DOFs into race-safe per-world velocity violations."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.params["joints"].index_domain != Model.AttributeFrequency.JOINT_DOF:
            raise ValueError("Velocity limits require a JOINT_DOF selector.")
        self._out = wp.zeros(env.num_envs, dtype=wp.int32, device=env.device)

    def __call__(self, env, joints: NewtonSelection) -> torch.Tensor:
        self._out.zero_()
        if isinstance(joints, MuJoCoSelection):
            wp.launch(
                _selected_velocity_limit_violation,
                dim=(env.num_envs, joints.width),
                inputs=[joints.scalar_field("state", "joint_qd"), joints.scalar_field("model", "joint_velocity_limit")],
                outputs=[self._out],
                device=self._out.device,
            )
        else:
            for part, worlds in joints.native_bindings:
                model, state = part.owner.model, part.owner.state
                wp.launch(
                    _velocity_limit_violation,
                    dim=part.capacity,
                    inputs=[
                        part.freq_ids,
                        part.env_ids,
                        part.world_start,
                        wp.from_torch(worlds),
                        state.joint_qd,
                        model.joint_velocity_limit,
                    ],
                    outputs=[self._out],
                    device=model.device,
                )
        return wp.to_torch(self._out).bool()


class illegal_contact(ManagerTermBase):
    """Threshold net normal force [N] on participating bodies after the physics frame.

    ``sensor_name=None`` consumes model-scoped sensor/contact pairs prepared in
    ``env.native_contacts`` by the composition root. This matches the task's
    one-frame contact history: intermediate substep maxima are not used.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        bodies = cfg.params["bodies"]
        sensor_name = cfg.params["sensor_name"]
        self._sensor = env.scene.sensors[sensor_name] if sensor_name is not None else None
        if self._sensor is None:
            if bodies.index_domain != Model.AttributeFrequency.BODY:
                raise ValueError("Keyboard contact termination requires selected bodies.")
            return
        native = self._sensor.contact_view
        if native.sensing_type != "body":
            raise ValueError("Keyboard contact termination requires a body contact sensor.")
        row_for_body = {body: row for row, body in enumerate(native.sensing_indices)}
        self._rows = torch.tensor([row_for_body[int(body)] for body in bodies.ids.numpy()], device=env.device).reshape(
            bodies.dense_ids().shape
        )

    def __call__(self, env, bodies, sensor_name: str | None, threshold: float):
        if isinstance(bodies, MuJoCoSelection):
            return (bodies.selected_net_normal_forces().norm(dim=-1) > threshold).any(dim=1)
        if self._sensor is not None:
            forces = self._sensor.data.net_normal_forces_w_history.torch
            magnitude = forces.norm(dim=-1).amax(dim=1).flatten()[self._rows]
            return ((magnitude > threshold) & bodies.dense_active()).any(dim=1)

        result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for part, worlds in bodies.native_bindings:
            owner = part.owner
            if owner.solver is None or owner.solver.model is not owner.model:
                raise ValueError("Native contact termination requires the selection's bound solver.")
            sensor, contacts = env.native_contacts[owner.model]
            owner.solver.update_contacts(contacts)
            # Poses and counterpart positions are unused; force accumulation is identical.
            sensor.update(None, contacts)
            normal = wp.to_torch(sensor.total_force) - wp.to_torch(sensor.total_force_friction)
            magnitude = normal.reshape(len(worlds), part.width, 3).norm(dim=-1)
            result[worlds] = ((magnitude > threshold) & part.dense_active()).any(dim=1)
        return result
