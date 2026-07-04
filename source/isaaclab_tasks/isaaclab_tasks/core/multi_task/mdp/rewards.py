# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain-agnostic reward terms shared across terrain and factory tasks.

- :func:`command_task_reward` — passes the command term's terminal multiplicative
  reward through to the reward manager. Works with any command term that
  exposes a ``task_reward`` attribute (notably :class:`MultiTaskCommand`).
- :func:`action_l2_clamped` / :func:`action_rate_l2_clamped` — generic L2
  action penalties with a saturation clamp.
- :func:`mechanical_power` — Σ |τⱼ · q̇ⱼ| across an articulation's joints,
  NaN-safe. Useful as a soft-safety reward signal for any actuated robot.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.envs.mdp import body_projected_gravity_b
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg
    from isaaclab.sensors import ContactSensor


def command_task_reward(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Expose the command term's terminal multiplicative reward as a reward term.

    For :class:`MultiTaskCommand`, ``task_reward`` is non-zero only on terminal
    steps — bind with ``weight=1.0`` to use it as the sole task reward.
    """
    return env.command_manager.get_term(command_name).task_reward


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(-5000, 5000)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-5000, 5000)


def mechanical_power(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Σ |τⱼ · q̇ⱼ| across the articulation's joints. NaN-safe.

    Total instantaneous absolute mechanical power [W]. NaN/Inf outputs (rare —
    seen briefly during reset on some backends) are clamped to 0.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    work = torch.sum((wp.to_torch(robot.data.applied_torque) * wp.to_torch(robot.data.joint_vel)).abs(), dim=1)
    return torch.where(torch.isfinite(work), work, torch.zeros_like(work))


def joint_position_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize selected joint positions relative to a target.

    Args:
        env: Owning manager-based environment.
        target: Target joint position [rad].
        asset_cfg: Articulation and joints whose positions are measured.

    Returns:
        Sum of squared joint-position errors [rad^2] for each environment.
    """
    position = env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids]
    return (position - target).square().sum(dim=-1)


def joint_position_limits(env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize selected joints outside a centered fraction of hard limits.

    Args:
        env: Owning manager-based environment.
        soft_ratio: Retained fraction of each hard joint-position range.
        asset_cfg: Articulation and joints whose limits are measured.

    Returns:
        Sum of joint-position limit excess [rad] for each environment.
    """
    asset = env.scene[asset_cfg.name]
    limits = asset.data.joint_pos_limits.torch[:, asset_cfg.joint_ids]
    midpoint = (limits[..., 0] + limits[..., 1]) * 0.5
    radius = (limits[..., 1] - limits[..., 0]) * (0.5 * soft_ratio)
    lower = midpoint - radius
    upper = midpoint + radius
    position = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    excess = torch.clamp_max(position - lower, 0.0).neg()
    excess.add_(torch.clamp_min(position - upper, 0.0))
    return excess.sum(dim=-1)


def contact_undesired(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Detect any selected current contact-force component above a threshold.

    Args:
        env: Owning manager-based environment.
        threshold: Absolute contact-force component threshold [N].
        sensor_cfg: Contact sensor and bodies included in the test.

    Returns:
        Binary undesired-contact indicator for each environment.
    """
    force = env.scene.sensors[sensor_cfg.name].data.net_forces_w.torch[:, sensor_cfg.body_ids]
    return torch.any(force.abs() > threshold, dim=(1, 2)).to(force.dtype)


def body_orientation_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize selected-body tilt while vertically loaded.

    The selected sensor bodies and articulation bodies must have the same order.

    Args:
        env: Owning manager-based environment.
        threshold: Vertical contact-force threshold [N].
        sensor_cfg: Contact sensor and bodies used for contact detection.
        asset_cfg: Articulation and corresponding bodies whose tilt is measured.

    Returns:
        Sum of contacted-body projected-gravity XY norms for each environment.
    """
    force_z = env.scene.sensors[sensor_cfg.name].data.net_forces_w.torch[:, sensor_cfg.body_ids, 2]
    gravity = body_projected_gravity_b(env, asset_cfg).view(env.num_envs, -1, 3)
    return (torch.linalg.vector_norm(gravity[..., :2], dim=-1) * (force_z > threshold)).sum(dim=-1)


def body_contact_velocity(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize selected-body center-of-mass speed while in contact.

    The selected sensor bodies and articulation bodies must have the same order.

    Args:
        env: Owning manager-based environment.
        threshold: Contact-force magnitude threshold [N].
        sensor_cfg: Contact sensor and bodies used for contact detection.
        asset_cfg: Articulation and corresponding bodies whose speed is measured.

    Returns:
        Sum of contacted-body three-dimensional speeds [m/s] for each environment.
    """
    force = env.scene.sensors[sensor_cfg.name].data.net_forces_w.torch[:, sensor_cfg.body_ids]
    velocity = env.scene[asset_cfg.name].data.body_com_lin_vel_w.torch[:, asset_cfg.body_ids]
    contact = torch.linalg.vector_norm(force, dim=-1) > threshold
    return (torch.linalg.vector_norm(velocity, dim=-1) * contact).sum(dim=-1)


def body_heading_alignment(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize selected-body heading error relative to the articulation root.

    Args:
        env: Owning manager-based environment.
        asset_cfg: Articulation and bodies whose headings are measured.

    Returns:
        Sum of absolute wrapped heading errors [rad] for each environment.
    """
    asset = env.scene[asset_cfg.name]
    body_quaternion = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    unit_forward = torch.zeros_like(body_quaternion[..., :3])
    unit_forward[..., 0] = 1.0
    body_forward = quat_apply(body_quaternion, unit_forward)

    root_forward = quat_apply(asset.data.root_quat_w.torch, unit_forward[:, 0])
    body_heading = torch.atan2(body_forward[..., 1], body_forward[..., 0])
    root_heading = torch.atan2(root_forward[:, 1], root_forward[:, 0])
    return wrap_to_pi(body_heading - root_heading[:, None]).abs().sum(dim=-1)


class contact_penalty(ManagerTermBase):
    """Penalize contacts on selected sensor bodies."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        contact_sensor_cfg: SceneEntityCfg | None = cfg.params.get("contact_sensor_cfg")
        exclude_contact_sensor_cfg: SceneEntityCfg | None = cfg.params.get("exclude_contact_sensor_cfg")
        if (contact_sensor_cfg is None) == (exclude_contact_sensor_cfg is None):
            raise ValueError("contact_penalty expects exactly one of contact_sensor_cfg or exclude_contact_sensor_cfg.")

        if contact_sensor_cfg is not None:
            self.contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
            self.body_ids = contact_sensor_cfg.body_ids
        else:
            self.contact_sensor: ContactSensor = env.scene.sensors[exclude_contact_sensor_cfg.name]
            if exclude_contact_sensor_cfg.body_ids == slice(None):
                self.body_ids = []
            else:
                exclude_body_ids = set(exclude_contact_sensor_cfg.body_ids)
                self.body_ids = [
                    body_id for body_id in range(self.contact_sensor.num_sensors) if body_id not in exclude_body_ids
                ]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        contact_sensor_cfg: SceneEntityCfg | None = None,
        exclude_contact_sensor_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        net_contact_forces = wp.to_torch(self.contact_sensor.data.net_forces_w_history)
        is_contact = torch.max(torch.linalg.norm(net_contact_forces[:, :, self.body_ids], dim=-1), dim=1)[0] > threshold
        return torch.sum(is_contact, dim=1)


class RewardScaled(ManagerTermBase):
    """Multiply one ordinary reward term by a lazily bound device scale."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        """Retain the reward call and defer binding until all managers exist.

        Args:
            cfg: Reward term configuration.
            env: Owning manager-based environment.
        """
        super().__init__(cfg, env)
        func = cfg.params["func"]
        func_params = cfg.params["func_params"]
        scale_bind = cfg.params["scale_bind"]
        if not callable(func):
            raise TypeError("Scaled reward func must be callable.")
        if not isinstance(func_params, Mapping):
            raise TypeError("Scaled reward func_params must be a mapping.")
        if not isinstance(scale_bind, str) or not scale_bind:
            raise TypeError("Scaled reward scale_bind must be a nonempty expression.")
        self._func: Callable[..., torch.Tensor] = func
        self._func_params = dict(func_params)
        self._scale_bind = scale_bind
        self._scale: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        func: Callable[..., torch.Tensor],
        func_params: Mapping[str, object],
        scale_bind: str,
    ) -> torch.Tensor:
        """Return one reward multiplied by its device-resident scale.

        Args:
            env: Owning manager-based environment.
            func: Stateless reward function retained during construction.
            func_params: Arguments retained for the reward function.
            scale_bind: Expression resolving once to a scalar or environment-major tensor.

        Returns:
            Scaled reward for each environment.
        """
        del func, func_params, scale_bind
        if self._scale is None:
            scale = eval(self._scale_bind, {}, {"env": env})  # noqa: S307
            if not isinstance(scale, torch.Tensor) or scale.device != torch.device(env.device):
                raise TypeError("Scaled reward bindings must resolve to a tensor on the environment device.")
            if scale.shape not in ((), (env.num_envs,)):
                raise ValueError("Scaled reward bindings must be scalar or environment-major.")
            self._scale = scale
        value = self._func(env, **self._func_params)
        if value.shape != (env.num_envs,):
            raise ValueError("Scaled reward functions must return one scalar per environment.")
        return value * self._scale
