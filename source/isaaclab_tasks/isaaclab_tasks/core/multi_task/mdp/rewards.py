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

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
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


def reach_reward(env: ManagerBasedRLEnv, held_asset_cfg: SceneEntityCfg, ee_cfg: SceneEntityCfg, std: float = 0.1):
    robot: Articulation = env.scene[ee_cfg.name]
    held_asset: RigidObject = env.scene[held_asset_cfg.name]
    ee_pos_w = wp.to_torch(robot.data.body_pos_w)[:, ee_cfg.body_ids].view(env.num_envs, -1)
    nut_pos_w = wp.to_torch(held_asset.data.root_pos_w)
    dist = torch.norm((nut_pos_w - ee_pos_w), dim=1)
    return 1 - torch.tanh(dist / std)


def orientation_reward(env: ManagerBasedRLEnv, std: float, context: str = "progress_context") -> torch.Tensor:
    context_term: ManagerTermBase = env.termination_manager.get_term_cfg(context).func  # type: ignore
    euler_xy_diff: torch.Tensor = getattr(context_term, "euler_xy_diff")
    return 1 - torch.tanh(euler_xy_diff / std)


def concentric_reward(env: ManagerBasedRLEnv, std: float, context: str = "progress_context") -> torch.Tensor:
    context_term: ManagerTermBase = env.termination_manager.get_term_cfg(context).func  # type: ignore
    xy_distance: torch.Tensor = getattr(context_term, "xy_distance")
    orientation_aligned: torch.Tensor = getattr(context_term, "orientation_aligned")
    return torch.where(orientation_aligned, 1 - torch.tanh(xy_distance / std), 0.0)


def progress_reward(env: ManagerBasedRLEnv, std: float, context: str = "progress_context") -> torch.Tensor:
    context_term: ManagerTermBase = env.termination_manager.get_term_cfg(context).func  # type: ignore
    orientation_aligned: torch.Tensor = getattr(context_term, "orientation_aligned")
    position_centered: torch.Tensor = getattr(context_term, "position_centered")
    z_distance: torch.Tensor = getattr(context_term, "z_distance")
    return torch.where(orientation_aligned & position_centered, 1 - torch.tanh(z_distance / std), 0.0)


def success_reward(env: ManagerBasedRLEnv, context: str = "progress_context") -> torch.Tensor:
    context_term: ManagerTermBase = env.termination_manager.get_term_cfg(context).func  # type: ignore
    orientation_aligned: torch.Tensor = getattr(context_term, "orientation_aligned")
    position_centered: torch.Tensor = getattr(context_term, "position_centered")
    z_distance_reached: torch.Tensor = getattr(context_term, "z_distance_reached")
    return torch.where(orientation_aligned & position_centered & z_distance_reached, 1.0, 0.0)


def gripper_asymetric_contact_penalty(env: ManagerBasedRLEnv, threshold: float = 1.0) -> torch.Tensor:
    left_finger_contact_sensor: ContactSensor = env.scene.sensors["panda_leftfinger_object_s"]
    right_finger_contact_sensor: ContactSensor = env.scene.sensors["panda_rightfinger_object_s"]

    left_finger_contact = wp.to_torch(left_finger_contact_sensor.data.net_forces_w).view(env.num_envs, 3)
    right_finger_contact = wp.to_torch(right_finger_contact_sensor.data.net_forces_w).view(env.num_envs, 3)

    left_finger_in_contact = torch.norm(left_finger_contact, dim=-1) > threshold
    right_finger_in_contact = torch.norm(right_finger_contact, dim=-1) > threshold
    return (left_finger_in_contact != right_finger_in_contact).float()


def gripper_firm_contact(env: ManagerBasedRLEnv, threshold: float = 1.0) -> torch.Tensor:
    left_finger_contact_sensor: ContactSensor = env.scene.sensors["panda_leftfinger_object_s"]
    right_finger_contact_sensor: ContactSensor = env.scene.sensors["panda_rightfinger_object_s"]

    left_finger_contact = wp.to_torch(left_finger_contact_sensor.data.net_forces_w).view(env.num_envs, 3)
    right_finger_contact = wp.to_torch(right_finger_contact_sensor.data.net_forces_w).view(env.num_envs, 3)

    left_finger_in_contact = torch.norm(left_finger_contact, dim=-1) > threshold
    right_finger_in_contact = torch.norm(right_finger_contact, dim=-1) > threshold

    return (left_finger_in_contact & right_finger_in_contact).float()


class unstable_manipulation(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        self.held_asset: RigidObject = env.scene[cfg.params.get("held_asset_cfg").name]
        self.robot: Articulation = env.scene[cfg.params.get("ee_cfg").name]
        self.last_relative_position = torch.zeros((env.num_envs, 6), device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, ee_cfg: SceneEntityCfg, held_asset_cfg: SceneEntityCfg) -> torch.Tensor:
        asset_vel_w = wp.to_torch(self.held_asset.data.root_vel_w)
        asset_pos_w = wp.to_torch(self.held_asset.data.root_pos_w)
        asset_quat_w = wp.to_torch(self.held_asset.data.root_quat_w)
        ee_pos_w = wp.to_torch(self.robot.data.body_link_pos_w)[:, ee_cfg.body_ids].view(-1, 3)
        ee_quat_w = wp.to_torch(self.robot.data.body_link_quat_w)[:, ee_cfg.body_ids].view(-1, 4)

        pos_e, rot_e = math_utils.compute_pose_error(asset_pos_w, asset_quat_w, ee_pos_w, ee_quat_w)
        current_relative_pose = torch.cat((pos_e, math_utils.wrap_to_pi(rot_e) * 0.25), dim=1)
        delta_pose = current_relative_pose - self.last_relative_position
        stable_manip = (delta_pose.abs().sum(dim=1) < 0.1) | (asset_vel_w.abs().sum(dim=1) < 0.1)
        self.last_relative_position[:] = current_relative_pose
        asset_initialized = env.episode_length_buf > 10
        return ((~stable_manip) & asset_initialized).float()
