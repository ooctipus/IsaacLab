# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor

# viz for debug, remove when done debugging
# from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
# frame_marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
# frame_marker_cfg.markers["frame"].scale = (0.025, 0.025, 0.025)
# pose_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/debug_transform"))


def reach_reward(env: ManagerBasedRLEnv, held_asset_cfg: SceneEntityCfg, ee_cfg: SceneEntityCfg, std: float = 0.1):
    robot: Articulation = env.scene[ee_cfg.name]
    held_asset: RigidObject = env.scene[held_asset_cfg.name]
    ee_pos_w = robot.data.body_pos_w[:, ee_cfg.body_ids].view(env.num_envs, -1)
    nut_pos_w = held_asset.data.root_pos_w
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


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(-5000, 5000)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-5000, 5000)


def gripper_asymetric_contact_penalty(env: ManagerBasedRLEnv, threshold: float = 1.0) -> torch.Tensor:
    left_finger_contact_sensor: ContactSensor = env.scene.sensors["panda_leftfinger_object_s"]
    right_finger_contact_sensor: ContactSensor = env.scene.sensors["panda_rightfinger_object_s"]

    # check if contact force is above threshold
    left_finger_contact = left_finger_contact_sensor.data.net_forces_w.view(env.num_envs, 3)
    right_finger_contact = right_finger_contact_sensor.data.net_forces_w.view(env.num_envs, 3)

    left_finger_in_contact = torch.norm(left_finger_contact, dim=-1) > threshold
    right_finger_in_contact = torch.norm(right_finger_contact, dim=-1) > threshold
    return (left_finger_in_contact != right_finger_in_contact).float()


def gripper_firm_contact(env: ManagerBasedRLEnv, threshold: float = 1.0) -> torch.Tensor:
    left_finger_contact_sensor: ContactSensor = env.scene.sensors["panda_leftfinger_object_s"]
    right_finger_contact_sensor: ContactSensor = env.scene.sensors["panda_rightfinger_object_s"]

    # check if contact force is above threshold
    left_finger_contact = left_finger_contact_sensor.data.net_forces_w.view(env.num_envs, 3)
    right_finger_contact = right_finger_contact_sensor.data.net_forces_w.view(env.num_envs, 3)

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
        # Return the unweighted reward for the termination terms
        asset_vel_w = self.held_asset.data.root_state_w[:, 7:13]
        asset_pose = self.held_asset.data.root_pose_w
        ee_pose = self.robot.data.body_link_pose_w[:, ee_cfg.body_ids, :7].view(-1, 7)

        pos_e, rot_e = math_utils.compute_pose_error(asset_pose[:, :3], asset_pose[:, 3:], ee_pose[:, :3], ee_pose[:, 3:])
        # scale 0.25 to rotation part because radian error can be a lot more amplified compared to position....
        current_relative_pose = torch.cat((pos_e, math_utils.wrap_to_pi(rot_e) * 0.25), dim=1)
        delta_pose = current_relative_pose - self.last_relative_position
        stable_manip = (delta_pose.abs().sum(dim=1) < 0.1) | (asset_vel_w.abs().sum(dim=1) < 0.1)
        self.last_relative_position[:] = current_relative_pose
        asset_initialized = (env.episode_length_buf > 10)
        return ((~stable_manip) & asset_initialized).float()
