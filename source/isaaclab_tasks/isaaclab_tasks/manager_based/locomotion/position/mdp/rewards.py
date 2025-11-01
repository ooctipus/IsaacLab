# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Reference: [Advanced Skills by Learning Locomotion and Local Navigation End-to-End, Nikita Rudin(s),
#             https://arxiv.org/pdf/2209.12827]

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def task_reward(env: ManagerBasedRLEnv, reward_window: float = 1.0):  # Represents Tr, the length of the reward window
    #
    # See section II.B (page 3) Exploration Reward for details.
    # Calculate the time step at which the reward window starts
    reward_start_step = env.max_episode_length * (1 - reward_window / env.max_episode_length_s)

    # Calculate the distance to the goal (‖xb − x∗b‖^2), squared L2 norm of the difference
    distance_to_goal = env.command_manager.get_command("goal_point")[:, :3].norm(2, -1).pow(2)

    # Calculate task reward as per the equation:
    # If within the reward window, r_task is non-zero
    task_reward = (1 / (1 + distance_to_goal)) * (env.episode_length_buf > reward_start_step).float()
    residue_task_reward = (1 / reward_window) * task_reward

    # TODO: Try no to change exploration weight here.
    # "The following line removes the exploration reward (r_bias) once the task reward (r_task)
    #  reaches 50% of its maximum value, as described in the paper." [II.B (page 3)]
    if task_reward.mean() > 0.5 and (env.reward_manager.get_term_cfg("exploration").weight > 0.0):
        env.reward_manager.get_term_cfg("exploration").weight = 0.0

    return residue_task_reward


def heading_tracking(env: ManagerBasedRLEnv, distance_threshold: float = 2.0, reward_window: float = 2.0):
    desired_heading = env.command_manager.get_command("goal_point")[:, 3]
    reward_start_step = env.max_episode_length * (1 - reward_window / env.max_episode_length_s)
    current_dist = env.command_manager.get_command("goal_point")[:, :2].norm(2, -1)
    r_heading_tracking = (
        1
        / reward_window
        * (1 / (1 + desired_heading.pow(2)))
        * (current_dist < distance_threshold).float()
        * (env.episode_length_buf > reward_start_step).float()
    )
    return r_heading_tracking


def exploration_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    # Retrieve the robot and target data
    robot: Articulation = env.scene[robot_cfg.name]
    base_velocity = robot.data.root_lin_vel_b  # Robot's current base velocity vector
    target_position = env.command_manager.get_command("goal_point")[:, :3]  # Target position relative to robot base

    # Compute the dot product of the robot's base velocity and target position vectors
    velocity_alignment = (base_velocity[:, :3] * target_position).sum(-1)

    # Calculate the norms (magnitudes) of the velocity and target position vectors
    velocity_magnitude = torch.norm(base_velocity, p=2, dim=-1)
    target_magnitude = torch.norm(target_position, p=2, dim=-1)

    # Calculate the exploration reward by normalizing the dot product (cosine similarity)
    # Small epsilon added in denominator to prevent division by zero
    exploration_reward = velocity_alignment / (velocity_magnitude * target_magnitude + 1e-6)
    return exploration_reward


def stall_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_vel_threshold: float = 0.1,
    distance_threshold: float = 0.5,
):
    robot: Articulation = env.scene[robot_cfg.name]
    base_vel = robot.data.root_lin_vel_b.norm(2, dim=-1)
    distance_to_goal = env.command_manager.get_command("goal_point")[:, :2].norm(2, dim=-1)
    return (base_vel < base_vel_threshold) & (distance_to_goal > distance_threshold)


def illegal_contact_penalty(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # type: ignore
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold,
        dim=1,  # type: ignore
    ).float()


def feet_lin_acc_l2(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[robot_cfg.name]
    feet_acc = torch.sum(torch.square(robot.data.body_lin_acc_w[..., robot_cfg.body_ids, :]), dim=(1, 2))
    return feet_acc


def feet_rot_acc_l2(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[robot_cfg.name]
    feet_acc = torch.sum(torch.square(robot.data.body_ang_acc_w[..., robot_cfg.body_ids, :]), dim=(1, 2))
    return feet_acc


def stand_penalty(
    env: ManagerBasedRLEnv,
    height_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    base_height = robot.data.root_link_pos_w[:, 2]  # z-coordinate of the base
    penalty = (base_height < height_threshold).float() * -1.0
    return penalty


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        max_iterations: int = 500,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        current_iter = int(env.common_step_counter / 48)
        if current_iter > max_iterations:
            return torch.zeros(self.num_envs, device=self.device)
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        distance = torch.norm(env.command_manager.get_command("goal_point")[:, :2], dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        return torch.where(
            torch.logical_or(distance > 0.4, body_vel > self.velocity_threshold), sync_reward * async_reward, 0.0
        )

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def forward_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    max_iter: int = 150,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    root_lin_vel_b = asset.data.root_lin_vel_b
    forward_velocity = root_lin_vel_b[:, 0]
    current_iter = int(env.common_step_counter / 48)
    distance = torch.norm(env.command_manager.get_command("goal_point")[:, :2], dim=1)
    return torch.where(distance > 0.4, torch.tanh(forward_velocity.clamp(-1, 1) / std) * (current_iter < max_iter), 0)


def air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    distance = torch.norm(env.command_manager.get_command("goal_point")[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
    body_vel = torch.linalg.norm(asset.data.root_com_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
    reward = torch.where(
        torch.logical_or(distance > 0.4, body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    return torch.sum(reward, dim=1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def foot_slip_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_com_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    distance = torch.norm(env.command_manager.get_command("goal_point")[:, :2], dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where((distance > 0.4) | (body_vel > velocity_threshold), reward, stand_still_scale * reward)