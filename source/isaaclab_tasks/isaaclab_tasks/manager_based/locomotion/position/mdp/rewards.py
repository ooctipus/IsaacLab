# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Reference: [Advanced Skills by Learning Locomotion and Local Navigation End-to-End, Nikita Rudin(s),
#             https://arxiv.org/pdf/2209.12827]

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp
from torch.nn import functional as F

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg
    from isaaclab.sensors import ContactSensor

    from .commands import RelativeStateCommand


def task_reward(env: ManagerBasedRLEnv, std: float = 0.5):
    distance_to_goal = env.command_manager.get_command("goal_point")[:, :3].norm(2, -1)
    return 1 - torch.tanh(distance_to_goal / std)


def heading_tracking(env: ManagerBasedRLEnv, std: float = 0.5):
    distance_to_goal = env.command_manager.get_command("goal_point")[:, :3].norm(2, -1)
    desired_heading = env.command_manager.get_command("goal_point")[:, 3].abs()
    return (1 - torch.tanh(desired_heading / std)) * (distance_to_goal < 0.4).float()


def exploration_reward(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), forward_only: bool = False
):
    robot: Articulation = env.scene[robot_cfg.name]
    base_velocity = wp.to_torch(robot.data.root_lin_vel_b)
    target_position = env.command_manager.get_command("goal_point")[:, :3]

    cos_align = F.cosine_similarity(base_velocity[:, :3], target_position, dim=-1, eps=1e-6)

    if not forward_only:
        return cos_align

    speed = torch.linalg.vector_norm(base_velocity, ord=2, dim=-1)
    forward_comp = base_velocity[:, 0].clamp_min(0)
    forward_weight = forward_comp / (speed + 1e-6)

    return cos_align * forward_weight


def forward_direction_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[robot_cfg.name]
    base_velocity = wp.to_torch(robot.data.root_lin_vel_b)  # [N, 3], in body frame

    speed = torch.linalg.vector_norm(base_velocity, dim=-1)  # ||v||
    cos_forward = base_velocity[:, 0] / (speed + 1e-6)  # alignment with +x

    # Only reward motion that has a forward component; backward/sideways → 0
    return cos_forward.clamp(min=0.0)


def mechanical_power(env: ManagerBasedRLEnv, robot_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    work = torch.sum((wp.to_torch(robot.data.applied_torque) * wp.to_torch(robot.data.joint_vel)).abs(), dim=1)
    work = torch.where(torch.isfinite(work), work, torch.zeros_like(work))
    return work


def command_success(env: ManagerBasedRLEnv):
    command_term: RelativeStateCommand = env.command_manager.get_term("goal_point")
    return command_term.get_task_reward()


class reward_compose(ManagerTermBase):
    """Compose sparse terminal success with episode-accumulated quality costs.

    The nested ``success`` term is evaluated only as a terminal gate and sets the
    maximum terminal reward through its ``weight``. Nested ``quality`` terms are
    accumulated every step with their own ``weight`` and ``env.step_dt``, matching
    the contribution they would have made as ordinary reward terms. At terminal
    steps the accumulated quality cost is mapped to a multiplier in ``[0, 1]``.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize accumulated quality state.

        Args:
            cfg: Reward composer configuration.
            env: Manager-based environment.
        """
        super().__init__(cfg, env)
        self._success_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        self._quality_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        self._quality_term_sums = {
            name: torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
            for name in cfg.params.get("quality", {})
        }
        self._step_quality = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        self._success_reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        self._success_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._quality_cost = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        self._quality_multiplier = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        self._composed_reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Log and clear accumulated composer state for reset environments.

        Args:
            env_ids: Environment ids to reset. If ``None``, all environments are reset.
        """
        if env_ids is None:
            env_ids = slice(None)

        log = self._env.extras.setdefault("log", {})
        self._log_episode_reward(log, "success", self._success_sum, env_ids)
        for name, value in self._quality_term_sums.items():
            self._log_episode_reward(log, f"quality/{name}", value, env_ids)

        self._success_sum[env_ids] = 0.0
        self._quality_sum[env_ids] = 0.0
        for value in self._quality_term_sums.values():
            value[env_ids] = 0.0

    def _log_episode_reward(
        self, log: dict[str, torch.Tensor], name: str, value: torch.Tensor, env_ids: Sequence[int] | slice
    ) -> None:
        """Log a composer subterm with RewardManager's episode-reward convention."""
        log[f"Episode_Reward/reward_composer/{name}"] = (
            torch.mean(value[env_ids]) * self._env.step_dt / self._env.max_episode_length_s
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        success: RewardTermCfg,
        quality: dict[str, RewardTermCfg],
    ) -> torch.Tensor:
        """Compute sparse success reward with a fixed tanh quality discount.

        Args:
            env: Manager-based environment.
            success: Sparse success term. Its ``weight`` is the maximum success reward.
            quality: Per-step quality terms to accumulate over the episode.

        Returns:
            ``success_reward * (1 - tanh(cost / success.weight))`` when success
            is non-zero, otherwise zero.
        """
        self._step_quality.zero_()
        for name, term_cfg in quality.items():
            contribution = term_cfg.func(env, **term_cfg.params)
            self._step_quality.add_(contribution, alpha=term_cfg.weight)
            self._quality_term_sums[name].add_(contribution, alpha=term_cfg.weight)
        self._quality_sum += self._step_quality

        torch.mul(success.func(env, **success.params), success.weight, out=self._success_reward)
        torch.gt(self._success_reward, 0.0, out=self._success_mask)

        torch.mul(self._success_reward, self._success_mask, out=self._composed_reward)
        self._success_sum += self._composed_reward

        self._quality_cost.copy_(self._quality_sum).mul_(env.step_dt).neg_()
        self._quality_multiplier.copy_(self._quality_cost).div_(float(success.weight)).tanh_().neg_().add_(1.0)
        torch.mul(self._success_reward, self._quality_multiplier, out=self._composed_reward)
        self._composed_reward.mul_(self._success_mask)
        return self._composed_reward


def position_tracking(env: ManagerBasedRLEnv, std: float):
    command_term: RelativeStateCommand = env.command_manager.get_term("goal_point")
    position_error = command_term.get_state_error()[:, 0]
    return 1 - torch.tanh(position_error / std)


def rotation_tracking(env: ManagerBasedRLEnv, std: float):
    command_term: RelativeStateCommand = env.command_manager.get_term("goal_point")
    rotation_error = command_term.get_state_error()[:, 1]
    return 1 - torch.tanh(rotation_error / std)


def lin_vel_tracking(env: ManagerBasedRLEnv, std: float):
    command_term: RelativeStateCommand = env.command_manager.get_term("goal_point")
    lin_vel_error = command_term.get_state_error()[:, 2]
    return 1 - torch.tanh(lin_vel_error / std)


def ang_vel_tracking(env: ManagerBasedRLEnv, std: float):
    command_term: RelativeStateCommand = env.command_manager.get_term("goal_point")
    ang_vel_error = command_term.get_state_error()[:, 3]
    return 1 - torch.tanh(ang_vel_error / std)


def speeding(env: ManagerBasedRLEnv, robot_cfg=SceneEntityCfg("robot"), speed_limit=1.5) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    speeding = torch.norm(wp.to_torch(robot.data.root_vel_w), dim=-1) > speed_limit
    return speeding.float()


def incoming_wrench(env: ManagerBasedRLEnv, robot_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    incoming_wrench = torch.norm(wp.to_torch(robot.data.body_incoming_joint_wrench_b), dim=-1)  # (B, num_bodies)
    return incoming_wrench.sum(dim=1)  # (B,)


def stall_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_vel_threshold: float = 0.1,
    distance_threshold: float = 0.5,
):
    robot: Articulation = env.scene[robot_cfg.name]
    base_vel = wp.to_torch(robot.data.root_lin_vel_b).norm(2, dim=-1)
    distance_to_goal = env.command_manager.get_command("goal_point")[:, :2].norm(2, dim=-1)
    return (base_vel < base_vel_threshold) & (distance_to_goal > distance_threshold)


def illegal_contact_penalty(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # type: ignore
    net_contact_forces = wp.to_torch(contact_sensor.data.net_forces_w_history)
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold,
        dim=1,  # type: ignore
    ).float()


def feet_lin_acc_l2(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[robot_cfg.name]
    feet_acc = torch.sum(torch.square(wp.to_torch(robot.data.body_lin_acc_w)[..., robot_cfg.body_ids, :]), dim=(1, 2))
    return feet_acc


def feet_rot_acc_l2(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[robot_cfg.name]
    feet_acc = torch.sum(torch.square(wp.to_torch(robot.data.body_ang_acc_w)[..., robot_cfg.body_ids, :]), dim=(1, 2))
    return feet_acc


def stand_penalty(
    env: ManagerBasedRLEnv,
    height_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    base_height = wp.to_torch(robot.data.root_link_pos_w)[:, 2]  # z-coordinate of the base
    penalty = (base_height < height_threshold).float() * -1.0
    return penalty


class foot_touchdown_impact(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.history_length: int = cfg.params.get("history_length", 3)

        # Use the sensor_cfg to decide how many feet we care about
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        num_feet = len(sensor_cfg.body_ids)

        self.foot_speed_history = torch.zeros((env.num_envs, num_feet, self.history_length), device=env.device)
        self._hist_idx: int = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        history_length: int,  # not actually needed, but harmless if passed from cfg
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

        # Current foot speeds: [N, B_feet]
        foot_vel = wp.to_torch(asset.data.body_com_lin_vel_w)[:, asset_cfg.body_ids, :]
        foot_speed = torch.linalg.vector_norm(foot_vel, dim=-1)

        # Ring buffer over the last `history_length` steps
        idx = self._hist_idx % self.history_length
        self.foot_speed_history[:, :, idx] = foot_speed
        self._hist_idx += 1

        # Touchdown detection via contact time
        contact_time = wp.to_torch(contact_sensor.data.current_contact_time)[:, sensor_cfg.body_ids]
        is_touchdown = (contact_time > 0.0) & (contact_time <= env.step_dt)

        # Max speed over history for each foot, store as per-foot impact at touchdown, zero otherwise
        max_hist_speed, _ = self.foot_speed_history.max(dim=-1)
        per_foot_impact = torch.where(is_touchdown, max_hist_speed, torch.zeros_like(max_hist_speed))

        # Sum over feet → per-env impact scalar
        impact = per_foot_impact.sum(dim=-1)  # [N]

        return impact


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in
    :attr:`synced_feet_pair_names` to bias the policy towards a desired gait, i.e trotting, bounding, or pacing.
    Note that this reward is only for quadrupedal gaits with two pairs of synchronized feet.
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
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
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
        body_vel = torch.linalg.norm(wp.to_torch(self.asset.data.root_com_lin_vel_b)[:, :2], dim=1)
        return torch.where(
            torch.logical_or(distance > 0.4, body_vel > self.velocity_threshold), sync_reward * async_reward, 0.0
        )

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = wp.to_torch(self.contact_sensor.data.current_air_time)
        contact_time = wp.to_torch(self.contact_sensor.data.current_contact_time)
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = wp.to_torch(self.contact_sensor.data.current_air_time)
        contact_time = wp.to_torch(self.contact_sensor.data.current_contact_time)
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def forward_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    root_lin_vel_b = wp.to_torch(asset.data.root_lin_vel_b)
    forward_velocity = root_lin_vel_b[:, 0]
    distance = torch.norm(env.command_manager.get_command("goal_point")[:, :2], dim=1)
    return torch.where(distance > 0.4, torch.tanh(forward_velocity.clamp(-1, 1) / std), 0)


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
    current_air_time = wp.to_torch(contact_sensor.data.current_air_time)[:, sensor_cfg.body_ids]
    current_contact_time = wp.to_torch(contact_sensor.data.current_contact_time)[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    distance = torch.norm(env.command_manager.get_command("goal_point")[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
    body_vel = (
        torch.linalg.norm(wp.to_torch(asset.data.root_com_lin_vel_b)[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
    )
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
    last_air_time = wp.to_torch(contact_sensor.data.last_air_time)[:, sensor_cfg.body_ids]
    last_contact_time = wp.to_torch(contact_sensor.data.last_contact_time)[:, sensor_cfg.body_ids]
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
    net_contact_forces = wp.to_torch(contact_sensor.data.net_forces_w_history)
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(
        wp.to_torch(asset.data.body_com_lin_vel_w)[:, asset_cfg.body_ids, :2], dim=2
    )

    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    distance = torch.norm(env.command_manager.get_command("goal_point")[:, :2], dim=1)
    body_vel = torch.linalg.norm(wp.to_torch(asset.data.root_lin_vel_b)[:, :2], dim=1)
    reward = torch.linalg.norm((wp.to_torch(asset.data.joint_pos) - wp.to_torch(asset.data.default_joint_pos)), dim=1)
    return torch.where((distance > 0.4) | (body_vel > velocity_threshold), reward, stand_still_scale * reward)
