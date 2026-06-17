# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

    from ...mdp.commands.state_command.state_command import StateCommand


def command_success(env: ManagerBasedRLEnv):
    command_term: StateCommand = env.command_manager.get_term("goal_point")
    return command_term.get_task_reward()


class reward_compose(ManagerTermBase):
    """Compose sparse terminal success with episode-accumulated quality costs."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
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
        log[f"Episode_Reward/reward_composer/{name}"] = (
            torch.mean(value[env_ids]) * self._env.step_dt / self._env.max_episode_length_s
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        success: RewardTermCfg,
        quality: dict[str, RewardTermCfg],
    ) -> torch.Tensor:
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

        torch.mul(self._quality_sum, env.step_dt, out=self._quality_cost).neg_()
        self._quality_multiplier.copy_(self._quality_cost).div_(float(success.weight)).tanh_().neg_().add_(1.0)
        torch.mul(self._success_reward, self._quality_multiplier, out=self._composed_reward)
        self._composed_reward.mul_(self._success_mask)
        return self._composed_reward


def exploration_reward(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), forward_only: bool = False
):
    # Retrieve the robot and target data
    robot: Articulation = env.scene[robot_cfg.name]
    base_velocity = robot.data.root_lin_vel_b  # Robot's current base velocity vector
    target_position = env.command_manager.get_command("goal_point")[:, :3]  # Target position relative to robot base

    # Base directional alignment (cosine similarity)
    cos_align = F.cosine_similarity(base_velocity[:, :3], target_position, dim=-1, eps=1e-6)

    if not forward_only:
        return cos_align

    # Forward preference weight: positive forward component relative to speed
    speed = torch.linalg.vector_norm(base_velocity, ord=2, dim=-1)
    forward_comp = base_velocity[:, 0].clamp_min(0)
    forward_weight = forward_comp / (speed + 1e-6)

    return cos_align * forward_weight
