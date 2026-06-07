# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Reference: [Advanced Skills by Learning Locomotion and Local Navigation End-to-End, Nikita Rudin(s),
#             https://arxiv.org/pdf/2209.12827]

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

    from .commands import RelativeStateCommand


def command_success(env: ManagerBasedRLEnv):
    command_term = cast("RelativeStateCommand", env.command_manager.get_term("goal_point"))
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
        env_id_selection: Sequence[int] | slice = slice(None) if env_ids is None else env_ids

        log = self._env.extras.setdefault("log", {})
        self._log_episode_reward(log, "success", self._success_sum, env_id_selection)
        for name, value in self._quality_term_sums.items():
            self._log_episode_reward(log, f"quality/{name}", value, env_id_selection)

        self._success_sum[env_id_selection] = 0.0
        self._quality_sum[env_id_selection] = 0.0
        for value in self._quality_term_sums.values():
            value[env_id_selection] = 0.0

    def _log_episode_reward(
        self, log: dict[str, torch.Tensor], name: str, value: torch.Tensor, env_ids: Sequence[int] | slice
    ) -> None:
        """Log a composer subterm with RewardManager's episode-reward convention."""
        log[f"Episode_Reward/reward_composer/{name}"] = (
            torch.mean(value[env_ids]) * self._env.step_dt / getattr(self._env, "max_episode_length_s")
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
