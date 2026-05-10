# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from tensordict import TensorDict

from isaaclab.utils.string import string_to_callable


class CommanderActorModel(MLPModel):
    """Actor model with a learned command feature."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        commander_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        commander_activation: str = "elu",
        commander_obs_normalization: bool = False,
        kinematic_reward_weight: float = 0.1,
        commander_loss_coef: float = 0.1,
        get_command_target_fn: Callable | str | None = None,
        log_error_fn: Callable | str | None = None,
        **kwargs,
    ) -> None:
        if get_command_target_fn is None:
            raise ValueError("get_command_target_fn must be provided for CommanderActorModel.")
        if isinstance(get_command_target_fn, str):
            target_fn = string_to_callable(get_command_target_fn)

            def get_command_target_fn():
                return target_fn()

        if isinstance(log_error_fn, str):
            log_error_fn = string_to_callable(log_error_fn)
        self._get_command_target_fn = get_command_target_fn
        self._log_error_fn = log_error_fn
        self.commander_loss_coef = commander_loss_coef
        self.kinematic_reward_weight = kinematic_reward_weight

        original_actor_groups = list(obs_groups[obs_set])
        self._commander_obs_groups = [g for g in original_actor_groups if g != "cmd_feat"]
        num_commander_obs = sum(obs[g].shape[-1] for g in self._commander_obs_groups)
        obs["cmd_feat"] = get_command_target_fn()
        num_commands = obs["cmd_feat"].shape[-1]
        obs_groups[obs_set] = [g for g in original_actor_groups if g != "task"] + ["cmd_feat"]
        if "critic" in obs_groups and "cmd_feat" not in obs_groups["critic"]:
            obs_groups["critic"].append("cmd_feat")
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        self.commander = MLP(num_commander_obs, num_commands, list(commander_hidden_dims), commander_activation)
        self._commander_obs_normalization = commander_obs_normalization
        if commander_obs_normalization:
            self.commander_obs_normalizer = EmpiricalNormalization(num_commander_obs)
            self.command_target_normalizer: EmpiricalNormalization | None = EmpiricalNormalization(num_commands)
        else:
            self.commander_obs_normalizer = nn.Identity()
            self.command_target_normalizer = None
        self.cmd_feat_cache: torch.Tensor | None = None
        self.cmd_target_cache: torch.Tensor | None = None

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        commander_obs = self.commander_obs_normalizer(torch.cat([obs[g] for g in self._commander_obs_groups], dim=-1))
        self.cmd_feat_cache = self.commander(commander_obs)
        obs["cmd_feat"] = self.cmd_feat_cache
        return super().get_latent(obs, masks, hidden_state)

    def update_normalization(self, obs: TensorDict) -> None:
        if self._commander_obs_normalization:
            commander_obs = torch.cat([obs[g] for g in self._commander_obs_groups], dim=-1)
            self.commander_obs_normalizer.update(commander_obs)  # type: ignore[union-attr]
            self.command_target_normalizer.update(self._get_command_target_fn())  # type: ignore[union-attr]
            obs["cmd_feat"] = self.commander(self.commander_obs_normalizer(commander_obs))
        super().update_normalization(obs)

    def get_kinematic_reward(self) -> torch.Tensor:
        self.cmd_target_cache = self._get_command_target_fn()
        target = (
            self.command_target_normalizer(self.cmd_target_cache)
            if self.command_target_normalizer
            else self.cmd_target_cache
        )
        err = torch.linalg.vector_norm(self.cmd_feat_cache - target, dim=-1)
        if self._log_error_fn is not None:
            proposed = (
                self.command_target_normalizer.inverse(self.cmd_feat_cache)
                if self.command_target_normalizer
                else self.cmd_feat_cache
            )
            self._log_error_fn(proposed, self.cmd_target_cache)
        return -err * self.kinematic_reward_weight
