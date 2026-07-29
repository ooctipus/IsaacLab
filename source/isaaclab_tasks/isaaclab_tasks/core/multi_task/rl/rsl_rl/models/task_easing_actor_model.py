# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, HiddenState
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDict

from .residual_mlp import ResidualMLP


class TaskEasingActorModel(MLPModel):
    """Actor model with learned goal-refinement blocks."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        task_easing_constraint_fn: str = "relu",
        task_easing_loss_coef: float = 0.0,
        task_easing_margin: float = 0.0,
        task_easing_network: str = "mlp",
        num_goal_refinements: int = 2,
        goal_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        goal_activation: str | None = None,
        **kwargs,
    ) -> None:
        if "task" not in obs_groups.get(obs_set, []):
            raise ValueError("TaskEasingActorModel requires 'task' in the actor obs_groups.")
        if "task" not in obs_groups.get("critic", []):
            raise ValueError("TaskEasingActorModel requires 'task' in the critic obs_groups.")
        if task_easing_constraint_fn not in ("relu", "softplus"):
            raise ValueError("task_easing_constraint_fn must be 'relu' or 'softplus'.")
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        self.task_easing_loss_coef = task_easing_loss_coef
        self.task_easing_margin = task_easing_margin
        self.task_easing_constraint_fn = resolve_nn_activation(task_easing_constraint_fn)
        self.num_goal_refinements = num_goal_refinements
        self.num_goals = num_goal_refinements + 1

        goal_dim = obs["task"].shape[-1]
        goal_input_dim = self.obs_dim
        activation = goal_activation if goal_activation is not None else kwargs.get("activation", "elu")
        if task_easing_network == "mlp":
            self.goal_blocks = nn.ModuleList(
                [MLP(goal_input_dim, goal_dim, list(goal_hidden_dims), activation) for _ in range(num_goal_refinements)]
            )
        elif task_easing_network == "residual":
            self.goal_blocks = nn.ModuleList(
                [
                    ResidualMLP(
                        goal_input_dim,
                        goal_dim,
                        goal_hidden_dims[0],
                        1,
                        1,
                        len(goal_hidden_dims),
                        activation,
                        None,
                        True,
                    )
                    for _ in range(num_goal_refinements)
                ]
            )
        else:
            raise ValueError("task_easing_network must be 'mlp' or 'residual'.")
        self._context_groups = [g for g in self.obs_groups if g != "task"]

    def _build_goal_chain(self, obs: TensorDict) -> tuple[list[torch.Tensor], torch.Tensor]:
        goal = obs["task"]
        context = torch.cat([obs[g] for g in self._context_groups], dim=-1)
        goals = [goal]
        for block in self.goal_blocks:
            goal = block(torch.cat([goal, context], dim=-1))
            goals.append(goal)
        return goals, context

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        goals, context = self._build_goal_chain(obs)
        return self.obs_normalizer(torch.cat([goals[-1], context], dim=-1))

    def evaluate_all_goals(self, obs: TensorDict, critic: nn.Module) -> torch.Tensor:
        goals, _ = self._build_goal_chain(obs)
        original_task = obs["task"]
        values = []
        for goal in goals:
            obs["task"] = goal
            values.append(critic(obs))
        obs["task"] = original_task
        return torch.cat(values, dim=-1)

    def monotonic_value_loss(self, values: torch.Tensor) -> torch.Tensor:
        diffs = values[:, 1:] - values[:, :-1]
        return (self.task_easing_loss_coef * self.task_easing_constraint_fn(self.task_easing_margin - diffs)).mean()
