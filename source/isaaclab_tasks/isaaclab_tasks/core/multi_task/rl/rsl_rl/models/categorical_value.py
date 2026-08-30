# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import Any, cast

import torch
from rsl_rl.modules import HiddenState
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict

from .residual_mlp_encoder_model import ResidualMLPEncoderModel


class CategoricalValueMixin:
    """Expose categorical value logits while returning their scalar expectation."""

    is_categorical_value = True
    value_support: torch.Tensor

    def _configure_categorical_value(
        self,
        num_bins: int,
        value_min: float,
        value_max: float,
        reward_scaling: bool,
        reward_scale_epsilon: float,
    ) -> None:
        if num_bins < 2:
            raise ValueError(f"num_bins must be at least 2; got {num_bins}.")
        if value_max <= value_min:
            raise ValueError(f"value_max ({value_max}) must exceed value_min ({value_min}).")
        if reward_scaling and not math.isclose(value_max, -value_min):
            raise ValueError("Reward-scaled categorical value support must be symmetric around zero.")
        if reward_scale_epsilon <= 0.0:
            raise ValueError(f"reward_scale_epsilon must be positive; got {reward_scale_epsilon}.")

        self.num_value_bins = num_bins
        self.value_min = value_min
        self.value_max = value_max
        self.reward_scaling = reward_scaling
        self.reward_scale_epsilon = reward_scale_epsilon
        self.reward_scale_max = max(abs(value_min), abs(value_max))
        model = cast(ResidualMLPEncoderModel, self)
        model.register_buffer("value_support", torch.linspace(value_min, value_max, num_bins))

    def get_value_logits(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Return unnormalized categorical value logits."""
        model = cast(ResidualMLPEncoderModel, self)
        if masks is not None and not model.is_recurrent:
            obs = unpad_trajectories(obs, masks)
        latent = model.get_latent(obs, masks, hidden_state)
        return model.mlp(latent)

    def value_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Return expected scalar values from categorical logits."""
        probabilities = torch.softmax(logits, dim=-1)
        return torch.sum(probabilities * self.value_support, dim=-1, keepdim=True)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,  # noqa: ARG002
    ) -> torch.Tensor:
        """Return expected scalar values for the standard RSL-RL critic contract."""
        return self.value_from_logits(self.get_value_logits(obs, masks, hidden_state))


class CategoricalResidualMLPEncoderModel(CategoricalValueMixin, ResidualMLPEncoderModel):
    """SimBa-v1 critic with a categorical value head."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,  # noqa: ARG002
        num_bins: int = 101,
        value_min: float = -5.0,
        value_max: float = 5.0,
        reward_scaling: bool = True,
        reward_scale_epsilon: float = 1.0e-8,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("distribution_cfg") is not None:
            raise ValueError("Categorical value critics do not support an output distribution.")
        super().__init__(obs, obs_groups, obs_set, num_bins, **kwargs)
        self._configure_categorical_value(
            num_bins,
            value_min,
            value_max,
            reward_scaling,
            reward_scale_epsilon,
        )
