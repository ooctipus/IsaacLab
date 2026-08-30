# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch.nn as nn
from rsl_rl.modules.distribution import GaussianDistribution
from tensordict import TensorDict

from .categorical_value import CategoricalValueMixin
from .residual_mlp_encoder_model import ResidualMLPEncoderModel
from .simba_v2 import SimbaV2Head


class SimbaV2EncoderModel(ResidualMLPEncoderModel):
    """Per-group observation encoders followed by a SimBaV2 backbone."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 1,
        expansion: int = 4,
        c_shift: float = 3.0,
        obs_normalization: bool = False,
        encoder_normalization: bool = False,
        head_layer_norm: bool = False,
        distribution_cfg: dict | None = None,
        encoder_cfg: dict[str, dict[str, Any]] | None = None,
        simplicial_group_size: int | None = None,
        simplicial_temperature: float = 1.0,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
        hidden_dims: list[int] | tuple[int, ...] | None = None,
        activation: str = "relu",  # noqa: ARG002
    ) -> None:
        if head_layer_norm:
            raise ValueError("SimBaV2 uses hyperspherical normalization; head_layer_norm must be False.")
        if simplicial_group_size is not None and obs_set != "actor":
            raise ValueError("SEM is actor-only; simplicial_group_size must be None for critic models.")

        # Reuse the established per-group encoder and normalization contract, then
        # replace the temporary residual head with the parallel SimBaV2 head.
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dim=hidden_dim,
            num_blocks=1,
            expand=expansion,
            activation="relu",
            norm=False,
            obs_normalization=obs_normalization,
            encoder_normalization=encoder_normalization,
            head_layer_norm=False,
            distribution_cfg=distribution_cfg,
            encoder_cfg=encoder_cfg,
            cnns=cnns,
            hidden_dims=hidden_dims,
        )
        if self.distribution is not None and type(self.distribution) is not GaussianDistribution:
            raise ValueError("SimBaV2 PPO actors support only the state-independent GaussianDistribution.")
        model_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.mlp = SimbaV2Head(
            input_dim=self._get_latent_dim(),
            output_dim=model_output_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            expansion=expansion,
            c_shift=c_shift,
            simplicial_group_size=simplicial_group_size,
            simplicial_temperature=simplicial_temperature,
        )

    @property
    def hyperspherical_backbone(self) -> bool:
        """Whether optimizer steps must project this model's normalized weights."""
        return True

    def project_hyperspherical_weights_(self) -> None:
        """Project all SimBaV2 linear weights onto unit row norms."""
        self.mlp.project_hyperspherical_weights_()


class CategoricalSimbaV2EncoderModel(CategoricalValueMixin, SimbaV2EncoderModel):
    """SimBaV2 critic with a categorical value head."""

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
