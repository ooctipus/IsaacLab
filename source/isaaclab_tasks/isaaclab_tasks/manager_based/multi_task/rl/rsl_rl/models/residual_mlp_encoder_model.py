# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch.nn as nn
from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

from .mlp_encoder_model import MLPEncoderModel
from .mlp_encoder_model import mlp_output_dim as get_mlp_output_dim
from .residual_mlp import ResidualMLP


class ResidualMLPEncoderModel(MLPEncoderModel):
    """MLPEncoderModel with a residual MLP head."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        expand: int = 4,
        activation: str = "relu",
        norm: bool = True,
        last_activation: str | None = None,
        obs_normalization: bool = False,
        encoder_normalization: bool = False,
        head_layer_norm: bool = False,
        distribution_cfg: dict | None = None,
        encoder_cfg: dict[str, dict[str, Any]] | None = None,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
        hidden_dims: list[int] | tuple[int, ...] | None = None,  # noqa: ARG002
    ) -> None:
        if encoder_cfg is None and cnns is None:
            raise ValueError(
                "ResidualMLPEncoderModel requires 'encoder_cfg' unless encoders are shared through 'cnns'."
            )
        nn.Module.__init__(self)
        self._encoded_obs_group_keys = list(encoder_cfg.keys() if encoder_cfg is not None else cnns.keys())
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        if cnns is not None:
            if set(cnns.keys()) != set(self.obs_groups_encoded):
                raise ValueError("Shared encoders must match encoded observation groups.")
            encoders = cnns
        else:
            if set(encoder_cfg) != set(self.obs_groups_encoded):
                raise ValueError("encoder_cfg keys must match encoded observation groups.")
            encoders = {
                obs_group: MLP(input_dim=self.obs_dims_encoded[i], **encoder_cfg[obs_group])
                for i, obs_group in enumerate(self.obs_groups_encoded)
            }
        self.obs_normalization = obs_normalization
        self.obs_normalizer = EmpiricalNormalization(self.obs_dim) if obs_normalization else nn.Identity()
        self.encoders = encoders if isinstance(encoders, nn.ModuleDict) else nn.ModuleDict(encoders)
        self.cnns = self.encoders
        self.encoder_normalization = encoder_normalization
        self.encoder_normalizers = nn.ModuleDict(
            {
                obs_group: EmpiricalNormalization(self.obs_dims_encoded[i]) if encoder_normalization else nn.Identity()
                for i, obs_group in enumerate(self.obs_groups_encoded)
            }
        )
        self.encoder_latent_dim = sum(get_mlp_output_dim(encoder) for encoder in self.encoders.values())
        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim
        latent_dim = self._get_latent_dim()
        self.mlp = ResidualMLP(
            latent_dim,
            mlp_output_dim,
            hidden_dim,
            num_blocks,
            expand,
            2,
            activation,
            last_activation,
            norm,
        )
        self.head_norm = nn.LayerNorm(latent_dim) if head_layer_norm else nn.Identity()
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)
