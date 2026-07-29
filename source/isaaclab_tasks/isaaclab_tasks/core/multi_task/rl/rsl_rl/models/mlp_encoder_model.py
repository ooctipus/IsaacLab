# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from tensordict import TensorDict


class MLPEncoderModel(MLPModel):
    """MLP model with per-observation-group MLP encoders."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        encoder_normalization: bool = False,
        head_layer_norm: bool = True,
        distribution_cfg: dict | None = None,
        encoder_cfg: dict[str, dict[str, Any]] | None = None,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
    ) -> None:
        if encoder_cfg is None and cnns is None:
            raise ValueError("MLPEncoderModel requires 'encoder_cfg' unless encoders are shared through 'cnns'.")
        self._encoded_obs_group_keys = list(encoder_cfg.keys() if encoder_cfg is not None else cnns.keys())
        self._get_obs_dim(obs, obs_groups, obs_set)
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
        self.encoder_latent_dim = sum(mlp_output_dim(encoder) for encoder in encoders.values())
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        self.encoders = encoders if isinstance(encoders, nn.ModuleDict) else nn.ModuleDict(encoders)
        self.cnns = self.encoders
        self.encoder_normalization = encoder_normalization
        normalizers = {
            obs_group: EmpiricalNormalization(self.obs_dims_encoded[i]) if encoder_normalization else nn.Identity()
            for i, obs_group in enumerate(self.obs_groups_encoded)
        }
        self.encoder_normalizers = nn.ModuleDict(normalizers)
        self.head_norm = nn.LayerNorm(self._get_latent_dim()) if head_layer_norm else nn.Identity()

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        latent_passthrough = super().get_latent(obs, masks, hidden_state)
        latent_encoded = torch.cat(
            [
                self.encoders[obs_group](self.encoder_normalizers[obs_group](obs[obs_group].flatten(start_dim=1)))
                for obs_group in self.obs_groups_encoded
            ],
            dim=-1,
        )
        return self.head_norm(torch.cat([latent_passthrough, latent_encoded], dim=-1))

    def update_normalization(self, obs: TensorDict) -> None:
        super().update_normalization(obs)
        if self.encoder_normalization:
            for obs_group in self.obs_groups_encoded:
                self.encoder_normalizers[obs_group].update(  # type: ignore[union-attr]
                    obs[obs_group].flatten(start_dim=1)
                )

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("JIT export is not implemented for MLPEncoderModel.")

    def as_onnx(self, verbose: bool) -> nn.Module:
        raise NotImplementedError("ONNX export is not implemented for MLPEncoderModel.")

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        passthrough_groups: list[str] = []
        passthrough_dim = 0
        encoded_groups: list[str] = []
        encoded_dims: list[int] = []
        for obs_group in obs_groups[obs_set]:
            shape = obs[obs_group].shape
            feature_dim = math.prod(shape[1:])
            if obs_group in self._encoded_obs_group_keys:
                encoded_groups.append(obs_group)
                encoded_dims.append(int(feature_dim))
            else:
                if len(shape) != 2:
                    raise ValueError(f"Observation '{obs_group}' must be encoded or flattened before MLPEncoderModel.")
                passthrough_groups.append(obs_group)
                passthrough_dim += int(shape[-1])
        missing = set(self._encoded_obs_group_keys) - set(encoded_groups)
        if missing:
            raise ValueError(f"encoder_cfg contains observation groups not present in {obs_set}: {sorted(missing)}")
        if not encoded_groups:
            raise ValueError("MLPEncoderModel needs at least one encoded observation group.")
        self.obs_groups_encoded = encoded_groups
        self.obs_dims_encoded = encoded_dims
        return passthrough_groups, passthrough_dim

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.encoder_latent_dim


def mlp_output_dim(mlp: nn.Module) -> int:
    last_linear: nn.Linear | None = None
    for module in mlp.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is None:
        raise ValueError("Could not determine MLP output dimension: no Linear layer found.")
    return last_linear.out_features
