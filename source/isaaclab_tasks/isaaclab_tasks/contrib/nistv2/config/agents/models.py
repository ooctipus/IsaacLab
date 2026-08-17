# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL models for Factory V2."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDict


class PointCloudModel(MLPModel):
    """Encode ordered asset point clouds before the policy MLP."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        point_cloud_mlp_cfg: dict,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        point_cloud_group: str = "perception",
    ) -> None:
        """Initialize the point-cloud model.

        Args:
            obs: Observation dictionary.
            obs_groups: Mapping from model inputs to environment observation groups.
            obs_set: Observation set used by this model.
            output_dim: Model output dimension.
            point_cloud_mlp_cfg: Point-cloud encoder configuration.
            hidden_dims: Hidden dimensions of the policy or value MLP.
            activation: Activation used by the policy or value MLP.
            obs_normalization: Whether to normalize non-geometric observations.
            distribution_cfg: Optional action-distribution configuration.
            point_cloud_group: Observation group containing flattened xyz points.
        """
        self.point_cloud_group = point_cloud_group
        self.point_latent_dim = point_cloud_mlp_cfg["output_dim"]
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
        self.point_cloud_dim = obs[point_cloud_group].shape[-1]
        encoder_dims = [self.point_cloud_dim, *point_cloud_mlp_cfg["hidden_dims"], self.point_latent_dim]
        encoder_layers: list[nn.Module] = []
        for input_dim, layer_output_dim in zip(encoder_dims, encoder_dims[1:]):
            encoder_layers.extend(
                [nn.Linear(input_dim, layer_output_dim), resolve_nn_activation(point_cloud_mlp_cfg["activation"])]
            )
        self.point_encoder = nn.Sequential(*encoder_layers)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Concatenate normalized state history with encoded scene geometry."""
        point_latent = self.point_encoder(obs[self.point_cloud_group])
        if not self.obs_groups:
            return point_latent
        return torch.cat((super().get_latent(obs, masks, hidden_state), point_latent), dim=-1)

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-compatible inference model."""
        return _TorchPointCloudModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return an ONNX-compatible inference model."""
        return _OnnxPointCloudModel(self, verbose)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_groups = obs_groups[obs_set]
        if self.point_cloud_group not in active_groups:
            raise ValueError(f"Observation set {obs_set!r} does not include {self.point_cloud_group!r}.")

        point_cloud = obs[self.point_cloud_group]
        if point_cloud.ndim != 2:
            raise ValueError(f"Point-cloud group {self.point_cloud_group!r} must be flattened per environment.")

        state_groups = [group for group in active_groups if group != self.point_cloud_group]
        state_dim = 0
        for group in state_groups:
            if obs[group].ndim != 2:
                raise ValueError(f"State observation {group!r} must be one-dimensional per environment.")
            state_dim += obs[group].shape[-1]
        return state_groups, state_dim

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.point_latent_dim


class _TorchPointCloudModel(nn.Module):
    def __init__(self, model: PointCloudModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.point_encoder = copy.deepcopy(model.point_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module() if model.distribution is not None else nn.Identity()
        )

    def forward(self, state: torch.Tensor, point_cloud: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference from state and point-cloud observations."""
        latent = torch.cat((self.obs_normalizer(state), self.point_encoder(point_cloud)), dim=-1)
        return self.deterministic_output(self.mlp(latent))

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent state (no-op for this feed-forward model)."""
        pass


class _OnnxPointCloudModel(_TorchPointCloudModel):
    def __init__(self, model: PointCloudModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.state_dim = model.obs_dim
        self.point_cloud_dim = model.point_cloud_dim
        self.point_cloud_group = model.point_cloud_group

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return representative state and point-cloud inputs."""
        return torch.zeros(1, self.state_dim), torch.zeros(1, self.point_cloud_dim)

    @property
    def input_names(self) -> list[str]:
        """Names of the ONNX inputs."""
        return ["obs", self.point_cloud_group]

    @property
    def output_names(self) -> list[str]:
        """Names of the ONNX outputs."""
        return ["actions"]
