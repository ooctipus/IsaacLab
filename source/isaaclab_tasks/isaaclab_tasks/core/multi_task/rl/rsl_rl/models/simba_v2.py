# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .simplicial_embedding import SimplicialEmbedding


def l2_normalize(value: torch.Tensor, dim: int = -1, epsilon: float = 1.0e-8) -> torch.Tensor:
    """Normalize a tensor along one dimension with a finite zero-vector fallback."""
    norm = torch.linalg.vector_norm(value, dim=dim, keepdim=True)
    return value / norm.clamp_min(epsilon)


class HypersphericalLinear(nn.Module):
    """Bias-free linear layer whose output-neuron weights are unit normalized."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.orthogonal_(self.weight)
        self.project_weights_()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.weight)

    @torch.no_grad()
    def project_weights_(self) -> None:
        """Project every output neuron's incoming weights onto the unit sphere."""
        self.weight.div_(torch.linalg.vector_norm(self.weight, dim=1, keepdim=True).clamp_min(1.0e-8))


class FeatureScaler(nn.Module):
    """Learnable feature-wise gain with source-matched gradient scaling."""

    def __init__(self, feature_dim: int, init: float = 1.0, scale: float = 1.0) -> None:
        super().__init__()
        if scale == 0.0:
            raise ValueError("FeatureScaler scale must be non-zero.")
        self.gain = nn.Parameter(torch.full((feature_dim,), scale))
        self.forward_scale = init / scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.gain * self.forward_scale * value


class HypersphericalEmbedder(nn.Module):
    """Shift, normalize, and embed observations on a unit hypersphere."""

    def __init__(self, input_dim: int, hidden_dim: int, c_shift: float = 3.0) -> None:
        super().__init__()
        if c_shift <= 0.0:
            raise ValueError(f"c_shift must be positive; got {c_shift}.")
        scaler_init = math.sqrt(2.0 / hidden_dim)
        self.linear = HypersphericalLinear(input_dim + 1, hidden_dim)
        self.scaler = FeatureScaler(hidden_dim, scaler_init, scaler_init)
        self.c_shift = c_shift

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shift = value.new_full((*value.shape[:-1], 1), self.c_shift)
        shifted = l2_normalize(torch.cat((value, shift), dim=-1))
        return l2_normalize(self.scaler(self.linear(shifted)))


class HypersphericalMLP(nn.Module):
    """Inverted-bottleneck SimBaV2 transformation."""

    def __init__(self, hidden_dim: int, expansion: int = 4) -> None:
        super().__init__()
        if expansion <= 0:
            raise ValueError(f"expansion must be positive; got {expansion}.")
        expanded_dim = hidden_dim * expansion
        scaler_init = math.sqrt(2.0 / expanded_dim)
        self.input_linear = HypersphericalLinear(hidden_dim, expanded_dim)
        self.scaler = FeatureScaler(expanded_dim, scaler_init, scaler_init)
        self.output_linear = HypersphericalLinear(expanded_dim, hidden_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.scaler(self.input_linear(value))
        hidden = F.relu(hidden) + 1.0e-8
        return l2_normalize(self.output_linear(hidden))


class HypersphericalLERPBlock(nn.Module):
    """SimBaV2 block with a learnable feature-wise interpolation."""

    def __init__(self, hidden_dim: int, num_blocks: int, expansion: int = 4) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive; got {num_blocks}.")
        self.mlp = HypersphericalMLP(hidden_dim, expansion)
        self.alpha = FeatureScaler(
            hidden_dim,
            init=1.0 / (num_blocks + 1),
            scale=1.0 / math.sqrt(hidden_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        transformed = self.mlp(value)
        return l2_normalize(value + self.alpha(transformed - value))


class HypersphericalPredictor(nn.Module):
    """Two-linear SimBaV2 output predictor with an optional SEM bottleneck."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int | Sequence[int],
        simplicial_group_size: int | None = None,
        simplicial_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.output_shape = (output_dim,) if isinstance(output_dim, int) else tuple(output_dim)
        flat_output_dim = math.prod(self.output_shape)
        self.hidden_linear = HypersphericalLinear(hidden_dim, hidden_dim)
        self.scaler = FeatureScaler(hidden_dim)
        self.simplicial_embedding = (
            SimplicialEmbedding(hidden_dim, simplicial_group_size, simplicial_temperature)
            if simplicial_group_size is not None
            else nn.Identity()
        )
        self.output_linear = HypersphericalLinear(hidden_dim, flat_output_dim)
        self.output_bias = nn.Parameter(torch.zeros(flat_output_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.scaler(self.hidden_linear(value))
        hidden = self.simplicial_embedding(hidden)
        output = self.output_linear(hidden) + self.output_bias
        if len(self.output_shape) > 1:
            output = output.unflatten(-1, self.output_shape)
        return output


class SimbaV2Head(nn.Module):
    """Paper-faithful SimBaV2 backbone and PPO-compatible output predictor."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | Sequence[int],
        hidden_dim: int = 128,
        num_blocks: int = 1,
        expansion: int = 4,
        c_shift: float = 3.0,
        simplicial_group_size: int | None = None,
        simplicial_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive; got {hidden_dim}.")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive; got {num_blocks}.")

        self.embedder = HypersphericalEmbedder(input_dim, hidden_dim, c_shift)
        self.blocks = nn.ModuleList(
            HypersphericalLERPBlock(hidden_dim, num_blocks, expansion) for _ in range(num_blocks)
        )
        self.predictor = HypersphericalPredictor(
            hidden_dim,
            output_dim,
            simplicial_group_size,
            simplicial_temperature,
        )

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        """Return the final unit-norm backbone representation."""
        hidden = self.embedder(value)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.encode(value))

    @torch.no_grad()
    def project_hyperspherical_weights_(self) -> None:
        """Project all SimBaV2 linear weights after an optimizer update."""
        for module in self.modules():
            if isinstance(module, HypersphericalLinear):
                module.project_weights_()
