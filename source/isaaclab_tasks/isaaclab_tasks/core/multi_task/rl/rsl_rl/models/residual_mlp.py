# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from functools import reduce

import torch
import torch.nn as nn
from rsl_rl.utils import resolve_nn_activation


def _lecun_uniform_(tensor: torch.Tensor) -> None:
    fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
    nn.init.uniform_(tensor, -1.0 / math.sqrt(fan_in), 1.0 / math.sqrt(fan_in))


class ResidualBlock(nn.Module):
    """Pre-norm residual feedforward block."""

    def __init__(
        self,
        hidden_dim: int,
        expand: int = 4,
        num_layers: int = 2,
        activation: str = "relu",
        norm: bool = True,
    ) -> None:
        super().__init__()
        if num_layers == 2:
            self.pre_norm = nn.LayerNorm(hidden_dim) if norm else nn.Identity()
            self.layers = nn.Sequential(
                nn.Linear(hidden_dim, expand * hidden_dim),
                resolve_nn_activation(activation),
                nn.Linear(expand * hidden_dim, hidden_dim),
            )
        else:
            self.pre_norm = nn.Identity()
            layers: list[nn.Module] = []
            for _ in range(num_layers):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(resolve_nn_activation(activation))
            self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(self.pre_norm(x))


class ResidualMLP(nn.Sequential):
    """Residual MLP body used by Octi RSL-RL presets."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dim: int,
        num_blocks: int = 2,
        expand: int = 4,
        num_layers_per_block: int = 2,
        activation: str = "relu",
        last_activation: str | None = None,
        norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim)]
        if num_layers_per_block > 2 and norm:
            layers.append(nn.LayerNorm(hidden_dim))
        if num_layers_per_block > 2:
            layers.append(resolve_nn_activation(activation))
        for _ in range(num_blocks):
            layers.append(ResidualBlock(hidden_dim, expand, num_layers_per_block, activation, norm))
        layers.append(nn.LayerNorm(hidden_dim) if norm else nn.Identity())
        if isinstance(output_dim, int):
            layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            layers.append(nn.Linear(hidden_dim, reduce(lambda x, y: x * y, output_dim)))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))
        if last_activation is not None:
            layers.append(resolve_nn_activation(last_activation))
        for i, layer in enumerate(layers):
            self.add_module(str(i), layer)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _lecun_uniform_(module.weight)
                nn.init.zeros_(module.bias)
