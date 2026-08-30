# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn


class SimplicialEmbedding(nn.Module):
    """Project feature groups onto a product of probability simplices.

    The final feature dimension is split into equal groups and softmax-normalized
    within each group. The original shape is restored after normalization.
    """

    def __init__(self, feature_dim: int, group_size: int = 64, temperature: float = 1.0) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive; got {feature_dim}.")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive; got {group_size}.")
        if feature_dim % group_size != 0:
            raise ValueError(f"feature_dim ({feature_dim}) must be divisible by group_size ({group_size}) for SEM.")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive; got {temperature}.")

        self.feature_dim = feature_dim
        self.group_size = group_size
        self.num_groups = feature_dim // group_size
        self.temperature = temperature

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply group-wise softmax while preserving all leading dimensions."""
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected SEM input width {self.feature_dim}; got {features.shape[-1]}.")
        grouped = features.reshape(*features.shape[:-1], self.num_groups, self.group_size)
        normalized = torch.softmax(grouped / self.temperature, dim=-1)
        return normalized.flatten(start_dim=-2)
