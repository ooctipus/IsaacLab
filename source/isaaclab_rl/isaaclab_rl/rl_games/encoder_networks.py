# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
import torch.nn as nn
from typing import TYPE_CHECKING
from rl_games.algos_torch.running_mean_std import RunningMeanStd

if TYPE_CHECKING:
    from .encoder_network_cfg import CNNCfg, MLPCfg


def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    if act_name == "elu": return torch.nn.ELU()  # noqa: E701
    elif act_name == "selu": return torch.nn.SELU()  # noqa: E701
    elif act_name == "relu": return torch.nn.ReLU()  # noqa: E701
    elif act_name == "crelu": return torch.nn.CELU()  # noqa: E701
    elif act_name == "lrelu": return torch.nn.LeakyReLU()  # noqa: E701
    elif act_name == "tanh": return torch.nn.Tanh()  # noqa: E701
    elif act_name == "sigmoid": return torch.nn.Sigmoid()  # noqa: E701
    elif act_name == "identity": return torch.nn.Identity()  # noqa: E701
    else: raise ValueError(f"Invalid activation function '{act_name}'.")  # noqa: E701


def make_norm2d(norm, c, groups_gn=16):
    if norm == "batch": return nn.BatchNorm2d(c)  # noqa: E701
    if norm == "group": return nn.GroupNorm(num_groups=min(groups_gn, c), num_channels=c)  # noqa: E701
    if norm == "layer": return nn.GroupNorm(1, c)  # noqa: E701
    return nn.Identity()  # noqa: E701


def make_norm1d(norm, c):
    if norm == "batch": return nn.BatchNorm1d(c)  # noqa: E701
    if norm == "layer": return nn.LayerNorm(c)  # noqa: E701
    return nn.Identity()  # noqa: E701


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor):
        return x.permute(*self.dims).contiguous()


class MLP(nn.Module):
    """
    Simple MLP encoder: flatten → (Linear + Norm + Act [+ Dropout]) × N → projector.
    Mirrors the style of CNN above (cfg-driven, optional input normalization & projector).
    """
    cfg: "MLPCfg"

    def __init__(self, input_shape: tuple[int, ...], cfg: "MLPCfg"):
        super().__init__()
        self.cfg: "MLPCfg" = cfg

        # flatten everything except batch
        self.flatten = nn.Flatten(1)

        # input normalization on the flattened vector
        flat_in = int(math.prod(input_shape))
        self.input_norm = RunningMeanStd((flat_in,)) if getattr(cfg, "input_norm", False) else nn.Identity()

        # build encoder stack
        layers: list[nn.Module] = []
        in_dim = flat_in
        for i, out_dim in enumerate(cfg.layers):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(make_norm1d(cfg.norm, out_dim))
            layers.append(resolve_nn_activation(cfg.activation.lower()))
            if getattr(cfg, "dropout", None):
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = out_dim

        self.encoder = nn.Sequential(*layers)

        # optional projector to a fixed feature size
        self.projector: nn.Module = (
            nn.Identity() if getattr(cfg, "feature_size", None) is None else nn.Linear(in_dim, cfg.feature_size[0])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)           # (B, flat_in)
        x = self.input_norm(x)        # (B, flat_in)
        h = self.encoder(x)           # (B, hidden_dim or flat_in if no layers)
        return self.projector(h)      # (B, feature_size) or (B, hidden_dim)


class CNN(nn.Module):

    def __init__(self, input_shape: tuple[int, int, int], cfg):
        super().__init__()
        self.cfg: CNNCfg = cfg
        assert len(cfg.channels) == len(cfg.kernel_sizes) == len(cfg.strides) == len(cfg.paddings), "must match length"

        C, H, W = input_shape
        if cfg.permute:
            H, W, C = input_shape

        self.permute = Permute(0, 3, 1, 2) if cfg.permute else nn.Identity()
        self.input_norm = RunningMeanStd((C, H, W)) if cfg.input_norm else nn.Identity()

        layers: list[nn.Module] = []
        in_c = C
        for i, out_c in enumerate(cfg.channels):
            layers.append(nn.Conv2d(in_c, out_c, cfg.kernel_sizes[i], cfg.strides[i], cfg.paddings[i]))
            layers.append(make_norm2d(cfg.norm, out_c))
            layers.append(resolve_nn_activation(cfg.activation.lower()))
            if cfg.use_maxpool: layers.append(nn.MaxPool2d(cfg.pool_size))  # noqa: E701
            in_c = out_c
        if cfg.gap: layers.append(nn.AdaptiveAvgPool2d((1, 1)))  # noqa: E701

        self.encoder = nn.Sequential(*layers)
        self.flatten = nn.Flatten(1)

        # figure out flattened dim with a dummy forward
        with torch.no_grad():
            p = next(self.encoder.parameters())
            dummy = torch.zeros(1, C, H, W, dtype=p.dtype, device=p.device)
            enc = self.encoder(dummy)
            flat_dim = enc.view(1, -1).shape[1]
        self.projector: nn.Module = nn.Identity() if cfg.feature_size is None else nn.Linear(flat_dim, cfg.feature_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.permute(x)
        x = self.input_norm(x)
        y = self.encoder(x)  # expect x as (B, C, H, W)
        y = self.flatten(y)  # (B, flat_dim)
        return self.projector(y)  # (B, out_dim)
