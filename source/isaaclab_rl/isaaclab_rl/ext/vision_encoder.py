# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from typing import TYPE_CHECKING, Any

# Import the activation resolver
from rsl_rl.utils import resolve_nn_activation

if TYPE_CHECKING:
    from . import actor_critic_vision_cfg as cfg


class VisionAdapter(nn.Module):
    """Base class for all vision encoders."""

    def __init__(self, obs_space: spaces.Box, adapter_cfg: cfg.ActorCriticVisionAdapterCfg):
        super().__init__()
        self.cfg = adapter_cfg
        self.obs_space = obs_space
        self._processors: list[callable] = []
        self._processor_descriptions: list[str] = []
        self._feature_dim = -1

        if obs_space.dtype != np.float32:
            self._processors.append(lambda x: x.float())
            self._processor_descriptions.append("cast to float32")

        if len(obs_space.shape) == 4:
            # determine whether permutation processor and dtype caster processor is needed
            # is normalize is false, permutation and dtype casting should be only processing needed
            if obs_space.shape[1] in [3, 1, 4]:
                self.num_channel = obs_space.shape[1]
            elif obs_space.shape[-1] in [3, 1, 4]:
                self._processors.append(lambda x: x.permute(0, 3, 1, 2))
                self._processor_descriptions.append("permute HWC->CHW")
                self.num_channel = obs_space.shape[-1]
            else:
                raise ValueError("did not detect correct channel")

            # if normalize is True we need more processors.
            if self.cfg.normalize:
                if obs_space.shape[1] == 3 or obs_space.shape[-1] == 3:  # rgb indicator
                    processors, descriptions = self._compile_rgb_processors(obs_space)
                    self._processors.extend(processors)
                    self._processor_descriptions.extend(descriptions)
                elif obs_space.shape[1] == 1 or obs_space.shape[-1] == 1:  # depth indicator
                    processors, descriptions = self._compile_depth_processors(obs_space)
                    self._processors.extend(processors)
                    self._processor_descriptions.extend(descriptions)
                elif obs_space.shape[1] == 4 or obs_space.shape[-1] == 4:  # rgbd indicator
                    processors, descriptions = self._compile_rgbd_processors(obs_space)
                    self._processors.extend(processors)
                    self._processor_descriptions.extend(descriptions)
        elif len(obs_space.shape) == 3:
            # expect (B, N_points, C)
            self.num_channel = obs_space.shape[-1]
            if self.cfg.normalize:
                procs, desc = self._compile_point_cloud_processors(obs_space)
                self._processors.extend(procs)
                self._processor_descriptions.extend(desc)
        else:
            raise ValueError(f"Does not recognize this perception with dim {len(obs_space.shape)}")

    def feature_dim(self):
        return self._feature_dim

    def freeze(self):
        """Freeze all parameters in the model."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def _compile_point_cloud_processors(self, obs_space: Any) -> tuple[list[callable], list[str]]:
        """Build processors for point-cloud inputs (center + unit-sphere scale)."""
        procs: list[callable] = []
        desc: list[str] = []
        # center each cloud at its centroid
        procs.append(lambda x: x - x.mean(dim=1, keepdim=True))
        desc.append("center to centroid")

        # scale to fit in unit sphere
        def scale_unit(x):
            d = torch.norm(x, dim=-1)  # (B, N)
            m = d.max(dim=1, keepdim=True)[0]  # (B, 1)
            return x / (m.unsqueeze(-1) + 1e-6)

        procs.append(scale_unit)
        desc.append("scale to unit sphere")
        return procs, desc

    def _compile_rgb_processors(self, obs_space: Any) -> list[Any]:
        """Build processors for 3-channel inputs."""
        procs: list[callable] = []
        desc: list[str] = []
        if self.cfg.normalize_style == "imagenet":
            # ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
            if obs_space.high == 255:
                procs.append(lambda x: x / 255.0)
                desc.append("rgb: scale / 255")
                procs.append(lambda x: (x - mean) / std)
                desc.append("ImageNet normalize")
            else:
                procs.append(lambda x: (x - mean) / std)
                desc.append("ImageNet normalize")
        elif self.cfg.normalize_style == "normal":
            # "normal": scale to [-1,1] then standardize
            if obs_space.high == 255:
                procs.append(lambda x: (x / 255.0) * 2 - 1)
                desc.append("rgb: scale [0 - 255]→[-1, 1]")
            procs.append(lambda x: (x - x.mean()) / x.std())
            desc.append("rgb: per-image standardize")
        return procs, desc

    def _compile_depth_processors(self, obs_space: Any) -> list[Any]:
        """Build processors for single-channel depth inputs."""
        procs: list[callable] = []
        desc: list[str] = []
        high = obs_space.high
        if np.all(high == np.inf):
            procs.append(lambda x: torch.tanh(x / 2.0) * 2 - 1)
            desc.append("depth: scale [0 - inf]→[-1, 1]")
        elif np.all(high == 255.0):
            procs.append(lambda x: (x / 255.0) * 2 - 1)
            desc.append("depth: scale [0 - 255]→[-1, 1]")
        elif np.all(high == 1.0) and np.all(obs_space.low == 0.0):
            procs.append(lambda x: x * 2 - 1)
            desc.append("depth: scale [0 - 1]→[-1, 1]")
        elif np.all(high == 1.0) and np.all(obs_space.low == -1.0):
            pass  # no need to do anything if already -1 to 1
        else:
            raise ValueError("Your depth image is not qualified for automatic normalization")
        procs.append(
            lambda x: (x - x.mean(dim=tuple(range(1, x.ndim)), keepdim=True))
            / (x.std(dim=tuple(range(1, x.ndim)), keepdim=True) + 1e-8)
        )
        desc.append("depth: per-image standardize")
        return procs, desc

    def _compile_rgbd_processors(self, obs_space: spaces.Box):
        raise NotImplementedError()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all preprocessing steps in sequence."""
        for fn in self._processors:
            x = fn(x)
        return x

    def initialize(self):
        """Hook called at the end of initialization to set up model."""
        if self.cfg.freeze:
            self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.
        Args:
            x: Input tensor in shape (B, C, H, W)
        Returns:
            Encoded features in shape (B, feature_dim)
        """
        raise NotImplementedError("Subclasses must implement forward method")


class CNNEncoder(VisionAdapter):
    """CNN encoder with configurable architecture."""

    cfg: cfg.CNNEncoderCfg

    def __init__(self, obs_space: spaces.Box, encoder_cfg: cfg.CNNEncoderCfg):
        super().__init__(obs_space, encoder_cfg)
        self._build_encoder(obs_space)
        self.initialize()

    def _build_encoder(self, obs_space):
        layers: list[nn.Module] = []
        in_c = self.num_channel
        ec = self.cfg
        activation = resolve_nn_activation(ec.activation)
        for i, out_c in enumerate(ec.channels):
            c = nn.Conv2d(in_c, out_c, kernel_size=ec.kernel_sizes[i], stride=ec.strides[i], padding=ec.paddings[i])
            layers.append(c)
            layers.append(activation)
            if ec.use_maxpool:
                layers.append(nn.MaxPool2d(ec.pool_size))
            in_c = out_c

        layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*layers)

        # determine flattened size via a dummy forward (after preprocessing)
        with torch.no_grad():
            if isinstance(obs_space, torch.Tensor):
                dummy = self.preprocess(obs_space.cpu())
            else:
                dummy = self.preprocess(torch.tensor(obs_space.sample()))
            self._feature_dim = self.encoder(dummy).shape[1]

        if self.cfg.output_dim is not None:
            self.projector = nn.Sequential(nn.Linear(self._feature_dim, self.cfg.output_dim), activation)
        else:
            self.projector = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(self.preprocess(x))
        return self.projector(feats)  # → (B, feature_dim)


class PointNetEncoder(VisionAdapter):
    """Point-cloud encoder: per-point MLP → global-pool → projector."""

    def __init__(self, obs_space: spaces.Box, adapter_cfg: cfg.ActorCriticVisionAdapterCfg):
        super().__init__(obs_space, adapter_cfg)
        self._build_encoder(obs_space)
        self.initialize()

    def _build_encoder(self, obs_space: spaces.Box):
        pc_cfg: cfg.PointNetEncoderCfg = self.cfg.encoder_cfg
        in_c = self.num_channel
        convs: list[nn.Module] = []
        for i, out_c in enumerate(pc_cfg.channels):
            convs.append(nn.Conv1d(in_c, out_c, kernel_size=1, stride=pc_cfg.strides[i]))
            convs.append(nn.BatchNorm1d(out_c))
            convs.append(torch.nn.ReLU())
            in_c = out_c
        self.point_mlp = nn.Sequential(*convs)

        # pooling
        if pc_cfg.use_global_feat:
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.pool = None

        if self.cfg.output_dim is not None:
            self.projector = nn.Sequential(nn.Linear(self._feature_dim, self.cfg.output_dim), torch.nn.ReLU())
        else:
            self.projector = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)  # (B,N,C)
        x = x.transpose(1, 2)  # (B,C,N)
        x = self.point_mlp(x)  # (B,hidden,N)
        if self.pool:
            x = self.pool(x).squeeze(-1)  # (B, hidden)
        else:
            B, C, N = x.shape
            x = x.view(B, C * N)
        return self.projector(x)  # (B, feature_dim)


class MLPEncoder(VisionAdapter):
    """Simple MLP encoder: flatten → hidden-layer MLP → projector."""

    cfg: cfg.MLPEncoderCfg

    def __init__(self, obs_space, encoder_cfg: cfg.MLPEncoderCfg):
        super().__init__(obs_space, encoder_cfg)
        self._build_encoder(obs_space)
        self.initialize()

    def _build_encoder(self, obs_space):
        # figure out total input dim (drop batch axis)
        flat_in = int(np.prod(obs_space.shape[1:]))
        layers = [nn.Flatten()]
        in_dim = flat_in
        for out_dim in self.cfg.layers:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(resolve_nn_activation(self.cfg.activation))
            in_dim = out_dim

        self.encoder = nn.Sequential(*layers)
        # final projector maps ‘in_dim’ → feature_dim if requested
        if self.cfg.output_dim is not None:
            self.projector = nn.Sequential(nn.Linear(self._feature_dim, self.cfg.output_dim), torch.nn.ReLU())
        else:
            self.projector = nn.Identity()

    def forward(self, x):
        x = self.preprocess(x)  # casting / normalization etc.
        h = self.encoder(x)  # → (B, hidden_sizes[-1])
        return self.projector(h)  # → (B, feature_dim) or (B, hidden)


class PreTrainedPointNetEncoder(VisionAdapter):
    """
    Only the SA layers from PointNet++, no classification head.
    Loads pretrained weights (ignores fc1/fc2/fc3 via strict=False).
    """

    def __init__(self, obs_space, adapter_cfg):
        super().__init__(obs_space, adapter_cfg)
        self.normal_vector = False
        # 1) instantiate the full model
        from .pointnet2_ssg_wo_normals import pointnet2_cls_ssg

        full = pointnet2_cls_ssg.get_model(num_class=40, normal_channel=False)

        # 2) load your checkpoint on THAT
        ckpt = torch.load(
            "logs/pointnet/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth", weights_only=False
        )
        full.load_state_dict(ckpt["model_state_dict"])

        # 3) copy over only the SA layers into our own modules
        self.sa1 = full.sa1
        self.sa2 = full.sa2
        self.sa3 = full.sa3

        # 4) optionally freeze them
        if self.cfg.freeze:
            for p in list(self.sa1.parameters()) + list(self.sa2.parameters()) + list(self.sa3.parameters()):
                p.requires_grad = False
            self.sa1.eval()
            self.sa2.eval()
            self.sa3.eval()

        # 5) build your projector 1024→feature_dim (or Identity)
        feat_dim = getattr(self.cfg.encoder_cfg, "feature_dim", None)
        if feat_dim is not None:
            self.projector = nn.Sequential(nn.Linear(1024, feat_dim), resolve_nn_activation("relu"))
        else:
            self.projector = nn.Identity()

    def forward(self, x):
        x = self.preprocess(x)
        x = x.transpose(1, 2)
        if self.normal_vector:
            norm = x[:, 3:, :]
            xyz = x[:, :3, :]
        else:
            norm = None
            xyz = x

        # 3) Stage 1: get new coords & features
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)

        feats = l3_points.view(l3_points.size(0), -1)  # (B,1024)
        return self.projector(feats)
