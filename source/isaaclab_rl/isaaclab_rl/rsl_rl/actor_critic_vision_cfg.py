# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from dataclasses import MISSING
from isaaclab.utils import configclass
from .ext.modules import vision_encoder as encoders 

@configclass
class ActorCriticVisionAdapterCfg:

    encoder_cfg: CNNEncoderCfg | PretrainedEncoderCfg | PointNetEncoderCfg = MISSING

    feature_dim: int | None = 128
    
    activation: str | None  = None
    
    normalize: bool = False
    
    normalize_style: str = "normal"
    
    freeze: bool = False


@configclass
class PointNetEncoderCfg:
    """Config for per-point MLP → max-pool PointNet encoder."""

    class_type: type[encoders.PointNetEncoder] = encoders.PointNetEncoder

    channels: list[int] = [64, 128, 256]
    
    strides: list[int] = [2, 2, 2]

    use_global_feat: bool = True
    
    feature_dim: int | None = None


@configclass
class CNNEncoderCfg:
    
    class_type: type[encoders.CNNEncoder] = encoders.CNNEncoder
    
    channels: list[int] = [32, 64, 128]
    
    kernel_sizes: list[int] = [3, 3, 3]
    
    strides: list[int] = [2, 2, 2]
    
    paddings: list[int] = [1, 1, 1]
    
    use_maxpool: bool = True
    
    pool_size: int = 2


@configclass
class PretrainedEncoderCfg:
    
    model_name: str = 'resnet18'
    
    
