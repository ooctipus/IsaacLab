# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from dataclasses import MISSING
from isaaclab.utils import configclass


@configclass
class ActorCriticVisionAdapterCfg:

    encoder_cfg: CNNEncoderCfg | PretrainedEncoderCfg = MISSING

    feature_dim: int = 128
    
    activation: str | None  = None
    
    normalize: bool = False
    
    normalize_style: str = "normal"
    
    freeze: bool = False


@configclass
class CNNEncoderCfg:
    
    channels: list[int] = [32, 64, 128]
    
    kernel_sizes: list[int] = [3, 3, 3]
    
    strides: list[int] = [2, 2, 2]
    
    paddings: list[int] = [1, 1, 1]
    
    use_maxpool: bool = True
    
    pool_size: int = 2


@configclass
class PretrainedEncoderCfg:
    
    model_name: str = 'resnet18'
    
    
