# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from isaaclab.utils import configclass
from .ext.modules import vision_encoder as encoders 

@configclass
class ActorCriticVisionEncoderCfg:

    encoder_cfg: CNNEncoderCfg | PretrainedEncoderCfg | PointNetEncoderCfg = MISSING

    encoder_type: str = "cnn"

    encoder_config: Optional[Dict] = dict(model_name="resnet18")

    freeze_encoder: bool = False
    
    normalize: bool = False
    
    normalize_style: str = "normal"
    
    freeze: bool = False


@configclass
class PointNetEncoderCfg:
    """Config for per-point MLP → max-pool PointNet encoder."""

    class_type: type[encoders.PointNetEncoder] = encoders.PointNetEncoder
    # sizes of each hidden MLP layer (1×1 conv dims)
    channels: list[int] = [64, 128, 256]
    # whether to use the last layer’s output as a global feature
    use_global_feat: bool = True


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
    
    
