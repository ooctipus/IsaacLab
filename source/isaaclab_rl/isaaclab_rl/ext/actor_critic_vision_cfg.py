# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from dataclasses import MISSING
from isaaclab.utils import configclass
from . import vision_encoder as encoders 

@configclass
class ActorCriticVisionAdapterCfg:

    encoder_cfgs: dict[str, CNNEncoderCfg | PretrainedEncoderCfg | PointNetEncoderCfg | MLPEncoderCfg] = MISSING
    """Encoders backbone that predicts feature for downstram usecases"""

    projectors_cfg: dict[str, ProjectorCfg] | None = None
    """Extra heads that predict secondary objectives for encoder, primary objective is policy objective."""

@configclass
class ProjectorCfg:
    
    prediction_group: list[str] = MISSING
    
    layers: list[int] = [64, 64]
    
    activation: str = 'elu'
    
    freeze = False


@configclass
class EncoderBaseCfg:
    
    encoding_groups: list[str] = MISSING
    
    output_dim: int | None = None
    
    activation: str | None  = None
    
    normalize: bool = False
    
    freeze: bool = False

@configclass
class PointNetEncoderCfg(EncoderBaseCfg):
    """Config for per-point MLP → max-pool PointNet encoder."""

    class_type: type[encoders.PointNetEncoder] = encoders.PointNetEncoder

    channels: list[int] = [64, 128, 256]
    
    strides: list[int] = [2, 2, 2]

    use_global_feat: bool = True


@configclass
class CNNEncoderCfg(EncoderBaseCfg):
    
    class_type: type[encoders.CNNEncoder] = encoders.CNNEncoder
    
    channels: list[int] = [32, 64, 128]
    
    kernel_sizes: list[int] = [3, 3, 3]
    
    strides: list[int] = [2, 2, 2]
    
    paddings: list[int] = [1, 1, 1]
    
    use_maxpool: bool = True
    
    pool_size: int = 2


@configclass
class MLPEncoderCfg(EncoderBaseCfg):
    
    class_type: type[encoders.MLPEncoder] = encoders.MLPEncoder
    
    layers: list[int] = [512, 256, 128]


@configclass
class PretrainedEncoderCfg(EncoderBaseCfg):
    # BUG!!!!
    class_type: type[encoders.MLPEncoder] = encoders.MLPEncoder
    
    model_name: str = 'resnet18'
    
    
