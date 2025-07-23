# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from isaaclab.utils import configclass
from typing import Dict, Optional


@configclass
class ActorCriticVisionEncoderCfg:

    image_feature_dim: int = 128

    encoder_type: str = "cnn"

    encoder_config: Optional[Dict] = dict(model_name="resnet18")

    freeze_encoder: bool = False
    
    normalize: bool = False
