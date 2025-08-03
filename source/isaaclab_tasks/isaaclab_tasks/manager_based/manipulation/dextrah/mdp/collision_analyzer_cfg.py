# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from .collision_analyzer import CollisionAnalyzer

@configclass
class CollisionAnalyzerCfg:
    
    class_type: type[CollisionAnalyzer] = CollisionAnalyzer

    num_points: int = 32
    
    max_dist: float = 0.5
    
    asset_cfg: SceneEntityCfg = MISSING
    
    obstacle_cfgs: list[SceneEntityCfg] = MISSING
