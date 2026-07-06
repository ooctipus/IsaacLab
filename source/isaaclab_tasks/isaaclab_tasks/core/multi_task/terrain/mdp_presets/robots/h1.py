# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree H1 humanoid preset. Activate with ``presets=h1``."""

__all__: list[str] = []

from isaaclab.assets import ArticulationCfg

from isaaclab_assets.robots.unitree import H1_CFG

from .robot_presets import (
    AsyncFootPairsCfg,
    BaseBodyNameCfg,
    ExperimentNameCfg,
    FootBodyNamesCfg,
    HeightScannerPrimPathCfg,
    NonFootContactBodyNamesCfg,
    RobotArticulationCfg,
    SyncFootPairsCfg,
)

_H1_CFG: ArticulationCfg = H1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
_H1_CFG.spawn.usd_path = (  # type: ignore[attr-defined]
    "https://uwlab-assets.s3.us-west-004.backblazeb2.com/Robots/Unitree/H1/h1.usd"
)
_H1_CFG.spawn.articulation_props.enabled_self_collisions = True  # type: ignore[union-attr]

RobotArticulationCfg.h1 = _H1_CFG
HeightScannerPrimPathCfg.h1 = "{ENV_REGEX_NS}/Robot/torso_link"
BaseBodyNameCfg.h1 = "torso_link"
FootBodyNamesCfg.h1 = ".*ankle_link"
NonFootContactBodyNamesCfg.h1 = "^(?!.*ankle_link).*$"
AsyncFootPairsCfg.h1 = (("left_ankle_link", "right_ankle_link"),)
SyncFootPairsCfg.h1 = ()
ExperimentNameCfg.h1 = "h1_position_command"
