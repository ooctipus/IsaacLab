# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree B2 robot preset. Activate with ``presets=b2``.

No-op on branches that do not ship ``isaaclab_assets.robots.unitree.B2_CFG``
(e.g. ``main``) -- the preset registration is skipped and ``presets=b2``
simply won't be recognised there.
"""

__all__: list[str] = []

import os

from isaaclab.assets import ArticulationCfg

import isaaclab_assets.robots.unitree as unitree
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

from .robot_presets import (
    BaseBodyNameCfg,
    ExperimentNameCfg,
    FootBodyNamesCfg,
    HeightScannerPrimPathCfg,
    NonFootContactBodyNamesCfg,
    RetargetLateralHipJointPatternCfg,
    RobotArticulationCfg,
)

B2_LATERAL_HIP_PATTERN = ".*hip_joint"
"""Regex matching B2 lateral hip joint names."""

if hasattr(unitree, "B2_CFG"):
    _B2_CFG: ArticulationCfg = unitree.B2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    _B2_CFG.spawn.usd_path = os.path.join(  # type: ignore[attr-defined]
        ISAACLAB_ASSETS_DATA_DIR, "Robots", "Unitree", "B2", "b2.usd"
    )

    RobotArticulationCfg.b2 = _B2_CFG
    HeightScannerPrimPathCfg.b2 = "{ENV_REGEX_NS}/Robot/base_link"
    BaseBodyNameCfg.b2 = "base_link"
    NonFootContactBodyNamesCfg.b2 = "^(?!.*foot$).*$"
    FootBodyNamesCfg.b2 = ".*foot"
    ExperimentNameCfg.b2 = "b2_position_command"
    RetargetLateralHipJointPatternCfg.b2 = B2_LATERAL_HIP_PATTERN
