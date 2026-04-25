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
    BaseContactBodyNamesCfg,
    ExperimentNameCfg,
    FootBodyNamesCfg,
    HeightScannerPrimPathCfg,
    NonFootBodyNamesCfg,
    RetargetFootBodyNamesCfg,
    RetargetHaaJointPatternCfg,
    RetargetJointRegularizeTargetsCfg,
    RobotArticulationCfg,
)

if hasattr(unitree, "B2_CFG"):
    _B2_CFG: ArticulationCfg = unitree.B2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    _B2_CFG.spawn.usd_path = os.path.join(  # type: ignore[attr-defined]
        ISAACLAB_ASSETS_DATA_DIR, "Robots", "Unitree", "B2", "b2.usd"
    )

    RobotArticulationCfg.b2 = _B2_CFG
    HeightScannerPrimPathCfg.b2 = "{ENV_REGEX_NS}/Robot/base_link"
    BaseBodyNameCfg.b2 = "base_link"
    BaseContactBodyNamesCfg.b2 = "base_link"
    FootBodyNamesCfg.b2 = ".*foot"
    NonFootBodyNamesCfg.b2 = "^(?!.*(?:foot)).*$"
    ExperimentNameCfg.b2 = "b2_position_command"
    RetargetFootBodyNamesCfg.b2 = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    RetargetHaaJointPatternCfg.b2 = ".*hip_joint"
    # Same HAA/knee naming convention as Go2.
    RetargetJointRegularizeTargetsCfg.b2 = {
        ".*hip_joint": 0.0,
        ".*_calf_joint": -0.873,
    }
