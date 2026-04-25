# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Boston Dynamics Spot preset. Activate with ``presets=spot``."""

__all__: list[str] = []

from isaaclab.assets import ArticulationCfg

from isaaclab_assets.robots.spot import SPOT_CFG

from .robot_presets import (
    AsyncFootPairsCfg,
    BaseBodyNameCfg,
    ExperimentNameCfg,
    FootBodyNamesCfg,
    HeightScannerPrimPathCfg,
    NonFootBodyNamesCfg,
    RetargetFootBodyNamesCfg,
    RetargetHaaJointPatternCfg,
    RetargetJointRegularizeTargetsCfg,
    RobotArticulationCfg,
    SyncFootPairsCfg,
)

_SPOT_CFG: ArticulationCfg = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

RobotArticulationCfg.spot = _SPOT_CFG
HeightScannerPrimPathCfg.spot = "{ENV_REGEX_NS}/Robot/body"
BaseBodyNameCfg.spot = "body"
# Spot feet are lowercase ``*_foot`` rather than the default ``.*FOOT.*``.
FootBodyNamesCfg.spot = ".*_foot"
NonFootBodyNamesCfg.spot = "^(?!.*_foot).*$"
AsyncFootPairsCfg.spot = (
    ("fl_foot", "fr_foot"),
    ("hr_foot", "hl_foot"),
    ("fl_foot", "hl_foot"),
    ("fr_foot", "hr_foot"),
)
SyncFootPairsCfg.spot = (("fl_foot", "hr_foot"), ("fr_foot", "hl_foot"))
ExperimentNameCfg.spot = "spot_position_command"


# ---------------------------------------------------------------------------
# Retarget constants for Spot
# ---------------------------------------------------------------------------

SPOT_HAA_PATTERN = ".*hip_x"
"""Regex matching Spot hip abduction/adduction joint names."""

RetargetFootBodyNamesCfg.spot = ["fl_foot", "fr_foot", "hl_foot", "hr_foot"]
RetargetHaaJointPatternCfg.spot = SPOT_HAA_PATTERN
# Pull HAA toward 0 and knees toward Spot's crouched init-pose angle; hip-swing
# is left free.
RetargetJointRegularizeTargetsCfg.spot = {
    ".*hip_x": 0.0,
    ".*_kn": -1.5,
}
