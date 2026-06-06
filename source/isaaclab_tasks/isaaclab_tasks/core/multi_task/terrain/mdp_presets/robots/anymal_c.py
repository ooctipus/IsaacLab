# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Anymal-C robot preset. Activate with ``presets=anymal_c``."""

from __future__ import annotations

__all__: list[str] = []

from isaaclab.assets import ArticulationCfg

import isaaclab_assets.robots.anymal as anymal

from ...retarget.criteria import BaseZError, FootPositionError, JointMargin
from .robot_presets import (
    AsyncFootPairsCfg,
    BaseBodyNameCfg,
    ExperimentNameCfg,
    FootBodyNamesCfg,
    HeightScannerPrimPathCfg,
    NonFootContactBodyNamesCfg,
    RetargetJointRegularizeTargetsCfg,
    RetargetLateralHipJointPatternCfg,
    RobotArticulationCfg,
    SyncFootPairsCfg,
)

_ANYMAL_C_CFG: ArticulationCfg = anymal.ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
_ANYMAL_C_CFG.spawn.usd_path = (  # type: ignore[attr-defined]
    "https://uwlab-assets.s3.us-west-004.backblazeb2.com/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
)

RobotArticulationCfg.anymal_c = _ANYMAL_C_CFG
HeightScannerPrimPathCfg.anymal_c = "{ENV_REGEX_NS}/Robot/base"
BaseBodyNameCfg.anymal_c = "base"
NonFootContactBodyNamesCfg.anymal_c = "^(?!.*FOOT).*$"
FootBodyNamesCfg.anymal_c = ".*FOOT.*"
AsyncFootPairsCfg.anymal_c = (
    ("LF_FOOT", "RF_FOOT"),
    ("RH_FOOT", "LH_FOOT"),
    ("LF_FOOT", "LH_FOOT"),
    ("RF_FOOT", "RH_FOOT"),
)
SyncFootPairsCfg.anymal_c = (("LF_FOOT", "RH_FOOT"), ("RF_FOOT", "LH_FOOT"))
ExperimentNameCfg.anymal_c = "anymal_c_position_command"


# ---------------------------------------------------------------------------
# Retarget validation criteria for ANYmal-C
# ---------------------------------------------------------------------------

ANYMAL_C_LATERAL_HIP_PATTERN = ".*HAA"
"""Regex matching ANYmal-C lateral hip joint names."""

RetargetLateralHipJointPatternCfg.anymal_c = ANYMAL_C_LATERAL_HIP_PATTERN
# Pull lateral hips toward 0 (base near support-polygon centroid) and knees
# toward their init-pose flexion (front knees tuck forward, hind knees back).
# Hip flexion/extension is left free so IK can adjust stride.
RetargetJointRegularizeTargetsCfg.anymal_c = {
    ANYMAL_C_LATERAL_HIP_PATTERN: 0.0,
    ".*F_KFE": -0.8,
    ".*H_KFE": 0.8,
}

# Re-export generic criteria used by existing terrain tuning scripts.
__all__ += ["FootPositionError", "JointMargin", "BaseZError"]
