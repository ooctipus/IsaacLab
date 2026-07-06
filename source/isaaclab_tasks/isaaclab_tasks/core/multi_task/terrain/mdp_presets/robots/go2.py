# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree Go2 robot preset. Activate with ``presets=go2``."""

__all__: list[str] = []

from isaaclab.assets import ArticulationCfg

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from .robot_presets import (
    ExperimentNameCfg,
    FootBodyNamesCfg,
    RetargetLateralHipJointPatternCfg,
    RobotArticulationCfg,
)

_GO2_CFG: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
_GO2_CFG.spawn.usd_path = (  # type: ignore[attr-defined]
    "https://uwlab-assets.s3.us-west-004.backblazeb2.com/Robots/Unitree/Go2/go2.usd"
)
_GO2_CFG.spawn.articulation_props.enabled_self_collisions = True  # type: ignore[union-attr]
# Stand higher (less squatted) than the default Unitree pose.
_GO2_CFG.init_state.joint_pos = {
    ".*L_hip_joint": 0.0,
    ".*R_hip_joint": 0.0,
    "F[L,R]_thigh_joint": 0.35,
    "R[L,R]_thigh_joint": 0.35,
    ".*_calf_joint": -0.873,
}

RobotArticulationCfg.go2 = _GO2_CFG
# Unitree feet are lowercase ``*_foot`` rather than the default ``.*FOOT.*``.
FootBodyNamesCfg.go2 = ".*_foot"
ExperimentNameCfg.go2 = "go2_position_command"


# ---------------------------------------------------------------------------
# Retarget constants for Go2
# ---------------------------------------------------------------------------

GO2_LATERAL_HIP_PATTERN = ".*hip_joint"
"""Regex matching Go2 lateral hip joint names."""

RetargetLateralHipJointPatternCfg.go2 = GO2_LATERAL_HIP_PATTERN
