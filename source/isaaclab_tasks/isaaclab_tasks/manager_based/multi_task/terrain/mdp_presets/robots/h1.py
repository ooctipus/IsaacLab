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
    RetargetJointRegularizeTargetsCfg,
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


# ---------------------------------------------------------------------------
# Retarget validation criteria for Unitree H1
# ---------------------------------------------------------------------------

# Regularize lateral-hip, torso, and arm DOFs toward the default stance.
# Hip pitch, knee, and ankle pitch are left free so IK can tune the stance
# for terrain.
RetargetJointRegularizeTargetsCfg.h1 = {
    ".*_hip_yaw": 0.0,
    ".*_hip_roll": 0.0,
    "torso": 0.0,
    ".*_shoulder_pitch": 0.28,
    ".*_shoulder_roll": 0.0,
    ".*_shoulder_yaw": 0.0,
    ".*_elbow": 0.52,
}
