# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct robot and exact reference-kinematics preset axes."""

from isaaclab.assets import ArticulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from .g1 import (
    G1_BEHAVIOR_BODY_NAMES,
    G1_BEHAVIOR_JOINT_NAMES,
    G1_MOTION_ARTICULATION_CFG,
    g1_reference_kinematics,
)
from .smpl import SMPL_MOTION_ARTICULATION_CFG, smpl_reference_kinematics


@configclass
class RobotArticulationCfg(PresetCfg):
    """Full simulation articulation selected by the shared motion preset name."""

    default: ArticulationCfg = SMPL_MOTION_ARTICULATION_CFG
    smpl_cmu: ArticulationCfg = SMPL_MOTION_ARTICULATION_CFG
    g1_lafan: ArticulationCfg = G1_MOTION_ARTICULATION_CFG
    g1_cmu: ArticulationCfg = G1_MOTION_ARTICULATION_CFG


__all__ = [
    "G1_MOTION_ARTICULATION_CFG",
    "SMPL_MOTION_ARTICULATION_CFG",
    "RobotArticulationCfg",
    "G1_BEHAVIOR_BODY_NAMES",
    "G1_BEHAVIOR_JOINT_NAMES",
    "g1_reference_kinematics",
    "smpl_reference_kinematics",
]
