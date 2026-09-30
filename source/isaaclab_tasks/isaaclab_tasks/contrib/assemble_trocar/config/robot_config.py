# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot configuration for the ``install_trocar`` task."""

import math

from isaaclab.assets import ArticulationCfg

from isaaclab_assets.robots.unitree import G129_CFG_WITH_DEX3_BASE_FIX

# Joint indices in the full robot joint vector for observation extraction.
# Body joints: 29 DOF (legs, waist, arms, wrists)
G1_29DOF_BODY_JOINT_INDICES: list[int] = [
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
]

# Dex3 hand joints: 14 DOF (left + right)
G1_DEX3_JOINT_INDICES: list[int] = [31, 37, 41, 30, 36, 29, 35, 34, 40, 42, 33, 39, 32, 38]

# Default joint positions for the supported setup (G1 29DOF + Dex3).
_DEFAULT_JOINT_POS: dict[str, float] = {
    # legs
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0,
    "left_ankle_pitch_joint": 0.0,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.0,
    "right_ankle_pitch_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    # waist
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    # arms
    "left_shoulder_pitch_joint": -0.754599,
    "left_shoulder_roll_joint": 0.550010,
    "left_shoulder_yaw_joint": -0.399298,
    "left_elbow_joint": 0.278886,
    "left_wrist_roll_joint": 0.320559,
    "left_wrist_pitch_joint": -0.203525,
    "left_wrist_yaw_joint": -0.387435,
    "right_shoulder_pitch_joint": -0.340858,
    "right_shoulder_roll_joint": -0.186152,
    "right_shoulder_yaw_joint": 0.015023,
    "right_elbow_joint": -0.777159,
    "right_wrist_roll_joint": 0.019805,
    "right_wrist_pitch_joint": 1.182285,
    "right_wrist_yaw_joint": -0.022848,
    # dex3 hands (left)
    "left_hand_index_0_joint": -60.0 * math.pi / 180.0,
    "left_hand_middle_0_joint": -60.0 * math.pi / 180.0,
    "left_hand_thumb_0_joint": 0.0,
    "left_hand_index_1_joint": -40.0 * math.pi / 180.0,
    "left_hand_middle_1_joint": -40.0 * math.pi / 180.0,
    "left_hand_thumb_1_joint": 0.0,
    "left_hand_thumb_2_joint": 0.0,
    # dexterous hand joint - right hand
    "right_hand_index_0_joint": 60.0 * math.pi / 180.0,
    "right_hand_middle_0_joint": 60.0 * math.pi / 180.0,
    "right_hand_thumb_0_joint": 0.0,
    "right_hand_index_1_joint": 40.0 * math.pi / 180.0,
    "right_hand_middle_1_joint": 40.0 * math.pi / 180.0,
    "right_hand_thumb_1_joint": 0.0,
    "right_hand_thumb_2_joint": 0.0,
}

G1_29DOF_DEX3_CFG = G129_CFG_WITH_DEX3_BASE_FIX.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.15, 0.0, 0.744),
        rot=(0.0, 0.0, 0.7071, 0.7071),
        joint_pos=_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
)
