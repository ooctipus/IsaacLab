# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native SMPL HumEnv asset paths, source ordering, and physical constants."""

from __future__ import annotations

import os

_SMPL_DIR = os.path.dirname(os.path.abspath(__file__))
SMPL_ROBOT_MJCF_PATH = os.path.join(_SMPL_DIR, "robot.xml")
SMPL_ROBOT_MJCF_SHA256 = "f91b6d512f9fcece8d781ccf49afbea2ba5dfd33be0be376e84d5b5579e79fe2"
SMPL_HUMENV_MJCF_PATH = os.path.join(_SMPL_DIR, "humenv.xml")
SMPL_HUMENV_MJCF_SHA256 = "a38c014814929152ddeb3e2a4f18a99c5eb861f0c7f6678416624c39dba9b024"
SMPL_NEWTON_VARIANT = "mujoco"
SMPL_ARTICULATION_ROOT_PRIM_PATH = "/Geometry/Pelvis"

NUM_JOINTS = 69
NUM_BODIES = 24

# Canonical qpos/qvel/body order in the packaged HumEnv MJCF.
MUJOCO_JOINT_NAMES: tuple[str, ...] = (
    "L_Hip_x",
    "L_Hip_y",
    "L_Hip_z",
    "L_Knee_x",
    "L_Knee_y",
    "L_Knee_z",
    "L_Ankle_x",
    "L_Ankle_y",
    "L_Ankle_z",
    "L_Toe_x",
    "L_Toe_y",
    "L_Toe_z",
    "R_Hip_x",
    "R_Hip_y",
    "R_Hip_z",
    "R_Knee_x",
    "R_Knee_y",
    "R_Knee_z",
    "R_Ankle_x",
    "R_Ankle_y",
    "R_Ankle_z",
    "R_Toe_x",
    "R_Toe_y",
    "R_Toe_z",
    "Torso_x",
    "Torso_y",
    "Torso_z",
    "Spine_x",
    "Spine_y",
    "Spine_z",
    "Chest_x",
    "Chest_y",
    "Chest_z",
    "Neck_x",
    "Neck_y",
    "Neck_z",
    "Head_x",
    "Head_y",
    "Head_z",
    "L_Thorax_x",
    "L_Thorax_y",
    "L_Thorax_z",
    "L_Shoulder_x",
    "L_Shoulder_y",
    "L_Shoulder_z",
    "L_Elbow_x",
    "L_Elbow_y",
    "L_Elbow_z",
    "L_Wrist_x",
    "L_Wrist_y",
    "L_Wrist_z",
    "L_Hand_x",
    "L_Hand_y",
    "L_Hand_z",
    "R_Thorax_x",
    "R_Thorax_y",
    "R_Thorax_z",
    "R_Shoulder_x",
    "R_Shoulder_y",
    "R_Shoulder_z",
    "R_Elbow_x",
    "R_Elbow_y",
    "R_Elbow_z",
    "R_Wrist_x",
    "R_Wrist_y",
    "R_Wrist_z",
    "R_Hand_x",
    "R_Hand_y",
    "R_Hand_z",
)

MUJOCO_BODY_NAMES: tuple[str, ...] = (
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
)


__all__ = [
    "MUJOCO_BODY_NAMES",
    "MUJOCO_JOINT_NAMES",
    "NUM_BODIES",
    "NUM_JOINTS",
    "SMPL_ARTICULATION_ROOT_PRIM_PATH",
    "SMPL_HUMENV_MJCF_PATH",
    "SMPL_HUMENV_MJCF_SHA256",
    "SMPL_NEWTON_VARIANT",
    "SMPL_ROBOT_MJCF_PATH",
    "SMPL_ROBOT_MJCF_SHA256",
]
