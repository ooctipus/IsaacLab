# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Concrete SMPL simulation articulation and exact reference model."""

from __future__ import annotations

import hashlib
from pathlib import Path

from isaaclab_assets.robots.smpl.smpl_cfg import SMPL_HUMANOID_CFG
from isaaclab_assets.robots.smpl.smpl_constants import SMPL_HUMENV_MJCF_PATH, SMPL_HUMENV_MJCF_SHA256

_ROBOT_PRIM_PATH = "{ENV_REGEX_NS}/Robot"
_SMPL_SIMULATOR_BODY_NAMES = (
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
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "Neck",
    "Head",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
)
_SMPL_SIMULATOR_JOINT_NAMES = tuple(
    f"{body}_x:{component}" for body in _SMPL_SIMULATOR_BODY_NAMES[1:] for component in range(3)
)

SMPL_MOTION_ARTICULATION_CFG = SMPL_HUMANOID_CFG.replace(prim_path=_ROBOT_PRIM_PATH)
"""Packaged exact SMPL articulation whose native asset owns control and passive terms."""


def smpl_reference_kinematics(env):
    """Build the hash-verified packaged HumEnv reference model on the environment device."""
    from ....kinematics import NewtonKinematics, NewtonKinematicsCfg

    path = Path(SMPL_HUMENV_MJCF_PATH)
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != SMPL_HUMENV_MJCF_SHA256:
        raise ValueError(f"SMPL reference MJCF hash differs: expected {SMPL_HUMENV_MJCF_SHA256}, got {actual}.")
    return NewtonKinematics(
        NewtonKinematicsCfg(
            usd_path=None,
            mjcf_path=str(path),
            device=str(env.device),
            collapse_fixed_joints=False,
        )
    )


__all__ = [
    "SMPL_MOTION_ARTICULATION_CFG",
    "smpl_reference_kinematics",
]
