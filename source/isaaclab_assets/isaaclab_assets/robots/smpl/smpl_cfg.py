# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native MuJoCo SMPL articulation used by the HumEnv motion preset."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

from .smpl_constants import (
    SMPL_ARTICULATION_ROOT_PRIM_PATH,
    SMPL_NEWTON_VARIANT,
    SMPL_ROBOT_MJCF_PATH,
)

SMPL_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.MjcfFileCfg(
        asset_path=SMPL_ROBOT_MJCF_PATH,
        variants={"Physics": SMPL_NEWTON_VARIANT},
        self_collision=True,
        activate_contact_sensors=False,
    ),
    articulation_root_prim_path=SMPL_ARTICULATION_ROOT_PRIM_PATH,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={},
)
"""SMPL humanoid converted directly from its robot-only native MJCF.

The 24-body source owns all 69 affine actuators, passive joint terms, force
limits, contacts, and armature. The surrounding environment owns world geometry.
Episode resets replace the fallback pose.
"""
