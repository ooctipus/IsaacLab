# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Concrete SMPL simulation articulation and exact reference model."""

from __future__ import annotations

from isaaclab_newton.sim import NewtonMjcfFileCfg

from isaaclab_assets.robots.smpl.smpl_cfg import SMPL_HUMANOID_CFG
from isaaclab_assets.robots.smpl.smpl_constants import SMPL_ROBOT_MJCF_PATH

_ROBOT_PRIM_PATH = "{ENV_REGEX_NS}/Robot"


SMPL_MOTION_ARTICULATION_CFG = SMPL_HUMANOID_CFG.replace(
    prim_path=_ROBOT_PRIM_PATH,
    spawn=NewtonMjcfFileCfg(asset_path=SMPL_ROBOT_MJCF_PATH, self_collision=True),
    articulation_root_prim_path="/humanoid",
)
"""Packaged exact SMPL articulation whose native asset owns control and passive terms."""
