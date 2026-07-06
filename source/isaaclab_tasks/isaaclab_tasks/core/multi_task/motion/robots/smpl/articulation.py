# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Concrete SMPL simulation articulation and exact reference model."""

from __future__ import annotations

import re

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


def smpl_live_joint_mujoco_names(live_joint_names: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve live articulation joint labels to packaged MuJoCo coordinate names."""
    mujoco_names: list[str] = []
    for name in live_joint_names:
        joint_name, separator, component = name.rpartition(":")
        match = re.fullmatch(r"(.+)_x_\1_y_\1_z", joint_name)
        if not separator or match is None or component not in ("0", "1", "2"):
            raise ValueError("SMPL live joints must use native Body_x_Body_y_Body_z:0/1/2 coordinate labels.")
        mujoco_names.append(f"{match.group(1)}_{'xyz'[int(component)]}")
    resolved = tuple(mujoco_names)
    if len(set(resolved)) != len(resolved):
        raise ValueError("SMPL live joints do not resolve to unique MuJoCo coordinates.")
    return resolved
