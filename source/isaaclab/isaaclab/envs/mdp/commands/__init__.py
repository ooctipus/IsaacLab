# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Various command terms that can be used in the environment."""

from isaaclab.utils.lazy_imports import lazy_export

lazy_export(
    ("commands_cfg", [
        "NullCommandCfg",
        "UniformVelocityCommandCfg",
        "NormalVelocityCommandCfg",
        "UniformPoseCommandCfg",
        "UniformPose2dCommandCfg",
        "TerrainBasedPose2dCommandCfg",
    ]),
    ("null_command", "NullCommand"),
    ("pose_2d_command", ["TerrainBasedPose2dCommand", "UniformPose2dCommand"]),
    ("pose_command", "UniformPoseCommand"),
    ("velocity_command", ["NormalVelocityCommand", "UniformVelocityCommand"]),
)
