# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pink IK controller package for IsaacLab.

This package provides integration between Pink inverse kinematics solver and IsaacLab.
"""

from isaaclab.utils.module import attach_cascading

__getattr__, __dir__ = attach_cascading(
    __name__,
    submodules=["null_space_posture_task", "pink_ik", "pink_ik_cfg"],
)


__all__ = ["NullSpacePostureTask", "PinkIKController", "PinkIKControllerCfg", "PinkIKTaskCfg"]
