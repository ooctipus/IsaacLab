# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the pick and place environments."""

from isaaclab.utils.module import attach_cascading

__getattr__, __dir__ = attach_cascading(
    __name__,
    submodules=["observations", "terminations"],
    packages=["isaaclab.envs.mdp"],
)
