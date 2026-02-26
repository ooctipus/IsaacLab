# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for different assets."""

from isaaclab.utils.lazy_imports import lazy_export

lazy_export(
    ("gelsight", ["GELSIGHT_R15_CFG", "GELSIGHT_MINI_CFG"]),
    ("velodyne", "VELODYNE_VLP_16_RAYCASTER_CFG"),
)
