# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawners that spawn sensors in the simulation.

Currently, the following sensors are supported:

* Camera: A USD camera prim with settings for pinhole or fisheye projections.

"""

from isaaclab.utils.lazy_imports import lazy_export

lazy_export(
    ("sensors", "spawn_camera"),
    ("sensors_cfg", ["FisheyeCameraCfg", "PinholeCameraCfg"]),
)
