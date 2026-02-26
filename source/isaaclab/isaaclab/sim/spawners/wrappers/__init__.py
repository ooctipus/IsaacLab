# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for wrapping spawner configurations.

Unlike the other spawner modules, this module provides a way to wrap multiple spawner configurations
into a single configuration. This is useful when the user wants to spawn multiple assets based on
different configurations.
"""

from isaaclab.utils.lazy_imports import lazy_export

lazy_export(
    ("wrappers", ["spawn_multi_asset", "spawn_multi_usd_file"]),
    ("wrappers_cfg", ["MultiAssetSpawnerCfg", "MultiUsdFileCfg"]),
)
