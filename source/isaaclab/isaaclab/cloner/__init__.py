# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for environment cloning utilities."""

from isaaclab.utils.lazy_imports import lazy_export

lazy_export(
    ("cloner_cfg", "TemplateCloneCfg"),
    ("cloner_strategies", ["random", "sequential"]),
    ("cloner_utils", [
        "clone_from_template",
        "make_clone_plan",
        "usd_replicate",
        "filter_collisions",
        "grid_transforms",
    ]),
)
