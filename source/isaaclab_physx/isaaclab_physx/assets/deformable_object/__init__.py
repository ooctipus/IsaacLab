# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for deformable object assets."""

from isaaclab.utils.lazy_imports import lazy_export

lazy_export(
    ("deformable_object", "DeformableObject"),
    ("deformable_object_cfg", "DeformableObjectCfg"),
    ("deformable_object_data", "DeformableObjectData"),
)
