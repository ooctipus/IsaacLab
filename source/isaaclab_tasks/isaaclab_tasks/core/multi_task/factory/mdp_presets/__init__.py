# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preset registry for the Factory task family.

Contents are declared in :file:`__init__.pyi`. The wildcard imports there force
eager loading of each robot's preset module so its class-attribute
registrations execute before the preset resolver runs.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
