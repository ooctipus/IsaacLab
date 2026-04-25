# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom Newton IK objectives for terrain-conforming retargeting.

Impl modules import ``newton`` at module load, which transitively pulls
``pxr``. The submodule-level ``cfg.py`` is pure dataclasses and must remain
importable during pre-Kit env-cfg construction — so this package defers
impl imports via ``lazy_export``.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
