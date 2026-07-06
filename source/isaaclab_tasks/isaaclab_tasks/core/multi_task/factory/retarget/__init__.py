# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-free Factory task-table configuration.

Declared families generate held/fixed asset evidence, solve a flat objective
tuple, apply independent criteria, and select exact per-board quotas. Newton
construction remains behind the task-table builder invocation.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
