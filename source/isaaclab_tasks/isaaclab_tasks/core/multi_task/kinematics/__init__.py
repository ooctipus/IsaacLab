# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-free Newton kinematics, topology, and IK objectives.

Public symbols are exposed lazily through ``__init__.pyi`` so configuration
imports do not pay for Newton/OpenUSD construction until a table builder needs
it. Newton and OpenUSD remain valid without Kit or a simulation context.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
