# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-based kinematics: model wrapper and IK objectives.

The module implementations (``newton_kinematics``, ``ik_objectives``) pull in
``newton`` → ``pxr``, which must not be loaded before Kit has launched during
env-cfg construction. Public symbols are exposed lazily via ``lazy_export``
(backed by ``__init__.pyi``) so importing this package does not trigger the
Newton → USD chain until a symbol is first accessed.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
