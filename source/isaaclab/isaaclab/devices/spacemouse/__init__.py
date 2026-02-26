# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spacemouse device for SE(2) and SE(3) control."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "se2_spacemouse": ["Se2SpaceMouse"],
        "se2_spacemouse_cfg": ["Se2SpaceMouseCfg"],
        "se3_spacemouse": ["Se3SpaceMouse"],
        "se3_spacemouse_cfg": ["Se3SpaceMouseCfg"],
    },
)
