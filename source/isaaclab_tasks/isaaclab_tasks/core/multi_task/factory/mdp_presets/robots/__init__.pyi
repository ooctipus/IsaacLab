# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "RobotContactSensorsCfg",
    "RobotActionsCfg",
    "RobotArticulationCfg",
]

from .robot_presets import (
    RobotContactSensorsCfg,
    RobotActionsCfg,
    RobotArticulationCfg,
)

# Wildcard imports force eager module load so each robot's class-attribute
# registrations execute. Each robot module sets ``__all__ = []`` so these
# wildcards do not re-export any names.
from .franka import *
