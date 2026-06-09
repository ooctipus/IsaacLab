# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-axis preset *classes* for the Factory task family.

Every :class:`PresetCfg` declared here represents one robot-specific knob that
the base environment / scene references inline. Per-robot modules in this
package (e.g. :mod:`franka`) populate their own field on these classes at
import time, and the preset resolver then substitutes the picked field at
every consumption site.

To register a new robot, drop a new module alongside this one and set class
attributes such as::

    from .robot_presets import RobotArticulationCfg, RobotActionsCfg, ...

    RobotArticulationCfg.<robot> = <ROBOT>_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    RobotActionsCfg.<robot> = <RobotActionsCfg>()
    ...

``RobotArticulationCfg.default`` and ``RobotActionsCfg.default`` are
:data:`MISSING` and will fail loudly if resolution happens with no robot
selected.
"""

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg


@configclass
class RobotArticulationCfg(PresetCfg):
    """Full :class:`ArticulationCfg` for the robot asset at ``scene.robot``.

    Required -- no sensible default exists, so resolving without picking a
    robot preset leaves ``scene.robot`` as :data:`MISSING`.
    """

    default: ArticulationCfg = MISSING  # type: ignore[assignment]


@configclass
class RobotActionsCfg(PresetCfg):
    """Action term group bound to the active robot.

    Each robot preset assigns an :func:`isaaclab.utils.configclass`-decorated
    actions cfg (typically with ``arm_action`` and ``gripper_action`` fields)
    to a class attribute named after the robot.
    """

    default: object = MISSING  # type: ignore[assignment]


@configclass
class RobotContactSensorsCfg(PresetCfg):
    """Robot-specific contact sensors."""

    default: ContactSensorCfg = MISSING  # type: ignore[assignment]
