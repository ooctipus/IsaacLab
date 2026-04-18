# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-axis preset *classes* for the position locomotion task.

Every :class:`PresetCfg` declared here represents one robot-specific scalar
that the base environment (and rsl_rl agent) references inline instead of
hard-coding.  Per-robot modules in this package (e.g. :mod:`anymal_c`,
:mod:`go2`) populate their own field on these classes at import time, and the
preset resolver then substitutes the picked field at every consumption site.

To register a new robot, drop a new module alongside this one and set class
attributes such as::

    from .robot_presets import BaseBodyNameCfg, RobotArticulationCfg, ...

    RobotArticulationCfg.<robot> = <ROBOT>_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    BaseBodyNameCfg.<robot> = "..."
    ...

Only add fields; **never reassign** ``default`` -- it is the fallback used
when no robot preset is selected.  Package-level defaults preserve the
strings previously hard-coded in the base env / reward / termination configs
so the task still loads without a robot picked (except for
:class:`RobotArticulationCfg.default`, which is :data:`MISSING` and will fail
loudly if resolution happens with no robot selected).
"""

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg


@configclass
class RobotArticulationCfg(PresetCfg):
    """Full :class:`ArticulationCfg` for the robot asset at ``scene.robot``.

    Required -- no sensible default exists, so resolving without picking a
    robot preset leaves ``scene.robot`` as :data:`MISSING`.
    """

    default: ArticulationCfg = MISSING  # type: ignore[assignment]


@configclass
class HeightScannerPrimPathCfg(PresetCfg):
    """USD prim path anchoring the height-scanner ray caster."""

    default: str = "{ENV_REGEX_NS}/Robot/base"


@configclass
class BaseBodyNameCfg(PresetCfg):
    """Robot base / torso body name.

    Used by events that target the base mass (``add_base_mass``) and by the
    viewer (``viewer.body_name``).
    """

    default: str = "base"


@configclass
class BaseContactBodyNamesCfg(PresetCfg):
    """Body-name regex for bodies that must not contact the terrain.

    Drives the ``base_contact`` termination's ``sensor_cfg.body_names``.
    """

    default: str = "^(?!.*foot).*$"


@configclass
class FootBodyNamesCfg(PresetCfg):
    """Body-name regex matching the robot's feet.

    Drives reward terms that index foot bodies (e.g. ``foot_touchdown``).
    """

    default: str = ".*FOOT.*"


@configclass
class NonFootBodyNamesCfg(PresetCfg):
    """Body-name regex matching every body **except** the feet.

    Drives the ``undesired_contact`` reward's ``sensor_cfg.body_names``.
    """

    default: str = "^(?!.*(?:(FOOT))).*$"


@configclass
class AsyncFootPairsCfg(PresetCfg):
    """Foot pairs whose air/contact times should differ (gait diagnostic)."""

    default: tuple[tuple[str, str], ...] = ()


@configclass
class SyncFootPairsCfg(PresetCfg):
    """Foot pairs whose air/contact times should match (gait diagnostic)."""

    default: tuple[tuple[str, str], ...] = ()


@configclass
class ExperimentNameCfg(PresetCfg):
    """rsl_rl experiment name, picked up by ``PositionLocomotionPPORunnerCfg``."""

    default: str = "position_command"
