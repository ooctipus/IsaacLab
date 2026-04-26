# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mewtwo humanoid preset. Activate with ``presets=mewtwo``.

No-op on branches that do not ship ``isaaclab_assets.robots.mewtwo`` -- the
preset registration is skipped and ``presets=mewtwo`` simply won't be
recognised there.

Behaviour note: the previous per-robot mixin removed ``events.add_base_mass``
because Mewtwo has no "base" body.  Here we instead point the event at the
``Pelvis`` body so base-mass randomisation stays active on the correct link.
"""

__all__: list[str] = []

from isaaclab.assets import ArticulationCfg

from .robot_presets import (
    AsyncFootPairsCfg,
    BaseBodyNameCfg,
    ExperimentNameCfg,
    HeightScannerPrimPathCfg,
    NonFootContactBodyNamesCfg,
    RobotArticulationCfg,
    SyncFootPairsCfg,
)

try:
    import isaaclab_assets.robots.mewtwo as mewtwo  # type: ignore[import-not-found]
except ImportError:
    mewtwo = None  # type: ignore[assignment]

if mewtwo is not None and hasattr(mewtwo, "MEWTWO_CFG"):
    _MEWTWO_CFG: ArticulationCfg = mewtwo.MEWTWO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    RobotArticulationCfg.mewtwo = _MEWTWO_CFG
    HeightScannerPrimPathCfg.mewtwo = "{ENV_REGEX_NS}/Robot/Pelvis"
    BaseBodyNameCfg.mewtwo = "Pelvis"
    NonFootContactBodyNamesCfg.mewtwo = "^(?!.*(?:Toe|Thumb|Index|Pinky|Coccyx.*)).*$"
    AsyncFootPairsCfg.mewtwo = (("RightToe", "LeftToe"),)
    SyncFootPairsCfg.mewtwo = ()
    ExperimentNameCfg.mewtwo = "mewtwo_position_command"
