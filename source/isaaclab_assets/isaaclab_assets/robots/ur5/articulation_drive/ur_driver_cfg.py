# Copyright (c) 2024-2025, The Octi Lab Project Developers.
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

from __future__ import annotations

from dataclasses import MISSING
from typing import Callable

from isaaclab.utils import configclass
from octilab.assets.articulation.articulation_drive import ArticulationDriveCfg

from .ur_driver import URDriver


@configclass
class URDriverCfg(ArticulationDriveCfg):
    class_type: Callable[..., URDriver] = URDriver

    ip: str = MISSING  # type: ignore

    port: int = MISSING  # type: ignore
