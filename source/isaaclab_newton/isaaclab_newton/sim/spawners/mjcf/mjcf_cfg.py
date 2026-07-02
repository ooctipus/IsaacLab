# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils.configclass import configclass


@configclass
class NewtonMjcfFileCfg(SpawnerCfg):
    """Configuration for loading MJCF directly into a Newton model builder.

    The spawner authors only an inspectable USD marker. Newton consumes that
    marker and parses :attr:`asset_path` natively when it builds the simulation
    model; no intermediate USD conversion or Kit application is required.
    """

    func: Callable | str = "{DIR}.mjcf:spawn_newton_mjcf"

    asset_path: str = MISSING
    """Local MJCF file loaded by Newton."""

    self_collision: bool = True
    """Whether bodies from this MJCF may collide with one another."""
