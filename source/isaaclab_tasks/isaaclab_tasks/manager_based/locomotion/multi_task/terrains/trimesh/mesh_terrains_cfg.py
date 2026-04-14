# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab import terrains
from isaaclab.utils import configclass

"""
Different trimesh terrain configurations.
"""


@configclass
class MeshObjTerrainCfg(terrains.SubTerrainBaseCfg):
    """Configuration for a plane mesh terrain."""

    function = "{DIR}.mesh_terrains:obj_terrain"

    obj_path: str = MISSING

    spawn_origin_path: str = MISSING


@configclass
class TerrainGenCfg(MeshObjTerrainCfg):
    """Configuration for a plane mesh terrain."""

    function = "{DIR}.mesh_terrains:terrain_gen"

    height: float = MISSING

    levels: float = MISSING

    include_overhang: bool = MISSING

    terrain_styles: list = MISSING

    yaml_path: str = (MISSING,)

    spawn_origin_path: str = MISSING

    python_script: str = MISSING


@configclass
class MeshStonesEverywhereTerrainCfg(terrains.SubTerrainBaseCfg):
    """A terrain with stones everywhere."""

    function = "{DIR}.mesh_terrains:stones_everywhere_terrain"

    w_gap: tuple[float, float] = MISSING
    """Stone gap width."""

    w_stone: tuple[float, float] = MISSING
    """Grid square stone size (width)."""

    s_max: tuple[float, float] = MISSING
    """Maximum shift, both x and y shift is uniformly sampled from [-s_max, s_max]."""

    h_max: tuple[float, float] = MISSING
    """Maximum height, uniformly sampled from [-h_max, h_max], default height is 1.0 m."""

    holes_depth: float = MISSING
    """Holes depth [m]."""

    platform_width: float = MISSING
    """Platform width [m]."""


@configclass
class MeshBalanceBeamsTerrainCfg(terrains.SubTerrainBaseCfg):
    """A terrain with balance-beams."""

    function = "{DIR}.mesh_terrains:balance_beams_terrain"

    platform_width: float = MISSING
    """Platform width [m]."""

    h_offset: tuple[float, float] = MISSING
    """Height offset [m]."""

    w_stone: tuple[float, float] = MISSING
    """Stone width [m]."""

    mid_gap: tuple[float, float] = MISSING
    """Gap between two beams [m]."""


@configclass
class MeshSteppingBeamsTerrainCfg(terrains.SubTerrainBaseCfg):
    """A terrain with stepping-beams."""

    function = "{DIR}.mesh_terrains:stepping_beams_terrain"

    platform_width: float = MISSING
    """Platform width [m]."""

    h_offset: tuple[float, float] = MISSING
    """Height offset [m]."""

    w_stone: tuple[float, float] = MISSING
    """Stone width [m]."""

    l_stone: tuple[float, float] = MISSING
    """Length of the stepping beams [m]."""

    gap: tuple[float, float] = MISSING
    """Gap between two beams [m]."""

    yaw: tuple[float, float] = MISSING
    """Yaw angle of the stepping beams [rad]."""


@configclass
class MeshDiversityBoxTerrainCfg(terrains.SubTerrainBaseCfg):
    """A terrain with boxes for anymal parkour."""

    function = "{DIR}.mesh_terrains:box_terrain"

    box_width_range: tuple[float, float] = MISSING
    """Box width range [m]."""

    box_length_range: tuple[float, float] = MISSING
    """Box length range [m]."""

    box_height_range: tuple[float, float] = MISSING
    """Box height range [m]."""

    box_gap_range: tuple[float, float] = None  # type: ignore
    """Gap between two boxes [m]."""

    up_or_down: str = None  # type: ignore
    """Flag for climbing up (box at origin) or down (box near origin)."""


@configclass
class MeshPassageTerrainCfg(terrains.SubTerrainBaseCfg):
    """A terrain with passage."""

    function = "{DIR}.mesh_terrains:passage_terrain"

    passage_width: float | tuple[float, float] = MISSING
    """Passage width (y direction) [m]."""

    passage_height: float | tuple[float, float] = MISSING
    """Passage height [m]."""

    passage_length: float | tuple[float, float] = MISSING
    """Passage length (x direction) [m]."""


@configclass
class MeshStructuredTerrainCfg(terrains.SubTerrainBaseCfg):
    """Configuration for a structured terrain."""

    function = "{DIR}.mesh_terrains:structured_terrain"
    terrain_type: Literal["stairs", "inverted_stairs", "obstacles", "walls"] = MISSING


@configclass
class MeshRadiatingBeamTerrainCfg(terrains.MeshStarTerrainCfg):
    """A terrain that creates beam bridges connecting a central cylindrical platform to the outer border.

    Improved upon :class:`isaaclab.terrains.MeshStarTerrainCfg`:
    1. Adds ``border_size`` parameter for user-specified terrain border.
    2. Allows generating an odd number of beams.
    """

    function = "{DIR}.mesh_terrains:beam_terrain"

    border_size: tuple[float, float] = MISSING
    """Inner length (x) and width (y) defining the inner square area enclosed by the border [m]."""

    num_bars: tuple[int, int] = MISSING
    """Range of number of beams."""

    beam_distribution: Literal["uniform", "random"] = MISSING
    """Distribution pattern of sampling beams."""
