# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab import terrains
from isaaclab.utils import configclass

from . import mesh_terrains

"""
Different trimesh terrain configurations.
"""


@configclass
class MeshObjTerrainCfg(terrains.SubTerrainBaseCfg):
    """Configuration for a plane mesh terrain."""

    function = mesh_terrains.obj_terrain

    obj_path: str = MISSING

    spawn_origin_path: str = MISSING


@configclass
class TerrainGenCfg(MeshObjTerrainCfg):
    """Configuration for a plane mesh terrain."""

    function = mesh_terrains.terrain_gen

    height: float = MISSING

    levels: float = MISSING

    include_overhang: bool = MISSING

    terrain_styles: list = MISSING

    yaml_path: str = (MISSING,)

    spawn_origin_path: str = MISSING

    python_script: str = MISSING


@configclass
class MeshStonesEverywhereTerrainCfg(terrains.SubTerrainBaseCfg):
    """
    A terrain with stones everywhere
    """

    function = mesh_terrains.stones_everywhere_terrain

    # stone gap width
    w_gap: tuple[float, float] = MISSING

    # grid square stone size (width)
    w_stone: tuple[float, float] = MISSING

    # the maximum shift, both x and y shift is uniformly sample from [-s_max, s_max]
    s_max: tuple[float, float] = MISSING

    # the maximum height, the height is uniformly sample from [-hmax, h_max], default height is 1.0 m
    h_max: tuple[float, float] = MISSING

    # holes depth
    holes_depth: float = MISSING

    # the platform width
    platform_width: float = MISSING


@configclass
class MeshBalanceBeamsTerrainCfg(terrains.SubTerrainBaseCfg):
    """
    A terrain with balance-beams
    """

    # balance beams terrain function
    function = mesh_terrains.balance_beams_terrain

    # the platform width
    platform_width: float = MISSING

    # the height offset
    h_offset: tuple[float, float] = MISSING

    # stone width
    w_stone: tuple[float, float] = MISSING

    # the gap between two beams
    mid_gap: tuple[float, float] = MISSING


@configclass
class MeshSteppingBeamsTerrainCfg(terrains.SubTerrainBaseCfg):
    """
    A terrain with stepping-beams
    """

    # stepping beams terrain function
    function = mesh_terrains.stepping_beams_terrain

    # the platform width
    platform_width: float = MISSING

    # the height offset
    h_offset: tuple[float, float] = MISSING

    # stone width
    w_stone: tuple[float, float] = MISSING

    # length of the stepping beams
    l_stone: tuple[float, float] = MISSING

    #  the gap between two beams
    gap: tuple[float, float] = MISSING

    # the yaw angle of the stepping beams
    yaw: tuple[float, float] = MISSING


@configclass
class MeshDiversityBoxTerrainCfg(terrains.SubTerrainBaseCfg):
    """
    A terrain with boxes for anymal parkour
    """

    function = mesh_terrains.box_terrain

    # the box width range
    box_width_range: tuple[float, float] = MISSING
    # the box length range
    box_length_range: tuple[float, float] = MISSING
    # the box height range
    box_height_range: tuple[float, float] = MISSING

    # the gap between two boxes
    box_gap_range: tuple[float, float] = None  # type: ignore

    # flag for climbing up (box is set at the origin ) or climb down (box is set near the origin)
    up_or_down: str = None  # type: ignore


@configclass
class MeshPassageTerrainCfg(terrains.SubTerrainBaseCfg):
    """
    A terrain with passage
    """

    function = mesh_terrains.passage_terrain

    # the passage width (y dir)
    passage_width: float | tuple[float, float] = MISSING

    # the passage height
    passage_height: float | tuple[float, float] = MISSING

    # the passage length (x dir)
    passage_length: float | tuple[float, float] = MISSING


@configclass
class MeshStructuredTerrainCfg(terrains.SubTerrainBaseCfg):
    """Configuration for a structured terrain."""

    function = mesh_terrains.structured_terrain
    terrain_type: Literal["stairs", "inverted_stairs", "obstacles", "walls"] = MISSING


@configclass
class MeshRadiatingBeamTerrainCfg(terrains.MeshStarTerrainCfg):
    """
    A terrain that creates beam bridges connecting a central cylindrical platform to the outer border. Improved
    upon the :class:`isaaclab.terrains.MeshStarTerrainCfg`:
    1. Add parameter border_size to allow user specify the border of the terrain
    2. Allow generating odd number of beams 
    """

    function = mesh_terrains.beam_terrain

    # The inner length (x) and width (y) defining the inner square area enclosed by the border (in m).
    border_size: tuple[float, float] = MISSING
