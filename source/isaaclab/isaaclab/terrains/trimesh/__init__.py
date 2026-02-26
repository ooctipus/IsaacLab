# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This sub-module provides methods to create different terrains using the ``trimesh`` library.

In contrast to the height-field representation, the trimesh representation does not
create arbitrarily small triangles. Instead, the terrain is represented as a single
tri-mesh primitive. Thus, this representation is more computationally and memory
efficient than the height-field representation, but it is not as flexible.
"""

from isaaclab.utils.lazy_imports import lazy_export

lazy_export(
    (
        "mesh_terrains_cfg",
        [
            "MeshBoxTerrainCfg",
            "MeshFloatingRingTerrainCfg",
            "MeshGapTerrainCfg",
            "MeshInvertedPyramidStairsTerrainCfg",
            "MeshPitTerrainCfg",
            "MeshPlaneTerrainCfg",
            "MeshPyramidStairsTerrainCfg",
            "MeshRailsTerrainCfg",
            "MeshRandomGridTerrainCfg",
            "MeshRepeatedBoxesTerrainCfg",
            "MeshRepeatedCylindersTerrainCfg",
            "MeshRepeatedPyramidsTerrainCfg",
            "MeshStarTerrainCfg",
        ],
    ),
    (
        "mesh_terrains",
        [
            "flat_terrain",
            "pyramid_stairs_terrain",
            "inverted_pyramid_stairs_terrain",
            "random_grid_terrain",
            "rails_terrain",
            "pit_terrain",
            "box_terrain",
            "gap_terrain",
            "floating_ring_terrain",
            "star_terrain",
            "repeated_objects_terrain",
        ],
    ),
)
