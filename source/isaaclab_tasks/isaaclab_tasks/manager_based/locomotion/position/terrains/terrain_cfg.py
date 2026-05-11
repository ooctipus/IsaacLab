# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab import terrains as terrain_cfg

from . import trimesh as isaaclab_terrain
from .utils import (
    CircleFootprintCfg,
    MorphologicalPatchSamplingCfg,
    RectFootprintCfg,
)

ROBOT_FOOTPRINT = RectFootprintCfg(length=0.8, width=0.5)
"""Default robot footprint for spawn patches. Sized for Anymal C (~80 cm long x 50 cm wide).

``length`` = nose-to-tail (+x forward), ``width`` = shoulder-to-shoulder (+y lateral).
Override this in robot-specific terrain configs if using a different platform.
"""


def circle_patch(*, radius: float, num_patches: int = 20, max_height_diff: float = 0.2, **kw):
    """Build a :class:`MorphologicalPatchSamplingCfg` with a circular footprint."""
    return MorphologicalPatchSamplingCfg(
        num_patches=num_patches,
        footprint=CircleFootprintCfg(radius=radius),
        max_height_diff=max_height_diff,
        **kw,
    )


def spawn_patch(*, num_patches: int = 20, max_height_diff: float = 0.2, **kw):
    """Build a :class:`MorphologicalPatchSamplingCfg` using :data:`ROBOT_FOOTPRINT`."""
    return MorphologicalPatchSamplingCfg(
        num_patches=num_patches,
        footprint=ROBOT_FOOTPRINT,
        max_height_diff=max_height_diff,
        **kw,
    )


FLAT = terrain_cfg.MeshPlaneTerrainCfg(
    flat_patch_sampling={
        "target": circle_patch(radius=0.5),
        "spawn": spawn_patch(),
    },
)


GAP = terrain_cfg.MeshGapTerrainCfg(
    platform_width=3.0,
    gap_width_range=(0.05, 1.5),
    flat_patch_sampling={
        "target": circle_patch(radius=0.5),
        "spawn": spawn_patch(),
    },
)

PIT = terrain_cfg.MeshPitTerrainCfg(
    platform_width=3.0,
    pit_depth_range=(0.05, 1.2),
    flat_patch_sampling={
        "target": circle_patch(radius=0.5),
        "spawn": spawn_patch(),
    },
)

SQUARE_PILLAR_OBSTACLE = terrain_cfg.HfDiscreteObstaclesTerrainCfg(
    num_obstacles=35,
    obstacle_height_mode="fixed",
    obstacle_width_range=(0.25, 0.75),
    obstacle_height_range=(1.0, 2.0),
    platform_width=0.5,
    flat_patch_sampling={
        "target": circle_patch(radius=0.2),
        "spawn": spawn_patch(),
    },
)

MAZE = isaaclab_terrain.MeshMazeTerrainCfg(
    grid_rows=(5, 5),
    grid_cols=(5, 5),
    wall_thickness=0.3,
    wall_height=(2.5, 3.5),
    open_ratio=0.3,
    grid_noise=0.3,
    flat_patch_sampling={
        "target": circle_patch(radius=0.5),
        "spawn": spawn_patch(),
    },
)


CONTOUR = isaaclab_terrain.MeshContourTerrainCfg(
    num_levels=(6, 12),
    level_height=(0.25, 0.25),
    noise_scale=3.0,
    noise_octaves=10,
    smoothing=0.3,
    stones=isaaclab_terrain.MeshContourTerrainCfg.StoneCfg(
        num_stones=(8, 20),
        size_range=(0.6, 1.2),
        height_scale=0.6,
        roughness=0.3,
    ),
    flat_patch_sampling={
        "target": circle_patch(radius=0.15, max_height_diff=0.1),
        "spawn": spawn_patch(max_height_diff=0.1),
    },
)

SLOPE_INV = terrain_cfg.HfInvertedPyramidSlopedTerrainCfg(
    slope_range=(0.0, 0.9),
    platform_width=2.0,
    border_width=1.5,
    flat_patch_sampling={
        "target": circle_patch(radius=0.5),
        "spawn": spawn_patch(),
    },
)

EXTREME_STAIR = terrain_cfg.HfPyramidStairsTerrainCfg(
    platform_width=3.0,
    step_height_range=(0.05, 0.2),
    step_width=0.3,
    inverted=True,
    border_width=1.0,
    flat_patch_sampling={
        "target": circle_patch(radius=0.4),
        "spawn": spawn_patch(),
    },
)


STEPPING_STONE = isaaclab_terrain.MeshStonesEverywhereTerrainCfg(
    w_gap=(0.04, 0.26),
    w_stone=(0.96, 0.2),
    s_max=(0.018, 0.118),
    h_max=(0.005, 0.1),
    holes_depth=-10.0,
    platform_width=1.5,
    flat_patch_sampling={
        "target": circle_patch(radius=0.01),
        "spawn": spawn_patch(),
    },
)

BALANCING_BEAM = isaaclab_terrain.MeshBalanceBeamsTerrainCfg(
    platform_width=2.0,
    h_offset=(0.01, 0.1),
    w_stone=(0.25, 0.25),
    mid_gap=(0.25, 0.25),
    flat_patch_sampling={
        "spawn": spawn_patch(max_height_diff=0.05, x_range=(4, 6), y_range=(-1, 1), z_range=(-0.05, 0.05)),
    },
)

NARROW_BEAM = isaaclab_terrain.MeshSteppingBeamsTerrainCfg(
    platform_width=2.0,
    h_offset=(0.01, 0.1),
    w_stone=(0.5, 0.2),
    l_stone=(0.8, 1.6),
    gap=(0.15, 0.5),
    yaw=(0, 15),
    flat_patch_sampling={
        "spawn": spawn_patch(max_height_diff=0.05, x_range=(4, 6), y_range=(-1, 1), z_range=(-0.05, 0.05)),
    },
)

RADIATING_BEAM = isaaclab_terrain.MeshRadiatingBeamTerrainCfg(
    platform_width=2.25,
    platform_height=(0.0, 0.25),
    num_bars=(12, 1),
    beam_distribution="random",
    border=isaaclab_terrain.MeshRadiatingBeamTerrainCfg.SquareBorderCfg(inner_size=(7.25, 7.25)),
    bar_width_range=(0.7, 0.4),
    bar_height_range=(1.5, 1.5),
    ground_plane=False,
    flat_patch_sampling={
        "target": circle_patch(radius=0.25),
        "spawn": spawn_patch(),
    },
)


ELEVATING_BEAM = isaaclab_terrain.MeshRadiatingBeamTerrainCfg(
    platform_width=2.5,
    platform_height=(0.0, 0.75),
    num_bars=(3, 3),
    beam_distribution="random",
    border=isaaclab_terrain.MeshRadiatingBeamTerrainCfg.SquareBorderCfg(inner_size=(7.0, 7.0)),
    bar_width_range=(0.7, 0.7),
    bar_height_range=(1.5, 1.5),
    ground_plane=False,
    flat_patch_sampling={
        "target": circle_patch(radius=0.25),
        "spawn": spawn_patch(),
    },
)

RADIATING_BEAM_BOX = isaaclab_terrain.MeshRadiatingBeamTerrainCfg(
    platform_width=2.25,
    platform_height_noise=1.0,
    num_bars=(2, 2),
    beam_distribution="random",
    border=isaaclab_terrain.MeshRadiatingBeamTerrainCfg.PlatformBorderCfg(inner_size=(8.5, 8.5), radius=1.0),
    bar_width_range=(0.8, 0.45),
    bar_height_range=(1.5, 1.5),
    ground_plane=False,
    beam_style=isaaclab_terrain.MeshRadiatingBeamTerrainCfg.BoxBeamCfg(
        box_length=(0.35, 0.2),
        box_gap=0.02,
        box_position_variation=(0.05, 0.05, 0.03),
    ),
    flat_patch_sampling={
        "target": circle_patch(radius=0.05),
        "spawn": spawn_patch(),
    },
)


RANDOM_JUMP_BOX = isaaclab_terrain.MeshRadiatingBeamTerrainCfg(
    platform_width=2.0,
    platform_height_noise=1.5,
    num_bars=(8, 2),
    beam_distribution="uniform",
    border=isaaclab_terrain.MeshRadiatingBeamTerrainCfg.PlatformBorderCfg(inner_size=(7.5, 7.5), radius=1.0),
    bar_width_range=(2.0, 2.0),
    bar_height_range=(1.5, 1.5),
    ground_plane=False,
    beam_style=isaaclab_terrain.MeshRadiatingBeamTerrainCfg.BoxBeamCfg(
        box_length=(1.75, 1.75), box_gap=0.5, box_position_variation=(0.2, 0.2, 0.2), box_yaw_variation=1.57
    ),
    flat_patch_sampling={
        "target": circle_patch(radius=0.1),
        "spawn": spawn_patch(),
    },
)

RANDOM_PARALLEL_BOX = isaaclab_terrain.MeshRadiatingBeamTerrainCfg(
    platform_width=1.5,
    platform_height_noise=1.5,
    num_bars=(6, 3),
    beam_distribution="uniform",
    border=None,
    bar_width_range=(3.5, 2.5),
    bar_height_range=(1.5, 1.5),
    ground_plane=False,
    beam_style=isaaclab_terrain.MeshRadiatingBeamTerrainCfg.BoxBeamCfg(
        box_length=(1.1, 1.1), box_gap=0.4, box_position_variation=(0.4, 0.4, 0.2), box_yaw_variation=1.57
    ),
    flat_patch_sampling={
        "target": circle_patch(radius=0.1),
        "spawn": spawn_patch(),
    },
)


CLIMBING_BOX = isaaclab_terrain.MeshRadiatingBeamTerrainCfg(
    platform_width=1.5,
    platform_height=2.5,
    platform_height_noise=1.0,
    num_bars=(5, 3),
    beam_distribution="uniform",
    border=None,
    bar_width_range=(5.5, 5.5),
    bar_height_range=(1.5, 1.5),
    ground_plane=False,
    beam_style=isaaclab_terrain.MeshRadiatingBeamTerrainCfg.BoxBeamCfg(
        box_length=(1.1, 1.1), box_gap=0.4, box_position_variation=(0.7, 0.7, 0.4), box_yaw_variation=1.57
    ),
    flat_patch_sampling={
        "target": circle_patch(radius=0.15, max_height_diff=0.3),
        "spawn": spawn_patch(max_height_diff=0.3),
    },
)


FLOATING_ISLAND = isaaclab_terrain.MeshFloatingIslandTerrainCfg(
    num_islands=(8, 5),
    island_style=isaaclab_terrain.MeshFloatingIslandTerrainCfg.CylinderIslandCfg(radius=(0.9, 0.9)),
    island_height=1.0,
    island_height_variation=2.0,
    island_margin=0.75,
    graph=isaaclab_terrain.MeshFloatingIslandTerrainCfg.DelaunayGraphCfg(),
    passway_style=isaaclab_terrain.MeshFloatingIslandTerrainCfg.BoxBeamCfg(
        box_length=(0.4, 0.25),
        box_gap=0.0,
        box_position_variation=(0.05, 0.05, 0.05),
    ),
    passway_width=0.5,
    passway_curvature=0.3,
    flat_patch_sampling={
        "target": circle_patch(radius=0.05, max_height_diff=0.3),
        "spawn": spawn_patch(max_height_diff=0.3),
    },
)
