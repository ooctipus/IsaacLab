# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.terrains.terrain_generator as terrain_generator
from isaaclab import terrains as terrain_cfg

from . import trimesh as octilab_terrain
from .utils import FlatPatchSamplingByRadiusCfg, PatchSamplingCfg


def patched_find_flat_patches(*args, **kwargs) -> None:
    patch_key = "patch_radius"
    kwargs[patch_key]["patched"] = True
    cfg_class = kwargs[patch_key]["cfg"]
    cfg_class_args = {key: val for key, val in kwargs[patch_key].items() if key not in ["func", "cfg"]}
    patch_sampling_cfg: PatchSamplingCfg = cfg_class(**cfg_class_args)
    return patch_sampling_cfg.func(kwargs["wp_mesh"], kwargs["origin"], patch_sampling_cfg)


terrain_generator.find_flat_patches = patched_find_flat_patches


GAP = terrain_cfg.MeshGapTerrainCfg(
    platform_width=3.0,
    gap_width_range=(0.05, 1.5),
    flat_patch_sampling={
        "spawn": FlatPatchSamplingByRadiusCfg(
            num_patches=10,
            patch_radius=0.5,
            radius_range=(0.5, 7.0),
            max_height_diff=0.2,
        ),
        "target": FlatPatchSamplingByRadiusCfg(
            num_patches=10,
            patch_radius=0.5,
            radius_range=(0.5, 7.0),
            max_height_diff=0.2,
        )
    },
)

PIT = terrain_cfg.MeshPitTerrainCfg(
    platform_width=3.0,
    pit_depth_range=(0.05, 1.2),
    flat_patch_sampling={
        "spawn": FlatPatchSamplingByRadiusCfg(
            num_patches=10,
            patch_radius=0.5,
            max_height_diff=0.2,
            radius_range=(0.5, 7.0),
        ),
        "target": FlatPatchSamplingByRadiusCfg(
            num_patches=10,
            patch_radius=0.5,
            max_height_diff=0.2,
            radius_range=(0.5, 7.0),
        )
    },
)

RADIATING_BEAM = octilab_terrain.MeshRadiatingBeamTerrainCfg(
    platform_width=3.0,
    num_bars=12,
    border_size=(6.5, 6.5),
    bar_width_range=(0.7, 0.3),
    bar_height_range=(1.5, 1.5),
    flat_patch_sampling={
        "spawn": FlatPatchSamplingByRadiusCfg(
            num_patches=10, patch_radius=0.4, radius_range=(0.2, 10.0), max_height_diff=0.2, z_range=(-1, 1)
        ),
        "target": FlatPatchSamplingByRadiusCfg(
            num_patches=10, patch_radius=0.05, radius_range=(0.2, 10.0), max_height_diff=0.2, z_range=(-1, 1)
        )
    },
)

SQUARE_PILLAR_OBSTACLE = terrain_cfg.HfDiscreteObstaclesTerrainCfg(
    num_obstacles=35,
    obstacle_height_mode="fixed",
    obstacle_width_range=(0.25, 0.75),
    obstacle_height_range=(1.0, 2.0),
    platform_width=0.5,
    flat_patch_sampling={
        "spawn": FlatPatchSamplingByRadiusCfg(
            num_patches=10,
            patch_radius=0.2,
            max_height_diff=0.2,
            radius_range=(0.2, 7.0),
        ),
        "target": FlatPatchSamplingByRadiusCfg(
            num_patches=10,
            patch_radius=0.2,
            max_height_diff=0.2,
            radius_range=(0.2, 7.0),
        )
    },
)

IRREGULAR_PILLAR_OBSTACLE = terrain_cfg.MeshRepeatedBoxesTerrainCfg(
    platform_width=1.0,
    max_height_noise=0.5,
    object_params_start=terrain_cfg.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        num_objects=5, height=4.0, size=(0.5, 0.5), max_yx_angle=0.0, degrees=True
    ),
    object_params_end=terrain_cfg.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        num_objects=10, height=6.0, size=(1.0, 1.0), max_yx_angle=0.0, degrees=True
    ),
    flat_patch_sampling={
        "spawn": FlatPatchSamplingByRadiusCfg(
            num_patches=10,
            patch_radius=0.5,
            max_height_diff=0.2,
            radius_range=(0.2, 7.0),
        ),
        "target": FlatPatchSamplingByRadiusCfg(
            num_patches=10,
            patch_radius=0.5,
            max_height_diff=0.2,
            radius_range=(0.2, 7.0),
        )
    },
)

SLOPE_INV = terrain_cfg.HfInvertedPyramidSlopedTerrainCfg(
    slope_range=(0.0, 0.9),
    platform_width=2.0,
    border_width=1.5,
    flat_patch_sampling={
        "spawn": FlatPatchSamplingByRadiusCfg(
            num_patches=10,
            patch_radius=0.5,
            max_height_diff=0.2,
            radius_range=(0.2, 7.0),
        ),
        "target": FlatPatchSamplingByRadiusCfg(
            num_patches=10,
            patch_radius=0.5,
            max_height_diff=0.2,
            radius_range=(0.2, 7.0),
        )
    },
)

EXTREME_STAIR = terrain_cfg.HfPyramidStairsTerrainCfg(
    platform_width=3.0,
    step_height_range=(0.05, 0.2),
    step_width=0.3,
    inverted=True,
    border_width=1.0,
    flat_patch_sampling={
        "spawn": FlatPatchSamplingByRadiusCfg(
            num_patches=10,
            patch_radius=0.4,
            max_height_diff=0.2,
            radius_range=(0.0, 7.0),
        ),
        "target": FlatPatchSamplingByRadiusCfg(
            num_patches=10,
            patch_radius=0.4,
            max_height_diff=0.2,
            radius_range=(0.0, 7.0),
        )
    },
)


STEPPING_STONE = octilab_terrain.MeshStonesEverywhereTerrainCfg(
    w_gap=(0.04, 0.26),
    w_stone=(0.96, 0.2),
    s_max=(0.018, 0.118),
    h_max=(0.005, 0.1),
    holes_depth=-10.0,
    platform_width=1.5,
    flat_patch_sampling={
        "spawn": FlatPatchSamplingByRadiusCfg(
            num_patches=10,
            patch_radius=0.4,
            max_height_diff=0.2,
            radius_range=(0.0, 7.0),
        ),
        "target": FlatPatchSamplingByRadiusCfg(
            num_patches=10,
            patch_radius=0.05,
            max_height_diff=0.2,
            radius_range=(2.5, 6.5),
        )
    },
)

BALANCING_BEAM = octilab_terrain.MeshBalanceBeamsTerrainCfg(
    platform_width=2.0,
    h_offset=(0.01, 0.1),
    w_stone=(0.25, 0.25),
    mid_gap=(0.25, 0.25),
    flat_patch_sampling={
        "target": FlatPatchSamplingByRadiusCfg(
            patch_radius=0.4,
            num_patches=10,
            x_range=(4, 6),
            y_range=(-1, 1),
            z_range=(-0.05, 0.05),
            max_height_diff=0.05,
        )
    },
)

NARROW_BEAM = octilab_terrain.MeshSteppingBeamsTerrainCfg(
    platform_width=2.0,
    h_offset=(0.01, 0.1),
    w_stone=(0.5, 0.2),
    l_stone=(0.8, 1.6),
    gap=(0.15, 0.5),
    yaw=(0, 15),
    flat_patch_sampling={
        "target": FlatPatchSamplingByRadiusCfg(
            patch_radius=0.4,
            num_patches=10,
            x_range=(4, 6),
            y_range=(-1, 1),
            z_range=(-0.05, 0.05),
            max_height_diff=0.05,
        )
    },
)
