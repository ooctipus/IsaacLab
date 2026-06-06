# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-terrain presets selectable via ``env.scene.terrain.terrain_generator.sub_terrains=<name>``."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from . import terrains
from .terrains.utils import CircleFootprintCfg, MorphologicalPatchSamplingCfg, RectFootprintCfg

ROBOT_FOOTPRINT = RectFootprintCfg(length=0.8, width=0.5)


def circle_patch(*, radius: float, num_patches: int = 20, max_height_diff: float = 0.2, **kwargs):
    """Build a circular morphological flat-patch sampler."""
    return MorphologicalPatchSamplingCfg(
        num_patches=num_patches,
        footprint=CircleFootprintCfg(radius=radius),
        max_height_diff=max_height_diff,
        **kwargs,
    )


def spawn_patch(*, num_patches: int = 20, max_height_diff: float = 0.2, **kwargs):
    """Build a robot-footprint morphological flat-patch sampler."""
    return MorphologicalPatchSamplingCfg(
        num_patches=num_patches,
        footprint=ROBOT_FOOTPRINT,
        max_height_diff=max_height_diff,
        **kwargs,
    )


def with_flat_patches(
    terrain_cfg,
    *,
    target_radius: float = 0.5,
    target_max_height_diff: float = 0.2,
    spawn_max_height_diff: float = 0.2,
):
    """Attach old flat-patch spawn/target samplers to a terrain config."""
    return terrain_cfg.replace(
        flat_patch_sampling={
            "target": circle_patch(radius=target_radius, max_height_diff=target_max_height_diff),
            "spawn": spawn_patch(max_height_diff=spawn_max_height_diff),
        },
    )


@configclass
class SubTerrainPresetCfg(PresetCfg):
    """Named sub-terrain configurations for the position locomotion task."""

    all = {
        "gap": terrains.GAP,
        "pit": terrains.PIT,
        "extreme_stair": terrains.EXTREME_STAIR,
        "slope_inv": terrains.SLOPE_INV,
        "stepping_stone": terrains.STEPPING_STONE,
        "radiating_beam": terrains.RADIATING_BEAM,
        "contour": terrains.CONTOUR,
        "climbing_box": terrains.CLIMBING_BOX,
        "floating_island": terrains.FLOATING_ISLAND,
        "maze": terrains.MAZE,
        "random_jump_box": terrains.RANDOM_JUMP_BOX,
        "random_parallel_box": terrains.RANDOM_PARALLEL_BOX,
        "balancing_beam": terrains.BALANCING_BEAM,
    }
    flat_patch_commands = {
        "gap": with_flat_patches(terrains.GAP),
        "pit": with_flat_patches(terrains.PIT),
        "extreme_stair": with_flat_patches(terrains.EXTREME_STAIR, target_radius=0.4),
        "slope_inv": with_flat_patches(terrains.SLOPE_INV),
        "stepping_stone": with_flat_patches(terrains.STEPPING_STONE, target_radius=0.01),
        "radiating_beam": with_flat_patches(terrains.RADIATING_BEAM, target_radius=0.25),
        "contour": with_flat_patches(terrains.CONTOUR, target_radius=0.15, target_max_height_diff=0.1),
        "climbing_box": with_flat_patches(terrains.CLIMBING_BOX, target_radius=0.05),
        "floating_island": with_flat_patches(terrains.FLOATING_ISLAND, target_radius=0.1),
        "maze": with_flat_patches(terrains.MAZE),
        "random_jump_box": with_flat_patches(terrains.RANDOM_JUMP_BOX, target_radius=0.1),
        "random_parallel_box": with_flat_patches(terrains.RANDOM_PARALLEL_BOX, target_radius=0.15),
        "balancing_beam": with_flat_patches(terrains.BALANCING_BEAM, target_radius=0.05, spawn_max_height_diff=0.05),
    }
    eval = {
        "gap": terrains.GAP.replace(gap_width_range=(1.0, 1.5)),
        "pit": terrains.PIT.replace(pit_depth_range=(0.8, 1.2)),
        "extreme_stair": terrains.EXTREME_STAIR.replace(step_height_range=(0.12, 0.2)),
        "slope_inv": terrains.SLOPE_INV.replace(slope_range=(0.6, 0.9)),
        "stepping_stone": terrains.STEPPING_STONE.replace(
            w_gap=(0.15, 0.26),
            w_stone=(0.4, 0.2),
            s_max=(0.080, 0.118),
            h_max=(0.075, 0.1),
        ),
        "radiating_beam": terrains.RADIATING_BEAM.replace(num_bars=(5, 1)),
    }
    gap = {"gap": terrains.GAP}
    pit = {"pit": terrains.PIT}
    extreme_stair = {"extreme_stair": terrains.EXTREME_STAIR}
    slope_inv = {"slope_inv": terrains.SLOPE_INV}
    square_pillar_obstacle = {"square_pillar_obstacle": terrains.SQUARE_PILLAR_OBSTACLE}
    stepping_stone = {"stepping_stone": terrains.STEPPING_STONE}
    radiating_beam = {"radiating_beam": terrains.RADIATING_BEAM}
    flat = {"flat": terrains.FLAT}
    maze = {"maze": terrains.MAZE}
    foot_sampled_commands = all
    default = all
