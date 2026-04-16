# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-terrain presets selectable via ``env.scene.terrain.terrain_generator.sub_terrains=<name>``."""

from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

from . import terrains


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
        "climbing_box":terrains.CLIMBING_BOX,
        "floating_island":terrains.FLOATING_ISLAND,
        "maze": terrains.MAZE,
        "random_jump_box": terrains.RANDOM_JUMP_BOX,
        "random_parallel_box":terrains.RANDOM_PARALLEL_BOX,
        "balancing_beam":terrains.BALANCING_BEAM
    }
    eval = {
        "gap": terrains.GAP.replace(gap_width_range=(1.0, 1.5)),
        "pit": terrains.PIT.replace(pit_depth_range=(0.8, 1.2)),
        "extreme_stair": terrains.EXTREME_STAIR.replace(step_height_range=(0.12, 0.2)),
        "slope_inv": terrains.SLOPE_INV.replace(slope_range=(0.6, 0.9)),
        "stepping_stone": terrains.STEPPING_STONE.replace(
            w_gap=(0.15, 0.26), w_stone=(0.4, 0.2), s_max=(0.080, 0.118), h_max=(0.075, 0.1),
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
    default = all