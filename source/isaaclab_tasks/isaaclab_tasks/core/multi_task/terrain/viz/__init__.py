# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-task-specific visualisation utilities."""

from .sampler_images import (
    SpawnGoalSamplerImageLogger,
    log_spawn_goal_sampler_images,
    spawn_goal_scatter_image,
    terrain_success_heatmap_image,
    wandb_log_image,
)
from .terrain_background import heightmap_to_rgb, render_terrain_background
from .trajectory_recorder import TrajectoryRecorder

__all__ = [
    "SpawnGoalSamplerImageLogger",
    "TrajectoryRecorder",
    "heightmap_to_rgb",
    "log_spawn_goal_sampler_images",
    "render_terrain_background",
    "spawn_goal_scatter_image",
    "terrain_success_heatmap_image",
    "wandb_log_image",
]
