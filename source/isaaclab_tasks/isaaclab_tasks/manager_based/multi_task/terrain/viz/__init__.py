# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-task-specific visualisation utilities."""

from .terrain_background import heightmap_to_rgb, render_terrain_background
from .trajectory_recorder import TrajectoryRecorder

__all__ = ["TrajectoryRecorder", "heightmap_to_rgb", "render_terrain_background"]
