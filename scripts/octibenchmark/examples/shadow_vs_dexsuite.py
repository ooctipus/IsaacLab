# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Comparison benchmark: Shadow Hand Vision vs Dexsuite single-camera at 4096 envs.

Both tasks use the Newton renderer for a fair FPS comparison.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from octibenchmark.bench_cfg import BenchmarkMatrix, Launcher

# Shadow Hand Vision — newton backend + newton renderer, RGB
SHADOW_VISION = BenchmarkMatrix(
    tasks=["Isaac-Repose-Cube-Shadow-Vision-Direct-v0"],
    num_envs=[4096],
    hydra_args=["presets=newton,newton_renderer,rgb"],
    launcher=Launcher.NON_RL,
    num_frames=100,
    warmup_frames=10,
)

# Dexsuite Kuka Allegro Lift — single camera, newton renderer
DEXSUITE_SINGLE_CAM = BenchmarkMatrix(
    tasks=["Isaac-Dexsuite-Kuka-Allegro-Lift-v0"],
    num_envs=[4096],
    hydra_args=["presets=newton,cube,single_camera,newton_renderer"],
    launcher=Launcher.NON_RL,
    num_frames=100,
    warmup_frames=10,
)

ALL_MATRICES = {
    "shadow_vision": SHADOW_VISION,
    "dexsuite_single_cam": DEXSUITE_SINGLE_CAM,
}
