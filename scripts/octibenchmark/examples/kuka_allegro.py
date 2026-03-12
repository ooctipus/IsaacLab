# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark matrices for Kuka Allegro Lift task."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from octibenchmark.bench_cfg import BenchmarkMatrix, Launcher

KUKA_SCALING = BenchmarkMatrix(
    tasks=["Isaac-Dexsuite-Kuka-Allegro-Lift-v0"],
    num_envs=[256, 512, 1024, 2048, 4096, 8192, 16384],
    hydra_sweeps={
        "preset": [
            "presets=cube",
            "presets=newton,cube",
            "presets=cube,single_camera,newton_renderer",
            "presets=newton,cube,single_camera,newton_renderer",
            "presets=cube,single_camera,newton_renderer,depth64",
            "presets=newton,cube,single_camera,newton_renderer,depth64",
        ],
    },
    launcher=Launcher.RSL_RL,
    num_frames=50,
    warmup_frames=5,
)

ALL_MATRICES = {
    "kuka_scaling": KUKA_SCALING,
}
