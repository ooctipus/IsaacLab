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
    num_envs=[64, 256, 1024],
    hydra_sweeps={
        "preset": [
            "presets=cube",
            "presets=newton,cube",
        ],
    },
    launcher=Launcher.NON_RL,
    num_frames=50,
    warmup_frames=5,
)

ALL_MATRICES = {
    "kuka_scaling": KUKA_SCALING,
}
