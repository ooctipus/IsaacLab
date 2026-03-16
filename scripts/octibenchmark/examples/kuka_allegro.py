# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark matrices for Kuka Allegro Lift task."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from octibenchmark.bench_cfg import BenchmarkMatrix, Launcher, ProfileLevel

_HYDRA_SWEEPS = {
    "preset": [
        "presets=cube",
        "presets=newton,cube",
        "presets=cube,single_camera,newton_renderer,rgb64",
        "presets=newton,cube,single_camera,newton_renderer,rgb64",
        "presets=cube,single_camera,newton_renderer,depth64",
        "presets=cube,single_camera,newton_renderer,albedo64",
        "presets=newton,cube,single_camera,newton_renderer,depth64",
        "presets=cube,single_camera,newton_renderer,rgb128",
        "presets=newton,cube,single_camera,newton_renderer,rgb128",
        "presets=cube,single_camera,newton_renderer,depth128",
        "presets=cube,single_camera,newton_renderer,albedo128",
        "presets=newton,cube,single_camera,newton_renderer,depth128",
    ],
}
_NUM_ENVS = [256, 512, 1024]
_TASK = "Isaac-Dexsuite-Kuka-Allegro-Lift-v0"

KUKA_NON_RL = BenchmarkMatrix(
    tasks=[_TASK],
    num_envs=_NUM_ENVS,
    hydra_sweeps=_HYDRA_SWEEPS,
    launcher=Launcher.NON_RL,
    num_frames=50,
    warmup_frames=5,
    profile_level=ProfileLevel.PLAIN,
)

KUKA_RSL_RL = BenchmarkMatrix(
    tasks=[_TASK],
    num_envs=_NUM_ENVS,
    hydra_sweeps=_HYDRA_SWEEPS,
    launcher=Launcher.RSL_RL,
    num_frames=50,
    warmup_frames=5,
    profile_level=ProfileLevel.PLAIN,
)

ALL_MATRICES = {
    "non_rl": KUKA_NON_RL,
    "rsl_rl": KUKA_RSL_RL,
}
