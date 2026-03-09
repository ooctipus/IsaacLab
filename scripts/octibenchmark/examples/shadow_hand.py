# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shadow Hand benchmark matrix definitions.

Defines the full benchmark sweep for Shadow Hand in-hand repose tasks,
covering vision and state-only environments across physics backends,
renderers, camera data types, resolutions, and environment counts.

These are pure data definitions — no CLI logic. To run them, use
:mod:`octibenchmark.run_matrix`::

    python scripts/octibenchmark/run_matrix.py \\
        --example shadow_hand --dry_run

    python scripts/octibenchmark/run_matrix.py \\
        --example shadow_hand \\
        --matrices shadow_vision_nonrl shadow_state_nonrl \\
        --output_dir /tmp/shadow_bench
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from octibenchmark.bench_cfg import BenchmarkMatrix, Launcher

# Vision-specific NVTX hooks for Shadow Hand environments
_SHADOW_VISION_HOOKS = [
    ("_compute_image_observations", "vision.compute_image_obs"),
    ("_compute_proprio_observations", "vision.compute_proprio_obs"),
    ("feature_extractor.step", "vision.feature_extractor.step"),
]


# ---------------------------------------------------------------------------
# Non-RL benchmarks: pure environment stepping (no network)
# ---------------------------------------------------------------------------

# Shadow Hand Vision (with CNN feature extractor)
SHADOW_VISION_NONRL = BenchmarkMatrix(
    tasks=["Isaac-Repose-Cube-Shadow-Vision-Direct-v0"],
    num_envs=[2048, 4096, 8192],
    hydra_sweeps={
        "preset": [
            "presets=newton,newton_renderer,rgb",
            "presets=newton,newton_renderer,depth",
        ],
        "resolution": [
            "env.tiled_camera.width=64 env.tiled_camera.height=64",
            "env.tiled_camera.width=128 env.tiled_camera.height=128",
            "env.tiled_camera.width=256 env.tiled_camera.height=256",
        ],
    },
    launcher=Launcher.NON_RL,
    num_frames=100,
    warmup_frames=10,
    extra_nvtx_hooks=_SHADOW_VISION_HOOKS,
)

# Shadow Hand Vision Benchmark (no CNN — raw camera output)
SHADOW_VISION_BENCH_NONRL = BenchmarkMatrix(
    tasks=["Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0"],
    num_envs=[2048, 4096, 8192],
    hydra_sweeps={
        "preset": [
            "presets=newton,newton_renderer,rgb",
            "presets=newton,newton_renderer,depth",
        ],
        "resolution": [
            "env.tiled_camera.width=64 env.tiled_camera.height=64",
            "env.tiled_camera.width=128 env.tiled_camera.height=128",
            "env.tiled_camera.width=256 env.tiled_camera.height=256",
        ],
    },
    launcher=Launcher.NON_RL,
    num_frames=100,
    warmup_frames=10,
    extra_nvtx_hooks=_SHADOW_VISION_HOOKS,
)

# Shadow Hand state-only (no camera, no resolution sweep)
SHADOW_STATE_NONRL = BenchmarkMatrix(
    tasks=["Isaac-Repose-Cube-Shadow-Direct-v0"],
    num_envs=[2048, 4096, 8192, 16384],
    hydra_sweeps={
        "preset": ["presets=newton"],
    },
    launcher=Launcher.NON_RL,
    num_frames=100,
    warmup_frames=10,
)


# ---------------------------------------------------------------------------
# RSL-RL training benchmarks
# ---------------------------------------------------------------------------

SHADOW_VISION_TRAIN = BenchmarkMatrix(
    tasks=["Isaac-Repose-Cube-Shadow-Vision-Direct-v0"],
    num_envs=[2048, 4096],
    hydra_sweeps={
        "preset": [
            "presets=newton,newton_renderer,rgb",
            "presets=newton,newton_renderer,depth",
        ],
        "resolution": [
            "env.tiled_camera.width=128 env.tiled_camera.height=128",
        ],
    },
    launcher=Launcher.RSL_RL,
    max_iterations=5,
    warmup_frames=0,
    extra_nvtx_hooks=_SHADOW_VISION_HOOKS,
)

SHADOW_STATE_TRAIN = BenchmarkMatrix(
    tasks=["Isaac-Repose-Cube-Shadow-Direct-v0"],
    num_envs=[4096, 8192],
    hydra_sweeps={
        "preset": ["presets=newton"],
    },
    launcher=Launcher.RSL_RL,
    max_iterations=5,
    warmup_frames=0,
)


# ---------------------------------------------------------------------------
# All matrices grouped for convenience
# ---------------------------------------------------------------------------

ALL_MATRICES = {
    "shadow_vision_nonrl": SHADOW_VISION_NONRL,
    "shadow_vision_bench_nonrl": SHADOW_VISION_BENCH_NONRL,
    "shadow_state_nonrl": SHADOW_STATE_NONRL,
    "shadow_vision_train": SHADOW_VISION_TRAIN,
    "shadow_state_train": SHADOW_STATE_TRAIN,
}
