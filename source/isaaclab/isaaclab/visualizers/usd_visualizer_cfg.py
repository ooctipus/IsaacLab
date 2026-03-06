# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the USD file visualizer."""

from __future__ import annotations

from isaaclab.utils import configclass

from .visualizer_cfg import VisualizerCfg


@configclass
class UsdVisualizerCfg(VisualizerCfg):
    """Configuration for USD file visualizer (offline export to .usd/.usda).

    This visualizer writes time-sampled transforms and meshes to a USD stage
    that can later be opened in Isaac Sim / Omniverse for look-dev, material
    assignment, and high-quality rendering via Movie Capture.
    """

    visualizer_type: str = "usd"
    """Type identifier for USD visualizer."""

    output_path: str = "output.usd"
    """File path for the exported USD stage."""

    fps: int = 60
    """Frames per second written into the USD stage metadata."""

    num_frames: int | None = None
    """Maximum number of frames to record. None means unlimited (record until close)."""

    max_worlds: int | None = 0
    """Maximum number of worlds/environments rendered by the viewer (0/None = all)."""
