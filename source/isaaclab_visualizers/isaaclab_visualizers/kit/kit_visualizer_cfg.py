# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Kit-based visualizer."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg


@configclass
class KitVisualizerCfg(VisualizerCfg):
    """Configuration for Kit visualizer using Isaac Sim viewport."""

    visualizer_type: str = "kit"
    """Type identifier for Kit visualizer."""

    cam_prim_path: str = "/OmniverseKit_Persp"
    """USD path to the Kit viewport camera when :attr:`cam_source` is ``"prim_path"``.

    Kit visualizer defaults to the standard viewport camera instead of assuming the
    scene contains a task-authored camera prim at ``/World/envs/env_0/Camera``.
    """

    viewport_name: str | None = None
    """Name for a new viewport window when :attr:`create_viewport` is ``True``.

    If ``None``, a default name (``"Visualizer Viewport"``) is used.
    """

    create_viewport: bool = False
    """If ``True``, create a new viewport window; if ``False``, use the active viewport window."""

    headless: bool = False
    """Run without creating viewport windows when supported by the app."""

    dock_position: str = "SAME"
    """Dock position for a new viewport. Options: 'LEFT', 'RIGHT', 'BOTTOM', 'SAME'."""

    window_width: int = 1280
    """Viewport width in pixels (when :attr:`create_viewport` is ``True``)."""

    window_height: int = 720
    """Viewport height in pixels (when :attr:`create_viewport` is ``True``)."""

    def __post_init__(self) -> None:
        """Normalize the inherited visualizer camera default for Kit viewports."""
        if self.cam_prim_path == "/World/envs/env_0/Camera":
            self.cam_prim_path = "/OmniverseKit_Persp"
