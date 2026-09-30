# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton ``ViewerRTX`` visualizer that renders the simulation's own cloned USD stage."""

from __future__ import annotations

import math
import os
import sys
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from isaaclab.sim import SimulationContext
from isaaclab.visualizers.base_visualizer import BaseVisualizer

if TYPE_CHECKING:
    from isaaclab.cloner import ClonePlan
    from isaaclab.scene_data import SceneDataProvider

    from .newton_visualizer_cfg import NewtonRTXStageVisualizerCfg


class NewtonRTXStageVisualizer(BaseVisualizer):
    """Draw the cloned OVStage stage with Newton's ``ViewerRTX`` while Newton drives the body poses.

    The OV clone context builds the stage from the clone plan, so every environment keeps its authored
    materials, and ``ViewerRTX`` borrows it: Newton's model is bound to the stage prims at the body
    labels, and only the body poses are written each step.
    """

    marker_type = None

    def __init__(self, cfg: NewtonRTXStageVisualizerCfg):
        super().__init__(cfg)
        from isaaclab_newton.cloner import NewtonReplicateContext
        from isaaclab_ov.cloner import OvReplicateContext

        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError("NewtonRTXStageVisualizer requires an active SimulationContext.")
        # Both contexts must join before the clone plan completes: Newton supplies the model and state,
        # and the OV context builds the stage that ViewerRTX draws.
        self._newton_backend = sim.get_or_create_backend(NewtonReplicateContext, sim, clone_role="scene")
        self._ov_backend = sim.get_or_create_backend(OvReplicateContext, sim, clone_role="scene")
        self._ov_backend._request_ovstage()
        self.cfg: NewtonRTXStageVisualizerCfg = cfg
        self._viewer = None
        self._stage = None
        self._sim_time = 0.0
        self._step_counter = 0

    def initialize(self, scene_data_provider: SceneDataProvider, clone_plan: ClonePlan) -> None:
        """Build the rendering stage and bind Newton's model to it."""
        if self._is_initialized:
            return
        self._set_scene_data_provider(scene_data_provider, clone_plan)
        from newton.viewer import ViewerRTX

        headless = self.cfg.headless or (sys.platform not in ("win32", "darwin") and not os.environ.get("DISPLAY"))
        self._stage = self._ov_backend.create_ovstage()
        self._viewer = ViewerRTX(
            width=self.cfg.window_width, height=self.cfg.window_height, headless=headless, ovstage=self._stage
        )
        self._viewer.set_model(self._newton_backend.get_model())
        self._apply_camera_pose()
        self._is_initialized = True

    def _apply_camera_pose(self) -> None:
        """Aim the viewer from :attr:`eye` at :attr:`lookat`."""
        eye, target = self.cfg.eye, self.cfg.lookat
        dx, dy, dz = (target[i] - eye[i] for i in range(3))
        yaw = math.degrees(math.atan2(dy, dx))
        pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
        self._viewer.set_camera(wp.vec3(*eye), pitch=pitch, yaw=yaw)

    def step(self, dt: float) -> None:
        """Write the current Newton body poses into the stage and draw one frame."""
        if not self._is_initialized or self._is_closed:
            return
        self._sim_time += dt
        self._step_counter += 1
        if self._step_counter % self.cfg.update_frequency != 0:
            return
        state = self._newton_backend.request_visualization_state(self._scene_data_provider)
        self._viewer.begin_frame(self._sim_time)
        self._viewer.log_state(state)
        self._viewer.end_frame()

    def render_rgb_array(self) -> np.ndarray:
        """Return the last rendered frame as an RGB array."""
        if self._viewer is None:
            raise RuntimeError("NewtonRTXStageVisualizer has not been initialized.")
        # ViewerRTX has no public pixel accessor yet; save_screenshot() is the only public capture.
        return self._viewer._capture_screenshot_pixels()

    def close(self) -> None:
        """Close the viewer and release the stage."""
        if self._is_closed:
            return
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._stage = None
        self._is_closed = True

    def is_running(self) -> bool:
        """Return whether the viewer window is still open."""
        return self._viewer is not None and self._viewer.is_running()
