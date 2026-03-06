# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD file visualizer that exports simulation to a time-sampled USD stage."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from newton.viewer import ViewerUSD

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .usd_visualizer_cfg import UsdVisualizerCfg
from .visualizer import Visualizer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.sim.scene_data_providers import SceneDataProvider


class UsdVisualizer(Visualizer):
    """Offline USD exporter that writes time-sampled transforms and meshes.

    The resulting ``.usd`` file can be referenced into an Omniverse look-dev
    template stage for material assignment, camera setup, and high-quality
    rendering via the Movie Capture extension.
    """

    def __init__(self, cfg: UsdVisualizerCfg):
        super().__init__(cfg)
        self.cfg: UsdVisualizerCfg = cfg
        self._viewer: ViewerUSD | None = None
        self._model = None
        self._state = None
        self._sim_time = 0.0
        self._scene_data_provider = None
        self._pbar = None

    def initialize(self, scene_data_provider: SceneDataProvider) -> None:
        if self._is_initialized:
            logger.debug("[UsdVisualizer] initialize() called while already initialized.")
            return
        if scene_data_provider is None:
            raise RuntimeError("USD visualizer requires a scene_data_provider.")

        self._scene_data_provider = scene_data_provider

        self._env_ids = self._compute_visualized_env_ids()
        if self._env_ids:
            get_filtered_model = getattr(scene_data_provider, "get_newton_model_for_env_ids", None)
            if callable(get_filtered_model):
                self._model = get_filtered_model(self._env_ids)
            else:
                self._model = scene_data_provider.get_newton_model()
        else:
            self._model = scene_data_provider.get_newton_model()
        self._state = scene_data_provider.get_newton_state(self._env_ids)

        try:
            self._viewer = ViewerUSD(
                output_path=self.cfg.output_path,
                fps=self.cfg.fps,
                up_axis="Z",
                num_frames=self.cfg.num_frames,
            )
            max_worlds = self.cfg.max_worlds
            self._viewer.set_model(self._model, max_worlds=None if max_worlds in (None, 0) else max_worlds)
            self._sim_time = 0.0

            logger.info(
                "[UsdVisualizer] initialized | output_path=%s fps=%d num_frames=%s max_worlds=%s",
                os.path.abspath(self.cfg.output_path),
                self.cfg.fps,
                self.cfg.num_frames,
                self.cfg.max_worlds,
            )
            if tqdm is not None:
                total = self.cfg.num_frames
                self._pbar = tqdm(
                    total=total,
                    desc="USD export",
                    unit="frame",
                )
            self._is_initialized = True
        except Exception as exc:
            logger.error("[UsdVisualizer] Failed to initialize viewer: %s", exc)
            raise

    def step(self, dt: float) -> None:
        if not self._is_initialized or self._viewer is None or self._scene_data_provider is None:
            return

        self._state = self._scene_data_provider.get_newton_state(self._env_ids)
        self._sim_time += dt

        self._viewer.begin_frame(self._sim_time)
        self._viewer.log_state(self._state)
        self._viewer.end_frame()

        if self._pbar is not None:
            self._pbar.update(1)

    def close(self) -> None:
        if not self._is_initialized:
            return

        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

        try:
            if self._viewer is not None:
                self._viewer.close()
                output = os.path.abspath(self.cfg.output_path)
                if os.path.exists(output):
                    size = os.path.getsize(output)
                    logger.info("[UsdVisualizer] USD saved: %s (%s bytes)", output, size)
                else:
                    logger.warning("[UsdVisualizer] USD file not found after close: %s", output)
        except Exception as exc:
            logger.warning("[UsdVisualizer] Error during close: %s", exc)

        self._viewer = None
        self._is_initialized = False
        self._is_closed = True

    def is_running(self) -> bool:
        if self._viewer is None:
            return False
        return self._viewer.is_running()

    def is_training_paused(self) -> bool:
        return False

    def supports_markers(self) -> bool:
        return False

    def supports_live_plots(self) -> bool:
        return False
