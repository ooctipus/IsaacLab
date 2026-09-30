# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for visualizers."""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import Callable

    from isaaclab.cloner import ClonePlan
    from isaaclab.managers import ManagerBase
    from isaaclab.renderers.base_renderer import VisualMaterialBatch
    from isaaclab.scene_data import SceneDataProvider

    from .visualizer_cfg import VisualizerCfg


logger = logging.getLogger(__name__)

_USD_DEFAULT_VERTICAL_APERTURE_MM = 15.2908


class BaseVisualizer(ABC):
    """Base class for all visualizer backends.

    Lifecycle: __init__() -> initialize() -> step() (repeated) -> close()
    """

    marker_type: ClassVar[type | None] = None
    """Backend state used for clone-plan-declared visualization markers."""

    def __init__(self, cfg: VisualizerCfg):
        """Initialize visualizer with config.

        Args:
            cfg: Visualizer configuration.
        """
        self.cfg = cfg
        self._scene_data_provider = None
        self._clone_plan = None
        self._is_initialized = False
        self._is_closed = False
        self._env_ids: list[int] | None = None
        self._deferred_startup_messages: list[str] = []
        self._live_plot_sources: list = []
        self._live_plot_env_idx: int = 0
        self._live_plots_step_counter: int = 0
        self._reset_requested: bool = False

    @property
    def visual_material_writer(self) -> Callable[[tuple[VisualMaterialBatch, ...]], Any] | None:
        """Return the backend's shared material-writer factory, if supported.

        Its writer accepts ``None`` for a full sync or channel-to-material-offset device arrays plus
        one environment-id device array for partial writes, and provides an idempotent ``close()``.
        """
        return None

    @abstractmethod
    def initialize(self, scene_data_provider: SceneDataProvider, clone_plan: ClonePlan) -> None:
        """Initialize visualizer resources.

        Args:
            scene_data_provider: Scene data provider used by the visualizer.
            clone_plan: Shared plan describing every cloned scene row.
        """
        raise NotImplementedError

    def _set_scene_data_provider(
        self, scene_data_provider: SceneDataProvider, clone_plan: ClonePlan
    ) -> SceneDataProvider:
        """Store the scene data provider and clone plan shared by all visualizers."""
        if scene_data_provider is None:
            raise RuntimeError(f"{self.__class__.__name__} requires a scene_data_provider.")
        if clone_plan.env_ids is None:
            raise RuntimeError(f"{self.__class__.__name__} requires a completed clone plan.")
        self._scene_data_provider = scene_data_provider
        self._clone_plan = clone_plan
        return scene_data_provider

    @abstractmethod
    def step(self, dt: float) -> None:
        """Update visualization for one step.

        Args:
            dt: Time step in seconds.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        raise NotImplementedError

    @abstractmethod
    def is_running(self) -> bool:
        """Check if visualizer is still running (e.g., window not closed).

        Returns:
            ``True`` if the visualizer is running, otherwise ``False``.
        """
        raise NotImplementedError

    def is_training_paused(self) -> bool:
        """Check if training is paused by visualizer controls.

        Returns:
            ``True`` if training is paused, otherwise ``False``.
        """
        return False

    def is_rendering_paused(self) -> bool:
        """Check if rendering is paused by visualizer controls.

        Returns:
            ``True`` if rendering is paused, otherwise ``False``.
        """
        return False

    def consume_reset_request(self) -> bool:
        """Return whether an episode reset was requested and clear the flag.

        Returns:
            ``True`` once when a reset was requested, then ``False`` until the next request.
        """
        requested = self._reset_requested
        self._reset_requested = False
        return requested

    @property
    def is_initialized(self) -> bool:
        """Check if initialize() has been called."""
        return self._is_initialized

    @property
    def is_closed(self) -> bool:
        """Check if close() has been called."""
        return self._is_closed

    def supports_live_plots(self) -> bool:
        """Check if visualizer supports live plots.

        Returns:
            ``True`` if live plots are supported, otherwise ``False``.
        """
        return False

    def add_live_plots(
        self,
        managers: dict[str, ManagerBase],
        scalars: dict[str, dict[str, Any]] | None = None,
        term_names: dict[str, list[str]] | None = None,
        env_idx: int = 0,
    ) -> None:
        """Register environment managers and direct scalars as live-plot data sources.

        Creates one :class:`~isaaclab.ui.live_plots.ManagerLivePlots` per manager and one
        :class:`~isaaclab.ui.live_plots.DirectScalarLivePlots` per scalar group, storing all
        sources for use inside :meth:`_render_live_plots`.  Does nothing when
        :meth:`supports_live_plots` returns ``False`` or ``cfg.enable_live_plots`` is ``False``.

        Args:
            managers: Mapping of manager name to manager instance.
            scalars: Optional mapping of group name to a dict of ``{term_name: callable}``.
                Each callable must take no arguments and return a numeric value.  Used to
                plot non-manager metrics such as episode reward or episode length.
            term_names: Optional per-manager allowlists of term names to include.
                ``None`` (default) collects all terms for every manager.
            env_idx: Environment index to sample each step.  Defaults to ``0``.
        """
        if not self.supports_live_plots():
            return
        if not self.cfg.enable_live_plots:
            return
        from isaaclab.ui.live_plots.manager_live_plots import DirectScalarLivePlots, ManagerLivePlots

        # Scalar groups (e.g. episode metrics) are placed first so they appear at
        # the top of every visualizer's plot list regardless of backend ordering.
        self._live_plot_sources = []
        if scalars:
            for group_name, scalar_dict in scalars.items():
                self._live_plot_sources.append(DirectScalarLivePlots(group_name, scalar_dict))
        for name, mgr in managers.items():
            # Skip managers that have no active terms — they contribute nothing to plots
            # and would create empty panels in Rerun, Viser, and the Kit live-plot window.
            active = getattr(mgr, "active_terms", None)
            if active is not None:
                has_terms = bool(active) if not isinstance(active, dict) else any(v for v in active.values())
                if not has_terms:
                    continue
            self._live_plot_sources.append(ManagerLivePlots(name, mgr, (term_names or {}).get(name)))
        self._live_plot_env_idx = env_idx

    def _render_live_plots(self) -> None:
        """Push live-plot data to the backend for the current step.

        Called from each backend's :meth:`step` implementation when live plots are active.
        The default implementation is a no-op; backends that support live plots override this
        method to forward collected term values to their native plotting API (e.g.
        ``viewer.log_scalar``).
        """
        pass

    def pumps_app_update(self) -> bool:
        """Whether this visualizer calls omni.kit.app.get_app().update() in step().

        Returns True for visualizers (e.g. KitVisualizer) that already pump the Kit
        app loop, so SimulationContext.render() can skip its own app.update() call
        and avoid double-rendering.
        """
        return False

    def get_visualized_env_ids(self) -> list[int] | None:
        """Return env IDs this visualizer is displaying, if any.

        Returns:
            Visualized environment ids, or ``None`` for all environments.
        """
        return self._env_ids

    def _compute_visualized_env_ids(self) -> list[int] | None:
        """Compute which environment indices to visualize from config.

        Returns:
            Selected environment ids, or ``None`` to visualize all environments.
        """
        if self._clone_plan is None:
            return None
        cfg = self.cfg
        num_envs = len(self._clone_plan.env_ids)
        if num_envs <= 0:
            logger.debug("[Visualizer] num_envs is 0 or missing from provider; env selection disabled.")
            return None
        # Explicit list wins; never combine with random cap-only mode.
        if cfg.visible_env_indices is not None:
            return [i for i in cfg.visible_env_indices if 0 <= i < num_envs]

        max_visible = cfg.max_visible_envs
        # Random subset only for cap-only mode: needs a cap and no explicit indices (see VisualizerCfg).
        if max_visible is not None and cfg.randomly_sample_visible_envs and int(max_visible) >= 0:
            k = min(int(max_visible), num_envs)
            # k == 0: sample(range(n), 0) is []; contiguous resolver used the same convention.
            return sorted(random.sample(range(num_envs), k))
        return None

    def get_rendering_dt(self) -> float | None:
        """Get rendering time step.

        Returns:
            Rendering time step override, or ``None`` to use interface default.
        """
        return None

    def set_camera_view(self, eye: tuple, target: tuple) -> None:
        """Set camera view position.

        Args:
            eye: Camera eye position.
            target: Camera target position.
        """
        pass

    def _resolve_cfg_camera_pose(
        self, _visualizer_name: str
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Resolve camera pose from cfg eye/lookat fields."""
        eye = tuple(float(v) for v in self.cfg.eye)
        lookat = tuple(float(v) for v in self.cfg.lookat)
        return eye, lookat

    def _focal_length_to_vertical_fov_degrees(self) -> float:
        """Convert cfg focal length to vertical FOV using USD's default aperture."""
        focal_length = float(self.cfg.focal_length)
        if focal_length <= 0.0:
            raise ValueError("VisualizerCfg.focal_length must be positive.")
        return math.degrees(2.0 * math.atan(_USD_DEFAULT_VERTICAL_APERTURE_MM / (2.0 * focal_length)))

    def reset(self, soft: bool = False) -> None:
        """Reset visualizer state.

        Args:
            soft: Whether to perform a soft reset.
        """
        pass

    def _log_initialization_table(self, logger: logging.Logger, title: str, rows: list[tuple[str, Any]]) -> None:
        """Log a compact initialization table for a visualizer.

        Args:
            logger: Logger used to emit the table.
            title: Table title.
            rows: Table row key/value pairs.
        """
        from prettytable import PrettyTable

        table = PrettyTable()
        table.title = title
        table.field_names = ["Field", "Value"]
        table.align["Field"] = "l"
        table.align["Value"] = "l"
        for key, value in rows:
            table.add_row([key, value])
        logger.debug("Visualizer initialization:\n%s", table.get_string())

    def _log_viewer_url(
        self,
        visualizer_name: str,
        viewer_url: str,
    ) -> None:
        """Queue a visible browser URL block for web-based visualizers.

        Args:
            visualizer_name: Name of the visualizer exposing the URL.
            viewer_url: Browser URL for the visualizer.
        """
        parsed_url = urlparse(viewer_url)
        visualizer_label = visualizer_name.removesuffix("Visualizer").lower()
        title = f" {visualizer_label} (listening *:{parsed_url.port}) " if parsed_url.port else f" {visualizer_label} "
        label = "URL"
        label_width = len(label)
        value_width = max(len(viewer_url), len(title) + 2, 21)
        inner_width = label_width + value_width + 9
        left_rule_width = max((inner_width - len(title)) // 2, 1)
        right_rule_width = max(inner_width - len(title) - left_rule_width, 1)

        lines = [
            f"╭{'─' * left_rule_width}{title}{'─' * right_rule_width}╮",
            f"│{' ' * (label_width + 4)}╷{' ' * (value_width + 4)}│",
            f"│   {label:<{label_width}} │ {viewer_url:<{value_width}}   │",
            f"│{' ' * (label_width + 4)}╵{' ' * (value_width + 4)}│",
            f"╰{'─' * inner_width}╯",
        ]
        self._deferred_startup_messages.append("\n" + "\n".join(lines) + "\n")

    def flush_startup_messages(self) -> None:
        """Print deferred startup messages immediately before the workflow update loop starts."""
        for message in self._deferred_startup_messages:
            print(message, flush=True)
        self._deferred_startup_messages.clear()

    def play(self) -> None:
        """Handle simulation play/start. No-op by default."""
        pass

    def pause(self) -> None:
        """Handle simulation pause. No-op by default."""
        pass

    def stop(self) -> None:
        """Handle simulation stop. No-op by default."""
        pass
