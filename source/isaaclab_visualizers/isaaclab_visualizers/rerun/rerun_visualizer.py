# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rerun visualizer implementation for Isaac Lab."""

from __future__ import annotations

import atexit
import contextlib
import inspect
import logging
import socket
import webbrowser
from typing import TYPE_CHECKING
from urllib.parse import quote

import newton
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from newton.viewer import ViewerRerun

from isaaclab.visualizers.base_visualizer import BaseVisualizer

from isaaclab_visualizers.newton.newton_visualization_markers import (
    NewtonVisualizationMarkers,
    render_newton_visualization_markers,
)
from isaaclab_visualizers.newton_adapter import (
    log_geo_with_expanded_plane_scale,
    log_state_particles,
    resolve_visible_env_indices,
)

from .rerun_visualizer_cfg import RerunVisualizerCfg

if TYPE_CHECKING:
    from isaaclab.cloner import ClonePlan
    from isaaclab.scene_data import SceneDataProvider
    from isaaclab.visualizers.streaming_view import StreamingView

logger = logging.getLogger(__name__)


def _is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Return whether a TCP port can be bound on host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, int(port)))
            return True
        except OSError:
            return False


def _is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Return whether a TCP port is currently accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, int(port))) == 0


def _normalize_host(addr: str) -> str:
    """Normalize bind host to loopback-friendly address for client URLs."""
    if addr in ("0.0.0.0", "127.0.0.1", "localhost"):
        return "127.0.0.1"
    return addr


def _ensure_rerun_server(app_id: str, bind_address: str, grpc_port: int, web_port: int) -> tuple[str, bool]:
    """Resolve rerun endpoint and whether viewer should start web/grpc server."""
    del app_id
    connect_host = _normalize_host(bind_address)
    expected_uri = f"rerun+http://{connect_host}:{int(grpc_port)}/proxy"

    if _is_port_open(grpc_port, host=connect_host):
        # Reuse existing endpoint; do not create a new server here.
        return expected_uri, False

    if not _is_port_free(web_port, host=connect_host):
        raise RuntimeError(f"Rerun web port {web_port} is in use. Free the port or choose a different `web_port`.")

    # No existing gRPC server: NewtonViewerRerun should start and own it.
    return expected_uri, True


def _open_rerun_web_viewer(host: str, web_port: int, connect_to: str) -> None:
    """Open rerun web UI and prefill endpoint connection URL."""
    url = _rerun_web_viewer_url(host, web_port, connect_to)
    try:
        if not webbrowser.open_new_tab(url):
            logger.info("[RerunVisualizer] Could not auto-open browser tab. Open manually: %s", url)
    except OSError:
        logger.info("[RerunVisualizer] Could not auto-open browser tab. Open manually: %s", url)


def _rerun_web_viewer_url(host: str, web_port: int, connect_to: str) -> str:
    """Return rerun web UI URL with prefilled endpoint."""
    # Keep the nested URL readable while still encoding '+' in the rerun+http scheme.
    return f"http://{host}:{int(web_port)}/?url={quote(connect_to, safe=':/')}"


class NewtonViewerRerun(ViewerRerun):
    """Wrapper around Newton's ViewerRerun with rendering pause controls."""

    #: Manager names set by :meth:`RerunVisualizer.add_live_plots`; when non-empty,
    #: ``_get_blueprint`` produces one ``TimeSeriesView`` per manager instead of the
    #: default single view.
    _live_plot_manager_names: list[str]

    def __init__(self, *args, open_browser: bool = False, streaming_view: bool = False, **kwargs):
        """Initialize viewer wrapper and Isaac Lab pause state."""
        self._live_plot_manager_names = []
        self._camera_pose: tuple | None = None
        self._streaming_view_active = streaming_view
        if open_browser:
            super().__init__(*args, **kwargs)
        else:
            original_serve_web_viewer = rr.serve_web_viewer

            # Rerun Viewer launches a browser automatically, so here we suppress that behavior
            def _serve_web_viewer_without_browser(*serve_args, **serve_kwargs):
                with contextlib.suppress(TypeError, ValueError):
                    supports_open_browser = "open_browser" in inspect.signature(original_serve_web_viewer).parameters
                    if supports_open_browser:
                        serve_kwargs.setdefault("open_browser", False)
                return original_serve_web_viewer(*serve_args, **serve_kwargs)

            with contextlib.ExitStack() as stack:
                rr.serve_web_viewer = _serve_web_viewer_without_browser
                stack.callback(setattr, rr, "serve_web_viewer", original_serve_web_viewer)
                super().__init__(*args, **kwargs)
        self._paused_rendering = False
        self._reset_requested = False

    def _get_blueprint(self):
        """Return a Rerun blueprint.

        When ``streaming_view`` is active the streaming composite
        (``Spatial2DView``) is the primary full-width panel and live-plot
        time-series views are appended as a narrow right column when registered.

        When streaming is **not** active the standard 3D Newton view is used,
        with live-plot time-series views appended when registered.

        The stored :attr:`_camera_pose` is forwarded to
        :class:`~rerun.blueprint.EyeControls3D` when the 3D view is included.
        """
        manager_views = (
            [rrb.TimeSeriesView(name=name, origin=f"/{name}") for name in self._live_plot_manager_names]
            if self._live_plot_manager_names
            else []
        )
        # TimePanel is always hidden (this viewer has no scrubbing UI use case).
        panel_states = [rrb.TimePanel(state="hidden")]

        # Streaming-view blueprint: 2D composite panel is dominant.
        if self._streaming_view_active:
            streaming_panel = rrb.Spatial2DView(name="Streaming View", origin="streaming/view")
            if manager_views:
                return rrb.Blueprint(
                    rrb.Horizontal(
                        streaming_panel,
                        rrb.Vertical(*manager_views),
                        column_shares=[4, 1],
                    ),
                    *panel_states,
                    collapse_panels=True,
                )
            return rrb.Blueprint(
                streaming_panel,
                *panel_states,
                collapse_panels=True,
            )

        # Standard 3D blueprint (no streaming).
        eye_controls = (
            rrb.EyeControls3D(position=self._camera_pose[0], look_target=self._camera_pose[1])
            if self._camera_pose
            else None
        )
        view_3d = (
            rrb.Spatial3DView(name="3D View", origin="/", eye_controls=eye_controls)
            if eye_controls
            else rrb.Spatial3DView(name="3D View", origin="/")
        )
        if manager_views:
            return rrb.Blueprint(
                rrb.Horizontal(
                    view_3d,
                    rrb.Vertical(*manager_views),
                    column_shares=[4, 1],
                ),
                *panel_states,
                collapse_panels=True,
            )
        return rrb.Blueprint(
            view_3d,
            *panel_states,
            collapse_panels=True,
        )

    def is_rendering_paused(self) -> bool:
        """Return whether rendering is paused by viewer controls."""
        return self._paused_rendering

    def consume_reset_request(self) -> bool:
        """Return whether an episode reset was requested and clear the flag."""
        requested = self._reset_requested
        self._reset_requested = False
        return requested

    def _render_ui(self):
        """Extend base UI with Isaac Lab rendering pause toggle."""
        super()._render_ui()

        if not self._has_imgui:
            return

        imgui = self._imgui
        if not imgui:
            return

        if imgui.collapsing_header("IsaacLab Controls"):
            if imgui.button("Pause Rendering" if not self._paused_rendering else "Resume Rendering"):
                self._paused_rendering = not self._paused_rendering
            if imgui.button("Reset Episode"):
                self._reset_requested = True

    def log_geo(
        self,
        name: str,
        geo_type: int,
        geo_scale: tuple[float, ...],
        geo_thickness: float,
        geo_is_solid: bool,
        geo_src=None,
        hidden: bool = False,
    ):
        """Log geometry, preserving large render extents for infinite ground planes."""
        return log_geo_with_expanded_plane_scale(
            super().log_geo,
            newton.GeoType.PLANE,
            name,
            geo_type,
            geo_scale,
            geo_thickness,
            geo_is_solid,
            geo_src,
            hidden,
        )

    def _log_particles(self, state):
        """Log particles from the requested scene-data publication."""
        log_state_particles(self, state)


class RerunVisualizer(BaseVisualizer):
    """Rerun visualizer for Isaac Lab."""

    marker_type = NewtonVisualizationMarkers

    def __init__(self, cfg: RerunVisualizerCfg):
        """Initialize Rerun visualizer state.

        Args:
            cfg: Rerun visualizer configuration.
        """
        super().__init__(cfg)
        from isaaclab.sim import SimulationContext

        simulation_context = SimulationContext.instance()
        if simulation_context is None:
            raise RuntimeError("RerunVisualizer requires an active SimulationContext.")
        self.marker_type = None if cfg.streaming_view else NewtonVisualizationMarkers
        if not cfg.streaming_view:
            from isaaclab_newton.cloner import NewtonReplicateContext

            self._newton_backend = simulation_context.get_or_create_backend(
                NewtonReplicateContext, simulation_context, clone_role="scene"
            )
            self._newton_backend.load_visual_shapes = True
        self.cfg: RerunVisualizerCfg = cfg
        self._viewer: NewtonViewerRerun | None = None
        self._sim_time = 0.0
        self._resolved_visible_env_ids: list[int] | None = None
        self._streaming: StreamingView | None = None

    def initialize(self, scene_data_provider: SceneDataProvider, clone_plan: ClonePlan) -> None:
        """Initialize rerun viewer and bind scene data provider.

        Args:
            scene_data_provider: Scene data provider used to fetch model/state data.
            clone_plan: Shared plan describing every cloned scene row.
        """
        if self._is_initialized:
            return

        self._set_scene_data_provider(scene_data_provider, clone_plan)
        num_envs = len(clone_plan.env_ids)
        self._env_ids = self._compute_visualized_env_ids()
        model = None if self.cfg.streaming_view else self._newton_backend.get_model()

        grpc_port = int(self.cfg.grpc_port)
        web_port = int(self.cfg.web_port)
        bind_address = self.cfg.bind_address or "0.0.0.0"
        rerun_address, start_server_in_viewer = _ensure_rerun_server(
            app_id=self.cfg.app_id,
            bind_address=bind_address,
            grpc_port=grpc_port,
            web_port=web_port,
        )
        if not start_server_in_viewer:
            logger.info("[RerunVisualizer] Reusing existing rerun server at %s.", rerun_address)

        viewer_address = None if start_server_in_viewer else rerun_address
        self._viewer = NewtonViewerRerun(
            app_id=self.cfg.app_id,
            address=viewer_address,
            serve_web_viewer=start_server_in_viewer,
            web_port=web_port,
            grpc_port=grpc_port,
            keep_historical_data=self.cfg.keep_historical_data,
            keep_scalar_history=self.cfg.keep_scalar_history or self.cfg.enable_live_plots,
            record_to_rrd=self.cfg.record_to_rrd,
            open_browser=self.cfg.open_browser,
            streaming_view=self.cfg.streaming_view,
        )
        if start_server_in_viewer:
            rerun_address = getattr(self._viewer, "_grpc_server_uri", rerun_address)
        viewer_host = _normalize_host(bind_address)
        viewer_url = _rerun_web_viewer_url(viewer_host, web_port, rerun_address)
        print()
        self._log_viewer_url("RerunVisualizer", viewer_url)
        if self.cfg.open_browser and not start_server_in_viewer:
            _open_rerun_web_viewer(viewer_host, web_port, rerun_address)
        self._resolved_visible_env_ids = resolve_visible_env_indices(self._env_ids, self.cfg.max_visible_envs, num_envs)
        if model is not None:
            self._viewer.set_model(model)
            self._viewer.show_particles = self.cfg.show_particles
            self._viewer.set_visible_worlds(self._resolved_visible_env_ids)
            # Preserve simulation world positions (env_spacing) rather than adding viewer-side offsets.
            self._viewer.set_world_offsets((0.0, 0.0, 0.0))
            self._apply_camera_pose(self._resolve_initial_camera_pose())
            self._viewer.up_axis = 2
            self._viewer.scaling = 1.0
        self._viewer._paused = False

        num_visualized_envs = (
            len(self._resolved_visible_env_ids) if self._resolved_visible_env_ids is not None else num_envs
        )
        self._log_initialization_table(
            logger=logger,
            title="RerunVisualizer Configuration",
            rows=[
                ("eye", self.cfg.eye),
                ("lookat", self.cfg.lookat),
                ("focal_length", f"{self.cfg.focal_length} (not applied: Rerun EyeControls3D has no FOV field)"),
                ("num_visualized_envs", num_visualized_envs),
                ("endpoint", f"http://{viewer_host}:{web_port}"),
                ("bind_address", bind_address),
                ("grpc_port", grpc_port),
                ("web_port", web_port),
                ("open_browser", self.cfg.open_browser),
                ("show_particles", self.cfg.show_particles),
                ("record_to_rrd", self.cfg.record_to_rrd or "<none>"),
            ],
        )

        self._setup_streaming_view()
        self._is_initialized = True
        atexit.register(self.close)

    def step(self, dt: float) -> None:
        """Advance visualization by one simulation step.

        Args:
            dt: Simulation time-step in seconds.
        """
        if not self._is_initialized or self._is_closed or self._viewer is None:
            return

        self._sim_time += dt

        if not self._viewer.is_paused():
            self._viewer.begin_frame(self._sim_time)
            try:
                if not self.cfg.streaming_view:
                    state = self._newton_backend.request_visualization_state(self._scene_data_provider)
                    if state is not None:
                        body_q = getattr(state, "body_q", None)
                        # Skip log_state for empty body arrays but do not return: _push_streaming_frame
                        # must still run after end_frame() so the streaming panel stays live.
                        if not (hasattr(body_q, "shape") and body_q.shape[0] == 0):
                            self._viewer.log_state(state)
                            if self.cfg.enable_markers:
                                render_newton_visualization_markers(
                                    self._viewer, self._resolved_visible_env_ids, num_envs=len(self._clone_plan.env_ids)
                                )
                self._render_live_plots()
            finally:
                self._viewer.end_frame()

        # Push streaming outside the pause-gate so it updates even when the Newton viewer is paused,
        # and outside begin/end_frame so the rr.log call is not constrained to the viewer's internal
        # time context. When paused, only compose (so render_tiled_rgb_array() stays current) without
        # re-logging to Rerun — the viewer already holds the last frame.
        if self._viewer.is_paused():
            self.render_tiled_rgb_array()
        else:
            self._push_streaming_frame()

    def close(self) -> None:
        """Close viewer/session resources."""
        if self._is_closed:
            return

        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

        self._streaming = None
        rr.disconnect()
        self._is_closed = True

    def is_running(self) -> bool:
        """Return whether the visualizer should continue stepping.

        Returns:
            ``True`` while the visualizer is active, otherwise ``False``.
        """
        if not self._is_initialized or self._is_closed:
            return False
        if self._viewer is None:
            return False
        return self._viewer.is_running()

    # ------------------------------------------------------------------
    # Streaming view
    # ------------------------------------------------------------------

    def _setup_streaming_view(self) -> None:
        """Resolve the camera the streaming panel shows."""
        from isaaclab.sim import SimulationContext
        from isaaclab.visualizers.streaming_view import StreamingView

        if not self.cfg.streaming_view:
            return
        sim = SimulationContext.instance()
        self._streaming = StreamingView(
            self.cfg, sim.get_camera_sensors(), visible_env_ids=self._resolved_visible_env_ids
        )
        self._viewer._streaming_view_active = True

    def _push_streaming_frame(self) -> None:
        """Compose the streaming frame and log it to Rerun."""
        composite = self.render_tiled_rgb_array()
        if composite is not None:
            rr.log("streaming/view", rr.Image(composite))

    def render_tiled_rgb_array(self) -> np.ndarray | None:
        """Return the composited streaming frame, all GT types side by side.

        Returns:
            ``uint8 (H, W, 3)`` composite array, or ``None`` when the streaming view is inactive or
            the camera produces no usable output.
        """
        return self._streaming.composite() if self._streaming is not None else None

    def _resolve_initial_camera_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Resolve initial camera pose from config."""
        return self._resolve_cfg_camera_pose("RerunVisualizer")

    def _apply_camera_pose(self, pose: tuple[tuple[float, float, float], tuple[float, float, float]]) -> None:
        """Apply camera pose to rerun's 3D view controls.

        Args:
            pose: Camera eye and target tuples.
        """
        if self._viewer is None:
            return
        cam_pos, cam_target = pose
        self._viewer._camera_pose = pose
        # Do not send a Spatial3DView blueprint when the streaming composite is active:
        # the streaming blueprint (Spatial2DView) from _get_blueprint() would be replaced
        # by a 3D view, hiding the streaming composite panel entirely.
        if self._streaming is not None:
            return
        panel_states = [rrb.TimePanel(state="hidden")]
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Spatial3DView(
                    name="3D View",
                    origin="/",
                    eye_controls=rrb.EyeControls3D(
                        position=cam_pos,
                        look_target=cam_target,
                    ),
                ),
                *panel_states,
                collapse_panels=True,
            )
        )

    def set_camera_view(
        self, eye: tuple[float, float, float] | list[float], target: tuple[float, float, float] | list[float]
    ) -> None:
        """Set the 3D view's camera eye/target.

        Args:
            eye: Camera eye position.
            target: Camera look-at target.
        """
        eye_t = (float(eye[0]), float(eye[1]), float(eye[2]))
        target_t = (float(target[0]), float(target[1]), float(target[2]))
        self._apply_camera_pose((eye_t, target_t))

    def supports_live_plots(self) -> bool:
        """Rerun backend supports live plots via :meth:`newton.Viewer.log_scalar` (mapped to ``rr.Scalars``)."""
        return True

    def add_live_plots(
        self,
        managers: dict,
        scalars: dict | None = None,
        term_names: dict[str, list[str]] | None = None,
        env_idx: int = 0,
    ) -> None:
        """Register managers for live plotting and send a per-manager blueprint.

        Calls the base implementation to populate :attr:`_live_plot_sources`, then sends a
        Rerun blueprint with one :class:`rerun.blueprint.TimeSeriesView` per manager so that
        each manager's terms appear in a separate chart panel rather than all sharing a single
        time-series view.

        Args:
            managers: Mapping of manager name to manager instance.
            scalars: Optional mapping of group name to a dict of ``{term_name: callable}``.
                Each callable must take no arguments and return a numeric value.
            term_names: Optional per-manager allowlists of term names to include.
            env_idx: Environment index to sample each step.  Defaults to ``0``.
        """
        super().add_live_plots(managers, scalars=scalars, term_names=term_names, env_idx=env_idx)
        if self._viewer is None or not self._live_plot_sources:
            return
        # Store manager names on the viewer so _get_blueprint() returns the per-manager
        # layout.  ViewerRerun.log_scalar calls _get_blueprint() on the first scalar logged,
        # which would overwrite any blueprint we send here — so we inject the layout into
        # the viewer's own blueprint factory instead of calling rr.send_blueprint directly.
        # Build the list of Rerun series-view names.  For manager sources, one view per
        # manager groups all their terms together.  For DirectScalarLivePlots (e.g. episode
        # metrics), each scalar gets its own view so they have independent Y axes — otherwise
        # episode_length (~160) and mean_reward (~0-1) share an axis, hiding the smaller one.
        from isaaclab.ui.live_plots.manager_live_plots import DirectScalarLivePlots

        names = []
        for source in self._live_plot_sources:
            if isinstance(source, DirectScalarLivePlots):
                for term in source._scalars:
                    names.append(f"{source.manager_name}/{term}")
            else:
                names.append(source.manager_name)
        self._viewer._live_plot_manager_names = names

    def _render_live_plots(self) -> None:
        """Push manager-term scalars to Rerun as time-series scalars."""
        if self._viewer is None or not self._live_plot_sources:
            return
        self._live_plots_step_counter += 1
        if self._live_plots_step_counter % max(1, self.cfg.live_plots_update_interval) != 0:
            return
        for source in self._live_plot_sources:
            for term_name, values in source.collect(self._live_plot_env_idx).items():
                if len(values) == 1:
                    self._viewer.log_scalar(f"{source.manager_name}/{term_name}", values[0])
                else:
                    for i, v in enumerate(values):
                        self._viewer.log_scalar(f"{source.manager_name}/{term_name}[{i}]", v)

    def is_training_paused(self) -> bool:
        """Return whether training is paused.

        Rerun viewer exposes rendering pause only.
        """
        return False

    def is_rendering_paused(self) -> bool:
        """Return whether rendering is paused from viewer controls."""
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.is_rendering_paused()

    def consume_reset_request(self) -> bool:
        """Return whether an episode reset was requested from viewer controls and clear the flag."""
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.consume_reset_request()
