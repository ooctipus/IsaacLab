# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Viser-based visualizer using Newton's ViewerViser."""

from __future__ import annotations

import contextlib
import io
import logging
import math
import os
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any

import newton
import numpy as np
from newton.viewer import ViewerViser

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

from .viser_visualizer_cfg import ViserVisualizerCfg

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from isaaclab.cloner import ClonePlan
    from isaaclab.scene_data import SceneDataProvider
    from isaaclab.visualizers.streaming_view import StreamingView


def _letterbox_16_9(image: np.ndarray) -> np.ndarray:
    """Pad *image* with black bars to 16:9 so Viser doesn't stretch it.

    Args:
        image: ``uint8 (H, W, 3)`` composite frame.

    Returns:
        ``uint8 (H', W', 3)`` image with ``W'/H' == 16/9``.
    """
    h, w = image.shape[:2]
    target_w = max(w, int(h * 16 / 9))
    target_h = max(h, int(w * 9 / 16))
    if target_w == w and target_h == h:
        return image
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - h) // 2
    x0 = (target_w - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = image
    return canvas


def _scalar_base_name(name: str) -> str:
    """Strip a trailing ``[N]`` component index from a scalar name to get the term base name."""
    if name.endswith("]") and "[" in name:
        bracket = name.rfind("[")
        if name[bracket + 1 : -1].isdigit():
            return name[:bracket]
    return name


def _disable_viser_runtime_client_rebuild_if_bundled() -> None:
    """Skip viser's runtime frontend rebuild when a bundled build is present."""
    try:
        import viser
        import viser._client_autobuild as client_autobuild
    except ImportError:
        return

    client_root = Path(viser.__file__).resolve().parent / "client"
    has_bundled_build = (client_root / "build" / "index.html").exists()
    if not has_bundled_build:
        return

    client_autobuild.ensure_client_is_built = lambda: None


def _open_viser_web_viewer(url: str) -> None:
    """Open the Viser web UI in a browser."""
    try:
        if not webbrowser.open_new_tab(url):
            logger.info("[ViserVisualizer] Could not auto-open browser tab. Open manually: %s", url)
    except OSError:
        logger.info("[ViserVisualizer] Could not auto-open browser tab. Open manually: %s", url)


def _viser_web_viewer_url(port: int, display_address: str) -> str:
    """Return Viser web UI URL for display to users."""
    return f"http://{display_address}:{int(port)}"


class NewtonViewerViser(ViewerViser):
    """Isaac Lab wrapper for Newton's ViewerViser."""

    def __init__(
        self,
        port: int = 8080,
        bind_address: str = "0.0.0.0",
        label: str | None = None,
        verbose: bool = True,
        share: bool = False,
        record_to_viser: str | None = None,
        metadata: dict | None = None,
    ):
        """Initialize Newton-backed viser viewer wrapper.

        Args:
            port: HTTP port for viser server.
            bind_address: Host/interface for the Viser server to bind.
            label: Optional viewer label.
            verbose: Whether to keep verbose startup output enabled.
            share: Whether to enable sharing/tunneling.
            record_to_viser: Optional recording destination.
            metadata: Optional metadata attached to the viewer.
        """
        _disable_viser_runtime_client_rebuild_if_bundled()
        try:
            viser = self._get_viser()
        except ImportError as exc:
            raise ImportError(
                "The Viser visualizer requires the optional 'viser' package. "
                "Run your command with: uv run --extra viser <command>."
            ) from exc
        original_viser_server = viser.ViserServer

        def _viser_server_with_bind_address(*args, **kwargs):
            kwargs["host"] = bind_address
            kwargs["verbose"] = verbose
            return original_viser_server(*args, **kwargs)

        with contextlib.ExitStack() as stack:
            viser.ViserServer = _viser_server_with_bind_address
            stack.callback(setattr, viser, "ViserServer", original_viser_server)
            if not verbose:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            super().__init__(
                port=port,
                label=label,
                verbose=verbose,
                share=share,
                record_to_viser=record_to_viser,
            )
        self._metadata = metadata or {}
        self._isaaclab_plane_grid_cache: dict[str, tuple] = {}
        self._per_plot_folders: dict[str, Any] = {}
        self._live_plots_folder: Any = None

    @property
    def share_url(self) -> str | None:
        """Return the public share URL created by Viser, if any."""
        return self._share_url

    def clear_model(self) -> None:
        """Clear cached state and remove per-plot GUI folders with the viewer model."""
        cache = getattr(self, "_isaaclab_plane_grid_cache", None)
        if cache is not None:
            cache.clear()
        super().clear_model()
        per_plot_folders = getattr(self, "_per_plot_folders", None)
        if per_plot_folders:
            for folder in list(per_plot_folders.values()):
                with contextlib.suppress(Exception):
                    folder.remove()
            per_plot_folders.clear()
        # Do NOT remove _live_plots_folder — it is a persistent structural element
        # created once in _setup_isaaclab_sidebar and should survive model reloads.
        # Only the per-term chart handles (in _per_plot_folders) are cleared above.

    @staticmethod
    def _array_signature(array) -> tuple[tuple[int, ...], bytes] | None:
        """Return a stable signature for small transform/scale arrays."""
        if array is None:
            return None
        array_np = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
        return tuple(int(dim) for dim in array_np.shape), array_np.tobytes()

    def _log_plane_instances(
        self,
        name: str,
        plane_info: dict[str, float | bool],
        xforms,
        scales,
        hidden: bool = False,
    ) -> None:
        """Avoid removing/re-adding unchanged Viser plane grids every frame."""
        cache = getattr(self, "_isaaclab_plane_grid_cache", None)
        if hidden or xforms is None:
            if cache is not None:
                cache.pop(name, None)
            return super()._log_plane_instances(name, plane_info, xforms, scales, hidden=hidden)

        xforms_np = self._to_numpy(xforms)
        if xforms_np is None or len(xforms_np) == 0:
            if cache is not None:
                cache.pop(name, None)
            return super()._log_plane_instances(name, plane_info, xforms, scales, hidden=hidden)

        scales_np = self._to_numpy(scales) if scales is not None else None
        signature = (
            float(plane_info["width"]),
            float(plane_info["length"]),
            self._array_signature(xforms_np),
            self._array_signature(scales_np),
        )
        if cache is not None and cache.get(name) == signature and name in self._plane_handles:
            return None
        if cache is not None:
            cache[name] = signature
        return super()._log_plane_instances(name, plane_info, xforms, scales, hidden=hidden)

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

    def _update_scalar_plots(self) -> None:
        """Create one collapsible folder per term, with one multi-series chart per term.

        Components of the same term (e.g. ``joint_pos[0]``, ``joint_pos[1]``) are grouped
        onto a single uPlot chart as separate series, matching the Kit visualizer's per-term
        grouping.  Single-value terms get a chart with one data series.

        Relies on private ViewerViser attributes (_plot_history_size, _scalar_buffers,
        _scalar_dirty, _plot_handles, _plot_folder).  If Newton refactors these internals
        this override should be updated or removed.
        """
        if not self._scalar_dirty:
            return
        from viser import uplot

        _SERIES_COLORS = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#f97316", "#06b6d4"]

        # Identify which term groups (base names) have at least one dirty component.
        dirty_bases: set[str] = {_scalar_base_name(name) for name in self._scalar_dirty}

        # Collect all known scalars grouped by base term name (insertion order preserved).
        all_groups: dict[str, list[str]] = {}
        for name in self._scalar_buffers:
            base = _scalar_base_name(name)
            all_groups.setdefault(base, []).append(name)

        for base_name, names in all_groups.items():
            if base_name not in dirty_bases:
                continue

            # Use only the filled portion of the buffer — no NaN padding — so the
            # chart visibly grows over time rather than appearing static at the right edge.
            bufs = [self._scalar_buffers.get(name) for name in names]
            n_actual = max((len(b) for b in bufs if b), default=0)
            if n_actual == 0:
                continue
            x = np.arange(n_actual, dtype=np.float64)
            ys = [np.array(list(b)[:n_actual], dtype=np.float64) if b else np.full(n_actual, np.nan) for b in bufs]
            data = (x, *ys)

            handle = self._plot_handles.get(names[0])
            if handle is None:
                folder_label = base_name.rsplit("/", 1)[-1]
                parent = self._live_plots_folder if self._live_plots_folder is not None else self._server.gui
                with parent:
                    folder = self._server.gui.add_folder(folder_label, expand_by_default=False)
                self._per_plot_folders[base_name] = folder
                if self._plot_folder is None:
                    self._plot_folder = folder

                series_list = [uplot.Series(label="step", show=False)]
                for i, name in enumerate(names):
                    suffix = name[len(base_name) :]  # "" for scalar, "[0]" etc. for vector
                    series_list.append(
                        uplot.Series(
                            label=suffix if suffix else folder_label,
                            stroke=_SERIES_COLORS[i % len(_SERIES_COLORS)],
                            width=1,
                        )
                    )
                with folder:
                    handle = self._server.gui.add_uplot(
                        data=data,
                        series=tuple(series_list),
                        scales={"x": uplot.Scale(time=False)},
                        aspect=1.33,
                    )
                for name in names:
                    self._plot_handles[name] = handle
            else:
                handle.data = data
        self._scalar_dirty.clear()


class ViserVisualizer(BaseVisualizer):
    """Viser web-based visualizer backed by Newton's ViewerViser."""

    marker_type = NewtonVisualizationMarkers

    def __init__(self, cfg: ViserVisualizerCfg):
        """Initialize Viser visualizer state.

        Args:
            cfg: Viser visualizer configuration.
        """
        super().__init__(cfg)
        from isaaclab.sim import SimulationContext

        simulation_context = SimulationContext.instance()
        if simulation_context is None:
            raise RuntimeError("ViserVisualizer requires an active SimulationContext.")
        self.marker_type = None if cfg.streaming_view else NewtonVisualizationMarkers
        if not cfg.streaming_view:
            from isaaclab_newton.cloner import NewtonReplicateContext

            self._newton_backend = simulation_context.get_or_create_backend(
                NewtonReplicateContext, simulation_context, clone_role="scene"
            )
            self._newton_backend.load_visual_shapes = True
        self.cfg: ViserVisualizerCfg = cfg
        self._viewer: NewtonViewerViser | None = None
        self._model: Any | None = None
        self._sim_time = 0.0
        self._active_record_path: str | None = None
        self._pending_camera_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        self._resolved_visible_env_ids: list[int] | None = None
        self._paused_rendering = False
        self._paused_simulation = False
        self._streaming: StreamingView | None = None

    def initialize(self, scene_data_provider: SceneDataProvider, clone_plan: ClonePlan) -> None:
        """Initialize viewer resources and bind scene data provider.

        Args:
            scene_data_provider: Scene data provider used to fetch model/state data.
            clone_plan: Shared plan describing every cloned scene row.
        """
        if self._is_initialized:
            logger.debug("[ViserVisualizer] initialize() called while already initialized.")
            return

        self._set_scene_data_provider(scene_data_provider, clone_plan)
        num_envs = len(clone_plan.env_ids)
        metadata = {"num_envs": num_envs}
        self._env_ids = self._compute_visualized_env_ids()
        self._model = None if self.cfg.streaming_view else self._newton_backend.get_model()

        self._active_record_path = self.cfg.record_to_viser
        self._resolved_visible_env_ids = resolve_visible_env_indices(self._env_ids, self.cfg.max_visible_envs, num_envs)
        self._create_viewer(record_to_viser=self.cfg.record_to_viser, metadata=metadata)
        num_visualized_envs = (
            len(self._resolved_visible_env_ids) if self._resolved_visible_env_ids is not None else num_envs
        )
        self._log_initialization_table(
            logger=logger,
            title="ViserVisualizer Configuration",
            rows=[
                ("eye", self.cfg.eye),
                ("lookat", self.cfg.lookat),
                ("focal_length", self.cfg.focal_length),
                ("num_visualized_envs", num_visualized_envs),
                ("bind_address", self.cfg.bind_address),
                ("display_address", self.cfg.display_address),
                ("port", self.cfg.port),
                ("record_to_viser", self.cfg.record_to_viser or "<none>"),
            ],
        )
        self._setup_streaming_view()
        self._is_initialized = True

    def step(self, dt: float) -> None:
        """Advance visualization by one simulation step.

        Args:
            dt: Simulation time-step in seconds.
        """
        if not self._is_initialized or self._viewer is None or self._scene_data_provider is None:
            return

        self._apply_pending_camera_pose()

        self._sim_time += dt

        # Skip all rendering when no browser clients are connected.
        server = getattr(self._viewer, "_server", None)
        has_clients = True
        if server is not None:
            get_clients = getattr(server, "get_clients", None)
            if callable(get_clients):
                has_clients = len(get_clients()) > 0

        if not has_clients:
            self._render_live_plots()  # still throttled internally; no-ops when no clients
            # No browser clients: skip compositing and pushing entirely.  If a
            # VideoRecorder calls render_tiled_rgb_array() it will compose on demand.
            return

        if self._paused_rendering:
            # Push streaming outside the pause-gate so it updates even when
            # rendering is paused, matching Rerun's behaviour.
            self._push_streaming_frame()
            return

        self._viewer.begin_frame(self._sim_time)
        try:
            # When streaming_view is active, skip the 3D Newton scene so the
            # background streaming composite is the only content visible.
            if not self.cfg.streaming_view:
                state = self._newton_backend.request_visualization_state(self._scene_data_provider)
                self._viewer.log_state(state)
                if self.cfg.enable_markers:
                    render_newton_visualization_markers(
                        self._viewer, self._resolved_visible_env_ids, num_envs=len(self._clone_plan.env_ids)
                    )
            self._render_live_plots()
            self._push_streaming_frame()
        finally:
            self._viewer.end_frame()

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

    def _push_streaming_frame(self) -> None:
        """Compose the streaming frame and push it to connected Viser clients."""
        composite = self.render_tiled_rgb_array()
        if composite is None:
            return
        # Letterbox to 16:9 so the composite isn't stretched when Viser fills
        # the browser canvas.  Black bars are added on whichever axis needs it.
        composite_display = _letterbox_16_9(composite)
        self._viewer._server.scene.set_background_image(composite_display, format="jpeg")

    def render_tiled_rgb_array(self) -> np.ndarray | None:
        """Return the composited streaming frame, all GT types side by side.

        This is the pre-letterbox composite, so a :class:`VideoRecorder` records the full content
        without black bars.

        Returns:
            ``uint8 (H, W, 3)`` composite array, or ``None`` when the streaming view is inactive or
            the camera produces no usable output.
        """
        return self._streaming.composite() if self._streaming is not None else None

    def close(self) -> None:
        """Close viewer resources and finalize optional recording."""
        if not self._is_initialized:
            return
        self._close_viewer(finalize_viser=bool(self.cfg.record_to_viser))
        self._streaming = None

        self._viewer = None
        self._is_initialized = False
        self._is_closed = True
        self._active_record_path = None
        self._pending_camera_pose = None

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

    def is_training_paused(self) -> bool:
        """Return whether simulation is paused from viewer controls."""
        return self._paused_simulation

    def is_rendering_paused(self) -> bool:
        """Return whether rendering is paused from viewer controls."""
        return self._paused_rendering

    def supports_live_plots(self) -> bool:
        """Viser backend supports live plots via :meth:`newton.Viewer.log_scalar` (uPlot sidebar charts)."""
        return True

    def add_live_plots(
        self,
        managers: dict,
        scalars: dict | None = None,
        term_names: dict[str, list[str]] | None = None,
        env_idx: int = 0,
    ) -> None:
        """Register managers for live plotting.

        Calls the base implementation to populate :attr:`_live_plot_sources`.  Checkboxes are
        created lazily on the first :meth:`_render_live_plots` call so each checkbox appears
        immediately above its plot charts in the Viser sidebar.

        Args:
            managers: Mapping of manager name to manager instance.
            scalars: Optional mapping of group name to a dict of ``{term_name: callable}``.
                Each callable must take no arguments and return a numeric value.
            term_names: Optional per-manager allowlists of term names to include.
            env_idx: Environment index to sample each step.  Defaults to ``0``.
        """
        super().add_live_plots(managers, scalars=scalars, term_names=term_names, env_idx=env_idx)

    def _render_live_plots(self) -> None:
        """Push manager-term scalars to the Viser viewer's per-term plot folders."""
        if self._viewer is None or not self._live_plot_sources:
            return
        # Skip when no browser clients are connected — nobody is watching the plots.
        server = getattr(self._viewer, "_server", None)
        if server is not None:
            get_clients = getattr(server, "get_clients", None)
            if callable(get_clients) and len(get_clients()) == 0:
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

    def _create_viewer(self, record_to_viser: str | None, metadata: dict | None = None) -> None:
        """Create Newton-backed Viser viewer and apply initial camera.

        Args:
            record_to_viser: Optional output path for viser recording.
            metadata: Optional metadata passed to viewer.
        """
        if self._model is None and not self.cfg.streaming_view:
            raise RuntimeError("Viser visualizer requires a Newton model.")

        self._viewer = NewtonViewerViser(
            port=self.cfg.port,
            bind_address=self.cfg.bind_address,
            label="Isaac Lab",
            verbose=False,
            share=self.cfg.share,
            record_to_viser=record_to_viser,
            metadata=metadata or {},
        )
        server = getattr(self._viewer, "_server", None)
        viewer_url = self._viewer.share_url or _viser_web_viewer_url(self.cfg.port, self.cfg.display_address)
        if self.cfg.verbose:
            print()
            self._log_viewer_url(
                "ViserVisualizer",
                viewer_url,
            )
        if self._model is not None:
            self._viewer.set_model(self._model)
            self._viewer.show_particles = self.cfg.show_particles
            self._viewer.set_visible_worlds(self._resolved_visible_env_ids)
            # Preserve simulation world positions (env_spacing) rather than adding viewer-side offsets.
            self._viewer.set_world_offsets((0.0, 0.0, 0.0))
        if server is not None:
            self._setup_isaaclab_sidebar(server)
        if self.cfg.open_browser:
            _open_viser_web_viewer(viewer_url)
        if self._model is not None:
            self._set_viser_camera_view(self._resolve_initial_camera_pose())
        self._sim_time = 0.0

    def _setup_isaaclab_sidebar(self, server) -> None:
        """Configure the Viser sidebar as the Isaac Lab panel.

        The panel is renamed to ``Isaac Lab``.  ``Live Plots`` and
        ``Visualization Markers`` are added as top-level collapsed folders
        directly inside the panel alongside the physics backend label.
        """
        viewer = self._viewer
        with contextlib.suppress(Exception):
            server.gui.set_panel_label("Isaac Lab")

            pause_rendering_btn = server.gui.add_button("Pause Rendering", color=None)

            @pause_rendering_btn.on_click
            def _(_):
                self._paused_rendering = not self._paused_rendering
                pause_rendering_btn.label = "Resume Rendering" if self._paused_rendering else "Pause Rendering"
                pause_rendering_btn.color = "orange" if self._paused_rendering else None

            pause_simulation_btn = server.gui.add_button("Pause Simulation", color=None)

            @pause_simulation_btn.on_click
            def _(_):
                self._paused_simulation = not self._paused_simulation
                pause_simulation_btn.label = "Resume Simulation" if self._paused_simulation else "Pause Simulation"
                pause_simulation_btn.color = "orange" if self._paused_simulation else None

            reset_button = server.gui.add_button("Reset Episode")

            @reset_button.on_click
            def _(_):
                self._reset_requested = True

            live_plots_folder = server.gui.add_folder("Live Plots", expand_by_default=False)
            viewer._live_plots_folder = live_plots_folder

            vis_folder = server.gui.add_folder("Visualization Markers", expand_by_default=False)

            _VIZ_FLAGS = [
                ("Joints", "show_joints"),
                ("Contacts", "show_contacts"),
                ("Center of Mass", "show_com"),
                ("Particles", "show_particles"),
                ("Visual", "show_visual"),
                ("Collision", "show_collision"),
                ("Springs", "show_springs"),
                ("Cloth", "show_triangles"),
                ("Inertia Boxes", "show_inertia_boxes"),
            ]
            with vis_folder:
                for label, attr in _VIZ_FLAGS:
                    if not hasattr(viewer, attr):
                        continue
                    cb = server.gui.add_checkbox(label, initial_value=getattr(viewer, attr, False))

                    def _make_cb(a=attr):
                        @cb.on_update
                        def _(event, _attr=a):
                            with contextlib.suppress(Exception):
                                setattr(viewer, _attr, event.target.value)

                    _make_cb()

    def _close_viewer(self, finalize_viser: bool = False) -> None:
        """Close viewer and log recording output when requested."""
        if self._viewer is None:
            return
        self._viewer.close()
        if finalize_viser and self._active_record_path:
            if os.path.exists(self._active_record_path):
                size = os.path.getsize(self._active_record_path)
                logger.info("[ViserVisualizer] Recording saved: %s (%s bytes)", self._active_record_path, size)
            else:
                logger.warning("[ViserVisualizer] Recording file not found: %s", self._active_record_path)
        self._viewer = None

    def _resolve_initial_camera_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Resolve initial camera pose from config."""
        return self._resolve_cfg_camera_pose("ViserVisualizer")

    def _try_apply_viser_camera_view(self, pose: tuple[tuple[float, float, float], tuple[float, float, float]]) -> bool:
        """Try applying camera pose to active viser clients.

        Returns:
            ``True`` if at least one client camera was updated, otherwise ``False``.
        """
        if self._viewer is None:
            return False
        server = getattr(self._viewer, "_server", None)
        get_clients = getattr(server, "get_clients", None) if server is not None else None
        if not callable(get_clients):
            return False

        clients = get_clients()

        client_iterable = clients.values() if isinstance(clients, dict) else clients
        cam_pos, cam_target = pose
        fov_radians = math.radians(self._focal_length_to_vertical_fov_degrees())
        applied = False
        for client in client_iterable:
            camera = getattr(client, "camera", None)
            if camera is None:
                continue
            if hasattr(camera, "fov"):
                camera.fov = fov_radians
                applied = True
            if hasattr(camera, "position"):
                camera.position = cam_pos
                applied = True
            if hasattr(camera, "look_at"):
                camera.look_at = cam_target
                applied = True
        return applied

    def _set_viser_camera_view(self, pose: tuple[tuple[float, float, float], tuple[float, float, float]]) -> None:
        """Apply or defer camera pose update depending on client readiness."""
        if self._try_apply_viser_camera_view(pose):
            self._pending_camera_pose = None
        else:
            self._pending_camera_pose = pose

    def _apply_pending_camera_pose(self) -> None:
        """Apply deferred camera pose once client cameras are available."""
        if self._pending_camera_pose is None:
            return
        if self._try_apply_viser_camera_view(self._pending_camera_pose):
            self._pending_camera_pose = None

    def set_camera_view(
        self, eye: tuple[float, float, float] | list[float], target: tuple[float, float, float] | list[float]
    ) -> None:
        """Set every connected client's camera eye/target.

        Args:
            eye: Camera eye position.
            target: Camera look-at target.
        """
        eye_t = (float(eye[0]), float(eye[1]), float(eye[2]))
        target_t = (float(target[0]), float(target[1]), float(target[2]))
        self._set_viser_camera_view((eye_t, target_t))
