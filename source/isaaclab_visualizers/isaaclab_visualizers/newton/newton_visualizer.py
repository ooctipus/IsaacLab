# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton visualizers and planned-camera presentation."""

from __future__ import annotations

import contextlib
import logging
import math
import os
import sys
from typing import TYPE_CHECKING

import numpy as np  # noqa: F401 — used in type hints and colorization helpers
import torch
import warp as wp

# On Linux without a display, set pyglet's headless option BEFORE importing newton.viewer
# so ViewerGL resolves to an EGL HeadlessWindow at class-definition time.  Only apply on
# headless Linux; on macOS/Windows or when DISPLAY is set the flag is left unset so
# interactive windows open normally.
if __import__("sys").platform not in ("win32", "darwin") and not __import__("os").environ.get("DISPLAY"):
    import pyglet as _pyglet_headless_init

    _pyglet_headless_init.options["headless"] = True
    del _pyglet_headless_init

from newton.viewer import ViewerGL
from pyglet.math import Vec3 as PygletVec3

from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.streaming_view import StreamingView

from isaaclab_visualizers.newton.newton_visualization_markers import (
    NewtonVisualizationMarkers,
    render_newton_visualization_markers,
)
from isaaclab_visualizers.newton_adapter import log_state_particles, resolve_visible_env_indices

from .newton_visualizer_cfg import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg, NewtonVisualizerCfg

logger = logging.getLogger(__name__)


def _newton_scalar_base_name(name: str) -> str:
    """Strip a trailing ``[N]`` component index from a scalar name to get the term base name."""
    if name.endswith("]") and "[" in name:
        bracket = name.rfind("[")
        if name[bracket + 1 : -1].isdigit():
            return name[:bracket]
    return name


if TYPE_CHECKING:
    from newton import State

    from isaaclab.cloner import ClonePlan
    from isaaclab.scene_data import SceneDataProvider


def _imgui_optional_checkbox(imgui, label: str, value: bool, available: bool, tip: str) -> bool:
    """Render a checkbox greyed out with a tooltip when *available* is False."""
    if not available:
        imgui.begin_disabled()
    _, new_val = imgui.checkbox(label, value)
    if not available:
        imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(tip)
        return value
    return new_val


# ---------------------------------------------------------------------------
# Newton viewer wrapper (adds Isaac Lab ImGui controls to Newton's GL viewer)
# ---------------------------------------------------------------------------


class _NewtonViewerUIMixin:
    """Mixin providing Isaac Lab UI for the Newton GL viewer wrapper."""

    CAMERA_SPEED_BOOST_MULTIPLIER = 2.0
    """Factor applied to :attr:`camera_speed` while the speed-boost modifier is held."""

    def _is_camera_speed_boost_active(self) -> bool:
        """Return whether the camera speed-boost modifier (Left/Right Shift) is held."""
        import pyglet

        return bool(self.is_key_down(pyglet.window.key.LSHIFT) or self.is_key_down(pyglet.window.key.RSHIFT))

    @property
    def camera_speed(self) -> float:
        """Keyboard camera translation speed [m/s], doubled while Shift is held."""
        base_speed = self._camera_speed
        if self._is_camera_speed_boost_active():
            return base_speed * self.CAMERA_SPEED_BOOST_MULTIPLIER
        return base_speed

    @camera_speed.setter
    def camera_speed(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("camera_speed must be finite and nonnegative")
        self._camera_speed = value

    def _register_isaaclab_ui_callbacks(self) -> None:
        """Register model-dependent Isaac Lab viewer controls."""
        self.register_ui_callback(self._render_training_controls, position="side")

    def _patch_scalar_plot_width(self) -> None:
        """Set up ImPlot and suppress Newton's built-in floating Plots window.

        Plots are rendered inline in the left panel by
        :meth:`~NewtonVisualizer._live_plots_panel_imgui` instead.
        """
        gui = self.gui

        # Initialise the optional ImPlot context once and link it to the active imgui context.
        try:
            from imgui_bundle import implot as _implot
        except ImportError:
            self._implot = self._implot_ctx = None
        else:
            self._implot_ctx = _implot.create_context()
            _implot.set_imgui_context(gui.ui.imgui.get_current_context())
            self._implot = _implot

        # Replace Newton's floating plots window with a no-op; rendering is in the panel.
        gui._render_scalar_plots = lambda: None

    def _patch_image_logger(self) -> None:
        """Patch the image logger for streaming view integration.

        Suppresses Newton's built-in ``draw_controls`` sidebar section
        (which labels itself "Logged Images (N)"), since :meth:`_draw_streaming_view_controls`
        provides the selection UI. Also overrides the initial window size to
        75 % of the available viewport area so the first-open panel is large.

        When no ``_image_logger`` attribute is present, this method returns immediately.
        """
        import types

        image_logger = getattr(self, "_image_logger", None)
        if image_logger is None:
            return

        # Suppress Newton's own "Logged Images" sidebar section.
        image_logger.draw_controls = lambda: None

        # Override draw() to open the floating panel sized to the composite aspect ratio.
        _orig_draw = type(image_logger).draw
        _viewer_ref = self  # capture for closure — used to read composite dimensions

        def _draw_large(self_logger: object) -> None:
            # Use our own flag (not Newton's entry.window_initialized) so Newton cannot
            # preempt our sizing by marking the window as initialized via the placeholder.
            if getattr(_viewer_ref, "_streaming_panel_needs_sizing", False):
                comp_w = getattr(_viewer_ref, "_streaming_composite_w", 0)
                comp_h = getattr(_viewer_ref, "_streaming_composite_h", 0)
                if comp_w > 0 and comp_h > 0:
                    from imgui_bundle import imgui as _imgui

                    vp = _imgui.get_main_viewport()
                    sidebar_w = float(self_logger._sidebar_width_px)
                    margin = 20.0
                    avail_w = max(320.0, vp.work_size.x - sidebar_w - 2.0 * margin)
                    avail_h = max(240.0, vp.work_size.y - 2.0 * margin)
                    title_h = 40.0
                    composite_wh = comp_w / comp_h

                    if avail_w / composite_wh + title_h <= avail_h:
                        w = avail_w
                        h = avail_w / composite_wh + title_h
                    else:
                        h = avail_h
                        w = (avail_h - title_h) * composite_wh

                    x = sidebar_w + margin + (avail_w - w) * 0.5
                    y = margin + (avail_h - h) * 0.5
                    # Cond_.always overrides whatever size Newton or imgui.ini gave the window.
                    _imgui.set_next_window_pos(_imgui.ImVec2(float(x), float(y)), _imgui.Cond_.always)
                    _imgui.set_next_window_size(_imgui.ImVec2(float(w), float(h)), _imgui.Cond_.always)
                    # Force the window uncollapsed — imgui.ini may have saved a collapsed state.
                    _imgui.set_next_window_collapsed(False, _imgui.Cond_.always)
                    _viewer_ref._streaming_panel_needs_sizing = False
            return _orig_draw(self_logger)

        image_logger.draw = types.MethodType(_draw_large, image_logger)

    def consume_reset_request(self) -> bool:
        """Return whether an episode reset was requested and clear the flag."""
        requested = self._reset_requested
        self._reset_requested = False
        return requested

    def _patch_viewer_panel(self) -> None:
        """Replace Newton's left panel with an IsaacLab-oriented layout.

        New section order:

        1. **Isaac Lab** (open) — physics backend, model info, training controls.
        2. **Live Plots** (closed) — injected when :meth:`~NewtonVisualizer.add_live_plots`
           is called.
        3. **Visualization Markers** (open) — Newton's debug overlays, renamed.
        4. **Rendering Options** (open) — VSync and renderer-specific options.
        5. **Wind** (closed) — only shown when ``viewer.wind`` is set.
        6. **Controls** (closed) — camera keyboard reference.
        7. **Selection API** (closed) — Newton's selection panel.

        The top-level Newton ``Pause / Step`` row is suppressed; pause/resume is
        handled by the IsaacLab training controls inside **Isaac Lab**.
        """
        import newton as nt

        gui = self.gui

        def _render_left_panel(_g=gui):
            if not _g.is_available:
                return

            viewer = _g._viewer
            imgui = _g.ui.imgui
            io = _g.ui.io
            s = _g.ui.dpi_scale
            nav_highlight_color = _g.ui.get_theme_color(imgui.Col_.nav_cursor, (1.0, 1.0, 1.0, 1.0))

            imgui.set_next_window_pos(imgui.ImVec2(10 * s, 10 * s), imgui.Cond_.first_use_ever)
            imgui.set_next_window_size(
                imgui.ImVec2(363 * s, io.display_size[1] - 20 * s),
                imgui.Cond_.first_use_ever,
            )
            panel_h = io.display_size[1] - 20 * s
            imgui.set_next_window_size_constraints(
                imgui.ImVec2(160 * s, panel_h),
                imgui.ImVec2(io.display_size[0], panel_h),
            )

            if not imgui.begin(f"Newton Viewer v{nt.__version__}"):
                imgui.end()
                return

            imgui.separator()

            # Layers panel callback (ViewerGL built-in, only shown with >1 layer).
            for callback in _g._ui_callbacks.get("panel", []):
                callback(imgui)

            # --- Simulation -------------------------------------------------
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Simulation"):
                imgui.separator()
                if viewer.model is not None:
                    axis_names = ["X", "Y", "Z"]
                    imgui.text(f"Up Axis: {axis_names[viewer.model.up_axis]}")
                    gravity = viewer.model.gravity.numpy()[0]
                    imgui.text(f"Gravity: ({gravity[0]:.2f}, {gravity[1]:.2f}, {gravity[2]:.2f})")
                imgui.separator()
                for callback in _g._ui_callbacks.get("side", []):
                    callback(imgui)

            # --- Streaming View ---------------------------------------------
            viewer._draw_streaming_view_controls()

            # --- Live Plots -------------------------------------------------
            live_plots_cb = getattr(viewer, "_live_plots_callback", None)
            if live_plots_cb is not None:
                live_plots_cb(imgui)

            # --- Visualization Markers -------------------------------------
            if viewer.model is not None:
                imgui.set_next_item_open(False, imgui.Cond_.appearing)
                if imgui.collapsing_header("Visualization Markers"):
                    imgui.separator()
                    renderer = getattr(viewer, "renderer", None)
                    _c, viewer.show_joints = imgui.checkbox("Show Joints", viewer.show_joints)
                    if viewer.show_joints and renderer is not None and hasattr(renderer, "joint_scale"):
                        _, renderer.joint_scale = imgui.slider_float("Joint Scale", renderer.joint_scale, 0.25, 5.0)
                    _c, viewer.show_contacts = imgui.checkbox("Show Contacts", viewer.show_contacts)
                    if viewer.show_contacts and renderer is not None:
                        if hasattr(renderer, "arrow_length_scale"):
                            _, renderer.arrow_length_scale = imgui.slider_float(
                                "Contact Length", renderer.arrow_length_scale, 0.25, 5.0
                            )
                        if hasattr(renderer, "arrow_scale"):
                            _, renderer.arrow_scale = imgui.slider_float(
                                "Contact Width", renderer.arrow_scale, 0.25, 5.0
                            )
                    _model = viewer.model
                    _has_particles = _model is not None and int(getattr(_model, "particle_count", 0)) > 0
                    _has_springs = _model is not None and int(getattr(_model, "spring_count", 0)) > 0
                    _has_cloth = _model is not None and int(getattr(_model, "tri_count", 0)) > 0
                    viewer.show_particles = _imgui_optional_checkbox(
                        imgui,
                        "Show Particles",
                        viewer.show_particles,
                        _has_particles,
                        "No particle bodies in this environment",
                    )
                    viewer.show_springs = _imgui_optional_checkbox(
                        imgui,
                        "Show Springs",
                        viewer.show_springs,
                        _has_springs,
                        "No spring constraints in this environment",
                    )
                    _c, viewer.show_com = imgui.checkbox("Show Center of Mass", viewer.show_com)
                    if viewer.show_com and renderer is not None and hasattr(renderer, "com_scale"):
                        _, renderer.com_scale = imgui.slider_float("COM Scale", renderer.com_scale, 0.25, 5.0)
                    viewer.show_triangles = _imgui_optional_checkbox(
                        imgui,
                        "Show Cloth",
                        viewer.show_triangles,
                        _has_cloth,
                        "No cloth/triangle meshes in this environment",
                    )
                    _c, viewer.show_collision = imgui.checkbox("Show Collision", viewer.show_collision)
                    if renderer is not None and hasattr(renderer, "draw_edges"):
                        _c, renderer.draw_edges = imgui.checkbox("Show Edges", renderer.draw_edges)
                    sdf_margin_mode = getattr(viewer, "sdf_margin_mode", None)
                    SDFMarginMode = getattr(type(viewer), "SDFMarginMode", None)
                    if sdf_margin_mode is not None and SDFMarginMode is not None:
                        _sdf_labels = ["Off", "Margin", "Margin + Gap"]
                        _, new_sdf_idx = imgui.combo("Gap + Margin", int(sdf_margin_mode), _sdf_labels)
                        viewer.sdf_margin_mode = SDFMarginMode(new_sdf_idx)
                        if viewer.sdf_margin_mode != SDFMarginMode.OFF and renderer is not None:
                            _, renderer.wireframe_line_width = imgui.slider_float(
                                "Wireframe Width (px)", renderer.wireframe_line_width, 0.5, 5.0
                            )
                    _c, viewer.show_visual = imgui.checkbox("Show Visual", viewer.show_visual)
                    _c, viewer.show_inertia_boxes = imgui.checkbox("Show Inertia Boxes", viewer.show_inertia_boxes)
                    from isaaclab.sim import SimulationContext

                    sim = SimulationContext.instance()
                    marker_groups = () if sim is None else sim.vis_marker_registry.get_groups()
                    for marker in marker_groups:
                        name = marker.cfg.prim_path.rsplit("/", 1)[-1].replace("_", " ")
                        changed, visible = imgui.checkbox(f"Show {name}##{marker.group_id}", marker.is_visible())
                        if changed:
                            marker.set_visibility(visible)

            # --- Rendering Options ------------------------------------------
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if imgui.collapsing_header("Rendering Options"):
                imgui.separator()
                _c, viewer.vsync = imgui.checkbox("VSync", viewer.vsync)
                for callback in _g._ui_callbacks.get("rendering", []):
                    callback(imgui)

            # --- Wind -------------------------------------------------------
            wind = getattr(viewer, "wind", None)
            if wind is not None:
                imgui.set_next_item_open(False, imgui.Cond_.once)
                if imgui.collapsing_header("Wind"):
                    imgui.separator()
                    changed, wind.amplitude = imgui.slider_float("Wind Amplitude", wind.amplitude, -2.0, 2.0, "%.2f")
                    changed, wind.period = imgui.slider_float("Wind Period", wind.period, 1.0, 30.0, "%.2f")
                    changed, wind.frequency = imgui.slider_float("Wind Frequency", wind.frequency, 0.1, 5.0, "%.2f")
                    direction = [wind.direction[0], wind.direction[1], wind.direction[2]]
                    changed, direction = imgui.slider_float3("Wind Direction", direction, -1.0, 1.0, "%.2f")
                    if changed:
                        wind.direction = direction

            # --- Controls ---------------------------------------------------
            imgui.set_next_item_open(False, imgui.Cond_.appearing)
            if imgui.collapsing_header("Controls"):
                imgui.separator()
                _g._render_camera_info()
                imgui.separator()
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*nav_highlight_color))
                imgui.text("Controls:")
                imgui.pop_style_color()
                imgui.text("WASD - Move camera")
                imgui.text("Shift + WASD - Move camera 2x speed")
                imgui.text("QE - Pan up/down")
                imgui.text("Left Click - Look around")
                imgui.text("Right Click - Pick and drag objects")
                imgui.text("Middle Click - Orbit")
                imgui.text("Shift + Middle Click - Pan")
                imgui.text("Ctrl + Middle Click - Dolly")
                imgui.text("Scroll - Dolly")
                imgui.text("Ctrl + Scroll - FOV zoom")
                imgui.text("Space - Pause/Resume Rendering")
                imgui.text(". - Step one frame (when paused)")
                imgui.text("H - Toggle UI")
                imgui.text("F - Frame camera around model")

            # --- Selection API ----------------------------------------------
            _g._render_selection_panel()

            imgui.end()

        gui._render_left_panel = _render_left_panel

    def _render_training_controls(self, imgui):
        """Render Isaac Lab training control widgets inside the Isaac Lab panel section."""
        pause_label = "Resume Simulation" if self._paused_training else "Pause Simulation"
        if imgui.button(pause_label):
            self._paused_training = not self._paused_training

        # Newton's Space handler toggles this same flag directly.
        rendering_label = "Resume Rendering" if self._paused else "Pause Rendering"
        if imgui.button(rendering_label):
            self._paused = not self._paused

        if imgui.button("Reset Episode"):
            self._reset_requested = True

        imgui.text("Visualizer Update Frequency")
        current_frequency = self._update_frequency
        changed, new_frequency = imgui.slider_int(
            "##VisualizerUpdateFreq", current_frequency, 1, 20, f"Every {current_frequency} frames"
        )
        if changed:
            self._update_frequency = new_frequency

        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Controls visualizer update frequency\nlower values -> more responsive visualizer but slower"
                " training\nhigher values -> less responsive visualizer but faster training"
            )

    def _draw_streaming_view_controls(self) -> None:
        """Render the streaming image panel selector in the HUD sidebar."""
        image_logger = getattr(self, "_image_logger", None)
        if image_logger is None:
            return

        if not image_logger._images:
            return

        imgui = self.ui.imgui
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Streaming View"):
            return

        names = list(image_logger._images.keys())
        # Display "Open" as the action label regardless of the underlying image key.
        display_items = ["Hide"] + ["Open" for _ in names]
        if image_logger._selected is not None and image_logger._selected in names:
            current = names.index(image_logger._selected) + 1
        else:
            current = 0

        imgui.text("Toggle")
        changed, new_idx = imgui.combo("##streaming_view", current, display_items)
        if changed:
            new_selected = None if new_idx == 0 else names[new_idx - 1]
            image_logger._selected = new_selected
            # Signal the image-logger draw hook to resize to the composite aspect ratio.
            if new_selected is not None:
                entry = image_logger._images.get(new_selected)
                if entry is not None:
                    entry.window_initialized = False
                # Set our flag so _draw_large applies correct aspect-ratio sizing.
                viewer = getattr(self, "_viewer", None) or self
                viewer._streaming_panel_needs_sizing = True

    def _coerce_color3(self, color) -> tuple[float, float, float]:
        """Normalize color values from imgui/renderer into an RGB tuple."""
        if hasattr(color, "x") and hasattr(color, "y") and hasattr(color, "z"):
            return (float(color.x), float(color.y), float(color.z))
        return (float(color[0]), float(color[1]), float(color[2]))

    def _log_particles(self, state):
        """Log particles from the requested scene-data publication."""
        log_state_particles(self, state)


class NewtonViewerGL(_NewtonViewerUIMixin, ViewerGL):
    """Wrapper around Newton's ViewerGL with training/rendering pause controls."""

    def __init__(self, *args, metadata: dict | None = None, update_frequency: int = 1, **kwargs):
        """Initialize Newton viewer wrapper state.

        Args:
            *args: Positional arguments forwarded to ``ViewerGL``.
            metadata: Optional metadata shown in viewer panels.
            update_frequency: Viewer refresh cadence in simulation frames.
            **kwargs: Keyword arguments forwarded to ``ViewerGL``.
        """
        super().__init__(*args, **kwargs)
        self._paused_training = False
        self._reset_requested = False
        self._metadata = metadata or {}
        self._update_frequency = update_frequency
        self.particle_color: tuple[float, float, float] | None = None
        self._particle_color_buffer: wp.array | None = None
        self._particle_color_buffer_count = 0
        self._particle_color_buffer_value: tuple[float, float, float] | None = None
        self._live_plots_callback = None

        with contextlib.suppress(AttributeError):
            self._patch_scalar_plot_width()
            self._patch_viewer_panel()
            self._patch_image_logger()

        self.register_ui_callback(self._render_training_controls, position="side")

    def is_training_paused(self) -> bool:
        """Return whether simulation is paused by viewer controls."""
        return self._paused_training

    def is_rendering_paused(self) -> bool:
        """Return whether rendering is paused by viewer controls.

        Mirrors ``self._paused`` directly since the Newton viewer's Space key handler toggles it
        in-place, outside the Isaac Lab "Pause Rendering" button.
        """
        return self._paused

    def on_key_press(self, symbol, modifiers):
        """Forward key presses unless UI is currently capturing input."""
        if self.ui.is_capturing():
            return
        super().on_key_press(symbol, modifiers)

    def _particle_color_array(self, count: int) -> wp.array:
        """Return a cached Warp color array for Newton's particle point batch."""
        color = self._coerce_color3(self.particle_color)
        if (
            self._particle_color_buffer is None
            or self._particle_color_buffer_count != count
            or self._particle_color_buffer_value != color
        ):
            self._particle_color_buffer = wp.full(
                shape=count,
                value=wp.vec3(*color),
                dtype=wp.vec3,
                device=self.device,
            )
            self._particle_color_buffer_count = count
            self._particle_color_buffer_value = color
        return self._particle_color_buffer

    def _particle_color_update_array(self, name: str, count: int) -> wp.array | None:
        """Return particle colors only when Newton needs the GL color buffer refreshed."""
        obj = self.objects.get(name)
        capacity = obj.num_instances if obj is not None else 0
        if (
            obj is None
            or count > capacity
            or self._particle_color_buffer_value != self._coerce_color3(self.particle_color)
        ):
            return self._particle_color_array(max(count, capacity))
        return None

    def log_points(self, name, points, radii=None, colors=None, hidden=False):
        """Apply configured model-particle appearance while preserving Newton's point logging.

        The configured particle color only applies to Newton's canonical
        ``/model/particles`` point batch. User-defined point clouds retain the
        colors provided by their own ``log_points`` calls.
        """
        if name != "/model/particles" or points is None or self.particle_color is None:
            return super().log_points(name, points, radii, colors, hidden)

        colors = self._particle_color_update_array(name, len(points))
        return super().log_points(name, points, radii, colors, hidden)


class NewtonVisualizer(BaseVisualizer):
    """Internal base class for Newton visualizers.

    Implements the shared ``initialize / step / close`` lifecycle. Subclasses
    override the viewer and presentation hooks:

    - :meth:`_create_viewer` — instantiate the correct Newton viewer class.
    - :meth:`_apply_viewer_post_init` — apply backend-specific post-init settings.
    - :meth:`_apply_camera_pose` — set camera position with the backend's API.
    - :meth:`_apply_camera_focal_length` — set or defer FOV.
    - :meth:`_pump_paused` — keep the event loop alive while simulation is paused.
    - :meth:`_pre_step` — per-frame hook before the render block (e.g. deferred FOV).
    - :meth:`render_rgb_array` — capture and return the current frame.
    - :meth:`_log_streaming_image` — push the composited streaming frame into the viewer image panel.
    - :meth:`_uses_streaming_view` — whether the streaming view is active.

    Do not instantiate this class directly; use :class:`NewtonGLVisualizer` or
    :class:`NewtonRTXVisualizer`.
    """

    marker_type = NewtonVisualizationMarkers

    class _ViewerPickingBinding:
        """Stable Newton-manager callback for viewer picking.

        CUDA graphs record picking arrays by address, so closing the window
        neutralizes and retains them until the captured graph is gone.
        """

        def __init__(self) -> None:
            self._viewer: NewtonViewerGL | None = None
            self._retained_picking = None

        def bind(self, viewer: NewtonViewerGL) -> None:
            """Bind picking to the current viewer model."""
            self._viewer = viewer
            self._retained_picking = None

        def apply(self, state: State) -> None:
            """Apply picking while the viewer is active."""
            if self._viewer is None:
                # Host callbacks do not run during graph replay, so reaching
                # this branch means captured inputs are no longer needed.
                self._retained_picking = None
                return
            self._viewer.apply_forces(state)

        def deactivate(self) -> None:
            """Make captured picking inert while preserving its inputs."""
            viewer = self._viewer
            if viewer is None:
                return

            picking = getattr(viewer, "picking", None)
            if picking is not None:
                viewer.picking_enabled = False
                picking.release()

            self._retained_picking = picking
            self._viewer = None

    def __init__(self, cfg: NewtonVisualizerCfg):
        """Initialize shared Newton visualizer state.

        Args:
            cfg: Newton visualizer configuration.
        """
        super().__init__(cfg)
        from isaaclab_newton.cloner import NewtonReplicateContext

        from isaaclab.sim import SimulationContext

        simulation_context = SimulationContext.instance()
        if simulation_context is None:
            raise RuntimeError("Newton visualizers require an active SimulationContext.")
        self._newton_backend = simulation_context.get_or_create_backend(
            NewtonReplicateContext, simulation_context, clone_role="scene"
        )
        self._newton_backend.load_visual_shapes = True
        self.cfg: NewtonVisualizerCfg = cfg
        self._viewer: NewtonViewerGL | None = None
        self._sim_time = 0.0
        self._step_counter = 0
        self._runtime_headless: bool = False
        self._model = None
        self._state = None
        self._update_frequency = cfg.update_frequency
        self._last_camera_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        self._resolved_visible_env_ids: list[int] | None = None
        self._streaming: StreamingView | None = None
        self._viewer_picking_binding = self._ViewerPickingBinding()
        self._picking_enabled = False
        self._live_plots_manager_visible: dict[str, bool] = {}

    @property
    def visual_material_writer(self):
        """Return the shared Newton model color-writer factory."""
        return self._newton_backend.create_visual_material_writer

    # ------------------------------------------------------------------
    # Shared lifecycle
    # ------------------------------------------------------------------

    def initialize(self, scene_data_provider: SceneDataProvider, clone_plan: ClonePlan) -> None:
        """Initialize viewer resources and bind scene data provider.

        Args:
            scene_data_provider: Scene data provider used to fetch model/state data.
            clone_plan: Shared plan describing every cloned scene row.
        """

        if self._is_initialized:
            logger.debug("[%s] initialize() called while already initialized.", type(self).__name__)
            return

        self._set_scene_data_provider(scene_data_provider, clone_plan)
        picking_supported = self._newton_backend.supports_rigid_body_force_input()
        num_envs = len(clone_plan.env_ids)
        metadata = {"num_envs": num_envs}
        self._env_ids = self._compute_visualized_env_ids()
        self._resolved_visible_env_ids = resolve_visible_env_indices(self._env_ids, self.cfg.max_visible_envs, num_envs)
        self._model = self._newton_backend.get_model()

        runtime_headless = self.cfg.headless or (
            sys.platform not in ("win32", "darwin") and not os.environ.get("DISPLAY")
        )
        if runtime_headless and not self.cfg.headless:
            # print() instead of logger.warning(): the kitless launch path does not
            # install a logging handler, so this user-facing notice would be swallowed.
            print(
                "[WARNING] [NewtonVisualizer] No display found (DISPLAY is unset); the Newton viewer runs"
                " headless via EGL and no window will open. Run from a session with a display (or set"
                " DISPLAY, e.g. 'export DISPLAY=:0') to see the viewer."
            )
        self._runtime_headless = runtime_headless

        # Use pyglet's EGL headless backend when requested or when no Linux X display is available.
        # NOTE: this call is only effective when ``DISPLAY`` is unset on Linux.  When a display
        # is present, importing ``ViewerGL`` at module-import time
        # already initialised pyglet (and resolved the ``Window`` class), so setting
        # ``pyglet.options["headless"]`` here is a no-op.  In that situation ``cfg.headless=True``
        # has no effect and a real windowed viewer is created.  To guarantee headless behaviour
        # when a display is present, unset DISPLAY before importing this module.
        if runtime_headless:
            import pyglet

            pyglet.options["headless"] = True

        self._picking_enabled = self.cfg.enable_picking and picking_supported and not runtime_headless
        self._viewer = self._create_viewer(runtime_headless, metadata)

        if self._viewer is not None:
            self._viewer.set_model(self._model)
            if self._picking_enabled:
                # Keep Newton's public force path scoped to picking for this integration.
                self._viewer.wind = None
            self._viewer.set_visible_worlds(self._resolved_visible_env_ids)
            self._viewer.set_world_offsets(self.cfg.world_spacing)
            self._apply_camera_focal_length()
            initial_pose = self._resolve_initial_camera_pose()
            self._apply_camera_pose(initial_pose)
            self._viewer._paused = False

            self._apply_model_visualization_options()
            self._viewer.picking_enabled = self._picking_enabled

            self._apply_viewer_post_init()

        self._setup_streaming_view()

        num_visualized_envs = (
            len(self._resolved_visible_env_ids) if self._resolved_visible_env_ids is not None else num_envs
        )
        try:
            current_eye = tuple(float(x) for x in self._viewer.camera.pos) if self._viewer is not None else self.cfg.eye
        except AttributeError:
            current_eye = self.cfg.eye
        self._log_initialization_table(
            logger=logger,
            title=f"{type(self).__name__} Configuration",
            rows=[
                ("eye", current_eye),
                ("lookat", self._last_camera_pose[1] if self._last_camera_pose else self.cfg.lookat),
                ("focal_length", self.cfg.focal_length),
                ("streaming_view", self.cfg.streaming_view),
                ("streaming_gt_types", list(self.cfg.streaming_gt_types)),
                ("num_visualized_envs", num_visualized_envs),
                ("headless", self.cfg.headless),
                ("show_particles", self.cfg.show_particles),
                ("enable_picking", self._picking_enabled),
            ],
        )
        if self._viewer is not None and self._picking_enabled:
            self._viewer_picking_binding.bind(self._viewer)
            self._newton_backend.register_state_force_callback(self._viewer_picking_binding.apply)
        if self._viewer is not None and self.cfg.enable_picking and not picking_supported:
            logger.info(
                "[NewtonVisualizer] Object dragging is disabled because the active physics solver does not support"
                " rigid-body force input."
            )
        self._is_initialized = True

    def _apply_model_visualization_options(self) -> None:
        """Apply configured options reset by Newton model changes."""
        if self._viewer is None:
            return
        self._viewer.show_joints = self.cfg.show_joints
        self._viewer.show_contacts = self.cfg.show_contacts
        self._viewer.show_collision = self.cfg.show_collision
        self._viewer.show_springs = self.cfg.show_springs
        self._viewer.show_inertia_boxes = self.cfg.show_inertia_boxes
        self._viewer.show_com = self.cfg.show_com
        self._viewer.show_particles = self.cfg.show_particles

    def step(self, dt: float) -> None:
        """Advance visualization by one simulation step.

        Args:
            dt: Simulation time-step in seconds.
        """
        if not self._is_initialized or self._is_closed:
            return

        self._sim_time += dt
        self._step_counter += 1

        if self._runtime_headless or self._viewer is None:
            return

        update_frequency = self._viewer._update_frequency if self._viewer else self._update_frequency
        if self._step_counter % update_frequency != 0:
            return

        self._pre_step()
        num_envs = len(self._clone_plan.env_ids)

        if not self._viewer.is_paused():
            self._state = self._newton_backend.request_visualization_state(self._scene_data_provider)
            self._viewer.begin_frame(self._sim_time)
            try:
                if self._state is not None:
                    body_q = getattr(self._state, "body_q", None)
                    if hasattr(body_q, "shape") and body_q.shape[0] == 0:
                        return
                    self._viewer.log_state(self._state)
                    if self.cfg.enable_markers:
                        render_newton_visualization_markers(
                            self._viewer, self._resolved_visible_env_ids, num_envs=num_envs
                        )
                    self._log_streaming_image()
                    self._render_live_plots()
            finally:
                self._viewer.end_frame()
                if not self._viewer.is_running():
                    self._viewer_picking_binding.deactivate()
        else:
            self._pump_paused()
            if not self._viewer.is_running():
                self._viewer_picking_binding.deactivate()

    def consume_reset_request(self) -> bool:
        """Return whether an episode reset was requested and clear the flag."""
        if self._viewer is not None:
            return self._viewer.consume_reset_request()
        return False

    def reset(self, soft: bool = False) -> None:
        """Rebind viewer resources after a hard Newton model reset."""
        if soft or not self._picking_enabled or not self._is_initialized or self._is_closed:
            return

        model = self._newton_backend.get_model()
        if model is self._model:
            return
        self._model = model
        self._state = self._newton_backend.request_visualization_state(self._scene_data_provider)
        if self._viewer is not None:
            self._viewer.set_model(self._model)
            if self._picking_enabled:
                self._viewer.wind = None
            self._viewer._register_isaaclab_ui_callbacks()
            self._viewer.set_visible_worlds(self._resolved_visible_env_ids)
            self._viewer.set_world_offsets(self.cfg.world_spacing)
            self._apply_model_visualization_options()
            self._viewer.picking_enabled = self._picking_enabled
            if self._picking_enabled:
                self._viewer_picking_binding.bind(self._viewer)

    def close(self) -> None:
        """Release viewer resources."""
        if self._is_closed:
            return
        if self._picking_enabled:
            # Keep the stable callback registered: captured graphs replay its
            # now-neutral device inputs without retaining the viewer.
            self._viewer_picking_binding.deactivate()
        if self._viewer is not None:
            try:
                self._viewer.close()
            finally:
                self._viewer = None
        self._streaming = None
        self._is_closed = True

    def is_running(self) -> bool:
        """Return whether the visualizer should continue stepping."""
        if not self._is_initialized or self._is_closed:
            return False
        if self._viewer is None:
            return False
        return self._viewer.is_running()

    def supports_live_plots(self) -> bool:
        """The base visualizer does not advertise a live-plot panel."""
        return False

    def is_training_paused(self) -> bool:
        """Return whether training is paused from viewer controls."""
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.is_training_paused()

    def is_rendering_paused(self) -> bool:
        """Return whether rendering is paused from viewer controls."""
        if not self._is_initialized or self._viewer is None:
            return False
        return self._viewer.is_rendering_paused()

    def set_camera_view(
        self, eye: tuple[float, float, float] | list[float], target: tuple[float, float, float] | list[float]
    ) -> None:
        """Set active viewer camera eye/target.

        Args:
            eye: Camera eye position.
            target: Camera look-at target.
        """
        eye_t = (float(eye[0]), float(eye[1]), float(eye[2]))
        target_t = (float(target[0]), float(target[1]), float(target[2]))
        self.cfg.eye = eye_t
        self.cfg.lookat = target_t
        self._apply_camera_pose((eye_t, target_t))

    # ------------------------------------------------------------------
    # Hook methods — override in subclasses
    # ------------------------------------------------------------------

    def _create_viewer(self, runtime_headless: bool, metadata: dict) -> NewtonViewerGL | None:
        """Create and return the backend viewer instance.

        Args:
            runtime_headless: Whether to run without a display.
            metadata: Metadata dict passed to the viewer constructor.
        """
        raise NotImplementedError

    def _apply_viewer_post_init(self) -> None:
        """Apply backend-specific settings after the viewer is constructed."""

    def _apply_camera_pose(
        self,
        pose: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> None:
        """Apply camera eye/target pose to the viewer.

        Args:
            pose: ``(eye, lookat)`` tuple.
        """
        raise NotImplementedError

    def _apply_camera_focal_length(self) -> None:
        """Apply cfg focal length to the viewer camera."""
        raise NotImplementedError

    def _pump_paused(self) -> None:
        """Keep the event loop alive while simulation is paused without advancing state."""
        raise NotImplementedError

    def _pre_step(self) -> None:
        """Per-frame hook called before the render block. No-op by default."""

    def render_rgb_array(self) -> np.ndarray | None:
        """Return the latest RGB frame as a uint8 array with shape ``(H, W, 3)``."""
        raise NotImplementedError

    def _log_streaming_image(self) -> None:
        """Push the composited streaming frame into the viewer image panel."""

    def _uses_streaming_view(self) -> bool:
        """Return whether the streaming camera view is active."""
        return bool(self.cfg.streaming_view)

    # ------------------------------------------------------------------
    # Shared internals
    # ------------------------------------------------------------------

    def _resolve_initial_camera_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Resolve initial camera pose from config or USD camera path."""
        return self._resolve_cfg_camera_pose(type(self).__name__)

    def _setup_streaming_view(self) -> None:
        """Resolve the camera the streaming panel shows."""
        from isaaclab.sim import SimulationContext

        if not self._uses_streaming_view():
            return
        sim = SimulationContext.instance()
        self._streaming = StreamingView(
            self.cfg,
            sim.get_camera_sensors(),
            visible_env_ids=self._resolved_visible_env_ids,
            target_aspect=self.cfg.window_width / max(1, self.cfg.window_height),
        )


# ---------------------------------------------------------------------------
# GL backend
# ---------------------------------------------------------------------------


class NewtonGLVisualizer(NewtonVisualizer):
    """Newton OpenGL rasterizer visualizer for Isaac Lab.

    Wraps :class:`NewtonViewerGL` — fast local window with the full Isaac Lab
    feature set: streaming camera panel, particle color override, live scalar and array
    plots (via Newton's ImGui sidebar), and :meth:`render_rgb_array` support.

    Use :class:`NewtonGLVisualizerCfg` (type ``"newton_gl"``) to select this backend.
    """

    def __init__(self, cfg: NewtonGLVisualizerCfg):
        """Initialize Newton GL visualizer.

        Args:
            cfg: GL visualizer configuration.
        """
        super().__init__(cfg)
        self.cfg: NewtonGLVisualizerCfg = cfg

        # Camera-selector dropdown state — populated in _build_streaming_camera_dropdown().
        # _streaming_camera_choices indexes the same order as the combo's scene camera names.
        self._streaming_camera_choices: list[str] = []
        self._streaming_camera_selection: int = 0

    def initialize(self, scene_data_provider: SceneDataProvider, clone_plan: ClonePlan) -> None:
        """Initialize the GL visualizer and build the streaming camera dropdown.

        Args:
            scene_data_provider: Provider for scene data and camera sensors.
            clone_plan: Shared plan describing every cloned scene row.
        """
        super().initialize(scene_data_provider, clone_plan)
        if self._is_initialized:
            self._build_streaming_camera_dropdown()

    def _build_streaming_camera_dropdown(self) -> None:
        """Populate the camera-selector combo from the streaming view's sources.

        Patches :meth:`~_NewtonViewerUIMixin._draw_streaming_view_controls` on the viewer instance to
        inject the combo into the existing sidebar section.
        """
        if self._streaming is None:
            return
        self._streaming_camera_choices = list(self._streaming.scene_cameras)
        self._streaming_camera_selection = self._streaming_camera_choices.index(self._camera_choice_name())
        if self._streaming_camera_choices and self._viewer is not None:
            self._patch_streaming_camera_controls()

    def _camera_choice_name(self) -> str:
        """Sensor name of the camera currently streamed, as it appears in the combo."""
        return next(name for name, cam in self._streaming.scene_cameras.items() if cam is self._streaming.camera)

    def _patch_streaming_camera_controls(self) -> None:
        """Inject the source-camera combo into the viewer's streaming sidebar section.

        Monkey-patches :meth:`_draw_streaming_view_controls` on the *viewer
        instance* (not the class) so that the closure captures ``self``
        (the visualizer) without modifying any Newton viewer code.
        """
        import types

        viewer = self._viewer
        _orig = type(viewer)._draw_streaming_view_controls
        _vis = self  # closure reference to the visualizer

        def _patched(self_viewer):
            # Re-implement the whole Streaming View accordion so Source Camera
            # can be rendered inside it (calling _orig first would close the
            # accordion before we could inject additional content).
            image_logger = getattr(self_viewer, "_image_logger", None)
            if image_logger is None or not image_logger._images:
                return

            imgui = self_viewer.ui.imgui
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if not imgui.collapsing_header("Streaming View"):
                return

            # Open / Hide image panel combo.
            names = list(image_logger._images.keys())
            display_items = ["Hide"] + ["Open" for _ in names]
            if image_logger._selected is not None and image_logger._selected in names:
                current = names.index(image_logger._selected) + 1
            else:
                current = 0
            imgui.text("Toggle")
            changed, new_idx = imgui.combo("##streaming_view", current, display_items)
            if changed:
                new_selected = None if new_idx == 0 else names[new_idx - 1]
                image_logger._selected = new_selected
                if new_selected is not None:
                    entry = image_logger._images.get(new_selected)
                    if entry is not None:
                        entry.window_initialized = False
                    # Signal _draw_large to apply aspect-ratio sizing.
                    self_viewer._streaming_panel_needs_sizing = True

            # Source Camera selector — inside the accordion, below Open/Hide.
            if _vis._streaming_camera_choices:
                imgui.separator()
                imgui.text("Source Camera")
                changed, new_cam_idx = imgui.combo(
                    "##streaming_cam_src",
                    _vis._streaming_camera_selection,
                    _vis._streaming_camera_choices,
                )
                if changed:
                    _vis._switch_streaming_camera(new_cam_idx)
                if imgui.is_item_hovered() and _vis._streaming is not None and _vis._streaming.camera is not None:
                    imgui.set_tooltip(_vis._streaming.camera.cfg.prim_path)

        viewer._draw_streaming_view_controls = types.MethodType(_patched, viewer)

    def _switch_streaming_camera(self, new_idx: int) -> None:
        """Stream from the combo selection at ``new_idx``.

        Args:
            new_idx: Index into :attr:`_streaming_camera_choices`.
        """
        if new_idx == self._streaming_camera_selection or self._streaming is None:
            return
        self._streaming_camera_selection = new_idx
        choice = self._streaming_camera_choices[new_idx]
        self._streaming.select(self._streaming.scene_cameras[choice])

        # Clear the panel's window_initialized flag so _draw_large re-sizes it to the new camera's
        # grid aspect ratio the next time it opens.
        if self._viewer is not None:
            self._viewer._streaming_composite_h = 0
            self._viewer._streaming_composite_w = 0
            image_logger = getattr(self._viewer, "_image_logger", None)
            if image_logger is not None:
                entry = image_logger._images.get("Streaming View")
                if entry is not None:
                    entry.window_initialized = False

    def _create_viewer(self, runtime_headless: bool, metadata: dict) -> NewtonViewerGL:
        return NewtonViewerGL(
            width=self.cfg.window_width,
            height=self.cfg.window_height,
            headless=runtime_headless,
            metadata=metadata,
            update_frequency=self.cfg.update_frequency,
        )

    def supports_live_plots(self) -> bool:
        """Newton GL supports live scalar/array plots via the ImGui sidebar."""
        return True

    def add_live_plots(
        self,
        managers: dict,
        scalars: dict | None = None,
        term_names: dict[str, list[str]] | None = None,
        env_idx: int = 0,
    ) -> None:
        """Register managers for live plotting and add per-manager sidebar toggles.

        Calls the base implementation to populate :attr:`_live_plot_sources`, then registers
        the Live Plots collapsing section in the Newton viewer sidebar.

        Args:
            managers: Mapping of manager name to manager instance.
            scalars: Optional mapping of group name to a dict of ``{term_name: callable}``.
            term_names: Optional per-manager allowlists of term names to include.
            env_idx: Environment index to sample each step.  Defaults to ``0``.
        """
        super().add_live_plots(managers, scalars=scalars, term_names=term_names, env_idx=env_idx)
        if not self._live_plot_sources or self._viewer is None:
            return
        self._live_plots_manager_visible = {source.manager_name: True for source in self._live_plot_sources}
        self._viewer._live_plots_callback = self._live_plots_panel_imgui

    def _live_plots_panel_imgui(self, imgui) -> None:
        """Render a Live Plots collapsing section in the Newton GL sidebar."""
        if not self._live_plot_sources or self._viewer is None:
            return
        viewer = self._viewer
        scalar_buffers = getattr(viewer, "_scalar_buffers", None)
        array_buffers = getattr(viewer, "_array_buffers", None)
        if not scalar_buffers and not array_buffers:
            return

        _ip = getattr(viewer, "_implot", None)
        if not hasattr(viewer, "_scalar_arrays"):
            viewer._scalar_arrays = {}
        scalar_arrays = viewer._scalar_arrays
        n = getattr(viewer, "_plot_history_size", 250)
        s = viewer.gui.ui.dpi_scale
        plot_h = 180 * s

        groups: dict[str, list[str]] = {}
        for name in scalar_buffers or {}:
            base = _newton_scalar_base_name(name)
            groups.setdefault(base, []).append(name)

        episode_keys = [k for k in groups if k.startswith("episode/")]
        other_keys = [k for k in groups if not k.startswith("episode/")]
        groups = {k: groups[k] for k in episode_keys + other_keys}

        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Live Plots"):
            return
        imgui.separator()

        for base_name, names in groups.items():
            term_label = base_name.rsplit("/", 1)[-1]
            if not imgui.collapsing_header(term_label):
                continue
            for name in names:
                buf = scalar_buffers.get(name, [])
                arr = scalar_arrays.get(name)
                if arr is None:
                    arr = np.full(n, np.nan, dtype=np.float32)
                    arr[n - len(buf) :] = np.array(buf, dtype=np.float32)
                    scalar_arrays[name] = arr
            if _ip is not None and _ip.begin_plot(f"##{base_name}", imgui.ImVec2(-1, plot_h)):
                _auto = _ip.AxisFlags_.auto_fit.value
                _ip.setup_axes("", "", _auto, _auto)
                _ip.setup_finish()
                for name in names:
                    arr = scalar_arrays.get(name)
                    if arr is not None:
                        suffix = name[len(base_name) :]
                        label = suffix if suffix else term_label
                        _ip.plot_line(label, arr)
                _ip.end_plot()
            else:
                graph_size = imgui.ImVec2(-1, 80 * s)
                for name in names:
                    arr = scalar_arrays.get(name)
                    if arr is not None:
                        buf = scalar_buffers.get(name, [])
                        overlay = f"{buf[-1]:.4g}" if buf else ""
                        imgui.plot_lines(f"##{name}", arr, graph_size=graph_size, overlay_text=overlay)

        render_heatmap = getattr(viewer, "_render_array_heatmap", None)
        if render_heatmap is not None:
            panel_width = imgui.get_content_region_avail().x
            for name, array in (array_buffers or {}).items():
                if imgui.collapsing_header(name):
                    render_heatmap(name, array, panel_width - 20.0 * s, dpi_scale=s)

    def _render_live_plots(self) -> None:
        """Push manager-term scalars to the Newton viewer's built-in plot panel."""
        if self._viewer is None or not self._live_plot_sources:
            return
        if getattr(self, "_runtime_headless", False):
            return
        self._live_plots_step_counter += 1
        if self._live_plots_step_counter % max(1, self.cfg.live_plots_update_interval) != 0:
            return
        for source in self._live_plot_sources:
            if not self._live_plots_manager_visible.get(source.manager_name, True):
                continue
            for term_name, values in source.collect(self._live_plot_env_idx).items():
                if len(values) == 1:
                    self._viewer.log_scalar(f"{source.manager_name}/{term_name}", values[0])
                else:
                    for i, v in enumerate(values):
                        self._viewer.log_scalar(f"{source.manager_name}/{term_name}[{i}]", v)

    def _apply_viewer_post_init(self) -> None:
        """Apply GL-specific renderer settings after viewer construction."""
        self._viewer.up_axis = 2  # Z-up
        self._viewer.scaling = 1.0
        self._viewer.particle_color = self.cfg.particle_color
        self._viewer.renderer.draw_shadows = self.cfg.enable_shadows
        self._viewer.renderer.draw_sky = self.cfg.enable_sky
        self._viewer.renderer.draw_wireframe = self.cfg.enable_wireframe
        # Accept list/tuple/array-like config colors; provide a stable tuple for nanobind conversion.
        self._viewer.renderer.sky_upper = self._viewer._coerce_color3(self.cfg.sky_upper_color)
        self._viewer.renderer.sky_lower = self._viewer._coerce_color3(self.cfg.sky_lower_color)
        self._viewer.renderer._light_color = self._viewer._coerce_color3(self.cfg.light_color)

    def _apply_camera_pose(
        self,
        pose: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> None:
        if self._viewer is None:
            return
        cam_pos, cam_target = pose
        # Match Newton's Camera native pos type: PygletVec3, not wp.vec3.
        self._viewer.camera.pos = PygletVec3(*cam_pos)
        self._viewer.camera.look_at(cam_target)
        self._last_camera_pose = (cam_pos, cam_target)

    def _apply_camera_focal_length(self) -> None:
        if self._viewer is None:
            return
        self._viewer.camera.fov = self._focal_length_to_vertical_fov_degrees()

    def _pump_paused(self) -> None:
        self._viewer._update()

    def render_rgb_array(self) -> np.ndarray:
        """Return the latest RGB frame rendered by the Newton GL viewer.

        In headless mode, current physics state is requested and a full render cycle is executed
        only when a frame is requested.

        Returns:
            The latest viewer framebuffer as a uint8 array with shape ``(H, W, 3)``.

        Raises:
            RuntimeError: If the visualizer has not been initialized.
        """
        if self._viewer is None:
            raise RuntimeError("NewtonGLVisualizer must be initialized before capturing an RGB frame.")
        if self._runtime_headless and not self._viewer.is_paused():
            self._state = self._newton_backend.request_visualization_state(self._scene_data_provider)
            self._pre_step()
            self._viewer.begin_frame(self._sim_time)
            try:
                if self._state is not None:
                    self._viewer.log_state(self._state)
                if self.cfg.enable_markers:
                    render_newton_visualization_markers(
                        self._viewer,
                        self._resolved_visible_env_ids,
                        num_envs=len(self._clone_plan.env_ids),
                    )
            finally:
                self._viewer.end_frame()
        return self._viewer.get_frame().numpy()

    def render_tiled_rgb_array(self) -> np.ndarray | None:
        """Return the last composited streaming frame (all GT types side-by-side).

        Returns the full multi-GT composite produced by the streaming camera panel —
        including depth (turbo colormap), segmentation, and normals when configured via
        :attr:`~isaaclab.visualizers.VisualizerCfg.streaming_gt_types`.

        When the streaming panel is hidden (headless training or panel closed by the
        user), this method builds the composite on demand so that :class:`VideoRecorder`
        and similar consumers always receive a valid frame.

        Returns:
            ``uint8 (H, W, 3)`` composite array, or ``None`` if no camera sensor has
            been configured or no usable GT output is available.
        """
        return self._build_streaming_composite()

    def _build_streaming_composite(self) -> np.ndarray | None:
        """Build (or return the cached) streaming composite for the current step.

        The composite is built at most once per visualizer step, so repeated calls within one step
        (from :meth:`_log_streaming_image` and :meth:`render_tiled_rgb_array` both) share a result.

        Returns:
            ``uint8 (H, W, 3)`` composite array, or ``None`` when the streaming view is inactive or
            the camera produces no usable output.
        """
        return self._streaming.composite(self._step_counter) if self._streaming is not None else None

    def _log_streaming_image(self) -> None:
        """Fetch GT frames, colorize, composite, and push to Newton's image panel.

        Skips all camera rendering work when the streaming panel is hidden (no image key
        selected in the sidebar combo).  The panel key is registered with a 1×1 placeholder
        on the first call so the combo always appears in the sidebar, but no GPU/CPU
        rendering is performed until the user opens the panel.

        When the panel is visible the composite is built via :meth:`_build_streaming_composite`
        (which caches by step counter) and pushed to the image logger.
        """
        if self._viewer is None or self._streaming is None:
            return

        _PANEL_KEY = "Streaming View"
        image_logger = getattr(self._viewer, "_image_logger", None)
        if image_logger is None:
            return

        # First call: register the panel key in the image logger so the sidebar combo
        # appears.  Use a 1×1 black placeholder — no camera work needed yet.
        # Clear _selected so the panel starts hidden; the user opens it via the
        # "Streaming View" → "Open" combo in the Newton sidebar.
        if image_logger is not None and _PANEL_KEY not in getattr(image_logger, "_images", {}):
            placeholder = wp.zeros((1, 1, 3), dtype=wp.uint8)
            self._viewer.log_image(_PANEL_KEY, placeholder)
            if hasattr(image_logger, "_selected"):
                image_logger._selected = None
            return

        # When the panel is hidden (selected=None), skip all camera rendering.
        # Work resumes the next step after the user selects "Open" in the combo.
        # render_tiled_rgb_array() calls _build_streaming_composite() directly for
        # headless VideoRecorder use-cases, bypassing this guard.
        if image_logger is not None and image_logger._selected is None:
            return

        composite = self._build_streaming_composite()
        if composite is None:
            return

        # Store actual dimensions so _draw_large can size the panel correctly.
        new_h, new_w = composite.shape[:2]
        prev_w = getattr(self._viewer, "_streaming_composite_w", 0)
        prev_h = getattr(self._viewer, "_streaming_composite_h", 0)
        self._viewer._streaming_composite_h = new_h
        self._viewer._streaming_composite_w = new_w
        # Trigger a panel resize whenever the composite first arrives (prev dims were
        # 0 or the 1×1 placeholder) so the window expands from the initial title-bar
        # state to fit the real frame.
        if prev_w <= 1 or prev_h <= 1:
            self._viewer._streaming_panel_needs_sizing = True
        composite_t = torch.from_numpy(composite).contiguous()
        self._viewer.log_image(_PANEL_KEY, wp.from_torch(composite_t))


# ---------------------------------------------------------------------------
# Planned-camera presenter
# ---------------------------------------------------------------------------


class NewtonRTXVisualizer(BaseVisualizer):
    """Present one planned camera stream in a Newton GL image sink.

    The camera's selected renderer owns rendering and publishes :class:`CameraData`; this visualizer
    only composites that output through :class:`StreamingView`. It therefore works with every camera
    renderer and never creates an OVRTX renderer or a second USD stage.
    """

    marker_type = None

    def __init__(self, cfg: NewtonRTXVisualizerCfg):
        super().__init__(cfg)
        self.cfg = cfg
        self._viewer: NewtonViewerGL | None = None
        self._streaming: StreamingView | None = None
        self._frame: np.ndarray | None = None
        self._sim_time = 0.0
        self._step_counter = 0

    def initialize(self, scene_data_provider: SceneDataProvider, clone_plan: ClonePlan) -> None:
        """Resolve the planned camera and create only its image sink."""
        if self._is_initialized:
            return
        if scene_data_provider is None or clone_plan.env_ids is None:
            raise RuntimeError("NewtonRTXVisualizer requires a completed clone plan.")
        if not self.cfg.streaming_view:
            raise ValueError("NewtonRTXVisualizer requires streaming_view=True and an explicit streaming_camera.")
        self._clone_plan = clone_plan
        self._env_ids = self._compute_visualized_env_ids()
        visible_env_ids = resolve_visible_env_indices(self._env_ids, self.cfg.max_visible_envs, len(clone_plan.env_ids))

        from isaaclab.sim import SimulationContext

        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError("NewtonRTXVisualizer requires an active SimulationContext.")
        self._streaming = StreamingView(
            self.cfg,
            sim.get_camera_sensors(),
            visible_env_ids=visible_env_ids,
            target_aspect=self.cfg.window_width / max(1, self.cfg.window_height),
        )
        runtime_headless = self.cfg.headless or (
            sys.platform not in ("win32", "darwin") and not os.environ.get("DISPLAY")
        )
        self._viewer = NewtonViewerGL(
            width=self.cfg.window_width,
            height=self.cfg.window_height,
            headless=runtime_headless,
            metadata={"num_envs": len(clone_plan.env_ids)},
            update_frequency=self.cfg.update_frequency,
        )
        self._is_initialized = True

    def step(self, dt: float) -> None:
        """Present the current planned-camera composite."""
        if not self._is_initialized or self._is_closed or self._viewer is None or self._streaming is None:
            return
        self._sim_time += dt
        self._step_counter += 1
        if self._step_counter % self._viewer._update_frequency != 0 or not self._viewer.is_running():
            return
        if not self._viewer.is_rendering_paused() or self._frame is None:
            self._frame = self._streaming.composite(self._step_counter)
        if self._frame is None:
            raise RuntimeError("NewtonRTXVisualizer's planned camera produced no composite.")
        self._viewer.begin_frame(self._sim_time)
        try:
            self._viewer.log_image("Camera", self._frame, fullscreen=True)
        finally:
            self._viewer.end_frame()

    def render_rgb_array(self) -> np.ndarray:
        """Return the current planned-camera composite."""
        if self._streaming is None:
            raise RuntimeError("NewtonRTXVisualizer requires an initialized planned camera stream.")
        self._frame = self._streaming.composite(self._step_counter)
        if self._frame is None:
            raise RuntimeError("NewtonRTXVisualizer's planned camera produced no composite.")
        return self._frame

    def reset(self, soft: bool = False) -> None:
        """Invalidate the cached camera composite."""
        del soft
        self._frame = None
        if self._streaming is not None:
            self._streaming.invalidate()

    def close(self) -> None:
        """Release the image sink."""
        if self._is_closed:
            return
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._streaming = None
        self._frame = None
        self._is_closed = True

    def is_running(self) -> bool:
        """Return whether the image sink remains open."""
        return bool(self._is_initialized and not self._is_closed and self._viewer and self._viewer.is_running())

    def consume_reset_request(self) -> bool:
        """Return and clear the image sink's reset request."""
        return self._viewer.consume_reset_request() if self._viewer is not None else False

    def is_training_paused(self) -> bool:
        """Return whether training is paused from the image sink."""
        return self._viewer.is_training_paused() if self._viewer is not None else False

    def is_rendering_paused(self) -> bool:
        """Return whether camera presentation is paused from the image sink."""
        return self._viewer.is_rendering_paused() if self._viewer is not None else False
