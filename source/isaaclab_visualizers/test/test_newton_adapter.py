# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Newton viewer adapter helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_visualizers.newton import (
    NewtonGLVisualizer,
    NewtonGLVisualizerCfg,
    NewtonRTXVisualizer,
    NewtonRTXVisualizerCfg,
)
from isaaclab_visualizers.newton import newton_visualization_markers as newton_markers
from isaaclab_visualizers.newton.newton_visualizer import NewtonViewerGL
from isaaclab_visualizers.newton_adapter import (
    VISUALIZER_INFINITE_PLANE_SIZE,
    expand_infinite_plane_scale,
    log_geo_with_expanded_plane_scale,
    resolve_visible_env_indices,
)
from isaaclab_visualizers.rerun import RerunVisualizer, RerunVisualizerCfg
from isaaclab_visualizers.viser import ViserVisualizer, ViserVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.vis_marker_registry import VisMarkerRegistry
from isaaclab.sim import SimulationContext


class _NewtonBackendResource:
    """Explicit simulation-scoped Newton resource for visualizer unit tests."""

    def __init__(self):
        self.model = None
        self.state = None
        self.supports_force_input = False

    def get_model(self):
        return self.model

    def request_visualization_state(self, _provider):
        return self.state

    def supports_rigid_body_force_input(self):
        return self.supports_force_input


@pytest.fixture(autouse=True)
def active_visualizer_owner(monkeypatch: pytest.MonkeyPatch):
    """Give directly constructed visualizers an explicit context-owned Newton resource."""
    from isaaclab_newton.cloner import NewtonReplicateContext

    resource = _NewtonBackendResource()
    context = object.__new__(SimulationContext)
    context.stage = object()
    context.cfg = SimpleNamespace(device="cpu")
    context._backend_registry = {NewtonReplicateContext: resource}
    context._backend_clone_roles = {}
    context._clone_plan = None
    context._camera_sensors = {}
    monkeypatch.setattr(SimulationContext, "_instance", context)
    yield resource


def test_expand_infinite_plane_scale_expands_non_positive_extents():
    assert expand_infinite_plane_scale((0.0, 0.0, 1.0, 0.0)) == (
        VISUALIZER_INFINITE_PLANE_SIZE,
        VISUALIZER_INFINITE_PLANE_SIZE,
        1.0,
        0.0,
    )
    assert expand_infinite_plane_scale((-1.0, 25.0)) == (
        VISUALIZER_INFINITE_PLANE_SIZE,
        25.0,
    )
    assert expand_infinite_plane_scale((25.0, 0.0)) == (
        25.0,
        VISUALIZER_INFINITE_PLANE_SIZE,
    )


def test_expand_infinite_plane_scale_preserves_finite_extents():
    assert expand_infinite_plane_scale((100.0, 50.0, 1.0)) == (100.0, 50.0, 1.0)


def test_log_geo_with_expanded_plane_scale_delegates_with_adjusted_plane_scale():
    calls = []

    def _log_geo(*args):
        calls.append(args)
        return "logged"

    assert log_geo_with_expanded_plane_scale(_log_geo, 1, "ground", 1, (0.0, 25.0), 0.0, True) == "logged"
    assert calls == [("ground", 1, (VISUALIZER_INFINITE_PLANE_SIZE, 25.0), 0.0, True, None, False)]


def test_log_geo_with_expanded_plane_scale_preserves_non_plane_scale():
    calls = []

    def _log_geo(*args):
        calls.append(args)

    log_geo_with_expanded_plane_scale(_log_geo, 1, "box", 2, (0.0, 25.0), 0.0, True, hidden=True)
    assert calls == [("box", 2, (0.0, 25.0), 0.0, True, None, True)]


def test_resolve_visible_env_indices_truncates_explicit_list():
    assert resolve_visible_env_indices([1, 3, 5], 2, 10) == [1, 3]
    assert resolve_visible_env_indices([1, 3], 1, 10) == [1]


def test_resolve_visible_env_indices_deduplicates_before_truncating():
    assert resolve_visible_env_indices([1, 1, 3, 5], 2, 10) == [1, 3]


def test_resolve_visible_env_indices_explicit_full_list_when_no_cap():
    assert resolve_visible_env_indices([1, 3], None, 10) == [1, 3]


def test_resolve_visible_env_indices_cap_when_no_filter():
    # When _compute_visualized_env_ids is None, cap is max_visible_envs.
    assert resolve_visible_env_indices(None, 3, 10) == [0, 1, 2]


def test_resolve_visible_env_indices_all_when_no_cap():
    assert resolve_visible_env_indices(None, None, 10) is None


def test_resolve_visible_env_indices_num_envs_zero_falls_through_like_newton():
    assert resolve_visible_env_indices(None, 5, 0) is None


def test_newton_visualizer_cfg_exposes_viewer_options():
    cfg = NewtonGLVisualizerCfg(enable_picking=False, show_particles=True, particle_color=(0.1, 0.2, 0.3))

    assert cfg.enable_picking is False
    assert cfg.show_particles is True
    assert cfg.particle_color == (0.1, 0.2, 0.3)


@pytest.mark.parametrize(
    ("visualizer_type", "cfg_type"),
    [
        (NewtonGLVisualizer, NewtonGLVisualizerCfg),
        (RerunVisualizer, RerunVisualizerCfg),
        (ViserVisualizer, ViserVisualizerCfg),
    ],
)
def test_newton_visualizer_reuses_backend_by_type(active_visualizer_owner, visualizer_type, cfg_type):
    from isaaclab_newton.cloner import NewtonReplicateContext

    context = SimulationContext.instance()
    key = NewtonReplicateContext

    visualizer = visualizer_type(cfg_type())

    assert visualizer._newton_backend is active_visualizer_owner
    assert context._backend_clone_roles == {key: {"scene"}}


def test_viser_without_clients_skips_scene_data_request():
    """An idle server must not convert physics state that it cannot display."""
    visualizer = object.__new__(ViserVisualizer)
    visualizer._is_initialized = True
    visualizer._viewer = SimpleNamespace(_server=SimpleNamespace(get_clients=lambda: {}))
    visualizer._scene_data_provider = object()
    visualizer._newton_backend = SimpleNamespace(request_visualization_state=Mock())
    visualizer._clone_plan = SimpleNamespace(env_ids=[0])
    visualizer._sim_time = 0.0
    visualizer._apply_pending_camera_pose = Mock()
    visualizer._render_live_plots = Mock()

    visualizer.step(0.25)

    assert visualizer._sim_time == 0.25
    visualizer._newton_backend.request_visualization_state.assert_not_called()
    visualizer._render_live_plots.assert_called_once_with()


def test_newton_marker_registry_lifecycle():
    """The plan-owned registry constructs and closes Newton marker state."""
    registry = VisMarkerRegistry()
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/test",
        markers={"sphere": sim_utils.SphereCfg(radius=1.0)},
    )

    registry.prepare((cfg,), (newton_markers.NewtonVisualizationMarkers,))

    planned_cfg, (marker,) = registry.get(cfg)
    assert planned_cfg is cfg
    assert isinstance(marker, newton_markers.NewtonVisualizationMarkers)
    assert registry.get_groups() == (marker,)

    registry.close()

    assert registry.get_groups() == ()


def test_newton_visualizer_cfg_exposes_world_spacing():
    cfg = NewtonGLVisualizerCfg(world_spacing=(2.0, 2.0, 0.0))

    assert cfg.world_spacing == (2.0, 2.0, 0.0)


def test_newton_visualizer_set_camera_view_updates_cfg_without_viewer():
    visualizer = NewtonGLVisualizer(NewtonGLVisualizerCfg())

    visualizer.set_camera_view((1, 2, 3), (0, 0, 1))

    assert visualizer.cfg.eye == (1.0, 2.0, 3.0)
    assert visualizer.cfg.lookat == (0.0, 0.0, 1.0)
    assert visualizer._resolve_initial_camera_pose() == ((1.0, 2.0, 3.0), (0.0, 0.0, 1.0))


def test_newton_visualizer_set_camera_view_updates_active_viewer():
    """NewtonGLVisualizer should honor SimulationContext camera updates."""

    class _FakeCamera:
        def __init__(self):
            self.pos = None
            self.look_at_calls = []

        def look_at(self, target):
            self.look_at_calls.append(tuple(target))

    class _FakeViewer:
        def __init__(self):
            self.camera = _FakeCamera()

    viewer = _FakeViewer()
    visualizer = NewtonGLVisualizer(NewtonGLVisualizerCfg())
    visualizer._viewer = viewer

    visualizer.set_camera_view((1, 2, 3), (0, 0, 1))

    assert (viewer.camera.pos.x, viewer.camera.pos.y, viewer.camera.pos.z) == (1.0, 2.0, 3.0)
    assert viewer.camera.look_at_calls == [(0.0, 0.0, 1.0)]
    assert visualizer.cfg.eye == (1.0, 2.0, 3.0)
    assert visualizer.cfg.lookat == (0.0, 0.0, 1.0)


def test_newton_visualizer_streams_from_the_planned_camera_path():
    """A visualizer selects a camera already declared by the scene."""
    task_camera = SimpleNamespace(
        num_instances=4,
        update=lambda **_kwargs: None,
        data=SimpleNamespace(output={"rgb": torch.zeros((4, 1, 1, 4), dtype=torch.uint8)}),
        cfg=SimpleNamespace(prim_path="/World/envs/env_[^/]+/Camera"),
    )
    streaming_camera = SimpleNamespace(
        num_instances=4,
        update=lambda **_kwargs: None,
        data=SimpleNamespace(output={"rgb": torch.zeros((4, 1, 1, 4), dtype=torch.uint8)}),
        cfg=SimpleNamespace(prim_path="/World/envs/env_[^/]+/StreamingCamera"),
    )

    cfg = NewtonGLVisualizerCfg(
        streaming_view=True,
        streaming_envs=4,
        streaming_camera="{ENV_REGEX_NS}/StreamingCamera",
    )
    task_path = "/World/envs/env_[^/]+/Camera"
    streaming_path = "/World/envs/env_[^/]+/StreamingCamera"
    visualizer = NewtonGLVisualizer(cfg)
    SimulationContext.instance()._camera_sensors = {task_path: task_camera, streaming_path: streaming_camera}

    visualizer._setup_streaming_view()

    assert visualizer._streaming.camera is streaming_camera
    assert list(visualizer._streaming.scene_cameras) == [task_path, streaming_path]


def test_newton_visualizer_requires_the_selected_scene_camera():
    """The panel resolves the exact camera selected by its cfg."""
    task_camera = SimpleNamespace(
        num_instances=4,
        update=lambda **_kwargs: None,
        data=SimpleNamespace(output={"rgb": torch.zeros((4, 1, 1, 4), dtype=torch.uint8)}),
        cfg=SimpleNamespace(prim_path="/World/envs/env_[^/]+/Camera"),
    )

    camera_path = "/World/envs/env_[^/]+/Camera"
    visualizer = NewtonGLVisualizer(
        NewtonGLVisualizerCfg(streaming_view=True, streaming_envs=4, streaming_camera="{ENV_REGEX_NS}/Camera")
    )
    SimulationContext.instance()._camera_sensors = {camera_path: task_camera}

    visualizer._setup_streaming_view()

    assert visualizer._streaming.camera is task_camera


def test_newton_visualizer_render_rgb_array_returns_viewer_frame():
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    viewer = SimpleNamespace(get_frame=lambda: SimpleNamespace(numpy=lambda: frame))
    visualizer = NewtonGLVisualizer(NewtonGLVisualizerCfg())
    visualizer._viewer = viewer

    assert visualizer.render_rgb_array() is frame


def test_newton_visualizer_render_rgb_array_requires_initialized_viewer():
    visualizer = NewtonGLVisualizer(NewtonGLVisualizerCfg())

    with pytest.raises(RuntimeError, match="must be initialized"):
        visualizer.render_rgb_array()


def test_newton_viewer_camera_speed_boost_when_shift_held(monkeypatch):
    viewer = NewtonViewerGL.__new__(NewtonViewerGL)
    viewer._camera_speed = 4.0

    monkeypatch.setattr(NewtonViewerGL, "is_key_down", lambda self, key: True)
    assert viewer.camera_speed == pytest.approx(8.0)

    monkeypatch.setattr(NewtonViewerGL, "is_key_down", lambda self, key: False)
    assert viewer.camera_speed == pytest.approx(4.0)


def test_newton_viewer_camera_speed_setter_validates(monkeypatch):
    viewer = NewtonViewerGL.__new__(NewtonViewerGL)
    monkeypatch.setattr(NewtonViewerGL, "is_key_down", lambda self, key: False)

    viewer.camera_speed = 6.0
    assert viewer._camera_speed == pytest.approx(6.0)

    with pytest.raises(ValueError, match="camera_speed must be finite and nonnegative"):
        viewer.camera_speed = -1.0


class _FakeTrainingControlsImgui:
    """Minimal imgui double that drives ``_render_training_controls`` by label."""

    def __init__(self, clicked_label: str | None = None):
        self._clicked_label = clicked_label

    def button(self, label):
        return label == self._clicked_label

    def text(self, _text):
        pass

    def slider_int(self, _label, value, _min_value, _max_value, _format):
        return False, value

    def is_item_hovered(self):
        return False

    def set_tooltip(self, _text):
        pass


def test_newton_gl_viewer_rendering_pause_state_stays_in_sync_with_space_key():
    """Space toggles ``_paused`` directly (Newton's own key handler); the "Pause Rendering"
    button and ``is_rendering_paused()`` must reflect that instead of a separately tracked flag,
    or the UI desyncs from the actual paused state that gates rendering.
    """
    viewer = NewtonViewerGL.__new__(NewtonViewerGL)
    viewer._paused = False
    viewer._paused_training = False
    viewer._reset_requested = False
    viewer._update_frequency = 1

    assert viewer.is_rendering_paused() is False

    # Simulate Newton's own Space key handler (newton/_src/viewer/viewer_gui.py), which
    # toggles ``_paused`` directly and bypasses the Isaac Lab "Pause Rendering" button.
    viewer._paused = not viewer._paused

    assert viewer.is_rendering_paused() is True
    viewer._render_training_controls(_FakeTrainingControlsImgui())  # must not raise: no click

    # The button must read the post-Space state and toggle it back correctly.
    resume_click = _FakeTrainingControlsImgui(clicked_label="Resume Rendering")
    viewer._render_training_controls(resume_click)
    assert viewer.is_rendering_paused() is False


def test_newton_viewer_particle_color_override(monkeypatch):
    from newton.viewer import ViewerGL

    viewer = NewtonViewerGL.__new__(NewtonViewerGL)
    viewer.device = "cpu"
    viewer.objects = {}
    viewer.model_changed = False
    viewer.particle_color = (0.1, 0.2, 0.3)
    viewer._particle_color_buffer = None
    viewer._particle_color_buffer_count = 0
    viewer._particle_color_buffer_value = None
    points = wp.zeros(4, dtype=wp.vec3, device="cpu")
    calls = []

    def _log_points(self, name, points, radii=None, colors=None, hidden=False):
        calls.append((name, points, radii, colors, hidden))

    monkeypatch.setattr(ViewerGL, "log_points", _log_points)

    viewer.log_points("/model/particles", points, colors=None)

    name, _, _, colors, hidden = calls[-1]
    assert name == "/model/particles"
    assert hidden is False
    assert isinstance(colors, wp.array)
    assert colors.shape[0] == 4
    np.testing.assert_allclose(colors.numpy()[0], np.array([0.1, 0.2, 0.3], dtype=np.float32), rtol=1.0e-6)


def test_newton_viewer_particle_color_override_reuses_existing_color_buffer(monkeypatch):
    from newton.viewer import ViewerGL

    viewer = NewtonViewerGL.__new__(NewtonViewerGL)
    viewer.device = "cpu"
    viewer.model_changed = False
    viewer.particle_color = (0.1, 0.2, 0.3)
    viewer._particle_color_buffer = wp.zeros(4, dtype=wp.vec3, device="cpu")
    viewer._particle_color_buffer_count = 4
    viewer._particle_color_buffer_value = (0.1, 0.2, 0.3)
    viewer.objects = {"/model/particles": SimpleNamespace(num_instances=4)}
    points = wp.zeros(4, dtype=wp.vec3, device="cpu")
    calls = []

    def _log_points(self, name, points, radii=None, colors=None, hidden=False):
        calls.append((name, points, radii, colors, hidden))

    monkeypatch.setattr(ViewerGL, "log_points", _log_points)

    viewer.log_points("/model/particles", points, colors=None)

    _, _, _, colors, _ = calls[-1]
    # When buffer is already valid and count matches, _particle_color_update_array returns None
    # (no new GPU upload needed; Newton retains existing colors from the previous frame).
    assert colors is None


def test_newton_viewer_particle_color_override_leaves_other_points_unchanged(monkeypatch):
    from newton.viewer import ViewerGL

    viewer = NewtonViewerGL.__new__(NewtonViewerGL)
    viewer.device = "cpu"
    viewer.model_changed = False
    viewer.particle_color = (0.1, 0.2, 0.3)
    viewer._particle_color_buffer = None
    viewer._particle_color_buffer_count = 0
    viewer._particle_color_buffer_value = None
    custom_colors = wp.zeros(3, dtype=wp.vec3, device="cpu")
    points = wp.zeros(3, dtype=wp.vec3, device="cpu")
    calls = []

    def _log_points(self, name, points, radii=None, colors=None, hidden=False):
        calls.append((name, points, radii, colors, hidden))

    monkeypatch.setattr(ViewerGL, "log_points", _log_points)

    viewer.log_points("/user/custom_points", points, colors=custom_colors)

    _, _, _, colors, _ = calls[-1]
    assert colors is custom_colors


def test_newton_viewer_fast_paths_all_active_mpm_particles(monkeypatch):
    import newton as nt

    viewer = NewtonViewerGL.__new__(NewtonViewerGL)
    viewer._mpm_particle_flags_cache_key = None
    viewer._mpm_particles_all_active = False
    viewer.model_changed = False
    viewer.particle_color = None
    viewer.show_particles = True
    viewer.model = SimpleNamespace(
        mpm=object(),
        particle_count=3,
        particle_radius=0.01,
        particle_flags=wp.array(
            [int(nt.ParticleFlags.ACTIVE)] * 3,
            dtype=wp.int32,
            device="cpu",
        ),
    )
    published_points = wp.full(3, wp.vec3(1.0, 2.0, 3.0), dtype=wp.vec3, device="cpu")
    state = SimpleNamespace(particle_q=published_points)
    log_points_calls = []

    monkeypatch.setattr(NewtonViewerGL, "_apply_layer_transform_to_points", lambda self, points: points)
    monkeypatch.setattr(NewtonViewerGL, "_qualify", lambda self, name: name)
    monkeypatch.setattr(NewtonViewerGL, "_layer_force_hidden", lambda self: False)
    monkeypatch.setattr(NewtonViewerGL, "log_points", lambda self, *a, **kw: log_points_calls.append((a, kw)))

    viewer._log_particles(state)

    # _log_particles calls self.log_points with all keyword args, so positional tuple is empty.
    assert len(log_points_calls) == 1
    assert log_points_calls[0][1]["name"] == "/model/particles"
    assert log_points_calls[0][1]["points"] is published_points
    assert log_points_calls[0][1]["points"] is state.particle_q


def test_newton_viewer_does_not_filter_the_sdp_pointer(monkeypatch):
    viewer = NewtonViewerGL.__new__(NewtonViewerGL)
    viewer.device = "cpu"
    viewer.model_changed = False
    viewer.show_particles = True
    radii = wp.array([0.1, 0.2], dtype=wp.float32, device="cpu")
    viewer.model = SimpleNamespace(
        mpm=object(),
        particle_count=2,
        particle_radius=radii,
        particle_flags=object(),
    )
    published_points = wp.array([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], dtype=wp.vec3, device="cpu")
    log_points_calls = []

    monkeypatch.setattr(NewtonViewerGL, "_apply_layer_transform_to_points", lambda self, points: points)
    monkeypatch.setattr(NewtonViewerGL, "_qualify", lambda self, name: name)
    monkeypatch.setattr(NewtonViewerGL, "_layer_force_hidden", lambda self: False)
    monkeypatch.setattr(NewtonViewerGL, "log_points", lambda self, *a, **kw: log_points_calls.append((a, kw)))

    viewer._log_particles(SimpleNamespace(particle_q=published_points))

    assert len(log_points_calls) == 1
    assert log_points_calls[0][1]["points"] is published_points
    assert log_points_calls[0][1]["radii"] is radii


@pytest.mark.parametrize("viewer_path", ["rerun", "viser"])
def test_web_newton_viewers_draw_the_sdp_ordered_state(monkeypatch, viewer_path):
    if viewer_path == "rerun":
        from isaaclab_visualizers.rerun.rerun_visualizer import NewtonViewerRerun as Viewer
    else:
        from isaaclab_visualizers.viser.viser_visualizer import NewtonViewerViser as Viewer

    viewer = Viewer.__new__(Viewer)
    viewer.device = "cpu"
    viewer.model_changed = False
    viewer.show_particles = True
    viewer.model = SimpleNamespace(mpm=None, particle_count=2, particle_radius=0.1, particle_flags=None)
    published_points = wp.array([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], dtype=wp.vec3, device="cpu")
    log_points_calls = []

    monkeypatch.setattr(Viewer, "_apply_layer_transform_to_points", lambda self, points: points)
    monkeypatch.setattr(Viewer, "_qualify", lambda self, name: name)
    monkeypatch.setattr(Viewer, "_layer_force_hidden", lambda self: False)
    monkeypatch.setattr(Viewer, "log_points", lambda self, *a, **kw: log_points_calls.append((a, kw)))

    viewer._log_particles(SimpleNamespace(particle_q=published_points))

    assert log_points_calls[0][1]["points"] is published_points


class _BodyQ:
    shape = (1,)


class _Viewer:
    _update_frequency = 1

    def __init__(self):
        self.device = "cpu"
        self.show_contacts = False
        self.logged_state = None
        self.closed = False

    def is_paused(self):
        return False

    def is_running(self):
        return True

    def begin_frame(self, _time):
        pass

    def log_state(self, state):
        self.logged_state = state

    def end_frame(self):
        pass

    def get_frame(self):
        return SimpleNamespace(numpy=lambda: np.zeros((4, 6, 3), dtype=np.uint8))

    def close(self):
        self.closed = True


class _SceneDataProvider:
    def __init__(self, num_envs=1):
        # The provider owns the env count -- it reads it off the clone plan -- so a visualizer
        # asks it rather than the physics engine.
        self.num_envs = num_envs


def _make_newton_visualizer(viewer, scene_data_provider=None):
    visualizer = NewtonGLVisualizer(NewtonGLVisualizerCfg(enable_markers=False))
    visualizer._is_initialized = True
    visualizer._is_closed = False
    visualizer._sim_time = 0.0
    visualizer._step_counter = 0
    visualizer._runtime_headless = False
    visualizer._viewer = viewer
    visualizer._scene_data_provider = scene_data_provider or _SceneDataProvider()
    visualizer._resolved_visible_env_ids = None
    visualizer._live_plot_sources = []
    if viewer is not None:
        visualizer._viewer_picking_binding.bind(viewer)
    visualizer._log_camera_sensor_image = lambda: None
    return visualizer


def test_newton_visualizer_forwards_and_neutralizes_picking():
    viewer = _Viewer()
    viewer.picking_enabled = True
    viewer.picking = SimpleNamespace(release=Mock())
    viewer.apply_forces = Mock()
    visualizer = _make_newton_visualizer(viewer)
    visualizer._picking_enabled = True
    callback = visualizer._viewer_picking_binding.apply

    state = object()
    callback(state)
    viewer.apply_forces.assert_called_once_with(state)

    visualizer.close()

    assert viewer.picking_enabled is False
    assert viewer.closed
    viewer.picking.release.assert_called_once_with()
    assert visualizer._viewer is None
    assert visualizer._viewer_picking_binding._viewer is None
    assert visualizer._viewer_picking_binding._retained_picking is viewer.picking

    callback(object())
    assert visualizer._viewer_picking_binding._retained_picking is None


def test_newton_visualizer_hard_reset_rebinds_viewer_model(active_visualizer_owner):
    new_model = object()
    new_state = object()
    active_visualizer_owner.model = new_model
    active_visualizer_owner.state = new_state

    viewer = _Viewer()
    viewer.picking_enabled = False
    viewer.set_model = Mock()
    viewer._register_isaaclab_ui_callbacks = Mock()
    viewer.set_visible_worlds = Mock()
    viewer.set_world_offsets = Mock()
    visualizer = _make_newton_visualizer(viewer)
    visualizer._resolved_visible_env_ids = [1, 3]
    visualizer._picking_enabled = True
    visualizer.cfg.world_spacing = (2.0, 0.0, 0.0)
    visualizer.cfg.show_contacts = True

    visualizer.reset(soft=False)
    visualizer.reset(soft=False)

    assert visualizer._model is new_model
    assert visualizer._state is new_state
    viewer.set_model.assert_called_once_with(new_model)
    viewer._register_isaaclab_ui_callbacks.assert_called_once_with()
    viewer.set_visible_worlds.assert_called_once_with([1, 3])
    viewer.set_world_offsets.assert_called_once_with((2.0, 0.0, 0.0))
    assert viewer.show_contacts is True
    assert viewer.picking_enabled is True
    assert viewer.wind is None
    assert visualizer._viewer_picking_binding._viewer is viewer


def test_newton_visualizer_headless_renders_frame_on_demand(active_visualizer_owner):
    """Headless EGL should defer rendering until a frame is requested."""
    state = SimpleNamespace(body_q=_BodyQ())
    viewer = _Viewer()
    active_visualizer_owner.request_visualization_state = Mock(return_value=state)

    visualizer = _make_newton_visualizer(viewer)
    visualizer._runtime_headless = True
    visualizer.step(0.1)

    assert viewer.logged_state is None
    active_visualizer_owner.request_visualization_state.assert_not_called()

    visualizer.render_rgb_array()

    assert viewer.logged_state is state
    active_visualizer_owner.request_visualization_state.assert_called_once_with(visualizer._scene_data_provider)


# ── USD marker inference and None-normal guard ────────────────────────


class UsdFileCfg:
    """Minimal stand-in that duck-types ``isaaclab.sim.spawners.UsdFileCfg``."""

    def __init__(self, usd_path, scale=None):
        self.usd_path = usd_path
        self.scale = scale


def test_infer_newton_marker_cfg_generic_usd_loads_mesh():
    import os

    import newton
    from isaaclab_visualizers.newton.newton_visualization_markers import _infer_newton_marker_cfg

    usd_path = os.path.join(os.path.dirname(newton.__file__), "tests", "assets", "cube_cylinder.usda")
    spec = _infer_newton_marker_cfg(UsdFileCfg(usd_path))

    assert spec.renderer == "mesh"
    assert spec.mesh_type == "usd"
    assert spec.preloaded_mesh is not None
    assert spec.preloaded_mesh.vertices.shape[0] > 0


def test_infer_newton_marker_cfg_missing_usd_fails_explicitly():
    from isaaclab_visualizers.newton.newton_visualization_markers import _infer_newton_marker_cfg

    with pytest.raises(FileNotFoundError, match="missing.usd"):
        _infer_newton_marker_cfg(UsdFileCfg("/nonexistent/missing.usd"))


def test_infer_newton_marker_cfg_unsupported_type_fails_explicitly():
    from isaaclab_visualizers.newton.newton_visualization_markers import _infer_newton_marker_cfg

    with pytest.raises(TypeError, match="UnsupportedMarkerCfg"):
        _infer_newton_marker_cfg(type("UnsupportedMarkerCfg", (), {})())


def test_infer_newton_marker_cfg_arrow_x_usd_still_maps_to_builtin_arrow():
    from isaaclab_visualizers.newton.newton_visualization_markers import _infer_newton_marker_cfg

    spec = _infer_newton_marker_cfg(UsdFileCfg("/assets/arrow_x.usd"))

    assert spec.renderer == "mesh"
    assert spec.mesh_type == "arrow"
    assert spec.preloaded_mesh is None


def test_infer_newton_marker_cfg_frame_prim_usd_still_maps_to_frame_renderer():
    from isaaclab_visualizers.newton.newton_visualization_markers import _infer_newton_marker_cfg

    spec = _infer_newton_marker_cfg(UsdFileCfg("/assets/frame_prim.usd"))

    assert spec.renderer == "frame"


def test_ensure_mesh_registered_handles_none_normals_and_uvs(monkeypatch):
    import isaaclab_visualizers.newton.newton_visualization_markers as _mod
    import numpy as np
    from isaaclab_visualizers.newton.newton_visualization_markers import (
        NewtonVisualizationMarkers,
        _NewtonMarkerSpec,
    )

    fake_mesh = SimpleNamespace(
        vertices=np.zeros((4, 3), dtype=np.float32),
        indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.int32),
        normals=None,
        uvs=None,
    )
    monkeypatch.setattr(_mod, "_create_mesh", lambda cfg: fake_mesh)

    log_calls = []

    class _LoggingViewer:
        def log_mesh(self, name, vertices, indices, normals=None, uvs=None, texture=None, hidden=True):
            log_calls.append({"normals": normals, "uvs": uvs})

    fake_self = SimpleNamespace(_registered_meshes=set())
    spec = _NewtonMarkerSpec(renderer="mesh", mesh_type="usd", preloaded_mesh=fake_mesh)

    NewtonVisualizationMarkers._ensure_mesh_registered(fake_self, _LoggingViewer(), "/test/mesh", spec)

    assert len(log_calls) == 1
    assert log_calls[0]["normals"] is None
    assert log_calls[0]["uvs"] is None


# ---------------------------------------------------------------------------
# Planned-camera presenter tests
# ---------------------------------------------------------------------------


def test_newton_visualizer_cfg_distinct_types():
    from isaaclab.visualizers import BaseVisualizer

    assert NewtonGLVisualizerCfg().visualizer_type == "newton_gl"
    cfg = NewtonRTXVisualizerCfg()
    assert cfg.visualizer_type == "newton_rtx"
    assert cfg.streaming_view
    assert not hasattr(cfg, "show_particles")
    assert not hasattr(cfg, "enable_picking")
    assert NewtonRTXVisualizer.__bases__ == (BaseVisualizer,)


def test_newton_rtx_visualizer_requires_streaming_view():
    visualizer = NewtonRTXVisualizer(NewtonRTXVisualizerCfg(streaming_view=False))

    with pytest.raises(ValueError, match="requires streaming_view=True"):
        visualizer.initialize(object(), SimpleNamespace(env_ids=[0]))


def test_newton_rtx_visualizer_requires_the_selected_planned_camera():
    visualizer = NewtonRTXVisualizer(
        NewtonRTXVisualizerCfg(streaming_camera="{ENV_REGEX_NS}/Camera", streaming_envs=[0])
    )

    with pytest.raises(RuntimeError, match="is not a registered camera"):
        visualizer.initialize(object(), SimpleNamespace(env_ids=[0]))


def test_newton_rtx_visualizer_presents_only_the_planned_camera_composite(monkeypatch):
    from isaaclab_visualizers.newton import newton_visualizer

    camera_path = "/World/envs/env_[^/]+/Camera"
    camera = SimpleNamespace(
        num_instances=1,
        data=SimpleNamespace(output={"rgb": torch.tensor([[[[10, 20, 30, 255]]]], dtype=torch.uint8)}),
        cfg=SimpleNamespace(prim_path=camera_path),
    )
    visualizer = NewtonRTXVisualizer(
        NewtonRTXVisualizerCfg(streaming_camera="{ENV_REGEX_NS}/Camera", streaming_envs=[0])
    )
    context = SimulationContext.instance()
    context._camera_sensors = {camera_path: camera}
    context.get_or_create_backend = Mock(side_effect=AssertionError("presenter requested a backend"))
    sink = Mock()
    sink._update_frequency = 1
    sink.is_running.return_value = True
    sink.is_rendering_paused.return_value = False
    monkeypatch.setattr(newton_visualizer, "NewtonViewerGL", lambda **_kwargs: sink)

    visualizer.initialize(object(), SimpleNamespace(env_ids=[0]))
    visualizer.step(1.0 / 60.0)

    assert visualizer._streaming.camera is camera
    assert {"_newton_backend", "_model", "_state"}.isdisjoint(vars(visualizer))
    sink.set_model.assert_not_called()
    sink.begin_frame.assert_called_once_with(1.0 / 60.0)
    name, image = sink.log_image.call_args.args
    assert name == "Camera"
    np.testing.assert_array_equal(image, np.array([[[10, 20, 30]]], dtype=np.uint8))
    assert sink.log_image.call_args.kwargs == {"fullscreen": True}
    sink.end_frame.assert_called_once_with()
    np.testing.assert_array_equal(visualizer.render_rgb_array(), np.array([[[10, 20, 30]]], dtype=np.uint8))


def test_newton_rtx_visualizer_has_no_native_ovrtx_path():
    import inspect

    from isaaclab_visualizers.newton import newton_visualizer

    source = inspect.getsource(newton_visualizer)
    forbidden = {"ViewerRTX", "_init_ovrtx", "_capture_screenshot_pixels", "from ovrtx", "import ovrtx"}
    assert not {token for token in forbidden if token in source}
    presenter = inspect.getsource(NewtonRTXVisualizer)
    forbidden_ownership = {
        "_newton_backend",
        ".set_model(",
        "request_visualization_state",
        "render_newton_visualization_markers",
    }
    assert not {token for token in forbidden_ownership if token in presenter}


def test_newton_gl_visualizer_set_camera_view_uses_look_at():
    """GL camera pose must use camera.look_at, not set_camera."""
    look_at_calls = []

    class _FakeGLCamera:
        def __init__(self):
            self.pos = None

        def look_at(self, target):
            look_at_calls.append(tuple(target))

    class _FakeGLViewer:
        def __init__(self):
            self.camera = _FakeGLCamera()

    visualizer = NewtonGLVisualizer(NewtonGLVisualizerCfg())
    visualizer._viewer = _FakeGLViewer()

    visualizer.set_camera_view((1.0, 2.0, 3.0), (0.0, 0.0, 1.0))

    assert look_at_calls == [(0.0, 0.0, 1.0)]
