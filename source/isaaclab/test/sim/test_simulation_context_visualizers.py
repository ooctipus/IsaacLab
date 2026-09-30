# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for SimulationContext visualizer orchestration."""

from __future__ import annotations

import inspect
import sys
from types import SimpleNamespace
from typing import Any, cast

import isaaclab_visualizers.kit.kit_visualizer as kit_visualizer
import isaaclab_visualizers.rerun.rerun_visualizer as rerun_visualizer
import isaaclab_visualizers.viser.viser_visualizer as viser_visualizer
import numpy as np
import pytest
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer_cfg import (
    NewtonGLVisualizerCfg,
    NewtonRTXVisualizerCfg,
    NewtonVisualizerCfg,
)
from isaaclab_visualizers.rerun.rerun_visualizer_cfg import RerunVisualizerCfg
from isaaclab_visualizers.viser.viser_visualizer_cfg import ViserVisualizerCfg

from pxr import Gf, Usd, UsdGeom

from isaaclab.markers.vis_marker_registry import VisMarkerRegistry
from isaaclab.sim.simulation_context import SimulationContext
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg

pytestmark = [pytest.mark.integration, pytest.mark.rendering]


def _completed_clone_plan(num_envs: int):
    from isaaclab.cloner import ClonePlan

    return ClonePlan(
        sources=(),
        destinations=(),
        clone_mask=np.empty((0, num_envs), dtype=np.bool_),
        env_ids=np.arange(num_envs, dtype=np.int64),
        is_complete=True,
        _env_ids_cpu=tuple(range(num_envs)),
    )


@pytest.fixture(autouse=True)
def active_visualizer_owner(monkeypatch: pytest.MonkeyPatch):
    """Give directly constructed Newton-backed visualizers their required owner."""
    context = object.__new__(SimulationContext)
    context.stage = Usd.Stage.CreateInMemory()
    context.cfg = type("Cfg", (), {"device": "cpu"})()
    context._backend_registry = {}
    context._backend_clone_roles = {}
    context._clone_plan = None
    context._renderers_initialized = False
    context._renderer_entries = []
    context._scene_data_provider = SimpleNamespace(_bind_point_plan=lambda _plan: None)
    context._camera_sensors = {}
    context.vis_marker_registry = VisMarkerRegistry()
    monkeypatch.setattr(SimulationContext, "_instance", context)


def test_web_visualizer_cfgs_do_not_open_browser_by_default():
    assert RerunVisualizerCfg().open_browser is False
    assert ViserVisualizerCfg().open_browser is False


def test_visualizer_cfgs_define_no_lifecycle_callbacks():
    """Visualizer configs stay declarative; construction and cloning belong to live engines."""
    cfg_classes = (
        VisualizerCfg,
        KitVisualizerCfg,
        NewtonVisualizerCfg,
        NewtonGLVisualizerCfg,
        NewtonRTXVisualizerCfg,
        RerunVisualizerCfg,
        ViserVisualizerCfg,
    )
    for cfg_class in cfg_classes:
        for callback in ("build", "build_visualizer", "clone_context", "create_visualizer", "get_visualizer_type"):
            assert callback not in cfg_class.__dict__
    assert "def __post_init__" not in inspect.getsource(NewtonVisualizerCfg)
    assert NewtonVisualizerCfg().class_type is None
    assert NewtonVisualizerCfg().visualizer_type is None


@pytest.mark.parametrize("consumer_name", ["renderer", "newton_gl", "rerun", "viser"])
def test_newton_visual_consumers_request_shared_model_geometry(consumer_name):
    """Every native Newton drawing consumer requests visuals on the one shared backend."""
    from isaaclab_newton.cloner import NewtonReplicateContext
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    ctx = SimulationContext.instance()
    physics_backend = ctx.get_or_create_backend(NewtonReplicateContext, ctx, clone_role="physics")
    assert physics_backend.load_visual_shapes is False

    cfg = {
        "renderer": NewtonWarpRendererCfg(),
        "newton_gl": NewtonGLVisualizerCfg(),
        "rerun": RerunVisualizerCfg(),
        "viser": ViserVisualizerCfg(),
    }[consumer_name]
    consumer = cfg.class_type(cfg)

    assert consumer._newton_backend is physics_backend
    assert physics_backend.load_visual_shapes is True
    assert ctx._backend_registry == {NewtonReplicateContext: physics_backend}
    assert ctx._backend_clone_roles == {NewtonReplicateContext: {"physics", "scene"}}


def test_newton_rtx_presenter_does_not_request_newton_model_geometry():
    """Newton RTX presents a planned camera image rather than drawing the Newton model."""
    from isaaclab_newton.cloner import NewtonReplicateContext

    ctx = SimulationContext.instance()
    physics_backend = ctx.get_or_create_backend(NewtonReplicateContext, ctx, clone_role="physics")

    cfg = NewtonRTXVisualizerCfg()
    cfg.class_type(cfg)

    assert physics_backend.load_visual_shapes is False
    assert ctx._backend_registry == {NewtonReplicateContext: physics_backend}


@pytest.mark.parametrize(
    "cfg",
    [
        RerunVisualizerCfg(streaming_view=True, streaming_camera="{ENV_REGEX_NS}/Camera"),
        ViserVisualizerCfg(streaming_view=True, streaming_camera="{ENV_REGEX_NS}/Camera"),
    ],
)
def test_streaming_web_visualizers_do_not_request_newton_scene(cfg):
    """Camera-only web presenters neither clone nor pull a Newton scene."""
    from isaaclab_newton.cloner import NewtonReplicateContext

    ctx = SimulationContext.instance()
    physics_backend = ctx.get_or_create_backend(NewtonReplicateContext, ctx, clone_role="physics")

    visualizer = cfg.class_type(cfg)

    assert not hasattr(visualizer, "_newton_backend")
    assert visualizer.marker_type is None
    assert physics_backend.load_visual_shapes is False
    assert ctx._backend_clone_roles == {NewtonReplicateContext: {"physics"}}


def test_viser_streaming_initializes_without_a_newton_backend(monkeypatch: pytest.MonkeyPatch):
    """Camera-only Viser initialization never touches the deliberately absent Newton backend."""
    visualizer = viser_visualizer.ViserVisualizer(
        ViserVisualizerCfg(streaming_view=True, streaming_camera="{ENV_REGEX_NS}/Camera")
    )
    monkeypatch.setattr(visualizer, "_create_viewer", lambda **_kwargs: setattr(visualizer, "_viewer", object()))
    monkeypatch.setattr(visualizer, "_setup_streaming_view", lambda: None)

    visualizer.initialize(cast(Any, object()), _completed_clone_plan(1))

    assert visualizer._model is None
    assert not hasattr(visualizer, "_newton_backend")


@pytest.mark.parametrize("order", [("kit", "rtx"), ("rtx", "kit")])
def test_kit_and_isaac_rtx_share_one_usd_clone_dispatch(order, monkeypatch):
    """USD consumers share one resource and one plan dispatch regardless of construction order."""
    import importlib
    from unittest.mock import MagicMock, patch

    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    import isaaclab.cloner.replicate_session as replicate_session
    from isaaclab.cloner import ClonePlan, UsdReplicateContext

    omni_module = sys.modules.get("omni", type(sys)("omni"))
    usd_module = type(sys)("omni.usd")
    monkeypatch.setitem(sys.modules, "omni", omni_module)
    monkeypatch.setitem(sys.modules, "omni.usd", usd_module)
    monkeypatch.setattr(omni_module, "usd", usd_module, raising=False)
    rtx_renderer = importlib.import_module("isaaclab_physx.renderers.isaac_rtx_renderer")

    ctx = SimulationContext.instance()
    registered = []

    def get_or_create_backend(backend_type, *args, clone_role=None, **kwargs):
        backend = SimulationContext.get_or_create_backend(ctx, backend_type, *args, clone_role=clone_role, **kwargs)
        registered.append((backend_type, backend, clone_role))
        return backend

    ctx.get_or_create_backend = get_or_create_backend
    prepared_renderer = SimpleNamespace(prepare_stage=MagicMock())
    ctx._renderer_entries = [prepared_renderer]
    settings = MagicMock()
    settings.get.return_value = False
    constructors = {
        "kit": lambda: kit_visualizer.KitVisualizer(KitVisualizerCfg()),
        "rtx": lambda: rtx_renderer.IsaacRtxRenderer(IsaacRtxRendererCfg()),
    }
    with (
        patch.object(rtx_renderer, "enable_extension"),
        patch.object(rtx_renderer, "get_settings_manager", return_value=settings),
        patch.object(rtx_renderer, "apply_isaac_rtx_global_settings"),
        patch.object(rtx_renderer, "ensure_rtx_hydra_engine_attached"),
    ):
        consumers = [constructors[name]() for name in order]

    backend = ctx._backend_registry[UsdReplicateContext]
    backend.replicate = MagicMock()
    UsdGeom.Xform.Define(ctx.stage, "/World/envs/env_0/Robot")
    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot",),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.arange(2, dtype=np.int64),
        root_layer_identifier=ctx.stage.GetRootLayer().identifier,
        _env_ids_cpu=(0, 1),
    )
    ctx.set_clone_plan(plan)
    completed = replicate_session.replicate(plan)

    assert len(consumers) == 2
    assert all(
        resource is backend and role == "scene" for key, resource, role in registered if key is UsdReplicateContext
    )
    assert sum(key is UsdReplicateContext for key, _resource, _role in registered) == 2
    assert ctx._backend_clone_roles == {UsdReplicateContext: {"scene"}}
    backend.replicate.assert_called_once_with(completed)
    prepared_renderer.prepare_stage.assert_called_once_with(ctx.stage, completed)


@pytest.mark.parametrize("cfg", [NewtonGLVisualizerCfg(), RerunVisualizerCfg(), ViserVisualizerCfg()])
def test_newton_backed_visualizers_require_active_simulation_context(cfg, monkeypatch):
    monkeypatch.setattr(SimulationContext, "_instance", None)
    with pytest.raises(RuntimeError, match="active SimulationContext"):
        cfg.class_type(cfg)


class _FakeProvider:
    """Minimal scene-data pointer provider for orchestration tests."""


class _FakeVisualizer(BaseVisualizer):
    """Minimal visualizer for orchestration tests."""

    def __init__(
        self,
        *,
        env_ids=None,
        running=True,
        closed=False,
        rendering_paused=False,
        training_paused_steps=0,
        raises_on_step=False,
        pumps_app_update=False,
    ):
        super().__init__(VisualizerCfg(enable_markers=False))
        self._env_ids = env_ids
        self._running = running
        self._closed = closed
        self._rendering_paused = rendering_paused
        self._training_paused_steps = training_paused_steps
        self._raises_on_step = raises_on_step
        self._pumps_app_update = pumps_app_update
        self.step_calls = []
        self.close_calls = 0

    @property
    def is_closed(self):
        return self._closed

    def is_running(self):
        return self._running

    def initialize(self, _provider, _plan):
        pass

    def is_rendering_paused(self):
        return self._rendering_paused

    def is_training_paused(self):
        if self._training_paused_steps > 0:
            self._training_paused_steps -= 1
            return True
        return False

    def step(self, dt):
        self.step_calls.append(dt)
        if self._raises_on_step:
            raise RuntimeError("step failed")

    def close(self):
        self.close_calls += 1
        self._closed = True

    def get_visualized_env_ids(self):
        return self._env_ids

    def pumps_app_update(self):
        return self._pumps_app_update

    def supports_live_plots(self):
        return False

    def flush_startup_messages(self):
        pass


def _make_context(visualizers, provider=None, dt=0.1):
    ctx = object.__new__(SimulationContext)
    ctx._visualizers = list(visualizers)
    ctx._scene_data_provider = provider
    ctx._viz_dt = dt
    ctx._render_callbacks = {}
    ctx.vis_marker_registry = VisMarkerRegistry()
    return ctx


def test_render_removes_closed_and_nonrunning_visualizers():
    provider = _FakeProvider()
    closed_viz = _FakeVisualizer(closed=True)
    stopped_viz = _FakeVisualizer(running=False)
    paused_viz = _FakeVisualizer(rendering_paused=True)
    healthy_viz = _FakeVisualizer(env_ids=[1])
    ctx = _make_context([closed_viz, stopped_viz, paused_viz, healthy_viz], provider=provider)

    ctx.render()

    assert ctx._visualizers == [paused_viz, healthy_viz]
    assert closed_viz.close_calls == 1
    assert stopped_viz.close_calls == 1
    assert paused_viz.close_calls == 0
    assert paused_viz.step_calls == [0.0]
    assert healthy_viz.step_calls == [0.1]


def test_render_propagates_visualizer_failure_without_removing_it():
    failing_viz = _FakeVisualizer(raises_on_step=True)
    ctx = _make_context([failing_viz])

    with pytest.raises(RuntimeError, match="step failed"):
        ctx.render()

    assert ctx._visualizers == [failing_viz]
    assert failing_viz.close_calls == 0


def test_render_skips_zero_dt_for_paused_app_pumping_visualizer():
    provider = _FakeProvider()
    paused_app_pumping_viz = _FakeVisualizer(rendering_paused=True, pumps_app_update=True)
    ctx = _make_context([paused_app_pumping_viz], provider=provider, dt=0.3)

    ctx.render()

    assert paused_app_pumping_viz.step_calls == []


def test_render_handles_training_pause_loop():
    provider = _FakeProvider()
    viz = _FakeVisualizer(training_paused_steps=1)
    ctx = _make_context([viz], provider=provider, dt=0.2)

    ctx.render()

    assert viz.step_calls == [0.0, 0.2]


class _LivePlotVisualizer(_FakeVisualizer):
    def __init__(self, *, enable_live_plots: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.cfg = VisualizerCfg(enable_markers=False, enable_live_plots=enable_live_plots)

    def supports_live_plots(self):
        return True


def test_render_dispatches_callbacks_for_live_plot_only_visualizer():
    """Live-plot panels share the marker registry, so dispatch must not require marker support."""
    dispatched = []
    ctx = _make_context([_LivePlotVisualizer()], provider=_FakeProvider())
    ctx.vis_marker_registry.add_callback("probe", dispatched.append)

    ctx.render()

    assert len(dispatched) == 1


def test_render_skips_dispatch_when_live_plots_disabled():
    """Live-plot support with the flag off consumes nothing, so callbacks stay idle."""
    dispatched = []
    ctx = _make_context([_LivePlotVisualizer(enable_live_plots=False)], provider=_FakeProvider())
    ctx.vis_marker_registry.add_callback("probe", dispatched.append)

    ctx.render()

    assert dispatched == []


def test_physics_ready_initializes_every_visualizer_without_cfg_filter():
    created = []

    class _Cfg:
        def __init__(self, visualizer_type):
            self.visualizer_type = visualizer_type

        def class_type(self, cfg):
            viz = _FakeVisualizer()
            viz.cfg = cfg
            viz.initialize = lambda _provider, _plan: created.append(cfg.visualizer_type)
            return viz

    ctx = _make_context_with_settings({}, visualizer_cfgs=[_Cfg("newton_gl"), _Cfg("kit"), _Cfg("rerun")])
    ctx._construct_visualizers()
    ctx.initialize_visualizers({})
    ctx.initialize_visualizers({})

    assert created == ["newton_gl", "kit", "rerun"]
    assert len(ctx._visualizers) == 3


def test_reset_relies_on_physics_ready_before_resetting_visualizers_and_playing():
    """The backend-neutral ready event initializes visualizers before reset/play hooks run."""
    events: list[str] = []
    ctx = object.__new__(SimulationContext)
    visualizer = _FakeVisualizer()
    visualizer.reset = lambda soft: events.append(f"visualizer_reset:{soft}")
    ctx._visualizers = [visualizer]

    class _PhysicsManager:
        @staticmethod
        def reset(soft=False):
            events.append(f"reset:{soft}")
            ctx.initialize_visualizers({})

        @staticmethod
        def play():
            events.append("play")

    def _initialize_visualizers(_payload=None):
        events.append("initialize_visualizers")

    ctx._physics_manager = _PhysicsManager()
    ctx.initialize_visualizers = _initialize_visualizers

    ctx.reset()

    assert events == ["reset:False", "initialize_visualizers", "visualizer_reset:False", "play"]
    assert ctx.is_playing()
    assert not ctx.is_stopped()


class _DummyViserViewer:
    def __init__(self):
        self.calls = []

    def begin_frame(self, sim_time: float) -> None:
        self.calls.append(("begin_frame", sim_time))

    def log_state(self, state) -> None:
        self.calls.append(("log_state", state))

    def end_frame(self) -> None:
        self.calls.append(("end_frame",))

    def is_running(self) -> bool:
        return True


def test_viser_visualizer_requests_state_through_sdp(monkeypatch: pytest.MonkeyPatch):
    provider = object()
    plan = _completed_clone_plan(4)
    viewer = _DummyViserViewer()

    def _fake_create_viewer(self, record_to_viser: str | None, metadata: dict | None = None):
        assert record_to_viser is None
        assert metadata == {"num_envs": len(plan.env_ids)}
        self._viewer = viewer

    monkeypatch.setattr(viser_visualizer.ViserVisualizer, "_create_viewer", _fake_create_viewer)

    state_calls: list[int] = []

    class _FakeNewtonBackend:
        def get_model(self):
            return "dummy-model"

        def request_visualization_state(self, requested_provider):
            assert requested_provider is provider
            state_calls.append(len(state_calls) + 1)
            return {"state_call": len(state_calls)}

    from isaaclab_newton.cloner import NewtonReplicateContext

    SimulationContext.instance()._backend_registry[NewtonReplicateContext] = _FakeNewtonBackend()

    visualizer = viser_visualizer.ViserVisualizer(ViserVisualizerCfg())
    visualizer.initialize(cast(Any, provider), plan)
    visualizer.step(0.25)

    assert visualizer.is_initialized
    assert state_calls == [1]
    assert visualizer._sim_time == pytest.approx(0.25)
    assert viewer.calls[0][0] == "begin_frame"
    assert viewer.calls[0][1] == pytest.approx(0.25)
    # log_state passes the state through as-is; no env_ids merged in.
    assert viewer.calls[1] == ("log_state", {"state_call": 1})
    assert viewer.calls[2] == ("end_frame",)


@pytest.mark.parametrize(
    ("cfg_max_visible_envs", "expected_visible"),
    [
        (None, None),
        (0, []),
        (3, [0, 1, 2]),
    ],
)
def test_viser_visualizer_create_viewer_applies_visible_worlds(
    monkeypatch: pytest.MonkeyPatch,
    cfg_max_visible_envs: int | None,
    expected_visible: list[int] | None,
):
    captured = {}

    class _FakeNewtonViewerViser:
        def __init__(
            self,
            *,
            port: int,
            bind_address: str,
            label: str | None,
            verbose: bool,
            share: bool,
            record_to_viser: str | None,
            metadata: dict | None = None,
        ):
            captured["init"] = {
                "port": port,
                "bind_address": bind_address,
                "label": label,
                "verbose": verbose,
                "share": share,
                "record_to_viser": record_to_viser,
                "metadata": metadata,
            }

        def set_model(self, model: Any) -> None:
            captured["set_model"] = model

        def set_visible_worlds(self, worlds) -> None:
            captured["visible_worlds"] = worlds

        def set_world_offsets(self, spacing) -> None:
            captured["set_world_offsets"] = tuple(spacing)

        @property
        def share_url(self) -> str | None:
            return None

    monkeypatch.setattr(viser_visualizer, "NewtonViewerViser", _FakeNewtonViewerViser)
    monkeypatch.setattr(
        viser_visualizer.ViserVisualizer,
        "_resolve_initial_camera_pose",
        lambda self: ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0)),
    )
    monkeypatch.setattr(viser_visualizer.ViserVisualizer, "_set_viser_camera_view", lambda self, pose: None)

    cfg = ViserVisualizerCfg(
        max_visible_envs=cfg_max_visible_envs,
        open_browser=False,
        randomly_sample_visible_envs=False,
    )
    visualizer = viser_visualizer.ViserVisualizer(cfg)
    visualizer._model = "dummy-model"
    visualizer._env_ids = None  # normally set by initialize() -> _compute_visualized_env_ids()
    visualizer._resolved_visible_env_ids = expected_visible
    visualizer._create_viewer(record_to_viser="record.viser", metadata={"num_envs": 8})

    assert captured["set_model"] == "dummy-model"
    assert captured["init"]["bind_address"] == cfg.bind_address
    assert captured["visible_worlds"] == expected_visible
    assert captured["set_world_offsets"] == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("cfg_max_visible_envs", "expected_visible"),
    [
        (None, None),
        (0, []),
        (3, [0, 1, 2]),
    ],
)
def test_rerun_visualizer_initialize_applies_visible_worlds_and_world_offsets(
    monkeypatch: pytest.MonkeyPatch,
    cfg_max_visible_envs: int | None,
    expected_visible: list[int] | None,
):
    captured = {}

    class _FakeNewtonViewerRerun:
        def __init__(
            self,
            *,
            app_id: str,
            address: str | None,
            serve_web_viewer: bool,
            web_port: int,
            grpc_port: int,
            keep_historical_data: bool,
            keep_scalar_history: bool,
            record_to_rrd: str | None,
            open_browser: bool,
            streaming_view: bool,
        ):
            captured["init"] = {
                "app_id": app_id,
                "address": address,
                "serve_web_viewer": serve_web_viewer,
                "web_port": web_port,
                "grpc_port": grpc_port,
                "keep_historical_data": keep_historical_data,
                "keep_scalar_history": keep_scalar_history,
                "record_to_rrd": record_to_rrd,
                "open_browser": open_browser,
                "streaming_view": streaming_view,
            }

        def set_model(self, model: Any) -> None:
            captured["set_model"] = model

        def set_visible_worlds(self, worlds) -> None:
            captured["visible_worlds"] = worlds

        def set_world_offsets(self, spacing) -> None:
            captured["set_world_offsets"] = tuple(spacing)

        def close(self) -> None:
            captured["closed"] = True

    class _FakeNewtonBackend:
        def get_model(self):
            return "dummy-model"

        def request_visualization_state(self, _provider):
            return {"ok": True}

    from isaaclab_newton.cloner import NewtonReplicateContext

    SimulationContext.instance()._backend_registry[NewtonReplicateContext] = _FakeNewtonBackend()

    monkeypatch.setattr(rerun_visualizer, "NewtonViewerRerun", _FakeNewtonViewerRerun)
    monkeypatch.setattr(
        rerun_visualizer, "_ensure_rerun_server", lambda **kwargs: ("rerun+http://127.0.0.1:9876/proxy", False)
    )
    monkeypatch.setattr(rerun_visualizer, "_open_rerun_web_viewer", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        rerun_visualizer.RerunVisualizer,
        "_resolve_initial_camera_pose",
        lambda self: ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0)),
    )
    monkeypatch.setattr(rerun_visualizer.RerunVisualizer, "_apply_camera_pose", lambda self, pose: None)

    cfg = RerunVisualizerCfg(
        open_browser=False,
        max_visible_envs=cfg_max_visible_envs,
        randomly_sample_visible_envs=False,
    )
    visualizer = rerun_visualizer.RerunVisualizer(cfg)
    visualizer.initialize(cast(Any, object()), _completed_clone_plan(4))

    assert captured["set_model"] == "dummy-model"
    assert captured["visible_worlds"] == expected_visible
    assert captured["set_world_offsets"] == (0.0, 0.0, 0.0)


def test_kit_visualizer_binds_the_cfg_owned_camera(monkeypatch: pytest.MonkeyPatch):
    """Kit authors and selects the single camera declared by its cfg."""

    class _FakeViewportApi:
        def __init__(self):
            self.set_active_camera_calls = []

        def set_active_camera(self, camera_path):
            self.set_active_camera_calls.append(camera_path)

    class _FakeViewportWindow:
        def __init__(self):
            self.viewport_api = _FakeViewportApi()

    viewport_window = _FakeViewportWindow()
    viewport_utility = type(
        "ViewportUtility",
        (),
        {
            "create_viewport_window": staticmethod(lambda **kwargs: viewport_window),
            "get_active_viewport_window": staticmethod(lambda: viewport_window),
        },
    )
    monkeypatch.setitem(sys.modules, "omni", type(sys)("omni"))
    monkeypatch.setitem(sys.modules, "omni.kit", type(sys)("omni.kit"))
    monkeypatch.setitem(sys.modules, "omni.kit.viewport", type(sys)("omni.kit.viewport"))
    monkeypatch.setitem(sys.modules, "omni.kit.viewport.utility", viewport_utility)
    monkeypatch.setitem(sys.modules, "omni.ui", type("OmniUi", (), {"DockPosition": object})())

    cfg = KitVisualizerCfg()
    visualizer = kit_visualizer.KitVisualizer(cfg)
    visualizer._runtime_headless = False

    visualizer._setup_viewport()

    assert not cfg.streaming_view
    assert UsdGeom.Camera.Get(SimulationContext.instance().stage, cfg.prim_path)
    assert viewport_window.viewport_api.set_active_camera_calls == [cfg.prim_path]


def test_kit_visualizer_camera_is_a_global_clone_plan_row() -> None:
    from isaaclab.cloner import make_clone_plan

    cfg = KitVisualizerCfg()
    plan = make_clone_plan((cfg,), num_clones=2, env_spacing=1.0)

    assert plan.sources == plan.destinations == (cfg.prim_path,)
    assert plan.cfg_rows == {id(cfg): (0,)}


def test_kit_visualizer_rejects_camera_outside_clone_plan() -> None:
    """Kit cannot draw a cfg-owned camera omitted from the shared clone lifecycle."""
    visualizer = kit_visualizer.KitVisualizer(KitVisualizerCfg())

    with pytest.raises(RuntimeError, match="camera is not covered by the clone plan"):
        visualizer.initialize(cast(Any, object()), _completed_clone_plan(2))


def test_kit_visualizer_cfg_camera_accepts_set_camera_view(monkeypatch: pytest.MonkeyPatch):
    """The cfg-owned Kit camera follows SimulationContext set_camera_view updates."""
    applied_camera_poses = []
    monkeypatch.setattr(
        kit_visualizer.KitVisualizer,
        "_set_viewport_camera",
        lambda self, eye, target: applied_camera_poses.append((tuple(eye), tuple(target))),
    )

    visualizer = kit_visualizer.KitVisualizer(KitVisualizerCfg())
    visualizer._is_initialized = True
    applied_camera_poses.clear()

    visualizer.set_camera_view((1.0, 2.0, 3.0), (0.0, 0.0, 1.0))

    assert applied_camera_poses == [((1.0, 2.0, 3.0), (0.0, 0.0, 1.0))]


def test_kit_streaming_view_requires_camera_rendering(monkeypatch: pytest.MonkeyPatch):
    """A requested Kit streaming view fails instead of silently disabling itself."""
    settings = type("Settings", (), {"get": lambda _self, _key, _default: False})()
    monkeypatch.setattr(kit_visualizer, "get_settings_manager", lambda: settings)
    visualizer = object.__new__(kit_visualizer.KitVisualizer)
    visualizer.cfg = KitVisualizerCfg(streaming_view=True)

    with pytest.raises(RuntimeError, match="requires camera rendering"):
        visualizer._setup_streaming_view()


def test_kit_visualizer_rejects_unknown_dock_position():
    """An invalid dock position cannot silently select the SAME dock."""
    with pytest.raises(ValueError, match="Unknown dock_position"):
        kit_visualizer.KitVisualizer(KitVisualizerCfg(dock_position="diagonal"))


@pytest.mark.parametrize("field", ["max_visible_envs", "visible_env_indices"])
def test_kit_visualizer_rejects_shared_stage_visibility_filters(field: str):
    """A viewport cannot implement view-local filtering by mutating the shared cloned stage."""
    value = 1 if field == "max_visible_envs" else [0]
    with pytest.raises(ValueError, match="does not support partial environment visibility"):
        kit_visualizer.KitVisualizer(KitVisualizerCfg(**{field: value}))


def test_kit_visualizer_app_failures_propagate(monkeypatch: pytest.MonkeyPatch):
    """A selected interactive Kit step cannot disappear behind an app-update fallback."""
    from types import ModuleType, SimpleNamespace

    omni = ModuleType("omni")
    omni.__path__ = []
    omni_kit = ModuleType("omni.kit")
    omni_kit.__path__ = []
    omni_app = ModuleType("omni.kit.app")
    omni.kit = omni_kit
    omni_kit.app = omni_app
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.kit", omni_kit)
    monkeypatch.setitem(sys.modules, "omni.kit.app", omni_app)

    writes = []
    settings = SimpleNamespace(set_bool=lambda _key, value: writes.append(value))
    monkeypatch.setattr(kit_visualizer, "get_settings_manager", lambda: settings)
    visualizer = object.__new__(kit_visualizer.KitVisualizer)
    visualizer.cfg = KitVisualizerCfg()
    visualizer._is_initialized = True
    visualizer._runtime_headless = False
    visualizer._app_pumped_this_step = False
    visualizer._sim_time = 0.0
    visualizer._step_counter = 0
    visualizer._scene_data_provider = SimpleNamespace(
        request_transforms=lambda *_args: None, request_points=lambda *_args: None
    )
    visualizer._clone_ctx = SimpleNamespace(_update_fabric_hierarchy=lambda: None)
    visualizer._clone_plan = _completed_clone_plan(1)
    visualizer.is_training_paused = lambda: False

    omni_app.get_app = lambda: SimpleNamespace(is_running=lambda: False)
    with pytest.raises(RuntimeError, match="app is not running"):
        visualizer.step(0.0)

    def fail_update():
        raise AttributeError("update failed")

    omni_app.get_app = lambda: SimpleNamespace(is_running=lambda: True, update=fail_update)
    with pytest.raises(AttributeError, match="update failed"):
        visualizer.step(0.0)
    assert writes == [False, True]


def test_headless_kit_defers_fabric_until_a_viewport_frame_is_requested():
    """A headless Kit sink does no SDP/Fabric work during an uncaptured step."""
    calls = []
    visualizer = object.__new__(kit_visualizer.KitVisualizer)
    visualizer.cfg = KitVisualizerCfg(headless=True)
    visualizer._is_initialized = True
    visualizer._runtime_headless = True
    visualizer._app_pumped_this_step = False
    visualizer._sim_time = 0.0
    visualizer._step_counter = 0
    visualizer._scene_data_provider = SimpleNamespace(
        request_transforms=lambda *_args: calls.append("transforms"),
        request_points=lambda *_args: calls.append("points"),
    )
    visualizer._clone_ctx = SimpleNamespace(_update_fabric_hierarchy=lambda: calls.append("hierarchy"))
    visualizer._clone_plan = _completed_clone_plan(1)

    visualizer.step(0.1)
    assert calls == []

    visualizer._prepare_viewport_frame()
    assert calls == ["transforms", "hierarchy"]


def test_kit_visualizer_pause_query_failures_propagate(monkeypatch: pytest.MonkeyPatch):
    """A Kit settings failure cannot be interpreted as an unpaused simulation."""

    class _Settings:
        @staticmethod
        def get(_key):
            raise RuntimeError("settings failed")

    monkeypatch.setattr(kit_visualizer, "get_settings_manager", _Settings)
    visualizer = object.__new__(kit_visualizer.KitVisualizer)

    with pytest.raises(RuntimeError, match="settings failed"):
        visualizer.is_training_paused()


def test_kit_visualizer_docking_failures_propagate(monkeypatch: pytest.MonkeyPatch):
    """Missing requested Kit windows cannot become successful initialization."""
    import asyncio
    from types import ModuleType, SimpleNamespace

    async def next_update_async():
        pass

    omni = ModuleType("omni")
    omni.__path__ = []
    omni_kit = ModuleType("omni.kit")
    omni_kit.__path__ = []
    omni_app = ModuleType("omni.kit.app")
    omni_app.get_app = lambda: SimpleNamespace(next_update_async=next_update_async)
    omni_ui = ModuleType("omni.ui")
    omni_ui.Workspace = SimpleNamespace(get_window=lambda _name: None)
    omni.kit = omni_kit
    omni.ui = omni_ui
    omni_kit.app = omni_app
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.kit", omni_kit)
    monkeypatch.setitem(sys.modules, "omni.kit.app", omni_app)
    monkeypatch.setitem(sys.modules, "omni.ui", omni_ui)

    visualizer = object.__new__(kit_visualizer.KitVisualizer)
    with pytest.raises(RuntimeError, match="Could not find viewport window"):
        asyncio.run(visualizer._dock_viewport_async("Missing Viewport", object()))
    with pytest.raises(RuntimeError, match="Could not dock streaming window"):
        asyncio.run(visualizer._dock_image_window_async("Missing Streaming", object()))


def test_kit_visualizer_render_product_failures_propagate(monkeypatch: pytest.MonkeyPatch):
    """A failed selected headless render product cannot return a stale frame."""
    from types import ModuleType, SimpleNamespace

    omni = ModuleType("omni")
    omni.__path__ = []
    omni_kit = ModuleType("omni.kit")
    omni_kit.__path__ = []
    omni_app = ModuleType("omni.kit.app")
    omni_replicator = ModuleType("omni.replicator")
    omni_replicator.__path__ = []
    omni_replicator_core = ModuleType("omni.replicator.core")
    omni.kit = omni_kit
    omni.kit.app = omni_app
    omni.replicator = omni_replicator
    omni.replicator.core = omni_replicator_core
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.kit", omni_kit)
    monkeypatch.setitem(sys.modules, "omni.kit.app", omni_app)
    monkeypatch.setitem(sys.modules, "omni.replicator", omni_replicator)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", omni_replicator_core)

    def fail_resume():
        raise RuntimeError("resume failed")

    visualizer = object.__new__(kit_visualizer.KitVisualizer)
    visualizer.cfg = KitVisualizerCfg(window_width=2, window_height=2)
    visualizer._runtime_headless = True
    visualizer._rgb_annotator = SimpleNamespace(get_data=lambda: np.zeros((2, 2, 4), dtype=np.uint8))
    visualizer._rgb_render_product = SimpleNamespace(resume=fail_resume, pause=lambda: None)
    visualizer._app_pumped_this_step = True

    with pytest.raises(RuntimeError, match="resume failed"):
        visualizer.render_rgb_array()


def test_kit_visualizer_rejects_unknown_origin_type():
    """An unknown requested camera origin is never changed to world implicitly."""
    visualizer = object.__new__(kit_visualizer.KitVisualizer)
    visualizer.cfg = KitVisualizerCfg(origin_type="invalid")

    with pytest.raises(ValueError, match="Unknown origin_type"):
        visualizer._setup_initial_camera_view()


def test_viser_streaming_upload_failure_propagates():
    """A requested Viser streaming frame cannot disappear behind an upload catch."""

    class _Scene:
        @staticmethod
        def set_background_image(*args, **kwargs):
            raise RuntimeError("upload failed")

    visualizer = object.__new__(viser_visualizer.ViserVisualizer)
    visualizer._streaming = type("Streaming", (), {"composite": lambda _self: np.zeros((2, 2, 3), dtype="uint8")})()
    visualizer._viewer = type("Viewer", (), {"_server": type("Server", (), {"scene": _Scene()})()})()

    with pytest.raises(RuntimeError, match="upload failed"):
        visualizer._push_streaming_frame()


@pytest.mark.parametrize(("headless", "pumps_app"), [(True, False), (False, True)])
def test_kit_visualizer_reports_whether_it_pumps_the_app(headless: bool, pumps_app: bool):
    visualizer = object.__new__(kit_visualizer.KitVisualizer)
    visualizer._runtime_headless = headless

    assert visualizer.pumps_app_update() is pumps_app


def test_headless_kit_keeps_the_cfg_owned_camera_without_a_viewport_import():
    visualizer = kit_visualizer.KitVisualizer(KitVisualizerCfg(headless=True))
    visualizer._runtime_headless = True

    visualizer._setup_viewport()

    assert visualizer._viewport_api is None
    assert UsdGeom.Camera.Get(SimulationContext.instance().stage, visualizer.cfg.prim_path)


def test_kit_asset_tracking_requests_the_planned_transform_through_sdp(monkeypatch: pytest.MonkeyPatch):
    """Asset tracking resolves a plan destination once and never reads an asset data object."""
    import torch
    import warp as wp

    from isaaclab.cloner import ClonePlan
    from isaaclab.cloner.clone_plan import RigidBodyLayout
    from isaaclab.scene_data import SceneDataFormat

    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot",),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.arange(2, dtype=np.int64),
        positions=np.zeros((2, 3), dtype=np.float32),
        is_complete=True,
        rigid_body_prototypes=(
            RigidBodyLayout(
                "/World/envs/env_{}/Robot/base",
                "/World/envs/env_*/Robot/base",
                0,
                None,
                "base",
                clone_mask=np.ones(2, dtype=np.bool_),
            ),
        ),
        _env_ids_cpu=(0, 1),
    )
    fake_sim_type = type("Sim", (), {"get_clone_plan": lambda _self: plan})
    monkeypatch.setattr(SimulationContext, "instance", classmethod(lambda _cls: fake_sim_type()))

    provider = type(
        "Provider",
        (),
        {
            "request_transforms": lambda _self, output_format: (
                type("Output", (), {"positions": wp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=wp.vec3)})()
                if output_format is SceneDataFormat.Vec3_Quat
                else None
            ),
        },
    )()
    visualizer = object.__new__(kit_visualizer.KitVisualizer)
    visualizer.cfg = KitVisualizerCfg(
        origin_type="asset", origin_track_path="/World/envs/env_.*/Robot", origin_env_index=1
    )
    visualizer._clone_plan = plan
    visualizer._scene_data_provider = provider
    visualizer._viewer_origin = None
    visualizer._origin_index = None
    visualizer._apply_viewer_origin_to_camera = lambda: None

    visualizer._setup_initial_camera_view()
    visualizer._update_asset_tracking_camera()

    assert visualizer._origin_index == 1
    assert torch.equal(visualizer.viewer_origin.cpu(), torch.tensor([1.0, 2.0, 3.0]))


def test_kit_visualizer_reuses_the_last_composite_while_training_is_paused():
    """Regression: a paused Kit panel reuses the scene camera's last composite.

    ``step()`` advances ``_step_counter`` whether or not training is paused, so the per-step
    composite cache misses every step while paused. The panel must keep the picture stable while
    the scene camera is not being updated.
    """
    from types import SimpleNamespace

    import torch

    from isaaclab.visualizers.streaming_view import StreamingView

    rgb = torch.zeros((2, 4, 4, 4), dtype=torch.uint8)
    camera = SimpleNamespace(
        num_instances=2,
        prim_paths=("/World/envs/env_0/VisualizerCamera", "/World/envs/env_1/VisualizerCamera"),
        cfg=SimpleNamespace(prim_path="/World/envs/env_[^/]+/VisualizerCamera"),
        data=SimpleNamespace(output={"rgb": rgb}),
    )
    camera_path = "/World/envs/env_[^/]+/VisualizerCamera"
    cfg = KitVisualizerCfg(streaming_view=True, streaming_envs=[0, 1], streaming_camera=camera_path)

    visualizer = object.__new__(kit_visualizer.KitVisualizer)
    visualizer.cfg = cfg
    visualizer._streaming = StreamingView(cfg, {camera_path: camera})
    visualizer._camera_image_provider = None
    visualizer._step_counter = 0

    # Running: each step composites the scene-owned camera's current output.
    visualizer.is_training_paused = lambda: False
    for _ in range(3):
        visualizer._step_counter += 1
        visualizer._update_camera_image_panel()
    assert visualizer._streaming.last_composite is not None

    previous = visualizer._streaming.last_composite
    visualizer.is_training_paused = lambda: True
    for _ in range(5):
        visualizer._step_counter += 1
        visualizer._update_camera_image_panel()
    assert visualizer._streaming.last_composite is previous


def test_kit_visualizer_authors_camera_pose_without_viewport_state() -> None:
    visualizer = kit_visualizer.KitVisualizer(KitVisualizerCfg())
    eye = (1.0, 2.0, 3.0)
    target = (4.0, 5.0, 6.0)

    visualizer._set_viewport_camera(eye, target)

    camera_to_world = visualizer._camera_xform_op.Get()
    assert tuple(camera_to_world.ExtractTranslation()) == pytest.approx(eye)
    actual_forward = camera_to_world.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized()
    expected_forward = (Gf.Vec3d(*target) - Gf.Vec3d(*eye)).GetNormalized()
    assert tuple(actual_forward) == pytest.approx(tuple(expected_forward))


# ---------------------------------------------------------------------------
# Shared helpers for config-resolution and initialize_visualizers tests
# ---------------------------------------------------------------------------


class _FakeVisualizerCfg:
    """Minimal visualizer config for testing initialize_visualizers."""

    def __init__(self, visualizer_type: str, *, fail_create: bool = False, fail_init: bool = False):
        self.visualizer_type = visualizer_type
        self.enable_markers = False
        self._fail_create = fail_create
        self._fail_init = fail_init

    def class_type(self, cfg):
        if self._fail_create:
            raise RuntimeError("create failed")
        visualizer = _FakeVisualizer() if not self._fail_init else _FailingInitVisualizer()
        visualizer.cfg = cfg
        return visualizer


class _FailingInitVisualizer(_FakeVisualizer):
    def initialize(self, provider, plan):
        raise RuntimeError("init failed")


def _make_context_with_settings(
    settings: dict,
    visualizer_cfgs=None,
    *,
    has_gui: bool = False,
    has_offscreen_render: bool = False,
):
    """Build a minimal SimulationContext for rendering-state and visualizer initialization tests.

    Centralises the ``object.__new__`` construction so new internal attributes only need to be added
    in one place when the production code changes.
    """
    cfg = type(
        "Cfg",
        (),
        {
            "visualizer_cfgs": visualizer_cfgs,
            "physics": type("PhysicsCfg", (), {"dt": 0.01})(),
            "dt": 0.01,
            "render_interval": 1,
        },
    )()
    ctx = object.__new__(SimulationContext)
    ctx.cfg = cfg
    ctx._has_gui = has_gui
    ctx._has_offscreen_render = has_offscreen_render
    ctx._xr_enabled = False
    ctx._pending_camera_view = None
    ctx._visualizers = []
    ctx._visualizer_cfgs = [] if visualizer_cfgs is None else visualizer_cfgs
    if not isinstance(ctx._visualizer_cfgs, list):
        ctx._visualizer_cfgs = [ctx._visualizer_cfgs]
    ctx._uninitialized_visualizers = []
    ctx._backend_registry = {}
    ctx._backend_clone_roles = {}
    ctx.stage = object()
    ctx._scene_data_provider = _FakeProvider()
    ctx._clone_plan = _completed_clone_plan(1)
    ctx._camera_sensors = {}
    ctx._viz_dt = 0.01
    ctx.get_setting = lambda name: settings.get(name)
    return ctx


def test_visualizer_construction_precedes_clone_and_initialization_follows_it():
    events = []
    cfg = _FakeVisualizerCfg("newton_gl")
    ctx = _make_context_with_settings({}, visualizer_cfgs=[cfg])

    class VisualizerProbe(_FakeVisualizer):
        def __init__(self):
            super().__init__()
            self.cfg = cfg

        def initialize(self, _provider, _plan):
            events.append("initialize")

    def construct(_cfg):
        events.append("construct")
        return VisualizerProbe()

    cfg.class_type = construct
    ctx._construct_visualizers()
    assert events == ["construct"]

    events.append("clone")
    ctx.initialize_visualizers()

    assert events == ["construct", "clone", "initialize"]


def test_visualizer_cfgs_are_only_the_declared_objects():
    first = _FakeVisualizerCfg("newton_gl")
    second = _FakeVisualizerCfg("kit")
    settings = {
        "/isaaclab/visualizer/types": "rerun",
        "/isaaclab/visualizer/explicit": True,
        "/isaaclab/visualizer/disable_all": True,
        "/isaaclab/visualizer/max_visible_envs": 1,
    }
    declared = [first, second]
    ctx = _make_context_with_settings(settings, visualizer_cfgs=declared)

    assert ctx._visualizer_cfgs is declared
    assert ctx._visualizer_cfgs == [first, second]


def test_settings_do_not_synthesize_visualizer_cfgs():
    settings = {
        "/isaaclab/visualizer/types": "newton_rtx",
        "/isaaclab/visualizer/explicit": True,
        "/isaaclab/visualizer/disable_all": False,
        "/isaaclab/visualizer/max_visible_envs": None,
    }
    ctx = _make_context_with_settings(settings)

    assert ctx._visualizer_cfgs == []
    ctx._construct_visualizers()
    assert ctx.visualizers == []


def test_is_rendering_uses_constructed_visualizers_not_settings():
    cfg = _FakeVisualizerCfg("newton_rtx")
    settings = {
        "/isaaclab/render/rtx_sensors": False,
        "/isaaclab/visualizer/types": "rerun",
        "/isaaclab/visualizer/disable_all": True,
    }
    ctx = _make_context_with_settings(settings, visualizer_cfgs=[cfg])
    assert ctx.is_rendering is False
    ctx._construct_visualizers()
    assert ctx.is_rendering is True


def test_declared_visualizer_construction_failure_propagates():
    failing_cfg = _FakeVisualizerCfg("newton_gl", fail_create=True)
    ctx = _make_context_with_settings({}, visualizer_cfgs=[failing_cfg])

    with pytest.raises(RuntimeError, match="create failed"):
        ctx._construct_visualizers()


def test_declared_visualizer_initialization_failure_propagates():
    failing_cfg = _FakeVisualizerCfg("newton_gl", fail_init=True)
    ctx = _make_context_with_settings({}, visualizer_cfgs=[failing_cfg])
    ctx._construct_visualizers()

    with pytest.raises(RuntimeError, match="init failed"):
        ctx.initialize_visualizers()


def test_visualizer_cfg_without_class_type_fails_explicitly():
    cfg = VisualizerCfg()
    ctx = _make_context_with_settings({}, visualizer_cfgs=[cfg])

    with pytest.raises(ValueError, match="class_type"):
        ctx._construct_visualizers()


# ---------------------------------------------------------------------------
# RerunVisualizer streaming-view tests
# ---------------------------------------------------------------------------


def test_rerun_visualizer_setup_streaming_view_sets_flag_and_blueprint_includes_spatial2d():
    """``_setup_streaming_view`` sets ``_streaming_view_active`` and the blueprint shows a Spatial2DView.

    Streams a named scene camera, so no Isaac Sim session is required.
    """
    from types import SimpleNamespace

    import torch

    camera = SimpleNamespace(
        num_instances=2,
        update=lambda **_kwargs: None,
        data=SimpleNamespace(output={"rgb": torch.zeros((2, 1, 1, 4), dtype=torch.uint8)}),
        cfg=SimpleNamespace(prim_path="/World/envs/env_[^/]+/Camera"),
    )

    cfg = RerunVisualizerCfg(open_browser=False, streaming_view=True, streaming_camera="{ENV_REGEX_NS}/Camera")
    visualizer = object.__new__(rerun_visualizer.RerunVisualizer)
    visualizer.cfg = cfg
    visualizer._resolved_visible_env_ids = None
    visualizer._streaming = None
    SimulationContext.instance()._camera_sensors = {"/World/envs/env_[^/]+/Camera": camera}

    # Fake _viewer with its own _streaming_view_active flag.
    class _FakeViewer:
        def __init__(self):
            self._streaming_view_active = False
            self._live_plot_manager_names = []
            self._camera_pose = None

    fake_viewer = _FakeViewer()
    visualizer._viewer = fake_viewer

    visualizer._setup_streaming_view()

    # The viewer flag drives the blueprint, so it has to follow the streaming view's own state.
    assert fake_viewer._streaming_view_active is True
    assert visualizer._streaming.camera is camera
    assert visualizer._streaming.env_ids == [0, 1]

    # --- Verify _get_blueprint returns a blueprint containing Spatial2DView ----

    import rerun.blueprint as rrb

    blueprint_viewer = object.__new__(rerun_visualizer.NewtonViewerRerun)
    blueprint_viewer._streaming_view_active = True
    blueprint_viewer._live_plot_manager_names = []
    blueprint_viewer._camera_pose = None

    bp = blueprint_viewer._get_blueprint()

    # The root container wraps a Spatial2DView for the streaming panel.
    contents = bp.root_container.contents
    flat = []
    stack = list(contents)
    while stack:
        item = stack.pop()
        flat.append(item)
        sub = getattr(item, "contents", None)
        if sub:
            stack.extend(sub)
    assert any(isinstance(item, rrb.Spatial2DView) for item in flat), (
        "_get_blueprint with streaming_view_active=True must include a Spatial2DView panel"
    )
