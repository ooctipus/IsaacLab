# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared physics-manager lifecycle behavior."""

import gc
import weakref
from types import SimpleNamespace

import pytest

from isaaclab.physics import PhysicsEvent, PhysicsManager


def test_close_runs_all_live_stop_listeners_and_aggregates_failures(monkeypatch):
    """STOP fan-out and shared-state cleanup survive an individual listener failure."""

    class TestManager(PhysicsManager):
        def _bind_context(self, sim_context):
            super()._bind_context(sim_context)

        def reset(self, soft=False):
            pass

        def forward(self):
            pass

        def get_scene_data_backend(self):
            return None

        def step(self):
            pass

    events = []
    manager = TestManager(object())
    manager._sim = SimpleNamespace(_physics_manager=manager)
    manager._sim_time = 1.0

    manager.register_callback(
        lambda _payload: events.append("first"),
        PhysicsEvent.STOP,
        order=0,
    )

    class CollectedListener:
        def callback(self, _payload):
            events.append("collected")

    collected_listener = CollectedListener()
    listener_ref = weakref.ref(collected_listener)
    manager.register_callback(collected_listener.callback, PhysicsEvent.STOP, order=1)
    del collected_listener
    gc.collect()
    assert listener_ref() is None

    def failing_listener(_payload):
        events.append("failed")
        raise ReferenceError("listener failure")

    manager.register_callback(
        failing_listener,
        PhysicsEvent.STOP,
        order=2,
    )
    manager.register_callback(
        lambda _payload: events.append("last"),
        PhysicsEvent.STOP,
        order=3,
    )

    with pytest.raises(RuntimeError, match=r"1 callback\(s\) failed") as exc_info:
        manager.close()

    assert isinstance(exc_info.value.__cause__, ReferenceError)
    assert events == ["first", "failed", "last"]
    assert manager._callbacks == {}
    assert manager._sim is None
    assert manager.cfg is manager._cfg
    assert manager._sim_time == 0.0


def test_clear_instance_finishes_teardown_after_physics_close_failure(monkeypatch):
    """A STOP failure is re-raised only after the remaining context teardown."""
    import isaaclab.sim.simulation_context as context_module
    from isaaclab.sim import SimulationContext

    events = []

    class FailingManager:
        def close(self):
            events.append("physics")
            raise RuntimeError("STOP failed")

    class Visualizer:
        def __init__(self, name, error=None):
            self.name = name
            self.error = error

        def close(self):
            events.append(self.name)
            if self.error is not None:
                raise self.error

    class Renderer:
        def __init__(self, name, error=None):
            self.name = name
            self.error = error

        def close(self):
            events.append(self.name)
            if self.error is not None:
                raise self.error

    class Backend:
        def clear(self):
            events.append("backend")

    backend = Backend()
    context = SimpleNamespace(
        _physics_manager=FailingManager(),
        _renderer_entries=[
            Renderer("renderer_failed", LookupError("renderer failed")),
            Renderer("renderer_last"),
        ],
        _visualizers=[
            Visualizer("visualizer_failed", ValueError("visualizer failed")),
            Visualizer("visualizer_last"),
        ],
        _uninitialized_visualizers=[],
        _backend_registry={Backend: backend},
        _backend_clone_roles={Backend: {"physics", "scene"}},
        _camera_sensors={},
        vis_marker_registry=Visualizer("markers"),
    )
    monkeypatch.setattr(SimulationContext, "_instance", context)
    monkeypatch.setattr(context_module.stage_utils, "close_stage", lambda: events.append("stage"))
    monkeypatch.setattr(context_module, "clear_resolve_matching_names_cache", lambda: events.append("cache"))
    monkeypatch.setattr(context_module.gc, "collect", lambda: events.append("gc"))

    with pytest.raises(RuntimeError, match=r"3 error\(s\) occurred during teardown") as exc_info:
        SimulationContext.clear_instance()
    SimulationContext.clear_instance()

    assert str(exc_info.value) == (
        "SimulationContext.clear_instance(): 3 error(s) occurred during teardown: RuntimeError: STOP failed; "
        "LookupError: renderer failed; ValueError: visualizer failed"
    )
    assert str(exc_info.value.__cause__) == "STOP failed"
    assert events == [
        "physics",
        "renderer_failed",
        "renderer_last",
        "visualizer_failed",
        "visualizer_last",
        "markers",
        "backend",
        "stage",
        "cache",
        "gc",
    ]
    assert context._renderer_entries == []
    assert context._visualizers == []
    assert context._backend_registry == {}
    assert context._backend_clone_roles == {}
    assert SimulationContext.instance() is None


def test_clear_instance_drops_owned_context_references_before_garbage_collection(monkeypatch):
    """The singleton and method-local context references are gone before garbage collection."""
    import isaaclab.sim.simulation_context as context_module
    from isaaclab.sim import SimulationContext

    class Manager:
        def close(self):
            pass

    class Renderer:
        def close(self):
            pass

    class Context:
        pass

    context = Context()
    context._physics_manager = Manager()
    context._renderer_entries = [Renderer()]
    context._visualizers = []
    context._uninitialized_visualizers = []
    context._backend_registry = {}
    context._backend_clone_roles = {}
    context._camera_sensors = {}
    context.vis_marker_registry = SimpleNamespace(close=lambda: None)
    context_ref = weakref.ref(context)
    context_alive_during_gc = []

    monkeypatch.setattr(SimulationContext, "_instance", context)
    monkeypatch.setattr(context_module.stage_utils, "close_stage", lambda: None)
    monkeypatch.setattr(context_module, "clear_resolve_matching_names_cache", lambda: None)
    monkeypatch.setattr(
        context_module.gc,
        "collect",
        lambda: context_alive_during_gc.append(context_ref() is not None),
    )
    del context

    SimulationContext.clear_instance()

    assert context_alive_during_gc == [False]
    assert context_ref() is None


@pytest.mark.parametrize("backend", ["physx", "ovphysx", "newton"])
def test_physics_backend_is_declared_by_the_resolved_cfg(backend):
    """The composition root reports cfg identity without inspecting implementation names."""
    from isaaclab.sim import SimulationContext

    context = SimpleNamespace(cfg=SimpleNamespace(physics=SimpleNamespace(backend=backend)))
    assert SimulationContext.physics_backend.fget(context) == backend


def test_physx_family_backends_answer_the_physx_membership_test():
    """Callers gate PhysX-family work with ``"physx" in backend``, and both PhysX backends match.

    The equality case above is the regression: callers used to match on the class name
    ``"physxmanager"``, which never equalled ``"physx"``, so branches written as
    ``if backend == "physx"`` fell through to their "not implemented in Newton" error while
    running on PhysX. Membership held either way; equality did not.
    """
    from isaaclab.sim import SimulationContext

    backend = SimulationContext.physics_backend.fget

    def cfg(name):
        return SimpleNamespace(cfg=SimpleNamespace(physics=SimpleNamespace(backend=name)))

    assert "physx" in backend(cfg("physx"))
    assert "physx" in backend(cfg("ovphysx"))
    assert "physx" not in backend(cfg("newton"))
