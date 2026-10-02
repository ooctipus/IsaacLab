# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Context ownership and lifecycle contracts for native GPU worlds."""

from __future__ import annotations

import ast
import inspect
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import NewtonWorldsBackend, NewtonWorldsBackendCfg, NewtonWorldsCfg, NewtonWorldsManager
from isaaclab_newton.physics import worlds as module

from isaaclab.physics import PhysicsEvent, PhysicsManager


@pytest.fixture
def native_backend(monkeypatch):
    """Isolate context orchestration; actual native physics has its independent GPU oracle."""
    device = wp.get_device("cpu")
    order = []
    stream = SimpleNamespace(
        cuda_stream=0, wait_event=lambda _: order.append("wait"), record_event=lambda _: order.append("record")
    )
    runtime = SimpleNamespace(
        device=device, capture=Mock(return_value=object()), close=Mock(side_effect=lambda **_: order.append("close"))
    )
    constructor = Mock(return_value=runtime)
    monkeypatch.setattr(module, "MuJoCoWorlds", constructor)
    monkeypatch.setattr(wp, "Stream", lambda _: stream)
    monkeypatch.setattr(wp, "Event", lambda _: object())
    monkeypatch.setattr(wp, "get_stream", lambda *_: stream)
    monkeypatch.setattr(wp, "ScopedStream", lambda *_, **__: nullcontext())
    monkeypatch.setattr(wp, "synchronize_device", lambda _: order.append("join"))
    model = SimpleNamespace(opt=SimpleNamespace(timestep=SimpleNamespace(numpy=lambda: np.array([0.005]))))
    cfg = NewtonWorldsBackendCfg(
        prototypes=((model, object()),),
        world_capacities=(8,),
        world_id_capacity=4,
        command_capacity=4,
        dt=0.005,
        substeps=2,
        initial_world_ready_capacities=(0,),
        memory_budget_bytes=2**24,
    )
    backend = NewtonWorldsBackend(cfg)
    yield backend, runtime, constructor, cfg, order
    backend.close()


def test_backend_owns_one_native_runtime_and_graph(native_backend, monkeypatch):
    """Verify coherent reset-only/step replay changes only the native physics permit."""
    backend, runtime, constructor, cfg, order = native_backend
    assert constructor.call_args.args == (cfg.prototypes,)
    assert constructor.call_args.kwargs["initial_world_ready_capacities"] == (0,)
    commands, results, payload = object(), object(), object()
    callbacks = dict(
        validate=Mock(), initialize=Mock(), before_step=Mock(), after_substep=Mock(), application_bindings=Mock()
    )
    backend.prepare(commands, results, retain=(payload,), **callbacks)
    assert runtime.capture.call_args.args == (commands, results)
    assert runtime.capture.call_args.kwargs["retain"] == (payload,)
    assert runtime.capture.call_args.kwargs["refresh_kinematics"] is True
    assert runtime.capture.call_args.kwargs["substeps"] == 2
    for name, callback in callbacks.items():
        assert runtime.capture.call_args.kwargs[name] is callback
    permits = []
    monkeypatch.setattr(wp, "capture_launch", lambda _: permits.append(int(backend._permit.numpy()[0])))
    order.clear()
    backend.forward()
    backend.step()
    assert permits == [0, 1]
    assert order == ["wait", "record", "wait", "record"]
    with pytest.raises(RuntimeError, match="already"):
        backend.prepare(commands, results)
    graph = backend.graph
    backend.close()
    assert backend.graph is None and runtime.capture.return_value is graph
    assert order[-2:] == ["join", "close"]
    runtime.close.assert_called_once_with(streams=(wp.get_stream(backend.device),))
    with pytest.raises(RuntimeError, match="open"):
        backend.step()


@pytest.mark.parametrize("change", [{"dt": 0.01}, {"dt": float("nan")}, {"substeps": 0}, {"substeps": True}])
def test_configuration_rejects_native_time_mismatch_before_runtime_allocation(native_backend, change):
    """Verify declared frame timing cannot diverge from prepared native timing."""
    _, _, constructor, cfg, _ = native_backend
    constructor.reset_mock()
    with pytest.raises(ValueError):
        NewtonWorldsBackend(cfg.replace(**change))
    constructor.assert_not_called()


def test_manager_borrows_context_resource_and_orders_reset_only_lifecycle(native_backend, monkeypatch):
    """Verify the manager neither allocates another runtime nor owns backend closure."""
    backend, runtime, _, _, _ = native_backend
    backend.prepare(object(), object())
    permits = []
    monkeypatch.setattr(wp, "capture_launch", lambda _: permits.append(int(backend._permit.numpy()[0])))
    sim = SimpleNamespace(
        cfg=SimpleNamespace(physics=NewtonWorldsCfg(), device="cpu", dt=0.01),
        physics_manager=NewtonWorldsManager,
        resolve_visualizer_types=lambda: [],
        get_setting=lambda _: None,
    )
    manager = NewtonWorldsManager
    manager.initialize(sim)
    events = []
    manager.register_callback(lambda _: events.append("model"), PhysicsEvent.MODEL_INIT)
    manager.register_callback(lambda _: events.append("ready"), PhysicsEvent.PHYSICS_READY)
    try:
        with pytest.raises(RuntimeError, match="Install"):
            manager.reset()
        manager.install(backend)
        manager.reset()
        manager.step()
        manager.reset(soft=True)
        assert permits == [0, 1, 0]
        assert events == ["model", "ready", "ready"]
        assert manager.get_simulation_time() == pytest.approx(0.01)
        manager.close()
        runtime.close.assert_not_called()
        assert PhysicsManager._sim is None
    finally:
        manager.close()


def test_native_backend_rejects_rendering_and_duplicate_state_authority():
    """Verify unimplemented rendering and dense Newton mirrors are explicit forbidden boundaries."""
    sim = SimpleNamespace(resolve_visualizer_types=lambda: ["newton_gl"], get_setting=lambda _: None)
    with pytest.raises(ValueError, match="headless"):
        NewtonWorldsManager.initialize(sim)
    source = inspect.getsource(module)
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"state_0", "state_1", "replicate", "get_model", "get_state"}
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "SolverMuJoCo"
    assert "SceneDataBackend" in source
