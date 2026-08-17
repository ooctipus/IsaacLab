# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton manager integration with physics incident capture."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import (
    MJWarpDebugOperationProvider,
    NewtonCfg,
    NewtonDebugCaptureCfg,
    NewtonDebugReplayCfg,
    NewtonKaminoManager,
    NewtonManager,
    NewtonMJWarpManager,
)
from isaaclab_newton.physics import newton_manager as newton_manager_module
from isaaclab_newton.physics._debug_capture import DebugCaptureError, DebugCapturePlan
from isaaclab_newton.physics._incident_recorder import _OperationProviderCleanupError
from newton import ModelBuilder
from newton.solvers import SolverBase, SolverKamino, SolverMuJoCo

from isaaclab.physics import PhysicsManager


class _ZeroMask:
    """Minimal reset mask used by manager step tests."""

    def __init__(self, values: tuple[bool, ...] = (False,)) -> None:
        self.values = np.asarray(values, dtype=np.bool_)

    def numpy(self) -> np.ndarray:
        """Return a host snapshot of the mask."""
        return self.values.copy()

    def zero_(self) -> None:
        """Consume the mask."""
        self.values.fill(False)


class _State:
    """Minimal Newton state used by substep tests."""

    def clear_forces(self) -> None:
        """Clear transient forces."""


def _solver_instance() -> SolverBase:
    """Create an uninitialized solver instance for storage-discovery tests."""
    return object.__new__(SolverBase)


def test_initialize_solver_binds_recorder_after_solver_and_contacts(monkeypatch):
    """Recorder construction sees finalized solver providers and live workflow context."""
    events: list[str] = []
    model = object()
    state = object()
    control = object()
    solver = _solver_instance()
    contacts = object()
    collision_pipeline = object()
    captured: dict[str, object] = {}

    class _Recorder:
        history_length = 3
        capture_per_substep = False

        def __init__(self, recorder_model, recorder_cfg, **providers) -> None:
            events.append("recorder")
            captured.update(providers)
            captured["model"] = recorder_model
            captured["cfg"] = recorder_cfg
            captured["context"] = providers["context_provider"]()

    def _build_solver(cls, built_model, solver_cfg) -> None:
        assert built_model is model
        events.append("solver")
        NewtonManager._solver = solver

    def _initialize_contacts(cls) -> None:
        events.append("contacts")
        NewtonManager._contacts = contacts
        NewtonManager._collision_pipeline = collision_pipeline

    monkeypatch.setattr(newton_manager_module, "PhysicsIncidentRecorder", _Recorder)
    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        NewtonCfg(debug_capture=NewtonDebugCaptureCfg(record_scene=False)),
        raising=False,
    )
    monkeypatch.setattr(PhysicsManager, "_sim", None, raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cpu", raising=False)
    monkeypatch.setattr(PhysicsManager, "_debug_context_providers", {"workflow_value": lambda: 7}, raising=False)
    monkeypatch.setattr(NewtonManager, "_model", model, raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", state, raising=False)
    monkeypatch.setattr(NewtonManager, "_control", control, raising=False)
    monkeypatch.setattr(NewtonManager, "_usdrt_stage", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_use_newton_actuators_active", True, raising=False)
    monkeypatch.setattr(NewtonManager, "_debug_operation_provider_cleanup_pending", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_debug_incident_triggers", {}, raising=False)
    monkeypatch.setattr(NewtonManager, "_build_solver", classmethod(_build_solver))
    monkeypatch.setattr(NewtonManager, "_initialize_contacts", classmethod(_initialize_contacts))
    monkeypatch.setattr(NewtonManager, "_eval_fk_impl", classmethod(lambda cls, *args: events.append("fk")))
    monkeypatch.setattr(NewtonManager, "_mark_transforms_dirty", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "get_physics_dt", classmethod(lambda cls: 0.02))

    NewtonManager.initialize_solver()

    assert events == ["solver", "contacts", "fk", "recorder"]
    assert captured["model"] is model
    assert captured["state"] is state
    assert captured["control"] is control
    assert captured["solver"] is solver
    assert captured["contacts"] is contacts
    assert captured["collision_pipeline"] is collision_pipeline
    assert captured["scene_exporter"] is None
    assert captured["context"] == {"workflow_value": 7}


def test_initialize_solver_freezes_recorder_after_graph_warmup(monkeypatch):
    """Lazy solver buffers created by graph warmup belong to the frozen schema."""
    events: list[str] = []
    solver = _solver_instance()

    class _Recorder:
        history_length = 3
        capture_per_substep = False

        def __init__(self, _model, _cfg, **providers) -> None:
            events.append("recorder")
            assert providers["solver"].lazy_buffer == "allocated"

    def _build_solver(cls, _model, _solver_cfg) -> None:
        events.append("solver")
        NewtonManager._solver = solver

    def _capture_graph(cls) -> None:
        events.append("graph")
        solver.lazy_buffer = "allocated"

    monkeypatch.setattr(newton_manager_module, "PhysicsIncidentRecorder", _Recorder)
    monkeypatch.setattr(PhysicsManager, "_cfg", NewtonCfg(debug_capture=NewtonDebugCaptureCfg()), raising=False)
    monkeypatch.setattr(PhysicsManager, "_sim", None, raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cuda:0", raising=False)
    monkeypatch.setattr(NewtonManager, "_model", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_control", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_incident_recorder", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_usdrt_stage", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_use_newton_actuators_active", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_debug_operation_provider_cleanup_pending", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_debug_incident_triggers", {}, raising=False)
    monkeypatch.setattr(NewtonManager, "_build_solver", classmethod(_build_solver))
    monkeypatch.setattr(NewtonManager, "_initialize_contacts", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "_eval_fk_impl", classmethod(lambda cls, *args: None))
    monkeypatch.setattr(NewtonManager, "_mark_transforms_dirty", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "_capture_or_defer_graph", classmethod(_capture_graph))
    monkeypatch.setattr(NewtonManager, "get_physics_dt", classmethod(lambda cls: 0.02))

    NewtonManager.initialize_solver()

    assert events == ["solver", "graph", "recorder"]


def test_initialize_solver_retains_provider_when_recorder_cleanup_fails(monkeypatch):
    """Manager retains the exact provider whose recorder cleanup must be retried."""

    class _Provider:
        def __init__(self) -> None:
            self.close_count = 0

        def bind(self, solver) -> None:
            pass

        def snapshot(self):
            return {"operation": np.asarray([1.0], dtype=np.float32)}

        def close(self) -> None:
            self.close_count += 1

    class _Recorder:
        def __init__(self, *args, **kwargs) -> None:
            raise _OperationProviderCleanupError(
                "capture-plan initialization",
                DebugCaptureError("schema binding failed"),
                DebugCaptureError("operation_provider.close() failed"),
            )

    provider = _Provider()
    solver = _solver_instance()

    def _build_solver(cls, model, solver_cfg) -> None:
        NewtonManager._solver = solver

    monkeypatch.setattr(newton_manager_module, "PhysicsIncidentRecorder", _Recorder)
    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        NewtonCfg(
            use_cuda_graph=False,
            debug_capture=NewtonDebugCaptureCfg(record_operations=True, record_scene=False),
        ),
        raising=False,
    )
    monkeypatch.setattr(PhysicsManager, "_sim", None, raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cpu", raising=False)
    monkeypatch.setattr(NewtonManager, "_model", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_control", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_contacts", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_collision_pipeline", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_incident_recorder", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_debug_operation_provider", provider, raising=False)
    monkeypatch.setattr(NewtonManager, "_debug_operation_provider_cleanup_pending", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_debug_incident_triggers", {}, raising=False)
    monkeypatch.setattr(NewtonManager, "_usdrt_stage", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_use_newton_actuators_active", True, raising=False)
    monkeypatch.setattr(NewtonManager, "_build_solver", classmethod(_build_solver))
    monkeypatch.setattr(NewtonManager, "_initialize_contacts", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "_eval_fk_impl", classmethod(lambda cls, *args: None))
    monkeypatch.setattr(NewtonManager, "_mark_transforms_dirty", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "get_physics_dt", classmethod(lambda cls: 0.02))

    with pytest.raises(_OperationProviderCleanupError, match="Provider ownership is retained"):
        NewtonManager.initialize_solver()

    assert NewtonManager._incident_recorder is None
    assert NewtonManager._debug_operation_provider is provider
    assert NewtonManager._debug_operation_provider_cleanup_pending is True

    NewtonManager.clear()

    assert provider.close_count == 1
    assert NewtonManager._debug_operation_provider is None
    assert NewtonManager._debug_operation_provider_cleanup_pending is False


def test_clear_retries_pending_provider_cleanup_without_dropping_ownership(monkeypatch):
    """A failed retry preserves manager ownership until a later clear succeeds."""

    class _Provider:
        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1
            if self.close_count == 1:
                raise ValueError("hook teardown still busy")

    provider = _Provider()
    monkeypatch.setattr(PhysicsManager, "_cfg", NewtonCfg(use_cuda_graph=False), raising=False)
    monkeypatch.setattr(NewtonManager, "_incident_recorder", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_debug_operation_provider", provider, raising=False)
    monkeypatch.setattr(NewtonManager, "_debug_operation_provider_cleanup_pending", True, raising=False)

    with pytest.raises(RuntimeError, match=r"cleanup is pending.*clear\(\)"):
        NewtonManager.initialize_solver()
    assert provider.close_count == 0

    with pytest.raises(RuntimeError, match="remains owned.*clear"):
        NewtonManager.clear()

    assert provider.close_count == 1
    assert NewtonManager._debug_operation_provider is provider
    assert NewtonManager._debug_operation_provider_cleanup_pending is True

    NewtonManager.clear()

    assert provider.close_count == 2
    assert NewtonManager._debug_operation_provider is None
    assert NewtonManager._debug_operation_provider_cleanup_pending is False


def test_step_orders_reset_capture_and_observation(monkeypatch):
    """A step resets the solver before pre-capture and observes only after simulation."""
    events: list[str] = []

    class _Solver:
        def notify_model_changed(self, change: object) -> None:
            events.append(f"notify:{change}")

    class _Recorder:
        capture_per_substep = False
        halted = False

        def rearm_reset_worlds(self, world_ids: tuple[int, ...], *, all_worlds: bool = False) -> None:
            events.append(f"rearm:{world_ids}:{all_worlds}")

        def capture_pre(self, state: object) -> None:
            events.append("capture_pre")

        def step(self, state: object, sim_time: float, *, trigger_results: dict) -> None:
            events.append("observe")

    cfg = NewtonCfg(debug_capture=NewtonDebugCaptureCfg(), use_cuda_graph=False)
    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(is_playing=lambda: True), raising=False)
    monkeypatch.setattr(PhysicsManager, "_cfg", cfg, raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cpu", raising=False)
    monkeypatch.setattr(PhysicsManager, "_sim_time", 0.0, raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", _Solver(), raising=False)
    monkeypatch.setattr(NewtonManager, "_model_changes", {9}, raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_model", SimpleNamespace(world_count=3), raising=False)
    monkeypatch.setattr(NewtonManager, "_world_reset_mask", _ZeroMask((False, True, False, False)), raising=False)
    monkeypatch.setattr(NewtonManager, "_fk_reset_mask", _ZeroMask(), raising=False)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_graph", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_solver_dt", 0.01, raising=False)
    monkeypatch.setattr(NewtonManager, "_num_substeps", 1, raising=False)
    monkeypatch.setattr(NewtonManager, "_decimation", 1, raising=False)
    monkeypatch.setattr(NewtonManager, "_incident_recorder", _Recorder(), raising=False)
    monkeypatch.setattr(NewtonManager, "_usdrt_stage", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_particle_visual_prims", {}, raising=False)
    monkeypatch.setattr(NewtonManager, "_reset_solver_internals_delegate", lambda mask: events.append("reset"))
    monkeypatch.setattr(NewtonManager, "_eval_fk", lambda world_mask, fk_mask: events.append("fk"))
    monkeypatch.setattr(NewtonManager, "_is_all_graphable", classmethod(lambda cls: True))
    monkeypatch.setattr(NewtonManager, "_simulate_full", classmethod(lambda cls: events.append("simulate")))
    monkeypatch.setattr(NewtonManager, "_log_solver_debug", classmethod(lambda cls: None))

    NewtonManager.step()

    assert events == [
        "reset",
        "notify:9",
        "rearm:(1,):False",
        "reset",
        "fk",
        "capture_pre",
        "simulate",
        "observe",
    ]


def test_per_substep_halt_stops_remaining_decimation(monkeypatch):
    """A per-substep incident halt stops later substeps, decimation, and sensor updates."""
    events: list[str] = []

    class _Recorder:
        capture_per_substep = True
        halted = False

        def record_step_replay_pre(self, state: object, *, sim_time: float, substep_idx: int) -> None:
            events.append("replay_pre")

        def record_step_replay_post(self, state: object) -> None:
            events.append("replay_post")

        def observe_substep(self, state: object, sim_time: float, *, substep_idx: int, trigger_results: dict) -> None:
            events.append("observe")
            self.halted = True

    recorder = _Recorder()
    monkeypatch.setattr(PhysicsManager, "_sim_time", 0.0, raising=False)
    monkeypatch.setattr(NewtonManager, "_incident_recorder", recorder, raising=False)
    monkeypatch.setattr(NewtonManager, "_use_single_state", True, raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", _State(), raising=False)
    monkeypatch.setattr(NewtonManager, "_control", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_solver_dt", 0.01, raising=False)
    monkeypatch.setattr(NewtonManager, "_num_substeps", 3, raising=False)
    monkeypatch.setattr(NewtonManager, "_decimation", 4, raising=False)
    monkeypatch.setattr(NewtonManager, "_collision_decimation", 0, raising=False)
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_adapter", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_post_actuator_callbacks", [], raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_step_solver",
        classmethod(lambda cls, *args: events.append("solve")),
    )
    monkeypatch.setattr(NewtonManager, "_update_sensors", classmethod(lambda cls, contacts: events.append("sensors")))

    NewtonManager._simulate_full()

    assert events == ["replay_pre", "solve", "replay_post", "observe"]


def test_per_substep_indices_are_monotonic_across_decimation(monkeypatch):
    """Per-substep capture uses one manager-step-wide sequence across solver dispatches."""
    replay_records: list[tuple[int, float]] = []
    observations: list[tuple[int, float]] = []

    class _Recorder:
        capture_per_substep = True
        halted = False

        def record_step_replay_pre(self, state: object, *, sim_time: float, substep_idx: int) -> None:
            replay_records.append((substep_idx, sim_time))

        def record_step_replay_post(self, state: object) -> None:
            pass

        def observe_substep(
            self,
            state: object,
            sim_time: float,
            *,
            substep_idx: int,
            trigger_results: dict,
        ) -> None:
            assert substep_idx == len(observations)
            observations.append((substep_idx, sim_time))

    monkeypatch.setattr(PhysicsManager, "_sim_time", 2.0, raising=False)
    monkeypatch.setattr(NewtonManager, "_incident_recorder", _Recorder(), raising=False)
    monkeypatch.setattr(NewtonManager, "_use_single_state", True, raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", _State(), raising=False)
    monkeypatch.setattr(NewtonManager, "_control", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_solver_dt", 0.1, raising=False)
    monkeypatch.setattr(NewtonManager, "_num_substeps", 2, raising=False)
    monkeypatch.setattr(NewtonManager, "_decimation", 3, raising=False)
    monkeypatch.setattr(NewtonManager, "_collision_decimation", 0, raising=False)
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_adapter", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_post_actuator_callbacks", [], raising=False)
    monkeypatch.setattr(NewtonManager, "_debug_incident_triggers", {}, raising=False)
    monkeypatch.setattr(NewtonManager, "_step_solver", classmethod(lambda cls, *args: None))
    monkeypatch.setattr(NewtonManager, "_update_sensors", classmethod(lambda cls, contacts: None))

    NewtonManager._simulate_full()

    assert [record[0] for record in replay_records] == list(range(6))
    assert [record[0] for record in observations] == list(range(6))
    assert [record[1] for record in replay_records] == pytest.approx([2.0, 2.1, 2.2, 2.3, 2.4, 2.5])
    assert [record[1] for record in observations] == pytest.approx([2.1, 2.2, 2.3, 2.4, 2.5, 2.6])


def test_discover_solver_provider_returns_single_solver_exactly(monkeypatch):
    """A conventional manager keeps the solver as the direct capture provider."""
    solver = _solver_instance()
    monkeypatch.setattr(NewtonManager, "_solver", solver, raising=False)

    assert NewtonManager._discover_solver_provider() is solver


def test_discover_solver_provider_returns_sorted_coupled_mapping(monkeypatch):
    """Coupled managers expose every unique MRO solver slot deterministically."""

    class _CoupledManager(NewtonManager):
        pass

    dummy = _solver_instance()
    rigid = _solver_instance()
    soft = _solver_instance()
    monkeypatch.setattr(NewtonManager, "_solver", dummy, raising=False)
    monkeypatch.setattr(_CoupledManager, "_rigid_solver", rigid, raising=False)
    monkeypatch.setattr(_CoupledManager, "_soft_solver", soft, raising=False)
    monkeypatch.setattr(_CoupledManager, "_solver_alias", dummy, raising=False)

    provider = _CoupledManager._discover_solver_provider()

    assert isinstance(provider, dict)
    assert list(provider) == ["rigid_solver", "soft_solver", "solver"]
    assert provider == {"rigid_solver": rigid, "soft_solver": soft, "solver": dummy}


def test_kamino_step_orders_reset_fk_and_incident_capture(monkeypatch):
    """Kamino preserves its reset/FK path around dispatch-level incident capture."""
    events: list[str] = []

    class _Recorder:
        capture_per_substep = False
        halted = False

        def capture_pre(self, state: object) -> None:
            events.append("capture_pre")

        def step(self, state: object, sim_time: float, *, trigger_results: dict) -> None:
            events.append("observe")

    state = object()
    cfg = NewtonCfg(debug_capture=NewtonDebugCaptureCfg(), use_cuda_graph=False)
    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(is_playing=lambda: True), raising=False)
    monkeypatch.setattr(PhysicsManager, "_cfg", cfg, raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cpu", raising=False)
    monkeypatch.setattr(PhysicsManager, "_sim_time", 0.0, raising=False)
    monkeypatch.setattr(NewtonKaminoManager, "_model_changes", set(), raising=False)
    monkeypatch.setattr(NewtonKaminoManager, "_state_0", state, raising=False)
    monkeypatch.setattr(NewtonKaminoManager, "_model", SimpleNamespace(world_count=1), raising=False)
    monkeypatch.setattr(NewtonManager, "_world_reset_mask", _ZeroMask((False, False)), raising=False)
    monkeypatch.setattr(NewtonManager, "_fk_reset_mask", _ZeroMask(), raising=False)
    monkeypatch.setattr(NewtonKaminoManager, "_graph_capture_pending", False, raising=False)
    monkeypatch.setattr(NewtonKaminoManager, "_graph", None, raising=False)
    monkeypatch.setattr(NewtonKaminoManager, "_solver_dt", 0.01, raising=False)
    monkeypatch.setattr(NewtonKaminoManager, "_num_substeps", 1, raising=False)
    monkeypatch.setattr(NewtonKaminoManager, "_incident_recorder", _Recorder(), raising=False)
    monkeypatch.setattr(NewtonKaminoManager, "_usdrt_stage", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_reset_solver_internals_delegate", lambda mask: None)
    monkeypatch.setattr(NewtonManager, "_eval_fk", lambda world_mask, fk_mask: events.append("kamino_fk"))
    monkeypatch.setattr(
        NewtonKaminoManager,
        "_simulate_physics_only",
        classmethod(lambda cls: events.append("simulate")),
    )
    monkeypatch.setattr(NewtonKaminoManager, "_log_solver_debug", classmethod(lambda cls: None))

    NewtonKaminoManager.step()

    assert events == ["kamino_fk", "capture_pre", "simulate", "observe"]
    assert PhysicsManager._sim_time == 0.01


def test_kamino_prepares_real_state_and_solver_schemas_before_recorder_binding(monkeypatch):
    """Kamino's state and solver schemas are complete before strict recorder binding."""
    builder = ModelBuilder()
    NewtonKaminoManager._register_builder_attributes(builder)
    NewtonKaminoManager._register_builder_attributes(builder)
    body = builder.add_body(mass=1.0, inertia=wp.mat33(1.0))
    builder.add_joint_revolute(parent=-1, child=body, axis=(0.0, 0.0, 1.0))
    builder.add_shape_sphere(body=body, radius=0.1)
    builder.add_ground_plane()
    model = builder.finalize(device="cpu")
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    solver = SolverKamino(model, SolverKamino.Config(use_fk_solver=True, use_collision_detector=True))
    monkeypatch.setattr(NewtonManager, "_solver", solver, raising=False)
    monkeypatch.setattr(NewtonManager, "_control", control, raising=False)

    NewtonKaminoManager._prepare_debug_capture_providers()
    state_plan = DebugCapturePlan.build(state_0, root_name="state", include_private=True)
    solver_plan = DebugCapturePlan.build(solver, root_name="solver", include_private=True)
    solver.step(state_0, state_1, control, None, 0.01)

    state_plan.validate_schema(state_0)
    state_plan.validate_schema(state_1)
    solver_plan.validate_schema(solver)
    assert state_0.body_f_total is not None
    assert state_0.joint_q_prev is not None
    assert state_0.joint_lambdas is not None


def test_debug_incident_trigger_registration_is_strict(monkeypatch):
    """Trigger registration requires capture, valid names, callables, and uniqueness."""
    monkeypatch.setattr(NewtonManager, "_debug_incident_triggers", {}, raising=False)
    monkeypatch.setattr(NewtonManager, "_incident_recorder", None, raising=False)
    monkeypatch.setattr(PhysicsManager, "_cfg", NewtonCfg(debug_capture=None), raising=False)

    with pytest.raises(RuntimeError, match="require NewtonCfg.debug_capture"):
        NewtonManager.set_debug_incident_trigger("velocity_spike", lambda context: None)

    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        NewtonCfg(debug_capture=NewtonDebugCaptureCfg()),
        raising=False,
    )
    with pytest.raises(ValueError, match="lower snake case"):
        NewtonManager.set_debug_incident_trigger("VelocitySpike", lambda context: None)
    with pytest.raises(TypeError, match="must be callable"):
        NewtonManager.set_debug_incident_trigger("velocity_spike", object())

    def trigger(context):
        return None

    NewtonManager.set_debug_incident_trigger("velocity_spike", trigger)
    with pytest.raises(ValueError, match="already registered"):
        NewtonManager.set_debug_incident_trigger("velocity_spike", trigger)

    monkeypatch.setattr(NewtonManager, "_incident_recorder", object(), raising=False)
    with pytest.raises(RuntimeError, match="before solver initialization"):
        NewtonManager.set_debug_incident_trigger("energy_spike", trigger)


def test_dispatch_post_triggers_receive_context_in_sorted_order(monkeypatch):
    """Regular capture evaluates sorted triggers once with dispatch-post context."""
    calls: list[tuple[str, NewtonManager.DebugTriggerContext]] = []
    received: dict[str, object] = {}
    solver = _solver_instance()

    class _Recorder:
        capture_per_substep = False
        halted = False

        def step(self, state: object, sim_time: float, *, trigger_results: dict) -> None:
            received["state"] = state
            received["sim_time"] = sim_time
            received["results"] = trigger_results

    monkeypatch.setattr(NewtonManager, "_debug_incident_triggers", {}, raising=False)
    monkeypatch.setattr(NewtonManager, "_incident_recorder", None, raising=False)
    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        NewtonCfg(debug_capture=NewtonDebugCaptureCfg()),
        raising=False,
    )
    monkeypatch.setattr(PhysicsManager, "_device", "cpu", raising=False)
    monkeypatch.setattr(PhysicsManager, "_sim_time", 1.25, raising=False)
    monkeypatch.setattr(NewtonManager, "_model", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_control", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", solver, raising=False)
    monkeypatch.setattr(NewtonManager, "_contacts", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_collision_pipeline", None, raising=False)

    def late(context: NewtonManager.DebugTriggerContext):
        calls.append(("late_trigger", context))
        return

    def early(context: NewtonManager.DebugTriggerContext):
        calls.append(("early_trigger", context))
        return NewtonManager.DebugTriggerResult(reason="velocity threshold", world_ids=(2,))

    NewtonManager.set_debug_incident_trigger("late_trigger", late)
    NewtonManager.set_debug_incident_trigger("early_trigger", early)
    monkeypatch.setattr(NewtonManager, "_incident_recorder", _Recorder(), raising=False)

    NewtonManager._capture_incident_post_state()

    assert [name for name, _ in calls] == ["early_trigger", "late_trigger"]
    context = calls[0][1]
    assert context.phase == "dispatch_post"
    assert context.substep_idx is None
    assert context.sim_time == 1.25
    assert context.solver is solver
    assert received["state"] is NewtonManager._state_0
    assert received["sim_time"] == 1.25
    assert received["results"] == {
        "early_trigger": NewtonManager.DebugTriggerResult(reason="velocity threshold", world_ids=(2,))
    }


def test_per_substep_capture_uses_exact_times_and_solver_post_triggers(monkeypatch):
    """Replay, trigger, and observation timestamps identify each exact solver substep."""
    replay_times: list[float] = []
    observed: list[tuple[int, float, dict]] = []
    trigger_contexts: list[NewtonManager.DebugTriggerContext] = []

    class _Recorder:
        capture_per_substep = True
        halted = False

        def record_step_replay_pre(self, state: object, *, sim_time: float, substep_idx: int) -> None:
            replay_times.append(sim_time)

        def record_step_replay_post(self, state: object) -> None:
            pass

        def observe_substep(
            self,
            state: object,
            sim_time: float,
            *,
            substep_idx: int,
            trigger_results: dict,
        ) -> None:
            observed.append((substep_idx, sim_time, trigger_results))

    solver = _solver_instance()
    monkeypatch.setattr(NewtonManager, "_debug_incident_triggers", {}, raising=False)
    monkeypatch.setattr(NewtonManager, "_incident_recorder", None, raising=False)
    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        NewtonCfg(debug_capture=NewtonDebugCaptureCfg(capture_per_substep=True), use_cuda_graph=False),
        raising=False,
    )
    monkeypatch.setattr(NewtonManager, "_model", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", _State(), raising=False)
    monkeypatch.setattr(NewtonManager, "_control", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", solver, raising=False)
    monkeypatch.setattr(NewtonManager, "_contacts", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_collision_pipeline", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_use_single_state", True, raising=False)
    monkeypatch.setattr(NewtonManager, "_solver_dt", 0.1, raising=False)
    monkeypatch.setattr(NewtonManager, "_num_substeps", 2, raising=False)
    monkeypatch.setattr(NewtonManager, "_collision_decimation", 0, raising=False)
    monkeypatch.setattr(NewtonManager, "_step_solver", classmethod(lambda cls, *args: None))

    def trigger(context: NewtonManager.DebugTriggerContext):
        trigger_contexts.append(context)
        return

    NewtonManager.set_debug_incident_trigger("velocity_spike", trigger)
    monkeypatch.setattr(NewtonManager, "_incident_recorder", _Recorder(), raising=False)

    assert NewtonManager._run_solver_substeps(None, dispatch_time=2.0) is False

    assert replay_times == pytest.approx([2.0, 2.1])
    assert [entry[0] for entry in observed] == [0, 1]
    assert [entry[1] for entry in observed] == pytest.approx([2.1, 2.2])
    assert [context.phase for context in trigger_contexts] == ["solver_post", "solver_post"]
    assert [context.substep_idx for context in trigger_contexts] == [0, 1]
    assert [context.sim_time for context in trigger_contexts] == pytest.approx([2.1, 2.2])
    assert [entry[2] for entry in observed] == [{}, {}]


def test_debug_incident_trigger_errors_are_actionable(monkeypatch):
    """Callback exceptions and return-type mismatches identify the failing trigger."""
    monkeypatch.setattr(NewtonManager, "_debug_incident_triggers", {}, raising=False)
    monkeypatch.setattr(NewtonManager, "_incident_recorder", None, raising=False)
    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        NewtonCfg(debug_capture=NewtonDebugCaptureCfg()),
        raising=False,
    )
    monkeypatch.setattr(NewtonManager, "_solver", _solver_instance(), raising=False)
    monkeypatch.setattr(NewtonManager, "_model", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_control", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_contacts", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_collision_pipeline", None, raising=False)

    NewtonManager.set_debug_incident_trigger("bad_result", lambda context: object())
    with pytest.raises(RuntimeError, match="bad_result.*expected NewtonManager.DebugTriggerResult"):
        NewtonManager._evaluate_debug_incident_triggers(phase="dispatch_post", substep_idx=None, sim_time=0.0)

    monkeypatch.setattr(NewtonManager, "_debug_incident_triggers", {}, raising=False)

    def broken(context):
        raise ValueError("threshold provider failed")

    NewtonManager.set_debug_incident_trigger("broken_trigger", broken)
    with pytest.raises(RuntimeError, match="broken_trigger.*threshold provider failed"):
        NewtonManager._evaluate_debug_incident_triggers(phase="dispatch_post", substep_idx=None, sim_time=0.0)


def _operation_capture_cfg() -> NewtonDebugCaptureCfg:
    """Create an operation-enabled capture config for provider resolution tests."""
    return NewtonDebugCaptureCfg(
        replay=NewtonDebugReplayCfg(enabled=True, record_operations=True),
    )


def _mujoco_solver(*, use_mujoco_cpu: bool) -> SolverMuJoCo:
    """Create an uninitialized SolverMuJoCo carrying only its backend mode."""
    solver = object.__new__(SolverMuJoCo)
    solver.use_mujoco_cpu = use_mujoco_cpu
    return solver


@pytest.mark.parametrize(
    "capture_cfg",
    [
        _operation_capture_cfg(),
        NewtonDebugCaptureCfg(record_operations=True),
    ],
)
@pytest.mark.parametrize("coupled", [False, True])
def test_operation_capture_auto_configures_mjwarp_provider(
    monkeypatch,
    capture_cfg: NewtonDebugCaptureCfg,
    coupled: bool,
):
    """Incident-only and replay operation capture auto-configure MJWarp on Warp CPU or CUDA."""
    solver = _mujoco_solver(use_mujoco_cpu=False)
    solver_provider = {"mjwarp": solver, "other": _solver_instance()} if coupled else solver
    monkeypatch.setattr(NewtonManager, "_debug_operation_provider", None, raising=False)

    provider = NewtonManager._resolve_debug_operation_provider(solver_provider, capture_cfg)

    assert isinstance(provider, MJWarpDebugOperationProvider)
    assert NewtonManager._debug_operation_provider is provider


@pytest.mark.parametrize("capture_cfg", [None, NewtonDebugCaptureCfg()])
def test_registered_operation_provider_requires_enabled_recording(
    monkeypatch,
    capture_cfg: NewtonDebugCaptureCfg | None,
):
    """A forgotten operation recording flag fails before an expensive debug run."""
    monkeypatch.setattr(NewtonManager, "_debug_operation_provider", object(), raising=False)

    with pytest.raises(RuntimeError, match="operation provider.*recording is disabled"):
        NewtonManager._validate_debug_operation_provider_configuration(capture_cfg)


def test_explicit_operation_provider_wins_over_automatic_selection(monkeypatch):
    """An explicitly registered provider is retained for every solver topology."""

    class _Provider:
        def bind(self, solver):
            pass

        def snapshot(self):
            return {}

        def close(self):
            pass

    explicit = _Provider()
    monkeypatch.setattr(NewtonManager, "_debug_operation_provider", explicit, raising=False)

    provider = NewtonManager._resolve_debug_operation_provider(_solver_instance(), _operation_capture_cfg())

    assert provider is explicit


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("cpu", "CPU SolverMuJoCo"),
        ("coupled_none", "0 compatible MJWarp-backed SolverMuJoCo"),
        ("coupled_many", "2 compatible MJWarp-backed SolverMuJoCo"),
        ("other", "SolverBase"),
    ],
)
def test_operation_capture_requires_explicit_provider_for_unsupported_solver(
    monkeypatch,
    kind: str,
    message: str,
):
    """Unsupported, CPU, and ambiguous coupled solvers require explicit registration."""
    if kind == "cpu":
        solver_provider = _mujoco_solver(use_mujoco_cpu=True)
    elif kind == "coupled_none":
        solver_provider = {
            "mujoco_cpu": _mujoco_solver(use_mujoco_cpu=True),
            "other": _solver_instance(),
        }
    elif kind == "coupled_many":
        solver_provider = {
            "mjwarp_a": _mujoco_solver(use_mujoco_cpu=False),
            "mjwarp_b": _mujoco_solver(use_mujoco_cpu=False),
        }
    else:
        solver_provider = _solver_instance()
    monkeypatch.setattr(NewtonManager, "_debug_operation_provider", None, raising=False)

    with pytest.raises(RuntimeError, match=message):
        NewtonManager._resolve_debug_operation_provider(solver_provider, _operation_capture_cfg())


@pytest.mark.parametrize("use_single_state", [True, False])
def test_post_solver_capture_precedes_force_clear(monkeypatch, use_single_state: bool):
    """Replay and incident snapshots retain raw post-solver forces in both state modes."""
    events: list[str] = []

    class _TrackedState:
        def clear_forces(self) -> None:
            events.append("clear_forces")

        def assign(self, other) -> None:
            events.append("assign")

    class _Recorder:
        capture_per_substep = True
        halted = False

        def record_step_replay_pre(self, state, *, sim_time: float, substep_idx: int) -> None:
            pass

        def record_step_replay_post(self, state) -> None:
            events.append("replay_post")

        def observe_substep(
            self,
            state,
            sim_time: float,
            *,
            substep_idx: int,
            trigger_results: dict,
        ) -> None:
            events.append("observe")

    state_0 = _TrackedState()
    state_1 = _TrackedState()
    monkeypatch.setattr(NewtonManager, "_debug_incident_triggers", {}, raising=False)
    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        NewtonCfg(debug_capture=NewtonDebugCaptureCfg(capture_per_substep=True), use_cuda_graph=False),
        raising=False,
    )
    monkeypatch.setattr(NewtonManager, "_state_0", state_0, raising=False)
    monkeypatch.setattr(NewtonManager, "_state_1", state_1, raising=False)
    monkeypatch.setattr(NewtonManager, "_control", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", _solver_instance(), raising=False)
    monkeypatch.setattr(NewtonManager, "_contacts", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_collision_pipeline", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_use_single_state", use_single_state, raising=False)
    monkeypatch.setattr(NewtonManager, "_solver_dt", 0.1, raising=False)
    monkeypatch.setattr(NewtonManager, "_num_substeps", 1, raising=False)
    monkeypatch.setattr(NewtonManager, "_collision_decimation", 0, raising=False)
    monkeypatch.setattr(NewtonManager, "_incident_recorder", _Recorder(), raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_step_solver",
        classmethod(lambda cls, *args: events.append("solve")),
    )

    assert NewtonManager._run_solver_substeps(None, dispatch_time=0.0) is False
    expected = ["solve"]
    if not use_single_state:
        expected.append("assign")
    assert events == [*expected, "replay_post", "observe", "clear_forces"]


@pytest.mark.parametrize(
    ("capture_cfg", "requested_flag"),
    [
        (
            NewtonDebugCaptureCfg(record_contacts=True),
            "debug_capture.record_contacts",
        ),
        (
            NewtonDebugCaptureCfg(
                replay=NewtonDebugReplayCfg(
                    enabled=True,
                    record_state=False,
                    record_control=False,
                    record_solver=False,
                    record_contacts=True,
                )
            ),
            "debug_capture.replay.record_contacts",
        ),
    ],
)
def test_internal_mjwarp_rejects_stale_contact_mirror_without_sensors(
    monkeypatch,
    capture_cfg: NewtonDebugCaptureCfg,
    requested_flag: str,
):
    """Internal MJWarp never presents its sensor-only contact mirror as live evidence."""
    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        NewtonCfg(use_cuda_graph=False, debug_capture=capture_cfg),
        raising=False,
    )
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_newton_contact_sensors", {}, raising=False)
    monkeypatch.setattr(NewtonManager, "_report_contacts", False, raising=False)

    with pytest.raises(ValueError, match=requested_flag):
        NewtonMJWarpManager._prepare_debug_capture_providers()


def test_external_mjwarp_pipeline_allows_live_contact_capture(monkeypatch):
    """Newton-pipeline contacts remain valid explicit evidence for MJWarp."""
    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        NewtonCfg(debug_capture=NewtonDebugCaptureCfg(record_contacts=True)),
        raising=False,
    )
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", True, raising=False)

    NewtonMJWarpManager._prepare_debug_capture_providers()


def test_debug_trigger_context_receives_latest_operation_snapshot(monkeypatch):
    """Custom triggers can inspect built-in transient collision and solver operations."""
    payload = SimpleNamespace(candidate_pair_count=np.asarray([3], dtype=np.int32))
    contexts: list[NewtonManager.DebugTriggerContext] = []

    class _Provider:
        def __init__(self) -> None:
            self.snapshot_count = 0

        def bind(self, solver) -> None:
            pass

        def snapshot(self):
            self.snapshot_count += 1
            return payload

        def close(self) -> None:
            pass

    provider = _Provider()
    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        NewtonCfg(use_cuda_graph=False, debug_capture=NewtonDebugCaptureCfg(record_operations=True)),
        raising=False,
    )
    monkeypatch.setattr(NewtonManager, "_incident_recorder", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_debug_operation_provider", provider, raising=False)
    monkeypatch.setattr(NewtonManager, "_debug_incident_triggers", {}, raising=False)
    monkeypatch.setattr(NewtonManager, "_model", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_control", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", _solver_instance(), raising=False)
    monkeypatch.setattr(NewtonManager, "_contacts", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_collision_pipeline", None, raising=False)

    def inspect_operations(context: NewtonManager.DebugTriggerContext):
        contexts.append(context)
        return

    monkeypatch.setattr(
        NewtonManager,
        "_debug_incident_triggers",
        {"inspect_operations": inspect_operations},
        raising=False,
    )

    results = NewtonManager._evaluate_debug_incident_triggers(
        phase="dispatch_post",
        substep_idx=None,
        sim_time=1.0,
    )

    assert results == {}
    assert provider.snapshot_count == 1
    assert contexts[0].operations is payload


def test_debug_trigger_operation_snapshot_errors_are_actionable(monkeypatch):
    """A provider failure cannot silently remove transient trigger evidence."""

    class _Provider:
        def snapshot(self):
            raise ValueError("collision hook lost")

    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        NewtonCfg(use_cuda_graph=False, debug_capture=NewtonDebugCaptureCfg(record_operations=True)),
        raising=False,
    )
    monkeypatch.setattr(NewtonManager, "_incident_recorder", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_debug_operation_provider", _Provider(), raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_debug_incident_triggers",
        {"inspect_operations": lambda context: None},
        raising=False,
    )
    monkeypatch.setattr(NewtonManager, "_model", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_control", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", _solver_instance(), raising=False)
    monkeypatch.setattr(NewtonManager, "_contacts", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_collision_pipeline", None, raising=False)

    with pytest.raises(RuntimeError, match="snapshot failed.*collision hook lost"):
        NewtonManager._evaluate_debug_incident_triggers(
            phase="dispatch_post",
            substep_idx=None,
            sim_time=1.0,
        )
