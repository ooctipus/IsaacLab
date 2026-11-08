# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The manager records a physics transition before task resets can erase it."""

from types import SimpleNamespace

import isaaclab_newton.physics.newton_manager as manager_module
import pytest
import warp as wp
from isaaclab_newton.physics import NewtonCfg, NewtonDebugCaptureCfg, NewtonManager
from isaaclab_newton.physics.newton_backend import NewtonBackend


def test_incident_capture_brackets_backend_step(monkeypatch):
    events = []
    state = object()
    recorder = SimpleNamespace(
        capture_pre=lambda value: events.append(("pre", value)),
        step=lambda value, time, **kwargs: events.append(("post", value, time, kwargs)),
        capture_per_substep=False,
        halted=False,
    )
    manager = NewtonManager(NewtonCfg(debug_capture=NewtonDebugCaptureCfg()))
    manager._sim = SimpleNamespace(is_playing=lambda: True)
    manager.backend = SimpleNamespace(
        device=wp.get_device("cpu"),
        decimation=1,
        handles_decimation=lambda: False,
        dt=0.01,
        state_0=state,
        step=lambda: events.append(("physics",)),
    )
    manager._incident_recorder = recorder
    monkeypatch.setattr(manager, "_mark_transforms_changed", lambda: events.append(("publish",)))

    manager.step()

    assert events == [("pre", state), ("physics",), ("post", state, 0.01, {"trigger_results": {}}), ("publish",)]


def test_incident_capture_keeps_post_state_when_solver_check_raises(monkeypatch):
    events = []
    state = object()
    manager = NewtonManager(NewtonCfg(debug_capture=NewtonDebugCaptureCfg()))
    manager._sim = SimpleNamespace(is_playing=lambda: True)
    manager.backend = SimpleNamespace(
        device=wp.get_device("cpu"),
        decimation=1,
        handles_decimation=lambda: False,
        dt=0.01,
        state_0=state,
        step=lambda: (_ for _ in ()).throw(RuntimeError("solver failed")),
    )
    manager._incident_recorder = SimpleNamespace(
        capture_pre=lambda value: events.append(("pre", value)),
        step=lambda value, time, **kwargs: events.append(("post", value, time, kwargs)),
        capture_per_substep=False,
    )
    monkeypatch.setattr(manager, "_mark_transforms_changed", lambda: events.append(("publish",)))

    with pytest.raises(RuntimeError, match="solver failed"):
        manager.step()

    assert events == [("pre", state), ("post", state, 0.01, {"trigger_results": {}})]


def test_incident_capture_rearms_reset_worlds_before_forward():
    events = []
    world_mask = wp.array([True, False, True, False], dtype=wp.bool, device="cpu")
    manager = NewtonManager(NewtonCfg(debug_capture=NewtonDebugCaptureCfg()))
    manager.backend = SimpleNamespace(
        kinematics_dirty=True,
        world_mask=world_mask,
        model=SimpleNamespace(world_count=3),
        forward=lambda: events.append(("forward",)),
    )
    manager._incident_recorder = SimpleNamespace(
        rearm_reset_worlds=lambda ids, all_worlds: events.append(("rearm", ids, all_worlds))
    )

    manager.forward()

    assert events == [("rearm", (0, 2), False), ("forward",)]


def test_backend_debug_hook_brackets_every_solver_substep():
    events = []
    state = SimpleNamespace(clear_forces=lambda: events.append("clear"))
    ops = SimpleNamespace(
        single_state=True,
        step=lambda *args, **kwargs: events.append("solver"),
        post_step=lambda state: events.append("post_step"),
    )
    backend = SimpleNamespace(
        solver_ops=ops,
        num_substeps=2,
        control=object(),
        solver_dt=0.005,
        cfg=SimpleNamespace(collision_decimation=0),
        collision_pipeline=None,
        staged_body_f=None,
        state_force_callbacks=[],
        state_0=state,
        state_1=None,
        state_version=0,
        _debug_substep_index=0,
        debug_substep=lambda state, post, index: events.append(("after" if post else "before", index)),
    )

    NewtonBackend._run_solver_substeps(backend, None, observables=None)

    assert events == [
        ("before", 0),
        "solver",
        ("after", 0),
        "clear",
        ("before", 1),
        "solver",
        ("after", 1),
        "clear",
        "post_step",
    ]


def test_per_substep_capture_records_replay_and_halts_before_next_solver_step(monkeypatch):
    events = []
    state = object()
    manager = NewtonManager(NewtonCfg(debug_capture=NewtonDebugCaptureCfg()))
    manager._sim = SimpleNamespace(is_playing=lambda: True)
    manager.backend = SimpleNamespace(
        device=wp.get_device("cpu"), solver_dt=0.005, dt=0.01, decimation=1,
        handles_decimation=lambda: False, state_0=state,
    )
    recorder = SimpleNamespace(
        capture_per_substep=True,
        halted=False,
        capture_pre=lambda state: events.append("dispatch_pre"),
        record_step_replay_pre=lambda state, **kwargs: events.append(("replay_pre", kwargs)),
        record_step_replay_post=lambda state: events.append("replay_post"),
        observe_substep=lambda state, time, **kwargs: events.append(("observe", time, kwargs)),
    )
    manager._incident_recorder = recorder

    def step():
        manager._capture_debug_substep(state, False, 0)
        events.append("solver")
        recorder.halted = True
        manager._capture_debug_substep(state, True, 0)
        events.append("next_solver")

    manager.backend.step = step
    monkeypatch.setattr(manager, "_mark_transforms_changed", lambda: events.append("publish"))

    with pytest.raises(RuntimeError, match="Physics incident artifacts"):
        manager.step()

    assert events == [
        "dispatch_pre",
        ("replay_pre", {"sim_time": 0.0, "substep_idx": 0}),
        "solver",
        "replay_post",
        ("observe", 0.005, {"substep_idx": 0, "trigger_results": {}}),
    ]


def test_trigger_registry_is_per_manager_and_evaluates_current_backend():
    capture = NewtonDebugCaptureCfg()
    manager = NewtonManager(NewtonCfg(debug_capture=capture))
    other = NewtonManager(NewtonCfg(debug_capture=capture))
    state = object()
    manager.backend = SimpleNamespace(
        model=object(), state_0=state, control=object(), solver=object(), contacts=None, collision_pipeline=None
    )
    seen = []

    def trigger(context):
        seen.append((context.state, context.phase, context.substep_idx))
        return NewtonManager.DebugTriggerResult("velocity spike", (1,))

    manager.set_debug_incident_trigger("velocity_spike", trigger)
    assert other._debug_incident_triggers == {}
    assert manager._evaluate_debug_incident_triggers("dispatch_post", None, 0.01) == {
        "velocity_spike": NewtonManager.DebugTriggerResult("velocity spike", (1,))
    }
    assert seen == [(state, "dispatch_post", None)]


def test_scene_export_uses_current_clone_plan(tmp_path, monkeypatch):
    from pxr import Usd

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Source", "Xform")
    topology = SimpleNamespace(world_prototype_layout=[0], world_prototype_starts=[0, 0, 1], world_prototypes=[0])
    plan = SimpleNamespace(topology=topology)
    manager = NewtonManager(NewtonCfg())
    manager._sim = SimpleNamespace(get_clone_plan=lambda: plan)
    monkeypatch.setattr(manager_module, "get_current_stage", lambda: stage)
    monkeypatch.setattr(
        manager_module.cloner.path, "get_asset_prototype_paths", lambda plan: ("/World/Source", "/World/Unused")
    )
    scene_path = tmp_path / "incident.usda"

    manager._export_incident_scene(str(scene_path), [0])

    assert Usd.Stage.Open(str(scene_path)).GetPrimAtPath("/World/Source").IsValid()
    with pytest.raises(ValueError, match="Incident world IDs"):
        manager._export_incident_scene(str(scene_path), [1])


def test_debugger_has_no_old_singleton_or_solver_manager_owners():
    from pathlib import Path

    physics = Path(manager_module.__file__).parent
    assert not hasattr(NewtonManager, "_incident_recorder")
    assert not hasattr(NewtonManager, "_debug_operation_provider")
    assert not (physics / "mjwarp_manager.py").exists()
    assert not (physics / "kamino_manager.py").exists()
    assert not (physics / "debug_state_buffer.py").exists()
