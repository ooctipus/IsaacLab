# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact native population ownership and simulation lifecycle boundaries."""

from __future__ import annotations

import ast
import inspect
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import population as module
from isaaclab_newton.physics.population import (
    NewtonPopulation,
    NewtonPopulationBackend,
    NewtonPopulationBackendCfg,
    NewtonPopulationCfg,
    NewtonPopulationManager,
)
from newton.sensors import SensorContact
from newton.solvers import SolverMuJoCo

from isaaclab.physics import PhysicsEvent, PhysicsManager


@wp.kernel
def _write_task_coordinates(q: wp.array[float], selected: wp.array[bool], value: float):
    i = wp.tid()
    if selected[i]:
        q[i] = value


@wp.kernel
def _write_reset_inputs(phase: int, q: wp.array[float], mask: wp.array[bool], roots: wp.array[wp.transform]):
    world = wp.tid()
    selected = phase % 4 == 0 or (phase % 4 == 1 and world % 2 == 0)
    mask[world] = selected
    if world == 0:
        mask[mask.shape[0] - 1] = phase % 4 == 3
    if selected:
        q[world] = 0.014 + 0.001 * float(phase)
        roots[world] = wp.transform(wp.vec3(0.0, 0.0, 0.0001 * float(phase)), wp.quat_identity())


@wp.kernel
def _delay_producer(values: wp.array[float], iterations: int):
    i = wp.tid()
    value = float(i) * 0.001
    for _ in range(iterations):
        value = wp.sin(value) + 0.01
    values[i] = value


def _model(device="cpu"):
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    builder.begin_world()
    body = builder.add_link()
    builder.add_shape_box(body, hx=0.03, hy=0.03, hz=0.03)
    joint = builder.add_joint_prismatic(-1, body, axis=(0.0, 0.0, 1.0), target_ke=10.0, target_kd=1.0)
    builder.add_articulation([joint])
    builder.add_ground_plane()
    builder.end_world()
    return builder.finalize(device)


def _cpu_source():
    """Mock only the native solver, retaining real models, buffers, cloning, and FK."""
    source = object.__new__(SolverMuJoCo)
    source.model = _model()
    source.update_data_interval = 2

    def replicate(count):
        solver = object.__new__(SolverMuJoCo)
        solver.model = source.model.replicate(count)
        solver.update_data_interval = 2
        solver.step = Mock(side_effect=lambda state, output, *_args: output.assign(state))
        solver.reset = Mock()
        solver.notify_model_changed = Mock()
        return solver

    source.replicate = Mock(side_effect=replicate)
    return source


def _backend(sources, counts, **kwargs):
    return NewtonPopulationBackend(NewtonPopulationBackendCfg(prototypes=sources, counts=counts, dt=0.005, **kwargs))


def test_exact_sizes_identity_reuse_and_no_readback():
    sources = (_cpu_source(), _cpu_source())
    backend = _backend(sources, (2, 1))
    try:
        original = backend.populations
        unchanged_graph = object()
        original[0].graph = unchanged_graph
        with (
            patch.object(wp.array, "numpy", side_effect=AssertionError("readback")),
            patch.object(newton.ModelBuilder, "finalize", side_effect=AssertionError("reconstruction")),
        ):
            backend.replace((2, 3))
        assert backend.counts == (2, 3)
        assert backend.populations[0] is original[0]
        assert backend.populations[0].graph is unchanged_graph
        assert backend.populations[1] is not original[1]
        current = backend.populations
        with patch.object(wp, "synchronize_device", side_effect=AssertionError("unchanged-count synchronization")):
            backend.replace((2, 3))
        assert backend.populations is current
        backend.replace((0, 3))
        assert backend.populations[0] is None
        backend.replace((2, 3))
        assert backend.populations[0] is not original[0]
        assert sources[0].replicate.call_count == 2
        np.testing.assert_array_equal(sources[0].model.joint_q.numpy(), [0.0])
    finally:
        backend.close()


def test_failed_replacement_preserves_complete_previous_tuple():
    sources = (_cpu_source(), _cpu_source())
    backend = _backend(sources, (2, 1))
    original = backend.populations
    sources[1].replicate.side_effect = RuntimeError("allocation failed")
    try:
        with pytest.raises(RuntimeError, match="allocation failed"):
            backend.replace((3, 2))
        assert backend.populations is original
        assert backend.counts == (2, 1)
        backend.step()
        assert original[0].solver.step.call_count == 2
    finally:
        backend.close()


def test_zero_counts_do_not_construct_native_populations():
    source = _cpu_source()
    backend = _backend((source,), (0,))
    assert backend.populations == (None,)
    source.replicate.assert_not_called()
    backend.step()
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.replace((1,))


@pytest.mark.parametrize("direct", (False, True))
@pytest.mark.parametrize(
    "options,message",
    [
        ({"dt": 0.0}, "dt must be finite and positive"),
        ({"dt": -0.005}, "dt must be finite and positive"),
        ({"dt": float("nan")}, "dt must be finite and positive"),
        ({"dt": float("inf")}, "dt must be finite and positive"),
        ({"substeps": 0}, "substeps must be a positive even integer"),
        ({"substeps": 1}, "substeps must be a positive even integer"),
        ({"substeps": 3}, "substeps must be a positive even integer"),
        ({"substeps": True}, "substeps must be a positive even integer"),
        ({"use_cuda_graph": True}, "Population graph capture requires CUDA"),
    ],
)
def test_population_configuration_fails_before_cloning(direct, options, message):
    source = _cpu_source()
    arguments = {"dt": 0.005, "substeps": 2, "use_cuda_graph": False} | options
    with pytest.raises(ValueError, match=message):
        if direct:
            NewtonPopulation(source, 2, **arguments)
        else:
            # An empty plan still validates the backend configuration.
            NewtonPopulationBackend(NewtonPopulationBackendCfg(prototypes=(source,), counts=(0,), **arguments))
    source.replicate.assert_not_called()


def test_direct_population_rejects_incompatible_capture_cadence_before_cloning():
    source = object.__new__(SolverMuJoCo)
    source.model = SimpleNamespace(device=SimpleNamespace(is_cuda=True))
    source.update_data_interval = 3
    source.replicate = Mock(side_effect=AssertionError("clone reached before cadence validation"))
    with pytest.raises(ValueError, match="update_data_interval to divide two"):
        NewtonPopulation(source, 2, dt=0.005, substeps=2, use_cuda_graph=True)
    source.replicate.assert_not_called()


@pytest.mark.parametrize(
    "counts,rows",
    [
        ((2,), (((0, 0), (0, 1)),)),
        ((2,), (((0,), (1,)),)),
        ((0,), (((0,), (0,)),)),
        ((3,), (((2,), (0,)),)),
        ((3,), (((True,), (0,)),)),
    ],
)
def test_invalid_survivor_maps_fail_before_allocation(counts, rows):
    source = _cpu_source()
    backend = _backend((source,), (2,))
    old = backend.populations
    source.replicate.reset_mock()
    try:
        with pytest.raises(ValueError):
            backend.replace(counts, survivors=rows)
        source.replicate.assert_not_called()
        assert backend.populations is old
    finally:
        backend.close()


def test_partial_fk_and_reset_preserve_explicit_mask_contract():
    backend = _backend((_cpu_source(),), (2,))
    runtime = backend.populations[0]
    try:
        runtime.state_0.joint_q.assign(np.array([0.1, 0.2], dtype=np.float32))
        original = runtime.state_0.body_q.numpy().copy()
        selected = wp.array([True, False, False], dtype=wp.bool, device="cpu")
        runtime.forward(selected)
        actual = runtime.state_0.body_q.numpy()
        assert actual[0, 2] == pytest.approx(0.1)
        np.testing.assert_array_equal(actual[1], original[1])
        runtime.reset(selected, flags=0)
        runtime.solver.reset.assert_called_once_with(runtime.state_0, world_mask=selected, flags=0)
        np.testing.assert_array_equal(runtime.state_1.joint_q.numpy(), np.array([0.1, 0.2], dtype=np.float32))
        with pytest.raises(ValueError, match="world_mask"):
            runtime.forward(wp.ones(2, dtype=wp.bool, device="cpu"))
    finally:
        backend.close()


def test_reconcile_validates_every_mask_before_mutating_any_population():
    backend = _backend((_cpu_source(),) * 3, (2, 0, 3))
    valid = wp.array([True, False, False], dtype=wp.bool, device="cpu")
    try:
        for masks in ((valid,), (valid, valid, None), (valid, None, valid)):
            with pytest.raises(ValueError):
                backend.reconcile_state(masks, newton.ModelFlags.JOINT_PROPERTIES)
        for population in backend.populations:
            if population is not None:
                population.solver.notify_model_changed.assert_not_called()
                population.solver.reset.assert_not_called()
        backend.reconcile_state((valid, None, None), newton.ModelFlags.JOINT_PROPERTIES)
        for index, mask in ((0, valid), (2, None)):
            population = backend.populations[index]
            population.solver.notify_model_changed.assert_called_once_with(
                newton.ModelFlags.JOINT_PROPERTIES, world_mask=mask
            )
            population.solver.reset.assert_called_once_with(population.state_0, world_mask=mask, flags=0)
    finally:
        backend.close()


def test_reconcile_joins_submitted_workers_when_an_operation_raises():
    trace = []
    producer = Mock()
    producer.record_event.side_effect = lambda event: trace.append(("fork", event))
    producer.wait_event.side_effect = lambda event: trace.append(("join", event))
    streams, events = (Mock(), Mock(), Mock()), (object(), object(), object())
    backend = object.__new__(NewtonPopulationBackend)
    backend.device, backend._closed = "cpu", False
    backend._streams, backend._complete, backend._fork = streams, events, object()
    backend._stream_groups = ((0,), (1, 2), (3,))
    backend.populations = tuple(Mock() for _ in range(4))
    backend.populations[2].reset.side_effect = ValueError("injected reset failure")
    with patch.object(wp, "get_stream", return_value=producer), patch.object(wp, "ScopedStream"):
        with pytest.raises(ValueError, match="injected reset failure"):
            backend.reconcile_state((None,) * 4)
    assert trace == [("fork", backend._fork), ("join", events[0]), ("join", events[1])]
    streams[0].record_event.assert_called_once_with(events[0])
    streams[1].record_event.assert_called_once_with(events[1])
    streams[2].wait_event.assert_not_called()
    backend.populations[3].reset.assert_not_called()


def test_manager_borrows_context_resource_and_orders_lifecycle():
    backend = _backend((_cpu_source(),), (2,))
    sim = SimpleNamespace(
        cfg=SimpleNamespace(physics=NewtonPopulationCfg(), device="cpu", dt=0.01),
        physics_manager=NewtonPopulationManager,
        resolve_visualizer_types=lambda: [],
        get_setting=lambda _name: None,
    )
    manager = NewtonPopulationManager
    manager.initialize(sim)
    events = []
    manager.register_callback(lambda _: events.append("model"), PhysicsEvent.MODEL_INIT)
    manager.register_callback(lambda _: events.append("ready"), PhysicsEvent.PHYSICS_READY)
    try:
        with pytest.raises(RuntimeError, match="Install"):
            manager.reset()
        manager.install(backend)
        manager.reset()
        assert events == ["model", "ready"]
        state = backend.populations[0].state_0
        manager.step()
        assert backend.populations[0].state_0 is state
        assert manager.get_simulation_time() == pytest.approx(0.01)
        assert not manager.handles_decimation()
        manager.close()
        assert backend.counts == (2,)  # Only SimulationContext closes owned resources.
        assert PhysicsManager._sim is None
    finally:
        manager.close()
        backend.close()


def test_reject_rendering_and_forbidden_ownership_dependencies():
    sim = SimpleNamespace(resolve_visualizer_types=lambda: ["newton_gl"], get_setting=lambda _name: None)
    with pytest.raises(ValueError, match="headless"):
        NewtonPopulationManager.initialize(sim)
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)):
            assert (node.id if isinstance(node, ast.Name) else node.attr) != "NewtonManager"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            dependencies = [name.name for name in node.names]
            if isinstance(node, ast.ImportFrom):
                dependencies.append(node.module or "")
            for dependency in dependencies:
                assert not set(dependency.split(".")) & {
                    "isaaclab_tasks",
                    "isaaclab_rl",
                    "rsl_rl",
                    "newton_manager",
                    "NewtonManager",
                    "ModelBuilder",
                }


def test_real_simulation_context_owns_allocation_and_teardown():
    from isaaclab.sim import SimulationCfg, SimulationContext

    source = _cpu_source()
    sim = SimulationContext(SimulationCfg(device="cpu", physics=NewtonPopulationCfg(), dt=0.01, visualizer_cfgs=[]))
    try:
        cfg = NewtonPopulationBackendCfg(prototypes=(source,), counts=(2,), dt=0.005)
        backend = sim.get_or_create_backend(cfg)
        assert sim.get_or_create_backend(cfg) is backend
        sim.physics_manager.install(backend)
        sim.reset()
        sim.step(render=False)
        assert sim.is_playing()
        assert sim.get_physics_step_count() == 1
    finally:
        sim.clear_instance()
    with pytest.raises(RuntimeError, match="closed"):
        backend.step()


@pytest.mark.skipif(not wp.is_cuda_available(), reason="Native MuJoCo Warp requires CUDA")
def test_native_graph_streams_match_eager_and_reuse_unchanged_graphs():
    device = wp.get_cuda_device()
    with wp.ScopedDevice(device):
        source = SolverMuJoCo(
            _model(device), use_mujoco_contacts=True, enable_sleeping=True, update_data_interval=2, njmax=64, nconmax=32
        )
        q_source, native_source = source.model.joint_q.numpy().copy(), source.mjw_data.qpos.numpy().copy()
        empty = _backend((source, source), (0, 0), stream_count=2, use_cuda_graph=True)
        empty.step()
        empty.close()
        eager = _backend((source, source), (2, 3), stream_count=1)
        with patch.object(wp.array, "numpy", side_effect=AssertionError("clone readback")):
            graph = _backend((source, source), (2, 3), stream_count=2, use_cuda_graph=True)
        producer = wp.Stream(device)
        try:
            for step in range(16):
                with wp.ScopedStream(producer):
                    for left, right in zip(eager.populations, graph.populations, strict=True):
                        target = 0.02 * (1 if step % 2 else -1)
                        left.control.joint_target_q.fill_(target)
                        right.control.joint_target_q.fill_(target)
                    eager.step()
                    graph.step()
                    if step in (3, 7, 11, 15):
                        for left, right in zip(eager.populations, graph.populations, strict=True):
                            count = left.model.world_count
                            mask = wp.array(
                                [step == 15 or i % 2 == step % 3 for i in range(count)] + [False], dtype=wp.bool
                            )
                            for runtime in (left, right):
                                wp.launch(_write_task_coordinates, count, [runtime.state_0.joint_q, mask, 0.015])
                            reset_mask = None if step == 15 else mask
                            left.reset(reset_mask, flags=0)
                            with patch.object(right.solver, "reset", side_effect=AssertionError("eager reset")):
                                right.reset(reset_mask, flags=0)
                    # Native reads after the join must observe every population.
                    observed = [wp.clone(runtime.state_0.joint_q) for runtime in graph.populations]
                wp.synchronize_stream(producer)
                for left, right, copied in zip(eager.populations, graph.populations, observed, strict=True):
                    np.testing.assert_allclose(left.state_0.joint_q.numpy(), copied.numpy(), rtol=1e-5, atol=1e-6)
                    np.testing.assert_allclose(left.state_0.joint_qd.numpy(), right.state_0.joint_qd.numpy(), atol=1e-6)
                    np.testing.assert_array_equal(left.solver.mjw_data.time.numpy(), right.solver.mjw_data.time.numpy())
                    np.testing.assert_array_equal(
                        left.solver.mjw_data.tree_asleep.numpy(), right.solver.mjw_data.tree_asleep.numpy()
                    )
            original = graph.populations
            graph.replace((2, 4))
            assert graph.populations[0] is original[0]
            assert graph.populations[0].graph is original[0].graph
            assert graph.populations[1] is not original[1]
            np.testing.assert_array_equal(source.model.joint_q.numpy(), q_source)
            np.testing.assert_array_equal(source.mjw_data.qpos.numpy(), native_source)
        finally:
            eager.close()
            graph.close()


@pytest.mark.skipif(not wp.is_cuda_available(), reason="Native MuJoCo Warp requires CUDA")
@pytest.mark.parametrize(
    "use_cuda_graph,missing_dependency", [(False, None), (True, None), (True, "fork"), (True, "join")]
)
def test_reconcile_streams_preserve_contacts_survivors_and_producer_order(use_cuda_graph, missing_dependency):
    device = wp.get_cuda_device()
    with wp.ScopedDevice(device):
        model = _model(device)
        prepared_sensor = SensorContact(model, sensing_bodies=[0], request_contact_attributes=False)
        source = SolverMuJoCo(
            model, use_mujoco_contacts=True, enable_sleeping=True, update_data_interval=2, njmax=64, nconmax=32
        )
        initial_root = model.joint_X_p.numpy().copy()
        serial = _backend((source,) * 4, (5, 0, 7, 0), stream_count=1, use_cuda_graph=use_cuda_graph)
        parallel = _backend((source,) * 4, (5, 0, 7, 0), stream_count=4, use_cuda_graph=use_cuda_graph)
        producer, scratch = wp.Stream(device), wp.empty(4096, device=device)
        masks = [
            [
                None if p is None else wp.zeros(p.model.world_count + 1, dtype=wp.bool, device=device)
                for p in b.populations
            ]
            for b in (serial, parallel)
        ]
        contacts = [
            [
                None
                if p is None
                else (
                    prepared_sensor.replicate(p.model),
                    newton.Contacts(p.solver.mjw_data.naconmax, 0, device=device, requested_attributes={"force"}),
                )
                for p in b.populations
            ]
            for b in (serial, parallel)
        ]
        real_wait = wp.Stream.wait_event

        def wait_event(stream, event, *args, **kwargs):
            if missing_dependency == "join" and stream is producer and any(event is e for e in parallel._complete):
                return None
            if missing_dependency == "fork" and event is parallel._fork:
                return None
            return real_wait(stream, event, *args, **kwargs)

        def observe(backend):
            # Device consumers are queued before any host readback; a missing
            # join must expose stale outputs instead of being hidden by a fence.
            return [
                tuple(wp.clone(value) for value in (p.state_0.body_q, p.state_1.joint_q, p.solver.mjw_data.qpos))
                for p in backend.populations
                if p is not None
            ]

        try:
            with pytest.raises(AssertionError) if missing_dependency else nullcontext():
                for phase in range(1 if missing_dependency else 16):
                    with wp.ScopedStream(producer):
                        before = [wp.clone(p.state_0.body_q) for p in parallel.populations if p is not None]
                        # The stronger negative-control delay ensures producer
                        # writes remain pending while the host submits workers.
                        wp.launch(
                            _delay_producer, scratch.size, [scratch, 262144 if missing_dependency == "fork" else 256]
                        )
                        for backend, world_masks in zip((serial, parallel), masks, strict=True):
                            for p, mask in zip(backend.populations, world_masks, strict=True):
                                if p is not None:
                                    wp.launch(
                                        _write_reset_inputs,
                                        p.model.world_count,
                                        [phase, p.state_0.joint_q, mask, p.model.joint_X_p],
                                    )
                                    p.control.joint_target_q.fill_(0.015 + 0.001 * phase)
                        serial.reconcile_state(masks[0], newton.ModelFlags.JOINT_PROPERTIES)
                        with patch.object(wp.Stream, "wait_event", wait_event):
                            parallel.reconcile_state(masks[1], newton.ModelFlags.JOINT_PROPERTIES)
                        expected, observed = observe(serial), observe(parallel)
                    wp.synchronize_stream(producer)
                    for left, right in zip(expected, observed, strict=True):
                        for a, b in zip(left, right, strict=True):
                            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-5, atol=1e-6)
                    active = [(p, m) for p, m in zip(parallel.populations, masks[1], strict=True) if p is not None]
                    for old, (p, mask) in zip(before, active, strict=True):
                        untouched = ~mask.numpy()[:-1]
                        np.testing.assert_array_equal(old.numpy()[untouched], p.state_0.body_q.numpy()[untouched])
                    with wp.ScopedStream(producer):
                        serial.step()
                        parallel.step()
                        expected, observed = observe(serial), observe(parallel)
                        forces = []
                        for backend, resources in zip((serial, parallel), contacts, strict=True):
                            values = []
                            for p, resource in zip(backend.populations, resources, strict=True):
                                if p is not None:
                                    sensor, contact = resource
                                    p.solver.update_contacts(contact)
                                    sensor.update(None, contact)
                                    values.append((wp.clone(sensor.total_force), wp.clone(sensor.total_force_friction)))
                            forces.append(values)
                    wp.synchronize_stream(producer)
                    for left, right in zip(expected + forces[0], observed + forces[1], strict=True):
                        for a, b in zip(left, right, strict=True):
                            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-4, atol=1e-5)
                    for backend in (serial, parallel):
                        for p in backend.populations:
                            if p is not None:
                                assert np.isfinite(p.state_0.joint_q.numpy()).all()
                                assert not np.any(p.solver.mjw_data.overflow.numpy())
            np.testing.assert_array_equal(model.joint_X_p.numpy(), initial_root)
        finally:
            serial.close()
            parallel.close()
