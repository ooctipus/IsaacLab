# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import MJWarpDebugOperationProvider
from isaaclab_newton.physics import mjwarp_debug as mjwarp_debug_module
from isaaclab_newton.physics._debug_capture import DebugCaptureError, DebugCapturePlan, DebugSchemaError
from newton.solvers import SolverMuJoCo


def _uninitialized_solver(*, use_mujoco_cpu: bool = False) -> SolverMuJoCo:
    """Construct a selection-only SolverMuJoCo without importing a model."""
    solver = object.__new__(SolverMuJoCo)
    solver.use_mujoco_cpu = use_mujoco_cpu
    return solver


@pytest.fixture(scope="module")
def mjwarp_solver():
    """Build one minimal MJWarp-backed solver on the Warp CPU device."""
    import newton

    builder = newton.ModelBuilder()
    body = builder.add_body(mass=1.0)
    builder.add_joint_revolute(parent=-1, child=body, axis=(0.0, 0.0, 1.0))
    builder.add_shape_sphere(body=body, radius=0.1)
    builder.add_ground_plane()
    model = builder.finalize(device="cpu")
    solver = SolverMuJoCo(
        model,
        use_mujoco_cpu=False,
        iterations=2,
        njmax=16,
        nconmax=16,
    )
    return solver, model


def _step(solver: SolverMuJoCo, model) -> None:
    """Run one direct MJWarp-backed Newton solver step."""
    solver.step(model.state(), model.state(), model.control(), None, 0.01)


def _sha256(path: Path) -> str:
    """Return the exact bytes hash for an installed source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_constructor_rejects_invalid_scan_patterns():
    """Iteration scan patterns are immutable, unique, and internally consistent."""
    with pytest.raises(TypeError, match="tuple"):
        MJWarpDebugOperationProvider(first_nonfinite_include_fields=["*"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        MJWarpDebugOperationProvider(first_nonfinite_include_fields=())
    with pytest.raises(ValueError, match="duplicate"):
        MJWarpDebugOperationProvider(first_nonfinite_include_fields=("*", "*"))
    with pytest.raises(ValueError, match="requires first_nonfinite_include_fields"):
        MJWarpDebugOperationProvider(first_nonfinite_exclude_fields=("*.unused",))


def test_bind_requires_exactly_one_mjwarp_backed_solver():
    """Selection rejects absent, ambiguous, and pure-CPU MuJoCo solvers before hooking."""
    provider = MJWarpDebugOperationProvider()
    with pytest.raises(DebugCaptureError, match="found none"):
        provider.bind({"soft": object()})  # type: ignore[arg-type]

    first = _uninitialized_solver()
    second = _uninitialized_solver()
    with pytest.raises(DebugCaptureError, match="found 2"):
        provider.bind({"rigid": first, "secondary": second})

    cpu_solver = _uninitialized_solver(use_mujoco_cpu=True)
    cpu_solver.mjw_model = object()
    cpu_solver.mjw_data = object()
    cpu_solver.model = SimpleNamespace(device="cpu")
    with pytest.raises(DebugCaptureError, match="use_mujoco_cpu=True"):
        provider.bind(cpu_solver)

    missing_selector = object.__new__(SolverMuJoCo)
    with pytest.raises(DebugCaptureError, match="missing use_mujoco_cpu"):
        provider.bind({"uninitialized": missing_selector})


def test_coupled_mapping_selects_mjwarp_solver_and_ignores_native_cpu(mjwarp_solver):
    """A native MuJoCo CPU entry does not make one MJWarp-backed entry ambiguous."""
    solver, _ = mjwarp_solver
    native_cpu = _uninitialized_solver(use_mujoco_cpu=True)
    provider = MJWarpDebugOperationProvider()

    provider.bind({"mjwarp": solver, "native_cpu": native_cpu})
    try:
        assert provider._target_solver is solver
    finally:
        provider.close()


def test_provider_schema_rejects_unsupported_leaves():
    """Provider roots retain stable None fields but reject unsupported leaves."""
    root = SimpleNamespace(
        captured=np.ones(1, dtype=np.float32),
        missing=None,
        callback=lambda: None,
    )
    plan = DebugCapturePlan.build(root, root_name="provider", include_private=True)

    with pytest.raises(DebugCaptureError, match="provider\\.callback") as error:
        mjwarp_debug_module._require_stable_plan(plan, "Test provider")
    assert "provider.missing" not in str(error.value)


def test_symbolic_world_extent_must_match_runtime_data():
    """An explicit leading nworld annotation cannot silently misattribute worlds."""
    root = SimpleNamespace(values=np.ones((2, 3), dtype=np.float32))
    plan = DebugCapturePlan.build(root, root_name="context", include_private=True)
    symbolic_field = dataclasses.replace(plan.fields[0], symbolic_shape=("nworld", None))

    with pytest.raises(DebugSchemaError, match="declares leading nworld.*expected 3"):
        mjwarp_debug_module._validate_scan_world_extents((symbolic_field,), root, nworld=3)


def test_bind_rejects_unmatched_scan_glob_without_mutation(mjwarp_solver):
    """A renamed scan path fails before model options or module functions change."""
    import mujoco_warp._src.solver as solver_module

    solver, _ = mjwarp_solver
    original_factory = solver_module._create_solver_context
    original_graph_conditional = solver.mjw_model.opt.graph_conditional
    provider = MJWarpDebugOperationProvider(
        first_nonfinite_include_fields=("mjwarp_solver_context.field_removed_upstream",)
    )

    with pytest.raises(DebugSchemaError, match="matched no fields"):
        provider.bind(solver)

    assert solver_module._create_solver_context is original_factory
    assert solver.mjw_model.opt.graph_conditional == original_graph_conditional


def test_pinned_contexts_are_auto_discovered_and_source_files_are_unchanged(mjwarp_solver):
    """A real step retains both contexts while restoring functions and source bytes exactly."""
    import mujoco_warp._src.collision_driver as collision_module
    import mujoco_warp._src.solver as solver_module

    solver, model = mjwarp_solver
    source_paths = (Path(inspect.getfile(solver_module)), Path(inspect.getfile(collision_module)))
    hashes_before = tuple(_sha256(path) for path in source_paths)
    original_functions = (
        solver_module.solve,
        solver_module._create_solver_context,
        collision_module.collision,
        collision_module.create_collision_context,
    )
    original_graph_conditional = solver.mjw_model.opt.graph_conditional

    provider = MJWarpDebugOperationProvider()
    provider.bind(solver)
    initial = provider.snapshot()
    assert initial is not provider.snapshot()
    assert not initial.solver_context_valid
    assert not initial.solver_completed
    assert initial.solver_call_count == 0
    assert not initial.collision_context_valid
    assert not initial.collision_completed
    assert initial.collision_call_count == 0
    for plan, context in (
        (provider._solver_context_plan, initial.solver_context),
        (provider._collision_context_plan, initial.collision_context),
    ):
        assert plan is not None
        discovered = {field.path[0] for field in plan.fields} | {entry.path[0] for entry in plan.unallocated}
        assert discovered == {field.name for field in dataclasses.fields(context)}
        assert all(np.count_nonzero(value) == 0 for value in plan.to_numpy(context).values())

    collision_paths = {field.display_path for field in provider._collision_context_plan.fields}
    assert {
        "mjwarp_collision_context.collision_pair",
        "mjwarp_collision_context.collision_pairid",
        "mjwarp_collision_context.collision_worldid",
    }.issubset(collision_paths)

    _step(solver, model)
    captured = provider.snapshot()
    assert captured.solver_context_valid and captured.solver_completed
    assert captured.solver_call_count == 1
    assert captured.collision_context_valid and captured.collision_completed
    assert captured.collision_call_count >= 1

    provider.close()
    provider.close()
    assert (
        provider._target_solver,
        provider._snapshot,
        provider._data_plan,
        provider._solver_context_plan,
        provider._collision_context_plan,
        provider._modules,
    ) == (None, None, None, None, None, None)
    assert not provider._original_functions
    assert not provider._installed_functions
    assert not provider._scan_fields
    assert original_functions == (
        solver_module.solve,
        solver_module._create_solver_context,
        collision_module.collision,
        collision_module.create_collision_context,
    )
    assert solver.mjw_model.opt.graph_conditional == original_graph_conditional
    assert tuple(_sha256(path) for path in source_paths) == hashes_before


def test_iteration_scan_retains_first_bad_context_with_clean_paths(mjwarp_solver, monkeypatch):
    """The first post-iteration context and Data stay frozen with clean paths."""
    import mujoco_warp._src.solver as solver_module

    solver, model = mjwarp_solver
    original_iteration = solver_module._solver_iteration
    iteration_count = [0]

    @functools.wraps(original_iteration)
    def inject_after_iteration(m, d, ctx, nsolving, compact=False):
        result = original_iteration(m, d, ctx, nsolving, compact=compact)
        values = ctx.h.numpy()
        values[0, 0, 0] = np.nan
        ctx.h.assign(values)
        data_values = d.efc.D.numpy()
        data_values[0, 0] = 123.0 if iteration_count[0] == 0 else 456.0
        d.efc.D.assign(data_values)
        iteration_count[0] += 1
        return result

    monkeypatch.setattr(solver_module, "_solver_iteration", inject_after_iteration)
    original_graph_conditional = solver.mjw_model.opt.graph_conditional
    provider = MJWarpDebugOperationProvider(
        first_nonfinite_include_fields=("mjwarp_solver_context.h",),
    )
    provider.bind(solver)
    assert solver.mjw_model.opt.graph_conditional is False

    _step(solver, model)
    snapshot = provider.snapshot()
    assert snapshot.first_nonfinite_valid
    assert snapshot.first_nonfinite_iteration == 0
    assert snapshot.first_nonfinite_path_flags == {"mjwarp_solver_context.h": True}
    assert snapshot.first_nonfinite_world_flags.tolist() == [False]
    assert snapshot.first_nonfinite_global
    assert np.isnan(snapshot.first_nonfinite_context.h.numpy()[0, 0, 0])
    assert snapshot.pre_solve_data_valid
    assert snapshot.pre_solve_data.efc.D.numpy()[0, 0] != pytest.approx(123.0)
    assert snapshot.first_nonfinite_data.efc.D.numpy()[0, 0] == pytest.approx(123.0)
    assert solver.mjw_data.efc.D.numpy()[0, 0] == pytest.approx(456.0)
    assert iteration_count[0] == 2
    operation_plan = DebugCapturePlan.build(snapshot, root_name="operations", include_private=True)
    operation_paths = {field.display_path for field in operation_plan.fields}
    assert "operations.solver_call_count" in operation_paths
    assert "operations.collision_call_count" in operation_paths
    assert "operations.first_nonfinite_context.h" in operation_paths
    assert "operations.pre_solve_data.efc.D" in operation_paths
    assert "operations.first_nonfinite_data.efc.D" in operation_paths
    assert not any("first_nonfinite_context[" in path for path in operation_paths)

    provider.close()
    assert solver.mjw_model.opt.graph_conditional == original_graph_conditional
    assert solver_module._solver_iteration is inject_after_iteration


def test_signature_drift_fails_before_interposition(mjwarp_solver, monkeypatch):
    """An upstream private-boundary signature change produces an actionable bind error."""
    import mujoco_warp._src.solver as solver_module

    solver, _ = mjwarp_solver

    def changed_solve_signature(model, data, new_required_argument):
        del model, data, new_required_argument

    monkeypatch.setattr(solver_module, "solve", changed_solve_signature)
    provider = MJWarpDebugOperationProvider()
    with pytest.raises(DebugCaptureError, match="signature drifted"):
        provider.bind(solver)
    assert solver_module.solve is changed_solve_signature


def test_iteration_scan_rejects_exact_nonfloating_fields_and_filters_wildcards(mjwarp_solver):
    """Exact bool scans fail while wildcard metadata contains only finite-checkable fields."""
    solver, _ = mjwarp_solver
    exact = MJWarpDebugOperationProvider(
        first_nonfinite_include_fields=("mjwarp_solver_context.done",),
    )
    with pytest.raises(DebugSchemaError, match="not floating/complex"):
        exact.bind(solver)

    wildcard = MJWarpDebugOperationProvider(
        first_nonfinite_include_fields=("mjwarp_solver_context.*",),
    )
    wildcard.bind(solver)
    try:
        snapshot = wildcard.snapshot()
        assert "mjwarp_solver_context.done" not in snapshot.first_nonfinite_path_flags
        assert wildcard._scan_fields
        assert all(
            field.kind in {"warp_array", "torch_tensor", "numpy_array", "scalar"} for field in wildcard._scan_fields
        )
    finally:
        wildcard.close()


def test_external_solver_factory_does_not_mutate_owned_snapshot(mjwarp_solver):
    """Only a SolverContext created inside the locked target solve is accepted."""
    import mujoco_warp._src.solver as solver_module

    solver, _ = mjwarp_solver
    provider = MJWarpDebugOperationProvider()
    provider.bind(solver)
    try:
        initial_context = provider._snapshot.solver_context
        with wp.ScopedDevice(solver.model.device):
            external = solver_module._create_solver_context(solver.mjw_model, solver.mjw_data)
        assert external is not initial_context
        assert provider._snapshot.solver_context is initial_context
        assert not provider._snapshot.solver_context_valid
    finally:
        provider.close()


def test_solver_depth_is_restored_when_boundary_leave_raises(mjwarp_solver, monkeypatch):
    """Solver thread-local ownership never leaks through a failing boundary teardown."""
    import mujoco_warp._src.solver as solver_module

    solver, _ = mjwarp_solver
    original_solve = solver_module.solve

    @functools.wraps(original_solve)
    def fast_solve(m, d):
        del m, d

    monkeypatch.setattr(solver_module, "solve", fast_solve)
    provider = MJWarpDebugOperationProvider()
    provider.bind(solver)

    def fail_leave(name):
        assert name == "solve"
        provider._active_boundary = None
        provider._runtime_lock.release()
        raise DebugCaptureError("injected leave failure")

    provider._leave_runtime_boundary = fail_leave
    try:
        with pytest.raises(DebugCaptureError, match="injected leave failure"):
            solver_module.solve(solver.mjw_model, solver.mjw_data)
        assert provider._solver_local.depth == 0
    finally:
        del provider._leave_runtime_boundary
        provider.close()
        monkeypatch.undo()
    assert solver_module.solve is original_solve


def test_solver_exception_keeps_context_valid_but_completion_false(mjwarp_solver, monkeypatch):
    """A post-factory solve exception publishes an accepted but incomplete context."""
    import mujoco_warp._src.solver as solver_module

    solver, _ = mjwarp_solver
    original_solve = solver_module.solve

    @functools.wraps(original_solve)
    def raise_after_factory(m, d):
        solver_module._create_solver_context(m, d)
        raise RuntimeError("injected solve failure")

    monkeypatch.setattr(solver_module, "solve", raise_after_factory)
    provider = MJWarpDebugOperationProvider()
    provider.bind(solver)
    try:
        with wp.ScopedDevice(solver.model.device), pytest.raises(RuntimeError, match="injected"):
            solver_module.solve(solver.mjw_model, solver.mjw_data)
        snapshot = provider.snapshot()
        assert snapshot.solver_context_valid
        assert not snapshot.solver_completed
    finally:
        provider.close()
    assert solver_module.solve is raise_after_factory


def test_overlap_blocks_collision_snapshot_and_close(mjwarp_solver):
    """Concurrent target work cannot race collision mutation, publication, or teardown."""
    import mujoco_warp._src.collision_driver as collision_module

    solver, _ = mjwarp_solver
    provider = MJWarpDebugOperationProvider()
    provider.bind(solver)
    provider._enter_runtime_boundary("test_solve")
    try:
        with pytest.raises(DebugCaptureError, match="overlaps active 'test_solve'"):
            collision_module.collision(solver.mjw_model, solver.mjw_data)
        with pytest.raises(DebugCaptureError, match="during active target boundary"):
            provider.snapshot()
        with pytest.raises(DebugCaptureError, match="still in flight"):
            provider.close()
    finally:
        provider._leave_runtime_boundary("test_solve")
    provider.close()


def test_context_schema_drift_fails_at_each_factory_boundary(mjwarp_solver):
    """A later SolverContext shape change is rejected instead of partially recorded."""
    import mujoco_warp._src.solver as solver_module

    solver, _ = mjwarp_solver
    provider = MJWarpDebugOperationProvider()
    provider.bind(solver)
    try:
        with wp.ScopedDevice(solver.model.device):
            changed = solver_module._create_solver_context(solver.mjw_model, solver.mjw_data)
            changed.h = wp.zeros((changed.h.shape[0], changed.h.shape[1] + 1, changed.h.shape[2]), dtype=float)
        with pytest.raises(DebugSchemaError, match="object graph no longer matches"):
            provider._accept_solver_context(changed)
    finally:
        provider.close()


def test_close_refuses_to_overwrite_another_function_owner(mjwarp_solver, monkeypatch):
    """Teardown reports lost ownership and never overwrites the conflicting function."""
    import mujoco_warp._src.solver as solver_module

    solver, _ = mjwarp_solver
    provider = MJWarpDebugOperationProvider()
    provider.bind(solver)
    installed = solver_module.solve

    def conflicting_solve(m, d):
        del m, d

    monkeypatch.setattr(solver_module, "solve", conflicting_solve)
    with pytest.raises(DebugCaptureError, match="ownership was lost"):
        provider.close()
    assert solver_module.solve is conflicting_solve

    monkeypatch.setattr(solver_module, "solve", installed)
    monkeypatch.undo()
    assert solver_module.solve is installed
    provider.close()


def test_close_can_retry_after_graph_option_restoration_fails(mjwarp_solver):
    """A graph-option setter failure leaves all hooks owned for a clean retry."""
    import mujoco_warp._src.solver as solver_module

    class _FailOnceOption:
        """Delegate every option except the first graph-conditional write."""

        def __init__(self, target):
            object.__setattr__(self, "_target", target)
            object.__setattr__(self, "_fail", True)

        def __getattr__(self, name):
            return getattr(self._target, name)

        def __setattr__(self, name, value):
            if name == "graph_conditional" and self._fail:
                object.__setattr__(self, "_fail", False)
                raise RuntimeError("injected graph option failure")
            setattr(self._target, name, value)

    solver, _ = mjwarp_solver
    real_option = solver.mjw_model.opt
    original_graph_conditional = real_option.graph_conditional
    original_solve = solver_module.solve
    provider = MJWarpDebugOperationProvider(
        first_nonfinite_include_fields=("mjwarp_solver_context.h",),
    )
    provider.bind(solver)
    installed_solve = solver_module.solve
    solver.mjw_model.opt = _FailOnceOption(real_option)
    try:
        with pytest.raises(DebugCaptureError, match="close may be retried"):
            provider.close()
        assert solver_module.solve is installed_solve
        assert provider._bound

        provider.close()
        assert solver_module.solve is original_solve
        assert real_option.graph_conditional == original_graph_conditional
    finally:
        solver.mjw_model.opt = real_option
        if provider._bound:
            provider.close()
