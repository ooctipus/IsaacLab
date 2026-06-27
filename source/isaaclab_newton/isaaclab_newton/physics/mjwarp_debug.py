# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Transient MuJoCo Warp operation capture without modifying the installation."""

from __future__ import annotations

import dataclasses
import enum
import functools
import importlib
import inspect
import threading
from collections.abc import Mapping
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from newton.solvers import SolverBase, SolverMuJoCo

from ._debug_capture import (
    DebugCaptureError,
    DebugCaptureField,
    DebugCapturePlan,
    DebugSchemaError,
    clone_debug_value,
    debug_value_to_numpy,
)

if TYPE_CHECKING:
    from mujoco_warp import Data
    from mujoco_warp._src.collision_core import CollisionContext
    from mujoco_warp._src.solver import SolverContext


_INTERPOSITION_LOCK = threading.RLock()
_ACTIVE_PROVIDER: MJWarpDebugOperationProvider | None = None
_OWNER_ATTRIBUTE = "__isaaclab_mjwarp_debug_owner__"


@dataclasses.dataclass(frozen=True, slots=True)
class _MJWarpModules:
    """Pinned private modules whose live globals form the instrumented call path."""

    forward: ModuleType
    solver: ModuleType
    collision_driver: ModuleType


class MJWarpDebugOperationProvider:
    """Expose transient MJWarp solver and collision workspaces to incident capture.

    The provider interposes live Python function boundaries in the installed
    ``mujoco_warp`` modules. It never edits those modules on disk. Every
    :class:`mujoco_warp._src.solver.SolverContext`,
    :class:`mujoco_warp._src.collision_core.CollisionContext`, and live MJWarp
    ``Data`` schema is discovered by
    :class:`~isaaclab_newton.physics._debug_capture.DebugCapturePlan`; no
    workspace field names are encoded here.

    Supply :attr:`first_nonfinite_include_fields` to enable the expensive
    per-iteration diagnostic. This mode temporarily disables MJWarp graph
    conditionals so every solver iteration crosses a Python boundary, scans the
    selected full paths, and preserves the first complete non-finite solver
    context and ``Data`` clone. :meth:`close` restores both the function bindings
    and the original graph-conditional option.

    Args:
        first_nonfinite_include_fields: Full-path glob patterns to scan after
            each MJWarp solver iteration. Paths start with
            ``mjwarp_solver_context``. ``None`` disables iteration scanning.
        first_nonfinite_exclude_fields: Full-path glob patterns removed from
            the iteration scan. Every pattern must match an included field.

    Raises:
        TypeError: If a glob collection is not a tuple of strings.
        ValueError: If a glob collection is empty, duplicated, or inconsistent.
    """

    @dataclasses.dataclass(slots=True)
    class Snapshot:
        """Stable operation root returned when iteration scanning is disabled.

        Attributes:
            solver_context: Latest schema-valid solver context.
            solver_context_valid: Whether the current solve created the context.
            solver_completed: Whether the current solve returned successfully.
            solver_call_count: Number of target solve calls observed since binding.
            collision_context: Latest schema-valid broad-phase collision context.
            collision_context_valid: Whether the current collision pass created the context.
            collision_completed: Whether the current collision pass returned successfully.
            collision_call_count: Number of target collision calls observed since binding.
        """

        solver_context: SolverContext
        solver_context_valid: bool
        solver_completed: bool
        solver_call_count: int
        collision_context: CollisionContext
        collision_context_valid: bool
        collision_completed: bool
        collision_call_count: int

    @dataclasses.dataclass(slots=True)
    class IterationSnapshot:
        """Stable operation root including the first non-finite iteration.

        Attributes:
            solver_context: Latest schema-valid solver context.
            solver_context_valid: Whether the current solve created the context.
            solver_completed: Whether the current solve returned successfully.
            solver_call_count: Number of target solve calls observed since binding.
            collision_context: Latest schema-valid broad-phase collision context.
            collision_context_valid: Whether the current collision pass created the context.
            collision_completed: Whether the current collision pass returned successfully.
            collision_call_count: Number of target collision calls observed since binding.
            pre_solve_data: Complete live ``Data`` clone from immediately before solve entry.
            pre_solve_data_valid: Whether the pre-solve clone completed successfully.
            first_nonfinite_data: Complete ``Data`` clone from the first matching iteration.
            first_nonfinite_context: Complete solver context from the first matching iteration.
            first_nonfinite_iteration: Zero-based matching iteration, or ``-1`` when none matched.
            first_nonfinite_path_flags: Whether each selected full path was non-finite.
            first_nonfinite_world_flags: Per-world matches for fields with explicit leading ``nworld``.
            first_nonfinite_global: Whether a matching field could not be attributed to one world.
            first_nonfinite_valid: Whether the current solve produced a matching iteration.
            graph_conditional_forced_off: Whether iteration scanning forced graph conditionals off.
            graph_conditional_original: Graph-conditional value saved before interposition.
        """

        solver_context: SolverContext
        solver_context_valid: bool
        solver_completed: bool
        solver_call_count: int
        collision_context: CollisionContext
        collision_context_valid: bool
        collision_completed: bool
        collision_call_count: int
        pre_solve_data: Data
        pre_solve_data_valid: bool
        first_nonfinite_data: Data
        first_nonfinite_context: SolverContext
        first_nonfinite_iteration: int
        first_nonfinite_path_flags: dict[str, bool]
        first_nonfinite_world_flags: np.ndarray
        first_nonfinite_global: bool
        first_nonfinite_valid: bool
        graph_conditional_forced_off: bool
        graph_conditional_original: bool

    def __init__(
        self,
        *,
        first_nonfinite_include_fields: tuple[str, ...] | None = None,
        first_nonfinite_exclude_fields: tuple[str, ...] = (),
    ) -> None:
        self._first_nonfinite_include_fields = _validate_patterns_or_none(
            first_nonfinite_include_fields,
            "first_nonfinite_include_fields",
        )
        self._first_nonfinite_exclude_fields = _validate_patterns(
            first_nonfinite_exclude_fields,
            "first_nonfinite_exclude_fields",
            allow_empty=True,
        )
        if first_nonfinite_include_fields is None and first_nonfinite_exclude_fields:
            raise ValueError("first_nonfinite_exclude_fields requires first_nonfinite_include_fields.")

        self._bound = False
        self._closed = False
        self._data_plan: DebugCapturePlan | None = None
        self._target_solver: SolverMuJoCo | None = None
        self._modules: _MJWarpModules | None = None
        self._solver_context_plan: DebugCapturePlan | None = None
        self._collision_context_plan: DebugCapturePlan | None = None
        self._scan_fields: tuple[DebugCaptureField, ...] = ()
        self._snapshot: (
            MJWarpDebugOperationProvider.Snapshot | MJWarpDebugOperationProvider.IterationSnapshot | None
        ) = None
        self._original_functions: dict[tuple[ModuleType, str], object] = {}
        self._installed_functions: dict[tuple[ModuleType, str], object] = {}
        self._graph_conditional_original: bool | None = None
        self._graph_conditional_restored: bool = False
        self._runtime_lock = threading.Lock()
        self._active_boundary: str | None = None
        self._collision_local = threading.local()
        self._solver_local = threading.local()
        self._iteration_index = 0

    def bind(self, solver: SolverBase | Mapping[str, SolverBase]) -> None:
        """Bind to exactly one live MJWarp-backed :class:`SolverMuJoCo` instance.

        Args:
            solver: One solver or a coupled-solver mapping supplied by
                :class:`~isaaclab_newton.physics.NewtonManager`.

        Raises:
            DebugCaptureError: If selection, pinned call-path validation,
                context discovery, or interposition fails.
        """
        if self._bound:
            raise DebugCaptureError("MJWarp debug operation provider is already bound.")
        if self._closed:
            raise DebugCaptureError("MJWarp debug operation provider was closed and cannot be rebound.")

        target = _select_mjwarp_solver(solver)
        modules = _load_mjwarp_modules()
        originals = _validate_mjwarp_call_path(target, modules, scan_iterations=self._scan_enabled)

        try:
            model = object.__getattribute__(target, "mjw_model")
            data = object.__getattribute__(target, "mjw_data")
            device = object.__getattribute__(object.__getattribute__(target, "model"), "device")
            with wp.ScopedDevice(device):
                solver_context = originals[(modules.solver, "create_solver_context")](model, data)
                collision_context = originals[(modules.collision_driver, "create_collision_context")](data.naconmax)
        except Exception as exc:
            raise DebugCaptureError(f"Failed to allocate deterministic MJWarp debug contexts: {exc}") from exc

        solver_plan = DebugCapturePlan.build(
            solver_context,
            root_name="mjwarp_solver_context",
            include_private=True,
        )
        collision_plan = DebugCapturePlan.build(
            collision_context,
            root_name="mjwarp_collision_context",
            include_private=True,
        )
        data_plan = DebugCapturePlan.build(data, root_name="mjwarp_data", include_private=True)
        _require_complete_plan(solver_plan, "MJWarp SolverContext")
        _require_complete_plan(collision_plan, "MJWarp CollisionContext")
        _require_complete_plan(data_plan, "MJWarp Data")
        _zero_context(solver_plan, solver_context)
        _zero_context(collision_plan, collision_context)
        solver_plan.validate_schema(solver_context)
        collision_plan.validate_schema(collision_context)

        scan_fields: tuple[DebugCaptureField, ...] = ()
        if self._scan_enabled:
            assert self._first_nonfinite_include_fields is not None
            selected_binding = solver_plan.bind(
                self._first_nonfinite_include_fields,
                self._first_nonfinite_exclude_fields,
            )
            scan_fields = tuple(
                field
                for field in selected_binding.fields
                if debug_value_to_numpy(field.validate(solver_context), field.display_path).dtype.kind in "fc"
            )
            exact_non_scannable = tuple(
                pattern
                for pattern in self._first_nonfinite_include_fields
                if not any(marker in pattern for marker in "*?[")
                and not any(field.display_path == pattern for field in scan_fields)
            )
            if exact_non_scannable:
                raise DebugSchemaError(f"Exact MJWarp scan paths are not floating/complex: {exact_non_scannable}.")
            if not scan_fields:
                raise DebugSchemaError("MJWarp first-nonfinite scan patterns select no floating or complex fields.")
            _validate_scan_world_extents(scan_fields, solver_context, int(data.nworld))
            first_context = _clone_dataclass_context(solver_plan, solver_context)
            pre_solve_data = _clone_dataclass_tree(data, "mjwarp_data")
            first_nonfinite_data = _clone_dataclass_tree(data, "mjwarp_data")
            data_plan.validate_schema(pre_solve_data)
            data_plan.validate_schema(first_nonfinite_data)
            _zero_array_fields(data_plan, pre_solve_data)
            _zero_array_fields(data_plan, first_nonfinite_data)
            snapshot: MJWarpDebugOperationProvider.Snapshot | MJWarpDebugOperationProvider.IterationSnapshot = (
                self.IterationSnapshot(
                    solver_context=solver_context,
                    solver_context_valid=False,
                    solver_completed=False,
                    solver_call_count=0,
                    collision_context=collision_context,
                    collision_context_valid=False,
                    collision_completed=False,
                    collision_call_count=0,
                    first_nonfinite_context=first_context,
                    first_nonfinite_iteration=-1,
                    pre_solve_data=pre_solve_data,
                    pre_solve_data_valid=False,
                    first_nonfinite_data=first_nonfinite_data,
                    first_nonfinite_path_flags={field.display_path: False for field in scan_fields},
                    first_nonfinite_world_flags=np.zeros(int(data.nworld), dtype=np.bool_),
                    first_nonfinite_global=False,
                    first_nonfinite_valid=False,
                    graph_conditional_forced_off=True,
                    graph_conditional_original=bool(model.opt.graph_conditional),
                )
            )
        else:
            snapshot = self.Snapshot(
                solver_context=solver_context,
                solver_context_valid=False,
                solver_completed=False,
                solver_call_count=0,
                collision_context=collision_context,
                collision_context_valid=False,
                collision_completed=False,
                collision_call_count=0,
            )

        self._target_solver = target
        self._modules = modules
        self._solver_context_plan = solver_plan
        self._collision_context_plan = collision_plan
        self._data_plan = data_plan
        self._scan_fields = scan_fields
        self._snapshot = snapshot
        self._original_functions = originals

        try:
            self._install_interposition()
        except Exception:
            self._target_solver = None
            self._modules = None
            self._solver_context_plan = None
            self._collision_context_plan = None
            self._data_plan = None
            self._scan_fields = ()
            self._snapshot = None
            self._original_functions = {}
            raise
        self._bound = True

    def snapshot(self) -> Snapshot | IterationSnapshot:
        """Return the stable discoverable operation root.

        Replay ``pre`` snapshots observe the previous completed contexts until
        the next MJWarp solve or collision factory replaces them. Validity flags
        distinguish deterministic initialization and skipped operations from a
        context produced by the current solve.

        Returns:
            Stable operation snapshot updated by the live MJWarp hooks.

        Raises:
            DebugCaptureError: If the provider is not actively bound.
        """
        if not self._bound or self._closed or self._snapshot is None:
            raise DebugCaptureError("MJWarp debug operation provider is not bound.")
        if not self._runtime_lock.acquire(blocking=False):
            raise DebugCaptureError(
                f"Cannot snapshot MJWarp operations during active target boundary {self._active_boundary!r}."
            )
        try:
            cloned = _clone_dataclass_tree(self._snapshot, "mjwarp_operations")
        finally:
            self._runtime_lock.release()
        if not isinstance(cloned, (self.Snapshot, self.IterationSnapshot)):
            raise DebugCaptureError(f"Unexpected cloned operation root type {type(cloned).__qualname__}.")
        return cloned

    def close(self) -> None:
        """Restore every owned function and model option exactly once.

        Raises:
            DebugCaptureError: If ownership was lost or restoration failed.
                Restoration failures leave the provider bound so callers can retry.
        """
        if self._closed:
            return
        if not self._bound:
            self._closed = True
            return

        global _ACTIVE_PROVIDER
        with _INTERPOSITION_LOCK:
            conflicts: list[str] = []
            runtime_acquired = self._runtime_lock.acquire(blocking=False)
            if not runtime_acquired:
                conflicts.append(f"target boundary {self._active_boundary!r} is still in flight")
            if _ACTIVE_PROVIDER is not self:
                conflicts.append("global interposition ownership changed")
            for (module, name), installed in self._installed_functions.items():
                if getattr(module, name, None) is not installed:
                    conflicts.append(f"{module.__name__}.{name} was replaced by another owner")
            if self._scan_enabled:
                assert self._target_solver is not None
                assert self._graph_conditional_original is not None
                model = object.__getattribute__(self._target_solver, "mjw_model")
                expected = self._graph_conditional_original if self._graph_conditional_restored else False
                if bool(model.opt.graph_conditional) != expected:
                    conflicts.append("mjw_model.opt.graph_conditional changed outside provider ownership")
            if conflicts:
                if runtime_acquired:
                    self._runtime_lock.release()
                raise DebugCaptureError(
                    "Cannot restore MJWarp debug interposition because ownership was lost: "
                    + "; ".join(conflicts)
                    + "."
                )

            restored_functions: list[tuple[ModuleType, str]] = []
            try:
                if self._scan_enabled:
                    assert self._target_solver is not None
                    assert self._graph_conditional_original is not None
                    model = object.__getattribute__(self._target_solver, "mjw_model")
                    if not self._graph_conditional_restored:
                        try:
                            model.opt.graph_conditional = self._graph_conditional_original
                        except Exception:
                            self._graph_conditional_restored = (
                                bool(model.opt.graph_conditional) == self._graph_conditional_original
                            )
                            raise
                        self._graph_conditional_restored = True
                for module_name, original in self._original_functions.items():
                    module, name = module_name
                    setattr(module, name, original)
                    restored_functions.append(module_name)
                _ACTIVE_PROVIDER = None
            except Exception as exc:
                rollback_errors: list[str] = []
                for module, name in reversed(restored_functions):
                    try:
                        setattr(module, name, self._installed_functions[(module, name)])
                    except Exception as rollback_exc:
                        rollback_errors.append(f"{module.__name__}.{name}: {rollback_exc}")
                detail = f" Rollback also failed: {'; '.join(rollback_errors)}." if rollback_errors else ""
                raise DebugCaptureError(
                    "Failed to restore MJWarp debug interposition; the provider remains bound and close may be retried."
                    + detail
                ) from exc
            finally:
                self._runtime_lock.release()

        self._installed_functions.clear()
        self._original_functions.clear()
        self._scan_fields = ()
        self._snapshot = None
        self._data_plan = None
        self._solver_context_plan = None
        self._collision_context_plan = None
        self._modules = None
        self._target_solver = None
        self._graph_conditional_original = None
        self._active_boundary = None
        self._bound = False
        self._closed = True

    @property
    def _scan_enabled(self) -> bool:
        return self._first_nonfinite_include_fields is not None

    def _install_interposition(self) -> None:
        """Install strict wrappers after all allocations and discovery succeed."""
        global _ACTIVE_PROVIDER
        assert self._target_solver is not None
        assert self._modules is not None
        assert self._snapshot is not None

        modules = self._modules
        original_solver_factory = self._original_functions[(modules.solver, "create_solver_context")]
        original_solver_solve = self._original_functions[(modules.solver, "solve")]
        original_collision = self._original_functions[(modules.collision_driver, "collision")]
        original_collision_factory = self._original_functions[(modules.collision_driver, "create_collision_context")]

        @functools.wraps(original_solver_factory)
        def create_solver_context(m, d):
            context = original_solver_factory(m, d)
            if self._matches_target(m, d) and getattr(self._solver_local, "depth", 0):
                self._accept_solver_context(context)
            return context

        @functools.wraps(original_solver_solve)
        def solve(m, d):
            if not self._matches_target(m, d):
                return original_solver_solve(m, d)
            self._enter_runtime_boundary("solve")
            depth = getattr(self._solver_local, "depth", 0)
            self._solver_local.depth = depth + 1
            try:
                self._begin_solver_call(d)
                result = original_solver_solve(m, d)
                self._snapshot.solver_completed = True
                return result
            finally:
                self._solver_local.depth = depth
                self._leave_runtime_boundary("solve")

        @functools.wraps(original_collision)
        def collision(m, d):
            if not self._matches_target(m, d):
                return original_collision(m, d)
            self._enter_runtime_boundary("collision")
            depth = getattr(self._collision_local, "depth", 0)
            self._collision_local.depth = depth + 1
            self._snapshot.collision_call_count += 1
            self._snapshot.collision_context_valid = False
            self._snapshot.collision_completed = False
            try:
                result = original_collision(m, d)
                self._snapshot.collision_completed = True
                return result
            finally:
                self._collision_local.depth = depth
                self._leave_runtime_boundary("collision")

        @functools.wraps(original_collision_factory)
        def create_collision_context(naconmax):
            context = original_collision_factory(naconmax)
            if getattr(self._collision_local, "depth", 0):
                self._accept_collision_context(context)
            return context

        wrappers: dict[tuple[ModuleType, str], object] = {
            (modules.solver, "create_solver_context"): create_solver_context,
            (modules.solver, "solve"): solve,
            (modules.collision_driver, "collision"): collision,
            (modules.collision_driver, "create_collision_context"): create_collision_context,
        }

        if self._scan_enabled:
            original_iteration = self._original_functions[(modules.solver, "_solver_iteration")]

            @functools.wraps(original_iteration)
            def solver_iteration(m, d, ctx, step_size_cost, nsolving):
                result = original_iteration(m, d, ctx, step_size_cost, nsolving)
                if self._matches_target(m, d):
                    self._scan_solver_iteration(ctx)
                    self._iteration_index += 1
                return result

            wrappers[(modules.solver, "_solver_iteration")] = solver_iteration

        for wrapper in wrappers.values():
            setattr(wrapper, _OWNER_ATTRIBUTE, self)

        with _INTERPOSITION_LOCK:
            if _ACTIVE_PROVIDER is not None:
                raise DebugCaptureError("Another MJWarp debug operation provider owns the global interposition.")
            for (module, name), original in self._original_functions.items():
                if getattr(module, name, None) is not original:
                    raise DebugCaptureError(
                        f"Cannot interpose {module.__name__}.{name}: the callable changed after validation."
                    )

            graph_original = bool(self._target_solver.mjw_model.opt.graph_conditional)
            installed: list[tuple[ModuleType, str]] = []
            try:
                for (module, name), wrapper in wrappers.items():
                    setattr(module, name, wrapper)
                    installed.append((module, name))
                if self._scan_enabled:
                    self._target_solver.mjw_model.opt.graph_conditional = False
                    self._graph_conditional_original = graph_original
                _ACTIVE_PROVIDER = self
            except Exception as install_error:
                rollback_errors: list[str] = []
                for module, name in reversed(installed):
                    try:
                        setattr(module, name, self._original_functions[(module, name)])
                    except Exception as exc:
                        rollback_errors.append(f"{module.__name__}.{name}: {exc}")
                if self._scan_enabled:
                    try:
                        self._target_solver.mjw_model.opt.graph_conditional = graph_original
                    except Exception as exc:
                        rollback_errors.append(f"graph_conditional: {exc}")
                detail = f" Rollback also failed: {'; '.join(rollback_errors)}." if rollback_errors else ""
                raise DebugCaptureError(
                    f"Failed to install MJWarp debug interposition: {install_error}.{detail}"
                ) from install_error

            self._installed_functions = wrappers

    def _matches_target(self, model: object, data: object) -> bool:
        assert self._target_solver is not None
        return model is self._target_solver.mjw_model and data is self._target_solver.mjw_data

    def _enter_runtime_boundary(self, name: str) -> None:
        if not self._runtime_lock.acquire(blocking=False):
            raise DebugCaptureError(
                f"Concurrent MJWarp target boundary {name!r} overlaps active {self._active_boundary!r}."
            )
        self._active_boundary = name

    def _leave_runtime_boundary(self, name: str) -> None:
        if self._active_boundary != name:
            active = self._active_boundary
            self._active_boundary = None
            self._runtime_lock.release()
            raise DebugCaptureError(f"MJWarp target boundary ownership changed from {name!r} to {active!r}.")
        self._active_boundary = None
        self._runtime_lock.release()

    def _begin_solver_call(self, data: object) -> None:
        assert self._snapshot is not None
        self._snapshot.solver_call_count += 1
        self._snapshot.solver_context_valid = False
        self._snapshot.solver_completed = False
        self._iteration_index = 0
        if not isinstance(self._snapshot, self.IterationSnapshot):
            return
        self._snapshot.first_nonfinite_iteration = -1
        assert self._data_plan is not None
        self._snapshot.pre_solve_data_valid = False
        _copy_discovered_object(self._data_plan, data, self._snapshot.pre_solve_data)
        self._snapshot.pre_solve_data_valid = True
        self._snapshot.first_nonfinite_world_flags.fill(False)
        self._snapshot.first_nonfinite_global = False
        self._snapshot.first_nonfinite_valid = False
        for path in self._snapshot.first_nonfinite_path_flags:
            self._snapshot.first_nonfinite_path_flags[path] = False
        _zero_context(self._solver_context_plan, self._snapshot.first_nonfinite_context)

        _zero_array_fields(self._data_plan, self._snapshot.first_nonfinite_data)

    def _accept_solver_context(self, context: object) -> None:
        assert self._solver_context_plan is not None
        assert self._snapshot is not None
        self._solver_context_plan.validate_schema(context)

        self._snapshot.solver_context = context
        self._snapshot.solver_context_valid = True

    def _accept_collision_context(self, context: object) -> None:
        assert self._collision_context_plan is not None
        assert self._snapshot is not None
        self._collision_context_plan.validate_schema(context)
        self._snapshot.collision_context = context
        self._snapshot.collision_context_valid = True

    def _scan_solver_iteration(self, context: object) -> None:
        assert self._solver_context_plan is not None
        assert self._scan_fields
        assert isinstance(self._snapshot, self.IterationSnapshot)
        self._solver_context_plan.validate_schema(context)
        if self._snapshot.first_nonfinite_valid:
            return

        bad_paths: list[str] = []
        bad_worlds = np.zeros_like(self._snapshot.first_nonfinite_world_flags)
        global_bad = False
        for field in self._scan_fields:
            array = debug_value_to_numpy(field.validate(context), field.display_path)
            if array.dtype.kind not in "fc":
                continue
            mask = ~np.isfinite(array)
            if not bool(mask.any()):
                continue
            bad_paths.append(field.display_path)
            if (
                field.symbolic_shape
                and field.symbolic_shape[0] == "nworld"
                and array.ndim > 0
                and array.shape[0] == bad_worlds.size
            ):
                bad_worlds |= mask.reshape(mask.shape[0], -1).any(axis=1)
            else:
                global_bad = True
        if not bad_paths:
            return

        _copy_discovered_object(
            self._solver_context_plan,
            context,
            self._snapshot.first_nonfinite_context,
        )
        assert self._target_solver is not None
        assert self._data_plan is not None
        _copy_discovered_object(
            self._data_plan,
            self._target_solver.mjw_data,
            self._snapshot.first_nonfinite_data,
        )
        for path in bad_paths:
            self._snapshot.first_nonfinite_path_flags[path] = True
        np.copyto(self._snapshot.first_nonfinite_world_flags, bad_worlds)
        self._snapshot.first_nonfinite_global = global_bad
        self._snapshot.first_nonfinite_iteration = self._iteration_index
        self._snapshot.first_nonfinite_valid = True


def _uses_native_mujoco_cpu(solver: SolverMuJoCo, label: str) -> bool:
    """Read the initialized backend selector with an actionable error."""
    try:
        return bool(object.__getattribute__(solver, "use_mujoco_cpu"))
    except AttributeError as exc:
        raise DebugCaptureError(f"SolverMuJoCo entry {label!r} is not initialized: missing use_mujoco_cpu.") from exc


def _select_mjwarp_solver(solver: SolverBase | Mapping[str, SolverBase]) -> SolverMuJoCo:
    """Select one unique live MJWarp-backed SolverMuJoCo from a manager provider."""
    candidates: list[tuple[str, SolverMuJoCo]] = []
    observed: list[str] = []
    native_cpu: list[str] = []
    if isinstance(solver, Mapping):
        for key, value in solver.items():
            observed.append(f"{key!r}={type(value).__module__}.{type(value).__qualname__}")
            if isinstance(value, SolverMuJoCo):
                if _uses_native_mujoco_cpu(value, str(key)):
                    native_cpu.append(str(key))
                else:
                    candidates.append((str(key), value))
    else:
        observed.append(f"{type(solver).__module__}.{type(solver).__qualname__}")
        if isinstance(solver, SolverMuJoCo):
            if _uses_native_mujoco_cpu(solver, "solver"):
                native_cpu.append("solver")
            else:
                candidates.append(("solver", solver))

    unique: dict[int, tuple[SolverMuJoCo, list[str]]] = {}
    for name, candidate in candidates:
        entry = unique.setdefault(id(candidate), (candidate, []))
        entry[1].append(name)
    if not unique:
        detail = ", ".join(observed) if observed else "empty mapping"
        if native_cpu:
            raise DebugCaptureError(
                "MJWarpDebugOperationProvider cannot bind only SolverMuJoCo(use_mujoco_cpu=True) entries "
                f"at {native_cpu}; select the mujoco_warp backend instead."
            )
        raise DebugCaptureError(
            "MJWarpDebugOperationProvider requires exactly one live newton.solvers.SolverMuJoCo; "
            f"found none in {detail}."
        )
    if len(unique) != 1:
        labels = ["/".join(names) for _, names in unique.values()]
        raise DebugCaptureError(
            f"MJWarpDebugOperationProvider requires exactly one live SolverMuJoCo; found {len(unique)} at {labels}."
        )

    target = next(iter(unique.values()))[0]
    try:
        use_mujoco_cpu = object.__getattribute__(target, "use_mujoco_cpu")
        mjw_model = object.__getattribute__(target, "mjw_model")
        mjw_data = object.__getattribute__(target, "mjw_data")
        model = object.__getattribute__(target, "model")
        object.__getattribute__(model, "device")
    except AttributeError as exc:
        raise DebugCaptureError(f"Selected SolverMuJoCo is not fully initialized: missing {exc}.") from exc
    if use_mujoco_cpu:
        raise DebugCaptureError(
            "MJWarpDebugOperationProvider cannot bind SolverMuJoCo(use_mujoco_cpu=True); "
            "select the mujoco_warp backend instead."
        )
    if mjw_model is None or mjw_data is None:
        raise DebugCaptureError("Selected MJWarp-backed SolverMuJoCo has no finalized mjw_model or mjw_data.")
    return target


def _load_mjwarp_modules() -> _MJWarpModules:
    """Import only the isolated private modules required by the adapter."""
    try:
        return _MJWarpModules(
            forward=importlib.import_module("mujoco_warp._src.forward"),
            solver=importlib.import_module("mujoco_warp._src.solver"),
            collision_driver=importlib.import_module("mujoco_warp._src.collision_driver"),
        )
    except Exception as exc:
        raise DebugCaptureError(f"Unable to import pinned MJWarp debug boundaries: {exc}") from exc


def _validate_mjwarp_call_path(
    target: SolverMuJoCo,
    modules: _MJWarpModules,
    *,
    scan_iterations: bool,
) -> dict[tuple[ModuleType, str], object]:
    """Validate signatures and live global lookups before any mutation."""
    expected = {
        (modules.solver, "create_solver_context"): ("m", "d"),
        (modules.solver, "solve"): ("m", "d"),
        (modules.collision_driver, "collision"): ("m", "d"),
        (modules.collision_driver, "create_collision_context"): ("naconmax",),
    }
    if scan_iterations:
        expected[(modules.solver, "_solver_iteration")] = (
            "m",
            "d",
            "ctx",
            "step_size_cost",
            "nsolving",
        )

    originals: dict[tuple[ModuleType, str], object] = {}
    for (module, name), parameter_names in expected.items():
        function = getattr(module, name, None)
        if not callable(function):
            raise DebugCaptureError(f"Pinned MJWarp boundary {module.__name__}.{name} is missing or not callable.")
        owner = getattr(function, _OWNER_ATTRIBUTE, None)
        if owner is not None:
            raise DebugCaptureError(
                f"Pinned MJWarp boundary {module.__name__}.{name} is already owned by another debug provider."
            )
        _validate_signature(function, module.__name__, name, parameter_names)
        originals[(module, name)] = function

    solver_body = inspect.unwrap(originals[(modules.solver, "solve")])
    if solver_body.__globals__.get("create_solver_context") is not originals[(modules.solver, "create_solver_context")]:
        raise DebugCaptureError(
            "MJWarp solver.solve no longer runtime-resolves solver.create_solver_context; "
            "the pinned debug adapter must be updated."
        )
    collision_body = inspect.unwrap(originals[(modules.collision_driver, "collision")])
    if (
        collision_body.__globals__.get("create_collision_context")
        is not originals[(modules.collision_driver, "create_collision_context")]
    ):
        raise DebugCaptureError(
            "MJWarp collision_driver.collision no longer runtime-resolves create_collision_context; "
            "the pinned debug adapter must be updated."
        )
    if getattr(modules.forward, "solver", None) is not modules.solver:
        raise DebugCaptureError(
            "MJWarp forward.solver module identity changed; the pinned debug adapter must be updated."
        )
    if getattr(modules.forward, "collision_driver", None) is not modules.collision_driver:
        raise DebugCaptureError(
            "MJWarp forward.collision_driver module identity changed; the pinned debug adapter must be updated."
        )
    target_module = object.__getattribute__(target, "_mujoco_warp")
    if getattr(target_module, "step", None) is not getattr(modules.forward, "step", None):
        raise DebugCaptureError("SolverMuJoCo no longer calls the pinned mujoco_warp._src.forward.step boundary.")
    return originals


def _validate_signature(function: object, module_name: str, name: str, expected: tuple[str, ...]) -> None:
    """Fail actionably when an upstream private function signature drifts."""
    try:
        signature = inspect.signature(function)
    except Exception as exc:
        raise DebugCaptureError(f"Cannot inspect {module_name}.{name}: {exc}") from exc
    parameters = tuple(signature.parameters.values())
    actual_names = tuple(parameter.name for parameter in parameters)
    positional = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }
    if actual_names != expected or any(parameter.kind not in positional for parameter in parameters):
        raise DebugCaptureError(
            f"MJWarp boundary {module_name}.{name} signature drifted: expected "
            f"({', '.join(expected)}), got {signature}. Update the isolated MJWarp debug adapter."
        )


def _validate_scan_world_extents(
    fields: tuple[DebugCaptureField, ...],
    root: object,
    nworld: int,
) -> None:
    """Validate explicit symbolic world ownership against the live extent."""
    for field in fields:
        if not field.symbolic_shape or field.symbolic_shape[0] != "nworld":
            continue
        array = debug_value_to_numpy(field.validate(root), field.display_path)
        actual = array.shape[0] if array.ndim > 0 else None
        if actual != nworld:
            raise DebugSchemaError(
                f"MJWarp scan field {field.display_path!r} declares leading nworld "
                f"but has extent {actual}; expected {nworld}."
            )


def _require_complete_plan(plan: DebugCapturePlan, provider_name: str) -> None:
    """Require every discovered provider field to be allocated and capturable."""
    incomplete = tuple(entry.display_path for entry in (*plan.unallocated, *plan.ignored))
    if not plan.fields or incomplete:
        raise DebugCaptureError(
            f"{provider_name} discovery must produce a complete allocated schema; incomplete paths: {list(incomplete)}."
        )


def _zero_context(plan: DebugCapturePlan, context: object) -> None:
    """Make every discovered context leaf deterministic without a field table."""
    for field in plan.fields:
        value = field.get(context)
        zeroed = _zero_debug_value(value)
        if zeroed is not value:
            raise DebugSchemaError(
                f"MJWarp context field '{field.display_path}' is immutable; deterministic seeding requires "
                "an upstream context factory value rather than replacement."
            )


def _zero_array_fields(plan: DebugCapturePlan, root: object) -> None:
    """Zero discovered array leaves while preserving deterministic scalar metadata."""
    for field in plan.fields:
        value = field.validate(root)
        if isinstance(value, (wp.array, torch.Tensor, np.ndarray)):
            _zero_debug_value(value)


def _zero_debug_value(value: object) -> object:
    """Zero one cloned debug leaf while preserving its exact schema."""
    if isinstance(value, wp.array):
        value.zero_()
        return value
    if isinstance(value, torch.Tensor):
        value.zero_()
        return value
    if isinstance(value, np.ndarray):
        value.fill(0)
        return value
    if isinstance(value, enum.Enum):
        return value
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return type(value)(0)
    if isinstance(value, float):
        return type(value)(0.0)
    if isinstance(value, complex):
        return type(value)(0.0)
    if isinstance(value, str):
        return ""
    if isinstance(value, list):
        return [_zero_debug_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_zero_debug_value(item) for item in value)
    raise DebugCaptureError(
        f"Cannot deterministically zero debug value of type {type(value).__module__}.{type(value).__qualname__}."
    )


def _clone_dataclass_context(plan: DebugCapturePlan, context: object) -> object:
    """Clone a flat workspace dataclass from discovery without a field table."""
    field_names = tuple(field.name for field in dataclasses.fields(context) if field.init)
    planned_names = tuple(
        field.path[0] for field in plan.fields if len(field.path) == 1 and isinstance(field.path[0], str)
    )
    if len(planned_names) != len(plan.fields) or set(planned_names) != set(field_names):
        raise DebugSchemaError(
            "MJWarp SolverContext is no longer a flat dataclass of capturable fields; "
            "update the isolated operation provider before using iteration snapshots."
        )
    values = {
        str(field.path[0]): clone_debug_value(field.validate(context), field.display_path) for field in plan.fields
    }
    try:
        return type(context)(**values)
    except Exception as exc:
        raise DebugSchemaError(f"Failed to reconstruct cloned MJWarp SolverContext: {exc}") from exc


def _clone_dataclass_tree(value: object, path: str) -> object:
    """Recursively clone a complete dataclass tree and every supported leaf."""
    try:
        return clone_debug_value(value, path)
    except DebugCaptureError as leaf_error:
        if isinstance(value, Mapping):
            return {key: _clone_dataclass_tree(item, f"{path}[{key!r}]") for key, item in value.items()}
        if isinstance(value, list):
            return [_clone_dataclass_tree(item, f"{path}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(_clone_dataclass_tree(item, f"{path}[{index}]") for index, item in enumerate(value))
        if not dataclasses.is_dataclass(value) or isinstance(value, type):
            raise leaf_error
    fields = dataclasses.fields(value)
    non_init = [field.name for field in fields if not field.init]
    if non_init:
        raise DebugSchemaError(f"MJWarp Data dataclass at '{path}' has non-init fields {non_init}.")
    values = {
        field.name: _clone_dataclass_tree(object.__getattribute__(value, field.name), f"{path}.{field.name}")
        for field in fields
    }
    try:
        return type(value)(**values)
    except Exception as exc:
        raise DebugSchemaError(f"Failed to reconstruct MJWarp Data dataclass at '{path}': {exc}") from exc


def _copy_discovered_object(
    plan: DebugCapturePlan,
    source_root: object,
    destination_root: object,
) -> None:
    """Copy one full discovered object into an existing same-type clone."""
    plan.validate_schema(source_root)
    plan.validate_schema(destination_root)
    for field in plan.fields:
        source = field.validate(source_root)
        target = field.validate(destination_root)
        if isinstance(source, wp.array) and isinstance(target, wp.array):
            wp.copy(target, source)
        elif isinstance(source, torch.Tensor) and isinstance(target, torch.Tensor):
            target.copy_(source)
        elif isinstance(source, np.ndarray) and isinstance(target, np.ndarray):
            np.copyto(target, source)
        else:
            if not field.path or any(not isinstance(step, str) for step in field.path):
                raise DebugSchemaError(f"Cannot replace indexed immutable debug field '{field.display_path}'.")
            parent = destination_root
            for step in field.path[:-1]:
                parent = object.__getattribute__(parent, step)
            setattr(parent, field.path[-1], clone_debug_value(source, field.display_path))


def _validate_patterns_or_none(
    patterns: tuple[str, ...] | None,
    name: str,
) -> tuple[str, ...] | None:
    if patterns is None:
        return None
    return _validate_patterns(patterns, name, allow_empty=False)


def _validate_patterns(
    patterns: tuple[str, ...],
    name: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(patterns, tuple):
        raise TypeError(f"{name} must be a tuple of strings.")
    if not patterns and not allow_empty:
        raise ValueError(f"{name} must not be empty.")
    if any(not isinstance(pattern, str) or not pattern for pattern in patterns):
        raise TypeError(f"{name} must contain only non-empty strings.")
    if len(patterns) != len(set(patterns)):
        raise ValueError(f"{name} must not contain duplicate patterns.")
    return patterns
