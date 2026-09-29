# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact homogeneous populations owned by one simulation context.

This experimental backend currently supports prepared native-contact MuJoCo Warp
solvers and headless execution. Task assignment, policy buffers, and random number
generation belong to the environment, independently of native population storage.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Sequence
from dataclasses import MISSING, field
from typing import TYPE_CHECKING

import newton
import warp as wp
from newton.solvers import SolverMuJoCo

from isaaclab.physics import PhysicsCfg, PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend
from isaaclab.sim import BackendCfg
from isaaclab.utils import checked_apply, configclass

from .newton_manager_cfg import NewtonCfg

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext

__all__ = [
    "NewtonPopulation",
    "NewtonPopulationBackend",
    "NewtonPopulationBackendCfg",
    "NewtonPopulationCfg",
    "NewtonPopulationManager",
]


@wp.kernel
def _select_articulations(worlds: wp.array[int], world_mask: wp.array[bool], selected: wp.array[bool]):
    i = wp.tid()
    selected[i] = world_mask[worlds[i]]


@wp.kernel
def _select_local_worlds(selected: wp.array[bool]):
    i = wp.tid()
    selected[i] = i < selected.shape[0] - 1


class NewtonPopulation:
    """One exact population and every native object used by its graph.

    Experimental. ``state_0`` is always the current state at a physics-frame
    boundary. Two-substep graph recordings return to the same buffer pointers.
    Property updates and partial resets follow the underlying solver's contracts.
    The timestep must be finite and positive, and each frame requires a positive
    even integer number of native substeps, including for eager execution.
    """

    def __init__(self, prototype: SolverMuJoCo, count: int, *, dt: float, substeps: int, use_cuda_graph: bool):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")
        if isinstance(substeps, bool) or operator.index(substeps) < 2 or substeps % 2:
            raise ValueError("substeps must be a positive even integer.")
        if use_cuda_graph:
            if not prototype.model.device.is_cuda:
                raise ValueError("Population graph capture requires CUDA.")
            if prototype.update_data_interval > 0 and 2 % prototype.update_data_interval:
                raise ValueError("Two-substep graphs require update_data_interval to divide two.")
        self.solver = prototype.replicate(count)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.dt = dt
        self.substeps = operator.index(substeps)
        self.graph: wp.Graph | None = None
        self._reset_graph: wp.Graph | None = None
        self._fk_mask = wp.empty(self.model.articulation_count, dtype=wp.bool, device=self.model.device)
        self._reset_mask = wp.zeros(self.model.world_count + 1, dtype=wp.bool, device=self.model.device)
        self.forward()
        self.state_1.assign(self.state_0)
        if use_cuda_graph:
            # Recording does not execute a physics step. In particular, do not
            # warm up and reset: that needlessly changes the prepared snapshot.
            # CUDA cannot capture Torch's legacy default stream. Record on an
            # owned nonblocking stream, then replay on the caller's producer stream.
            with wp.ScopedStream(wp.Stream(self.model.device), sync_exit=True):
                with wp.ScopedCapture(device=self.model.device) as capture:
                    self._simulate(2)
                self.graph = capture.graph
                with wp.ScopedCapture(device=self.model.device) as capture:
                    self.reset(self._reset_mask, flags=0)
                self._reset_graph = capture.graph

    @property
    def model(self) -> newton.Model:
        """The solver-owned model, without a second model reference to update."""
        return self.solver.model

    def _simulate(self, substeps: int) -> None:
        for _ in range(substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        """Advance one physics frame while holding the current control inputs."""
        if self.graph is None:
            self._simulate(self.substeps)
        else:
            for _ in range(self.substeps // 2):
                wp.capture_launch(self.graph)

    def forward(self, world_mask: wp.array | None = None) -> None:
        """Update body poses [m, xyzw] and twists from current joint state."""
        mask = None
        if world_mask is not None:
            self._validate_world_mask(world_mask)
            wp.launch(
                _select_articulations,
                dim=self.model.articulation_count,
                inputs=[self.model.articulation_world, world_mask, self._fk_mask],
                device=self.model.device,
            )
            mask = self._fk_mask
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0, mask=mask)

    def reset(self, world_mask: wp.array | None = None, flags: newton.StateFlags | int | None = None) -> None:
        """Reset selected worlds; ``flags=0`` preserves task-written joints.

        Property edits must be notified before this call. This follows the native
        solver's reset semantics; it does not restore a complete episode snapshot.
        """
        if world_mask is not None:
            self._validate_world_mask(world_mask)
        if self._reset_graph is not None and flags is not None and int(flags) == 0:
            if world_mask is None:
                wp.launch(_select_local_worlds, self._reset_mask.size, [self._reset_mask], device=self.model.device)
            else:
                wp.copy(self._reset_mask, world_mask)
            wp.capture_launch(self._reset_graph)
            return
        self.solver.reset(self.state_0, world_mask=world_mask, flags=flags)
        self.forward(world_mask)
        self.state_1.assign(self.state_0)

    def notify_model_changed(self, flags: newton.ModelFlags | int, *, world_mask: wp.array | None = None) -> None:
        """Synchronize in-place physical-property edits for the selected worlds."""
        if world_mask is not None:
            self._validate_world_mask(world_mask)
        self.solver.notify_model_changed(flags, world_mask=world_mask)

    def _validate_world_mask(self, mask: wp.array) -> None:
        if mask.dtype != wp.bool or mask.shape != (self.model.world_count + 1,) or mask.device != self.model.device:
            raise ValueError("world_mask must be a same-device bool array of shape (world_count + 1,).")


@configclass
class NewtonPopulationBackendCfg(BackendCfg):
    """Experimental prepared population allocation inputs, frozen after registration."""

    class_type: type = "{DIR}.population:NewtonPopulationBackend"
    prototypes: tuple[SolverMuJoCo, ...] = field(kw_only=True, metadata={"copy": False})
    """Prepared pristine one-world solvers, borrowed read-only and retained for the backend lifetime."""
    counts: tuple[int, ...] = MISSING
    """Exact initial world counts; zero omits that prototype's runtime population."""
    dt: float = MISSING
    """Native solver timestep [s]. One physics frame advances ``dt * substeps``."""
    substeps: int = 2
    """Positive even native substeps per physics frame, preserving canonical state pointers."""
    use_cuda_graph: bool = False
    """Record a repeating two-substep CUDA graph from each untouched prepared replica."""
    stream_count: int = 8
    """Maximum concurrent population streams; one executes on the caller's stream."""


class NewtonPopulationBackend:
    """Simulation-owned prepared sources and exact independently mutable populations.

    Experimental. ``replace`` preserves every object for unchanged counts. Optional
    native row maps transfer ongoing worlds into changed populations before they
    become visible. Logical task actor IDs remain entirely outside this backend.
    """

    def __init__(self, cfg: NewtonPopulationBackendCfg):
        if not cfg.prototypes or any(not isinstance(source, SolverMuJoCo) for source in cfg.prototypes):
            raise ValueError("At least one prepared SolverMuJoCo prototype is required.")
        self.prototypes = tuple(cfg.prototypes)
        self.device = self.prototypes[0].model.device
        if any(source.model.device != self.device for source in self.prototypes):
            raise ValueError("All population prototypes must use the same device.")
        if not math.isfinite(cfg.dt) or cfg.dt <= 0:
            raise ValueError("dt must be finite and positive.")
        if isinstance(cfg.substeps, bool) or operator.index(cfg.substeps) < 2 or cfg.substeps % 2:
            raise ValueError("substeps must be a positive even integer.")
        if isinstance(cfg.stream_count, bool) or operator.index(cfg.stream_count) < 1:
            raise ValueError("stream_count must be a positive integer.")
        if cfg.use_cuda_graph:
            if not self.device.is_cuda:
                raise ValueError("Population graph capture requires CUDA.")
            if any(source.update_data_interval > 0 and 2 % source.update_data_interval for source in self.prototypes):
                raise ValueError("Two-substep graphs require update_data_interval to divide two.")
        self.dt, self.substeps, self.use_cuda_graph = cfg.dt, operator.index(cfg.substeps), cfg.use_cuda_graph
        stream_count = min(operator.index(cfg.stream_count), len(self.prototypes)) if self.device.is_cuda else 1
        self._streams = tuple(wp.Stream(self.device) for _ in range(stream_count)) if stream_count > 1 else ()
        self._complete = tuple(wp.Event(self.device) for _ in self._streams)
        self._fork = wp.Event(self.device) if self._streams else None
        self._last_work = wp.Event(self.device) if self.device.is_cuda else None
        self._transfer_status = (
            wp.empty(len(self.prototypes), dtype=wp.int32, device="cpu", pinned=True) if self.device.is_cuda else None
        )
        self._stream_groups: tuple[tuple[int, ...], ...] = tuple(() for _ in self._streams)
        self.populations: tuple[NewtonPopulation | None, ...] = (None,) * len(self.prototypes)
        self._closed = False
        self.replace(cfg.counts)

    @property
    def counts(self) -> tuple[int, ...]:
        """Current exact world counts, derived from the owned models."""
        return tuple(0 if population is None else population.model.world_count for population in self.populations)

    def replace(
        self, counts: tuple[int, ...], *, survivors: Sequence[tuple[Sequence[int], Sequence[int]]] | None = None
    ) -> None:
        """Build all changed populations before publishing; fence before retirement.

        A construction failure leaves the previous population tuple live and valid.
        Equal counts reuse the existing model, state, control, solver, and graph.
        ``survivors`` supplies one ``(source_rows, target_rows)`` pair per prototype.
        These are host integer sequences describing physics rows, not task IDs.
        For unchanged counts, surviving rows must retain their original indices.
        """
        if self._closed:
            raise RuntimeError("The population backend is closed.")
        if len(counts) != len(self.prototypes):
            raise ValueError("Supply one world count per prepared prototype.")
        if any(isinstance(count, bool) or operator.index(count) < 0 for count in counts):
            raise ValueError("Population counts must be nonnegative integers.")
        counts = tuple(operator.index(count) for count in counts)
        if survivors is not None:
            if len(survivors) != len(counts):
                raise ValueError("Supply one survivor row map per prototype.")
            for count, old_count, (source_rows, target_rows) in zip(counts, self.counts, survivors, strict=True):
                if len(source_rows) != len(target_rows):
                    raise ValueError("Source and target survivor row maps must have equal lengths.")
                for rows, limit in ((source_rows, old_count), (target_rows, count)):
                    if any(isinstance(row, bool) or not 0 <= operator.index(row) < limit for row in rows):
                        raise ValueError("Survivor rows must be valid native world indices.")
                    if len(set(rows)) != len(rows):
                        raise ValueError("Survivor rows must be unique within each population.")
                if count == old_count and tuple(source_rows) != tuple(target_rows):
                    raise ValueError("Unchanged populations retain their native row indices.")
        if counts == self.counts:
            return
        old_counts = self.counts
        replacements = [
            population if count else None for population, count in zip(self.populations, counts, strict=True)
        ]
        stream_groups = self._stream_groups
        if self._streams:
            groups, loads = [[] for _ in self._streams], [0] * len(self._streams)
            active = [i for i, count in enumerate(counts) if count]
            # Prepared sources contain one local world; exact replication scales
            # their native DOF count without reading the new device allocations.
            weights = [
                source.model.joint_dof_count * count for source, count in zip(self.prototypes, counts, strict=True)
            ]
            for index in sorted(active, key=weights.__getitem__, reverse=True):
                stream = min(range(len(loads)), key=loads.__getitem__)
                groups[stream].append(index)
                loads[stream] += max(1, weights[index])
            stream_groups = tuple(tuple(group) for group in groups)
        build_groups = stream_groups if self._streams else (tuple(i for i, count in enumerate(counts) if count),)
        producer = wp.get_stream(self.device) if self.device.is_cuda else None
        statuses = []
        try:
            if producer is not None:
                producer.wait_event(self._last_work)
            if self._streams:
                producer.record_event(self._fork)
            for group, indices in enumerate(build_groups):
                changed = [index for index in indices if counts[index] != old_counts[index]]
                if not changed:
                    continue
                stream = self._streams[group] if self._streams else None
                if stream is not None:
                    stream.wait_event(self._fork)
                # One host thread records independent graphs sequentially. Only
                # queued construction and migration work can overlap on device.
                with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
                    for index in changed:
                        replacements[index] = NewtonPopulation(
                            self.prototypes[index],
                            counts[index],
                            dt=self.dt,
                            substeps=self.substeps,
                            use_cuda_graph=self.use_cuda_graph,
                        )
                        if survivors is not None and old_counts[index]:
                            source_rows, target_rows = survivors[index]
                            if len(source_rows):
                                old, new = self.populations[index], replacements[index]
                                status = new.solver.copy_worlds_from(
                                    old.solver,
                                    source_rows,
                                    target_rows,
                                    states=((old.state_0, new.state_0), (old.state_1, new.state_1)),
                                    controls=((old.control, new.control),),
                                )
                                # Retain the status and its native descriptor
                                # owners until this transaction has completed.
                                statuses.append((index, status))
                                if self._transfer_status is not None:
                                    wp.copy(self._transfer_status, status, dest_offset=index, count=1)
                if stream is not None:
                    stream.record_event(self._complete[group])
                    producer.wait_event(self._complete[group])
            # The previous-work dependency and new joins cover initialization,
            # capture setup and pinned status copies, even if the caller changed
            # streams or this plan removes every population.
            if producer is not None:
                producer.record_event(self._last_work)
                wp.synchronize_stream(producer)
            else:
                wp.synchronize_device(self.device)
            if statuses:
                values = self._transfer_status.numpy() if self._transfer_status is not None else None
                if any(
                    int(values[index] if values is not None else status.numpy()[0]) != 0 for index, status in statuses
                ):
                    raise RuntimeError("Native world transfer failed; the previous populations remain active.")
        except BaseException:
            # Construction can fail before a worker completion event is recorded.
            # Keep the conservative failure fence before releasing partial owners.
            wp.synchronize_device(self.device)
            raise
        self.populations, self._stream_groups = tuple(replacements), stream_groups

    def step(self) -> None:
        """Advance populations, ordering inputs and outputs on the caller's Warp stream."""
        self._run(lambda index: self.populations[index].step())

    def reconcile_state(self, world_masks: Sequence[wp.array | None], model_flags: newton.ModelFlags | int = 0) -> None:
        """Apply authored properties and coordinates, ordered on the caller's Warp stream.

        Supply one mask per prototype, using ``None`` for absent populations or to
        select all worlds in an active population. Masks use each model's native
        world order and include the global-world sentinel. Property notification
        precedes ``reset(flags=0)``, which preserves task-written joint coordinates.
        Empty masks retain the solver's normal reset/cache-refresh semantics.
        """
        if self._closed:
            raise RuntimeError("The population backend is closed.")
        if len(world_masks) != len(self.populations):
            raise ValueError("Supply one world mask per population.")
        for population, mask in zip(self.populations, world_masks, strict=True):
            if population is None:
                if mask is not None:
                    raise ValueError("Absent populations require a None world mask.")
            elif mask is not None:
                population._validate_world_mask(mask)

        def reconcile(index):
            population, mask = self.populations[index], world_masks[index]
            if model_flags:
                population.notify_model_changed(model_flags, world_mask=mask)
            population.reset(mask, flags=0)

        self._run(reconcile)

    def _run(self, operation: Callable[[int], None]) -> None:
        """Fork producer inputs, execute independent populations, and join their outputs."""
        if self._closed:
            raise RuntimeError("The population backend is closed.")
        producer = wp.get_stream(self.device) if self._last_work is not None else None
        if producer is not None:
            producer.wait_event(self._last_work)
        try:
            if not self._streams:
                for index, population in enumerate(self.populations):
                    if population is not None:
                        operation(index)
                return
            producer.record_event(self._fork)
            submitted = []
            try:
                for stream, indices, complete in zip(self._streams, self._stream_groups, self._complete, strict=True):
                    if indices:
                        stream.wait_event(self._fork)
                        try:
                            with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
                                for index in indices:
                                    operation(index)
                        finally:
                            stream.record_event(complete)
                            submitted.append(complete)
            finally:
                for complete in submitted:
                    producer.wait_event(complete)
        finally:
            if producer is not None:
                producer.record_event(self._last_work)

    def forward(self) -> None:
        """Update forward kinematics for every active population."""
        if self._closed:
            raise RuntimeError("The population backend is closed.")
        producer = wp.get_stream(self.device) if self._last_work is not None else None
        if producer is not None:
            producer.wait_event(self._last_work)
        try:
            for population in self.populations:
                if population is not None:
                    population.forward()
        finally:
            if producer is not None:
                producer.record_event(self._last_work)

    def close(self) -> None:
        """Wait for queued work before releasing graphs and their native storage."""
        if not self._closed:
            wp.synchronize_device(self.device)
            self.populations = ()
            self.prototypes = ()
            self._stream_groups = self._streams = self._complete = ()
            self._fork = self._last_work = self._transfer_status = None
            self._closed = True


@configclass
class NewtonPopulationCfg(PhysicsCfg):
    """Experimental headless physics lifecycle for an explicitly installed population backend."""

    class_type: type = "{DIR}.population:NewtonPopulationManager"
    prototype_physics: NewtonCfg | None = None
    """Optional authoring settings for a caller preparing native one-world prototypes.

    The caller creates prototypes once and registers the resulting backend. This
    manager never authors assets or invokes a second global Newton manager.
    """


class NewtonPopulationManager(PhysicsManager):
    """Bind the simulation lifecycle to one context-owned population backend.

    Install the resource returned by ``sim.get_or_create_backend(cfg)`` before
    calling ``sim.reset()``. No global Newton model or selection view is exposed.
    """

    _backend: NewtonPopulationBackend | None = None
    _scene_data: SceneDataBackend | None = None
    _ready = False

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        if sim_context.resolve_visualizer_types() or sim_context.get_setting("/isaaclab/cameras_enabled"):
            raise ValueError("Newton populations currently support headless execution without visualizers or cameras.")
        super().initialize(sim_context)
        cls._backend = None
        cls._scene_data = SceneDataBackend()
        cls._ready = False

    @classmethod
    def create_builder(cls, up_axis: str = "Z") -> newton.ModelBuilder:
        """Create a prototype builder with this manager's solver schema and configured physics defaults."""
        physics = cls._cfg.prototype_physics
        if physics is None:
            raise ValueError("Prototype authoring requires NewtonPopulationCfg.prototype_physics.")
        builder = newton.ModelBuilder(up_axis=up_axis, gravity=cls._sim.cfg.gravity)
        SolverMuJoCo.register_custom_attributes(builder)
        checked_apply(physics.default_shape_cfg, builder.default_shape_cfg)
        builder.default_bvh_cfg = newton.ModelBuilder.BvhConfig(
            mesh_constructor=physics.bvh_constructor_geometry,
            gaussian_constructor=physics.bvh_constructor_gaussian,
            shape_constructor=physics.bvh_constructor_scene,
        )
        return builder

    @classmethod
    def install(cls, backend: NewtonPopulationBackend) -> None:
        """Borrow the explicitly composed, simulation-owned native resource."""
        if cls._scene_data is None:
            raise RuntimeError("Initialize the simulation context before installing a population backend.")
        if cls._backend is not None and cls._backend is not backend:
            raise RuntimeError("A population backend is already installed; resize it with replace().")
        if backend.device != wp.get_device(cls.get_device()):
            raise ValueError("The population backend must use the simulation device.")
        if not math.isclose(backend.dt * backend.substeps, cls.get_physics_dt(), rel_tol=1e-7):
            raise ValueError("Population dt * substeps must equal SimulationCfg.dt.")
        cls._backend = backend

    @classmethod
    def reset(cls, soft: bool = False) -> None:
        if cls._backend is None:
            raise RuntimeError("Install a context-owned NewtonPopulationBackend before sim.reset().")
        if not cls._ready:
            cls.dispatch_event(PhysicsEvent.MODEL_INIT)
        elif not soft:
            for population in cls._backend.populations:
                if population is not None:
                    population.reset()
        cls._backend.forward()
        cls._ready = True
        cls.dispatch_event(PhysicsEvent.PHYSICS_READY)

    @classmethod
    def forward(cls) -> None:
        if cls._backend is not None:
            cls._backend.forward()

    @classmethod
    def step(cls) -> None:
        if not cls._ready or cls._backend is None:
            raise RuntimeError("The population simulation must be reset before stepping.")
        cls._backend.step()
        PhysicsManager._sim_time += cls.get_physics_dt()

    @classmethod
    def get_scene_data_backend(cls) -> SceneDataBackend:
        return cls._scene_data

    @classmethod
    def close(cls) -> None:
        try:
            super().close()
        finally:
            # SimulationContext owns close() of resources in its backend registry.
            cls._backend = None
            cls._scene_data = None
            cls._ready = False
