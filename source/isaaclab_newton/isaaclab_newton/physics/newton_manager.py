# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics manager for Isaac Lab."""

from __future__ import annotations

import contextlib
import gc
import inspect
import logging
import re
from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import warp as wp


@contextlib.contextmanager
def _paused_gc():
    """Pause Python garbage collection for the duration of a CUDA graph capture.

    A garbage-collection pass inside a capture window can drop the last
    reference to an array allocated earlier in the capture. While the capture
    is paused for a ``wp.capture_while``/``wp.capture_if`` conditional body,
    Warp then inserts the memory free node into the body graph with dependency
    nodes from the parent graph, which fails and latches a sticky CUDA error
    that poisons a later, unrelated copy. Reference-count-driven frees are
    deterministic solver behavior and remain allowed; only cyclic collection
    is paused until the capture window closes.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


from newton import (
    Axis,
    CollisionPipeline,
    Contacts,
    Control,
    Model,
    ModelBuilder,
    ModelFlags,
    State,
    eval_fk,
)
from newton.geometry import HydroelasticSDF
from newton.sensors import SensorContact as NewtonContactSensor
from newton.sensors import SensorFrameTransform
from newton.sensors import SensorIMU as NewtonSensorIMU
from newton.solvers import SolverBase, SolverKamino

from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend, SceneDataFormat, SceneDataPublication
from isaaclab.sim import SimulationContext
from isaaclab.utils.timer import Timer
from isaaclab.utils.warp.index_kernel import IndexKernelDispatcher

from isaaclab_newton.cloner.replicate import NewtonReplicateContext
from isaaclab_newton.physics.featherstone_manager_cfg import FeatherstoneSolverCfg
from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg
from isaaclab_newton.physics.newton_manager_cfg import NewtonSolverCfg
from isaaclab_newton.physics.xpbd_manager_cfg import XPBDSolverCfg

if TYPE_CHECKING:
    from isaaclab.actuators.newton import NewtonActuatorAdapter
    from isaaclab.cloner import ClonePlan

    from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg


_SENSORS_BY_STATE_ATTRIBUTE = {
    "body_qdd": "the IMU or PVA sensor",
    "body_parent_f": "the joint-wrench sensor",
}
_SENSOR_STAGE_STATE_ATTRIBUTES = frozenset(_SENSORS_BY_STATE_ATTRIBUTE)


def _compile_label_pattern(expr: str | list[str] | None) -> re.Pattern[str] | None:
    """Compile selector expressions for Newton's full label matching."""
    if not expr:
        return None
    return re.compile("|".join((expr,) if isinstance(expr, str) else expr))


logger = logging.getLogger(__name__)

# Tagged union for entries in _cl_site_index_map.
# _GlobalSite: (global_shape_idx, None)           — body_pattern was None
# _LocalSite:  (None, [[env0_idx, ...], ...])     — per-world site indices
_GlobalSite = tuple[int, None]
_LocalSite = tuple[None, list[list[int]]]
_SiteEntry = _GlobalSite | _LocalSite


@wp.kernel(enable_backward=False)
def _or_reset_masks_from_mask(
    env_mask: wp.array(dtype=wp.bool),
    articulation_ids: wp.array2d(dtype=int),
    world_mask: wp.array(dtype=wp.bool),
    fk_mask: wp.array(dtype=wp.bool),
):
    """OR env_mask into world_mask and set corresponding articulation bits in fk_mask."""
    world, arti = wp.tid()
    if env_mask[world]:
        world_mask[world] = True
        fk_mask[articulation_ids[world, arti]] = True


@wp.kernel(enable_backward=False)
def _scatter_reset_masks_from_ids(
    env_ids: wp.array(dtype=Any),
    articulation_ids: wp.array2d(dtype=int),
    world_mask: wp.array(dtype=wp.bool),
    fk_mask: wp.array(dtype=wp.bool),
):
    """Scatter-set world_mask and fk_mask from sparse env_ids."""
    i, arti = wp.tid()
    world = wp.int32(env_ids[i])
    world_mask[world] = True
    fk_mask[articulation_ids[world, arti]] = True


_SCATTER_RESET_MASKS_FROM_IDS_DISPATCHER = IndexKernelDispatcher(_scatter_reset_masks_from_ids, ("env_ids",))


def _scatter_reset_masks_from_ids_kernel(env_ids: wp.array | torch.Tensor) -> wp.Kernel:
    """Select the reset-mask writer matching the environment selector dtype."""
    return _SCATTER_RESET_MASKS_FROM_IDS_DISPATCHER.select(env_ids)


@wp.kernel(enable_backward=False)
def _or_world_reset_mask_from_mask(env_mask: wp.array(dtype=wp.bool), world_mask: wp.array(dtype=wp.bool)):
    """Mark masked worlds for solver reset without requesting FK."""
    world = wp.tid()
    if env_mask[world]:
        world_mask[world] = True


@wp.kernel(enable_backward=False)
def _scatter_world_reset_mask_from_ids(env_ids: wp.array(dtype=wp.int32), world_mask: wp.array(dtype=wp.bool)):
    """Mark selected worlds for solver reset without requesting FK."""
    world_mask[env_ids[wp.tid()]] = True


class NewtonSceneDataBackend(SceneDataBackend):
    """Scene data backend that reads rigid body transforms from Newton's simulation state.

    The backend reads ``body_q`` (an array of :class:`wp.transformf`) from
    Newton's current state and exposes its native pointer plus the clone-plan-to-native
    index map as :class:`SceneDataFormat.IndexedTransform`.
    """

    def __init__(self, manager: NewtonManager):
        self._manager = manager
        self._transform_publication = SceneDataPublication(SceneDataFormat.IndexedTransform(), dirty=True)
        self._particle_publication = SceneDataPublication(SceneDataFormat.Points(), dirty=True)
        self._cable_publication = SceneDataPublication(SceneDataFormat.CablePoints(), dirty=True)

    def setup(self, model: Model, plan: ClonePlan, device: str) -> None:
        """Bind canonical clone-plan bodies and native state pointers once."""
        native_indices: dict[str, int] = {}
        for index, label in enumerate(model.body_label):
            if label is not None:
                if label in native_indices:
                    raise RuntimeError(f"Newton model contains duplicate body label {label!r}.")
                native_indices[label] = index
        try:
            indices = [native_indices[path] for path in plan.iter_rigid_body_paths()]
        except KeyError as exc:
            raise RuntimeError(f"Clone-plan body {exc.args[0]!r} is absent from the Newton model.") from exc
        self._transform_publication.data.source_indices = wp.array(indices, dtype=wp.int32, device=device)
        state = self._manager._newton._state_0
        self._transform_publication.data.transforms = state.body_q
        self._particle_publication.data.points = state.particle_q
        self._cable_publication.data.body_q = state.body_q

    @property
    def transform_publication(self) -> SceneDataPublication:
        """Return the current Newton rigid-body pointer and dirty latch."""
        return self._transform_publication

    @property
    def point_publications(self) -> dict[str, SceneDataPublication]:
        """Publish particle pointers and native cable state as independent dirty streams."""
        return (
            {"points": self._particle_publication, "cables": self._cable_publication}
            if self._cable_publication.data.segment_counts is not None
            else {"points": self._particle_publication}
        )


def _eval_fk_unbound(world_reset_mask: wp.array | None, fk_mask: wp.array | None) -> None:
    """Default :attr:`NewtonManager._eval_fk` value before a solver is initialized.

    Raises so a stray ``forward()`` / ``step()`` before ``initialize_solver()`` fails loudly
    instead of silently running a wrong (or no) FK.
    """
    raise RuntimeError(
        "FK hook is not bound. NewtonManager.initialize_solver() must run "
        "(via reset()) before forward()/step() can run forward kinematics."
    )


def _reset_solver_internals_unbound(world_mask: wp.array | None) -> None:
    """Default reset-hook delegate value before a solver is initialized."""
    raise RuntimeError(
        "Solver reset hook is not bound. NewtonManager.initialize_solver() must run "
        "(via reset()) before forward()/step() can reset solver internals."
    )


class NewtonManager(PhysicsManager):
    """Abstract Newton physics manager for Isaac Lab.

    Owns solver lifecycle, collision handling, sensors, and CUDA-graph orchestration for one
    simulation context. Its registry-owned :class:`NewtonReplicateContext` owns all native and
    clone state.
    Concrete subclasses (one per solver) implement :meth:`_build_solver` and
    may extend :meth:`_initialize_contacts`, :meth:`_prepare_builder_for_finalize`,
    :meth:`_step_solver`, :meth:`_supports_cuda_graph_capture`,
    :meth:`_reset_solver_internals`,
    :meth:`_solver_specific_clear`, :meth:`_check_solver_status`, and
    :meth:`_log_solver_debug`.

    Concrete :class:`NewtonSolverCfg` subclasses declare their matching manager
    through :attr:`~isaaclab.physics.PhysicsCfg.class_type`.

    Lifecycle: construction -> clone -> ``reset() -> step()`` (repeated) ``-> close()``.

    Physics, renderers, and visualizers independently resolve the same
    :class:`NewtonReplicateContext` registry key. Construction order therefore does not change
    which object owns the model and state.
    """

    _newton: NewtonReplicateContext | None
    _builder_attribute_solvers: ClassVar[tuple[type[SolverBase], ...]] = ()
    _solver_dt: float
    _num_substeps: int
    _decimation: int
    _collision_decimation: int
    _deterministic_mode: wp.DeterministicMode

    _solver: SolverBase | None
    _use_single_state: bool | None
    """Use only one state for both input and output for solver stepping. Requires solver support."""

    # Physics settings
    _gravity_vector: tuple[float, float, float]

    # Collision and contacts
    _needs_collision_pipeline: bool
    _collision_pipeline: CollisionPipeline | None
    _collision_cfg: NewtonCollisionPipelineCfg | None
    _newton_contact_sensors: dict  # Maps sensor_key to NewtonContactSensor
    _newton_frame_transform_sensors: list  # List of SensorFrameTransform
    _newton_imu_sensors: list  # List of NewtonSensorIMU
    _report_contacts: bool
    _supports_contact_sensors: bool

    # Per-world reset masks (allocated in start_simulation, consumed in step/forward).
    # Newton reserves the final slot for global entities in world -1.
    _world_reset_mask: wp.array | None  # (num_envs + 1,) wp.bool
    _fk_reset_mask: wp.array | None  # (articulation_count,) wp.bool — for eval_fk(mask=...)
    _reconciliation_pending: bool
    _reconciliation_replayable: bool
    # Solver-specialized FK delegate. Bound in initialize_solver() to the active subclass's choice of FK implementation.
    _eval_fk: Callable[[wp.array | None, wp.array | None], None]
    # Solver-specialized reset delegate. Like _eval_fk, this must dispatch correctly through the base manager.
    _reset_solver_internals_delegate: Callable[[wp.array | None], None]

    # Newton actuator adapter (owns actuators and double-buffered states)
    _adapter: NewtonActuatorAdapter | None
    # In-graph hooks invoked after the actuator step and before the solver
    # substeps, in registration order. Multiple articulations register their
    # implicit-DOF telemetry / FF-routing kernels here.
    _post_actuator_callbacks: list[Callable[[], None]]
    # In-graph hooks invoked after the last solver substep and before sensors,
    # in registration order. Articulations with non-identity ordering register
    # their backend-to-user state republish kernels here so the reorders are
    # recorded into every captured graph.
    _post_step_callbacks: list[Callable[[], None]]

    # CUDA graphing
    _graph: wp.Graph | None
    _graph_capture_pending: bool
    # Scene data backend
    _scene_data_backend: NewtonSceneDataBackend | None

    def __init__(self, cfg: NewtonSolverCfg):
        super().__init__(cfg)
        self._newton = None
        self._solver_dt = 1.0 / 200.0
        self._num_substeps = 1
        self._collision_decimation = 0
        self._gravity_vector = (0.0, 0.0, -9.81)
        self.clear()

    def _bind_context(self, sim_context: SimulationContext) -> None:
        """Bind the manager and register its shared clone resource.

        Args:
            sim_context: Parent simulation context.
        """
        super()._bind_context(sim_context)
        self._newton = sim_context.get_or_create_backend(NewtonReplicateContext, sim_context, clone_role="physics")
        nested_solvers = tuple(
            solver_type
            for entry in getattr(self.cfg, "entries", ())
            for solver_type in getattr(entry.solver_cfg.class_type, "_builder_attribute_solvers", ())
        )
        self._newton.bind_physics(self.cfg, tuple(dict.fromkeys((*self._builder_attribute_solvers, *nested_solvers))))

        # Newton-specific setup: get gravity from SimulationCfg (not physics manager cfg)
        sim = self._sim
        if sim is not None:
            self._gravity_vector = sim.cfg.gravity  # type: ignore[union-attr]

        self._scene_data_backend = NewtonSceneDataBackend(self)

    def create_builder(self, up_axis: str | None = None, **kwargs) -> ModelBuilder:
        """Create a builder configured by this manager on its native resource."""
        return self._newton.create_builder(up_axis, **kwargs)

    def set_builder(self, builder: ModelBuilder) -> None:
        """Set the builder owned by this manager's native resource."""
        self._newton.set_builder(builder)

    def get_model(self) -> Model | None:
        """Return the shared native model."""
        return self._newton.get_model()

    def get_state_0(self) -> State | None:
        """Return the shared current state."""
        return self._newton.get_state_0()

    def reset(self, soft: bool = False) -> None:
        """Initialize the cloned model once and preserve its native identity across resets.

        Args:
            soft: Retained for the backend-neutral reset interface.
        """
        if self._newton._model is None:
            self.start_simulation()
            self.initialize_solver()

    def _eval_fk_impl(self, world_reset_mask: wp.array | None, fk_mask: wp.array | None) -> None:
        """Update body states from joint coordinates.

        Solver-specialized FK implementation. The base implementation runs Newton's generic
        ``eval_fk`` over the articulations selected by ``fk_mask``. Subclasses may override
        this method to use a solver-specific FK.

        Args:
            world_reset_mask: Per-world mask of environments to reset (``None`` means all).
                Unused by the base implementation; consumed by solver-specific overrides such as
                :meth:`NewtonKaminoManager._eval_fk_impl`.
            fk_mask: Per-articulation mask of articulations to update (``None`` means all).
        """
        eval_fk(
            self._newton._model,
            self._newton._state_0.joint_q,
            self._newton._state_0.joint_qd,
            self._newton._state_0,
            fk_mask,
        )

    def forward(self) -> None:
        """Update articulation kinematics without stepping physics.

        Update body poses from joint coordinates via the solver-specialized FK delegate
        (:attr:`_eval_fk`, bound to the active subclass's :meth:`_eval_fk_impl` in
        :meth:`initialize_solver`). Only the articulations flagged dirty in
        :attr:`_fk_reset_mask` and :attr:`_world_reset_mask` (see :meth:`invalidate_fk`) are
        updated. The masks are consumed (zeroed) afterwards so the next :meth:`step` does not
        redundantly re-solve them.

        The delegate (rather than a direct ``self._eval_fk_impl`` call) is required because the
        composition root invokes ``self.forward()`` through the base manager; the bound delegate
        dispatches to the concrete subclass override.
        """
        self._reconcile_state()
        for callback in self._post_step_callbacks:
            callback()
        self._mark_transforms_dirty()

    def _reconcile_state(self) -> None:
        """Reconcile authored state without republishing user-order shadows."""
        if not (self._reconciliation_pending or self._reconciliation_replayable):
            return
        self._reset_solver_internals_delegate(self._world_reset_mask)
        self._eval_fk(self._world_reset_mask, self._fk_reset_mask)
        if self._fk_reset_mask is not None:
            self._fk_reset_mask.zero_()
        if self._world_reset_mask is not None:
            self._world_reset_mask.zero_()
        self._reconciliation_pending = False

    def _mark_transforms_dirty(self) -> None:
        """Flag that rigid-body transforms and native cable state have changed."""
        self._scene_data_backend._transform_publication.dirty = True
        self._scene_data_backend._cable_publication.dirty = True

    def _mark_particles_dirty(self) -> None:
        """Flag that particle positions have changed."""
        self._scene_data_backend._particle_publication.dirty = True

    def _mark_state_dirty(self) -> None:
        """Flag that all scene-data publications have changed.

        Convenience method that marks both transforms and particles dirty.
        Called after stepping.
        """
        self._mark_transforms_dirty()
        self._mark_particles_dirty()

    def step(self) -> None:
        """Step the physics simulation.

        The stepping logic follows one of two paths depending on whether
        **all** actuators are CUDA-graph-safe:

        **All-graphable path** (:meth:`_simulate_full`):

        Actuators and solver substeps are captured together in a single
        CUDA graph containing the full
        ``decimation x (actuators + solver substeps)`` loop.

        **Eager-actuator path** (fallback, some actuators not graph-safe):

        Actuators are stepped eagerly on the CPU timeline (outside the
        graph), then a graph containing only the solver substeps is
        launched via :meth:`_simulate_physics_only`.

        In both paths the sequence within one physics step is::

            zero actuated DOFs in control.joint_f
            -> actuator.step (computes effort, writes to control.joint_f)
            -> solver.step x num_substeps (integrates, reads control.joint_f)
            -> sensors.update
        """
        sim = self._sim
        if sim is None or not sim.is_playing():
            return

        # Notify solver of model changes
        if self._newton._model_changes:
            with wp.ScopedDevice(self._device):
                for change in self._newton._model_changes:
                    self._solver.notify_model_changed(change)
                self._newton._model_changes.clear()

        # Lazy CUDA graph capture
        cfg = self._cfg
        device = self._device
        capture_pending = (
            self._graph_capture_pending and cfg is not None and cfg.use_cuda_graph and "cuda" in device  # type: ignore[union-attr]
        )
        self._reconcile_state()
        if capture_pending:
            self._capture_cuda_graph()

        physics_dt = self._solver_dt * self._num_substeps
        use_graph = cfg is not None and cfg.use_cuda_graph and "cuda" in device  # type: ignore[union-attr]
        if use_graph and self._graph is None:
            raise RuntimeError("Newton CUDA graph execution was selected, but capture produced no graph.")

        if self._is_all_graphable():
            # --- All actuators are graph-safe: actuators + solver in one graph ---
            if use_graph:
                wp.capture_launch(self._graph)
            else:
                with wp.ScopedDevice(device):
                    self._simulate_full()
            self._sim_time += physics_dt * self._decimation
        else:
            # --- Some actuators not graph-safe: step them eagerly, graph solver only ---
            if self._adapter is not None:
                self._adapter.step(self._newton._state_0, self._newton._control, physics_dt)
            for cb in self._post_actuator_callbacks:
                cb()

            if use_graph:
                wp.capture_launch(self._graph)
            else:
                with wp.ScopedDevice(device):
                    self._simulate_physics_only()
            self._sim_time += physics_dt

        self._mark_state_dirty()

        self._check_solver_status()

        # Launch solver-specific debug logging after stepping.
        self._log_solver_debug()

    def close(self) -> None:
        """Clean up Newton physics resources."""
        super().close()
        self.clear()

    def get_scene_data_backend(self) -> SceneDataBackend | None:
        """Return the SceneDataBackend for the SceneDataProvider."""
        return self._scene_data_backend

    def get_physics_sim_view(self) -> list:
        """Return the articulation views owned by the shared Newton resource."""
        return list(self._newton._articulation_views.values())

    def clear(self):
        """Clear manager-owned Newton state (callbacks cleared by super().close())."""
        self._solver = None
        self._use_single_state = None
        self._needs_collision_pipeline = False
        self._deterministic_mode = wp.DeterministicMode.NOT_GUARANTEED
        self._eval_fk = _eval_fk_unbound
        self._reset_solver_internals_delegate = _reset_solver_internals_unbound
        self._collision_pipeline = None
        self._collision_cfg = None
        self._newton_contact_sensors = {}
        self._newton_frame_transform_sensors = []
        self._newton_imu_sensors = []
        self._report_contacts = False
        self._supports_contact_sensors = True
        self._adapter = None
        self._post_actuator_callbacks = []
        self._post_step_callbacks = []
        # Set by an articulation that took the ``use_newton_actuators=True``
        # branch in ``_process_actuators_cfg``.  Together with the adapter
        # check, this gates whether the decimation loop can be captured into
        # a CUDA graph (see :meth:`_is_all_graphable`).
        self._use_newton_actuators_active = False
        self._decimation = 1
        # Per-world reset masks
        self._world_reset_mask = None
        self._fk_reset_mask = None
        self._reconciliation_pending = False
        self._reconciliation_replayable = False
        self._graph = None
        self._graph_capture_pending = False
        self._scene_data_backend = None
        self._solver_specific_clear()

    def _prepare_builder_for_finalize(self, builder: ModelBuilder) -> None:
        """Subclass hook to normalize *builder* before model finalization.

        Override in solver subclasses that need to adapt imported or replicated
        builder data before :meth:`ModelBuilder.finalize` allocates model arrays.
        The default implementation is a no-op.
        """

    def add_model_change(self, change: ModelFlags) -> None:
        """Register a model change to notify the solver."""
        self._newton.add_model_change(change)

    def invalidate_fk(
        self,
        env_mask: wp.array | None = None,
        env_ids: wp.array | None = None,
        articulation_ids: wp.array | None = None,
    ) -> None:
        """Mark environments as needing FK recomputation and solver reset.

        Called by asset write methods that modify joint coordinates or root
        transforms. The masks are consumed by the next forward or physics-step boundary.

        Args:
            env_mask: Boolean mask of dirtied environments. Shape ``(num_envs,)``.
                Used by ``_mask`` write methods.
            env_ids: Integer indices of dirtied environments.
                Used by ``_index`` write methods.
            articulation_ids: Mapping from ``(world, arti)`` to model articulation
                index. Shape ``(world_count, count_per_world)``. Obtained from
                ``ArticulationView.articulation_ids``.
        """
        if self._world_reset_mask is None or self._fk_reset_mask is None:
            return

        if articulation_ids is not None and env_mask is not None:
            wp.launch(
                _or_reset_masks_from_mask,
                dim=articulation_ids.shape,
                inputs=[env_mask, articulation_ids],
                outputs=[self._world_reset_mask, self._fk_reset_mask],
                device=self._device,
            )
        elif articulation_ids is not None and env_ids is not None:
            wp.launch(
                _scatter_reset_masks_from_ids_kernel(env_ids),
                dim=(env_ids.shape[0], articulation_ids.shape[1]),
                inputs=[env_ids, articulation_ids],
                outputs=[self._world_reset_mask, self._fk_reset_mask],
                device=self._device,
            )
        else:
            # Fallback: no topology info — mark everything dirty
            self._world_reset_mask[: self._newton._model.world_count].fill_(True)
            self._fk_reset_mask.fill_(True)
        self._reconciliation_pending = True
        self._reconciliation_replayable |= bool(wp.get_device(self._device).is_capturing)

    def invalidate_body_state(
        self,
        env_ids: wp.array(dtype=wp.int32) | None = None,
        env_mask: wp.array(dtype=wp.bool) | None = None,
    ) -> None:
        """Mark selected maximal-coordinate body state as changed without requesting FK.

        Args:
            env_ids: Integer indices of dirtied environments. Used by index write methods.
            env_mask: Boolean mask of dirtied environments. Used by mask write methods.
        """
        if self._world_reset_mask is None:
            return
        if env_mask is not None:
            wp.launch(
                _or_world_reset_mask_from_mask,
                dim=env_mask.shape[0],
                inputs=[env_mask],
                outputs=[self._world_reset_mask],
                device=self._device,
            )
        elif env_ids is not None:
            wp.launch(
                _scatter_world_reset_mask_from_ids,
                dim=env_ids.shape[0],
                inputs=[env_ids],
                outputs=[self._world_reset_mask],
                device=self._device,
            )
        else:
            self._world_reset_mask[: self._newton._model.world_count].fill_(True)
        self._reconciliation_pending = True
        self._reconciliation_replayable |= bool(wp.get_device(self._device).is_capturing)

    def _drain_stale_cuda_error(self) -> None:
        """Clear a stale CUDA error latched on the device before (re)initialization.

        Warp 1.15 leaves the per-thread CUDA error uncleared when
        ``wp_free_device_async`` fails to add a graph memory free node while a
        capture is still registered (its "capture ended" sibling branch clears
        the identical error as benign), and the next Warp array copy then
        surfaces that stale error as its own failure, aborting simulation
        initialization. Draining here keeps a prior simulation lifecycle's
        latched error from poisoning this one. Remove once the upstream Warp
        fix lands.
        """
        device = wp.get_device(str(self._device))
        if not device.is_cuda:
            return
        # Private Warp API: the drain primitives are not exposed publicly; getting the
        # device above guarantees the runtime is initialized. Guard the whole private
        # interaction so a future Warp internals reshuffle degrades to a skipped drain
        # rather than hard-failing simulation start.
        try:
            from warp._src.context import runtime as _wp_runtime

            core = _wp_runtime.core
            # wp_cuda_context_check drains via cudaGetLastError() but returns the
            # post-sync error state (0 once a non-sticky error was drained), and its
            # internal check_cuda() prints the drained error verbatim to stderr;
            # suppress the print and diff Warp's error buffer to report the drain.
            before = core.wp_get_error_string()
            was_enabled = bool(core.wp_is_error_output_enabled())
            core.wp_set_error_output_enabled(0)
            try:
                persistent = core.wp_cuda_context_check(device.context)
            finally:
                core.wp_set_error_output_enabled(1 if was_enabled else 0)
            after = core.wp_get_error_string()
        except (ImportError, AttributeError) as exc:
            logger.warning("Skipping stale CUDA error drain; Warp internals unavailable: %s", exc)
            return

        if persistent != 0:
            logger.error(
                "CUDA error %d persists after drain; the device context is likely unrecoverable: %s",
                persistent,
                after.decode(errors="replace"),
            )
        elif after != before:
            logger.warning(
                "Drained stale CUDA error latched by a prior lifecycle: %s (last Warp error recorded before drain: %s)",
                after.decode(errors="replace"),
                before.decode(errors="replace") or "<none>",
            )

    def start_simulation(self) -> None:
        """Start simulation by finalizing model and initializing state.

        This function finalizes the model and initializes the simulation state.
        Note: Collision pipeline is initialized later in initialize_solver() after
        we determine whether the solver needs external collision detection.
        """
        logger.debug(f"Builder: {self._newton._builder}")

        self._drain_stale_cuda_error()

        if self._newton._builder is None:
            raise RuntimeError("Newton initialization requires clone-plan replication to produce a model builder.")

        logger.info("Dispatching MODEL_INIT callbacks")
        self.dispatch_event(PhysicsEvent.MODEL_INIT)

        if self._newton._cl_pending_sites:
            raise RuntimeError("Newton sites must be registered before clone-plan replication.")

        device = self._device
        logger.info(f"Finalizing model on device: {device}")
        self._newton._builder.up_axis = Axis.from_string(self._newton._up_axis)
        # Forward pending extended attribute requests to builder and clear them
        if self._newton._pending_extended_state_attributes:
            self._newton._builder.request_state_attributes(*self._newton._pending_extended_state_attributes)
            self._newton._active_extended_state_attributes |= self._newton._pending_extended_state_attributes
            self._newton._pending_extended_state_attributes = set()
        self._prepare_builder_for_finalize(self._newton._builder)
        with Timer(name="newton_finalize_builder", msg="Finalize builder took:", activity="Finalizing physics model"):
            self._newton._model = self._newton._builder.finalize(device=device)
            cfg = self._cfg
            if cfg.soft_contact_cfg is not None:
                self._newton._model.soft_contact_ke = float(cfg.soft_contact_cfg.soft_contact_ke)
                self._newton._model.soft_contact_kd = float(cfg.soft_contact_cfg.soft_contact_kd)
                self._newton._model.soft_contact_mu = float(cfg.soft_contact_cfg.soft_contact_mu)
            self._newton._model.set_gravity(self._gravity_vector)
            self._newton._model.num_envs = self._newton._num_envs

        self._newton._state_0 = self._newton._model.state()
        self._newton._control = self._newton._model.control()
        plan = self._sim.get_clone_plan()
        if plan is None or not plan.is_complete:
            raise RuntimeError("Newton initialization requires a completed clone plan.")
        self._scene_data_backend.setup(self._newton._model, plan, device)
        # The initial body-state update from joint coordinates is deferred to the tail of
        # initialize_solver(), where it runs through the solver-specialized FK delegate after the solver is initialized.

        # The single global actuator adapter is built lazily on the first
        # call to ``activate_newton_actuator_path`` from any Newton-fast-path
        # articulation after this point. Assign through the explicit base
        # class so external readers (which import ``NewtonManager`` directly)
        # observe the canonical state regardless of which subclass is active.
        self._adapter = None
        self._use_newton_actuators_active = False

        # Newton's final reset-mask slot selects global entities in world -1.
        # Isaac Lab resets local environments only, so that slot remains false.
        self._world_reset_mask = wp.zeros(self._newton._model.world_count + 1, dtype=wp.bool, device=device)
        self._fk_reset_mask = wp.zeros(self._newton._model.articulation_count, dtype=wp.bool, device=device)

        self._initialize_cable_publication(plan)
        self._mark_state_dirty()

    def _initialize_cable_publication(self, plan: ClonePlan) -> None:
        """Publish native Newton cable pointers with topology declared by the clone plan."""
        bindings = plan.point_bindings("cables")
        if not bindings:
            return
        declared = {binding.path: binding.source_count - 1 for binding in bindings}
        cable_shapes: dict[str, dict[int, int]] = {}
        for shape_id, label in enumerate(self._newton._model.shape_label):
            if label is None:
                continue
            prim_path, separator, suffix = label.rpartition("_edge_capsule_")
            if not separator or not suffix.isdigit():
                continue
            if prim_path not in declared:
                raise RuntimeError(f"Newton model contains undeclared cable geometry {prim_path!r}.")
            segment = int(suffix)
            segments = cable_shapes.setdefault(prim_path, {})
            if segment in segments:
                raise RuntimeError(f"Cable publication requires one Newton shape labeled {label}.")
            segments[segment] = shape_id

        shape_ids: list[int] = []
        shape_offsets: list[int] = []
        segment_counts: list[int] = []
        for binding in bindings:
            prim_path = binding.path
            segment_count = binding.source_count - 1
            segments = cable_shapes.get(prim_path)
            if segments is None:
                raise RuntimeError(f"Clone-plan cable {prim_path!r} has no Newton segment shapes.")
            if set(segments) != set(range(segment_count)):
                raise RuntimeError(f"Cable publication requires {segment_count} ordered segment shapes.")
            segment_shape_ids = [segments[segment] for segment in range(segment_count)]
            shape_offsets.append(len(shape_ids))
            segment_counts.append(segment_count)
            shape_ids.extend(segment_shape_ids)

        publication = self._scene_data_backend._cable_publication
        if not shape_ids:
            publication.data = SceneDataFormat.CablePoints()
            return
        model = self._newton._model
        state = self._newton._state_0
        device = state.body_q.device
        publication.data = SceneDataFormat.CablePoints(
            body_q=state.body_q,
            shape_body=model.shape_body,
            shape_transform=model.shape_transform,
            shape_scale=model.shape_scale,
            shape_ids=wp.array(shape_ids, dtype=wp.int32, device=device),
            shape_offsets=wp.array(shape_offsets, dtype=wp.int32, device=device),
            segment_counts=wp.array(segment_counts, dtype=wp.int32, device=device),
            binding_ids=wp.array(range(len(bindings)), dtype=wp.int32, device=device),
        )
        publication.dirty = True

    def _initialize_contacts(self) -> None:
        """Initialize contacts using Newton's :class:`CollisionPipeline`.

        This default implementation handles solvers that rely on Newton's
        unified collision pipeline (XPBD, Featherstone, and MuJoCo with
        ``use_mujoco_contacts=False``).  Solver subclasses with internal
        contact handling (e.g. :class:`NewtonMJWarpManager` when
        ``use_mujoco_contacts=True``) override this method to allocate a
        :class:`Contacts` object sized to the solver's internal contact buffer.
        """
        if not self._needs_collision_pipeline:
            return
        pipeline_args = {"broad_phase": "explicit"}
        if self._collision_cfg is not None:
            pipeline_args = self._collision_cfg.to_dict()
            hydro_cfg = pipeline_args.pop("sdf_hydroelastic_config", None)
            if hydro_cfg is not None:
                pipeline_args["sdf_hydroelastic_config"] = HydroelasticSDF.Config(**hydro_cfg)
        pipeline_args["deterministic"] = self._deterministic_mode != wp.DeterministicMode.NOT_GUARANTEED
        if self._collision_pipeline is None:
            self._collision_pipeline = CollisionPipeline(self._newton._model, **pipeline_args)
        if self._newton._contacts is None:
            self._newton._contacts = self._collision_pipeline.contacts()
            # Grow the collision-pipeline contact buffer to the solver's max when the
            # solver (e.g. MuJoCo/mujoco_warp) requires more contacts than the pipeline
            # auto-estimate. Without this, the RSL-RL sensor path (use_mujoco_contacts=
            # False) sizes _contacts from the pipeline alone and solver.update_contacts()
            # raises when naconmax (nconmax * num_envs) exceeds rigid_contact_max.
            # Mirrors the mjwarp_manager.py override for the use_mujoco_contacts=True path.
            _solver = self._solver
            if _solver is not None and hasattr(_solver, "get_max_contact_count"):
                _need = _solver.get_max_contact_count()
                if _need > self._newton._contacts.rigid_contact_max:
                    if self._deterministic_mode != wp.DeterministicMode.NOT_GUARANTEED:
                        # In deterministic mode, CollisionPipeline sizes _sort_key_array from rigid_contact_max at
                        # construction. Rebuild so the sort and contact buffers retain matching capacity; replacing
                        # Contacts alone would leave the sorting buffer undersized.
                        pipeline_args["rigid_contact_max"] = _need
                        self._collision_pipeline = CollisionPipeline(self._newton._model, **pipeline_args)
                        self._newton._contacts = self._collision_pipeline.contacts()
                    else:
                        self._newton._contacts = Contacts(
                            rigid_contact_max=_need,
                            soft_contact_max=0,
                            device=self._device,
                            requested_attributes=self._newton._model.get_requested_contact_attributes(),
                        )

    # ----- Solver construction (subclass contract) ------------------------

    def _create_solver(self, model: Model, solver_cfg) -> SolverBase:
        """Construct a solver without changing the active manager state.

        Solver-manager subclasses override this hook so nested consumers can
        reuse their typed construction logic through ``solver_cfg.class_type``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement solver construction.")

    @abstractmethod
    def _build_solver(self, model: Model, solver_cfg) -> None:
        """Construct the solver this manager owns and assign it onto the base class.

        Subclasses must populate the canonical :class:`NewtonManager` slots:

        * :attr:`self._solver` — the constructed :class:`SolverBase`
          instance.
        * :attr:`self._use_single_state` — ``True`` if the solver
          steps in-place on a single :class:`State` (e.g. MuJoCo); ``False``
          if it needs separate input/output states (e.g. XPBD, Featherstone,
          Kamino).
        * :attr:`self._needs_collision_pipeline` — ``True`` if the
          manager owns Newton's :class:`CollisionPipeline` for contact
          generation; ``False`` if the solver runs internal collision
          detection (MuJoCo internal contacts, Kamino with its own detector).
        The solver also records on the shared Newton resource whether it consumes
        external rigid-body forces from :attr:`State.body_f`.

        Args:
            model: Finalized Newton model the solver should run on.
            solver_cfg: The concrete :class:`NewtonSolverCfg`.
        """
        raise NotImplementedError("NewtonManager subclasses must implement _build_solver()")

    def _filter_solver_kwargs(self, solver_cls: type, solver_cfg) -> dict:
        """Return cfg fields that match ``solver_cls.__init__`` parameters.

        Drops keys that the solver constructor doesn't accept (e.g. cfg-only
        metadata like ``class_type``). ``self`` and ``model``
        are always excluded — ``model`` is passed positionally at construction.
        """
        valid = set(inspect.signature(solver_cls.__init__).parameters) - {"self", "model"}
        kwargs = {k: v for k, v in solver_cfg.to_dict().items() if k in valid}
        if "deterministic" in valid:
            kwargs["deterministic"] = self._deterministic_mode
        return kwargs

    def _validate_deterministic_solver_cfg(
        self, solver_cfg: NewtonSolverCfg, deterministic_mode: wp.DeterministicMode
    ) -> None:
        """Validate that a solver can provide the requested determinism guarantee."""
        if deterministic_mode == wp.DeterministicMode.NOT_GUARANTEED:
            return
        solver_cfg_type = type(solver_cfg).__name__
        if not isinstance(solver_cfg, (FeatherstoneSolverCfg, MJWarpSolverCfg, XPBDSolverCfg)):
            raise ValueError(
                f"Newton deterministic mode {deterministic_mode.name} is not supported by {solver_cfg_type}. "
                "Use MJWarp on the GPU, XPBD, or Featherstone, or disable deterministic mode."
            )
        if getattr(solver_cfg, "use_mujoco_cpu", False):
            raise ValueError(
                f"Newton deterministic mode {deterministic_mode.name} is not supported by the MuJoCo CPU backend. "
                "Set MJWarpSolverCfg.use_mujoco_cpu=False or disable deterministic mode."
            )
        if isinstance(solver_cfg, MJWarpSolverCfg) and not solver_cfg.disable_sensors:
            raise ValueError(
                f"Newton deterministic mode {deterministic_mode.name} is not supported while MuJoCo Warp's "
                "internal sensor computation is enabled. Set MJWarpSolverCfg.disable_sensors=True or disable "
                "deterministic mode."
            )
        blocked = self._newton._active_extended_state_attributes & _SENSOR_STAGE_STATE_ATTRIBUTES
        if isinstance(solver_cfg, MJWarpSolverCfg) and blocked:
            sensors = sorted({_SENSORS_BY_STATE_ATTRIBUTE[attr] for attr in blocked})
            raise ValueError(
                f"This task does not support deterministic physics: it uses {' and '.join(sensors)},"
                f" reading {sorted(blocked)}. Those attributes require MuJoCo's disabled sensor stage."
            )

    def _apply_deterministic_request(self, cfg: NewtonSolverCfg) -> wp.DeterministicMode:
        """Translate the backend-neutral determinism request into Newton settings."""
        if getattr(cfg, "use_mujoco_cpu", False):
            if cfg.deterministic or cfg.deterministic_mode != "not_guaranteed":
                logger.info("MuJoCo CPU is already reproducible; Newton deterministic mode is not applied.")
            return wp.DeterministicMode.NOT_GUARANTEED
        if cfg.deterministic_mode != "not_guaranteed":
            return self._resolve_deterministic_mode(cfg.deterministic_mode)
        if not cfg.deterministic:
            return wp.DeterministicMode.NOT_GUARANTEED
        if isinstance(cfg, MJWarpSolverCfg):
            cfg.disable_sensors = True
        return wp.DeterministicMode.RUN_TO_RUN

    @staticmethod
    def _resolve_deterministic_mode(deterministic_mode: str) -> wp.DeterministicMode:
        """Convert a Newton config value to Warp's deterministic-mode enum."""
        return {
            "not_guaranteed": wp.DeterministicMode.NOT_GUARANTEED,
            "run_to_run": wp.DeterministicMode.RUN_TO_RUN,
            "gpu_to_gpu": wp.DeterministicMode.GPU_TO_GPU,
        }[deterministic_mode]

    def _step_solver(
        self, state_0: State, state_1: State, control: Control, contacts: Contacts | None, substep_dt: float
    ) -> None:
        """Run one solver substep.

        Default invokes :attr:`_solver` once.  Subclasses can override to
        batch multiple solvers within a single substep.
        """
        self._solver.step(state_0, state_1, control, contacts, substep_dt)

    def _solver_specific_clear(self) -> None:
        """Solver-specific cleanup hook called from :meth:`clear`.

        Default no-op.  Subclasses override to release sub-solver references
        or other solver-specific resources.
        """

    def _check_solver_status(self) -> None:
        """Raise solver-specific asynchronous failures after stepping.

        Default no-op. Subclasses override when a solver requires a host-side
        status check after CUDA graph replay.
        """

    def _log_solver_debug(self) -> None:
        """Solver-specific debug logging after stepping.

        Default no-op.  Subclasses override to log solver-specific debug info
        (e.g. constraint violations, contact forces, etc.) after stepping.
        """

    def _reset_solver_internals(self, world_mask: wp.array | None) -> None:
        """Clear solver-internal state for environments reset since the last boundary.

        The hook runs immediately before reset masks are consumed by :meth:`step`
        and :meth:`forward`. The base implementation delegates to
        :meth:`SolverBase.reset` with ``flags=0``, preserving the joint state
        authored by Isaac Lab while clearing solver-owned buffers. Solvers with
        no reset implementation are unaffected.

        Args:
            world_mask: Per-world reset mask, or ``None`` when no simulation
                state is available.
        """
        if world_mask is None:
            return
        self._solver.reset(self._newton._state_0, world_mask=world_mask, flags=0)

    # ----- Lifecycle orchestration ----------------------------------------

    def initialize_solver(self) -> None:
        """Initialize the solver and collision pipeline.

        Thin orchestrator: delegates solver construction to
        :meth:`_build_solver` (overridden by each solver subclass), lets
        ``PHYSICS_READY`` consumers declare contact requirements, allocates the
        collision pipeline once, then schedules CUDA graph capture for the first
        :meth:`step` call.

        """
        cfg = self._cfg
        if cfg is None:
            return

        with Timer(name="newton_initialize_solver", msg="Initialize solver took:", activity="Initializing solver"):
            self._num_substeps = cfg.num_substeps  # type: ignore[union-attr]
            self._collision_decimation = cfg.collision_decimation  # type: ignore[union-attr]
            deterministic_mode = self._apply_deterministic_request(cfg)
            self._validate_deterministic_solver_cfg(cfg, deterministic_mode)
            self._deterministic_mode = deterministic_mode
            self._solver_dt = self.get_physics_dt() / self._num_substeps
            self._collision_cfg = cfg.collision_cfg  # type: ignore[union-attr]

            self._build_solver(self._newton._model, cfg)
            if self._solver is None:
                raise RuntimeError(
                    f"{type(self).__name__}._build_solver did not assign self._solver. "
                    "Subclasses of NewtonManager must populate self._solver, "
                    "self._use_single_state, self._needs_collision_pipeline, and "
                    "the shared resource's force-input capability."
                )
            self._newton._state_1 = self._newton._state_0 if self._use_single_state else self._newton._model.state()
        # Bind the solver-specialized FK delegate to the active subclass's _eval_fk_impl so
        # forward()/step() dispatch through the concrete manager selected by SimulationContext.
        self._eval_fk = self._eval_fk_impl
        self._reset_solver_internals_delegate = self._reset_solver_internals

        # Establish the initial kinematically-consistent body state through the
        # solver-specialized FK delegate, now that the solver and the delegate both exist.
        # Runs before graph capture below so it sees a valid body_q.
        self._eval_fk(None, None)
        self._mark_transforms_dirty()
        logger.info("Dispatching PHYSICS_READY callbacks")
        self.dispatch_event(PhysicsEvent.PHYSICS_READY)
        self._initialize_contacts()

        # The graphable actuator path receives its decimation after initialization;
        # set_decimation schedules its capture with the final loop count.
        if not self._use_newton_actuators_active:
            self._schedule_graph_capture()

    def _schedule_graph_capture(self) -> None:
        """Capture a CUDA graph, deferring only for reset-dependent solvers.

        Called by :meth:`initialize_solver` and :meth:`set_decimation`
        whenever the graph needs to be (re-)captured.
        """
        cfg = self._cfg
        device = self._device
        if cfg is None or device is None:
            return

        use_cuda_graph = cfg.use_cuda_graph and "cuda" in device
        if use_cuda_graph and not self._supports_cuda_graph_capture():
            self._graph = None
            self._graph_capture_pending = False
            raise RuntimeError(
                f"{type(self).__name__} does not support CUDA graph capture for the current solver configuration."
            )

        if use_cuda_graph:
            self._graph = None
            self._graph_capture_pending = bool(self._newton._mpm_object_registry)
            if self._graph_capture_pending:
                logger.info("Newton CUDA graph capture deferred until first step()")
            else:
                self._capture_cuda_graph()
        else:
            self._graph = None
            self._graph_capture_pending = False

    def _capture_cuda_graph(self) -> None:
        """Capture the selected simulation path on the simulation device."""
        self._graph_capture_pending = False
        with Timer(name="newton_cuda_graph", msg="CUDA graph took:"):
            simulate = self._simulate_full if self._is_all_graphable() else self._simulate_physics_only
            with _paused_gc(), wp.ScopedCapture(device=self._device, force_module_load=False) as capture:
                simulate()
            self._graph = capture.graph
        if self._graph is None:
            raise RuntimeError("Newton CUDA capture produced no graph.")
        if isinstance(self._solver, SolverKamino):
            wp.capture_launch(self._graph)
        logger.info("Newton CUDA graph captured")

    def _supports_cuda_graph_capture(self) -> bool:
        """Return whether the active solver configuration supports CUDA graph capture."""
        return True

    # ------------------------------------------------------------------
    # Building blocks — used by _simulate_full / _simulate_physics_only
    # ------------------------------------------------------------------

    def _run_solver_substeps(self, contacts) -> None:
        """Run ``num_substeps`` solver iterations, handling double-buffered state swap."""
        collide_every = self._collision_decimation
        # Last substep is skipped: its contact set would only feed the next tick's
        # top-of-loop collide(), not this one.
        collide_mid_loop = collide_every > 0 and self._needs_collision_pipeline and contacts is not None

        if self._use_single_state:
            for i in range(self._num_substeps):
                for callback in self._newton._state_force_callbacks:
                    callback(self._newton._state_0)
                self._step_solver(
                    self._newton._state_0, self._newton._state_0, self._newton._control, contacts, self._solver_dt
                )
                self._newton._state_0.clear_forces()
                if collide_mid_loop and (i + 1) % collide_every == 0 and i + 1 < self._num_substeps:
                    self._collision_pipeline.collide(self._newton._state_0, contacts)
        else:
            cfg = self._cfg
            need_copy_on_last = cfg is not None and self._num_substeps % 2 == 1
            for i in range(self._num_substeps):
                for callback in self._newton._state_force_callbacks:
                    callback(self._newton._state_0)
                self._step_solver(
                    self._newton._state_0, self._newton._state_1, self._newton._control, contacts, self._solver_dt
                )
                if need_copy_on_last and i == self._num_substeps - 1:
                    self._newton._state_0.assign(self._newton._state_1)
                else:
                    self._newton._state_0, self._newton._state_1 = self._newton._state_1, self._newton._state_0
                self._newton._state_0.clear_forces()
                if collide_mid_loop and (i + 1) % collide_every == 0 and i + 1 < self._num_substeps:
                    self._collision_pipeline.collide(self._newton._state_0, contacts)

    def _update_sensors(self, contacts) -> None:
        """Push latest state to all registered Newton sensors."""
        if self._newton_frame_transform_sensors:
            for sensor in self._newton_frame_transform_sensors:
                sensor.update(self._newton._state_0)
        if self._newton_imu_sensors:
            for sensor in self._newton_imu_sensors:
                sensor.update(self._newton._state_0)
        if self._report_contacts:
            eval_contacts = contacts if contacts is not None else self._newton._contacts
            self._solver.update_contacts(eval_contacts, self._newton._state_0)
            for sensor in self._newton_contact_sensors.values():
                sensor.update(self._newton._state_0, eval_contacts)

    # ------------------------------------------------------------------
    # Composite stepping routines
    # ------------------------------------------------------------------

    def _simulate_full(self) -> None:
        """Run ``decimation x (actuators + solver substeps)``, then sensors.

        Works for any decimation count (including 1).  All actuators must be
        graph-safe so the entire loop can be captured as a single CUDA graph.
        """
        physics_dt = self._solver_dt * self._num_substeps
        contacts = self._newton._contacts if self._needs_collision_pipeline else None

        for _ in range(self._decimation):
            if self._needs_collision_pipeline:
                self._collision_pipeline.collide(self._newton._state_0, self._newton._contacts)

            if self._adapter is not None:
                self._adapter.step(self._newton._state_0, self._newton._control, physics_dt)
            for cb in self._post_actuator_callbacks:
                cb()

            self._run_solver_substeps(contacts)

        for cb in self._post_step_callbacks:
            cb()
        self._update_sensors(contacts)

    def _simulate_physics_only(self) -> None:
        """Collision + solver substeps + sensors (no actuators, no USD sync).

        Used when actuators are stepped eagerly outside the graph, or when
        there are no actuators at all.
        """
        if self._needs_collision_pipeline:
            self._collision_pipeline.collide(self._newton._state_0, self._newton._contacts)
            contacts = self._newton._contacts
        else:
            contacts = None

        self._run_solver_substeps(contacts)
        for cb in self._post_step_callbacks:
            cb()
        self._update_sensors(contacts)

    # State accessors (used extensively by articulation/rigid object data)

    def get_solver_dt(self) -> float:
        """Get the solver substep timestep."""
        return self._solver_dt

    def _is_all_graphable(self) -> bool:
        """``True`` when the decimation loop can be captured into a CUDA graph.

        Requires:
          1. An articulation took the ``use_newton_actuators=True`` branch
             (signalled via :meth:`activate_newton_actuator_path`).
          2. Either no actuator adapter was needed (all-implicit) or every
             actuator in the adapter is CUDA-graph-safe.
        """
        if not self._use_newton_actuators_active:
            return False
        return self._adapter is None or self._adapter.is_all_graphable

    def activate_newton_actuator_path(self) -> None:
        """Opt an articulation into the Newton actuator fast path.

        Idempotent — called by every Newton-fast-path articulation's
        ``_process_actuators_cfg``:

        1. Sets :attr:`_use_newton_actuators_active`, which
           :meth:`_is_all_graphable` checks (adapter presence alone
           cannot distinguish the fast path from the standard Lab path).
        2. On first call, builds the single sim-level
           :class:`NewtonActuatorAdapter` over the full flat DOF layout;
           later calls reuse it.
        """
        # Shared state lives on the base class so all readers (including
        # framework code that imports ``NewtonManager`` directly) see the
        # same flag regardless of which solver subclass is active.
        self._use_newton_actuators_active = True

        if self._adapter is not None:
            return
        if self._newton._model is None or not self._newton._model.actuators:
            return
        from isaaclab.actuators.newton import NewtonActuatorAdapter  # noqa: PLC0415

        dofs_per_env = self._newton._model.joint_dof_count // self._newton._num_envs
        self._adapter = NewtonActuatorAdapter(
            actuators=list(self._newton._model.actuators),
            num_envs=self._newton._num_envs,
            num_joints=dofs_per_env,
            dof_offset=0,
            device=self._device,
        )
        self._adapter.finalize(self._newton._control)

    def register_post_actuator_callback(self, callback: Callable[[], None]) -> None:
        """Append a hook to the list invoked after the actuator step on every iteration.

        Each callback runs inside the captured CUDA graph (when
        :meth:`_is_all_graphable` is ``True``) right after
        :meth:`NewtonActuatorAdapter.step` and before the solver substeps,
        so kernel writes to ``state``/``control`` are visible to the
        integrator on the same iteration. Multiple articulations register
        their own implicit-DOF telemetry / FF-routing kernels here; all
        registered callbacks fire in registration order each step.
        """
        self._post_actuator_callbacks.append(callback)

    def register_post_step_callback(self, callback: Callable[[], None]) -> None:
        """Append a hook to the list invoked after the last solver substep on every step.

        Each callback runs inside the stepped (and, when
        :meth:`_is_all_graphable` is ``True``, captured) region right after the
        final solver substep of the decimation loop and before
        :meth:`_update_sensors`, so the launches it issues are recorded into
        every captured CUDA graph and replayed on each tick. The hook fires
        exactly once per :meth:`step` call, reflecting the state after all
        decimation iterations (and their solver substeps) have completed -- not
        once per substep and not once per decimation iteration. Callbacks must be
        graph-safe (fixed shapes, no host branching on device data) and must be
        registered before capture. Articulations with non-identity ordering
        register their backend-to-user state republish here; all registered
        callbacks fire in registration order each step.
        """
        self._post_step_callbacks.append(callback)

    def unregister_post_step_callback(self, callback: Callable[[], None]) -> None:
        """Remove a previously registered post-step callback.

        Symmetric to :meth:`register_post_step_callback`, this lets an
        articulation deregister its republish hook when its callbacks are
        cleared so the bound method does not linger after the articulation is
        gone. Removing a callback that was never
        registered (or was already removed) is a safe no-op, matching the
        tolerant deregistration of other handles.
        """
        with contextlib.suppress(ValueError):
            self._post_step_callbacks.remove(callback)

    def set_decimation(self, decimation: int) -> None:
        """Set the decimation count and re-capture the CUDA graph.

        When all actuators are graphable the entire decimation loop
        (actuators + solver substeps, repeated *decimation* times)
        is captured as a single CUDA graph.

        If a CUDA graph was previously captured, it is re-captured on the next
        :meth:`step` with the new decimation count.
        """
        self._decimation = max(1, decimation)
        if self._is_all_graphable():
            self._schedule_graph_capture()

    def handles_decimation(self) -> bool:
        """``True`` when :meth:`step` executes the full decimation loop internally.

        This is the case when all Newton actuators are CUDA-graph-safe.
        The full decimation loop (including the trivial ``decimation=1`` case)
        is folded into a single :meth:`step` call.
        """
        return self._is_all_graphable()

    def add_contact_sensor(
        self,
        body_names_expr: str | list[str] | None = None,
        shape_names_expr: str | list[str] | None = None,
        contact_partners_body_expr: str | list[str] | None = None,
        contact_partners_shape_expr: str | list[str] | None = None,
        verbose: bool = False,
    ) -> tuple[str | list[str] | None, str | list[str] | None, str | list[str] | None, str | list[str] | None]:
        """Add a contact sensor for reporting contacts between bodies/shapes.

        Compiles the Isaac Lab regular expressions and delegates to
        :class:`newton.sensors.SensorContact`, which full-matches compiled patterns
        against model labels.

        Args:
            body_names_expr: Expression for body names to sense.
            shape_names_expr: Expression for shape names to sense.
            contact_partners_body_expr: Expression for contact partner body names.
            contact_partners_shape_expr: Expression for contact partner shape names.
            verbose: Print verbose information.
        """
        if not self._supports_contact_sensors:
            raise NotImplementedError(
                "Newton contact sensors are not yet supported by the active coupled solver because its "
                "contact forces live in per-entry buffers."
            )
        if body_names_expr is None and shape_names_expr is None:
            raise ValueError("At least one of body_names_expr or shape_names_expr must be provided")
        if body_names_expr is not None and shape_names_expr is not None:
            raise ValueError("Only one of body_names_expr or shape_names_expr must be provided")
        if contact_partners_body_expr is not None and contact_partners_shape_expr is not None:
            raise ValueError("Only one of contact_partners_body_expr or contact_partners_shape_expr must be provided")

        sensor_target = body_names_expr or shape_names_expr
        partner_filter = contact_partners_body_expr or contact_partners_shape_expr or "all bodies/shapes"
        logger.info(f"Adding contact sensor for {sensor_target} with filter {partner_filter}")

        def _hashable_key(x):
            return tuple(x) if isinstance(x, list) else x

        sensor_key = (
            _hashable_key(body_names_expr),
            _hashable_key(shape_names_expr),
            _hashable_key(contact_partners_body_expr),
            _hashable_key(contact_partners_shape_expr),
        )

        with Timer(name="newton_contact_sensor", msg="Contact sensor construction took:"):
            sensor = NewtonContactSensor(
                self._newton._model,
                sensing_bodies=_compile_label_pattern(body_names_expr),
                sensing_shapes=_compile_label_pattern(shape_names_expr),
                counterpart_bodies=_compile_label_pattern(contact_partners_body_expr),
                counterpart_shapes=_compile_label_pattern(contact_partners_shape_expr),
                measure_total=True,
                verbose=verbose,
            )

        self._newton_contact_sensors[sensor_key] = sensor
        self._report_contacts = True

        return sensor_key

    def add_frame_transform_sensor(self, shapes: list[int], reference_sites: list[int]) -> int:
        """Add a frame transform sensor for measuring relative transforms.

        Creates a :class:`SensorFrameTransform` from pre-resolved shape and reference
        site indices, appends it to the internal list, and returns its index.

        Args:
            shapes: Ordered list of shape indices to measure.
            reference_sites: 1:1 list of reference site indices (same length as shapes).

        Returns:
            Index of the newly created sensor in :attr:`_newton_frame_transform_sensors`.
        """
        sensor = SensorFrameTransform(
            self._newton._model,
            shapes=shapes,
            reference_sites=reference_sites,
        )
        idx = len(self._newton_frame_transform_sensors)
        self._newton_frame_transform_sensors.append(sensor)
        logger.info(f"Added frame transform sensor (index={idx}, shapes={len(shapes)})")
        return idx

    def add_imu_sensor(self, sites: list[int]) -> int:
        """Add an IMU sensor for measuring acceleration and angular velocity at sites.

        Creates a ``newton.sensors.SensorIMU`` from pre-resolved site indices,
        appends it to the internal list, and returns its index.

        Args:
            sites: Ordered list of site indices (one per environment).

        Returns:
            Index of the newly created sensor in the internal IMU sensor list.
        """
        if self._newton._model is None:
            raise RuntimeError("add_imu_sensor called before model finalization (start_simulation).")
        sensor = NewtonSensorIMU(
            self._newton._model,
            sites=sites,
            request_state_attributes=False,  # Already requested via NewtonManager
        )
        idx = len(self._newton_imu_sensors)
        self._newton_imu_sensors.append(sensor)
        logger.info(f"Added IMU sensor (index={idx}, sites={len(sites)})")
        return idx
