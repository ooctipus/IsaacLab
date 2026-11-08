# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton physics manager for Isaac Lab."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import warp as wp
from newton import Axis, ModelBuilder, ModelFlags, State
from newton.selection import ArticulationView
from newton.sensors import SensorContact, SensorIMU
from newton.solvers import SolverMuJoCo

from pxr import UsdGeom

import isaaclab.cloner as cloner
from isaaclab.physics import CallbackHandle, PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend, SceneDataProvider
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.timer import Timer
from isaaclab.utils.version import has_kit

from isaaclab_newton.physics.newton_backend_cfg import NewtonBackendCfg
from isaaclab_newton.physics.newton_builder_cfg import NewtonBuilderCfg
from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg
from isaaclab_newton.renderers.visual_material import (
    VisualMaterialWriter,
    VisualShapeColorWriter,
)

from .newton_backend import NewtonBackend, NewtonCloneRecord, SiteEntry
from .newton_builder import create_builder
from .newton_scene_data import NewtonSceneDataBackend

if TYPE_CHECKING:
    from isaaclab.assets import BaseArticulation
    from isaaclab.renderers.base_renderer import VisualMaterialBatch

    from ._incident_recorder import PhysicsIncidentRecorder


logger = logging.getLogger(__name__)
_TRIGGER_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class _DebugHalt(Exception):
    """Stop the current solver dispatch after a captured incident."""


def create_newton_builder(cfg: NewtonBuilderCfg) -> ModelBuilder:
    """Construct the shared builder: solver-aware under Newton physics, plain otherwise.

    Args:
        cfg: The selected physics configuration.

    Returns:
        An empty builder.
    """
    if isinstance(cfg.physics_cfg, NewtonCfg):
        return create_builder(cfg.physics_cfg)
    return ModelBuilder()


class NewtonManager(PhysicsManager):
    """Newton physics manager for Isaac Lab.

    Runs one :class:`NewtonBackend` through the physics lifecycle and keeps what survives hard resets: the builder,
    site requests, replication outputs and decimation.

    Lifecycle: ``initialize() -> reset() -> step()`` (repeated) ``-> close()``.
    """

    backend_name = "newton"

    class DebugOperationProvider(Protocol):
        """Solver-specific transient operation recorder."""

        def bind(self, solver: object) -> None: ...
        def snapshot(self) -> object: ...
        def close(self) -> None: ...

    @dataclass(frozen=True, slots=True)
    class DebugTriggerContext:
        """Live state supplied to an incident trigger."""

        model: object
        state: State
        control: object
        solver: object
        contacts: object | None
        collision_pipeline: object | None
        operations: object | None
        sim_time: float
        phase: str
        substep_idx: int | None

    @dataclass(frozen=True, slots=True)
    class DebugTriggerResult:
        """Worlds, or global state, affected by a custom incident."""

        reason: str
        world_ids: tuple[int, ...] = ()
        global_scope: bool = False

    def __init__(self, cfg: NewtonCfg):
        super().__init__(cfg)
        self._cfg: NewtonCfg = cfg
        self.builder: ModelBuilder | None = None
        """Shared construction builder, acquired at :meth:`initialize` and kept across hard resets."""
        self.backend: NewtonBackend | None = None
        """Data bound to the finalized model; replaced on every hard reset."""
        self._incident_recorder: PhysicsIncidentRecorder | None = None
        self._debug_operation_provider: NewtonManager.DebugOperationProvider | None = None
        self._auto_debug_operation_provider = False
        self._pending_operation_provider: NewtonManager.DebugOperationProvider | None = None
        self._debug_incident_triggers: dict[str, Callable] = {}
        self.clear()
        self._scene_data_backend = NewtonSceneDataBackend(self)

    @property
    def shares_materials(self) -> bool:
        """Whether contacts read one material table shared by all environments instead of per-shape values."""
        solver_cfg = self._cfg.solver_cfg
        return solver_cfg.class_type.shares_materials(solver_cfg)

    @classmethod
    def provides_implicit_damping(cls) -> bool:
        # Newton's symplectic integrator has no implicit damping.
        return False

    @classmethod
    def setup_deformable_body(cls, prim: Any, deformable_type: str, sim_mesh_prim: Any, vis_mesh_prim: Any) -> None:
        """Apply Newton's token deformable anchor schemas and sync the visual mesh geometry."""
        sim_mesh_path = sim_mesh_prim.GetPath().pathString
        token = "PhysicsVolumeDeformableSimAPI" if deformable_type == "volume" else "PhysicsSurfaceDeformableSimAPI"
        if not sim_mesh_prim.AddAppliedSchema(token):
            raise RuntimeError(f"Failed to set {deformable_type} deformable sim API on prim '{sim_mesh_path}'.")
        # Newton renders the simulation mesh directly: overwrite the visual mesh geometry until
        # separate visual/simulation meshes are supported.
        vis_mesh = UsdGeom.Mesh(vis_mesh_prim)
        if deformable_type == "volume":
            tet_mesh = UsdGeom.TetMesh(sim_mesh_prim)
            surface_indices = tet_mesh.GetSurfaceFaceVertexIndicesAttr().Get()
            if surface_indices is None or len(surface_indices) == 0:
                raise ValueError(
                    f"Deformable body at '{prim.GetPath().pathString}' has no surface indices on its TetMesh"
                    " prim; cannot sync to visual mesh."
                )
            vis_mesh.GetPointsAttr().Set(tet_mesh.GetPointsAttr().Get())
            vis_mesh.GetFaceVertexIndicesAttr().Set(np.asarray(surface_indices).flatten())
            vis_mesh.GetFaceVertexCountsAttr().Set([3] * len(surface_indices))
        else:
            sim_mesh = UsdGeom.Mesh(sim_mesh_prim)
            vis_mesh.GetFaceVertexIndicesAttr().Set(sim_mesh.GetFaceVertexIndicesAttr().Get())
            vis_mesh.GetFaceVertexCountsAttr().Set(sim_mesh.GetFaceVertexCountsAttr().Get())
        if not prim.AddAppliedSchema("PhysicsDeformableBodyAPI"):
            raise RuntimeError(f"Failed to set deformable body API on prim '{prim.GetPath().pathString}'.")

    def initialize(self, sim_context: SimulationContext) -> None:
        """Initialize the manager with simulation context.

        Args:
            sim_context: Parent simulation context.
        """
        super().initialize(sim_context)
        self.builder = sim_context.get_or_create_backend(NewtonBuilderCfg(physics_cfg=self._cfg))

        # This context imports NewtonManager, so it can only be imported after this module initializes.
        from isaaclab_newton.cloner import NewtonReplicateContext  # noqa: PLC0415

        self.clone_context_type = NewtonReplicateContext

    def reset(self, soft: bool = False) -> None:
        """Reset physics simulation.

        A hard reset (``soft=False``) closes the backend, with its solver, contacts, step callbacks and captured graph,
        and finalizes a new one. Consumers bind to it again on ``PHYSICS_READY``; reacquire ``asset.data`` and its
        arrays afterwards. A soft reset keeps the backend.

        Args:
            soft: If True, skip full reinitialization.
        """
        if not soft:
            self._close_backend()
            self.start_simulation()
            self.initialize_solver()

    def _close_backend(self, *, dispatch_stop: bool = True) -> None:
        """Stop consumers, drop their views, and close the backend."""
        if self._incident_recorder is not None:
            self._incident_recorder.clear()
            self._incident_recorder = None
        if self._pending_operation_provider is not None:
            self._pending_operation_provider.close()
            self._pending_operation_provider = None
        if self._auto_debug_operation_provider:
            self._debug_operation_provider = None
            self._auto_debug_operation_provider = False
        backend = self.backend
        if backend is None:
            return
        if dispatch_stop:
            self.dispatch_event(PhysicsEvent.STOP)
        self.views.clear()
        self.backend = None
        SimulationContext.instance().close_backend(backend)

    def forward(self) -> None:
        """Update articulation kinematics without stepping physics; see :meth:`NewtonBackend.forward`."""
        if self.backend is not None:
            recorder = self._incident_recorder
            if recorder is not None and self.backend.kinematics_dirty:
                reset_worlds = self.backend.world_mask.numpy()[: self.backend.model.world_count].nonzero()[0]
                if len(reset_worlds):
                    recorder.rearm_reset_worlds(
                        tuple(int(world) for world in reset_worlds),
                        all_worlds=len(reset_worlds) == self.backend.model.world_count,
                    )
            self.backend.forward()

    def video_capture_backend(self) -> str:
        """Newton GL headless perspective video capture."""
        return "newton_gl"

    def _mark_transforms_changed(self) -> None:
        """Publish authored rigid-body changes and invalidate cable geometry."""
        self._scene_data_backend.transforms_timestamp += 1
        self._scene_data_backend.geometry_timestamp += 1
        self._flag_outer_capture()

    def _mark_particles_dirty(self) -> None:
        """Invalidate SDP geometry after native particle writes."""
        self._scene_data_backend.geometry_timestamp += 1
        self._flag_outer_capture()

    def _flag_outer_capture(self) -> None:
        """Flag writes recorded into an outer CUDA graph capture."""
        if self.backend.device.is_capturing:
            self.backend.transforms_may_change_on_graph_replay = True

    def step(self) -> None:
        """Advance physics by one environment step, or one physics step when the environment runs decimation."""
        sim = self._sim
        if sim is None or not sim.is_playing():
            return
        backend = self.backend
        physics_steps = backend.decimation if backend.handles_decimation() else 1
        recorder = self._incident_recorder
        if recorder is not None:
            with wp.ScopedDevice(backend.device):
                recorder.capture_pre(backend.state_0)
        debug_halt = False
        try:
            backend.step()
        except _DebugHalt:
            debug_halt = True
        finally:
            if recorder is not None and not recorder.capture_per_substep:
                end_time = self._sim_time + backend.dt * physics_steps
                with wp.ScopedDevice(backend.device):
                    results = self._evaluate_debug_incident_triggers("dispatch_post", None, end_time)
                    recorder.step(backend.state_0, end_time, trigger_results=results)
        self._sim_time += backend.dt * physics_steps
        if recorder is not None:
            if recorder.halted or debug_halt:
                raise RuntimeError(
                    f"Physics incident artifacts were exported to {self._cfg.debug_capture.output_dir};"
                    " halting simulation."
                )
        self._mark_transforms_changed()

    def close(self) -> None:
        """Clean up Newton physics resources."""
        super().close()
        self.clear()
        self.builder = None
        self._debug_operation_provider = None
        self._debug_incident_triggers.clear()

    def get_scene_data_backend(self) -> SceneDataBackend | None:
        """Return the SceneDataBackend for the SceneDataProvider."""
        return self._scene_data_backend

    def register_callback(
        self,
        callback: Callable,
        event: PhysicsEvent,
        order: int = 0,
        name: str | None = None,
        wrap_weak_ref: bool = True,
    ) -> CallbackHandle:
        """Register a callback. Passes event to parent class."""
        return super().register_callback(callback, event, order, name, wrap_weak_ref)

    def get_physics_sim_view(self) -> list:
        """Return the registered articulation views."""
        return list(self.views.values())

    def is_fabric_enabled(self) -> bool:
        """Check if fabric interface is enabled (not applicable for Newton)."""
        return False

    def clear(self):
        """Clear all Newton-specific state (callbacks cleared by super().close())."""
        self._close_backend(dispatch_stop=False)
        self._decimation = 1
        # Site requests and replication outputs survive hard resets.
        # (body_pattern, per_world, xform floats) -> (label, transform)
        self._site_requests: dict[tuple[str | None, bool, tuple[float, ...]], tuple[str, wp.transform]] = {}
        self._site_index_map: dict[str, SiteEntry] = {}
        self._clone: NewtonCloneRecord | None = None
        self.views.clear()

    def set_debug_operation_provider(self, provider: DebugOperationProvider) -> None:
        """Register an operation provider before solver initialization."""
        methods = ("bind", "snapshot", "close")
        if provider is None or any(not callable(getattr(provider, name, None)) for name in methods):
            raise TypeError("Debug operation provider must define bind(), snapshot(), and close().")
        if self._debug_operation_provider is not None or self._incident_recorder is not None:
            raise RuntimeError("Debug operation provider is already registered or capture has started.")
        self._debug_operation_provider = provider

    def set_debug_incident_trigger(
        self, name: str, trigger: Callable[[DebugTriggerContext], DebugTriggerResult | None]
    ) -> None:
        """Register a named trigger before the recorder freezes its schema."""
        if not isinstance(name, str):
            raise TypeError("Debug trigger name must be a string.")
        if _TRIGGER_NAME.fullmatch(name) is None:
            raise ValueError(f"Debug trigger name {name!r} must be lower snake case.")
        if not callable(trigger):
            raise TypeError(f"Debug trigger {name!r} must be callable.")
        if self._cfg.debug_capture is None or self._incident_recorder is not None:
            raise RuntimeError("Debug triggers require capture enabled before solver initialization.")
        if name in self._debug_incident_triggers:
            raise ValueError(f"Debug trigger {name!r} is already registered.")
        self._debug_incident_triggers[name] = trigger

    def _evaluate_debug_incident_triggers(self, phase: str, substep_idx: int | None, sim_time: float) -> dict:
        """Evaluate triggers in name order against the current backend state."""
        if not self._debug_incident_triggers:
            return {}
        backend = self.backend
        operations = None
        cfg = self._cfg.debug_capture
        if cfg.record_operations or cfg.replay.record_operations:
            provider = self._debug_operation_provider
            if provider is None:
                raise RuntimeError("Debug operation provider is unavailable for trigger evaluation.")
            operations = provider.snapshot()
            if operations is None:
                raise RuntimeError("Debug operation provider returned no trigger snapshot.")
        context = self.DebugTriggerContext(
            backend.model,
            backend.state_0,
            backend.control,
            backend.solver,
            backend.contacts,
            backend.collision_pipeline,
            operations,
            float(sim_time),
            phase,
            substep_idx,
        )
        results = {}
        for name, trigger in sorted(self._debug_incident_triggers.items()):
            try:
                result = trigger(context)
            except Exception as exc:
                raise RuntimeError(f"Debug incident trigger {name!r} failed: {exc}") from exc
            if result is not None:
                if not isinstance(result, self.DebugTriggerResult):
                    raise TypeError(f"Debug incident trigger {name!r} must return DebugTriggerResult or None.")
                results[name] = result
        return results

    def _capture_debug_substep(self, state: State, post: bool, index: int) -> None:
        """Record replay transitions and inspect a solver substep when configured."""
        recorder = self._incident_recorder
        backend = self.backend
        sim_time = self._sim_time + (index + int(post)) * backend.solver_dt
        if not post:
            recorder.record_step_replay_pre(state, sim_time=sim_time, substep_idx=index)
            return
        recorder.record_step_replay_post(state)
        if recorder.capture_per_substep:
            results = self._evaluate_debug_incident_triggers("solver_post", index, sim_time)
            recorder.observe_substep(state, sim_time, substep_idx=index, trigger_results=results)
            if recorder.halted:
                raise _DebugHalt

    def _export_incident_scene(self, usd_path: str, world_ids: list[int]) -> None:
        """Export a flattened static USD scene; recorded arrays carry the dynamic state."""
        from pxr import Usd  # noqa: PLC0415

        stage = get_current_stage()
        plan = self._sim.get_clone_plan() if self._sim is not None else None
        if stage is None or plan is None:
            raise RuntimeError("Incident scene export requires a USD stage and clone plan.")
        topology = plan.topology
        world_count = len(topology.world_prototype_layout)
        if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= world_count for i in world_ids):
            raise ValueError(f"Incident world IDs must be in [0, {world_count}).")
        starts, assets = topology.world_prototype_starts, topology.world_prototypes
        selected = {int(asset) for asset in assets[int(starts[0]) : int(starts[1])]}
        for world in world_ids or range(world_count):
            prototype = int(topology.world_prototype_layout[world])
            selected.update(int(asset) for asset in assets[int(starts[prototype + 1]) : int(starts[prototype + 2])])
        paths = cloner.path.get_asset_prototype_paths(plan)
        source_paths = list(dict.fromkeys(paths[asset] for asset in sorted(selected)))
        if any(not source for source in source_paths):
            raise RuntimeError("Clone plan contains a selected asset without an authored scene source.")
        if not source_paths:
            raise RuntimeError("Clone plan contains no authored scene sources.")
        flat_layer = stage.Flatten()
        flat_stage = Usd.Stage.Open(flat_layer)
        if flat_stage is None:
            raise RuntimeError("Cannot open the flattened incident scene.")
        for source in source_paths:
            if not flat_stage.GetPrimAtPath(source).IsValid():
                raise RuntimeError(f"Clone-plan source {source!r} is absent from the flattened scene.")
        flat_stage.SetDefaultPrim(flat_stage.GetPrimAtPath(source_paths[0]))
        if flat_layer.Export(usd_path) is False:
            raise RuntimeError(f"Failed to export incident scene to {usd_path!r}.")

    def register_site(self, body_pattern: str | None, xform: wp.transform, *, per_world: bool = False) -> str:
        """Request a site for injection into prototypes before replication.

        Sensors call this during ``__init__``. Identical ``(body_pattern, per_world, transform)`` requests share a site,
        and requests persist until :meth:`close`, so sites survive hard resets without being requested again.
        The pattern matches prototype-local body labels (e.g. ``"Robot/finger.*"``) during replication, and wildcard
        patterns create one site per matched body.

        Args:
            body_pattern: Regex matched against body labels, or ``None`` for a global site.
            xform: Site transform relative to the body.
            per_world: Create one bodyless site in each cloned world's frame. Requires ``body_pattern=None``.

        Returns:
            The assigned site label.
        """
        if per_world and body_pattern is not None:
            raise ValueError("per_world site registration requires body_pattern=None.")
        requests = self._site_requests
        key = (body_pattern, per_world, tuple(xform))
        if key not in requests:
            requests[key] = (f"ft_{len(requests)}", xform)
        return requests[key][0]

    def inject_sites(
        self, main_builder: ModelBuilder, source_builders: dict[str, ModelBuilder]
    ) -> tuple[dict[str, int], dict[int, dict[str, list[int]]], dict[str, wp.transform]]:
        """Add unresolved sites to source builders, or to the main builder for global and shared-asset sites.

        Args:
            main_builder: Top-level builder that receives global sites.
            source_builders: Source builders by source path.

        Returns:
            ``(global_site_indices, source_site_indices, env_root_sites)``: global site shape indices by label,
            builder-local site indices by ``id(builder)`` and label, and world-root site transforms by label.
        """
        global_sites: dict[str, int] = {}
        source_sites: dict[int, dict[str, list[int]]] = {}
        root_sites: dict[str, wp.transform] = {}
        for (body_pattern, per_world, _), (label, xform) in self._site_requests.items():
            if label in self._site_index_map:
                continue
            if per_world:
                root_sites[label] = xform
                continue
            if body_pattern is None:
                global_sites[label] = main_builder.add_site(body=-1, xform=xform, label=label)
                continue
            matched_any = False
            for builder in (*source_builders.values(), main_builder):
                if builder is main_builder and matched_any:
                    break
                indices, names = resolve_matching_names(body_pattern, builder.body_label, raise_when_no_match=False)
                if not indices:
                    continue
                matched_any = True
                sites = [
                    builder.add_site(body=body, xform=xform, label=f"{name}/{label}")
                    for body, name in zip(indices, names)
                ]
                source_sites.setdefault(id(builder), {})[label] = sites
            if not matched_any:
                raise ValueError(f"Site '{label}' with body_pattern '{body_pattern}' matched no builder bodies.")
        return global_sites, source_sites, root_sites

    def record_clone(self, record: NewtonCloneRecord, site_index_map: dict[str, SiteEntry]) -> None:
        """Record native replication outputs for model finalization and consumers.

        Args:
            record: Replication outputs.
            site_index_map: Sites resolved during replication, by label.
        """
        self._clone = record
        self._site_index_map.update(site_index_map)

    def get_site_index_map(self) -> dict[str, SiteEntry]:
        """Return resolved sites by label."""
        return self._site_index_map

    def get_world_xforms(self) -> list[wp.transform] | None:
        """Return the root transform of each cloned world, or ``None`` without replication."""
        clone = self._clone
        return None if clone is None else clone.world_xforms

    def get_clone_source_builders(self) -> dict[str, ModelBuilder]:
        """Return per-source builders retained from replication, keyed by clone-plan source path."""
        clone = self._clone
        return {} if clone is None else clone.source_builders

    def request_extended_state_attribute(self, attr: str) -> None:
        """Request an extended state attribute (e.g. ``"body_qdd"``) from the builder before finalization.

        Args:
            attr: State attribute name (must be in ``State.EXTENDED_ATTRIBUTES``).
        """
        self.builder.request_state_attributes(attr)

    def request_extended_contact_attribute(self, attr: str) -> None:
        """Request an extended contact attribute (e.g. ``"force"``) from the builder before finalization.

        Args:
            attr: Contact attribute name.
        """
        self.builder.request_contact_attributes(attr)

    def add_model_change(self, change: ModelFlags) -> None:
        """Register a model change to notify the solver at the next step."""
        self.backend.model_changes.add(change)
        if change & ModelFlags.SHAPE_TYPE_AND_SOURCE:
            self.backend.geometry_change_version += 1

    def invalidate_fk(
        self,
        env_mask: wp.array | None = None,
        articulation_ids: wp.array | None = None,
        *,
        articulation_mask: wp.array | None = None,
        consume_mask: bool = False,
    ) -> None:
        """Publish an authored-state change and schedule FK; see :meth:`NewtonBackend.invalidate_fk`.

        Asset writers invalidate their own caches separately. Pending work is consumed at the next
        forward, raw-state, rendering, or physics-step boundary.
        """
        self._mark_transforms_changed()
        self.backend.invalidate_fk(
            env_mask, articulation_ids, articulation_mask=articulation_mask, consume_mask=consume_mask
        )

    def invalidate_body_state(self, env_mask: wp.array | None = None, *, consume_mask: bool = False) -> None:
        """Publish directly authored body state; see :meth:`NewtonBackend.invalidate_body_state`."""
        self._mark_transforms_changed()
        self.backend.invalidate_body_state(env_mask, consume_mask=consume_mask)

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
        sim = self._sim
        builder = self.builder
        self._drain_stale_cuda_error()

        logger.info("Dispatching MODEL_INIT callbacks")
        self.dispatch_event(PhysicsEvent.MODEL_INIT)

        # The builder survives hard resets, so sites already added to it are skipped.
        global_sites, body_sites, root_sites = self.inject_sites(builder, {})
        site_map = self._site_index_map
        site_map.update((label, (index, None)) for label, index in global_sites.items())
        site_map.update((label, (None, [indices])) for label, indices in body_sites.get(id(builder), {}).items())
        for label, xform in root_sites.items():
            site_map[label] = (None, [[builder.add_site(body=-1, xform=xform, label=label)]])

        device = self._device
        logger.info(f"Finalizing model on device: {device}")
        builder.up_axis = Axis.Z
        if (solver_cfg := self._cfg.solver_cfg) is not None:
            solver_cfg.class_type.prepare_builder(solver_cfg, builder)
        with Timer(
            name="newton_finalize_builder",
            msg="Finalize builder took:",
            activity="Finalizing physics model",
            synchronize="both",
            device=device,
        ):
            cfg = NewtonBackendCfg(physics_cfg=self._cfg, device=sim.device)
            initial = () if self._clone is None else self._clone.initial_body_variants
            self.backend = sim.get_or_create_backend(
                cfg, builder=builder, dt=self.get_physics_dt(), initial_body_variants=initial
            )
            self.backend.model.set_gravity(sim.cfg.gravity)

        self.backend.decimation = self._decimation
        clone = self._clone
        if clone is not None:
            self.backend.particle_ranges = clone.particle_ranges
        self._scene_data_backend.initialize_geometry(
            [] if clone is None else clone.geometry_batches, {} if clone is None else clone.cable_bindings
        )
        logger.info("Dispatching PHYSICS_READY callbacks")
        self.dispatch_event(PhysicsEvent.PHYSICS_READY)

    def instantiate_builder_from_stage(self):
        """Import the explicitly declared clone plan into the Newton builder."""
        cloner.replicate(self._sim.get_clone_plan())

    def initialize_solver(self) -> None:
        """Build the solver, its contacts and the initial body state; see :meth:`NewtonBackend.init_solver`."""
        backend = self.backend
        sim = self._sim
        plan = sim.get_clone_plan() if sim is not None else None
        if sim is not None:
            backend.solver_init_callbacks.append(self._prepare_visualizer_for_capture)
        with Timer(
            name="newton_initialize_solver",
            msg="Initialize solver took:",
            activity="Initializing solver",
            synchronize="both",
            device=self.get_device(),
        ):
            backend.init_solver(None if plan is None else plan.env_template, relaxed_capture=self._relaxed_capture)
        capture_cfg = self._cfg.debug_capture
        if capture_cfg is not None:
            from ._incident_recorder import PhysicsIncidentRecorder, _OperationProviderCleanupError  # noqa: PLC0415

            operation_requested = capture_cfg.record_operations or capture_cfg.replay.record_operations
            provider = self._debug_operation_provider
            if provider is not None and not operation_requested:
                raise RuntimeError("An operation provider is registered but operation recording is disabled.")
            if operation_requested and provider is None:
                if not isinstance(backend.solver, SolverMuJoCo) or backend.solver.use_mujoco_cpu:
                    raise RuntimeError(
                        "Automatic operation capture requires a MJWarp-backed SolverMuJoCo; "
                        "register an explicit provider."
                    )
                from .mjwarp_debug import MJWarpDebugOperationProvider  # noqa: PLC0415

                provider = MJWarpDebugOperationProvider()
                self._debug_operation_provider = provider
                self._auto_debug_operation_provider = True
            try:
                self._incident_recorder = PhysicsIncidentRecorder(
                    backend.model,
                    capture_cfg,
                    state=backend.state_0,
                    control=backend.control,
                    solver=backend.solver,
                    contacts=backend.contacts,
                    collision_pipeline=backend.collision_pipeline,
                    scene_exporter=self._export_incident_scene if capture_cfg.record_scene else None,
                    context_provider=self.get_debug_context,
                    operation_provider=provider,
                    triggers=dict(self._debug_incident_triggers),
                )
            except _OperationProviderCleanupError:
                self._pending_operation_provider = provider
                raise
            if capture_cfg.capture_per_substep or capture_cfg.replay.enabled:
                backend.debug_substep = self._capture_debug_substep
        self._mark_transforms_changed()

    def _relaxed_capture(self) -> bool:
        """Whether Kit rendering requires relaxed CUDA graph capture."""
        sim = self._sim
        return sim is not None and has_kit() and (sim.has_gui or sim.has_offscreen_render)

    def _prepare_visualizer_for_capture(self) -> None:
        """Let the Newton visualizer register picking forces before the step is captured."""
        if self.backend.solver_ops.applies_body_forces():
            self._sim._prepare_newton_visualizer_for_capture()

    def create_visual_material_writer(self, batches: tuple[VisualMaterialBatch, ...]) -> VisualMaterialWriter:
        """Compile material-to-shape addresses for the active Newton model."""
        return self.backend.create_visual_material_writer(batches)

    def create_visual_shape_color_writer(
        self, asset: BaseArticulation, body_names: tuple[str, ...]
    ) -> VisualShapeColorWriter:
        """Compile selected articulation-body shape addresses for the active Newton model."""
        model = self.backend.model
        view = asset.root_view
        if not isinstance(view, ArticulationView):
            root_expr = asset.cfg.prim_path
            root_expr += (
                "(?:/.*)?" if asset.cfg.articulation_root_prim_path is None else asset.cfg.articulation_root_prim_path
            )
            prim_paths = [path for path in model.articulation_label if re.fullmatch(root_expr, path)]
            view = ArticulationView(model, prim_paths, verbose=False)
        return VisualShapeColorWriter(model, view, body_names)

    def get_scene_data_provider(self) -> SceneDataProvider:
        """Return the active scene data provider."""
        return SimulationContext.instance().get_scene_data_provider()

    def activate_newton_actuator_path(self) -> None:
        """Run the model's Newton actuators inside the step; see :meth:`NewtonBackend.activate_actuators`."""
        self.backend.activate_actuators()

    def register_solver_init_callback(self, callback: Callable[[], None]) -> None:
        """Append a one-shot hook invoked once the solver exists, before the initial FK and any CUDA graph capture."""
        self.backend.solver_init_callbacks.append(callback)

    def register_pre_actuator_callback(self, callback: Callable[[], None]) -> None:
        """Append a hook invoked before the Newton actuators of every physics step.

        Each callback is recorded into the captured step, so it must be graph-safe. Hard resets discard these hooks;
        register them again when the new model is ready.
        """
        self.backend.pre_actuator_callbacks.append(callback)
        self.backend.invalidate_graph()

    def register_post_actuator_callback(self, callback: Callable[[], None]) -> None:
        """Append a hook invoked after the Newton actuators of every physics step.

        Each callback is recorded into the captured step right after the Newton actuators and before the solver
        substeps, so its writes to ``state``/``control`` are visible to the integrator on the same iteration.
        Callbacks fire in registration order. Hard resets discard these hooks; register them again when the new model
        is ready.
        """
        self.backend.post_actuator_callbacks.append(callback)
        self.backend.invalidate_graph()

    def register_state_force_callback(self, callback: Callable[[State], None]) -> None:
        """Register a graph-safe callback that applies forces before every solver substep.

        Hard resets discard these hooks; register them again when the new model is ready.

        Args:
            callback: Function that adds forces [N, N·m] to the provided state.
        """
        if callback not in self.backend.state_force_callbacks:
            self.backend.state_force_callbacks.append(callback)
            self.backend.invalidate_graph()

    def register_post_step_callback(self, callback: Callable[[], None]) -> None:
        """Append a hook invoked once per :meth:`step`, after the last solver substep and before sensors update.

        The hook reflects the state after all decimation iterations of the step. Callbacks must be graph-safe (fixed
        shapes, no host branching on device data) because they are recorded into the captured step.
        Hard resets discard these hooks; register them again when the new model is ready.
        """
        self.backend.post_step_callbacks.append(callback)
        self.backend.invalidate_graph()

    def unregister_post_step_callback(self, callback: Callable[[], None]) -> None:
        """Remove a post-step callback; removing an absent callback is a no-op."""
        if self.backend is not None and callback in self.backend.post_step_callbacks:
            self.backend.post_step_callbacks.remove(callback)
            self.backend.invalidate_graph()

    def set_decimation(self, decimation: int) -> None:
        """Set the decimation count; see :meth:`NewtonBackend.set_decimation`.

        Once Newton actuators run in the step, one :meth:`step` advances the whole decimation loop.
        """
        self._decimation = max(1, decimation)
        if self.backend is not None:
            self.backend.set_decimation(self._decimation)

    def handles_decimation(self) -> bool:
        """Whether :meth:`step` executes the full decimation loop; see :meth:`NewtonBackend.handles_decimation`."""
        return self.backend.handles_decimation()

    def add_contact_sensor(
        self,
        body_names_expr: str | list[str] | None = None,
        shape_names_expr: str | list[str] | None = None,
        contact_partners_body_expr: str | list[str] | None = None,
        contact_partners_shape_expr: str | list[str] | None = None,
        verbose: bool = False,
    ) -> SensorContact:
        """Add a contact sensor between bodies or shapes; see :meth:`NewtonBackend.add_contact_sensor`."""
        return self.backend.add_contact_sensor(
            body_names_expr,
            shape_names_expr,
            contact_partners_body_expr,
            contact_partners_shape_expr,
            verbose,
        )

    def add_imu_sensor(self, sites: list[int]) -> SensorIMU:
        """Add an IMU sensor at sites; see :meth:`NewtonBackend.add_imu_sensor`."""
        return self.backend.add_imu_sensor(sites)
