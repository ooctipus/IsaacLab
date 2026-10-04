# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation-owned native worlds with GPU lifecycle commands and stable graphs."""

from __future__ import annotations

import math
import operator
from dataclasses import MISSING, field
from typing import TYPE_CHECKING

import newton
import numpy as np
import warp as wp
from gpu_components.directory_data import InstanceCommands, InstanceResults
from newton.solvers import SolverMuJoCo, mujoco_worlds_capture, mujoco_worlds_close, mujoco_worlds_prepare

from isaaclab.physics import PhysicsCfg, PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend
from isaaclab.sim import BackendCfg
from isaaclab.utils import checked_apply, configclass

from .newton_manager_cfg import NewtonCfg

if TYPE_CHECKING:
    from mujoco_warp import Data, Model

    from isaaclab.sim import SimulationContext

__all__ = ["NewtonWorldsBackend", "NewtonWorldsBackendCfg", "NewtonWorldsCfg", "NewtonWorldsManager"]


@configclass
class NewtonWorldsBackendCfg(BackendCfg):
    """Experimental native preparation inputs, immutable after context registration."""

    class_type: type = "{DIR}.worlds:NewtonWorldsBackend"
    prototypes: tuple[tuple[Model, Data], ...] = field(kw_only=True, metadata={"copy": False})
    """Prepared immutable native models and one-world defaults, borrowed without copying."""
    world_capacities: tuple[int, ...] = MISSING
    """Virtual world row limits, independently of physical readiness and live counts."""
    world_id_capacity: int = MISSING
    """Maximum simultaneous logical world identities."""
    command_capacity: int = MISSING
    """Maximum GPU lifecycle requests in one batch."""
    dt: float = MISSING
    """Native timestep [s], matching every prepared prototype's immutable timestep."""
    substeps: int = 2
    """Native substeps per physics frame; ``dt * substeps`` equals SimulationCfg.dt."""
    initial_world_ready_capacities: tuple[int, ...] | None = None
    """Initial physical world readiness; lifetime creation belongs to task commands."""
    contact_capacities: tuple[int, ...] | None = None
    """Optional virtual candidate/contact limits per prototype."""
    ccd_capacities: tuple[int, ...] | None = None
    """Optional virtual CCD scratch limits per prototype."""
    memory_budget_bytes: int | None = None
    """Shared physical VMM budget [byte]; None selects fixed backing."""


class NewtonWorldsBackend:
    """Own one native runtime and executable for a simulation context.

    Experimental. Native arrays and world identities live only in ``runtime``.
    Tasks prepare prototypes, command/payload buffers and recording callbacks.
    They publish coherent commands before any replay, including ``forward``.
    """

    def __init__(self, cfg: NewtonWorldsBackendCfg):
        if not math.isfinite(cfg.dt) or cfg.dt <= 0:
            raise ValueError("Native timestep must be finite and positive.")
        if isinstance(cfg.substeps, bool) or operator.index(cfg.substeps) < 1:
            raise ValueError("Native substeps must be a positive integer.")
        prepared = tuple(cfg.prototypes)
        if not prepared:
            raise ValueError("At least one prepared native prototype is required.")
        for model, _ in prepared:
            if not np.allclose(model.opt.timestep.numpy(), cfg.dt, rtol=1e-7, atol=0):
                raise ValueError("Every prepared native timestep must equal the backend dt.")
        self.dt, self.substeps = cfg.dt, operator.index(cfg.substeps)
        self.runtime = mujoco_worlds_prepare(
            prepared,
            world_capacities=cfg.world_capacities,
            id_capacity=cfg.world_id_capacity,
            command_capacity=cfg.command_capacity,
            contact_capacities=cfg.contact_capacities,
            ccd_capacities=cfg.ccd_capacities,
            memory_budget_bytes=cfg.memory_budget_bytes,
            initial_world_ready_capacities=cfg.initial_world_ready_capacities,
        )
        self.graph: wp.Graph | None = None
        self._closed = False
        try:
            with wp.ScopedDevice(self.device):
                self._permit = wp.zeros(1, dtype=int, device=self.device)
                self._last_work = wp.Event(self.device)
                wp.get_stream(self.device).record_event(self._last_work)
        except BaseException:
            mujoco_worlds_close(self.runtime, streams=(wp.get_stream(self.device),))
            raise

    @property
    def device(self) -> wp.context.Device:
        """The authoritative native runtime device."""
        return self.runtime.device

    def prepare(
        self,
        commands: InstanceCommands,
        results: InstanceResults,
        *,
        validate=None,
        initialize=None,
        before_step=None,
        after_substep=None,
        application_bindings=None,
        retain=(),
    ) -> None:
        """Record task callbacks and one graph before the first simulation reset.

        Callbacks run only during preparation; their recorded kernels consume
        explicitly retained buffers. ``application_bindings`` separately declares
        numeric count relations for their exact captured records. The task initializes each request's payload
        before publishing its new command sequence. Poses refresh on every valid
        replay, including reset-only replay with no native advancement.
        """
        if self._closed:
            raise RuntimeError("The native worlds backend is closed.")
        if self.graph is not None:
            raise RuntimeError("The native worlds backend already has its prepared graph.")
        with wp.ScopedDevice(self.device), wp.ScopedStream(wp.Stream(self.device), sync_exit=True):
            self.graph = mujoco_worlds_capture(
                self.runtime,
                commands,
                results,
                permit=self._permit,
                validate=validate,
                initialize=initialize,
                before_step=before_step,
                after_substep=after_substep,
                application_bindings=application_bindings,
                retain=retain,
                substeps=self.substeps,
                refresh_kinematics=True,
            )
            wp.get_stream(self.device).record_event(self._last_work)

    def _replay(self, advance: bool) -> None:
        if self._closed or self.graph is None:
            raise RuntimeError("Prepare an open native worlds backend before replay.")
        with wp.ScopedDevice(self.device):
            producer = wp.get_stream(self.device)
            producer.wait_event(self._last_work)
            try:
                self._permit.fill_(int(advance))
                wp.capture_launch(self.graph)
            finally:
                producer.record_event(self._last_work)

    def step(self) -> None:
        """Apply coherent pending commands, advance one physics frame and refresh poses."""
        self._replay(True)

    def forward(self) -> None:
        """Apply coherent pending commands and refresh poses without advancing time."""
        self._replay(False)

    def close(self) -> None:
        """Join queued consumers, drop graph borrowers and close native storage."""
        if not self._closed:
            with wp.ScopedDevice(self.device):
                wp.synchronize_device(self.device)
                self.graph = None
                mujoco_worlds_close(self.runtime, streams=(wp.get_stream(self.device),))
                self._permit = self._last_work = None
                self._closed = True


@configclass
class NewtonWorldsCfg(PhysicsCfg):
    """Experimental physics lifecycle for explicitly prepared native worlds."""

    class_type: type = "{DIR}.worlds:NewtonWorldsManager"
    prototype_physics: NewtonCfg | None = None
    """Authoring defaults for the manager's one-world builder factory."""


class NewtonWorldsManager(PhysicsManager):
    """Borrow the SimulationContext-owned backend without a global Newton state mirror."""

    _backend: NewtonWorldsBackend | None = None
    _scene_data: SceneDataBackend | None = None
    _ready = False

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        if sim_context.resolve_visualizer_types() or sim_context.get_setting("/isaaclab/cameras_enabled"):
            raise ValueError("Native worlds currently require headless execution without cameras or visualizers.")
        super().initialize(sim_context)
        cls._backend = None
        cls._scene_data = SceneDataBackend()
        cls._ready = False

    @classmethod
    def create_builder(cls, up_axis: str = "Z") -> newton.ModelBuilder:
        """Create a one-world builder with registered solver schema and authoring defaults."""
        physics = cls._cfg.prototype_physics
        if physics is None:
            raise ValueError("Prototype authoring requires NewtonWorldsCfg.prototype_physics.")
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
    def install(cls, backend: NewtonWorldsBackend) -> None:
        """Borrow the explicitly registered context resource before simulation reset."""
        if cls._scene_data is None:
            raise RuntimeError("Initialize the simulation context before installing native worlds.")
        if cls._backend is not None and cls._backend is not backend:
            raise RuntimeError("A native worlds backend is already installed.")
        if backend.device != wp.get_device(cls.get_device()):
            raise ValueError("The native worlds backend must use the simulation device.")
        if not math.isclose(backend.dt * backend.substeps, cls.get_physics_dt(), rel_tol=1e-7):
            raise ValueError("Native dt * substeps must equal SimulationCfg.dt.")
        cls._backend = backend

    @classmethod
    def reset(cls, soft: bool = False) -> None:
        if cls._backend is None:
            raise RuntimeError("Install a context-owned NewtonWorldsBackend before sim.reset().")
        if not cls._ready:
            cls.dispatch_event(PhysicsEvent.MODEL_INIT)
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
            raise RuntimeError("Reset the native worlds simulation before stepping.")
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
            cls._backend = cls._scene_data = None
            cls._ready = False
