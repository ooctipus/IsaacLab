# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for physics managers with unified callback system."""

from __future__ import annotations

import logging
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

from isaaclab.utils._device import set_cuda_device

if TYPE_CHECKING:
    from isaaclab.scene_data import SceneDataBackend
    from isaaclab.sim.simulation_context import SimulationContext

logger = logging.getLogger(__name__)


class PhysicsEvent(Enum):
    """Physics simulation lifecycle events.

    Lifecycle order: MODEL_INIT -> PHYSICS_READY -> STOP. PRIM_DELETION may occur
    any time after model creation.
    """

    MODEL_INIT = "model_init"
    """Physics model is being constructed.
    Fired during scene building, before simulation can run. Use this to register
    physics representations (rigid bodies, joints, constraints) with the solver.
    """

    PHYSICS_READY = "physics_ready"
    """Physics is initialized and queryable.
    Fired after all physics data structures are created and the simulation is
    ready to step. Assets can now read initial state (positions, velocities).
    """

    PRIM_DELETION = "prim_deletion"
    """A physics prim is being deleted.

    The payload is a mapping containing its ``prim_path``. A path of ``/`` invalidates
    every stage-bound object.
    """

    STOP = "stop"
    """Simulation is stopping."""


class CallbackHandle:
    """Handle for a registered callback, allowing deregistration."""

    def __init__(self, callback_id: int, manager: PhysicsManager):
        self._id = callback_id
        self._manager = manager

    @property
    def id(self) -> int:
        return self._id

    def deregister(self) -> None:
        """Remove this callback from the manager."""
        self._manager.deregister_callback(self._id)


class PhysicsManager(ABC):
    """Abstract base class for physics simulation managers.

    Physics managers handle the lifecycle of a physics simulation backend,
    including initialization, stepping, and cleanup.

    This base class provides:
    - Unified callback management system
    - Common state variables (_sim, _cfg, _device)
    - Default accessor implementations

    Lifecycle: construction -> clone -> reset() -> step() (repeated) -> close()
    """

    supports_anim_recording: ClassVar[bool] = False
    """Whether this backend can service ``--anim_recording_enabled`` (OVD Recorder).

    Overridden by backends that implement the recorder (currently PhysX-only).
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._cfg = cfg
        self._sim: SimulationContext | None = None
        self._device = "cuda:0"
        self._sim_time = 0.0
        self._callbacks: dict[int, tuple[PhysicsEvent, Callable, int, str | None]] = {}
        self._callback_id = 0

    def _prepare_stage_creation(self) -> None:
        """Perform backend-specific setup required before the USD stage is created."""
        pass

    @staticmethod
    def fix_articulation_root(articulation_prim: Any, stage: Any) -> Any:
        """Ensure that an articulation root has one enabled world fixed joint.

        The base implementation leaves the root in place. Backends whose parser requires a different
        root topology may relocate it and return the resulting root prim.

        Args:
            articulation_prim: The articulation-root prim to fix.
            stage: The stage containing the prim.

        Returns:
            The articulation-root prim after backend normalization.

        Raises:
            NotImplementedError: If a new joint is needed and the root is not a rigid body.
        """
        # Keep these imports local. Hoisting the isaaclab.sim ones closes a real import cycle:
        # isaaclab.physics -> sim.schemas.schemas -> sim.utils.prims -> sim.utils.queries ->
        # sim.simulation_context -> isaaclab.physics (partially initialized). Keeping pxr local
        # also keeps USD out of the config-definition path, which env configs import through
        # managers.manager_base before the simulation app starts.
        from pxr import UsdPhysics  # noqa: PLC0415

        from isaaclab.sim.schemas.schemas import create_world_fixed_joint  # noqa: PLC0415
        from isaaclab.sim.utils import find_global_fixed_joint_prim  # noqa: PLC0415

        root_path = articulation_prim.GetPath().pathString
        joint = find_global_fixed_joint_prim(root_path, stage=stage)
        if joint is not None:
            joint.GetJointEnabledAttr().Set(True)
            return articulation_prim
        if not articulation_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise NotImplementedError(f"Cannot fix non-rigid articulation root '{root_path}'.")

        create_world_fixed_joint(articulation_prim, stage)
        return articulation_prim

    @staticmethod
    def _relocate_articulation_root(
        articulation_prim: Any,
        companion_schema: str,
        companion_namespace: str,
    ) -> Any:
        """Move root-bearing schemas and authored properties to the root link's parent."""
        # Keep pxr local: this module is imported while environment configs load (via the manager
        # classes), and config loading must not pull USD/omni modules before the simulation app
        # starts.
        from pxr import Usd, UsdPhysics  # noqa: PLC0415

        new_root = articulation_prim.GetParent()
        if new_root.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise RuntimeError(
                f"Cannot relocate '{articulation_prim.GetPath()}' to existing articulation root '{new_root.GetPath()}'."
            )

        # Keep this import local for the same reason as the pxr imports above.
        from isaaclab.sim.schemas._backend_hooks import _articulation_root_companion_namespace  # noqa: PLC0415

        registry = Usd.SchemaRegistry()
        root_schema = UsdPhysics.Tokens.PhysicsArticulationRootAPI
        schemas_to_move = []
        for schema_name in articulation_prim.GetPrimTypeInfo().GetAppliedAPISchemas():
            definition = registry.FindAppliedAPIPrimDefinition(schema_name)
            companion_namespace_override = _articulation_root_companion_namespace(schema_name)
            if schema_name == companion_schema:
                properties = list(articulation_prim.GetAuthoredPropertiesInNamespace(companion_namespace))
            elif companion_namespace_override is not None:
                # a backend-registered schema, possibly an unregistered token the registry cannot
                # describe, so take the namespace the backend declared for it
                properties = list(articulation_prim.GetAuthoredPropertiesInNamespace(companion_namespace_override))
            elif schema_name == root_schema or (
                definition is not None and root_schema in definition.GetAppliedAPISchemas()
            ):
                properties = []
                if definition is not None:
                    for property_name in definition.GetPropertyNames():
                        prop = articulation_prim.GetProperty(property_name)
                        if prop and prop.IsAuthored():
                            properties.append(prop)
            else:
                continue
            schemas_to_move.append((schema_name, properties))

        for schema_name, properties in schemas_to_move:
            if not new_root.AddAppliedSchema(schema_name):
                raise RuntimeError(f"Failed to apply '{schema_name}' to '{new_root.GetPath()}'.")
            for prop in properties:
                if not prop.FlattenTo(new_root):
                    raise RuntimeError(f"Failed to move '{prop.GetPath()}' to '{new_root.GetPath()}'.")
        for schema_name, _ in schemas_to_move:
            if not articulation_prim.RemoveAppliedSchema(schema_name):
                raise RuntimeError(f"Failed to remove '{schema_name}' from '{articulation_prim.GetPath()}'.")
        if articulation_prim.HasAPI(UsdPhysics.ArticulationRootAPI) or not new_root.HasAPI(
            UsdPhysics.ArticulationRootAPI
        ):
            raise RuntimeError(
                f"Failed to relocate articulation root '{articulation_prim.GetPath()}' to '{new_root.GetPath()}'."
            )
        return new_root

    def register_callback(
        self,
        callback: Callable[[Any], None],
        event: PhysicsEvent,
        order: int = 0,
        name: str | None = None,
    ) -> CallbackHandle:
        """Register a callback for a physics event.

        Args:
            callback: The callback function. Receives event payload as argument.
            event: The event to listen for.
            order: Priority order (lower = earlier). Default 0.
            name: Optional name for debugging.

        Returns:
            CallbackHandle that can be used to deregister the callback.

        Example:
            >>> def on_physics_ready(payload):
            ...     print("Physics is ready!")
            >>> handle = sim._register_physics_callback(on_physics_ready, PhysicsEvent.PHYSICS_READY)
            >>> # Later, to remove:
            >>> handle.deregister()
        """
        cid = self._callback_id
        self._callback_id += 1

        callback = self._wrap_weak_ref(callback)

        self._callbacks[cid] = (event, callback, order, name)
        return CallbackHandle(cid, self)

    def deregister_callback(self, callback_id: int | CallbackHandle) -> None:
        """Remove a registered callback.

        Args:
            callback_id: The ID or CallbackHandle returned by register_callback().
        """
        cid = callback_id.id if isinstance(callback_id, CallbackHandle) else callback_id
        if cid not in self._callbacks:
            return

        self._callbacks.pop(cid)

    def dispatch_event(self, event: PhysicsEvent, payload: Any = None) -> None:
        """Dispatch an event to all registered callbacks.

        This is the default implementation using simple callback lists.
        Subclasses may override or extend with platform-specific dispatch.

        Args:
            event: The event to dispatch.
            payload: Optional data to pass to callbacks.
        """
        matching = [(cid, cb, order) for cid, (ev, cb, order, _name) in self._callbacks.items() if ev == event]
        matching.sort(key=lambda x: x[2])

        for _, callback, _ in matching:
            callback(payload)

    def clear_callbacks(self) -> None:
        """Remove all registered callbacks.

        Do NOT reset ``_callback_id`` — handle IDs must remain monotonically
        unique across the lifetime of the process.  Resetting the counter
        would let a future :meth:`register_callback` hand out an ID that an
        old, still-alive :class:`CallbackHandle` (e.g. on a sensor that has
        not been garbage-collected yet) holds, so when the old object
        eventually finalizes its ``__del__`` would deregister the new
        callback.  This bit ovphysx's kitless multi-context tests where two
        ``InteractiveScene``s are created in sequence: the first scene's
        sensor would post-GC deregister the second scene's
        ``_initialize_callback`` by ID collision, leaving the second sensor
        forever uninitialized.
        """
        for cid in list(self._callbacks):
            self.deregister_callback(cid)
        self._callbacks.clear()

    @staticmethod
    def _wrap_weak_ref(callback: Callable) -> Callable:
        """Wrap bound methods with weak references to prevent leaks.

        Args:
            callback: The callback to wrap.

        Returns:
            Wrapped callback if it's a bound method, otherwise original.
        """
        owner = getattr(callback, "__self__", None)
        if owner is not None:
            obj_ref = weakref.ref(owner)
            method_name = callback.__name__

            def weak_callback(payload: Any) -> Any:
                obj = obj_ref()
                if obj is None:
                    return None
                return getattr(obj, method_name)(payload)

            return weak_callback
        return callback

    @abstractmethod
    def _bind_context(self, sim_context: SimulationContext) -> None:
        """Bind the physics manager to its simulation before cloning.

        Subclasses should call ``super()._bind_context()`` first, then register backend resources
        needed by cloning. Runtime model initialization belongs to the post-clone hard reset.

        Args:
            sim_context: Parent simulation context.
        """
        self._sim = sim_context
        self._device = sim_context.cfg.device
        self._sim_time = 0.0

        # Synchronize the process-wide CUDA device before backend-specific
        # initialization allocates state. PyTorch must select the device before
        # Warp so that both runtimes retain the same primary CUDA context.
        if "cuda" in self._device:
            set_cuda_device(self._device)

        # The OVD Recorder (omni.physx.pvd) only records PhysX simulations. On other backends the
        # recording would silently never start, so the process would run until manually killed
        # instead of stopping at `--anim_recording_stop_time` and saving the animation.
        # ``get_setting`` may be absent on lightweight sim_context test doubles that only
        # implement the ``cfg``/``device`` surface this method also reads above.
        get_setting = getattr(sim_context, "get_setting", None)
        if get_setting and get_setting("/isaaclab/anim_recording/enabled") and not self.supports_anim_recording:
            raise ValueError(
                f"'--anim_recording_enabled' was set, but the active physics backend ('{type(self).__name__}') does not"
                " support the OVD Recorder. Select the PhysX backend, e.g. by appending"
                " 'physics=isaacsim_physx' to the command line."
            )

    @abstractmethod
    def reset(self, soft: bool = False) -> None:
        """Reset physics simulation.

        Args:
            soft: If True, skip full reinitialization.
        """
        pass

    @abstractmethod
    def forward(self) -> None:
        """Update kinematics without stepping physics."""
        pass

    @abstractmethod
    def get_scene_data_backend(self) -> SceneDataBackend:
        """Return the SceneDataBackend for the SceneDataProvider."""
        pass

    @abstractmethod
    def step(self) -> None:
        """Step physics simulation by one timestep (physics only, no rendering)."""
        pass

    def close(self) -> None:
        """Clean up physics resources.

        Subclasses whose STOP listeners own backend handles should call
        ``super().close()`` before backend-specific cleanup so those listeners
        can invalidate their handles while the backend is still live.

        All STOP listeners are given a chance to run. If one or more listeners
        fail, callback and shared simulation state is still cleared before an
        aggregate :class:`RuntimeError` is raised from the first failure.
        """
        is_active_manager = self._sim is not None and self._sim._physics_manager is self
        callback_errors = self._dispatch_event_collect_errors(PhysicsEvent.STOP) if is_active_manager else []

        try:
            self.clear_callbacks()
        finally:
            if is_active_manager:
                self._sim = None
                self._sim_time = 0.0

        if callback_errors:
            raise RuntimeError(
                f"{len(callback_errors)} callback(s) failed during PhysicsEvent.STOP dispatch."
            ) from callback_errors[0]

    def _dispatch_event_collect_errors(self, event: PhysicsEvent, payload: Any = None) -> list[Exception]:
        """Dispatch an event to every listener and collect direct or backend-stored failures."""
        matching = [
            (callback, order)
            for registered_event, callback, order, _name in self._callbacks.values()
            if registered_event == event
        ]
        matching.sort(key=lambda item: item[1])
        callback_errors: list[Exception] = []

        for callback, _order in matching:
            try:
                callback(payload)
            except Exception as exc:
                callback_errors.append(exc)
        return callback_errors

    def get_physics_dt(self) -> float:
        """Get the physics timestep in seconds."""
        return self._sim.cfg.dt if self._sim else 1.0 / 60.0

    def get_device(self) -> str:
        """Get the physics simulation device."""
        return self._device

    def get_simulation_time(self) -> float:
        """Get the current simulation time in seconds."""
        return self._sim_time

    def get_physics_sim_view(self) -> Any:
        """Get the physics simulation view. Override in subclasses."""
        return None

    def play(self) -> None:
        """Start or resume physics simulation. Default is no-op."""
        pass

    def pause(self) -> None:
        """Pause physics simulation. Default is no-op."""
        pass

    def stop(self) -> None:
        """Stop physics simulation. Default is no-op."""
        pass

    def wait_for_playing(self) -> None:
        """Block until the timeline is playing. Default is no-op."""
        pass

    def set_decimation(self, decimation: int) -> None:
        """Inform the physics backend how many substeps the environment runs per policy step.

        Backends that can fold the full decimation loop into a single
        :meth:`step` call (e.g. Newton with all-graphable actuators) use this
        to size their internal loop / CUDA graph.  The default implementation
        is a no-op.

        Args:
            decimation: Number of physics steps per environment step.
        """
        pass

    def handles_decimation(self) -> bool:
        """``True`` when :meth:`step` executes the full decimation loop internally.

        When this returns ``True`` the environment should call :meth:`step`
        once per policy step instead of looping ``decimation`` times.
        """
        return False

    def get_backend(self) -> str:
        """Get the tensor backend being used ("numpy" or "torch")."""
        return "torch" if "cuda" in self._device else "numpy"
