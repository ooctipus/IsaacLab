# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OvPhysX Manager for Isaac Lab.

This module manages an ovphysx-based physics simulation lifecycle without Kit dependencies.
It attaches the simulation's shared OV snapshot through OVStage and steps the simulation
using the ovphysx C/Python API.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import stat
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from pxr import Sdf, UsdPhysics

from isaaclab.cloner.clone_plan import ClonePlan
from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend, SceneDataFormat, SceneDataPublication

from isaaclab_ov._runtime import import_ovphysx
from isaaclab_ov.cloner import OvReplicateContext
from isaaclab_ov.stage import create_ovstage

from .ovphysx_compat import OVPHYSX_LIFECYCLE_ENTRY_POINTS
from .ovphysx_manager_cfg import DEFAULT_COOKED_COLLIDER_CACHE_DIR

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

    from .ovphysx_manager_cfg import OvPhysxCfg

__all__ = ["OvPhysxManager", "OvPhysxSceneDataBackend"]


def _prepare_default_cache_dir(cache_dir: str) -> str:
    """Create the default collider cache and reject an unsafe shared-temporary path."""
    try:
        os.makedirs(cache_dir, mode=0o700)
        return cache_dir
    except FileExistsError:
        pass
    entry = os.lstat(cache_dir)
    if stat.S_ISLNK(entry.st_mode):
        raise RuntimeError(f"OVPhysX cache directory '{cache_dir}' is a symlink; refusing to write through it.")
    if not stat.S_ISDIR(entry.st_mode):
        raise RuntimeError(f"OVPhysX cache directory '{cache_dir}' exists and is not a directory.")
    if hasattr(os, "getuid") and entry.st_uid != os.getuid():
        raise RuntimeError(f"OVPhysX cache directory '{cache_dir}' is owned by another user; refusing to use it.")
    return cache_dir


logger = logging.getLogger(__name__)
_locked_ovphysx_device: str | None = None


class OvPhysxSceneDataBackend(SceneDataBackend):
    """Publish OVPhysX state through bindings declared by the clone plan."""

    def __init__(self):
        self._rigid_view: Any = None
        self._rigid_spec: tuple[Any, tuple[str, ...], str] = (None, (), "")
        self._transform_publication = SceneDataPublication(SceneDataFormat.Transform(), dirty=True)
        self._point_publication = SceneDataPublication(SceneDataFormat.BodyPoints(), dirty=True)
        self._deformable_reads: list[tuple[Any, Any, wp.array]] = []

    def setup(self, physx: Any, plan: ClonePlan, device: str) -> None:
        """Prepare publications solely from the completed clone plan.

        Args:
            physx: Live OVPhysX instance.
            plan: Completed clone plan.
            device: Warp device string used to allocate native staging buffers.
        """
        self._rigid_view = None
        paths = tuple(plan.iter_rigid_body_paths())
        self._rigid_spec = (physx, paths, device)
        self._deformable_reads = []
        self._transform_publication.data.transforms = None
        self._point_publication.data.points = ()
        self._point_publication.data.binding_ids = ()
        self._transform_publication.dirty = True
        self._point_publication.dirty = True

        if paths:
            self._transform_publication.data.transforms = wp.zeros(len(paths), dtype=wp.transformf, device=device)

        self._setup_deformable_bindings(physx, plan, device)

    def _setup_deformable_bindings(self, physx: Any, plan: ClonePlan, device: str) -> None:
        """Wire exact plan-declared OVPhysX nodal-position bindings."""
        from isaaclab_ov import tensor_types as TT
        from isaaclab_ov.assets.deformable_object.views import OvPhysxDeformableBodyView

        if plan.point_clouds:
            raise RuntimeError("OVPhysX cannot publish the plan's authored point clouds.")
        point_bindings = plan.point_bindings()
        binding_ids = {binding.path: index for index, binding in enumerate(point_bindings)}
        point_buffers = []
        view_binding_ids = []

        for deformable_type in ("volume", "surface"):
            if deformable_type == "volume":
                sim_nodal_position_type = TT.DEFORMABLE_SIM_NODAL_POSITION
                sim_element_indices_type = TT.DEFORMABLE_SIM_ELEMENT_INDICES
                collision_element_indices_type = TT.DEFORMABLE_COLLISION_ELEMENT_INDICES
                tensor_types = [
                    sim_nodal_position_type,
                    sim_element_indices_type,
                    collision_element_indices_type,
                ]
            else:
                sim_nodal_position_type = TT.SURFACE_DEFORMABLE_SIM_POSITION
                sim_element_indices_type = TT.SURFACE_DEFORMABLE_SIM_ELEMENT_INDICES
                collision_element_indices_type = None
                tensor_types = [sim_nodal_position_type, sim_element_indices_type]

            groups: dict[int, list] = {}
            for entry in plan.deformables:
                if entry.deformable_type == deformable_type:
                    groups.setdefault(entry.vertex_count, []).append(entry)
            for declared_count, entries in groups.items():
                view = OvPhysxDeformableBodyView(
                    physx,
                    prim_paths=[entry.root_path for entry in entries],
                    device=device,
                    tensor_types=tensor_types,
                    eager=True,
                    simulation_nodal_position_type=sim_nodal_position_type,
                    simulation_element_indices_type=sim_element_indices_type,
                    collision_element_indices_type=collision_element_indices_type,
                )
                ordered = plan.match_deformables(deformable_type, view.prim_paths, entries)
                max_nodes = int(view.max_simulation_nodes_per_body)
                if max_nodes != declared_count:
                    raise RuntimeError(
                        f"Clone plan declares {declared_count} nodes for {[entry.root_path for entry in entries]!r}, "
                        f"but OVPhysX exposes {max_nodes}."
                    )
                position_buf = wp.zeros((view.count, max_nodes), dtype=wp.vec3f, device=device)
                self._deformable_reads.append((view, sim_nodal_position_type, position_buf))
                point_buffers.append(position_buf)
                view_binding_ids.append(
                    wp.array(
                        [binding_ids[entry.vis_mesh_path] for entry in ordered],
                        dtype=wp.int32,
                        device=device,
                    )
                )

        self._point_publication.data.points = tuple(point_buffers)
        self._point_publication.data.binding_ids = tuple(view_binding_ids)

    @property
    def point_publications(self) -> dict[str, SceneDataPublication]:
        """Return native padded deformable-body pointers and their dirty latch."""
        return {"points": self._point_publication}

    def _invalidate(self, *, points: bool = True) -> None:
        self._transform_publication.dirty = True
        self._point_publication.dirty |= points

    def _materialize(self, publication: SceneDataPublication) -> None:
        if publication is self._transform_publication and publication.data.transforms is not None:
            if self._rigid_view is None:
                from isaaclab_ov import tensor_types as TT
                from isaaclab_ov.sim.views.ovphysx_view import OvPhysxView

                physx, paths, device = self._rigid_spec
                view = OvPhysxView(
                    physx, prim_paths=list(paths), device=device, tensor_types=[TT.RIGID_BODY_POSE], eager=True
                )
                binding = view.binding_for(TT.RIGID_BODY_POSE)
                if tuple(binding.prim_paths) != paths or tuple(binding.shape) != (len(paths), 7):
                    view.close()
                    raise RuntimeError("OVPhysX rigid-body binding does not match the clone plan.")
                self._rigid_view = view
            self._rigid_view.read_into("rigid_body_pose", publication.data.transforms)
        elif publication is self._point_publication:
            for view, tensor_type, position_buf in self._deformable_reads:
                view.read_into(tensor_type, position_buf)

    @property
    def transform_publication(self) -> SceneDataPublication:
        """Return the current OVPhysX rigid-body pointer and dirty latch."""
        return self._transform_publication


class OvPhysxManager(PhysicsManager):
    """Manages an ovphysx-backed physics simulation lifecycle.

    Unlike PhysxManager, this manager does not depend on a host Kit or
    Carbonite runtime, or on the Omniverse timeline. It drives the simulation
    through the OVPhysX Python wheel and its packaged runtime.

    Lifecycle: construction -> clone -> reset() -> step() (repeated) -> close()
    """

    def __init__(self, cfg: OvPhysxCfg):
        super().__init__(cfg)
        self._physx = None
        self._ovstage = None
        self._warmup_done = False
        self._clone_ctx: OvReplicateContext | None = None
        self._atexit_registered = False
        self._scene_data_backend: OvPhysxSceneDataBackend | None = None
        self._physx_schemas_registered = False
        self._next_control_ordinal = 2
        self._gravity: tuple[float, float, float] | None = None

    def fix_articulation_root(self, articulation_prim: Any, stage: Any) -> Any:
        """Fix and normalize an articulation root for the OVPhysX parser."""
        root = super().fix_articulation_root(articulation_prim, stage)
        if root.HasAPI(UsdPhysics.RigidBodyAPI):
            return self._relocate_articulation_root(
                root,
                companion_schema="PhysxArticulationAPI",
                companion_namespace="physxArticulation",
            )
        return root

    def _prepare_stage_creation(self) -> None:
        """Register OvPhysX USD schemas before creating the selected backend's stage."""
        self._ensure_physx_schemas_registered()

    def _ensure_physx_schemas_registered(self) -> None:
        """Register the codeless USD plugins published by the OVPhysX wheel.

        The wheel's public paths include both ``PhysxSchema`` and
        ``OmniUsdPhysicsDeformableSchema``. A host may already provide one of
        those plugins from a compiled library, so only missing plugin names are
        registered from the wheel's codeless resource paths.
        """
        if self._physx_schemas_registered:
            return
        try:
            import ovphysx  # noqa: PLC0415

            from pxr import Plug  # noqa: PLC0415
        except ImportError:
            return
        registry = Plug.Registry()
        registered_names = {plugin.name.casefold() for plugin in registry.GetAllPlugins()}
        # The wheel documents ``<module>/resources`` as its stable layout and its
        # bundled plugin names match those module directory names case-insensitively.
        schema_paths = [
            str(path) for path in ovphysx.codeless_schema_paths() if path.parent.name.casefold() not in registered_names
        ]
        if schema_paths:
            registry.RegisterPlugins(schema_paths)
        self._physx_schemas_registered = True

    def _bind_context(self, sim_context: SimulationContext) -> None:
        """Bind the physics manager to its simulation before cloning.

        This stores the config and device but does not load the USD stage yet --
        the stage may not be fully populated at this point.  The actual load
        happens lazily in :meth:`reset`.

        ``self._physx`` is intentionally not cleared here: if the current
        :class:`SimulationContext` already constructed it and has not been
        closed, the manager reuses that instance. ``self._locked_device`` carries
        IsaacLab's conservative first-device policy for this process.
        """
        super()._bind_context(sim_context)
        self._clone_ctx = sim_context.get_or_create_backend(OvReplicateContext, sim_context, clone_role="physics")
        self._ensure_physx_schemas_registered()
        device = "gpu" if "cuda" in self._device else "cpu"
        self._configure_physx_scene_prim(sim_context._physics_scene_prim, self._cfg, device)
        self._gravity = tuple(sim_context.cfg.gravity)
        self._warmup_done = False
        # The provider captures this object now; clone-plan bindings are attached at warmup.
        self._scene_data_backend = OvPhysxSceneDataBackend()

    def reset(self, soft: bool = False) -> None:
        """Reset physics simulation.

        On the first (non-soft) reset the method:
        - Loads the clone context's cached USD snapshot
        - Creates the ovphysx.PhysX instance
        - Populates and attaches an OVStage
        - Warms up GPU buffers (if on CUDA)
        - Dispatches PHYSICS_READY

        A forced re-warm dispatches :attr:`~isaaclab.physics.PhysicsEvent.STOP`
        before replacing the attached stage so listeners discard stale bindings.
        """
        if not soft:
            if not self._warmup_done:
                if self._ovstage is not None:
                    self.dispatch_event(PhysicsEvent.STOP, payload={})
                self._warmup_and_load()
        self._scene_data_backend._invalidate()
        if not soft:
            self.dispatch_event(PhysicsEvent.PHYSICS_READY, payload={})

    def forward(self) -> None:
        """Refresh kinematics and invalidate transforms without stepping physics."""
        if self._physx is not None:
            self._physx.update_articulations_kinematic()
            self._scene_data_backend._invalidate(points=False)

    def step(self) -> None:
        """Step the simulation by one physics timestep."""
        if self._physx is None:
            return
        dt = self.get_physics_dt()
        self._step_physx(self._physx, dt=dt)
        self._sim_time += dt
        self._scene_data_backend._invalidate()

    @staticmethod
    def _step_physx(physx: Any, dt: float) -> None:
        """Step the pinned OVPhysX runtime synchronously."""
        physx.step_sync(dt=dt)

    @staticmethod
    def _reset_physx_stage(physx: Any) -> None:
        """Clear the loaded stage through the pinned OVPhysX runtime API."""
        operation = physx.reset_stage()
        physx.wait_op(operation)

    @staticmethod
    def _warmup_physx(physx: Any) -> None:
        """Warm a runtime through its version-selected API."""
        entry_point = OVPHYSX_LIFECYCLE_ENTRY_POINTS["warmup"]
        warmup = getattr(physx, entry_point, None)
        if warmup is None:
            raise AttributeError(f"OVPhysX does not expose the selected {entry_point}() lifecycle entry point")
        warmup()

    @staticmethod
    def _destroy_physx(physx: Any) -> None:
        """Tear a runtime down through its version-selected API."""
        entry_point = OVPHYSX_LIFECYCLE_ENTRY_POINTS["destroy"]
        destroy = getattr(physx, entry_point, None)
        if destroy is None:
            raise AttributeError(f"OVPhysX does not expose the selected {entry_point}() lifecycle entry point")
        destroy()

    def close(self) -> None:
        """Release ovphysx resources and clean up."""
        # Dispatch STOP while the runtime is still live. Asset and sensor callbacks
        # invalidate raw native handles before the view registry drains the remaining
        # binding caches and the runtime is released.
        try:
            super().close()
        finally:
            try:
                self._release_physx()
            finally:
                self._warmup_done = False
                self._clone_ctx = None
                self._gravity = None
                # Drop the SceneDataBackend singleton: its cached bindings and buffers
                # belong to the runtime instance just released. The next
                # SimulationContext re-creates it during binding.
                self._scene_data_backend = None

    def _release_physx(self) -> None:
        """Release the OVPhysX runtime instance and its owned OVStage.

        Safe to call multiple times. ``_locked_device`` intentionally survives
        release so later IsaacLab contexts keep the process's first device
        choice; this is required for CPU-first processes and conservative for
        GPU-first processes.
        """
        physx = self._physx
        if physx is None:
            self._destroy_ovstage()
            return

        # Preserve the legacy 0.5.11 behavior: release both owners even when
        # cleanup raises. Only OVPhysX 0.6 destroy failures can remain retryable.
        destroy_entry_point = OVPHYSX_LIFECYCLE_ENTRY_POINTS["destroy"]
        release_owners = destroy_entry_point == "release"
        try:
            try:
                self._close_physx_views(physx)
            finally:
                try:
                    self._reset_physx_stage(physx)
                finally:
                    try:
                        self._destroy_physx(physx)
                    except Exception:
                        if destroy_entry_point == "destroy":
                            # OVPhysX 0.6 keeps ``handle`` valid when destroy raises
                            # before native teardown. Preserve both owners so a later
                            # close can retry. A RuntimeError from ``handle`` means
                            # destruction reached its terminal state.
                            try:
                                physx.handle
                            except RuntimeError:
                                release_owners = True
                            except Exception:
                                # An unfamiliar handle probe must not replace the
                                # original destroy error or release either owner.
                                release_owners = False
                        raise
                    else:
                        release_owners = True
        finally:
            if release_owners:
                self._physx = None
                self._destroy_ovstage()

    def _attach_ovstage(self) -> None:
        """Populate and attach this simulation's shared OV snapshot."""
        import ovstage  # noqa: PLC0415

        clone_ctx = self._clone_ctx
        if clone_ctx is None:
            raise RuntimeError("OvPhysxManager has no simulation-scoped OV clone context.")
        stage = create_ovstage("isaaclab")
        try:
            ovstage.population.open_usd_from_string(
                stage,
                clone_ctx.stage_usda,
                ordinal=1,
                # FIXME: Use PHYSICS once OVStage includes native-instance collider
                # dependencies in physics-only population.
                domains=ovstage.PopulationDomain.ALL,
            )
            # ovphysx reads sealed data only: population completes the writes but never
            # commits the ordinal, so attaching at an unsealed ordinal fails the parse
            # and silently yields an empty scene.
            stage.advance_write_floor(ordinal=1).wait()
            self._physx.attach_ovstage(stage, read_ordinal=1)
        except Exception:
            stage.destroy()
            raise
        self._ovstage = stage
        self._next_control_ordinal = 2

    def _destroy_ovstage(self) -> None:
        """Destroy the attached OVStage after PhysX has released its stage."""
        if self._ovstage is not None:
            self._ovstage.destroy()
            self._ovstage = None
        self._next_control_ordinal = 2

    @staticmethod
    def _close_physx_views(physx: Any) -> None:
        """Destroy every cached :class:`~isaaclab_ov.sim.views.OvPhysxView` binding for ``physx``."""
        from isaaclab_ov.sim.views.ovphysx_view import OvPhysxView  # noqa: PLC0415

        OvPhysxView._close_all_for(physx)

    def _prepare_physx_for_stage_reuse(self) -> None:
        """Drain stage-bound handles before reusing the active runtime for another stage."""
        physx = self._physx
        if physx is None:
            return
        self._close_physx_views(physx)
        self._reset_physx_stage(physx)
        self._destroy_ovstage()

    def get_physx_instance(self) -> Any:
        """Return the underlying ovphysx.PhysX instance (or None if not yet created)."""
        return self._physx

    def get_gravity(self) -> tuple[float, float, float]:
        """Return the world-frame gravity vector [m/s^2] currently applied to the scene.

        Mirrors PhysX's ``SimulationView.get_gravity()`` so backend-agnostic sensor code can
        read gravity through one method. The value tracks :meth:`set_gravity`,
        falling back to the simulation cfg until the first live update.

        Raises:
            RuntimeError: If no simulation is active. Call :meth:`initialize` first.
        """
        if self._sim is None or not hasattr(self._sim, "cfg"):
            raise RuntimeError("OvPhysxManager has not been initialized yet.")
        return tuple(self._sim.cfg.gravity) if self._gravity is None else self._gravity

    def set_gravity(self, gravity: tuple[float, float, float]) -> None:
        """Set the scene-wide gravity vector through OVStage [m/s^2].

        Args:
            gravity: World-frame gravity vector [m/s^2].

        Raises:
            RuntimeError: If the OVPhysX simulation has not been initialized.
            ValueError: If gravity does not contain three finite values.
        """
        if self._sim is None or self._physx is None or self._ovstage is None:
            raise RuntimeError("OvPhysxManager has not been initialized yet.")
        gravity_array = np.asarray(gravity, dtype=np.float32)
        if gravity_array.shape != (3,) or not np.all(np.isfinite(gravity_array)):
            raise ValueError("Gravity must contain three finite values.")
        magnitude = float(np.linalg.norm(gravity_array))
        direction = (
            np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
            if magnitude == 0.0
            else (gravity_array / magnitude).reshape(1, 3)
        )
        ordinal = self._next_control_ordinal
        self._next_control_ordinal += 1

        import ovstage  # noqa: PLC0415

        with contextlib.ExitStack() as cleanup:
            paths = cleanup.enter_context(ovstage.PathDictionary(self._ovstage))
            path_list = paths.create_path_list_from_strings([self._sim.cfg.physics_prim_path])
            cleanup.callback(paths.destroy_path_list, path_list)
            query = cleanup.enter_context(self._ovstage.query_from_path_list(path_list))
            self._ovstage.write_attribute(query, "physics:gravityDirection", ordinal, direction, is_array=False).wait()
            self._ovstage.write_attribute(
                query, "physics:gravityMagnitude", ordinal, np.array([magnitude], dtype=np.float32), is_array=False
            ).wait()
            self._ovstage.advance_write_floor(ordinal=ordinal).wait()
            self._physx.update_from_ovstage(ordinal, ordinal)

        # Only publish once the ordinal has been applied, so a failed write leaves
        # :meth:`get_gravity` reporting the gravity the scene is still running with.
        self._gravity = (float(gravity_array[0]), float(gravity_array[1]), float(gravity_array[2]))

    def get_scene_data_backend(self) -> SceneDataBackend:
        """Return the plan-bound SceneDataBackend for the central SceneDataProvider."""
        return self._scene_data_backend

    def _warmup_and_load(self) -> None:
        """Attach the shared clone snapshot to the OVPhysX runtime.

        When no runtime is active, constructs a new :class:`ovphysx.PhysX`
        instance. The first construction also records IsaacLab's process device
        choice and registers process-exit cleanup. On a forced re-warm before
        :meth:`close`, it reuses the active instance, rebuilds its OVStage from the
        same snapshot and clone rows, and re-runs the version-selected warmup entry
        point so the new stage's bodies are resident.

        Raises:
            RuntimeError: If ``SimulationContext`` is not set, or if a device
                different from IsaacLab's first device choice is requested.
                OVPhysX CPU-only mode is process-wide and cannot be reversed;
                IsaacLab applies the same conservative policy in both
                directions for a predictable lifecycle.
        """
        sim = self._sim
        if sim is None:
            raise RuntimeError("OvPhysxManager: SimulationContext is not set.")
        plan = sim.get_clone_plan()
        if plan is None or not plan.is_complete:
            raise RuntimeError("OVPhysX initialization requires a completed clone plan.")

        device_str = self._device
        if "cuda" in device_str:
            parts = device_str.split(":")
            gpu_index = int(parts[1]) if len(parts) > 1 else 0
            ovphysx_device = "gpu"
        else:
            gpu_index = 0
            ovphysx_device = "cpu"

        global _locked_ovphysx_device
        if _locked_ovphysx_device is not None and ovphysx_device != _locked_ovphysx_device:
            raise RuntimeError(
                f"OvPhysxManager is locked to device {_locked_ovphysx_device!r} for the lifetime of this process; "
                f"cannot switch to {ovphysx_device!r}. IsaacLab pins the first OVPhysX device choice because "
                "CPU-only mode cannot be reversed; restart the process to use a different device."
            )

        if self._physx is None:
            self._construct_physx(ovphysx_device, gpu_index)
            _locked_ovphysx_device = ovphysx_device
        else:
            # Bindings are tied to the realized objects of one stage. Invalidate
            # asset/sensor handles and drain generic views before resetting the
            # cached runtime; PHYSICS_READY after this method rebuilds them.
            self._prepare_physx_for_stage_reuse()

        self._attach_ovstage()
        logger.info("OvPhysxManager: attached OVStage to ovphysx (device=%s)", ovphysx_device)
        for source, targets, transforms in self._clone_ctx.physics_clone_rows:
            operation = self._physx.clone(source, list(targets), list(transforms))
            self._physx.wait_op(operation)

        # GPU bodies must be re-warmed after every OVStage attachment: the cached PhysX
        # instance carries its old buffer layout from the previous stage.
        if ovphysx_device == "gpu":
            self._warmup_physx(self._physx)

        # Bind native views only after physics exists, using the plan published before cloning.
        self._scene_data_backend.setup(self._physx, plan, self._device)

        self.dispatch_event(PhysicsEvent.MODEL_INIT, payload={})
        self._clone_ctx._physics_initialized = True
        self._warmup_done = True

    def _construct_physx(self, ovphysx_device: str, gpu_index: int) -> None:
        """Bootstrap the ``ovphysx`` wheel and create the :class:`ovphysx.PhysX` instance.

        The pinned OVPhysX wheel documents :func:`ovphysx.bootstrap` as
        idempotent, so every explicit runtime construction invokes it. This
        method also configures worker threads, stores the result on
        ``self._physx``, and registers process-exit cleanup once.
        """
        ovphysx = import_ovphysx()
        ovphysx.bootstrap()
        cache_dir = self._cfg.cooked_collider_cache_dir
        if cache_dir == DEFAULT_COOKED_COLLIDER_CACHE_DIR:
            cache_dir = _prepare_default_cache_dir(cache_dir)
        self._physx = self._create_physx_instance(ovphysx, ovphysx_device, gpu_index, cache_dir)
        if not self._atexit_registered:
            # Globally retained environments may otherwise keep TensorBinding DLPack
            # caches alive until Python module finalization. Normal atexit cleanup runs
            # before that phase and preserves the process's real exit status.
            atexit.register(self._close_at_exit)
            self._atexit_registered = True

    def _close_at_exit(self) -> None:
        """Release a live OVPhysX runtime without leaking an atexit exception."""
        if self._physx is None:
            return
        try:
            self.close()
        except Exception:
            logger.exception("Failed to close OVPhysX during process exit.")

    @staticmethod
    def _create_physx_instance(
        ovphysx: Any, ovphysx_device: str, gpu_index: int, cooked_collider_cache_dir: str | None
    ) -> Any:
        """Create a PhysX instance through the pinned OVPhysX runtime API.

        Args:
            ovphysx: Imported OVPhysX runtime module.
            ovphysx_device: Physics device, either ``"cpu"`` or ``"gpu"``.
            gpu_index: CUDA device ordinal selected for GPU physics.
            cooked_collider_cache_dir: Directory for the cooked-collider cache, or ``None`` to use
                the runtime default.

        Returns:
            The configured ``ovphysx.PhysX`` instance.
        """

        carbonite_overrides = {
            "/physics/physxDispatcher": True,
            "/physics/updateToUsd": False,
            "/physics/updateVelocitiesToUsd": False,
            "/physics/updateParticlesToUsd": False,
        }
        if ovphysx_device == "gpu":
            carbonite_overrides.update(
                {
                    "/physics/suppressReadback": True,
                    "/physics/suppressFabricUpdate": True,
                }
            )
        ovphysx.PhysX.set_cpu_mode(ovphysx_device == "cpu")
        physx_kwargs = {
            "config": ovphysx.PhysXConfig(
                num_threads=8,
                cooked_collider_cache_dir=cooked_collider_cache_dir,
                carbonite_overrides=carbonite_overrides,
            ),
        }
        if ovphysx_device == "gpu":
            physx_kwargs["active_cuda_gpus"] = str(gpu_index)
        return ovphysx.PhysX(**physx_kwargs)

    def _configure_physx_scene_prim(self, scene_prim, cfg, device: str) -> None:
        """Apply PhysxSceneAPI schema and device-specific scene attributes to the
        scene prim.

        The PhysxSchema USD plugin may not be loaded in standalone ovphysx mode,
        so we write the apiSchemas list entry and scene attributes directly via
        raw Sdf metadata manipulation instead of using the high-level USD API.

        The schema, scene-query-support, and solver-determinism/accuracy attributes are applied
        regardless of device. The GPU-specific dynamics/broadphase/capacity attributes are
        applied only when ``device == "gpu"`` — without them PhysX defaults to
        CPU broadphase even when OVPhysX is configured for GPU execution.

        Args:
            scene_prim: The /World/PhysicsScene prim to configure.
            cfg: The :class:`OvPhysxCfg` carrying solver-determinism flags and GPU buffer-capacity
                values. The GPU buffer-capacity values are only consulted when ``device == "gpu"``.
            device: Resolved physics device — one of ``"cpu"`` or ``"gpu"``.
        """
        schemas = Sdf.TokenListOp()
        current = scene_prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
        items = list(current.prependedItems) if current.prependedItems else []
        if "PhysxSceneAPI" not in items:
            items.append("PhysxSceneAPI")
        schemas.prependedItems = items
        scene_prim.SetMetadata("apiSchemas", schemas)

        scene_prim.CreateAttribute("physxScene:enableSceneQuerySupport", Sdf.ValueTypeNames.Bool).Set(
            self._sim.cfg.enable_scene_query_support
        )
        scene_prim.CreateAttribute("physxScene:envIdInBoundsBitCount", Sdf.ValueTypeNames.Int).Set(4)

        if cfg is not None:
            # OvPhysX answers the backend-agnostic determinism request with enhanced determinism.
            # This is best-effort: reproducibility is not verified end to end.
            scene_prim.CreateAttribute("physxScene:enableEnhancedDeterminism", Sdf.ValueTypeNames.Bool).Set(
                cfg.enable_enhanced_determinism or cfg.deterministic
            )
            scene_prim.CreateAttribute("physxScene:enableExternalForcesEveryIteration", Sdf.ValueTypeNames.Bool).Set(
                cfg.enable_external_forces_every_iteration
            )

        if device == "gpu":
            scene_prim.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
            scene_prim.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.String).Set("GPU")

            if cfg is not None:
                for attr, val in [
                    ("gpuMaxRigidContactCount", cfg.gpu_max_rigid_contact_count),
                    ("gpuMaxRigidPatchCount", cfg.gpu_max_rigid_patch_count),
                    ("gpuFoundLostPairsCapacity", cfg.gpu_found_lost_pairs_capacity),
                    ("gpuFoundLostAggregatePairsCapacity", cfg.gpu_found_lost_aggregate_pairs_capacity),
                    ("gpuTotalAggregatePairsCapacity", cfg.gpu_total_aggregate_pairs_capacity),
                    ("gpuCollisionStackSize", cfg.gpu_collision_stack_size),
                ]:
                    scene_prim.CreateAttribute(f"physxScene:{attr}", Sdf.ValueTypeNames.UInt).Set(val)
