# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX Manager for Isaac Lab.

This module manages PhysX physics simulation lifecycle, configuration, callbacks, and physics views.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import time
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any

import torch
import warp as wp

import carb
import omni.kit.app
import omni.physics.tensors
import omni.physx
import omni.timeline
from pxr import Sdf, UsdPhysics, UsdUtils

import isaaclab.sim as sim_utils
from isaaclab.cloner.clone_plan import ClonePlan
from isaaclab.physics import PhysicsEvent, PhysicsManager
from isaaclab.scene_data import SceneDataBackend, SceneDataFormat, SceneDataPublication
from isaaclab.utils.string import to_camel_case

from isaaclab_physx.cloner import PhysxReplicateContext

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

    from .physx_manager_cfg import PhysxCfg

__all__ = ["PhysxManager"]

logger = logging.getLogger(__name__)


class AnimationRecorder:
    """Handles animation recording using PhysX PVD interface."""

    def __init__(self, sim_context: SimulationContext):
        self._sim = sim_context
        self._enabled = bool(sim_context.get_setting("/isaaclab/anim_recording/enabled"))
        self._started_at: float | None = None
        self._physx_pvd = None

        if self._enabled:
            self._start_time = sim_context.get_setting("/isaaclab/anim_recording/start_time")
            self._stop_time = sim_context.get_setting("/isaaclab/anim_recording/stop_time")
            self._setup_output_dir()

    def _setup_output_dir(self) -> None:
        """Initialize recording directory and PVD interface."""
        from omni.physxpvd.bindings import _physxPvd

        repo_path = os.path.join(carb.tokens.get_tokens_interface().resolve("${app}"), "..")
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self._output_dir = os.path.join(repo_path, "anim_recordings", timestamp).replace("\\", "/").rstrip("/") + "/"
        os.makedirs(self._output_dir, exist_ok=True)

        self._physx_pvd = _physxPvd.acquire_physx_pvd_interface()
        self._sim.set_setting("/persistent/physics/omniPvdOvdRecordingDirectory", self._output_dir)
        self._sim.set_setting("/physics/omniPvdOutputEnabled", True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def update(self) -> bool:
        """Update recording state. Returns True if recording finished."""
        if not self._enabled:
            return False
        if self._started_at is None:
            self._started_at = time.time()
        if time.time() - self._started_at > self._stop_time:
            self._finish()
            return True
        return False

    def _finish(self) -> None:
        """Finalize and export the recording."""
        logger.warning("[AnimationRecorder] Finishing recording. This may take a few minutes.")

        physx = omni.physx.get_physx_simulation_interface()
        physx.detach_stage()

        stage_path = os.path.join(self._output_dir, "stage_simulation.usdc")
        sim_utils.save_stage(stage_path, save_and_reload_in_place=False)

        ovd_files = [f for f in glob.glob(os.path.join(self._output_dir, "*.ovd")) if not f.endswith("tmp.ovd")]
        if ovd_files and self._physx_pvd:
            input_ovd = max(ovd_files, key=os.path.getctime)
            self._physx_pvd.ovd_to_usd_over_with_layer_creation(
                input_ovd,
                stage_path,
                self._output_dir,
                "baked_animation_recording.usda",
                self._start_time,
                self._stop_time,
                True,
                False,
            )
            self._update_usda_start_time(os.path.join(self._output_dir, "baked_animation_recording.usda"))

        self._sim.set_setting("/physics/omniPvdOutputEnabled", False)

    def _update_usda_start_time(self, file_path: str) -> None:
        """Patch the start time in the exported USDA file."""
        with open(file_path) as f:
            content = f.read()
        match = re.search(r"timeCodesPerSecond\s*=\s*(\d+)", content)
        if match:
            fps = int(match.group(1))
            new_start = int(self._start_time * fps)
            content = re.sub(r"startTimeCode\s*=\s*\d+", f"startTimeCode = {new_start}", content)
            with open(file_path, "w") as f:
                f.write(content)


class PhysxSceneDataBackend(SceneDataBackend):
    def __init__(self, device: str):
        self._device = device
        self._rigid_body_view: omni.physics.tensors.RigidBodyView | None = None
        self._transform_publication = SceneDataPublication(SceneDataFormat.Transform(), dirty=True)
        self._point_publication = SceneDataPublication(SceneDataFormat.BodyPoints(), dirty=True)
        self._deformable_views: list[Any] = []

    def setup(self, simulation_view: omni.physics.tensors.SimulationView, plan: ClonePlan) -> None:
        """Create native views solely from the completed clone plan."""
        self._rigid_body_view = None
        self._deformable_views = []
        self._transform_publication.data.transforms = None
        self._point_publication.data.points = ()
        self._point_publication.data.binding_ids = ()
        self._transform_publication.dirty = True
        self._point_publication.dirty = True

        if plan.rigid_body_prototypes:
            paths = list(plan.iter_rigid_body_paths())
            self._rigid_body_view = simulation_view.create_rigid_body_view(paths)
            if self._rigid_body_view.prim_paths != paths:
                raise RuntimeError("PhysX rigid-body view did not preserve clone-plan order.")
            self._transform_publication.data.transforms = self._rigid_body_view.get_transforms().view(wp.transformf)

        if plan.point_clouds:
            raise RuntimeError("PhysX cannot publish the plan's authored point clouds.")
        point_bindings = plan.point_bindings()
        binding_ids = {binding.path: index for index, binding in enumerate(point_bindings)}
        view_binding_ids = []

        for deformable_type in ("volume", "surface"):
            entries = tuple(entry for entry in plan.deformables if entry.deformable_type == deformable_type)
            if not entries:
                continue
            patterns = list(dict.fromkeys(entry.view_path for entry in entries))
            if deformable_type == "volume":
                view = simulation_view.create_volume_deformable_body_view(patterns)
            else:
                view = simulation_view.create_surface_deformable_body_view(patterns)
            if view is None or view._backend is None:
                raise RuntimeError(f"PhysX did not create the declared {deformable_type} deformable view.")

            ordered_entries = plan.match_deformables(deformable_type, view.prim_paths)
            max_nodes = int(view.max_simulation_nodes_per_body)
            for entry in ordered_entries:
                count = entry.vertex_count
                if count > max_nodes:
                    raise RuntimeError(
                        f"Clone plan declares {count} nodes for {entry.root_path!r}, but PhysX exposes {max_nodes}."
                    )
            self._deformable_views.append(view)
            view_binding_ids.append(
                wp.array(
                    [binding_ids[entry.vis_mesh_path] for entry in ordered_entries],
                    dtype=wp.int32,
                    device=self._device,
                )
            )

        self._point_publication.data.points = tuple(
            view.get_simulation_nodal_positions().view(wp.vec3f) for view in self._deformable_views
        )
        self._point_publication.data.binding_ids = tuple(view_binding_ids)

    def publish(self, *, transforms: bool = True, points: bool = True) -> None:
        """Publish current native pointers at the physics lifecycle boundary."""
        if transforms:
            if self._rigid_body_view is not None:
                self._rigid_body_view.get_transforms()
            self._transform_publication.dirty = True
        if points:
            for view in self._deformable_views:
                view.get_simulation_nodal_positions()
            self._point_publication.dirty = True

    @property
    def point_publications(self) -> dict[str, SceneDataPublication]:
        """Return native padded deformable-body pointers and their dirty latch."""
        return {"points": self._point_publication}

    @property
    def transform_publication(self) -> SceneDataPublication:
        """Return the current PhysX rigid-body pointer and dirty latch."""
        return self._transform_publication


class PhysxManager(PhysicsManager):
    """Manages PhysX physics simulation lifecycle.

    Lifecycle: construction -> clone -> reset() -> step() (repeated) -> close()
    """

    supports_anim_recording = True

    def __init__(self, cfg: PhysxCfg):
        super().__init__(cfg)
        self._timeline: omni.timeline.ITimeline = omni.timeline.get_timeline_interface()
        self._scene_data_backend: PhysxSceneDataBackend | None = None
        self._view: omni.physics.tensors.SimulationView | None = None
        self._articulation_views: dict[str, omni.physics.tensors.ArticulationView] = {}
        self._subscriptions: dict[str, Any] = {}
        self._anim_recorder: AnimationRecorder | None = None

    def _bind_context(self, sim_context: SimulationContext) -> None:
        """Bind the manager and register resources needed during cloning."""
        from isaaclab_physx import _patch_isaacsim_simulation_manager, _subscribe_to_simulation_manager_enable

        _subscribe_to_simulation_manager_enable()
        _patch_isaacsim_simulation_manager()

        super()._bind_context(sim_context)
        from isaaclab.cloner import UsdReplicateContext  # noqa: PLC0415

        sim_context.get_or_create_backend(UsdReplicateContext, sim_context.stage, clone_role="physics")
        sim_context.get_or_create_backend(PhysxReplicateContext, sim_context.stage, clone_role="physics")
        self._setup_subscriptions()
        self._configure_physics()
        self._anim_recorder = AnimationRecorder(sim_context)
        self._scene_data_backend = PhysxSceneDataBackend(self._device)

        # force update cycle to apply dt
        sim = self._sim
        sim.set_setting("/app/player/playSimulations", False)  # type: ignore[union-attr]
        omni.kit.app.get_app().update()
        sim.set_setting("/app/player/playSimulations", True)  # type: ignore[union-attr]

    def fix_articulation_root(self, articulation_prim: Any, stage: Any) -> Any:
        """Fix and normalize an articulation root for the PhysX parser."""
        root = super().fix_articulation_root(articulation_prim, stage)
        if root.HasAPI(UsdPhysics.RigidBodyAPI):
            return self._relocate_articulation_root(
                root,
                companion_schema="PhysxArticulationAPI",
                companion_namespace="physxArticulation",
            )
        return root

    def reset(self, soft: bool = False) -> None:
        """Reset the physics simulation."""
        if not soft and self._view is None:
            self._warmup_and_create_views()

        device = self._device
        if "cuda" in device:
            torch.cuda.set_device(device)

        if self._view is not None:
            self._view._backend.initialize_kinematic_bodies()

        self._scene_data_backend.publish()

    def forward(self) -> None:
        """Update articulation kinematics.

        Does not push state into Fabric. Consumers request transforms through the scene data
        provider, which converts the current pointer only when needed.
        """
        sim = self._sim
        if self._view is not None and sim is not None and sim.is_playing():
            self._view.update_articulations_kinematic()
            self._scene_data_backend.publish(points=False)

    def get_scene_data_backend(self) -> SceneDataBackend:
        """Return the SceneDataBackend for the SceneDataProvider."""
        return self._scene_data_backend

    def step(self) -> None:
        """Step the physics simulation."""
        sim = self._sim
        if sim is None:
            return

        if self._anim_recorder and self._anim_recorder.enabled and self._anim_recorder.update():
            logger.warning("Animation recording finished. Shutting down.")
            omni.kit.app.get_app().shutdown()
            return

        physx_sim = omni.physx.get_physx_simulation_interface()
        physx_sim.simulate(sim.cfg.dt, 0.0)
        physx_sim.fetch_results()
        self._sim_time += sim.cfg.dt
        device = self._device
        if "cuda" in device:
            torch.cuda.set_device(device)
        self._scene_data_backend.publish()

    def play(self) -> None:
        """Start or resume the timeline."""
        self._timeline.play()
        # Pump events so timeline callbacks fire synchronously
        omni.kit.app.get_app().update()
        if self._view is not None:
            self._scene_data_backend.publish(points=False)

    def pause(self) -> None:
        """Pause the timeline."""
        self._timeline.pause()
        # Pump events so timeline callbacks fire synchronously
        omni.kit.app.get_app().update()

    def stop(self) -> None:
        """Stop the timeline."""
        self._timeline.stop()
        # Pump events so timeline callbacks fire synchronously
        omni.kit.app.get_app().update()

    def wait_for_playing(self) -> None:
        """Block until the timeline is playing, keeping the GUI responsive."""
        if self._timeline.is_playing():
            return
        app = omni.kit.app.get_app()
        while not self._timeline.is_playing():
            app.update()
            if self._timeline.is_stopped():
                break
        if self._view is not None:
            self._scene_data_backend.publish(points=False)

    def close(self) -> None:
        """Clean up physics resources."""
        # Detach PhysX from the stage FIRST to prevent shape/actor cleanup errors
        # This disconnects PhysX from USD before any deletion events are fired
        if physx_sim := omni.physx.get_physx_simulation_interface():
            physx_sim.detach_stage()
            # Pump the app to flush pending PhysX cleanup operations
            omni.kit.app.get_app().update()

        # Now invalidate views (they're already disconnected from PhysX)
        self._invalidate_views()
        self._subscriptions.clear()

        self.dispatch_event(PhysicsEvent.PRIM_DELETION, payload={"prim_path": "/"})

        self._anim_recorder = None
        super().close()

    def get_physics_sim_view(self) -> omni.physics.tensors.SimulationView | None:
        return self._view

    def _setup_subscriptions(self) -> None:
        """Subscribe to timeline events."""
        if "play" in self._subscriptions:
            return
        stream = self._timeline.get_timeline_event_stream()
        self._subscriptions["play"] = stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY), self._on_play
        )
        self._subscriptions["stop"] = stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP), self._on_stop
        )

    def _configure_physics(self) -> None:
        """Apply all physics settings."""
        sim = self._sim
        cfg = self._cfg
        if sim is None or cfg is None:
            return

        device = sim.device
        is_gpu = "cuda" in device
        if cfg.enable_ccd and is_gpu:
            raise ValueError("PhysxCfg.enable_ccd is unsupported with GPU dynamics; disable CCD or use CPU PhysX.")

        # global settings (via SettingsManager)
        sim.set_setting("/persistent/omnihydra/useSceneGraphInstancing", True)  # type: ignore[union-attr]
        sim.set_setting("/physics/physxDispatcher", True)  # type: ignore[union-attr]
        sim.set_setting("/physics/disableContactProcessing", True)  # type: ignore[union-attr]
        sim.set_setting("/physics/collisionConeCustomGeometry", False)  # type: ignore[union-attr]
        sim.set_setting("/physics/collisionCylinderCustomGeometry", False)  # type: ignore[union-attr]
        sim.set_setting("/physics/autoPopupSimulationOutputWindow", False)  # type: ignore[union-attr]
        for key in (
            "updateToUsd",
            "updateParticlesToUsd",
            "updateVelocitiesToUsd",
            "updateForceSensorsToUsd",
            "updateResidualsToUsd",
        ):
            sim.set_setting(f"/physics/{key}", False)  # type: ignore[union-attr]
        sim.set_setting("/physics/visualizationDisplaySimulationOutput", False)  # type: ignore[union-attr]

        # Device setup for this manager instance.
        if is_gpu:
            parts = device.split(":")
            cuda_device = sim.get_setting("/physics/cudaDevice")  # type: ignore[union-attr]
            device_id = int(parts[1]) if len(parts) > 1 else max(0, int(cuda_device) if cuda_device is not None else 0)
            sim.set_setting("/physics/cudaDevice", device_id)  # type: ignore[union-attr]
            sim.set_setting("/physics/suppressReadback", True)  # type: ignore[union-attr]
            self._device = f"cuda:{device_id}"
        else:
            sim.set_setting("/physics/cudaDevice", -1)  # type: ignore[union-attr]
            sim.set_setting("/physics/suppressReadback", False)  # type: ignore[union-attr]
            self._device = "cpu"

        # physx scene api (use sim.cfg for shared parameters like physics_prim_path, dt, physics_material)
        # apply schema and set attributes by name
        sim_cfg = sim.cfg
        scene_prim = sim._physics_scene_prim
        if "PhysxSceneAPI" not in scene_prim.GetAppliedSchemas():
            scene_prim.AddAppliedSchema("PhysxSceneAPI")
        scene_prim.CreateAttribute("physxScene:envIdInBoundsBitCount", Sdf.ValueTypeNames.Int).Set(4)

        # timestep
        steps_per_sec = int(1.0 / sim_cfg.dt)
        sim_utils.safe_set_attribute_on_usd_prim(
            scene_prim, "physxScene:timeStepsPerSecond", steps_per_sec, camel_case=False
        )
        # gpu dynamics
        sim_utils.safe_set_attribute_on_usd_prim(
            scene_prim, "physxScene:broadphaseType", "GPU" if is_gpu else "MBP", camel_case=False
        )
        sim_utils.safe_set_attribute_on_usd_prim(scene_prim, "physxScene:enableGPUDynamics", is_gpu, camel_case=False)

        sim_utils.safe_set_attribute_on_usd_prim(scene_prim, "physxScene:enableCCD", cfg.enable_ccd, camel_case=False)

        # solver
        sim_utils.safe_set_attribute_on_usd_prim(
            scene_prim, "physxScene:solverType", "TGS" if cfg.solver_type == 1 else "PGS", camel_case=False
        )
        scene_prim.CreateAttribute("physxScene:solveArticulationContactLast", Sdf.ValueTypeNames.Bool).Set(
            cfg.solve_articulation_contact_last
        )

        sim_utils.safe_set_attribute_on_usd_prim(
            scene_prim,
            "physxScene:enableSceneQuerySupport",
            sim_cfg.enable_scene_query_support,
            camel_case=False,
        )

        # PhysX answers the backend-agnostic determinism request with enhanced determinism. An
        # explicitly enabled flag stays enabled.
        if cfg.deterministic:
            cfg.enable_enhanced_determinism = True

        # apply remaining cfg attributes to scene (physxScene:*)
        skip = {
            # generic request, translated above; PhysX has no physxScene:deterministic attribute
            "deterministic",
            "solver_type",
            "enable_ccd",
            "solve_articulation_contact_last",
            "class_type",
            "backend",
        }
        for key, value in cfg.to_dict().items():  # type: ignore
            if key not in skip:
                attr_name = "bounce_threshold" if key == "bounce_threshold_velocity" else key
                sim_utils.safe_set_attribute_on_usd_prim(
                    scene_prim,
                    f"physxScene:{to_camel_case(attr_name, 'cC')}",
                    value,
                    camel_case=False,
                )

        # default physics material
        mat_path = f"{sim_cfg.physics_prim_path}/defaultMaterial"
        sim_cfg.physics_material.func(mat_path, sim_cfg.physics_material)
        sim_utils.bind_physics_material(sim_cfg.physics_prim_path, mat_path)

        # warnings
        if not cfg.enable_external_forces_every_iteration:
            warning_message = (
                "PhysxCfg.enable_external_forces_every_iteration is deprecated and will be removed in a future "
                "PhysX release. External forces are applied every iteration by default; remove this override."
            )
            if cfg.solver_type == 1:
                warning_message += " Disabling this behavior with the TGS solver may cause noisy velocities."
            warnings.warn(
                warning_message,
                DeprecationWarning,
                stacklevel=2,
            )
        if not cfg.enable_stabilization and sim_cfg.dt > 0.0333:
            logger.warning("Large timestep without stabilization may cause physics issues.")

    def _warmup_and_create_views(self) -> None:
        """Warm-start physics and create simulation views."""
        if self._view is not None:
            return
        plan = self._sim.get_clone_plan()
        if plan is None or not plan.is_complete:
            raise RuntimeError("PhysX initialization requires a completed clone plan.")

        stage_id = UsdUtils.StageCache.Get().GetId(self._sim.stage).ToLongInt()

        is_gpu = "cuda" in self.get_device()

        physx = omni.physx.get_physx_interface()
        physx_sim = omni.physx.get_physx_simulation_interface()

        # Attach stage to PhysX BEFORE loading/starting - only needed for GPU pipeline.
        # For CPU, the old SimulationManager never called attach_stage() explicitly.
        # Calling attach_stage() + force_load_physics_from_usd() together causes a
        # double-initialization that corrupts the CPU broadphase (MBP) collision setup,
        # causing objects to fall through surfaces non-deterministically.
        if is_gpu:
            physx_sim.attach_stage(stage_id)

        # warmup physx
        self.dispatch_event(PhysicsEvent.MODEL_INIT, payload={})
        physx.force_load_physics_from_usd()
        physx.start_simulation()
        physx.update_simulation(self.get_physics_dt(), 0.0)
        physx_sim.fetch_results()

        # Create tensor views
        self._view = omni.physics.tensors.create_simulation_view("warp", stage_id=stage_id)
        if self._view is None:
            raise RuntimeError("PhysX did not create the simulation view required by scene data.")
        self._view.set_subspace_roots("/")

        # Final update after view creation
        physx.update_simulation(self.get_physics_dt(), 0.0)
        self._scene_data_backend.setup(self._view, plan)

        self.dispatch_event(PhysicsEvent.PHYSICS_READY, payload={})

    def _invalidate_views(self) -> None:
        """Invalidate and clear simulation views."""
        if self._view:
            self._view.invalidate()
        self._view = None
        self._articulation_views = {}

    def _on_play(self, event: Any) -> None:
        sim = self._sim
        if sim is not None and sim.get_setting("/app/player/playSimulations"):  # type: ignore[union-attr]
            self._warmup_and_create_views()

    def _on_stop(self, event: Any) -> None:
        self.dispatch_event(PhysicsEvent.STOP, payload={})
        omni.physx.get_physx_simulation_interface().detach_stage()
        self._invalidate_views()
