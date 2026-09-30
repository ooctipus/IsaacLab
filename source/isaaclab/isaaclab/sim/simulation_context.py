# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gc
import logging
import traceback
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import fields
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import torch

import isaaclab.sim as sim_utils
import isaaclab.sim.utils.stage as stage_utils
from isaaclab.app.logging_utils import force_log_level
from isaaclab.app.settings_manager import SettingsManager
from isaaclab.markers.vis_marker_registry import VisMarkerRegistry
from isaaclab.physics import CallbackHandle, PhysicsEvent, PhysicsManager
from isaaclab.renderers import BaseRenderer, RendererCfg
from isaaclab.scene_data import SceneDataProvider
from isaaclab.sim.utils import create_new_stage
from isaaclab.utils.string import clear_resolve_matching_names_cache
from isaaclab.utils.version import has_kit
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.cloner.clone_plan import ClonePlan
    from isaaclab.sensors.camera import Camera

from .simulation_cfg import SimulationCfg

logger = logging.getLogger(__name__)

_BackendT = TypeVar("_BackendT")


def _apply_visualizer_defaults(default_cfg: VisualizerCfg | None, cfg: VisualizerCfg) -> None:
    """Apply explicitly customized shared defaults to one concrete visualizer cfg."""
    if default_cfg is None:
        return
    base_defaults = VisualizerCfg()
    concrete_defaults = type(cfg)()
    for field in fields(default_cfg):
        name = field.name
        if name in ("class_type", "visualizer_type") or not hasattr(cfg, name):
            continue
        value = getattr(default_cfg, name)
        if value != getattr(base_defaults, name) and getattr(cfg, name) == getattr(concrete_defaults, name):
            setattr(cfg, name, value)


class SimulationContext:
    """Controls simulation lifecycle including physics stepping and rendering.

    This singleton class manages:

    * Physics configuration (time-step, solver parameters via :class:`isaaclab.sim.SimulationCfg`)
    * Simulation state (play, pause, step, stop)
    * Rendering and visualization

    The singleton instance can be accessed using the ``instance()`` class method.
    """

    # SINGLETON PATTERN

    _instance: SimulationContext | None = None

    def __new__(cls, cfg: SimulationCfg):
        """Enforce singleton pattern."""
        if cls._instance is not None:
            raise RuntimeError("A SimulationContext already exists; use SimulationContext.instance().")
        return super().__new__(cls)

    @classmethod
    def instance(cls) -> SimulationContext | None:
        """Get the singleton instance, or None if not created."""
        return cls._instance

    def __init__(self, cfg: SimulationCfg):
        """Initialize the simulation context.

        Args:
            cfg: Simulation configuration with a concrete physics backend.
        """
        cfg.validate()
        from pxr import UsdUtils  # noqa: PLC0415

        # Store config
        self.cfg = cfg
        visualizer_cfgs = self.cfg.visualizer_cfgs
        self._visualizer_cfgs: list[Any] = visualizer_cfgs if isinstance(visualizer_cfgs, list) else [visualizer_cfgs]
        for visualizer_cfg in self._visualizer_cfgs:
            _apply_visualizer_defaults(self.cfg.default_visualizer_cfg, visualizer_cfg)

        use_isaac_sim = has_kit()
        kit_context = None
        if use_isaac_sim:
            import omni.usd

            kit_context = omni.usd.get_context()
        self._physics_manager: PhysicsManager = self.cfg.physics.class_type(self.cfg.physics)
        self._physics_manager._prepare_stage_creation()

        # Get or create stage based on config
        stage_cache = UsdUtils.StageCache.Get()
        if self.cfg.create_stage_in_memory:
            self.stage = create_new_stage()
        else:
            current = getattr(stage_utils._context, "stage", None)
            if current is not None:
                self.stage = current
            elif kit_context is not None:
                self.stage = kit_context.get_stage()
                if self.stage is None:
                    if not kit_context.new_stage():
                        raise RuntimeError("Kit failed to create the simulation stage.")
                    self.stage = kit_context.get_stage()
            else:
                self.stage = create_new_stage()

        # Ensure stage is in the USD cache
        stage_id = stage_cache.GetId(self.stage).ToLongInt()  # type: ignore[union-attr]
        if stage_id < 0:
            stage_cache.Insert(self.stage)  # type: ignore[union-attr]

        # Set as current stage in thread-local context for get_current_stage()
        stage_utils._context.stage = self.stage

        # When Kit is running, attach the stage to Kit's USD context so that
        # Kit extensions (PhysX views, Articulation, viewport) can discover it.
        if kit_context is not None:
            from pxr import Usd

            if kit_context.get_stage() is not self.stage:
                kit_context.attach_stage_with_callback(stage_cache.GetId(self.stage).ToLongInt())
            with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
                for path in (
                    "/OmniverseKit_Persp",
                    "/OmniverseKit_Front",
                    "/OmniverseKit_Top",
                    "/OmniverseKit_Right",
                ):
                    self.stage.RemovePrim(path)

        # Acquire settings interface (SettingsManager: standalone dict or Omniverse when available)
        self.settings = SettingsManager.instance()

        # Initialize USD physics scene and physics manager
        self._physics_scene_prim = self._init_usd_physics_scene()

        # Normalize "cuda" -> "cuda:<id>" now that the USD physics scene is initialized
        # and /physics/cudaDevice is available. Update cfg.device in-place so all
        # downstream code (physics backends, assets, sensors) sees a consistent value.
        if "cuda" in self.cfg.device and ":" not in self.cfg.device:
            cuda_device = self.get_setting("/physics/cudaDevice")
            device_id = max(0, int(cuda_device) if cuda_device is not None else 0)
            self.cfg.device = f"cuda:{device_id}"

        self._render_callbacks: dict[str, tuple[int, Callable[[Any], None]]] = {}
        self._backend_registry: dict[type[object], object] = {}
        self._backend_clone_roles: dict[type[object], set[str]] = {}
        self._camera_sensors: weakref.WeakValueDictionary[str, Any] = weakref.WeakValueDictionary()
        # Published once as a preliminary plan, then replaced once with its declared topology.
        self._clone_plan: ClonePlan | None = None
        self._physics_manager._bind_context(self)

        # Construct visualizers before cloning; initialize their runtime resources only after it.
        self._scene_data_provider = SceneDataProvider(self._physics_manager.get_scene_data_backend())
        self._renderer_entries: list[BaseRenderer] = []
        self._renderers_initialized = False
        self._visualizers: list[BaseVisualizer] = []
        self._uninitialized_visualizers: list[BaseVisualizer] = []
        self._reset_requested: bool = False
        # Default visualization dt used before/without visualizer initialization.
        self._viz_dt = self.cfg.dt * self.cfg.render_interval

        # Cache commonly-used settings (these don't change during runtime)
        self._has_gui = bool(self.get_setting("/isaaclab/has_gui"))
        self._has_offscreen_render = bool(self.get_setting("/isaaclab/render/offscreen"))
        self._xr_enabled = bool(self.get_setting("/isaaclab/xr/enabled"))
        # Renderer construction sets this process-global flag for the current context.
        self.set_setting("/isaaclab/render/rtx_sensors", False)
        self._pending_camera_view: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
        self.vis_marker_registry = VisMarkerRegistry()

        # Simulation state
        self._is_playing = False
        self._is_stopped = True

        type(self)._instance = self
        try:
            self._construct_visualizers()
            self._physics_manager.register_callback(
                self._initialize_renderers,
                PhysicsEvent.PHYSICS_READY,
                order=5,
                name="initialize_renderers",
            )
            self._physics_manager.register_callback(
                self.initialize_visualizers,
                PhysicsEvent.PHYSICS_READY,
                order=30,
                name="initialize_visualizers",
            )
        except Exception:
            type(self)._instance = None
            raise

    def _init_usd_physics_scene(self) -> Usd.Prim:
        """Create and configure the USD physics scene."""
        from pxr import Gf, UsdGeom, UsdPhysics  # noqa: PLC0415

        cfg = self.cfg
        with sim_utils.use_stage(self.stage):
            # Set stage conventions for metric units
            UsdGeom.SetStageUpAxis(self.stage, "Z")
            UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
            UsdPhysics.SetStageKilogramsPerUnit(self.stage, 1.0)

            physics_scene = UsdPhysics.Scene.Define(self.stage, cfg.physics_prim_path)

            # Pre-create gravity tensor to avoid torch heap corruption issues (torch 2.1+)
            gravity = torch.tensor(cfg.gravity, dtype=torch.float32, device=self.cfg.device)
            gravity_magnitude = torch.norm(gravity).item()

            if gravity_magnitude == 0.0:
                gravity_direction = [0.0, 0.0, -1.0]
            else:
                gravity_direction = (gravity / gravity_magnitude).tolist()

            physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(*gravity_direction))
            physics_scene.CreateGravityMagnitudeAttr(gravity_magnitude)
            return physics_scene.GetPrim()

    @property
    def physics_sim_view(self):
        """Returns the physics simulation view."""
        return self._physics_manager.get_physics_sim_view()

    @property
    def device(self) -> str:
        """Returns the device on which the simulation is running."""
        return self._physics_manager.get_device()

    @property
    def backend(self) -> str:
        """Returns the tensor backend being used ("numpy" or "torch")."""
        return self._physics_manager.get_backend()

    @property
    def has_gui(self) -> bool:
        """Returns whether GUI is enabled (cached at init)."""
        return self._has_gui

    @property
    def physics_backend(self) -> str:
        """Canonical identity declared by the resolved physics configuration."""
        return self.cfg.physics.backend

    @property
    def has_offscreen_render(self) -> bool:
        """Returns whether offscreen rendering is enabled (cached at init)."""
        return self._has_offscreen_render

    def has_active_visualizers(self) -> bool:
        """Return whether any visualizer path is active for rendering/camera control."""
        return bool(self._visualizers)

    def is_headless_or_exist_active_visualizer(self) -> bool:
        """Return whether the simulation should keep stepping without visualizers or with an active visualizer."""
        return not self._visualizers or any(viz.is_running() and not viz.is_closed for viz in self._visualizers)

    def can_render_rgb_array(self) -> bool:
        """Return whether rgb-array rendering is currently available."""
        return self.has_gui or self.has_offscreen_render or self.has_active_visualizers()

    @property
    def is_rendering(self) -> bool:
        """Returns whether *continuous* rendering is active (GUI, RTX sensors, visualizers, or XR).

        This drives the per-step render/Kit-pump loop, so it deliberately excludes headless
        offscreen rendering (``--video`` / ``rgb_array``). Offscreen frames are produced on
        demand when a frame is actually requested (via :meth:`render`), not on every step; see
        :meth:`has_offscreen_render` and :meth:`can_render_rgb_array` for the capability checks.
        """
        return (
            self._has_gui
            or self.get_setting("/isaaclab/render/rtx_sensors")
            or bool(self._visualizers)
            or self._xr_enabled
        )

    def get_physics_dt(self) -> float:
        """Returns the physics time step."""
        return self._physics_manager.get_physics_dt()

    def _configure_decimation(self, decimation: int) -> bool:
        """Configure backend-owned stepping and return whether it consumes the full loop."""
        self._physics_manager.set_decimation(decimation)
        return self._physics_manager.handles_decimation()

    def _register_physics_callback(
        self, callback: Callable[[Any], None], event: PhysicsEvent, *, order: int = 0, name: str | None = None
    ) -> CallbackHandle:
        """Register an internal physics-lifecycle consumer without exposing the manager."""
        return self._physics_manager.register_callback(callback, event, order=order, name=name)

    def _fix_articulation_root(self, articulation_prim: Any, stage: Any) -> Any:
        """Apply the active backend's articulation-root normalization."""
        return self._physics_manager.fix_articulation_root(articulation_prim, stage)

    def get_renderer(self, cfg: RendererCfg) -> BaseRenderer:
        """Construct and track the renderer for one camera configuration.

        Args:
            cfg: Renderer configuration from a camera.

        Returns:
            The newly constructed renderer.
        """
        if self._renderers_initialized or (self._clone_plan is not None and self._clone_plan.is_complete):
            raise RuntimeError("Renderers must be constructed from cfg before the clone plan completes.")
        if cfg.class_type is None:
            raise ValueError(f"{type(cfg).__name__}.class_type must name a renderer implementation.")
        renderer = cfg.class_type(cfg)
        if not isinstance(renderer, BaseRenderer):
            raise TypeError(f"{type(cfg).__name__}.class_type returned {type(renderer).__name__}.")
        self._renderer_entries.append(renderer)
        with force_log_level(logging.INFO):
            logger.info("Created new renderer for simulation: %s", type(renderer).__name__)
        return renderer

    def _initialize_renderers(self, _payload: Any = None) -> None:
        """Initialize the complete renderer set once, after cloning and before sensors."""
        if self._renderers_initialized:
            return
        if self._renderer_entries and (self._clone_plan is None or not self._clone_plan.is_complete):
            raise RuntimeError("Renderer initialization requires a completed clone plan.")
        self._renderers_initialized = True
        for renderer in self._renderer_entries:
            renderer.initialize()

    def get_or_create_backend(
        self,
        backend_type: type[_BackendT],
        *args: Any,
        clone_role: Literal["physics", "scene"] | None = None,
        **kwargs: Any,
    ) -> _BackendT:
        """Return the simulation-scoped native backend for ``backend_type``.

        Physics, renderers, and visualizers requesting the same backend type receive one shared
        resource rather than building state to synchronize.

        Args:
            backend_type: Backend class to construct when the resource is first registered.
            *args: Positional constructor arguments used when the backend is first registered.
            clone_role: ``"physics"`` when this consumer is the active physics engine, or
                ``"scene"`` when it needs the clone plan independently of physics.
            **kwargs: Keyword constructor arguments used when the backend is first registered.

        Returns:
            The existing or newly constructed native backend.
        """
        registered_roles = self._backend_clone_roles.get(backend_type, ())
        if (
            self._clone_plan is not None
            and self._clone_plan.is_complete
            and (
                backend_type not in self._backend_registry
                or (clone_role is not None and clone_role not in registered_roles)
            )
        ):
            raise RuntimeError("Backend resources and clone roles must be registered before the clone plan completes.")
        if backend_type not in self._backend_registry:
            self._backend_registry[backend_type] = backend_type(*args, **kwargs)
        if clone_role is not None:
            self._backend_clone_roles.setdefault(backend_type, set()).add(clone_role)
        return cast(_BackendT, self._backend_registry[backend_type])

    def _construct_visualizers(self) -> None:
        """Construct every requested visualizer without initializing runtime resources."""
        for cfg in self._visualizer_cfgs:
            if cfg.class_type is None:
                raise ValueError(f"{type(cfg).__name__}.class_type must name a visualizer implementation.")
            visualizer = cfg.class_type(cfg)
            if not isinstance(visualizer, BaseVisualizer):
                raise TypeError(f"{type(cfg).__name__}.class_type returned {type(visualizer).__name__}.")
            self._visualizers.append(visualizer)
            self._uninitialized_visualizers.append(visualizer)

    def initialize_visualizers(self, _payload: Any = None) -> None:
        """Initialize every visualizer constructed before cloning."""
        if not self._uninitialized_visualizers:
            return
        if self._clone_plan is None or self._clone_plan.env_ids is None or not self._clone_plan.is_complete:
            raise RuntimeError("Visualizers require the completed clone plan.")
        self._viz_dt = self.cfg.dt * self.cfg.render_interval

        waiting = list(self._uninitialized_visualizers)
        new_visualizers = []
        for visualizer in waiting:
            visualizer.initialize(self._scene_data_provider, self._clone_plan)
            self._uninitialized_visualizers.remove(visualizer)
            new_visualizers.append(visualizer)

        # Replay any camera pose requested before visualizers were initialized.
        pending = self._pending_camera_view
        if pending is not None:
            eye, target = pending
            for viz in new_visualizers:
                viz.set_camera_view(eye, target)
            if not self._uninitialized_visualizers:
                self._pending_camera_view = None

    def get_scene_data_provider(self) -> SceneDataProvider:
        return self._scene_data_provider

    def register_camera_sensor(self, camera: Camera) -> None:
        """Register a plan-validated camera for visualizer streaming."""
        self._camera_sensors[camera.cfg.prim_path] = camera

    def get_camera_sensors(self) -> dict[str, Camera]:
        """Return plan-validated cameras by prim-path expression."""
        return dict(self._camera_sensors)

    def get_clone_plan(self) -> ClonePlan | None:
        """Return the clone plan published by the scene.

        Set when a :class:`~isaaclab.cloner.ReplicateSession` opens, so an asset built inside the
        session can read where it is about to be cloned to. Registered backend resources, renderers,
        and visualizers consume that same plan. ``None`` until the scene starts replicating.
        """
        return self._clone_plan

    def set_clone_plan(self, plan: ClonePlan) -> None:
        """Publish the preliminary plan or refine it once with its declared topology."""
        if self._renderers_initialized:
            raise RuntimeError("The clone plan must be published before backend initialization.")
        if self._clone_plan is not None:
            construction_fields = (
                "sources",
                "destinations",
                "clone_mask",
                "env_ids",
                "positions",
                "env_template",
                "cfg_rows",
                "semantic_tags",
                "geometry_requests",
                "root_layer_identifier",
            )
            same_plan = all(getattr(self._clone_plan, name) is getattr(plan, name) for name in construction_fields)
            if self._clone_plan.is_complete or not plan.is_complete or not same_plan:
                raise RuntimeError("A SimulationContext owns exactly one clone-plan lifecycle.")
        self._clone_plan = plan
        if plan.is_complete:
            self._scene_data_provider._bind_point_plan(plan)

    @property
    def visualizers(self) -> list[BaseVisualizer]:
        """Returns the list of active visualizers."""
        return self._visualizers

    def get_rendering_dt(self) -> float:
        """Return rendering dt, allowing visualizer-specific override."""
        for viz in self._visualizers:
            viz_dt = viz.get_rendering_dt()
            if viz_dt is not None and viz_dt > 0:
                return float(viz_dt)
        return self._viz_dt

    def set_camera_view(self, eye: tuple, target: tuple) -> None:
        """Set camera view on all visualizers that support it."""
        self._pending_camera_view = (tuple(eye), tuple(target))
        for viz in self._visualizers:
            viz.set_camera_view(eye, target)

    def add_render_callback(self, name: str, fn: Callable[[Any], None], order: int = 0) -> None:
        """Register a callback to fire after every render step.

        Args:
            name: Unique identifier. Silently replaces any existing callback with the same name.
            fn: Callable invoked with a single ``None`` argument after each :meth:`render` call.
            order: Execution order relative to other callbacks. Lower values fire first.
        """
        self._render_callbacks[name] = (order, fn)

    def remove_render_callback(self, name: str) -> None:
        """Unregister a previously registered render callback.

        Args:
            name: Identifier passed to :meth:`add_render_callback`. No-op if not found.
        """
        self._render_callbacks.pop(name, None)

    def forward(self) -> None:
        """Update kinematics without stepping physics."""
        self._physics_manager.forward()

    def reset(self, soft: bool = False) -> None:
        """Reset the simulation.

        Args:
            soft: If True, skip full reinitialization.
        """
        self._physics_manager.reset(soft)
        for viz in self._visualizers:
            viz.reset(soft)
        # Start the timeline so the play button is pressed
        self._physics_manager.play()
        self._is_playing = True
        self._is_stopped = False

    def step(self, render: bool = True) -> None:
        """Step physics and optionally render.

        If the timeline is paused (e.g. via the GUI), this method blocks and keeps
        the visualizer responsive until the timeline is resumed or stopped.

        Args:
            render: Whether to render the scene after stepping. Defaults to True.
        """
        # Block while the GUI timeline is paused so the entire training loop freezes.
        # See: https://github.com/isaac-sim/IsaacLab/issues/4279
        self._physics_manager.wait_for_playing()
        self._physics_manager.step()
        if render and self.is_rendering:
            self.render()

    def render(self, skip_app_pumping: bool = False) -> None:
        """Update visualizers and render the scene.

        Visualizers run at render cadence, while camera sensors drive their configured renderer
        when fetching data. Registered render observers fire after visualizers update.

        **App-loop vs. standalone visualizers:**  The app loop (``app.update()``) is the
        only way to drive camera/RTX sensor rendering and viewport GUI updates; it
        cannot be split into "cameras only" and "GUI only". Standalone visualizers
        have self-contained ``step()`` methods that never call
        ``app.update()``, so they can run independently of camera rendering.  The
        ``skip_app_pumping`` flag exploits this distinction: when True, app-loop visualizers are skipped
        while standalone visualizers continue to update.

        Args:
            skip_app_pumping: When True, skip visualizers whose :meth:`~BaseVisualizer.pumps_app_update`
                returns True. This disables the app loop and camera updates while still stepping standalone visualizers.
                Used by environment ``step()`` when ``render_enabled`` is False.
        """
        if self._visualizers:
            for viz in self._visualizers:
                viz.flush_startup_messages()
            if any(
                viz.cfg.enable_markers or (viz.supports_live_plots() and viz.cfg.enable_live_plots)
                for viz in self._visualizers
            ):
                self.vis_marker_registry.dispatch_callbacks()

            visualizers_to_remove = []
            dt = self.get_rendering_dt()
            for viz in self._visualizers:
                if skip_app_pumping and viz.pumps_app_update():
                    continue
                if viz.is_closed or not viz.is_running():
                    logger.info(
                        "Visualizer %s: %s",
                        "closed" if viz.is_closed else "not running",
                        type(viz).__name__,
                    )
                    visualizers_to_remove.append(viz)
                    continue
                if viz.is_rendering_paused():
                    if not viz.pumps_app_update():
                        viz.step(0.0)
                    continue
                while viz.is_training_paused() and viz.is_running():
                    viz.step(0.0)
                viz.step(dt)

            for viz in visualizers_to_remove:
                viz.close()
                self._visualizers.remove(viz)
                logger.info("Removed visualizer: %s", type(viz).__name__)

        for _, callback in sorted(self._render_callbacks.values(), key=lambda x: x[0]):
            callback(None)

    def play(self) -> None:
        """Start or resume the simulation."""
        self._physics_manager.play()
        for viz in self._visualizers:
            viz.play()
        self._is_playing = True
        self._is_stopped = False

    def pause(self) -> None:
        """Pause the simulation (can be resumed with play)."""
        self._physics_manager.pause()
        for viz in self._visualizers:
            viz.pause()
        self._is_playing = False

    def stop(self) -> None:
        """Stop the simulation completely."""
        self._physics_manager.stop()
        for viz in self._visualizers:
            viz.stop()
        self._is_playing = False
        self._is_stopped = True

    def request_reset(self) -> None:
        """Request an episode reset from a UI control (e.g. the Kit window button).

        The request is consumed on the next call to :meth:`consume_reset_request`.
        """
        self._reset_requested = True

    def consume_reset_request(self) -> bool:
        """Return ``True`` if any visualizer or UI control requested an episode reset and clear the flag.

        Checks both the simulation-context-level flag (set by :meth:`request_reset`) and
        each visualizer's own flag. All flags are cleared atomically so a single reset
        is triggered even when multiple sources fire in the same step.

        Returns:
            ``True`` once when a reset was requested, then ``False`` until the next request.
        """
        requested = self._reset_requested
        self._reset_requested = False
        for viz in self._visualizers:
            requested |= viz.consume_reset_request()
        return requested

    def is_playing(self) -> bool:
        """Returns True if simulation is playing (not paused or stopped)."""
        return self._is_playing

    def is_stopped(self) -> bool:
        """Returns True if simulation is stopped (not just paused)."""
        return self._is_stopped

    def set_setting(self, name: str, value: Any) -> None:
        """Set a setting value."""
        self.settings.set(name, value)

    def get_setting(self, name: str) -> Any:
        """Get a setting value."""
        return self.settings.get(name)

    @classmethod
    def clear_instance(cls) -> None:
        """Clean up resources and clear the singleton instance."""
        instance = cls._instance
        if instance is not None:
            teardown_errors: list[Exception] = []

            def run_cleanup(callback: Callable[[], Any]) -> None:
                try:
                    callback()
                except Exception as exc:
                    teardown_errors.append(exc)

            try:
                # Close physics manager FIRST to detach PhysX from the stage.
                run_cleanup(instance._physics_manager.close)

                # Close camera renderers after STOP invalidates camera-owned render data and
                # before the stage is closed so stage-bound renderer resources remain valid.
                for renderer in list(instance._renderer_entries):
                    run_cleanup(renderer.close)
                instance._renderer_entries.clear()

                # Give every visualizer a chance to release its resources.
                for viz in list(instance._visualizers):
                    run_cleanup(viz.close)
                instance._visualizers.clear()
                instance._uninitialized_visualizers.clear()
                run_cleanup(instance.vis_marker_registry.close)
                for resource in instance._backend_registry.values():
                    clear = getattr(resource, "clear", None)
                    if clear is not None:
                        run_cleanup(clear)
                instance._backend_registry.clear()
                instance._backend_clone_roles.clear()
                instance._camera_sensors.clear()

                # Tear down the stage. We skip clear_stage() (prim-by-prim deletion) since
                # close_stage() + app shutdown destroy the entire stage at once.
                run_cleanup(stage_utils.close_stage)

                # Discard cached name-resolution data from destroyed assets.
                run_cleanup(clear_resolve_matching_names_cache)
            finally:
                cls._instance = None
                del instance

            run_cleanup(gc.collect)

            logger.info("SimulationContext cleared")

            if len(teardown_errors) == 1:
                raise teardown_errors[0]
            if teardown_errors:
                details = "; ".join(f"{type(error).__name__}: {error}" for error in teardown_errors)
                msg = (
                    f"SimulationContext.clear_instance(): {len(teardown_errors)} error(s) occurred during teardown:"
                    f" {details}"
                )
                raise RuntimeError(msg) from teardown_errors[0]

    @classmethod
    def clear_stage(cls) -> None:
        """Clear the current USD stage (preserving /World and PhysicsScene).

        Uses a predicate that preserves /World and PhysicsScene while also
        respecting the default deletability checks (ancestral prims, etc.).
        """
        if cls._instance is None:
            return

        def _predicate(prim: Usd.Prim) -> bool:
            path = prim.GetPath().pathString
            if path == "/World":
                return False
            if prim.GetTypeName() == "PhysicsScene":
                return False
            return True

        sim_utils.clear_stage(predicate=_predicate)


@contextmanager
def build_simulation_context(
    sim_cfg: SimulationCfg,
    *,
    create_new_stage: bool = True,
    device: str | None = None,
) -> Iterator[SimulationContext]:
    """Context manager to build a simulation context with the provided settings.

    Args:
        sim_cfg: Simulation configuration with a concrete physics backend.
        create_new_stage: Whether to create a new stage. Defaults to True.
        device: Device override. Defaults to ``None``, preserving ``sim_cfg.device``.

    Yields:
        The simulation context to use for the simulation.
    """
    sim: SimulationContext | None = None
    try:
        if create_new_stage:
            # ``create_new_stage`` is shadowed here by the bool parameter, so call via the namespace.
            sim_utils.create_new_stage()

        if device is not None:
            sim_cfg.device = device

        sim = SimulationContext(sim_cfg)

        yield sim

    except Exception:
        logger.error(traceback.format_exc())
        raise
    finally:
        if sim is not None:
            if not sim.get_setting("/isaaclab/has_gui"):
                sim.stop()
            sim.clear_instance()
