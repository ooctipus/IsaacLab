# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-based visualizer using Isaac Sim viewport."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from pxr import Gf, UsdGeom

from isaaclab import cloner
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.cloner import UsdReplicateContext
from isaaclab.cloner.cloner_cfg import expand_env_regex_ns
from isaaclab.scene_data import SceneDataFormat
from isaaclab.sim import SimulationContext
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.streaming_view import StreamingView

from .kit_visualization_markers import KitVisualizationMarkers
from .kit_visualizer_cfg import KitVisualizerCfg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.cloner import ClonePlan
    from isaaclab.scene_data import SceneDataProvider

_DEFAULT_VIEWPORT_NAME = "Visualizer Viewport"


class KitVisualizer(BaseVisualizer):
    """Kit visualizer using Isaac Sim viewport."""

    marker_type = KitVisualizationMarkers

    def __init__(self, cfg: KitVisualizerCfg):
        """Initialize Kit visualizer state.

        Args:
            cfg: Kit visualizer configuration.
        """
        if cfg.dock_position.upper() not in {"LEFT", "RIGHT", "BOTTOM", "SAME"}:
            raise ValueError(f"[KitVisualizer] Unknown dock_position: {cfg.dock_position!r}.")
        if cfg.max_visible_envs is not None or cfg.visible_env_indices is not None:
            raise ValueError("KitVisualizer does not support partial environment visibility.")
        super().__init__(cfg)
        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError("KitVisualizer requires an active SimulationContext.")
        self._clone_ctx = sim.get_or_create_backend(UsdReplicateContext, sim.stage, clone_role="scene")
        self.cfg: KitVisualizerCfg = cfg
        camera = UsdGeom.Camera.Define(sim.stage, cfg.prim_path)
        camera.GetFocalLengthAttr().Set(cfg.focal_length)
        self._camera_xform_op = UsdGeom.Xformable(camera).MakeMatrixXform()

        self._simulation_app = None
        self._viewport_window = None
        self._viewport_api = None
        self._is_initialized = False
        self._sim_time = 0.0
        self._step_counter = 0
        self._runtime_headless = bool(cfg.headless)
        # Lazy Replicator render product + annotator for render_rgb_array().
        self._rgb_render_product = None
        self._rgb_annotator = None
        self._streaming: StreamingView | None = None
        self._camera_image_provider = None
        self._camera_image_window = None
        self._camera_gpu_upload_tensor = None
        # Guard flag: True once app.update() has been called in the current step() invocation.
        # render_rgb_array() skips its own pump when the step already pumped the app.
        self._app_pumped_this_step: bool = False
        self._viewer_origin: torch.Tensor | None = None  # world-space origin offset for eye/lookat
        self._origin_index: int | None = None
        self._set_viewport_camera(cfg.eye, cfg.lookat)

    # ---- Lifecycle ------------------------------------------------------------------------

    @property
    def visual_material_writer(self):
        """Write material channels directly through Fabric."""
        return self._clone_ctx.create_fabric_visual_material_writer

    def initialize(self, scene_data_provider: SceneDataProvider, clone_plan: ClonePlan) -> None:
        """Initialize viewport resources and bind scene data provider.

        Args:
            scene_data_provider: Scene data provider used by the visualizer.
            clone_plan: Shared plan describing every cloned scene row.
        """
        if self._is_initialized:
            logger.debug("[KitVisualizer] initialize() called while already initialized.")
            return

        scene_data_provider = self._set_scene_data_provider(scene_data_provider, clone_plan)
        sim = SimulationContext.instance()
        if not clone_plan.is_complete:
            raise RuntimeError("KitVisualizer requires a completed clone plan.")
        if id(self.cfg) not in clone_plan.cfg_rows:
            raise RuntimeError("KitVisualizer camera is not covered by the clone plan.")
        self._clone_ctx._prepare_fabric(scene_data_provider, sim.cfg.device, clone_plan)
        num_envs = len(clone_plan.env_ids)

        self._ensure_simulation_app()
        self._setup_viewport()

        self._log_initialization_table(
            logger=logger,
            title="KitVisualizer Configuration",
            rows=[
                ("eye", self.cfg.eye),
                ("lookat", self.cfg.lookat),
                ("streaming_view", self.cfg.streaming_view),
                ("streaming_gt_types", list(self.cfg.streaming_gt_types)),
                ("num_visualized_envs", num_envs),
                ("create_viewport", self.cfg.create_viewport),
                ("headless", self._runtime_headless),
            ],
        )
        self._setup_streaming_view()

        self._is_initialized = True
        self._setup_initial_camera_view()

    def step(self, dt: float) -> None:
        """Advance visualizer/UI updates for one simulation step.

        Args:
            dt: Simulation time-step in seconds.
        """
        if not self._is_initialized:
            return
        self._app_pumped_this_step = False
        self._sim_time += dt
        self._step_counter += 1
        if self._runtime_headless:
            return
        self._prepare_viewport_frame()

        _externally_paused = self.is_training_paused()
        if not _externally_paused:
            import omni.kit.app

            app = omni.kit.app.get_app()
            if app is None or not app.is_running():
                raise RuntimeError("[KitVisualizer] Isaac Sim app is not running.")
            # Keep app pumping for viewport/UI updates only; physics is owned by SimulationContext.
            settings = get_settings_manager()
            settings.set_bool("/app/player/playSimulations", False)
            try:
                app.update()
            finally:
                settings.set_bool("/app/player/playSimulations", True)
            self._app_pumped_this_step = True
        self._update_camera_image_panel()

    def _prepare_viewport_frame(self) -> None:
        """Write the current SDP state to Fabric before a viewport frame."""
        self._scene_data_provider.request_transforms(SceneDataFormat.FabricMatrix44)
        self._clone_ctx._update_fabric_hierarchy()
        for stream in self._clone_plan.point_stream_names:
            self._scene_data_provider.request_points(SceneDataFormat.FabricMeshPoints, stream)
        if self.cfg.origin_type == "asset":
            self._update_asset_tracking_camera()

    def close(self) -> None:
        """Close viewport resources."""
        if not self._is_initialized:
            return
        self._streaming = None
        self._camera_image_provider = None
        self._camera_image_window = None
        self._simulation_app = None
        self._viewport_window = None
        self._viewport_api = None
        import contextlib

        if self._rgb_annotator is not None:
            with contextlib.suppress(Exception):
                self._rgb_annotator.detach()
        if self._rgb_render_product is not None:
            with contextlib.suppress(Exception):
                self._rgb_render_product.destroy()
        self._rgb_annotator = None
        self._rgb_render_product = None
        self._is_initialized = False
        self._is_closed = True

    def render_rgb_array(self) -> np.ndarray:
        """Return an RGB frame captured from the Kit viewport camera.

        Uses the Replicator annotator bound to the camera declared by :class:`KitVisualizerCfg`.
        Lazily creates the render product and annotator on the first call. Returns a blank frame
        while the RTX pipeline warms up.

        Returns:
            RGB image array of shape ``(window_height, window_width, 3)``, dtype ``uint8``.
        """
        camera_path = self.cfg.prim_path
        import omni.kit.app
        import omni.replicator.core as rep

        w, h = self.cfg.window_width, self.cfg.window_height

        # Create the render product and annotator before the app update so the first
        # captured frame contains real rendered output, not empty/blank data.
        if self._rgb_annotator is None:
            self._rgb_render_product = rep.create.render_product(camera_path, (w, h))
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([self._rgb_render_product])
        elif self._runtime_headless and self._rgb_render_product is not None:
            # In headless mode the render product is paused between captures (see below).
            # Resume it now so RTX can produce a fresh frame before we read the annotator.
            self._rgb_render_product.resume()

        if self._runtime_headless:
            self._prepare_viewport_frame()
        if not self._app_pumped_this_step:
            settings = get_settings_manager()
            play_flag = settings.get("/app/player/playSimulations")
            settings.set_bool("/app/player/playSimulations", False)
            omni.kit.app.get_app().update()
            settings.set_bool("/app/player/playSimulations", bool(play_flag))

        raw = self._rgb_annotator.get_data()
        if isinstance(raw, dict):
            raw = raw.get("data", np.array([], dtype=np.uint8))
        raw = np.asarray(raw, dtype=np.uint8)
        if raw.size == 0:
            return np.zeros((h, w, 3), dtype=np.uint8)
        if raw.ndim == 1:
            raw = raw.reshape(h, w, -1)
        result = raw[:, :, :3]

        # Headless on-demand: pause RTX after the frame is captured so it does not
        # render between recording windows.  resume() is called at the top of the
        # next render_rgb_array() call.
        if self._runtime_headless and self._rgb_render_product is not None:
            self._rgb_render_product.pause()

        return result

    # ---- Capabilities ---------------------------------------------------------------------

    def is_running(self) -> bool:
        """Return whether Kit app/runtime is still running.

        Returns:
            ``True`` when the visualizer can continue stepping, otherwise ``False``.
        """
        if self._simulation_app is not None:
            return self._simulation_app.is_running()
        try:
            import omni.kit.app

            app = omni.kit.app.get_app()
            return app is not None and app.is_running()
        except (ImportError, AttributeError):
            return False

    def is_training_paused(self) -> bool:
        """Return whether simulation play flag is paused in Kit settings."""
        return get_settings_manager().get("/app/player/playSimulations") is False

    def supports_live_plots(self) -> bool:
        """Kit backend hosts live plot widgets via :class:`~isaaclab.ui.widgets.ManagerLiveVisualizer`."""
        return True

    def add_live_plots(
        self,
        managers: dict,
        scalars: dict | None = None,
        term_names: dict[str, list[str]] | None = None,
        env_idx: int = 0,
    ) -> None:
        """Register managers for live plotting using the Kit omni.ui widget path.

        Creates a :class:`~isaaclab.ui.widgets.ManagerLiveVisualizer` per manager and stores
        them in :attr:`kit_manager_visualizers` so that :class:`~isaaclab.envs.ui.BaseEnvWindow`
        can wire them into the viewport panel.  Also calls the base implementation to populate
        :attr:`_live_plot_sources` for any non-omni.ui consumers.

        Note:
            Scalar groups (e.g. episode metrics) are stored in :attr:`_live_plot_sources` via
            the base implementation but are not yet wired into the omni.ui viewport panel.

        Args:
            managers: Mapping of manager name to manager instance.
            scalars: Optional mapping of group name to a dict of ``{term_name: callable}``.
                Each callable must take no arguments and return a numeric value.
            term_names: Optional per-manager allowlists of term names to include.
            env_idx: Environment index to sample each step.  Defaults to ``0``.
        """
        super().add_live_plots(managers, scalars=scalars, term_names=term_names, env_idx=env_idx)
        from isaaclab.ui.live_plots.manager_live_plots import DirectScalarLivePlots
        from isaaclab.ui.widgets.manager_live_visualizer import (
            DirectScalarLiveVisualizer,
            ManagerLiveVisualizer,
            ManagerLiveVisualizerCfg,
        )

        self.kit_manager_visualizers: dict[str, ManagerLiveVisualizer | DirectScalarLiveVisualizer] = {
            name: ManagerLiveVisualizer(
                manager=mgr,
                cfg=ManagerLiveVisualizerCfg(
                    manager_name=name,
                    term_names=(term_names or {}).get(name),
                ),
            )
            for name, mgr in managers.items()
        }
        # Wire scalar groups (e.g. episode metrics) into the Kit UI panel.
        for source in self._live_plot_sources:
            if isinstance(source, DirectScalarLivePlots):
                self.kit_manager_visualizers[source.manager_name] = DirectScalarLiveVisualizer(source)

    def pumps_app_update(self) -> bool:
        """Return whether :meth:`step` pumps the Kit app loop."""
        return not self._runtime_headless

    def set_camera_view(
        self, eye: tuple[float, float, float] | list[float], target: tuple[float, float, float] | list[float]
    ) -> None:
        """Set active viewport camera eye/target.

        Args:
            eye: Camera eye position.
            target: Camera look-at target.
        """
        if not self._is_initialized:
            logger.debug("[KitVisualizer] set_camera_view() ignored because visualizer is not initialized.")
            return
        self._set_viewport_camera(tuple(eye), tuple(target))

    def render_tiled_rgb_array(self) -> np.ndarray | None:
        """Return the composited streaming frame, all GT types side by side.

        This is the full multi-GT composite the streaming camera panel shows — depth (turbo
        colormap), segmentation and normals included when
        :attr:`~isaaclab.visualizers.VisualizerCfg.streaming_gt_types` asks for them.

        Returns:
            ``uint8 (H, W, 3)`` composite array, or ``None`` when the streaming view is inactive.
        """
        if self._streaming is None:
            return None
        if self._runtime_headless:
            self._update_camera_image_panel()
        return self._streaming.composite(self._step_counter)

    def reapply_origin(self) -> None:
        """Recompute the camera position from the current :attr:`~KitVisualizerCfg.origin_type` and push it to
        the viewport.

        Call this after mutating :attr:`cfg.origin_type`, :attr:`cfg.origin_env_index`, or
        :attr:`cfg.origin_track_path` so the viewport reflects the new origin immediately rather than
        waiting for the next :meth:`step` call.

        For ``"asset"`` origins the camera update is deferred to the
        next :meth:`step` because asset state is not available until after
        :meth:`~isaaclab.sim.SimulationContext.reset`.
        """
        self._setup_initial_camera_view()

    @property
    def viewer_origin(self) -> torch.Tensor | None:
        """Current world-space origin offset applied to :attr:`~KitVisualizerCfg.eye` and
        :attr:`~KitVisualizerCfg.lookat` when computing the absolute camera position.

        Returns ``None`` before :meth:`initialize` is called or when no valid origin has been
        established yet (e.g. asset-tracking before the first :meth:`step`).
        """
        return self._viewer_origin

    # ---- Viewport + camera ----------------------------------------------------------------

    def _ensure_simulation_app(self) -> None:
        """Ensure a running Isaac Sim app is available and cache runtime mode."""
        import omni.kit.app

        app = omni.kit.app.get_app()
        if app is None or not app.is_running():
            raise RuntimeError("[KitVisualizer] Isaac Sim app is not running.")

        try:
            from isaacsim import SimulationApp

            sim_app = None
            if hasattr(SimulationApp, "_instance") and SimulationApp._instance is not None:
                sim_app = SimulationApp._instance
            elif hasattr(SimulationApp, "instance") and callable(SimulationApp.instance):
                sim_app = SimulationApp.instance()

            if sim_app is not None:
                self._simulation_app = sim_app
                self._runtime_headless = bool(self.cfg.headless or self._simulation_app.config.get("headless", False))
                if self._runtime_headless:
                    logger.warning("[KitVisualizer] Running in headless mode. Viewport may not display.")
        except ImportError:
            pass

    def _setup_viewport(self) -> None:
        """Create/resolve viewport and configure initial camera."""
        if self._runtime_headless:
            self._viewport_window = None
            self._viewport_api = None
            if self._uses_streaming_view():
                logger.debug("[KitVisualizer] Camera image view requested in headless mode; no UI panel is created.")
            return

        import omni.kit.viewport.utility as vp_utils
        from omni.ui import DockPosition

        effective_viewport_name = (
            self.cfg.viewport_name if self.cfg.viewport_name is not None else _DEFAULT_VIEWPORT_NAME
        )
        if self.cfg.create_viewport:
            if not str(effective_viewport_name).strip():
                raise RuntimeError(
                    "[KitVisualizer] viewport_name must be a non-empty string when create_viewport=True."
                )
            dock_position_name = self.cfg.dock_position.upper()
            dock_position_map = {
                "LEFT": DockPosition.LEFT,
                "RIGHT": DockPosition.RIGHT,
                "BOTTOM": DockPosition.BOTTOM,
                "SAME": DockPosition.SAME,
            }
            dock_pos = dock_position_map[dock_position_name]

            self._viewport_window = vp_utils.create_viewport_window(
                name=effective_viewport_name,
                width=self.cfg.window_width,
                height=self.cfg.window_height,
                position_x=50,
                position_y=50,
                docked=True,
                camera_path=self.cfg.prim_path,
            )

            asyncio.ensure_future(self._dock_viewport_async(effective_viewport_name, dock_pos))
        else:
            self._viewport_window = vp_utils.get_active_viewport_window()

        if self._viewport_window is None:
            raise RuntimeError("[KitVisualizer] Interactive mode requires an active viewport window.")
        self._viewport_api = self._viewport_window.viewport_api
        if not self.cfg.create_viewport:
            self._viewport_api.set_active_camera(self.cfg.prim_path)

    def _uses_streaming_view(self) -> bool:
        """Return whether Kit should display a streaming camera image panel."""
        return bool(self.cfg.streaming_view)

    def _setup_streaming_view(self) -> None:
        """Resolve the scene Camera sensor backing the streaming image panel."""
        if not self._uses_streaming_view():
            return
        cameras_enabled = get_settings_manager().get("/isaaclab/cameras_enabled", False)
        if not cameras_enabled:
            raise RuntimeError(
                "Kit streaming_view requires camera rendering. Construct AppLauncher with enable_cameras=True or use "
                "launch_simulation(), which enables cameras for streaming_view=True."
            )

        self._streaming = StreamingView(
            self.cfg,
            SimulationContext.instance().get_camera_sensors(),
            visible_env_ids=None,
            target_aspect=self.cfg.window_width / max(1, self.cfg.window_height),
        )
        if not self._runtime_headless:
            self._setup_camera_image_window()
        else:
            logger.debug("[KitVisualizer] Camera image window skipped in headless mode.")

    def _setup_camera_image_window(self) -> None:
        """Create a dockable Kit UI image panel for streaming camera output."""
        import omni.ui

        dock_position_name = self.cfg.dock_position.upper()
        dock_position_map = {
            "LEFT": omni.ui.DockPosition.LEFT,
            "RIGHT": omni.ui.DockPosition.RIGHT,
            "BOTTOM": omni.ui.DockPosition.BOTTOM,
            "SAME": omni.ui.DockPosition.SAME,
        }
        dock_position = dock_position_map[dock_position_name]
        title = self.cfg.viewport_name or "Streaming View"
        self._camera_image_provider = omni.ui.ByteImageProvider()
        self._camera_image_window = omni.ui.Window(title, width=self.cfg.window_width, height=self.cfg.window_height)
        with self._camera_image_window.frame:
            omni.ui.ImageWithProvider(self._camera_image_provider)
        asyncio.ensure_future(self._dock_image_window_async(title, dock_position))

    async def _dock_image_window_async(self, window_name: str, dock_position) -> None:
        """Dock the camera image panel next to the main viewport."""
        import omni.kit.app
        import omni.ui

        image_window = None
        for _ in range(10):
            image_window = omni.ui.Workspace.get_window(window_name)
            if image_window:
                break
            await omni.kit.app.get_app().next_update_async()
        main_viewport = omni.ui.Workspace.get_window("Viewport")
        if image_window is None or main_viewport is None:
            raise RuntimeError(f"[KitVisualizer] Could not dock streaming window '{window_name}'.")
        if image_window != main_viewport:
            image_window.dock_in(main_viewport, dock_position, 0.5)

    def _update_camera_image_panel(self) -> None:
        """Refresh the streaming image panel with composited multi-GT output."""
        if self._streaming is None:
            return

        # The step counter advances while training is paused, so the per-step composite cache would
        # miss every step. Re-upload the last picture instead: nothing moved, so re-driving the
        # camera would only burn GPU time and jitter the panel with floating-point differences.
        if self.is_training_paused():
            if self._streaming.last_composite is not None and self._camera_image_provider is not None:
                self._upload_camera_image_to_panel(self._streaming.last_composite)
            return

        composite = self._streaming.composite(self._step_counter)
        if composite is not None and self._camera_image_provider is not None:
            self._upload_camera_image_to_panel(composite)

    def _upload_camera_image_to_panel(self, image: np.ndarray | torch.Tensor) -> None:
        """Upload an RGB/RGBA image to the Kit image provider."""
        if isinstance(image, torch.Tensor):
            if image.is_cuda:
                import omni.gpu_foundation_factory as gf

                if image.ndim == 3 and image.shape[2] == 3:
                    alpha = torch.full((*image.shape[:2], 1), 255, dtype=torch.uint8, device=image.device)
                    image = torch.cat((image, alpha), dim=2)
                image = image.to(dtype=torch.uint8).contiguous()
                self._camera_gpu_upload_tensor = image
                self._camera_image_provider.set_bytes_data_from_gpu(
                    int(image.data_ptr()), [int(image.shape[1]), int(image.shape[0])], gf.TextureFormat.RGBA8_UNORM
                )
                return
            image = image.detach().contiguous().cpu().numpy()

        image = image.astype("uint8", copy=False)
        if image.ndim == 3 and image.shape[2] == 3:
            alpha = np.full((*image.shape[:2], 1), 255, dtype=np.uint8)
            image = np.concatenate((image, alpha), axis=2)
        image = np.ascontiguousarray(image)
        self._camera_image_provider.set_bytes_data(image.flatten().data, [image.shape[1], image.shape[0]])

    async def _dock_viewport_async(self, viewport_name: str, dock_position) -> None:
        """Dock a created viewport window relative to main viewport."""
        import omni.kit.app
        import omni.ui

        viewport_window = None
        for _ in range(10):
            viewport_window = omni.ui.Workspace.get_window(viewport_name)
            if viewport_window:
                break
            await omni.kit.app.get_app().next_update_async()

        if not viewport_window:
            raise RuntimeError(f"[KitVisualizer] Could not find viewport window '{viewport_name}'.")

        main_viewport = omni.ui.Workspace.get_window("Viewport")
        if not main_viewport:
            for alt_name in ["/OmniverseKit/Viewport", "Viewport Next"]:
                main_viewport = omni.ui.Workspace.get_window(alt_name)
                if main_viewport:
                    break

        if not main_viewport:
            raise RuntimeError("[KitVisualizer] Could not find the main viewport window.")
        if main_viewport != viewport_window:
            viewport_window.dock_in(main_viewport, dock_position, 0.5)
            await omni.kit.app.get_app().next_update_async()
            viewport_window.focus()
            viewport_window.visible = True
            await omni.kit.app.get_app().next_update_async()
            viewport_window.focus()

    def _set_viewport_camera(self, position: tuple[float, float, float], target: tuple[float, float, float]) -> None:
        """Author an eye/target pose on the cfg-owned viewport camera."""
        eye = Gf.Vec3d(*map(float, position))
        center = Gf.Vec3d(*map(float, target))
        forward = center - eye
        if forward.GetLength() == 0.0:
            raise ValueError("[KitVisualizer] Camera eye and target must differ.")
        forward.Normalize()
        up = Gf.Vec3d(0.0, 0.0, 1.0)
        if abs(Gf.Dot(forward, up)) > 0.999:
            up = Gf.Vec3d(0.0, 1.0, 0.0)
        self._camera_xform_op.Set(Gf.Matrix4d().SetLookAt(eye, center, up).GetInverse())

    def _setup_initial_camera_view(self) -> None:
        """Position the viewport camera according to :attr:`KitVisualizerCfg.origin_type`.

        Called once at the end of :meth:`initialize`. For ``"world"`` and ``"env"`` origins the
        camera is positioned immediately. For asset-tracking origins the first update is deferred
        to :meth:`step` because asset state is not yet available at initialization time.
        """
        if self.cfg.origin_type == "world":
            self._viewer_origin = torch.zeros(3)
        elif self.cfg.origin_type == "env":
            plan = self._clone_plan
            if plan.positions is None or not (0 <= self.cfg.origin_env_index < len(plan.positions)):
                raise ValueError(
                    f"[KitVisualizer] clone plan does not contain origin environment {self.cfg.origin_env_index}."
                )
            self._viewer_origin = plan.positions[self.cfg.origin_env_index]
        elif self.cfg.origin_type == "asset":
            if self.cfg.origin_track_path is None:
                raise ValueError("[KitVisualizer] origin_type='asset' requires origin_track_path to be set.")
            plan = self._clone_plan
            if not plan.is_complete:
                raise RuntimeError("[KitVisualizer] asset tracking requires a completed clone plan.")
            paths = cloner.query.destination_paths(plan, expand_env_regex_ns(self.cfg.origin_track_path))
            path = paths.get(self.cfg.origin_env_index)
            if path is None:
                raise ValueError(
                    f"[KitVisualizer] clone plan does not place {self.cfg.origin_track_path!r} in "
                    f"environment {self.cfg.origin_env_index}."
                )
            body_paths = tuple(plan.iter_rigid_body_paths())
            body_path = next(
                (candidate for candidate in body_paths if candidate == path or candidate.startswith(path + "/")), None
            )
            if body_path is None:
                raise ValueError(
                    f"[KitVisualizer] clone plan declares no rigid body below {self.cfg.origin_track_path!r}."
                )
            self._origin_index = body_paths.index(body_path)
            # Physics state is available after reset; defer the first request to step().
            return
        else:
            raise ValueError(f"[KitVisualizer] Unknown origin_type: {self.cfg.origin_type!r}.")

        self._apply_viewer_origin_to_camera()

    def _update_asset_tracking_camera(self) -> None:
        """Update the viewport camera from the planned transform requested through SDP."""
        output = self._scene_data_provider.request_transforms(SceneDataFormat.Vec3_Quat)
        self._viewer_origin = wp.to_torch(output.positions)[self._origin_index]
        self._apply_viewer_origin_to_camera()

    def _apply_viewer_origin_to_camera(self) -> None:
        """Compute absolute eye/target from :attr:`_viewer_origin` and push to the viewport."""
        origin = self._viewer_origin.detach().cpu().numpy()
        eye = np.array(self.cfg.eye, dtype=float) + origin
        target = np.array(self.cfg.lookat, dtype=float) + origin
        self.set_camera_view(tuple(float(v) for v in eye), tuple(float(v) for v in target))
