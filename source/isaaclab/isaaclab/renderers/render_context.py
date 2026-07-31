# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation-scoped renderers for camera sensors."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, cast

from isaaclab.sensors.camera.camera_data import CameraData

from .base_renderer import BaseRenderer
from .renderer import Renderer
from .renderer_cfg import RendererCfg

logger = logging.getLogger(__name__)


class RenderContext:
    """Holds :class:`BaseRenderer` instances for all :class:`Camera` sensors in a simulation.

    A camera reuses a backend when a prior camera registered a config equal under ``==`` (value
    equality) and the same concrete ``RendererCfg`` subclass. A distinct ``RendererCfg`` that
    maps to a different implementation (e.g. Isaac RTX vs Newton) produces another backend; each
    has :meth:`BaseRenderer.prepare_stage` run before use.

    :meth:`update_scene_state` is invoked at most once per :meth:`get_physics_step_count` for the
    context;
    """

    __slots__ = (
        "_renderer_entries",
        "_physics_initialized",
        "_prepared_renderer_ids",
        "_prepared_num_envs",
        "_last_scene_state_step",
        "_declared_texture_paths",
        "_visual_material_listeners",
    )

    def __init__(self) -> None:
        self._renderer_entries: list[tuple[RendererCfg, BaseRenderer]] = []
        self._physics_initialized: bool = False  # Set to True after the first PHYSICS_READY callback fires.
        self._prepared_renderer_ids: set[int] = set()
        self._prepared_num_envs: int | None = None
        self._last_scene_state_step: int | None = None
        self._declared_texture_paths: list[str] = []
        self._visual_material_listeners: list = []

    def _check_global_settings_compatible(self, cfg: RendererCfg) -> None:
        """Reject conflicting process-global renderer settings."""
        if getattr(cfg, "renderer_type", None) != "isaac_rtx" or not hasattr(cfg, "global_settings"):
            return
        for stored_cfg, _renderer in self._renderer_entries:
            if getattr(stored_cfg, "renderer_type", None) != "isaac_rtx" or not hasattr(stored_cfg, "global_settings"):
                continue
            if stored_cfg.global_settings != cfg.global_settings:
                raise ValueError(
                    "Isaac RTX global settings differ across camera renderer configs. "
                    "These settings are process-global; configure the same "
                    "IsaacRtxRendererCfg.global_settings for every Isaac RTX camera."
                )

    def get_renderer(self, cfg: RendererCfg) -> BaseRenderer:
        """Return a backend for this configuration, reusing a matching instance if present.

        Lookups use ``==`` and concrete ``RendererCfg`` type, so :func:`hash` is not used (configs
        are typically not hashable).

        Args:
            cfg: Renderer configuration from the initializing camera.

        Returns:
            A shared or newly created renderer backend.
        """
        self._check_global_settings_compatible(cfg)
        for stored_cfg, r in self._renderer_entries:
            if type(stored_cfg) is type(cfg) and stored_cfg == cfg:
                return r
        new_renderer = cast(BaseRenderer, Renderer(cfg))  # type: ignore[misc]
        self._renderer_entries.append((cfg, new_renderer))
        if self._declared_texture_paths:
            new_renderer.register_visual_material_textures(list(self._declared_texture_paths))
        logger.info(
            "Created new renderer for simulation: %s",
            type(new_renderer).__name__,
        )
        if self._physics_initialized:
            new_renderer.initialize()
        return new_renderer

    def ensure_initialize(self) -> None:
        """Idempotent call fired after PHYSICS_READY callback."""
        if self._physics_initialized:
            return
        self._physics_initialized = True
        for _cfg, renderer in self._renderer_entries:
            renderer.initialize()

    def notify_visual_material_written(self, writes: Sequence[Any]) -> None:
        """Broadcast batched visual-material channel writes to every registered backend.

        No-op before the PHYSICS_READY callback: values authored earlier are picked up by each
        backend's own stage export or model build (:meth:`ensure_prepare_stage`), so there is
        nothing to synchronize yet.

        Args:
            writes: :class:`~isaaclab.renderers.base_renderer.VisualMaterialWrite` entries, one
                per written shader attribute.
        """
        if not self._physics_initialized:
            return
        for _cfg, renderer in self._renderer_entries:
            renderer.notify_visual_material_written(writes)
        for listener in self._visual_material_listeners:
            listener(writes)

    def _register_visual_material_listener(self, listener) -> list[str]:
        """Register a callback for visual-material writes and return the declared texture paths.

        Consumers that mirror material writes outside the renderer registry subscribe here — the
        Newton viewer uses it to swap its logged-mesh textures. The returned texture paths are
        everything declared so far (see :meth:`register_visual_material_textures`), so the
        listener can preload its own pool at registration.

        Args:
            listener: Callable invoked with the list of
                :class:`~isaaclab.renderers.base_renderer.VisualMaterialWrite` on every write.
        """
        self._visual_material_listeners.append(listener)
        return list(self._declared_texture_paths)

    def register_visual_material_textures(self, texture_paths: Sequence[str]) -> None:
        """Declare candidate ``texture``-channel textures to every registered backend.

        Declarations are also replayed to backends registered later (entities typically declare
        their pools before cameras create their renderers), so ordering between material entities
        and camera construction does not matter.

        Args:
            texture_paths: Asset paths of the candidate textures.
        """
        self._declared_texture_paths.extend(texture_paths)
        for _cfg, renderer in self._renderer_entries:
            renderer.register_visual_material_textures(texture_paths)

    def ensure_prepare_stage(self, stage: Any, num_envs: int) -> None:
        """Call :meth:`BaseRenderer.prepare_stage` for each registered backend (once per backend).

        If a new backend is added after the first :meth:`prepare_stage` call, this method ensures
        that new backend is prepared for the same ``stage`` and ``num_envs`` when the camera
        that owns it is initialized.

        Args:
            stage: USD stage passed to each backend.
            num_envs: Environment count.

        Raises:
            RuntimeError: If :meth:`get_renderer` was never called, or ``num_envs`` disagrees with
                a value already used for a prepared backend in this context.
        """
        if not self._renderer_entries:
            raise RuntimeError("get_renderer must be called at least once before ensure_prepare_stage.")
        if self._prepared_num_envs is not None and self._prepared_num_envs != num_envs:
            raise RuntimeError(
                "RenderContext prepare_stage was used with a different num_envs "
                f"({self._prepared_num_envs} vs {num_envs})."
            )
        for _cfg, renderer in self._renderer_entries:
            rid = id(renderer)
            if rid not in self._prepared_renderer_ids:
                renderer.prepare_stage(stage, num_envs)
                self._prepared_renderer_ids.add(rid)
        if self._prepared_num_envs is None:
            self._prepared_num_envs = num_envs

    def update_scene_state(self, physics_step_count: int) -> None:
        """Update scene state on all backends (at most once per step).

        Invokes :meth:`BaseRenderer.update_transforms` and then
        :meth:`BaseRenderer.update_geometries` on each registered renderer.
        """
        if not self._renderer_entries:
            return

        if self._last_scene_state_step == physics_step_count:
            return

        for _cfg, renderer in self._renderer_entries:
            renderer.update_transforms()
            renderer.update_geometries()

        self._last_scene_state_step = physics_step_count

    def render_into_camera(
        self,
        renderer: BaseRenderer,
        render_data: Any,
        camera_data: CameraData,
        physics_step_count: int,
    ) -> None:
        """Sync scene state, render, and read outputs into ``camera_data``."""
        self.update_scene_state(physics_step_count)
        renderer.render(render_data)
        renderer.read_output(render_data, camera_data)

    def reset_stage_prepare_flag(self) -> None:
        """Allow :meth:`ensure_prepare_stage` to run ``prepare_stage`` again (e.g. a new USD stage)."""
        self._prepared_renderer_ids.clear()
        self._prepared_num_envs = None

    def reset_scene_state_cadence(self) -> None:
        """Clear per-step scene state update dedupe (e.g. a long pause with no physics)."""
        self._last_scene_state_step = None
