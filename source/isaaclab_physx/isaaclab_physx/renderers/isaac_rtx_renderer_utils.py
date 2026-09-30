# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for Isaac RTX renderer integration."""

from __future__ import annotations

import logging
import time

import omni.usd

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import SettingsManager, get_settings_manager

from .isaac_rtx_renderer_cfg import IsaacRtxRendererGlobalSettingsCfg

logger = logging.getLogger(__name__)

_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING = "/rtx/scenePartitioning/showAllPartitionsByDefault"
_RTX_FIELD_TO_SETTING = {
    "enable_translucency": "/rtx/translucency/enabled",
    "enable_reflections": "/rtx/reflections/enabled",
    "enable_global_illumination": "/rtx/indirectDiffuse/enabled",
    "enable_dlssg": "/rtx-transient/dlssg/enabled",
    "enable_dl_denoiser": "/rtx-transient/dldenoiser/enabled",
    "dlss_mode": "/rtx/post/dlss/execMode",
    "enable_direct_lighting": "/rtx/directLighting/enabled",
    "samples_per_pixel": "/rtx/directLighting/sampledLighting/samplesPerPixel",
    "enable_shadows": "/rtx/shadows/enabled",
    "enable_ambient_occlusion": "/rtx/ambientOcclusion/enabled",
    "dome_light_upper_lower_strategy": "/rtx/domeLight/upperLowerStrategy",
    "ambient_light_intensity": "/rtx/sceneDb/ambientLightIntensity",
    "ambient_occlusion_denoiser_mode": "/rtx/ambientOcclusion/denoiserMode",
    "subpixel_mode": "/rtx/raytracing/subpixel/mode",
    "enable_cached_raytracing": "/rtx/raytracing/cached/enabled",
    "max_samples_per_launch": "/rtx/pathtracing/maxSamplesPerLaunch",
    "view_tile_limit": "/rtx/viewTile/limit",
    "show_all_partitions_by_default": _SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING,
    # RT2 path tracing settings
    "max_bounces": "/rtx/rtpt/maxBounces",
    "split_glass": "/rtx/rtpt/splitGlass",
    "split_clearcoat": "/rtx/rtpt/splitClearcoat",
    "split_rough_reflection": "/rtx/rtpt/splitRoughReflection",
}

_STREAMING_WAIT_TIMEOUT_S: float = 30.0


def _setting_path_from_key(key: str) -> str:
    """Convert a user-friendly carb setting key to a carb path."""
    if key.startswith("/"):
        return key
    if "_" in key:
        return "/" + key.replace("_", "/")
    if "." in key:
        return "/" + key.replace(".", "/")
    return key


def apply_isaac_rtx_determinism_settings(settings: SettingsManager | None = None) -> None:
    """Apply Isaac RTX settings for reproducible rendering.

    Selects RealTimePathTracing and disables the RTPT color and light caches.

    Args:
        settings: Settings manager to apply settings through. If None, the global settings manager is used.
    """
    if settings is None:
        settings = get_settings_manager()
    settings.set("/rtx/rendermode", "RealTimePathTracing")
    settings.set("/rtx/rtpt/cached/enabled", False)
    settings.set("/rtx/rtpt/lightcache/cached/enabled", False)
    logger.info("Applied Isaac RTX settings for deterministic rendering.")


def apply_isaac_rtx_global_settings(
    global_settings: IsaacRtxRendererGlobalSettingsCfg,
    settings: SettingsManager | None = None,
) -> None:
    """Apply global Isaac RTX settings before renderer initialization.

    Args:
        global_settings: Global Isaac RTX settings to apply.
        settings: Settings manager to apply settings through. If None, the global settings manager is used.
    """
    if settings is None:
        settings = get_settings_manager()
    _apply_isaac_rtx_global_settings(global_settings, settings)


def _apply_isaac_rtx_global_settings(
    global_settings: IsaacRtxRendererGlobalSettingsCfg,
    settings: SettingsManager,
) -> None:
    """Apply global Isaac RTX settings to the provided settings manager."""

    for field_name, setting_path in _RTX_FIELD_TO_SETTING.items():
        value = getattr(global_settings, field_name)
        if value is not None:
            settings.set(setting_path, value)

    if global_settings.carb_settings:
        for key, value in global_settings.carb_settings.items():
            settings.set(_setting_path_from_key(key), value)

    if global_settings.antialiasing_mode is not None:
        import omni.replicator.core as rep

        rep.settings.set_render_rtx_realtime(antialiasing=global_settings.antialiasing_mode)


def _get_stage_streaming_busy() -> bool:
    """Synchronously query whether RTX stage streaming is still in progress."""
    import omni.usd

    usd_context = omni.usd.get_context()
    if usd_context is None:
        raise RuntimeError("Isaac RTX requires an active USD context.")
    return usd_context.get_stage_streaming_status()


def _wait_for_streaming_complete() -> None:
    """Pump ``app.update()`` until RTX streaming reports idle or timeout.

    After streaming finishes a final ``app.update()`` is issued so that the
    frame captured by downstream annotators reflects the newly loaded textures.
    """
    import omni.kit.app

    start = time.monotonic()
    while _get_stage_streaming_busy() and (time.monotonic() - start) < _STREAMING_WAIT_TIMEOUT_S:
        omni.kit.app.get_app().update()

    elapsed = time.monotonic() - start
    if _get_stage_streaming_busy():
        raise TimeoutError(f"RTX streaming did not complete within {_STREAMING_WAIT_TIMEOUT_S:.1f} s.")
    if elapsed > 0.01:
        logger.info("RTX streaming completed in %.2f s.", elapsed)

    omni.kit.app.get_app().update()


def ensure_rtx_hydra_engine_attached() -> None:
    """Attach the RTX Hydra engine to the USD context if not already attached.

    ``ViewportWindow`` usually performs this during startup, but callers can also
    reach this code path before a viewport has attached an RTX engine to the
    :class:`omni.usd.UsdContext`. Without that attachment the first Replicator tiled
    render product runs against a cold pipeline. On some GPUs this manifests as
    ``cudaErrorIllegalAddress`` inside ``omni.rtx`` (CUDA ``freeAsync``) and/or all
    tiles rendering as black.

    This helper is idempotent: when the engine is already attached (e.g. app files
    that load ``omni.kit.viewport.window``, or a previous call already attached it)
    the function is a no-op.
    """
    ctx = omni.usd.get_context()
    if ctx is None:
        raise RuntimeError("Isaac RTX requires an active USD context.")
    if "rtx" not in ctx.get_attached_hydra_engine_names():
        omni.usd.create_hydra_engine("rtx", ctx)


def ensure_isaac_rtx_render_update() -> None:
    """Pump the Isaac RTX renderer for a requested camera frame.

    This keeps the Kit-specific ``app.update()`` logic inside the renderers
    package rather than in the backend-agnostic ``SimulationContext``.

    After the initial ``app.update()`` the streaming subsystem is queried
    synchronously via ``UsdContext.get_stage_streaming_status()``.  If textures
    are still loading, additional ``app.update()`` calls are pumped until the
    subsystem reports idle (or a timeout is reached).

    The active renderer and simulation are required to be ready when a frame is requested.
    """
    sim = sim_utils.SimulationContext.instance()
    if sim is None or not sim.is_rendering:
        raise RuntimeError("Isaac RTX cannot render without an active rendering SimulationContext.")

    import omni.kit.app

    sim.set_setting("/app/player/playSimulations", False)
    try:
        omni.kit.app.get_app().update()
        if _get_stage_streaming_busy():
            _wait_for_streaming_complete()
    finally:
        sim.set_setting("/app/player/playSimulations", True)
