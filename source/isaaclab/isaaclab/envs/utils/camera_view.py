# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for visualizer and recorder camera image views."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ...cloner.cloner_cfg import DEFAULT_ENV_TEMPLATE, expand_env_regex_ns
from ...utils.images import sensor_key_for_gt_type
from ...visualizers.visualizer_cfg import PerspectiveCameraCfg, SceneCameraCfg

if TYPE_CHECKING:
    from ...sensors.camera import Camera
    from ...visualizers.visualizer_cfg import VisualizerCfg


def resolve_camera_sources(
    cfg: VisualizerCfg, cameras: dict[str, Camera], *, env_template: str = DEFAULT_ENV_TEMPLATE
) -> list[PerspectiveCameraCfg | Camera]:
    """Bind display sources before initializing a visualizer, without reading sensor frames.

    Args:
        cfg: Requested camera sources and display channels. The configuration is not modified.
        cameras: Scene-owned sensors keyed by scene name.
        env_template: Environment namespace used to expand ``{ENV_REGEX_NS}`` references.

    Returns:
        Ordered perspective settings and borrowed sensors. Explicit scene references must support
        every requested channel; automatic discovery skips incompatible sensors.
    """
    sources = list(cfg.cameras or [PerspectiveCameraCfg(eye=cfg.eye, lookat=cfg.lookat, focal_length=cfg.focal_length)])
    if not cfg.streaming_view:
        return sources
    gt_types = cfg.streaming_gt_types
    for gt_type in gt_types:
        sensor_key_for_gt_type(gt_type)
    if cfg.cameras is None:
        if cfg.streaming_sensor_prim_path is not None:
            sources.insert(0, SceneCameraCfg(prim_path=cfg.streaming_sensor_prim_path))
        else:
            for camera in cameras.values():
                available = frozenset(camera.cfg.data_types)
                if all(sensor_key_for_gt_type(gt, available, required=False) is not None for gt in gt_types):
                    sources.append(camera)
    for index, source in enumerate(sources):
        if not isinstance(source, SceneCameraCfg):
            continue
        path = expand_env_regex_ns(source.prim_path, env_template)
        pattern = path.replace("%d", "[^/]+").replace("{}", "[^/]+")
        pattern = pattern.replace("/World/envs/*", "/World/envs/env_[^/]+")
        for camera in cameras.values():
            if camera.cfg.prim_path == path or (
                camera._view is not None
                and any(re.fullmatch(pattern, str(prim.GetPath())) for prim in camera._view.prims)
            ):
                break
        else:
            available_paths = sorted(camera.cfg.prim_path for camera in cameras.values())
            raise ValueError(
                f"No scene Camera matches prim_path={path!r}. "
                f"Declare a CameraCfg in the scene; available paths: {available_paths}."
            )
        available = frozenset(camera.cfg.data_types)
        for gt_type in gt_types:
            sensor_key_for_gt_type(gt_type, available)
        sources[index] = camera
    return sources
