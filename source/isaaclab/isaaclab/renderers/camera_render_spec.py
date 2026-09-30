# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable description of a tiled camera passed to render backends."""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab.sensors.camera.camera_cfg import CameraCfg


@dataclass(frozen=True)
class CameraRenderSpec:
    """Stable inputs for :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.create_render_data`.

    Backends use this instead of holding a reference to the :class:`~isaaclab.sensors.camera.Camera`
    sensor instance, avoiding circular dependencies between sensors and render data.

    Args:
        cfg: Camera configuration (data types, resolution, filters, etc.).
        device: Torch device string (e.g. ``"cuda:0"``) used by GPU annotators and Warp.
        camera_source_prim_paths: Absolute USD paths of the camera prototypes named by the clone
            plan, in plan row order.
        camera_prim_paths: Absolute USD paths for each environment's camera prim, in ascending
            environment order. A heterogeneous scene populates only the environments its clone
            plan row covers, so the first path is not necessarily in environment 0.
    """

    cfg: CameraCfg
    device: str
    camera_source_prim_paths: tuple[str, ...]
    camera_prim_paths: tuple[str, ...]

    @property
    def num_instances(self) -> int:
        """Number of tiled camera instances, one per environment the camera is cloned to."""
        return len(self.camera_prim_paths)
