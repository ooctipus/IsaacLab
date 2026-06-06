# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cfg for the fast terrain scanner."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.sensors.sensor_base_cfg import SensorBaseCfg
from isaaclab.sim.spawners.sensors import SensorFrameCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .fast_terrain_scanner import FastTerrainScanner


@configclass
class FastTerrainScannerCfg(SensorBaseCfg):
    """Configuration for :class:`FastTerrainScanner`.

    Specialized API surface for the height-scanner-on-static-terrain use case. Compared
    to :class:`isaaclab.sensors.MultiMeshRayCasterCfg`, this config drops fields we
    don't need (per-target ``RaycastTargetCfg``, ``track_mesh_transforms``,
    ``merge_prim_meshes``, ``is_shared``, ``reference_meshes``, ``update_mesh_ids``)
    and the corresponding code paths.
    """

    @configclass
    class OffsetCfg:
        """Per-sensor offset relative to the spawned prim."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Position offset [m]."""
        rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        """Quaternion offset (x, y, z, w)."""

    class_type: type[FastTerrainScanner] | str = "{DIR}.fast_terrain_scanner:FastTerrainScanner"

    spawn: SensorFrameCfg | None = SensorFrameCfg()
    """Spawn config — keeps API parity with the upstream raycaster sensors."""

    mesh_prim_paths: list[str] = MISSING
    """List of prim-path expressions to ray-cast against. Each entry may use ``{ENV_REGEX_NS}``
    to match per-env terrain prims (e.g. ``"{ENV_REGEX_NS}/ground"``).

    For each resolved prim, the first child prim of type ``Mesh`` is parsed (via
    :func:`isaaclab.utils.mesh.create_trimesh_from_geom_mesh`) and uploaded as a Warp mesh.
    Identical mesh data is deduplicated across envs by ``(prim_path, device)`` cache key.
    """

    offset: OffsetCfg = OffsetCfg()
    """Per-sensor offset (translation + rotation) applied on top of the FrameView pose."""

    ray_alignment: Literal["world", "yaw", "base"] = "yaw"
    """Frame the ray pattern is expressed in. ``"yaw"`` is the height-scanner default —
    rotate ray *starts* by the body yaw so the scan footprint follows the heading, but
    keep ray *directions* fixed in world (so straight-down stays straight-down)."""

    pattern_cfg: PatternBaseCfg = MISSING
    """Pattern that produces ``(ray_starts, ray_directions)`` torch tensors at init time."""

    max_distance: float = 1e6
    """Maximum ray-cast distance [m] passed to ``mesh_query_ray``."""

    drift_range: tuple[float, float] = (0.0, 0.0)
    """Uniform-sampled per-env sensor-position drift [m] (x, y, z components share the range)."""

    ray_cast_drift_range: dict[str, tuple[float, float]] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
    }
    """Per-axis uniform-sampled drift applied to ray starts."""

    visualizer_cfg: VisualizationMarkersCfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/FastTerrainScanner")
    """The configuration object for the ray-hit visualization markers. Defaults to RAY_CASTER_MARKER_CFG.

    .. note::
        This attribute is only used when debug visualization is enabled (``debug_vis=True``).
    """
