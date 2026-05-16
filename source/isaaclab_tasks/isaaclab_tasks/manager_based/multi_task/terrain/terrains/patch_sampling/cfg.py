# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from . import morph as morph_funcs
from . import rejection as rejection_funcs


@configclass
class PatchSamplingCfg:
    """Configuration for sampling patches on the sub-terrain."""

    func: Callable = MISSING
    """The function to use for sampling patches."""

    num_patches: int = MISSING
    """Number of patches to sample."""

    patch_radius: float | list[float] = MISSING
    """Radius of the patches."""

    patched = False

    def __post_init__(self):
        if not self.patched:
            cfg = self.to_dict()
            cfg["cfg"] = self.__class__
            setattr(self, "patch_radius", cfg)


@configclass
class FlatPatchSamplingCfg(PatchSamplingCfg):
    func: Callable = rejection_funcs.find_flat_patches
    """The function to use for sampling patches."""

    patch_radius: float | list[float] = MISSING
    """Radius of the patches.

    A list of radii can be provided to check for patches of different sizes. This is useful to deal with
    cases where the terrain may have holes or obstacles in some areas.
    """

    x_range: tuple[float, float] = (-1e6, 1e6)
    """The range of x-coordinates to sample from. Defaults to (-1e6, 1e6).

    This range is internally clamped to the size of the terrain mesh.
    """

    y_range: tuple[float, float] = (-1e6, 1e6)
    """The range of y-coordinates to sample from. Defaults to (-1e6, 1e6).

    This range is internally clamped to the size of the terrain mesh.
    """

    z_range: tuple[float, float] = (-1e6, 1e6)
    """Allowed range of z-coordinates for the sampled patch. Defaults to (-1e6, 1e6)."""

    max_height_diff: float = MISSING
    """Maximum allowed height difference between the highest and lowest points on the patch."""


@configclass
class PieceWiseRangeFlatPatchSamplingCfg(PatchSamplingCfg):
    """Configuration for sampling flat patches on the sub-terrain with piece-wise ranges."""

    func: Callable = rejection_funcs.find_piecewise_range_flat_patches
    """The function to use for sampling patches with piece wise ranges."""

    patch_radius: float | list[float] = MISSING
    """Radius of the patches.

    A list of radii can be provided to check for patches of different sizes. This is useful to deal with
    cases where the terrain may have holes or obstacles in some areas.
    """

    x_range: list[tuple[float, float]] | tuple[float, float] = (-1e6, 1e6)
    """The list of (min, max) intervals for X sampling (in mesh frame)."""

    y_range: list[tuple[float, float]] | tuple[float, float] = (-1e6, 1e6)
    """The list of (min, max) intervals for Y sampling (in mesh frame)."""

    z_range: list[tuple[float, float]] | tuple[float, float] = (-1e6, 1e6)
    """The list of (min, max) intervals for Z filtering (in mesh frame)."""

    max_height_diff: float = MISSING
    """Maximum allowed height difference between the highest and lowest points on the patch."""

    max_iterations: int = 100


@configclass
class FlatPatchSamplingByRadiusCfg(PatchSamplingCfg):
    func: Callable = rejection_funcs.find_flat_patches_by_radius

    patch_radius: float | list[float] = MISSING

    radius_range: tuple[float, float] = MISSING

    x_range: list[tuple[float, float]] | tuple[float, float] = (-1e6, 1e6)
    """The list of (min, max) intervals for X sampling (in mesh frame)."""

    y_range: list[tuple[float, float]] | tuple[float, float] = (-1e6, 1e6)
    """The list of (min, max) intervals for Y sampling (in mesh frame)."""

    z_range: tuple[float, float] = (-1e6, 1e6)
    """The list of (min, max) intervals for Z filtering (in mesh frame)."""

    max_height_diff: float = MISSING

    max_iterations: int = 100


# ---------------------------------------------------------------------------
# Morphological patch sampling (deterministic, GPU-batched)
# ---------------------------------------------------------------------------


@configclass
class CircleFootprintCfg:
    """Circular robot footprint (yaw-invariant)."""

    radius: float = MISSING
    """Footprint radius [m]."""


@configclass
class RectFootprintCfg:
    """Rectangular robot footprint (axis-aligned with the body frame).

    The robot's forward direction is +x.  At yaw = 0 the footprint's
    ``length`` lies along the world x-axis and ``width`` along the
    world y-axis.
    """

    length: float = MISSING
    """Nose-to-tail extent along the robot's forward (+x) axis [m]."""

    width: float = MISSING
    """Shoulder-to-shoulder extent along the robot's lateral (+y) axis [m]."""


@configclass
class MorphologicalPatchSamplingCfg(PatchSamplingCfg):
    """Deterministic patch sampling via morphological heightmap filtering.

    Rasterizes the mesh to a 2D heightmap, computes a validity mask using
    max/min pooling with the robot footprint kernel, then samples from the
    valid region with optional farthest-point refinement.
    """

    func: Callable = morph_funcs.find_flat_patches_morphological

    patch_radius: float | list[float] = 0.0
    """Unused by this sampler (kept for base-class compatibility)."""

    footprint: CircleFootprintCfg | RectFootprintCfg = MISSING
    """Robot footprint shape used to evaluate local flatness."""

    x_range: tuple[float, float] = (-1e6, 1e6)
    """Search range for x-coordinates [m], relative to origin."""

    y_range: tuple[float, float] = (-1e6, 1e6)
    """Search range for y-coordinates [m], relative to origin."""

    z_range: tuple[float, float] = (-1e6, 1e6)
    """Allowed height range for valid patches [m], relative to origin."""

    max_height_diff: float = MISSING
    """Maximum height variation within the footprint to accept a cell [m]."""

    horizontal_scale: float = 0.1
    """Rasterization grid spacing [m].

    Lower values grow the number of valid cells the sampler can draw from —
    reach for this knob first when the morphological filter raises with
    ``found only N valid cells but M patches requested``.
    """

    oversample_ratio: float = 2.0
    """Oversample by this factor, then apply farthest-point sampling.

    Set to 1.0 to disable farthest-point refinement.
    """
