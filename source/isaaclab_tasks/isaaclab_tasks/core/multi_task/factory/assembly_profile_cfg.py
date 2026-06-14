# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assembly profile configuration classes.

An assembly profile describes the full geometric path from assembled (fraction=0)
to disassembled (fraction=1) as a sequence of contiguous segments.  Two segment
types are provided:

* :class:`EndPointsSegmentCfg` — defined by explicit start/end poses and optional
  extra revolutions.
* :class:`IncrementalSegmentCfg` — defined by start pose, travel distance, and
  a screw-pitch ratio (m/rad).

Start-sampler configs define noise applied on top of each segment's start pose:
:class:`SymmetryOrbitCfg` (symmetry-backed rotational equivalents) and
:class:`UniformPoseNoiseCfg` (free 6-DoF noise).  ``None`` means no noise.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from ..utils.symmetry import AssetSymmetryCfg
from .assembly_keypoints import Offset
from .assembly_profile import (
    AssemblyProfile,
    EndPointsSegment,
    IncrementalSegment,
    SymmetryOrbit,
    UniformPoseNoise,
)

# ---------------------------------------------------------------------------
# Start-sampler Cfg classes
# ---------------------------------------------------------------------------


@configclass
class SymmetryOrbitCfg:
    """Sample a symmetry-equivalent start rotation from a held-asset symmetry
    definition -- continuous yaw or N-fold rotation about the asset's axis, so the
    asset's symmetry is declared once and shared with the success criterion. No
    position noise."""

    class_type: type = SymmetryOrbit
    """Class of the sampler implementation."""

    symmetry: AssetSymmetryCfg = MISSING
    """The held asset's symmetry; the start rotation is sampled from its orbit."""


@configclass
class UniformPoseNoiseCfg:
    """Uniform noise over user-defined position [m] and euler-angle [rad] ranges."""

    class_type: type = UniformPoseNoise
    """Class of the sampler implementation."""

    x: tuple[float, float] = (0.0, 0.0)
    """Position noise range along x [m]."""

    y: tuple[float, float] = (0.0, 0.0)
    """Position noise range along y [m]."""

    z: tuple[float, float] = (0.0, 0.0)
    """Position noise range along z [m]."""

    roll: tuple[float, float] = (0.0, 0.0)
    """Roll noise range [rad]."""

    pitch: tuple[float, float] = (0.0, 0.0)
    """Pitch noise range [rad]."""

    yaw: tuple[float, float] = (0.0, 0.0)
    """Yaw noise range [rad]."""


# ---------------------------------------------------------------------------
# Segment and profile Cfg classes
# ---------------------------------------------------------------------------


@configclass
class EndPointsSegmentCfg:
    """Segment defined by explicit start and end poses.

    See :class:`EndPointsSegment` for the runtime implementation.
    """

    class_type: type = EndPointsSegment
    """Class of the segment implementation."""

    fraction: tuple[float, float] = (0.0, 1.0)
    """Fraction range ``(lo, hi)`` this segment covers. ``0`` is assembled."""

    start_sampler: SymmetryOrbitCfg | UniformPoseNoiseCfg | None = None
    """Noise config applied on top of the interpolated pose. ``None`` means no noise."""

    start_pose: Offset = Offset()
    """Offset at ``fraction[0]`` (assembled end) relative to the fixed asset."""

    end_pose: Offset = Offset()
    """Offset at ``fraction[1]`` (disassembled end) relative to the fixed asset."""

    revolutions: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Extra full turns ``(roll, pitch, yaw)`` between the two endpoints."""


@configclass
class IncrementalSegmentCfg:
    """Segment defined by start pose, travel distance, and rotation ratio.

    See :class:`IncrementalSegment` for the runtime implementation.
    """

    class_type: type = IncrementalSegment
    """Class of the segment implementation."""

    fraction: tuple[float, float] = (0.0, 1.0)
    """Fraction range ``(lo, hi)`` this segment covers. ``0`` is assembled."""

    start_sampler: SymmetryOrbitCfg | UniformPoseNoiseCfg | None = None
    """Noise config applied on top of the interpolated pose. ``None`` means no noise."""

    start_pose: Offset = Offset()
    """Offset at ``fraction[0]`` (assembled end) relative to the fixed asset."""

    distance: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Linear travel vector from start to end [m]."""

    ratio: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Screw pitch per axis: meters of travel per radian [m/rad]. Zero means no rotation."""


@configclass
class AssemblyProfileCfg:
    """Complete assembly path as a list of contiguous segment configs.

    ``fraction=0`` means fully assembled; increasing fraction moves toward the
    disassembled state. See :class:`AssemblyProfile` for the runtime implementation.
    """

    class_type: type = AssemblyProfile
    """Class of the profile implementation."""

    segments: list[EndPointsSegmentCfg | IncrementalSegmentCfg] | None = None
    """Ordered list of segment configs covering the full fraction range."""
