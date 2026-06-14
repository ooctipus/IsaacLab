# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Symmetry config dataclasses, with no torch or warp imports.

A held asset's success/observation symmetry is the set of orientations that are
indistinguishable goals. It is declared as a list of *elements*. Built-in
elements cover axis generators (continuous or N-fold rotation about a local
axis) and manually-authored ("semantic") equivalent rotations; custom
elements can participate by providing their own ``class_type(cfg)``
implementation class. :mod:`.asset_symmetry` expands and deduplicates
these into the flat tables the Warp reducer consumes; the identity is always implied.

This is the single source of truth for "what is an equivalent goal", shared by
the success criterion, the command observation, and (eventually) the assembly
sampler — so a continuous-yaw asset is one definition, not a ``UniformYawCfg``
in the sampler plus a ``symmetry_order`` in the success term.
"""

from __future__ import annotations

from dataclasses import MISSING, field
from typing import TYPE_CHECKING, Protocol

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .asset_symmetry import AssetSymmetry, SymmetryElement


class SymmetryElementCfg(Protocol):
    """Protocol for cfg elements compiled by :class:`AssetSymmetryCfg`."""

    class_type: type[SymmetryElement] | str
    """Implementation class called as ``class_type(cfg)`` to compile this element."""


@configclass
class AxisSymmetryCfg:
    """Rotational symmetry about a local-frame axis.

    ``order`` selects the kind: ``0`` is continuous (any rotation about
    :attr:`axis` is equivalent; the reducer handles it analytically, no
    enumeration); ``N >= 1`` is N-fold discrete (``Raxis(2*pi*k/N)`` for
    ``k in [0, N)``); ``1`` is the identity (no symmetry)."""

    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Local rotation axis (unit; the held asset's insertion axis is local z)."""

    order: int = 1
    """N-fold order. ``0`` = continuous about :attr:`axis`; ``1`` = identity only."""

    class_type: type[SymmetryElement] | str = "{DIR}.asset_symmetry:AxisSymmetry"
    """Implementation class called as ``class_type(cfg)`` to compile this element."""


@configclass
class SemanticSymmetryCfg:
    """Manually-authored equivalent rotations that are NOT axis-generated.

    Each quaternion ``q`` (x, y, z, w) asserts that ``target * q`` is an
    indistinguishable assembled goal — e.g. a connector that mates in two
    mirrored keyings declares ``[(0,0,0,1), (0,1,0,0)]`` (identity + 180° about
    y). Compared by the identical ``geodesic(held, target * q)`` reduction as
    axis members; the identity is implied and need not be listed."""

    offsets: list[tuple[float, float, float, float]] = MISSING
    """Equivalent local-frame rotations as (x, y, z, w) quaternions."""

    class_type: type[SymmetryElement] | str = "{DIR}.asset_symmetry:SemanticSymmetry"
    """Implementation class called as ``class_type(cfg)`` to compile this element."""


@configclass
class AssetSymmetryCfg:
    """One asset's full symmetry profile.

    A single cyclic element is compiled to the O(1) closed-form path; anything
    else expands to a deduplicated finite orbit reduced by a bounded loop.
    Mixing a continuous axis with any other element is rejected at build time.
    """

    elements: list[SymmetryElementCfg] = field(default_factory=lambda: [AxisSymmetryCfg(order=1)])
    """Symmetry generators. Each element must expose ``class_type(cfg)``."""

    class_type: type[AssetSymmetry] | str = "{DIR}.asset_symmetry:AssetSymmetry"
    """Implementation class called as ``class_type(cfg)`` to compile one per-type table entry."""
