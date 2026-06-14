# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-asset symmetry compilation."""

from __future__ import annotations

import itertools
import math
from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .symmetry_cfg import AssetSymmetryCfg, AxisSymmetryCfg, SemanticSymmetryCfg, SymmetryElementCfg

KIND_CYCLIC = 0
KIND_GENERAL = 1

_DEDUP_EPS = 1e-4


@dataclass(frozen=True)
class SymmetryElementTable:
    """Compiled data for one symmetry element.

    Attributes:
        kind: Dispatch tag. ``0`` = cyclic axis, ``1`` = finite offset set.
        axis: Local symmetry axis [3], float32 (cyclic only; unit).
        order: N-fold order (cyclic only; ``0`` = continuous).
        offset_quat: Element quaternions [M, 4], float32 (x, y, z, w).
    """

    kind: int
    axis: np.ndarray
    order: int
    offset_quat: np.ndarray


@dataclass(frozen=True)
class SymmetryTableEntry:
    """Compiled data for one asset type.

    Attributes:
        kind: Dispatch tag. ``0`` = cyclic closed-form, ``1`` = general orbit loop.
        axis: Local symmetry axis [3], float32 (cyclic only; unit).
        order: N-fold order (cyclic only; ``0`` = continuous).
        offset_quat: Orbit quaternions [M, 4], float32 (x, y, z, w).
    """

    kind: int
    axis: np.ndarray
    order: int
    offset_quat: np.ndarray


def _identity_quat() -> np.ndarray:
    """Return the identity quaternion (x, y, z, w)."""
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _unit_vector(values: object, size: int, name: str) -> np.ndarray:
    """Return a finite, nonzero vector normalized to unit length."""
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape [{size}]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{name} must be nonzero")
    return (array / norm).astype(np.float32, copy=False)


def _unit_quats(values: object, name: str) -> np.ndarray:
    """Return finite, nonzero quaternions normalized to unit length."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"{name} must have shape [M, 4]")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one quaternion")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    if float(norm.min()) <= 1e-12:
        raise ValueError(f"{name} must contain nonzero quaternions")
    return (array / norm).astype(np.float32, copy=False)


def _axis_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return the unit quaternion for ``angle`` [rad] about ``axis``."""
    s = math.sin(0.5 * angle)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(0.5 * angle)], dtype=np.float32)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two unit quaternions in xyzw convention."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return _unit_vector(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        4,
        "composed symmetry quaternion",
    )


def _dedupe(quats: np.ndarray) -> np.ndarray:
    """Canonicalize signs and drop near-duplicate quaternions."""
    q = _unit_quats(quats, "symmetry orbit")
    sign = np.sign(q[:, 3])
    for row_idx in np.flatnonzero(np.abs(q[:, 3]) < 1e-6):
        lead = next((math.copysign(1.0, float(v)) for v in q[row_idx, :3] if abs(float(v)) > 1e-6), 1.0)
        sign[row_idx] = lead
    sign[sign == 0.0] = 1.0
    q = q * sign[:, None]
    keyed = np.round(q / _DEDUP_EPS) * _DEDUP_EPS
    _, keep = np.unique(keyed, axis=0, return_index=True)
    return q[np.sort(keep)].astype(np.float32, copy=False)


def _compose_orbit(tables: list[SymmetryElementTable]) -> np.ndarray:
    """Compose one quaternion from each element into a finite orbit."""
    if any(table.kind == KIND_CYCLIC and table.order == 0 for table in tables):
        raise ValueError("continuous axis (order=0) must be the sole element of an AssetSymmetryCfg")

    composed = [_identity_quat()]
    for combo in itertools.product(*(table.offset_quat for table in tables)):
        q = combo[0]
        for nxt in combo[1:]:
            q = _quat_mul(q, nxt)
        composed.append(q)
    return _dedupe(np.stack(composed))


def _element_table(element: SymmetryElementCfg) -> SymmetryElementTable:
    """Instantiate one element symmetry and return its compiled table."""
    symmetry = element.class_type(element)
    if not isinstance(symmetry, SymmetryElement):
        raise TypeError(f"{type(element).__name__}.class_type must construct a SymmetryElement")
    table = symmetry.table
    if not isinstance(table, SymmetryElementTable):
        raise TypeError(f"{type(element).__name__}.class_type must produce a SymmetryElementTable")

    kind = int(table.kind)
    if kind not in (KIND_CYCLIC, KIND_GENERAL):
        raise ValueError("SymmetryElementTable.kind must be KIND_CYCLIC or KIND_GENERAL")
    order = int(table.order)
    if order < 0:
        raise ValueError("SymmetryElementTable.order must be >= 0")
    if kind == KIND_GENERAL and order != 0:
        raise ValueError("general SymmetryElementTable.order must be 0")

    return SymmetryElementTable(
        kind=kind,
        axis=_unit_vector(table.axis, 3, "SymmetryElementTable.axis"),
        order=order,
        offset_quat=_unit_quats(table.offset_quat, "SymmetryElementTable.offset_quat"),
    )


class SymmetryElement:
    """Base class for compiled symmetry elements."""

    table: SymmetryElementTable


class AxisSymmetry(SymmetryElement):
    """Rotational symmetry element compiled from :class:`~.symmetry_cfg.AxisSymmetryCfg`."""

    def __init__(self, cfg: AxisSymmetryCfg):
        axis = _unit_vector(cfg.axis, 3, "AxisSymmetryCfg.axis")
        order = int(cfg.order)
        if order < 0:
            raise ValueError("AxisSymmetryCfg.order must be >= 0")

        if order == 0:
            offset_quat = _identity_quat()[None, :]
        else:
            offset_quat = np.stack([_axis_quat(axis, 2.0 * math.pi * k / order) for k in range(order)])
        self.table = SymmetryElementTable(KIND_CYCLIC, axis, order, offset_quat.astype(np.float32, copy=False))


class SemanticSymmetry(SymmetryElement):
    """Semantic symmetry element compiled from :class:`~.symmetry_cfg.SemanticSymmetryCfg`."""

    def __init__(self, cfg: SemanticSymmetryCfg):
        if cfg.offsets is MISSING:
            raise ValueError("SemanticSymmetryCfg.offsets must be set")
        if not cfg.offsets:
            raise ValueError("SemanticSymmetryCfg.offsets must contain at least one quaternion")
        self.table = SymmetryElementTable(
            KIND_GENERAL,
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            0,
            _unit_quats(cfg.offsets, "SemanticSymmetryCfg.offsets quaternions"),
        )


class AssetSymmetry:
    """Asset-level symmetry compiled from :class:`~.symmetry_cfg.AssetSymmetryCfg`."""

    def __init__(self, cfg: AssetSymmetryCfg):
        elements = list(cfg.elements)
        if not elements:
            raise ValueError("AssetSymmetryCfg.elements is empty; use AxisSymmetryCfg(order=1) for no symmetry")

        tables = [_element_table(element) for element in elements]
        if len(tables) == 1 and tables[0].kind == KIND_CYCLIC:
            self.table = SymmetryTableEntry(KIND_CYCLIC, tables[0].axis, tables[0].order, _identity_quat()[None, :])
            self.finite_orbit_quat = tables[0].offset_quat
        else:
            self.table = SymmetryTableEntry(
                KIND_GENERAL, np.array([0.0, 0.0, 1.0], dtype=np.float32), 0, _compose_orbit(tables)
            )
            self.finite_orbit_quat = self.table.offset_quat
