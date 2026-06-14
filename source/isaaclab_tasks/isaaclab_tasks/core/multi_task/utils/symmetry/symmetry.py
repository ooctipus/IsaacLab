# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batch Warp reduction over compiled asset symmetry tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .asset_symmetry import KIND_CYCLIC, AssetSymmetry, SymmetryTableEntry

if TYPE_CHECKING:
    from .symmetry_cfg import AssetSymmetryCfg


@wp.func
def _nearest_cyclic(held: wp.quatf, target: wp.quatf, axis: wp.vec3, order: int) -> wp.quatf:
    """Return the nearest target-equivalent orientation for one cyclic symmetry."""
    r = wp.mul(wp.quat_inverse(target), held)
    twist = r[0] * axis[0] + r[1] * axis[1] + r[2] * axis[2]
    phi = 2.0 * wp.atan2(twist, r[3])
    if order == 0:
        theta = phi
    else:
        n = float(order)
        theta = (2.0 * wp.pi / n) * wp.round(phi * n / (2.0 * wp.pi))
    s = wp.sin(0.5 * theta)
    offset = wp.quatf(axis[0] * s, axis[1] * s, axis[2] * s, wp.cos(0.5 * theta))
    return wp.mul(target, offset)


@wp.kernel
def _reduce_orientation_single_cyclic_kernel(
    held_quat: wp.array(dtype=wp.quatf),
    target_quat: wp.array(dtype=wp.quatf),
    axis: wp.vec3,
    order: int,
    orientation_error: wp.array(dtype=wp.float32),
    nearest: wp.array(dtype=wp.quatf),
):
    """Single-type cyclic reduction with no type lookup and no finite-orbit loop."""
    i = wp.tid()
    held = held_quat[i]
    target = target_quat[i]
    q_best = _nearest_cyclic(held, target, axis, order)

    orientation_error[i] = 2.0 * wp.acos(wp.min(wp.abs(wp.dot(held, q_best)), 1.0))
    nearest[i] = q_best


@wp.kernel
def _reduce_orientation_cyclic_kernel(
    held_quat: wp.array(dtype=wp.quatf),
    target_quat: wp.array(dtype=wp.quatf),
    type_id: wp.array(dtype=wp.int32),
    axis: wp.array(dtype=wp.vec3),
    order: wp.array(dtype=wp.int32),
    orientation_error: wp.array(dtype=wp.float32),
    nearest: wp.array(dtype=wp.quatf),
):
    """All-cyclic reduction with no kind branch and no finite-orbit loop."""
    i = wp.tid()
    t = type_id[i]
    held = held_quat[i]
    target = target_quat[i]
    q_best = _nearest_cyclic(held, target, axis[t], order[t])

    orientation_error[i] = 2.0 * wp.acos(wp.min(wp.abs(wp.dot(held, q_best)), 1.0))
    nearest[i] = q_best


@wp.kernel
def _reduce_orientation_kernel(
    held_quat: wp.array(dtype=wp.quatf),
    target_quat: wp.array(dtype=wp.quatf),
    type_id: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    axis: wp.array(dtype=wp.vec3),
    order: wp.array(dtype=wp.int32),
    indptr: wp.array(dtype=wp.int32),
    offset_quat: wp.array(dtype=wp.quatf),
    orientation_error: wp.array(dtype=wp.float32),
    nearest: wp.array(dtype=wp.quatf),
):
    """Mixed cyclic/general reduction over packed finite-orbit slices."""
    i = wp.tid()
    t = type_id[i]
    held = held_quat[i]
    target = target_quat[i]

    if kind[t] == 0:
        q_best = _nearest_cyclic(held, target, axis[t], order[t])
    else:
        lo = indptr[t]
        hi = indptr[t + 1]
        best_dot = float(-1.0)
        q_best = target
        for k in range(lo, hi):
            q_k = wp.mul(target, offset_quat[k])
            d = wp.abs(wp.dot(held, q_k))
            if d > best_dot:
                best_dot = d
                q_best = q_k

    orientation_error[i] = 2.0 * wp.acos(wp.min(wp.abs(wp.dot(held, q_best)), 1.0))
    nearest[i] = q_best


class Symmetry:
    """Symmetry-reduced orientation alignment over a flat batch of asset instances.

    Args:
        cfgs: Per-type symmetry definitions in type-id order.
        device: Warp device for the table arrays and kernel launch.
    """

    def __init__(self, cfgs: list[AssetSymmetryCfg], device: str):
        self.device = str(device)
        entries = []
        for cfg in cfgs:
            asset_symmetry = cfg.class_type(cfg)
            if not isinstance(asset_symmetry, AssetSymmetry):
                raise TypeError("AssetSymmetryCfg.class_type must construct an AssetSymmetry")
            if not isinstance(asset_symmetry.table, SymmetryTableEntry):
                raise TypeError("AssetSymmetryCfg.class_type must produce a SymmetryTableEntry")
            entries.append(asset_symmetry.table)
        self.num_types = len(entries)

        if entries:
            kind = np.asarray([entry.kind for entry in entries], dtype=np.int32)
            axis = np.stack([np.asarray(entry.axis, dtype=np.float32) for entry in entries]).astype(np.float32)
            order = np.asarray([entry.order for entry in entries], dtype=np.int32)
            segments = [
                np.asarray(entry.offset_quat, dtype=np.float32) for entry in entries if entry.kind != KIND_CYCLIC
            ]
            counts = np.asarray(
                [0 if entry.kind == KIND_CYCLIC else entry.offset_quat.shape[0] for entry in entries], dtype=np.int64
            )
            indptr = np.zeros(len(entries) + 1, dtype=np.int32)
            indptr[1:] = np.cumsum(counts, dtype=np.int64).astype(np.int32)
            offset_quat = (
                np.concatenate(segments, axis=0).astype(np.float32, copy=False)
                if segments
                else np.zeros((0, 4), dtype=np.float32)
            )
        else:
            kind = np.zeros(0, dtype=np.int32)
            axis = np.zeros((0, 3), dtype=np.float32)
            order = np.zeros(0, dtype=np.int32)
            indptr = np.zeros(1, dtype=np.int32)
            offset_quat = np.zeros((0, 4), dtype=np.float32)

        self._all_cyclic = self.num_types > 0 and bool(np.all(kind == KIND_CYCLIC))
        self._single_cyclic = self.num_types == 1 and self._all_cyclic
        if self._single_cyclic:
            self._single_axis = wp.vec3(float(axis[0, 0]), float(axis[0, 1]), float(axis[0, 2]))
            self._single_order = int(order[0])
        else:
            self._single_axis = wp.vec3(0.0, 0.0, 1.0)
            self._single_order = 1

        self._kind = wp.array(kind, dtype=wp.int32, device=self.device)
        self._axis = wp.array(axis, dtype=wp.vec3, device=self.device)
        self._order = wp.array(order, dtype=wp.int32, device=self.device)
        self._indptr = wp.array(indptr, dtype=wp.int32, device=self.device)
        self._offset_quat = wp.array(offset_quat, dtype=wp.quatf, device=self.device)

    def reduce_orientation(
        self,
        held_quat: wp.array(dtype=wp.quatf),
        target_quat: wp.array(dtype=wp.quatf),
        type_id: wp.array(dtype=wp.int32),
        out_error: wp.array(dtype=wp.float32),
        out_nearest: wp.array(dtype=wp.quatf),
    ) -> None:
        """Symmetry-reduce a flat instance batch (all args ``wp.array`` of length ``T``).

        For each instance, finds the nearest symmetry-equivalent of ``target_quat``
        to ``held_quat`` and writes the geodesic angle [rad] to ``out_error`` and
        that orientation (x, y, z, w) to ``out_nearest``. ``type_id`` indexes the
        per-type tables except for the single-cyclic fast path, where it is
        intentionally unused. No world/grid structure is assumed.
        """
        if self._single_cyclic:
            wp.launch(
                _reduce_orientation_single_cyclic_kernel,
                dim=held_quat.shape[0],
                inputs=[
                    held_quat,
                    target_quat,
                    self._single_axis,
                    self._single_order,
                    out_error,
                    out_nearest,
                ],
                device=self.device,
            )
            return

        if self._all_cyclic:
            wp.launch(
                _reduce_orientation_cyclic_kernel,
                dim=held_quat.shape[0],
                inputs=[
                    held_quat,
                    target_quat,
                    type_id,
                    self._axis,
                    self._order,
                    out_error,
                    out_nearest,
                ],
                device=self.device,
            )
            return

        wp.launch(
            _reduce_orientation_kernel,
            dim=held_quat.shape[0],
            inputs=[
                held_quat,
                target_quat,
                type_id,
                self._kind,
                self._axis,
                self._order,
                self._indptr,
                self._offset_quat,
                out_error,
                out_nearest,
            ],
            device=self.device,
        )
