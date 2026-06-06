# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Synthetic Warp benchmark for heterogeneous command dispatch.

This is intentionally standalone research code. It compares three execution
shapes for a workload that looks like ``MultiTaskCommand`` dispatch:

* ``mega``: one thread per work item, with a large runtime branch tree over
  64 synthetic state kernels.
* ``kind_queue``: one queue per synthetic state kernel. Kernels are branch-free,
  but launch count scales with the number of state kernels.
* ``primitive_queue_local_synth``: synthetic proxy for the production
  ``primitive_queue_local`` backend. Queues are grouped by the deeper shared
  primitive: vec3, scalar, quat, reductions, contact predicates, contact count
  differences, and local-frame vec3 projection.
* ``primitive_graph_local_synth``: synthetic proxy for the production
  ``primitive_graph_local`` backend. Primitive-family queues also use
  target-independent producer nodes: current vec3, current quat, reductions,
  contact mask, and frame basis.
* ``graph_packed_local_synth``: future-target variant of
  ``primitive_graph_local`` with shared producer nodes, parallel consumer
  kernels, and packed local outputs. This tests the real target layout:
  preserve consumer parallelism while avoiding dense scatter.

The point is not exact command semantics; it is to quantify whether the
homogeneity hierarchy is worth pursuing before touching production kernels.
Run with:

    ./isaaclab.sh -p -m isaaclab_tasks.core.multi_task.mdp.commands.benchmark.bench_dispatch_homogeneity
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
import warp as wp
import warp.utils as wpu

NUM_SYNTH_KINDS = 64
NUM_PRIMITIVES = 8
KINDS_PER_PRIMITIVE = NUM_SYNTH_KINDS // NUM_PRIMITIVES
SRC_WIDTH = 64
TGT_WIDTH = 8


@wp.kernel
def prepare_primitive_sort_kernel(
    kind: wp.array(dtype=wp.int32),
    keys: wp.array(dtype=wp.int64),
    values: wp.array(dtype=wp.int32),
    kinds_per_primitive: int,
):
    i = wp.tid()
    keys[i] = wp.int64(int(kind[i]) // kinds_per_primitive)
    values[i] = i


@wp.kernel
def prepare_kind_sort_kernel(
    kind: wp.array(dtype=wp.int32),
    keys: wp.array(dtype=wp.int64),
    values: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    keys[i] = wp.int64(kind[i])
    values[i] = i


@wp.func
def _act(err: float, param: float) -> float:
    return 1.0 - wp.tanh(err / param)


@wp.func
def _write_vec3(
    i: int,
    scale: float,
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
):
    d0 = (tgt[i, 0] - src[i, 0]) * scale
    d1 = (tgt[i, 1] - src[i, 1]) * scale
    d2 = (tgt[i, 2] - src[i, 2]) * scale
    err = wp.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    delta[i, 0] = d0
    delta[i, 1] = d1
    delta[i, 2] = d2
    delta[i, 3] = 0.0
    error[i] = err
    activation[i] = _act(err, act_param[i])


@wp.func
def _write_scalar(
    i: int,
    scale: float,
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
):
    d0 = (tgt[i, 0] - src[i, 0]) * scale
    err = wp.abs(d0)
    delta[i, 0] = d0
    delta[i, 1] = 0.0
    delta[i, 2] = 0.0
    delta[i, 3] = 0.0
    error[i] = err
    activation[i] = _act(err, act_param[i])


@wp.func
def _write_quat(
    i: int,
    scale: float,
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
):
    cx = src[i, 0]
    cy = src[i, 1]
    cz = src[i, 2]
    cw = src[i, 3]
    tx = tgt[i, 0]
    ty = tgt[i, 1]
    tz = tgt[i, 2]
    tw = tgt[i, 3]
    dw = cw * tw + cx * tx + cy * ty + cz * tz
    dx = (cw * tx - cx * tw - cy * tz + cz * ty) * scale
    dy = (cw * ty + cx * tz - cy * tw - cz * tx) * scale
    dz = (cw * tz - cx * ty + cy * tx - cz * tw) * scale
    v = wp.sqrt(dx * dx + dy * dy + dz * dz)
    err = 2.0 * wp.atan2(v, wp.abs(dw))
    delta[i, 0] = dx
    delta[i, 1] = dy
    delta[i, 2] = dz
    delta[i, 3] = dw
    error[i] = err
    activation[i] = _act(err, act_param[i])


@wp.func
def _write_reduce8(
    i: int,
    scale: float,
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
):
    total = float(0.0)
    for j in range(8):
        total = total + src[i, j]
    d0 = (tgt[i, 0] - total) * scale
    err = wp.abs(d0)
    delta[i, 0] = d0
    delta[i, 1] = 0.0
    delta[i, 2] = 0.0
    delta[i, 3] = 0.0
    error[i] = err
    activation[i] = _act(err, act_param[i])


@wp.func
def _write_reduce32(
    i: int,
    scale: float,
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
):
    total = float(0.0)
    for j in range(32):
        total = total + src[i, j] * src[i, j + 32]
    d0 = (tgt[i, 0] - total) * scale
    err = wp.abs(d0)
    delta[i, 0] = d0
    delta[i, 1] = 0.0
    delta[i, 2] = 0.0
    delta[i, 3] = 0.0
    error[i] = err
    activation[i] = _act(err, act_param[i])


@wp.func
def _write_contact_any16(
    i: int,
    scale: float,
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
):
    hit = float(0.0)
    threshold = tgt[i, 1]
    for j in range(16):
        if src[i, j + 16] > threshold:
            hit = 1.0
    d0 = (tgt[i, 0] - hit) * scale
    err = wp.abs(d0)
    delta[i, 0] = d0
    delta[i, 1] = 0.0
    delta[i, 2] = 0.0
    delta[i, 3] = 0.0
    error[i] = err
    activation[i] = _act(err, act_param[i])


@wp.func
def _write_contact_diff16(
    i: int,
    scale: float,
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
):
    left = float(0.0)
    right = float(0.0)
    threshold = tgt[i, 1]
    for j in range(8):
        if src[i, j + 16] > threshold:
            left = left + 1.0
        if src[i, j + 24] > threshold:
            right = right + 1.0
    d0 = (tgt[i, 0] - (left - right)) * scale
    err = wp.abs(d0)
    delta[i, 0] = d0
    delta[i, 1] = 0.0
    delta[i, 2] = 0.0
    delta[i, 3] = 0.0
    error[i] = err
    activation[i] = _act(err, act_param[i])


@wp.func
def _write_local_frame_vec3(
    i: int,
    scale: float,
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
):
    # Rotate target-position error into a body frame. This approximates the
    # factory-style "relative to fixed asset" path that mixes vec3 and quat work.
    vx = tgt[i, 0] - src[i, 0]
    vy = tgt[i, 1] - src[i, 1]
    vz = tgt[i, 2] - src[i, 2]
    qx = -src[i, 8]
    qy = -src[i, 9]
    qz = -src[i, 10]
    qw = src[i, 11]
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    d0 = (vx + qw * tx + qy * tz - qz * ty) * scale
    d1 = (vy + qw * ty + qz * tx - qx * tz) * scale
    d2 = (vz + qw * tz + qx * ty - qy * tx) * scale
    err = wp.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    delta[i, 0] = d0
    delta[i, 1] = d1
    delta[i, 2] = d2
    delta[i, 3] = 0.0
    error[i] = err
    activation[i] = _act(err, act_param[i])


@wp.func
def _write_vec3_scatter(
    q: int,
    dst: int,
    scale: float,
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
):
    d0 = (tgt[q, 0] - src[q, 0]) * scale
    d1 = (tgt[q, 1] - src[q, 1]) * scale
    d2 = (tgt[q, 2] - src[q, 2]) * scale
    err = wp.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    delta[dst, 0] = d0
    delta[dst, 1] = d1
    delta[dst, 2] = d2
    delta[dst, 3] = 0.0
    error[dst] = err
    activation[dst] = _act(err, act_param[q])


@wp.func
def _write_scalar_scatter(
    q: int,
    dst: int,
    scale: float,
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
):
    d0 = (tgt[q, 0] - src[q, 0]) * scale
    err = wp.abs(d0)
    delta[dst, 0] = d0
    delta[dst, 1] = 0.0
    delta[dst, 2] = 0.0
    delta[dst, 3] = 0.0
    error[dst] = err
    activation[dst] = _act(err, act_param[q])


@wp.func
def _write_quat_scatter(
    q: int,
    dst: int,
    scale: float,
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
):
    cx = src[q, 0]
    cy = src[q, 1]
    cz = src[q, 2]
    cw = src[q, 3]
    tx = tgt[q, 0]
    ty = tgt[q, 1]
    tz = tgt[q, 2]
    tw = tgt[q, 3]
    dw = cw * tw + cx * tx + cy * ty + cz * tz
    dx = (cw * tx - cx * tw - cy * tz + cz * ty) * scale
    dy = (cw * ty + cx * tz - cy * tw - cz * tx) * scale
    dz = (cw * tz - cx * ty + cy * tx - cz * tw) * scale
    v = wp.sqrt(dx * dx + dy * dy + dz * dz)
    err = 2.0 * wp.atan2(v, wp.abs(dw))
    delta[dst, 0] = dx
    delta[dst, 1] = dy
    delta[dst, 2] = dz
    delta[dst, 3] = dw
    error[dst] = err
    activation[dst] = _act(err, act_param[q])


@wp.func
def _write_reduce8_scatter(
    q: int,
    dst: int,
    scale: float,
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
):
    total = float(0.0)
    for j in range(8):
        total = total + src[q, j]
    d0 = (tgt[q, 0] - total) * scale
    err = wp.abs(d0)
    delta[dst, 0] = d0
    delta[dst, 1] = 0.0
    delta[dst, 2] = 0.0
    delta[dst, 3] = 0.0
    error[dst] = err
    activation[dst] = _act(err, act_param[q])


@wp.func
def _write_reduce32_scatter(
    q: int,
    dst: int,
    scale: float,
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
):
    total = float(0.0)
    for j in range(32):
        total = total + src[q, j] * src[q, j + 32]
    d0 = (tgt[q, 0] - total) * scale
    err = wp.abs(d0)
    delta[dst, 0] = d0
    delta[dst, 1] = 0.0
    delta[dst, 2] = 0.0
    delta[dst, 3] = 0.0
    error[dst] = err
    activation[dst] = _act(err, act_param[q])


@wp.func
def _write_contact_any16_scatter(
    q: int,
    dst: int,
    scale: float,
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
):
    hit = float(0.0)
    threshold = tgt[q, 1]
    for j in range(16):
        if src[q, j + 16] > threshold:
            hit = 1.0
    d0 = (tgt[q, 0] - hit) * scale
    err = wp.abs(d0)
    delta[dst, 0] = d0
    delta[dst, 1] = 0.0
    delta[dst, 2] = 0.0
    delta[dst, 3] = 0.0
    error[dst] = err
    activation[dst] = _act(err, act_param[q])


@wp.func
def _write_contact_diff16_scatter(
    q: int,
    dst: int,
    scale: float,
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
):
    left = float(0.0)
    right = float(0.0)
    threshold = tgt[q, 1]
    for j in range(8):
        if src[q, j + 16] > threshold:
            left = left + 1.0
        if src[q, j + 24] > threshold:
            right = right + 1.0
    d0 = (tgt[q, 0] - (left - right)) * scale
    err = wp.abs(d0)
    delta[dst, 0] = d0
    delta[dst, 1] = 0.0
    delta[dst, 2] = 0.0
    delta[dst, 3] = 0.0
    error[dst] = err
    activation[dst] = _act(err, act_param[q])


@wp.func
def _write_local_frame_vec3_scatter(
    q: int,
    dst: int,
    scale: float,
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
):
    vx = tgt[q, 0] - src[q, 0]
    vy = tgt[q, 1] - src[q, 1]
    vz = tgt[q, 2] - src[q, 2]
    qx = -src[q, 8]
    qy = -src[q, 9]
    qz = -src[q, 10]
    qw = src[q, 11]
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    d0 = (vx + qw * tx + qy * tz - qz * ty) * scale
    d1 = (vy + qw * ty + qz * tx - qx * tz) * scale
    d2 = (vz + qw * tz + qx * ty - qy * tx) * scale
    err = wp.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    delta[dst, 0] = d0
    delta[dst, 1] = d1
    delta[dst, 2] = d2
    delta[dst, 3] = 0.0
    error[dst] = err
    activation[dst] = _act(err, act_param[q])


@wp.kernel
def mega_dispatch_kernel(  # noqa: C901 - intentionally bad mega-kernel baseline.
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    k = int(kind[i])
    if k == 0:
        _write_vec3(i, scale[0], src, tgt, act_param, delta, error, activation)
    elif k == 1:
        _write_vec3(i, scale[1], src, tgt, act_param, delta, error, activation)
    elif k == 2:
        _write_vec3(i, scale[2], src, tgt, act_param, delta, error, activation)
    elif k == 3:
        _write_vec3(i, scale[3], src, tgt, act_param, delta, error, activation)
    elif k == 4:
        _write_vec3(i, scale[4], src, tgt, act_param, delta, error, activation)
    elif k == 5:
        _write_vec3(i, scale[5], src, tgt, act_param, delta, error, activation)
    elif k == 6:
        _write_vec3(i, scale[6], src, tgt, act_param, delta, error, activation)
    elif k == 7:
        _write_vec3(i, scale[7], src, tgt, act_param, delta, error, activation)
    elif k == 8:
        _write_scalar(i, scale[8], src, tgt, act_param, delta, error, activation)
    elif k == 9:
        _write_scalar(i, scale[9], src, tgt, act_param, delta, error, activation)
    elif k == 10:
        _write_scalar(i, scale[10], src, tgt, act_param, delta, error, activation)
    elif k == 11:
        _write_scalar(i, scale[11], src, tgt, act_param, delta, error, activation)
    elif k == 12:
        _write_scalar(i, scale[12], src, tgt, act_param, delta, error, activation)
    elif k == 13:
        _write_scalar(i, scale[13], src, tgt, act_param, delta, error, activation)
    elif k == 14:
        _write_scalar(i, scale[14], src, tgt, act_param, delta, error, activation)
    elif k == 15:
        _write_scalar(i, scale[15], src, tgt, act_param, delta, error, activation)
    elif k == 16:
        _write_quat(i, scale[16], src, tgt, act_param, delta, error, activation)
    elif k == 17:
        _write_quat(i, scale[17], src, tgt, act_param, delta, error, activation)
    elif k == 18:
        _write_quat(i, scale[18], src, tgt, act_param, delta, error, activation)
    elif k == 19:
        _write_quat(i, scale[19], src, tgt, act_param, delta, error, activation)
    elif k == 20:
        _write_quat(i, scale[20], src, tgt, act_param, delta, error, activation)
    elif k == 21:
        _write_quat(i, scale[21], src, tgt, act_param, delta, error, activation)
    elif k == 22:
        _write_quat(i, scale[22], src, tgt, act_param, delta, error, activation)
    elif k == 23:
        _write_quat(i, scale[23], src, tgt, act_param, delta, error, activation)
    elif k == 24:
        _write_reduce8(i, scale[24], src, tgt, act_param, delta, error, activation)
    elif k == 25:
        _write_reduce8(i, scale[25], src, tgt, act_param, delta, error, activation)
    elif k == 26:
        _write_reduce8(i, scale[26], src, tgt, act_param, delta, error, activation)
    elif k == 27:
        _write_reduce8(i, scale[27], src, tgt, act_param, delta, error, activation)
    elif k == 28:
        _write_reduce8(i, scale[28], src, tgt, act_param, delta, error, activation)
    elif k == 29:
        _write_reduce8(i, scale[29], src, tgt, act_param, delta, error, activation)
    elif k == 30:
        _write_reduce8(i, scale[30], src, tgt, act_param, delta, error, activation)
    elif k == 31:
        _write_reduce8(i, scale[31], src, tgt, act_param, delta, error, activation)
    elif k == 32:
        _write_reduce32(i, scale[32], src, tgt, act_param, delta, error, activation)
    elif k == 33:
        _write_reduce32(i, scale[33], src, tgt, act_param, delta, error, activation)
    elif k == 34:
        _write_reduce32(i, scale[34], src, tgt, act_param, delta, error, activation)
    elif k == 35:
        _write_reduce32(i, scale[35], src, tgt, act_param, delta, error, activation)
    elif k == 36:
        _write_reduce32(i, scale[36], src, tgt, act_param, delta, error, activation)
    elif k == 37:
        _write_reduce32(i, scale[37], src, tgt, act_param, delta, error, activation)
    elif k == 38:
        _write_reduce32(i, scale[38], src, tgt, act_param, delta, error, activation)
    elif k == 39:
        _write_reduce32(i, scale[39], src, tgt, act_param, delta, error, activation)
    elif k == 40:
        _write_contact_any16(i, scale[40], src, tgt, act_param, delta, error, activation)
    elif k == 41:
        _write_contact_any16(i, scale[41], src, tgt, act_param, delta, error, activation)
    elif k == 42:
        _write_contact_any16(i, scale[42], src, tgt, act_param, delta, error, activation)
    elif k == 43:
        _write_contact_any16(i, scale[43], src, tgt, act_param, delta, error, activation)
    elif k == 44:
        _write_contact_any16(i, scale[44], src, tgt, act_param, delta, error, activation)
    elif k == 45:
        _write_contact_any16(i, scale[45], src, tgt, act_param, delta, error, activation)
    elif k == 46:
        _write_contact_any16(i, scale[46], src, tgt, act_param, delta, error, activation)
    elif k == 47:
        _write_contact_any16(i, scale[47], src, tgt, act_param, delta, error, activation)
    elif k == 48:
        _write_contact_diff16(i, scale[48], src, tgt, act_param, delta, error, activation)
    elif k == 49:
        _write_contact_diff16(i, scale[49], src, tgt, act_param, delta, error, activation)
    elif k == 50:
        _write_contact_diff16(i, scale[50], src, tgt, act_param, delta, error, activation)
    elif k == 51:
        _write_contact_diff16(i, scale[51], src, tgt, act_param, delta, error, activation)
    elif k == 52:
        _write_contact_diff16(i, scale[52], src, tgt, act_param, delta, error, activation)
    elif k == 53:
        _write_contact_diff16(i, scale[53], src, tgt, act_param, delta, error, activation)
    elif k == 54:
        _write_contact_diff16(i, scale[54], src, tgt, act_param, delta, error, activation)
    elif k == 55:
        _write_contact_diff16(i, scale[55], src, tgt, act_param, delta, error, activation)
    elif k == 56:
        _write_local_frame_vec3(i, scale[56], src, tgt, act_param, delta, error, activation)
    elif k == 57:
        _write_local_frame_vec3(i, scale[57], src, tgt, act_param, delta, error, activation)
    elif k == 58:
        _write_local_frame_vec3(i, scale[58], src, tgt, act_param, delta, error, activation)
    elif k == 59:
        _write_local_frame_vec3(i, scale[59], src, tgt, act_param, delta, error, activation)
    elif k == 60:
        _write_local_frame_vec3(i, scale[60], src, tgt, act_param, delta, error, activation)
    elif k == 61:
        _write_local_frame_vec3(i, scale[61], src, tgt, act_param, delta, error, activation)
    elif k == 62:
        _write_local_frame_vec3(i, scale[62], src, tgt, act_param, delta, error, activation)
    else:
        _write_local_frame_vec3(i, scale[63], src, tgt, act_param, delta, error, activation)


@wp.kernel
def indexed_mega_dispatch_kernel(  # noqa: C901 - intentionally bad indexed mega-kernel baseline.
    work_ids: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    k = int(kind[i])
    if k == 0:
        _write_vec3(i, scale[0], src, tgt, act_param, delta, error, activation)
    elif k == 1:
        _write_vec3(i, scale[1], src, tgt, act_param, delta, error, activation)
    elif k == 2:
        _write_vec3(i, scale[2], src, tgt, act_param, delta, error, activation)
    elif k == 3:
        _write_vec3(i, scale[3], src, tgt, act_param, delta, error, activation)
    elif k == 4:
        _write_vec3(i, scale[4], src, tgt, act_param, delta, error, activation)
    elif k == 5:
        _write_vec3(i, scale[5], src, tgt, act_param, delta, error, activation)
    elif k == 6:
        _write_vec3(i, scale[6], src, tgt, act_param, delta, error, activation)
    elif k == 7:
        _write_vec3(i, scale[7], src, tgt, act_param, delta, error, activation)
    elif k == 8:
        _write_scalar(i, scale[8], src, tgt, act_param, delta, error, activation)
    elif k == 9:
        _write_scalar(i, scale[9], src, tgt, act_param, delta, error, activation)
    elif k == 10:
        _write_scalar(i, scale[10], src, tgt, act_param, delta, error, activation)
    elif k == 11:
        _write_scalar(i, scale[11], src, tgt, act_param, delta, error, activation)
    elif k == 12:
        _write_scalar(i, scale[12], src, tgt, act_param, delta, error, activation)
    elif k == 13:
        _write_scalar(i, scale[13], src, tgt, act_param, delta, error, activation)
    elif k == 14:
        _write_scalar(i, scale[14], src, tgt, act_param, delta, error, activation)
    elif k == 15:
        _write_scalar(i, scale[15], src, tgt, act_param, delta, error, activation)
    elif k == 16:
        _write_quat(i, scale[16], src, tgt, act_param, delta, error, activation)
    elif k == 17:
        _write_quat(i, scale[17], src, tgt, act_param, delta, error, activation)
    elif k == 18:
        _write_quat(i, scale[18], src, tgt, act_param, delta, error, activation)
    elif k == 19:
        _write_quat(i, scale[19], src, tgt, act_param, delta, error, activation)
    elif k == 20:
        _write_quat(i, scale[20], src, tgt, act_param, delta, error, activation)
    elif k == 21:
        _write_quat(i, scale[21], src, tgt, act_param, delta, error, activation)
    elif k == 22:
        _write_quat(i, scale[22], src, tgt, act_param, delta, error, activation)
    elif k == 23:
        _write_quat(i, scale[23], src, tgt, act_param, delta, error, activation)
    elif k == 24:
        _write_reduce8(i, scale[24], src, tgt, act_param, delta, error, activation)
    elif k == 25:
        _write_reduce8(i, scale[25], src, tgt, act_param, delta, error, activation)
    elif k == 26:
        _write_reduce8(i, scale[26], src, tgt, act_param, delta, error, activation)
    elif k == 27:
        _write_reduce8(i, scale[27], src, tgt, act_param, delta, error, activation)
    elif k == 28:
        _write_reduce8(i, scale[28], src, tgt, act_param, delta, error, activation)
    elif k == 29:
        _write_reduce8(i, scale[29], src, tgt, act_param, delta, error, activation)
    elif k == 30:
        _write_reduce8(i, scale[30], src, tgt, act_param, delta, error, activation)
    elif k == 31:
        _write_reduce8(i, scale[31], src, tgt, act_param, delta, error, activation)
    elif k == 32:
        _write_reduce32(i, scale[32], src, tgt, act_param, delta, error, activation)
    elif k == 33:
        _write_reduce32(i, scale[33], src, tgt, act_param, delta, error, activation)
    elif k == 34:
        _write_reduce32(i, scale[34], src, tgt, act_param, delta, error, activation)
    elif k == 35:
        _write_reduce32(i, scale[35], src, tgt, act_param, delta, error, activation)
    elif k == 36:
        _write_reduce32(i, scale[36], src, tgt, act_param, delta, error, activation)
    elif k == 37:
        _write_reduce32(i, scale[37], src, tgt, act_param, delta, error, activation)
    elif k == 38:
        _write_reduce32(i, scale[38], src, tgt, act_param, delta, error, activation)
    elif k == 39:
        _write_reduce32(i, scale[39], src, tgt, act_param, delta, error, activation)
    elif k == 40:
        _write_contact_any16(i, scale[40], src, tgt, act_param, delta, error, activation)
    elif k == 41:
        _write_contact_any16(i, scale[41], src, tgt, act_param, delta, error, activation)
    elif k == 42:
        _write_contact_any16(i, scale[42], src, tgt, act_param, delta, error, activation)
    elif k == 43:
        _write_contact_any16(i, scale[43], src, tgt, act_param, delta, error, activation)
    elif k == 44:
        _write_contact_any16(i, scale[44], src, tgt, act_param, delta, error, activation)
    elif k == 45:
        _write_contact_any16(i, scale[45], src, tgt, act_param, delta, error, activation)
    elif k == 46:
        _write_contact_any16(i, scale[46], src, tgt, act_param, delta, error, activation)
    elif k == 47:
        _write_contact_any16(i, scale[47], src, tgt, act_param, delta, error, activation)
    elif k == 48:
        _write_contact_diff16(i, scale[48], src, tgt, act_param, delta, error, activation)
    elif k == 49:
        _write_contact_diff16(i, scale[49], src, tgt, act_param, delta, error, activation)
    elif k == 50:
        _write_contact_diff16(i, scale[50], src, tgt, act_param, delta, error, activation)
    elif k == 51:
        _write_contact_diff16(i, scale[51], src, tgt, act_param, delta, error, activation)
    elif k == 52:
        _write_contact_diff16(i, scale[52], src, tgt, act_param, delta, error, activation)
    elif k == 53:
        _write_contact_diff16(i, scale[53], src, tgt, act_param, delta, error, activation)
    elif k == 54:
        _write_contact_diff16(i, scale[54], src, tgt, act_param, delta, error, activation)
    elif k == 55:
        _write_contact_diff16(i, scale[55], src, tgt, act_param, delta, error, activation)
    elif k == 56:
        _write_local_frame_vec3(i, scale[56], src, tgt, act_param, delta, error, activation)
    elif k == 57:
        _write_local_frame_vec3(i, scale[57], src, tgt, act_param, delta, error, activation)
    elif k == 58:
        _write_local_frame_vec3(i, scale[58], src, tgt, act_param, delta, error, activation)
    elif k == 59:
        _write_local_frame_vec3(i, scale[59], src, tgt, act_param, delta, error, activation)
    elif k == 60:
        _write_local_frame_vec3(i, scale[60], src, tgt, act_param, delta, error, activation)
    elif k == 61:
        _write_local_frame_vec3(i, scale[61], src, tgt, act_param, delta, error, activation)
    elif k == 62:
        _write_local_frame_vec3(i, scale[62], src, tgt, act_param, delta, error, activation)
    else:
        _write_local_frame_vec3(i, scale[63], src, tgt, act_param, delta, error, activation)


@wp.kernel
def queue_vec3_kernel(
    work_ids: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    _write_vec3(i, scale[int(kind[i])], src, tgt, act_param, delta, error, activation)


@wp.kernel
def queue_scalar_kernel(
    work_ids: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    _write_scalar(i, scale[int(kind[i])], src, tgt, act_param, delta, error, activation)


@wp.kernel
def queue_quat_kernel(
    work_ids: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    _write_quat(i, scale[int(kind[i])], src, tgt, act_param, delta, error, activation)


@wp.kernel
def queue_reduce8_kernel(
    work_ids: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    _write_reduce8(i, scale[int(kind[i])], src, tgt, act_param, delta, error, activation)


@wp.kernel
def queue_reduce32_kernel(
    work_ids: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    _write_reduce32(i, scale[int(kind[i])], src, tgt, act_param, delta, error, activation)


@wp.kernel
def queue_contact_any16_kernel(
    work_ids: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    _write_contact_any16(i, scale[int(kind[i])], src, tgt, act_param, delta, error, activation)


@wp.kernel
def queue_contact_diff16_kernel(
    work_ids: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    _write_contact_diff16(i, scale[int(kind[i])], src, tgt, act_param, delta, error, activation)


@wp.kernel
def queue_local_frame_vec3_kernel(
    work_ids: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    _write_local_frame_vec3(i, scale[int(kind[i])], src, tgt, act_param, delta, error, activation)


@wp.kernel
def packed_vec3_scatter_kernel(
    work_ids: wp.array(dtype=wp.int32),
    packed_kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    _write_vec3_scatter(q, int(work_ids[q]), scale[int(packed_kind[q])], src, tgt, act_param, delta, error, activation)


@wp.kernel
def packed_scalar_scatter_kernel(
    work_ids: wp.array(dtype=wp.int32),
    packed_kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    _write_scalar_scatter(
        q, int(work_ids[q]), scale[int(packed_kind[q])], src, tgt, act_param, delta, error, activation
    )


@wp.kernel
def packed_quat_scatter_kernel(
    work_ids: wp.array(dtype=wp.int32),
    packed_kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    _write_quat_scatter(q, int(work_ids[q]), scale[int(packed_kind[q])], src, tgt, act_param, delta, error, activation)


@wp.kernel
def packed_reduce8_scatter_kernel(
    work_ids: wp.array(dtype=wp.int32),
    packed_kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    _write_reduce8_scatter(
        q, int(work_ids[q]), scale[int(packed_kind[q])], src, tgt, act_param, delta, error, activation
    )


@wp.kernel
def packed_reduce32_scatter_kernel(
    work_ids: wp.array(dtype=wp.int32),
    packed_kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    _write_reduce32_scatter(
        q, int(work_ids[q]), scale[int(packed_kind[q])], src, tgt, act_param, delta, error, activation
    )


@wp.kernel
def packed_contact_any16_scatter_kernel(
    work_ids: wp.array(dtype=wp.int32),
    packed_kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    _write_contact_any16_scatter(
        q, int(work_ids[q]), scale[int(packed_kind[q])], src, tgt, act_param, delta, error, activation
    )


@wp.kernel
def packed_contact_diff16_scatter_kernel(
    work_ids: wp.array(dtype=wp.int32),
    packed_kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    _write_contact_diff16_scatter(
        q, int(work_ids[q]), scale[int(packed_kind[q])], src, tgt, act_param, delta, error, activation
    )


@wp.kernel
def packed_local_frame_vec3_scatter_kernel(
    work_ids: wp.array(dtype=wp.int32),
    packed_kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    _write_local_frame_vec3_scatter(
        q, int(work_ids[q]), scale[int(packed_kind[q])], src, tgt, act_param, delta, error, activation
    )


@wp.kernel
def contact_predicate16_graph_kernel(
    node_work_ids: wp.array(dtype=wp.int32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    contact_mask: wp.array2d(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(node_work_ids[q])
    threshold = tgt[i, 1]
    for j in range(16):
        hit = float(0.0)
        if src[i, j + 16] > threshold:
            hit = 1.0
        contact_mask[q, j] = hit


@wp.kernel
def graph_contact_any16_kernel(
    work_ids: wp.array(dtype=wp.int32),
    node_ids_by_item: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    contact_mask: wp.array2d(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    node = int(node_ids_by_item[i])
    hit = float(0.0)
    for j in range(16):
        if contact_mask[node, j] > 0.0:
            hit = 1.0
    d0 = (tgt[i, 0] - hit) * scale[int(kind[i])]
    err = wp.abs(d0)
    delta[i, 0] = d0
    delta[i, 1] = 0.0
    delta[i, 2] = 0.0
    delta[i, 3] = 0.0
    error[i] = err
    activation[i] = _act(err, act_param[i])


@wp.kernel
def graph_contact_diff16_kernel(
    work_ids: wp.array(dtype=wp.int32),
    node_ids_by_item: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    contact_mask: wp.array2d(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    node = int(node_ids_by_item[i])
    left = float(0.0)
    right = float(0.0)
    for j in range(8):
        left = left + contact_mask[node, j]
        right = right + contact_mask[node, j + 8]
    d0 = (tgt[i, 0] - (left - right)) * scale[int(kind[i])]
    err = wp.abs(d0)
    delta[i, 0] = d0
    delta[i, 1] = 0.0
    delta[i, 2] = 0.0
    delta[i, 3] = 0.0
    error[i] = err
    activation[i] = _act(err, act_param[i])


@wp.kernel
def current_vec3_graph_kernel(
    node_work_ids: wp.array(dtype=wp.int32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    graph_vec3: wp.array2d(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(node_work_ids[q])
    graph_vec3[q, 0] = src[i, 0]
    graph_vec3[q, 1] = src[i, 1]
    graph_vec3[q, 2] = src[i, 2]


@wp.kernel
def vec3_delta_graph_kernel(
    node_work_ids: wp.array(dtype=wp.int32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    graph_vec3: wp.array2d(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(node_work_ids[q])
    graph_vec3[q, 0] = tgt[i, 0] - src[i, 0]
    graph_vec3[q, 1] = tgt[i, 1] - src[i, 1]
    graph_vec3[q, 2] = tgt[i, 2] - src[i, 2]


@wp.kernel
def graph_current_vec3_kernel(
    work_ids: wp.array(dtype=wp.int32),
    node_ids_by_item: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    graph_vec3: wp.array2d(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    node = int(node_ids_by_item[i])
    s = scale[int(kind[i])]
    d0 = (tgt[i, 0] - graph_vec3[node, 0]) * s
    d1 = (tgt[i, 1] - graph_vec3[node, 1]) * s
    d2 = (tgt[i, 2] - graph_vec3[node, 2]) * s
    err = wp.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    delta[i, 0] = d0
    delta[i, 1] = d1
    delta[i, 2] = d2
    delta[i, 3] = 0.0
    error[i] = err
    activation[i] = _act(err, act_param[i])


@wp.kernel
def current_quat_graph_kernel(
    node_work_ids: wp.array(dtype=wp.int32),
    src: wp.array2d(dtype=wp.float32),
    graph_quat: wp.array2d(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(node_work_ids[q])
    graph_quat[q, 0] = src[i, 0]
    graph_quat[q, 1] = src[i, 1]
    graph_quat[q, 2] = src[i, 2]
    graph_quat[q, 3] = src[i, 3]


@wp.kernel
def graph_quat_kernel(
    work_ids: wp.array(dtype=wp.int32),
    node_ids_by_item: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    graph_quat: wp.array2d(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    node = int(node_ids_by_item[i])
    cx = graph_quat[node, 0]
    cy = graph_quat[node, 1]
    cz = graph_quat[node, 2]
    cw = graph_quat[node, 3]
    tx = tgt[i, 0]
    ty = tgt[i, 1]
    tz = tgt[i, 2]
    tw = tgt[i, 3]
    s = scale[int(kind[i])]
    dw = cw * tw + cx * tx + cy * ty + cz * tz
    dx = (cw * tx - cx * tw - cy * tz + cz * ty) * s
    dy = (cw * ty + cx * tz - cy * tw - cz * tx) * s
    dz = (cw * tz - cx * ty + cy * tx - cz * tw) * s
    v = wp.sqrt(dx * dx + dy * dy + dz * dz)
    err = 2.0 * wp.atan2(v, wp.abs(dw))
    delta[i, 0] = dx
    delta[i, 1] = dy
    delta[i, 2] = dz
    delta[i, 3] = dw
    error[i] = err
    activation[i] = _act(err, act_param[i])


@wp.kernel
def reduce8_graph_kernel(
    node_work_ids: wp.array(dtype=wp.int32),
    src: wp.array2d(dtype=wp.float32),
    graph_scalar: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(node_work_ids[q])
    total = float(0.0)
    for j in range(8):
        total = total + src[i, j]
    graph_scalar[q] = total


@wp.kernel
def reduce32_graph_kernel(
    node_work_ids: wp.array(dtype=wp.int32),
    src: wp.array2d(dtype=wp.float32),
    graph_scalar: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(node_work_ids[q])
    total = float(0.0)
    for j in range(32):
        total = total + src[i, j] * src[i, j + 32]
    graph_scalar[q] = total


@wp.kernel
def graph_scalar_reduce_kernel(
    work_ids: wp.array(dtype=wp.int32),
    node_ids_by_item: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    graph_scalar: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    node = int(node_ids_by_item[i])
    d0 = (tgt[i, 0] - graph_scalar[node]) * scale[int(kind[i])]
    err = wp.abs(d0)
    delta[i, 0] = d0
    delta[i, 1] = 0.0
    delta[i, 2] = 0.0
    delta[i, 3] = 0.0
    error[i] = err
    activation[i] = _act(err, act_param[i])


@wp.kernel
def local_frame_vec3_graph_kernel(
    node_work_ids: wp.array(dtype=wp.int32),
    src: wp.array2d(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    graph_vec3: wp.array2d(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(node_work_ids[q])
    vx = tgt[i, 0] - src[i, 0]
    vy = tgt[i, 1] - src[i, 1]
    vz = tgt[i, 2] - src[i, 2]
    qx = -src[i, 8]
    qy = -src[i, 9]
    qz = -src[i, 10]
    qw = src[i, 11]
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    graph_vec3[q, 0] = vx + qw * tx + qy * tz - qz * ty
    graph_vec3[q, 1] = vy + qw * ty + qz * tx - qx * tz
    graph_vec3[q, 2] = vz + qw * tz + qx * ty - qy * tx


@wp.kernel
def frame_basis_graph_kernel(
    node_work_ids: wp.array(dtype=wp.int32),
    src: wp.array2d(dtype=wp.float32),
    graph_frame: wp.array2d(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(node_work_ids[q])
    qx = -src[i, 8]
    qy = -src[i, 9]
    qz = -src[i, 10]
    qw = src[i, 11]
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    graph_frame[q, 0] = src[i, 0]
    graph_frame[q, 1] = src[i, 1]
    graph_frame[q, 2] = src[i, 2]
    graph_frame[q, 3] = 1.0 - 2.0 * (yy + zz)
    graph_frame[q, 4] = 2.0 * (xy - wz)
    graph_frame[q, 5] = 2.0 * (xz + wy)
    graph_frame[q, 6] = 2.0 * (xy + wz)
    graph_frame[q, 7] = 1.0 - 2.0 * (xx + zz)
    graph_frame[q, 8] = 2.0 * (yz - wx)
    graph_frame[q, 9] = 2.0 * (xz - wy)
    graph_frame[q, 10] = 2.0 * (yz + wx)
    graph_frame[q, 11] = 1.0 - 2.0 * (xx + yy)


@wp.kernel
def graph_frame_vec3_kernel(
    work_ids: wp.array(dtype=wp.int32),
    node_ids_by_item: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    graph_frame: wp.array2d(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    node = int(node_ids_by_item[i])
    vx = tgt[i, 0] - graph_frame[node, 0]
    vy = tgt[i, 1] - graph_frame[node, 1]
    vz = tgt[i, 2] - graph_frame[node, 2]
    s = scale[int(kind[i])]
    d0 = (graph_frame[node, 3] * vx + graph_frame[node, 4] * vy + graph_frame[node, 5] * vz) * s
    d1 = (graph_frame[node, 6] * vx + graph_frame[node, 7] * vy + graph_frame[node, 8] * vz) * s
    d2 = (graph_frame[node, 9] * vx + graph_frame[node, 10] * vy + graph_frame[node, 11] * vz) * s
    err = wp.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    delta[i, 0] = d0
    delta[i, 1] = d1
    delta[i, 2] = d2
    delta[i, 3] = 0.0
    error[i] = err
    activation[i] = _act(err, act_param[i])


@wp.kernel
def graph_vec3_local_kernel(
    work_ids: wp.array(dtype=wp.int32),
    node_ids_by_item: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    graph_vec3: wp.array2d(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    node = int(node_ids_by_item[i])
    s = scale[int(kind[i])]
    d0 = (tgt[i, 0] - graph_vec3[node, 0]) * s
    d1 = (tgt[i, 1] - graph_vec3[node, 1]) * s
    d2 = (tgt[i, 2] - graph_vec3[node, 2]) * s
    err = wp.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    delta[q, 0] = d0
    delta[q, 1] = d1
    delta[q, 2] = d2
    delta[q, 3] = 0.0
    error[q] = err
    activation[q] = _act(err, act_param[i])


@wp.kernel
def graph_quat_local_kernel(
    work_ids: wp.array(dtype=wp.int32),
    node_ids_by_item: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    graph_quat: wp.array2d(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    node = int(node_ids_by_item[i])
    cx = graph_quat[node, 0]
    cy = graph_quat[node, 1]
    cz = graph_quat[node, 2]
    cw = graph_quat[node, 3]
    tx = tgt[i, 0]
    ty = tgt[i, 1]
    tz = tgt[i, 2]
    tw = tgt[i, 3]
    s = scale[int(kind[i])]
    dw = cw * tw + cx * tx + cy * ty + cz * tz
    dx = (cw * tx - cx * tw - cy * tz + cz * ty) * s
    dy = (cw * ty + cx * tz - cy * tw - cz * tx) * s
    dz = (cw * tz - cx * ty + cy * tx - cz * tw) * s
    v = wp.sqrt(dx * dx + dy * dy + dz * dz)
    err = 2.0 * wp.atan2(v, wp.abs(dw))
    delta[q, 0] = dx
    delta[q, 1] = dy
    delta[q, 2] = dz
    delta[q, 3] = dw
    error[q] = err
    activation[q] = _act(err, act_param[i])


@wp.kernel
def graph_scalar_reduce_local_kernel(
    work_ids: wp.array(dtype=wp.int32),
    node_ids_by_item: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    graph_scalar: wp.array(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    node = int(node_ids_by_item[i])
    d0 = (tgt[i, 0] - graph_scalar[node]) * scale[int(kind[i])]
    err = wp.abs(d0)
    delta[q, 0] = d0
    delta[q, 1] = 0.0
    delta[q, 2] = 0.0
    delta[q, 3] = 0.0
    error[q] = err
    activation[q] = _act(err, act_param[i])


@wp.kernel
def graph_frame_vec3_local_kernel(
    work_ids: wp.array(dtype=wp.int32),
    node_ids_by_item: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    graph_frame: wp.array2d(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    node = int(node_ids_by_item[i])
    vx = tgt[i, 0] - graph_frame[node, 0]
    vy = tgt[i, 1] - graph_frame[node, 1]
    vz = tgt[i, 2] - graph_frame[node, 2]
    s = scale[int(kind[i])]
    d0 = (graph_frame[node, 3] * vx + graph_frame[node, 4] * vy + graph_frame[node, 5] * vz) * s
    d1 = (graph_frame[node, 6] * vx + graph_frame[node, 7] * vy + graph_frame[node, 8] * vz) * s
    d2 = (graph_frame[node, 9] * vx + graph_frame[node, 10] * vy + graph_frame[node, 11] * vz) * s
    err = wp.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    delta[q, 0] = d0
    delta[q, 1] = d1
    delta[q, 2] = d2
    delta[q, 3] = 0.0
    error[q] = err
    activation[q] = _act(err, act_param[i])


@wp.kernel
def graph_contact_any16_local_kernel(
    work_ids: wp.array(dtype=wp.int32),
    node_ids_by_item: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    contact_mask: wp.array2d(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    node = int(node_ids_by_item[i])
    hit = float(0.0)
    for j in range(16):
        if contact_mask[node, j] > 0.0:
            hit = 1.0
    d0 = (tgt[i, 0] - hit) * scale[int(kind[i])]
    err = wp.abs(d0)
    delta[q, 0] = d0
    delta[q, 1] = 0.0
    delta[q, 2] = 0.0
    delta[q, 3] = 0.0
    error[q] = err
    activation[q] = _act(err, act_param[i])


@wp.kernel
def graph_contact_diff16_local_kernel(
    work_ids: wp.array(dtype=wp.int32),
    node_ids_by_item: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    scale: wp.array(dtype=wp.float32),
    tgt: wp.array2d(dtype=wp.float32),
    act_param: wp.array(dtype=wp.float32),
    contact_mask: wp.array2d(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),
    count: int,
):
    q = wp.tid()
    if q >= count:
        return
    i = int(work_ids[q])
    node = int(node_ids_by_item[i])
    left = float(0.0)
    right = float(0.0)
    for j in range(8):
        left = left + contact_mask[node, j]
        right = right + contact_mask[node, j + 8]
    d0 = (tgt[i, 0] - (left - right)) * scale[int(kind[i])]
    err = wp.abs(d0)
    delta[q, 0] = d0
    delta[q, 1] = 0.0
    delta[q, 2] = 0.0
    delta[q, 3] = 0.0
    error[q] = err
    activation[q] = _act(err, act_param[i])


@dataclass
class VariantOutput:
    delta: torch.Tensor
    error: torch.Tensor
    activation: torch.Tensor


@dataclass
class Workload:
    kind: torch.Tensor
    src: torch.Tensor
    tgt: torch.Tensor
    act_param: torch.Tensor
    scale: torch.Tensor
    sort_keys: torch.Tensor
    sort_values: torch.Tensor
    sort_values_live: torch.Tensor
    primitive_queues: list[torch.Tensor]
    kind_queues: list[torch.Tensor]
    schedule_ordered_ids: torch.Tensor
    primitive_sorted_ids: torch.Tensor
    kind_sorted_ids: torch.Tensor
    packed_src: list[torch.Tensor]
    packed_tgt: list[torch.Tensor]
    packed_act_param: list[torch.Tensor]
    packed_kind: list[torch.Tensor]
    packed_local_ids: list[torch.Tensor]
    vec3_node_work_ids: torch.Tensor
    vec3_node_ids_by_item: torch.Tensor
    graph_vec3: torch.Tensor
    quat_node_work_ids: torch.Tensor
    quat_node_ids_by_item: torch.Tensor
    graph_quat: torch.Tensor
    reduce8_node_work_ids: torch.Tensor
    reduce8_node_ids_by_item: torch.Tensor
    graph_reduce8: torch.Tensor
    reduce32_node_work_ids: torch.Tensor
    reduce32_node_ids_by_item: torch.Tensor
    graph_reduce32: torch.Tensor
    contact_node_work_ids: torch.Tensor
    contact_node_ids_by_item: torch.Tensor
    contact_mask: torch.Tensor
    local_frame_node_work_ids: torch.Tensor
    local_frame_node_ids_by_item: torch.Tensor
    graph_frame: torch.Tensor


@dataclass
class PackedOutput:
    delta: list[torch.Tensor]
    error: list[torch.Tensor]
    activation: list[torch.Tensor]


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)


def make_kind_ids(n_work: int, num_kinds: int, pattern: str, device: torch.device) -> torch.Tensor:
    if pattern == "grouped":
        n_per_kind = (n_work + num_kinds - 1) // num_kinds
        base = torch.arange(num_kinds, device=device, dtype=torch.int32).repeat_interleave(n_per_kind)
        return base[:n_work].clone()
    base_pattern = pattern
    sort_mode = ""
    if pattern.endswith("_kind_sorted"):
        base_pattern = pattern.removesuffix("_kind_sorted")
        sort_mode = "kind"
    elif pattern.endswith("_primitive_sorted"):
        base_pattern = pattern.removesuffix("_primitive_sorted")
        sort_mode = "primitive"
    if base_pattern == "skew":
        ranks = torch.arange(1, num_kinds + 1, device=device, dtype=torch.float32)
        probs = 1.0 / ranks
        probs = probs / probs.sum()
        kind = torch.multinomial(probs, n_work, replacement=True).to(torch.int32)
    else:
        kind = torch.randint(0, num_kinds, (n_work,), device=device, dtype=torch.int32)
    if sort_mode == "kind":
        return kind[torch.argsort(kind)].contiguous()
    if sort_mode == "primitive":
        return kind[torch.argsort(kind // KINDS_PER_PRIMITIVE)].contiguous()
    return kind


def make_schedule_ordered_ids(kind: torch.Tensor, slots_per_env: int) -> torch.Tensor:
    """Return work ids sorted by primitive family within each env-local slot row."""
    n_work = int(kind.numel())
    n_envs = (n_work + slots_per_env - 1) // slots_per_env
    padded = n_envs * slots_per_env
    primitive = kind // KINDS_PER_PRIMITIVE
    work_ids = torch.arange(n_work, device=kind.device, dtype=torch.int32)
    if padded != n_work:
        pad = padded - n_work
        primitive = torch.cat(
            [primitive, torch.full((pad,), NUM_PRIMITIVES, device=kind.device, dtype=primitive.dtype)]
        )
        work_ids = torch.cat([work_ids, torch.zeros(pad, device=kind.device, dtype=work_ids.dtype)])
    primitive_rows = primitive.view(n_envs, slots_per_env)
    work_id_rows = work_ids.view(n_envs, slots_per_env)
    order = torch.argsort(primitive_rows, dim=1, stable=True)
    return torch.gather(work_id_rows, 1, order).flatten()[:n_work].contiguous()


def build_shared_node_inputs(
    n_work: int,
    item_ids: torch.Tensor,
    graph_fanout: int,
    copy_ranges: tuple[tuple[torch.Tensor, int, int], ...],
    node_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build shared-node ids and force exact shared-input semantics."""
    node_ids_by_item = torch.full((n_work,), -1, device=item_ids.device, dtype=torch.int32)
    if item_ids.numel() == 0:
        return (
            torch.empty(0, device=item_ids.device, dtype=torch.int32),
            node_ids_by_item,
            torch.empty((0, node_width), device=item_ids.device, dtype=torch.float32),
        )

    fanout = max(1, int(graph_fanout))
    item_pos = torch.arange(item_ids.numel(), device=item_ids.device)
    node_ids = (item_pos // fanout).to(torch.int32)
    node_starts = torch.arange(0, item_ids.numel(), fanout, device=item_ids.device)
    node_work_ids = item_ids[node_starts].contiguous()
    representative_ids = node_work_ids[node_ids.long()].long()
    item_ids_long = item_ids.long()
    for tensor, start, stop in copy_ranges:
        tensor[item_ids_long, start:stop] = tensor[representative_ids, start:stop]
    node_ids_by_item[item_ids_long] = node_ids

    scratch = torch.empty((node_work_ids.numel(), node_width), device=item_ids.device, dtype=torch.float32)
    return node_work_ids, node_ids_by_item, scratch


def build_contact_graph_inputs(
    src: torch.Tensor,
    tgt: torch.Tensor,
    primitive_queues: list[torch.Tensor],
    graph_fanout: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build shared contact-predicate nodes for primitive graph benchmarking."""
    any_ids = primitive_queues[5]
    diff_ids = primitive_queues[6]
    if any_ids.numel() != 0 and diff_ids.numel() != 0:
        n_pairs = min(int(any_ids.numel()), int(diff_ids.numel()))
        paired_ids = torch.stack([any_ids[:n_pairs], diff_ids[:n_pairs]], dim=1).flatten()
        contact_ids = torch.cat([paired_ids, any_ids[n_pairs:], diff_ids[n_pairs:]])
    else:
        contact_ids = torch.cat([any_ids, diff_ids])

    return build_shared_node_inputs(src.shape[0], contact_ids, graph_fanout, ((src, 16, 32), (tgt, 1, 2)), 16)


def make_workload(
    n_work: int,
    num_kinds: int,
    pattern: str,
    seed: int,
    device: torch.device,
    slots_per_env: int,
    graph_fanout: int,
) -> Workload:
    torch.manual_seed(seed)
    kind = make_kind_ids(n_work, num_kinds, pattern, device)
    src = torch.randn(n_work, SRC_WIDTH, device=device, dtype=torch.float32)
    tgt = torch.randn(n_work, TGT_WIDTH, device=device, dtype=torch.float32)
    src[:, :4] = _normalize_quat(src[:, :4])
    src[:, 8:12] = _normalize_quat(src[:, 8:12])
    tgt[:, :4] = _normalize_quat(tgt[:, :4])
    act_param = torch.empty(n_work, device=device, dtype=torch.float32).uniform_(0.5, 2.0)
    scale = torch.linspace(0.40, 1.45, NUM_SYNTH_KINDS, device=device, dtype=torch.float32)
    sort_keys = torch.empty(2 * n_work, device=device, dtype=torch.int64)
    sort_values = torch.empty(2 * n_work, device=device, dtype=torch.int32)
    kind_queues = [(kind == k).nonzero(as_tuple=False).flatten().to(torch.int32).contiguous() for k in range(num_kinds)]
    primitive_sorted_ids = torch.argsort(kind // KINDS_PER_PRIMITIVE).to(torch.int32).contiguous()
    kind_sorted_ids = torch.argsort(kind).to(torch.int32).contiguous()
    schedule_ordered_ids = make_schedule_ordered_ids(kind, slots_per_env)
    primitive_queues = [
        ((kind >= p * KINDS_PER_PRIMITIVE) & (kind < (p + 1) * KINDS_PER_PRIMITIVE))
        .nonzero(as_tuple=False)
        .flatten()
        .to(torch.int32)
        .contiguous()
        for p in range(NUM_PRIMITIVES)
    ]
    vec3_node_work_ids, vec3_node_ids_by_item, graph_vec3 = build_shared_node_inputs(
        n_work, primitive_queues[0], graph_fanout, ((src, 0, 3),), 3
    )
    quat_node_work_ids, quat_node_ids_by_item, graph_quat = build_shared_node_inputs(
        n_work, primitive_queues[2], graph_fanout, ((src, 0, 4),), 4
    )
    reduce8_node_work_ids, reduce8_node_ids_by_item, graph_reduce8 = build_shared_node_inputs(
        n_work, primitive_queues[3], graph_fanout, ((src, 0, 8),), 1
    )
    graph_reduce8 = graph_reduce8.flatten()
    reduce32_node_work_ids, reduce32_node_ids_by_item, graph_reduce32 = build_shared_node_inputs(
        n_work, primitive_queues[4], graph_fanout, ((src, 0, 64),), 1
    )
    graph_reduce32 = graph_reduce32.flatten()
    contact_node_work_ids, contact_node_ids_by_item, contact_mask = build_contact_graph_inputs(
        src, tgt, primitive_queues, graph_fanout
    )
    local_frame_node_work_ids, local_frame_node_ids_by_item, graph_frame = build_shared_node_inputs(
        n_work, primitive_queues[7], graph_fanout, ((src, 0, 3), (src, 8, 12)), 12
    )
    packed_src = [src[ids.long()].contiguous() for ids in primitive_queues]
    packed_tgt = [tgt[ids.long()].contiguous() for ids in primitive_queues]
    packed_act_param = [act_param[ids.long()].contiguous() for ids in primitive_queues]
    packed_kind = [kind[ids.long()].contiguous() for ids in primitive_queues]
    packed_local_ids = [torch.arange(ids.numel(), device=device, dtype=torch.int32) for ids in primitive_queues]
    return Workload(
        kind=kind,
        src=src,
        tgt=tgt,
        act_param=act_param,
        scale=scale,
        sort_keys=sort_keys,
        sort_values=sort_values,
        sort_values_live=sort_values[:n_work],
        primitive_queues=primitive_queues,
        kind_queues=kind_queues,
        schedule_ordered_ids=schedule_ordered_ids,
        primitive_sorted_ids=primitive_sorted_ids,
        kind_sorted_ids=kind_sorted_ids,
        packed_src=packed_src,
        packed_tgt=packed_tgt,
        packed_act_param=packed_act_param,
        packed_kind=packed_kind,
        packed_local_ids=packed_local_ids,
        vec3_node_work_ids=vec3_node_work_ids,
        vec3_node_ids_by_item=vec3_node_ids_by_item,
        graph_vec3=graph_vec3,
        quat_node_work_ids=quat_node_work_ids,
        quat_node_ids_by_item=quat_node_ids_by_item,
        graph_quat=graph_quat,
        reduce8_node_work_ids=reduce8_node_work_ids,
        reduce8_node_ids_by_item=reduce8_node_ids_by_item,
        graph_reduce8=graph_reduce8,
        reduce32_node_work_ids=reduce32_node_work_ids,
        reduce32_node_ids_by_item=reduce32_node_ids_by_item,
        graph_reduce32=graph_reduce32,
        contact_node_work_ids=contact_node_work_ids,
        contact_node_ids_by_item=contact_node_ids_by_item,
        contact_mask=contact_mask,
        local_frame_node_work_ids=local_frame_node_work_ids,
        local_frame_node_ids_by_item=local_frame_node_ids_by_item,
        graph_frame=graph_frame,
    )


def make_output(n_work: int, device: torch.device) -> VariantOutput:
    return VariantOutput(
        delta=torch.empty(n_work, 4, device=device, dtype=torch.float32),
        error=torch.empty(n_work, device=device, dtype=torch.float32),
        activation=torch.empty(n_work, device=device, dtype=torch.float32),
    )


def make_packed_output(work: Workload) -> PackedOutput:
    return PackedOutput(
        delta=[
            torch.empty(ids.numel(), 4, device=work.kind.device, dtype=torch.float32) for ids in work.primitive_queues
        ],
        error=[torch.empty(ids.numel(), device=work.kind.device, dtype=torch.float32) for ids in work.primitive_queues],
        activation=[
            torch.empty(ids.numel(), device=work.kind.device, dtype=torch.float32) for ids in work.primitive_queues
        ],
    )


def launch_mega(work: Workload, out: VariantOutput) -> None:
    wp.launch(
        mega_dispatch_kernel,
        dim=work.kind.numel(),
        inputs=[
            wp.from_torch(work.kind, dtype=wp.int32),
            wp.from_torch(work.scale, dtype=wp.float32),
            wp.from_torch(work.src, dtype=wp.float32),
            wp.from_torch(work.tgt, dtype=wp.float32),
            wp.from_torch(work.act_param, dtype=wp.float32),
            wp.from_torch(out.delta, dtype=wp.float32),
            wp.from_torch(out.error, dtype=wp.float32),
            wp.from_torch(out.activation, dtype=wp.float32),
        ],
        device=str(work.kind.device),
    )


def launch_indexed_mega(work: Workload, out: VariantOutput, ids: torch.Tensor) -> None:
    wp.launch(
        indexed_mega_dispatch_kernel,
        dim=ids.numel(),
        inputs=[
            wp.from_torch(ids, dtype=wp.int32),
            wp.from_torch(work.kind, dtype=wp.int32),
            wp.from_torch(work.scale, dtype=wp.float32),
            wp.from_torch(work.src, dtype=wp.float32),
            wp.from_torch(work.tgt, dtype=wp.float32),
            wp.from_torch(work.act_param, dtype=wp.float32),
            wp.from_torch(out.delta, dtype=wp.float32),
            wp.from_torch(out.error, dtype=wp.float32),
            wp.from_torch(out.activation, dtype=wp.float32),
            ids.numel(),
        ],
        device=str(work.kind.device),
    )


def launch_sort_indexed_mega(work: Workload, out: VariantOutput, sort_by: str) -> None:
    wp_kind = wp.from_torch(work.kind, dtype=wp.int32)
    wp_sort_keys = wp.from_torch(work.sort_keys, dtype=wp.int64)
    wp_sort_values = wp.from_torch(work.sort_values, dtype=wp.int32)
    if sort_by == "primitive":
        wp.launch(
            prepare_primitive_sort_kernel,
            dim=work.kind.numel(),
            inputs=[wp_kind, wp_sort_keys, wp_sort_values, KINDS_PER_PRIMITIVE],
            device=str(work.kind.device),
        )
    else:
        wp.launch(
            prepare_kind_sort_kernel,
            dim=work.kind.numel(),
            inputs=[wp_kind, wp_sort_keys, wp_sort_values],
            device=str(work.kind.device),
        )
    wpu.radix_sort_pairs(wp_sort_keys, wp_sort_values, work.kind.numel())
    launch_indexed_mega(work, out, work.sort_values_live)


def _launch_queue_kernel(kernel, ids: torch.Tensor, work: Workload, out: VariantOutput) -> None:
    if ids.numel() == 0:
        return
    wp.launch(
        kernel,
        dim=ids.numel(),
        inputs=[
            wp.from_torch(ids, dtype=wp.int32),
            wp.from_torch(work.kind, dtype=wp.int32),
            wp.from_torch(work.scale, dtype=wp.float32),
            wp.from_torch(work.src, dtype=wp.float32),
            wp.from_torch(work.tgt, dtype=wp.float32),
            wp.from_torch(work.act_param, dtype=wp.float32),
            wp.from_torch(out.delta, dtype=wp.float32),
            wp.from_torch(out.error, dtype=wp.float32),
            wp.from_torch(out.activation, dtype=wp.float32),
            ids.numel(),
        ],
        device=str(work.kind.device),
    )


def launch_kind_queue(work: Workload, out: VariantOutput) -> None:
    kernels = [
        queue_vec3_kernel,
        queue_scalar_kernel,
        queue_quat_kernel,
        queue_reduce8_kernel,
        queue_reduce32_kernel,
        queue_contact_any16_kernel,
        queue_contact_diff16_kernel,
        queue_local_frame_vec3_kernel,
    ]
    for k, ids in enumerate(work.kind_queues):
        _launch_queue_kernel(kernels[k // KINDS_PER_PRIMITIVE], ids, work, out)


def launch_primitive_queue(work: Workload, out: VariantOutput) -> None:
    _launch_queue_kernel(queue_vec3_kernel, work.primitive_queues[0], work, out)
    _launch_queue_kernel(queue_scalar_kernel, work.primitive_queues[1], work, out)
    _launch_queue_kernel(queue_quat_kernel, work.primitive_queues[2], work, out)
    _launch_queue_kernel(queue_reduce8_kernel, work.primitive_queues[3], work, out)
    _launch_queue_kernel(queue_reduce32_kernel, work.primitive_queues[4], work, out)
    _launch_queue_kernel(queue_contact_any16_kernel, work.primitive_queues[5], work, out)
    _launch_queue_kernel(queue_contact_diff16_kernel, work.primitive_queues[6], work, out)
    _launch_queue_kernel(queue_local_frame_vec3_kernel, work.primitive_queues[7], work, out)


def _launch_graph_contact_kernel(kernel, ids: torch.Tensor, work: Workload, out: VariantOutput) -> None:
    if ids.numel() == 0:
        return
    wp.launch(
        kernel,
        dim=ids.numel(),
        inputs=[
            wp.from_torch(ids, dtype=wp.int32),
            wp.from_torch(work.contact_node_ids_by_item, dtype=wp.int32),
            wp.from_torch(work.kind, dtype=wp.int32),
            wp.from_torch(work.scale, dtype=wp.float32),
            wp.from_torch(work.tgt, dtype=wp.float32),
            wp.from_torch(work.act_param, dtype=wp.float32),
            wp.from_torch(work.contact_mask, dtype=wp.float32),
            wp.from_torch(out.delta, dtype=wp.float32),
            wp.from_torch(out.error, dtype=wp.float32),
            wp.from_torch(out.activation, dtype=wp.float32),
            ids.numel(),
        ],
        device=str(work.kind.device),
    )


def _launch_graph_vec3_kernel(
    kernel,
    node_ids: torch.Tensor,
    node_ids_by_item: torch.Tensor,
    graph_vec3_out: torch.Tensor,
    ids: torch.Tensor,
    work: Workload,
    out: VariantOutput,
) -> None:
    if node_ids.numel() != 0:
        wp.launch(
            kernel,
            dim=node_ids.numel(),
            inputs=[
                wp.from_torch(node_ids, dtype=wp.int32),
                wp.from_torch(work.src, dtype=wp.float32),
                wp.from_torch(work.tgt, dtype=wp.float32),
                wp.from_torch(graph_vec3_out, dtype=wp.float32),
                node_ids.numel(),
            ],
            device=str(work.kind.device),
        )
    if ids.numel() == 0:
        return
    wp.launch(
        graph_current_vec3_kernel,
        dim=ids.numel(),
        inputs=[
            wp.from_torch(ids, dtype=wp.int32),
            wp.from_torch(node_ids_by_item, dtype=wp.int32),
            wp.from_torch(work.kind, dtype=wp.int32),
            wp.from_torch(work.scale, dtype=wp.float32),
            wp.from_torch(work.tgt, dtype=wp.float32),
            wp.from_torch(work.act_param, dtype=wp.float32),
            wp.from_torch(graph_vec3_out, dtype=wp.float32),
            wp.from_torch(out.delta, dtype=wp.float32),
            wp.from_torch(out.error, dtype=wp.float32),
            wp.from_torch(out.activation, dtype=wp.float32),
            ids.numel(),
        ],
        device=str(work.kind.device),
    )


def _launch_graph_quat_kernel(ids: torch.Tensor, work: Workload, out: VariantOutput) -> None:
    if work.quat_node_work_ids.numel() != 0:
        wp.launch(
            current_quat_graph_kernel,
            dim=work.quat_node_work_ids.numel(),
            inputs=[
                wp.from_torch(work.quat_node_work_ids, dtype=wp.int32),
                wp.from_torch(work.src, dtype=wp.float32),
                wp.from_torch(work.graph_quat, dtype=wp.float32),
                work.quat_node_work_ids.numel(),
            ],
            device=str(work.kind.device),
        )
    if ids.numel() == 0:
        return
    wp.launch(
        graph_quat_kernel,
        dim=ids.numel(),
        inputs=[
            wp.from_torch(ids, dtype=wp.int32),
            wp.from_torch(work.quat_node_ids_by_item, dtype=wp.int32),
            wp.from_torch(work.kind, dtype=wp.int32),
            wp.from_torch(work.scale, dtype=wp.float32),
            wp.from_torch(work.tgt, dtype=wp.float32),
            wp.from_torch(work.act_param, dtype=wp.float32),
            wp.from_torch(work.graph_quat, dtype=wp.float32),
            wp.from_torch(out.delta, dtype=wp.float32),
            wp.from_torch(out.error, dtype=wp.float32),
            wp.from_torch(out.activation, dtype=wp.float32),
            ids.numel(),
        ],
        device=str(work.kind.device),
    )


def _launch_graph_reduce_kernel(
    kernel,
    node_ids: torch.Tensor,
    node_ids_by_item: torch.Tensor,
    graph_scalar_out: torch.Tensor,
    ids: torch.Tensor,
    work: Workload,
    out: VariantOutput,
) -> None:
    if node_ids.numel() != 0:
        wp.launch(
            kernel,
            dim=node_ids.numel(),
            inputs=[
                wp.from_torch(node_ids, dtype=wp.int32),
                wp.from_torch(work.src, dtype=wp.float32),
                wp.from_torch(graph_scalar_out, dtype=wp.float32),
                node_ids.numel(),
            ],
            device=str(work.kind.device),
        )
    if ids.numel() == 0:
        return
    wp.launch(
        graph_scalar_reduce_kernel,
        dim=ids.numel(),
        inputs=[
            wp.from_torch(ids, dtype=wp.int32),
            wp.from_torch(node_ids_by_item, dtype=wp.int32),
            wp.from_torch(work.kind, dtype=wp.int32),
            wp.from_torch(work.scale, dtype=wp.float32),
            wp.from_torch(work.tgt, dtype=wp.float32),
            wp.from_torch(work.act_param, dtype=wp.float32),
            wp.from_torch(graph_scalar_out, dtype=wp.float32),
            wp.from_torch(out.delta, dtype=wp.float32),
            wp.from_torch(out.error, dtype=wp.float32),
            wp.from_torch(out.activation, dtype=wp.float32),
            ids.numel(),
        ],
        device=str(work.kind.device),
    )


def _launch_graph_frame_kernel(ids: torch.Tensor, work: Workload, out: VariantOutput) -> None:
    if work.local_frame_node_work_ids.numel() != 0:
        wp.launch(
            frame_basis_graph_kernel,
            dim=work.local_frame_node_work_ids.numel(),
            inputs=[
                wp.from_torch(work.local_frame_node_work_ids, dtype=wp.int32),
                wp.from_torch(work.src, dtype=wp.float32),
                wp.from_torch(work.graph_frame, dtype=wp.float32),
                work.local_frame_node_work_ids.numel(),
            ],
            device=str(work.kind.device),
        )
    if ids.numel() == 0:
        return
    wp.launch(
        graph_frame_vec3_kernel,
        dim=ids.numel(),
        inputs=[
            wp.from_torch(ids, dtype=wp.int32),
            wp.from_torch(work.local_frame_node_ids_by_item, dtype=wp.int32),
            wp.from_torch(work.kind, dtype=wp.int32),
            wp.from_torch(work.scale, dtype=wp.float32),
            wp.from_torch(work.tgt, dtype=wp.float32),
            wp.from_torch(work.act_param, dtype=wp.float32),
            wp.from_torch(work.graph_frame, dtype=wp.float32),
            wp.from_torch(out.delta, dtype=wp.float32),
            wp.from_torch(out.error, dtype=wp.float32),
            wp.from_torch(out.activation, dtype=wp.float32),
            ids.numel(),
        ],
        device=str(work.kind.device),
    )


def launch_primitive_graph(work: Workload, out: VariantOutput) -> None:
    _launch_graph_vec3_kernel(
        current_vec3_graph_kernel,
        work.vec3_node_work_ids,
        work.vec3_node_ids_by_item,
        work.graph_vec3,
        work.primitive_queues[0],
        work,
        out,
    )
    _launch_queue_kernel(queue_scalar_kernel, work.primitive_queues[1], work, out)
    _launch_graph_quat_kernel(work.primitive_queues[2], work, out)
    _launch_graph_reduce_kernel(
        reduce8_graph_kernel,
        work.reduce8_node_work_ids,
        work.reduce8_node_ids_by_item,
        work.graph_reduce8,
        work.primitive_queues[3],
        work,
        out,
    )
    _launch_graph_reduce_kernel(
        reduce32_graph_kernel,
        work.reduce32_node_work_ids,
        work.reduce32_node_ids_by_item,
        work.graph_reduce32,
        work.primitive_queues[4],
        work,
        out,
    )
    if work.contact_node_work_ids.numel() != 0:
        wp.launch(
            contact_predicate16_graph_kernel,
            dim=work.contact_node_work_ids.numel(),
            inputs=[
                wp.from_torch(work.contact_node_work_ids, dtype=wp.int32),
                wp.from_torch(work.src, dtype=wp.float32),
                wp.from_torch(work.tgt, dtype=wp.float32),
                wp.from_torch(work.contact_mask, dtype=wp.float32),
                work.contact_node_work_ids.numel(),
            ],
            device=str(work.kind.device),
        )
    _launch_graph_contact_kernel(graph_contact_any16_kernel, work.primitive_queues[5], work, out)
    _launch_graph_contact_kernel(graph_contact_diff16_kernel, work.primitive_queues[6], work, out)
    _launch_graph_frame_kernel(work.primitive_queues[7], work, out)


def _launch_graph_vec3_local_kernel(
    kernel,
    node_ids: torch.Tensor,
    node_ids_by_item: torch.Tensor,
    graph_vec3_out: torch.Tensor,
    ids: torch.Tensor,
    work: Workload,
    out: PackedOutput,
    p: int,
) -> None:
    if node_ids.numel() != 0:
        wp.launch(
            kernel,
            dim=node_ids.numel(),
            inputs=[
                wp.from_torch(node_ids, dtype=wp.int32),
                wp.from_torch(work.src, dtype=wp.float32),
                wp.from_torch(work.tgt, dtype=wp.float32),
                wp.from_torch(graph_vec3_out, dtype=wp.float32),
                node_ids.numel(),
            ],
            device=str(work.kind.device),
        )
    if ids.numel() == 0:
        return
    wp.launch(
        graph_vec3_local_kernel,
        dim=ids.numel(),
        inputs=[
            wp.from_torch(ids, dtype=wp.int32),
            wp.from_torch(node_ids_by_item, dtype=wp.int32),
            wp.from_torch(work.kind, dtype=wp.int32),
            wp.from_torch(work.scale, dtype=wp.float32),
            wp.from_torch(work.tgt, dtype=wp.float32),
            wp.from_torch(work.act_param, dtype=wp.float32),
            wp.from_torch(graph_vec3_out, dtype=wp.float32),
            wp.from_torch(out.delta[p], dtype=wp.float32),
            wp.from_torch(out.error[p], dtype=wp.float32),
            wp.from_torch(out.activation[p], dtype=wp.float32),
            ids.numel(),
        ],
        device=str(work.kind.device),
    )


def _launch_graph_quat_local_kernel(ids: torch.Tensor, work: Workload, out: PackedOutput, p: int) -> None:
    if work.quat_node_work_ids.numel() != 0:
        wp.launch(
            current_quat_graph_kernel,
            dim=work.quat_node_work_ids.numel(),
            inputs=[
                wp.from_torch(work.quat_node_work_ids, dtype=wp.int32),
                wp.from_torch(work.src, dtype=wp.float32),
                wp.from_torch(work.graph_quat, dtype=wp.float32),
                work.quat_node_work_ids.numel(),
            ],
            device=str(work.kind.device),
        )
    if ids.numel() == 0:
        return
    wp.launch(
        graph_quat_local_kernel,
        dim=ids.numel(),
        inputs=[
            wp.from_torch(ids, dtype=wp.int32),
            wp.from_torch(work.quat_node_ids_by_item, dtype=wp.int32),
            wp.from_torch(work.kind, dtype=wp.int32),
            wp.from_torch(work.scale, dtype=wp.float32),
            wp.from_torch(work.tgt, dtype=wp.float32),
            wp.from_torch(work.act_param, dtype=wp.float32),
            wp.from_torch(work.graph_quat, dtype=wp.float32),
            wp.from_torch(out.delta[p], dtype=wp.float32),
            wp.from_torch(out.error[p], dtype=wp.float32),
            wp.from_torch(out.activation[p], dtype=wp.float32),
            ids.numel(),
        ],
        device=str(work.kind.device),
    )


def _launch_graph_reduce_local_kernel(
    kernel,
    node_ids: torch.Tensor,
    node_ids_by_item: torch.Tensor,
    graph_scalar_out: torch.Tensor,
    ids: torch.Tensor,
    work: Workload,
    out: PackedOutput,
    p: int,
) -> None:
    if node_ids.numel() != 0:
        wp.launch(
            kernel,
            dim=node_ids.numel(),
            inputs=[
                wp.from_torch(node_ids, dtype=wp.int32),
                wp.from_torch(work.src, dtype=wp.float32),
                wp.from_torch(graph_scalar_out, dtype=wp.float32),
                node_ids.numel(),
            ],
            device=str(work.kind.device),
        )
    if ids.numel() == 0:
        return
    wp.launch(
        graph_scalar_reduce_local_kernel,
        dim=ids.numel(),
        inputs=[
            wp.from_torch(ids, dtype=wp.int32),
            wp.from_torch(node_ids_by_item, dtype=wp.int32),
            wp.from_torch(work.kind, dtype=wp.int32),
            wp.from_torch(work.scale, dtype=wp.float32),
            wp.from_torch(work.tgt, dtype=wp.float32),
            wp.from_torch(work.act_param, dtype=wp.float32),
            wp.from_torch(graph_scalar_out, dtype=wp.float32),
            wp.from_torch(out.delta[p], dtype=wp.float32),
            wp.from_torch(out.error[p], dtype=wp.float32),
            wp.from_torch(out.activation[p], dtype=wp.float32),
            ids.numel(),
        ],
        device=str(work.kind.device),
    )


def _launch_graph_contact_local_kernel(kernel, ids: torch.Tensor, work: Workload, out: PackedOutput, p: int) -> None:
    if ids.numel() == 0:
        return
    wp.launch(
        kernel,
        dim=ids.numel(),
        inputs=[
            wp.from_torch(ids, dtype=wp.int32),
            wp.from_torch(work.contact_node_ids_by_item, dtype=wp.int32),
            wp.from_torch(work.kind, dtype=wp.int32),
            wp.from_torch(work.scale, dtype=wp.float32),
            wp.from_torch(work.tgt, dtype=wp.float32),
            wp.from_torch(work.act_param, dtype=wp.float32),
            wp.from_torch(work.contact_mask, dtype=wp.float32),
            wp.from_torch(out.delta[p], dtype=wp.float32),
            wp.from_torch(out.error[p], dtype=wp.float32),
            wp.from_torch(out.activation[p], dtype=wp.float32),
            ids.numel(),
        ],
        device=str(work.kind.device),
    )


def _launch_graph_frame_local_kernel(ids: torch.Tensor, work: Workload, out: PackedOutput, p: int) -> None:
    if work.local_frame_node_work_ids.numel() != 0:
        wp.launch(
            frame_basis_graph_kernel,
            dim=work.local_frame_node_work_ids.numel(),
            inputs=[
                wp.from_torch(work.local_frame_node_work_ids, dtype=wp.int32),
                wp.from_torch(work.src, dtype=wp.float32),
                wp.from_torch(work.graph_frame, dtype=wp.float32),
                work.local_frame_node_work_ids.numel(),
            ],
            device=str(work.kind.device),
        )
    if ids.numel() == 0:
        return
    wp.launch(
        graph_frame_vec3_local_kernel,
        dim=ids.numel(),
        inputs=[
            wp.from_torch(ids, dtype=wp.int32),
            wp.from_torch(work.local_frame_node_ids_by_item, dtype=wp.int32),
            wp.from_torch(work.kind, dtype=wp.int32),
            wp.from_torch(work.scale, dtype=wp.float32),
            wp.from_torch(work.tgt, dtype=wp.float32),
            wp.from_torch(work.act_param, dtype=wp.float32),
            wp.from_torch(work.graph_frame, dtype=wp.float32),
            wp.from_torch(out.delta[p], dtype=wp.float32),
            wp.from_torch(out.error[p], dtype=wp.float32),
            wp.from_torch(out.activation[p], dtype=wp.float32),
            ids.numel(),
        ],
        device=str(work.kind.device),
    )


def launch_primitive_graph_packed_local(work: Workload, out: PackedOutput) -> None:
    _launch_graph_vec3_local_kernel(
        current_vec3_graph_kernel,
        work.vec3_node_work_ids,
        work.vec3_node_ids_by_item,
        work.graph_vec3,
        work.primitive_queues[0],
        work,
        out,
        0,
    )
    _launch_packed_local_kernel(packed_scalar_scatter_kernel, 1, work, out)
    _launch_graph_quat_local_kernel(work.primitive_queues[2], work, out, 2)
    _launch_graph_reduce_local_kernel(
        reduce8_graph_kernel,
        work.reduce8_node_work_ids,
        work.reduce8_node_ids_by_item,
        work.graph_reduce8,
        work.primitive_queues[3],
        work,
        out,
        3,
    )
    _launch_graph_reduce_local_kernel(
        reduce32_graph_kernel,
        work.reduce32_node_work_ids,
        work.reduce32_node_ids_by_item,
        work.graph_reduce32,
        work.primitive_queues[4],
        work,
        out,
        4,
    )
    if work.contact_node_work_ids.numel() != 0:
        wp.launch(
            contact_predicate16_graph_kernel,
            dim=work.contact_node_work_ids.numel(),
            inputs=[
                wp.from_torch(work.contact_node_work_ids, dtype=wp.int32),
                wp.from_torch(work.src, dtype=wp.float32),
                wp.from_torch(work.tgt, dtype=wp.float32),
                wp.from_torch(work.contact_mask, dtype=wp.float32),
                work.contact_node_work_ids.numel(),
            ],
            device=str(work.kind.device),
        )
    _launch_graph_contact_local_kernel(graph_contact_any16_local_kernel, work.primitive_queues[5], work, out, 5)
    _launch_graph_contact_local_kernel(graph_contact_diff16_local_kernel, work.primitive_queues[6], work, out, 6)
    _launch_graph_frame_local_kernel(work.primitive_queues[7], work, out, 7)


def _launch_packed_scatter_kernel(kernel, p: int, work: Workload, out: VariantOutput) -> None:
    ids = work.primitive_queues[p]
    if ids.numel() == 0:
        return
    wp.launch(
        kernel,
        dim=ids.numel(),
        inputs=[
            wp.from_torch(ids, dtype=wp.int32),
            wp.from_torch(work.packed_kind[p], dtype=wp.int32),
            wp.from_torch(work.scale, dtype=wp.float32),
            wp.from_torch(work.packed_src[p], dtype=wp.float32),
            wp.from_torch(work.packed_tgt[p], dtype=wp.float32),
            wp.from_torch(work.packed_act_param[p], dtype=wp.float32),
            wp.from_torch(out.delta, dtype=wp.float32),
            wp.from_torch(out.error, dtype=wp.float32),
            wp.from_torch(out.activation, dtype=wp.float32),
            ids.numel(),
        ],
        device=str(work.kind.device),
    )


def launch_packed_primitive_scatter(work: Workload, out: VariantOutput) -> None:
    _launch_packed_scatter_kernel(packed_vec3_scatter_kernel, 0, work, out)
    _launch_packed_scatter_kernel(packed_scalar_scatter_kernel, 1, work, out)
    _launch_packed_scatter_kernel(packed_quat_scatter_kernel, 2, work, out)
    _launch_packed_scatter_kernel(packed_reduce8_scatter_kernel, 3, work, out)
    _launch_packed_scatter_kernel(packed_reduce32_scatter_kernel, 4, work, out)
    _launch_packed_scatter_kernel(packed_contact_any16_scatter_kernel, 5, work, out)
    _launch_packed_scatter_kernel(packed_contact_diff16_scatter_kernel, 6, work, out)
    _launch_packed_scatter_kernel(packed_local_frame_vec3_scatter_kernel, 7, work, out)


def _launch_packed_local_kernel(kernel, p: int, work: Workload, out: PackedOutput) -> None:
    ids = work.packed_local_ids[p]
    if ids.numel() == 0:
        return
    wp.launch(
        kernel,
        dim=ids.numel(),
        inputs=[
            wp.from_torch(ids, dtype=wp.int32),
            wp.from_torch(work.packed_kind[p], dtype=wp.int32),
            wp.from_torch(work.scale, dtype=wp.float32),
            wp.from_torch(work.packed_src[p], dtype=wp.float32),
            wp.from_torch(work.packed_tgt[p], dtype=wp.float32),
            wp.from_torch(work.packed_act_param[p], dtype=wp.float32),
            wp.from_torch(out.delta[p], dtype=wp.float32),
            wp.from_torch(out.error[p], dtype=wp.float32),
            wp.from_torch(out.activation[p], dtype=wp.float32),
            ids.numel(),
        ],
        device=str(work.kind.device),
    )


def launch_packed_primitive_local(work: Workload, out: PackedOutput) -> None:
    _launch_packed_local_kernel(packed_vec3_scatter_kernel, 0, work, out)
    _launch_packed_local_kernel(packed_scalar_scatter_kernel, 1, work, out)
    _launch_packed_local_kernel(packed_quat_scatter_kernel, 2, work, out)
    _launch_packed_local_kernel(packed_reduce8_scatter_kernel, 3, work, out)
    _launch_packed_local_kernel(packed_reduce32_scatter_kernel, 4, work, out)
    _launch_packed_local_kernel(packed_contact_any16_scatter_kernel, 5, work, out)
    _launch_packed_local_kernel(packed_contact_diff16_scatter_kernel, 6, work, out)
    _launch_packed_local_kernel(packed_local_frame_vec3_scatter_kernel, 7, work, out)


def time_variant(name: str, fn, warmup: int, runs: int, graph: bool) -> float:
    for _ in range(warmup):
        fn()
    wp.synchronize()
    if graph:
        with wp.ScopedCapture(device="cuda:0") as capture:
            fn()
        graph_obj = capture.graph
        for _ in range(warmup):
            wp.capture_launch(graph_obj)
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            wp.capture_launch(graph_obj)
        wp.synchronize()
        dt = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        for _ in range(runs):
            fn()
        wp.synchronize()
        dt = time.perf_counter() - t0
    ms = dt * 1000.0 / runs
    print(f"{name:>29}: {ms:8.4f} ms")
    return ms


def verify(work: Workload) -> None:
    ref = make_output(work.kind.numel(), work.kind.device)
    q = make_output(work.kind.numel(), work.kind.device)
    g = make_output(work.kind.numel(), work.kind.device)
    gl = make_packed_output(work)
    launch_mega(work, ref)
    launch_packed_primitive_scatter(work, q)
    launch_primitive_graph(work, g)
    launch_primitive_graph_packed_local(work, gl)
    wp.synchronize()
    torch.testing.assert_close(q.error, ref.error, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(q.activation, ref.activation, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(g.error, ref.error, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(g.activation, ref.activation, atol=1.0e-6, rtol=1.0e-6)
    for p, ids in enumerate(work.primitive_queues):
        torch.testing.assert_close(gl.error[p], ref.error[ids.long()], atol=1.0e-6, rtol=1.0e-6)
        torch.testing.assert_close(gl.activation[p], ref.activation[ids.long()], atol=1.0e-6, rtol=1.0e-6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-work", type=int, default=1_048_576)
    parser.add_argument("--num-kinds", type=int, default=64, choices=[8, 16, 32, 64])
    parser.add_argument(
        "--pattern",
        choices=[
            "random",
            "grouped",
            "skew",
            "random_kind_sorted",
            "random_primitive_sorted",
            "skew_kind_sorted",
            "skew_primitive_sorted",
        ],
        default="random",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--slots-per-env", type=int, default=8)
    parser.add_argument("--graph-fanout", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    wp.init()
    device = torch.device("cuda:0")
    work = make_workload(
        args.n_work, args.num_kinds, args.pattern, args.seed, device, args.slots_per_env, args.graph_fanout
    )
    if not args.no_verify:
        verify(work)
    mega_out = make_output(args.n_work, device)
    indexed_primitive_out = make_output(args.n_work, device)
    indexed_kind_out = make_output(args.n_work, device)
    schedule_ordered_out = make_output(args.n_work, device)
    sort_indexed_primitive_out = make_output(args.n_work, device)
    sort_indexed_kind_out = make_output(args.n_work, device)
    kind_out = make_output(args.n_work, device)
    primitive_out = make_output(args.n_work, device)
    primitive_graph_out = make_output(args.n_work, device)
    graph_packed_local_out = make_packed_output(work)
    packed_out = make_output(args.n_work, device)
    packed_local_out = make_packed_output(work)
    counts = torch.bincount(work.kind.to(torch.long), minlength=args.num_kinds).detach().cpu().tolist()
    primitive_counts = [int(q.numel()) for q in work.primitive_queues]
    print(
        f"# dispatch homogeneity benchmark: n_work={args.n_work}, num_kinds={args.num_kinds}, "
        f"pattern={args.pattern}, slots_per_env={args.slots_per_env}, "
        f"graph_fanout={args.graph_fanout}, graph={args.graph}"
    )
    print(f"# cuda={torch.cuda.get_device_name(0)}, warp={wp.__version__}, torch={torch.__version__}")
    print(f"# kind_counts_head={counts[: min(len(counts), 12)]}")
    print(f"# primitive_counts={primitive_counts}")
    print(
        "# synthetic proxy labels: primitive_queue_local_synth ~= production "
        "primitive_queue_local dispatch; primitive_graph_local_synth ~= production "
        "primitive_graph_local dispatch; graph_packed_local_synth is the future "
        "packed-output graph target."
    )
    print("# public backend comparison: run benchmark/bench_multi_task_command_backends.py --preset shared_direct")
    print(
        "# graph_nodes="
        f"vec3:{int(work.vec3_node_work_ids.numel())}, "
        f"reduce8:{int(work.reduce8_node_work_ids.numel())}, "
        f"reduce32:{int(work.reduce32_node_work_ids.numel())}, "
        f"contact:{int(work.contact_node_work_ids.numel())}, "
        f"local_frame:{int(work.local_frame_node_work_ids.numel())}"
    )
    t_mega = time_variant("mega", lambda: launch_mega(work, mega_out), args.warmup, args.runs, args.graph)
    t_indexed_primitive = time_variant(
        "idx_mega_primitive",
        lambda: launch_indexed_mega(work, indexed_primitive_out, work.primitive_sorted_ids),
        args.warmup,
        args.runs,
        args.graph,
    )
    t_indexed_kind = time_variant(
        "idx_mega_kind",
        lambda: launch_indexed_mega(work, indexed_kind_out, work.kind_sorted_ids),
        args.warmup,
        args.runs,
        args.graph,
    )
    t_schedule_ordered = time_variant(
        "schedule_ordered_mega",
        lambda: launch_indexed_mega(work, schedule_ordered_out, work.schedule_ordered_ids),
        args.warmup,
        args.runs,
        args.graph,
    )
    t_sort_indexed_primitive = time_variant(
        "sort_idx_primitive",
        lambda: launch_sort_indexed_mega(work, sort_indexed_primitive_out, "primitive"),
        args.warmup,
        args.runs,
        args.graph,
    )
    t_sort_indexed_kind = time_variant(
        "sort_idx_kind",
        lambda: launch_sort_indexed_mega(work, sort_indexed_kind_out, "kind"),
        args.warmup,
        args.runs,
        args.graph,
    )
    t_kind = time_variant("kind_queue", lambda: launch_kind_queue(work, kind_out), args.warmup, args.runs, args.graph)
    t_primitive = time_variant(
        "primitive_queue_local_synth",
        lambda: launch_primitive_queue(work, primitive_out),
        args.warmup,
        args.runs,
        args.graph,
    )
    t_primitive_graph = time_variant(
        "primitive_graph_local_synth",
        lambda: launch_primitive_graph(work, primitive_graph_out),
        args.warmup,
        args.runs,
        args.graph,
    )
    t_graph_packed_local = time_variant(
        "graph_packed_local_synth",
        lambda: launch_primitive_graph_packed_local(work, graph_packed_local_out),
        args.warmup,
        args.runs,
        args.graph,
    )
    t_packed = time_variant(
        "packed_scatter", lambda: launch_packed_primitive_scatter(work, packed_out), args.warmup, args.runs, args.graph
    )
    t_local = time_variant(
        "packed_local",
        lambda: launch_packed_primitive_local(work, packed_local_out),
        args.warmup,
        args.runs,
        args.graph,
    )
    print(
        f"# speedup primitive_queue_local_synth/mega={t_mega / t_primitive:.3f}x, "
        f"primitive_graph_local_synth/mega={t_mega / t_primitive_graph:.3f}x, "
        f"graph_packed_local_synth/mega={t_mega / t_graph_packed_local:.3f}x, "
        f"packed/mega={t_mega / t_packed:.3f}x, local/mega={t_mega / t_local:.3f}x, "
        f"idx_primitive/mega={t_mega / t_indexed_primitive:.3f}x, "
        f"idx_kind/mega={t_mega / t_indexed_kind:.3f}x, "
        f"schedule_ordered/mega={t_mega / t_schedule_ordered:.3f}x, "
        f"sort_idx_primitive/mega={t_mega / t_sort_indexed_primitive:.3f}x, "
        f"sort_idx_kind/mega={t_mega / t_sort_indexed_kind:.3f}x, kind/mega={t_mega / t_kind:.3f}x"
    )


if __name__ == "__main__":
    main()
