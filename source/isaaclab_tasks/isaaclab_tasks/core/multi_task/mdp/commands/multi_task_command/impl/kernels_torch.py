# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PyTorch reference implementations for every kernel id.

Kernel ids and metadata schemas (:class:`BufferKindDef`, :class:`StateKernelDef`)
live in :mod:`..kernel_ids` — the shared source of truth. This module provides
the PyTorch function bodies and binds them to the ids via the
:data:`STATE_KERNELS`, :data:`BUFFER_KINDS`, :data:`ACTIVATION_KERNELS`,
:data:`METRIC_KERNELS`, :data:`DELTA_KERNELS`, :data:`SAMPLER_KERNELS`
registry tuples (each indexed by the corresponding enum). The Warp mirror
(:mod:`.kernels_wp`) imports the same enums and reproduces the math as
``@wp.func``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import axis_angle_from_quat, quat_from_euler_xyz, quat_inv, quat_mul

from ..kernel_ids import (
    BUFFER_KIND,
    BufferKindDef,
    StateKernelDef,
)

__all__ = [
    "ACTIVATION_KERNELS",
    "BUFFER_KINDS",
    "BUFFER_KIND_READERS",
    "DELTA_KERNELS",
    "METRIC_KERNELS",
    "SAMPLER_KERNELS",
    "STATE_KERNEL_BUFFER_KIND",
    "STATE_KERNEL_COMPUTES",
    "STATE_KERNELS",
    "buffer_kind_is_body_indexed",
    "buffer_kind_per_element_stride",
    "state_kernel_intra_body_offset",
    "state_kernel_intra_body_stride",
]

# ``ManagerBasedRLEnv`` appears only in type annotations on the reader
# signatures below. Kept TYPE_CHECKING-only so this module is import-clean
# before SimulationApp launches — see ``test_env_cfg_no_forbidden_imports``
# for the contract. ``from __future__ import annotations`` above makes the
# string-quoted annotations resolve lazily at runtime.
if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


# Kernel ids and the BUFFER_KIND enum live in ``kernel_ids.py`` (single source of
# truth for both PyTorch and Warp paths). This module only registers the
# PyTorch implementation against those ids.
#
# Canonical layout assembly lives in ``spec._compute_canonical_layout`` — it's
# cfg-driven on both the entity axis (which entities appear) and the channel
# axis (which state-kernel slices per entity). This module defines only the
# kernel function bodies themselves.


# --- activation kernels (error -> score/predicate) ---
def activation_tanh(error: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.tanh(error / std)


def activation_less(error: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    return error < threshold


def activation_greater(error: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    return error > threshold


def activation_gaussian(error: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """``exp(-(error / σ)²)``. Smooth peak at error=0 (zero slope), steepest near error≈σ/√2."""
    z = error / sigma
    return torch.exp(-(z * z))


ACTIVATION_KERNELS = (activation_tanh, activation_less, activation_greater, activation_gaussian)


# --- metric kernels (x_cur, x_target -> scalar error) ---
def metric_geometric(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x, dim=-1)


def metric_quaternion(quat: torch.Tensor) -> torch.Tensor:
    assert quat.shape[-1] == 4
    angle_axis = axis_angle_from_quat(quat)
    return torch.linalg.vector_norm(angle_axis, dim=-1)


METRIC_KERNELS = (metric_geometric, metric_quaternion)


# --- delta kernels (order: that - this)---
def delta_geometric(x_cur: torch.Tensor, x_tgt: torch.Tensor) -> torch.Tensor:
    return x_tgt - x_cur


def delta_quaternion(quat_cur: torch.Tensor, quat_tgt: torch.Tensor) -> torch.Tensor:
    assert quat_cur.shape[-1] == 4 and quat_tgt.shape[-1] == 4
    return quat_mul(quat_inv(quat_cur), quat_tgt)


DELTA_KERNELS = (delta_geometric, delta_quaternion)


# ---------------------------------------------------------------------------
# State kernels — two-dispatch pipeline.
#
# 1. READ DISPATCH — for each unique ``(buffer_kind, asset_name)`` consumed by
#    the cfg, call the registered reader to produce a post-processed ``[N, *]``
#    tensor and write it into one unified per-step buffer. After this step the
#    state data the compute phase needs lives in a single contiguous tensor;
#    no asset dispatch happens downstream.
#
# 2. EXECUTE DISPATCH — per read group, advanced-index-gather the unified
#    buffer at precomputed absolute indices to produce ``[M, N, slice_size]``,
#    then run one batched ``compute_fn`` over it.
#
# Readers produce fully post-processed data (e.g. env-local body positions),
# so the execute side is pure slicing + kernel math — it has no notion of
# which asset its data came from.
# ---------------------------------------------------------------------------


def _read_joint_pos(env: ManagerBasedRLEnv, asset_name: str) -> torch.Tensor:
    articulation: Articulation = env.scene[asset_name]
    return articulation.data.joint_pos.torch


def _read_joint_vel(env: ManagerBasedRLEnv, asset_name: str) -> torch.Tensor:
    articulation: Articulation = env.scene[asset_name]
    return articulation.data.joint_vel.torch


def _read_body_pos_w(env: ManagerBasedRLEnv, asset_name: str) -> torch.Tensor:
    """World-frame body positions, ``[N, num_bodies, 3]``.

    Returns the raw scene tensor as a zero-copy torch view. The env-origin
    subtraction that makes body pos env-local is applied by the dispatch
    (Warp kernel for Warp path, inline PyTorch for the Torch reference
    path) so this reader stays allocation-free — essential for CUDA Graph
    capture and for keeping the "reader = pure read" contract clean.
    """
    articulation: Articulation = env.scene[asset_name]
    return articulation.data.body_pos_w.torch


def _read_body_quat_w(env: ManagerBasedRLEnv, asset_name: str) -> torch.Tensor:
    articulation: Articulation = env.scene[asset_name]
    return articulation.data.body_quat_w.torch


def _read_body_lin_vel_w(env: ManagerBasedRLEnv, asset_name: str) -> torch.Tensor:
    articulation: Articulation = env.scene[asset_name]
    return articulation.data.body_lin_vel_w.torch


def _read_body_ang_vel_w(env: ManagerBasedRLEnv, asset_name: str) -> torch.Tensor:
    articulation: Articulation = env.scene[asset_name]
    return articulation.data.body_ang_vel_w.torch


def _read_contact_net_forces_w(env: ManagerBasedRLEnv, asset_name: str) -> torch.Tensor:
    sensor = env.scene.sensors[asset_name]
    return sensor.data.net_forces_w.torch


def _read_joint_mech_power_abs(env: ManagerBasedRLEnv, asset_name: str) -> torch.Tensor:
    """Per-joint absolute mechanical power, ``[N, num_joints]``.

    Computes ``|applied_effort · joint_vel|`` element-wise. NaN-safe:
    non-finite entries (which can show up briefly during reset on some
    backends) are clamped to 0. Unlike the other readers this allocates a
    fresh tensor each step — the elementwise product can't be represented
    as a stride view.
    """
    articulation: Articulation = env.scene[asset_name]
    tau = articulation.actuators.applied_effort.torch
    qd = articulation.data.joint_vel.torch
    power = (tau * qd).abs()
    return torch.where(torch.isfinite(power), power, torch.zeros_like(power))


# Single source of truth for buffer-kind metadata. Indexed by :class:`BUFFER_KIND`.
BUFFER_KINDS: tuple[BufferKindDef, ...] = (
    BufferKindDef(_read_joint_pos, per_element_stride=1, is_body_indexed=False),
    BufferKindDef(_read_joint_vel, per_element_stride=1, is_body_indexed=False),
    BufferKindDef(_read_body_pos_w, per_element_stride=3, is_body_indexed=True),
    BufferKindDef(_read_body_quat_w, per_element_stride=4, is_body_indexed=True),
    BufferKindDef(_read_body_lin_vel_w, per_element_stride=3, is_body_indexed=True),
    BufferKindDef(_read_body_ang_vel_w, per_element_stride=3, is_body_indexed=True),
    BufferKindDef(_read_contact_net_forces_w, per_element_stride=3, is_body_indexed=True),
    BufferKindDef(_read_joint_mech_power_abs, per_element_stride=1, is_body_indexed=False),
)


# Flat reader tuple — exposed as a separate module attribute so tests can
# ``patch.object(mtc_mod, "BUFFER_KIND_READERS", ...)`` to inject mock readers.
BUFFER_KIND_READERS: tuple[Callable[[ManagerBasedRLEnv, str], torch.Tensor], ...] = tuple(
    d.reader for d in BUFFER_KINDS
)


def buffer_kind_per_element_stride(buffer_kind: int) -> int:
    """Floats per body / joint in a slab of this buffer kind."""
    return BUFFER_KINDS[int(buffer_kind)].per_element_stride


def buffer_kind_is_body_indexed(buffer_kind: int) -> bool:
    """True if the buffer's second axis is body-indexed (uses ``asset_cfg.body_ids``)."""
    return BUFFER_KINDS[int(buffer_kind)].is_body_indexed


def _state_identity(stacked: torch.Tensor) -> torch.Tensor:
    """Pass through. ``[M, N, stride]`` → ``[M, N, stride]``.

    Unified-buffer reads have already extracted the exact floats the kernel
    needs at the spec-precomputed absolute indices, so no reshape is required
    for kernels whose state_stride equals the gathered slice size. Used by
    body / joint kernels that read the entire per-body block, including
    BODY_POS_Z (which the gather has already projected to a single z float
    via intra-body offset 2, stride 1).
    """
    return stacked


def _contact_predicates_from_flat(stacked_flat: torch.Tensor) -> torch.Tensor:
    """``[M, N, K*3]`` flat contact forces → ``[M, N, K]`` binary predicates.

    The unified-buffer gather emits K bodies × 3 floats per subtask as a flat
    slice; reshape to ``[M, N, K, 3]`` internally to compute per-body norms.
    """
    m, n, total = stacked_flat.shape
    if total % 3 != 0:
        raise ValueError(f"contact compute expected K*3 trailing floats; got {total}")
    k = total // 3
    forces = stacked_flat.view(m, n, k, 3)
    mag = torch.linalg.norm(forces, dim=-1)
    return (mag > 1.0).to(mag.dtype)


def state_body_contact(stacked: torch.Tensor) -> torch.Tensor:
    """``[M, N, K*3]`` → ``[M, N, K]`` — per-body contact predicate."""
    return _contact_predicates_from_flat(stacked)


def state_body_contact_count(stacked: torch.Tensor) -> torch.Tensor:
    """``[M, N, K*3]`` → ``[M, N, 1]`` — count of bodies in contact."""
    return _contact_predicates_from_flat(stacked).sum(dim=-1, keepdim=True)


def state_body_contact_count_diff(stacked: torch.Tensor) -> torch.Tensor:
    """``[M, N, K*3]`` → ``[M, N, 1]`` — ``count(first K/2) − count(last K/2)``."""
    pred = _contact_predicates_from_flat(stacked)
    k = int(pred.shape[-1])
    if k % 2 != 0:
        raise ValueError(f"state_body_contact_count_diff requires even body count; got K={k}")
    half = k // 2
    return pred[..., :half].sum(dim=-1, keepdim=True) - pred[..., half:].sum(dim=-1, keepdim=True)


def state_joint_mech_power(stacked: torch.Tensor) -> torch.Tensor:
    """``[M, N, J]`` → ``[M, N, 1]`` — instantaneous total mechanical power [W].

    Reduces ``|τ_j · q̇_j|`` across the J joints into a single per-env scalar.
    The reduction is over **joints**, not time — so this is power (W), not
    work (J), and the value is independent of episode length. Pair with
    GAUSSIAN activation (``param`` = power scale σ in W) for a soft-safety
    factor in ``[0, 1]`` regardless of robot scale.
    """
    return stacked.sum(dim=-1, keepdim=True)


# Single source of truth for state-kernel metadata. Indexed by :class:`STATE_KERNEL_ID`.
# Kernels sharing a ``buffer_kind`` share the unified-buffer slab read
# (e.g. BODY_POS / BODY_POS_Z → BODY_POS_W; CONTACT / CONTACT_COUNT /
# CONTACT_COUNT_DIFF → CONTACT_NET_FORCES_W).
# Debug visualization for the four base-tracking kernels (POS, QUAT, LIN_VEL,
# ANG_VEL) is bound in :mod:`.kernels_viz` — see :data:`~.kernels_viz.VIZ_REGISTRY`.
STATE_KERNELS: tuple[StateKernelDef, ...] = (
    StateKernelDef(BUFFER_KIND.JOINT_POS, 0, 1, _state_identity),
    StateKernelDef(BUFFER_KIND.JOINT_VEL, 0, 1, _state_identity),
    StateKernelDef(BUFFER_KIND.BODY_POS_W, 0, 3, _state_identity),
    StateKernelDef(BUFFER_KIND.BODY_QUAT_W, 0, 4, _state_identity),
    StateKernelDef(BUFFER_KIND.BODY_LIN_VEL_W, 0, 3, _state_identity),
    StateKernelDef(BUFFER_KIND.BODY_ANG_VEL_W, 0, 3, _state_identity),
    StateKernelDef(BUFFER_KIND.BODY_POS_W, 2, 1, _state_identity),  # BODY_POS_Z — z component
    StateKernelDef(BUFFER_KIND.CONTACT_NET_FORCES_W, 0, 3, state_body_contact),
    StateKernelDef(BUFFER_KIND.CONTACT_NET_FORCES_W, 0, 3, state_body_contact_count),
    StateKernelDef(BUFFER_KIND.CONTACT_NET_FORCES_W, 0, 3, state_body_contact_count_diff),
    StateKernelDef(BUFFER_KIND.JOINT_MECH_POWER_ABS, 0, 1, state_joint_mech_power),
)


# Flat tuples derived from STATE_KERNELS for hot-path indexing. Exposed as
# module attributes so test patching keeps working.
STATE_KERNEL_BUFFER_KIND: tuple[BUFFER_KIND, ...] = tuple(d.buffer_kind for d in STATE_KERNELS)
STATE_KERNEL_COMPUTES: tuple[Callable[[torch.Tensor], torch.Tensor], ...] = tuple(d.compute_fn for d in STATE_KERNELS)


def state_kernel_intra_body_offset(state_kernel_id: int) -> int:
    """First float this kernel reads within one body's slot of its source buffer."""
    return STATE_KERNELS[int(state_kernel_id)].intra_body_offset


def state_kernel_intra_body_stride(state_kernel_id: int) -> int:
    """Number of floats per body this kernel reads from its source buffer."""
    return STATE_KERNELS[int(state_kernel_id)].intra_body_stride


# --- sampler kernels (params -> target) ---
def sampler_uniform(params: torch.Tensor) -> torch.Tensor:
    """Per-dim uniform sample. ``params`` is ``[..., 2*Dmax]`` interleaved ``[min, range]``."""
    mn = params[..., 0::2]  # [..., Dmax]
    rg = params[..., 1::2]  # [..., Dmax]
    return mn + torch.rand_like(mn) * rg


def sampler_euler_uniform_to_quat(params: torch.Tensor) -> torch.Tensor:
    """Sample Euler (roll, pitch, yaw) uniformly and convert to a unit quaternion.

    ``params`` is ``[..., 2*Dmax]`` interleaved ``[min, range]``; the first 3 (min, range)
    pairs encode the Euler angle bounds (rad) and any trailing pairs are padding reserved
    for aligning ``target_dim_max`` with the 4-dim quaternion output. Output is
    ``[..., 4]`` in xyzw ordering.
    """
    mn = params[..., 0:6:2]  # [..., 3]
    rg = params[..., 1:6:2]  # [..., 3]
    euler = mn + torch.rand_like(mn) * rg
    return quat_from_euler_xyz(euler[..., 0], euler[..., 1], euler[..., 2])


SAMPLER_KERNELS = (sampler_uniform, sampler_euler_uniform_to_quat)
