# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single source of truth for kernel ids and metadata schemas.

The multi-task command term has two parallel implementations of every kernel:
a PyTorch reference (:mod:`.impl.kernels_torch`) and Warp kernels
(:mod:`.impl.kernels_wp`). Both must agree on the integer ids the spec routes by
— a desync silently dispatches to the wrong projection.

This module is the **only** place ids are declared. Both implementation
modules import their constants from here:

- :mod:`.impl.kernels_torch` builds its registry tuples (``STATE_KERNELS``,
  ``BUFFER_KINDS``, …) keyed by :class:`STATE_KERNEL_ID` / :class:`BUFFER_KIND`.
- :mod:`.impl.kernels_wp` derives its ``STATE_*`` / ``ACTIVATION_*`` ``wp.constant``
  values from the same enums (``wp.constant(int(STATE_KERNEL_ID.X))``), so
  Warp branch ids cannot drift from PyTorch.

Schema dataclasses (:class:`BufferKindDef`, :class:`StateKernelDef`) also
live here — they describe the metadata each id must carry, and are
populated by :mod:`.impl.kernels_torch` with concrete PyTorch function references.

Adding a new kernel:

  1. Add an entry to the relevant enum in this file (with docstring).
  2. Add a row to the corresponding registry tuple in :mod:`.impl.kernels_torch`,
     pointing at a new PyTorch implementation function.
  3. Add a ``@wp.func`` projection (and an ``elif`` branch in
     ``dispatch_mega``) in :mod:`.impl.kernels_wp`.

Step 1 is what makes step 2 and step 3 stay in sync; both modules import
the same enum and the same int value flows through.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# Activation / metric / sampler / state kernel ids.
# -----------------------------------------------------------------------------


class ACTIVATION_KERNEL_ID(enum.IntEnum):
    """Activation kernel — maps scalar error → activation in [0, 1]."""

    TANH = 0
    """``1 - tanh(error / σ)``. Steepest at error=0 (slope = −1/σ), then flattens
    out toward 0 as error grows. ``σ`` controls scale. Use for tracking-style
    subtasks where you want consistent gradient pressure across the whole
    error range — small errors and large errors both contribute to learning."""
    LESS = 1
    """``error < threshold`` → ``1.0``, else ``0.0``. Hard predicate; for instant subtasks."""
    GREATER = 2
    """``error > threshold`` → ``1.0``, else ``0.0``. Hard predicate (mirror of ``LESS``)."""
    GAUSSIAN = 3
    """``exp(-(error / σ)²)``. Smooth peak of 1 at error=0 with zero slope —
    small violations don't penalize at all. Slope reaches its steepest value
    at error = σ/√2 ≈ 0.71·σ (the "budget edge"), then flattens back out as
    error → ∞ where the activation is essentially 0.

    Use for soft-safety / budget-style subtasks where:
      - small excursions from zero violation are *expected* and shouldn't
        impose any meaningful gradient cost (e.g. ~10W of standing power
        on a quadruped; an inevitable single contact event during a fall);
      - violations near the configured budget σ should sharply penalize
        — that's where the function transitions from "fine" to "not fine";
      - violations well past 2σ are saturated at near-zero, so the gradient
        doesn't get pulled toward "endless reduction" once the budget is
        already blown — there's no benefit to going from 5σ to 4σ.

    Compare with TANH which is sharpest at error=0 (immediately punishes
    any nonzero error) and flattens out at large errors. The two shapes
    target different supervision regimes."""


class METRIC_KERNEL_ID(enum.IntEnum):
    """Metric kernel — maps a delta vector to a scalar non-negative error."""

    GEOMETRIC = 0
    """L2 norm. For stride-1 inputs this is ``|delta|``; for stride-n, the Euclidean norm."""
    QUATERNION = 1
    """Angle magnitude of a delta quaternion. Maps ``[w, x, y, z]`` delta-quat to ``[0, π]``."""


class STATE_KERNEL_ID(enum.IntEnum):
    """State kernel — projects raw scene data into the per-subtask state vector."""

    JOINT_POS = 0
    JOINT_VEL = 1
    BODY_POS = 2
    BODY_QUAT = 3
    BODY_LIN_VEL = 4
    BODY_ANG_VEL = 5
    BODY_POS_Z = 6
    """Z-component of env-local body position, stride 1. Projects into the ``z`` slot of
    the canonical per-entity block — lets foot tasks target height without over-constraining
    xy. Note: spawn-plane-relative, NOT terrain-aware — prefer ``BODY_CONTACT`` for
    "foot on ground" semantics."""
    BODY_CONTACT = 7
    """Binary contact predicate per body, stride 1. ``1.0`` iff net contact force on
    the body exceeds a small threshold, else ``0.0``. Terrain-agnostic — directly
    reads the physical contact signal rather than inferring it from geometry. Use
    for "foot on ground / in air" tasks that must hold under arbitrary terrain."""
    BODY_CONTACT_COUNT = 8
    """Count of bodies in contact across the asset_cfg's body set, stride 1.
    Single scalar per env — permutation-invariant over the listed bodies. Use for
    gait-style tasks like "exactly 3 of the 4 feet are on the ground," where the
    identity of the lifted foot is irrelevant (the walking-tripod stance)."""
    BODY_CONTACT_COUNT_DIFF = 9
    """Signed difference of contact counts between two equal-sized body groups, stride 1.
    ``asset_cfg.body_ids`` must list an even number of bodies; the first half is one
    group, the second half the other. Returns ``count(first_half) - count(second_half)``.
    Combined with metric ``GEOMETRIC`` (|diff|) and activation ``GREATER``, this
    expresses "one group fully planted, the other fully airborne" without hard-coding
    which of the two phases — e.g. trot (one diagonal vs the other) or bound (front
    pair vs hind pair)."""
    JOINT_MECH_POWER = 10
    """Instantaneous total mechanical power ``Σ_j |τ_j · q̇_j|`` [W] across the
    asset_cfg's joint set, stride 1. Single scalar per env — the ``Σ_j`` sums
    across **joints**, not across time, so this is a per-step quantity in watts
    (not joules). The composer's transit-mean over per-step activations is then
    bounded in ``[0, 1]`` independent of episode length.

    Pair with a :class:`~.multi_task_cfg.MultiTaskCfg.TrackingTaskCfg`
    (``expose_in_obs=False``, target=0) + GAUSSIAN activation
    (``param`` = power scale σ in W) to apply a soft-safety multiplicative
    factor on the composer's terminal reward — penalizing high-actuation gaits
    without per-step shaping that breaks the bootstrap contract. Joint-indexed;
    ``asset_cfg.joint_ids = slice(None)`` (all joints) is the typical setting."""


class SAMPLER_KERNEL_ID(enum.IntEnum):
    """Sampler kernel — maps interleaved ``[min, range]`` params to a target draw."""

    UNIFORM = 0
    """Per-dim uniform draw. Output dim = number of (min, range) pairs."""
    EULER_UNIFORM_TO_QUAT = 1
    """Sample 3 Euler angles uniformly, convert to a unit quaternion (xyzw, dim 4)."""


class BUFFER_KIND(enum.IntEnum):
    """Identifies a scene-data field. One tensor view per ``(kind, asset_name)`` per step.

    Contract: every buffer / state kernel here operates in **world frame**. The
    composer's ``delta`` and ``error`` are world-frame too. Obs consumers are
    responsible for rotating spatial slots into their local robot frame when
    they read ``command_reach`` / ``command_track`` — see the ``asset_cfg=``
    argument on :func:`~..observations.command_track_b`.
    """

    JOINT_POS = 0
    JOINT_VEL = 1
    BODY_POS_W = 2
    BODY_QUAT_W = 3
    BODY_LIN_VEL_W = 4
    BODY_ANG_VEL_W = 5
    CONTACT_NET_FORCES_W = 6
    JOINT_MECH_POWER_ABS = 7
    """Per-joint ``|τ_j · q̇_j|`` (absolute mechanical power), shape ``[N, num_joints]``.

    Reader allocates a per-step torch tensor (``applied_effort * joint_vel``,
    abs); this is the one buffer in the registry that does NOT meet the
    "zero-copy view" contract — the elementwise product cannot be expressed
    as a stride view of any underlying scene array.
    """


# -----------------------------------------------------------------------------
# Schema dataclasses — populated by :mod:`.impl.kernels_torch` with PyTorch function refs.
# Warp side imports the *enums* (above) and reproduces the math via @wp.func.
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BufferKindDef:
    """Static metadata for one :class:`BUFFER_KIND` slab.

    Args:
        reader: Returns the slab's raw tensor as a zero-copy torch view over
            the scene's underlying warp storage, shape ``[N, *]``.
            **Contract: no allocations, no transformations.** Any frame
            adjustment (e.g. env-local body pos) is applied by the dispatch
            (Warp kernel or inline PyTorch) so this path stays pointer-stable
            across steps — necessary for CUDA Graph capture.
        per_element_stride: Floats contributed by each indexed element (body or
            joint) to a slab of this kind.
        is_body_indexed: ``True`` if the buffer's second axis is body-indexed
            (uses ``asset_cfg.body_ids``); ``False`` for joint-indexed buffers.
    """

    reader: Callable[[ManagerBasedRLEnv, str], torch.Tensor]
    per_element_stride: int
    is_body_indexed: bool


@dataclass(frozen=True)
class StateKernelDef:
    """Static metadata for one :class:`STATE_KERNEL_ID`.

    Args:
        buffer_kind: Which slab this kernel reads from. Multiple kernels can
            share a kind (e.g. POS / POS_Z both read BODY_POS_W).
        intra_body_offset: First float this kernel reads within one body's slot
            of its source buffer (e.g. POS_Z reads offset 2 = z component).
        intra_body_stride: Floats per body the kernel reads (1 for POS_Z;
            equals the buffer's full per-body stride for kernels reading the
            whole block).
        compute_fn: Pure tensor op ``[M, N, slice_size] → [M, N, stride]``.
            Runs on the stacked unified-buffer gather for one read group.

    Debug visualization is bound separately in :mod:`.impl.kernels_viz` via
    :data:`~.impl.kernels_viz.VIZ_REGISTRY` so the math registry stays Kit-free.
    """

    buffer_kind: BUFFER_KIND
    intra_body_offset: int
    intra_body_stride: int
    compute_fn: Callable[[torch.Tensor], torch.Tensor]
