# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: SIM109
# Warp DSL does not support ``x in (tuple)`` membership tests inside ``@wp.kernel``
# bodies — chained ``or`` equalities are the required form for branch gating.

"""Warp dispatch for :class:`MultiTaskCommand`.

The per-step pipeline is two ``wp.launch`` phases — read (slab → unified) and
execute (per-slot project → delta → metric → activation → scatter). Both
phases collapse what was ~440 PyTorch kernel launches into ≈8 Warp launches.

Argument bundles (Warp structs):

The execute kernel's inputs are grouped into four domain-specific structs so
the signature is readable and helpers can accept one struct instead of many
loose arrays:

- :class:`EnvSlots` — per-env live routing (refreshed on resample). Shape
  ``[N, k_max]`` and ``[N]`` arrays.
- :class:`SubtaskSpec` — per-subtask metadata. Immutable after spec build.
  Shape ``[M]`` arrays.
- :class:`StateAccess` — per-step read-side buffers: the unified state
  tensor + the CSR gather indices into it + the flat targets tensor.
- :class:`Outputs` — the four per-step output tensors.

Architecture (execute phase):

- :func:`dispatch_mega` is a thin shell (~60 lines) that resolves per-slot
  metadata and routes to one of the ``_project_*`` functions based on
  ``SubtaskSpec.state_kernel_id``. The shell owns gather metadata, scatter,
  and activation; everything kernel-specific lives in the ``@wp.func`` leaves.
- Each state kernel has its own ``_project_*`` ``@wp.func`` that reads from
  ``StateAccess`` and returns its post-projection vector (typed as
  :class:`wp.vec3` / :class:`wp.vec4` / ``float`` as appropriate).
- Error computation is split into one ``@wp.func`` per metric kind
  (:func:`_metric_geometric_vec3`, :func:`_metric_geometric_scalar`, :func:`_metric_quaternion`).
- Delta scatter to the outputs lives in :func:`_scatter_delta` — stride-gated,
  routes by ``is_instant``.
- Activation applied via :func:`_apply_activation` (one branch per kid).

Adding a new state kernel is three steps, each local:
  1. Add a ``_project_*`` ``@wp.func``.
  2. Add an ``elif skid == STATE_NEW:`` branch in :func:`dispatch_mega`
     that calls the new projection and the appropriate error helper.
  3. Register the kernel id in :mod:`..kernel_ids`.

No changes to the shell signature, no changes to scatter / activation, no
changes to the struct definitions.

Conventions:

- Quaternions are xyzw throughout (matches :mod:`isaaclab.utils.math`).
- Float arithmetic in ``wp.float32``.
- Warp inlines ``@wp.func`` calls during JIT; the struct + function
  decomposition has zero runtime cost relative to a monolithic kernel.
"""

from __future__ import annotations

import warp as wp

# ---------------------------------------------------------------------------
# Branch keys — derived directly from the enums in ``kernel_ids.py`` so a new
# kernel id added in one place automatically lands here without any hand-copy.
# Warp's ``wp.constant`` requires a literal integer, so each ``STATE_*`` /
# ``ACTIVATION_*`` is a constant *whose value comes from the central enum*. If the
# enum is renumbered or a new entry is inserted, every Warp branch using
# these constants picks up the new value at module import time — no chance
# of drift between PyTorch and Warp branch ids.
# ---------------------------------------------------------------------------
from ..kernel_ids import ACTIVATION_KERNEL_ID, STATE_KERNEL_ID

# state kernel ids
STATE_JOINT_POS = wp.constant(int(STATE_KERNEL_ID.JOINT_POS))
STATE_JOINT_VEL = wp.constant(int(STATE_KERNEL_ID.JOINT_VEL))
STATE_BODY_POS = wp.constant(int(STATE_KERNEL_ID.BODY_POS))
STATE_BODY_QUAT = wp.constant(int(STATE_KERNEL_ID.BODY_QUAT))
STATE_BODY_LIN_VEL = wp.constant(int(STATE_KERNEL_ID.BODY_LIN_VEL))
STATE_BODY_ANG_VEL = wp.constant(int(STATE_KERNEL_ID.BODY_ANG_VEL))
STATE_BODY_POS_Z = wp.constant(int(STATE_KERNEL_ID.BODY_POS_Z))
STATE_BODY_CONTACT = wp.constant(int(STATE_KERNEL_ID.BODY_CONTACT))
STATE_BODY_CONTACT_COUNT = wp.constant(int(STATE_KERNEL_ID.BODY_CONTACT_COUNT))
STATE_BODY_CONTACT_COUNT_DIFF = wp.constant(int(STATE_KERNEL_ID.BODY_CONTACT_COUNT_DIFF))
STATE_JOINT_MECH_POWER = wp.constant(int(STATE_KERNEL_ID.JOINT_MECH_POWER))

# activation kernel ids
ACTIVATION_TANH = wp.constant(int(ACTIVATION_KERNEL_ID.TANH))
ACTIVATION_LESS = wp.constant(int(ACTIVATION_KERNEL_ID.LESS))
ACTIVATION_GREATER = wp.constant(int(ACTIVATION_KERNEL_ID.GREATER))
ACTIVATION_GAUSSIAN = wp.constant(int(ACTIVATION_KERNEL_ID.GAUSSIAN))

# Contact-force threshold — matches ``_contact_predicates_from_flat`` in
# ``kernels.py`` (norm > 1 N ⇒ contact).
_CONTACT_MAG_THRESHOLD_SQ = wp.constant(1.0)

# Packed-scatter fused-pipeline ids. These are computation shapes, not semantic
# state kernels. For example, contact-count subtasks lower to a
# vec3-threshold-sum pipeline instead of being treated as a "contact primitive".
PIPELINE_DIRECT_VEC3_DELTA = wp.constant(0)
PIPELINE_DIRECT_SCALAR_DELTA = wp.constant(1)
PIPELINE_DIRECT_QUAT_DELTA = wp.constant(2)
PIPELINE_VEC3_THRESHOLD_VECTOR_DELTA = wp.constant(3)
PIPELINE_VEC3_THRESHOLD_SUM_DELTA = wp.constant(4)
PIPELINE_VEC3_THRESHOLD_PAIR_DIFF_DELTA = wp.constant(5)
PIPELINE_SCALAR_SUM_DELTA = wp.constant(6)


# ---------------------------------------------------------------------------
# Argument bundles — the domain-specific groupings passed to the kernel.
# ---------------------------------------------------------------------------


@wp.struct
class EnvSlots:
    """Per-env live slot routing.

    Refreshed by :meth:`MultiTaskCommand._resample_command` whenever an env
    gets a fresh task assignment. All three arrays are int32 because Warp
    requires int32 for index arrays; the base class keeps int64 companions
    for PyTorch advanced indexing and the Warp subclass ``copy_`` s those
    into these int32 mirrors each step.
    """

    subtask_ids: wp.array2d(dtype=int)
    """``[N, k_max]`` — which subtask fills each of the env's active slots."""
    slot_count: wp.array(dtype=int)
    """``[N]`` — number of active slots per env. Slots ``>= slot_count`` are padded."""
    slot_offsets: wp.array2d(dtype=int)
    """``[N, k_max]`` — offset into :attr:`StateAccess.targets_flat` for each slot."""


@wp.struct
class PackedScatterQueue:
    """Flat fused-pipeline-sorted queue that scatters into legacy outputs."""

    env_ids: wp.array(dtype=int)
    """``[max_work]`` — env id for each queued work item."""
    slot_ids: wp.array(dtype=int)
    """``[max_work]`` — slot id for each queued work item."""
    subtask_ids: wp.array(dtype=int)
    """``[max_work]`` — subtask id for each queued work item."""
    target_offsets: wp.array(dtype=int)
    """``[max_work]`` — target slice offset for each queued work item."""
    pipeline_ids: wp.array(dtype=int)
    """``[max_work]`` — fused primitive-pipeline id for each queued work item."""
    count: wp.array(dtype=int)
    """``[1]`` — total valid work items."""


@wp.struct
class PrimitiveLocalQueue:
    """Flat fused-pipeline-sorted queue with env-slot to local-output mapping."""

    env_ids: wp.array(dtype=int)
    """``[max_work]`` — env id for each queued work item."""
    slot_ids: wp.array(dtype=int)
    """``[max_work]`` — slot id for each queued work item."""
    subtask_ids: wp.array(dtype=int)
    """``[max_work]`` — subtask id for each queued work item."""
    target_offsets: wp.array(dtype=int)
    """``[max_work]`` — target slice offset for each queued work item."""
    schedule_offsets: wp.array(dtype=int)
    """``[num_schedules]`` — first local output row for each primitive schedule."""
    schedule_counts: wp.array(dtype=int)
    """``[num_schedules]`` — queued item count for each primitive schedule."""
    count: wp.array(dtype=int)
    """``[1]`` — total valid work items."""


@wp.struct
class PrimitiveProducerQueue:
    """Static signature lookup tables for one producer kind.

    Each producer kind groups subtasks by gather signature (target-independent
    state read). The dense graph kernels read ``signature_subtask`` to find the
    representative subtask whose gather block defines the producer, and
    ``subtask_signature`` to map a consumer's subtask back to its signature.
    Both tables are spec-derived and constant across resamples.
    """

    subtask_signature: wp.array(dtype=int)
    """``[num_subtasks]`` — producer signature id for each subtask, or ``-1``."""
    signature_subtask: wp.array(dtype=int)
    """``[num_signatures]`` — representative subtask for each signature."""


@wp.struct
class SubtaskSpec:
    """Per-subtask metadata — immutable after spec build.

    Int32 copies of the spec tables live in the mega backend's
    ``MegaKernelPlan`` and are wrapped into this struct once at construction
    time.
    """

    state_kernel_id: wp.array(dtype=int)
    """``[M]`` — which state kernel routes this subtask's projection branch."""
    metric_kernel_id: wp.array(dtype=int)
    """``[M]`` — reserved for future branches that need runtime metric selection."""
    activation_kernel_id: wp.array(dtype=int)
    """``[M]`` — which activation branch (:data:`ACTIVATION_TANH` / LESS / GREATER) to apply."""
    activation_kernel_param: wp.array(dtype=float)
    """``[M]`` — scalar param passed to the activation kernel (std for tanh, threshold for less/greater)."""
    state_stride: wp.array(dtype=int)
    """``[M]`` — output dim after projection (1, 3, or 4). Drives the scatter loop bound."""
    canonical_offset: wp.array(dtype=int)
    """``[M]`` — offset into reach / track tensors for canonical scatter, or ``-1`` to skip."""
    is_instant_flag: wp.array(dtype=int)
    """``[M]`` — routes the canonical scatter: ``1`` → ``command_reach``, ``0`` → ``command_track``.

    Also drives the composer's instant-vs-tracking classification. For every
    BaseTaskCfg in the current hierarchy, exactly one of ``is_instant_flag``
    / ``is_tracking_flag`` is ``1``; the composer guards against future
    extensions by reading both rather than assuming ``1 - is_instant``.
    """
    is_tracking_flag: wp.array(dtype=int)
    """``[M]`` — ``1`` iff this subtask is a tracking (quality) subtask.
    Includes both ordinary tracking goals and soft-safety constraints —
    they're unified at the reward-composition level. The composer accumulates
    every tracking subtask's transit-mean into the eased multiplicative
    quality factor ``( ∏_k mean_t A_k(t) ) ^ quality_easing``."""
    gather_offset: wp.array(dtype=int)
    """``[M]`` — start of this subtask's block in :attr:`SubtaskSpec.gather_indices_flat`."""
    gather_count: wp.array(dtype=int)
    """``[M]`` — length of this subtask's gather block (in floats read from unified)."""
    gather_indices_flat: wp.array(dtype=int)
    """``[sum(gather_count)]`` — concatenated absolute indices into :attr:`StateAccess.unified`.

    Per-subtask CSR: subtask ``sid`` reads indices ``[gather_offset[sid],
    gather_offset[sid] + gather_count[sid])``. Stored here (not in
    :class:`StateAccess`) because it's spec-time immutable.
    """


@wp.struct
class StateAccess:
    """Per-step read-side inputs — unified state tensor + flat targets.

    The unified buffer is refilled by the read-dispatch phase
    (``fill_slab_copy`` launches) before :func:`dispatch_mega` runs. The
    targets tensor is written at resample time by the sampler dispatch.
    """

    unified: wp.array2d(dtype=float)
    """``[N, unified_width]`` — per-step post-read state buffer."""
    targets_flat: wp.array2d(dtype=float)
    """``[N, max_task_total_stride]`` — per-env flat targets written at resample."""


@wp.struct
class Outputs:
    """Per-step output tensors written by :func:`dispatch_mega` and :func:`compose_reward`."""

    buf_error: wp.array2d(dtype=float)
    """``[N, k_max]`` — per-slot scalar error from the metric kernel. Written by dispatch."""
    buf_activation: wp.array2d(dtype=float)
    """``[N, k_max]`` — per-slot activation score from the activation kernel. Written by dispatch."""
    command_reach: wp.array2d(dtype=float)
    """``[N, reach_canonical_width]`` — canonical deltas from instant subtasks. Written by dispatch."""
    command_track: wp.array2d(dtype=float)
    """``[N, track_canonical_width]`` — canonical deltas from tracking subtasks. Written by dispatch."""
    task_reward: wp.array(dtype=float)
    """``[N]`` — multiplicative terminal reward. Written by composer."""
    task_done_success: wp.array(dtype=wp.bool)
    """``[N]`` — per-env success flag. Written by composer."""
    progress: wp.array(dtype=float)
    """``[N]`` — mean of active-slot activations ∈ [0, 1]. Written by composer."""


@wp.struct
class ComposerState:
    """In-place per-step composer state.

    Advanced each step by :func:`compose_reward`:

    - ``sum_activation`` accumulates per-step activations per slot.
    - ``transit_steps`` increments by 1.
    - ``instant_achieved`` latches ``True`` once an instant slot's
      activation exceeds the threshold; stays ``True`` until the env's
      resample clears it.

    All three are reset in the base class's ``_resample_command`` via
    PyTorch ``[env_ids] = 0/False`` assignments.
    """

    sum_activation: wp.array2d(dtype=float)
    """``[N, k_max]`` — running sum of activation per slot."""
    transit_steps: wp.array(dtype=int)
    """``[N]`` — steps since last resample; caller guarantees ``≥ 0`` after reset."""
    instant_achieved: wp.array2d(dtype=wp.bool)
    """``[N, k_max]`` — latched per-slot achievement flags."""


# ---------------------------------------------------------------------------
# Read dispatch — slab → unified buffer copy.
# ---------------------------------------------------------------------------


@wp.kernel
def fill_slab_copy(
    source: wp.array2d(dtype=float),
    unified: wp.array2d(dtype=float),
    offset: int,
):
    """Copy ``source[env, i]`` → ``unified[env, offset + i]``.

    Source and destination share the per-env element count (``size``),
    implied by the 2D launch shape ``dim=(num_envs, size)``. One launch per
    slab (5–7 per step for the production cfg) — fixed-arity Warp kernels
    can't accept variable lists of source arrays.
    """
    env, i = wp.tid()
    unified[env, offset + i] = source[env, i]


# ---------------------------------------------------------------------------
# Typed slab fills — read scene wp.array views directly without laundering
# through Torch + reshape. The vec3/quat variants handle PhysX's padded body
# buffer layout natively (Warp typed indexing respects the per-element stride),
# so no compaction copy is needed and the launches are capture-safe.
# ---------------------------------------------------------------------------


@wp.kernel
def fill_slab_vec3(
    source: wp.array2d(dtype=wp.vec3),
    unified: wp.array2d(dtype=float),
    offset: int,
):
    """Copy a ``vec3`` scene slab into the unified float buffer.

    Launch shape: ``dim=(num_envs, num_elements)``. Reads one ``vec3`` per
    thread (Warp handles any per-element padding via the source's stride),
    writes the 3 components contiguously into ``unified[env, offset + b*3..]``.
    """
    env, b = wp.tid()
    v = source[env, b]
    base = offset + b * 3
    unified[env, base] = v[0]
    unified[env, base + 1] = v[1]
    unified[env, base + 2] = v[2]


@wp.kernel
def fill_slab_vec3_env_local(
    body_pos_w: wp.array2d(dtype=wp.vec3),
    env_origins: wp.array(dtype=wp.vec3),
    unified: wp.array2d(dtype=float),
    offset: int,
):
    """Vec3 body-pos slab with env-origin subtraction.

    Launch shape: ``dim=(num_envs, num_bodies)``.
    """
    env, b = wp.tid()
    p = body_pos_w[env, b] - env_origins[env]
    base = offset + b * 3
    unified[env, base] = p[0]
    unified[env, base + 1] = p[1]
    unified[env, base + 2] = p[2]


@wp.kernel
def fill_slab_quat(
    source: wp.array2d(dtype=wp.quat),
    unified: wp.array2d(dtype=float),
    offset: int,
):
    """Copy a ``quat`` scene slab into the unified float buffer.

    Launch shape: ``dim=(num_envs, num_elements)``. Writes 4 components
    contiguously per body.
    """
    env, b = wp.tid()
    q = source[env, b]
    base = offset + b * 4
    unified[env, base] = q[0]
    unified[env, base + 1] = q[1]
    unified[env, base + 2] = q[2]
    unified[env, base + 3] = q[3]


@wp.kernel
def fill_slab_joint_mech_power_abs(
    applied_effort: wp.array2d(dtype=float),
    joint_vel: wp.array2d(dtype=float),
    unified: wp.array2d(dtype=float),
    offset: int,
):
    """Compute ``|τ · q̇|`` element-wise into the unified buffer.

    Reads the two underlying ``wp.array`` directly so the Warp dispatch never
    materializes a computed reader's result.

    NaN-safe: non-finite products are clamped to 0 to match the Torch
    reference's defensive handling of reset transients on some physics
    backends.

    Launch shape: ``dim=(num_envs, num_joints)``.
    """
    env, j = wp.tid()
    p = wp.abs(applied_effort[env, j] * joint_vel[env, j])
    if wp.isfinite(p):
        unified[env, offset + j] = p
    else:
        unified[env, offset + j] = 0.0


# ---------------------------------------------------------------------------
# Projections — one ``@wp.func`` per state kernel family. Each reads from
# ``StateAccess`` + ``SubtaskSpec.gather_indices_flat`` and returns the
# projected state in the smallest typed container that fits.
# ---------------------------------------------------------------------------


@wp.func
def _project_xyz(state: StateAccess, spec: SubtaskSpec, env: int, gbase: int) -> wp.vec3:
    """Three contiguous floats — used by POS, LIN_VEL, ANG_VEL."""
    return wp.vec3(
        state.unified[env, spec.gather_indices_flat[gbase]],
        state.unified[env, spec.gather_indices_flat[gbase + 1]],
        state.unified[env, spec.gather_indices_flat[gbase + 2]],
    )


@wp.func
def _project_scalar(state: StateAccess, spec: SubtaskSpec, env: int, gbase: int) -> float:
    """Single float — used by JOINT_POS, JOINT_VEL, BODY_POS_Z."""
    return state.unified[env, spec.gather_indices_flat[gbase]]


@wp.func
def _project_quat_xyzw(state: StateAccess, spec: SubtaskSpec, env: int, gbase: int) -> wp.vec4:
    """Four floats in xyzw order — used by BODY_QUAT."""
    return wp.vec4(
        state.unified[env, spec.gather_indices_flat[gbase]],
        state.unified[env, spec.gather_indices_flat[gbase + 1]],
        state.unified[env, spec.gather_indices_flat[gbase + 2]],
        state.unified[env, spec.gather_indices_flat[gbase + 3]],
    )


@wp.func
def _state_body_contact(state: StateAccess, spec: SubtaskSpec, env: int, gbase: int, gcount: int) -> wp.vec4:
    """Per-body contact predicates — K bodies, K ≤ 4. Returns a padded vec4.

    ``gcount = K · 3`` (force xyz per body). Each lane is ``1.0`` iff that
    body's contact force magnitude² exceeds :data:`_CONTACT_MAG_THRESHOLD_SQ`.
    Lanes beyond K stay ``0.0`` so downstream delta math (target - current)
    has well-defined padded zeros.

    The K ≤ 4 cap comes from the shared :func:`_scatter_delta` which writes
    up to 4 channels (``d0..d3``) per slot. Relaxing this would require
    either a scatter variant with a wider output or multiple launches;
    current cfgs max out at K=4 (four feet).
    """
    k = gcount / 3
    c0 = float(0.0)
    c1 = float(0.0)
    c2 = float(0.0)
    c3 = float(0.0)
    for bi in range(k):
        fx = state.unified[env, spec.gather_indices_flat[gbase + bi * 3]]
        fy = state.unified[env, spec.gather_indices_flat[gbase + bi * 3 + 1]]
        fz = state.unified[env, spec.gather_indices_flat[gbase + bi * 3 + 2]]
        is_contact = float(0.0)
        if fx * fx + fy * fy + fz * fz > _CONTACT_MAG_THRESHOLD_SQ:
            is_contact = 1.0
        if bi == 0:
            c0 = is_contact
        if bi == 1:
            c1 = is_contact
        if bi == 2:
            c2 = is_contact
        if bi == 3:
            c3 = is_contact
    return wp.vec4(c0, c1, c2, c3)


@wp.func
def _state_body_contact_count(state: StateAccess, spec: SubtaskSpec, env: int, gbase: int, gcount: int) -> float:
    """Per-body contact predicates summed over K bodies — returns ``cnt``.

    ``gcount = K·3`` (3 force components per body). Reads each body's xyz
    and counts those with magnitude² > threshold.
    """
    k = gcount / 3
    cnt = float(0.0)
    for bi in range(k):
        fx = state.unified[env, spec.gather_indices_flat[gbase + bi * 3]]
        fy = state.unified[env, spec.gather_indices_flat[gbase + bi * 3 + 1]]
        fz = state.unified[env, spec.gather_indices_flat[gbase + bi * 3 + 2]]
        if fx * fx + fy * fy + fz * fz > _CONTACT_MAG_THRESHOLD_SQ:
            cnt = cnt + 1.0
    return cnt


@wp.func
def _state_joint_mech_power(state: StateAccess, spec: SubtaskSpec, env: int, gbase: int, gcount: int) -> float:
    """Σ_j |τ_j · q̇_j| [W] — instantaneous total mechanical power across joints.

    The reader has already computed |τ·q̇| per joint and written it into the
    unified buffer; this just sums along the joint axis. ``gcount`` is the
    number of joints (per_element_stride = 1). Reduction is over joints, not
    time — output is power (W), not work (J).
    """
    total = float(0.0)
    for j in range(gcount):
        total = total + state.unified[env, spec.gather_indices_flat[gbase + j]]
    return total


@wp.func
def _state_body_contact_count_diff(state: StateAccess, spec: SubtaskSpec, env: int, gbase: int, gcount: int) -> float:
    """``count(first K/2) - count(last K/2)`` — used by gait subtasks.

    Splits the K bodies into two halves and returns the signed difference.
    """
    k = gcount / 3
    half = k / 2
    cnt_a = float(0.0)
    cnt_b = float(0.0)
    for bi in range(k):
        fx = state.unified[env, spec.gather_indices_flat[gbase + bi * 3]]
        fy = state.unified[env, spec.gather_indices_flat[gbase + bi * 3 + 1]]
        fz = state.unified[env, spec.gather_indices_flat[gbase + bi * 3 + 2]]
        is_contact = float(0.0)
        if fx * fx + fy * fy + fz * fz > _CONTACT_MAG_THRESHOLD_SQ:
            is_contact = 1.0
        if bi < half:
            cnt_a = cnt_a + is_contact
        else:
            cnt_b = cnt_b + is_contact
    return cnt_a - cnt_b


# ---------------------------------------------------------------------------
# Target readers — pull one subtask's target slice from ``StateAccess.targets_flat``.
# ---------------------------------------------------------------------------


@wp.func
def _read_target_xyz(state: StateAccess, env: int, tgt_off: int) -> wp.vec3:
    return wp.vec3(
        state.targets_flat[env, tgt_off],
        state.targets_flat[env, tgt_off + 1],
        state.targets_flat[env, tgt_off + 2],
    )


@wp.func
def _read_target_quat_xyzw(state: StateAccess, env: int, tgt_off: int) -> wp.vec4:
    return wp.vec4(
        state.targets_flat[env, tgt_off],
        state.targets_flat[env, tgt_off + 1],
        state.targets_flat[env, tgt_off + 2],
        state.targets_flat[env, tgt_off + 3],
    )


# ---------------------------------------------------------------------------
# Deltas — one per metric kind.
# ---------------------------------------------------------------------------


@wp.func
def _delta_quaternion(c: wp.vec4, t: wp.vec4) -> wp.vec4:
    """Delta quaternion ``inv(c) · t`` (xyzw). Unit-quat inverse negates xyz.

    Expanding the Hamilton product with (x₁,y₁,z₁,w₁) = (−cx, −cy, −cz, cw)
    and (x₂,y₂,z₂,w₂) = (tx, ty, tz, tw):
    """
    cx = c[0]
    cy = c[1]
    cz = c[2]
    cw = c[3]
    tx = t[0]
    ty = t[1]
    tz = t[2]
    tw = t[3]
    dw = cw * tw + cx * tx + cy * ty + cz * tz
    dx = cw * tx - cx * tw - cy * tz + cz * ty
    dy = cw * ty + cx * tz - cy * tw - cz * tx
    dz = cw * tz - cx * ty + cy * tx - cz * tw
    return wp.vec4(dx, dy, dz, dw)


@wp.func
def _quat_apply_inverse_xyzw(q: wp.vec4, v: wp.vec3) -> wp.vec3:
    """Rotate vector ``v`` by ``q^-1`` for unit quaternions in xyzw order."""
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    vx = v[0]
    vy = v[1]
    vz = v[2]

    ox = (1.0 - 2.0 * (y * y + z * z)) * vx + 2.0 * (x * y + z * w) * vy + 2.0 * (x * z - y * w) * vz
    oy = 2.0 * (x * y - z * w) * vx + (1.0 - 2.0 * (x * x + z * z)) * vy + 2.0 * (y * z + x * w) * vz
    oz = 2.0 * (x * z + y * w) * vx + 2.0 * (y * z - x * w) * vy + (1.0 - 2.0 * (x * x + y * y)) * vz
    return wp.vec3(ox, oy, oz)


# ---------------------------------------------------------------------------
# Errors — one per metric kind.
# ---------------------------------------------------------------------------


@wp.func
def _metric_geometric_vec3(d: wp.vec3) -> float:
    """Euclidean norm of a 3-vector. Matches :func:`kernels.metric_geometric`."""
    return wp.length(d)


@wp.func
def _metric_geometric_scalar(d: float) -> float:
    """``|d|`` — L2 norm of a 1-vector. Same as ``wp.abs``."""
    return wp.abs(d)


@wp.func
def _metric_quaternion(d: wp.vec4) -> float:
    """Angle magnitude of a delta quaternion — ``2 · atan2(‖v‖, |w|) ∈ [0, π]``.

    Mirrors :func:`isaaclab.utils.math.axis_angle_from_quat` composed with an
    L2 norm; see :func:`kernels.metric_quaternion`.
    """
    dx = d[0]
    dy = d[1]
    dz = d[2]
    dw = d[3]
    v = wp.sqrt(dx * dx + dy * dy + dz * dz)
    return 2.0 * wp.atan2(v, wp.abs(dw))


# ---------------------------------------------------------------------------
# Activation — one ``@wp.func`` per ID, grep-symmetric with PyTorch
# ``activation_*`` in :mod:`.kernels_torch`. Plus a thin dispatcher.
# ---------------------------------------------------------------------------


@wp.func
def _activation_tanh(err: float, param: float) -> float:
    """``1 - tanh(err / param)``. Mirrors :func:`kernels_torch.activation_tanh`."""
    return 1.0 - wp.tanh(err / param)


@wp.func
def _activation_less(err: float, param: float) -> float:
    """``err < param`` → 1.0, else 0.0. Mirrors :func:`kernels_torch.activation_less`."""
    if err < param:
        return 1.0
    return 0.0


@wp.func
def _activation_greater(err: float, param: float) -> float:
    """``err > param`` → 1.0, else 0.0. Mirrors :func:`kernels_torch.activation_greater`."""
    if err > param:
        return 1.0
    return 0.0


@wp.func
def _activation_gaussian(err: float, param: float) -> float:
    """``exp(-(err / param)²)``. Mirrors :func:`kernels_torch.activation_gaussian`."""
    z = err / param
    return wp.exp(-(z * z))


@wp.func
def _apply_activation(kid: int, err: float, param: float) -> float:
    """Dispatch on activation kid → one of ``_activation_*``.

    Warp inlines ``@wp.func`` calls; this thin dispatcher costs nothing
    relative to the previous monolithic ``if/elif`` body but makes each
    activation directly grep-able under its ID name (and symmetric with
    the PyTorch path).
    """
    if kid == ACTIVATION_TANH:
        return _activation_tanh(err, param)
    if kid == ACTIVATION_LESS:
        return _activation_less(err, param)
    if kid == ACTIVATION_GREATER:
        return _activation_greater(err, param)
    # GAUSSIAN (implicit else — spec guarantees kid ∈ {0, 1, 2, 3}).
    return _activation_gaussian(err, param)


# ---------------------------------------------------------------------------
# Scatter — writes the delta vector to the correct canonical tensor slice.
# ---------------------------------------------------------------------------


@wp.func
def _scatter_delta(
    outputs: Outputs,
    env: int,
    canon_off: int,
    stride: int,
    instant: int,
    d0: float,
    d1: float,
    d2: float,
    d3: float,
):
    """Write the delta vector at ``[canon_off, canon_off + stride)``.

    Joint kernels mark themselves unscattered with ``canon_off = -1`` (no
    canonical projection). The ``instant`` flag routes the write to
    ``outputs.command_reach`` (instant subtask) or ``outputs.command_track``
    (tracking).
    """
    if canon_off < 0:
        return
    if instant != 0:
        if stride >= 1:
            outputs.command_reach[env, canon_off] = d0
        if stride >= 2:
            outputs.command_reach[env, canon_off + 1] = d1
        if stride >= 3:
            outputs.command_reach[env, canon_off + 2] = d2
        if stride >= 4:
            outputs.command_reach[env, canon_off + 3] = d3
    else:
        if stride >= 1:
            outputs.command_track[env, canon_off] = d0
        if stride >= 2:
            outputs.command_track[env, canon_off + 1] = d1
        if stride >= 3:
            outputs.command_track[env, canon_off + 2] = d2
        if stride >= 4:
            outputs.command_track[env, canon_off + 3] = d3


# ---------------------------------------------------------------------------
# The shell kernel — resolves metadata, dispatches to a projection +
# error helper by state kernel id, runs activation, scatters.
# ---------------------------------------------------------------------------


@wp.func
def _dispatch_slot(
    env: int,
    slot: int,
    env_slots: EnvSlots,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
):
    # Metric kid is implicit — the Phase 1 branch tree pairs BODY_QUAT with
    # QUATERNION and every other state kernel with GEOMETRIC. Loading
    # ``spec.metric_kernel_id`` is reserved for a future branch that
    # supports mixing metrics on the same state kind (not used today).
    sid = env_slots.subtask_ids[env, slot]
    skid = spec.state_kernel_id[sid]
    akid = spec.activation_kernel_id[sid]
    act_param = spec.activation_kernel_param[sid]
    canon_off = spec.canonical_offset[sid]
    instant = spec.is_instant_flag[sid]
    tgt_off = env_slots.slot_offsets[env, slot]
    stride = spec.state_stride[sid]
    gbase = spec.gather_offset[sid]
    gcount = spec.gather_count[sid]

    # Delta components default to zero so unused lanes don't leak into the
    # scatter. Only ``[0, stride)`` are meaningful per state kernel.
    d0 = float(0.0)
    d1 = float(0.0)
    d2 = float(0.0)
    d3 = float(0.0)
    err = float(0.0)

    # --- state-kernel branch tree: project → delta → error ---
    # Warp requires static types per symbol; each branch uses a distinct
    # local name to avoid type collisions across branches.
    if skid == STATE_BODY_POS or skid == STATE_BODY_LIN_VEL or skid == STATE_BODY_ANG_VEL:
        x3 = _project_xyz(state, spec, env, gbase)
        t3 = _read_target_xyz(state, env, tgt_off)
        d3v = t3 - x3
        d0 = d3v[0]
        d1 = d3v[1]
        d2 = d3v[2]
        err = _metric_geometric_vec3(d3v)
    elif skid == STATE_JOINT_POS or skid == STATE_JOINT_VEL or skid == STATE_BODY_POS_Z:
        x_s = _project_scalar(state, spec, env, gbase)
        t_s = state.targets_flat[env, tgt_off]
        d0 = t_s - x_s
        err = _metric_geometric_scalar(d0)
    elif skid == STATE_BODY_QUAT:
        cq = _project_quat_xyzw(state, spec, env, gbase)
        tq = _read_target_quat_xyzw(state, env, tgt_off)
        dq = _delta_quaternion(cq, tq)
        d0 = dq[0]
        d1 = dq[1]
        d2 = dq[2]
        d3 = dq[3]
        err = _metric_quaternion(dq)
    elif skid == STATE_BODY_CONTACT:
        # Per-body predicate kernel — K ≤ 4. stride (= K) drives how many
        # targets / deltas are meaningful; unused lanes zero out below.
        # Distinct local names (``tc0..tc3``, ``xcb``) avoid Warp's
        # static-type collisions with the xyz / quat branches' ``t3`` etc.
        xcb = _state_body_contact(state, spec, env, gbase, gcount)
        tc0 = state.targets_flat[env, tgt_off]
        tc1 = float(0.0)
        tc2 = float(0.0)
        tc3 = float(0.0)
        if stride >= 2:
            tc1 = state.targets_flat[env, tgt_off + 1]
        if stride >= 3:
            tc2 = state.targets_flat[env, tgt_off + 2]
        if stride >= 4:
            tc3 = state.targets_flat[env, tgt_off + 3]
        d0 = tc0 - xcb[0]
        d1 = tc1 - xcb[1]
        d2 = tc2 - xcb[2]
        d3 = tc3 - xcb[3]
        # L2 over just the live lanes — padded d's are already zero.
        err = wp.sqrt(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3)
    elif skid == STATE_BODY_CONTACT_COUNT:
        x_cnt = _state_body_contact_count(state, spec, env, gbase, gcount)
        tgt_cnt = state.targets_flat[env, tgt_off]
        d0 = tgt_cnt - x_cnt
        err = _metric_geometric_scalar(d0)
    elif skid == STATE_BODY_CONTACT_COUNT_DIFF:
        x_diff = _state_body_contact_count_diff(state, spec, env, gbase, gcount)
        tgt_diff = state.targets_flat[env, tgt_off]
        d0 = tgt_diff - x_diff
        err = _metric_geometric_scalar(d0)
    elif skid == STATE_JOINT_MECH_POWER:
        x_pwr = _state_joint_mech_power(state, spec, env, gbase, gcount)
        tgt_pwr = state.targets_flat[env, tgt_off]
        d0 = tgt_pwr - x_pwr
        err = _metric_geometric_scalar(d0)

    # (BODY_CONTACT per-body predicate not wired — variable stride = K.)

    _scatter_delta(outputs, env, canon_off, stride, instant, d0, d1, d2, d3)
    outputs.buf_error[env, slot] = err
    outputs.buf_activation[env, slot] = _apply_activation(akid, err, act_param)


@wp.kernel
def dispatch_mega(
    env_slots: EnvSlots,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
):
    """One thread per ``(env, slot)``. Early-return on padded slots.

    The shell owns gather metadata + activation + scatter. Each state kernel
    contributes one projection ``@wp.func`` and reuses the error helper that
    matches its metric. Adding a new state kernel leaves this shell
    untouched except for one new ``elif`` branch.
    """
    env, slot = wp.tid()
    if slot >= env_slots.slot_count[env]:
        return
    _dispatch_slot(env, slot, env_slots, spec, state, outputs)


@wp.func
def _dispatch_packed_pipeline_item(
    pipeline_id: int,
    env: int,
    slot: int,
    sid: int,
    tgt_off: int,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
):
    """Dispatch one fused-pipeline queue item."""
    act_param = spec.activation_kernel_param[sid]
    akid = spec.activation_kernel_id[sid]
    canon_off = spec.canonical_offset[sid]
    instant = spec.is_instant_flag[sid]
    stride = spec.state_stride[sid]
    gbase = spec.gather_offset[sid]
    gcount = spec.gather_count[sid]

    d0 = float(0.0)
    d1 = float(0.0)
    d2 = float(0.0)
    d3 = float(0.0)
    err = float(0.0)

    if pipeline_id == PIPELINE_DIRECT_VEC3_DELTA:
        x3 = _project_xyz(state, spec, env, gbase)
        t3 = _read_target_xyz(state, env, tgt_off)
        d3v = t3 - x3
        d0 = d3v[0]
        d1 = d3v[1]
        d2 = d3v[2]
        err = _metric_geometric_vec3(d3v)
    elif pipeline_id == PIPELINE_DIRECT_SCALAR_DELTA:
        d0 = state.targets_flat[env, tgt_off] - _project_scalar(state, spec, env, gbase)
        err = _metric_geometric_scalar(d0)
    elif pipeline_id == PIPELINE_DIRECT_QUAT_DELTA:
        dq = _delta_quaternion(_project_quat_xyzw(state, spec, env, gbase), _read_target_quat_xyzw(state, env, tgt_off))
        d0 = dq[0]
        d1 = dq[1]
        d2 = dq[2]
        d3 = dq[3]
        err = _metric_quaternion(dq)
    elif pipeline_id == PIPELINE_VEC3_THRESHOLD_VECTOR_DELTA:
        xcb = _state_body_contact(state, spec, env, gbase, gcount)
        tc0 = state.targets_flat[env, tgt_off]
        tc1 = float(0.0)
        tc2 = float(0.0)
        tc3 = float(0.0)
        if stride >= 2:
            tc1 = state.targets_flat[env, tgt_off + 1]
        if stride >= 3:
            tc2 = state.targets_flat[env, tgt_off + 2]
        if stride >= 4:
            tc3 = state.targets_flat[env, tgt_off + 3]
        d0 = tc0 - xcb[0]
        d1 = tc1 - xcb[1]
        d2 = tc2 - xcb[2]
        d3 = tc3 - xcb[3]
        err = wp.sqrt(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3)
    elif pipeline_id == PIPELINE_VEC3_THRESHOLD_SUM_DELTA:
        d0 = state.targets_flat[env, tgt_off] - _state_body_contact_count(state, spec, env, gbase, gcount)
        err = _metric_geometric_scalar(d0)
    elif pipeline_id == PIPELINE_VEC3_THRESHOLD_PAIR_DIFF_DELTA:
        d0 = state.targets_flat[env, tgt_off] - _state_body_contact_count_diff(state, spec, env, gbase, gcount)
        err = _metric_geometric_scalar(d0)
    elif pipeline_id == PIPELINE_SCALAR_SUM_DELTA:
        d0 = state.targets_flat[env, tgt_off] - _state_joint_mech_power(state, spec, env, gbase, gcount)
        err = _metric_geometric_scalar(d0)

    _scatter_delta(outputs, env, canon_off, stride, instant, d0, d1, d2, d3)
    outputs.buf_error[env, slot] = err
    outputs.buf_activation[env, slot] = _apply_activation(akid, err, act_param)


@wp.func
def _write_primitive_local_outputs(
    env: int,
    slot: int,
    sid: int,
    spec: SubtaskSpec,
    outputs: Outputs,
    d0: float,
    d1: float,
    d2: float,
    d3: float,
    err: float,
):
    """Write public command/debug surfaces for one primitive item."""
    act = _apply_activation(spec.activation_kernel_id[sid], err, spec.activation_kernel_param[sid])
    _scatter_delta(
        outputs,
        env,
        spec.canonical_offset[sid],
        spec.state_stride[sid],
        spec.is_instant_flag[sid],
        d0,
        d1,
        d2,
        d3,
    )
    outputs.buf_error[env, slot] = err
    outputs.buf_activation[env, slot] = act


@wp.func
def _dispatch_direct_vec3_local(
    env: int,
    slot: int,
    sid: int,
    tgt_off: int,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
):
    gbase = spec.gather_offset[sid]
    x3 = _project_xyz(state, spec, env, gbase)
    t3 = _read_target_xyz(state, env, tgt_off)
    d3v = t3 - x3
    _write_primitive_local_outputs(
        env,
        slot,
        sid,
        spec,
        outputs,
        d3v[0],
        d3v[1],
        d3v[2],
        0.0,
        _metric_geometric_vec3(d3v),
    )


@wp.func
def _dispatch_direct_scalar_local(
    env: int,
    slot: int,
    sid: int,
    tgt_off: int,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
):
    d0 = state.targets_flat[env, tgt_off] - _project_scalar(state, spec, env, spec.gather_offset[sid])
    _write_primitive_local_outputs(
        env,
        slot,
        sid,
        spec,
        outputs,
        d0,
        0.0,
        0.0,
        0.0,
        _metric_geometric_scalar(d0),
    )


@wp.func
def _dispatch_direct_quat_local(
    env: int,
    slot: int,
    sid: int,
    tgt_off: int,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
):
    gbase = spec.gather_offset[sid]
    dq = _delta_quaternion(_project_quat_xyzw(state, spec, env, gbase), _read_target_quat_xyzw(state, env, tgt_off))
    _write_primitive_local_outputs(
        env,
        slot,
        sid,
        spec,
        outputs,
        dq[0],
        dq[1],
        dq[2],
        dq[3],
        _metric_quaternion(dq),
    )


@wp.func
def _dispatch_vec3_threshold_vector_local(
    env: int,
    slot: int,
    sid: int,
    tgt_off: int,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
):
    stride = spec.state_stride[sid]
    xcb = _state_body_contact(state, spec, env, spec.gather_offset[sid], spec.gather_count[sid])
    tc0 = state.targets_flat[env, tgt_off]
    tc1 = float(0.0)
    tc2 = float(0.0)
    tc3 = float(0.0)
    if stride >= 2:
        tc1 = state.targets_flat[env, tgt_off + 1]
    if stride >= 3:
        tc2 = state.targets_flat[env, tgt_off + 2]
    if stride >= 4:
        tc3 = state.targets_flat[env, tgt_off + 3]
    d0 = tc0 - xcb[0]
    d1 = tc1 - xcb[1]
    d2 = tc2 - xcb[2]
    d3 = tc3 - xcb[3]
    _write_primitive_local_outputs(
        env,
        slot,
        sid,
        spec,
        outputs,
        d0,
        d1,
        d2,
        d3,
        wp.sqrt(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3),
    )


@wp.func
def _contact_mask_vec4(mask: wp.array2d(dtype=float), node_id: int) -> wp.vec4:
    return wp.vec4(mask[node_id, 0], mask[node_id, 1], mask[node_id, 2], mask[node_id, 3])


@wp.kernel
def compute_dense_graph_producers(
    vec3_nodes: PrimitiveProducerQueue,
    scalar_nodes: PrimitiveProducerQueue,
    quat_nodes: PrimitiveProducerQueue,
    scalar_sum_nodes: PrimitiveProducerQueue,
    contact_nodes: PrimitiveProducerQueue,
    spec: SubtaskSpec,
    state: StateAccess,
    direct_vec3: wp.array2d(dtype=float),
    direct_scalar: wp.array(dtype=float),
    direct_quat: wp.array2d(dtype=float),
    scalar_sum: wp.array(dtype=float),
    contact_mask: wp.array2d(dtype=float),
    vec3_signature_count: int,
    scalar_signature_count: int,
    quat_signature_count: int,
    scalar_sum_signature_count: int,
    contact_signature_count: int,
):
    """Compute all dense producer rows in one signature-grouped launch."""
    env, signature = wp.tid()

    cursor = int(0)
    if signature < vec3_signature_count:
        sid = vec3_nodes.signature_subtask[signature]
        node_id = env * vec3_signature_count + signature
        x3 = _project_xyz(state, spec, env, spec.gather_offset[sid])
        direct_vec3[node_id, 0] = x3[0]
        direct_vec3[node_id, 1] = x3[1]
        direct_vec3[node_id, 2] = x3[2]
        return

    cursor = cursor + vec3_signature_count
    if signature < cursor + scalar_signature_count:
        local_signature = signature - cursor
        sid = scalar_nodes.signature_subtask[local_signature]
        direct_scalar[env * scalar_signature_count + local_signature] = _project_scalar(
            state, spec, env, spec.gather_offset[sid]
        )
        return

    cursor = cursor + scalar_signature_count
    if signature < cursor + quat_signature_count:
        local_signature = signature - cursor
        sid = quat_nodes.signature_subtask[local_signature]
        node_id = env * quat_signature_count + local_signature
        q = _project_quat_xyzw(state, spec, env, spec.gather_offset[sid])
        direct_quat[node_id, 0] = q[0]
        direct_quat[node_id, 1] = q[1]
        direct_quat[node_id, 2] = q[2]
        direct_quat[node_id, 3] = q[3]
        return

    cursor = cursor + quat_signature_count
    if signature < cursor + scalar_sum_signature_count:
        local_signature = signature - cursor
        sid = scalar_sum_nodes.signature_subtask[local_signature]
        scalar_sum[env * scalar_sum_signature_count + local_signature] = _state_joint_mech_power(
            state, spec, env, spec.gather_offset[sid], spec.gather_count[sid]
        )
        return

    cursor = cursor + scalar_sum_signature_count
    if signature < cursor + contact_signature_count:
        local_signature = signature - cursor
        sid = contact_nodes.signature_subtask[local_signature]
        node_id = env * contact_signature_count + local_signature
        xcb = _state_body_contact(state, spec, env, spec.gather_offset[sid], spec.gather_count[sid])
        contact_mask[node_id, 0] = xcb[0]
        contact_mask[node_id, 1] = xcb[1]
        contact_mask[node_id, 2] = xcb[2]
        contact_mask[node_id, 3] = xcb[3]


@wp.func
def _dispatch_vec3_threshold_sum_local(
    env: int,
    slot: int,
    sid: int,
    tgt_off: int,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
):
    d0 = state.targets_flat[env, tgt_off] - _state_body_contact_count(
        state, spec, env, spec.gather_offset[sid], spec.gather_count[sid]
    )
    _write_primitive_local_outputs(
        env,
        slot,
        sid,
        spec,
        outputs,
        d0,
        0.0,
        0.0,
        0.0,
        _metric_geometric_scalar(d0),
    )


@wp.func
def _dispatch_contact_vector_from_mask_local(
    env: int,
    slot: int,
    sid: int,
    tgt_off: int,
    node_id: int,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
    contact_mask: wp.array2d(dtype=float),
):
    stride = spec.state_stride[sid]
    xcb = _contact_mask_vec4(contact_mask, node_id)
    tc0 = state.targets_flat[env, tgt_off]
    tc1 = float(0.0)
    tc2 = float(0.0)
    tc3 = float(0.0)
    if stride >= 2:
        tc1 = state.targets_flat[env, tgt_off + 1]
    if stride >= 3:
        tc2 = state.targets_flat[env, tgt_off + 2]
    if stride >= 4:
        tc3 = state.targets_flat[env, tgt_off + 3]
    d0 = tc0 - xcb[0]
    d1 = tc1 - xcb[1]
    d2 = tc2 - xcb[2]
    d3 = tc3 - xcb[3]
    _write_primitive_local_outputs(
        env,
        slot,
        sid,
        spec,
        outputs,
        d0,
        d1,
        d2,
        d3,
        wp.sqrt(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3),
    )


@wp.func
def _dispatch_contact_sum_from_mask_local(
    env: int,
    slot: int,
    sid: int,
    tgt_off: int,
    node_id: int,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
    contact_mask: wp.array2d(dtype=float),
):
    count = spec.gather_count[sid] / 3
    xcb = _contact_mask_vec4(contact_mask, node_id)
    x_cnt = float(0.0)
    for i in range(count):
        x_cnt = x_cnt + xcb[i]
    d0 = state.targets_flat[env, tgt_off] - x_cnt
    _write_primitive_local_outputs(
        env,
        slot,
        sid,
        spec,
        outputs,
        d0,
        0.0,
        0.0,
        0.0,
        _metric_geometric_scalar(d0),
    )


@wp.func
def _dispatch_contact_pair_diff_from_mask_local(
    env: int,
    slot: int,
    sid: int,
    tgt_off: int,
    node_id: int,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
    contact_mask: wp.array2d(dtype=float),
):
    count = spec.gather_count[sid] / 3
    half = count / 2
    xcb = _contact_mask_vec4(contact_mask, node_id)
    cnt_a = float(0.0)
    cnt_b = float(0.0)
    for i in range(count):
        if i < half:
            cnt_a = cnt_a + xcb[i]
        else:
            cnt_b = cnt_b + xcb[i]
    d0 = state.targets_flat[env, tgt_off] - (cnt_a - cnt_b)
    _write_primitive_local_outputs(
        env,
        slot,
        sid,
        spec,
        outputs,
        d0,
        0.0,
        0.0,
        0.0,
        _metric_geometric_scalar(d0),
    )


@wp.func
def _dispatch_vec3_threshold_pair_diff_local(
    env: int,
    slot: int,
    sid: int,
    tgt_off: int,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
):
    d0 = state.targets_flat[env, tgt_off] - _state_body_contact_count_diff(
        state, spec, env, spec.gather_offset[sid], spec.gather_count[sid]
    )
    _write_primitive_local_outputs(
        env,
        slot,
        sid,
        spec,
        outputs,
        d0,
        0.0,
        0.0,
        0.0,
        _metric_geometric_scalar(d0),
    )


@wp.func
def _dispatch_scalar_sum_local(
    env: int,
    slot: int,
    sid: int,
    tgt_off: int,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
):
    d0 = state.targets_flat[env, tgt_off] - _state_joint_mech_power(
        state, spec, env, spec.gather_offset[sid], spec.gather_count[sid]
    )
    _write_primitive_local_outputs(
        env,
        slot,
        sid,
        spec,
        outputs,
        d0,
        0.0,
        0.0,
        0.0,
        _metric_geometric_scalar(d0),
    )


@wp.kernel
def dispatch_primitive_local_direct_vec3(
    queue: PrimitiveLocalQueue,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
):
    q = wp.tid()
    if q >= queue.schedule_counts[PIPELINE_DIRECT_VEC3_DELTA]:
        return
    index = queue.schedule_offsets[PIPELINE_DIRECT_VEC3_DELTA] + q
    _dispatch_direct_vec3_local(
        queue.env_ids[index],
        queue.slot_ids[index],
        queue.subtask_ids[index],
        queue.target_offsets[index],
        spec,
        state,
        outputs,
    )


@wp.kernel
def dispatch_primitive_local_direct_scalar(
    queue: PrimitiveLocalQueue,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
):
    q = wp.tid()
    if q >= queue.schedule_counts[PIPELINE_DIRECT_SCALAR_DELTA]:
        return
    index = queue.schedule_offsets[PIPELINE_DIRECT_SCALAR_DELTA] + q
    _dispatch_direct_scalar_local(
        queue.env_ids[index],
        queue.slot_ids[index],
        queue.subtask_ids[index],
        queue.target_offsets[index],
        spec,
        state,
        outputs,
    )


@wp.kernel
def dispatch_primitive_local_direct_quat(
    queue: PrimitiveLocalQueue,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
):
    q = wp.tid()
    if q >= queue.schedule_counts[PIPELINE_DIRECT_QUAT_DELTA]:
        return
    index = queue.schedule_offsets[PIPELINE_DIRECT_QUAT_DELTA] + q
    _dispatch_direct_quat_local(
        queue.env_ids[index],
        queue.slot_ids[index],
        queue.subtask_ids[index],
        queue.target_offsets[index],
        spec,
        state,
        outputs,
    )


@wp.kernel
def dispatch_primitive_local_vec3_threshold_vector(
    queue: PrimitiveLocalQueue,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
):
    q = wp.tid()
    if q >= queue.schedule_counts[PIPELINE_VEC3_THRESHOLD_VECTOR_DELTA]:
        return
    index = queue.schedule_offsets[PIPELINE_VEC3_THRESHOLD_VECTOR_DELTA] + q
    _dispatch_vec3_threshold_vector_local(
        queue.env_ids[index],
        queue.slot_ids[index],
        queue.subtask_ids[index],
        queue.target_offsets[index],
        spec,
        state,
        outputs,
    )


@wp.kernel
def dispatch_primitive_local_vec3_threshold_sum(
    queue: PrimitiveLocalQueue,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
):
    q = wp.tid()
    if q >= queue.schedule_counts[PIPELINE_VEC3_THRESHOLD_SUM_DELTA]:
        return
    index = queue.schedule_offsets[PIPELINE_VEC3_THRESHOLD_SUM_DELTA] + q
    _dispatch_vec3_threshold_sum_local(
        queue.env_ids[index],
        queue.slot_ids[index],
        queue.subtask_ids[index],
        queue.target_offsets[index],
        spec,
        state,
        outputs,
    )


@wp.kernel
def dispatch_primitive_local_vec3_threshold_pair_diff(
    queue: PrimitiveLocalQueue,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
):
    q = wp.tid()
    if q >= queue.schedule_counts[PIPELINE_VEC3_THRESHOLD_PAIR_DIFF_DELTA]:
        return
    index = queue.schedule_offsets[PIPELINE_VEC3_THRESHOLD_PAIR_DIFF_DELTA] + q
    _dispatch_vec3_threshold_pair_diff_local(
        queue.env_ids[index],
        queue.slot_ids[index],
        queue.subtask_ids[index],
        queue.target_offsets[index],
        spec,
        state,
        outputs,
    )


@wp.kernel
def dispatch_primitive_local_scalar_sum(
    queue: PrimitiveLocalQueue,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
):
    q = wp.tid()
    if q >= queue.schedule_counts[PIPELINE_SCALAR_SUM_DELTA]:
        return
    index = queue.schedule_offsets[PIPELINE_SCALAR_SUM_DELTA] + q
    _dispatch_scalar_sum_local(
        queue.env_ids[index],
        queue.slot_ids[index],
        queue.subtask_ids[index],
        queue.target_offsets[index],
        spec,
        state,
        outputs,
    )


@wp.kernel
def dispatch_graph_dense(
    env_slots: EnvSlots,
    subtask_schedule_ids: wp.array(dtype=int),
    vec3_nodes: PrimitiveProducerQueue,
    scalar_nodes: PrimitiveProducerQueue,
    quat_nodes: PrimitiveProducerQueue,
    scalar_sum_nodes: PrimitiveProducerQueue,
    contact_nodes: PrimitiveProducerQueue,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
    direct_vec3: wp.array2d(dtype=float),
    direct_scalar: wp.array(dtype=float),
    direct_quat: wp.array2d(dtype=float),
    scalar_sum: wp.array(dtype=float),
    contact_mask: wp.array2d(dtype=float),
    vec3_signature_count: int,
    scalar_signature_count: int,
    quat_signature_count: int,
    scalar_sum_signature_count: int,
    contact_signature_count: int,
):
    """Consume dense graph producers in schedule-ordered ``[env, slot]`` layout."""
    env, slot = wp.tid()
    if slot >= env_slots.slot_count[env]:
        return

    sid = env_slots.subtask_ids[env, slot]
    tgt_off = env_slots.slot_offsets[env, slot]
    pipeline_id = subtask_schedule_ids[sid]

    if pipeline_id == PIPELINE_DIRECT_VEC3_DELTA:
        signature = vec3_nodes.subtask_signature[sid]
        node_id = env * vec3_signature_count + signature
        x3 = wp.vec3(direct_vec3[node_id, 0], direct_vec3[node_id, 1], direct_vec3[node_id, 2])
        d3v = _read_target_xyz(state, env, tgt_off) - x3
        _write_primitive_local_outputs(
            env,
            slot,
            sid,
            spec,
            outputs,
            d3v[0],
            d3v[1],
            d3v[2],
            0.0,
            _metric_geometric_vec3(d3v),
        )
    elif pipeline_id == PIPELINE_DIRECT_SCALAR_DELTA:
        signature = scalar_nodes.subtask_signature[sid]
        node_id = env * scalar_signature_count + signature
        d0 = state.targets_flat[env, tgt_off] - direct_scalar[node_id]
        _write_primitive_local_outputs(
            env,
            slot,
            sid,
            spec,
            outputs,
            d0,
            0.0,
            0.0,
            0.0,
            _metric_geometric_scalar(d0),
        )
    elif pipeline_id == PIPELINE_DIRECT_QUAT_DELTA:
        signature = quat_nodes.subtask_signature[sid]
        node_id = env * quat_signature_count + signature
        q_current = wp.vec4(
            direct_quat[node_id, 0],
            direct_quat[node_id, 1],
            direct_quat[node_id, 2],
            direct_quat[node_id, 3],
        )
        dq = _delta_quaternion(q_current, _read_target_quat_xyzw(state, env, tgt_off))
        _write_primitive_local_outputs(
            env,
            slot,
            sid,
            spec,
            outputs,
            dq[0],
            dq[1],
            dq[2],
            dq[3],
            _metric_quaternion(dq),
        )
    elif pipeline_id == PIPELINE_VEC3_THRESHOLD_VECTOR_DELTA:
        signature = contact_nodes.subtask_signature[sid]
        _dispatch_contact_vector_from_mask_local(
            env,
            slot,
            sid,
            tgt_off,
            env * contact_signature_count + signature,
            spec,
            state,
            outputs,
            contact_mask,
        )
    elif pipeline_id == PIPELINE_VEC3_THRESHOLD_SUM_DELTA:
        signature = contact_nodes.subtask_signature[sid]
        _dispatch_contact_sum_from_mask_local(
            env,
            slot,
            sid,
            tgt_off,
            env * contact_signature_count + signature,
            spec,
            state,
            outputs,
            contact_mask,
        )
    elif pipeline_id == PIPELINE_VEC3_THRESHOLD_PAIR_DIFF_DELTA:
        signature = contact_nodes.subtask_signature[sid]
        _dispatch_contact_pair_diff_from_mask_local(
            env,
            slot,
            sid,
            tgt_off,
            env * contact_signature_count + signature,
            spec,
            state,
            outputs,
            contact_mask,
        )
    elif pipeline_id == PIPELINE_SCALAR_SUM_DELTA:
        signature = scalar_sum_nodes.subtask_signature[sid]
        node_id = env * scalar_sum_signature_count + signature
        d0 = state.targets_flat[env, tgt_off] - scalar_sum[node_id]
        _write_primitive_local_outputs(
            env,
            slot,
            sid,
            spec,
            outputs,
            d0,
            0.0,
            0.0,
            0.0,
            _metric_geometric_scalar(d0),
        )


@wp.kernel
def dispatch_packed_scatter_flat(
    queue: PackedScatterQueue,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
):
    """Dispatch a flat fused-pipeline-sorted packed queue in one launch."""
    index = wp.tid()
    if index >= queue.count[0]:
        return
    _dispatch_packed_pipeline_item(
        queue.pipeline_ids[index],
        queue.env_ids[index],
        queue.slot_ids[index],
        queue.subtask_ids[index],
        queue.target_offsets[index],
        spec,
        state,
        outputs,
    )


@wp.kernel
def rotate_canonical_vec3_pair(
    root_quat_w: wp.array(dtype=wp.quat),
    command_reach: wp.array2d(dtype=float),
    reach_offsets: wp.array(dtype=int),
    num_reach_offsets: int,
    command_track: wp.array2d(dtype=float),
    track_offsets: wp.array(dtype=int),
):
    """Rotate reach and track vec3 command slots for one root asset."""
    env, offset_id = wp.tid()
    qq = root_quat_w[env]
    q = wp.vec4(qq[0], qq[1], qq[2], qq[3])
    if offset_id < num_reach_offsets:
        off = reach_offsets[offset_id]
        v = wp.vec3(command_reach[env, off], command_reach[env, off + 1], command_reach[env, off + 2])
        out = _quat_apply_inverse_xyzw(q, v)
        command_reach[env, off] = out[0]
        command_reach[env, off + 1] = out[1]
        command_reach[env, off + 2] = out[2]
    else:
        off = track_offsets[offset_id - num_reach_offsets]
        v = wp.vec3(command_track[env, off], command_track[env, off + 1], command_track[env, off + 2])
        out = _quat_apply_inverse_xyzw(q, v)
        command_track[env, off] = out[0]
        command_track[env, off + 1] = out[1]
        command_track[env, off + 2] = out[2]


# ---------------------------------------------------------------------------
# Composer kernel — multiplicative terminal reward.
#
# Mirrors :func:`~.reward_composer.multiplicative_terminal_reward`. One
# thread per env; the thread iterates over its env's active slots, folds
# them into running accumulators, and writes reward + success + progress.
# Unlike the PyTorch reference path:
#
# - ``sum_activation`` and ``transit_steps`` are updated in place (no
#   separate ``.add_()`` step).
# - ``instant_achieved`` is latched in place (no ``.copy_()`` after).
# - Progress is folded in (no separate PyTorch ``sum / div`` pair).
#
# Net effect: eliminates every PyTorch op the composer used to do, at the
# cost of one more ``wp.launch`` per step (still cheap — ≈3 µs).
# ---------------------------------------------------------------------------


@wp.kernel
def compose_reward(
    env_slots: EnvSlots,
    spec: SubtaskSpec,
    composer_state: ComposerState,
    outputs: Outputs,
    episode_length_buf: wp.array(dtype=wp.int64),
    effective_max_episode_length: wp.array(dtype=int),
    instant_threshold: float,
    quality_easing: float,
):
    """One thread per env. Advances composer state + writes terminal outputs.

    Semantics match :func:`~.reward_composer.multiplicative_terminal_reward`:

    - Latch each instant slot's achievement once ``activation > threshold``.
    - ``success = (all active-instant slots achieved) AND (has any instant slot)``.
    - ``quality_factor = (∏_{k ∈ tracking ∪ safety} mean_t A_k(t)) ^ quality_easing``
      — eased product over tracking and safety slots; vacuously 1 when neither.
    - ``terminal_value = gate · quality_factor``.
    - ``reward = terminal_value`` on success-or-timeout, else 0.

    Slots ``>= env_slots.slot_count[env]`` are padded and skipped — their
    ``instant_achieved`` entries retain whatever the last resample wrote
    (zero after reset), which is safe because they never contribute to
    any reduction here.
    """
    env = wp.tid()
    n_slots = env_slots.slot_count[env]

    # Advance transit_steps in place — done once per env, not per slot.
    composer_state.transit_steps[env] = composer_state.transit_steps[env] + 1
    tsteps = float(composer_state.transit_steps[env])

    # Per-slot reductions. Warp requires explicit ``int(…)`` / ``float(…)``
    # wrappers to declare these as mutable locals (rather than compile-time
    # constants) so they can be updated inside the dynamic for-loop.
    all_instant_ok_int = int(1)  # 1 = all ok so far; 0 once any instant slot is missed
    has_instant_int = int(0)
    activation_sum = float(0.0)  # for progress
    # Quality: per-env product of per-quality-slot (tracking ∪ safety)
    # transit means, raised to ``quality_easing`` at the end. Vacuous 1
    # when no quality subtasks (multiplicative identity, ``1^easing = 1``).
    quality_product = float(1.0)

    for slot in range(n_slots):
        sid = env_slots.subtask_ids[env, slot]
        is_instant = spec.is_instant_flag[sid]
        is_tracking = spec.is_tracking_flag[sid]
        act = outputs.buf_activation[env, slot]
        activation_sum = activation_sum + act

        # Update sum_activation in place (both branches use it).
        new_sum = composer_state.sum_activation[env, slot] + act
        composer_state.sum_activation[env, slot] = new_sum

        if is_instant != 0:
            has_instant_int = 1
            prev_achieved = composer_state.instant_achieved[env, slot]
            achieved_int = int(0)
            if prev_achieved:
                achieved_int = 1
            if act > instant_threshold:
                achieved_int = 1
            composer_state.instant_achieved[env, slot] = achieved_int != 0
            if achieved_int == 0:
                all_instant_ok_int = 0
        elif is_tracking != 0:
            # Tracking subtasks (including soft-safety constraints) are
            # the unified quality dimension: per-slot transit-mean,
            # multiplied into the per-env product. Empty case (no
            # tracking slots) leaves ``quality_product = 1`` (multiplicative
            # identity), so ``1^easing = 1`` correctly handles pure-instant.
            quality_product = quality_product * (new_sum / tsteps)

    # Success = all instant slots achieved AND the task has at least one
    # instant slot. Pure-tracking tasks never satisfy the second clause,
    # so they never fire success — only timeout.
    success_int = all_instant_ok_int * has_instant_int

    # Timeout check — per-env effective length to honor the adaptive
    # tracking-episode-length curriculum. Reads from
    # ``effective_max_episode_length[env]`` (int32 array) so a pure-tracking
    # env on a randomized short window latches at its own ``len-1`` step,
    # not at the global cap. Cast to int64 to match the buffer dtype.
    # We compare against ``len - 1`` because ``_update_command`` runs at the
    # end of env.step AFTER reward read AND AFTER ``_reset_idx`` zeroes
    # ``episode_length_buf`` for timed-out envs. Latching the terminal value
    # at ``buf == len - 1`` guarantees the reward read at the outer terminal
    # step (``buf == len``) sees it. See the matching guard in
    # :class:`~.multi_task_command_torch.MultiTaskCommandTorch`.
    timeout_int = int(0)
    if episode_length_buf[env] >= wp.int64(effective_max_episode_length[env] - 1):
        timeout_int = 1
    done_now_int = success_int
    if timeout_int == 1:
        done_now_int = 1

    # Quality factor: ``(∏ quality_means) ^ quality_easing``. Vacuous 1 for
    # pure-instant tasks (no tracking, no safety). When ``has_quality_int == 0``
    # the product stayed at the multiplicative identity 1, so ``1^easing = 1``
    # is correct without a special case.
    quality_factor = wp.pow(quality_product, quality_easing)

    # Gate = 1 iff all instant slots achieved (vacuous for pure-tracking —
    # ``all_instant_ok_int`` stays 1 because the loop doesn't touch it).
    terminal_value = float(all_instant_ok_int) * quality_factor

    reward = float(0.0)
    if done_now_int == 1:
        reward = terminal_value

    outputs.task_reward[env] = reward
    outputs.task_done_success[env] = success_int != 0

    # Progress — mean of active-slot activations. Empty slot_count → 0.
    progress_val = float(0.0)
    if n_slots > 0:
        progress_val = activation_sum / float(n_slots)
    outputs.progress[env] = progress_val


# ---------------------------------------------------------------------------
# Parallel composer kernel — block-per-env with cooperative slot reductions.
#
# The serial compose loop is the dominant cost at high ``k_max`` (≈ 60 % of
# the future-synthetic step). This variant launches one block per env with
# ``block_dim = k_max`` so each thread handles exactly one slot's per-slot
# state update. Cross-slot reductions go through Warp tile primitives
# (``tile_sum`` / ``tile_min`` / ``tile_max``); the multiplicative quality
# factor uses ``log → sum → exp`` to keep the math numerically well-behaved
# when ``k_max`` is large (288+ slots). Thread 0 writes the per-env outputs.
#
# Threads with ``slot >= n_slots`` contribute neutral values to each
# reduction (0 for sums, 1 for AND-style mins, 0 for max-of-flags, log(1)=0
# for the log-sum-exp), so the reduction result equals the original serial
# loop's output.
# ---------------------------------------------------------------------------


@wp.func
def _compose_reduce_op(a: wp.vec4, b: wp.vec4) -> wp.vec4:
    """Combined reduction op for the parallel composer.

    Component-wise: sum, max, min, multiply. Collapses the four per-slot
    reductions (``activation_sum``, ``has_instant``, ``all_instant_ok``,
    ``quality_product``) into one cooperative pass. Multiplying the quality
    ratios directly is mathematically equivalent to exp(sum(log(...))) but
    avoids per-slot ``wp.log`` and the per-env ``wp.exp``; underflow to 0
    matches the behavior of ``log → -inf → exp → 0``.
    """
    return wp.vec4(a[0] + b[0], wp.max(a[1], b[1]), wp.min(a[2], b[2]), a[3] * b[3])


@wp.kernel
def compose_reward_parallel(
    env_slots: EnvSlots,
    spec: SubtaskSpec,
    composer_state: ComposerState,
    outputs: Outputs,
    episode_length_buf: wp.array(dtype=wp.int64),
    effective_max_episode_length: wp.array(dtype=int),
    instant_threshold: float,
    quality_easing: float,
):
    """Block-per-env composer. Same outputs as :func:`compose_reward`.

    Launch as ``wp.launch_tiled(compose_reward_parallel, dim=[num_envs],
    block_dim=k_max)``. ``block_dim`` must be ≥ the spec's ``k_max`` so every
    active slot has a thread; threads beyond ``slot_count[env]`` short-circuit
    to neutral reduction values.
    """
    env, slot = wp.tid()
    n_slots = env_slots.slot_count[env]

    # All threads load the new transit_steps; only thread 0 writes back.
    old_transit = composer_state.transit_steps[env]
    new_transit = old_transit + 1
    tsteps = float(new_transit)

    # Neutral defaults so threads beyond ``n_slots`` don't perturb reductions.
    local_act = float(0.0)
    local_has_instant = int(0)
    local_instant_ok = int(1)  # min reduction → 0 only if some instant slot failed
    local_quality = float(1.0)  # multiplicative identity for tile_reduce(mul)

    if slot < n_slots:
        sid = env_slots.subtask_ids[env, slot]
        is_instant = spec.is_instant_flag[sid]
        is_tracking = spec.is_tracking_flag[sid]
        act = outputs.buf_activation[env, slot]
        local_act = act

        # Per-slot ``sum_activation`` update (parallel; no contention).
        new_sum = composer_state.sum_activation[env, slot] + act
        composer_state.sum_activation[env, slot] = new_sum

        if is_instant != 0:
            local_has_instant = 1
            prev_achieved = composer_state.instant_achieved[env, slot]
            achieved_int = int(0)
            if prev_achieved:
                achieved_int = 1
            if act > instant_threshold:
                achieved_int = 1
            composer_state.instant_achieved[env, slot] = achieved_int != 0
            local_instant_ok = achieved_int
        elif is_tracking != 0:
            # Multiplicative quality factor: feed the per-slot ratio directly
            # into the product reduction. Underflow to 0 across many slots is
            # the desired semantic — matches the original log-sum-exp path.
            local_quality = new_sum / tsteps

    # Pack the four per-slot reduction inputs into a single vec4 and reduce
    # in one cooperative pass. Component-wise op: sum / max / min / multiply.
    combined = wp.vec4(local_act, float(local_has_instant), float(local_instant_ok), local_quality)
    reduced_t = wp.tile_reduce(_compose_reduce_op, wp.tile(combined, preserve_type=True))

    # Thread 0 finalizes per-env outputs.
    if slot == 0:
        composer_state.transit_steps[env] = new_transit

        reduced = wp.tile_extract(reduced_t, 0)
        activation_sum = reduced[0]
        has_instant = int(reduced[1])
        all_instant_ok = int(reduced[2])
        quality_product = reduced[3]

        success_int = all_instant_ok * has_instant
        timeout_int = int(0)
        if episode_length_buf[env] >= wp.int64(effective_max_episode_length[env] - 1):
            timeout_int = 1
        done_now_int = success_int
        if timeout_int == 1:
            done_now_int = 1

        # quality_factor = quality_product ^ quality_easing.
        # ``wp.pow(0, easing) == 0`` for easing > 0, so an underflowed
        # product still gives a 0 reward as expected.
        quality_factor = wp.pow(quality_product, quality_easing)
        terminal_value = float(all_instant_ok) * quality_factor

        reward = float(0.0)
        if done_now_int == 1:
            reward = terminal_value

        outputs.task_reward[env] = reward
        outputs.task_done_success[env] = success_int != 0

        progress_val = float(0.0)
        if n_slots > 0:
            progress_val = activation_sum / float(n_slots)
        outputs.progress[env] = progress_val


# ---------------------------------------------------------------------------
# Fused dispatch + compose kernel — block-per-env, only worth it at high k_max.
#
# The split between :func:`dispatch_mega` and :func:`compose_reward_parallel`
# requires a global memory roundtrip on ``outputs.buf_activation``: dispatch
# writes per-slot activation, compose reads it back. At high ``k_max`` the
# bandwidth and latency of that roundtrip is real (~20 µs on future_synthetic
# 16k envs).
#
# Fusing the two into one block-per-env kernel keeps the activation in
# registers between the two phases. Writes to ``outputs.buf_activation`` and
# ``outputs.buf_error`` are still issued for compatibility with metrics and
# tests, but compose's reduction reads from the in-register value directly
# instead of going back through global memory.
#
# Threshold: same as the parallel composer — only enabled when ``k_max`` is
# large enough to fill warps with one block per env. Backends select between
# the (dispatch_mega + parallel compose) and the fused path at construction
# time based on ``compose_select.use_parallel_compose``.
# ---------------------------------------------------------------------------


@wp.kernel
def dispatch_compose_fused(
    env_slots: EnvSlots,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
    composer_state: ComposerState,
    episode_length_buf: wp.array(dtype=wp.int64),
    effective_max_episode_length: wp.array(dtype=int),
    instant_threshold: float,
    quality_easing: float,
    inline_root_quat: wp.array(dtype=wp.quat),
    subtask_is_rotatable: wp.array(dtype=int),
    use_inline_rotation: int,
):
    """Fused dispatch + compose. Same outputs as ``dispatch_mega`` + ``compose_reward_parallel``.

    When ``use_inline_rotation != 0``, vec3 deltas in the BODY_POS / BODY_LIN_VEL /
    BODY_ANG_VEL branch are rotated to body frame in-register before scatter,
    using ``inline_root_quat[env]``. Saves the standalone
    ``rotate_canonical_vec3_pair`` launch in the captured graph.
    """
    env, slot = wp.tid()
    n_slots = env_slots.slot_count[env]

    old_transit = composer_state.transit_steps[env]
    new_transit = old_transit + 1
    tsteps = float(new_transit)

    local_act = float(0.0)
    local_has_instant = int(0)
    local_instant_ok = int(1)
    local_quality = float(1.0)

    if slot < n_slots:
        # ---- Dispatch phase ----
        sid = env_slots.subtask_ids[env, slot]
        skid = spec.state_kernel_id[sid]
        akid = spec.activation_kernel_id[sid]
        act_param = spec.activation_kernel_param[sid]
        canon_off = spec.canonical_offset[sid]
        instant = spec.is_instant_flag[sid]
        is_tracking_i = spec.is_tracking_flag[sid]
        tgt_off = env_slots.slot_offsets[env, slot]
        stride = spec.state_stride[sid]
        gbase = spec.gather_offset[sid]
        gcount = spec.gather_count[sid]

        d0 = float(0.0)
        d1 = float(0.0)
        d2 = float(0.0)
        d3 = float(0.0)
        err = float(0.0)

        if skid == STATE_BODY_POS or skid == STATE_BODY_LIN_VEL or skid == STATE_BODY_ANG_VEL:
            x3 = _project_xyz(state, spec, env, gbase)
            t3 = _read_target_xyz(state, env, tgt_off)
            d3v = t3 - x3
            # Optional inline body-frame rotation for rotatable vec3 outputs.
            # ``err`` uses L2 norm which is rotation-invariant — equivalent
            # output as the standalone ``rotate_canonical_vec3_pair`` kernel.
            if use_inline_rotation != 0 and subtask_is_rotatable[sid] != 0:
                qq = inline_root_quat[env]
                q = wp.vec4(qq[0], qq[1], qq[2], qq[3])
                d3v = _quat_apply_inverse_xyzw(q, d3v)
            d0 = d3v[0]
            d1 = d3v[1]
            d2 = d3v[2]
            err = _metric_geometric_vec3(d3v)
        elif skid == STATE_JOINT_POS or skid == STATE_JOINT_VEL or skid == STATE_BODY_POS_Z:
            x_s = _project_scalar(state, spec, env, gbase)
            t_s = state.targets_flat[env, tgt_off]
            d0 = t_s - x_s
            err = _metric_geometric_scalar(d0)
        elif skid == STATE_BODY_QUAT:
            cq = _project_quat_xyzw(state, spec, env, gbase)
            tq = _read_target_quat_xyzw(state, env, tgt_off)
            dq = _delta_quaternion(cq, tq)
            d0 = dq[0]
            d1 = dq[1]
            d2 = dq[2]
            d3 = dq[3]
            err = _metric_quaternion(dq)
        elif skid == STATE_BODY_CONTACT:
            xcb = _state_body_contact(state, spec, env, gbase, gcount)
            tc0 = state.targets_flat[env, tgt_off]
            tc1 = float(0.0)
            tc2 = float(0.0)
            tc3 = float(0.0)
            if stride >= 2:
                tc1 = state.targets_flat[env, tgt_off + 1]
            if stride >= 3:
                tc2 = state.targets_flat[env, tgt_off + 2]
            if stride >= 4:
                tc3 = state.targets_flat[env, tgt_off + 3]
            d0 = tc0 - xcb[0]
            d1 = tc1 - xcb[1]
            d2 = tc2 - xcb[2]
            d3 = tc3 - xcb[3]
            err = wp.sqrt(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3)
        elif skid == STATE_BODY_CONTACT_COUNT:
            x_cnt = _state_body_contact_count(state, spec, env, gbase, gcount)
            tgt_cnt = state.targets_flat[env, tgt_off]
            d0 = tgt_cnt - x_cnt
            err = _metric_geometric_scalar(d0)
        elif skid == STATE_BODY_CONTACT_COUNT_DIFF:
            x_diff = _state_body_contact_count_diff(state, spec, env, gbase, gcount)
            tgt_diff = state.targets_flat[env, tgt_off]
            d0 = tgt_diff - x_diff
            err = _metric_geometric_scalar(d0)
        elif skid == STATE_JOINT_MECH_POWER:
            x_pwr = _state_joint_mech_power(state, spec, env, gbase, gcount)
            tgt_pwr = state.targets_flat[env, tgt_off]
            d0 = tgt_pwr - x_pwr
            err = _metric_geometric_scalar(d0)

        # Activation in register — used by both the buf_activation write AND
        # the compose reduction, no roundtrip needed.
        act = _apply_activation(akid, err, act_param)
        local_act = act

        # Dense output writes (kept for tests/metrics compatibility).
        _scatter_delta(outputs, env, canon_off, stride, instant, d0, d1, d2, d3)
        outputs.buf_error[env, slot] = err
        outputs.buf_activation[env, slot] = act

        # ---- Compose phase ----
        new_sum = composer_state.sum_activation[env, slot] + act
        composer_state.sum_activation[env, slot] = new_sum

        if instant != 0:
            local_has_instant = 1
            prev_achieved = composer_state.instant_achieved[env, slot]
            achieved_int = int(0)
            if prev_achieved:
                achieved_int = 1
            if act > instant_threshold:
                achieved_int = 1
            composer_state.instant_achieved[env, slot] = achieved_int != 0
            local_instant_ok = achieved_int
        elif is_tracking_i != 0:
            local_quality = new_sum / tsteps

    combined = wp.vec4(local_act, float(local_has_instant), float(local_instant_ok), local_quality)
    reduced_t = wp.tile_reduce(_compose_reduce_op, wp.tile(combined, preserve_type=True))

    if slot == 0:
        composer_state.transit_steps[env] = new_transit

        reduced = wp.tile_extract(reduced_t, 0)
        activation_sum = reduced[0]
        has_instant = int(reduced[1])
        all_instant_ok = int(reduced[2])
        quality_product = reduced[3]

        success_int = all_instant_ok * has_instant
        timeout_int = int(0)
        if episode_length_buf[env] >= wp.int64(effective_max_episode_length[env] - 1):
            timeout_int = 1
        done_now_int = success_int
        if timeout_int == 1:
            done_now_int = 1

        quality_factor = wp.pow(quality_product, quality_easing)
        terminal_value = float(all_instant_ok) * quality_factor

        reward = float(0.0)
        if done_now_int == 1:
            reward = terminal_value

        outputs.task_reward[env] = reward
        outputs.task_done_success[env] = success_int != 0

        progress_val = float(0.0)
        if n_slots > 0:
            progress_val = activation_sum / float(n_slots)
        outputs.progress[env] = progress_val


# ---------------------------------------------------------------------------
# Fused dispatch_graph_dense + compose — same pattern as dispatch_compose_fused
# but for the primitive_graph_local backend's dense-consumer path. Used when
# ``k_max`` clears the parallel-compose threshold (see ``compose_select``).
# ---------------------------------------------------------------------------


@wp.kernel
def dispatch_graph_dense_compose_fused(
    env_slots: EnvSlots,
    subtask_schedule_ids: wp.array(dtype=int),
    vec3_nodes: PrimitiveProducerQueue,
    scalar_nodes: PrimitiveProducerQueue,
    quat_nodes: PrimitiveProducerQueue,
    scalar_sum_nodes: PrimitiveProducerQueue,
    contact_nodes: PrimitiveProducerQueue,
    spec: SubtaskSpec,
    state: StateAccess,
    outputs: Outputs,
    composer_state: ComposerState,
    direct_vec3: wp.array2d(dtype=float),
    direct_scalar: wp.array(dtype=float),
    direct_quat: wp.array2d(dtype=float),
    scalar_sum: wp.array(dtype=float),
    contact_mask: wp.array2d(dtype=float),
    episode_length_buf: wp.array(dtype=wp.int64),
    effective_max_episode_length: wp.array(dtype=int),
    instant_threshold: float,
    quality_easing: float,
    vec3_signature_count: int,
    scalar_signature_count: int,
    quat_signature_count: int,
    scalar_sum_signature_count: int,
    contact_signature_count: int,
):
    """Block-per-env fused kernel: dense graph consumer + parallel compose."""
    env, slot = wp.tid()
    n_slots = env_slots.slot_count[env]

    old_transit = composer_state.transit_steps[env]
    new_transit = old_transit + 1
    tsteps = float(new_transit)

    local_act = float(0.0)
    local_has_instant = int(0)
    local_instant_ok = int(1)
    local_quality = float(1.0)

    if slot < n_slots:
        sid = env_slots.subtask_ids[env, slot]
        tgt_off = env_slots.slot_offsets[env, slot]
        pipeline_id = subtask_schedule_ids[sid]
        akid = spec.activation_kernel_id[sid]
        act_param = spec.activation_kernel_param[sid]
        canon_off = spec.canonical_offset[sid]
        instant = spec.is_instant_flag[sid]
        is_tracking_i = spec.is_tracking_flag[sid]
        stride = spec.state_stride[sid]
        gcount = spec.gather_count[sid]

        d0 = float(0.0)
        d1 = float(0.0)
        d2 = float(0.0)
        d3 = float(0.0)
        err = float(0.0)

        if pipeline_id == PIPELINE_DIRECT_VEC3_DELTA:
            signature = vec3_nodes.subtask_signature[sid]
            node_id = env * vec3_signature_count + signature
            x3 = wp.vec3(direct_vec3[node_id, 0], direct_vec3[node_id, 1], direct_vec3[node_id, 2])
            d3v = _read_target_xyz(state, env, tgt_off) - x3
            d0 = d3v[0]
            d1 = d3v[1]
            d2 = d3v[2]
            err = _metric_geometric_vec3(d3v)
        elif pipeline_id == PIPELINE_DIRECT_SCALAR_DELTA:
            signature = scalar_nodes.subtask_signature[sid]
            node_id = env * scalar_signature_count + signature
            d0 = state.targets_flat[env, tgt_off] - direct_scalar[node_id]
            err = _metric_geometric_scalar(d0)
        elif pipeline_id == PIPELINE_DIRECT_QUAT_DELTA:
            signature = quat_nodes.subtask_signature[sid]
            node_id = env * quat_signature_count + signature
            q_current = wp.vec4(
                direct_quat[node_id, 0],
                direct_quat[node_id, 1],
                direct_quat[node_id, 2],
                direct_quat[node_id, 3],
            )
            dq = _delta_quaternion(q_current, _read_target_quat_xyzw(state, env, tgt_off))
            d0 = dq[0]
            d1 = dq[1]
            d2 = dq[2]
            d3 = dq[3]
            err = _metric_quaternion(dq)
        elif pipeline_id == PIPELINE_VEC3_THRESHOLD_VECTOR_DELTA:
            signature = contact_nodes.subtask_signature[sid]
            node_id = env * contact_signature_count + signature
            xcb = _contact_mask_vec4(contact_mask, node_id)
            tc0 = state.targets_flat[env, tgt_off]
            tc1 = float(0.0)
            tc2 = float(0.0)
            tc3 = float(0.0)
            if stride >= 2:
                tc1 = state.targets_flat[env, tgt_off + 1]
            if stride >= 3:
                tc2 = state.targets_flat[env, tgt_off + 2]
            if stride >= 4:
                tc3 = state.targets_flat[env, tgt_off + 3]
            d0 = tc0 - xcb[0]
            d1 = tc1 - xcb[1]
            d2 = tc2 - xcb[2]
            d3 = tc3 - xcb[3]
            err = wp.sqrt(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3)
        elif pipeline_id == PIPELINE_VEC3_THRESHOLD_SUM_DELTA:
            signature = contact_nodes.subtask_signature[sid]
            node_id = env * contact_signature_count + signature
            count = gcount / 3
            xcb = _contact_mask_vec4(contact_mask, node_id)
            x_cnt = float(0.0)
            for i in range(count):
                x_cnt = x_cnt + xcb[i]
            d0 = state.targets_flat[env, tgt_off] - x_cnt
            err = _metric_geometric_scalar(d0)
        elif pipeline_id == PIPELINE_VEC3_THRESHOLD_PAIR_DIFF_DELTA:
            signature = contact_nodes.subtask_signature[sid]
            node_id = env * contact_signature_count + signature
            count = gcount / 3
            half = count / 2
            xcb = _contact_mask_vec4(contact_mask, node_id)
            cnt_a = float(0.0)
            cnt_b = float(0.0)
            for i in range(count):
                if i < half:
                    cnt_a = cnt_a + xcb[i]
                else:
                    cnt_b = cnt_b + xcb[i]
            d0 = state.targets_flat[env, tgt_off] - (cnt_a - cnt_b)
            err = _metric_geometric_scalar(d0)
        elif pipeline_id == PIPELINE_SCALAR_SUM_DELTA:
            signature = scalar_sum_nodes.subtask_signature[sid]
            node_id = env * scalar_sum_signature_count + signature
            d0 = state.targets_flat[env, tgt_off] - scalar_sum[node_id]
            err = _metric_geometric_scalar(d0)

        act = _apply_activation(akid, err, act_param)
        local_act = act

        _scatter_delta(outputs, env, canon_off, stride, instant, d0, d1, d2, d3)
        outputs.buf_error[env, slot] = err
        outputs.buf_activation[env, slot] = act

        new_sum = composer_state.sum_activation[env, slot] + act
        composer_state.sum_activation[env, slot] = new_sum

        if instant != 0:
            local_has_instant = 1
            prev_achieved = composer_state.instant_achieved[env, slot]
            achieved_int = int(0)
            if prev_achieved:
                achieved_int = 1
            if act > instant_threshold:
                achieved_int = 1
            composer_state.instant_achieved[env, slot] = achieved_int != 0
            local_instant_ok = achieved_int
        elif is_tracking_i != 0:
            local_quality = new_sum / tsteps

    combined = wp.vec4(local_act, float(local_has_instant), float(local_instant_ok), local_quality)
    reduced_t = wp.tile_reduce(_compose_reduce_op, wp.tile(combined, preserve_type=True))

    if slot == 0:
        composer_state.transit_steps[env] = new_transit

        reduced = wp.tile_extract(reduced_t, 0)
        activation_sum = reduced[0]
        has_instant = int(reduced[1])
        all_instant_ok = int(reduced[2])
        quality_product = reduced[3]

        success_int = all_instant_ok * has_instant
        timeout_int = int(0)
        if episode_length_buf[env] >= wp.int64(effective_max_episode_length[env] - 1):
            timeout_int = 1
        done_now_int = success_int
        if timeout_int == 1:
            done_now_int = 1

        quality_factor = wp.pow(quality_product, quality_easing)
        terminal_value = float(all_instant_ok) * quality_factor

        reward = float(0.0)
        if done_now_int == 1:
            reward = terminal_value

        outputs.task_reward[env] = reward
        outputs.task_done_success[env] = success_int != 0

        progress_val = float(0.0)
        if n_slots > 0:
            progress_val = activation_sum / float(n_slots)
        outputs.progress[env] = progress_val


# ---------------------------------------------------------------------------
# Combined slab-copy kernel. Replaces N separate ``fill_slab_copy`` launches
# with a single launch, using a 2D grid (env, position) where ``position``
# enumerates floats across all copy slabs concatenated. Per-thread routing
# uses ``cumulative_sizes`` to find the slab and intra-slab offset.
#
# The kernel takes a fixed ``MAX_COPY_SLABS=8`` source-array slots; unused
# slots use a 1-element dummy that's never touched (their cumulative size
# range is empty).
# ---------------------------------------------------------------------------
