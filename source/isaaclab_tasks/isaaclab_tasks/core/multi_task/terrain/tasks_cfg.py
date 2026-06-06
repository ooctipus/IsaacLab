# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Subtask cfg templates + authoring helpers for the multi-task env cfg.

Kept separate from ``multi_task_env_cfg`` so the env cfg itself stays focused on
composition (which tasks exist, how their subtasks combine) rather than on the
per-subtask numeric defaults.

Authoring contract:

- Each template is a fully-formed :class:`MultiTaskCfg.BaseTaskCfg` subclass with
  representative values. Tasks in the env cfg derive variants via
  :meth:`configclass.replace` — never through factory functions.
- Templates have module-public names (uppercase). Import from this module; don't
  inline the cfgs.
"""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.core.multi_task.mdp.commands.impl.kernels_torch import (
    ACTIVATION_KERNEL_ID,
    METRIC_KERNEL_ID,
    SAMPLER_KERNEL_ID,
    STATE_KERNEL_ID,
)
from isaaclab_tasks.core.multi_task.mdp.commands.impl.multi_task_cfg import MinMaxSampler, MultiTaskCfg

# Standing height for Anymal-C — the nominal base z for position/pose targets.
STANDING_Z = (0.4, 0.7)

BASE_ENTITY = SceneEntityCfg("robot", body_names="base")


# ---------------------------------------------------------------------------
# Subtask cfg templates.
# ---------------------------------------------------------------------------

# Tracking — base body linear velocity in world frame (xyz target).
LIN_VEL_TRACKING = MultiTaskCfg.TrackingTaskCfg(
    asset_cfg=BASE_ENTITY,
    state_kernel=int(STATE_KERNEL_ID.BODY_LIN_VEL),
    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
    activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
    # std chosen so the initial-policy regime (err ~1-3 m/s for a random
    # legged-robot controller) produces a graded signal rather than a
    # saturated ~0. std=1.0 gives A(err=1)=0.24, A(err=2)=0.04, A(err=0.5)=0.54
    # — reducible by better policies. Tighten toward 0.3 via curriculum
    # once the policy can produce coherent body velocities.
    activation_kernel_param=1.0,
    sampler=MinMaxSampler(
        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
        minimum=[-1.0, -1.0, 0.0],
        maximum=[1.0, 1.0, 0.0],
    ),
)

# Tracking — base body angular velocity in world frame (yaw-rate target).
ANG_VEL_TRACKING = MultiTaskCfg.TrackingTaskCfg(
    asset_cfg=BASE_ENTITY,
    state_kernel=int(STATE_KERNEL_ID.BODY_ANG_VEL),
    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
    activation_kernel=int(ACTIVATION_KERNEL_ID.TANH),
    # See ``LIN_VEL_TRACKING`` — matching std so both tracking tasks have
    # comparable gradient landscapes at initialization. Ang-vel sampler is
    # ±1.5 rad/s which fits the same regime.
    activation_kernel_param=1.0,
    sampler=MinMaxSampler(
        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
        minimum=[0.0, 0.0, -1.5],
        maximum=[0.0, 0.0, 1.5],
    ),
)

# Instant — base body position target at standing height. Variants override the
# sampler's min/max (to shift the target region, e.g. ground-level vs airborne)
# and ``activation_kernel_param`` (threshold for "achieved").
BODY_POS_INSTANT = MultiTaskCfg.InstantaneousTaskCfg(
    asset_cfg=BASE_ENTITY,
    state_kernel=int(STATE_KERNEL_ID.BODY_POS),
    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
    activation_kernel_param=0.2,
    sampler=MinMaxSampler(
        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
        minimum=[-3.0, -3.0, STANDING_Z[0]],
        maximum=[3.0, 3.0, STANDING_Z[1]],
    ),
)

# Instant — base body orientation target (Euler angles sampled, converted to quat).
# ``out_dim=4`` on the sampler aligns ``target_dim_max`` with the 4-vec quaternion.
BODY_QUAT_INSTANT = MultiTaskCfg.InstantaneousTaskCfg(
    asset_cfg=BASE_ENTITY,
    state_kernel=int(STATE_KERNEL_ID.BODY_QUAT),
    metric_kernel=int(METRIC_KERNEL_ID.QUATERNION),
    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
    activation_kernel_param=0.3,
    sampler=MinMaxSampler(
        kernel=int(SAMPLER_KERNEL_ID.EULER_UNIFORM_TO_QUAT),
        minimum=[0.0, 0.0, -3.14],
        maximum=[0.0, 0.0, 3.14],
        out_dim=4,
    ),
)

# Instant — count of feet in contact. Target defaults to 3, so the subtask fires
# when any three of the four feet are on the ground — permutation-invariant across
# which foot is lifted. Expresses the walking-tripod stance directly (the iconic
# Anymal "walk" gait) without hard-coding which leg takes the step.
FEET_CONTACT_COUNT_INSTANT = MultiTaskCfg.InstantaneousTaskCfg(
    asset_cfg=SceneEntityCfg(
        "contact_forces",
        body_names=["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"],
    ),
    state_kernel=int(STATE_KERNEL_ID.BODY_CONTACT_COUNT),
    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
    activation_kernel=int(ACTIVATION_KERNEL_ID.LESS),
    # State ∈ {0, 1, 2, 3, 4}, target = 3 → |delta| = |3 - state|. Threshold 0.5
    # → matched only when exactly 3 feet are in contact.
    activation_kernel_param=0.5,
    sampler=MinMaxSampler(
        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
        minimum=[3.0],
        maximum=[3.0],
    ),
)

# Tracking — "one pair planted, the other airborne" maintained over the transit
# window. Tracking (not instant) because the gait semantic is "maintain this
# condition while moving toward the goal", not "hit this condition once."
# The composer time-averages its activation across transit steps, so a policy
# that trots *during* transit scores high while a policy that briefly achieves
# one phase then stands still does not. Physics naturally forces alternation
# when the body is translating, so "|count_diff| == 2 most steps" corresponds
# to a real trot cycle.
#
# ``body_names`` is ordered so the first half and second half split into the
# two opposing groups; the kernel returns ``count(first) - count(second)``,
# which is ``±2`` when one group is fully planted and the other airborne.
# Sampler target ``0`` + ``GREATER`` activation at threshold ``1.5`` fires on
# either sign, so both gait phases activate the same subtask.
PAIR_CONTACT_DIFF_TRACKING = MultiTaskCfg.TrackingTaskCfg(
    asset_cfg=SceneEntityCfg(
        "contact_forces",
        body_names=["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"],
    ),
    state_kernel=int(STATE_KERNEL_ID.BODY_CONTACT_COUNT_DIFF),
    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
    activation_kernel=int(ACTIVATION_KERNEL_ID.GREATER),
    # With state = count(first) - count(second), |state| = 2 on either phase.
    # delta = target - state (target=0) so |delta| = |state|. Threshold 1.5 →
    # activated iff |state| > 1.5. The composer averages this over transit.
    activation_kernel_param=1.5,
    sampler=MinMaxSampler(
        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
        minimum=[0.0],
        maximum=[0.0],
    ),
)


# ---------------------------------------------------------------------------
# Soft-safety subtasks — multiplicative discount on terminal G.
#
# Authoring contract: each safety subtask uses a state kernel whose value is a
# scalar "violation" amount, target 0 (the no-violation point), and TANH
# activation with param = the violation scale. Per-step activation is
# ``1 − tanh(violation/scale)`` ∈ ``(0, 1]``; the composer accumulates these,
# takes the episode mean per subtask, and multiplies all per-subtask means
# together into ``G``.
#
# Because it's multiplicative on the terminal value, the safety penalty:
#   1) Discounts but never makes ``G`` negative (monotonicity preserved).
#   2) Vanishes at reach-truncate (``gate=0`` zeroes terminal regardless),
#      so the rsl_rl bootstrap path stays clean of safety-bias.
#   3) Accumulates over the whole transit window — there's no per-step
#      reward shaping that would couple V(s) to expected future safety cost.
#
# Authored as factory functions, NOT module constants. Reason: every safety
# subtask carries an :class:`SceneEntityCfg`, and ``SceneEntityCfg.resolve``
# mutates ``body_ids`` / ``joint_ids`` on first call. Sharing a single
# instance across tasks would trip the resolver's regex/id consistency
# check on the second call. A factory hands every task site a fresh
# instance — clean and explicit. Subtask dedup in :func:`spec.build_spec`
# still collapses identical safety subtasks into a single shared spec row,
# so per-step compute cost is unchanged.
# ---------------------------------------------------------------------------


# Soft-safety subtasks are :class:`MultiTaskCfg.TrackingTaskCfg` instances with
# ``expose_in_obs=False`` — they contribute to ``G``'s multiplicative quality
# factor exactly like a tracking goal, but their delta channels never reach
# the policy obs (the policy learns to satisfy them implicitly via the
# reward gradient on ``G``).

UNDESIRED_CONTACT = MultiTaskCfg.TrackingTaskCfg(
    asset_cfg=SceneEntityCfg("contact_forces", body_names="^(?!.*FOOT).*$"),
    state_kernel=int(STATE_KERNEL_ID.BODY_CONTACT_COUNT),
    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
    activation_kernel=int(ACTIVATION_KERNEL_ID.GAUSSIAN),
    activation_kernel_param=8.0,
    sampler=MinMaxSampler(
        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
        minimum=[0.0],
        maximum=[0.0],
    ),
    expose_in_obs=False,
)


# Soft-safety on total mechanical actuation power. Use GAUSSIAN, not TANH:
# locomotion at 200-500W should not be penalized at all (Gaussian's flat plateau
# near 0 vs TANH's steepest slope at 0); the gradient should sharpen as the
# policy approaches the budget edge σ — that's where the function transitions
# from "fine" to "not fine"; past 2σ the activation saturates near 0 so the
# gradient doesn't keep pulling toward "endless reduction" once the budget is
# already blown. σ = 1000W → steepest at ~707W (σ/√2), essentially 0 past
# 2000W. Pair with target = 0 so the metric ``|target − power| = power``.
MECH_POWER = MultiTaskCfg.TrackingTaskCfg(
    asset_cfg=SceneEntityCfg("robot"),
    state_kernel=int(STATE_KERNEL_ID.JOINT_MECH_POWER),
    metric_kernel=int(METRIC_KERNEL_ID.GEOMETRIC),
    activation_kernel=int(ACTIVATION_KERNEL_ID.GAUSSIAN),
    activation_kernel_param=2000.0,
    sampler=MinMaxSampler(
        kernel=int(SAMPLER_KERNEL_ID.UNIFORM),
        minimum=[0.0],
        maximum=[0.0],
    ),
    expose_in_obs=False,
)
