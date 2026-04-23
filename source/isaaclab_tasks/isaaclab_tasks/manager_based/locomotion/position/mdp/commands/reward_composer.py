# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multiplicative terminal-reward composer for multi-task command terms.

Task return: ``G = reached · mean_activation_of_tracking_subtasks_over_transit``.

- ``reached`` is the AND of all instantaneous subtasks' achievement flags (vacuously
  ``True`` if the task has no instant subtasks).
- ``mean_activation`` is the running mean over all tracking subtasks (vacuously ``1``
  if the task has no tracking subtasks).
- ``G ∈ [0, 1]`` for every task kind. Emitted only on terminal steps
  (success or timeout); per-step reward is zero.

Why multiplicative rather than additive: transit-scoped tracking (e.g. "maintain 1 m/s
while walking to point B") is physically gated by reaching B. A policy that holds
velocity forever without arriving should score zero, not the same as arrival-with-quality.
The AND-gate encodes that dependency directly.

Finite-horizon framing — all terminations (success, timeout, failure) are real;
the reward stream carries the whole task value so ``time_out=False`` everywhere and
``V(s_terminal) = 0`` is correct by Bellman. See the plan at
``~/.claude/plans/id-prefer-you-study-cryptic-quail.md`` for the full derivation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ComposerOutput:
    """Single-step output of :func:`multiplicative_terminal_reward`."""

    reward: torch.Tensor
    """Per-env task reward [num_envs], float. Nonzero only on terminal steps."""

    done_success: torch.Tensor
    """Per-env success-termination flag [num_envs], bool. ``True`` when all
    instantaneous subtasks have been achieved (and the task has any instant subtask)."""

    instant_achieved_next: torch.Tensor
    """Updated persistent achievement flags [num_envs, num_subtasks], bool. The
    caller stores this back as the new state for the next step."""


def multiplicative_terminal_reward(
    activation_scores: torch.Tensor,
    is_instant_subtask: torch.Tensor,
    is_tracking_subtask: torch.Tensor,
    sum_activation: torch.Tensor,
    transit_steps: torch.Tensor,
    instant_achieved: torch.Tensor,
    is_timeout: torch.Tensor,
    instant_threshold: float = 0.5,
) -> ComposerOutput:
    """Compute the multiplicative terminal task reward for a single step.

    Notation (for one env):
        - ``sum_A[s] = Σ_{t ≤ now} activation[s]`` for every subtask ``s`` — running sum.
        - ``transit_steps = now + 1`` — total steps elapsed since reset, inclusive of
          the current step.
        - ``instant_achieved[s]`` — latched ``True`` once subtask ``s`` passes its
          threshold; stays ``True`` until reset.

    Semantics:
        - success = all active-instant subtasks have ``instant_achieved = True`` AND
          the task has at least one active-instant subtask.
        - gate = ``1`` if all active-instant subtasks achieved OR the task has no
          active-instant subtasks (vacuous), else ``0``.
        - transit_mean = mean of ``sum_A / transit_steps`` across active-tracking
          subtasks, vacuously ``1`` when the task has no active-tracking subtasks.
        - G = gate · transit_mean ∈ [0, 1].
        - Reward emitted only when ``done_now = success OR is_timeout``; zero otherwise.

    Active-vs-inactive subtasks: subtasks not belonging to the env's assigned task
    must have ``is_instant_subtask = False`` AND ``is_tracking_subtask = False``.
    Such slots are silently ignored (treated as non-instant, non-tracking).

    Idempotence contract: after ``success = True`` fires for an env, calling this
    function again without resetting ``instant_achieved`` / ``sum_activation`` /
    ``transit_steps`` will keep emitting ``success = True`` and a non-zero reward
    each step. The caller (command term) owns the reset on done.

    Args:
        activation_scores: Per-step activation of every subtask [num_envs,
            num_subtasks], float in [0, 1] from the activation kernel.
        is_instant_subtask: Mask [num_envs, num_subtasks], bool — ``True`` iff the
            subtask is an instant subtask of the env's assigned task.
        is_tracking_subtask: Mask [num_envs, num_subtasks], bool — ``True`` iff the
            subtask is a tracking subtask of the env's assigned task.
        sum_activation: Running sum of activation per subtask [num_envs,
            num_subtasks], float, updated outside this function (caller
            increments before the call).
        transit_steps: Step count since reset [num_envs], long or float. Caller
            guarantees ``≥ 1`` on every call (increment before composer).
        instant_achieved: Latched achievement mask [num_envs, num_subtasks], bool
            (caller persists across steps, clears on reset).
        is_timeout: Per-env timeout-this-step flag [num_envs], bool.
        instant_threshold: Activation score strictly above which an instant subtask
            is considered achieved.

    Returns:
        :class:`ComposerOutput` carrying the reward, success flag, and the updated
        ``instant_achieved`` mask (latched with this step's new achievements).
    """
    # Latch per-subtask achievement: True once threshold is crossed, stays True.
    newly_achieved = (activation_scores > instant_threshold) & is_instant_subtask
    instant_achieved_next = instant_achieved | newly_achieved

    # Instant "OK" per subtask: either achieved, or the slot isn't an instant-subtask.
    # For a task with no instant subtasks, every slot is non-instant → all OK → gate=1.
    per_subtask_instant_ok = instant_achieved_next | ~is_instant_subtask
    all_instant_ok = per_subtask_instant_ok.all(dim=-1)  # [num_envs]
    has_instant = is_instant_subtask.any(dim=-1)  # [num_envs]
    has_tracking = is_tracking_subtask.any(dim=-1)  # [num_envs]

    success = all_instant_ok & has_instant  # [num_envs]
    done_now = success | is_timeout  # [num_envs]

    # Gate: 1 iff all instant subtasks achieved (vacuous for pure-tracking tasks).
    instant_gate = all_instant_ok.float()  # [num_envs]

    # Transit mean over tracking subtasks.
    # sum_A is accumulated for every slot but we only average tracking ones.
    # transit_steps is [num_envs]; broadcast along subtask axis.
    steps_denom = transit_steps.to(sum_activation.dtype).unsqueeze(-1)  # [num_envs, 1]
    per_subtask_mean = sum_activation / steps_denom  # [num_envs, num_subtasks]
    tracking_count = is_tracking_subtask.sum(dim=-1)  # [num_envs]
    tracking_mean_sum = (per_subtask_mean * is_tracking_subtask).sum(dim=-1)  # [num_envs]
    # Vacuous value 1 for envs with no tracking subtasks; guard the /0 for those rows.
    safe_count = tracking_count.clamp(min=1).to(tracking_mean_sum.dtype)
    tracking_mean = torch.where(
        has_tracking,
        tracking_mean_sum / safe_count,
        torch.ones_like(tracking_mean_sum),
    )

    terminal_value = instant_gate * tracking_mean  # [num_envs]
    reward = torch.where(done_now, terminal_value, torch.zeros_like(terminal_value))

    return ComposerOutput(reward=reward, done_success=success, instant_achieved_next=instant_achieved_next)
