# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multiplicative terminal-reward composer for multi-task command terms.

Task return:
    ``G = reached · quality_factor``

where:

- ``reached`` is the AND of all instantaneous subtasks' achievement latches
  (vacuously ``True`` if the task has no instant subtasks).
- ``quality_factor ∈ [0, 1]`` is the **geometric mean** over all per-step-
  averaged tracking and safety activations:

      ``quality_factor = ( ∏_{k ∈ tracking ∪ safety} mean_t A_k(t) )^{1/K}``

  with ``A_k(t) = 1 − tanh(error_k(t) / σ_k)`` for tracking and
  ``A_k(t) = 1 − tanh(violation_k(t) / σ_k)`` for safety. Tracking and safety
  subtasks are reduced into a single quality scalar — both are ongoing
  conditions evaluated over the transit window, both produce activations in
  ``[0, 1]``, both should discount G but not invert preferences.

  Caller (command term) computes the geometric mean per env and passes it in;
  vacuously ``1`` when the task has neither tracking nor safety subtasks.

- ``G ∈ [0, 1]`` for every task kind. Emitted only at the terminal step
  (success OR timeout); zero on every other step.

Why geometric mean over the quality terms (not arithmetic, not unbounded
product): the composer was originally ``gate · arithmetic_mean(tracking) ·
∏(safety)``. Two independent compounding-related issues forced a redesign:

  1. Mixing arithmetic and multiplicative reductions makes the relative
     contribution of one safety dim depend on whether the task happens to
     have N tracking dims or 1 — fragile across cfg variants.
  2. Plain ``∏ safety_k`` compounds: at K=3 with each f_k=0.5, contribution
     drops to 0.125 — gradient signal becomes too weak.

  Geometric mean ``(∏ f_k)^{1/K}`` solves both: tracking and safety are
  symmetric (both are "quality dimensions"), and at K=3 with f_k=0.5 the
  result is 0.5 — proportional to typical factor magnitude, not their
  product.

Why multiplicative gate rather than additive: transit-scoped quality is
physically *gated* by task progress. A policy that maintains velocity forever
without arriving should score zero, not the same as arrival-with-quality; a
policy that arrives with massive chassis abuse should score *less* than clean
arrival but not negative. Multiplicative composition preserves both: ``gate ·
quality`` is monotonically non-negative, bounded by 1, and never makes success
worse than no-success.

Finite-horizon framing — all terminations (success, timeout, failure) are real;
the reward stream carries the whole task value so ``time_out=False`` everywhere
and ``V(s_terminal) = 0`` is correct by Bellman.
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
    instant_achieved: torch.Tensor,
    quality_factor: torch.Tensor,
    is_timeout: torch.Tensor,
    instant_threshold: float = 0.5,
) -> ComposerOutput:
    """Compute the multiplicative terminal task reward for a single step.

    Semantics:
        - success = all active-instant subtasks have ``instant_achieved = True``
          AND the task has at least one active-instant subtask.
        - gate = ``1`` if all active-instant subtasks achieved OR the task has
          no active-instant subtasks (vacuous), else ``0``.
        - quality_factor (caller-supplied) ∈ ``[0, 1]`` per env — the
          ``easing``-eased product of all per-dimension tracking and safety
          activations:

              ``( ∏_{k ∈ tracking ∪ safety} mean_t A_k(t) ) ^ easing``

          Vacuously ``1`` when the task has neither tracking nor safety
          subtasks (e.g. pure-instant). Caller (command term) owns the
          per-subtask accumulation, the product, and the easing exponent.
        - G = gate · quality_factor ∈ [0, 1].
        - Reward emitted only when ``done_now = success OR is_timeout``; zero
          otherwise.

    Active-vs-inactive subtasks: subtasks not belonging to the env's assigned
    task must have ``is_instant_subtask = False``. Padded slots are silently
    ignored.

    Idempotence contract: after ``success = True`` fires for an env, calling
    this function again without resetting ``instant_achieved`` will keep
    emitting ``success = True`` and a non-zero reward each step. The caller
    (command term) owns the reset on done.

    Multiplicative-composition rationale:
        Three RL-correctness properties that an additive ``-λ · violation``
        per-step penalty does not preserve:

        1. ``G ∈ [0, 1]`` always — V(s) is bounded, value bootstrap stays
           well-conditioned.
        2. Monotonic preference for success: ``success + bad_quality > no_success``
           because ``quality > 0``. The agent never refuses to terminate
           because the penalty would exceed the gain.
        3. Bootstrap immunity at reach-truncate: when ``gate = 0``,
           ``terminal_value = 0`` regardless of quality, so the bootstrap
           of ``γ·V(s_T)`` carries pure future-task-value, never
           future-quality contamination.

    Args:
        activation_scores: Per-step activation of every subtask [num_envs,
            num_subtasks], float in [0, 1] from the activation kernel.
        is_instant_subtask: Mask [num_envs, num_subtasks], bool — ``True`` iff
            the subtask is an instant subtask of the env's assigned task.
        instant_achieved: Latched achievement mask [num_envs, num_subtasks],
            bool (caller persists across steps, clears on reset).
        quality_factor: Per-env scalar [num_envs] in ``[0, 1]`` — the eased
            product of all tracking and safety per-subtask transit means.
            Vacuously 1 when the task has neither tracking nor safety.
        is_timeout: Per-env timeout-this-step flag [num_envs], bool.
        instant_threshold: Activation score strictly above which an instant
            subtask is considered achieved.

    Returns:
        :class:`ComposerOutput` carrying the reward, success flag, and the
        updated ``instant_achieved`` mask (latched with this step's new
        achievements).
    """
    # Latch per-subtask achievement: True once threshold is crossed, stays True.
    newly_achieved = (activation_scores > instant_threshold) & is_instant_subtask
    instant_achieved_next = instant_achieved | newly_achieved

    # Instant "OK" per subtask: either achieved, or the slot isn't an instant-subtask.
    # For a task with no instant subtasks, every slot is non-instant → all OK → gate=1.
    per_subtask_instant_ok = instant_achieved_next | ~is_instant_subtask
    all_instant_ok = per_subtask_instant_ok.all(dim=-1)  # [num_envs]
    has_instant = is_instant_subtask.any(dim=-1)  # [num_envs]

    success = all_instant_ok & has_instant  # [num_envs]
    done_now = success | is_timeout  # [num_envs]

    # Gate: 1 iff all instant subtasks achieved (vacuous for pure-tracking tasks).
    instant_gate = all_instant_ok.float()  # [num_envs]

    terminal_value = instant_gate * quality_factor.to(instant_gate.dtype)  # [num_envs]
    reward = torch.where(done_now, terminal_value, torch.zeros_like(terminal_value))

    return ComposerOutput(reward=reward, done_success=success, instant_achieved_next=instant_achieved_next)
