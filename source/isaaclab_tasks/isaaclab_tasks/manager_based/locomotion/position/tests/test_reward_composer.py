# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`multiplicative_terminal_reward`.

These tests exercise the core Stage-3 decision (terminal-only multiplicative reward)
without any env dependency. Synthetic activation trajectories are replayed through the
composer's step function and episodic return is verified against the closed-form
expectation ``G = reached · mean_tracking_activation``.
"""

from __future__ import annotations

import torch

from isaaclab_tasks.manager_based.locomotion.position.mdp.commands.reward_composer import (
    multiplicative_terminal_reward,
)


def _run_episode(
    activation_per_step: torch.Tensor,
    is_instant_subtask: torch.Tensor,
    is_tracking_subtask: torch.Tensor,
    instant_threshold: float = 0.5,
    record_non_terminal_rewards: bool = False,
) -> tuple[torch.Tensor, int, bool, list[torch.Tensor] | None]:
    """Drive the composer step-by-step on a synthetic trajectory.

    Mimics the per-step state update the command term performs:
        sum_activation += activation_scores
        transit_steps  += 1
        instant_achieved = output.instant_achieved_next

    Stops at the first step where every env has either succeeded or timed out.

    Args:
        activation_per_step: [T, num_envs, num_subtasks] float, synthetic scores.
        is_instant_subtask: [num_envs, num_subtasks] bool.
        is_tracking_subtask: [num_envs, num_subtasks] bool.
        instant_threshold: Forwarded to the composer.
        record_non_terminal_rewards: If ``True``, returns the full per-step reward
            tensor list (for asserting zero-on-non-terminal).

    Returns:
        Tuple ``(final_reward [num_envs], final_step_index, was_timeout,
        per_step_rewards)`` where ``per_step_rewards`` is a list with length
        ``final_step_index + 1`` when ``record_non_terminal_rewards=True``, else
        ``None``. ``was_timeout`` is ``True`` iff every env reached the timeout
        step without any succeeding.
    """
    T, num_envs, num_subtasks = activation_per_step.shape
    device = activation_per_step.device

    sum_activation = torch.zeros(num_envs, num_subtasks, device=device)
    transit_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    instant_achieved = torch.zeros(num_envs, num_subtasks, dtype=torch.bool, device=device)

    final_reward = torch.zeros(num_envs, device=device)
    final_step = T - 1
    timed_out = True
    per_step_rewards: list[torch.Tensor] | None = [] if record_non_terminal_rewards else None

    for step in range(T):
        activation = activation_per_step[step]
        sum_activation = sum_activation + activation
        transit_steps = transit_steps + 1

        is_timeout = torch.zeros(num_envs, dtype=torch.bool, device=device)
        if step == T - 1:
            is_timeout.fill_(True)

        out = multiplicative_terminal_reward(
            activation_scores=activation,
            is_instant_subtask=is_instant_subtask,
            is_tracking_subtask=is_tracking_subtask,
            sum_activation=sum_activation,
            transit_steps=transit_steps,
            instant_achieved=instant_achieved,
            is_timeout=is_timeout,
            instant_threshold=instant_threshold,
        )
        instant_achieved = out.instant_achieved_next

        if per_step_rewards is not None:
            per_step_rewards.append(out.reward.clone())

        if bool((out.done_success | is_timeout).all()):
            final_reward = out.reward
            final_step = step
            timed_out = bool(is_timeout.all()) and not bool(out.done_success.any())
            break

    return final_reward, final_step, timed_out, per_step_rewards


# -----------------------------------------------------------------------------
# Pure-instant tasks
# -----------------------------------------------------------------------------


def test_pure_instant_success_at_step_k():
    """Single instant subtask; activation 1.0 fires at step ``k`` → terminal G = 1."""
    T = 100
    k = 37
    activation = torch.zeros(T, 1, 1)
    activation[k:] = 1.0

    reward, final_step, timed_out, _ = _run_episode(
        activation_per_step=activation,
        is_instant_subtask=torch.tensor([[True]]),
        is_tracking_subtask=torch.tensor([[False]]),
    )
    assert not timed_out
    assert final_step == k
    assert torch.allclose(reward, torch.tensor([1.0]))


def test_pure_instant_never_succeeds():
    """Instant never crosses threshold → timeout with reward 0."""
    T = 50
    activation = torch.full((T, 1, 1), 0.3)  # below threshold forever

    reward, final_step, timed_out, _ = _run_episode(
        activation_per_step=activation,
        is_instant_subtask=torch.tensor([[True]]),
        is_tracking_subtask=torch.tensor([[False]]),
    )
    assert timed_out
    assert final_step == T - 1
    assert torch.allclose(reward, torch.tensor([0.0]))


def test_pure_instant_threshold_is_strict():
    """Activation exactly at threshold is NOT achievement (strict ``>`` semantics)."""
    T = 20
    activation = torch.full((T, 1, 1), 0.5)  # exactly at threshold

    reward, _, timed_out, _ = _run_episode(
        activation_per_step=activation,
        is_instant_subtask=torch.tensor([[True]]),
        is_tracking_subtask=torch.tensor([[False]]),
    )
    assert timed_out
    assert torch.allclose(reward, torch.tensor([0.0]))


# -----------------------------------------------------------------------------
# Pure-tracking tasks
# -----------------------------------------------------------------------------


def test_pure_tracking_constant_activation():
    """Constant activation ``c`` → timeout with reward = ``c``."""
    T = 80
    c = 0.75
    activation = torch.full((T, 1, 1), c)

    reward, final_step, timed_out, _ = _run_episode(
        activation_per_step=activation,
        is_instant_subtask=torch.tensor([[False]]),
        is_tracking_subtask=torch.tensor([[True]]),
    )
    assert timed_out
    assert final_step == T - 1
    assert torch.allclose(reward, torch.tensor([c]))


def test_pure_tracking_zero_activation():
    """Pure tracking with zero activation → timeout with reward 0."""
    T = 30
    activation = torch.zeros(T, 1, 1)

    reward, _, timed_out, _ = _run_episode(
        activation_per_step=activation,
        is_instant_subtask=torch.tensor([[False]]),
        is_tracking_subtask=torch.tensor([[True]]),
    )
    assert timed_out
    assert torch.allclose(reward, torch.tensor([0.0]))


def test_pure_tracking_varying_activation():
    """Activation alternating {0, 1} → reward = observed mean ≈ 0.5."""
    T = 40
    activation = torch.zeros(T, 1, 1)
    activation[::2] = 1.0  # even-indexed steps have activation 1, odd-indexed 0
    expected_mean = activation.flatten().mean().item()

    reward, _, timed_out, _ = _run_episode(
        activation_per_step=activation,
        is_instant_subtask=torch.tensor([[False]]),
        is_tracking_subtask=torch.tensor([[True]]),
    )
    assert timed_out
    assert abs(reward.item() - expected_mean) < 1e-6


# -----------------------------------------------------------------------------
# Mixed tasks (core Stage-3 behavior)
# -----------------------------------------------------------------------------


def test_mixed_gating_success():
    """Reach instant at step ``k`` while tracking constant ``c`` → G = c."""
    T = 120
    k = 42
    c = 0.6
    # [T, 1 env, 2 subtasks]: col 0 = instant, col 1 = tracking
    activation = torch.zeros(T, 1, 2)
    activation[:, 0, 1] = c
    activation[k:, 0, 0] = 1.0

    reward, final_step, timed_out, _ = _run_episode(
        activation_per_step=activation,
        is_instant_subtask=torch.tensor([[True, False]]),
        is_tracking_subtask=torch.tensor([[False, True]]),
    )
    assert not timed_out
    assert final_step == k
    assert torch.allclose(reward, torch.tensor([c]))


def test_mixed_never_reach_zero_reward():
    """Perfect tracking but instant never fires → timeout with reward 0.

    Policy-C case: "never arrive, perfect vel → 0" from the plan's design table.
    """
    T = 60
    c = 1.0
    activation = torch.zeros(T, 1, 2)
    activation[:, 0, 1] = c

    reward, _, timed_out, _ = _run_episode(
        activation_per_step=activation,
        is_instant_subtask=torch.tensor([[True, False]]),
        is_tracking_subtask=torch.tensor([[False, True]]),
    )
    assert timed_out
    assert torch.allclose(reward, torch.tensor([0.0]))


def test_mixed_timing_invariance():
    """Same tracking quality, different success steps → same episodic reward."""
    T = 150
    c = 0.55
    for k in (1, 10, 50, 100, 148):
        activation = torch.zeros(T, 1, 2)
        activation[:, 0, 1] = c
        activation[k:, 0, 0] = 1.0

        reward, final_step, timed_out, _ = _run_episode(
            activation_per_step=activation,
            is_instant_subtask=torch.tensor([[True, False]]),
            is_tracking_subtask=torch.tensor([[False, True]]),
        )
        assert not timed_out, f"k={k}: expected success, got timeout"
        assert final_step == k, f"k={k}: expected final_step={k}, got {final_step}"
        assert torch.allclose(reward, torch.tensor([c]), atol=1e-6), f"k={k}: expected reward {c}, got {reward.item()}"


def test_mixed_tracking_averages_up_to_success():
    """Mean activation computed over the exact transit window (first ``k+1`` steps)."""
    T = 100
    k = 9  # small so the arithmetic is exact
    # Tracking = 1.0 for first 5 steps, 0.0 after → mean over first 10 = 0.5.
    activation = torch.zeros(T, 1, 2)
    activation[:5, 0, 1] = 1.0
    activation[k:, 0, 0] = 1.0  # instant achieves at step k=9

    reward, final_step, timed_out, _ = _run_episode(
        activation_per_step=activation,
        is_instant_subtask=torch.tensor([[True, False]]),
        is_tracking_subtask=torch.tensor([[False, True]]),
    )
    # At step k=9, sum_tracking = 5.0 (from first 5 steps), transit_steps = 10 → mean = 0.5
    assert not timed_out
    assert final_step == k
    assert torch.allclose(reward, torch.tensor([0.5]), atol=1e-6)


# -----------------------------------------------------------------------------
# Multi-instant / multi-tracking composition
# -----------------------------------------------------------------------------


def test_multi_instant_requires_latching():
    """Two instant subtasks — first flashes achievement then drops; second achieves later.

    Without latching, at step ``k_second`` only the second subtask would be achieved,
    so success would not fire. With latching, the first's early flash persists and
    success fires at ``k_second``. This test fails if latching is broken.
    """
    T = 100
    k_first_flash = 15
    k_second = 60
    activation = torch.zeros(T, 1, 2)
    activation[k_first_flash, 0, 0] = 1.0  # single-step flash above threshold
    # col 0 drops back to 0 for every step after k_first_flash (default zeros)
    activation[k_second:, 0, 1] = 1.0  # second achieves at k_second

    reward, final_step, timed_out, _ = _run_episode(
        activation_per_step=activation,
        is_instant_subtask=torch.tensor([[True, True]]),
        is_tracking_subtask=torch.tensor([[False, False]]),
    )
    assert not timed_out
    assert final_step == k_second  # success fires only when the last instant achieves
    assert torch.allclose(reward, torch.tensor([1.0]))


def test_multi_instant_one_missing_times_out():
    """Two instant subtasks; only one achieves → no success, timeout with reward 0."""
    T = 40
    activation = torch.zeros(T, 1, 2)
    activation[5:, 0, 0] = 1.0

    reward, _, timed_out, _ = _run_episode(
        activation_per_step=activation,
        is_instant_subtask=torch.tensor([[True, True]]),
        is_tracking_subtask=torch.tensor([[False, False]]),
    )
    assert timed_out
    assert torch.allclose(reward, torch.tensor([0.0]))


def test_multi_tracking_mean_of_means():
    """Two tracking subtasks with constants c1, c2 → G = (c1 + c2) / 2."""
    T = 60
    c1, c2 = 0.8, 0.3
    activation = torch.zeros(T, 1, 2)
    activation[:, 0, 0] = c1
    activation[:, 0, 1] = c2

    reward, _, timed_out, _ = _run_episode(
        activation_per_step=activation,
        is_instant_subtask=torch.tensor([[False, False]]),
        is_tracking_subtask=torch.tensor([[True, True]]),
    )
    assert timed_out
    assert torch.allclose(reward, torch.tensor([(c1 + c2) / 2]), atol=1e-6)


# -----------------------------------------------------------------------------
# Terminal-only emission contract
# -----------------------------------------------------------------------------


def test_reward_is_zero_on_every_non_terminal_step():
    """Per-step reward must be exactly zero on every step where done is False.

    Verifies the terminal-only emission contract of the composer: the task reward
    has no per-step component. Any dense gradient must come from separate shaping
    terms in the reward manager.
    """
    T = 50
    c = 0.7
    k = 30
    activation = torch.zeros(T, 1, 2)
    activation[:, 0, 1] = c
    activation[k:, 0, 0] = 1.0

    reward, final_step, timed_out, per_step_rewards = _run_episode(
        activation_per_step=activation,
        is_instant_subtask=torch.tensor([[True, False]]),
        is_tracking_subtask=torch.tensor([[False, True]]),
        record_non_terminal_rewards=True,
    )
    assert not timed_out
    assert final_step == k
    assert per_step_rewards is not None
    # Every step before k must have reward exactly 0.0.
    for step_idx in range(k):
        assert torch.allclose(per_step_rewards[step_idx], torch.zeros(1)), (
            f"step {step_idx}: reward {per_step_rewards[step_idx].item()} != 0.0"
        )
    # Terminal step carries the actual return.
    assert torch.allclose(per_step_rewards[k], torch.tensor([c]), atol=1e-6)


def test_pure_tracking_reward_only_at_timeout():
    """Pure-tracking emits zero every step except the timeout step."""
    T = 25
    c = 0.5
    activation = torch.full((T, 1, 1), c)

    _, _, timed_out, per_step_rewards = _run_episode(
        activation_per_step=activation,
        is_instant_subtask=torch.tensor([[False]]),
        is_tracking_subtask=torch.tensor([[True]]),
        record_non_terminal_rewards=True,
    )
    assert timed_out
    assert per_step_rewards is not None
    for step_idx in range(T - 1):
        assert torch.allclose(per_step_rewards[step_idx], torch.zeros(1))
    assert torch.allclose(per_step_rewards[T - 1], torch.tensor([c]))


# -----------------------------------------------------------------------------
# Idempotence contract (caller must reset after success)
# -----------------------------------------------------------------------------


def test_success_repeats_without_reset():
    """After success fires, continuing without reset keeps emitting success.

    Documents the caller contract: the composer is idempotent — once ``instant_achieved``
    is fully latched, every subsequent call produces ``done_success=True`` and a
    non-zero reward. The command-term caller is responsible for resetting
    per-env state after any terminal step.
    """
    num_envs, num_subtasks = 1, 1
    activation_achieved = torch.tensor([[1.0]])  # above threshold
    is_instant = torch.tensor([[True]])
    is_tracking = torch.tensor([[False]])

    sum_activation = torch.zeros(num_envs, num_subtasks)
    transit_steps = torch.zeros(num_envs, dtype=torch.long)
    instant_achieved = torch.zeros(num_envs, num_subtasks, dtype=torch.bool)

    for step in range(5):
        sum_activation = sum_activation + activation_achieved
        transit_steps = transit_steps + 1
        is_timeout = torch.zeros(num_envs, dtype=torch.bool)

        out = multiplicative_terminal_reward(
            activation_scores=activation_achieved,
            is_instant_subtask=is_instant,
            is_tracking_subtask=is_tracking,
            sum_activation=sum_activation,
            transit_steps=transit_steps,
            instant_achieved=instant_achieved,
            is_timeout=is_timeout,
        )
        instant_achieved = out.instant_achieved_next
        assert bool(out.done_success.item()), f"step {step}: success must keep firing without reset"
        assert torch.allclose(out.reward, torch.tensor([1.0]))


# -----------------------------------------------------------------------------
# Batched heterogeneous envs
# -----------------------------------------------------------------------------


def test_batched_mixed_independent_envs():
    """Two envs in parallel with different task configurations.

    Env 0 is mixed and succeeds at step k=15 with tracking 0.8 → G=0.8.
    Env 1 is pure-tracking with constant 0.4 → G=0.4 at timeout.
    """
    T = 60
    c_env1 = 0.4
    k_env0 = 15
    c_env0_track = 0.8

    activation = torch.zeros(T, 2, 2)
    activation[k_env0:, 0, 0] = 1.0
    activation[:, 0, 1] = c_env0_track
    activation[:, 1, 1] = c_env1

    is_instant = torch.tensor(
        [
            [True, False],
            [False, False],
        ]
    )
    is_tracking = torch.tensor(
        [
            [False, True],
            [False, True],
        ]
    )

    sum_activation = torch.zeros(2, 2)
    transit_steps = torch.zeros(2, dtype=torch.long)
    instant_achieved = torch.zeros(2, 2, dtype=torch.bool)

    env0_final_reward: float | None = None
    env1_final_reward: float | None = None

    for step in range(T):
        sum_activation = sum_activation + activation[step]
        transit_steps = transit_steps + 1
        is_timeout = torch.tensor([step == T - 1, step == T - 1])

        out = multiplicative_terminal_reward(
            activation_scores=activation[step],
            is_instant_subtask=is_instant,
            is_tracking_subtask=is_tracking,
            sum_activation=sum_activation,
            transit_steps=transit_steps,
            instant_achieved=instant_achieved,
            is_timeout=is_timeout,
        )
        instant_achieved = out.instant_achieved_next

        if env0_final_reward is None and bool(out.done_success[0]):
            env0_final_reward = out.reward[0].item()
        if step == T - 1:
            env1_final_reward = out.reward[1].item()

    assert env0_final_reward is not None
    assert abs(env0_final_reward - c_env0_track) < 1e-5
    assert env1_final_reward is not None
    assert abs(env1_final_reward - c_env1) < 1e-5


def test_batched_mixed_success_and_failure():
    """Two envs: one succeeds, one never reaches its instant goal.

    Env 0 achieves both instant subtasks and has tracking 0.9 → G=0.9 at success.
    Env 1 achieves only the first instant (second never crosses threshold) → G=0 at timeout.
    Tracking is the same across envs; gating separates them.
    """
    T = 80
    k_env0_first = 5
    k_env0_second = 20
    k_env1_first = 5

    # [T, 2 envs, 3 subtasks]: col 0 instant A, col 1 instant B, col 2 tracking
    activation = torch.zeros(T, 2, 3)
    # Env 0: both instants fire
    activation[k_env0_first:, 0, 0] = 1.0
    activation[k_env0_second:, 0, 1] = 1.0
    activation[:, 0, 2] = 0.9
    # Env 1: only first instant fires
    activation[k_env1_first:, 1, 0] = 1.0
    activation[:, 1, 2] = 0.9

    is_instant = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
        ]
    )
    is_tracking = torch.tensor(
        [
            [False, False, True],
            [False, False, True],
        ]
    )

    sum_activation = torch.zeros(2, 3)
    transit_steps = torch.zeros(2, dtype=torch.long)
    instant_achieved = torch.zeros(2, 3, dtype=torch.bool)

    env0_final_reward: float | None = None
    env1_final_reward: float | None = None

    for step in range(T):
        sum_activation = sum_activation + activation[step]
        transit_steps = transit_steps + 1
        is_timeout = torch.tensor([step == T - 1, step == T - 1])

        out = multiplicative_terminal_reward(
            activation_scores=activation[step],
            is_instant_subtask=is_instant,
            is_tracking_subtask=is_tracking,
            sum_activation=sum_activation,
            transit_steps=transit_steps,
            instant_achieved=instant_achieved,
            is_timeout=is_timeout,
        )
        instant_achieved = out.instant_achieved_next

        if env0_final_reward is None and bool(out.done_success[0]):
            env0_final_reward = out.reward[0].item()
        if step == T - 1:
            env1_final_reward = out.reward[1].item()

    assert env0_final_reward is not None
    assert abs(env0_final_reward - 0.9) < 1e-5
    assert env1_final_reward is not None
    assert env1_final_reward == 0.0
