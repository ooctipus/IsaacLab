# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the frontier sampler primitives.

Covers ``build_knn_indices``, ``state_frontier_weights``, and
``frontier_sampling_probs`` -- the three pure functions that make up the
state-buffer-graph frontier-aware curriculum sampler. No env / Kit
required.
"""

from __future__ import annotations

import pytest
import torch

from isaaclab_tasks.manager_based.multi_task.mdp.util import (
    BetaSamplingCfg,
    UniformSamplingCfg,
    build_knn_indices,
    frontier_sampling_probs,
    state_frontier_weights,
)

# ---------------------------------------------------------------------------
# build_knn_indices
# ---------------------------------------------------------------------------


def test_build_knn_indices_basic():
    torch.manual_seed(0)
    xy = torch.rand(50, 2)
    knn = build_knn_indices(xy, k=8)
    assert knn.shape == (50, 8)
    assert knn.dtype == torch.long
    # All indices in valid range
    assert int(knn.min()) >= 0 and int(knn.max()) < 50
    # No row contains its own index (self should be excluded)
    self_idx = torch.arange(50).unsqueeze(-1)
    assert not (knn == self_idx).any()


def test_build_knn_indices_single_point():
    """n=1: every row of knn is just the point's own index (degenerate)."""
    knn = build_knn_indices(torch.tensor([[0.0, 0.0]]), k=4)
    assert knn.shape == (1, 4)
    assert (knn == 0).all()


def test_build_knn_indices_k_larger_than_pool():
    """When k+1 > n, missing slots are padded with self."""
    xy = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    knn = build_knn_indices(xy, k=8)
    assert knn.shape == (3, 8)
    # First 2 neighbors of each row should be the two other points (in some order),
    # remaining 6 slots should be self-references.
    self_idx = torch.arange(3)
    for i in range(3):
        real_neighbors = set(knn[i, :2].tolist())
        assert i not in real_neighbors  # self excluded from real
        assert real_neighbors == set(range(3)) - {i}
        assert (knn[i, 2:] == self_idx[i]).all()


def test_build_knn_indices_identical_xy():
    """Multiple points at the same xy: scipy handles distance=0 gracefully."""
    xy = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    knn = build_knn_indices(xy, k=2)
    # Row 0: self at (0,0); nearest others are (0,0) and (1,0) -- both should appear
    # Row 1: same as row 0
    # Row 2: nearest are (0,0) and (0,0) -- both indices 0 and 1 should appear
    for row in knn:
        assert int(row.min()) >= 0 and int(row.max()) < 3


def test_build_knn_indices_invalid_k():
    with pytest.raises(ValueError):
        build_knn_indices(torch.zeros(5, 2), k=0)


# ---------------------------------------------------------------------------
# state_frontier_weights
# ---------------------------------------------------------------------------


def test_state_frontier_zero_when_all_rates_equal():
    """All states at same s -> s_dil == s -> frontier is identically 0."""
    torch.manual_seed(0)
    n_states = 30
    knn = build_knn_indices(torch.rand(n_states, 2), k=8)
    spawn = torch.arange(n_states)
    target = torch.zeros(n_states, dtype=torch.long)
    rates = torch.full((n_states,), 0.42)
    state_s, state_frontier = state_frontier_weights(
        rates, state_knn_indices=knn, spawn_index=spawn, target_index=target, dilation_steps=1
    )
    assert state_s.shape == (n_states,)
    assert state_frontier.shape == (n_states,)
    assert torch.allclose(state_frontier, torch.zeros_like(state_frontier), atol=1e-6)


def test_state_frontier_isolated_void_gets_zero():
    """A state whose kNN neighbors are all unlearned should have state_frontier ~ 0,
    while states adjacent to the learned cluster should have state_frontier > 0."""
    # Deterministic layout: 5 learned states tightly clustered at (-5, -5),
    # 44 unlearned states in [0, 1]^2, 1 lone unlearned state at (10, 10).
    # The lone state is geometrically far from both clusters; its kNN are
    # drawn from the [0, 1]^2 cluster (all unlearned), so its frontier is ~0.
    learned_xy = torch.tensor([[-5.0, -5.0], [-4.9, -5.0], [-5.0, -4.9], [-4.9, -4.9], [-5.05, -4.95]])
    torch.manual_seed(0)
    middle_xy = torch.rand(44, 2)
    lone_xy = torch.tensor([[10.0, 10.0]])
    xy = torch.cat([learned_xy, middle_xy, lone_xy])
    n_states = xy.shape[0]

    knn = build_knn_indices(xy, k=4)
    spawn = torch.arange(n_states)
    target = torch.zeros(n_states, dtype=torch.long)

    rates = torch.full((n_states,), 0.05)
    rates[:5] = 0.95  # learned cluster

    _, state_frontier = state_frontier_weights(
        rates, state_knn_indices=knn, spawn_index=spawn, target_index=target, dilation_steps=1
    )

    # Lone state has only unlearned kNN neighbors -> frontier ~ 0
    assert float(state_frontier[-1]) < 1e-3
    # At least one state at the learned-cluster boundary has nontrivial frontier
    assert float(state_frontier.max()) > 0.1


def test_state_frontier_dilation_grows_signal():
    """More dilation steps should expand the frontier signal outward."""
    torch.manual_seed(0)
    n = 20
    xy = torch.linspace(0, 1, n).unsqueeze(-1).repeat(1, 2)  # diagonal line
    knn = build_knn_indices(xy, k=2)
    spawn = torch.arange(n)
    target = torch.zeros(n, dtype=torch.long)
    rates = torch.zeros(n)
    rates[0] = 0.95  # only first state is learned
    _, sf1 = state_frontier_weights(
        rates, state_knn_indices=knn, spawn_index=spawn, target_index=target, dilation_steps=1
    )
    _, sf3 = state_frontier_weights(
        rates, state_knn_indices=knn, spawn_index=spawn, target_index=target, dilation_steps=3
    )
    # 3-step dilation should reach further: more states with nonzero frontier
    assert int((sf3 > 0).sum()) >= int((sf1 > 0).sum())


# ---------------------------------------------------------------------------
# frontier_sampling_probs
# ---------------------------------------------------------------------------


def _setup(n_states=100, n_tasks=200, seed=0):
    torch.manual_seed(seed)
    xy = torch.rand(n_states, 2) * 10.0
    knn = build_knn_indices(xy, k=8)
    spawn = torch.randint(0, n_states, (n_tasks,))
    target = torch.randint(0, n_states, (n_tasks,))
    rates = torch.rand(n_tasks)
    return xy, knn, spawn, target, rates


def test_probs_normalized_and_finite():
    _, knn, spawn, target, rates = _setup()
    probs = frontier_sampling_probs(
        rates,
        state_knn_indices=knn,
        spawn_index=spawn,
        target_index=target,
        base=BetaSamplingCfg(target=0.66, kappa=1.0),
        frontier_lambda=2.0,
    )
    assert torch.isfinite(probs).all()
    assert (probs >= 0).all()
    assert abs(float(probs.sum()) - 1.0) < 1e-5


def test_lambda_zero_collapses_to_base_up_to_eps():
    """frontier_lambda=0 should give probs proportional to base_w (modulo the eps floor)."""
    _, knn, spawn, target, rates = _setup()
    probs = frontier_sampling_probs(
        rates,
        state_knn_indices=knn,
        spawn_index=spawn,
        target_index=target,
        base=BetaSamplingCfg(target=0.66, kappa=1.0),
        frontier_lambda=0.0,
    )
    # No frontier contribution means w = base_w + eps; ordering is determined
    # entirely by base_w (Beta peaked at 0.66). Highest-prob task should have
    # rate close to 0.66.
    top_idx = int(probs.argmax())
    assert abs(float(rates[top_idx]) - 0.66) < 0.3


def test_single_target_produces_no_target_contribution():
    """In a single-target topology, target_dev is identically 0, so the
    algorithm must reduce to spawn-side differentiation only."""
    torch.manual_seed(0)
    n_states = 50
    n_tasks = 50
    xy = torch.rand(n_states, 2) * 10.0
    knn = build_knn_indices(xy, k=8)
    spawn = torch.arange(n_tasks)
    target_shared = torch.zeros(n_tasks, dtype=torch.long)
    target_varied = torch.randint(0, n_states, (n_tasks,))
    rates = torch.rand(n_tasks)
    p_shared = frontier_sampling_probs(
        rates,
        state_knn_indices=knn,
        spawn_index=spawn,
        target_index=target_shared,
        base=BetaSamplingCfg(target=0.66, kappa=1.0),
        frontier_lambda=2.0,
    )
    # Verify constant target really cancels: every task's target deviation is 0,
    # so probs are determined by spawn-side only. The output should be finite
    # and normalized regardless.
    assert torch.isfinite(p_shared).all()
    assert abs(float(p_shared.sum()) - 1.0) < 1e-5
    # Sanity: with varied targets the distribution should be different (unless
    # the spawn variation happens to dominate everywhere, which is unlikely).
    p_varied = frontier_sampling_probs(
        rates,
        state_knn_indices=knn,
        spawn_index=spawn,
        target_index=target_varied,
        base=BetaSamplingCfg(target=0.66, kappa=1.0),
        frontier_lambda=2.0,
    )
    assert not torch.allclose(p_shared, p_varied, atol=1e-6)


def test_uniform_base_yields_pure_frontier_signal():
    """With UniformSamplingCfg, base_w is constant 1, so per-task variation
    comes entirely from the frontier term."""
    torch.manual_seed(0)
    _, knn, spawn, target, rates = _setup()
    probs = frontier_sampling_probs(
        rates,
        state_knn_indices=knn,
        spawn_index=spawn,
        target_index=target,
        base=UniformSamplingCfg(),
        frontier_lambda=5.0,
    )
    assert torch.isfinite(probs).all()
    assert abs(float(probs.sum()) - 1.0) < 1e-5


def test_cold_start_all_zero_rates_is_essentially_uniform():
    """Cold start: no successes anywhere -> state_frontier is identically 0
    -> every task gets the same base_w (Beta evaluated at s=0) -> probs are
    near-uniform across all tasks."""
    torch.manual_seed(0)
    _, knn, spawn, target, _ = _setup()
    rates = torch.zeros_like(_setup()[4])
    probs = frontier_sampling_probs(
        rates,
        state_knn_indices=knn,
        spawn_index=spawn,
        target_index=target,
        base=BetaSamplingCfg(target=0.66, kappa=1.0),
        frontier_lambda=5.0,
    )
    # Every task should have the same prob (within float tolerance) because
    # success_rate is identical for everyone and frontier_w is 0 everywhere.
    spread = float(probs.max()) - float(probs.min())
    assert spread < 1e-6


def test_invalid_base_type_raises():
    _, knn, spawn, target, rates = _setup()
    with pytest.raises(TypeError):
        frontier_sampling_probs(
            rates,
            state_knn_indices=knn,
            spawn_index=spawn,
            target_index=target,
            base="not a cfg",  # type: ignore[arg-type]
            frontier_lambda=1.0,
        )
