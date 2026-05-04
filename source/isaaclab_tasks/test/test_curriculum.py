# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :class:`WeightedCurriculum` and :func:`make_curriculum`.

The numerical-equivalence tests are the load-bearing safety net for the
upcoming terrain / factory migrations: they assert that
``make_curriculum(legacy_cfg).probabilities(rates)`` matches the legacy
``beta_sampling_probs`` / ``frontier_sampling_probs`` /
``uniform_sampling_probs`` outputs to machine precision.
"""

from __future__ import annotations

import pytest
import torch

from isaaclab_tasks.manager_based.multi_task.mdp.util import (
    BetaSamplingCfg,
    BetaSignal,
    FrontierSamplingCfg,
    FrontierSignal,
    StateLayout,
    UniformSamplingCfg,
    UniformSignal,
    WeightedCurriculum,
    beta_sampling_probs,
    build_knn_indices,
    frontier_sampling_probs,
    make_curriculum,
    uniform_sampling_probs,
)


def _layout_terrain(num_states: int = 50, num_items: int = 200, seed: int = 0) -> StateLayout:
    torch.manual_seed(seed)
    coords = torch.rand(num_states, 2) * 5.0
    spawn = torch.randint(0, num_states, (num_items,), dtype=torch.long)
    target = torch.randint(0, num_states, (num_items,), dtype=torch.long)
    return StateLayout(coords=coords, spawn_index=spawn, target_index=target)


def _layout_factory(num_states: int = 64, seed: int = 0) -> StateLayout:
    torch.manual_seed(seed)
    coords = torch.rand(num_states, 3)
    spawn = torch.arange(num_states, dtype=torch.long)
    return StateLayout(coords=coords, spawn_index=spawn)


# ---------------------------------------------------------------------------
# WeightedCurriculum
# ---------------------------------------------------------------------------


def test_weighted_curriculum_uniform_alone():
    """Single UniformSignal with eps=0 -> exact 1/N probabilities."""
    layout = _layout_terrain(num_items=100)
    rates = torch.rand(100)
    curr = WeightedCurriculum(signals=[(UniformSignal(layout), 1.0)], eps=0.0)
    probs = curr.probabilities(rates)
    assert torch.allclose(probs, torch.full_like(probs, 1.0 / 100))


def test_weighted_curriculum_probs_sum_to_one():
    layout = _layout_terrain()
    rates = torch.rand(layout.num_items)
    curr = WeightedCurriculum(
        signals=[
            (BetaSignal(layout, target=0.66, kappa=1.0, eps=1e-3), 1.0),
            (FrontierSignal(layout, k=8), 2.0),
        ],
        eps=1e-3,
    )
    probs = curr.probabilities(rates)
    assert torch.isfinite(probs).all()
    assert (probs >= 0).all()
    assert abs(float(probs.sum()) - 1.0) < 1e-6


def test_weighted_curriculum_signal_scores_dict():
    layout = _layout_terrain()
    rates = torch.rand(layout.num_items)
    curr = WeightedCurriculum(
        signals=[
            (BetaSignal(layout, target=0.66, kappa=1.0), 1.0),
            (FrontierSignal(layout, k=8), 2.0),
        ],
        eps=1e-3,
    )
    scores = curr.signal_scores(rates)
    assert set(scores.keys()) == {"beta", "frontier"}
    for v in scores.values():
        assert v.shape == (layout.num_items,)
    assert curr.signal_names == ["beta", "frontier"]


def test_weighted_curriculum_negative_weight_clamped():
    """Negative signal weights are clamped to zero, not subtracted."""
    layout = _layout_terrain()
    rates = torch.rand(layout.num_items)
    pos = WeightedCurriculum(signals=[(UniformSignal(layout), 1.0)], eps=0.0).probabilities(rates)
    neg = WeightedCurriculum(signals=[(UniformSignal(layout), -1.0)], eps=1e-6).probabilities(rates)
    # With weight clamped to 0, only eps remains -> still uniform.
    assert torch.allclose(pos, neg, atol=1e-6)


# ---------------------------------------------------------------------------
# make_curriculum -- legacy equivalence
# ---------------------------------------------------------------------------


def test_make_curriculum_uniform_matches_legacy():
    """UniformSamplingCfg -> identical probs to uniform_sampling_probs."""
    layout = _layout_terrain(num_items=200)
    rates = torch.rand(200)
    new_probs = make_curriculum(UniformSamplingCfg(), layout).probabilities(rates)
    legacy_probs = uniform_sampling_probs(rates)
    assert torch.allclose(new_probs, legacy_probs)


def test_make_curriculum_beta_matches_legacy():
    """BetaSamplingCfg -> identical probs to beta_sampling_probs."""
    layout = _layout_terrain(num_items=200)
    rates = torch.rand(200)
    new_probs = make_curriculum(BetaSamplingCfg(target=0.66, kappa=1.0), layout).probabilities(rates)
    legacy_probs = beta_sampling_probs(rates, target=0.66, kappa=1.0)
    assert torch.allclose(new_probs, legacy_probs, atol=1e-7)


def test_make_curriculum_beta_various_targets():
    """Equivalence holds across (target, kappa) settings."""
    layout = _layout_terrain(num_items=200)
    rates = torch.rand(200)
    for target in (0.3, 0.5, 0.66, 0.9):
        for kappa in (0.5, 1.0, 4.0):
            new_probs = make_curriculum(BetaSamplingCfg(target=target, kappa=kappa), layout).probabilities(rates)
            legacy_probs = beta_sampling_probs(rates, target=target, kappa=kappa)
            assert torch.allclose(new_probs, legacy_probs, atol=1e-7), f"mismatch for target={target}, kappa={kappa}"


def test_make_curriculum_frontier_with_beta_base_matches_legacy():
    """FrontierSamplingCfg(Beta base) -> identical probs to frontier_sampling_probs."""
    torch.manual_seed(0)
    n_states, n_items = 50, 200
    coords = torch.rand(n_states, 2) * 5.0
    spawn = torch.randint(0, n_states, (n_items,), dtype=torch.long)
    target = torch.randint(0, n_states, (n_items,), dtype=torch.long)
    rates = torch.rand(n_items)
    layout = StateLayout(coords=coords, spawn_index=spawn, target_index=target)

    cfg = FrontierSamplingCfg(
        base=BetaSamplingCfg(target=0.66, kappa=1.0),
        k=8,
        frontier_lambda=2.0,
        dilation_steps=1,
        eps=1e-3,
    )
    new_probs = make_curriculum(cfg, layout).probabilities(rates)
    legacy_probs = frontier_sampling_probs(
        rates,
        state_knn_indices=build_knn_indices(coords, k=8),
        spawn_index=spawn,
        target_index=target,
        base=BetaSamplingCfg(target=0.66, kappa=1.0),
        frontier_lambda=2.0,
        dilation_steps=1,
        eps=1e-3,
    )
    assert torch.allclose(new_probs, legacy_probs, atol=1e-6)


def test_make_curriculum_frontier_with_uniform_base_matches_legacy():
    """FrontierSamplingCfg(Uniform base) matches the legacy uniform-base path."""
    torch.manual_seed(0)
    n_states, n_items = 50, 200
    coords = torch.rand(n_states, 2) * 5.0
    spawn = torch.randint(0, n_states, (n_items,), dtype=torch.long)
    target = torch.randint(0, n_states, (n_items,), dtype=torch.long)
    rates = torch.rand(n_items)
    layout = StateLayout(coords=coords, spawn_index=spawn, target_index=target)

    cfg = FrontierSamplingCfg(
        base=UniformSamplingCfg(),
        k=8,
        frontier_lambda=5.0,
        dilation_steps=1,
        eps=1e-3,
    )
    new_probs = make_curriculum(cfg, layout).probabilities(rates)
    legacy_probs = frontier_sampling_probs(
        rates,
        state_knn_indices=build_knn_indices(coords, k=8),
        spawn_index=spawn,
        target_index=target,
        base=UniformSamplingCfg(),
        frontier_lambda=5.0,
        dilation_steps=1,
        eps=1e-3,
    )
    assert torch.allclose(new_probs, legacy_probs, atol=1e-6)


def test_make_curriculum_frontier_slot_eq_item():
    """Factory-style slot==item layout (target_index=None) exercises the
    target_index-None branch end-to-end through make_curriculum."""
    layout = _layout_factory(num_states=64)
    rates = torch.rand(64)
    cfg = FrontierSamplingCfg(
        base=BetaSamplingCfg(target=0.66, kappa=1.0),
        k=8,
        frontier_lambda=2.0,
        dilation_steps=1,
        eps=1e-3,
    )
    probs = make_curriculum(cfg, layout).probabilities(rates)
    assert torch.isfinite(probs).all()
    assert abs(float(probs.sum()) - 1.0) < 1e-6


def test_make_curriculum_frontier_dilation_passes_through():
    """dilation_steps from cfg actually reaches the FrontierSignal."""
    torch.manual_seed(0)
    n = 20
    coords = torch.linspace(0, 1, n).unsqueeze(-1).repeat(1, 2)
    layout = StateLayout(
        coords=coords,
        spawn_index=torch.arange(n, dtype=torch.long),
        target_index=None,
    )
    rates = torch.zeros(n)
    rates[0] = 0.95
    p1 = make_curriculum(
        FrontierSamplingCfg(base=UniformSamplingCfg(), k=2, frontier_lambda=2.0, dilation_steps=1, eps=1e-3),
        layout,
    ).probabilities(rates)
    p3 = make_curriculum(
        FrontierSamplingCfg(base=UniformSamplingCfg(), k=2, frontier_lambda=2.0, dilation_steps=3, eps=1e-3),
        layout,
    ).probabilities(rates)
    # 3-step dilation reaches further -> more items get above-uniform probability.
    threshold = 1.0 / n + 1e-6
    assert int((p3 > threshold).sum()) >= int((p1 > threshold).sum())


def test_make_curriculum_invalid_cfg_raises():
    layout = _layout_terrain()
    with pytest.raises(TypeError):
        make_curriculum("not a cfg", layout)  # type: ignore[arg-type]


def test_make_curriculum_invalid_frontier_base_raises():
    layout = _layout_terrain()

    class _BogusBase:
        pass

    cfg = FrontierSamplingCfg(base=_BogusBase(), k=8, frontier_lambda=1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        make_curriculum(cfg, layout)
