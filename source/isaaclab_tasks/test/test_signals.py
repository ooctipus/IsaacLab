# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for informativeness signals."""

from __future__ import annotations

import torch

from isaaclab_tasks.manager_based.multi_task.curriculum import (
    BetaSignal,
    FrontierSignal,
    StateLayout,
    UniformSignal,
)


def _layout_terrain(num_states: int = 50, num_items: int = 200, seed: int = 0) -> StateLayout:
    """Terrain-style layout: 2D coords, paired spawn+target items."""
    torch.manual_seed(seed)
    coords = torch.rand(num_states, 2)
    spawn = torch.randint(0, num_states, (num_items,), dtype=torch.long)
    target = torch.randint(0, num_states, (num_items,), dtype=torch.long)
    return StateLayout(coords=coords, spawn_index=spawn, target_index=target)


def _layout_factory(num_states: int = 64, seed: int = 0) -> StateLayout:
    """Factory-style layout: 3D coords, slot==item, target_index=None."""
    torch.manual_seed(seed)
    coords = torch.rand(num_states, 3)
    spawn = torch.arange(num_states, dtype=torch.long)
    return StateLayout(coords=coords, spawn_index=spawn)


# ---------------------------------------------------------------------------
# BetaSignal
# ---------------------------------------------------------------------------


def test_beta_peaks_at_target():
    """Beta kernel maximum is near the target rate."""
    layout = _layout_terrain()
    signal = BetaSignal(layout, target=0.5, kappa=4.0)
    rates = torch.linspace(0.0, 1.0, 21)
    scores = signal.score(rates)
    assert int(scores.argmax()) == 10  # index 10 is rate=0.5


def test_beta_uniform_input_uniform_output():
    """All-equal rates -> all-equal scores (Beta is per-item-only)."""
    layout = _layout_terrain()
    signal = BetaSignal(layout, target=0.66, kappa=1.0)
    rates = torch.full((100,), 0.42)
    scores = signal.score(rates)
    assert float(scores.std()) < 1e-7


def test_beta_independent_of_layout():
    """Beta scores depend only on rates, not the layout topology."""
    layout_a = _layout_terrain(num_states=20, num_items=100)
    layout_b = _layout_factory(num_states=100)
    rates = torch.rand(100)
    s_a = BetaSignal(layout_a, target=0.66, kappa=1.0).score(rates)
    s_b = BetaSignal(layout_b, target=0.66, kappa=1.0).score(rates)
    assert torch.allclose(s_a, s_b)


def test_beta_score_non_negative():
    """Score is always >= 0."""
    layout = _layout_terrain()
    signal = BetaSignal(layout, target=0.66, kappa=1.0)
    rates = torch.rand(100)
    scores = signal.score(rates)
    assert (scores >= 0).all()


# ---------------------------------------------------------------------------
# FrontierSignal
# ---------------------------------------------------------------------------


def test_frontier_zero_when_all_rates_equal():
    """All-equal rates -> identically zero frontier score."""
    layout = _layout_terrain()
    signal = FrontierSignal(layout, k=8)
    rates = torch.full((layout.num_items,), 0.42)
    scores = signal.score(rates)
    assert torch.allclose(scores, torch.zeros_like(scores), atol=1e-6)


def test_frontier_dilation_grows_signal():
    """More dilation steps -> more items with nonzero score."""
    layout = _layout_terrain(num_states=20, num_items=100)
    rates = torch.zeros(100)
    rates[0] = 0.95
    s1 = FrontierSignal(layout, k=2, dilation_steps=1).score(rates)
    s3 = FrontierSignal(layout, k=2, dilation_steps=3).score(rates)
    assert int((s3 > 0).sum()) >= int((s1 > 0).sum())


def test_frontier_slot_eq_item_no_target():
    """Factory's slot==item topology (target_index=None) produces valid scores."""
    layout = _layout_factory(num_states=64)
    signal = FrontierSignal(layout, k=8)
    rates = torch.rand(64)
    rates[:5] = 0.9  # learned cluster
    scores = signal.score(rates)
    assert torch.isfinite(scores).all()
    assert (scores >= 0).all()


def test_frontier_score_non_negative():
    """Score is always >= 0 (above-mean-deviation is clamped)."""
    layout = _layout_terrain()
    signal = FrontierSignal(layout, k=8)
    rates = torch.rand(layout.num_items)
    scores = signal.score(rates)
    assert (scores >= 0).all()


# ---------------------------------------------------------------------------
# UniformSignal
# ---------------------------------------------------------------------------


def test_uniform_returns_ones():
    layout = _layout_terrain(num_items=100)
    signal = UniformSignal(layout)
    rates = torch.rand(100)
    scores = signal.score(rates)
    assert torch.equal(scores, torch.ones(100))


def test_uniform_ignores_rates():
    layout = _layout_terrain(num_items=100)
    signal = UniformSignal(layout)
    s_a = signal.score(torch.zeros(100))
    s_b = signal.score(torch.rand(100))
    assert torch.equal(s_a, s_b)


def test_uniform_dtype_matches_input():
    layout = _layout_terrain(num_items=64)
    signal = UniformSignal(layout)
    for dtype in (torch.float32, torch.float64):
        scores = signal.score(torch.rand(64, dtype=dtype))
        assert scores.dtype == dtype
