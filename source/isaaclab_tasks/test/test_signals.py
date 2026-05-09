# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for informativeness signals."""

from __future__ import annotations

import torch

from isaaclab_tasks.manager_based.multi_task.curriculum import (
    BetaSamplingStrategyCfg,
    FrontierSamplingStrategyCfg,
    StateLayout,
    UniformSamplingStrategyCfg,
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


def _score(strategy, rates: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(rates)
    strategy.score(rates, out)
    return out


# ---------------------------------------------------------------------------
# BetaSamplingStrategy
# ---------------------------------------------------------------------------


def test_beta_peaks_at_target():
    """Beta kernel maximum is near the target rate."""
    layout = _layout_terrain()
    cfg = BetaSamplingStrategyCfg(target=0.5, kappa=4.0)
    signal = cfg.class_type(cfg, layout)
    rates = torch.linspace(0.0, 1.0, 21)
    scores = _score(signal, rates)
    assert int(scores.argmax()) == 10  # index 10 is rate=0.5


def test_beta_uniform_input_uniform_output():
    """All-equal rates -> all-equal scores (Beta is per-item-only)."""
    layout = _layout_terrain()
    cfg = BetaSamplingStrategyCfg(target=0.66, kappa=1.0)
    signal = cfg.class_type(cfg, layout)
    rates = torch.full((100,), 0.42)
    scores = _score(signal, rates)
    assert float(scores.std()) < 1e-7


def test_beta_independent_of_layout():
    """Beta scores depend only on rates, not the layout topology."""
    layout_a = _layout_terrain(num_states=20, num_items=100)
    layout_b = _layout_factory(num_states=100)
    cfg = BetaSamplingStrategyCfg(target=0.66, kappa=1.0)
    rates = torch.rand(100)
    s_a = _score(cfg.class_type(cfg, layout_a), rates)
    s_b = _score(cfg.class_type(cfg, layout_b), rates)
    assert torch.allclose(s_a, s_b)


def test_beta_score_non_negative():
    """Score is always >= 0."""
    layout = _layout_terrain()
    cfg = BetaSamplingStrategyCfg(target=0.66, kappa=1.0)
    signal = cfg.class_type(cfg, layout)
    rates = torch.rand(100)
    scores = _score(signal, rates)
    assert (scores >= 0).all()


# ---------------------------------------------------------------------------
# FrontierSamplingStrategy
# ---------------------------------------------------------------------------


def test_frontier_zero_when_all_rates_equal():
    """All-equal rates -> identically zero frontier score."""
    layout = _layout_terrain()
    cfg = FrontierSamplingStrategyCfg(k=8)
    signal = cfg.class_type(cfg, layout)
    rates = torch.full((layout.num_items,), 0.42)
    scores = _score(signal, rates)
    assert torch.allclose(scores, torch.zeros_like(scores), atol=1e-6)


def test_frontier_dilation_grows_signal():
    """More dilation steps -> more items with nonzero score."""
    layout = _layout_terrain(num_states=20, num_items=100)
    rates = torch.zeros(100)
    rates[0] = 0.95
    cfg1 = FrontierSamplingStrategyCfg(k=2, dilation_steps=1)
    cfg3 = FrontierSamplingStrategyCfg(k=2, dilation_steps=3)
    s1 = _score(cfg1.class_type(cfg1, layout), rates)
    s3 = _score(cfg3.class_type(cfg3, layout), rates)
    assert int((s3 > 0).sum()) >= int((s1 > 0).sum())


def test_frontier_slot_eq_item_no_target():
    """Factory's slot==item topology (target_index=None) produces valid scores."""
    layout = _layout_factory(num_states=64)
    cfg = FrontierSamplingStrategyCfg(k=8)
    signal = cfg.class_type(cfg, layout)
    rates = torch.rand(64)
    rates[:5] = 0.9  # learned cluster
    scores = _score(signal, rates)
    assert torch.isfinite(scores).all()
    assert (scores >= 0).all()


def test_frontier_score_non_negative():
    """Score is always >= 0 (above-mean-deviation is clamped)."""
    layout = _layout_terrain()
    cfg = FrontierSamplingStrategyCfg(k=8)
    signal = cfg.class_type(cfg, layout)
    rates = torch.rand(layout.num_items)
    scores = _score(signal, rates)
    assert (scores >= 0).all()


def test_frontier_isolated_unlearned_task_stays_zero():
    """A task whose feature-space neighbourhood has no learned task scores zero.

    Per-task frontier propagates rate via task-feature kNN, so an isolated
    cluster of unlearned tasks gets no inheritance from a far-away cluster
    of learned tasks (the kNN graph wires them up to similar tasks, not
    arbitrary "shares-a-state" tasks).
    """
    # 4 states, 4 items in two well-separated (spawn_xy, target_xy) clusters.
    # Cluster A (items 0,1): tasks among learned states, high rates.
    # Cluster B (items 2,3): tasks among unlearned states, far in feature space.
    coords = torch.tensor(
        [
            [0.00, 0.00],  # state 0 -- learned cluster
            [0.05, 0.05],  # state 1 -- learned cluster
            [100.0, 100.0],  # state 2 -- isolated cluster
            [100.0, 100.1],  # state 3 -- isolated cluster
        ]
    )
    spawn_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    target_idx = torch.tensor([1, 0, 3, 2], dtype=torch.long)
    layout = StateLayout(coords=coords, spawn_index=spawn_idx, target_index=target_idx)

    # Items 0/1 are learned, items 2/3 are unreached. Cluster B is far enough
    # in feature space that neither A item is its kNN neighbour.
    rates = torch.tensor([0.9, 0.9, 0.0, 0.0])
    cfg = FrontierSamplingStrategyCfg(k=1, dilation_steps=1)
    signal = cfg.class_type(cfg, layout)
    scores = _score(signal, rates)

    # Items 2/3 stay zero -- their kNN neighbours are within their own
    # cluster (also at rate 0). No spurious inheritance from cluster A.
    assert float(scores[2]) == 0.0
    assert float(scores[3]) == 0.0


def test_frontier_propagates_to_neighbour_in_task_space():
    """An unlearned task whose feature-space neighbour is learned scores positive."""
    # Two items at nearby task features. Item 0 has high rate, item 1 has zero.
    # With per-task kNN and dilation, item 1 should inherit from item 0.
    coords = torch.tensor([[0.0, 0.0], [0.1, 0.1]])
    spawn_idx = torch.tensor([0, 1], dtype=torch.long)
    target_idx = torch.tensor([1, 0], dtype=torch.long)
    layout = StateLayout(coords=coords, spawn_index=spawn_idx, target_index=target_idx)

    rates = torch.tensor([0.9, 0.0])
    cfg = FrontierSamplingStrategyCfg(k=1, dilation_steps=1)
    signal = cfg.class_type(cfg, layout)
    scores = _score(signal, rates)

    # Item 0 is itself learned -> (1 - 0.9) factor gives small score.
    # Item 1 is unlearned with item 0 as kNN neighbour -> picks up frontier.
    assert float(scores[1]) > 0.0
    # Item 0's score is small (already mostly learned).
    assert float(scores[0]) <= float(scores[1])


def test_frontier_partition_isolates_mechanics():
    """Partitioned kNN keeps mechanically-distinct task families independent.

    Two tasks share spatial endpoints but live in different ``task_partition``
    classes (e.g. walking-A→B vs flying-A→B). Frontier from the learned
    family must NOT propagate into the unlearned family.
    """
    coords = torch.tensor([[0.0, 0.0], [0.1, 0.1]])
    # 4 items: same (spawn, target) endpoints, two partitions of two items each.
    spawn_idx = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    target_idx = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    # Items 0,1 -> partition 0 (e.g. walking); items 2,3 -> partition 1 (flying).
    task_partition = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    layout = StateLayout(
        coords=coords,
        spawn_index=spawn_idx,
        target_index=target_idx,
        task_partition=task_partition,
    )

    # Walking learned (item 0 high), flying still unlearned. With unpartitioned
    # kNN, item 2 would have item 0 as a feature-space neighbour and inherit
    # frontier; with partition, items 2/3 see only each other (both at 0).
    rates = torch.tensor([0.9, 0.0, 0.0, 0.0])
    cfg = FrontierSamplingStrategyCfg(k=1, dilation_steps=1)
    signal = cfg.class_type(cfg, layout)
    scores = _score(signal, rates)

    assert float(scores[2]) == 0.0
    assert float(scores[3]) == 0.0
    # Partition 0 still propagates internally: item 1 picks up from item 0.
    assert float(scores[1]) > 0.0


# ---------------------------------------------------------------------------
# UniformSamplingStrategy
# ---------------------------------------------------------------------------


def test_uniform_returns_ones():
    layout = _layout_terrain(num_items=100)
    cfg = UniformSamplingStrategyCfg()
    signal = cfg.class_type(cfg, layout)
    rates = torch.rand(100)
    scores = _score(signal, rates)
    assert torch.equal(scores, torch.ones(100))


def test_uniform_ignores_rates():
    layout = _layout_terrain(num_items=100)
    cfg = UniformSamplingStrategyCfg()
    signal = cfg.class_type(cfg, layout)
    s_a = _score(signal, torch.zeros(100))
    s_b = _score(signal, torch.rand(100))
    assert torch.equal(s_a, s_b)


def test_uniform_dtype_matches_input():
    layout = _layout_terrain(num_items=64)
    cfg = UniformSamplingStrategyCfg()
    signal = cfg.class_type(cfg, layout)
    for dtype in (torch.float32, torch.float64):
        scores = _score(signal, torch.rand(64, dtype=dtype))
        assert scores.dtype == dtype
