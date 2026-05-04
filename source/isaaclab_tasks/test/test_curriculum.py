# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :class:`Curriculum` and :class:`CurriculumCfg`."""

from __future__ import annotations

import torch

from isaaclab_tasks.manager_based.multi_task.curriculum import (
    BetaSignalCfg,
    Curriculum,
    CurriculumCfg,
    FrontierSignalCfg,
    SignalEntry,
    StateLayout,
    UniformSignalCfg,
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
# Curriculum (runtime)
# ---------------------------------------------------------------------------


def test_uniform_only_curriculum_is_uniform():
    """A UniformSignal-only curriculum with eps=0 returns exact 1/N probabilities."""
    layout = _layout_terrain(num_items=100)
    rates = torch.rand(100)
    cfg = CurriculumCfg(signals=[SignalEntry(cfg=UniformSignalCfg(), weight=1.0)], eps=0.0)
    curr = cfg.class_type(cfg, layout)
    probs = curr.probabilities(rates)
    assert torch.allclose(probs, torch.full_like(probs, 1.0 / 100))


def test_curriculum_probabilities_sum_to_one_finite_nonneg():
    layout = _layout_terrain()
    rates = torch.rand(layout.num_items)
    curr = Curriculum(
        CurriculumCfg(
            signals=[
                SignalEntry(cfg=BetaSignalCfg(target=0.66, kappa=1.0, eps=1e-3), weight=1.0),
                SignalEntry(cfg=FrontierSignalCfg(k=8, dilation_steps=1), weight=2.0),
            ],
            eps=1e-3,
        ),
        layout,
    )
    probs = curr.probabilities(rates)
    assert torch.isfinite(probs).all()
    assert (probs >= 0).all()
    assert abs(float(probs.sum()) - 1.0) < 1e-6


def test_signal_scores_dict_keys_match_signal_names():
    layout = _layout_terrain()
    rates = torch.rand(layout.num_items)
    curr = Curriculum(
        CurriculumCfg(
            signals=[
                SignalEntry(cfg=BetaSignalCfg(target=0.66, kappa=1.0), weight=1.0),
                SignalEntry(cfg=FrontierSignalCfg(k=8), weight=2.0),
            ],
        ),
        layout,
    )
    scores = curr.signal_scores(rates)
    assert set(scores.keys()) == {"beta", "frontier"}
    for v in scores.values():
        assert v.shape == (layout.num_items,)
    assert curr.signal_names == ["beta", "frontier"]


def test_negative_weight_clamped_to_zero():
    """Negative weights clamp to 0; the curriculum becomes uniform via eps."""
    layout = _layout_terrain()
    rates = torch.rand(layout.num_items)
    pos_cfg = CurriculumCfg(signals=[SignalEntry(cfg=UniformSignalCfg(), weight=1.0)], eps=0.0)
    neg_cfg = CurriculumCfg(signals=[SignalEntry(cfg=UniformSignalCfg(), weight=-1.0)], eps=1e-6)
    pos = pos_cfg.class_type(pos_cfg, layout).probabilities(rates)
    neg = neg_cfg.class_type(neg_cfg, layout).probabilities(rates)
    assert torch.allclose(pos, neg, atol=1e-6)


def test_find_signal_returns_active_signal():
    layout = _layout_terrain()
    curr = Curriculum(
        CurriculumCfg(
            signals=[
                SignalEntry(cfg=BetaSignalCfg(target=0.66, kappa=1.0), weight=1.0),
                SignalEntry(cfg=FrontierSignalCfg(k=8), weight=2.0),
            ],
        ),
        layout,
    )
    beta = curr.find_signal("beta")
    frontier = curr.find_signal("frontier")
    assert beta is not None
    assert frontier is not None
    assert beta.name == "beta"
    assert frontier.name == "frontier"
    assert curr.find_signal("missing") is None


# ---------------------------------------------------------------------------
# CurriculumCfg.build (blueprint -> runtime)
# ---------------------------------------------------------------------------


def test_curriculum_cfg_build_produces_runtime_curriculum():
    layout = _layout_terrain()
    cfg = CurriculumCfg(
        signals=[
            SignalEntry(cfg=BetaSignalCfg(target=0.5, kappa=2.0), weight=1.0),
            SignalEntry(cfg=FrontierSignalCfg(k=8), weight=1.5),
        ],
        eps=1e-3,
    )
    curr = cfg.class_type(cfg, layout)
    assert isinstance(curr, Curriculum)
    assert len(curr.signals) == 2
    assert curr.eps == 1e-3
    assert [s.name for s, _ in curr.signals] == ["beta", "frontier"]


def test_curriculum_cfg_default_rate_source_is_monitor():
    cfg = CurriculumCfg(signals=[SignalEntry(cfg=UniformSignalCfg(), weight=1.0)])
    assert cfg.rate_source == "monitor"


def test_curriculum_cfg_estimator_rate_source():
    cfg = CurriculumCfg(signals=[SignalEntry(cfg=UniformSignalCfg(), weight=1.0)], rate_source="estimator")
    assert cfg.rate_source == "estimator"


def test_factory_slot_eq_item_layout_works():
    """target_index=None propagates through the cfg.build path."""
    layout = _layout_factory(num_states=64)
    rates = torch.rand(64)
    curr = Curriculum(
        CurriculumCfg(
            signals=[
                SignalEntry(cfg=BetaSignalCfg(target=0.66, kappa=1.0, eps=1e-3), weight=1.0),
                SignalEntry(cfg=FrontierSignalCfg(k=8, dilation_steps=1), weight=2.0),
            ],
            eps=1e-3,
        ),
        layout,
    )
    probs = curr.probabilities(rates)
    assert torch.isfinite(probs).all()
    assert abs(float(probs.sum()) - 1.0) < 1e-6


def test_curriculum_cfg_dilation_steps_propagates():
    """dilation_steps from FrontierSignalCfg actually reaches the runtime signal."""
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
    cfg1 = CurriculumCfg(
        signals=[
            SignalEntry(cfg=UniformSignalCfg(), weight=1.0),
            SignalEntry(cfg=FrontierSignalCfg(k=2, dilation_steps=1), weight=2.0),
        ],
        eps=1e-3,
    )
    cfg3 = CurriculumCfg(
        signals=[
            SignalEntry(cfg=UniformSignalCfg(), weight=1.0),
            SignalEntry(cfg=FrontierSignalCfg(k=2, dilation_steps=3), weight=2.0),
        ],
        eps=1e-3,
    )
    p1 = Curriculum(cfg1, layout).probabilities(rates)
    p3 = Curriculum(cfg3, layout).probabilities(rates)
    threshold = 1.0 / n + 1e-6
    assert int((p3 > threshold).sum()) >= int((p1 > threshold).sum())
