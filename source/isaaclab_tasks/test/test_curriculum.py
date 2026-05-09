# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :class:`Sampler` and :class:`SamplerCfg`."""

from __future__ import annotations

import pytest
import torch

from isaaclab_tasks.manager_based.multi_task.curriculum import (
    BetaSamplingStrategyCfg,
    FrontierSamplingStrategyCfg,
    Sampler,
    SamplerCfg,
    StateLayout,
    UniformSamplingStrategyCfg,
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
# Sampler (runtime)
# ---------------------------------------------------------------------------


def test_uniform_only_sampler_is_uniform():
    """A UniformSamplingStrategy-only sampler with eps=0 returns exact 1/N probabilities."""
    layout = _layout_terrain(num_items=100)
    rates = torch.rand(100)
    cfg = SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=1.0)], eps=0.0)
    curr = cfg.class_type(cfg, layout)
    probs = curr.probabilities(rates)
    assert torch.allclose(probs, torch.full_like(probs, 1.0 / 100))


def test_sampler_probabilities_sum_to_one_finite_nonneg():
    layout = _layout_terrain()
    rates = torch.rand(layout.num_items)
    curr = Sampler(
        SamplerCfg(
            strategies=[
                BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0),
                FrontierSamplingStrategyCfg(k=8, dilation_steps=1, weight=2.0),
            ],
            eps=1e-3,
        ),
        layout,
    )
    probs = curr.probabilities(rates)
    assert torch.isfinite(probs).all()
    assert (probs >= 0).all()
    assert abs(float(probs.sum()) - 1.0) < 1e-6


def test_scores_rows_match_names():
    layout = _layout_terrain()
    rates = torch.rand(layout.num_items)
    curr = Sampler(
        SamplerCfg(
            strategies=[
                BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0),
                FrontierSamplingStrategyCfg(k=8, weight=2.0),
            ],
        ),
        layout,
    )
    scores = curr.scores(rates)
    assert scores.shape == (2, layout.num_items)
    assert curr.names == ["beta", "frontier"]


def test_negative_weight_clamped_to_zero():
    """Negative weights clamp to 0; the sampler becomes uniform via eps."""
    layout = _layout_terrain()
    rates = torch.rand(layout.num_items)
    pos_cfg = SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=1.0)], eps=0.0)
    neg_cfg = SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=-1.0)], eps=1e-6)
    pos = pos_cfg.class_type(pos_cfg, layout).probabilities(rates)
    neg = neg_cfg.class_type(neg_cfg, layout).probabilities(rates)
    assert torch.allclose(pos, neg, atol=1e-6)


def test_names_return_active_strategies_in_order():
    layout = _layout_terrain()
    curr = Sampler(
        SamplerCfg(
            strategies=[
                BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0),
                FrontierSamplingStrategyCfg(k=8, weight=2.0),
            ],
        ),
        layout,
    )
    assert curr.names == ["beta", "frontier"]


# ---------------------------------------------------------------------------
# SamplerCfg (blueprint -> runtime)
# ---------------------------------------------------------------------------


def test_sampler_cfg_build_produces_runtime_sampler():
    layout = _layout_terrain()
    cfg = SamplerCfg(
        strategies=[
            BetaSamplingStrategyCfg(target=0.5, kappa=2.0, weight=1.0),
            FrontierSamplingStrategyCfg(k=8, weight=1.5),
        ],
        eps=1e-3,
    )
    curr = cfg.class_type(cfg, layout)
    assert isinstance(curr, Sampler)
    assert curr.names == ["beta", "frontier"]
    assert torch.isfinite(curr.probabilities(torch.rand(layout.num_items))).all()


def test_sampler_cfg_default_rate_source_is_monitor():
    cfg = SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=1.0)])
    assert cfg.rate_source == "monitor"


def test_sampler_cfg_estimator_rate_source():
    cfg = SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=1.0)], rate_source="estimator")
    assert cfg.rate_source == "estimator"


def test_factory_slot_eq_item_layout_works():
    """target_index=None propagates through the cfg.build path."""
    layout = _layout_factory(num_states=64)
    rates = torch.rand(64)
    curr = Sampler(
        SamplerCfg(
            strategies=[
                BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0),
                FrontierSamplingStrategyCfg(k=8, dilation_steps=1, weight=2.0),
            ],
            eps=1e-3,
        ),
        layout,
    )
    probs = curr.probabilities(rates)
    assert torch.isfinite(probs).all()
    assert abs(float(probs.sum()) - 1.0) < 1e-6


def test_sampler_cfg_dilation_steps_propagates():
    """dilation_steps from FrontierSamplingStrategyCfg actually reaches the runtime signal."""
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
    cfg1 = SamplerCfg(
        strategies=[
            UniformSamplingStrategyCfg(weight=1.0),
            FrontierSamplingStrategyCfg(k=2, dilation_steps=1, weight=2.0),
        ],
        eps=1e-3,
    )
    cfg3 = SamplerCfg(
        strategies=[
            UniformSamplingStrategyCfg(weight=1.0),
            FrontierSamplingStrategyCfg(k=2, dilation_steps=3, weight=2.0),
        ],
        eps=1e-3,
    )
    p1 = Sampler(cfg1, layout).probabilities(rates)
    p3 = Sampler(cfg3, layout).probabilities(rates)
    threshold = 1.0 / n + 1e-6
    assert int((p3 > threshold).sum()) >= int((p1 > threshold).sum())


def test_warp_sampler_matches_torch_scores_and_probabilities():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Warp sampler backend.")

    layout_cpu = _layout_terrain(num_states=24, num_items=96, seed=3)
    layout_cuda = StateLayout(
        coords=layout_cpu.coords.cuda(),
        spawn_index=layout_cpu.spawn_index.cuda(),
        target_index=layout_cpu.target_index.cuda(),
    )
    rates_cpu = torch.rand(layout_cpu.num_items)
    rates_cuda = rates_cpu.cuda()

    strategies = [
        BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0),
        FrontierSamplingStrategyCfg(k=4, dilation_steps=1, weight=0.25),
        FrontierSamplingStrategyCfg(k=4, dilation_steps=2, weight=0.5),
        FrontierSamplingStrategyCfg(k=4, dilation_steps=5, weight=0.25),
        UniformSamplingStrategyCfg(weight=0.1),
    ]
    torch_cfg = SamplerCfg(strategies=strategies, eps=1e-3)
    warp_cfg = SamplerCfg(strategies=strategies, eps=1e-3, warp=True, max_samples=16)
    torch_sampler = torch_cfg.class_type(torch_cfg, layout_cpu)
    warp_sampler = warp_cfg.class_type(warp_cfg, layout_cuda)

    scores_t = torch_sampler.scores(rates_cpu)
    probs_t = torch_sampler.probabilities(rates_cpu)
    scores_w = warp_sampler.scores(rates_cuda).cpu()
    probs_w = warp_sampler.probabilities(rates_cuda).cpu()

    assert torch.allclose(scores_w, scores_t, atol=1e-6)
    assert torch.allclose(probs_w, probs_t, atol=1e-6)
    samples = warp_sampler.sample(warp_sampler.probabilities(rates_cuda), 16)
    assert samples.shape == (16,)
    assert samples.min() >= 0
    assert samples.max() < layout_cpu.num_items
    probs_g, samples_g = warp_sampler.probabilities_and_sample(rates_cuda, 16)
    assert torch.allclose(probs_g.cpu(), probs_t, atol=1e-6)
    assert samples_g.shape == (16,)
    assert samples_g.min() >= 0
    assert samples_g.max() < layout_cpu.num_items
