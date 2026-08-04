# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`log_sampler_bins`."""

from __future__ import annotations

import torch

from isaaclab_tasks.core.multi_task.utils import (
    BetaSamplingStrategyCfg,
    FrontierSamplingStrategyCfg,
    Sampler,
    SamplerCfg,
    StateLayout,
    UniformSamplingStrategyCfg,
    log_sampler_bins,
)


def _layout_terrain(num_states: int = 50, num_items: int = 200, seed: int = 0) -> StateLayout:
    torch.manual_seed(seed)
    coords = torch.rand(num_states, 2) * 5.0
    spawn = torch.randint(0, num_states, (num_items,), dtype=torch.long)
    target = torch.randint(0, num_states, (num_items,), dtype=torch.long)
    return StateLayout(coords=coords, spawn_index=spawn, target_index=target)


def test_log_writes_per_signal_aggregate_keys():
    """Every active signal contributes Sampler/<name>/{mean,p90}."""
    layout = _layout_terrain()
    rates = torch.rand(layout.num_items)
    curriculum = Sampler(
        SamplerCfg(
            strategies=[
                BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0, success_rate_bind="success_rates"),
                FrontierSamplingStrategyCfg(k=8, dilation_steps=1, weight=2.0, success_rate_bind="success_rates"),
            ],
            eps=1e-3,
        ),
        layout,
        success_rates=rates,
    )
    probs = curriculum.probabilities()
    log: dict[str, float] = {}
    log_sampler_bins(curriculum, success_rates=rates, probs=probs, log_dict=log)
    assert "Sampler/beta/mean" in log
    assert "Sampler/beta/p90" in log
    assert "Sampler/frontier/mean" in log
    assert "Sampler/frontier/p90" in log


def test_log_writes_frontier_bin_keys_when_frontier_present():
    """Frontier bin breakdown emitted when curriculum contains a frontier signal."""
    layout = _layout_terrain()
    rates = torch.rand(layout.num_items)
    curriculum = Sampler(
        SamplerCfg(
            strategies=[
                UniformSamplingStrategyCfg(weight=1.0),
                FrontierSamplingStrategyCfg(k=8, dilation_steps=1, weight=2.0, success_rate_bind="success_rates"),
            ],
            eps=1e-3,
        ),
        layout,
        success_rates=rates,
    )
    probs = curriculum.probabilities()
    log: dict[str, float] = {}
    log_sampler_bins(curriculum, success_rates=rates, probs=probs, log_dict=log)
    bin_count_keys = [k for k in log if k.startswith("Frontier/bin_") and k.endswith("_count")]
    assert len(bin_count_keys) >= 1


def test_log_skips_bin_table_when_bin_strategy_absent():
    """When the requested bin_strategy isn't in the sampler, no Frontier/bin_* keys are written."""
    layout = _layout_terrain()
    rates = torch.rand(layout.num_items)
    curriculum = Sampler(
        SamplerCfg(
            strategies=[
                BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0, success_rate_bind="success_rates"),
            ],
            eps=1e-8,
        ),
        layout,
        success_rates=rates,
    )
    probs = curriculum.probabilities()
    log: dict[str, float] = {}
    log_sampler_bins(curriculum, success_rates=rates, probs=probs, log_dict=log)
    assert not any(k.startswith("Frontier/bin_") for k in log)
    assert "Sampler/beta/mean" in log


def test_log_bin_mass_sums_match_probs():
    """Sum of per-bin mass approximates the total probability mass (1.0)."""
    layout = _layout_terrain()
    rates = torch.rand(layout.num_items)
    curriculum = Sampler(
        SamplerCfg(
            strategies=[
                BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0, success_rate_bind="success_rates"),
                FrontierSamplingStrategyCfg(k=8, dilation_steps=1, weight=2.0, success_rate_bind="success_rates"),
            ],
            eps=1e-3,
        ),
        layout,
        success_rates=rates,
    )
    probs = curriculum.probabilities()
    log: dict[str, float] = {}
    log_sampler_bins(curriculum, success_rates=rates, probs=probs, log_dict=log)
    masses = [v for k, v in log.items() if k.startswith("Frontier/bin_") and k.endswith("_mass")]
    assert abs(sum(masses) - float(probs.sum())) < 1e-5


def test_log_bin_strategy_kwarg_selects_strategy():
    """bin_strategy='beta' should bucket by the Beta score instead of frontier."""
    layout = _layout_terrain()
    rates = torch.rand(layout.num_items)
    cfg = SamplerCfg(
        strategies=[BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0, success_rate_bind="success_rates")],
        eps=1e-3,
    )
    curriculum = cfg.class_type(cfg, layout, success_rates=rates)
    probs = curriculum.probabilities()
    log: dict[str, float] = {}
    log_sampler_bins(
        curriculum,
        success_rates=rates,
        probs=probs,
        log_dict=log,
        bin_strategy="beta",
    )
    assert any(k.startswith("Frontier/bin_") for k in log)


def test_log_uniform_only_curriculum_no_crash():
    """A trivial uniform-only curriculum should diagnose without error."""
    layout = _layout_terrain()
    rates = torch.rand(layout.num_items)
    cfg = SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=1.0)], eps=0.0)
    curriculum = cfg.class_type(cfg, layout, success_rates=rates)
    probs = curriculum.probabilities()
    log: dict[str, float] = {}
    log_sampler_bins(curriculum, success_rates=rates, probs=probs, log_dict=log)
    assert "Sampler/uniform/mean" in log
