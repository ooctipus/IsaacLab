# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..sampling_strategies import SamplingStrategy

if TYPE_CHECKING:
    from ...state_layout import StateLayout
    from ..sampler_cfg import SamplerCfg


class SamplerTorch:
    """Torch backend for weighted sampling strategies."""

    def __init__(self, cfg: SamplerCfg, layout: StateLayout) -> None:
        self.strategies: list[tuple[SamplingStrategy, float]] = [
            (strategy_cfg.class_type(strategy_cfg, layout), float(strategy_cfg.weight))
            for strategy_cfg in cfg.strategies
        ]
        self.eps = float(cfg.eps)
        self.rate_source = cfg.rate_source
        self.names = [strategy.name for strategy, _ in self.strategies]

    def scores(self, success_rates: torch.Tensor) -> torch.Tensor:
        """Return contiguous per-strategy score rows shaped ``[num_strategies, num_items]``."""
        out = torch.empty(
            (len(self.strategies), success_rates.shape[0]), device=success_rates.device, dtype=success_rates.dtype
        )
        for i, (strategy, _) in enumerate(self.strategies):
            strategy.score(success_rates, out[i])
        return out

    def probabilities(self, success_rates: torch.Tensor) -> torch.Tensor:
        """Return ``[num_items]`` probability vector summing to 1."""
        scores = self.scores(success_rates)
        w = torch.zeros_like(success_rates)
        for i, (_, weight) in enumerate(self.strategies):
            w.add_(scores[i], alpha=max(0.0, weight))
        w.add_(self.eps)
        return w / w.sum()

    def sample(self, probs: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Sample item indices from ``probs``."""
        return torch.multinomial(probs, num_samples, replacement=True)

    def probabilities_and_sample(
        self, success_rates: torch.Tensor, num_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return probabilities and sampled item indices."""
        probs = self.probabilities(success_rates)
        return probs, self.sample(probs, num_samples)
