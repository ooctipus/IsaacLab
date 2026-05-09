# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-selecting sampler over a fixed state layout."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .impl.sampler_torch import SamplerTorch
from .impl.sampler_warp import SamplerWarp

if TYPE_CHECKING:
    from ..state_layout import StateLayout
    from .sampler_cfg import SamplerCfg


class Sampler:
    """Backend-selecting weighted sampler built from sampling-strategy cfgs."""

    def __init__(self, cfg: SamplerCfg, layout: StateLayout) -> None:
        backend = SamplerWarp if cfg.warp else SamplerTorch
        self._impl = backend(cfg, layout)
        self.rate_source = self._impl.rate_source

    @property
    def names(self) -> list[str]:
        """Names of the active strategies, in declaration order."""
        return self._impl.names

    def scores(self, success_rates: torch.Tensor) -> torch.Tensor:
        """Return contiguous per-strategy score rows shaped ``[num_strategies, num_items]``."""
        return self._impl.scores(success_rates)

    def probabilities(self, success_rates: torch.Tensor) -> torch.Tensor:
        """Return ``[num_items]`` probability vector summing to 1."""
        return self._impl.probabilities(success_rates)

    def sample(self, probs: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Sample item indices from ``probs``."""
        return self._impl.sample(probs, num_samples)

    def probabilities_and_sample(
        self, success_rates: torch.Tensor, num_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return probabilities and sampled item indices."""
        return self._impl.probabilities_and_sample(success_rates, num_samples)
