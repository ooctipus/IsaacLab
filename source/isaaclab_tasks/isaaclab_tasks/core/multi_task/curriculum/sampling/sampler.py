# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-selecting sampler over a fixed state layout."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from .impl.sampler_kernels_warp import (
    STRATEGY_BETA,
    STRATEGY_FRONTIER,
    STRATEGY_UNIFORM,
)
from .impl.sampler_kernels_warp import (
    probabilities as probabilities_warp,
)
from .impl.sampler_kernels_warp import (
    sample as sample_warp,
)
from .impl.sampler_kernels_warp import (
    scores as scores_warp,
)
from .impl.sampler_kernels_warp import (
    update_frontier as update_frontier_warp,
)
from .impl.sampler_torch import SamplerTorch
from .sampling_strategies import BetaSamplingStrategy, FrontierSamplingStrategy
from .sampling_strategies_cfg import (
    BetaSamplingStrategyCfg,
    FrontierSamplingStrategyCfg,
    UniformSamplingStrategyCfg,
)

if TYPE_CHECKING:
    from ..state_layout import StateLayout
    from .sampler_cfg import SamplerCfg


def _task_features(layout: StateLayout) -> torch.Tensor:
    """Return task feature rows used for frontier locality ordering."""
    spawn_features = layout.coords[layout.spawn_index]
    if layout.target_index is None:
        return spawn_features
    target_features = layout.coords[layout.target_index]
    return torch.cat([spawn_features, target_features], dim=-1)


def _spatial_order(features: torch.Tensor, partition: torch.Tensor | None) -> torch.Tensor:
    """Sort item ids by a coarse spatial key, optionally grouped by partition."""
    num_items = int(features.shape[0])
    if num_items == 0:
        return torch.empty(0, device=features.device, dtype=torch.long)

    features = features.detach().to(dtype=torch.float32)
    dimension = int(features.shape[1])
    bits = max(1, min(10, 60 // max(dimension, 1)))
    bins = 1 << bits
    lower = features.amin(dim=0)
    span = (features.amax(dim=0) - lower).clamp_min(1.0e-12)
    quantized = (((features - lower) / span) * float(bins - 1)).clamp_(0, bins - 1).to(torch.int64)

    key = torch.zeros(num_items, device=features.device, dtype=torch.int64)
    for axis in range(dimension):
        key = key * bins + quantized[:, axis]
    order = torch.argsort(key, stable=True)
    if partition is not None:
        partition = partition.to(device=features.device, dtype=torch.long)
        order = order[torch.argsort(partition[order], stable=True)]
    return order


class Sampler:
    """Backend-selecting weighted sampler built from sampling-strategy cfgs.

    Each strategy binds its own runtime input signals via configclass
    ``*_bind`` fields; the caller (e.g. a curriculum term) supplies the
    binding namespace via ``**bind_ns`` keyword arguments. Conventional
    callers inject the env handle as ``env=...`` and a success-rate tensor
    as ``success_rates=...``.

    This composition root owns all Torch state and Torch-to-Warp views used
    by the Warp backend. The Warp implementation module only launches kernels
    over these caller-owned arrays.
    """

    def __init__(self, cfg: SamplerCfg, layout: StateLayout, **bind_ns) -> None:
        self._warp = bool(cfg.warp)
        if not self._warp:
            self._impl = SamplerTorch(cfg, layout, **bind_ns)
            self._names = self._impl.names
            self._plot_strategy_indices = self._impl._plot_strategy_indices
            self._weights = self._impl._weights
            self.eps = self._impl.eps
            return

        self._init_warp(cfg, layout, bind_ns)

    @property
    def names(self) -> list[str]:
        """Names of the active strategies, in declaration order."""
        return self._names

    def scores(self) -> torch.Tensor:
        """Return contiguous per-strategy score rows shaped ``[num_strategies, num_items]``."""
        if not self._warp:
            return self._impl.scores()
        self._update_frontier_warp()
        scores_warp(
            self._wp_success_rates,
            self._wp_score_rows,
            self._wp_strategy_kind,
            self._wp_beta_a,
            self._wp_beta_b,
            self._wp_frontier_ids,
            self._wp_frontier_results,
            self._num_strategies,
            self._num_items,
            self._device,
        )
        return self._score_rows

    def probabilities(self) -> torch.Tensor:
        """Return ``[num_items]`` probability vector summing to 1."""
        if not self._warp:
            return self._impl.probabilities()
        self._probabilities_warp()
        return self._probs

    def sample(self, probs: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Sample item indices from ``probs``."""
        if not self._warp:
            return self._impl.sample(probs, num_samples)
        if num_samples > self._samples.shape[0]:
            raise ValueError(
                f"Sampler.sample received {num_samples} samples, but max_samples={self._samples.shape[0]}."
            )
        self._validate_probabilities(probs)
        self._sample_warp(probs, num_samples)
        return self._samples[:num_samples]

    def probabilities_and_sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return probabilities and sampled item indices."""
        if not self._warp:
            return self._impl.probabilities_and_sample(num_samples)
        if num_samples > self._samples.shape[0]:
            raise ValueError(
                f"Sampler.probabilities_and_sample received {num_samples} samples, "
                f"but max_samples={self._samples.shape[0]}."
            )

        key = (self._success_rates.data_ptr(), int(num_samples))
        if self._graph is None or self._graph_key != key:
            with wp.ScopedCapture(device=self._device) as capture:
                self._probabilities_warp()
                self._sample_warp(self._probs, num_samples)
            self._graph = capture.graph
            self._graph_key = key

        wp.capture_launch(self._graph)
        return self._probs, self._samples[:num_samples]

    def _init_warp(self, cfg: SamplerCfg, layout: StateLayout, bind_ns: dict) -> None:
        supported_cfg_types = (BetaSamplingStrategyCfg, FrontierSamplingStrategyCfg, UniformSamplingStrategyCfg)
        for strategy_cfg in cfg.strategies:
            if not isinstance(strategy_cfg, supported_cfg_types):
                raise NotImplementedError(
                    f"Warp sampler does not implement strategy {type(strategy_cfg).__name__};"
                    " use the Torch backend (set ``warp=False`` on SamplerCfg)."
                )

        wp.init()
        self.eps = float(cfg.eps)
        self._seed = int(cfg.seed)
        self._names: list[str] = []
        self._plot_strategy_indices = [i for i, strategy_cfg in enumerate(cfg.strategies) if strategy_cfg.plot]
        self._num_items = int(layout.spawn_index.shape[0])

        success_rate_binds = {
            strategy_cfg.success_rate_bind
            for strategy_cfg in cfg.strategies
            if isinstance(strategy_cfg, (BetaSamplingStrategyCfg, FrontierSamplingStrategyCfg))
        }
        if len(success_rate_binds) > 1:
            raise ValueError(
                f"Warp sampler requires a single shared success_rate_bind; got {sorted(success_rate_binds)}."
            )
        if success_rate_binds:
            (bind_expression,) = success_rate_binds
            success_rates = eval(bind_expression, bind_ns)  # noqa: S307
            if not isinstance(success_rates, torch.Tensor):
                raise TypeError("Warp sampler success_rate_bind must resolve to a torch.Tensor.")
            self._success_rates = success_rates
        else:
            self._success_rates = torch.zeros(self._num_items, device=layout.coords.device)

        if self._success_rates.dtype != torch.float32:
            raise TypeError(f"Warp sampler requires float32 success rates, got {self._success_rates.dtype}.")
        if self._success_rates.device != layout.coords.device:
            raise ValueError(
                f"Warp sampler success rates must be on device {layout.coords.device}, "
                f"got {self._success_rates.device}."
            )
        if self._success_rates.shape != (self._num_items,):
            raise ValueError(
                f"Warp sampler success rates must have shape ({self._num_items},), "
                f"got {tuple(self._success_rates.shape)}."
            )
        if not self._success_rates.is_contiguous():
            raise ValueError("Warp sampler requires contiguous success rates.")

        kinds: list[int] = []
        weights: list[float] = []
        beta_a: list[float] = []
        beta_b: list[float] = []
        frontier_ids: list[int] = []
        frontier_group_by_k: dict[int, int] = {}
        frontier_group_knn: list[torch.Tensor] = []
        frontier_group_k: list[int] = []
        frontier_group_max_dilation_steps: list[int] = []
        frontier_result_by_group_step: dict[tuple[int, int], int] = {}

        device = layout.coords.device
        frontier_order = _spatial_order(_task_features(layout), layout.task_partition)
        frontier_inverse = torch.empty_like(frontier_order)
        frontier_inverse[frontier_order] = torch.arange(self._num_items, device=device, dtype=torch.long)

        for strategy_cfg in cfg.strategies:
            weights.append(float(strategy_cfg.weight))
            if isinstance(strategy_cfg, BetaSamplingStrategyCfg):
                strategy = BetaSamplingStrategy(strategy_cfg, layout, **bind_ns)
                self._names.append(strategy.name)
                kinds.append(STRATEGY_BETA)
                beta_a.append(float(strategy._a))
                beta_b.append(float(strategy._b))
                frontier_ids.append(-1)
            elif isinstance(strategy_cfg, FrontierSamplingStrategyCfg):
                strategy = FrontierSamplingStrategy(strategy_cfg, layout, **bind_ns)
                k = int(strategy_cfg.k)
                dilation_steps = int(strategy._dilation_steps)
                group_id = frontier_group_by_k.get(k)
                if group_id is None:
                    group_id = len(frontier_group_knn)
                    frontier_group_by_k[k] = group_id
                    frontier_group_knn.append(strategy._knn.to(dtype=torch.int64).contiguous())
                    frontier_group_k.append(k)
                    frontier_group_max_dilation_steps.append(dilation_steps)
                else:
                    frontier_group_max_dilation_steps[group_id] = max(
                        frontier_group_max_dilation_steps[group_id], dilation_steps
                    )

                self._names.append(strategy.name)
                kinds.append(STRATEGY_FRONTIER)
                beta_a.append(1.0)
                beta_b.append(1.0)
                result_key = (group_id, dilation_steps)
                result_id = frontier_result_by_group_step.get(result_key)
                if result_id is None:
                    result_id = len(frontier_result_by_group_step)
                    frontier_result_by_group_step[result_key] = result_id
                frontier_ids.append(result_id)
            elif isinstance(strategy_cfg, UniformSamplingStrategyCfg):
                strategy = strategy_cfg.class_type(strategy_cfg, layout, **bind_ns)
                self._names.append(strategy.name)
                kinds.append(STRATEGY_UNIFORM)
                beta_a.append(1.0)
                beta_b.append(1.0)
                frontier_ids.append(-1)

        self._num_strategies = len(kinds)
        self._num_frontier_groups = len(frontier_group_knn)
        self._num_frontier_results = len(frontier_result_by_group_step)
        self._max_k = max(frontier_group_k) if frontier_group_k else 1
        self._max_dilation_steps = max(frontier_group_max_dilation_steps) if frontier_group_max_dilation_steps else 0

        self._strategy_kind = torch.tensor(kinds, device=device, dtype=torch.int32)
        self._weights = torch.tensor(weights, device=device, dtype=torch.float32)
        self._beta_a = torch.tensor(beta_a, device=device, dtype=torch.float32)
        self._beta_b = torch.tensor(beta_b, device=device, dtype=torch.float32)
        self._frontier_ids = torch.tensor(frontier_ids, device=device, dtype=torch.int32)
        self._frontier_k = torch.tensor(frontier_group_k or [1], device=device, dtype=torch.int32)
        self._frontier_group_max_dilation_steps = torch.tensor(
            frontier_group_max_dilation_steps or [0], device=device, dtype=torch.int32
        )
        self._frontier_order = frontier_order.to(dtype=torch.int32)
        frontier_result_for_step = torch.full(
            (max(self._num_frontier_groups, 1), max(self._max_dilation_steps, 1)),
            -1,
            device=device,
            dtype=torch.int32,
        )
        for (group_id, dilation_steps), result_id in frontier_result_by_group_step.items():
            frontier_result_for_step[group_id, dilation_steps - 1] = result_id
        self._frontier_result_for_step = frontier_result_for_step

        if frontier_group_knn:
            frontier_knn = torch.empty(
                (self._num_frontier_groups, self._num_items, self._max_k), device=device, dtype=torch.int32
            )
            for group_id, indices in enumerate(frontier_group_knn):
                k = int(indices.shape[1])
                internal_knn = frontier_inverse[indices[frontier_order].to(dtype=torch.long)].to(dtype=torch.int32)
                frontier_knn[group_id, :, :k] = internal_knn
                if k < self._max_k:
                    self_indices = torch.arange(self._num_items, device=device, dtype=torch.int32).unsqueeze(-1)
                    frontier_knn[group_id, :, k:] = self_indices.expand(self._num_items, self._max_k - k)
        else:
            frontier_knn = torch.zeros((1, self._num_items, 1), device=device, dtype=torch.int32)
        self._frontier_knn = frontier_knn

        self._score_rows = torch.empty((self._num_strategies, self._num_items), device=device, dtype=torch.float32)
        self._weighted = torch.empty(self._num_items, device=device, dtype=torch.float32)
        self._probs = torch.empty(self._num_items, device=device, dtype=torch.float32)
        self._cdf = torch.empty(self._num_items, device=device, dtype=torch.float32)
        self._sum = torch.empty(1, device=device, dtype=torch.float32)
        self._frontier_prev = torch.empty(
            (max(self._num_frontier_groups, 1), self._num_items), device=device, dtype=torch.float32
        )
        self._frontier_next = torch.empty_like(self._frontier_prev)
        self._frontier_results = torch.empty(
            (max(self._num_frontier_results, 1), self._num_items), device=device, dtype=torch.float32
        )
        max_samples = int(cfg.max_samples) if cfg.max_samples is not None else 1
        self._samples = torch.empty(max_samples, device=device, dtype=torch.int64)
        self._sample_counter = torch.zeros(1, device=device, dtype=torch.int64)
        self._sample_base = torch.zeros(1, device=device, dtype=torch.int64)
        self._graph = None
        self._graph_key: tuple[int, int] | None = None
        self._device = str(device)

        self._wp_success_rates = wp.from_torch(self._success_rates, dtype=wp.float32)
        self._wp_strategy_kind = wp.from_torch(self._strategy_kind, dtype=wp.int32)
        self._wp_weights = wp.from_torch(self._weights, dtype=wp.float32)
        self._wp_beta_a = wp.from_torch(self._beta_a, dtype=wp.float32)
        self._wp_beta_b = wp.from_torch(self._beta_b, dtype=wp.float32)
        self._wp_frontier_ids = wp.from_torch(self._frontier_ids, dtype=wp.int32)
        self._wp_frontier_k = wp.from_torch(self._frontier_k, dtype=wp.int32)
        self._wp_frontier_group_max_dilation_steps = wp.from_torch(
            self._frontier_group_max_dilation_steps, dtype=wp.int32
        )
        self._wp_frontier_order = wp.from_torch(self._frontier_order, dtype=wp.int32)
        self._wp_frontier_result_for_step = wp.from_torch(self._frontier_result_for_step, dtype=wp.int32)
        self._wp_frontier_knn = wp.from_torch(self._frontier_knn, dtype=wp.int32)
        self._wp_score_rows = wp.from_torch(self._score_rows, dtype=wp.float32)
        self._wp_weighted = wp.from_torch(self._weighted, dtype=wp.float32)
        self._wp_probs = wp.from_torch(self._probs, dtype=wp.float32)
        self._wp_cdf = wp.from_torch(self._cdf, dtype=wp.float32)
        self._wp_sum = wp.from_torch(self._sum, dtype=wp.float32)
        self._wp_frontier_prev = wp.from_torch(self._frontier_prev, dtype=wp.float32)
        self._wp_frontier_next = wp.from_torch(self._frontier_next, dtype=wp.float32)
        self._wp_frontier_results = wp.from_torch(self._frontier_results, dtype=wp.float32)
        self._wp_sample_counter = wp.from_torch(self._sample_counter, dtype=wp.int64)
        self._wp_sample_base = wp.from_torch(self._sample_base, dtype=wp.int64)
        self._wp_samples = wp.from_torch(self._samples, dtype=wp.int64)

    def _update_frontier_warp(self) -> None:
        update_frontier_warp(
            self._wp_success_rates,
            self._wp_frontier_order,
            self._wp_frontier_result_for_step,
            self._wp_frontier_knn,
            self._wp_frontier_k,
            self._wp_frontier_group_max_dilation_steps,
            self._wp_frontier_prev,
            self._wp_frontier_next,
            self._wp_frontier_results,
            self._num_items,
            self._num_frontier_groups,
            self._max_dilation_steps,
            self._max_k,
            self._device,
        )

    def _probabilities_warp(self) -> None:
        self._update_frontier_warp()
        probabilities_warp(
            self._wp_success_rates,
            self._wp_weighted,
            self._wp_probs,
            self._wp_sum,
            self._wp_strategy_kind,
            self._wp_weights,
            self._wp_beta_a,
            self._wp_beta_b,
            self._wp_frontier_ids,
            self._wp_frontier_results,
            self.eps,
            self._num_strategies,
            self._num_items,
            self._device,
        )

    def _sample_warp(self, probs: torch.Tensor, num_samples: int) -> None:
        wp_probs = (
            self._wp_probs if probs.data_ptr() == self._probs.data_ptr() else wp.from_torch(probs, dtype=wp.float32)
        )
        sample_warp(
            wp_probs,
            self._wp_cdf,
            self._wp_samples,
            self._wp_sample_counter,
            self._wp_sample_base,
            num_samples,
            self._num_items,
            self._seed,
            self._device,
        )

    def _validate_probabilities(self, probs: torch.Tensor) -> None:
        if probs.dtype != torch.float32:
            raise TypeError(f"Warp sampler probabilities must have dtype torch.float32, got {probs.dtype}.")
        if probs.device != self._probs.device:
            raise ValueError(f"Warp sampler probabilities must be on device {self._probs.device}, got {probs.device}.")
        if probs.shape != (self._num_items,):
            raise ValueError(
                f"Warp sampler probabilities must have shape ({self._num_items},), got {tuple(probs.shape)}."
            )
        if not probs.is_contiguous():
            raise ValueError("Warp sampler requires contiguous probabilities.")
