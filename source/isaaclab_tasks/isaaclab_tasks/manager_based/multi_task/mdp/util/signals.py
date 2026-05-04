# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Informativeness signals for curriculum sampling.

Each signal scores per-item informativeness for policy improvement using
some proxy:

- :class:`BetaSignal` -- Bernoulli outcome variance peaked at a target
  success rate (high gradient magnitude proxy).
- :class:`FrontierSignal` -- spatial gradient over the state-pool kNN
  graph (high "neighbor learned, this state isn't" proxy).
- :class:`UniformSignal` -- constant 1.0 baseline / floor.

A :class:`WeightedCurriculum` (see :mod:`.curriculum`) composes any
number of signals into a single normalized probability distribution
over items. Signals receive their :class:`StateLayout` at construction
so per-call cost is just the score computation; per-call they consume
only ``success_rates``.
"""

from __future__ import annotations

from typing import Protocol

import torch

from .sampling import build_knn_indices, state_frontier_weights
from .state_layout import StateLayout


class InformativenessSignal(Protocol):
    """Per-item informativeness scorer for curriculum sampling.

    A signal turns ``success_rates`` into a non-negative score per item;
    the curriculum sums weighted scores across signals and normalizes.
    """

    name: str
    """Short identifier used in diagnostic / log keys."""

    def score(self, success_rates: torch.Tensor) -> torch.Tensor:
        """Return ``[num_items]`` non-negative unnormalized scores."""
        ...


class BetaSignal:
    """Per-item Beta-kernel score peaked at a target success rate.

    High when the item's own rate is near :paramref:`target` -- the
    regime where Bernoulli outcome variance and expected gradient
    magnitude are both highest. Independent of layout topology.

    Args:
        layout: State layout (unused by this signal; kept in the
            constructor so all signals share a uniform interface).
        target: Desired success-rate peak in ``[0, 1]``.
        kappa: Concentration around :paramref:`target`. Larger values
            sharpen the peak.
        eps: Soft kernel floor; prevents ``0`` ** ``negative`` when
            ``kappa * target < 1``. ``1e-3`` matches the legacy
            ``frontier_sampling_probs`` kernel; ``1e-8`` matches the
            legacy standalone ``beta_sampling_probs``.
    """

    name = "beta"

    def __init__(
        self,
        layout: StateLayout,
        *,
        target: float = 0.66,
        kappa: float = 1.0,
        eps: float = 1e-3,
    ) -> None:
        del layout  # unused -- Beta is layout-agnostic
        self._target = max(0.0, min(1.0, float(target)))
        self._kappa = max(0.0, float(kappa))
        self._eps = float(eps)
        self._a = 1.0 + self._kappa * self._target
        self._b = 1.0 + self._kappa * (1.0 - self._target)

    def score(self, success_rates: torch.Tensor) -> torch.Tensor:
        eps = self._eps
        return ((success_rates + eps).pow(self._a - 1.0) * (1.0 - success_rates + eps).pow(self._b - 1.0)).clamp_min(
            eps
        )


class FrontierSignal:
    """Per-item score from spatial gradient over a state-pool kNN graph.

    Aggregates per-item rates onto the state pool (via spawn + optional
    target endpoints), computes per-state frontier
    ``(1 - state_s) * (s_dilated - state_s).clamp_min(0)`` via graph
    max-pool over a kNN graph on ``layout.coords``, then sums
    above-mean-deviation back to per-item at the spawn endpoint plus
    (if present) target. Constant endpoints (e.g. shared targets under
    ``single_target_per_cell``) contribute zero and auto-cancel.

    The kNN graph is built once at construction from
    ``layout.coords``; per-call cost is only the scatter / max-pool /
    deviation computation.

    Args:
        layout: State layout providing ``coords`` and the ``spawn_index``
            / ``target_index`` mapping.
        k: Number of nearest neighbors per state.
        dilation_steps: Number of graph max-pool iterations.
    """

    name = "frontier"

    def __init__(
        self,
        layout: StateLayout,
        *,
        k: int = 8,
        dilation_steps: int = 1,
    ) -> None:
        self._spawn_index = layout.spawn_index
        self._target_index = layout.target_index
        self._dilation_steps = max(1, int(dilation_steps))
        self._knn = build_knn_indices(layout.coords, k=k)

    def score(self, success_rates: torch.Tensor) -> torch.Tensor:
        state_frontier = self.state_frontier(success_rates)
        spawn_f = state_frontier[self._spawn_index]
        score = (spawn_f - spawn_f.mean()).clamp_min(0.0)
        if self._target_index is not None:
            target_f = state_frontier[self._target_index]
            score = score + (target_f - target_f.mean()).clamp_min(0.0)
        return score

    def state_frontier(self, success_rates: torch.Tensor) -> torch.Tensor:
        """Per-state frontier ``(1 - state_s) * (s_dil - state_s).clamp_min(0)``.

        Returns the un-aggregated ``[num_states]`` spatial signal -- the
        intermediate that :meth:`score` then aggregates back to
        ``[num_items]`` via above-mean-deviation. Surfaced for state-
        pool diagnostics (terrain spawn scatter, factory wandb 3D
        scatter) so consumers can show the spatial signal directly
        without recomputing it.
        """
        _, state_frontier = state_frontier_weights(
            success_rates,
            state_knn_indices=self._knn,
            spawn_index=self._spawn_index,
            target_index=self._target_index,
            dilation_steps=self._dilation_steps,
        )
        return state_frontier


class UniformSignal:
    """Constant 1.0 per item -- the trivial baseline / floor.

    Used as the curriculum's per-item floor when no shape preference is
    desired (so only spatial-frontier or other signals differentiate
    items).

    Args:
        layout: State layout (unused except to record ``num_items``).
    """

    name = "uniform"

    def __init__(self, layout: StateLayout) -> None:
        del layout  # unused
        # success_rates' shape carries num_items at score time; nothing
        # to cache.

    def score(self, success_rates: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(success_rates)
