# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``WeightedCurriculum``: composes informativeness signals into a probability over items.

The curriculum's probability over items is::

    probs[i] = (Σ_s weight_s * signal_s.score(rates)[i] + eps) / Z

where ``Z`` normalizes globally. This is the user-facing object the
factory accumulator and the terrain curriculum both consume; it
replaces the legacy ``isinstance(cfg, BetaSamplingCfg | FrontierSamplingCfg
| UniformSamplingCfg)`` branching with one polymorphic call.

The :func:`make_curriculum` factory bridges the legacy cfg classes
(:class:`BetaSamplingCfg`, :class:`FrontierSamplingCfg`,
:class:`UniformSamplingCfg`) to a :class:`WeightedCurriculum` with the
matching ``eps`` so the new path produces numerically identical
probabilities to the legacy ``beta_sampling_probs`` /
``frontier_sampling_probs`` / ``uniform_sampling_probs`` functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .sampling_cfg import BetaSamplingCfg, FrontierSamplingCfg, UniformSamplingCfg
from .signals import BetaSignal, FrontierSignal, InformativenessSignal, UniformSignal
from .state_layout import StateLayout


@dataclass
class WeightedCurriculum:
    """Weighted-sum composition of informativeness signals.

    Attributes:
        signals: ``[(signal, weight), ...]`` -- each signal contributes
            ``weight * signal.score(rates)`` to the unnormalized item
            weights. Negative weights are clamped to 0.
        eps: Soft floor added to the summed weights before
            normalization; prevents zero-probability items so a
            success monitor keeps refreshing every item. Defaults to
            ``1e-3`` (matches the legacy frontier path); :func:`make_curriculum`
            overrides per legacy cfg type to keep numerical equivalence.
    """

    signals: list[tuple[InformativenessSignal, float]]
    eps: float = 1e-3

    def probabilities(self, success_rates: torch.Tensor) -> torch.Tensor:
        """Return ``[num_items]`` probability vector summing to 1."""
        w = torch.zeros_like(success_rates)
        for signal, weight in self.signals:
            w = w + max(0.0, float(weight)) * signal.score(success_rates)
        w = w + self.eps
        return w / w.sum()

    def signal_scores(self, success_rates: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-signal raw score, keyed by ``signal.name`` -- for diagnostics."""
        return {sig.name: sig.score(success_rates) for sig, _ in self.signals}

    @property
    def signal_names(self) -> list[str]:
        """Names of the active signals, in declaration order."""
        return [sig.name for sig, _ in self.signals]

    def find_signal(self, name: str) -> InformativenessSignal | None:
        """Return the first active signal with ``signal.name == name``, or ``None``.

        Used by diagnostics that want to surface a specific signal's
        internals (e.g. :meth:`FrontierSignal.state_frontier`) without
        having to know whether the curriculum was constructed from a
        Beta-only or Frontier cfg.
        """
        for sig, _ in self.signals:
            if sig.name == name:
                return sig
        return None


def make_curriculum(
    cfg: BetaSamplingCfg | FrontierSamplingCfg | UniformSamplingCfg,
    layout: StateLayout,
) -> WeightedCurriculum:
    """Bridge a legacy sampling cfg to a :class:`WeightedCurriculum`.

    Picks ``eps`` per legacy cfg so the resulting curriculum produces
    numerically identical probabilities to the corresponding legacy
    function. Specifically:

    - :class:`UniformSamplingCfg` -> single :class:`UniformSignal`,
      ``eps=0`` (matches ``uniform_sampling_probs`` exactly).
    - :class:`BetaSamplingCfg` -> single :class:`BetaSignal` with kernel
      ``eps=1e-8``, curriculum ``eps=1e-8`` (matches ``beta_sampling_probs``).
    - :class:`FrontierSamplingCfg` -> ``[base_signal, FrontierSignal]``
      with the base's eps and curriculum eps both set to ``cfg.eps``
      (matches ``frontier_sampling_probs``).

    Args:
        cfg: Legacy sampling cfg.
        layout: State layout the signals will operate over.

    Returns:
        A :class:`WeightedCurriculum` ready to call
        :meth:`~WeightedCurriculum.probabilities` on success rates.
    """
    if isinstance(cfg, FrontierSamplingCfg):
        if isinstance(cfg.base, BetaSamplingCfg):
            base_signal: InformativenessSignal = BetaSignal(
                layout,
                target=cfg.base.target,
                kappa=cfg.base.kappa,
                eps=cfg.eps,
            )
        elif isinstance(cfg.base, UniformSamplingCfg):
            base_signal = UniformSignal(layout)
        else:
            raise TypeError(
                f"Unsupported FrontierSamplingCfg.base '{type(cfg.base).__name__}'; "
                "expected BetaSamplingCfg or UniformSamplingCfg."
            )
        frontier = FrontierSignal(layout, k=cfg.k, dilation_steps=cfg.dilation_steps)
        return WeightedCurriculum(
            signals=[(base_signal, 1.0), (frontier, cfg.frontier_lambda)],
            eps=cfg.eps,
        )
    if isinstance(cfg, BetaSamplingCfg):
        return WeightedCurriculum(
            signals=[(BetaSignal(layout, target=cfg.target, kappa=cfg.kappa, eps=1e-8), 1.0)],
            eps=1e-8,
        )
    if isinstance(cfg, UniformSamplingCfg):
        return WeightedCurriculum(
            signals=[(UniformSignal(layout), 1.0)],
            eps=0.0,
        )
    raise TypeError(
        f"Unsupported sampling cfg '{type(cfg).__name__}'; expected "
        "BetaSamplingCfg, FrontierSamplingCfg, or UniformSamplingCfg."
    )
