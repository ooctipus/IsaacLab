# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``Curriculum``: weighted sum of informativeness signals over a state layout.

The curriculum's probability over items is::

    probs[i] = (Σ_s weight_s * signal_s.score(rates)[i] + eps) / Z

where ``Z`` normalizes globally. This is the user-facing object the
factory accumulator and the terrain curriculum both consume.

:class:`CurriculumCfg` is the blueprint: a list of signal cfgs +
weights + ``eps`` + an optional ``rate_source`` selector. Once a
:class:`StateLayout` is available, ``cfg.build(layout)`` returns a
runtime :class:`Curriculum`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from isaaclab.utils import configclass

from .signals import InformativenessSignal, SignalCfg
from .state_layout import StateLayout


@dataclass
class Curriculum:
    """Runtime weighted-sum composition of informativeness signals.

    Constructed via :meth:`CurriculumCfg.build`; consumers call
    :meth:`probabilities` per sample step. Holds runtime signals (with
    cached precomputation like the kNN graph) plus the global ``eps``.

    Attributes:
        signals: ``[(signal, weight), ...]`` -- each signal contributes
            ``weight * signal.score(rates)`` to the unnormalized item
            weights. Negative weights clamp to 0.
        eps: Soft floor added to summed weights before normalization;
            prevents zero-probability items.
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

        Used by diagnostics (e.g. the wandb 3D scatter) that surface a
        specific signal's internals without having to know which cfg
        the curriculum was built from.
        """
        for sig, _ in self.signals:
            if sig.name == name:
                return sig
        return None


@configclass
class CurriculumCfg:
    """Blueprint for a :class:`Curriculum`.

    Attributes:
        signals: ``[(signal_cfg, weight), ...]`` -- each entry pairs a
            signal blueprint (``BetaSignalCfg``, ``FrontierSignalCfg``,
            ``UniformSignalCfg``) with its weight in the weighted sum.
        eps: Soft floor on per-item probability; prevents
            zero-probability items so the success monitor keeps
            refreshing every item.
        rate_source: Per-consumer rate-source selector. Factory
            consumers route ``"monitor"`` to ``self.monitor_success_rate``
            and ``"estimator"`` to ``self.state_buffer.success_rates``;
            consumers with a single source (terrain, TermChoice) ignore
            this field.
    """

    signals: list[tuple[SignalCfg, float]] = field(default_factory=list)
    eps: float = 1e-3
    rate_source: Literal["monitor", "estimator"] = "monitor"

    def build(self, layout: StateLayout) -> Curriculum:
        return Curriculum(
            signals=[(sig_cfg.build(layout), w) for sig_cfg, w in self.signals],
            eps=self.eps,
        )
