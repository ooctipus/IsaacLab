# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``Curriculum``: weighted sum of informativeness signals over a state layout.

The curriculum's probability over items is::

    probs[i] = (Σ_s weight_s * signal_s.score(rates)[i] + eps) / Z

where ``Z`` normalizes globally.

Pattern: :class:`CurriculumCfg` is pure data with a ``class_type`` field
pointing at :class:`Curriculum`; :class:`Curriculum`'s ``__init__``
takes ``(cfg, layout)`` and constructs each runtime signal via the
same idiom (``sig_cfg.class_type(sig_cfg, layout)``). Consumers
instantiate with::

    curriculum = sampling_cfg.class_type(sampling_cfg, layout)
"""

from __future__ import annotations

from dataclasses import field
from typing import Any

import torch

from isaaclab.utils import configclass

from .sampling import InformativenessSignal
from .state_layout import StateLayout


class Curriculum:
    """Runtime weighted-sum composition of informativeness signals.

    Built from a :class:`CurriculumCfg` + :class:`StateLayout`. Holds
    runtime signals (with cached precomputation like the kNN graph) plus
    the global ``eps`` and rate-source selector.
    """

    def __init__(self, cfg: CurriculumCfg, layout: StateLayout) -> None:
        self.signals: list[tuple[InformativenessSignal, float]] = [
            (entry.cfg.class_type(entry.cfg, layout), float(entry.weight)) for entry in cfg.signals
        ]
        self.eps: float = float(cfg.eps)
        self.rate_source: str = cfg.rate_source

    def probabilities(self, success_rates: torch.Tensor) -> torch.Tensor:
        """Return ``[num_items]`` probability vector summing to 1."""
        w = torch.zeros_like(success_rates)
        for signal, weight in self.signals:
            w = w + max(0.0, weight) * signal.score(success_rates)
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
class SignalEntry:
    """Pairs a signal cfg with its weight in a :class:`CurriculumCfg`.

    A small wrapper around the ``(cfg, weight)`` pairing rather than a
    raw tuple. Reason: ``class_to_dict`` (the serialiser hydra uses to
    feed env cfgs into OmegaConf) does not recurse into tuple inputs
    -- it bails on tuples because they have no ``__dict__``. Wrapping
    each entry in a configclass forces recursion into the nested
    signal cfg so its ``class_type`` callable gets converted to a
    resolvable string before OmegaConf validates the tree.

    Attributes:
        cfg: A ``BetaSignalCfg`` / ``FrontierSignalCfg`` /
            ``UniformSignalCfg``. Annotated as ``Any`` so OmegaConf
            does not try to structurally validate the union.
        weight: Multiplier on the signal's score in the curriculum's
            weighted sum.
    """

    cfg: Any = None
    weight: float = 1.0


@configclass
class CurriculumCfg:
    """Blueprint for a :class:`Curriculum`.

    Attributes:
        class_type: Runtime class to instantiate. Subclasses can override.
        signals: List of :class:`SignalEntry` -- each pairs a signal
            blueprint with its weight in the weighted sum.
        eps: Soft floor on per-item probability; prevents
            zero-probability items so the success monitor keeps
            refreshing every item.
        rate_source: ``"monitor"`` or ``"estimator"`` -- per-consumer
            rate-source selector. Factory consumers route ``"monitor"``
            to ``self.monitor_success_rate`` and ``"estimator"`` to
            ``self.state_buffer.success_rates``; consumers with a single
            source (terrain, TermChoice) ignore this field.
    """

    class_type: type[Curriculum] | str = "{DIR}.curriculum:Curriculum"
    signals: list = field(default_factory=list)
    eps: float = 1e-3
    rate_source: str = "monitor"
