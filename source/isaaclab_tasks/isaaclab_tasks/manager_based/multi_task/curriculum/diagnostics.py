# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnostic helpers for :class:`Sampler`.

:func:`log_sampler_bins` writes per-bin probability stats (bucketed by
one chosen strategy) plus per-strategy aggregate stats into ``log_dict``.
Consumed by both terrain and factory training loops; downstream wandb
panels read the dict keys directly.
"""

from __future__ import annotations

import torch

from .sampling import Sampler


def log_sampler_bins(
    sampler: Sampler,
    *,
    success_rates: torch.Tensor,
    probs: torch.Tensor,
    log_dict: dict[str, float],
    bin_strategy: str = "frontier",
) -> None:
    """Bucket per-item ``probs`` by ``bin_strategy``'s score; write to ``log_dict``.

    Writes two kinds of keys into ``log_dict``:

    - ``Frontier/bin_<label>_{count,mean_prob,mass,mean_self_s}`` -- the
      frontier-bin breakdown, bucketed by ``bin_strategy``'s raw score.
    - ``Sampler/<strategy>/{mean,p90}`` -- per-strategy aggregate stats
      so you can see each strategy's strength and tail behavior over
      time.

    When ``bin_strategy`` isn't present in the sampler the binned
    section is skipped silently and only the per-strategy stats are
    written.

    Args:
        sampler: The active :class:`Sampler`.
        success_rates: ``[num_items]`` per-item rates passed to the
            sampler this step.
        probs: ``[num_items]`` probabilities the sampler produced.
        log_dict: Mutable dict (typically ``env.extras["log"]``) that
            receives the diagnostic keys.
        bin_strategy: ``strategy.name`` to bucket by. Defaults to
            ``"frontier"``; pass any strategy name present in the
            sampler to bin by it instead.
    """
    scores = sampler.scores(success_rates)
    names = sampler.names
    # Per-strategy aggregate stats -- always emitted.
    for i, name in enumerate(names):
        score = scores[i]
        log_dict[f"Sampler/{name}/mean"] = float(score.mean())
        log_dict[f"Sampler/{name}/p90"] = float(score.quantile(0.9))

    if bin_strategy not in names:
        return  # no binned table when the chosen strategy isn't present

    key = scores[names.index(bin_strategy)]
    bins = [
        ("ftr<0.01", 0.0, 0.01),
        ("0.01-0.05", 0.01, 0.05),
        ("0.05-0.20", 0.05, 0.20),
        ("0.20-0.50", 0.20, 0.50),
        ("0.50+", 0.50, float("inf")),
    ]
    for label, lo, hi in bins:
        mask = (key >= lo) & (key < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        log_dict[f"Frontier/bin_{label}_count"] = float(n)
        log_dict[f"Frontier/bin_{label}_mean_prob"] = float(probs[mask].mean())
        log_dict[f"Frontier/bin_{label}_mass"] = float(probs[mask].sum())
        log_dict[f"Frontier/bin_{label}_mean_self_s"] = float(success_rates[mask].mean())
