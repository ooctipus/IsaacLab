# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnostic helpers for :class:`Curriculum`.

:func:`log_curriculum_bins` writes per-bin probability stats (bucketed
by one chosen signal) plus per-signal aggregate stats into ``log_dict``.
Consumed by both terrain and factory training loops; downstream wandb
panels read the dict keys directly.
"""

from __future__ import annotations

import torch

from .curriculum import Curriculum


def log_curriculum_bins(
    curriculum: Curriculum,
    *,
    success_rates: torch.Tensor,
    probs: torch.Tensor,
    log_dict: dict[str, float],
    step_counter: int,
    bin_signal: str = "frontier",
) -> None:
    """Bucket per-item ``probs`` by ``bin_signal``'s score; write to ``log_dict``.

    Writes two kinds of keys into ``log_dict``:

    - ``Frontier/bin_<label>_{count,mean_prob,mass,mean_self_s}`` -- the
      legacy frontier-bin breakdown, bucketed by ``bin_signal``'s raw
      score (defaults to ``frontier``). Existing wandb dashboards keep
      working unchanged.
    - ``Curriculum/<signal>/{mean,p90}`` -- per-signal aggregate stats
      so you can see each signal's strength and tail behavior over
      time.

    When ``bin_signal`` isn't present in the curriculum the binned
    section is skipped silently and only the per-signal stats are
    written.

    Args:
        curriculum: The active :class:`Curriculum`.
        success_rates: ``[num_items]`` per-item rates passed to the
            sampler this step.
        probs: ``[num_items]`` probabilities the sampler produced.
        log_dict: Mutable dict (typically ``env.extras["log"]``) that
            receives the diagnostic keys.
        step_counter: Iteration counter, retained as a parameter for
            call-site compatibility but no longer surfaced.
        bin_signal: ``signal.name`` to bucket by. Defaults to
            ``"frontier"`` for backward-compatible dashboards; pass any
            signal name present in the curriculum to bin by it instead.
    """
    del step_counter  # retained for call-site compatibility; no stdout output
    scores = curriculum.signal_scores(success_rates)

    # Per-signal aggregate stats -- always emitted.
    for name, score in scores.items():
        log_dict[f"Curriculum/{name}/mean"] = float(score.mean())
        log_dict[f"Curriculum/{name}/p90"] = float(score.quantile(0.9))

    if bin_signal not in scores:
        return  # no binned table when the chosen signal isn't present

    key = scores[bin_signal]
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
