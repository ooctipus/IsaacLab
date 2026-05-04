# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnostic helpers for :class:`WeightedCurriculum`.

:func:`log_curriculum_bins` is the per-signal generalization of the
legacy ``log_frontier_bins``: it writes per-bin probability stats
(bucketed by one chosen signal) plus per-signal aggregate stats into
``log_dict`` and prints a tabular stdout dump. Used identically by
both terrain and factory consumers so the diagnostic stays in lockstep
with whatever the curriculum is doing.
"""

from __future__ import annotations

import torch

from .curriculum import WeightedCurriculum


def log_curriculum_bins(
    curriculum: WeightedCurriculum,
    *,
    success_rates: torch.Tensor,
    probs: torch.Tensor,
    log_dict: dict[str, float],
    step_counter: int,
    bin_signal: str = "frontier",
) -> None:
    """Bucket per-item ``probs`` by ``bin_signal``'s score; log + print.

    Writes two kinds of keys into ``log_dict``:

    - ``Frontier/bin_<label>_{count,mean_prob,mass,mean_self_s}`` -- the
      legacy frontier-bin breakdown, bucketed by ``bin_signal``'s raw
      score (defaults to ``frontier``). Existing wandb dashboards keep
      working unchanged.
    - ``Curriculum/<signal>/{mean,p90}`` -- per-signal aggregate stats
      so you can see each signal's strength and tail behavior over
      time.

    Stdout prints a per-signal summary line followed by the binned
    table (only when ``bin_signal`` is present in the curriculum). When
    a curriculum doesn't contain ``bin_signal``, the binned section is
    skipped silently and only the per-signal stats are emitted.

    Args:
        curriculum: The active :class:`WeightedCurriculum`.
        success_rates: ``[num_items]`` per-item rates passed to the
            sampler this step.
        probs: ``[num_items]`` probabilities the sampler produced.
        log_dict: Mutable dict (typically ``env.extras["log"]``) that
            receives the diagnostic keys.
        step_counter: Iteration counter for the stdout header.
        bin_signal: ``signal.name`` to bucket by. Defaults to
            ``"frontier"`` for backward-compatible dashboards; pass any
            signal name present in the curriculum to bin by it instead.
    """
    scores = curriculum.signal_scores(success_rates)

    # Per-signal aggregate stats -- always emitted.
    for name, score in scores.items():
        log_dict[f"Curriculum/{name}/mean"] = float(score.mean())
        log_dict[f"Curriculum/{name}/p90"] = float(score.quantile(0.9))

    total_p = float(probs.sum())
    print(
        f"[CURRICULUM DIAG] step={step_counter}  total_p={total_p:.3f}  signals=[{', '.join(curriculum.signal_names)}]",
        flush=True,
    )
    if scores:
        per_signal = "  ".join(
            f"{name}: mean={float(s.mean()):.3f} p90={float(s.quantile(0.9)):.3f}" for name, s in scores.items()
        )
        print(f"  per-signal:  {per_signal}", flush=True)

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
    rows = []
    for label, lo, hi in bins:
        mask = (key >= lo) & (key < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append((label, 0, 0.0, 0.0, 0.0, 0.0))
            continue
        mean_p = float(probs[mask].mean())
        mass = float(probs[mask].sum())
        mean_self_s = float(success_rates[mask].mean())
        min_key = float(key[mask].min())
        rows.append((label, n, mean_p, mass, mean_self_s, min_key))
        log_dict[f"Frontier/bin_{label}_count"] = float(n)
        log_dict[f"Frontier/bin_{label}_mean_prob"] = mean_p
        log_dict[f"Frontier/bin_{label}_mass"] = mass
        log_dict[f"Frontier/bin_{label}_mean_self_s"] = mean_self_s

    header = f"  {'bin':12s} {'count':>6s} {'mean_p':>10s} {'mass':>8s} {'mean_self_s':>11s} {'min_key':>8s}"
    print(f"  binned by {bin_signal}:", flush=True)
    print(header, flush=True)
    for label, n, mean_p, mass, mean_self_s, min_k in rows:
        print(
            f"  {label:12s} {n:6d} {mean_p:10.3e} {mass:8.3f} {mean_self_s:11.3f} {min_k:8.3f}",
            flush=True,
        )
