# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


def beta_sampling_probs(
    success_rates: torch.Tensor,
    target: float = 0.5,
    kappa: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert per-slot success rates into sampling probabilities peaked at ``target``.

    Uses a Beta-distribution kernel: slots whose success rate is near ``target``
    receive the highest weight. ``kappa`` controls concentration.

    ``eps`` doubles as a soft floor on the per-slot weight: the kernel
    evaluates ``(rate + eps)^(a-1) * (1 - rate + eps)^(b-1)``, so slots at
    the extremes (``rate = 0`` or ``1``) get weight ``eps^(b-1)`` /
    ``eps^(a-1)`` rather than 0. With the default ``eps = 1e-8`` extreme
    slots are ~``10⁶×`` less likely than ``target``, which effectively
    abandons them — their monitor window goes stale and policy regression
    on those slots is silently masked. Raise ``eps`` (e.g. ``0.01``-``0.05``)
    to flatten the tail and guarantee every slot keeps refreshing.
    """
    t = max(0.0, min(1.0, target))
    k = max(0.0, kappa)
    a = 1.0 + k * t
    b = 1.0 + k * (1.0 - t)
    w = ((success_rates + eps).pow(a - 1.0) * (1.0 - success_rates + eps).pow(b - 1.0)).clamp_min(eps)
    return torch.softmax(torch.log(w + eps), dim=0)


def uniform_sampling_probs(success_rates: torch.Tensor) -> torch.Tensor:
    """Return a uniform sampling distribution over slots.

    Every slot gets equal probability ``1 / N`` regardless of its success rate.
    Useful as a non-curriculum baseline or when the goal is to refresh the
    monitor window evenly across all slots.
    """
    n = success_rates.numel()
    return torch.full_like(success_rates, 1.0 / float(n))


def tagged_report(
    values: torch.Tensor,
    tags: torch.Tensor,
    tag_names: list[str],
    reduction: str = "sum",
) -> dict[str, float]:
    """Aggregate per-slot values by tag.

    Args:
        values: Per-slot tensor to aggregate.
        tags: Per-slot tag IDs (int, -1 = untagged).
        tag_names: Human-readable name for each tag ID.
        reduction: ``"sum"`` for probability mass, ``"mean"`` for averages.
    """
    out: dict[str, float] = {}
    for i, name in enumerate(tag_names):
        mask = tags == i
        if not mask.any():
            out[name] = 0.0
        elif reduction == "mean":
            out[name] = values[mask].mean().item()
        else:
            out[name] = values[mask].sum().item()
    return out
