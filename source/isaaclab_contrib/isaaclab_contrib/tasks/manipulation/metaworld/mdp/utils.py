# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Torch ports of Meta-World's reward primitives.

These mirror :mod:`metaworld.utils.reward_utils` (the dm_control-derived
:func:`tolerance` and :func:`hamacher_product`) with the same numerics, but
batched: each input may be ``(B,)`` or scalar, and the output is the same
shape. The default sigmoid is ``"long_tail"`` which Meta-World V2 uses
exclusively.
"""

from __future__ import annotations

from typing import Literal

import torch

_DEFAULT_VALUE_AT_MARGIN = 0.1


SigmoidType = Literal["gaussian", "long_tail", "linear", "hyperbolic"]


def _sigmoid(x: torch.Tensor, value_at_1: float, sigmoid: SigmoidType) -> torch.Tensor:
    """Map ``x`` to ``[0, 1]`` with ``f(0) = 1`` and ``f(±1) = value_at_1``."""
    if sigmoid == "gaussian":
        scale = torch.sqrt(torch.tensor(-2.0 * torch.log(torch.tensor(value_at_1))))
        return torch.exp(-0.5 * (x * scale) ** 2)
    if sigmoid == "long_tail":
        scale = (1.0 / value_at_1 - 1.0) ** 0.5
        return 1.0 / ((x * scale) ** 2 + 1.0)
    if sigmoid == "linear":
        scale = 1.0 - value_at_1
        ramp = 1.0 - torch.abs(x) * scale
        return torch.clamp(ramp, min=0.0)
    if sigmoid == "hyperbolic":
        scale = torch.acosh(torch.tensor(1.0 / value_at_1))
        return 1.0 / torch.cosh(x * scale)
    raise ValueError(f"Unknown sigmoid type: {sigmoid!r}")


def tolerance(
    x: torch.Tensor,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float | torch.Tensor = 0.0,
    value_at_margin: float = _DEFAULT_VALUE_AT_MARGIN,
    sigmoid: SigmoidType = "gaussian",
) -> torch.Tensor:
    """Tolerance reward — 1 inside ``bounds``, decaying via ``sigmoid`` outside.

    Args:
        x: Input tensor [unitless], any shape.
        bounds: ``(lower, upper)`` interval [unitless] where the reward is 1.
        margin: Distance outside ``bounds`` at which reward equals
            ``value_at_margin`` [unitless]. Scalar or broadcastable tensor.
        value_at_margin: Reward value at ``margin`` distance from ``bounds``.
        sigmoid: Decay shape outside ``bounds``. Meta-World V2 uses ``"long_tail"``.

    Returns:
        Tensor of the same shape as ``x``, values in ``[0, 1]``.
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError(f"tolerance bounds must satisfy lower <= upper, got {bounds}")

    in_bounds = (x >= lower) & (x <= upper)
    if isinstance(margin, torch.Tensor):
        zero_margin = torch.where(margin > 0, torch.zeros_like(margin), torch.ones_like(margin))
        margin_eff = torch.where(margin > 0, margin, torch.ones_like(margin))
    else:
        zero_margin = 0.0 if margin > 0 else 1.0
        margin_eff = margin if margin > 0 else 1.0

    d = torch.where(x < lower, (lower - x) / margin_eff, (x - upper) / margin_eff)
    decayed = _sigmoid(d, value_at_margin, sigmoid)
    out_of_bounds_value = torch.where(in_bounds, torch.ones_like(decayed), decayed)
    # When margin <= 0, fall back to a hard 0/1 indicator.
    if isinstance(margin, torch.Tensor):
        return torch.where(zero_margin > 0, in_bounds.to(out_of_bounds_value.dtype), out_of_bounds_value)
    if zero_margin > 0:
        return in_bounds.to(out_of_bounds_value.dtype)
    return out_of_bounds_value


def hamacher_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamacher product ``a * b / (a + b - a * b)`` — soft AND on ``[0, 1]`` inputs.

    Args:
        a: First input tensor in ``[0, 1]``.
        b: Second input tensor in ``[0, 1]``, broadcastable to ``a``.

    Returns:
        Tensor with the same broadcast shape, values in ``[0, 1]``.
    """
    denom = a + b - a * b
    safe_denom = torch.where(denom > 1e-12, denom, torch.ones_like(denom))
    out = (a * b) / safe_denom
    return torch.where(denom > 1e-12, out, torch.zeros_like(out))
