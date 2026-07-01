# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Temporal operators used only during trajectory construction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gradient_time(values: torch.Tensor, step_seconds: float) -> torch.Tensor:
    """Differentiate axis one with first-order edges and a central interior."""
    if values.ndim < 2 or values.shape[1] < 2:
        raise ValueError("Time gradients require at least two samples on axis one.")
    result = torch.empty_like(values)
    result[:, 0] = (values[:, 1] - values[:, 0]) / step_seconds
    result[:, -1] = (values[:, -1] - values[:, -2]) / step_seconds
    if values.shape[1] > 2:
        result[:, 1:-1] = (values[:, 2:] - values[:, :-2]) / (2.0 * step_seconds)
    return result


def gaussian_filter_time(values: torch.Tensor, *, sigma: float = 2.0) -> torch.Tensor:
    """Filter axis one with a radius-4-sigma Gaussian and nearest-edge extension."""
    radius = round(4.0 * sigma)
    coordinate = torch.arange(-radius, radius + 1, dtype=torch.float64, device=values.device)
    kernel = torch.exp(-0.5 * (coordinate / sigma) ** 2)
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    output_dtype = values.dtype
    moved = values.to(torch.float64).movedim(1, -1)
    flattened = moved.reshape(-1, 1, moved.shape[-1])
    filtered = F.conv1d(F.pad(flattened, (radius, radius), mode="replicate"), kernel)
    return filtered.reshape(moved.shape).movedim(-1, 1).to(output_dtype)


__all__ = ["gaussian_filter_time", "gradient_time"]
