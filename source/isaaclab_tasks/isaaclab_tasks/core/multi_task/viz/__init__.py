# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnostic visualization helpers for the multi-task package.

Split by data dimensionality, not by task family — terrain locomotion, factory
manipulation, and any future task share the same primitives:

- :mod:`.scatter_2d` — multi-panel matplotlib scatter dashboard over a 2D
  layout (one dot per point in world xy, per-panel colormap of a metric,
  optional background image). The terrain locomotion curriculum uses this
  directly; callers pass their own background (heightmap, blueprint, etc).
- :mod:`.scatter_3d` — 3D variant: per-panel ``[N, 6]`` (xyz + rgb) arrays
  for ``wandb.Object3D``. Wandb's native 3D viewer provides the rotate/zoom
  + history slider, so cross-run comparison falls out of the panel system
  with no extra wiring.

The shared :class:`PanelSpec` and :func:`aggregate_endpoints` work with both
dashboards. Endpoint arity is flexible: pass any number of endpoint index
tensors (single, pair, or N) so "fixed goal" and "paired endpoints"
curricula go through the same code path.
"""

from .scatter_2d import PanelSpec, ScatterDashboard2D, aggregate_endpoints
from .scatter_3d import ScatterDashboard3D

__all__ = [
    "PanelSpec",
    "ScatterDashboard2D",
    "ScatterDashboard3D",
    "aggregate_endpoints",
]
