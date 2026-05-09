# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""3D analog of :mod:`.scatter_2d` — colored point clouds for ``wandb.Object3D``.

Mirrors the 2D dashboard's contract one method to one output: each call to
:meth:`ScatterDashboard3D.to_object3d` turns one :class:`PanelSpec` into a
``[N_valid, 6]`` array (xyz + rgb) ready to drop into ``wandb.Object3D``.
Wandb's native 3D viewer handles rotate/zoom and the per-step history slider,
so cross-run comparison and step-through are inherited from the wandb panel
system without any extra wiring.

Aggregation reuses :func:`scatter_2d.aggregate_endpoints` — dimensionality- and
arity-agnostic, so "fixed-goal" (single endpoint) and "paired" (spawn/target)
curricula go through the same code path.
"""

from __future__ import annotations

import numpy as np

from .scatter_2d import PanelSpec


class ScatterDashboard3D:
    """Per-metric ``wandb.Object3D`` builder over a fixed 3D point layout.

    Geometry (per-point ``(x, y, z)``) is fixed at construction and lives in
    cached numpy. Each :meth:`to_object3d` call colormaps one panel's values
    onto those points and returns a wandb-ready array — the dashboard does
    no figure work and never touches matplotlib for layout.

    Args:
        positions: Per-point world ``(x, y, z)``, shape ``[N, 3]`` on CPU.
        valid_mask: Optional boolean mask shape ``[N]``; ``False`` rows are
            dropped from every output array. Defaults to all-True.

    Example:

    .. code-block:: python

        dashboard = ScatterDashboard3D(positions=keypoints_xyz)
        arr = dashboard.to_object3d(panel_success)
        wandb.log({"Sampler/success_3d": wandb.Object3D(arr)})
    """

    def __init__(
        self,
        positions: np.ndarray,
        *,
        valid_mask: np.ndarray | None = None,
    ):
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be [N, 3], got {tuple(positions.shape)}")
        self._xyz = np.ascontiguousarray(positions)
        self._n = positions.shape[0]
        self._valid = (
            np.ones(self._n, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).reshape(-1)
        )
        if self._valid.shape != (self._n,):
            raise ValueError(f"valid_mask must be [{self._n}], got {self._valid.shape}")

    @property
    def num_points(self) -> int:
        return self._n

    @property
    def num_valid(self) -> int:
        return int(self._valid.sum())

    def to_object3d(
        self,
        panel: PanelSpec,
        *,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build a ``wandb.Object3D``-shaped point cloud from one panel.

        For each valid point, samples ``panel.cmap`` at the normalized
        ``panel.values`` (clipped into ``[panel.vmin, panel.vmax]``) and
        emits one ``(x, y, z, r, g, b)`` row.

        Args:
            panel: Metric to render. Only ``values``, ``cmap``, ``vmin``,
                and ``vmax`` are read — ``legend_entries`` and
                ``stats_text`` aren't surfaced by wandb's 3D viewer, so
                they're ignored here.
            valid_mask: Optional per-frame override of the constructor
                mask. Validity often depends on which patches have been
                touched this frame, so callers pass the current frame's
                mask here instead of rebuilding the dashboard.

        Returns:
            ``[N_valid, 6]`` ``float32`` array. Columns 0-2 are world
            ``xyz``; columns 3-5 are RGB in ``[0, 255]``. Pass directly to
            ``wandb.Object3D`` (or stash under ``extras["log_object3d"]``
            and let your runner do the wrapping).
        """
        import matplotlib.pyplot as plt

        if panel.values.shape[0] != self._n:
            raise ValueError(
                f"panel '{panel.title}' values shape [{panel.values.shape[0]}] doesn't match num_points={self._n}"
            )
        valid = self._valid if valid_mask is None else np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid.shape != (self._n,):
            raise ValueError(f"valid_mask must be [{self._n}], got {valid.shape}")

        cmap = plt.get_cmap(panel.cmap)
        denom = max(panel.vmax - panel.vmin, 1e-12)
        normalized = np.clip((panel.values - panel.vmin) / denom, 0.0, 1.0)
        rgb = cmap(normalized)[:, :3] * 255.0
        return np.concatenate([self._xyz[valid], rgb[valid]], axis=1).astype(np.float32)
