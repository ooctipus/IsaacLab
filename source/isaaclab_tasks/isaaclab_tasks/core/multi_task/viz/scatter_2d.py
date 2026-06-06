# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-panel scatter dashboard over a 2D point layout.

The dashboard renders ``M`` panels side-by-side; every panel shows the same
``N`` points (each a dot at a fixed ``(x, y)`` from the dashboard's geometry)
but colors them by a different per-point metric. Optional background image
(e.g. a top-down terrain heightmap) is shared across all panels.

Designed for periodic wandb logging: ``render()`` returns a single ``[H, W, 3]``
``uint8`` numpy array suitable for ``wandb.Image``. Cost is dominated by
matplotlib (~100-300 ms at 2k×1.8k); the dashboard does no GPU/CPU transfers
itself — values are passed in already on CPU. The companion helper
:func:`aggregate_endpoints` does the typical "reduce per-edge values to
per-point sums/counts" GPU pattern and is intended to run *before* render so
the periodic logging gate stays cheap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch


def _draw_panel(
    ax,
    panel: PanelSpec,
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    *,
    background_image: np.ndarray | None,
    background_extent: tuple[float, float, float, float] | None,
    background_cmap: str,
    background_alpha: float,
    dot_size: float,
    xlabel: str,
    ylabel: str | None,
) -> None:
    """Draw one :class:`PanelSpec` onto a matplotlib axis.

    Caller owns the figure layout — this helper just plots: optional
    background image, scatter, axis labels, legend, stats text. Same styling
    is applied uniformly so panels in a multi-cell figure stay in lock-step.
    """
    from matplotlib.lines import Line2D

    if background_image is not None:
        ax.imshow(
            background_image,
            extent=background_extent,
            origin="lower",
            cmap=background_cmap,
            alpha=background_alpha,
        )

    panel_valid = valid & np.isfinite(panel.values)
    ax.scatter(
        x[panel_valid],
        y[panel_valid],
        c=panel.values[panel_valid],
        s=dot_size,
        cmap=panel.cmap,
        vmin=panel.vmin,
        vmax=panel.vmax,
        edgecolors="none",
    )
    if panel.outline_mask is not None:
        outline_valid = valid & np.asarray(panel.outline_mask, dtype=bool).reshape(-1)
        ax.scatter(
            x[outline_valid],
            y[outline_valid],
            s=max(dot_size * 2.0, dot_size + 4.0),
            facecolors="none",
            edgecolors=panel.outline_color,
            linewidths=0.5,
        )

    ax.set_title(panel.title, fontsize=11)
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    if background_extent is not None:
        xmn, xmx, ymn, ymx = background_extent
        ax.set_xlim(xmn, xmx)
        ax.set_ylim(ymn, ymx)

    if panel.legend_entries:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,  # pyright: ignore[reportArgumentType]
                markersize=8,
                label=label,
            )
            for label, color in panel.legend_entries
        ]
        leg = ax.legend(
            handles=handles,
            loc="upper right",
            framealpha=0.85,
            fontsize=8,
            handlelength=1.2,
            borderpad=0.4,
        )
        leg.get_frame().set_edgecolor("0.5")

    if panel.stats_text:
        ax.text(
            0.02,
            0.98,
            panel.stats_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="0.5", alpha=0.85, pad=3),
        )


@dataclass
class PanelSpec:
    """One panel in a :class:`ScatterDashboard2D` render.

    Attributes:
        values: Per-point metric, shape ``[N]``, already on CPU.
        cmap: Matplotlib colormap name (e.g. ``"RdYlGn"``, ``"viridis"``).
        vmin: Lower bound of the colormap.
        vmax: Upper bound of the colormap.
        title: Panel title.
        legend_entries: List of ``(label, color)`` swatches for the legend.
            Three entries (top / mid / bottom of the colormap) is the
            convention; ``color`` is anything matplotlib accepts (RGBA
            tuple or hex).
        stats_text: Small text block (multi-line allowed) shown in the
            upper-left corner. Use for ``N=...``, ``mean=...``, etc.
        outline_mask: Optional boolean mask for points to ring on top of
            the colored scatter.
        outline_color: Matplotlib color for :attr:`outline_mask`.
    """

    values: np.ndarray
    cmap: str
    vmin: float
    vmax: float
    title: str
    legend_entries: list[tuple[str, object]] = field(default_factory=list)
    stats_text: str = ""
    outline_mask: np.ndarray | None = None
    outline_color: object = "black"


def aggregate_endpoints(
    values: torch.Tensor,
    endpoint_indices: Sequence[torch.Tensor],
    n_points: int,
    *,
    sums_buf: torch.Tensor | None = None,
    counts_buf: torch.Tensor | None = None,
    ones_buf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum-reduce ``values`` into ``[n_points]`` bins via ``scatter_add_``.

    For each tensor in ``endpoint_indices``, ``values[k]`` contributes to the
    bin ``endpoint_indices[i][k]``. Typical use: each command has a ``spawn``
    and a ``target`` patch index — passing both index tensors aggregates
    each command's metric into both endpoint patches symmetrically.

    Args:
        values: Per-edge metric, shape ``[E]``.
        endpoint_indices: Iterable of long-tensors each shape ``[E]``;
            element ``k`` of every tensor must be a valid index into
            ``[0, n_points)``.
        n_points: Number of output bins.
        sums_buf: Optional preallocated ``[n_points]`` buffer to reuse.
            Zeroed in place; created on ``values.device`` if ``None``.
        counts_buf: Same, for the count.
        ones_buf: Optional preallocated ``[E]`` tensor of ones with the
            same dtype/device as :paramref:`values`. Reusing it avoids a
            logging-time allocation for every panel aggregation.

    Returns:
        ``(sums, counts)`` — each shape ``[n_points]`` on
        ``values.device``. Computed as
        :math:`\\text{sums}_p = \\sum_{i} \\sum_{k:\\text{idx}_i[k]=p} \\text{values}[k]`
        and ``counts`` is the number of contributions to each bin.
    """
    if sums_buf is None:
        sums_buf = torch.zeros(n_points, device=values.device, dtype=values.dtype)
    else:
        sums_buf.zero_()
    if counts_buf is None:
        counts_buf = torch.zeros(n_points, device=values.device, dtype=values.dtype)
    else:
        counts_buf.zero_()
    if ones_buf is None:
        ones = torch.ones_like(values)
    else:
        if ones_buf.shape != values.shape:
            raise ValueError(f"ones_buf must have shape {tuple(values.shape)}, got {tuple(ones_buf.shape)}")
        if ones_buf.device != values.device or ones_buf.dtype != values.dtype:
            raise ValueError("ones_buf must have the same device and dtype as values")
        ones = ones_buf
    for idx in endpoint_indices:
        sums_buf.scatter_add_(0, idx, values)
        counts_buf.scatter_add_(0, idx, ones)
    return sums_buf, counts_buf


class ScatterDashboard2D:
    """Multi-panel scatter render over a fixed 2D point layout.

    Geometry (per-point ``(x, y)``) is fixed at construction so it can be
    cached once and reused on every ``render()`` call. The same applies to
    the optional background image — heightmap, schematic, etc. — which is
    rasterized once by the caller and passed in as a numpy array plus
    extent.

    Args:
        positions: Per-point world ``(x, y)``, shape ``[N, 2]`` on CPU.
        valid_mask: Optional boolean mask shape ``[N]``; points with
            ``False`` are skipped on every panel. Defaults to all-True.
        background_image: Optional ``[Hbg, Wbg]`` 2D array (e.g. heightmap)
            drawn under every panel via ``imshow``.
        background_extent: ``(xmin, xmax, ymin, ymax)`` for ``imshow``;
            also constrains the visible axis range so every panel shows
            the same area.
        background_cmap: Colormap for the background. Defaults to grayscale.
        background_alpha: Alpha for the background image.
    """

    def __init__(
        self,
        positions: np.ndarray,
        *,
        valid_mask: np.ndarray | None = None,
        background_image: np.ndarray | None = None,
        background_extent: tuple[float, float, float, float] | None = None,
        background_cmap: str = "gray",
        background_alpha: float = 0.7,
    ):
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"positions must be [N, 2], got {tuple(positions.shape)}")
        self._xy = np.ascontiguousarray(positions)
        self._n = positions.shape[0]
        self._valid = (
            np.ones(self._n, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).reshape(-1)
        )
        if self._valid.shape != (self._n,):
            raise ValueError(f"valid_mask must be [{self._n}], got {self._valid.shape}")
        self._bg_image = background_image
        self._bg_extent = background_extent
        self._bg_cmap = background_cmap
        self._bg_alpha = background_alpha

    @property
    def num_points(self) -> int:
        return self._n

    @property
    def num_valid(self) -> int:
        return int(self._valid.sum())

    def render(
        self,
        panels: Sequence[PanelSpec],
        *,
        valid_mask: np.ndarray | None = None,
        figsize: tuple[float, float] = (15, 13),
        dpi: int = 140,
        suptitle: str = "",
        dot_size: float | None = None,
        dot_size_data: float | None = None,
        dot_fill_ratio: float = 0.20,
    ) -> np.ndarray:
        """Render all panels into a single RGB image.

        Args:
            panels: Per-panel metric specs (left to right).
            valid_mask: Optional ``[N]`` bool override of the constructor
                mask. Validity often varies frame to frame (a patch only
                receives data once it's been sampled), so callers pass the
                current frame's mask here instead of rebuilding the
                dashboard.
            figsize: Matplotlib figsize in inches.
            dpi: Output dpi.
            suptitle: Optional figure-level title.
            dot_size: Scatter marker size in matplotlib ``s=`` units (points
                squared). Mutually exclusive with :paramref:`dot_size_data`.
                When set, overrides the fill-ratio default outright.
            dot_size_data: Dot diameter in *data units* (meters). Marker
                ``s`` is computed per-render from the panel's data extent
                so the same physical radius reads correctly across terrain
                sizes. Mutually exclusive with :paramref:`dot_size`.
            dot_fill_ratio: Default sizing: every dot is sized so that all
                visible dots collectively cover this fraction of the
                rendered axis area. Math: ``s = ratio × axis_area / N``
                (in points²). Auto-grows dots when ``N`` is small (sparse
                renders are visible) and auto-shrinks when ``N`` is large
                (dense renders don't blob). Default ``0.20`` (20% fill —
                color reads strongly even at small thumbnail sizes); lower
                toward ``0.05–0.10`` to emphasise position over color,
                raise toward ``0.30`` for dense-color heatmap-like reads.

        Resolution priority: ``dot_size`` → ``dot_size_data`` →
        ``dot_fill_ratio``.

        Returns:
            ``[H, W, 3]`` ``uint8`` array. Pass directly to
            ``wandb.Image(...)`` or ``PIL.Image.fromarray``.
        """
        import math

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if len(panels) == 0:
            raise ValueError("panels must contain at least one PanelSpec")
        if dot_size is not None and dot_size_data is not None:
            raise ValueError("dot_size and dot_size_data are mutually exclusive")

        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=figsize,
            dpi=dpi,
            layout="constrained",
            sharey=True,
            squeeze=False,
        )
        axes = axes[0]

        if valid_mask is None:
            valid = self._valid
        else:
            valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
            if valid.shape != (self._n,):
                raise ValueError(f"valid_mask must be [{self._n}], got {valid.shape}")

        # Resolve marker ``s``:
        #   1. explicit ``dot_size`` — wins outright (raw points²).
        #   2. ``dot_size_data`` — data-unit diameter, converted via figure / extent.
        #   3. ``dot_fill_ratio`` — sized so all visible dots cover that fraction
        #      of the rendered axis area. Auto-grows for sparse, shrinks for dense.
        if dot_size is None:
            panel_w_inches = figsize[0] / max(len(panels), 1)
            panel_h_inches = figsize[1]
            if self._bg_extent is not None:
                xmin, xmax, ymin, ymax = self._bg_extent
                extent_x = max(xmax - xmin, 1e-9)
                extent_y = max(ymax - ymin, 1e-9)
            else:
                extent_x = max(float(np.ptp(self._xy[:, 0])) if self._n > 0 else 1.0, 1e-9)
                extent_y = max(float(np.ptp(self._xy[:, 1])) if self._n > 0 else 1.0, 1e-9)
            # ``aspect="equal"`` constrains the axis to whichever dim's scale is
            # tighter; the actual rendered axis dimensions follow from there.
            scale_in_per_m = min(panel_w_inches / extent_x, panel_h_inches / extent_y)
            axis_w_in = scale_in_per_m * extent_x
            axis_h_in = scale_in_per_m * extent_y

            if dot_size_data is not None:
                # diameter [points] = diameter_data * (axis_w_in / extent_x) * 72
                # s [points²] = π × (diameter_points / 2)²
                diameter_points = dot_size_data * scale_in_per_m * 72.0
                dot_size = math.pi * (diameter_points / 2.0) ** 2
            else:
                # Fill-ratio: total dot area = ratio × axis area, split evenly across N.
                axis_area_points_sq = (axis_w_in * 72.0) * (axis_h_in * 72.0)
                n_visible = max(int(valid.sum()), 1)
                dot_size = max(dot_fill_ratio * axis_area_points_sq / n_visible, 1e-3)

        for i, (ax, panel) in enumerate(zip(axes, panels)):
            if panel.values.shape[0] != self._n:
                raise ValueError(
                    f"panel '{panel.title}' values shape [{panel.values.shape[0]}] doesn't match num_points={self._n}"
                )
            _draw_panel(
                ax,
                panel,
                self._xy[:, 0],
                self._xy[:, 1],
                valid,
                background_image=self._bg_image,
                background_extent=self._bg_extent,
                background_cmap=self._bg_cmap,
                background_alpha=self._bg_alpha,
                dot_size=dot_size,
                xlabel="World x [m]",
                ylabel="World y [m]" if i == 0 else None,
            )

        if suptitle:
            fig.suptitle(suptitle, fontsize=12)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()  # pyright: ignore[reportAttributeAccessIssue]
        plt.close(fig)
        return img
