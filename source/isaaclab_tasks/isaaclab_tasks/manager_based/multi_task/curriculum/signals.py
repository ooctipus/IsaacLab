# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Informativeness signals for curriculum sampling.

Each signal scores per-item informativeness for policy improvement using
some proxy:

- :class:`BetaSignal` -- Bernoulli outcome variance peaked at a target
  success rate (high gradient magnitude proxy).
- :class:`FrontierSignal` -- spatial gradient over the state-pool kNN
  graph (high "neighbor learned, this state isn't" proxy).
- :class:`UniformSignal` -- constant 1.0 baseline / floor.

Pattern: each runtime class is paired with a ``*SignalCfg`` configclass
whose ``class_type`` field points at the runtime class. The cfg carries
all parameters; the runtime class's ``__init__(cfg, layout)`` reads
them. Construction follows the standard IsaacLab idiom:
``cfg.class_type(cfg, layout)`` -- no ``.build()`` method on cfgs, no
isinstance branching, no resolver helpers.

A :class:`Curriculum` (see :mod:`.curriculum`) composes any number of
signals into a single normalized probability distribution over items.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from isaaclab.utils import configclass

from .state_layout import StateLayout


class InformativenessSignal(Protocol):
    """Per-item informativeness scorer for curriculum sampling.

    A signal turns ``success_rates`` into a non-negative score per item;
    the curriculum sums weighted scores across signals and normalizes.
    """

    name: str
    """Short identifier used in diagnostic / log keys."""

    def score(self, success_rates: torch.Tensor) -> torch.Tensor:
        """Return ``[num_items]`` non-negative unnormalized scores."""
        ...


# ---------------------------------------------------------------------------
# Spatial helpers used by FrontierSignal
# ---------------------------------------------------------------------------


def build_knn_indices(coords: torch.Tensor, k: int) -> torch.Tensor:
    """Build a k-NN index table over a point set in arbitrary dimensions.

    Intended for use over the *state pool* (the pool of physically-valid
    state positions, e.g. ``task_table.spawn_states[:, :2]`` for terrain
    locomotion or ``held_asset.xyz`` relative to the goal pose for
    factory assembly), not over tasks. The state pool is the natural
    spatial domain regardless of how tasks combine endpoints; this
    function is therefore topology-agnostic.

    For each point the returned row contains the indices of its ``k``
    nearest other points (self excluded). When the pool has fewer than
    ``k+1`` points, missing slots are padded with the point's own index
    so frontier evaluations on those slots collapse to 0 without
    spurious cross-talk.

    Uses :class:`scipy.spatial.cKDTree`, so the cost is
    :math:`O(n \\log n)` and memory is :math:`O(n)`.

    Args:
        coords: ``[num_points, coord_dim]`` positions of the point set;
            ``coord_dim`` can be any positive size (2 for xy, 3 for xyz).
        k: Number of nearest neighbors to record per point.

    Returns:
        ``[num_points, k]`` long tensor of point-pool indices.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}.")
    from scipy.spatial import cKDTree

    num_points = int(coords.shape[0])
    device = coords.device
    self_idx = torch.arange(num_points, device=device, dtype=torch.long).unsqueeze(-1)
    knn = self_idx.expand(num_points, k).clone()
    if num_points <= 1:
        return knn

    coords_np = coords.detach().cpu().numpy()
    k_eff = min(k, num_points - 1)
    tree = cKDTree(coords_np)
    _, idx = tree.query(coords_np, k=k_eff + 1)
    idx = np.atleast_2d(idx)
    if idx.shape[0] != num_points:
        idx = idx.T
    knn_block = idx[:, 1 : k_eff + 1]
    if k_eff < k:
        pad = np.tile(np.arange(num_points, dtype=knn_block.dtype).reshape(-1, 1), (1, k - k_eff))
        knn_block = np.concatenate([knn_block, pad], axis=1)
    return torch.from_numpy(knn_block).to(device=device, dtype=torch.long)


def state_frontier_weights(
    success_rates: torch.Tensor,
    *,
    state_knn_indices: torch.Tensor,
    spawn_index: torch.Tensor,
    target_index: torch.Tensor | None = None,
    dilation_steps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-state success rates and per-state frontier weights.

    Per-state success ``state_s`` is the mean per-item success rate
    across all items touching the state in either endpoint role. The
    frontier weight is::

        state_frontier = (1 - state_s) * (s_dil - state_s).clamp_min(0)

    where ``s_dil`` is ``state_s`` propagated outward via
    ``dilation_steps`` graph-max iterations on ``state_knn_indices``.
    The ``(1 - state_s)`` factor downweights states that are themselves
    well-learned, even if a more-learned neighbor exists -- the frontier
    is "I'm not yet solved AND a neighbor is more solved", not just
    "a neighbor is more solved".

    Args:
        success_rates: ``[num_items]`` per-item rolling success rate.
        state_knn_indices: ``[num_states, k]`` long tensor from
            :func:`build_knn_indices` over the state pool coords.
        spawn_index: ``[num_items]`` per-item spawn state index.
        target_index: ``[num_items]`` per-item target state index, or
            ``None`` when items have no separate target endpoint
            (e.g. factory's slot==item case). With ``None`` the
            aggregation uses only ``spawn_index``.
        dilation_steps: Number of graph-max iterations.

    Returns:
        ``(state_s, state_frontier)`` -- both ``[num_states]`` on the
        same device/dtype as ``success_rates``.
    """
    n_states = int(state_knn_indices.shape[0])
    device = success_rates.device
    state_sums = torch.zeros(n_states, device=device, dtype=success_rates.dtype)
    state_counts = torch.zeros(n_states, device=device, dtype=success_rates.dtype)
    ones = torch.ones_like(success_rates)
    state_sums.scatter_add_(0, spawn_index, success_rates)
    state_counts.scatter_add_(0, spawn_index, ones)
    if target_index is not None:
        state_sums.scatter_add_(0, target_index, success_rates)
        state_counts.scatter_add_(0, target_index, ones)
    state_s = state_sums / state_counts.clamp_min(1.0)

    s_dil = state_s
    for _ in range(max(1, int(dilation_steps))):
        neighbor_max = s_dil[state_knn_indices].amax(dim=-1)
        s_dil = torch.maximum(s_dil, neighbor_max)
    state_frontier = (1.0 - state_s) * (s_dil - state_s).clamp_min(0.0)
    return state_s, state_frontier


# ---------------------------------------------------------------------------
# Signal classes
# ---------------------------------------------------------------------------


class BetaSignal:
    """Per-item Beta-kernel score peaked at a target success rate.

    High when the item's own rate is near :paramref:`BetaSignalCfg.target` --
    the regime where Bernoulli outcome variance and expected gradient
    magnitude are both highest. Independent of layout topology.
    """

    name = "beta"

    def __init__(self, cfg: BetaSignalCfg, layout: StateLayout) -> None:
        del layout  # unused -- Beta is layout-agnostic
        self._target = max(0.0, min(1.0, float(cfg.target)))
        self._kappa = max(0.0, float(cfg.kappa))
        self._eps = float(cfg.eps)
        self._a = 1.0 + self._kappa * self._target
        self._b = 1.0 + self._kappa * (1.0 - self._target)

    def score(self, success_rates: torch.Tensor) -> torch.Tensor:
        eps = self._eps
        return ((success_rates + eps).pow(self._a - 1.0) * (1.0 - success_rates + eps).pow(self._b - 1.0)).clamp_min(
            eps
        )


class FrontierSignal:
    """Per-item score from spatial gradient over a state-pool kNN graph.

    Aggregates per-item rates onto the state pool (via spawn + optional
    target endpoints), computes per-state frontier
    ``(1 - state_s) * (s_dilated - state_s).clamp_min(0)`` via graph
    max-pool over a kNN graph on ``layout.coords``, then sums
    above-mean-deviation back to per-item at the spawn endpoint plus
    (if present) target. Constant endpoints (e.g. shared targets)
    contribute zero and auto-cancel.

    The kNN graph is built once at construction; per-call cost is only
    the scatter / max-pool / deviation computation.
    """

    name = "frontier"

    def __init__(self, cfg: FrontierSignalCfg, layout: StateLayout) -> None:
        self._spawn_index = layout.spawn_index
        self._target_index = layout.target_index
        self._dilation_steps = max(1, int(cfg.dilation_steps))
        self._knn = build_knn_indices(layout.coords, k=cfg.k)

    def score(self, success_rates: torch.Tensor) -> torch.Tensor:
        state_frontier = self.state_frontier(success_rates)
        spawn_f = state_frontier[self._spawn_index]
        score = (spawn_f - spawn_f.mean()).clamp_min(0.0)
        if self._target_index is not None:
            target_f = state_frontier[self._target_index]
            score = score + (target_f - target_f.mean()).clamp_min(0.0)
        return score

    def state_frontier(self, success_rates: torch.Tensor) -> torch.Tensor:
        """Per-state frontier ``(1 - state_s) * (s_dil - state_s).clamp_min(0)``.

        Returns the un-aggregated ``[num_states]`` spatial signal --
        the intermediate that :meth:`score` then aggregates back to
        ``[num_items]`` via above-mean-deviation. Surfaced for
        state-pool diagnostics (terrain spawn scatter, factory wandb
        3D scatter) so consumers can show the spatial signal directly
        without recomputing it.
        """
        _, state_frontier = state_frontier_weights(
            success_rates,
            state_knn_indices=self._knn,
            spawn_index=self._spawn_index,
            target_index=self._target_index,
            dilation_steps=self._dilation_steps,
        )
        return state_frontier


class UniformSignal:
    """Constant 1.0 per item -- the trivial baseline / floor."""

    name = "uniform"

    def __init__(self, cfg: UniformSignalCfg, layout: StateLayout) -> None:
        del cfg, layout  # unused

    def score(self, success_rates: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(success_rates)


# ---------------------------------------------------------------------------
# Signal cfgs (pure data; ``class_type`` points at the runtime class)
# ---------------------------------------------------------------------------


@configclass
class BetaSignalCfg:
    """Blueprint for a :class:`BetaSignal`.

    ``class_type`` is annotated as ``type[BetaSignal] | str`` and given a
    ``"{DIR}.signals:BetaSignal"`` string value so that hydra /
    OmegaConf can serialise the cfg through ``OmegaConf.create``
    without rejecting the unsupported ``type[X]`` annotation. The
    ``ResolvableString`` wrapper is resolved to the runtime class by
    :func:`isaaclab.utils.configclass.validate` before consumers call
    ``cfg.class_type(cfg, layout)``.
    """

    class_type: type[BetaSignal] | str = "{DIR}.signals:BetaSignal"
    target: float = 0.66
    kappa: float = 1.0
    eps: float = 1e-3


@configclass
class FrontierSignalCfg:
    """Blueprint for a :class:`FrontierSignal`."""

    class_type: type[FrontierSignal] | str = "{DIR}.signals:FrontierSignal"
    k: int = 8
    dilation_steps: int = 1


@configclass
class UniformSignalCfg:
    """Blueprint for a :class:`UniformSignal`."""

    class_type: type[UniformSignal] | str = "{DIR}.signals:UniformSignal"


SignalCfg = BetaSignalCfg | FrontierSignalCfg | UniformSignalCfg
"""Discriminated union of signal cfg types. Used as the element type of
``CurriculumCfg.signals``."""
