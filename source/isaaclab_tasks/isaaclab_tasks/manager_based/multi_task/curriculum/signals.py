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


def task_frontier_weights(
    success_rates: torch.Tensor,
    *,
    task_knn_indices: torch.Tensor,
    dilation_steps: int = 1,
) -> torch.Tensor:
    """Compute per-task frontier weights from per-task rates and a task kNN graph.

    The frontier weight is::

        task_frontier = (1 - s) * (s_dil - s).clamp_min(0)

    where ``s_dil`` is ``success_rates`` propagated outward via
    ``dilation_steps`` graph-max iterations on ``task_knn_indices``.
    The ``(1 - s)`` factor downweights tasks that are themselves
    already learned, even if a more-learned neighbor exists -- the
    frontier is "I'm not yet solved AND a similar task is more
    solved", not just "a similar task is more solved".

    Working at task-level (rather than aggregating per-task rates onto
    states first) preserves task identity end-to-end. There is no
    free-rider pathology where a learned target inherits sampling
    pressure onto unrelated spawns paired with it -- two tasks share
    propagation only if they are actually neighbors in
    ``(spawn, target)`` feature space.

    Args:
        success_rates: ``[num_items]`` per-task rolling success rate.
        task_knn_indices: ``[num_items, k]`` long tensor from
            :func:`build_knn_indices` over per-task feature vectors
            (e.g. ``concat(spawn_xy, target_xy)`` for terrain or
            ``slot_xyz`` for factory's slot==item topology).
        dilation_steps: Number of graph-max iterations along the task
            kNN graph.

    Returns:
        ``[num_items]`` task frontier weights on the same device/dtype
        as ``success_rates``.
    """
    s_dil = success_rates
    for _ in range(max(1, int(dilation_steps))):
        neighbor_max = s_dil[task_knn_indices].amax(dim=-1)
        s_dil = torch.maximum(s_dil, neighbor_max)
    return (1.0 - success_rates) * (s_dil - success_rates).clamp_min(0.0)


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
        self._a = 1.0 + self._kappa * self._target
        self._b = 1.0 + self._kappa * (1.0 - self._target)

    def score(self, success_rates: torch.Tensor) -> torch.Tensor:
        return success_rates.pow(self._a - 1.0) * (1.0 - success_rates).pow(self._b - 1.0)


class FrontierSignal:
    """Per-task frontier score from a kNN graph over the task feature space.

    For each task ``i`` with rolling success rate ``s_i``::

        s_dil_i = max over kNN neighbours of i  (in task-feature space)
        score_i = (1 - s_i) * (s_dil_i - s_i).clamp_min(0)

    A task scores positive only when its own rate is below the maximum
    rate among its kNN-similar tasks -- i.e. "this specific task is not
    yet solved, but a similar task is." Because the signal lives on
    tasks (never aggregates rates onto states), task identity is
    preserved end-to-end and there is no "free-rider" pathology where
    a learned target inherits sampling pressure onto unrelated spawns
    paired with it.

    The task feature is ``concat(spawn_xy, target_xy)`` when the layout
    has a separate target endpoint, or just the spawn/slot feature
    when ``layout.target_index is None`` (factory's slot==item case,
    where the task feature degenerates to the slot's own coords).

    When ``layout.task_partition`` is set, the kNN is built **within
    each partition** rather than globally -- two tasks in different
    partitions are never neighbours regardless of how close they are
    in feature space. This is how mechanically-distinct task families
    (walking-A→B vs. flying-A→B vs. terrain-pose vs. velocity-cmd)
    keep independent frontier manifolds: their spatial endpoints
    overlap but their dynamics don't, so frontier propagation across
    families would be a misleading signal.

    The feature space is built once at construction; per-call cost is
    the max-pool dilation plus an elementwise multiply.
    """

    name = "frontier"

    def __init__(self, cfg: FrontierSignalCfg, layout: StateLayout) -> None:
        self._dilation_steps = max(1, int(cfg.dilation_steps))
        spawn_feat = layout.coords[layout.spawn_index]
        if layout.target_index is None:
            task_features = spawn_feat
        else:
            target_feat = layout.coords[layout.target_index]
            task_features = torch.cat([spawn_feat, target_feat], dim=-1)
        self._knn = _build_partitioned_knn(task_features, layout.task_partition, k=cfg.k)

    def score(self, success_rates: torch.Tensor) -> torch.Tensor:
        return task_frontier_weights(
            success_rates,
            task_knn_indices=self._knn,
            dilation_steps=self._dilation_steps,
        )

    def task_frontier(self, success_rates: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`score` -- the per-task frontier weight tensor.

        Surfaced separately so diagnostic consumers can pull the
        per-task spatial signal out of the curriculum without
        recomputing it. Equivalent to ``self.score(success_rates)``.
        """
        return self.score(success_rates)


def _build_partitioned_knn(
    features: torch.Tensor,
    partition: torch.Tensor | None,
    k: int,
) -> torch.Tensor:
    """Build a per-partition kNN graph over ``features``.

    For each task, returned neighbours come exclusively from tasks
    sharing the same ``partition`` key. With ``partition is None``
    behaves identically to :func:`build_knn_indices`.

    Args:
        features: ``[num_items, feature_dim]`` per-task feature vectors.
        partition: ``[num_items]`` long tensor of partition keys, or
            ``None`` for a single global partition.
        k: Number of nearest neighbours per task.

    Returns:
        ``[num_items, k]`` long tensor of *global* task indices.
    """
    if partition is None:
        return build_knn_indices(features, k=k)

    n = int(features.shape[0])
    knn = torch.empty((n, k), dtype=torch.long, device=features.device)
    for p in torch.unique(partition).tolist():
        mask = partition == p
        member_idx = mask.nonzero(as_tuple=False).squeeze(-1)
        if member_idx.numel() == 0:
            continue
        local_features = features[member_idx]
        local_knn = build_knn_indices(local_features, k=k)
        # ``local_knn`` indexes into the partition's local feature list;
        # remap to the global task indices that ``score`` gathers from.
        knn[member_idx] = member_idx[local_knn]
    return knn


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
