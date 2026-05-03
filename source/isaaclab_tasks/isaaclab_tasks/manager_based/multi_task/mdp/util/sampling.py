# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch

from .sampling_cfg import BetaSamplingCfg, UniformSamplingCfg


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


def build_knn_indices(
    xy_coords: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Build a k-NN index table over an xy point set.

    Intended for use over the *state buffer* (the pool of physically-valid
    robot xy positions, e.g. ``task_table.spawn_states[:, :2]``), not over
    tasks. The state pool is the natural spatial domain regardless of how
    tasks combine spawn/target endpoints; this function is therefore
    topology-agnostic.

    For each point the returned row contains the indices of its ``k``
    nearest other points in xy (self excluded). When the pool has fewer
    than ``k+1`` points, missing slots are padded with the point's own
    index so frontier evaluations on those slots collapse to 0 without
    spurious cross-talk.

    Uses :class:`scipy.spatial.cKDTree` so the cost is
    :math:`O(n \\log n)` and memory is :math:`O(n)` (avoids the
    :math:`O(n^2)` blow-up of a full :func:`torch.cdist` for very large
    pools).

    Args:
        xy_coords: ``[num_points, 2]`` xy positions of the point set.
        k: Number of nearest neighbors to record per point.

    Returns:
        ``[num_points, k]`` long tensor of point-pool indices.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}.")
    from scipy.spatial import cKDTree

    num_points = int(xy_coords.shape[0])
    device = xy_coords.device
    self_idx = torch.arange(num_points, device=device, dtype=torch.long).unsqueeze(-1)
    knn = self_idx.expand(num_points, k).clone()

    if num_points <= 1:
        return knn

    xy_np = xy_coords.detach().cpu().numpy()
    k_eff = min(k, num_points - 1)
    tree = cKDTree(xy_np)
    _, idx = tree.query(xy_np, k=k_eff + 1)
    idx = np.atleast_2d(idx)
    if idx.shape[0] != num_points:
        idx = idx.T
    knn_block = idx[:, 1 : k_eff + 1]
    if k_eff < k:
        pad = np.tile(np.arange(num_points, dtype=knn_block.dtype).reshape(-1, 1), (1, k - k_eff))
        knn_block = np.concatenate([knn_block, pad], axis=1)
    knn = torch.from_numpy(knn_block).to(device=device, dtype=torch.long)
    return knn


def state_frontier_weights(
    success_rates: torch.Tensor,
    *,
    state_knn_indices: torch.Tensor,
    spawn_index: torch.Tensor,
    target_index: torch.Tensor,
    dilation_steps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-state success rates and per-state frontier weights.

    The single source of truth for the state-buffer spatial-frontier
    computation. Both :func:`frontier_sampling_probs` (the algorithm)
    and the curriculum's diagnostic / dashboard paths call this so they
    can never silently diverge.

    Per-state success ``state_s`` is the mean per-task success rate
    across all tasks touching the state in either endpoint role. The
    frontier weight is::

        state_frontier = (1 - state_s) * (s_dil - state_s).clamp_min(0)

    where ``s_dil`` is ``state_s`` propagated outward via
    ``dilation_steps`` graph-max iterations on ``state_knn_indices``.
    The ``(1 - state_s)`` factor downweights states that are themselves
    well-learned, even if a more-learned neighbor exists -- the frontier
    is "I'm not yet solved AND a neighbor is more solved", not just
    "a neighbor is more solved".

    Args:
        success_rates: ``[num_tasks]`` per-task rolling success rate.
        state_knn_indices: ``[num_states, k]`` long tensor from
            :func:`build_knn_indices` over the state pool xy.
        spawn_index: ``[num_tasks]`` per-task spawn state index.
        target_index: ``[num_tasks]`` per-task target state index.
        dilation_steps: Number of graph-max iterations.

    Returns:
        ``(state_s, state_frontier)`` -- both ``[num_states]`` on the
        same device/dtype as ``success_rates``.
    """
    device = success_rates.device
    n_states = int(state_knn_indices.shape[0])
    state_sums = torch.zeros(n_states, device=device, dtype=success_rates.dtype)
    state_counts = torch.zeros(n_states, device=device, dtype=success_rates.dtype)
    ones = torch.ones_like(success_rates)
    state_sums.scatter_add_(0, spawn_index, success_rates)
    state_sums.scatter_add_(0, target_index, success_rates)
    state_counts.scatter_add_(0, spawn_index, ones)
    state_counts.scatter_add_(0, target_index, ones)
    state_s = state_sums / state_counts.clamp_min(1.0)

    s_dil = state_s
    for _ in range(max(1, int(dilation_steps))):
        neighbor_max = s_dil[state_knn_indices].amax(dim=-1)
        s_dil = torch.maximum(s_dil, neighbor_max)
    state_frontier = (1.0 - state_s) * (s_dil - state_s).clamp_min(0.0)

    return state_s, state_frontier


def frontier_sampling_probs(
    success_rates: torch.Tensor,
    *,
    state_knn_indices: torch.Tensor,
    spawn_index: torch.Tensor,
    target_index: torch.Tensor,
    base: BetaSamplingCfg | UniformSamplingCfg,
    frontier_lambda: float = 0.5,
    dilation_steps: int = 1,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Combine a per-task base sampler with a state-buffer-graph frontier weight.

    The frontier signal lives in the *state buffer* (the underlying pool
    of physically-valid robot xy positions, e.g. ``spawn_states``), not
    in task space. This makes the algorithm topology-agnostic: it works
    identically whether you have one spawn / many targets, many spawns
    / one target, or many of both. Per-task quantities are aggregated
    onto the state pool, the spatial frontier is computed there
    (see :func:`state_frontier_weights`), and per-task weights are
    recovered by summing each endpoint's *above-mean* deviation -- a
    constant endpoint (e.g. shared target under
    ``single_target_per_cell=True``) auto-cancels.

    Steps:

    1. Per-task base weight from the plugged-in ``base`` sampler.
    2. Per-state ``state_frontier`` from :func:`state_frontier_weights`.
    3. Per-task frontier weight = sum of above-mean deviations at the
       task's two endpoints. Constant endpoints contribute 0.
    4. ``w = base_w + lambda * task_frontier_w + eps``; normalize.

    Args:
        success_rates: ``[num_tasks]`` per-task rolling success rate.
        state_knn_indices: ``[num_states, k]`` long tensor of state-pool
            indices from :func:`build_knn_indices` over
            ``spawn_states[:, :2]``.
        spawn_index: ``[num_tasks]`` mapping each task to its spawn
            state index (``task_table.spawn_index``).
        target_index: ``[num_tasks]`` mapping each task to its target
            state index (``task_table.target_index``).
        base: Per-task sampler whose unnormalized kernel forms the
            "this task itself is borderline" term.
        frontier_lambda: Mixing weight for the spatial-frontier term.
            ``0`` reproduces ``base`` alone.
        dilation_steps: Number of graph-max iterations on the state-pool
            kNN graph. ``1`` looks at immediate neighbors; ``k_steps``
            propagates the local max ``k_steps`` hops out.
        eps: Floor on the per-task weight so the success monitor keeps
            refreshing every task.

    Returns:
        ``[num_tasks]`` probability tensor, normalized over all tasks.
    """
    # 1) Per-task baseline weight from the plugged-in sampler cfg.
    if isinstance(base, BetaSamplingCfg):
        t = max(0.0, min(1.0, base.target))
        k = max(0.0, base.kappa)
        a = 1.0 + k * t
        b = 1.0 + k * (1.0 - t)
        base_w = (success_rates + eps).pow(a - 1.0) * (1.0 - success_rates + eps).pow(b - 1.0)
    elif isinstance(base, UniformSamplingCfg):
        base_w = torch.ones_like(success_rates)
    else:
        raise TypeError(
            f"Unsupported base sampler '{type(base).__name__}'; expected BetaSamplingCfg or UniformSamplingCfg."
        )
    base_w = base_w.clamp_min(eps)

    # 2) Per-state frontier via the shared helper. Single source of truth.
    _, state_frontier = state_frontier_weights(
        success_rates,
        state_knn_indices=state_knn_indices,
        spawn_index=spawn_index,
        target_index=target_index,
        dilation_steps=dilation_steps,
    )

    # 3) Per-task frontier = sum of per-endpoint above-mean frontier deviations.
    #    A constant endpoint has zero variance, so its deviation is
    #    identically 0 and it contributes nothing. An endpoint that varies
    #    across tasks contributes its positive-side deviation, so tasks
    #    whose endpoint is above-average-frontier get credit. This
    #    generalizes without auto-detection: single-target topologies
    #    reduce to spawn-only, single-spawn to target-only, all-pair /
    #    N-pair contribute on both axes.
    spawn_frontier = state_frontier[spawn_index]
    target_frontier = state_frontier[target_index]
    task_frontier_w = (spawn_frontier - spawn_frontier.mean()).clamp_min(0.0) + (
        target_frontier - target_frontier.mean()
    ).clamp_min(0.0)

    # 4) Combine and normalize globally over all tasks.
    w = base_w + max(0.0, frontier_lambda) * task_frontier_w + eps
    return w / w.sum()


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
