# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class UniformSamplingCfg:
    """Sample slots uniformly at random. No success rates needed."""

    pass


@configclass
class BetaSamplingCfg:
    """Sample slots using Beta-weighted probabilities peaked near a target success rate.

    Attributes:
        success_rate_bind: Eval expression resolved against ``self`` (the calling
            ManagerTermBase instance) to obtain the per-slot success rate tensor.
        target: Desired success rate peak in [0, 1].
        kappa: Concentration around target.
        temperature: Softmax temperature controlling sharpness.
    """

    success_rate_bind: str = "self.success_rate"
    target: float = 0.5
    kappa: float = 1.0
    temperature: float = 2.0


@configclass
class FrontierSamplingCfg:
    """Frontier-aware sampler combining per-task value with per-tile spatial frontier.

    The ``base`` field plugs in any per-task sampler -- :class:`BetaSamplingCfg`
    for a Beta-shaped per-task weight, or :class:`UniformSamplingCfg` for a
    uniform per-task floor (only the spatial-frontier term distinguishes
    tasks). The per-tile term aggregates success rates onto the terrain
    grid, dilates them with one or more 3x3 max-pool steps, and treats
    ``s_dilated_neighbor - s_tile`` as a frontier score that boosts every
    task in that tile.

    Attributes:
        success_rate_bind: Eval expression resolved against ``self`` to
            obtain the per-task success rate tensor (same convention as
            :class:`BetaSamplingCfg`). Used both for the spatial-frontier
            computation and forwarded to the ``base`` sampler.
        base: Per-task sampler whose unnormalized weight is added to the
            spatial-frontier term. :class:`BetaSamplingCfg` rewards
            tasks at a target success rate; :class:`UniformSamplingCfg`
            gives every task an equal floor.
        k: Number of nearest neighbors per state in xy. Density-adaptive
            by construction: a tightly-packed cluster has ``k`` close
            neighbors; a sparse cluster has ``k`` farther ones. Smaller
            ``k`` = sharper, more local frontier; larger ``k`` =
            smoother. ``8`` is a reasonable default for most layouts.
        frontier_lambda: Mixing weight on the spatial-frontier term;
            ``0`` reproduces ``base`` alone.
        dilation_steps: Number of graph-max iterations over the kNN
            graph. ``1`` covers immediate neighbors, ``k_steps`` covers
            up to ``k_steps`` graph hops out (BFS-style propagation).
        eps: Soft floor on per-task weight so the success monitor keeps
            refreshing every task.
    """

    success_rate_bind: str = "self.success_rate"
    base: BetaSamplingCfg | UniformSamplingCfg = BetaSamplingCfg(target=0.66, kappa=1.0)
    k: int = 8
    frontier_lambda: float = 0.5
    dilation_steps: int = 1
    eps: float = 1e-3
