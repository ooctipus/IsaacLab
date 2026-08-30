# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure Warp kernels and launches for curriculum sampling."""

from __future__ import annotations

import warp as wp
import warp.utils as wpu

STRATEGY_UNIFORM = 0
STRATEGY_BETA = 1
STRATEGY_FRONTIER = 2


@wp.kernel
def _frontier_init_kernel(
    success_rates: wp.array(dtype=wp.float32),
    frontier_order: wp.array(dtype=wp.int32),
    frontier_prev: wp.array2d(dtype=wp.float32),
):
    f, i = wp.tid()
    frontier_prev[f, i] = success_rates[int(frontier_order[i])]


@wp.kernel
def _frontier_dilate_kernel(
    frontier_prev: wp.array2d(dtype=wp.float32),
    frontier_next: wp.array2d(dtype=wp.float32),
    frontier_results: wp.array2d(dtype=wp.float32),
    frontier_order: wp.array(dtype=wp.int32),
    frontier_result_for_step: wp.array2d(dtype=wp.int32),
    frontier_knn: wp.array3d(dtype=wp.int32),
    frontier_k: wp.array(dtype=wp.int32),
    frontier_group_max_dilation_steps: wp.array(dtype=wp.int32),
    step: int,
    max_k: int,
):
    g, i = wp.tid()
    if step >= int(frontier_group_max_dilation_steps[g]):
        return

    value = frontier_prev[g, i]
    k = int(frontier_k[g])
    for j in range(max_k):
        if j < k:
            neighbor = int(frontier_knn[g, i, j])
            value = wp.max(value, frontier_prev[g, neighbor])
    frontier_next[g, i] = value

    result_row = int(frontier_result_for_step[g, step])
    if result_row >= 0:
        frontier_results[result_row, int(frontier_order[i])] = value


@wp.kernel
def _sampler_score_kernel(
    success_rates: wp.array(dtype=wp.float32),
    score_rows: wp.array2d(dtype=wp.float32),
    strategy_kind: wp.array(dtype=wp.int32),
    beta_a: wp.array(dtype=wp.float32),
    beta_b: wp.array(dtype=wp.float32),
    frontier_ids: wp.array(dtype=wp.int32),
    frontier_dilated: wp.array2d(dtype=wp.float32),
):
    strategy, item = wp.tid()
    rate = success_rates[item]
    kind = int(strategy_kind[strategy])
    if kind == STRATEGY_UNIFORM:
        score_rows[strategy, item] = 1.0
    elif kind == STRATEGY_BETA:
        score_rows[strategy, item] = wp.pow(rate, beta_a[strategy] - 1.0) * wp.pow(1.0 - rate, beta_b[strategy] - 1.0)
    else:
        frontier = int(frontier_ids[strategy])
        delta = frontier_dilated[frontier, item] - rate
        if delta < 0.0:
            delta = 0.0
        score_rows[strategy, item] = (1.0 - rate) * delta


@wp.kernel
def _sampler_weight_kernel(
    success_rates: wp.array(dtype=wp.float32),
    weighted: wp.array(dtype=wp.float32),
    strategy_kind: wp.array(dtype=wp.int32),
    weights: wp.array(dtype=wp.float32),
    beta_a: wp.array(dtype=wp.float32),
    beta_b: wp.array(dtype=wp.float32),
    frontier_ids: wp.array(dtype=wp.int32),
    frontier_dilated: wp.array2d(dtype=wp.float32),
    eps: float,
    num_strategies: int,
):
    item = wp.tid()
    rate = success_rates[item]
    weighted_score = eps
    for strategy in range(num_strategies):
        kind = int(strategy_kind[strategy])
        score = float(1.0)
        if kind == STRATEGY_BETA:
            score = wp.pow(rate, beta_a[strategy] - 1.0) * wp.pow(1.0 - rate, beta_b[strategy] - 1.0)
        elif kind == STRATEGY_FRONTIER:
            frontier = int(frontier_ids[strategy])
            delta = frontier_dilated[frontier, item] - rate
            if delta < 0.0:
                delta = 0.0
            score = (1.0 - rate) * delta
        weight = weights[strategy]
        if weight > 0.0:
            weighted_score += weight * score
    weighted[item] = weighted_score


@wp.kernel
def _sampler_normalize_kernel(
    weighted: wp.array(dtype=wp.float32),
    total: wp.array(dtype=wp.float32),
    probabilities: wp.array(dtype=wp.float32),
):
    item = wp.tid()
    probabilities[item] = weighted[item] / total[0]


@wp.kernel
def _sample_counter_kernel(counter: wp.array(dtype=wp.int64), base: wp.array(dtype=wp.int64), num_samples: int):
    base[0] = counter[0]
    counter[0] = counter[0] + wp.int64(num_samples)


@wp.kernel
def _sample_cdf_kernel(
    cdf: wp.array(dtype=wp.float32),
    samples: wp.array(dtype=wp.int64),
    base: wp.array(dtype=wp.int64),
    seed: int,
    num_items: int,
):
    index = wp.tid()
    rng = wp.rand_init(seed, int(base[0]) + index)
    sample = wp.randf(rng)

    low = int(0)
    high = int(num_items - 1)
    while low < high:
        middle = (low + high) // 2
        if sample <= cdf[middle]:
            high = middle
        else:
            low = middle + 1
    samples[index] = wp.int64(low)


def scores(
    success_rates: wp.array(dtype=wp.float32),
    score_rows: wp.array2d(dtype=wp.float32),
    strategy_kind: wp.array(dtype=wp.int32),
    beta_a: wp.array(dtype=wp.float32),
    beta_b: wp.array(dtype=wp.float32),
    frontier_ids: wp.array(dtype=wp.int32),
    frontier_results: wp.array2d(dtype=wp.float32),
    num_strategies: int,
    num_items: int,
    device: str,
) -> None:
    """Write per-strategy score rows into caller-owned Warp arrays."""
    wp.launch(
        _sampler_score_kernel,
        dim=(num_strategies, num_items),
        inputs=[success_rates, score_rows, strategy_kind, beta_a, beta_b, frontier_ids, frontier_results],
        device=device,
    )


def probabilities(
    success_rates: wp.array(dtype=wp.float32),
    weighted: wp.array(dtype=wp.float32),
    probabilities_out: wp.array(dtype=wp.float32),
    total: wp.array(dtype=wp.float32),
    strategy_kind: wp.array(dtype=wp.int32),
    weights: wp.array(dtype=wp.float32),
    beta_a: wp.array(dtype=wp.float32),
    beta_b: wp.array(dtype=wp.float32),
    frontier_ids: wp.array(dtype=wp.int32),
    frontier_results: wp.array2d(dtype=wp.float32),
    eps: float,
    num_strategies: int,
    num_items: int,
    device: str,
) -> None:
    """Write normalized probabilities into caller-owned Warp arrays."""
    wp.launch(
        _sampler_weight_kernel,
        dim=num_items,
        inputs=[
            success_rates,
            weighted,
            strategy_kind,
            weights,
            beta_a,
            beta_b,
            frontier_ids,
            frontier_results,
            eps,
            num_strategies,
        ],
        device=device,
    )
    wpu.array_sum(weighted, out=total, value_count=num_items)
    wp.launch(
        _sampler_normalize_kernel,
        dim=num_items,
        inputs=[weighted, total, probabilities_out],
        device=device,
    )


def sample(
    probabilities: wp.array(dtype=wp.float32),
    cdf: wp.array(dtype=wp.float32),
    samples: wp.array(dtype=wp.int64),
    sample_counter: wp.array(dtype=wp.int64),
    sample_base: wp.array(dtype=wp.int64),
    num_samples: int,
    num_items: int,
    seed: int,
    device: str,
) -> None:
    """Sample item indices into a caller-owned fixed output array."""
    wpu.array_scan(probabilities, cdf, inclusive=True)
    wp.launch(
        _sample_counter_kernel,
        dim=1,
        inputs=[sample_counter, sample_base, num_samples],
        device=device,
    )
    wp.launch(
        _sample_cdf_kernel,
        dim=num_samples,
        inputs=[cdf, samples, sample_base, seed, num_items],
        device=device,
    )


def update_frontier(
    success_rates: wp.array(dtype=wp.float32),
    frontier_order: wp.array(dtype=wp.int32),
    frontier_result_for_step: wp.array2d(dtype=wp.int32),
    frontier_knn: wp.array3d(dtype=wp.int32),
    frontier_k: wp.array(dtype=wp.int32),
    frontier_group_max_dilation_steps: wp.array(dtype=wp.int32),
    frontier_prev: wp.array2d(dtype=wp.float32),
    frontier_next: wp.array2d(dtype=wp.float32),
    frontier_results: wp.array2d(dtype=wp.float32),
    num_items: int,
    num_frontier_groups: int,
    max_dilation_steps: int,
    max_k: int,
    device: str,
) -> None:
    """Update frontier result rows in caller-owned Warp arrays."""
    if num_frontier_groups == 0:
        return
    wp.launch(
        _frontier_init_kernel,
        dim=(num_frontier_groups, num_items),
        inputs=[success_rates, frontier_order, frontier_prev],
        device=device,
    )
    for step in range(max_dilation_steps):
        wp.launch(
            _frontier_dilate_kernel,
            dim=(num_frontier_groups, num_items),
            inputs=[
                frontier_prev,
                frontier_next,
                frontier_results,
                frontier_order,
                frontier_result_for_step,
                frontier_knn,
                frontier_k,
                frontier_group_max_dilation_steps,
                step,
                max_k,
            ],
            device=device,
        )
        frontier_prev, frontier_next = frontier_next, frontier_prev
