# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact uniform-assignment kernels for GPU-resident sequence metrics."""

from string import Template

import warp as wp

UNIFORM_ASSIGNMENT_BLOCK_DIM = 256


@wp.kernel(enable_backward=False)
def uniform_assignment_prepare_bucket(
    observed: wp.array(dtype=wp.float32, ndim=3),
    target: wp.array(dtype=wp.float32, ndim=3),
    row_indices: wp.array(dtype=wp.int64),
    source_frame_count: wp.int32,
    source_feature_width: wp.int32,
    destination_feature_width: wp.int32,
    bucket_observed: wp.array(dtype=wp.float32, ndim=3),
    bucket_target: wp.array(dtype=wp.float32, ndim=3),
):
    """Gather one fixed bucket and zero-pad its dense feature rows."""
    bucket_row, frame = wp.tid()
    source_row = int(row_indices[bucket_row])
    for feature in range(destination_feature_width):
        observed_value = float(0.0)
        target_value = float(0.0)
        if frame < source_frame_count and feature < source_feature_width:
            observed_value = observed[source_row, frame, feature]
            target_value = target[source_row, frame, feature]
        bucket_observed[bucket_row, frame, feature] = observed_value
        bucket_target[bucket_row, frame, feature] = target_value


@wp.kernel(enable_backward=False)
def uniform_assignment_prepare_flat_bucket(
    observed: wp.array(dtype=wp.float32, ndim=2),
    target: wp.array(dtype=wp.float32, ndim=2),
    observed_starts: wp.array(dtype=wp.int64),
    target_starts: wp.array(dtype=wp.int64),
    row_indices: wp.array(dtype=wp.int64),
    lengths: wp.array(dtype=wp.int64),
    source_feature_width: wp.int32,
    destination_feature_width: wp.int32,
    bucket_observed: wp.array(dtype=wp.float32, ndim=3),
    bucket_target: wp.array(dtype=wp.float32, ndim=3),
):
    """Gather compact per-clip rows into one fixed dense assignment bucket."""
    bucket_row, frame = wp.tid()
    pair_row = int(row_indices[bucket_row])
    count = int(lengths[bucket_row])
    for feature in range(destination_feature_width):
        observed_value = float(0.0)
        target_value = float(0.0)
        if frame < count and feature < source_feature_width:
            observed_value = observed[int(observed_starts[pair_row]) + frame, feature]
            target_value = target[int(target_starts[pair_row]) + frame, feature]
        bucket_observed[bucket_row, frame, feature] = observed_value
        bucket_target[bucket_row, frame, feature] = target_value


@wp.kernel(enable_backward=False)
def uniform_assignment_pairwise_distance(
    observed: wp.array(dtype=wp.float32, ndim=3),
    target: wp.array(dtype=wp.float32, ndim=3),
    feature_width: wp.int32,
    cost: wp.array(dtype=wp.float32, ndim=3),
):
    """Write translation-invariant Euclidean frame costs directly from feature differences."""
    batch, observed_frame, target_frame = wp.tid()
    squared_distance = float(0.0)
    for feature in range(feature_width):
        difference = observed[batch, observed_frame, feature] - target[batch, target_frame, feature]
        squared_distance += difference * difference
    cost[batch, observed_frame, target_frame] = wp.sqrt(squared_distance)


@wp.kernel(enable_backward=False)
def uniform_assignment_cost_scalar(
    cost: wp.array(dtype=wp.float32, ndim=3),
    lengths: wp.array(dtype=wp.int64),
    potential_rows: wp.array(dtype=wp.float64, ndim=2),
    potential_columns: wp.array(dtype=wp.float64, ndim=2),
    matching: wp.array(dtype=wp.int32, ndim=2),
    previous: wp.array(dtype=wp.int32, ndim=2),
    minimum: wp.array(dtype=wp.float64, ndim=2),
    used: wp.array(dtype=wp.int32, ndim=2),
    output: wp.array(dtype=wp.float64),
):
    """Solve one exact assignment per thread with stable lowest-column ties."""
    batch = wp.tid()
    count = int(lengths[batch])
    for column in range(count + 1):
        potential_rows[batch, column] = wp.float64(0.0)
        potential_columns[batch, column] = wp.float64(0.0)
        matching[batch, column] = 0
        previous[batch, column] = 0

    for row in range(1, count + 1):
        matching[batch, 0] = row
        for column in range(count + 1):
            minimum[batch, column] = wp.float64(1.0e300)
            used[batch, column] = 0

        column = int(0)
        for _ in range(count + 1):
            used[batch, column] = 1
            source_row = matching[batch, column]
            delta = wp.float64(1.0e300)
            next_column = int(0)
            for candidate in range(1, count + 1):
                if used[batch, candidate] == 0:
                    reduced = (
                        wp.float64(cost[batch, source_row - 1, candidate - 1])
                        - potential_rows[batch, source_row]
                        - potential_columns[batch, candidate]
                    )
                    if reduced < minimum[batch, candidate]:
                        minimum[batch, candidate] = reduced
                        previous[batch, candidate] = column
                    if minimum[batch, candidate] < delta:
                        delta = minimum[batch, candidate]
                        next_column = candidate

            for candidate in range(count + 1):
                if used[batch, candidate] != 0:
                    matched_row = matching[batch, candidate]
                    potential_rows[batch, matched_row] = potential_rows[batch, matched_row] + delta
                    potential_columns[batch, candidate] = potential_columns[batch, candidate] - delta
                elif candidate > 0:
                    minimum[batch, candidate] = minimum[batch, candidate] - delta
            column = next_column
            if matching[batch, column] == 0:
                break

        for _ in range(count + 1):
            next_column = previous[batch, column]
            matching[batch, column] = matching[batch, next_column]
            column = next_column
            if column == 0:
                break

    total = wp.float64(0.0)
    for column in range(1, count + 1):
        total = total + wp.float64(cost[batch, matching[batch, column] - 1, column - 1])
    output[batch] = total / wp.float64(count)


_UNIFORM_ASSIGNMENT_NATIVE = Template(
    r"""
    __shared__ double reduce_value[$block_dim];
    __shared__ int reduce_column[$block_dim];
    __shared__ double selected_delta;
    __shared__ int selected_column;

    const int count = (int)(*wp::address(lengths, batch));
    const double infinity = 1.0e300;

    for (int column = lane; column <= count; column += $block_dim) {
        *wp::address(potential_rows, batch, column) = 0.0;
        *wp::address(potential_columns, batch, column) = 0.0;
        *wp::address(matching, batch, column) = 0;
        *wp::address(previous, batch, column) = 0;
    }
    __syncthreads();

    for (int row = 1; row <= count; ++row) {
        if (lane == 0) {
            *wp::address(matching, batch, 0) = row;
            selected_column = 0;
        }
        for (int column = lane; column <= count; column += $block_dim) {
            *wp::address(minimum, batch, column) = infinity;
            *wp::address(used, batch, column) = 0;
        }
        __syncthreads();

        while (true) {
            const int column = selected_column;
            if (lane == 0) {
                *wp::address(used, batch, column) = 1;
            }
            __syncthreads();

            const int source_row = *wp::address(matching, batch, column);
            double local_delta = infinity;
            int local_column = 0;
            for (int candidate = lane + 1; candidate <= count; candidate += $block_dim) {
                if (*wp::address(used, batch, candidate) == 0) {
                    const double reduced =
                        (double)(*wp::address(cost, batch, source_row - 1, candidate - 1))
                        - *wp::address(potential_rows, batch, source_row)
                        - *wp::address(potential_columns, batch, candidate);
                    double candidate_minimum = *wp::address(minimum, batch, candidate);
                    if (reduced < candidate_minimum) {
                        candidate_minimum = reduced;
                        *wp::address(minimum, batch, candidate) = reduced;
                        *wp::address(previous, batch, candidate) = column;
                    }
                    if (
                        candidate_minimum < local_delta
                        || (
                            candidate_minimum == local_delta
                            && (local_column == 0 || candidate < local_column)
                        )
                    ) {
                        local_delta = candidate_minimum;
                        local_column = candidate;
                    }
                }
            }
            reduce_value[lane] = local_delta;
            reduce_column[lane] = local_column;
            __syncthreads();

            for (int stride = $reduce_start; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    const double other_value = reduce_value[lane + stride];
                    const int other_column = reduce_column[lane + stride];
                    const double own_value = reduce_value[lane];
                    const int own_column = reduce_column[lane];
                    if (
                        other_value < own_value
                        || (
                            other_value == own_value
                            && other_column != 0
                            && (own_column == 0 || other_column < own_column)
                        )
                    ) {
                        reduce_value[lane] = other_value;
                        reduce_column[lane] = other_column;
                    }
                }
                __syncthreads();
            }

            if (lane == 0) {
                selected_delta = reduce_value[0];
                selected_column = reduce_column[0];
            }
            __syncthreads();

            const double delta = selected_delta;
            for (int candidate = lane; candidate <= count; candidate += $block_dim) {
                if (*wp::address(used, batch, candidate) != 0) {
                    const int matched_row = *wp::address(matching, batch, candidate);
                    *wp::address(potential_rows, batch, matched_row) += delta;
                    *wp::address(potential_columns, batch, candidate) -= delta;
                }
                else if (candidate > 0) {
                    *wp::address(minimum, batch, candidate) -= delta;
                }
            }
            __syncthreads();

            if (*wp::address(matching, batch, selected_column) == 0) {
                break;
            }
        }

        if (lane == 0) {
            int column = selected_column;
            while (true) {
                const int next_column = *wp::address(previous, batch, column);
                *wp::address(matching, batch, column) = *wp::address(matching, batch, next_column);
                column = next_column;
                if (column == 0) {
                    break;
                }
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
        double total = 0.0;
        for (int column = 1; column <= count; ++column) {
            total += (double)(*wp::address(
                cost,
                batch,
                *wp::address(matching, batch, column) - 1,
                column - 1));
        }
        *wp::address(output, batch) = total / (double)count;
    }
"""
).substitute(
    {
        "block_dim": UNIFORM_ASSIGNMENT_BLOCK_DIM,
        "reduce_start": UNIFORM_ASSIGNMENT_BLOCK_DIM // 2,
    }
)


@wp.func_native(_UNIFORM_ASSIGNMENT_NATIVE)
def _uniform_assignment_cost_native(
    cost: wp.array(dtype=wp.float32, ndim=3),
    lengths: wp.array(dtype=wp.int64),
    potential_rows: wp.array(dtype=wp.float64, ndim=2),
    potential_columns: wp.array(dtype=wp.float64, ndim=2),
    matching: wp.array(dtype=wp.int32, ndim=2),
    previous: wp.array(dtype=wp.int32, ndim=2),
    minimum: wp.array(dtype=wp.float64, ndim=2),
    used: wp.array(dtype=wp.int32, ndim=2),
    output: wp.array(dtype=wp.float64),
    batch: wp.int32,
    lane: wp.int32,
): ...


@wp.kernel(enable_backward=False)
def uniform_assignment_cost(
    cost: wp.array(dtype=wp.float32, ndim=3),
    lengths: wp.array(dtype=wp.int64),
    potential_rows: wp.array(dtype=wp.float64, ndim=2),
    potential_columns: wp.array(dtype=wp.float64, ndim=2),
    matching: wp.array(dtype=wp.int32, ndim=2),
    previous: wp.array(dtype=wp.int32, ndim=2),
    minimum: wp.array(dtype=wp.float64, ndim=2),
    used: wp.array(dtype=wp.int32, ndim=2),
    output: wp.array(dtype=wp.float64),
):
    """Solve one variable-size assignment per CUDA block with stable lowest-column ties."""
    batch, lane = wp.tid()
    _uniform_assignment_cost_native(
        cost,
        lengths,
        potential_rows,
        potential_columns,
        matching,
        previous,
        minimum,
        used,
        output,
        batch,
        lane,
    )
