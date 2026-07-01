# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact uniform assignment kernel for GPU-resident motion EMD."""

import warp as wp


@wp.kernel
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
    """Solve one variable-size square assignment per thread with stable lowest-column ties."""
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
