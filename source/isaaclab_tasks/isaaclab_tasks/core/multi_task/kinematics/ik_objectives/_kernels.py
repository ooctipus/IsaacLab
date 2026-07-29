# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Warp kernels used by multiple IK objectives."""

import warp as wp


@wp.kernel
def jac_fill_row(
    q_grad: wp.array2d(dtype=wp.float32),
    n_dofs: int,
    row_idx: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    """Copy autodiff gradient for one residual row into the Jacobian."""
    batch_idx = wp.tid()
    for d in range(n_dofs):
        jacobian[batch_idx, row_idx, d] = q_grad[batch_idx, d]
