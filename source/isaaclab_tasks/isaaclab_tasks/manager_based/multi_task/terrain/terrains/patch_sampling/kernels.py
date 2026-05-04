# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels used by the morphological patch-sampling pipeline.

These kernels are launched from :mod:`morph`; they are
factored into their own module so the morph host code stays readable
and the compiled kernels remain importable in isolation for tests.
"""

from __future__ import annotations

import warp as wp


@wp.kernel(enable_backward=False)
def rasterize_grid_kernel(
    mesh: wp.uint64,
    x0: wp.float32,
    y0: wp.float32,
    scale: wp.float32,
    ray_start_z: wp.float32,
    max_dist: wp.float32,
    out_heightmap: wp.array2d(dtype=wp.float32),
):
    """Cast a downward ray from each grid cell centre; writes hit z or leaves ``inf``.

    The origin for cell ``(i, j)`` is constructed on-the-fly from the thread
    indices, so no ``[H*W, 3]`` ray-start / direction / hit tensors need to
    be materialized in GPU memory. Peak memory collapses from ~800 MiB at
    production scale to just the output heightmap.
    """
    i, j = wp.tid()
    origin = wp.vec3(
        x0 + (wp.float32(i) + 0.5) * scale,
        y0 + (wp.float32(j) + 0.5) * scale,
        ray_start_z,
    )
    direction = wp.vec3(0.0, 0.0, -1.0)
    query = wp.mesh_query_ray(mesh, origin, direction, max_dist)
    if query.result:
        out_heightmap[i, j] = ray_start_z - query.t


@wp.kernel
def morph_validity_kernel(
    heightmap: wp.array2d(dtype=wp.float32),
    mask: wp.array2d(dtype=wp.uint8),
    max_height_diff: wp.float32,
    z_min: wp.float32,
    z_max: wp.float32,
    pad: wp.int32,
    out_valid: wp.array2d(dtype=wp.uint8),
    out_h_range: wp.array2d(dtype=wp.float32),
):
    """Fused max-min-over-footprint morphological validity + height-range kernel.

    Replaces ``F.unfold(heightmap, k)`` + masked-max/min reductions, which
    materialize a ``[1, k*k, H*W]`` patch tensor (~800 MiB at production
    heightmap sizes). Each thread owns one output cell and streams the
    ``k*k`` footprint through registers, so peak memory is ``O(H*W)``
    regardless of kernel size.
    """
    i, j = wp.tid()
    H = heightmap.shape[0]
    W = heightmap.shape[1]
    k = mask.shape[0]

    # Border cells whose k x k footprint would clip outside [0, H) x [0, W).
    if i < pad or i >= H - pad or j < pad or j >= W - pad:
        out_valid[i, j] = wp.uint8(0)
        out_h_range[i, j] = wp.float32(0.0)
        return

    local_max = wp.float32(-1.0e18)
    local_min = wp.float32(1.0e18)
    any_miss = wp.uint8(0)

    for di in range(k):
        for dj in range(k):
            if mask[di, dj] == wp.uint8(0):
                continue
            ii = i + di - pad
            jj = j + dj - pad
            h = heightmap[ii, jj]
            if wp.isinf(h):
                any_miss = wp.uint8(1)
                continue
            if h > local_max:
                local_max = h
            if h < local_min:
                local_min = h

    h_range = local_max - local_min
    is_valid = wp.uint8(0)
    if any_miss == wp.uint8(0) and h_range <= max_height_diff and local_min >= z_min and local_max <= z_max:
        is_valid = wp.uint8(1)

    out_valid[i, j] = is_valid
    # For miss/border cells ``h_range`` is meaningless; callers gate on ``is_valid``.
    out_h_range[i, j] = h_range
