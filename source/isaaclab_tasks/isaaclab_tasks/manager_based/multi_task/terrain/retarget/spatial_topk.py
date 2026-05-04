# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-query top-``k`` nearest-point lookup via Warp ``HashGrid``.

A simple wrapper around :class:`warp.HashGrid` that, for each query xy
point, returns the ``k`` nearest points from a fixed set within a
search radius. Used by the retarget sampler's per-foot patch lookup so
the cost is ``O(Q * k)``-ish instead of ``O(Q * N_p)``: on large
terrains the cdist-then-topk path materializes a ``[Q, N_p]`` distance
matrix that can hit hundreds of GiB.

Inputs / outputs are ``torch`` tensors so callers don't need to know
about Warp; we convert to/from Warp arrays internally.

Limitations:
* 2D only (xy). The grid is built with ``z=0`` for both points and
  queries; the third grid dimension stays at 1 cell.
* ``k`` is a Python int constant baked into the kernel via a constant.
  Pass at most :data:`MAX_K` (the kernel's compile-time cap).
"""

from __future__ import annotations

import torch
import warp as wp

# Compile-time cap on top-k width. The kernel hot-loops use fixed-size
# scratch arrays so ``k`` must be ``<= MAX_K``. Bump if needed; for the
# retarget sampler nc <= 8 covers every supported robot.
MAX_K: int = 16


@wp.kernel
def _topk_query_kernel(
    grid_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    radius: float,
    k: int,
    out_idx: wp.array2d(dtype=wp.int32),
    out_dist: wp.array2d(dtype=wp.float32),
):
    """One thread per query. Streaming "replace the max" top-k, then
    a constant-cost bubble-sort finalisation.

    The hash grid yields all points within ``radius`` of the query;
    for each, we evict the current top-k slot with the largest distance
    if the new distance is smaller. After the iteration we bubble-sort
    the small ``k``-element array. Avoids dynamic-bound ``range`` and
    ``break``-inside-dynamic-loop, both of which Warp's codegen rejects.
    Slots not filled during iteration keep ``idx=-1``, ``dist=+inf``.
    """
    q = wp.tid()
    qp = queries[q]

    # Initialise top-k slots with sentinels.
    for j in range(k):
        out_idx[q, j] = -1
        out_dist[q, j] = wp.inf

    # Streaming pass: for each candidate within radius, find the slot
    # currently holding the largest distance and replace if smaller.
    nb_iter = wp.hash_grid_query(grid_id, qp, radius)
    nb = int(0)
    while wp.hash_grid_query_next(nb_iter, nb):
        d = wp.length(points[nb] - qp)
        if d > radius:
            continue
        # Find slot with current max distance.
        max_dist = out_dist[q, 0]
        max_slot = int(0)
        for j in range(1, k):
            if out_dist[q, j] > max_dist:
                max_dist = out_dist[q, j]
                max_slot = j
        if d < max_dist:
            out_dist[q, max_slot] = d
            out_idx[q, max_slot] = nb

    # Bubble sort the (small) top-k array ascending so the caller sees
    # sorted neighbours. ``k`` is bounded so this is cheap.
    for i in range(k):
        for j in range(k - 1):
            if out_dist[q, j] > out_dist[q, j + 1]:
                td = out_dist[q, j]
                out_dist[q, j] = out_dist[q, j + 1]
                out_dist[q, j + 1] = td
                ti = out_idx[q, j]
                out_idx[q, j] = out_idx[q, j + 1]
                out_idx[q, j + 1] = ti


class SpatialGrid:
    """Built ``warp.HashGrid`` over a 2D point cloud, reusable across queries.

    Holds the warp grid plus the CUDA buffers it indexes into, so the
    grid stays valid for the lifetime of this object. Construct via
    :func:`build_spatial_grid_xy`; query via
    :func:`spatial_topk_xy_with_grid`. Reusing one grid for many query
    batches lets callers chunk a large query set without rebuilding
    the spatial structure each chunk.
    """

    __slots__ = ("grid", "_pts_wp", "_points3", "device")

    def __init__(self, grid: wp.HashGrid, pts_wp: wp.array, points3: torch.Tensor) -> None:
        self.grid = grid
        # Hold strong refs so the warp grid keeps indexing into live storage.
        self._pts_wp = pts_wp
        self._points3 = points3
        self.device = points3.device


def build_spatial_grid_xy(
    points_xy: torch.Tensor,
    *,
    radius: float,
    cell_size: float | None = None,
) -> SpatialGrid:
    """Build a reusable hash grid over ``points_xy``.

    Args:
        points_xy: Reference point positions [m], shape ``[N, 2]``, float.
        radius: Query radius [m] this grid will be queried against. Used
            to size ``cell_size`` and the grid extent.
        cell_size: Hash-grid cell edge [m]. ``None`` defaults to
            ``radius * 1.5`` (slightly bigger than the query radius so
            points on the outer boundary aren't excluded by FP rounding;
            the per-point distance filter inside the kernel still
            enforces the exact ``radius``).

    Returns:
        :class:`SpatialGrid` ready to feed
        :func:`spatial_topk_xy_with_grid`.
    """
    if points_xy.dim() != 2 or points_xy.shape[1] != 2:
        raise ValueError(f"points_xy must be [N, 2], got {tuple(points_xy.shape)}")
    if not points_xy.is_cuda:
        raise ValueError("build_spatial_grid_xy requires CUDA tensors")

    device = points_xy.device
    n = int(points_xy.shape[0])
    if cell_size is None:
        cell_size = float(radius) * 1.5

    # Embed 2D points in z=0 so warp's 3D HashGrid handles them.
    points3 = torch.cat([points_xy, torch.zeros(n, 1, device=device, dtype=points_xy.dtype)], dim=1).contiguous()
    pts_wp = wp.from_torch(points3, dtype=wp.vec3)

    grid = wp.HashGrid(
        dim_x=max(1, int(2.0 * (points_xy[:, 0].abs().max().item() + radius) / cell_size) + 8),
        dim_y=max(1, int(2.0 * (points_xy[:, 1].abs().max().item() + radius) / cell_size) + 8),
        dim_z=1,
        device=str(device),
    )
    grid.build(pts_wp, cell_size)
    return SpatialGrid(grid=grid, pts_wp=pts_wp, points3=points3)


def spatial_topk_xy_with_grid(
    spatial_grid: SpatialGrid,
    queries_xy: torch.Tensor,
    *,
    k: int,
    radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a top-``k`` lookup against a prebuilt :class:`SpatialGrid`.

    Splitting build vs. query lets a caller process a many-query workload
    in chunks without re-paying the O(N) hash-grid build for every chunk.
    """
    if k > MAX_K:
        raise ValueError(f"k={k} exceeds MAX_K={MAX_K}; bump MAX_K in spatial_topk.py")
    if queries_xy.dim() != 2 or queries_xy.shape[1] != 2:
        raise ValueError(f"queries_xy must be [Q, 2], got {tuple(queries_xy.shape)}")
    if not queries_xy.is_cuda:
        raise ValueError("spatial_topk_xy_with_grid requires CUDA tensors")

    device = spatial_grid.device
    q = int(queries_xy.shape[0])

    queries3 = torch.cat([queries_xy, torch.zeros(q, 1, device=device, dtype=queries_xy.dtype)], dim=1).contiguous()
    qs_wp = wp.from_torch(queries3, dtype=wp.vec3)

    out_idx = torch.full((q, k), -1, dtype=torch.int32, device=device)
    out_dist = torch.full((q, k), float("inf"), dtype=torch.float32, device=device)
    out_idx_wp = wp.from_torch(out_idx, dtype=wp.int32)
    out_dist_wp = wp.from_torch(out_dist, dtype=wp.float32)

    wp.launch(
        _topk_query_kernel,
        dim=q,
        inputs=[spatial_grid.grid.id, spatial_grid._pts_wp, qs_wp, float(radius), int(k)],
        outputs=[out_idx_wp, out_dist_wp],
        device=str(device),
    )
    return out_idx, out_dist


def spatial_topk_xy(
    points_xy: torch.Tensor,
    queries_xy: torch.Tensor,
    k: int,
    radius: float,
    cell_size: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``k`` nearest ``points_xy`` for each ``queries_xy``, within ``radius``.

    Convenience wrapper that builds a single-use :class:`SpatialGrid`
    and queries it. For chunked workloads, prefer
    :func:`build_spatial_grid_xy` + :func:`spatial_topk_xy_with_grid`
    so the grid is paid for once.

    Args:
        points_xy: Reference point positions [m], shape ``[N, 2]``, float.
        queries_xy: Query positions [m], shape ``[Q, 2]``, float.
        k: Number of nearest points to return per query (``<= MAX_K``).
        radius: Search radius [m]. Points farther than ``radius`` are
            never returned. Sized to ``cell_size`` for grid efficiency.
        cell_size: Hash-grid cell edge [m]. ``None`` defaults to
            ``radius * 1.5``.

    Returns:
        Tuple ``(idx, dist)`` of shape ``[Q, k]``:

        * ``idx``: ``int32`` indices into ``points_xy``. ``-1`` for
          slots where no neighbour was found.
        * ``dist``: ``float32`` Euclidean distances. ``+inf`` for
          unfilled slots.
    """
    grid = build_spatial_grid_xy(points_xy, radius=radius, cell_size=cell_size)
    return spatial_topk_xy_with_grid(grid, queries_xy, k=k, radius=radius)
