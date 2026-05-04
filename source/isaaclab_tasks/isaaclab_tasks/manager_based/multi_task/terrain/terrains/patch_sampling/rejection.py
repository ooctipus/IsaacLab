# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rejection-sampling patch finders (pre-morphological path).

These functions sample candidate XY positions uniformly in the configured
range, then validate each candidate by ray-casting a small ring of probe
points. They predate :func:`find_flat_patches_morphological` and remain
the default for terrains whose heightmap is either small or degenerate.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from isaaclab.utils.warp import raycast_mesh

from . import cfg as patch_cfg


def uniform_sample_multiple_ranges(
    ranges: list[tuple[float, float]],
    sample_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample ``sample_size`` values uniformly from a union of intervals.

    Each sample is drawn from one of ``ranges`` chosen uniformly at random.

    Args:
        ranges: Non-empty list of ``(low, high)`` intervals.
        sample_size: Number of samples to draw.
        device: Output tensor device.

    Returns:
        Float tensor of shape ``[sample_size]``.
    """
    if not ranges:
        raise ValueError("`ranges` cannot be empty")

    num_intervals = len(ranges)
    interval_indices = torch.randint(low=0, high=num_intervals, size=(sample_size,), device=device)
    samples = torch.empty(sample_size, device=device, dtype=torch.float32)

    for i in range(num_intervals):
        mask = interval_indices == i
        count_i = mask.sum()
        if count_i > 0:
            low, high = ranges[i]
            samples[mask] = torch.empty(count_i, device=device).uniform_(low, high)

    return samples


def find_piecewise_range_flat_patches(
    wp_mesh: wp.Mesh,
    origin: np.ndarray | torch.Tensor | tuple[float, float, float],
    cfg: patch_cfg.PieceWiseRangeFlatPatchSamplingCfg,
) -> torch.Tensor:
    """Find flat patches with piece-wise X/Y/Z ranges (rejection sampling).

    Unlike :func:`find_flat_patches`, each of the three axes may be given
    as a *list* of ``(low, high)`` intervals; candidates are drawn from
    the union and the z-check accepts a ring that lies fully inside
    *any* of the z-intervals.

    Args:
        wp_mesh: Warp mesh to cast against.
        origin: Origin in the mesh frame (shift applied to all ranges).
        cfg: :class:`~cfg.PieceWiseRangeFlatPatchSamplingCfg`.

    Returns:
        Tensor of shape ``(num_patches, 3)`` with patch centres in the
        mesh frame, offset so that ``origin`` is subtracted.

    Raises:
        ValueError: If no valid x- or y-ranges remain after clipping.
        RuntimeError: If ``max_iterations`` is exhausted before success.
    """
    device = wp.device_to_torch(wp_mesh.device)

    if isinstance(cfg.patch_radius, float):
        patch_radius = [cfg.patch_radius]
    else:
        patch_radius = cfg.patch_radius

    if isinstance(origin, np.ndarray):
        origin = torch.from_numpy(origin).float().to(device)
    elif isinstance(origin, torch.Tensor):
        origin = origin.float().to(device)
    else:
        origin = torch.tensor(origin, dtype=torch.float, device=device)

    # Clip each interval to the mesh bounding box and shift by origin.
    mesh_pts = wp_mesh.points.numpy()
    mesh_xmin, mesh_xmax = mesh_pts[:, 0].min(), mesh_pts[:, 0].max()
    mesh_ymin, mesh_ymax = mesh_pts[:, 1].min(), mesh_pts[:, 1].max()

    x_range = [cfg.x_range] if isinstance(cfg.x_range, tuple) else cfg.x_range
    y_range = [cfg.y_range] if isinstance(cfg.y_range, tuple) else cfg.y_range
    z_range = [cfg.z_range] if isinstance(cfg.z_range, tuple) else cfg.z_range

    x_range_clipped = []
    for low, high in x_range:
        new_low = max(low + origin[0].item(), mesh_xmin)
        new_high = min(high + origin[0].item(), mesh_xmax)
        if new_low < new_high:
            x_range_clipped.append((new_low, new_high))

    y_range_clipped = []
    for low, high in y_range:
        new_low = max(low + origin[1].item(), mesh_ymin)
        new_high = min(high + origin[1].item(), mesh_ymax)
        if new_low < new_high:
            y_range_clipped.append((new_low, new_high))

    z_range_shifted = [(low + origin[2].item(), high + origin[2].item()) for low, high in z_range]

    if not x_range_clipped:
        raise ValueError("No valid x-ranges remain after clipping to bounding box.")
    if not y_range_clipped:
        raise ValueError("No valid y-ranges remain after clipping to bounding box.")
    if not z_range_shifted:
        raise ValueError("z_range cannot be empty.")

    # Build a ring of probe points around (0, 0) to test each candidate.
    angle = torch.linspace(0, 2 * np.pi, 10, device=device)
    query_x = []
    query_y = []
    for radius in patch_radius:
        query_x.append(radius * torch.cos(angle))
        query_y.append(radius * torch.sin(angle))
    query_x = torch.cat(query_x).unsqueeze(1)
    query_y = torch.cat(query_y).unsqueeze(1)
    query_points = torch.cat([query_x, query_y, torch.zeros_like(query_x)], dim=-1)

    points_ids = torch.arange(cfg.num_patches, device=device)
    flat_patches = torch.zeros(cfg.num_patches, 3, device=device)

    iter_count = 0
    while len(points_ids) > 0 and iter_count < cfg.max_iterations:
        pos_x = uniform_sample_multiple_ranges(x_range_clipped, len(points_ids), device)
        pos_y = uniform_sample_multiple_ranges(y_range_clipped, len(points_ids), device)

        flat_patches[points_ids, 0] = pos_x
        flat_patches[points_ids, 1] = pos_y

        # Ray-cast each ring point straight down from 100 m above.
        points = flat_patches[points_ids].unsqueeze(1) + query_points
        points[..., 2] = 100.0
        dirs = torch.zeros_like(points)
        dirs[..., 2] = -1.0

        ray_hits = raycast_mesh(points.view(-1, 3), dirs.view(-1, 3), wp_mesh)[0]
        heights = ray_hits.view(points.shape)[..., 2]

        flat_patches[points_ids, 2] = heights[..., -1]

        # Valid iff the whole ring fits inside *any* z-interval and is flat.
        z_ok_mask = torch.zeros(len(points_ids), dtype=torch.bool, device=device)
        for zlow, zhigh in z_range_shifted:
            in_this_range = (heights >= zlow) & (heights <= zhigh)
            fully_in_this_interval = in_this_range.all(dim=1)
            z_ok_mask |= fully_in_this_interval

        height_diff = heights.max(dim=1)[0] - heights.min(dim=1)[0]
        not_valid = (~z_ok_mask) | (height_diff > cfg.max_height_diff)

        points_ids = points_ids[not_valid]
        iter_count += 1

    if len(points_ids) > 0:
        raise RuntimeError(
            "Failed to find valid patches within the maximum number of iterations!\n"
            f"  Iterations: {iter_count}\n"
            f"  Still invalid patches: {len(points_ids)}\n"
            "  Consider adjusting your ranges or max_height_diff."
        )

    return flat_patches - origin


def find_flat_patches(
    wp_mesh: wp.Mesh,
    origin: np.ndarray | torch.Tensor | tuple[float, float, float],
    cfg: patch_cfg.FlatPatchSamplingCfg,
) -> torch.Tensor:
    """Find flat patches via rejection sampling against ``wp_mesh``.

    The search space is defined by an ``origin`` in the mesh frame plus
    scalar X/Y/Z ranges. At each iteration the function samples candidate
    centres, ray-casts a ring of probe points per candidate, and rejects
    those outside ``z_range`` or whose ring height span exceeds
    ``max_height_diff``.

    Args:
        wp_mesh: Warp mesh to cast against.
        origin: Origin in the mesh frame.
        cfg: :class:`~cfg.FlatPatchSamplingCfg`.

    Returns:
        Tensor of shape ``(num_patches, 3)`` with patch centres in the
        mesh frame, offset so that ``origin`` is subtracted.

    Raises:
        RuntimeError: If the function fails to find valid patches within
            10 000 iterations.
    """
    device = wp.device_to_torch(wp_mesh.device)

    patch_radius = [cfg.patch_radius] if isinstance(cfg.patch_radius, float) else cfg.patch_radius

    if isinstance(origin, np.ndarray):
        origin = torch.from_numpy(origin).to(torch.float).to(device)
    elif isinstance(origin, torch.Tensor):
        origin = origin.to(device)
    else:
        origin = torch.tensor(origin, dtype=torch.float, device=device)

    # Clip ranges to the mesh bounding box and shift by origin.
    x_range = (
        max(cfg.x_range[0] + origin[0].item(), wp_mesh.points.numpy()[:, 0].min()),
        min(cfg.x_range[1] + origin[0].item(), wp_mesh.points.numpy()[:, 0].max()),
    )
    y_range = (
        max(cfg.y_range[0] + origin[1].item(), wp_mesh.points.numpy()[:, 1].min()),
        min(cfg.y_range[1] + origin[1].item(), wp_mesh.points.numpy()[:, 1].max()),
    )
    z_range = (
        cfg.z_range[0] + origin[2].item(),
        cfg.z_range[1] + origin[2].item(),
    )

    angle = torch.linspace(0, 2 * np.pi, 10, device=device)
    query_x = []
    query_y = []
    for radius in patch_radius:
        query_x.append(radius * torch.cos(angle))
        query_y.append(radius * torch.sin(angle))
    query_x = torch.cat(query_x).unsqueeze(1)
    query_y = torch.cat(query_y).unsqueeze(1)
    query_points = torch.cat([query_x, query_y, torch.zeros_like(query_x)], dim=-1)

    points_ids = torch.arange(cfg.num_patches, device=device)
    flat_patches = torch.zeros(cfg.num_patches, 3, device=device)

    iter_count = 0
    while len(points_ids) > 0 and iter_count < 10000:
        pos_x = torch.empty(len(points_ids), device=device).uniform_(*x_range)
        pos_y = torch.empty(len(points_ids), device=device).uniform_(*y_range)
        flat_patches[points_ids, :2] = torch.stack([pos_x, pos_y], dim=-1)

        points = flat_patches[points_ids].unsqueeze(1) + query_points
        points[..., 2] = 100.0
        dirs = torch.zeros_like(points)
        dirs[..., 2] = -1.0

        ray_hits = raycast_mesh(points.view(-1, 3), dirs.view(-1, 3), wp_mesh)[0]
        heights = ray_hits.view(points.shape)[..., 2]
        flat_patches[points_ids, 2] = heights[..., -1]

        not_valid = torch.any(torch.logical_or(heights < z_range[0], heights > z_range[1]), dim=1)
        not_valid = torch.logical_or(not_valid, (heights.max(dim=1)[0] - heights.min(dim=1)[0]) > cfg.max_height_diff)

        points_ids = points_ids[not_valid]
        iter_count += 1

    if len(points_ids) > 0:
        # Diagnose why each remaining patch is invalid.
        diag_pos = flat_patches[points_ids]
        diag_pts = diag_pos.unsqueeze(1) + query_points
        diag_pts[..., 2] = 100.0
        diag_dirs = torch.zeros_like(diag_pts)
        diag_dirs[..., 2] = -1.0
        diag_hits = raycast_mesh(diag_pts.view(-1, 3), diag_dirs.view(-1, 3), wp_mesh)[0]
        diag_h = diag_hits.view(diag_pts.shape)[..., 2]
        diag_lines = []
        for i in range(len(points_ids)):
            h = diag_h[i]
            h_min, h_max = h.min().item(), h.max().item()
            h_diff = h_max - h_min
            out_z = torch.any(torch.logical_or(h < z_range[0], h > z_range[1])).item()
            diag_lines.append(
                f"  patch_id={points_ids[i].item()} pos=({diag_pos[i, 0]:.2f}, {diag_pos[i, 1]:.2f})"
                f" h_min={h_min:.3f} h_max={h_max:.3f} h_diff={h_diff:.3f}"
                f" z_out={out_z} heights={h.cpu().tolist()}"
            )
        raise RuntimeError(
            "Failed to find valid patches! Please check the input parameters."
            f"\n\tMaximum number of iterations reached: {iter_count}"
            f"\n\tNumber of invalid patches: {len(points_ids)}"
            f"\n\tMaximum height difference: {cfg.max_height_diff}"
            f"\n\tpatch_radius: {patch_radius}"
            f"\n\torigin: ({origin[0].item():.2f}, {origin[1].item():.2f}, {origin[2].item():.2f})"
            f"\n\tx_range: {x_range}, y_range: {y_range}, z_range: {z_range}"
            f"\n\tInvalid patch details:\n" + "\n".join(diag_lines)
        )

    return flat_patches - origin


def find_flat_patches_by_radius(
    wp_mesh: wp.Mesh,
    origin: np.ndarray | torch.Tensor | tuple[float, float, float],
    cfg: patch_cfg.FlatPatchSamplingByRadiusCfg,
) -> torch.Tensor:
    """Find flat patches whose centres are sampled in a polar annulus.

    Draws ``(radius, angle)`` uniformly from ``cfg.radius_range`` x [0, 2pi)
    around ``origin`` instead of a Cartesian box. Valid patches have all
    ring probe points inside ``z_range`` and a ring height span no larger
    than ``max_height_diff``.

    Args:
        wp_mesh: Warp mesh to cast against.
        origin: Origin in the mesh frame (centre of the annulus).
        cfg: :class:`~cfg.FlatPatchSamplingByRadiusCfg`.

    Returns:
        Tensor of shape ``(num_patches, 3)`` with patch centres in the
        mesh frame, offset so that ``origin`` is subtracted.

    Raises:
        RuntimeError: If ``max_iterations`` is exhausted before success.
    """
    device = wp.device_to_torch(wp_mesh.device)

    if isinstance(cfg.patch_radius, float):
        patch_radius = [cfg.patch_radius]
    else:
        patch_radius = cfg.patch_radius

    if isinstance(origin, np.ndarray):
        origin = torch.from_numpy(origin).float().to(device)
    elif isinstance(origin, torch.Tensor):
        origin = origin.float().to(device)
    else:
        origin = torch.tensor(origin, dtype=torch.float, device=device)

    z_range_shifted = (cfg.z_range[0] + origin[2].item(), cfg.z_range[1] + origin[2].item())

    num_angles = getattr(cfg, "ring_azimuth_samples", 10)
    angle = torch.linspace(0, 2 * np.pi, num_angles, device=device)
    ring_x = []
    ring_y = []
    for radius in patch_radius:
        ring_x.append(radius * torch.cos(angle))
        ring_y.append(radius * torch.sin(angle))

    ring_x = torch.cat(ring_x).unsqueeze(1)
    ring_y = torch.cat(ring_y).unsqueeze(1)
    ring_points = torch.cat([ring_x, ring_y, torch.zeros_like(ring_x)], dim=-1)

    flat_patches = torch.zeros((cfg.num_patches, 3), device=device)
    remaining_ids = torch.arange(cfg.num_patches, device=device)

    oversample_factor = float(getattr(cfg, "oversample_factor", 2.0))
    max_batch_size = getattr(cfg, "max_batch_size", None)
    max_batch_size = int(max_batch_size) if max_batch_size is not None else None

    iteration = 0
    while len(remaining_ids) > 0 and iteration < cfg.max_iterations:
        n_remaining = len(remaining_ids)
        pool = max(int(np.ceil(n_remaining * oversample_factor)), n_remaining)
        if max_batch_size is not None:
            pool = min(pool, max_batch_size)

        r_min, r_max = cfg.radius_range
        cand_radius = torch.empty(pool, device=device).uniform_(r_min, r_max)
        cand_angle = torch.empty(pool, device=device).uniform_(0, 2 * np.pi)

        cand_x = cand_radius * torch.cos(cand_angle) + origin[0]
        cand_y = cand_radius * torch.sin(cand_angle) + origin[1]
        cand_xy = torch.stack([cand_x, cand_y], dim=-1)

        ring_in_world = torch.zeros((pool, ring_points.shape[0], 3), device=device, dtype=torch.float32)
        ring_in_world[..., :2] = cand_xy.unsqueeze(1) + ring_points[..., :2]
        ring_in_world[..., 2] = 100.0
        dirs = torch.zeros_like(ring_in_world)
        dirs[..., 2] = -1.0

        ray_hits = raycast_mesh(ring_in_world.view(-1, 3), dirs.view(-1, 3), wp_mesh)[0]
        ring_hits_3d = ray_hits.view(ring_in_world.shape)

        heights = ring_hits_3d[..., 2]
        out_of_range = (heights < z_range_shifted[0]) | (heights > z_range_shifted[1])
        height_diff = heights.max(dim=1)[0] - heights.min(dim=1)[0]
        valid = (~out_of_range.any(dim=1)) & (height_diff <= cfg.max_height_diff)

        if valid.any():
            valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            take = min(valid_idx.shape[0], n_remaining)
            sel = valid_idx[:take]
            target = remaining_ids[:take]

            flat_patches[target, 0] = cand_xy[sel, 0]
            flat_patches[target, 1] = cand_xy[sel, 1]
            flat_patches[target, 2] = heights[sel, -1]

            remaining_ids = remaining_ids[take:]

        iteration += 1

    if len(remaining_ids) > 0:
        raise RuntimeError(
            f"Failed to find valid patches within {cfg.max_iterations} iterations.\n"
            f"Still invalid patches: {len(remaining_ids)}.\n"
            "Consider relaxing your constraints, increasing oversample_factor, or increasing max_iterations."
        )

    return flat_patches - origin
