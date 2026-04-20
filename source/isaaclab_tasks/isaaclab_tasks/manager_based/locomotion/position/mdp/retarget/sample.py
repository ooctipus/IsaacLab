# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 1: Sample contact points on geometry and select support polygons.

Fully batched torch implementation -- no Python for-loops over candidates.
All geometry filtering (neighbor search, permutation assignment, convex
hull, plane fit) runs as vectorized tensor ops.
"""

from __future__ import annotations

from itertools import permutations as _itertools_perms

import numpy as np
import torch
import warp as wp

from isaaclab.utils.math import quat_from_euler_xyz

from .buffer import RetargetBuffer
from .cfg import SupportSamplingCfg

# Pre-computed permutation index table for 4 contacts (24 permutations)
_PERMS_4 = torch.tensor(list(_itertools_perms(range(4))), dtype=torch.long)  # [24, 4]


def sample_contacts(
    wp_mesh: wp.Mesh,
    origin: np.ndarray,
    buffer: RetargetBuffer,
    cfg: SupportSamplingCfg,
    foot_offsets: np.ndarray,
    foot_ground_offset: float,
    standing_height: float,
    default_joint_q: np.ndarray,
    n_desired: int,
) -> tuple[int, dict[str, int]]:
    """Sample contact points on geometry, select support polygons, write to buffer.

    Fully batched: generates K candidate support polygons in parallel,
    filters with vectorized masks, and writes survivors to the buffer.

    Args:
        wp_mesh: Terrain warp mesh.
        origin: Terrain origin offset ``[3]``.
        buffer: Pre-allocated retarget buffer (written in-place).
        cfg: Sampling configuration.
        foot_offsets: Nominal contact offsets relative to base ``[num_contacts, 3]``.
        foot_ground_offset: Height of contact body frame above ground [m].
        standing_height: Base height above contact centroid [m].
        default_joint_q: Default joint coordinates ``[joint_coord_count]``.
        n_desired: Number of valid candidates to aim for.

    Returns:
        ``(num_written, rejection_stats)`` where rejection_stats maps reason to count.
    """
    from isaaclab_tasks.manager_based.locomotion.position.terrains.utils.patch_sampling_cfg import (
        CircleFootprintCfg,
        MorphologicalPatchSamplingCfg,
    )

    nc = buffer.num_contacts
    max_n = buffer.max_candidates

    fc_cfg = MorphologicalPatchSamplingCfg(
        num_patches=cfg.num_candidates,
        footprint=CircleFootprintCfg(radius=cfg.contact_radius),
        max_height_diff=cfg.max_height_diff,
        horizontal_scale=cfg.horizontal_scale,
        oversample_ratio=cfg.oversample_ratio,
    )
    fp = fc_cfg.func(wp_mesh, origin, fc_cfg)
    origin_t = torch.tensor(origin, dtype=torch.float, device=fp.device)
    fp[:, :3] += origin_t
    contact_pts = fp[:, :3].cpu()  # [n_pts, 3]
    n_pts = contact_pts.shape[0]

    K = min(5000, n_pts)
    target_n = min(n_desired * cfg.oversample_candidates, max_n)

    torch.manual_seed(42)
    offsets_xy = torch.from_numpy(foot_offsets[:, :2]).float()  # [nc, 2]
    PERMS = _PERMS_4[:, :nc].contiguous() if nc <= 4 else torch.tensor(
        list(_itertools_perms(range(nc))), dtype=torch.long,
    )

    # ------------------------------------------------------------------
    # Step 1: Generate K random 4-point selections
    # ------------------------------------------------------------------
    centers = contact_pts[torch.randperm(n_pts)[:K]]  # [K, 3]
    dists = torch.cdist(centers[:, :2], contact_pts[:, :2])  # [K, n_pts]
    in_range = (dists < cfg.search_radius) & (dists > cfg.min_center_dist)
    neighbor_counts = in_range.sum(dim=1)  # [K]
    too_few = neighbor_counts < nc

    weights = in_range.float()
    weights[too_few] = 1.0  # avoid multinomial error; masked out later
    sel_idx = torch.multinomial(weights, nc)  # [K, nc]
    pts = contact_pts[sel_idx]  # [K, nc, 3]

    # ------------------------------------------------------------------
    # Step 2: Batched PCA for polygon orientation
    # ------------------------------------------------------------------
    xy = pts[:, :, :2]  # [K, nc, 2]
    centered = xy - xy.mean(dim=1, keepdim=True)
    cov = centered.transpose(1, 2) @ centered  # [K, 2, 2]
    _, eigvecs = torch.linalg.eigh(cov)
    principal = eigvecs[:, :, -1]  # [K, 2]
    yaw = torch.atan2(principal[:, 1], principal[:, 0])  # [K]

    # ------------------------------------------------------------------
    # Step 3: Batched permutation assignment
    # ------------------------------------------------------------------
    cos_y, sin_y = yaw.cos(), yaw.sin()
    rot = torch.stack([cos_y, -sin_y, sin_y, cos_y], dim=-1).view(K, 2, 2)
    nom = centers[:, :2].unsqueeze(1) + torch.einsum("kij,cj->kci", rot, offsets_xy)  # [K, nc, 2]

    n_perms = PERMS.shape[0]
    pts_permed = xy[:, PERMS]  # [K, n_perms, nc, 2]
    costs = (pts_permed - nom.unsqueeze(1)).norm(dim=-1).sum(dim=-1)  # [K, n_perms]
    best_perm_idx = costs.argmin(dim=1)  # [K]
    best_perm = PERMS[best_perm_idx]  # [K, nc]
    assigned = torch.gather(pts, 1, best_perm.unsqueeze(-1).expand(-1, -1, 3))  # [K, nc, 3]

    # ------------------------------------------------------------------
    # Step 4: Batched quality checks
    # ------------------------------------------------------------------
    a_xy = assigned[:, :, :2]  # [K, nc, 2]
    d1 = (a_xy[:, 0] - a_xy[:, 3]).norm(dim=-1)
    d2 = (a_xy[:, 1] - a_xy[:, 2]).norm(dim=-1)
    dr = torch.minimum(d1, d2) / (torch.maximum(d1, d2) + 1e-6)
    front = (a_xy[:, 0] + a_xy[:, 2]) / 2
    rear = (a_xy[:, 1] + a_xy[:, 3]) / 2
    lon = (front - rear).norm(dim=-1)
    lat = ((a_xy[:, 2] - a_xy[:, 0]).norm(dim=-1)
           + (a_xy[:, 3] - a_xy[:, 1]).norm(dim=-1)) / 2
    quality_ok = (
        (dr >= cfg.min_diagonal_ratio)
        & (lon >= cfg.min_longitudinal_spread)
        & (lat >= cfg.min_lateral_spread)
        & (torch.minimum(d1, d2) >= cfg.min_diagonal_length)
    )

    # ------------------------------------------------------------------
    # Step 5: Batched convex hull check
    # ------------------------------------------------------------------
    center_xy = a_xy.mean(dim=1, keepdim=True)  # [K, 1, 2]
    angles = torch.atan2(
        a_xy[..., 1] - center_xy[..., 1],
        a_xy[..., 0] - center_xy[..., 0],
    )  # [K, nc]
    order = angles.argsort(dim=1)
    sorted_xy = torch.gather(a_xy, 1, order.unsqueeze(-1).expand(-1, -1, 2))
    edges = torch.roll(sorted_xy, -1, dims=1) - sorted_xy
    next_e = torch.roll(edges, -1, dims=1)
    cross = edges[..., 0] * next_e[..., 1] - edges[..., 1] * next_e[..., 0]
    convex_ok = (cross > 0).all(dim=1) | (cross < 0).all(dim=1)

    # ------------------------------------------------------------------
    # Step 6: Base height check
    # ------------------------------------------------------------------
    contact_ik = assigned.clone()
    contact_ik[:, :, 2] += foot_ground_offset
    centroid = contact_ik.mean(dim=1)  # [K, 3]
    base_target = centroid.clone()
    base_target[:, 2] += standing_height
    base_above = base_target[:, 2] - contact_ik[:, :, 2].mean(dim=1)
    height_ok = (base_above > cfg.min_base_above_contacts) & (base_above < cfg.max_base_above_contacts)

    # ------------------------------------------------------------------
    # Step 7: Combine masks, compact, write
    # ------------------------------------------------------------------
    valid = ~too_few & convex_ok & quality_ok & height_ok
    valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
    n_valid = min(valid_idx.shape[0], target_n, max_n)
    valid_idx = valid_idx[:n_valid]

    reject = {
        "too_few": int(too_few.sum()),
        "hull<4": int((~too_few & ~convex_ok).sum()),
        "quality": int((~too_few & convex_ok & ~quality_ok).sum()),
        "base_height": int((~too_few & convex_ok & quality_ok & ~height_ok).sum()),
    }

    if n_valid == 0:
        buffer.num_written = 0
        buffer.num_geometry_valid = 0
        return 0, reject

    # Compact valid data
    v_contact = contact_ik[valid_idx]  # [n_valid, nc, 3]
    v_base = base_target[valid_idx]  # [n_valid, 3]
    v_assigned = assigned[valid_idx]  # [n_valid, nc, 3]
    v_front = front[valid_idx]
    v_rear = rear[valid_idx]

    # Batched plane fit via SVD
    c = v_contact - v_contact.mean(dim=1, keepdim=True)
    z_range = c[:, :, 2].max(dim=1).values - c[:, :, 2].min(dim=1).values
    flat = z_range < 1e-4
    _, _, Vh = torch.linalg.svd(c)
    normals = Vh[:, -1]
    normals = torch.where(normals[:, 2:3] < 0, -normals, normals)
    roll = torch.where(flat, torch.zeros_like(z_range), torch.atan2(normals[:, 1], normals[:, 2]))
    pitch = torch.where(flat, torch.zeros_like(z_range), torch.atan2(-normals[:, 0], normals[:, 2]))

    # Yaw from front-rear axis
    fwd = v_front - v_rear
    yaw_v = torch.atan2(fwd[:, 1], fwd[:, 0])

    # Build joint_q_init
    default_jq_t = torch.from_numpy(default_joint_q).float()
    ji = default_jq_t.unsqueeze(0).expand(n_valid, -1).clone()
    ji[:, 0:3] = v_base
    ji[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw_v)

    # Base rotation target (roll damped)
    br = quat_from_euler_xyz(roll * 0.3, pitch, yaw_v)

    # Write to buffer
    gpu = buffer.device
    buffer.contact_targets_t[:n_valid * nc] = v_contact.view(-1, 3).to(gpu)
    buffer.joint_q_init_t[:n_valid] = ji.to(gpu)
    buffer.base_target_pos_t[:n_valid] = v_base.to(gpu)
    buffer.base_target_rot_t[:n_valid] = br.to(gpu)
    buffer._geom_valid[:n_valid] = True

    buffer.num_written = n_valid
    buffer.num_geometry_valid = n_valid

    return n_valid, reject
