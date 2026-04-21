# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Support polygon sampling strategy for the retarget pipeline.

Fully batched torch implementation -- no Python for-loops over candidates.
Geometry filtering (neighbor search, convex hull sort, quality checks,
plane fit) runs as vectorized tensor ops.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from ..mdp.retarget.buffer import RetargetBuffer
from ..mdp.retarget.cfg import SamplerBaseCfg
from ..mdp.retarget.pipeline import SamplerBase

N_ROTATIONS = 4
"""Number of cyclic rotations per support polygon (one per hull corner)."""


@configclass
class SupportPolygonSamplerCfg(SamplerBaseCfg):
    """Configuration for support polygon terrain sampling."""

    class_type: type = None  # type: ignore[assignment]
    """Resolved after class definition below."""

    num_candidates: int = 5000
    """Number of flat contact patches to sample on the terrain."""

    contact_radius: float = 0.04
    """Contact patch radius for morphological flatness check [m]."""

    max_height_diff: float = 0.03
    """Maximum height variation within a contact patch [m]."""

    horizontal_scale: float = 0.03
    """Heightmap rasterization grid spacing [m]."""

    oversample_ratio: float = 3.0
    """Oversample factor for farthest-point refinement of candidates."""

    search_radius: float = 0.5
    """Radius around each center point to search for support contacts [m]."""

    min_center_dist: float = 0.05
    """Minimum distance from center to candidate (avoid overlapping) [m]."""

    min_diagonal_ratio: float = 0.3
    """Minimum ratio of shorter to longer diagonal (reject extreme skew)."""

    min_longitudinal_spread: float = 0.1
    """Minimum front-to-back spread [m]."""

    min_lateral_spread: float = 0.05
    """Minimum left-to-right spread [m]."""

    min_diagonal_length: float = 0.15
    """Minimum diagonal length [m]."""

    min_base_above_contacts: float = 0.2
    """Minimum standing height above contact centroid [m]."""

    max_base_above_contacts: float = 1.5
    """Maximum standing height above contact centroid [m]."""

    oversample_candidates: int = 3
    """Geometry candidates per desired output (for oversampling before IK)."""


class SupportPolygonSampler(SamplerBase):
    """Terrain contact sampling via support polygon construction.

    Generates candidate support polygons on a terrain mesh by:

    1. Sampling flat contact patches via morphological filtering.
    2. Sorting each 4-point polygon into convex hull winding order.
    3. Generating 4 cyclic rotations per polygon (each assigns a
       different hull corner to the front-left foot).
    4. Filtering by polygon quality and base height feasibility.

    Each rotation is written to the buffer as an independent candidate.
    The pipeline's IK solver evaluates all rotations in a single batched
    solve, and the best rotation per polygon wins naturally through
    validation and FPS selection.

    Args:
        cfg: Sampling configuration.
        foot_offsets: Nominal contact offsets relative to base ``[num_contacts, 3]`` [m].
        foot_ground_offset: Height of contact body frame above ground [m].
        standing_height: Base height above contact centroid [m].
        default_joint_q: Default joint coordinates ``[joint_coord_count]``.
        reference_poses: Optional library of pre-computed joint configurations
            ``[M, joint_coord_count]`` representing natural stances (walking,
            crouching, turning, etc.).  When provided, each polygon candidate
            is expanded to ``4 * M`` IK problems (4 rotations x M reference
            inits), and the IK seed for each is set to the corresponding
            reference pose.  The pipeline's existing IK + validation + FPS
            selection picks the best (rotation, pose) combination, producing
            more natural variation across placed robots.
            **Not yet implemented** -- accepted but ignored.
    """

    def __init__(
        self,
        cfg: SupportPolygonSamplerCfg,
        *,
        foot_offsets: np.ndarray,
        foot_ground_offset: float,
        standing_height: float,
        default_joint_q: np.ndarray,
        reference_poses: np.ndarray | None = None,
    ):
        super().__init__(cfg)
        self.foot_offsets = foot_offsets
        self.foot_ground_offset = foot_ground_offset
        self.standing_height = standing_height
        self.default_joint_q = default_joint_q
        self.reference_poses = reference_poses

        # Pre-compute Counter Clock Wise winding order of the robot's feet from their
        # default XY offsets.  This maps hull Counter Clock Wise vertex positions to
        # the correct foot indices so the winding directions match.
        angles = np.arctan2(foot_offsets[:, 1], foot_offsets[:, 0])
        self._foot_ccw_order = np.argsort(angles).tolist()

        # Front/rear foot index pairs from X offsets.
        x_sorted = np.argsort(foot_offsets[:, 0])
        self._rear_pair = x_sorted[:2].tolist()
        self._front_pair = x_sorted[2:].tolist()

    @property
    def group_size(self) -> int:
        return N_ROTATIONS

    def __call__(
        self,
        wp_mesh: wp.Mesh,
        origin: np.ndarray,
        buffer: RetargetBuffer,
        n_desired: int,
    ) -> tuple[int, dict[str, int]]:
        from isaaclab_tasks.manager_based.locomotion.position.terrains.utils.patch_sampling_cfg import (
            CircleFootprintCfg,
            MorphologicalPatchSamplingCfg,
        )

        cfg = self.cfg  # type: SupportPolygonSamplerCfg
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
        contact_pts = fp[:, :3].cpu()
        n_pts = contact_pts.shape[0]

        K = min(5000, n_pts)
        target_n = min(n_desired * cfg.oversample_candidates, max_n)

        torch.manual_seed(42)

        # Step 1: Generate K random 4-point selections
        centers = contact_pts[torch.randperm(n_pts)[:K]]
        dists = torch.cdist(centers[:, :2], contact_pts[:, :2])
        in_range = (dists < cfg.search_radius) & (dists > cfg.min_center_dist)
        neighbor_counts = in_range.sum(dim=1)
        too_few = neighbor_counts < nc

        weights = in_range.float()
        weights[too_few] = 1.0
        sel_idx = torch.multinomial(weights, nc)
        pts = contact_pts[sel_idx]  # [K, nc, 3]

        # Step 2: Sort into convex hull winding order
        xy = pts[:, :, :2]  # [K, nc, 2]
        centroid_xy = xy.mean(dim=1, keepdim=True)
        hull_angles = torch.atan2(
            xy[..., 1] - centroid_xy[..., 1],
            xy[..., 0] - centroid_xy[..., 0],
        )
        hull_order = hull_angles.argsort(dim=1)  # [K, nc]
        hull_pts = torch.gather(pts, 1, hull_order.unsqueeze(-1).expand(-1, -1, 3))

        # Step 3: 4 cyclic rotations -- each assigns a different hull
        # corner to the first foot in CCW winding order.  The CCW foot
        # order (pre-computed from default foot offsets) ensures the
        # hull winding matches the robot's body perimeter.
        foot_ccw = self._foot_ccw_order  # e.g. [3,1,0,2] for Go2

        # For each rotation r, hull CCW position j maps to foot foot_ccw[j]
        # assigned[:, foot_ccw[j]] = hull_pts[:, (j+r) % nc]
        assigned_rot = []
        for r in range(N_ROTATIONS):
            a = torch.empty_like(pts)  # [K, nc, 3]
            for j in range(nc):
                a[:, foot_ccw[j]] = hull_pts[:, (j + r) % nc]
            assigned_rot.append(a)

        # Expand all 4 rotations as separate candidates [K*4, nc, 3].
        # The pipeline runs IK on all of them, then collapses to the
        # best per polygon using solver.costs.
        assigned = torch.stack(assigned_rot, dim=1).view(K * N_ROTATIONS, nc, 3)
        too_few = too_few.unsqueeze(1).expand(-1, N_ROTATIONS).reshape(K * N_ROTATIONS)

        # Step 4: Quality checks
        fp, rp = self._front_pair, self._rear_pair
        a_xy = assigned[:, :, :2]  # [K*4, nc, 2]
        d1 = (a_xy[:, fp[0]] - a_xy[:, rp[1]]).norm(dim=-1)
        d2 = (a_xy[:, fp[1]] - a_xy[:, rp[0]]).norm(dim=-1)
        dr = torch.minimum(d1, d2) / (torch.maximum(d1, d2) + 1e-6)
        front = (a_xy[:, fp[0]] + a_xy[:, fp[1]]) / 2
        rear = (a_xy[:, rp[0]] + a_xy[:, rp[1]]) / 2
        lon = (front - rear).norm(dim=-1)
        lat = ((a_xy[:, fp[0]] - a_xy[:, fp[1]]).norm(dim=-1)
               + (a_xy[:, rp[0]] - a_xy[:, rp[1]]).norm(dim=-1)) / 2
        quality_ok = (
            (dr >= cfg.min_diagonal_ratio)
            & (lon >= cfg.min_longitudinal_spread)
            & (lat >= cfg.min_lateral_spread)
            & (torch.minimum(d1, d2) >= cfg.min_diagonal_length)
        )

        # Step 5: Base height check
        contact_ik = assigned.clone()
        contact_ik[:, :, 2] += self.foot_ground_offset
        centroid = contact_ik.mean(dim=1)
        base_target = centroid.clone()
        base_target[:, 2] += self.standing_height
        base_above = base_target[:, 2] - contact_ik[:, :, 2].mean(dim=1)
        height_ok = (
            (base_above > cfg.min_base_above_contacts) & (base_above < cfg.max_base_above_contacts)
        )

        # Step 6: Combine masks, compact, write
        valid = ~too_few & quality_ok & height_ok
        valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
        n_valid = min(valid_idx.shape[0], target_n, max_n)
        valid_idx = valid_idx[:n_valid]

        reject = {
            "too_few": int(too_few.sum()),
            "quality": int((~too_few & ~quality_ok).sum()),
            "base_height": int((~too_few & quality_ok & ~height_ok).sum()),
        }

        if n_valid == 0:
            buffer.num_written = 0
            buffer.num_geometry_valid = 0
            return 0, reject

        v_contact = contact_ik[valid_idx]
        v_base = base_target[valid_idx]
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

        # Yaw from the assigned front-rear axis (per rotation)
        fwd = v_front - v_rear
        yaw_v = torch.atan2(fwd[:, 1], fwd[:, 0])

        default_jq_t = torch.from_numpy(self.default_joint_q).float()
        ji = default_jq_t.unsqueeze(0).expand(n_valid, -1).clone()
        ji[:, 0:3] = v_base
        ji[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw_v)

        br = quat_from_euler_xyz(roll, pitch, yaw_v)

        gpu = buffer.device
        buffer.contact_targets_t[:n_valid * nc] = v_contact.view(-1, 3).to(gpu)
        buffer.joint_q_init_t[:n_valid] = ji.to(gpu)
        buffer.base_target_pos_t[:n_valid] = v_base.to(gpu)
        buffer.base_target_rot_t[:n_valid] = br.to(gpu)
        buffer._geom_valid[:n_valid] = True

        buffer.num_written = n_valid
        buffer.num_geometry_valid = n_valid

        return n_valid, reject


SupportPolygonSamplerCfg.class_type = SupportPolygonSampler
