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

from .kinematic import NewtonKinematics

from ..mdp.retarget.buffer import RetargetBuffer
from ..mdp.retarget.cfg import SamplerBaseCfg
from ..mdp.retarget.pipeline import SamplerBase
from ..terrains.utils.patch_sampling_cfg import CircleFootprintCfg, MorphologicalPatchSamplingCfg

N_ROTATIONS = 4
"""Number of cyclic rotations per support polygon (one per hull corner)."""


@configclass
class SupportPolygonSamplerCfg(SamplerBaseCfg):
    """Configuration for support polygon terrain sampling."""

    class_type: type | str = "{DIR}.sampling:SupportPolygonSampler"
    """Sampler implementation class."""

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

    Derives robot geometry (foot offsets, standing height) from
    :class:`NewtonKinematics` and ``foot_body_ids`` automatically.

    Generates candidate support polygons on a terrain mesh by:

    1. Sampling flat contact patches via morphological filtering.
    2. Sorting each 4-point polygon into convex hull winding order.
    3. Generating 4 cyclic rotations per polygon (each assigns a
       different hull corner to the front-left foot).
    4. Filtering by polygon quality and base height feasibility.

    All rotations go to IK individually. FPS handles spatial
    deduplication at the end.

    Args:
        cfg: Sampling configuration.
        kin: Newton kinematics model (provides default stance).
        foot_body_ids: Newton body indices for the feet.
    """

    def __init__(self, cfg: SupportPolygonSamplerCfg, kin: NewtonKinematics, foot_body_ids: list[int]):
        super().__init__(cfg, kin, foot_body_ids)

        geom = kin.foot_geometry(foot_body_ids)
        self.foot_offsets = geom["foot_offsets"]
        self.foot_ground_offset = geom["foot_ground_offset"]
        self.standing_height = geom["standing_height"]
        self.default_joint_q = kin.default_joint_q

        angles = np.arctan2(self.foot_offsets[:, 1], self.foot_offsets[:, 0])
        self._foot_ccw_order = np.argsort(angles).tolist()

        x_sorted = np.argsort(self.foot_offsets[:, 0])
        self._rear_pair = x_sorted[:2].tolist()
        self._front_pair = x_sorted[2:].tolist()

    @property
    def group_size(self) -> int:
        return 1

    def __call__(
        self,
        wp_mesh: wp.Mesh,
        origin: np.ndarray,
        buffer: RetargetBuffer,
        n_desired: int,
    ) -> tuple[int, dict[str, int]]:
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
        xy = pts[:, :, :2]
        centroid_xy = xy.mean(dim=1, keepdim=True)
        hull_angles = torch.atan2(
            xy[..., 1] - centroid_xy[..., 1],
            xy[..., 0] - centroid_xy[..., 0],
        )
        hull_order = hull_angles.argsort(dim=1)
        hull_pts = torch.gather(pts, 1, hull_order.unsqueeze(-1).expand(-1, -1, 3))

        # Step 3: 4 cyclic rotations
        foot_ccw = self._foot_ccw_order
        assigned_rot = []
        for r in range(N_ROTATIONS):
            a = torch.empty_like(pts)
            for j in range(nc):
                a[:, foot_ccw[j]] = hull_pts[:, (j + r) % nc]
            assigned_rot.append(a)

        assigned = torch.stack(assigned_rot, dim=1).view(K * N_ROTATIONS, nc, 3)
        too_few = too_few.unsqueeze(1).expand(-1, N_ROTATIONS).reshape(K * N_ROTATIONS)

        # Step 4: Quality checks
        fp_pair, rp = self._front_pair, self._rear_pair
        a_xy = assigned[:, :, :2]
        d1 = (a_xy[:, fp_pair[0]] - a_xy[:, rp[1]]).norm(dim=-1)
        d2 = (a_xy[:, fp_pair[1]] - a_xy[:, rp[0]]).norm(dim=-1)
        dr = torch.minimum(d1, d2) / (torch.maximum(d1, d2) + 1e-6)
        front = (a_xy[:, fp_pair[0]] + a_xy[:, fp_pair[1]]) / 2
        rear = (a_xy[:, rp[0]] + a_xy[:, rp[1]]) / 2
        lon = (front - rear).norm(dim=-1)
        lat = ((a_xy[:, fp_pair[0]] - a_xy[:, fp_pair[1]]).norm(dim=-1)
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
        all_valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
        n_all_valid = all_valid_idx.shape[0]
        n_valid = min(n_all_valid, target_n, max_n)

        if n_all_valid > n_valid:
            # FPS to select the most spatially diverse candidates
            from pytorch3d.ops import sample_farthest_points

            base_xyz = base_target[all_valid_idx, :3]
            _, fps_idx = sample_farthest_points(base_xyz.unsqueeze(0), K=n_valid)
            valid_idx = all_valid_idx[fps_idx.squeeze(0)]
        else:
            valid_idx = all_valid_idx

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

        fwd = v_front - v_rear
        yaw_v = torch.atan2(fwd[:, 1], fwd[:, 0])

        default_jq_t = torch.from_numpy(self.default_joint_q).float()
        ji = default_jq_t.unsqueeze(0).expand(n_valid, -1).clone()
        ji[:, 0:3] = v_base
        ji[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw_v)

        br = quat_from_euler_xyz(roll * 0.5, pitch, yaw_v)

        # Reference polygon area (shoelace in CCW order)
        ccw = self._foot_ccw_order
        xy_ref = v_contact[:, :, :2]
        area = torch.zeros(n_valid)
        for i in range(nc):
            j = (i + 1) % nc
            pi = xy_ref[:, ccw[i]]
            pj = xy_ref[:, ccw[j]]
            area = area + (pi[:, 0] * pj[:, 1] - pj[:, 0] * pi[:, 1])
        self.reference_area = (area.abs() * 0.5).to(buffer.device)

        # Active mask (all feet active for quad)
        self.active_mask = torch.ones(n_valid, nc, dtype=torch.int32, device=buffer.device)

        gpu = buffer.device
        buffer.contact_targets_t[:n_valid * nc] = v_contact.view(-1, 3).to(gpu)
        buffer.joint_q_init_t[:n_valid] = ji.to(gpu)
        buffer.base_target_pos_t[:n_valid] = v_base.to(gpu)
        buffer.base_target_rot_t[:n_valid] = br.to(gpu)
        buffer._geom_valid[:n_valid] = True

        buffer.num_written = n_valid
        buffer.num_geometry_valid = n_valid

        return n_valid, reject
