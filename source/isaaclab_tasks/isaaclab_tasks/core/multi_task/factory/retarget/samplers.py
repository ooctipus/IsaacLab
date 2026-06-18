# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Two-level sampling for the offline factory IK pipeline.

The pipeline mirrors the terrain foot-sampling structure:

* :class:`NutPlacementSampler` -- sub-worlds (the analog of sub-terrains). Each
  sub-world is one nut placement: concentric on the bolt at a sampled assembly
  fraction (``on_bolt``), resting on the board (``on_table``), or floating freely
  (``in_air``). Always *nut-first*.
* :class:`GraspPairSampler` -- antipodal contact pairs on the held-asset mesh
  (the analog of terrain contact patches). No annotated grasp keypoint and no
  asset-specific parameterization: pairs are surface points with opposed normals
  (within the friction cone) whose separation fits the gripper aperture range.
  Compression (pinch) and expansion (e.g. inside the bore) pairs both qualify;
  region/mode tagging records which grasp family each pair belongs to. An FK
  library over ALL gripper orientations seeds each IK problem from the nearest
  template by pair geometry.

Together they emit ``W`` placements x ``G`` pairs = ``W*G`` per-fingertip IK
targets solved as one batched problem against the single Franka at the origin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

import isaaclab.utils.math as math_utils

from ...utils.grid_downsample import grid_bucket_downsample
from ..assembly_keypoints import NIST_BOARD_CFG
from ..assembly_profile import AssemblyProfile
from .criteria import points_min_sd, posed_points

if TYPE_CHECKING:
    from .cfg import FactoryIKPipelineCfg
    from .model import FactoryIKModel

# per-contact surface regions (held body frame, hole axis = z)
_REGIONS = ("outer", "bore", "axial")
# family id = (min_region * 3 + max_region) * 2 + mode; slots with region_a >
# region_b are never produced (contacts are min/max-sorted) and stay empty.
FAMILY_NAMES = [""] * (len(_REGIONS) * len(_REGIONS) * 2)
for _a in range(len(_REGIONS)):
    for _b in range(_a, len(_REGIONS)):
        for _m, _mode in enumerate(("pinch", "expand")):
            FAMILY_NAMES[(_a * len(_REGIONS) + _b) * 2 + _m] = f"{_REGIONS[_a]}-{_REGIONS[_b]} {_mode}"


def pair_features(p_a: torch.Tensor, p_b: torch.Tensor, axis_scale: float) -> torch.Tensor:
    """Pair-geometry features (midpoint + sign-canonicalized axis) for seeding/FPS."""
    mid = 0.5 * (p_a + p_b)
    axis = torch.nn.functional.normalize(p_b - p_a, dim=-1)
    flip_key = torch.tensor([1.0, 0.7, 0.3], device=p_a.device)
    axis = axis * torch.sign((axis * flip_key).sum(-1, keepdim=True))
    return torch.cat([mid, axis * axis_scale], dim=-1)


class NutPlacementSampler:
    """Sample sub-world placements: a board+bolt assembly group pose, then the nut.

    Per sub-world, the nistboard pose is sampled around its canonical scene pose
    (position + tilt), rejecting boards that penetrate the table or the robot
    base. The fixed asset (bolt) rides the board at its ``NIST_BOARD_CFG``
    keypoint offset -- the same composition ``reset_fixed_assets`` uses live -- so
    board and bolt always move together at the same relative pose. The nut then
    places on the bolt (assembly bands), on the board (resting), or in the air.

    The full tag list is ``list(assembly_bands) + ["on_table", "in_air"]`` so tag
    indices stay stable regardless of which placement types are weighted in.
    """

    def __init__(self, model: FactoryIKModel, cfg: FactoryIKPipelineCfg):
        self.model = model
        self.cfg = cfg
        self.device = cfg.device
        self.bands = cfg.placement.assembly_bands
        self.tag_names = list(self.bands.keys()) + ["on_table", "in_air"]
        self._align_offset = cfg.placement.align_offset
        self._profile = AssemblyProfile(cfg.placement.assembly_profile)
        # bolt-in-board keypoint: the same source of truth reset_fixed_assets used live
        kp = getattr(NIST_BOARD_CFG, cfg.board.fixed_asset_map[cfg.board.fixed_asset_cfg.name])
        self._kp_bolt_pos = torch.tensor(kp.pos, device=self.device).unsqueeze(0)
        self._kp_bolt_quat = torch.tensor(kp.quat, device=self.device).unsqueeze(0)  # xyzw
        self._board_init_pos = torch.tensor(model.board_init_pos, device=self.device).unsqueeze(0)
        self._board_init_quat = torch.tensor(model.board_init_quat, device=self.device).unsqueeze(0)
        self._board_probes = torch.tensor(model.board_probes, device=self.device)

    def _counts(self, num_placements: int) -> dict[str, int]:
        """Split ``num_placements`` across placement types by weight (largest-remainder)."""
        weights = {k: v for k, v in self.cfg.placement.placement_weights.items() if v > 0.0}
        total = sum(weights.values())
        raw = {k: num_placements * v / total for k, v in weights.items()}
        counts = {k: int(np.floor(r)) for k, r in raw.items()}
        leftover = num_placements - sum(counts.values())
        for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[:leftover]:
            counts[k] += 1
        return counts

    def _sample_board(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Board poses around the canonical scene pose, clear of the table and robot base.

        Locomotion-style single-shot sampling: oversample once, reject boards that
        collide with the table or the robot base, then FPS-downsample the
        survivors in (position, tilt) space -- the kept boards are SPREAD over the
        pose range rather than being the first ``n`` valid draws, and there is no
        resample loop. Fails loud when the clear rate is too low for the budget.

        Returns ``(board_pose[n, 7], bolt_pose[n, 7])`` -- the bolt composed at its
        board keypoint, so the pair always shares the same relative pose.
        """
        rng = self.cfg.board.pose_range
        tol = self.cfg.board.clear_tol
        m = max(int(n * self.cfg.board.oversample), 64)

        def u(key: str) -> torch.Tensor:
            lo, hi = rng.get(key, (0.0, 0.0))
            return torch.empty(m, device=self.device).uniform_(lo, hi)

        roll, pitch, yaw = u("roll"), u("pitch"), u("yaw")
        dq = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        pos = self._board_init_pos + torch.stack([u("x"), u("y"), u("z")], dim=-1)
        quat = math_utils.quat_mul(dq, self._board_init_quat.expand(m, -1))
        pts = posed_points(self._board_probes, torch.cat([pos, quat], dim=-1))
        ok = torch.ones(m, dtype=torch.bool, device=self.device)
        for mesh in (*(mesh for mesh in self.model.static_obstacles.values()), self.model.base_mesh):
            # 0.05 m radius: the obstacle colliders are thin closed boxes, so this
            # covers containment; the board never needs deep-interior visibility
            sd = points_min_sd(pts, mesh.id, 0.05, self.device)
            ok &= sd >= -tol
        self.board_stats = {"attempted": m, "clear": int(ok.sum())}
        idx = ok.nonzero(as_tuple=False).squeeze(-1)
        if idx.shape[0] < n:
            raise RuntimeError(
                f"board sampling: only {idx.shape[0]}/{m} oversampled poses are collision-clear but {n} are"
                " needed -- raise board.oversample or shrink board.pose_range"
            )
        if idx.shape[0] > n:
            # FPS in (position [m], rotation delta scaled to 0.1 m/rad) so the kept
            # boards cover the pose range instead of clustering
            feats = torch.cat([pos[idx], 0.1 * torch.stack([roll, pitch, yaw], dim=-1)[idx]], dim=-1)
            idx = idx[grid_bucket_downsample(feats, n)]
        board = torch.cat([pos[idx], quat[idx]], dim=-1)
        bp, bq = math_utils.combine_frame_transforms(
            board[:, :3],
            board[:, 3:7],
            self._kp_bolt_pos.expand(n, -1),
            self._kp_bolt_quat.expand(n, -1),
        )
        return board, torch.cat([bp, bq], dim=-1)

    def _on_bolt(self, bolt_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Nut concentric on each world's bolt, split across the assembly-fraction bands."""
        n = bolt_pose.shape[0]
        pos_list, quat_list, tag_list = [], [], []
        band_counts = self._split_even(n, len(self.bands))
        start = 0
        for tag_id, (band, count) in enumerate(zip(self.bands.values(), band_counts)):
            if count == 0:
                continue
            ap, aq = self._profile.sample(band, count, self.device)
            apw, aqw = math_utils.combine_frame_transforms(
                bolt_pose[start : start + count, :3], bolt_pose[start : start + count, 3:7], ap, aq
            )
            hp, hq = self._align_offset.subtract(apw, aqw)
            pos_list.append(hp)
            quat_list.append(hq)
            tag_list.append(torch.full((count,), tag_id, device=self.device, dtype=torch.long))
            start += count
        return torch.cat(pos_list), torch.cat(quat_list), torch.cat(tag_list)

    def _free(self, n: int, rng: dict, pin_z: float | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Nut at a uniform pose in ``rng`` (roll/pitch default 0; ``pin_z`` overrides z)."""

        def u(key: str) -> torch.Tensor:
            lo, hi = rng.get(key, (0.0, 0.0))
            return torch.empty(n, device=self.device).uniform_(lo, hi)

        z = torch.full((n,), pin_z, device=self.device) if pin_z is not None else u("z")
        pos = torch.stack([u("x"), u("y"), z], dim=-1)
        quat = math_utils.quat_from_euler_xyz(u("roll"), u("pitch"), u("yaw"))
        return pos, quat

    @staticmethod
    def _split_even(n: int, k: int) -> list[int]:
        base, rem = divmod(n, k)
        return [base + (1 if i < rem else 0) for i in range(k)]

    def sample(
        self, num_placements: int, board_library: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample ``W = num_placements`` nut placements bound to board configurations.

        Args:
            num_placements: Nut placements to sample (split across the library).
            board_library: Fixed ``(board_pose[B, 7], bolt_pose[B, 7])``
                configuration library to bind to; ``None`` samples a fresh one of
                size :attr:`BoardLibraryCfg.num_boards`.

        Returns:
            ``(nut_pose[W, 7], tag[W], board_pose[W, 7], bolt_pose[W, 7],
            board_index[W])`` -- poses as pos [m] + quat xyzw in the franka base
            frame, ``board_index`` into the library. The bolt rides the board
            keypoint; on-bolt and on-board nut placements compose through their
            configuration's group pose, in-air placements are world-frame.
        """
        counts = self._counts(num_placements)
        if board_library is None:
            board_library = self._sample_board(self.cfg.board.num_boards)
        lib_board, lib_bolt = board_library
        # bind sub-worlds to configurations cycling through the library, so every
        # placement type covers every board evenly
        board_index = torch.arange(num_placements, device=self.device) % lib_board.shape[0]
        board_pose = lib_board[board_index]
        bolt_pose = lib_bolt[board_index]
        pos_list, quat_list, tag_list, used = [], [], [], 0
        if counts.get("on_bolt", 0) > 0:
            c = counts["on_bolt"]
            p, q, t = self._on_bolt(bolt_pose[used : used + c])
            pos_list.append(p)
            quat_list.append(q)
            tag_list.append(t)
            used += c
        if counts.get("on_table", 0) > 0:
            c = counts["on_table"]
            # sampled in WORLD at the canonical board pose (the existing knob
            # semantics), then re-expressed in the board frame to ride this
            # world's sampled board.
            p, q = self._free(c, self.cfg.placement.on_table_pose_range, self.cfg.placement.table_height)
            lp, lq = math_utils.subtract_frame_transforms(
                self._board_init_pos.expand(c, -1), self._board_init_quat.expand(c, -1), p, q
            )
            board_c = board_pose[used : used + c]
            p, q = math_utils.combine_frame_transforms(board_c[:, :3], board_c[:, 3:7], lp, lq)
            pos_list.append(p)
            quat_list.append(q)
            tag_list.append(torch.full((c,), len(self.bands), device=self.device, dtype=torch.long))
            used += c
        if counts.get("in_air", 0) > 0:
            c = counts["in_air"]
            p, q = self._free(c, self.cfg.placement.in_air_pose_range, None)
            pos_list.append(p)
            quat_list.append(q)
            tag_list.append(torch.full((c,), len(self.bands) + 1, device=self.device, dtype=torch.long))
            used += c
        nut_pose = torch.cat([torch.cat(pos_list), torch.cat(quat_list)], dim=-1)
        return nut_pose, torch.cat(tag_list), board_pose, bolt_pose, board_index


class GraspPairSampler:
    """Antipodal contact-pair sampling on the held asset + FK seed library.

    Besides the retained pairs (``pair_a/pair_b/pair_aperture/pair_family``), the
    sampler keeps its inspection-stage tensors -- ``surface_points/surface_normals``
    (the raw surface samples) and ``candidate_pair_a/candidate_pair_b`` (every
    antipodal pair before FPS thinning) -- so the viser tooling can show the full
    sampling funnel. All are in the held body frame.
    """

    def __init__(self, model: FactoryIKModel, cfg: FactoryIKPipelineCfg):
        self.model = model
        self.cfg = cfg
        self.device = cfg.device
        self._build_pairs()
        self.tpl_feats, self.tpl_arm = self._build_library()
        self.stats["fk_samples"] = cfg.placement.grasp.fk_num_samples
        self.stats["templates"] = int(self.tpl_arm.shape[0])

    def _build_pairs(self) -> None:
        """Sample surface points + normals and keep antipodal pairs (FPS-thinned).

        Stores ``pair_a/pair_b [R, 3]`` (held frame), ``pair_aperture [R]`` [m], and
        ``pair_family [R]`` (region/mode family index into :data:`FAMILY_NAMES`).
        """
        g = self.cfg.placement.grasp
        verts = torch.tensor(self.model.held_verts, device=self.device)
        faces = torch.tensor(self.model.held_faces.astype(np.int64), device=self.device)
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        fn = torch.linalg.cross(v1 - v0, v2 - v0)
        areas = 0.5 * fn.norm(dim=-1)
        fi = torch.multinomial(areas, g.n_surface_samples, replacement=True)
        u = torch.rand(g.n_surface_samples, device=self.device)
        v = torch.rand(g.n_surface_samples, device=self.device)
        flip = u + v > 1.0
        u, v = torch.where(flip, 1.0 - u, u), torch.where(flip, 1.0 - v, v)
        pts = v0[fi] + u.unsqueeze(-1) * (v1 - v0)[fi] + v.unsqueeze(-1) * (v2 - v0)[fi]
        nrm = torch.nn.functional.normalize(fn[fi], dim=-1)

        ii, jj = torch.triu_indices(pts.shape[0], pts.shape[0], offset=1, device=self.device)
        d = pts[jj] - pts[ii]
        dist = d.norm(dim=-1)
        ap_ok = (dist > g.aperture_range[0]) & (dist < g.aperture_range[1])
        dhat = d / dist.clamp_min(1e-9).unsqueeze(-1)
        cos_th = float(np.cos(np.arctan(g.friction_mu)))
        # Compression: pad forces point inward along the pair axis, so the surface
        # normals must oppose it (n_i ~ -dhat, n_j ~ +dhat) within the friction cone.
        ci = -(nrm[ii] * dhat).sum(-1)
        cj = (nrm[jj] * dhat).sum(-1)
        pinch = ap_ok & (ci >= cos_th) & (cj >= cos_th)
        expand = ap_ok & (-ci >= cos_th) & (-cj >= cos_th)
        keep = pinch | expand
        if not bool(keep.any()):
            raise RuntimeError("no antipodal pairs found -- check mesh winding / aperture range / friction_mu")
        self.stats = {"surface": g.n_surface_samples, "pinch": int(pinch.sum()), "expand": int(expand.sum())}
        ii, jj, mode = ii[keep], jj[keep], expand[keep].long()

        # per-contact region (0 outer side / 1 bore / 2 axial face) -> pair family
        r_hat = torch.nn.functional.normalize(pts[:, :2], dim=-1)
        radial = (nrm[:, :2] * r_hat).sum(-1)
        region = torch.full((pts.shape[0],), 2, dtype=torch.long, device=self.device)
        side = nrm[:, 2].abs() < 0.7
        region[side & (radial > 0.5)] = 0
        region[side & (radial < -0.5)] = 1
        ra, rb = region[ii], region[jj]
        family = (torch.minimum(ra, rb) * 3 + torch.maximum(ra, rb)) * 2 + mode

        feats = pair_features(pts[ii], pts[jj], g.seed_axis_scale)
        sel = grid_bucket_downsample(feats, min(g.n_pairs_retained, ii.shape[0]))
        self.surface_points = pts
        self.surface_normals = nrm
        self.candidate_pair_a = pts[ii]
        self.candidate_pair_b = pts[jj]
        self.pair_a = pts[ii[sel]].contiguous()
        self.pair_b = pts[jj[sel]].contiguous()
        self.pair_aperture = (self.pair_b - self.pair_a).norm(dim=-1)
        self.pair_family = family[sel].contiguous()
        self.family_names = FAMILY_NAMES
        self.stats["retained"] = int(self.pair_a.shape[0])

    def _build_library(self) -> tuple[torch.Tensor, torch.Tensor]:
        """FK seed library over ALL gripper orientations (no downward filter).

        Returns ``(features[n_ret, 6], arm_seeds[n_ret, 7])`` -- world pad-pair
        midpoint + canonicalized axis per template, and the arm config behind it.
        """
        m, g = self.model, self.cfg.placement.grasp
        arm = m.arm_coords
        lo = wp.to_torch(m.model.joint_limit_lower).clone()
        hi = wp.to_torch(m.model.joint_limit_upper).clone()
        base = m.arm_stance
        armlo = torch.maximum(lo[arm], base - g.fk_joint_range)
        armhi = torch.minimum(hi[arm], base + g.fk_joint_range)
        jq = torch.zeros(g.fk_num_samples, m.nq, device=self.device)
        jq[:, arm] = armlo + torch.rand(g.fk_num_samples, len(arm), device=self.device) * (armhi - armlo)
        jq[:, m.finger_coords] = 0.02
        b = m.eval_fk(jq)
        pads = torch.stack(
            [
                math_utils.quat_apply(b[:, fb, 3:7], m.pad_offsets[k].expand(g.fk_num_samples, 3)) + b[:, fb, :3]
                for k, fb in enumerate(m.pad_bodies)
            ],
            dim=1,
        )
        feats = pair_features(pads[:, 0], pads[:, 1], g.seed_axis_scale)
        keep = grid_bucket_downsample(feats, min(g.fk_num_retained, feats.shape[0]))
        return feats[keep].contiguous(), jq[keep][:, arm].contiguous()

    def sample(
        self, nut_pose: torch.Tensor, grasps_per_placement: int, ik_seeds_per_grasp: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cast ``G`` antipodal pairs per placement to world and seed each IK problem.

        Args:
            nut_pose: Sub-world nut placements [W, 7] (pos [m] + quat xyzw).
            grasps_per_placement: Pairs ``G`` sampled (with replacement) per placement.
            ik_seeds_per_grasp: Nearby FK templates used as IK seeds per grasp.

        Returns:
            ``(t_plus[W*G, 3], t_minus[W*G, 3], seed_arm[W*G, 7], world_idx[W*G],
            aperture[W*G], family[W*G])`` -- world fingertip targets for the
            (+jaw-y, -jaw-y) pads, the seeded arm config, the originating sub-world
            index, the pair separation [m], and the grasp-family index.
        """
        w, g = nut_pose.shape[0], grasps_per_placement
        pick = torch.randint(0, self.pair_a.shape[0], (w * g,), device=self.device)
        npos = nut_pose[:, :3].repeat_interleave(g, 0)
        nquat = nut_pose[:, 3:7].repeat_interleave(g, 0)
        p_a = math_utils.quat_apply(nquat, self.pair_a[pick]) + npos
        p_b = math_utils.quat_apply(nquat, self.pair_b[pick]) + npos
        # random pad assignment so both wrist-roll branches are explored
        swap = torch.rand(w * g, device=self.device) < 0.5
        t_plus = torch.where(swap.unsqueeze(-1), p_b, p_a).contiguous()
        t_minus = torch.where(swap.unsqueeze(-1), p_a, p_b).contiguous()
        # k nearest FK templates per pair: each grasp becomes k IK problems started
        # from different arm poses (a gradient solve only explores the basin its
        # seed lands in; the 6-dim pad features barely constrain the 7-dof arm, so
        # near templates already span arm branches -- measured equal to explicit
        # arm-spread selection)
        k = max(1, min(int(ik_seeds_per_grasp), int(self.tpl_feats.shape[0])))
        dist = torch.cdist(pair_features(t_plus, t_minus, self.cfg.placement.grasp.seed_axis_scale), self.tpl_feats)
        seed_idx = dist.topk(min(k, dist.shape[1]), dim=1, largest=False).indices.reshape(-1)
        world_idx = torch.arange(w, device=self.device).repeat_interleave(g)
        return (
            t_plus.repeat_interleave(k, 0),
            t_minus.repeat_interleave(k, 0),
            self.tpl_arm[seed_idx],
            world_idx.repeat_interleave(k, 0),
            self.pair_aperture[pick].repeat_interleave(k, 0),
            self.pair_family[pick].repeat_interleave(k, 0),
        )
