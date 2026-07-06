# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Held-asset placement and grasp-pair sampling for Factory task tables.

* :class:`HeldAssetPlacementSampler` binds each held pose to one board and fixed
  asset configuration, then generates assembly, support, or free-space poses.
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
from .criteria import (
    edges_vs_posed_mesh_hit,
    points_min_sd,
    points_vs_body_meshes_min_sd,
    posed_collision_min_sd,
    posed_points,
)

if TYPE_CHECKING:
    from .cfg import (
        FactoryAssemblyPoseGenerateCfg,
        FactoryFreePoseGenerateCfg,
        FactoryGeometryCfg,
        FactorySupportPoseGenerateCfg,
        GraspSamplingCfg,
    )
    from .model import FactoryGeometry

# Bound only the temporary query-by-template distance matrix. Query-row
# chunking preserves the exact nearest-template problem and its source order.
_SEED_DISTANCE_WORKSPACE_BYTES = 256 * 1024 * 1024

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


class HeldAssetPlacementSampler:
    """Sample a board+fixed-asset configuration, then one held-asset pose.

    Per sub-world, the nistboard pose is sampled around its canonical scene pose
    (position + tilt), rejecting board groups that intersect the table or any
    default-pose robot collider. The fixed asset rides the board at its
    ``NIST_BOARD_CFG`` keypoint offset -- the same composition used live -- so
    board and fixed asset always move together. The held asset is then generated
    along the assembly path, on the board, or in free space.

    Placement kinds are separate family stages. This sampler owns only shared
    board geometry and direct per-kind generation operations.
    """

    def __init__(
        self,
        model: FactoryGeometry,
        cfg: FactoryGeometryCfg,
        generator: torch.Generator,
        default_robot_body_q: torch.Tensor,
    ):
        self.model = model
        self.cfg = cfg
        self.device = model.device
        self.generator = generator
        self._board_asset_offsets = {
            name: (
                torch.tensor(getattr(NIST_BOARD_CFG, keypoint).pos, device=self.device).unsqueeze(0),
                torch.tensor(getattr(NIST_BOARD_CFG, keypoint).quat, device=self.device).unsqueeze(0),
            )
            for name, keypoint in cfg.board.fixed_asset_map.items()
        }
        self._board_init_pos = torch.tensor(model.board_init_pos, device=self.device).unsqueeze(0)
        self._board_init_quat = torch.tensor(model.board_init_quat, device=self.device).unsqueeze(0)
        self._board_group_probes = tuple(
            torch.tensor(points, device=self.device) for points in model.board_group_probes
        )
        self._board_group_edge_bodies = wp.array(model.board_group_edge_bodies, dtype=wp.int32, device=self.device)
        self._board_group_edge_p0 = wp.array(model.board_group_edge_p0, dtype=wp.vec3, device=self.device)
        self._board_group_edge_p1 = wp.array(model.board_group_edge_p1, dtype=wp.vec3, device=self.device)
        self._board_group_meshes = (
            model.board_mesh,
            *(model.board_asset_meshes[name] for name in model.board_group_names[1:]),
        )
        self._default_robot_body_q = default_robot_body_q
        target_body = torch.as_tensor(model.robot_target_bodies, dtype=torch.int64, device=self.device)
        target_tf = torch.as_tensor(model.robot_target_tf, device=self.device)
        target_position, target_rotation = math_utils.combine_frame_transforms(
            default_robot_body_q[0, target_body, :3],
            default_robot_body_q[0, target_body, 3:7],
            target_tf[:, :3],
            target_tf[:, 3:7],
        )
        self._default_robot_shape_pose = torch.cat((target_position, target_rotation), dim=-1).contiguous()

    def _default_robot_clear(
        self,
        group_poses: tuple[torch.Tensor, ...],
        group_points: torch.Tensor,
        group_body_q: torch.Tensor,
        tolerance: float,
    ) -> torch.Tensor:
        """Return board groups clear of every default-pose robot collider."""
        count = group_body_q.shape[0]
        robot_body_q = self._default_robot_body_q.expand(count, -1, -1).contiguous()
        clear = torch.ones(count, dtype=torch.bool, device=self.device)
        for mesh, pose in zip(self._board_group_meshes, group_poses, strict=True):
            signed_distance = posed_collision_min_sd(
                robot_body_q,
                self.model.robot_full_probe_bodies_wp,
                self.model.robot_full_probes_wp,
                mesh.id,
                pose,
                0.05,
                self.device,
            )
            crossing = edges_vs_posed_mesh_hit(
                robot_body_q,
                self.model.robot_full_edge_bodies_wp,
                self.model.robot_full_edge_p0_wp,
                self.model.robot_full_edge_p1_wp,
                mesh.id,
                pose,
                self.device,
            )
            clear &= (signed_distance >= -tolerance) & ~crossing

        reverse_distance = points_vs_body_meshes_min_sd(
            group_points,
            robot_body_q,
            self.model.robot_target_body_wp,
            self.model.robot_target_mesh_wp,
            self.model.robot_target_tf_wp,
            0.05,
            self.device,
        )
        clear &= reverse_distance >= -tolerance
        for mesh_id, target_pose in zip(self.model.robot_target_meshes, self._default_robot_shape_pose, strict=True):
            crossing = edges_vs_posed_mesh_hit(
                group_body_q,
                self._board_group_edge_bodies,
                self._board_group_edge_p0,
                self._board_group_edge_p1,
                int(mesh_id),
                target_pose.expand(count, -1),
                self.device,
            )
            clear &= ~crossing
        return clear

    def _sample_board(self, n: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Board poses around the canonical scene pose, clear of table and default robot.

        Locomotion-style single-shot sampling: oversample once, reject boards that
        collide with the table or any default-pose robot collider, then FPS-downsample the
        survivors in (position, tilt) space -- the kept boards are SPREAD over the
        pose range rather than being the first ``n`` valid draws, and there is no
        resample loop. Fails loud when the clear rate is too low for the budget.

        Returns board poses and every mapped board-attached asset pose [m, xyzw].
        """
        rng = self.cfg.board.pose_range
        tol = self.cfg.board.clear_tol
        m = max(int(n * self.cfg.board.oversample), 64)

        def u(key: str) -> torch.Tensor:
            lo, hi = rng.get(key, (0.0, 0.0))
            return torch.empty(m, device=self.device).uniform_(lo, hi, generator=self.generator)

        roll, pitch, yaw = u("roll"), u("pitch"), u("yaw")
        dq = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        pos = self._board_init_pos + torch.stack([u("x"), u("y"), u("z")], dim=-1)
        quat = math_utils.quat_mul(dq, self._board_init_quat.expand(m, -1))
        board = torch.cat((pos, quat), dim=-1)
        attached = {}
        for name, (offset_position, offset_rotation) in self._board_asset_offsets.items():
            position, rotation = math_utils.combine_frame_transforms(
                board[:, :3],
                board[:, 3:7],
                offset_position.expand(m, -1),
                offset_rotation.expand(m, -1),
            )
            attached[name] = torch.cat((position, rotation), dim=-1)
        group_poses = (board, *(attached[name] for name in self.model.board_group_names[1:]))
        group_points = torch.cat(
            [posed_points(probes, pose) for probes, pose in zip(self._board_group_probes, group_poses, strict=True)],
            dim=1,
        )
        group_body_q = torch.stack(group_poses, dim=1).contiguous()
        identity = torch.zeros(m, 7, device=self.device)
        identity[:, 6] = 1.0
        ok = torch.ones(m, dtype=torch.bool, device=self.device)
        for mesh in self.model.static_obstacles.values():
            signed_distance = points_min_sd(group_points, mesh.id, 0.05, self.device)
            crossing = edges_vs_posed_mesh_hit(
                group_body_q,
                self._board_group_edge_bodies,
                self._board_group_edge_p0,
                self._board_group_edge_p1,
                mesh.id,
                identity,
                self.device,
            )
            ok &= (signed_distance >= -tol) & ~crossing
        ok &= self._default_robot_clear(group_poses, group_points, group_body_q, tol)
        idx = ok.nonzero(as_tuple=False).squeeze(-1)
        if idx.shape[0] < n:
            raise RuntimeError(
                f"board-group sampling: only {idx.shape[0]}/{m} oversampled poses are collision-clear but {n} are"
                " needed -- raise board.oversample or shrink board.pose_range"
            )
        if idx.shape[0] > n:
            features = torch.cat((pos[idx], 0.1 * torch.stack((roll, pitch, yaw), dim=-1)[idx]), dim=-1)
            idx = idx[grid_bucket_downsample(features, n, generator=self.generator)]
        return board[idx].contiguous(), {name: pose[idx].contiguous() for name, pose in attached.items()}

    def _on_fixed_asset(
        self, cfg: FactoryAssemblyPoseGenerateCfg, fixed_pose: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate held poses along each fixed asset's assembly path."""
        bands = cfg.assembly_bands
        profile = AssemblyProfile(cfg.assembly_profile)
        n = fixed_pose.shape[0]
        pos_list, quat_list, tag_list = [], [], []
        band_counts = self._split_even(n, len(bands))
        start = 0
        for tag_id, (band, count) in enumerate(zip(bands.values(), band_counts)):
            if count == 0:
                continue
            ap, aq = profile.sample(band, count, self.device, generator=self.generator)
            apw, aqw = math_utils.combine_frame_transforms(
                fixed_pose[start : start + count, :3], fixed_pose[start : start + count, 3:7], ap, aq
            )
            hp, hq = cfg.align_offset.subtract(apw, aqw)
            pos_list.append(hp)
            quat_list.append(hq)
            tag_list.append(torch.full((count,), tag_id, device=self.device, dtype=torch.long))
            start += count
        return torch.cat(pos_list), torch.cat(quat_list), torch.cat(tag_list)

    def _free(self, n: int, rng: dict, pin_z: float | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a held pose uniformly, optionally pinning its world z coordinate [m]."""

        def u(key: str) -> torch.Tensor:
            lo, hi = rng.get(key, (0.0, 0.0))
            return torch.empty(n, device=self.device).uniform_(lo, hi, generator=self.generator)

        z = torch.full((n,), pin_z, device=self.device) if pin_z is not None else u("z")
        pos = torch.stack([u("x"), u("y"), z], dim=-1)
        quat = math_utils.quat_from_euler_xyz(u("roll"), u("pitch"), u("yaw"))
        return pos, quat

    @staticmethod
    def _split_even(n: int, k: int) -> list[int]:
        base, rem = divmod(n, k)
        return [base + (1 if i < rem else 0) for i in range(k)]

    def _bind_boards(
        self, num_placements: int, board_library: tuple[torch.Tensor, dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Bind placements evenly to one declared board library."""
        lib_board, lib_attached = board_library
        board_index = torch.arange(num_placements, device=self.device) % lib_board.shape[0]
        return lib_board[board_index], {name: poses[board_index] for name, poses in lib_attached.items()}, board_index

    def sample_assembly(
        self,
        cfg: FactoryAssemblyPoseGenerateCfg,
        num_placements: int,
        board_library: tuple[torch.Tensor, dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Generate assembly-path placements on every candidate board."""
        board_pose, attached_poses, board_index = self._bind_boards(num_placements, board_library)
        primary_fixed_pose = attached_poses[self.cfg.board.fixed_asset_cfg.name]
        position, rotation, tag = self._on_fixed_asset(cfg, primary_fixed_pose)
        return torch.cat((position, rotation), dim=-1), tag, board_pose, attached_poses, board_index

    def sample_support(
        self,
        cfg: FactorySupportPoseGenerateCfg,
        num_placements: int,
        board_library: tuple[torch.Tensor, dict[str, torch.Tensor]],
        tag_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Generate held-object placements supported by each candidate board."""
        board_pose, attached_poses, board_index = self._bind_boards(num_placements, board_library)
        position, rotation = self._free(num_placements, cfg.pose_range, cfg.table_height)
        local_position, local_rotation = math_utils.subtract_frame_transforms(
            self._board_init_pos.expand(num_placements, -1),
            self._board_init_quat.expand(num_placements, -1),
            position,
            rotation,
        )
        position, rotation = math_utils.combine_frame_transforms(
            board_pose[:, :3], board_pose[:, 3:7], local_position, local_rotation
        )
        tag = torch.full((num_placements,), tag_index, device=self.device, dtype=torch.long)
        return torch.cat((position, rotation), dim=-1), tag, board_pose, attached_poses, board_index

    def sample_free(
        self,
        cfg: FactoryFreePoseGenerateCfg,
        num_placements: int,
        board_library: tuple[torch.Tensor, dict[str, torch.Tensor]],
        tag_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Generate free-space held-object placements for each candidate board."""
        board_pose, attached_poses, board_index = self._bind_boards(num_placements, board_library)
        position, rotation = self._free(num_placements, cfg.pose_range, None)
        tag = torch.full((num_placements,), tag_index, device=self.device, dtype=torch.long)
        return torch.cat((position, rotation), dim=-1), tag, board_pose, attached_poses, board_index


class GraspPairSampler:
    """Antipodal contact-pair sampling on the held asset + FK seed library.

    Retained pairs and their semantic family indices are in the held body frame.
    """

    def __init__(
        self,
        model: FactoryGeometry,
        cfg: GraspSamplingCfg,
        generator: torch.Generator,
    ):
        self.model = model
        self.cfg = cfg
        self.device = model.device
        self.generator = generator
        self._build_pairs()
        self.tpl_feats, self.tpl_approach, self.tpl_arm = self._build_library()

    def _build_pairs(self) -> None:
        """Sample surface points + normals and keep antipodal pairs (FPS-thinned).

        Stores ``pair_a/pair_b [R, 3]`` in the held frame and ``pair_family [R]``
        (region/mode family index into :data:`FAMILY_NAMES`).
        """
        g = self.cfg
        verts = torch.tensor(self.model.held_verts, device=self.device)
        faces = torch.tensor(self.model.held_faces.astype(np.int64), device=self.device)
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        fn = torch.linalg.cross(v1 - v0, v2 - v0)
        areas = 0.5 * fn.norm(dim=-1)
        fi = torch.multinomial(areas, g.n_surface_samples, replacement=True, generator=self.generator)
        u = torch.rand(g.n_surface_samples, device=self.device, generator=self.generator)
        v = torch.rand(g.n_surface_samples, device=self.device, generator=self.generator)
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
        sel = grid_bucket_downsample(feats, min(g.n_pairs_retained, ii.shape[0]), generator=self.generator)
        self.pair_a = pts[ii[sel]].contiguous()
        self.pair_b = pts[jj[sel]].contiguous()
        self.pair_family = family[sel].contiguous()
        self.family_names = FAMILY_NAMES

    def _build_library(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build geometry- and roll-diverse FK seeds over all gripper orientations.

        Returns:
            Pair features, gripper approach directions, and arm seeds. Pair
            features match target geometry; approach directions retain the roll
            degree of freedom that two point targets leave unconstrained.
        """
        m, g = self.model, self.cfg
        arm = m.arm_coords
        lo = torch.tensor(m.kinematics.topology.joint_limit_lower, device=self.device)
        hi = torch.tensor(m.kinematics.topology.joint_limit_upper, device=self.device)
        base = m.arm_stance
        armlo = torch.maximum(lo[m.arm_dofs], base - g.fk_joint_range)
        armhi = torch.minimum(hi[m.arm_dofs], base + g.fk_joint_range)
        jq = torch.zeros(g.fk_num_samples, m.nq, device=self.device)
        jq[:, arm] = armlo + torch.rand(g.fk_num_samples, len(arm), device=self.device, generator=self.generator) * (
            armhi - armlo
        )
        jq[:, m.finger_coords] = 0.02
        from .model import factory_eval_fk

        body_q = factory_eval_fk(m.kinematics, jq)
        pads = torch.stack(
            [
                math_utils.quat_apply(body_q[:, body, 3:7], m.pad_offsets[index].expand(g.fk_num_samples, 3))
                + body_q[:, body, :3]
                for index, body in enumerate(m.pad_bodies)
            ],
            dim=1,
        )
        features = pair_features(pads[:, 0], pads[:, 1], g.seed_axis_scale)
        approach = math_utils.quat_apply(
            body_q[:, m.ee_body, 3:7],
            torch.tensor((0.0, 0.0, 1.0), device=self.device).expand(g.fk_num_samples, -1),
        )
        library_features = torch.cat((features, g.seed_axis_scale * approach), dim=-1)
        keep = grid_bucket_downsample(
            library_features, min(g.fk_num_retained, features.shape[0]), generator=self.generator
        )
        return features[keep].contiguous(), approach[keep].contiguous(), jq[keep][:, arm].contiguous()

    def sample_targets(
        self, held_pose: torch.Tensor, grasps_per_placement: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cast antipodal contact-pair targets onto each held-object pose.

        Args:
            held_pose: Held-asset poses [W, 7] (position [m] and quaternion xyzw).
            grasps_per_placement: Pairs ``G`` sampled (with replacement) per placement.
        Returns:
            World fingertip targets, source-placement indices, and grasp-family indices.
        """
        w, g = held_pose.shape[0], grasps_per_placement
        pick = torch.randint(
            0,
            self.pair_a.shape[0],
            (w * g,),
            device=self.device,
            generator=self.generator,
        )
        held_position = held_pose[:, :3].repeat_interleave(g, 0)
        held_rotation = held_pose[:, 3:7].repeat_interleave(g, 0)
        p_a = math_utils.quat_apply(held_rotation, self.pair_a[pick]) + held_position
        p_b = math_utils.quat_apply(held_rotation, self.pair_b[pick]) + held_position
        # random pad assignment so both wrist-roll branches are explored
        swap = torch.rand(w * g, device=self.device, generator=self.generator) < 0.5
        t_plus = torch.where(swap.unsqueeze(-1), p_b, p_a).contiguous()
        t_minus = torch.where(swap.unsqueeze(-1), p_a, p_b).contiguous()
        world_idx = torch.arange(w, device=self.device).repeat_interleave(g)
        return t_plus, t_minus, world_idx, self.pair_family[pick]

    def seed_targets(
        self, t_plus: torch.Tensor, t_minus: torch.Tensor, ik_seeds_per_grasp: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Repeat targets across nearby arm seeds that span unconstrained roll."""
        k = max(1, min(int(ik_seeds_per_grasp), int(self.tpl_feats.shape[0])))
        target_features = pair_features(t_plus, t_minus, self.cfg.seed_axis_scale)
        pool_size = min(8 * k, int(self.tpl_feats.shape[0]))
        distance_row_bytes = self.tpl_feats.shape[0] * self.tpl_feats.element_size()
        chunk_rows = max(1, _SEED_DISTANCE_WORKSPACE_BYTES // distance_row_bytes)
        seed_indices = torch.empty((t_plus.shape[0], k), dtype=torch.long, device=self.device)

        for start in range(0, t_plus.shape[0], chunk_rows):
            stop = min(start + chunk_rows, t_plus.shape[0])
            feature_distance = torch.cdist(target_features[start:stop], self.tpl_feats)
            pool_indices = feature_distance.topk(pool_size, dim=1, largest=False).indices
            axis = torch.nn.functional.normalize(t_plus[start:stop] - t_minus[start:stop], dim=-1)
            pool_approach = self.tpl_approach[pool_indices]
            pool_approach = pool_approach - (pool_approach * axis[:, None]).sum(dim=-1, keepdim=True) * axis[:, None]
            pool_approach = torch.nn.functional.normalize(pool_approach, dim=-1)
            selected = torch.empty((stop - start, k), dtype=torch.long, device=self.device)
            selected[:, 0] = 0
            roll_distance = 2.0 - 2.0 * (pool_approach * pool_approach[:, :1]).sum(dim=-1)
            roll_distance[:, 0] = -1.0
            rows = torch.arange(stop - start, device=self.device)
            for index in range(1, k):
                next_index = roll_distance.argmax(dim=-1)
                selected[:, index] = next_index
                next_approach = pool_approach[rows, next_index]
                next_distance = 2.0 - 2.0 * (pool_approach * next_approach[:, None]).sum(dim=-1)
                roll_distance = torch.minimum(roll_distance, next_distance)
                roll_distance[rows, next_index] = -1.0
            seed_indices[start:stop].copy_(pool_indices.gather(1, selected))

        seed_index = seed_indices.reshape(-1)
        source_index = torch.arange(t_plus.shape[0], device=self.device).repeat_interleave(k)
        return (
            t_plus.repeat_interleave(k, 0),
            t_minus.repeat_interleave(k, 0),
            self.tpl_arm[seed_index],
            source_index,
        )
