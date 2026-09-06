# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Template-projection contact sampling for the Position terrain-stance family.

Build-time: random-joint FK (with joints clamped to ``default ± fk_joint_range``
to avoid wrap-around leg configurations) produces a library of canonical
foot-polygon shapes; samples whose ``nc`` feet don't form a convex polygon
(self-intersecting CCW ordering or one foot inside the triangle of the
others) are dropped. Per-foot nominal angles and the standing height are
derived from the same FK batch.

Query-time: each candidate picks a random morph patch (center) + random
yaw + random template, un-canonicalizes the template at that pose, and
assigns each foot to a morph patch via a constrained linear-sum-assignment
over ``nc^nc`` combinations. The LSA minimises total distance subject to
(1) distinct patches per contact foot, (2) the snapped stance preserving
the template's slot→CCW-rank permutation (no crossed feet), and (3) the
stance walked in angular order being strictly convex (no foot inside the
triangle of the others). Feet within ``terrain_snap_distance`` of their
assigned patch become **contact** targets (patch + ``foot_ground_offset``);
feet farther away become **air** targets (projected template xy, z clamped
above the local terrain surface).
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from ...kinematics import NewtonKinematics
from ...kinematics.collider_geometry import model_body_collider_z_min
from ...utils.grid_downsample import grid_bucket_downsample
from ..terrains.patch_sampling.cfg import CircleFootprintCfg, MorphologicalPatchSamplingCfg
from ..terrains.patch_sampling.morph import find_flat_patches_morphological
from .buffer import RetargetBuffer
from .canonical_shape import canonicalize_shape
from .cfg import PatchSamplingCfg, SamplerCfg, SamplerSizingCfg
from .fused_sampler_kernel import run_fused_sampler
from .sampler_base import SamplerBase, SamplerOutput, SamplerSizing, compute_sampler_sizing


def _prepare_ik_batched(
    v_contact: torch.Tensor,
    v_base: torch.Tensor,
    v_yaw: torch.Tensor,
    joint_q_seed: torch.Tensor,
    buffer: RetargetBuffer,
) -> None:
    """Plane-fit foot polygon → base pitch/roll → per-problem IK seed.

    Writes ``contact_targets``, ``joint_q_init``, ``base_target_pos`` and
    ``base_target_rot`` into ``buffer`` for the first ``v_contact.shape[0]``
    rows.

    Plane-fit is a closed-form least-squares solve on the foot polygon's
    mean-centered ``(x, y, z)`` covariance. The 2×2 system can be
    ill-conditioned when the foot xy layout is near-collinear (nc=2 biped
    or degenerate polygon), so we only take the fit when
    ``det > 1e-6 * xx * yy`` — a relative rank check that matches the
    degeneracy guard in :func:`~.canonical_shape.canonicalize_shape` so
    shape-match and IK seed agree.

    The base-target quaternion uses **half** of the plane-fit roll
    (``roll * 0.25`` after half-angle), keeping the base more upright on
    sloped terrain; full roll pulled the base near joint limits and
    spiked foot-error.

    Args:
        v_contact: Per-placement foot target positions [m],
            shape ``[n_ik, nc, 3]``.
        v_base: Per-placement base target position [m],
            shape ``[n_ik, 3]``.
        v_yaw: Per-placement base target yaw [rad], shape ``[n_ik]``.
        joint_q_seed: Per-placement generalized-position IK seed
            [m or rad, depending on joint type], shape ``[n_ik, joint_q_count]``.
            Populated from the matched FK
            template's joint configuration so IK starts in the same
            basin of attraction as the target stance, avoiding the
            crossed-leg local minimum that a uniform default-stance
            seed is vulnerable to on twisted poses.
        buffer: Retarget buffer to scatter the prepared IK problem into.
    """
    n_ik, nc, _ = v_contact.shape

    centroid = v_contact.mean(dim=-2, keepdim=True)  # [n_ik, 1, 3]
    delta = v_contact - centroid  # [n_ik, nc, 3]
    dx = delta[..., 0]
    dy = delta[..., 1]
    dz = delta[..., 2]

    xx = (dx * dx).sum(dim=-1)
    yy = (dy * dy).sum(dim=-1)
    xym = (dx * dy).sum(dim=-1)
    xz = (dx * dz).sum(dim=-1)
    yz = (dy * dz).sum(dim=-1)

    det = xx * yy - xym * xym
    plane_rank_ok = det > 1.0e-6 * xx * yy
    det_safe = det.clamp_min(1.0e-12)
    a_raw = (yy * xz - xym * yz) / det_safe
    b_raw = (xx * yz - xym * xz) / det_safe

    # Gate on both rank and z range: flat or degenerate → zero tilt.
    z_range = dz.amax(dim=-1) - dz.amin(dim=-1)
    use_fit = plane_rank_ok & (z_range >= 1.0e-4)
    zero = torch.zeros_like(a_raw)
    a = torch.where(use_fit, a_raw, zero)
    b = torch.where(use_fit, b_raw, zero)

    # Rotate plane slopes ``(a, b) = (dz/dx, dz/dy)_world`` into base frame
    # by ``-yaw``; atan then gives body-frame pitch/roll.
    cy = torch.cos(v_yaw)
    sy = torch.sin(v_yaw)
    pitch = -torch.atan(a * cy + b * sy)
    roll = torch.atan(-a * sy + b * cy)

    # Two quaternions from (yaw, pitch, roll): joint_q_init uses full roll
    # (matches the foot-plane exactly), base_target_rot uses half roll.
    def _euler_to_quat(y: torch.Tensor, p: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        cy2 = torch.cos(y * 0.5)
        sy2 = torch.sin(y * 0.5)
        cp2 = torch.cos(p * 0.5)
        sp2 = torch.sin(p * 0.5)
        cr2 = torch.cos(r * 0.5)
        sr2 = torch.sin(r * 0.5)
        qw = cy2 * cr2 * cp2 + sy2 * sr2 * sp2
        qx = cy2 * sr2 * cp2 - sy2 * cr2 * sp2
        qy = cy2 * cr2 * sp2 + sy2 * sr2 * cp2
        qz = sy2 * cr2 * cp2 - cy2 * sr2 * sp2
        return torch.stack([qx, qy, qz, qw], dim=-1)

    quat_full = _euler_to_quat(v_yaw, pitch, roll)
    quat_half = _euler_to_quat(v_yaw, pitch, roll * 0.5)

    buffer.contact_targets_t[: n_ik * nc] = v_contact.reshape(-1, 3)
    buffer.base_target_pos_t[:n_ik] = v_base
    buffer.base_target_rot_t[:n_ik] = quat_half
    buffer.joint_q_init_t[:n_ik].copy_(joint_q_seed)
    buffer.joint_q_init_t[:n_ik, :3] = v_base
    buffer.joint_q_init_t[:n_ik, 3:7] = quat_full


class Sampler(SamplerBase):
    """Template-projection contact sampler.

    Builds a library of canonical foot-polygon shapes from random-joint
    FK samples (joints clamped to ``default ± fk_joint_range`` to avoid
    wrap-around leg configurations; non-convex-hull samples dropped). At
    query time, each candidate projects a random template at a random
    ``(center, yaw)`` and runs a constrained bipartite assignment over
    ``nc^nc`` combinations to snap feet to morph patches without
    flipping the slot→CCW-rank permutation or collapsing the hull.

    Args:
        cfg: Sampling configuration.
        kin: Newton kinematics model (provides default stance + joint limits).
        foot_body_ids: Newton body indices for the feet.
    """

    def __init__(self, cfg: SamplerCfg, kin: NewtonKinematics, foot_body_ids: list[int], generator: torch.Generator):
        super().__init__(cfg, kin, foot_body_ids, generator)

        foot_z_min = model_body_collider_z_min(kin.builder, tuple(foot_body_ids))
        self.foot_ground_offset = float(-foot_z_min.min())
        self.default_joint_q = kin.default_joint_q

        # Per-foot nominal angle, canonical-shape library, and standing
        # height are derived from the FK distribution inside
        # :meth:`_compute_foot_reachability`. The derivation is invariant
        # to the URDF default base pose (translation + yaw): base held
        # fixed during FK sampling cancels out when every quantity is
        # measured in the polygon-centroid frame.
        self._compute_foot_reachability()

    def _compute_foot_reachability(self) -> None:
        """Build the canonical FK shape library and derived scalars.

        Samples ``cfg.fk_num_samples`` random scalar-joint configurations
        (base held at default pose; per-joint range clamped to
        ``default ± cfg.fk_joint_range`` intersected with URDF limits to
        avoid wrap-around jq whose FK foot positions look plausible but
        whose underlying configuration puts legs on the wrong side of
        the chassis), runs batched FK, canonicalises each polygon
        (centroid-center + plane-fit pitch/roll + per-foot de-nominal),
        drops non-convex-hull samples, and stride-thins the remainder
        to ``cfg.fk_num_retained`` shape samples.

        Stored attributes (all tensors on ``kin.device``):

        * ``_nominal_angle_t`` — per-foot circular-mean angle around
          the polygon centroid [rad], shape ``[nc]``. Defines the
          per-slot "de-nominal" rotation inside
          :func:`~.canonical_shape.canonicalize_shape`.
        * ``_fk_shape_samples`` — canonical foot-polygon shapes,
          ``float32 [n_retained, nc, 3]``.
        * ``_fk_joint_q_seed`` — full generalized-position configurations that
          generated each retained shape, ``float32 [n_retained, joint_q_count]``.
          Used at query time to seed IK from the matched template's
          own joint configuration, avoiding crossed-leg local minima
          that a uniform default-stance seed is vulnerable to.
        * ``standing_height`` — p95 of ``base_z - min_foot_z`` across
          the FK batch [m]. Drives the IK-seed base z lift.
        """
        kin = self.kin
        device = kin.device
        n_samples = int(self.cfg.fk_num_samples)

        jl_lo = torch.tensor(kin.topology.joint_limit_lower, device=device)
        jl_hi = torch.tensor(kin.topology.joint_limit_upper, device=device)
        # Clamp to ``default ± fk_joint_range`` (with per-pattern overrides)
        # so continuous joints (URDF limits ``±1e10`` after USD conversion)
        # don't produce wrap-around jq where e.g. a lateral hip is swung 90° and the
        # leg routes across the chassis. The foot positions of such samples
        # look fine as a convex quad, but the underlying jq makes IK land
        # on a physically crossed-leg pose.
        import re

        default_q = torch.from_numpy(kin.default_joint_q).float().to(device)
        coordinates, velocities, coordinate_names = kin.find_joint_scalar_coordinates(".*")
        if not coordinates:
            raise ValueError("Position reachability sampling requires at least one scalar joint coordinate.")
        coordinate_indices = torch.tensor(coordinates, dtype=torch.long, device=device)
        velocity_indices = torch.tensor(velocities, dtype=torch.long, device=device)
        scalar_default = default_q[coordinate_indices]
        num_scalar = len(coordinates)
        joint_range = torch.full((num_scalar,), float(self.cfg.fk_joint_range), device=device)
        overrides = dict(self.cfg.fk_joint_range_overrides)
        if overrides:
            for pattern, clamp in overrides.items():
                regex = re.compile(pattern)
                for i, name in enumerate(coordinate_names):
                    if regex.fullmatch(name):
                        joint_range[i] = float(clamp)
        scalar_lo = torch.maximum(jl_lo[velocity_indices], scalar_default - joint_range)
        scalar_hi = torch.minimum(jl_hi[velocity_indices], scalar_default + joint_range)

        # Uniform random joint sampling across the clamped physical range.
        jq = default_q.unsqueeze(0).expand(n_samples, -1).contiguous()
        rand_u = torch.rand(n_samples, num_scalar, device=device, generator=self.generator)
        jq[:, coordinate_indices] = rand_u * (scalar_hi - scalar_lo) + scalar_lo

        body_q_wp, _ = kin.eval_fk_batched(wp.from_torch(jq))
        body_q_t = wp.to_torch(body_q_wp).view(n_samples, -1, 7)  # type: ignore[arg-type]
        foot_ids_t = torch.tensor(self.foot_body_ids, device=device, dtype=torch.long)
        foot_xy = body_q_t[:, foot_ids_t, :2]  # [n_samples, nc, 2]

        # Per-foot nominal angle: circular mean of the foot's angle around
        # the per-sample polygon centroid. Robust to ±π wraparound and
        # independent of URDF-declared foot offsets — each foot's random-
        # joint-induced xy distribution is unimodal around its hip-outward
        # direction, so the circular mean recovers that direction.
        centroid_xy = foot_xy.mean(dim=1, keepdim=True)  # [n_samples, 1, 2]
        rel = foot_xy - centroid_xy
        theta = torch.atan2(rel[..., 1], rel[..., 0])
        nominal = torch.atan2(torch.sin(theta).mean(dim=0), torch.cos(theta).mean(dim=0))  # [nc]
        self._nominal_angle_t = nominal.float().contiguous()

        # Standing-height IK seed prior: p95 of ``base_z - min_foot_z``
        # across the FK batch. Using base held fixed at default and
        # quantile-aggregating over random joints isolates the joint-
        # induced variation from the URDF default base z.
        foot_ids_t_world = torch.tensor(self.foot_body_ids, device=device, dtype=torch.long)
        foot_z_min = body_q_t[:, foot_ids_t_world, 2].amin(dim=-1)  # [n_samples]
        base_z = body_q_t[:, 0, 2]  # [n_samples]
        self.standing_height = torch.quantile(base_z - foot_z_min, 0.95)

        # Per-foot canonical shape. Canonicalize each FK-produced polygon
        # the same way query time will (centroid-center + plane-fit +
        # per-foot de-nominal); NN in this space is a pure shape match.
        foot_xyz = body_q_t[:, foot_ids_t, :3]  # [n_samples, nc, 3]

        # Drop samples whose ``nc`` feet don't form a convex polygon in xy
        # (self-intersecting CCW-around-centroid walk, or one foot inside
        # the triangle of the others) OR whose per-slot CCW ordering
        # disagrees with the robot's nominal ordering — the latter catches
        # fully-mirrored stances (e.g., all front legs crossed + all back
        # legs crossed) that keep the hull convex but route every leg
        # across the chassis. Mirrored templates would seed IK into a
        # legs-across-body basin even though ``tgt_perm == tpl_perm`` at
        # query time.
        xy = foot_xyz[..., :2]
        centroid = xy.mean(dim=1, keepdim=True)
        rel_xy = xy - centroid
        angles = torch.atan2(rel_xy[..., 1], rel_xy[..., 0])
        order_ccw = angles.argsort(dim=-1)
        sorted_xy = torch.gather(xy, 1, order_ccw.unsqueeze(-1).expand(-1, -1, 2))
        edges = sorted_xy.roll(-1, dims=1) - sorted_xy
        next_edges = edges.roll(-1, dims=1)
        cross = edges[..., 0] * next_edges[..., 1] - edges[..., 1] * next_edges[..., 0]
        hull_convex = (cross > 0).all(dim=-1) | (cross < 0).all(dim=-1)  # [n_samples]

        # Nominal slot→CCW-rank permutation (the robot's "correct" layout).
        nominal_perm = self._nominal_angle_t.argsort()  # [nc]
        perm_correct = (order_ccw == nominal_perm.unsqueeze(0)).all(dim=-1)  # [n_samples]

        hull_valid = hull_convex & perm_correct
        foot_xyz = foot_xyz[hull_valid]
        joint_q_seed = jq[hull_valid]

        canon_all = canonicalize_shape(foot_xyz, self._nominal_angle_t)
        n_retained = int(min(self.cfg.fk_num_retained, canon_all.shape[0]))
        if n_retained < canon_all.shape[0]:
            # FPS-thin on the flattened canonical shape so each retained
            # template represents a geometrically distinct stance. Stride
            # subsampling would just give another uniform draw from the
            # joint-space distribution — FPS ensures random ``tpl_idx``
            # at query time evenly covers the FK shape manifold.
            flat = canon_all.reshape(canon_all.shape[0], -1)
            keep_idx = grid_bucket_downsample(flat, n_retained, generator=self.generator)
            self._fk_shape_samples = canon_all[keep_idx].contiguous()
            self._fk_joint_q_seed = joint_q_seed[keep_idx].contiguous()
        else:
            self._fk_shape_samples = canon_all.contiguous()
            self._fk_joint_q_seed = joint_q_seed.contiguous()

        # Cache the maximum per-foot distance from template centroid:
        # this bounds how far a projected foot can sit from the candidate
        # centre, and (plus snap_distance with a margin) becomes the
        # ``spatial_topk_xy`` query radius. Using the FK-derived bound
        # keeps the hash-grid search tight for small robots and elastic
        # for larger ones without hard-coded heuristics.
        self._fk_max_foot_reach = float(self._fk_shape_samples[..., :2].norm(dim=-1).max().item())

    def sizing(self, n_desired: int) -> SamplerSizing:
        """Back-derive stage sizes for a given final-robot target.

        Uses yield-rate knobs from :class:`SamplerSizingCfg` so a single
        ``n_desired`` determines every internal stage size.
        """
        sz: SamplerSizingCfg = self.cfg.sizing
        return compute_sampler_sizing(
            n_desired,
            final_fps_oversample=sz.final_fps_oversample,
            criteria_yield=sz.criteria_yield,
            polygon_fps_oversample=sz.polygon_fps_oversample,
            polygon_assembly_yield=sz.polygon_assembly_yield,
            morph_patch_oversample=sz.morph_patch_oversample,
            patches_per_polygon=len(self.foot_body_ids),
        )

    def __call__(
        self,
        wp_mesh: wp.Mesh,
        origin: np.ndarray,
        buffer: RetargetBuffer,
        n_desired: int,
        *,
        seed: int,
    ) -> SamplerOutput:
        """Template-projection query path.

        For each candidate: pick a random morph patch as center, a random
        yaw, and a random template; un-canonicalize the template at that
        pose; solve a constrained LSA that snaps feet to morph patches
        without flipping slot→CCW-rank or collapsing the hull. Feet within
        ``terrain_snap_distance`` of their assigned patch become contact
        targets; feet farther away become air targets (z clamped above
        local terrain).
        """
        patch: PatchSamplingCfg = self.cfg.patch
        nc = buffer.num_contacts
        max_n = buffer.max_candidates

        sizing = self.sizing(n_desired)
        num_patches = max(200, sizing.n_morph_patches)

        fc_cfg = MorphologicalPatchSamplingCfg(
            num_patches=num_patches,
            footprint=CircleFootprintCfg(radius=patch.contact_radius),
            max_height_diff=patch.max_height_diff,
            horizontal_scale=patch.horizontal_scale,
            oversample_ratio=patch.oversample_ratio,
            x_range=patch.x_range if patch.x_range is not None else (-1e6, 1e6),
            y_range=patch.y_range if patch.y_range is not None else (-1e6, 1e6),
        )
        fp = find_flat_patches_morphological(wp_mesh, origin, fc_cfg, generator=self.generator)
        dev_str = fp.device
        origin_t = torch.tensor(origin, dtype=torch.float, device=dev_str)
        fp[:, :3] += origin_t
        patch_pts = fp[:, :3].contiguous()  # [N_p, 3]

        # ``K`` (polygon pool) lives in cheap per-polygon scratches local
        # to this sampler — independent of the buffer's per-body slots.
        # ``target_n`` (post-FPS, what writes into the buffer) is the only
        # value clamped to ``max_n``.
        K = sizing.max_neighborhoods
        target_n = min(n_desired * sizing.oversample_candidates, max_n)

        # Sampler config -- pulled from cfg once and forwarded to the fused kernel.
        radius = float(self.cfg.terrain_snap_distance)
        outward_pen = float(self.cfg.outward_snap_penalty)
        raw_mc = int(self.cfg.min_contacts)
        min_contacts = nc if (raw_mc < 0 or raw_mc > nc) else max(1, raw_mc)
        force_all_snap = min_contacts >= nc
        query_radius = self._fk_max_foot_reach + 4.0 * radius

        outputs = run_fused_sampler(
            seed=seed,
            K=K,
            patch_pts=patch_pts,
            fk_shape_samples=self._fk_shape_samples,
            nominal_angles=self._nominal_angle_t,
            radius=radius,
            query_radius=query_radius,
            outward_pen=outward_pen,
            force_all_snap=force_all_snap,
            foot_ground_offset=self.foot_ground_offset,
        )
        yaws = outputs["yaws"]
        tpl_idx = outputs["tpl_idx"]
        is_contact_full = outputs["is_contact_full"]
        contact_ik = outputs["contact_ik"]
        n_found = outputs["n_found"]
        no_convex = outputs["no_convex"]

        out_of_reach = n_found < min_contacts
        valid = ~out_of_reach & ~no_convex
        all_valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
        n_all_valid = all_valid_idx.shape[0]
        n_valid = min(n_all_valid, target_n, max_n)

        if n_all_valid > n_valid:
            # Reduce before indexing to avoid a transient [n_valid, nc, 3] tensor.
            centroid_xyz = contact_ik.mean(dim=-2)[all_valid_idx]
            local_idx = grid_bucket_downsample(centroid_xyz, n_valid, generator=self.generator)
            valid_idx = all_valid_idx[local_idx]
        else:
            valid_idx = all_valid_idx

        reject = {
            "out_of_reach": int(out_of_reach.sum()),
            "non_convex_stance": int((no_convex & ~out_of_reach).sum()),
        }
        if n_valid == 0:
            buffer.num_written = 0
            buffer.num_geometry_valid = 0
            return SamplerOutput(num_written=0, reject_stats=reject)

        v_contact = contact_ik[valid_idx].contiguous()  # [n_valid, nc, 3]
        is_contact_ik = is_contact_full[valid_idx].contiguous()
        centroid_sel = v_contact.mean(dim=-2)
        v_base = torch.stack(
            [centroid_sel[..., 0], centroid_sel[..., 1], centroid_sel[..., 2] + self.standing_height],
            dim=-1,
        )
        v_yaw = yaws[valid_idx].contiguous()
        joint_q_seed = self._fk_joint_q_seed[tpl_idx[valid_idx]].contiguous()

        n_ik = v_contact.shape[0]
        _prepare_ik_batched(v_contact, v_base, v_yaw, joint_q_seed, buffer)
        buffer._geom_valid[:n_ik] = True
        buffer.num_written = n_ik
        buffer.num_geometry_valid = n_ik

        return SamplerOutput(
            num_written=n_ik,
            reject_stats=reject,
            is_contact=is_contact_ik,
        )
