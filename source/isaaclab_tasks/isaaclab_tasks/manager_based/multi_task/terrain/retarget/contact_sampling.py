# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Template-projection contact sampling strategy for the retarget pipeline.

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

import time
from contextlib import contextmanager

import numpy as np
import torch
import warp as wp

from ...kinematics import NewtonKinematics
from ..grid_downsample import grid_bucket_downsample
from ..terrains.patch_sampling.cfg import CircleFootprintCfg, MorphologicalPatchSamplingCfg
from ..terrains.patch_sampling.morph import MORPH_TIMINGS
from .buffer import RetargetBuffer
from .canonical_shape import canonicalize_shape
from .cfg import PatchSamplingCfg, SamplerCfg, SamplerSizingCfg
from .sampler_base import SamplerBase, SamplerOutput, SamplerSizing, compute_sampler_sizing
from .spatial_topk import build_spatial_grid_xy, spatial_topk_xy_with_grid


def _prepare_ik_batched(
    v_contact: torch.Tensor,
    v_base: torch.Tensor,
    v_yaw: torch.Tensor,
    jq_rev_seed: torch.Tensor,
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
        jq_rev_seed: Per-placement revolute-joint IK seed [rad],
            shape ``[n_ik, n_rev]``. Populated from the matched FK
            template's joint configuration so IK starts in the same
            basin of attraction as the target stance, avoiding the
            crossed-leg local minimum that a uniform default-stance
            seed is vulnerable to on twisted poses.
        buffer: Retarget buffer to scatter the prepared IK problem into.
    """
    n_ik, nc, _ = v_contact.shape
    n_rev = jq_rev_seed.shape[1]
    jc = 7 + n_rev

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
    buffer.joint_q_init_t[:n_ik, :3] = v_base
    buffer.joint_q_init_t[:n_ik, 3:7] = quat_full
    buffer.joint_q_init_t[:n_ik, 7:jc] = jq_rev_seed


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

    def __init__(self, cfg: SamplerCfg, kin: NewtonKinematics, foot_body_ids: list[int]):
        super().__init__(cfg, kin, foot_body_ids)

        # ``foot_ground_offset`` is the foot-body-to-sole z offset, derived
        # from the foot's actual collision geometry (sphere/capsule/box/mesh).
        geom = kin.foot_geometry(foot_body_ids)
        self.foot_ground_offset = geom["foot_ground_offset"]
        self.default_joint_q = kin.default_joint_q

        # Per-foot nominal angle, canonical-shape library, and standing
        # height are derived from the FK distribution inside
        # :meth:`_compute_foot_reachability`. The derivation is invariant
        # to the URDF default base pose (translation + yaw): base held
        # fixed during FK sampling cancels out when every quantity is
        # measured in the polygon-centroid frame.
        self._compute_foot_reachability()

    def _compute_foot_reachability(self, seed: int = 0) -> None:
        """Build the canonical FK shape library and derived scalars.

        Samples ``cfg.fk_num_samples`` random revolute joint configurations
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
        * ``_fk_joint_q_rev`` — revolute-joint configurations that
          generated each retained shape, ``float32 [n_retained, n_rev]``.
          Used at query time to seed IK from the matched template's
          own joint configuration, avoiding crossed-leg local minima
          that a uniform default-stance seed is vulnerable to.
        * ``standing_height`` — p95 of ``base_z - min_foot_z`` across
          the FK batch [m]. Drives the IK-seed base z lift.
        """
        kin = self.kin
        device = kin.device
        n_samples = int(self.cfg.fk_num_samples)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        jl_lo = wp.to_torch(kin.model.joint_limit_lower)  # type: ignore[arg-type]
        jl_hi = wp.to_torch(kin.model.joint_limit_upper)  # type: ignore[arg-type]
        # Clamp to ``default ± fk_joint_range`` (with per-pattern overrides)
        # so continuous joints (URDF limits ``±1e10`` after USD conversion)
        # don't produce wrap-around jq where e.g. a lateral hip is swung 90° and the
        # leg routes across the chassis. The foot positions of such samples
        # look fine as a convex quad, but the underlying jq makes IK land
        # on a physically crossed-leg pose.
        import re

        default_q = torch.from_numpy(kin.default_joint_q).float().to(device)
        rev_default = default_q[7:]
        n_rev = rev_default.shape[0]
        joint_range = torch.full((n_rev,), float(self.cfg.fk_joint_range), device=device)
        overrides = dict(self.cfg.fk_joint_range_overrides)
        if overrides:
            # ``joint_names[0]`` is the floating-base root; revolute names
            # follow in 1-based slot order matching jl_lo[6:] / jl_hi[6:].
            rev_names = kin.joint_names[1 : 1 + n_rev]
            for pattern, clamp in overrides.items():
                regex = re.compile(pattern)
                for i, name in enumerate(rev_names):
                    if regex.fullmatch(name):
                        joint_range[i] = float(clamp)
        rev_lo = torch.maximum(jl_lo[6:], rev_default - joint_range)
        rev_hi = torch.minimum(jl_hi[6:], rev_default + joint_range)

        # Uniform random joint sampling across the clamped physical range.
        gen = torch.Generator(device=device).manual_seed(seed)
        jq = default_q.unsqueeze(0).expand(n_samples, -1).contiguous()
        rand_u = torch.rand(n_samples, n_rev, device=device, generator=gen)
        jq[:, 7 : 7 + n_rev] = rand_u * (rev_hi - rev_lo) + rev_lo

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
        self.standing_height = float(torch.quantile(base_z - foot_z_min, 0.95).item())

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
        n_hull_valid = int(hull_valid.sum().item())
        if n_hull_valid < n_samples:
            foot_xyz = foot_xyz[hull_valid]
            jq_rev = jq[hull_valid, 7 : 7 + n_rev]
        else:
            jq_rev = jq[:, 7 : 7 + n_rev]

        canon_all = canonicalize_shape(foot_xyz, self._nominal_angle_t)
        n_retained = int(min(self.cfg.fk_num_retained, canon_all.shape[0]))
        if n_retained < canon_all.shape[0]:
            # FPS-thin on the flattened canonical shape so each retained
            # template represents a geometrically distinct stance. Stride
            # subsampling would just give another uniform draw from the
            # joint-space distribution — FPS ensures random ``tpl_idx``
            # at query time evenly covers the FK shape manifold.
            flat = canon_all.reshape(canon_all.shape[0], -1)
            keep_idx = grid_bucket_downsample(flat, n_retained)
            self._fk_shape_samples = canon_all[keep_idx].contiguous()
            self._fk_joint_q_rev = jq_rev[keep_idx].contiguous()
        else:
            self._fk_shape_samples = canon_all.contiguous()
            self._fk_joint_q_rev = jq_rev.contiguous()

        # Cache the maximum per-foot distance from template centroid:
        # this bounds how far a projected foot can sit from the candidate
        # centre, and (plus snap_distance with a margin) becomes the
        # ``spatial_topk_xy`` query radius. Using the FK-derived bound
        # keeps the hash-grid search tight for small robots and elastic
        # for larger ones without hard-coded heuristics.
        self._fk_max_foot_reach = float(self._fk_shape_samples[..., :2].norm(dim=-1).max().item())

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.init_info = (
            f"reachability: {n_samples} FK samples in {dt:.3f}s"
            f" ({n_samples - n_hull_valid} dropped for non-convex hull,"
            f" retained {self._fk_shape_samples.shape[0]} shape samples)"
        )

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

    @contextmanager
    def _time(self, name: str):
        """Record wall time for a sampler sub-phase, with CUDA sync on enter/exit."""
        dev = self.kin.device
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            self.sub_timings[name] = self.sub_timings.get(name, 0.0) + (time.perf_counter() - t0)

    def __call__(
        self,
        wp_mesh: wp.Mesh,
        origin: np.ndarray,
        buffer: RetargetBuffer,
        n_desired: int,
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
        self.sub_timings.clear()
        patch: PatchSamplingCfg = self.cfg.patch
        nc = buffer.num_contacts
        max_n = buffer.max_candidates
        device = self.kin.device

        sizing = self.sizing(n_desired)
        num_patches = max(200, sizing.n_morph_patches)

        with self._time("morph"):
            fc_cfg = MorphologicalPatchSamplingCfg(
                num_patches=num_patches,
                footprint=CircleFootprintCfg(radius=patch.contact_radius),
                max_height_diff=patch.max_height_diff,
                horizontal_scale=patch.horizontal_scale,
                oversample_ratio=patch.oversample_ratio,
                x_range=patch.x_range if patch.x_range is not None else (-1e6, 1e6),
                y_range=patch.y_range if patch.y_range is not None else (-1e6, 1e6),
            )
            fp = fc_cfg.func(wp_mesh, origin, fc_cfg)
            dev_str = fp.device
            origin_t = torch.tensor(origin, dtype=torch.float, device=dev_str)
            fp[:, :3] += origin_t
            patch_pts = fp[:, :3].contiguous()  # [N_p, 3]
            n_pts = patch_pts.shape[0]
        for _sub_name, _sub_dt in MORPH_TIMINGS.items():
            key = f"morph.{_sub_name}"
            self.sub_timings[key] = self.sub_timings.get(key, 0.0) + _sub_dt

        # ``K`` (polygon pool) lives in cheap per-polygon scratches local
        # to this sampler — independent of the buffer's per-body slots.
        # ``target_n`` (post-FPS, what writes into the buffer) is the only
        # value clamped to ``max_n``.
        K = sizing.max_neighborhoods
        target_n = min(n_desired * sizing.oversample_candidates, max_n)

        torch.manual_seed(42)
        if torch.device(dev_str).type == "cuda":
            torch.cuda.manual_seed_all(42)

        with self._time("project"):
            # Constants for the un-canonicalization step. The actual
            # ``[K, nc, ...]`` candidate generation runs INSIDE the
            # chunk loop below so the working set is bounded by
            # ``chunk_K`` rather than full ``K``. Holding ``yaws``,
            # ``tpl_canon``, ``projected_world`` upfront at the dense
            # ``pool_spacing=0.05 m`` setting (K~60M) costs several GiB
            # of intermediates *before* the LSA stage.
            cos_n = torch.cos(self._nominal_angle_t)  # [nc]
            sin_n = torch.sin(self._nominal_angle_t)
            n_tpl = self._fk_shape_samples.shape[0]

            # Optimal one-patch-per-foot assignment (brute-force bipartite).
            #
            # Contact is terrain-determined: if a morph patch is within
            # ``terrain_snap_distance`` of a foot's projected xy, that foot
            # contacts the patch. Each patch can be claimed by at most one
            # foot per candidate. Greedy "closest-foot-wins" isn't optimal
            # when two feet compete for the same patch, so we solve the
            # full linear-sum-assignment: for each of ``nc^nc`` combinations
            # of "each foot picks one of its nc-nearest patches", score
            # total distance (contact cost = dist; air cost = radius) with
            # distinctness, convexity, and winding-preservation constraints;
            # pick the combo with min total cost. For nc=4 this is 256
            # combinations per candidate; fully vectorized over K.
            patch_xy = patch_pts[:, :2].contiguous()
            radius = float(self.cfg.terrain_snap_distance)
            outward_pen = float(self.cfg.outward_snap_penalty)
            raw_mc = int(self.cfg.min_contacts)
            min_contacts = nc if (raw_mc < 0 or raw_mc > nc) else max(1, raw_mc)
            force_all_snap = min_contacts >= nc

            # Per-foot top-``nc`` patch lookup via Warp ``HashGrid``. The
            # cdist+topk path materialises an ``[K*nc, N_p]`` distance
            # matrix that on large terrains (e.g. 200x200 m pit with
            # K~500k, N_p~100k) hits hundreds of GiB; a spatial hash
            # collapses the cost to ``O(K * nc * patches_in_radius)``
            # with bounded memory regardless of K or N_p. Search radius
            # = max-foot-reach + a few snap-distances so force-snap
            # candidates always find their nearest patches even when the
            # template projects a foot at the edge of the reach envelope.
            # ``force_all_snap`` allows snapping up to the full query
            # radius; soft-polygon paths still gate contact at the
            # configured ``terrain_snap_distance``.
            query_radius = self._fk_max_foot_reach + 4.0 * radius
            effective_radius = query_radius if force_all_snap else radius

            # Build the per-foot patch hash grid ONCE; query it per chunk
            # below. Holding the full ``[K * nc, nc]`` topk output upfront
            # would peak at K * nc * nc * 8 B (idx+dist), which on dense
            # pools (e.g. ``pool_spacing=0.05 m``, K~60M placements) is
            # several GiB before any LSA work runs.
            patch_grid = build_spatial_grid_xy(patch_xy, radius=query_radius)

            # Enumerate nc^nc combinations once: ``combo[c, i]`` = rank
            # (0..nc-1) that foot ``i`` picks under combination ``c``.
            rank_axes = [torch.arange(nc, device=device)] * nc
            combo = torch.cartesian_prod(*rank_axes).view(-1, nc)  # [nc^nc, nc]
            C = combo.shape[0]
            neg_sentinel = -torch.arange(1, nc + 1, device=device).view(1, 1, nc)

            # Chunk K through the candidate-projection + spatial-topk +
            # LSA pipeline. The hot rows are ``[Kc, C, nc]`` float / long
            # tensors (gather, cost, contact-mask, target, edges, ...)
            # plus the wider ``[Kc, C, nc, 2]`` tensors (proj_xy_exp,
            # target_xy, tgt_rel, target_sorted, edges, next_edges).
            # Empirically ~16 of these float-sized scratches coexist at
            # peak during the convex-validity step, plus a handful of
            # long-typed gather/permute tensors at 2x the float
            # footprint, giving an effective per-row peak around
            # ``20 × C × nc × 4`` bytes.
            #
            # Budget = ``0.9 × free GPU bytes``, after a one-shot
            # ``empty_cache`` to release any cached blocks the upstream
            # allocator has been hoarding. Aggressive on purpose: the
            # downstream FPS + IK stages stream their work and don't
            # need much simultaneous headroom, and the per-chunk peak
            # inside the loop is bounded by ``per_row_bytes``. No hard
            # cap -- on a clean 50+ GiB GPU we want a chunk in the tens
            # of GiB so ``n_chunks`` stays in the low hundreds.
            torch_device = torch.device(device) if isinstance(device, str) else device
            per_row_bytes = 32 * C * nc * 4 + nc * nc * 8  # LSA + topk
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
                free_bytes, _ = torch.cuda.mem_get_info(torch_device)
                lsa_budget = max(256 * 1024 * 1024, int(0.90 * free_bytes))
            else:
                lsa_budget = 1 * 1024 * 1024 * 1024
            chunk_K = max(1, min(K, lsa_budget // max(1, per_row_bytes)))
            n_chunks = (K + chunk_K - 1) // chunk_K
            print(
                f"[contact_sampling] candidate gen + spatial-topk + LSA over K={K} placements: "
                f"chunk_K={chunk_K}, n_chunks={n_chunks}, "
                f"budget={lsa_budget // (1024 * 1024)} MiB, per_row={per_row_bytes // 1024} KiB",
                flush=True,
            )

            is_contact_chunks: list[torch.Tensor] = []
            contact_ik_chunks: list[torch.Tensor] = []
            n_found_chunks: list[torch.Tensor] = []
            no_convex_chunks: list[torch.Tensor] = []
            yaws_chunks: list[torch.Tensor] = []
            tpl_idx_chunks: list[torch.Tensor] = []
            for k0 in range(0, K, chunk_K):
                k1 = min(k0 + chunk_K, K)
                Kc = k1 - k0

                # Per-chunk candidate generation. Each chunk samples its
                # own (center, yaw, template_idx) triples and produces a
                # ``[Kc, nc, 3]`` ``proj_chunk`` -- so the ``[K, ...]``
                # tensors that previously dominated the projection step
                # never exist at full size.
                centers_c = patch_pts[torch.randint(0, n_pts, (Kc,), device=dev_str)]  # [Kc, 3]
                yaws_c = torch.rand(Kc, device=dev_str) * (2.0 * np.pi)
                tpl_idx_c = torch.randint(0, n_tpl, (Kc,), device=dev_str)
                tpl_canon_c = self._fk_shape_samples[tpl_idx_c]  # [Kc, nc, 3]

                cx = tpl_canon_c[..., 0]  # [Kc, nc]
                cy_ = tpl_canon_c[..., 1]
                cz = tpl_canon_c[..., 2]
                mid_x = cos_n * cx - sin_n * cy_  # [Kc, nc]
                mid_y = sin_n * cx + cos_n * cy_
                cos_y = torch.cos(yaws_c).unsqueeze(-1)  # [Kc, 1]
                sin_y = torch.sin(yaws_c).unsqueeze(-1)
                world_x = cos_y * mid_x - sin_y * mid_y + centers_c[:, 0:1]  # [Kc, nc]
                world_y = sin_y * mid_x + cos_y * mid_y + centers_c[:, 1:2]
                world_z = cz + centers_c[:, 2:3]
                proj_chunk = torch.stack([world_x, world_y, world_z], dim=-1)  # [Kc, nc, 3]
                proj_xy = proj_chunk[..., :2]  # [Kc, nc, 2]

                # Per-chunk top-nc lookup. ``proj_xy_chunk_flat`` is the
                # chunk's ``[Kc * nc, 2]`` foot-projection queries against
                # the prebuilt grid; the output ``[Kc * nc, nc]`` is
                # bounded by ``chunk_K``, never the full ``K``.
                proj_xy_chunk_flat = proj_xy.reshape(-1, 2).contiguous()
                topk_i_chunk_flat, topk_d_chunk_flat = spatial_topk_xy_with_grid(
                    patch_grid, proj_xy_chunk_flat, k=nc, radius=query_radius
                )
                topk_i = topk_i_chunk_flat.view(Kc, nc, nc).to(torch.long).clamp(min=0)
                topk_d = topk_d_chunk_flat.view(Kc, nc, nc)

                # Expand into combo space.
                gather_rank = combo.view(1, C, nc, 1).expand(Kc, C, nc, 1)
                chosen_i = torch.gather(topk_i.unsqueeze(1).expand(-1, C, -1, -1), 3, gather_rank).squeeze(-1)
                chosen_d = torch.gather(topk_d.unsqueeze(1).expand(-1, C, -1, -1), 3, gather_rank).squeeze(-1)

                # Contact mask: forced True under ``force_all_snap``.
                contact_mask_c = chosen_d < effective_radius  # [Kc, C, nc]

                # Outward-snap-aware contact cost.
                tpl_centroid_xy = proj_xy.mean(dim=1, keepdim=True)  # [Kc, 1, 2]
                tpl_r = (proj_xy - tpl_centroid_xy).norm(dim=-1)  # [Kc, nc]
                patch_xy_gather = patch_xy[chosen_i.view(-1)].view(Kc, C, nc, 2)
                patch_r = (patch_xy_gather - tpl_centroid_xy.unsqueeze(1)).norm(dim=-1)
                outward = (patch_r - tpl_r.unsqueeze(1)).clamp(min=0)
                contact_cost = chosen_d + outward_pen * outward
                cost_per_foot = torch.where(contact_mask_c, contact_cost, torch.full_like(chosen_d, radius))
                total_cost = cost_per_foot.sum(dim=-1)  # [Kc, C]

                if not force_all_snap:
                    foot_has_contact_option = topk_d[:, :, 0] < radius  # [Kc, nc]
                    combo_violates_must_contact = (foot_has_contact_option.unsqueeze(1) & ~contact_mask_c).any(dim=-1)
                    total_cost = torch.where(
                        combo_violates_must_contact, torch.full_like(total_cost, float("inf")), total_cost
                    )

                # Distinctness penalty (contact feet must pick distinct patches).
                eff_idx = torch.where(contact_mask_c, chosen_i, neg_sentinel)
                sorted_eff = eff_idx.sort(dim=-1).values
                has_dup = (sorted_eff[..., 1:] == sorted_eff[..., :-1]).any(dim=-1)
                total_cost = torch.where(has_dup, torch.full_like(total_cost, float("inf")), total_cost)

                # Winding preservation.
                tpl_rel = proj_xy - tpl_centroid_xy
                tpl_perm = torch.atan2(tpl_rel[..., 1], tpl_rel[..., 0]).argsort(dim=-1)  # [Kc, nc]

                proj_xy_exp = proj_xy.unsqueeze(1).expand(-1, C, -1, -1)
                target_xy = torch.where(contact_mask_c.unsqueeze(-1), patch_xy_gather, proj_xy_exp)
                tgt_rel = target_xy - target_xy.mean(dim=2, keepdim=True)
                tgt_perm = torch.atan2(tgt_rel[..., 1], tgt_rel[..., 0]).argsort(dim=-1)
                perm_match = (tgt_perm == tpl_perm.unsqueeze(1)).all(dim=-1)

                # Hull validity (strictly convex in CCW order).
                target_sorted = torch.gather(target_xy, 2, tgt_perm.unsqueeze(-1).expand(-1, -1, -1, 2))
                edges = target_sorted.roll(-1, dims=2) - target_sorted
                next_edges = edges.roll(-1, dims=2)
                cross = edges[..., 0] * next_edges[..., 1] - edges[..., 1] * next_edges[..., 0]
                convex_sorted = (cross > 0).all(dim=-1) | (cross < 0).all(dim=-1)
                convex_valid = perm_match & convex_sorted
                total_cost = torch.where(convex_valid, total_cost, torch.full_like(total_cost, float("inf")))

                # Pick best combo; emit per-K outputs.
                best_c = total_cost.argmin(dim=-1)  # [Kc]
                row_idx = torch.arange(Kc, device=device)
                no_convex_chunks.append(total_cost[row_idx, best_c].isinf())
                nearest_idx_c = chosen_i[row_idx, best_c]  # [Kc, nc]
                nearest_dist_c = chosen_d[row_idx, best_c]
                # is_contact mirrors the LSA's effective radius: under
                # force_all_snap a foot is contact iff the hash grid found
                # any patch within the (generous) query radius; otherwise
                # the foot is genuinely out of reach and the candidate
                # falls through ``out_of_reach`` below.
                is_contact_c = nearest_dist_c < effective_radius
                patch_gather = patch_pts[nearest_idx_c.view(-1)].view(Kc, nc, 3)
                patch_gather[..., 2] = patch_gather[..., 2] + self.foot_ground_offset

                # Air-foot z clamp above local terrain.
                local_terrain_z = patch_pts[topk_i[:, :, 0].reshape(-1), 2].view(Kc, nc)
                air_floor_z = local_terrain_z + self.foot_ground_offset
                proj_clamped = proj_chunk.clone()
                proj_clamped[..., 2] = torch.maximum(proj_chunk[..., 2], air_floor_z)
                contact_ik_c = torch.where(is_contact_c.unsqueeze(-1), patch_gather, proj_clamped)

                is_contact_chunks.append(is_contact_c)
                contact_ik_chunks.append(contact_ik_c)
                n_found_chunks.append(is_contact_c.sum(dim=-1))
                yaws_chunks.append(yaws_c)
                tpl_idx_chunks.append(tpl_idx_c)

            is_contact_full = torch.cat(is_contact_chunks, dim=0)  # [K, nc]
            contact_ik = torch.cat(contact_ik_chunks, dim=0)  # [K, nc, 3]
            n_found = torch.cat(n_found_chunks, dim=0)  # [K]
            no_convex = torch.cat(no_convex_chunks, dim=0)  # [K]
            yaws = torch.cat(yaws_chunks, dim=0)  # [K]
            tpl_idx = torch.cat(tpl_idx_chunks, dim=0)  # [K]
            out_of_reach = n_found < min_contacts

            valid = ~out_of_reach & ~no_convex
            all_valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
            n_all_valid = all_valid_idx.shape[0]
            n_valid = min(n_all_valid, target_n, max_n)

        with self._time("sampler_fps"):
            if n_all_valid > n_valid:
                centroid_xyz = contact_ik[all_valid_idx].mean(dim=-2)
                local_idx = grid_bucket_downsample(centroid_xyz, n_valid)
                valid_idx = all_valid_idx[local_idx]
            else:
                valid_idx = all_valid_idx

        reject = {
            "out_of_reach": int(out_of_reach.sum()),
            "non_convex_stance": int((no_convex & ~out_of_reach).sum()),
        }
        diagnostics: dict[str, object] = {
            "n_contact_histogram": torch.bincount(n_found[valid_idx], minlength=nc + 1).detach().clone(),
        }

        if n_valid == 0:
            buffer.num_written = 0
            buffer.num_geometry_valid = 0
            return SamplerOutput(num_written=0, reject_stats=reject, diagnostics=diagnostics)

        with self._time("prepare_ik"):
            v_contact = contact_ik[valid_idx].contiguous()  # [n_valid, nc, 3]
            is_contact_ik = is_contact_full[valid_idx].contiguous()
            centroid_sel = v_contact.mean(dim=-2)
            v_base = torch.stack(
                [centroid_sel[..., 0], centroid_sel[..., 1], centroid_sel[..., 2] + self.standing_height],
                dim=-1,
            )
            v_yaw = yaws[valid_idx].contiguous()
            # Seed IK from the matched template's own joint configuration
            # so the solver starts in the correct basin of attraction —
            # crucial for avoiding crossed-leg local minima on twisted
            # poses where default-stance seeds have to route legs through
            # each other.
            jq_rev_seed = self._fk_joint_q_rev[tpl_idx[valid_idx]].contiguous()

            n_ik = v_contact.shape[0]
            _prepare_ik_batched(v_contact, v_base, v_yaw, jq_rev_seed, buffer)
            buffer._geom_valid[:n_ik] = True
            buffer.num_written = n_ik
            buffer.num_geometry_valid = n_ik

        return SamplerOutput(
            num_written=n_ik,
            reject_stats=reject,
            is_contact=is_contact_ik,
            diagnostics=diagnostics,
        )
