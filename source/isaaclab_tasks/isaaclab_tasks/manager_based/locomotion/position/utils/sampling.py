# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Support polygon sampling strategy for the retarget pipeline.

Per-foot reachability envelopes are derived once at init via random joint
FK sampling; at runtime each foot draws from its own annulus-and-sector
of the terrain, so the ``nc``-contact polygon sits within the robot's
actual workspace. Each accepted polygon becomes one IK problem; the
matcher (see :meth:`TerrainFirstSampler._match_candidates` /
:meth:`TemplateMatchedSampler._match_candidates`) emits an explicit
per-placement slot permutation that reorders the query polygon into
matched-template order before IK.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import torch
import warp as wp

from ..mdp.retarget.buffer import RetargetBuffer
from ..mdp.retarget.cfg import (
    PatchSamplingCfg,
    SamplerSizingCfg,
    TemplateMatchedSamplerCfg,
    TerrainFirstSamplerCfg,
)
from ..mdp.retarget.sampler_base import SamplerBase, SamplerOutput, SamplerSizing, compute_sampler_sizing
from ..terrains.utils.grid_downsample import grid_bucket_downsample
from ..terrains.utils.patch_sampling_cfg import CircleFootprintCfg, MorphologicalPatchSamplingCfg
from ..terrains.utils.patch_sampling_morph import MORPH_TIMINGS
from .kinematic import NewtonKinematics
from .shape_canonical import canonicalize_shape, yaw_from_foot_xy


@wp.kernel
def _per_foot_reachability_sample(
    centers_xy: wp.array(dtype=wp.vec2),
    yaws: wp.array(dtype=wp.float32),
    contact_xy: wp.array(dtype=wp.vec2),
    r_min_sq_ccw: wp.array(dtype=wp.float32),
    r_max_sq_ccw: wp.array(dtype=wp.float32),
    theta_lo_ccw: wp.array(dtype=wp.float32),
    sector_width_ccw: wp.array(dtype=wp.float32),
    foot_ccw_order: wp.array(dtype=wp.int32),
    n_pts: int,
    nc: int,
    seed: int,
    sel_idx: wp.array2d(dtype=wp.int64),
    slot_found: wp.array2d(dtype=wp.int32),
):
    """Reservoir-sample one contact point per (center, foot) from its annulus-sector.

    One thread per ``(k, ccw_sector)``. Streams through all contact
    points doing O(1) per-point work and maintains a size-1 uniform
    reservoir over points lying in that foot's (annulus, sector)
    envelope. Replaces the torch path's ``[K, n_pts]`` distance/angle
    tensor materialization -- on production K*n_pts is ~40e9 float
    cells per chunk, so even 500MB-chunked torch allocations bottleneck
    on HBM bandwidth. A streaming kernel fuses the whole scan into
    registers and a single ``contact_xy`` L2-shared read stream.

    Outputs per ``(candidate, foot)``:

    * ``sel_idx`` — reservoir pick into ``contact_xy`` (or ``0`` if the
      sector was empty; callers consult ``slot_found`` to distinguish).
    * ``slot_found`` — ``1`` iff at least one in-envelope point was
      seen, else ``0``. Enables soft polygon builders that accept
      partial assemblies when the caller sets ``min_contacts < nc``.
    """
    tid = wp.tid()
    k = tid // nc
    j = tid - k * nc

    c = centers_xy[k]
    yaw = yaws[k]
    r_min_sq = r_min_sq_ccw[j]
    r_max_sq = r_max_sq_ccw[j]
    theta_lo_j = theta_lo_ccw[j]
    sw_j = sector_width_ccw[j]

    state = wp.rand_init(seed, tid)
    selected = int(0)
    count = int(0)
    two_pi = float(6.2831853071795864)

    for p in range(n_pts):
        pt = contact_xy[p]
        dx = pt[0] - c[0]
        dy = pt[1] - c[1]
        dist_sq = dx * dx + dy * dy
        if dist_sq >= r_min_sq and dist_sq <= r_max_sq:
            angle = wp.atan2(dy, dx)
            rel = angle - yaw - theta_lo_j
            rel = rel - wp.floor(rel / two_pi) * two_pi
            if rel < sw_j:
                count = count + 1
                if wp.randf(state) * float(count) < 1.0:
                    selected = p

    foot_idx = foot_ccw_order[j]
    sel_idx[k, foot_idx] = wp.int64(selected)
    if count > 0:
        slot_found[k, foot_idx] = 1


def _prepare_ik_batched(
    v_contact: torch.Tensor,
    v_base: torch.Tensor,
    v_yaw: torch.Tensor,
    default_joint_q: torch.Tensor,
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
    degeneracy guard in :func:`~.shape_canonical.canonicalize_shape` so
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
        default_joint_q: Robot's default-stance joint coordinates [m or rad,
            depending on joint type] (used to seed revolute joints),
            shape ``[jc]``.
        buffer: Retarget buffer to scatter the prepared IK problem into.
    """
    n_ik, nc, _ = v_contact.shape
    jc = default_joint_q.shape[0]

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
    buffer.joint_q_init_t[:n_ik, 7:jc] = default_joint_q[7:jc].unsqueeze(0).expand(n_ik, -1)


class TerrainFirstSampler(SamplerBase):
    """Terrain-first contact sampling via per-foot reachability envelopes.

    Each foot's ``(r_min, r_max, theta_lo, theta_hi)`` envelope is
    measured at init by running batched FK on random joint configs
    (within joint limits) and taking percentile bounds on the resulting
    per-foot radial/angular distribution. At sample time, for each
    random center + random base yaw, every foot draws one contact
    point from its own annulus-sector intersection; sectors are
    clipped to inter-foot midpoints so they tile without overlap.

    Each accepted polygon is one IK problem with identity slot
    assignment. Subclasses that maintain a template library can
    override :meth:`_match_candidates` to return a non-trivial slot
    permutation (see :class:`TemplateMatchedSampler`).

    Args:
        cfg: Sampling configuration.
        kin: Newton kinematics model (provides default stance + joint limits).
        foot_body_ids: Newton body indices for the feet.
    """

    def __init__(self, cfg: TerrainFirstSamplerCfg, kin: NewtonKinematics, foot_body_ids: list[int]):
        super().__init__(cfg, kin, foot_body_ids)

        # ``foot_ground_offset`` is the only remaining URDF-default-base-derived
        # scalar; it's the foot-body-to-sole z offset and cannot be recovered
        # from FK without foot collision geometry. Assumes the URDF default
        # stance places soles at z = 0 (standard practice).
        geom = kin.foot_geometry(foot_body_ids)
        self.foot_ground_offset = geom["foot_ground_offset"]
        self.default_joint_q = kin.default_joint_q
        # Pre-stage default stance on GPU so prepare_ik doesn't re-copy
        # every sampler call (small tensor but the sync is not free).
        self._default_joint_q_t = torch.from_numpy(kin.default_joint_q).float().to(kin.device)

        # Per-foot nominal angle, CCW ordering, reach envelope, shape
        # library, and standing height are all derived from the FK
        # distribution inside :meth:`_compute_foot_reachability`. The
        # derivation is invariant to the URDF default base pose (translation
        # + yaw): base held fixed during FK sampling cancels out when every
        # quantity is measured in the polygon-centroid frame.
        self._compute_foot_reachability()

    def _compute_foot_reachability(self, seed: int = 0) -> None:
        """Derive per-foot reachability via random-joint FK.

        Samples ``cfg.fk_num_samples`` random revolute joint configurations
        (base held at default pose), runs batched FK, and derives two
        artifacts:

        1. **2D (annulus, sector) proposal bounds** per foot -- p5/p95
           radial/angular quantiles, clipped to CCW inter-foot midpoints
           so sectors tile the disk without overlap. Drives
           :func:`_per_foot_reachability_sample` at runtime.
        2. **FK polygon shape samples** -- the per-foot canonical-frame
           shape of each FK-produced polygon, stored in
           :attr:`_fk_shape_samples` as a tensor of shape
           ``[n_retained, nc, 3]``. At query time, a candidate polygon is
           canonicalised the same way and accepted iff its maximum
           per-foot distance to some FK sample is below
           :attr:`_fk_shape_tol`. This is joint-feasibility NN in shape
           space: a match *witnesses* a joint configuration that produces
           a compatible polygon (necessary AND sufficient up to the
           tolerance), replacing the per-foot-marginal voxel union of
           the prior implementation.

        Stored attributes (all tensors on ``kin.device``):

        * ``_foot_r_min``, ``_foot_r_max``: radial annulus [m], shape ``[nc]``.
        * ``_foot_theta_lo``, ``_foot_theta_hi``: absolute angle range [rad]
          in base frame, already clipped to non-overlapping sectors.
        * ``_fk_shape_samples``: per-foot canonical polygon shapes [m],
          shape ``[n_retained, nc, 3]``.
        * ``_fk_shape_tol``: NN acceptance radius [m] in canonical shape
          space (per-foot L2, taken as the worst foot).

        Also populates :attr:`init_info` with a one-line summary so the
        pipeline ``rejection_summary`` can report how many FK samples
        the reachability envelope was built from and how long that took.

        Args:
            seed: RNG seed for reproducibility.
        """
        kin = self.kin
        device = kin.device
        nc = len(self.foot_body_ids)
        n_samples = int(self.cfg.fk_num_samples)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        jl_lo = wp.to_torch(kin.model.joint_limit_lower)  # type: ignore[arg-type]
        jl_hi = wp.to_torch(kin.model.joint_limit_upper)  # type: ignore[arg-type]
        rev_lo, rev_hi = jl_lo[6:], jl_hi[6:]
        n_rev = rev_lo.shape[0]

        # Uniform random joint sampling across the absolute joint limits.
        # An earlier 3-band scheme (1/3 uniform + 2/3 Gaussian around
        # URDF default) was dropped after an A/B showed identical
        # canonical-shape distributions: clustering in joint space around
        # URDF default doesn't translate into density at any particular
        # canonical shape (the canonicalization averages base-frame bias out).
        gen = torch.Generator(device=device).manual_seed(seed)
        jq = torch.from_numpy(kin.default_joint_q).float().to(device)
        jq = jq.unsqueeze(0).expand(n_samples, -1).contiguous()
        rand_u = torch.rand(n_samples, n_rev, device=device, generator=gen)
        jq[:, 7 : 7 + n_rev] = rand_u * (rev_hi - rev_lo) + rev_lo

        body_q_wp, _ = kin.eval_fk_batched(wp.from_torch(jq))
        body_q_t = wp.to_torch(body_q_wp).view(n_samples, -1, 7)  # type: ignore[arg-type]
        foot_ids_t = torch.tensor(self.foot_body_ids, device=device, dtype=torch.long)
        foot_xy = body_q_t[:, foot_ids_t, :2]  # [n_samples, nc, 2]
        # Measure each foot's (r, theta) relative to the per-sample polygon
        # centroid, not the base. This makes every derived quantity invariant
        # to the URDF default base pose (translation: cancels; yaw: rotates
        # all angles by a constant, which drops out of the CCW ordering).
        centroid_xy = foot_xy.mean(dim=1, keepdim=True)  # [n_samples, 1, 2]
        rel = foot_xy - centroid_xy  # [n_samples, nc, 2]
        r = rel.norm(dim=-1)
        theta = torch.atan2(rel[..., 1], rel[..., 0])

        # Per-foot nominal angle from the FK distribution itself: circular
        # mean of ``theta`` across samples. Robust to ±π wraparound and
        # independent of URDF-declared ``foot_offsets``. Each foot's random-
        # joint-induced xy distribution is unimodal around its hip-outward
        # direction, so the circular mean recovers that direction.
        nominal = torch.atan2(torch.sin(theta).mean(dim=0), torch.cos(theta).mean(dim=0))  # [nc]
        self._nominal_angle_t = nominal.float().contiguous()
        nominal_np = nominal.detach().cpu().numpy().astype(np.float64)

        # Feet sorted CCW by their FK-derived nominal angle. The relative
        # ordering is invariant to any constant rotation of all nominals
        # (i.e., base yaw at FK time).
        self._foot_ccw_order = np.argsort(nominal_np).tolist()

        # Wrap theta relative to nominal so quantile doesn't straddle +/-pi.
        delta = torch.remainder(theta - nominal + np.pi, 2 * np.pi) - np.pi

        r_lo = torch.quantile(r, 0.05, dim=0)
        r_hi = torch.quantile(r, 0.95, dim=0)
        theta_lo_emp = nominal + torch.quantile(delta, 0.05, dim=0)
        theta_hi_emp = nominal + torch.quantile(delta, 0.95, dim=0)

        # CCW inter-foot midpoints: sector for CCW index j spans
        # [mid(j-1,j), mid(j,j+1)]. ``nominal_ccw`` is monotonic by
        # construction; ``ext`` wraps it with one copy on either side.
        order = np.asarray(self._foot_ccw_order, dtype=np.int64)
        nominal_ccw = nominal_np[order]
        ext = np.concatenate([nominal_ccw[-1:] - 2 * np.pi, nominal_ccw, nominal_ccw[:1] + 2 * np.pi])
        mid_lo_ccw = (ext[:-2] + ext[1:-1]) * 0.5
        mid_hi_ccw = (ext[1:-1] + ext[2:]) * 0.5

        clip_lo_np = np.empty(nc, dtype=np.float32)
        clip_hi_np = np.empty(nc, dtype=np.float32)
        clip_lo_np[order] = mid_lo_ccw
        clip_hi_np[order] = mid_hi_ccw
        clip_lo = torch.from_numpy(clip_lo_np).to(device)
        clip_hi = torch.from_numpy(clip_hi_np).to(device)

        self._foot_r_min = torch.clamp(r_lo, min=float(self.cfg.patch.min_center_dist))
        self._foot_r_max = r_hi
        self._foot_theta_lo = torch.maximum(theta_lo_emp, clip_lo)
        self._foot_theta_hi = torch.minimum(theta_hi_emp, clip_hi)

        # Standing-height IK seed prior from the FK distribution. Using
        # p95 of ``base_z - min_foot_z`` across random joint configs gives
        # the robot's typical fully-extended stance height; this is
        # invariant to the URDF default base z because ``base_z`` is held
        # fixed during FK sampling and every foot position shifts rigidly
        # with the base (so the difference ``base_z - foot_z`` depends
        # only on joint angles, not base pose).
        foot_z_min = body_q_t[:, foot_ids_t, 2].amin(dim=-1)  # [n_samples]
        base_z = body_q_t[:, 0, 2]  # [n_samples]
        self.standing_height = float(torch.quantile(base_z - foot_z_min, 0.95).item())

        # FK polygon shape samples. Canonicalize each FK-produced polygon
        # the same way query time will (centroid-center + yaw + plane-fit
        # pitch/roll + per-foot de-nominal); NN in this space is a pure
        # shape match. Retain a uniform subsample of size
        # ``cfg.fk_num_retained`` as the empirical support of
        # ``p_robot(polygon_shape)``; query-time NN against this set
        # witnesses a joint configuration that realizes the accepted
        # polygon up to the :attr:`cfg.fk_shape_tol` tolerance.
        foot_xyz = body_q_t[:, foot_ids_t, :3]  # [n_samples, nc, 3]
        canon_all = canonicalize_shape(foot_xyz, self._nominal_angle_t)
        n_retained = int(self.cfg.fk_num_retained)
        stride = max(1, n_samples // n_retained)
        self._fk_shape_samples = canon_all[::stride][:n_retained].contiguous()
        # Cache stride-matched WORLD-frame foot positions so symmetry-
        # augmented template libraries (:class:`TemplateMatchedSampler`) can
        # apply slot permutations pre-canonicalization and re-canonicalize.
        self._fk_foot_xyz_world = foot_xyz[::stride][:n_retained].contiguous()
        self._fk_shape_tol = float(self.cfg.fk_shape_tol)
        # Template-intrinsic on-plane flag: canonical |z| within on_plane_tol
        # → slot is on the stance plane for this FK pose. Used by the
        # template-projection path (cfg.use_template_projection).
        self._fk_is_on_plane = (self._fk_shape_samples[..., 2].abs() < float(self.cfg.on_plane_tol)).contiguous()
        # Filter templates to stance-like (at least ``template_min_on_plane``
        # feet coplanar). FK joint-uniform sampling spans the reachable
        # foot workspace, but most joint configs produce legs sticking out
        # in directions unrelated to stance. Only the coplanar subset makes
        # sense to stamp onto a terrain's morph-patch cloud.
        min_on_plane = int(self.cfg.template_min_on_plane)
        if min_on_plane > 0:
            keep_mask = self._fk_is_on_plane.sum(dim=-1) >= min_on_plane
            self._fk_shape_samples = self._fk_shape_samples[keep_mask].contiguous()
            self._fk_foot_xyz_world = self._fk_foot_xyz_world[keep_mask].contiguous()
            self._fk_is_on_plane = self._fk_is_on_plane[keep_mask].contiguous()

        # CCW-reordered envelope buffers keyed by sector index -- fed into
        # :func:`_per_foot_reachability_sample` each sampler call. Built
        # once here to avoid rebuilding the reorder on every pipeline tick.
        order_np = np.asarray(self._foot_ccw_order, dtype=np.int64)
        r_min_ccw = self._foot_r_min.detach().cpu().numpy()[order_np].astype(np.float32)
        r_max_ccw = self._foot_r_max.detach().cpu().numpy()[order_np].astype(np.float32)
        theta_lo_ccw_np = self._foot_theta_lo.detach().cpu().numpy()[order_np].astype(np.float32)
        sw_ccw_np = (self._foot_theta_hi - self._foot_theta_lo).detach().cpu().numpy()[order_np].astype(np.float32)
        self._r_min_sq_ccw_wp = wp.array(r_min_ccw * r_min_ccw, dtype=wp.float32, device=device)
        self._r_max_sq_ccw_wp = wp.array(r_max_ccw * r_max_ccw, dtype=wp.float32, device=device)
        self._theta_lo_ccw_wp = wp.array(theta_lo_ccw_np, dtype=wp.float32, device=device)
        self._sector_width_ccw_wp = wp.array(sw_ccw_np, dtype=wp.float32, device=device)
        self._foot_ccw_order_wp = wp.array(
            np.asarray(self._foot_ccw_order, dtype=np.int32), dtype=wp.int32, device=device
        )

        # Body-frame sector midpoint per foot -- drives the air-slot
        # target fallback when the soft polygon builder accepts a
        # candidate with fewer than ``nc`` found slots. Using the
        # sector midpoint (rather than a raw FK mean) keeps the
        # fallback consistent with the reachability envelope's
        # acceptance region: an air target at the sector's "middle"
        # is always within the foot's reach envelope regardless of
        # base yaw, so IK has a feasible pose to aim for.
        r_mid = (self._foot_r_min + self._foot_r_max) * 0.5  # [nc]
        theta_mid = (self._foot_theta_lo + self._foot_theta_hi) * 0.5  # [nc]
        self._sector_mid_body = torch.stack(
            [r_mid * torch.cos(theta_mid), r_mid * torch.sin(theta_mid)],
            dim=-1,
        ).contiguous()  # [nc, 2]

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.init_info = (
            f"reachability: {n_samples} FK samples in {dt:.3f}s"
            f" (retained {self._fk_shape_samples.shape[0]} shape samples, tol={self._fk_shape_tol * 100:.1f}cm)"
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

    def _match_candidates(self, query_shape: torch.Tensor) -> tuple[torch.Tensor, dict[str, object], torch.Tensor]:
        """Accept/reject query polygons against the FK-reachable shape support.

        Default implementation: each foot must have *some* FK sample
        within tolerance in canonical shape space — independent per-foot
        NN, then worst-foot max. Polygon passes iff the worst foot's
        distance to its own nearest FK sample is under
        :attr:`_fk_shape_tol`. Returns identity slot assignment because
        this matcher has no template library to pick a permutation from.

        Subclasses can override this to swap in a
        nearest-template-with-index match that populates per-placement
        slot assignment (see :class:`TemplateMatchedSampler`).

        Args:
            query_shape: Canonicalised polygon shapes, ``[K, nc, 3]``.

        Returns:
            Tuple ``(accept_mask, diagnostics, matched_perm)``:

            * ``accept_mask`` — ``bool[K]``, ``True`` where the polygon
              matches some FK shape within tolerance.
            * ``diagnostics`` — per-call metrics for the offline
              sampler-metrics harness.
            * ``matched_perm`` — ``int64[K, nc]`` identity slot
              assignment (the matcher has no template library to pick a
              non-trivial permutation from).
        """
        K = query_shape.shape[0]
        nc = query_shape.shape[1]
        device = query_shape.device
        fk_samples = self._fk_shape_samples
        # Chunk over K so peak working-memory is
        # ``K_CHUNK * n_samples * nc * 3 * 4B`` rather than the full
        # ``K * N`` product (K~500, N~25k blows past HBM if materialised
        # in one shot).
        K_CHUNK = 32
        max_foot_nn = torch.empty(K, device=device)
        for k0 in range(0, K, K_CHUNK):
            k1 = min(k0 + K_CHUNK, K)
            diff = query_shape[k0:k1].unsqueeze(1) - fk_samples.unsqueeze(0)
            foot_dist = diff.norm(dim=-1)  # [chunk, N, nc]
            # Per-foot nearest-neighbour distance, then worst foot.
            # A polygon passes iff every foot has *some* FK sample
            # within tolerance (need not be the same sample).
            max_foot_nn[k0:k1] = foot_dist.amin(dim=1).amax(dim=-1)
        accept = max_foot_nn < self._fk_shape_tol
        diagnostics: dict[str, object] = {
            "nn_distance_all": max_foot_nn.detach().clone(),
            "nn_distance_accepted": max_foot_nn[accept].detach().clone(),
        }
        identity_perm = torch.arange(nc, dtype=torch.long, device=device).expand(K, nc).contiguous()
        return accept, diagnostics, identity_perm

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
        if self.cfg.use_template_projection:
            return self._project_call(wp_mesh, origin, buffer, n_desired)
        self.sub_timings.clear()
        patch: PatchSamplingCfg = self.cfg.patch
        nc = buffer.num_contacts
        max_n = buffer.max_candidates

        # Size every stage from ``n_desired`` via the yield-rate cascade.
        # A 200-patch floor keeps tiny unit-test meshes from degenerating to
        # zero candidates; no upper cap -- the buffer is already sized from
        # the same sizing, so downstream allocations fit by construction.
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
            device = fp.device
            origin_t = torch.tensor(origin, dtype=torch.float, device=device)
            fp[:, :3] += origin_t
            contact_pts = fp[:, :3]
            n_pts = contact_pts.shape[0]
        for _sub_name, _sub_dt in MORPH_TIMINGS.items():
            key = f"morph.{_sub_name}"
            self.sub_timings[key] = self.sub_timings.get(key, 0.0) + _sub_dt

        # One IK problem per accepted polygon.
        K = min(sizing.max_neighborhoods, max_n)
        target_n = min(n_desired * sizing.oversample_candidates, max_n)

        torch.manual_seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(42)

        with self._time("neighbors"):
            # Per-foot reachability sampling. For each random center +
            # random base yaw, every foot draws one contact from its own
            # (annulus-sector) intersection where the bounds come from
            # :meth:`_compute_foot_reachability`. The matcher (see
            # :meth:`_match_candidates`) returns an explicit per-placement
            # slot permutation that reorders the query polygon before IK.
            #
            # Sample centers with replacement from the morph-patch pool so
            # ``K`` is driven by the sizing cascade rather than the
            # terrain's patch count -- on rough terrains ``n_pts`` can be
            # small, and reusing a center with a different yaw still gives
            # a distinct polygon.
            centers = contact_pts[torch.randint(0, n_pts, (K,), device=device)].contiguous()
            two_pi = 2.0 * np.pi
            yaws = (torch.rand(K, device=device) * two_pi).contiguous()

            centers_xy_t = centers[:, :2].contiguous()
            contact_xy_t = contact_pts[:, :2].contiguous()

            # ``sel_idx[k, foot_idx]`` = contact-point index for that foot.
            # ``slot_found[k, foot_idx]`` = ``1`` iff the reservoir hit
            # at least one in-envelope patch for that slot. A candidate
            # is out-of-reach when *no* slot was found; the soft polygon
            # path downgrades this to a per-slot air classification.
            sel_idx = torch.empty(K, nc, dtype=torch.long, device=device)
            slot_found_int = torch.zeros(K, nc, dtype=torch.int32, device=device)

            wp.launch(
                _per_foot_reachability_sample,
                dim=K * nc,
                inputs=[
                    wp.from_torch(centers_xy_t, dtype=wp.vec2),
                    wp.from_torch(yaws, dtype=wp.float32),
                    wp.from_torch(contact_xy_t, dtype=wp.vec2),
                    self._r_min_sq_ccw_wp,
                    self._r_max_sq_ccw_wp,
                    self._theta_lo_ccw_wp,
                    self._sector_width_ccw_wp,
                    self._foot_ccw_order_wp,
                    n_pts,
                    nc,
                    42,
                ],
                outputs=[
                    wp.from_torch(sel_idx, dtype=wp.int64),
                    wp.from_torch(slot_found_int, dtype=wp.int32),
                ],
                device=self.kin.device,
            )
            slot_found = slot_found_int.to(torch.bool)  # [K, nc]
            n_found = slot_found.sum(dim=-1)  # [K]
            # ``cfg.min_contacts < 0`` (the default) disables the soft
            # polygon path: every slot must find a patch. Positive values
            # are clamped to ``[1, nc]`` so we never accept a fully-air
            # polygon.
            raw = int(self.cfg.min_contacts)
            min_contacts = nc if (raw < 0 or raw > nc) else max(1, raw)
            out_of_reach = n_found < min_contacts  # [K]

            pts = contact_pts[sel_idx]  # [K, nc, 3], indexed by foot_idx

        with self._time("polygon_build"):
            # Per-slot patch target: reservoir pick + per-foot ground
            # offset. Valid only where ``slot_found`` is ``True``; the
            # classifier ignores entries where it's ``False`` and swaps
            # in the template-projected fallback computed below.
            patch_pos = pts.clone()
            patch_pos[..., 2] += self.foot_ground_offset  # [K, nc, 3]

            # Air-slot fallback target: body-frame sector midpoint rotated
            # by yaw into world, then placed at the candidate's terrain
            # center z. Only used for slots marked air (patch not found in
            # sector); IK treats these as soft targets, so the exact z is
            # not critical as long as it sits at the local terrain height.
            cos_y = torch.cos(yaws).unsqueeze(-1)
            sin_y = torch.sin(yaws).unsqueeze(-1)
            off_x = self._sector_mid_body[:, 0]  # [nc]
            off_y = self._sector_mid_body[:, 1]
            air_x = centers_xy_t[:, 0:1] + cos_y * off_x - sin_y * off_y  # [K, nc]
            air_y = centers_xy_t[:, 1:2] + sin_y * off_x + cos_y * off_y  # [K, nc]
            air_z = centers[:, 2:3].expand(-1, nc)  # [K, nc]
            template_targets_world = torch.stack([air_x, air_y, air_z], dim=-1)

            # Binary per-slot contact/air decision keyed on terrain
            # reachability: slot is contact iff its reach sector yielded
            # a patch. Criteria consume ``is_contact_full`` to ignore
            # air slots.
            is_contact_full = slot_found
            contact_ik = torch.where(is_contact_full.unsqueeze(-1), patch_pos, template_targets_world)

            # Shape-space feasibility. Canonicalization folds translation,
            # yaw, plane-fit pitch+roll, and per-foot hip azimuth out of
            # the polygon so NN in this space is a pure shape match.
            query_shape = canonicalize_shape(contact_ik, self._nominal_angle_t)
            shape_ok, match_diag, matched_perm = self._match_candidates(query_shape)

            valid = ~out_of_reach & shape_ok
            all_valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
            n_all_valid = all_valid_idx.shape[0]
            n_valid = min(n_all_valid, target_n, max_n)

        with self._time("sampler_fps"):
            if n_all_valid > n_valid:
                # FPS keyed on polygon centroid -- the natural geometric
                # reference now that base_target is post-filter.
                centroid_xyz = contact_ik[all_valid_idx].mean(dim=-2)
                local_idx = grid_bucket_downsample(centroid_xyz, n_valid)
                valid_idx = all_valid_idx[local_idx]
            else:
                valid_idx = all_valid_idx

        reject = {
            "out_of_reach": int(out_of_reach.sum()),
            "shape_infeasible": int((~out_of_reach & ~shape_ok).sum()),
        }
        diagnostics: dict[str, object] = dict(match_diag)

        if n_valid == 0:
            buffer.num_written = 0
            buffer.num_geometry_valid = 0
            return SamplerOutput(num_written=0, reject_stats=reject, diagnostics=diagnostics)

        with self._time("prepare_ik"):
            v_contact_sel = contact_ik[valid_idx]  # [n_valid, nc, 3]
            is_contact_sel = is_contact_full[valid_idx]  # [n_valid, nc]
            # Base target is a soft IK prior (pipeline weights it 0.05),
            # not a feasibility gate -- derive it from the polygon
            # centroid with a ``standing_height`` lift. IK is free to
            # move the base from here; this just seeds it geometrically.
            centroid_sel = v_contact_sel.mean(dim=-2)  # [n_valid, 3]
            v_base_sel = torch.stack(
                [centroid_sel[..., 0], centroid_sel[..., 1], centroid_sel[..., 2] + self.standing_height],
                dim=-1,
            )

            # Gather per-placement slot permutation from the matcher.
            # ``matched_perm[t, f]`` tells us "foot ``f`` should receive the
            # query polygon's slot-``matched_perm[t, f]`` point", so we
            # reorder the query polygon before IK sees it. The subclass
            # template-matched sampler emits a non-trivial permutation; the
            # default FK-sample matcher emits identity.
            tpl_for_valid = matched_perm[valid_idx]  # [n_valid, nc]
            gather_idx = tpl_for_valid.unsqueeze(-1).expand(-1, -1, 3)
            v_contact = torch.gather(v_contact_sel, dim=1, index=gather_idx).contiguous()
            is_contact_ik = torch.gather(is_contact_sel, dim=1, index=tpl_for_valid).contiguous()
            slot_assignment = tpl_for_valid.to(torch.int32).contiguous()
            v_base = v_base_sel
            v_yaw = yaw_from_foot_xy(v_contact, self._nominal_angle_t, ref_xy=None).contiguous()

            n_ik = v_contact.shape[0]
            _prepare_ik_batched(v_contact, v_base, v_yaw, self._default_joint_q_t, buffer)
            buffer._geom_valid[:n_ik] = True

            buffer.num_written = n_ik
            buffer.num_geometry_valid = n_ik

        return SamplerOutput(
            num_written=n_ik,
            reject_stats=reject,
            slot_assignment=slot_assignment,
            is_contact=is_contact_ik,
            diagnostics=diagnostics,
        )

    def _project_call(
        self,
        wp_mesh: wp.Mesh,
        origin: np.ndarray,
        buffer: RetargetBuffer,
        n_desired: int,
    ) -> SamplerOutput:
        """Template-projection query path.

        Active when :attr:`TerrainFirstSamplerCfg.use_template_projection`
        is ``True``. Samples one FK template per candidate, un-canonicalizes
        it at a random ``(center, yaw)``, and classifies each foot as
        contact iff the template slot is on-plane AND a morphological
        patch is within :attr:`foot_contact_radius` of its projected xy.
        No reach envelope, no shape NN match, no cyclic expansion --
        contact count emerges from template + terrain geometry.
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

        K = min(sizing.max_neighborhoods, max_n)
        target_n = min(n_desired * sizing.oversample_candidates, max_n)

        torch.manual_seed(42)
        if torch.device(dev_str).type == "cuda":
            torch.cuda.manual_seed_all(42)

        with self._time("project"):
            # Per-candidate (center, yaw, template_idx).
            centers = patch_pts[torch.randint(0, n_pts, (K,), device=dev_str)]  # [K, 3]
            yaws = torch.rand(K, device=dev_str) * (2.0 * np.pi)
            n_tpl = self._fk_shape_samples.shape[0]
            tpl_idx = torch.randint(0, n_tpl, (K,), device=dev_str)
            tpl_canon = self._fk_shape_samples[tpl_idx]  # [K, nc, 3]

            # Un-canonicalize: undo per-foot de-nominal, then apply candidate yaw,
            # then translate by candidate center. Canonical z is already polygon-
            # centroid-relative; sum with candidate terrain z for world z.
            cos_n = torch.cos(self._nominal_angle_t)  # [nc]
            sin_n = torch.sin(self._nominal_angle_t)
            cx = tpl_canon[..., 0]  # [K, nc]
            cy = tpl_canon[..., 1]
            cz = tpl_canon[..., 2]
            mid_x = cos_n * cx - sin_n * cy  # [K, nc]
            mid_y = sin_n * cx + cos_n * cy
            cos_y = torch.cos(yaws).unsqueeze(-1)  # [K, 1]
            sin_y = torch.sin(yaws).unsqueeze(-1)
            world_x = cos_y * mid_x - sin_y * mid_y + centers[:, 0:1]  # [K, nc]
            world_y = sin_y * mid_x + cos_y * mid_y + centers[:, 1:2]
            world_z = cz + centers[:, 2:3]
            projected_world = torch.stack([world_x, world_y, world_z], dim=-1)  # [K, nc, 3]

            # Optimal one-patch-per-foot assignment (brute-force bipartite).
            #
            # Contact is purely terrain-determined: if a morph patch is within
            # ``terrain_presence_radius`` of a foot's projected xy, that foot
            # contacts the patch. The template's per-slot ``is_on_plane`` flag
            # is NOT used — a foot at a different world z (one step higher on
            # stairs) is still a valid contact. Template provides the xy
            # *layout*; terrain decides z and contact/air per foot.
            #
            # Each patch can be claimed by at most one foot per candidate.
            # Greedy "closest-foot-wins" isn't optimal: if foot 0's nearest is
            # 1 cm away from patch A and foot 1's nearest is 2 cm from A, a
            # greedy assignment gives A to foot 0 even when foot 1 has no good
            # 2nd choice — leaving foot 1 as air when swapping would keep
            # both as contact. We solve the proper linear-sum-assignment: for
            # each of ``nc^nc`` combinations of "each foot picks one of its
            # nc-nearest patches", score total distance (contact cost = dist;
            # air cost = radius) with a distinctness penalty, and pick the
            # combo with min total cost. For nc=4 this is 256 combinations
            # per candidate; fully vectorized over K.
            projected_xy = projected_world[..., :2].reshape(-1, 2)  # [K*nc, 2]
            patch_xy = patch_pts[:, :2].contiguous()
            dists = torch.cdist(projected_xy, patch_xy).view(K, nc, -1)  # [K, nc, N_p]
            radius = float(self.cfg.terrain_presence_radius)

            # Each foot's nc-nearest patches (the only ones worth considering).
            topk_dists, topk_idx = dists.topk(nc, dim=-1, largest=False)  # [K, nc, nc]

            # Enumerate nc^nc combinations: ``combo[c, i]`` = rank (0..nc-1)
            # that foot ``i`` picks under combination ``c``.
            rank_axes = [torch.arange(nc, device=device)] * nc
            combo = torch.cartesian_prod(*rank_axes).view(-1, nc)  # [nc^nc, nc]
            C = combo.shape[0]
            gather_rank = combo.view(1, C, nc, 1).expand(K, C, nc, 1)
            chosen_idx = torch.gather(topk_idx.unsqueeze(1).expand(-1, C, -1, -1), 3, gather_rank).squeeze(-1)
            chosen_dist = torch.gather(topk_dists.unsqueeze(1).expand(-1, C, -1, -1), 3, gather_rank).squeeze(-1)

            # Score: per-foot cost = dist if contact, else radius (air penalty).
            # Sum over feet; min-total combination wins.
            contact_mask_c = chosen_dist < radius  # [K, C, nc]
            cost_per_foot = torch.where(contact_mask_c, chosen_dist, torch.full_like(chosen_dist, radius))
            total_cost = cost_per_foot.sum(dim=-1)  # [K, C]

            # Distinctness penalty: contact feet must pick distinct patches.
            # Air feet don't conflict (they don't actually land on a patch) —
            # encode them with unique negative sentinels per slot so the
            # pairwise-equal check only sees real contact conflicts.
            neg_sentinel = -torch.arange(1, nc + 1, device=device).view(1, 1, nc)
            eff_idx = torch.where(contact_mask_c, chosen_idx, neg_sentinel)
            sorted_eff = eff_idx.sort(dim=-1).values
            has_dup = (sorted_eff[..., 1:] == sorted_eff[..., :-1]).any(dim=-1)  # [K, C]
            total_cost = torch.where(has_dup, torch.full_like(total_cost, float("inf")), total_cost)

            best_c = total_cost.argmin(dim=-1)  # [K]
            row_idx = torch.arange(K, device=device)
            nearest_idx = chosen_idx[row_idx, best_c]  # [K, nc]
            nearest_dist = chosen_dist[row_idx, best_c]
            is_contact_full = nearest_dist < radius  # [K, nc]
            patch_gather = patch_pts[nearest_idx.view(-1)].view(K, nc, 3)
            patch_gather[..., 2] = patch_gather[..., 2] + self.foot_ground_offset
            contact_ik = torch.where(is_contact_full.unsqueeze(-1), patch_gather, projected_world)

            n_found = is_contact_full.sum(dim=-1)
            raw = int(self.cfg.min_contacts)
            min_contacts = nc if (raw < 0 or raw > nc) else max(1, raw)
            out_of_reach = n_found < min_contacts

            valid = ~out_of_reach
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

        reject = {"out_of_reach": int(out_of_reach.sum())}
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
            # Identity slot assignment: template positions are already slot-ordered.
            slot_assignment = (
                torch.arange(nc, dtype=torch.int32, device=device).expand(v_contact.shape[0], nc).contiguous()
            )
            centroid_sel = v_contact.mean(dim=-2)
            v_base = torch.stack(
                [centroid_sel[..., 0], centroid_sel[..., 1], centroid_sel[..., 2] + self.standing_height],
                dim=-1,
            )
            v_yaw = yaws[valid_idx].contiguous()

            n_ik = v_contact.shape[0]
            _prepare_ik_batched(v_contact, v_base, v_yaw, self._default_joint_q_t, buffer)
            buffer._geom_valid[:n_ik] = True
            buffer.num_written = n_ik
            buffer.num_geometry_valid = n_ik

        return SamplerOutput(
            num_written=n_ik,
            reject_stats=reject,
            slot_assignment=slot_assignment,
            is_contact=is_contact_ik,
            diagnostics=diagnostics,
        )


class TemplateMatchedSampler(TerrainFirstSampler):
    """Hybrid sampler: terrain-first polygon + NN-to-templates match.

    Inherits the terrain-first polygon-assembly machinery from
    :class:`TerrainFirstSampler` (per-foot reach-envelope sampling,
    morphological-patch pool, plane-fit IK seeding) and replaces the
    pass/fail NN against the dense 25k-sample FK shape distribution
    with a NN match against a smaller FPS-thinned template library
    whose entries carry per-slot permutations. Build-time symmetry
    augmentation provides permutation coverage for symmetric robots:
    each matched template id directly determines the slot assignment
    for IK.

    Two stored tensors drive the match:

    * :attr:`_templates` — ``[N_tpl_aug, nc, 3]`` canonicalised template
      shapes.
    * :attr:`_template_perms` — ``[N_tpl_aug, nc]`` ``int64`` gather
      indices. For template ``t``, ``_template_perms[t, f] = s`` means
      foot ``f`` receives the query polygon's slot-``s`` point when
      that template is matched. Identity row for every unpermuted
      template; non-identity rows for symmetry-augmented copies.
    """

    def __init__(
        self,
        cfg: TemplateMatchedSamplerCfg,
        kin: NewtonKinematics,
        foot_body_ids: list[int],
    ):
        super().__init__(cfg, kin, foot_body_ids)
        self._build_templates()

    def _build_templates(self) -> None:
        """Build the FPS-thinned + symmetry-augmented template library.

        Starts from the stride-matched FK world foot positions cached
        by :meth:`_compute_foot_reachability` (as
        :attr:`_fk_foot_xyz_world`). Applies FPS thinning over
        canonical-shape flattened representations for spatial diversity,
        then for each retained sample, produces one template per
        configured symmetry permutation (identity + :attr:`cfg.symmetry_permutations`).
        Each permuted template is constructed by applying the permutation
        to the FK *world* foot positions and re-canonicalising, and it
        stores the permutation as its slot-to-query gather index.

        Stored attributes (all on ``kin.device``):

        * ``_templates`` — canonicalised shapes, ``float32 [N_aug, nc, 3]``.
        * ``_template_perms`` — slot-to-query gather indices,
          ``int64 [N_aug, nc]``.

        Validates :attr:`cfg.symmetry_permutations` entries are valid
        permutations of ``[0, nc-1]``; raises :class:`ValueError` otherwise.
        """
        cfg: TemplateMatchedSamplerCfg = self.cfg  # type: ignore[assignment]
        device = self.kin.device
        nc = len(self.foot_body_ids)

        if self._fk_foot_xyz_world is None:
            raise RuntimeError("parent did not cache _fk_foot_xyz_world; template build cannot proceed")

        # FPS-thin via grid-bucket downsample on flattened canonical
        # shapes. Bucket FPS in the canonical space promotes diversity
        # in the matching geometry rather than in world pose.
        base_canon = self._fk_shape_samples  # [N_fk, nc, 3]
        base_world = self._fk_foot_xyz_world  # [N_fk, nc, 3]
        n_fk = base_canon.shape[0]
        n_tpl = int(min(cfg.n_templates, n_fk))
        if n_tpl < n_fk:
            flat = base_canon.reshape(n_fk, nc * 3)
            keep_idx = grid_bucket_downsample(flat, n_tpl)
            base_canon = base_canon[keep_idx].contiguous()
            base_world = base_world[keep_idx].contiguous()

        # Validate and collect symmetry permutations (always prepend identity).
        perms: list[list[int]] = [list(range(nc))]
        for p in cfg.symmetry_permutations:
            if len(p) != nc or sorted(p) != list(range(nc)):
                raise ValueError(f"symmetry_permutations entry {p} is not a permutation of [0, {nc - 1}]")
            perms.append(list(p))

        # Apply each permutation to the WORLD foot positions, re-canonicalise,
        # and record the permutation with the resulting shape. Concatenation
        # gives ``[N_aug, nc, 3]`` with ``N_aug = len(perms) * n_tpl``.
        shapes_per_perm: list[torch.Tensor] = []
        perm_idx_per_perm: list[torch.Tensor] = []
        for perm in perms:
            perm_t = torch.tensor(perm, dtype=torch.long, device=device)
            rotated_world = base_world[:, perm_t, :]  # [n_tpl, nc, 3]
            rotated_canon = canonicalize_shape(rotated_world, self._nominal_angle_t).contiguous()
            shapes_per_perm.append(rotated_canon)
            perm_idx_per_perm.append(perm_t.unsqueeze(0).expand(rotated_canon.shape[0], nc).contiguous())

        self._templates = torch.cat(shapes_per_perm, dim=0).contiguous()
        self._template_perms = torch.cat(perm_idx_per_perm, dim=0).contiguous()

        self.init_info = (
            self.init_info or ""
        ) + f"; template lib: {self._templates.shape[0]} ({n_tpl} FPS-thinned × {len(perms)} perms)"

    def _match_candidates(
        self, query_shape: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, object], torch.Tensor | None]:
        """NN-match each query against the augmented template library.

        Two quantities are computed jointly from the ``[K, N_aug, nc]``
        foot-distance tensor to avoid a second scan:

        * **Acceptance gate** — per-foot *independent* NN: for each foot
          slot, the nearest template distance regardless of which
          template the other feet matched. The worst foot's distance
          governs acceptance. Matches :class:`TerrainFirstSampler`'s
          semantics so acceptance yield is comparable (the tolerance
          may need bumping when the template library is small, since
          per-slot neighbours are sparser than in a 25k-sample FK
          distribution).
        * **Slot-assignment lookup** — per-template *joint* worst-foot
          distance; argmin over templates gives the single template
          that explains *all* feet together. That template's
          permutation is gathered as the foot-to-query slot index.
          With ``|G| = 1`` every template has identity permutation, so
          the lookup is a no-op; with symmetry augmentation, the joint
          match selects the right rotation.

        Args:
            query_shape: Canonicalised polygon shapes, ``[K, nc, 3]``.

        Returns:
            Tuple ``(accept_mask, diagnostics, matched_perm)``:

            * ``accept_mask`` — ``bool[K]``.
            * ``diagnostics`` — ``nn_distance_{all,accepted}`` tensors
              (per-foot-indep worst-foot distance) and the matched
              template id per placement (from the joint match).
            * ``matched_perm`` — ``int64[K, nc]`` foot-to-query gather.
        """
        cfg: TemplateMatchedSamplerCfg = self.cfg  # type: ignore[assignment]
        K = query_shape.shape[0]
        device = query_shape.device
        templates = self._templates  # [N_aug, nc, 3]

        # Chunked over K; per-chunk peak memory is
        # ``K_CHUNK * N_aug * nc * 3 * 4B``. With N_aug ~= 2k this is
        # small enough that a larger K_CHUNK is safe vs. the parent's
        # 25k-sample library, keeping wall-time close to the parent.
        K_CHUNK = 64
        per_foot_worst = torch.empty(K, device=device)
        matched_tpl_id = torch.empty(K, dtype=torch.long, device=device)
        for k0 in range(0, K, K_CHUNK):
            k1 = min(k0 + K_CHUNK, K)
            diff = query_shape[k0:k1].unsqueeze(1) - templates.unsqueeze(0)
            foot_dist = diff.norm(dim=-1)  # [chunk, N_aug, nc]
            # Acceptance: per-foot-independent NN across templates, worst foot.
            per_foot_worst[k0:k1] = foot_dist.amin(dim=1).amax(dim=-1)
            # Slot-assignment lookup: joint worst-foot, best template.
            tpl_score = foot_dist.amax(dim=-1)  # [chunk, N_aug]
            matched_tpl_id[k0:k1] = tpl_score.argmin(dim=1)

        accept = per_foot_worst < cfg.template_shape_tol
        # Gather per-placement slot permutation. For rejected polygons
        # the permutation is meaningless but still harmless (caller
        # masks on ``accept`` before using it).
        matched_perm = self._template_perms[matched_tpl_id].contiguous()  # [K, nc]

        diagnostics: dict[str, object] = {
            "nn_distance_all": per_foot_worst.detach().clone(),
            "nn_distance_accepted": per_foot_worst[accept].detach().clone(),
            "matched_template_id": matched_tpl_id.detach().clone(),
        }
        return accept, diagnostics, matched_perm
