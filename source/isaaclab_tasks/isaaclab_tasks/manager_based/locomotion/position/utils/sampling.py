# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Support polygon sampling strategy for the retarget pipeline.

Per-foot reachability envelopes are derived once at init via random joint
FK sampling; at runtime each foot draws from its own annulus-and-sector
of the terrain, so the ``nc``-contact polygon sits within the robot's
actual workspace. Each selected polygon is then cyclically enumerated
over all ``nc`` CCW rotations of the foot-to-point assignment; the
pipeline's :attr:`group_size` collapse keeps the lowest-cost rotation
per polygon so IK can pick the orientation that best matches the
terrain patch even if the reachability-pinned assignment was
sub-optimal.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import torch
import warp as wp

from ..mdp.retarget.buffer import RetargetBuffer
from ..mdp.retarget.cfg import PatchSamplingCfg, SamplerSizingCfg, SupportPolygonSamplerCfg
from ..mdp.retarget.pipeline import SamplerBase, SamplerSizing, compute_sampler_sizing
from ..terrains.utils.grid_downsample import grid_bucket_downsample
from ..terrains.utils.patch_sampling_cfg import CircleFootprintCfg, MorphologicalPatchSamplingCfg
from ..terrains.utils.patch_sampling_morph import MORPH_TIMINGS
from .kinematic import NewtonKinematics


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
    out_of_reach: wp.array(dtype=wp.int32),
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
    if count == 0:
        wp.atomic_max(out_of_reach, k, 1)


@wp.kernel
def _prepare_ik_kernel(
    v_contact: wp.array2d(dtype=wp.vec3),
    v_base: wp.array(dtype=wp.vec3),
    v_yaw: wp.array(dtype=wp.float32),
    default_joint_q: wp.array(dtype=wp.float32),
    contact_targets: wp.array(dtype=wp.vec3),
    joint_q_init: wp.array2d(dtype=wp.float32),
    base_target_pos: wp.array(dtype=wp.vec3),
    base_target_rot: wp.array(dtype=wp.vec4),
    nc: int,
    jc: int,
):
    """Plane-fit + Euler-to-quat + buffer scatter, fused per-problem.

    Replaces the ~60-torch-kernel tensor chain in ``prepare_ik`` with a
    single launch. Plane fit uses closed-form 2x2 normal equations on
    mean-centered foot positions; base target orientation uses half
    roll (milder roll on inclined terrain).
    """
    i = wp.tid()

    # Centroid of foot positions.
    cx = float(0.0)
    cy = float(0.0)
    cz = float(0.0)
    for f in range(nc):
        p = v_contact[i, f]
        cx = cx + p[0]
        cy = cy + p[1]
        cz = cz + p[2]
    inv_nc = 1.0 / float(nc)
    cx = cx * inv_nc
    cy = cy * inv_nc
    cz = cz * inv_nc

    # Covariance elements on mean-centered (x, y, z) + z range.
    xx = float(0.0)
    yy = float(0.0)
    xym = float(0.0)
    xz = float(0.0)
    yz = float(0.0)
    z_min = float(1.0e9)
    z_max = float(-1.0e9)
    for f in range(nc):
        p = v_contact[i, f]
        dx = p[0] - cx
        dy = p[1] - cy
        dz = p[2] - cz
        xx = xx + dx * dx
        yy = yy + dy * dy
        xym = xym + dx * dy
        xz = xz + dx * dz
        yz = yz + dy * dz
        if dz > z_max:
            z_max = dz
        if dz < z_min:
            z_min = dz

    det = xx * yy - xym * xym
    if det < 1.0e-12:
        det = 1.0e-12
    a = (yy * xz - xym * yz) / det
    b = (xx * yz - xym * xz) / det

    yaw = v_yaw[i]

    # Flat-terrain short-circuit so pure-flat samples get zero roll/pitch.
    # Sign convention: ``a = dz/dx``, ``b = dz/dy`` are WORLD-frame plane
    # slopes. In ZYX composition ``R = Rz(yaw)*Ry(pitch)*Rx(roll)``, pitch
    # and roll are body-frame Euler angles, so we must rotate the plane
    # slopes into base frame by ``-yaw`` before taking atan. Otherwise the
    # base tilt is correct only at ``yaw = 0`` -- the cyclic-rotation
    # expansion draws variants with ``yaw`` offset by ``2*pi/nc``, and a
    # yaw-unaware kernel tilts the base into the wrong direction, sending
    # feet outside the reach envelope. Derivation: require body-z (=
    # column-2 of ``R``) align with plane normal ``(-a, -b, 1)``; solving
    # the small-angle linearisation gives ``pitch = -a_b`` and ``roll =
    # b_b`` where ``(a_b, b_b) = R_yaw^T @ (a, b)``.
    z_range_val = z_max - z_min
    roll = float(0.0)
    pitch = float(0.0)
    if z_range_val >= 1.0e-4:
        cy = wp.cos(yaw)
        sy = wp.sin(yaw)
        a_b = a * cy + b * sy
        b_b = -a * sy + b * cy
        roll = wp.atan(b_b)
        pitch = -wp.atan(a_b)

    # Joint-q-init base quaternion from (roll, pitch, yaw).
    cy2 = wp.cos(yaw * 0.5)
    sy2 = wp.sin(yaw * 0.5)
    cp2 = wp.cos(pitch * 0.5)
    sp2 = wp.sin(pitch * 0.5)
    cr2 = wp.cos(roll * 0.5)
    sr2 = wp.sin(roll * 0.5)
    ji_qw = cy2 * cr2 * cp2 + sy2 * sr2 * sp2
    ji_qx = cy2 * sr2 * cp2 - sy2 * cr2 * sp2
    ji_qy = cy2 * cr2 * sp2 + sy2 * sr2 * cp2
    ji_qz = sy2 * cr2 * cp2 - cy2 * sr2 * sp2

    # Base-target quaternion: half roll keeps the base more upright over
    # sloped terrain so IK has slack for the tilted leg placements; full
    # roll pulled the base too close to the joint-limit envelope and
    # spiked foot_err.
    cr3 = wp.cos(roll * 0.25)
    sr3 = wp.sin(roll * 0.25)
    br_qw = cy2 * cr3 * cp2 + sy2 * sr3 * sp2
    br_qx = cy2 * sr3 * cp2 - sy2 * cr3 * sp2
    br_qy = cy2 * cr3 * sp2 + sy2 * sr3 * cp2
    br_qz = sy2 * cr3 * cp2 - cy2 * sr3 * sp2

    base = v_base[i]
    base_target_pos[i] = base
    base_target_rot[i] = wp.vec4(br_qx, br_qy, br_qz, br_qw)

    for f in range(nc):
        contact_targets[i * nc + f] = v_contact[i, f]

    joint_q_init[i, 0] = base[0]
    joint_q_init[i, 1] = base[1]
    joint_q_init[i, 2] = base[2]
    joint_q_init[i, 3] = ji_qx
    joint_q_init[i, 4] = ji_qy
    joint_q_init[i, 5] = ji_qz
    joint_q_init[i, 6] = ji_qw
    for j in range(7, jc):
        joint_q_init[i, j] = default_joint_q[j]


class SupportPolygonSampler(SamplerBase):
    """Terrain contact sampling via per-foot reachability envelopes.

    Each foot's ``(r_min, r_max, theta_lo, theta_hi)`` envelope is
    measured at init by running batched FK on random joint configs
    (within joint limits) and taking percentile bounds on the resulting
    per-foot radial/angular distribution. At sample time, for each
    random center + random base yaw, every foot draws one contact
    point from its own annulus-sector intersection; sectors are
    clipped to inter-foot midpoints so they tile without overlap.

    Per-foot reachability pins a plausible assignment at sample time,
    but the sector bounds are empirical (p5/p95 over random-joint FK)
    and slightly overlap between adjacent feet. To let IK pick the
    best match, each accepted polygon is expanded into ``nc`` CCW-
    rotated assignments and the pipeline's :attr:`group_size` collapse
    keeps the lowest-cost rotation.

    Args:
        cfg: Sampling configuration.
        kin: Newton kinematics model (provides default stance + joint limits).
        foot_body_ids: Newton body indices for the feet.
    """

    def __init__(self, cfg: SupportPolygonSamplerCfg, kin: NewtonKinematics, foot_body_ids: list[int]):
        super().__init__(cfg, kin, foot_body_ids)

        geom = kin.foot_geometry(foot_body_ids)
        self.foot_offsets = geom["foot_offsets"]
        self.foot_ground_offset = geom["foot_ground_offset"]
        self.standing_height = geom["standing_height"]
        self.default_joint_q = kin.default_joint_q
        # Pre-stage default stance on the GPU so prepare_ik doesn't re-copy
        # every sampler call (small tensor but the sync is not free). The
        # wp view is cached alongside it for use by :func:`_prepare_ik_kernel`.
        self._default_joint_q_t = torch.from_numpy(kin.default_joint_q).float().to(kin.device)
        self._default_joint_q_wp = wp.from_torch(self._default_joint_q_t, dtype=wp.float32)

        # Feet sorted CCW by their nominal angle in base frame -- used to
        # map "sector j" (CCW index) to a specific foot body.
        nominal_np = np.arctan2(self.foot_offsets[:, 1], self.foot_offsets[:, 0])
        self._foot_ccw_order = np.argsort(nominal_np).tolist()

        # Cached torch tensors for cyclic-rotation expansion (hot path).
        nc = len(self.foot_body_ids)
        foot_ccw_t = torch.tensor(self._foot_ccw_order, dtype=torch.long, device=kin.device)
        inv_ccw_t = torch.empty(nc, dtype=torch.long, device=kin.device)
        inv_ccw_t[foot_ccw_t] = torch.arange(nc, device=kin.device)
        rr = torch.arange(nc, device=kin.device).view(nc, 1)
        ff = torch.arange(nc, device=kin.device).view(1, nc)
        # ``_cyclic_perm[r, f]`` = the foot whose reachability-sampled point
        # is reassigned to foot ``f`` under rotation ``r``.
        self._cyclic_perm = foot_ccw_t[(inv_ccw_t[ff] + rr) % nc].contiguous()
        self._nominal_angle_t = torch.from_numpy(nominal_np.astype(np.float32)).to(kin.device)

        self._compute_foot_reachability()

    @property
    def group_size(self) -> int:
        return len(self.foot_body_ids)

    def _compute_foot_reachability(self, n_samples: int = 100000, seed: int = 0) -> None:
        """Derive per-foot reachability via random-joint FK.

        Samples ``n_samples`` random revolute joint configurations (base
        held at default pose), runs batched FK, and derives two artifacts:

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
            n_samples: FK batch size.
            seed: RNG seed for reproducibility.
        """
        kin = self.kin
        device = kin.device
        nc = len(self.foot_body_ids)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        jl_lo = wp.to_torch(kin.model.joint_limit_lower)  # type: ignore[arg-type]
        jl_hi = wp.to_torch(kin.model.joint_limit_upper)  # type: ignore[arg-type]
        rev_lo, rev_hi = jl_lo[6:], jl_hi[6:]
        n_rev = rev_lo.shape[0]

        # Mix uniform-random joint sampling with perturbations around the
        # default stance. Uniform sampling alone under-populates the
        # "near-default stance" region of the workspace (rel_z close to
        # ``-standing_height`` requires a specific hip+knee combo), so
        # the voxels where runtime feet land on flat terrain are sparsely
        # covered and the grid lookup over-rejects. Three bands:
        #   - 1/3 uniform:  explore the full workspace.
        #   - 1/3 broad perturb (1/3 joint range): realistic gait poses.
        #   - 1/3 tight perturb (1/8 joint range): dense default cluster.
        # The tight band guarantees the default-stance voxel and its
        # immediate neighbours are firmly occupied, and the broader
        # bands extend the envelope into typical gait territory.
        gen = torch.Generator(device=device).manual_seed(seed)
        jq = torch.from_numpy(kin.default_joint_q).float().to(device)
        default_rev = jq[7 : 7 + n_rev].clone()
        jq = jq.unsqueeze(0).expand(n_samples, -1).contiguous()
        n_third = n_samples // 3
        rand_u = torch.rand(n_third, n_rev, device=device, generator=gen)
        jq[:n_third, 7 : 7 + n_rev] = rand_u * (rev_hi - rev_lo) + rev_lo
        broad_scale = (rev_hi - rev_lo) / 3.0
        rand_b = torch.randn(n_third, n_rev, device=device, generator=gen)
        jq_b = default_rev.unsqueeze(0) + rand_b * broad_scale
        jq[n_third : 2 * n_third, 7 : 7 + n_rev] = torch.clamp(jq_b, rev_lo, rev_hi)
        n_tight = n_samples - 2 * n_third
        tight_scale = (rev_hi - rev_lo) / 8.0
        rand_t = torch.randn(n_tight, n_rev, device=device, generator=gen)
        jq_t = default_rev.unsqueeze(0) + rand_t * tight_scale
        jq[2 * n_third :, 7 : 7 + n_rev] = torch.clamp(jq_t, rev_lo, rev_hi)

        body_q_wp, _ = kin.eval_fk_batched(wp.from_torch(jq))
        body_q_t = wp.to_torch(body_q_wp).view(n_samples, -1, 7)  # type: ignore[arg-type]
        base_xy = body_q_t[:, 0, :2]
        foot_ids_t = torch.tensor(self.foot_body_ids, device=device, dtype=torch.long)
        rel = body_q_t[:, foot_ids_t, :2] - base_xy.unsqueeze(1)  # [n_samples, nc, 2]
        r = rel.norm(dim=-1)
        theta = torch.atan2(rel[..., 1], rel[..., 0])

        nominal_np = np.arctan2(self.foot_offsets[:, 1], self.foot_offsets[:, 0])
        nominal = torch.from_numpy(nominal_np).float().to(device)
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

        # FK polygon shape samples. Canonicalise each FK-produced polygon
        # the same way query time will: centroid-center, derive yaw from
        # foot layout, plane-fit pitch/roll, rotate into body frame, and
        # per-foot de-nominal-yaw. Two polygons with nearby canonical
        # shapes differ by rigid-body motion only, so NN in this space
        # is pure shape-match. Retain a stride-4 subsample (~25k of the
        # 100k FK configs) as the empirical support of
        # ``p_robot(polygon_shape)``; query-time NN against this set
        # witnesses a joint configuration that realises the accepted
        # polygon up to the tolerance below.
        foot_xyz = body_q_t[:, foot_ids_t, :3]  # [n_samples, nc, 3]
        canon_all = self._canonicalize_shape(foot_xyz)
        fk_stride = 4
        self._fk_shape_samples = canon_all[::fk_stride].contiguous()
        # Worst-foot L2 acceptance radius. The prior voxel-grid used a
        # 4 cm per-foot marginal tolerance, but that was checked
        # independently per foot; the joint NN here is the max over
        # feet, which is strictly tighter under the same per-foot
        # budget. 8 cm gives comparable acceptance semantics for
        # near-default polygons (inter-foot radius spread after
        # centroid-referencing is typically 3-6 cm) while still
        # rejecting shapes no FK configuration can realise.
        self._fk_shape_tol = 0.08

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
        ``n_desired`` determines every internal stage size. The buffer
        must hold :attr:`group_size` rotational variants per polygon,
        so ``max_polygons`` is scaled by ``gs`` on top of the base
        cascade.
        """
        sz: SamplerSizingCfg = self.cfg.sizing
        base = compute_sampler_sizing(
            n_desired,
            final_fps_oversample=sz.final_fps_oversample,
            criteria_yield=sz.criteria_yield,
            polygon_fps_oversample=sz.polygon_fps_oversample,
            polygon_assembly_yield=sz.polygon_assembly_yield,
            morph_patch_oversample=sz.morph_patch_oversample,
            patches_per_polygon=len(self.foot_body_ids),
        )
        gs = self.group_size
        return SamplerSizing(
            n_final=base.n_final,
            oversample_candidates=base.oversample_candidates,
            max_neighborhoods=base.max_neighborhoods,
            n_morph_patches=base.n_morph_patches,
            max_polygons=base.max_polygons * gs,
        )

    def _canonicalize_shape(self, feet_xyz: torch.Tensor) -> torch.Tensor:
        """Rigid-body-invariant polygon shape descriptor.

        Transforms foot world positions into a per-foot canonical frame:
        polygon centroid at origin, base yaw from
        :meth:`_yaw_from_foot_xy` (centroid-referenced symmetric best-fit),
        pitch and roll from least-squares plane-fit (half-roll, full-pitch
        -- matches the IK ``base_target_rot`` convention used in
        :func:`_prepare_ik_kernel`), and per-foot ``(x, y)`` rotated by
        ``-nominal_angle[f]`` so the foot lies in its own hip-outward
        frame. Polygons differing by rigid body motion produce the same
        canonical shape, so nearest-neighbour in this space is a pure
        shape-match query.

        Used at build time on FK-sampled polygons to populate
        :attr:`_fk_shape_samples`, and at sample time on terrain-
        proposed polygons to check whether a kinematically feasible
        witness exists.

        Args:
            feet_xyz: Foot positions, shape ``[..., nc, 3]`` [m].

        Returns:
            Per-foot canonical coordinates, shape ``[..., nc, 3]`` [m].
        """
        centroid = feet_xyz.mean(dim=-2, keepdim=True)
        delta = feet_xyz - centroid
        yaw = self._yaw_from_foot_xy(feet_xyz, ref_xy=None)
        dxp, dyp, dzp = delta[..., 0], delta[..., 1], delta[..., 2]
        xx = (dxp * dxp).sum(dim=-1)
        yy = (dyp * dyp).sum(dim=-1)
        xym = (dxp * dyp).sum(dim=-1)
        xzm = (dxp * dzp).sum(dim=-1)
        yzm = (dyp * dzp).sum(dim=-1)
        det = (xx * yy - xym * xym).clamp_min(1.0e-12)
        a = (yy * xzm - xym * yzm) / det
        b = (xx * yzm - xym * xzm) / det
        flat = (dzp.amax(dim=-1) - dzp.amin(dim=-1)) < 1.0e-4
        a = torch.where(flat, torch.zeros_like(a), a)
        b = torch.where(flat, torch.zeros_like(b), b)
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        a_b = a * cos_y + b * sin_y
        b_b = -a * sin_y + b * cos_y
        pitch_t = -torch.atan(a_b)
        roll_t = 0.5 * torch.atan(b_b)
        cos_y2 = cos_y.unsqueeze(-1)
        sin_y2 = sin_y.unsqueeze(-1)
        dx_by = cos_y2 * dxp + sin_y2 * dyp
        dy_by = -sin_y2 * dxp + cos_y2 * dyp
        cos_p = torch.cos(pitch_t).unsqueeze(-1)
        sin_p = torch.sin(pitch_t).unsqueeze(-1)
        cos_r = torch.cos(roll_t).unsqueeze(-1)
        sin_r = torch.sin(roll_t).unsqueeze(-1)
        rel_x = cos_p * dx_by - sin_p * dzp
        v2y = sin_p * dx_by + cos_p * dzp
        rel_y = cos_r * dy_by + sin_r * v2y
        rel_z = -sin_r * dy_by + cos_r * v2y
        cos_n = torch.cos(self._nominal_angle_t)
        sin_n = torch.sin(self._nominal_angle_t)
        canon_x = cos_n * rel_x + sin_n * rel_y
        canon_y = -sin_n * rel_x + cos_n * rel_y
        return torch.stack([canon_x, canon_y, rel_z], dim=-1).contiguous()

    def _yaw_from_foot_xy(self, foot_xyz: torch.Tensor, ref_xy: torch.Tensor | None = None) -> torch.Tensor:
        """Derive a best-fit base yaw from per-foot world positions.

        Each foot ``f`` sits at world direction ``yaw + nominal[f]`` from
        the base origin. Rotating ``(foot_xy - base_xy)`` by
        ``-nominal[f]`` gives a vector at angle ``yaw``; summing across
        feet and taking ``atan2`` yields a robust weighted estimate that
        averages out per-foot reachability-sampling noise.

        Args:
            foot_xyz: Foot positions ``[..., nc, 3]``. Only the ``xy``
                columns are consulted.
            ref_xy: Base reference position ``[..., 2]`` (broadcastable
                to ``foot_xyz[..., :2]`` after unsqueezing the foot dim).
                If ``None``, uses the polygon centroid.

        Returns:
            Best-fit yaw, shape ``foot_xyz.shape[:-2]``.
        """
        nc = len(self.foot_body_ids)
        if ref_xy is None:
            ref_xy = foot_xyz[..., :2].mean(dim=-2)
        v_xy = foot_xyz[..., :2] - ref_xy.unsqueeze(-2)
        cos_n = torch.cos(self._nominal_angle_t).view(*([1] * (v_xy.dim() - 2)), nc)
        sin_n = torch.sin(self._nominal_angle_t).view(*([1] * (v_xy.dim() - 2)), nc)
        rot_vx = cos_n * v_xy[..., 0] + sin_n * v_xy[..., 1]
        rot_vy = -sin_n * v_xy[..., 0] + cos_n * v_xy[..., 1]
        return torch.atan2(rot_vy.sum(dim=-1), rot_vx.sum(dim=-1))

    def _expand_cyclic(
        self, v_contact_sel: torch.Tensor, v_base_sel: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand each polygon into ``gs`` cyclic-rotated foot assignments.

        For rotation ``r`` the point originally sampled for ``foot_ccw[(j+r) % nc]``
        is reassigned to ``foot_ccw[j]`` (shape ``[n_valid, gs, nc, 3]``),
        flattened to ``[n_valid * gs, nc, 3]`` with the ``(k, r)`` pair
        laid out contiguously so the pipeline's group-collapse indexing
        (``view(n_groups, gs)``) picks the lowest-cost rotation per polygon.

        The base target is broadcast from the caller (polygon-centroid
        based), not re-derived from the rotated foot layout. Since the
        centroid is invariant under foot reassignment, every cyclic
        variant of a polygon shares the same base position. Yaw is
        derived from the rotated feet so cyclic variants pose the base
        at different yaw angles (IK's group-collapse picks the best
        rotation post-solve).

        Args:
            v_contact_sel: Selected polygon foot positions (already shifted
                by :attr:`foot_ground_offset`), shape ``[n_valid, nc, 3]``.
            v_base_sel: Per-polygon base target position (polygon centroid
                lifted by :attr:`standing_height`), shape ``[n_valid, 3]``.
            device: Torch device for the returned tensors.

        Returns:
            Tuple ``(v_contact, v_base, v_yaw)``:

            * ``v_contact``: ``[n_valid * gs, nc, 3]`` rotated foot positions.
            * ``v_base``: ``[n_valid * gs, 3]`` base target position.
            * ``v_yaw``: ``[n_valid * gs]`` base target yaw [rad].
        """
        nc = len(self.foot_body_ids)
        gs = nc
        # [n_valid, nc, 3] -> [n_valid, gs, nc, 3] -> [n_valid * gs, nc, 3]
        v_contact = v_contact_sel[:, self._cyclic_perm, :].reshape(-1, nc, 3).contiguous()
        # Base is the same for every cyclic variant of a polygon.
        v_base = v_base_sel.unsqueeze(1).expand(-1, gs, -1).reshape(-1, 3).contiguous()
        # Yaw derivation uses the polygon centroid (``ref_xy=None``) --
        # consistent with the shape filter in :meth:`__call__`, which
        # canonicalises polygons using the centroid-referenced yaw. For
        # cyclic variants the derived yaw differs because the
        # foot-to-hip assignment changes, so variants pose the base at
        # different orientations and the pipeline's group-collapse
        # picks the lowest-cost one post-IK.
        v_yaw = self._yaw_from_foot_xy(v_contact, ref_xy=None).contiguous()
        return v_contact, v_base, v_yaw

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
    ) -> tuple[int, dict[str, int]]:
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

        # Buffer holds ``gs`` cyclic-rotation variants per polygon after the
        # expansion below, so polygon-stage caps divide ``max_n`` by ``gs``
        # and the expansion fits by construction.
        gs = self.group_size
        max_polys = max(1, max_n // gs)
        K = min(sizing.max_neighborhoods, max_polys)
        target_n = min(n_desired * sizing.oversample_candidates, max_polys)

        torch.manual_seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(42)

        with self._time("neighbors"):
            # Per-foot reachability sampling. For each random center +
            # random base yaw, every foot draws one contact from its own
            # (annulus-sector) intersection where the bounds come from
            # :meth:`_compute_foot_reachability`. The sampled assignment
            # is a plausible starting point; each accepted polygon is
            # cyclically enumerated into ``gs`` rotations in
            # :func:`_expand_cyclic`, and the pipeline's group-collapse
            # keeps the lowest-cost rotation per polygon.
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
            sel_idx = torch.empty(K, nc, dtype=torch.long, device=device)
            out_of_reach_int = torch.zeros(K, dtype=torch.int32, device=device)

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
                    wp.from_torch(out_of_reach_int, dtype=wp.int32),
                ],
                device=self.kin.device,
            )
            out_of_reach = out_of_reach_int != 0

            pts = contact_pts[sel_idx]  # [K, nc, 3], indexed by foot_idx

        with self._time("polygon_build"):
            contact_ik = pts.clone()
            contact_ik[:, :, 2] += self.foot_ground_offset

            # Per-foot shape-space NN against the FK empirical
            # workspace. The base target is *not* committed here --
            # it's a soft IK prior derived from the polygon centroid
            # post-filter. Canonicalisation folds translation / yaw /
            # plane-fit pitch+roll / per-foot hip azimuth out of the
            # polygon, so each foot's canonical ``(x, y, z)`` is a
            # rigid-body-invariant descriptor. ``_fk_shape_samples``
            # is the empirical support of the FK-reachable workspace;
            # the per-foot nearest-FK distance is the same
            # "is each foot in the feasible set" test the prior voxel
            # grid performed, but in continuous centroid-referenced
            # space (no voxel quantisation, and the centroid reference
            # avoids the off-centroid over-rejection that came from
            # pinning the query to ``sampled_center``).
            query_shape = self._canonicalize_shape(contact_ik)
            fk_samples = self._fk_shape_samples
            # Chunk over K so peak working-memory is
            # ``K_CHUNK * n_samples * nc * 3 * 4B`` rather than the
            # full ``K * N`` product (K~500, N~25k blows past HBM if
            # materialised in one shot).
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
            shape_ok = max_foot_nn < self._fk_shape_tol

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

        if n_valid == 0:
            buffer.num_written = 0
            buffer.num_geometry_valid = 0
            return 0, reject

        with self._time("prepare_ik"):
            v_contact_sel = contact_ik[valid_idx]  # [n_valid, nc, 3]
            # Base target is a soft IK prior (pipeline weights it 0.05),
            # not a feasibility gate -- derive it from the polygon
            # centroid with a ``standing_height`` lift. IK is free to
            # move the base from here; this just seeds it geometrically.
            centroid_sel = v_contact_sel.mean(dim=-2)  # [n_valid, 3]
            v_base_sel = torch.stack(
                [centroid_sel[..., 0], centroid_sel[..., 1], centroid_sel[..., 2] + self.standing_height],
                dim=-1,
            )

            # Cyclic-nc expansion. For each polygon, emit ``gs`` variants
            # where rotation ``r`` reassigns foot ``foot_ccw[j]`` to the
            # point that was originally sampled for ``foot_ccw[(j+r) % nc]``.
            # Laid out contiguously as ``(k=0,r=0), (k=0,r=1), ...,
            # (k=0,r=gs-1), (k=1,r=0), ...`` so the pipeline's
            # ``group_size`` collapse (``view(n_groups, gs)``) picks the
            # lowest-cost rotation per polygon.
            v_contact, v_base, v_yaw = self._expand_cyclic(v_contact_sel, v_base_sel, device)
            n_ik = v_contact.shape[0]

            wp.launch(
                _prepare_ik_kernel,
                dim=n_ik,
                inputs=[
                    wp.from_torch(v_contact, dtype=wp.vec3),
                    wp.from_torch(v_base, dtype=wp.vec3),
                    wp.from_torch(v_yaw, dtype=wp.float32),
                    self._default_joint_q_wp,
                    buffer.contact_targets,
                    buffer.joint_q_init,
                    buffer.base_target_pos,
                    buffer.base_target_rot,
                    nc,
                    self.kin.model.joint_coord_count,
                ],
                device=self.kin.device,
            )
            buffer._geom_valid[:n_ik] = True

            buffer.num_written = n_ik
            buffer.num_geometry_valid = n_ik

        return n_ik, reject
