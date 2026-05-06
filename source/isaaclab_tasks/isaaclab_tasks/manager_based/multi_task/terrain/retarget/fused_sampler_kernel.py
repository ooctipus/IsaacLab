# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fused single-kernel polygon sampler for the contact-sampling stage.

This module contains the **end-to-end** Warp kernel that, in a single
``wp.launch`` over ``K`` candidate slots, does:

1. Per-thread RNG init (deterministic from ``seed`` + thread id).
2. Sample one ``(center_idx, yaw, tpl_idx)`` triple per slot.
3. Un-canonicalize the chosen FK template into world-frame foot xyz
   under the sampled yaw + center.
4. Query the prebuilt morph-patch ``HashGrid`` for the top-``nc``
   nearest patches per foot, in registers.
5. Iterate every foot-rank permutation, evaluating cost +
   distinctness + force-must-contact + winding + convex-hull validity
   inline.
6. Emit per-slot outputs (``is_contact``, ``contact_ik``, ``n_found``,
   ``no_convex``, ``yaws``, ``tpl_idx``).

Because every per-slot intermediate lives in registers, this fused
kernel **eliminates the chunking infrastructure** the previous
implementation needed: there is no ``[Kc, C, nc, ...]`` family of
tensors to chunk around, no ``per_row_bytes`` budget, no
``empty_cache`` dance with the Warp allocator. The only allocations
the sampler has to do are the K-sized output tensors that the
downstream FPS + IK stages consume.

**Scope**: nc=4 (quadruped) only. The LSA hull check assumes a
4-point polygon and uses an unrolled 4-element sorting network.
Other foot counts raise :class:`NotImplementedError` from the
launcher and the caller falls back to the chunked path in
``contact_sampling.py``.
"""

from __future__ import annotations

import math

import torch
import warp as wp

from .spatial_topk import build_spatial_grid_xy

# Compile-time constants -- kernel inner loops unroll cleanly at nc=4.
NC: wp.constant = wp.constant(4)
N_COMBOS: wp.constant = wp.constant(24)  # 4! permutations -- distinctness penalty makes the rest waste.
TWO_PI: wp.constant = wp.constant(float(2.0 * math.pi))


# ----------------------------------------------------------------- structs


@wp.struct
class _FootTopK:
    """Top-``nc`` patch hits for one foot. ``idx[i]`` and ``dist[i]``
    give the i-th-nearest patch index + distance. Slots not filled
    keep ``idx=-1``, ``dist=+inf``.
    """

    idx: wp.vec4i
    dist: wp.vec4


# ---------------------------------------------------------------- helpers


@wp.func
def _topk4_query(
    grid_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    qx: wp.float32,
    qy: wp.float32,
    radius: wp.float32,
) -> _FootTopK:
    """Query ``grid_id`` for the 4 nearest patches around ``(qx, qy)``.

    Streaming "replace-the-max" top-4 followed by a 4-element bubble
    sort. Returns indices + distances in ascending-distance order;
    empty slots are left at ``(-1, +inf)``.
    """
    out = _FootTopK()
    out.idx = wp.vec4i(-1, -1, -1, -1)
    out.dist = wp.vec4(wp.inf, wp.inf, wp.inf, wp.inf)

    nb_iter = wp.hash_grid_query(grid_id, wp.vec3(qx, qy, 0.0), radius)
    nb = int(0)
    while wp.hash_grid_query_next(nb_iter, nb):
        p = points[nb]
        dx = p[0] - qx
        dy = p[1] - qy
        d = wp.sqrt(dx * dx + dy * dy)
        if d <= radius:
            # Find slot with the current max distance.
            max_d = out.dist[0]
            max_slot = int(0)
            for j in range(1, 4):
                if out.dist[j] > max_d:
                    max_d = out.dist[j]
                    max_slot = j
            if d < max_d:
                out.dist[max_slot] = d
                out.idx[max_slot] = nb

    # Bubble sort the 4 slots ascending so the caller sees neighbours
    # in distance order (matches the previous `.sort(...).values`
    # behaviour the LSA loop assumed).
    for i in range(4):
        for j in range(3):
            if out.dist[j] > out.dist[j + 1]:
                td = out.dist[j]
                out.dist[j] = out.dist[j + 1]
                out.dist[j + 1] = td
                ti = out.idx[j]
                out.idx[j] = out.idx[j + 1]
                out.idx[j + 1] = ti

    return out


@wp.func
def _argsort4(a: wp.float32, b: wp.float32, c: wp.float32, d: wp.float32) -> wp.vec4i:
    """Sorting-network argsort of 4 floats. Returns indices ascending."""
    v0 = a
    v1 = b
    v2 = c
    v3 = d
    i0 = int(0)
    i1 = int(1)
    i2 = int(2)
    i3 = int(3)
    if v0 > v1:
        tv = v0
        v0 = v1
        v1 = tv
        ti = i0
        i0 = i1
        i1 = ti
    if v2 > v3:
        tv = v2
        v2 = v3
        v3 = tv
        ti = i2
        i2 = i3
        i3 = ti
    if v0 > v2:
        tv = v0
        v0 = v2
        v2 = tv
        ti = i0
        i0 = i2
        i2 = ti
    if v1 > v3:
        tv = v1
        v1 = v3
        v3 = tv
        ti = i1
        i1 = i3
        i3 = ti
    if v1 > v2:
        tv = v1
        v1 = v2
        v2 = tv
        ti = i1
        i1 = i2
        i2 = ti
    return wp.vec4i(i0, i1, i2, i3)


@wp.func
def _has_dup4(a: wp.int32, b: wp.int32, c: wp.int32, d: wp.int32) -> int:
    """Return 1 if any two of the four ints are equal, else 0."""
    if a == b:
        return 1
    if a == c:
        return 1
    if a == d:
        return 1
    if b == c:
        return 1
    if b == d:
        return 1
    if c == d:
        return 1
    return 0


# ----------------------------------------------------------------- kernel


@wp.kernel
def _fused_sampler_kernel(
    # ---- scalars / RNG ----
    seed: int,
    K: int,
    n_pts: int,
    n_tpl: int,
    radius: wp.float32,
    effective_radius: wp.float32,
    query_radius: wp.float32,
    outward_pen: wp.float32,
    force_all_snap: int,
    foot_ground_offset: wp.float32,
    # ---- precomputed tables ----
    patch_pts: wp.array(dtype=wp.vec3),  # [N_p] real xyz
    patch_pts_z0: wp.array(dtype=wp.vec3),  # [N_p] same xyz with z=0 (hashgrid points)
    patch_grid_id: wp.uint64,  # built over patch_pts_z0
    fk_shape_samples: wp.array2d(dtype=wp.vec3),  # [n_tpl, NC] canonical foot xyz
    cos_n: wp.vec4,  # per-foot cos(nominal angle)
    sin_n: wp.vec4,  # per-foot sin(nominal angle)
    combo: wp.array2d(dtype=wp.int32),  # [N_COMBOS, NC] permutations
    # ---- outputs (allocated by the launcher) ----
    yaws_out: wp.array(dtype=wp.float32),  # [K]
    tpl_idx_out: wp.array(dtype=wp.int32),  # [K]
    is_contact_out: wp.array2d(dtype=wp.int32),  # [K, NC]
    contact_ik_out: wp.array2d(dtype=wp.vec3),  # [K, NC]
    n_found_out: wp.array(dtype=wp.int32),  # [K]
    no_convex_out: wp.array(dtype=wp.int32),  # [K]
):
    """One thread per candidate slot. Does the full sampler work end-to-end."""
    k = wp.tid()
    if k >= K:
        return

    # ----- 1. Per-thread RNG -----
    state = wp.rand_init(seed, k)
    center_idx = wp.randi(state, 0, n_pts)
    yaw = wp.randf(state) * TWO_PI
    tpl_idx_val = wp.randi(state, 0, n_tpl)
    yaws_out[k] = yaw
    tpl_idx_out[k] = tpl_idx_val

    center = patch_pts[center_idx]
    cos_y = wp.cos(yaw)
    sin_y = wp.sin(yaw)

    # ----- 2. Un-canonicalize template into world-frame foot positions -----
    tc0 = fk_shape_samples[tpl_idx_val, 0]
    tc1 = fk_shape_samples[tpl_idx_val, 1]
    tc2 = fk_shape_samples[tpl_idx_val, 2]
    tc3 = fk_shape_samples[tpl_idx_val, 3]

    # mid = R(nominal_angle) @ tpl_canon (per-foot rotation about z).
    mx0 = cos_n[0] * tc0[0] - sin_n[0] * tc0[1]
    my0 = sin_n[0] * tc0[0] + cos_n[0] * tc0[1]
    mx1 = cos_n[1] * tc1[0] - sin_n[1] * tc1[1]
    my1 = sin_n[1] * tc1[0] + cos_n[1] * tc1[1]
    mx2 = cos_n[2] * tc2[0] - sin_n[2] * tc2[1]
    my2 = sin_n[2] * tc2[0] + cos_n[2] * tc2[1]
    mx3 = cos_n[3] * tc3[0] - sin_n[3] * tc3[1]
    my3 = sin_n[3] * tc3[0] + cos_n[3] * tc3[1]

    # world = R(yaw) @ mid + center (per-candidate yaw rotation + translation).
    wx0 = cos_y * mx0 - sin_y * my0 + center[0]
    wy0 = sin_y * mx0 + cos_y * my0 + center[1]
    wz0 = tc0[2] + center[2]
    wx1 = cos_y * mx1 - sin_y * my1 + center[0]
    wy1 = sin_y * mx1 + cos_y * my1 + center[1]
    wz1 = tc1[2] + center[2]
    wx2 = cos_y * mx2 - sin_y * my2 + center[0]
    wy2 = sin_y * mx2 + cos_y * my2 + center[1]
    wz2 = tc2[2] + center[2]
    wx3 = cos_y * mx3 - sin_y * my3 + center[0]
    wy3 = sin_y * mx3 + cos_y * my3 + center[1]
    wz3 = tc3[2] + center[2]

    # ----- 3. Top-NC hashgrid query per foot -----
    top0 = _topk4_query(patch_grid_id, patch_pts_z0, wx0, wy0, query_radius)
    top1 = _topk4_query(patch_grid_id, patch_pts_z0, wx1, wy1, query_radius)
    top2 = _topk4_query(patch_grid_id, patch_pts_z0, wx2, wy2, query_radius)
    top3 = _topk4_query(patch_grid_id, patch_pts_z0, wx3, wy3, query_radius)

    # Sentinel-safe: empty slots come back as ``idx=-1, dist=+inf``.
    # Clamp idx to 0 so the patch-table reads below are bounds-safe; the
    # ``dist=+inf`` keeps the contact mask correctly false for those slots.
    if top0.idx[0] < 0:
        top0.idx[0] = 0
    if top0.idx[1] < 0:
        top0.idx[1] = 0
    if top0.idx[2] < 0:
        top0.idx[2] = 0
    if top0.idx[3] < 0:
        top0.idx[3] = 0
    if top1.idx[0] < 0:
        top1.idx[0] = 0
    if top1.idx[1] < 0:
        top1.idx[1] = 0
    if top1.idx[2] < 0:
        top1.idx[2] = 0
    if top1.idx[3] < 0:
        top1.idx[3] = 0
    if top2.idx[0] < 0:
        top2.idx[0] = 0
    if top2.idx[1] < 0:
        top2.idx[1] = 0
    if top2.idx[2] < 0:
        top2.idx[2] = 0
    if top2.idx[3] < 0:
        top2.idx[3] = 0
    if top3.idx[0] < 0:
        top3.idx[0] = 0
    if top3.idx[1] < 0:
        top3.idx[1] = 0
    if top3.idx[2] < 0:
        top3.idx[2] = 0
    if top3.idx[3] < 0:
        top3.idx[3] = 0

    # Per-foot template centroid + radii (used by the outward-snap cost).
    tcx = (wx0 + wx1 + wx2 + wx3) * 0.25
    tcy = (wy0 + wy1 + wy2 + wy3) * 0.25
    tpl_r0 = wp.sqrt((wx0 - tcx) * (wx0 - tcx) + (wy0 - tcy) * (wy0 - tcy))
    tpl_r1 = wp.sqrt((wx1 - tcx) * (wx1 - tcx) + (wy1 - tcy) * (wy1 - tcy))
    tpl_r2 = wp.sqrt((wx2 - tcx) * (wx2 - tcx) + (wy2 - tcy) * (wy2 - tcy))
    tpl_r3 = wp.sqrt((wx3 - tcx) * (wx3 - tcx) + (wy3 - tcy) * (wy3 - tcy))

    # Template winding order around tpl_centroid.
    tpl_a0 = wp.atan2(wy0 - tcy, wx0 - tcx)
    tpl_a1 = wp.atan2(wy1 - tcy, wx1 - tcx)
    tpl_a2 = wp.atan2(wy2 - tcy, wx2 - tcx)
    tpl_a3 = wp.atan2(wy3 - tcy, wx3 - tcx)
    tpl_perm = _argsort4(tpl_a0, tpl_a1, tpl_a2, tpl_a3)

    # Per-foot "is there at least one in-radius patch" (force-contact gate).
    has_opt0 = int(0)
    if top0.dist[0] < radius:
        has_opt0 = 1
    has_opt1 = int(0)
    if top1.dist[0] < radius:
        has_opt1 = 1
    has_opt2 = int(0)
    if top2.dist[0] < radius:
        has_opt2 = 1
    has_opt3 = int(0)
    if top3.dist[0] < radius:
        has_opt3 = 1

    # ----- 4. LSA permutation loop -----
    best_cost = float(1.0e30)
    best_c = int(-1)

    for c in range(N_COMBOS):
        r0 = combo[c, 0]
        r1 = combo[c, 1]
        r2 = combo[c, 2]
        r3 = combo[c, 3]

        ci0 = top0.idx[r0]
        ci1 = top1.idx[r1]
        ci2 = top2.idx[r2]
        ci3 = top3.idx[r3]

        cd0 = top0.dist[r0]
        cd1 = top1.dist[r1]
        cd2 = top2.dist[r2]
        cd3 = top3.dist[r3]

        contact0 = int(0)
        if cd0 < effective_radius:
            contact0 = 1
        contact1 = int(0)
        if cd1 < effective_radius:
            contact1 = 1
        contact2 = int(0)
        if cd2 < effective_radius:
            contact2 = 1
        contact3 = int(0)
        if cd3 < effective_radius:
            contact3 = 1

        # Force-must-contact gate.
        if force_all_snap == 0:
            if has_opt0 == 1 and contact0 == 0:
                continue
            if has_opt1 == 1 and contact1 == 0:
                continue
            if has_opt2 == 1 and contact2 == 0:
                continue
            if has_opt3 == 1 and contact3 == 0:
                continue

        # Distinctness: contact feet must pick distinct patches; air feet
        # use a per-foot negative sentinel so they never collide.
        e0 = -1
        if contact0 == 1:
            e0 = ci0
        e1 = -2
        if contact1 == 1:
            e1 = ci1
        e2 = -3
        if contact2 == 1:
            e2 = ci2
        e3 = -4
        if contact3 == 1:
            e3 = ci3
        if _has_dup4(e0, e1, e2, e3) == 1:
            continue

        # Per-foot cost + target xy (for hull check below).
        cost = float(0.0)
        if contact0 == 1:
            px = patch_pts_z0[ci0]
            patch_r = wp.sqrt((px[0] - tcx) * (px[0] - tcx) + (px[1] - tcy) * (px[1] - tcy))
            outward = wp.max(patch_r - tpl_r0, 0.0)
            cost = cost + cd0 + outward_pen * outward
            tx0 = px[0]
            ty0 = px[1]
        else:
            cost = cost + radius
            tx0 = wx0
            ty0 = wy0
        if contact1 == 1:
            px = patch_pts_z0[ci1]
            patch_r = wp.sqrt((px[0] - tcx) * (px[0] - tcx) + (px[1] - tcy) * (px[1] - tcy))
            outward = wp.max(patch_r - tpl_r1, 0.0)
            cost = cost + cd1 + outward_pen * outward
            tx1 = px[0]
            ty1 = px[1]
        else:
            cost = cost + radius
            tx1 = wx1
            ty1 = wy1
        if contact2 == 1:
            px = patch_pts_z0[ci2]
            patch_r = wp.sqrt((px[0] - tcx) * (px[0] - tcx) + (px[1] - tcy) * (px[1] - tcy))
            outward = wp.max(patch_r - tpl_r2, 0.0)
            cost = cost + cd2 + outward_pen * outward
            tx2 = px[0]
            ty2 = px[1]
        else:
            cost = cost + radius
            tx2 = wx2
            ty2 = wy2
        if contact3 == 1:
            px = patch_pts_z0[ci3]
            patch_r = wp.sqrt((px[0] - tcx) * (px[0] - tcx) + (px[1] - tcy) * (px[1] - tcy))
            outward = wp.max(patch_r - tpl_r3, 0.0)
            cost = cost + cd3 + outward_pen * outward
            tx3 = px[0]
            ty3 = px[1]
        else:
            cost = cost + radius
            tx3 = wx3
            ty3 = wy3

        if cost >= best_cost:
            continue

        # Hull validity: target points form a convex polygon whose winding
        # order matches the template's projected polygon.
        ttx = (tx0 + tx1 + tx2 + tx3) * 0.25
        tty = (ty0 + ty1 + ty2 + ty3) * 0.25
        ta0 = wp.atan2(ty0 - tty, tx0 - ttx)
        ta1 = wp.atan2(ty1 - tty, tx1 - ttx)
        ta2 = wp.atan2(ty2 - tty, tx2 - ttx)
        ta3 = wp.atan2(ty3 - tty, tx3 - ttx)
        tgt_perm = _argsort4(ta0, ta1, ta2, ta3)

        if (
            tgt_perm[0] != tpl_perm[0]
            or tgt_perm[1] != tpl_perm[1]
            or tgt_perm[2] != tpl_perm[2]
            or tgt_perm[3] != tpl_perm[3]
        ):
            continue

        # Sort target xy by tgt_perm and check strict convexity.
        sx0 = float(0.0)
        sy0 = float(0.0)
        sx1 = float(0.0)
        sy1 = float(0.0)
        sx2 = float(0.0)
        sy2 = float(0.0)
        sx3 = float(0.0)
        sy3 = float(0.0)
        if tgt_perm[0] == 0:
            sx0 = tx0
            sy0 = ty0
        if tgt_perm[0] == 1:
            sx0 = tx1
            sy0 = ty1
        if tgt_perm[0] == 2:
            sx0 = tx2
            sy0 = ty2
        if tgt_perm[0] == 3:
            sx0 = tx3
            sy0 = ty3
        if tgt_perm[1] == 0:
            sx1 = tx0
            sy1 = ty0
        if tgt_perm[1] == 1:
            sx1 = tx1
            sy1 = ty1
        if tgt_perm[1] == 2:
            sx1 = tx2
            sy1 = ty2
        if tgt_perm[1] == 3:
            sx1 = tx3
            sy1 = ty3
        if tgt_perm[2] == 0:
            sx2 = tx0
            sy2 = ty0
        if tgt_perm[2] == 1:
            sx2 = tx1
            sy2 = ty1
        if tgt_perm[2] == 2:
            sx2 = tx2
            sy2 = ty2
        if tgt_perm[2] == 3:
            sx2 = tx3
            sy2 = ty3
        if tgt_perm[3] == 0:
            sx3 = tx0
            sy3 = ty0
        if tgt_perm[3] == 1:
            sx3 = tx1
            sy3 = ty1
        if tgt_perm[3] == 2:
            sx3 = tx2
            sy3 = ty2
        if tgt_perm[3] == 3:
            sx3 = tx3
            sy3 = ty3

        ex0 = sx1 - sx0
        ey0 = sy1 - sy0
        ex1 = sx2 - sx1
        ey1 = sy2 - sy1
        ex2 = sx3 - sx2
        ey2 = sy3 - sy2
        ex3 = sx0 - sx3
        ey3 = sy0 - sy3
        cr0 = ex0 * ey1 - ey0 * ex1
        cr1 = ex1 * ey2 - ey1 * ex2
        cr2 = ex2 * ey3 - ey2 * ex3
        cr3 = ex3 * ey0 - ey3 * ex0

        all_pos = int(0)
        if cr0 > 0.0 and cr1 > 0.0 and cr2 > 0.0 and cr3 > 0.0:
            all_pos = 1
        all_neg = int(0)
        if cr0 < 0.0 and cr1 < 0.0 and cr2 < 0.0 and cr3 < 0.0:
            all_neg = 1
        if all_pos == 0 and all_neg == 0:
            continue

        best_cost = cost
        best_c = c

    # ----- 5. Emit outputs from best_c -----
    if best_c < 0:
        no_convex_out[k] = 1
        n_found_out[k] = 0
        is_contact_out[k, 0] = 0
        is_contact_out[k, 1] = 0
        is_contact_out[k, 2] = 0
        is_contact_out[k, 3] = 0
        contact_ik_out[k, 0] = wp.vec3(0.0, 0.0, 0.0)
        contact_ik_out[k, 1] = wp.vec3(0.0, 0.0, 0.0)
        contact_ik_out[k, 2] = wp.vec3(0.0, 0.0, 0.0)
        contact_ik_out[k, 3] = wp.vec3(0.0, 0.0, 0.0)
        return

    no_convex_out[k] = 0

    # Per-foot air-floor: each foot's nearest patch z plus the offset.
    af0 = patch_pts[top0.idx[0]][2] + foot_ground_offset
    af1 = patch_pts[top1.idx[0]][2] + foot_ground_offset
    af2 = patch_pts[top2.idx[0]][2] + foot_ground_offset
    af3 = patch_pts[top3.idx[0]][2] + foot_ground_offset

    n_count = int(0)

    # Foot 0
    rb = combo[best_c, 0]
    cib = top0.idx[rb]
    cdb = top0.dist[rb]
    if cdb < effective_radius:
        pp = patch_pts[cib]
        contact_ik_out[k, 0] = wp.vec3(pp[0], pp[1], pp[2] + foot_ground_offset)
        is_contact_out[k, 0] = 1
        n_count += 1
    else:
        z = wp.max(wz0, af0)
        contact_ik_out[k, 0] = wp.vec3(wx0, wy0, z)
        is_contact_out[k, 0] = 0
    # Foot 1
    rb = combo[best_c, 1]
    cib = top1.idx[rb]
    cdb = top1.dist[rb]
    if cdb < effective_radius:
        pp = patch_pts[cib]
        contact_ik_out[k, 1] = wp.vec3(pp[0], pp[1], pp[2] + foot_ground_offset)
        is_contact_out[k, 1] = 1
        n_count += 1
    else:
        z = wp.max(wz1, af1)
        contact_ik_out[k, 1] = wp.vec3(wx1, wy1, z)
        is_contact_out[k, 1] = 0
    # Foot 2
    rb = combo[best_c, 2]
    cib = top2.idx[rb]
    cdb = top2.dist[rb]
    if cdb < effective_radius:
        pp = patch_pts[cib]
        contact_ik_out[k, 2] = wp.vec3(pp[0], pp[1], pp[2] + foot_ground_offset)
        is_contact_out[k, 2] = 1
        n_count += 1
    else:
        z = wp.max(wz2, af2)
        contact_ik_out[k, 2] = wp.vec3(wx2, wy2, z)
        is_contact_out[k, 2] = 0
    # Foot 3
    rb = combo[best_c, 3]
    cib = top3.idx[rb]
    cdb = top3.dist[rb]
    if cdb < effective_radius:
        pp = patch_pts[cib]
        contact_ik_out[k, 3] = wp.vec3(pp[0], pp[1], pp[2] + foot_ground_offset)
        is_contact_out[k, 3] = 1
        n_count += 1
    else:
        z = wp.max(wz3, af3)
        contact_ik_out[k, 3] = wp.vec3(wx3, wy3, z)
        is_contact_out[k, 3] = 0

    n_found_out[k] = n_count


# --------------------------------------------------------------- launcher


def run_fused_sampler(
    *,
    seed: int,
    K: int,
    patch_pts: torch.Tensor,  # [N, 3] float -- real xyz
    fk_shape_samples: torch.Tensor,  # [n_tpl, NC, 3] float
    nominal_angles: torch.Tensor,  # [NC] float -- per-foot rotation about z
    radius: float,
    query_radius: float,
    outward_pen: float,
    force_all_snap: bool,
    foot_ground_offset: float,
) -> dict[str, torch.Tensor]:
    """Single-launch end-to-end sampler.

    Allocates the K-sized output tensors, builds the morph-patch hashgrid,
    and runs the fused kernel once over ``K`` candidate slots. Returns a
    dict with all the K-sized outputs the contact_sampling caller needs.

    Restrictions:
        - ``nc=4`` only (quadruped). Other foot counts raise.
        - ``patch_pts`` must be on CUDA -- the hashgrid and kernel are
          GPU-only.
    """
    nc = int(fk_shape_samples.shape[1])
    if nc != 4:
        raise NotImplementedError(
            f"Fused sampler kernel supports only nc=4; got nc={nc}. "
            "Use the chunked Python LSA path for other foot counts."
        )

    device = patch_pts.device
    if not patch_pts.is_cuda:
        raise ValueError("Fused sampler requires CUDA tensors.")

    # ---- Build the patch hashgrid (z=0 projection so radius is xy-only). ----
    patch_xy = patch_pts[:, :2].contiguous()
    patch_grid = build_spatial_grid_xy(patch_xy, radius=query_radius)

    # ---- Build permutation table (24 combos at nc=4). ----
    import itertools

    combo = torch.tensor(list(itertools.permutations(range(nc))), device=device, dtype=torch.int32)

    # ---- Per-foot nominal-angle constants packed into wp.vec4. ----
    cos_n_t = torch.cos(nominal_angles).to(torch.float32).cpu().tolist()
    sin_n_t = torch.sin(nominal_angles).to(torch.float32).cpu().tolist()
    cos_n_v = wp.vec4(cos_n_t[0], cos_n_t[1], cos_n_t[2], cos_n_t[3])
    sin_n_v = wp.vec4(sin_n_t[0], sin_n_t[1], sin_n_t[2], sin_n_t[3])

    # ---- Output tensors (the only K-sized memory we need). ----
    yaws = torch.empty((K,), dtype=torch.float32, device=device)
    tpl_idx = torch.empty((K,), dtype=torch.int32, device=device)
    is_contact = torch.empty((K, nc), dtype=torch.int32, device=device)
    contact_ik = torch.empty((K, nc, 3), dtype=torch.float32, device=device)
    n_found = torch.empty((K,), dtype=torch.int32, device=device)
    no_convex = torch.empty((K,), dtype=torch.int32, device=device)

    wp.launch(
        _fused_sampler_kernel,
        dim=K,
        inputs=[
            int(seed),
            int(K),
            int(patch_pts.shape[0]),
            int(fk_shape_samples.shape[0]),
            float(radius),
            float(query_radius if force_all_snap else radius),  # effective_radius
            float(query_radius),
            float(outward_pen),
            int(1 if force_all_snap else 0),
            float(foot_ground_offset),
            wp.from_torch(patch_pts.contiguous(), dtype=wp.vec3),
            patch_grid._pts_wp,  # the z=0 grid points (shared with grid build)
            patch_grid.grid.id,
            wp.from_torch(fk_shape_samples.contiguous(), dtype=wp.vec3),
            cos_n_v,
            sin_n_v,
            wp.from_torch(combo, dtype=wp.int32),
        ],
        outputs=[
            wp.from_torch(yaws, dtype=wp.float32),
            wp.from_torch(tpl_idx, dtype=wp.int32),
            wp.from_torch(is_contact, dtype=wp.int32),
            wp.from_torch(contact_ik, dtype=wp.vec3),
            wp.from_torch(n_found, dtype=wp.int32),
            wp.from_torch(no_convex, dtype=wp.int32),
        ],
        device=str(device),
    )

    return {
        "yaws": yaws,
        "tpl_idx": tpl_idx.to(torch.int64),
        "is_contact_full": is_contact.bool(),
        "contact_ik": contact_ik,
        "n_found": n_found.to(torch.int64),
        "no_convex": no_convex.bool(),
    }


__all__ = ["run_fused_sampler"]
