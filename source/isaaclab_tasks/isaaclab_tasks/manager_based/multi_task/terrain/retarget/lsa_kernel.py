# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fused linear-sum-assignment + hull-validity kernel for the contact sampler.

The LSA stage of :class:`SamplerContactPolygon` evaluates every
foot-rank permutation against several penalties (cost, distinctness,
force-contact, winding match, convex-hull validity) and picks the
lowest-cost survivor per candidate. The reference Python path
materialises ``[Kc, C, nc]`` and ``[Kc, C, nc, 2]`` tensors for every
intermediate, which dominates GPU memory at dense pool settings
(``pool_spacing=0.05`` gives K~60M and forces aggressive chunking).

This module provides a single Warp kernel that fuses every per-combo
step into per-thread registers, so the only allocations the LSA stage
needs are the ``[Kc, ...]`` outputs themselves. Memory drops from
``O(Kc × C × nc)`` to ``O(Kc)``, which lets the chunk loop process
tens of millions of candidates per pass instead of thousands.

**Scope**: nc=4 (quadruped) only. For other foot counts the launcher
raises :class:`NotImplementedError` and the caller should fall back
to the Python LSA path. The hull-validity check assumes a 4-point
polygon and uses an unrolled 4-element sorting network -- generalising
to arbitrary nc would require either runtime sort (at the cost of
register pressure) or a per-nc kernel.

The kernel is tagged ``module="unique"`` so warp recompiles it once
per process instead of forking with each launch.
"""

from __future__ import annotations

import math

import torch
import warp as wp


# Compile-time constants (unrolled inside the kernel).
NC: wp.constant = wp.constant(4)
N_COMBOS: wp.constant = wp.constant(24)  # 4! permutations


@wp.func
def _argsort4(a: float, b: float, c: float, d: float) -> wp.vec4i:
    """Sort 4 floats ascending and return the *indices* of the sort.

    Implements an optimal-comparator sorting network (5 swaps for n=4)
    on (value, index) pairs. Returns the permutation as a ``vec4i``
    where ``out[k]`` is the original index of the k-th smallest value.
    """
    # values
    v0 = a
    v1 = b
    v2 = c
    v3 = d
    # original indices
    i0 = int(0)
    i1 = int(1)
    i2 = int(2)
    i3 = int(3)

    # compare-swap pairs (0,1) and (2,3)
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
    # compare-swap (0,2) and (1,3)
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
    # compare-swap (1,2)
    if v1 > v2:
        tv = v1
        v1 = v2
        v2 = tv
        ti = i1
        i1 = i2
        i2 = ti

    return wp.vec4i(i0, i1, i2, i3)


@wp.func
def _has_dup4(a: int, b: int, c: int, d: int) -> int:
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


@wp.kernel
def _lsa_kernel(
    # ---- per-K inputs ----
    topk_i: wp.array2d(dtype=wp.int32),       # (Kc * NC, NC)
    topk_d: wp.array2d(dtype=wp.float32),     # (Kc * NC, NC)
    proj_pos: wp.array2d(dtype=wp.vec3),      # (Kc, NC) world-frame foot xyz
    # ---- per-call constants ----
    patch_xy: wp.array(dtype=wp.vec2),        # (N_patches,)
    patch_pts: wp.array(dtype=wp.vec3),       # (N_patches,)
    combo: wp.array2d(dtype=wp.int32),        # (N_COMBOS, NC)
    # ---- scalars ----
    radius: float,
    effective_radius: float,
    outward_pen: float,
    force_all_snap: int,                      # 0 or 1
    foot_ground_offset: float,
    # ---- outputs ----
    is_contact_c: wp.array2d(dtype=wp.int32), # (Kc, NC)
    contact_ik_c: wp.array2d(dtype=wp.vec3),  # (Kc, NC)
    n_found: wp.array(dtype=wp.int32),         # (Kc,)
    no_convex: wp.array(dtype=wp.int32),       # (Kc,)
):
    """One thread per Kc row. Iterates the ``N_COMBOS`` permutations,
    keeps the lowest-cost valid one, then emits per-foot outputs.
    """
    k = wp.tid()

    # ----- Per-foot constants for this Kc row -----
    p0 = proj_pos[k, 0]
    p1 = proj_pos[k, 1]
    p2 = proj_pos[k, 2]
    p3 = proj_pos[k, 3]

    # Template centroid (mean of projected feet).
    cx = (p0[0] + p1[0] + p2[0] + p3[0]) * 0.25
    cy = (p0[1] + p1[1] + p2[1] + p3[1]) * 0.25
    tpl_cen = wp.vec2(cx, cy)

    # Template winding order (atan2 around tpl_cen, then argsort).
    tpl_a0 = wp.atan2(p0[1] - cy, p0[0] - cx)
    tpl_a1 = wp.atan2(p1[1] - cy, p1[0] - cx)
    tpl_a2 = wp.atan2(p2[1] - cy, p2[0] - cx)
    tpl_a3 = wp.atan2(p3[1] - cy, p3[0] - cx)
    tpl_perm = _argsort4(tpl_a0, tpl_a1, tpl_a2, tpl_a3)

    # Per-foot template radius (used by the outward cost).
    tpl_r0 = wp.length(wp.vec2(p0[0] - cx, p0[1] - cy))
    tpl_r1 = wp.length(wp.vec2(p1[0] - cx, p1[1] - cy))
    tpl_r2 = wp.length(wp.vec2(p2[0] - cx, p2[1] - cy))
    tpl_r3 = wp.length(wp.vec2(p3[0] - cx, p3[1] - cy))

    # Per-foot "is there at least one in-radius patch" (force-contact gate).
    # ``topk[k*NC+foot, 0]`` is the closest patch for that foot.
    has_opt0 = int(0)
    if topk_d[k * NC + 0, 0] < radius:
        has_opt0 = 1
    has_opt1 = int(0)
    if topk_d[k * NC + 1, 0] < radius:
        has_opt1 = 1
    has_opt2 = int(0)
    if topk_d[k * NC + 2, 0] < radius:
        has_opt2 = 1
    has_opt3 = int(0)
    if topk_d[k * NC + 3, 0] < radius:
        has_opt3 = 1

    # ----- Combo loop: track best valid -----
    best_cost = float(1.0e30)  # +inf-ish
    best_c = int(-1)

    for c in range(N_COMBOS):
        r0 = combo[c, 0]
        r1 = combo[c, 1]
        r2 = combo[c, 2]
        r3 = combo[c, 3]

        # Per-foot picked patch.
        ci0 = topk_i[k * NC + 0, r0]
        ci1 = topk_i[k * NC + 1, r1]
        ci2 = topk_i[k * NC + 2, r2]
        ci3 = topk_i[k * NC + 3, r3]

        cd0 = topk_d[k * NC + 0, r0]
        cd1 = topk_d[k * NC + 1, r1]
        cd2 = topk_d[k * NC + 2, r2]
        cd3 = topk_d[k * NC + 3, r3]

        # Contact mask under the active radius (loose for force_all_snap).
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

        # ---- Force-must-contact: if a foot has a real contact option but
        # this combo's pick is air, the combo is invalid. Skipped under
        # ``force_all_snap`` (every foot is forced to contact then).
        if force_all_snap == 0:
            if has_opt0 == 1 and contact0 == 0:
                continue
            if has_opt1 == 1 and contact1 == 0:
                continue
            if has_opt2 == 1 and contact2 == 0:
                continue
            if has_opt3 == 1 and contact3 == 0:
                continue

        # ---- Distinctness: any two contact feet sharing a patch -> invalid.
        # Air feet use a per-foot negative sentinel so they never collide.
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

        # ---- Cost: per-foot contact cost or fallback ``radius`` for air.
        cost = float(0.0)
        # Foot 0
        if contact0 == 1:
            px = patch_xy[ci0]
            patch_r = wp.length(wp.vec2(px[0] - cx, px[1] - cy))
            outward = wp.max(patch_r - tpl_r0, 0.0)
            cost = cost + cd0 + outward_pen * outward
            tx0 = px[0]
            ty0 = px[1]
        else:
            cost = cost + radius
            tx0 = p0[0]
            ty0 = p0[1]
        # Foot 1
        if contact1 == 1:
            px = patch_xy[ci1]
            patch_r = wp.length(wp.vec2(px[0] - cx, px[1] - cy))
            outward = wp.max(patch_r - tpl_r1, 0.0)
            cost = cost + cd1 + outward_pen * outward
            tx1 = px[0]
            ty1 = px[1]
        else:
            cost = cost + radius
            tx1 = p1[0]
            ty1 = p1[1]
        # Foot 2
        if contact2 == 1:
            px = patch_xy[ci2]
            patch_r = wp.length(wp.vec2(px[0] - cx, px[1] - cy))
            outward = wp.max(patch_r - tpl_r2, 0.0)
            cost = cost + cd2 + outward_pen * outward
            tx2 = px[0]
            ty2 = px[1]
        else:
            cost = cost + radius
            tx2 = p2[0]
            ty2 = p2[1]
        # Foot 3
        if contact3 == 1:
            px = patch_xy[ci3]
            patch_r = wp.length(wp.vec2(px[0] - cx, px[1] - cy))
            outward = wp.max(patch_r - tpl_r3, 0.0)
            cost = cost + cd3 + outward_pen * outward
            tx3 = px[0]
            ty3 = px[1]
        else:
            cost = cost + radius
            tx3 = p3[0]
            ty3 = p3[1]

        # Skip if already worse than current best -- saves the hull check.
        if cost >= best_cost:
            continue

        # ---- Hull validity: target points form a convex polygon whose
        # winding order matches the template's projected polygon.
        tcx = (tx0 + tx1 + tx2 + tx3) * 0.25
        tcy = (ty0 + ty1 + ty2 + ty3) * 0.25
        ta0 = wp.atan2(ty0 - tcy, tx0 - tcx)
        ta1 = wp.atan2(ty1 - tcy, tx1 - tcx)
        ta2 = wp.atan2(ty2 - tcy, tx2 - tcx)
        ta3 = wp.atan2(ty3 - tcy, tx3 - tcx)
        tgt_perm = _argsort4(ta0, ta1, ta2, ta3)

        if (
            tgt_perm[0] != tpl_perm[0]
            or tgt_perm[1] != tpl_perm[1]
            or tgt_perm[2] != tpl_perm[2]
            or tgt_perm[3] != tpl_perm[3]
        ):
            continue

        # Sorted polygon vertices (using the matching permutation).
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

        # Edge vectors (in CCW or CW order depending on the permutation).
        ex0 = sx1 - sx0
        ey0 = sy1 - sy0
        ex1 = sx2 - sx1
        ey1 = sy2 - sy1
        ex2 = sx3 - sx2
        ey2 = sy3 - sy2
        ex3 = sx0 - sx3
        ey3 = sy0 - sy3

        # Cross products of consecutive edges (z-component of 3D cross).
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

        # Combo passed every check -- update best.
        best_cost = cost
        best_c = c

    # ----- Emit outputs from best_c -----
    if best_c < 0:
        # No valid combo. Mark the row as no_convex and clear outputs.
        no_convex[k] = 1
        n_found[k] = 0
        is_contact_c[k, 0] = 0
        is_contact_c[k, 1] = 0
        is_contact_c[k, 2] = 0
        is_contact_c[k, 3] = 0
        contact_ik_c[k, 0] = wp.vec3(0.0, 0.0, 0.0)
        contact_ik_c[k, 1] = wp.vec3(0.0, 0.0, 0.0)
        contact_ik_c[k, 2] = wp.vec3(0.0, 0.0, 0.0)
        contact_ik_c[k, 3] = wp.vec3(0.0, 0.0, 0.0)
        return

    no_convex[k] = 0

    # Air-foot z floor: each foot's nearest patch z, plus the ground offset.
    af0 = patch_pts[topk_i[k * NC + 0, 0]][2] + foot_ground_offset
    af1 = patch_pts[topk_i[k * NC + 1, 0]][2] + foot_ground_offset
    af2 = patch_pts[topk_i[k * NC + 2, 0]][2] + foot_ground_offset
    af3 = patch_pts[topk_i[k * NC + 3, 0]][2] + foot_ground_offset

    n_count = int(0)
    # Foot 0
    rb = combo[best_c, 0]
    cib = topk_i[k * NC + 0, rb]
    cdb = topk_d[k * NC + 0, rb]
    if cdb < effective_radius:
        pp = patch_pts[cib]
        contact_ik_c[k, 0] = wp.vec3(pp[0], pp[1], pp[2] + foot_ground_offset)
        is_contact_c[k, 0] = 1
        n_count += 1
    else:
        z = wp.max(p0[2], af0)
        contact_ik_c[k, 0] = wp.vec3(p0[0], p0[1], z)
        is_contact_c[k, 0] = 0
    # Foot 1
    rb = combo[best_c, 1]
    cib = topk_i[k * NC + 1, rb]
    cdb = topk_d[k * NC + 1, rb]
    if cdb < effective_radius:
        pp = patch_pts[cib]
        contact_ik_c[k, 1] = wp.vec3(pp[0], pp[1], pp[2] + foot_ground_offset)
        is_contact_c[k, 1] = 1
        n_count += 1
    else:
        z = wp.max(p1[2], af1)
        contact_ik_c[k, 1] = wp.vec3(p1[0], p1[1], z)
        is_contact_c[k, 1] = 0
    # Foot 2
    rb = combo[best_c, 2]
    cib = topk_i[k * NC + 2, rb]
    cdb = topk_d[k * NC + 2, rb]
    if cdb < effective_radius:
        pp = patch_pts[cib]
        contact_ik_c[k, 2] = wp.vec3(pp[0], pp[1], pp[2] + foot_ground_offset)
        is_contact_c[k, 2] = 1
        n_count += 1
    else:
        z = wp.max(p2[2], af2)
        contact_ik_c[k, 2] = wp.vec3(p2[0], p2[1], z)
        is_contact_c[k, 2] = 0
    # Foot 3
    rb = combo[best_c, 3]
    cib = topk_i[k * NC + 3, rb]
    cdb = topk_d[k * NC + 3, rb]
    if cdb < effective_radius:
        pp = patch_pts[cib]
        contact_ik_c[k, 3] = wp.vec3(pp[0], pp[1], pp[2] + foot_ground_offset)
        is_contact_c[k, 3] = 1
        n_count += 1
    else:
        z = wp.max(p3[2], af3)
        contact_ik_c[k, 3] = wp.vec3(p3[0], p3[1], z)
        is_contact_c[k, 3] = 0

    n_found[k] = n_count


def run_lsa(
    *,
    topk_i: torch.Tensor,        # [Kc, NC, NC] long (will be cast to int32)
    topk_d: torch.Tensor,        # [Kc, NC, NC] float
    proj_pos: torch.Tensor,      # [Kc, NC, 3] float (xyz)
    patch_xy: torch.Tensor,      # [N_patches, 2] float
    patch_pts: torch.Tensor,     # [N_patches, 3] float
    combo: torch.Tensor,         # [N_COMBOS, NC] long
    radius: float,
    effective_radius: float,
    outward_pen: float,
    force_all_snap: bool,
    foot_ground_offset: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch :func:`_lsa_kernel` on a single chunk's data.

    Returns ``(is_contact_c, contact_ik_c, n_found, no_convex)``:
    * ``is_contact_c``: ``[Kc, NC]`` bool.
    * ``contact_ik_c``: ``[Kc, NC, 3]`` float -- final foot xyz.
    * ``n_found``: ``[Kc]`` long.
    * ``no_convex``: ``[Kc]`` bool -- True iff no combo passed every
      check for that row.
    """
    Kc, nc, _ = topk_i.shape
    if nc != 4:
        raise NotImplementedError(
            f"LSA Warp kernel only supports nc=4 (quadruped); got nc={nc}. "
            "Use the Python LSA path for other foot counts."
        )
    if combo.shape != (24, 4):
        raise ValueError(f"combo must be (24, 4) for nc=4; got {tuple(combo.shape)}")

    device = topk_i.device

    topk_i_flat = topk_i.reshape(Kc * nc, nc).to(torch.int32).contiguous()
    topk_d_flat = topk_d.reshape(Kc * nc, nc).contiguous()
    proj_pos_c = proj_pos.contiguous()
    combo_c = combo.to(torch.int32).contiguous()

    is_contact_c = torch.empty((Kc, nc), dtype=torch.int32, device=device)
    contact_ik_c = torch.empty((Kc, nc, 3), dtype=torch.float32, device=device)
    n_found = torch.empty((Kc,), dtype=torch.int32, device=device)
    no_convex = torch.empty((Kc,), dtype=torch.int32, device=device)

    wp.launch(
        _lsa_kernel,
        dim=Kc,
        inputs=[
            wp.from_torch(topk_i_flat, dtype=wp.int32),
            wp.from_torch(topk_d_flat, dtype=wp.float32),
            wp.from_torch(proj_pos_c, dtype=wp.vec3),
            wp.from_torch(patch_xy.contiguous(), dtype=wp.vec2),
            wp.from_torch(patch_pts.contiguous(), dtype=wp.vec3),
            wp.from_torch(combo_c, dtype=wp.int32),
            float(radius),
            float(effective_radius),
            float(outward_pen),
            int(1 if force_all_snap else 0),
            float(foot_ground_offset),
        ],
        outputs=[
            wp.from_torch(is_contact_c, dtype=wp.int32),
            wp.from_torch(contact_ik_c, dtype=wp.vec3),
            wp.from_torch(n_found, dtype=wp.int32),
            wp.from_torch(no_convex, dtype=wp.int32),
        ],
        device=str(device),
    )

    return is_contact_c.bool(), contact_ik_c, n_found.to(torch.int64), no_convex.bool()


__all__ = ["run_lsa"]
