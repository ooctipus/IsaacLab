# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rigid-body-invariant shape descriptor for an indexed point set.

Pure tensor math. Given ``n`` rigidly-attached points (any contact set,
keypoint set, vertex polygon, …) and a per-index nominal azimuth that
encodes which slot each point occupies in the parent body's frame,
:func:`canonicalize_shape` returns per-point coordinates in a frame that
cancels rigid-body translation, yaw, and plane-fit pitch/roll. Two point
sets that differ only by rigid-body motion of the parent produce
identical canonical descriptors, so nearest-neighbour queries in this
space are pure shape-match queries.

The "nominal azimuth" terminology is intentionally generic — it's the
expected angle of slot ``i`` measured from the centroid in the parent
frame, derived externally (e.g. from a sample distribution over parent
poses). Callers supply whatever per-slot reference angles encode their
slot-identity convention.

No domain semantics — usable for legged contact polygons, factory
assembly keypoint sets, or any rigid-body shape descriptor.
"""

from __future__ import annotations

import torch


def yaw_from_xy_layout(
    xyz: torch.Tensor,
    nominal_angles: torch.Tensor,
    ref_xy: torch.Tensor | None = None,
) -> torch.Tensor:
    """Best-fit parent yaw from per-point world ``xy`` positions.

    Each point ``i`` sits at parent-frame azimuth ``yaw + nominal_angles[i]``
    from the reference ``xy``. Rotating ``(xy - ref_xy)`` by
    ``-nominal_angles[i]`` gives a vector at angle ``yaw``; summing across
    points and taking ``atan2`` yields a robust weighted estimate that
    averages out per-point noise.

    Args:
        xyz: Point positions ``[..., n, 3]``. Only ``xy`` columns are used.
        nominal_angles: Per-point nominal azimuth ``[n]`` [rad] in the
            centroid frame, expressing slot identity. Supplied externally.
        ref_xy: Reference ``xy`` ``[..., 2]`` (broadcastable). If ``None``,
            the centroid of ``xyz[..., :2]`` is used.

    Returns:
        Best-fit yaw [rad], shape ``xyz.shape[:-2]``.
    """
    n = xyz.shape[-2]
    if ref_xy is None:
        ref_xy = xyz[..., :2].mean(dim=-2)
    v_xy = xyz[..., :2] - ref_xy.unsqueeze(-2)
    cos_n = torch.cos(nominal_angles).view(*([1] * (v_xy.dim() - 2)), n)
    sin_n = torch.sin(nominal_angles).view(*([1] * (v_xy.dim() - 2)), n)
    rot_vx = cos_n * v_xy[..., 0] + sin_n * v_xy[..., 1]
    rot_vy = -sin_n * v_xy[..., 0] + cos_n * v_xy[..., 1]
    return torch.atan2(rot_vy.sum(dim=-1), rot_vx.sum(dim=-1))


def canonicalize_shape(
    xyz: torch.Tensor,
    nominal_angles: torch.Tensor,
) -> torch.Tensor:
    """Rigid-body-invariant per-point coordinates.

    Cancels rigid-body motion of the parent in three stages:

    1. **Translate** so the centroid is at the origin.
    2. **De-rotate**: apply the yaw from :func:`yaw_from_xy_layout`, then
       a plane-fit pitch / half-roll (matches the IK ``base_target_rot``
       convention used by the original locomotion caller).
    3. **Per-slot rotate**: rotate point ``i`` by ``-nominal_angles[i]`` so
       it lies in its own slot-relative frame.

    Point sets differing only by rigid-body motion of the parent produce
    identical canonical descriptors.

    Args:
        xyz: Point positions ``[..., n, 3]``.
        nominal_angles: Per-point nominal azimuth ``[n]`` [rad] in the
            centroid frame, expressing slot identity.

    Returns:
        Per-point canonical coordinates ``[..., n, 3]``.
    """
    centroid = xyz.mean(dim=-2, keepdim=True)
    delta = xyz - centroid
    yaw = yaw_from_xy_layout(xyz, nominal_angles, ref_xy=None)
    dxp, dyp, dzp = delta[..., 0], delta[..., 1], delta[..., 2]

    # Plane-fit pitch / roll from the centred xy-z scatter via least
    # squares on the 2x2 normal matrix. The xy covariance has det == 0
    # algebraically for n=2 or any colinear xy layout; floor-clamping
    # det to ``1e-12`` and dividing would amplify rounding noise in the
    # (also-zero) numerators into huge ``a, b`` whose ``atan`` saturates
    # to ±π/2, placing the descriptor in the wrong pitch/roll basin.
    # The rank check below detects degeneracy and zeros the tilt.
    xx = (dxp * dxp).sum(dim=-1)
    yy = (dyp * dyp).sum(dim=-1)
    xym = (dxp * dyp).sum(dim=-1)
    xzm = (dxp * dzp).sum(dim=-1)
    yzm = (dyp * dzp).sum(dim=-1)
    raw_det = xx * yy - xym * xym
    plane_rank_ok = raw_det > 1.0e-6 * xx * yy
    det = raw_det.clamp_min(1.0e-12)
    a = (yy * xzm - xym * yzm) / det
    b = (xx * yzm - xym * xzm) / det
    flat = (dzp.amax(dim=-1) - dzp.amin(dim=-1)) < 1.0e-4
    degenerate = flat | ~plane_rank_ok
    a = torch.where(degenerate, torch.zeros_like(a), a)
    b = torch.where(degenerate, torch.zeros_like(b), b)

    # Rotate the world-frame slopes ``(a, b)`` by ``-yaw`` into parent
    # frame, then atan to body pitch / roll. Half-roll matches the IK
    # convention upstream — keeps the parent more upright on tilted layouts.
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    a_b = a * cos_y + b * sin_y
    b_b = -a * sin_y + b * cos_y
    pitch_t = -torch.atan(a_b)
    roll_t = 0.5 * torch.atan(b_b)

    # Apply yaw, pitch, roll inverses to centred deltas.
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

    # Per-slot rotation by ``-nominal_angles[i]``: each point lands in
    # its own slot-aligned frame. Two points with the same nominal angle
    # collapse to the same canonical xy axis.
    cos_n = torch.cos(nominal_angles)
    sin_n = torch.sin(nominal_angles)
    canon_x = cos_n * rel_x + sin_n * rel_y
    canon_y = -sin_n * rel_x + cos_n * rel_y
    return torch.stack([canon_x, canon_y, rel_z], dim=-1).contiguous()
