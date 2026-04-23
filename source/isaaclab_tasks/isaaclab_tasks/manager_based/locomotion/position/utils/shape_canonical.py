# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rigid-body-invariant shape descriptor for contact polygons.

Pure tensor operations shared by samplers that need a rotation- and
translation-invariant descriptor of a contact tuple. Given ``nc`` world-
frame contact positions and the robot's per-slot nominal angles (the
circular mean of each slot's angle relative to the polygon centroid
across the FK sample distribution — a base-pose-invariant quantity
derived in :meth:`TerrainFirstSampler._compute_foot_reachability`),
:func:`canonicalize_shape` returns per-contact coordinates in a frame
that cancels all rigid-body motion, so two tuples differing only by
base translation/yaw/pitch/roll produce identical canonical shapes.

The module has no sampler-specific state: both :class:`TerrainFirstSampler`
and :class:`TemplateMatchedSampler` import it.
"""

from __future__ import annotations

import torch


def yaw_from_foot_xy(
    foot_xyz: torch.Tensor,
    nominal_angles: torch.Tensor,
    ref_xy: torch.Tensor | None = None,
) -> torch.Tensor:
    """Derive a best-fit base yaw from per-foot world positions.

    Each contact ``f`` sits at polygon-frame direction
    ``yaw + nominal_angles[f]`` from the centroid. Rotating
    ``(foot_xy - ref_xy)`` by ``-nominal_angles[f]`` gives a vector at
    angle ``yaw``; summing across contacts and taking ``atan2`` yields a
    robust weighted estimate that averages out per-contact sampling noise.

    Args:
        foot_xyz: Contact positions ``[..., nc, 3]`` [m]. Only ``xy``
            columns are consulted.
        nominal_angles: Per-contact nominal azimuth ``[nc]`` [rad] in
            the polygon-centroid frame, derived from the FK sample
            distribution (not the URDF default base pose).
        ref_xy: Base reference position ``[..., 2]`` [m] (broadcastable to
            ``foot_xyz[..., :2]`` after unsqueezing the contact dim). If
            ``None``, the polygon centroid is used.

    Returns:
        Best-fit base yaw [rad], shape ``foot_xyz.shape[:-2]``.
    """
    nc = foot_xyz.shape[-2]
    if ref_xy is None:
        ref_xy = foot_xyz[..., :2].mean(dim=-2)
    v_xy = foot_xyz[..., :2] - ref_xy.unsqueeze(-2)
    cos_n = torch.cos(nominal_angles).view(*([1] * (v_xy.dim() - 2)), nc)
    sin_n = torch.sin(nominal_angles).view(*([1] * (v_xy.dim() - 2)), nc)
    rot_vx = cos_n * v_xy[..., 0] + sin_n * v_xy[..., 1]
    rot_vy = -sin_n * v_xy[..., 0] + cos_n * v_xy[..., 1]
    return torch.atan2(rot_vy.sum(dim=-1), rot_vx.sum(dim=-1))


def canonicalize_shape(
    feet_xyz: torch.Tensor,
    nominal_angles: torch.Tensor,
) -> torch.Tensor:
    """Rigid-body-invariant polygon shape descriptor.

    Transforms contact world positions into a per-contact canonical frame
    by cancelling rigid-body motion in three stages:

    1. Centre at polygon centroid.
    2. Apply the yaw returned by :func:`yaw_from_foot_xy` (symmetric
       best-fit over contacts), then plane-fit pitch/roll (half-roll,
       full-pitch -- matches the IK ``base_target_rot`` convention).
    3. Rotate each contact by ``-nominal_angles[f]`` so the contact lies in
       its own hip-outward frame.

    Polygons differing by base rigid-body motion produce identical
    canonical shapes, so nearest-neighbour queries in this space are
    pure shape-match queries.

    Args:
        feet_xyz: Contact positions ``[..., nc, 3]`` [m].
        nominal_angles: Per-contact nominal azimuth ``[nc]`` [rad] in
            the polygon-centroid frame, derived from the FK sample
            distribution (not the URDF default base pose).

    Returns:
        Per-contact canonical coordinates ``[..., nc, 3]`` [m].
    """
    centroid = feet_xyz.mean(dim=-2, keepdim=True)
    delta = feet_xyz - centroid
    yaw = yaw_from_foot_xy(feet_xyz, nominal_angles, ref_xy=None)
    dxp, dyp, dzp = delta[..., 0], delta[..., 1], delta[..., 2]
    xx = (dxp * dxp).sum(dim=-1)
    yy = (dyp * dyp).sum(dim=-1)
    xym = (dxp * dyp).sum(dim=-1)
    xzm = (dxp * dzp).sum(dim=-1)
    yzm = (dyp * dzp).sum(dim=-1)
    # Rank check on the xy covariance: nc=2 (or any colinear layout) has
    # ``det == 0`` algebraically; floor-clamping to ``1e-12`` and dividing
    # amplifies rounding noise in the (also-zero) numerators into huge
    # ``a, b`` whose ``atan`` saturates to +/-pi/2, placing canonical
    # shapes in the wrong pitch/roll basin. Mirrors the same guard in
    # :func:`_prepare_ik_kernel` so IK-target and shape-match spaces
    # agree.
    raw_det = xx * yy - xym * xym
    plane_rank_ok = raw_det > 1.0e-6 * xx * yy
    det = raw_det.clamp_min(1.0e-12)
    a = (yy * xzm - xym * yzm) / det
    b = (xx * yzm - xym * xzm) / det
    flat = (dzp.amax(dim=-1) - dzp.amin(dim=-1)) < 1.0e-4
    degenerate = flat | ~plane_rank_ok
    a = torch.where(degenerate, torch.zeros_like(a), a)
    b = torch.where(degenerate, torch.zeros_like(b), b)
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
    cos_n = torch.cos(nominal_angles)
    sin_n = torch.sin(nominal_angles)
    canon_x = cos_n * rel_x + sin_n * rel_y
    canon_y = -sin_n * rel_x + cos_n * rel_y
    return torch.stack([canon_x, canon_y, rel_z], dim=-1).contiguous()
