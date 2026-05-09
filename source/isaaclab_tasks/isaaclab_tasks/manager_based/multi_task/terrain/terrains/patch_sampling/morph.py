# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Morphological (deterministic, GPU-batched) patch sampling.

Rasterizes a mesh into a 2D heightmap, runs a max-min morphological
filter with the robot footprint as kernel, then samples ``num_patches``
cells from the valid region. Replaces the earlier rejection-sampling
path for large, dense terrains.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager

import numpy as np
import torch
import warp as wp

from ....utils.grid_downsample import grid_bucket_downsample
from . import cfg as patch_cfg
from .kernels import morph_validity_kernel, rasterize_grid_kernel

MORPH_TIMINGS: dict[str, float] = {}
"""Cumulative wall-time per sub-phase of :func:`find_flat_patches_morphological`.

Populated only while :func:`_morph_time` is active. Cleared at the start of
each call to :func:`find_flat_patches_morphological`. Callers can read this
dict after invocation to report a breakdown (see RetargetPipeline).
"""


@contextmanager
def _morph_time(name: str, device):
    """Record wall time for a morphological-sampling sub-phase with CUDA sync.

    ``device`` may be a :class:`torch.device` or a device-string such as
    ``"cuda:0"`` (what :func:`warp.device_to_torch` returns on Warp meshes).
    """
    dev_str = device.type if isinstance(device, torch.device) else str(device)
    is_cuda = dev_str.startswith("cuda")
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if is_cuda:
            torch.cuda.synchronize()
        MORPH_TIMINGS[name] = MORPH_TIMINGS.get(name, 0.0) + (time.perf_counter() - t0)


def _resolve_footprint(cfg):
    """Ensure *cfg* is a footprint configclass, not a plain dict.

    The monkey-patching in ``terrain_cfg.py`` round-trips configs through
    ``to_dict()`` / ``cfg_class(**dict)``, which turns nested configclasses
    into plain dicts.  This helper reconstitutes the correct type.
    """
    if isinstance(cfg, (patch_cfg.CircleFootprintCfg, patch_cfg.RectFootprintCfg)):
        return cfg
    if isinstance(cfg, dict):
        if "length" in cfg and "width" in cfg:
            return patch_cfg.RectFootprintCfg(length=cfg["length"], width=cfg["width"])
        return patch_cfg.CircleFootprintCfg(radius=cfg["radius"])
    raise TypeError(f"Unknown footprint type: {type(cfg)}")


def _build_footprint_mask(
    cfg: patch_cfg.CircleFootprintCfg | patch_cfg.RectFootprintCfg, scale: float, device: torch.device
) -> torch.Tensor:
    """Convert a footprint config into a 2D boolean kernel mask.

    Args:
        cfg: Footprint configuration (or a dict that will be auto-resolved).
        scale: Grid cell size [m] (horizontal_scale).
        device: Torch device for the output tensor.

    Returns:
        Boolean tensor of shape ``[K, K]`` where K is always odd.
    """
    cfg = _resolve_footprint(cfg)

    if isinstance(cfg, patch_cfg.CircleFootprintCfg):
        r_cells = math.ceil(cfg.radius / scale)
        k = 2 * r_cells + 1
        y, x = torch.meshgrid(
            torch.arange(k, device=device) - r_cells,
            torch.arange(k, device=device) - r_cells,
            indexing="ij",
        )
        mask = (x.float() * scale) ** 2 + (y.float() * scale) ** 2 <= cfg.radius**2
    elif isinstance(cfg, patch_cfg.RectFootprintCfg):
        hl = math.ceil(cfg.length / (2.0 * scale))  # half-length along +x (forward)
        hw = math.ceil(cfg.width / (2.0 * scale))  # half-width along +y (lateral)
        k = 2 * max(hl, hw) + 1
        y, x = torch.meshgrid(
            torch.arange(k, device=device) - k // 2,
            torch.arange(k, device=device) - k // 2,
            indexing="ij",
        )
        mask = (x.float().abs() * scale <= cfg.length / 2.0) & (y.float().abs() * scale <= cfg.width / 2.0)
    else:
        raise TypeError(f"Unknown footprint type: {type(cfg)}")

    if not mask.any():
        mask[k // 2, k // 2] = True
    return mask


def _build_rotated_rect_masks(
    footprint, scale: float, yaw_angles: torch.Tensor, device: torch.device
) -> list[torch.Tensor]:
    """Build rotated rectangular footprint masks for each yaw angle.

    Returns a list of ``[K, K]`` boolean masks, one per yaw.
    """
    hl = footprint.length / 2.0  # half-length along +x (forward)
    hw = footprint.width / 2.0  # half-width along +y (lateral)
    r_max = math.sqrt(hl**2 + hw**2)
    r_cells = math.ceil(r_max / scale)
    k = 2 * r_cells + 1
    y, x = torch.meshgrid(
        torch.arange(k, device=device, dtype=torch.float32) - r_cells,
        torch.arange(k, device=device, dtype=torch.float32) - r_cells,
        indexing="ij",
    )
    wx = x * scale
    wy = y * scale

    masks = []
    for yaw in yaw_angles:
        c, s = float(yaw.cos()), float(yaw.sin())
        lx = wx * c + wy * s  # local x (forward)
        ly = -wx * s + wy * c  # local y (lateral)
        masks.append((lx.abs() <= hl) & (ly.abs() <= hw))
    return masks


def _yaw_to_quat_xyzw(yaw: torch.Tensor) -> torch.Tensor:
    """Convert yaw angles [rad] to quaternions in ``(x, y, z, w)`` convention.

    Args:
        yaw: Tensor of yaw angles, any shape.

    Returns:
        Quaternion tensor with shape ``(*yaw.shape, 4)``.
    """
    half = yaw * 0.5
    zeros = torch.zeros_like(half)
    return torch.stack([zeros, zeros, half.sin(), half.cos()], dim=-1)


def _rasterize_mesh(
    wp_mesh: wp.Mesh, x_range: tuple[float, float], y_range: tuple[float, float], scale: float, device: torch.device
) -> tuple[torch.Tensor, float, float]:
    """Rasterize a warp mesh to a 2D heightmap via one grid-shaped Warp launch.

    Args:
        wp_mesh: The warp mesh.
        x_range: World-space x bounds ``(min, max)`` [m].
        y_range: World-space y bounds ``(min, max)`` [m].
        scale: Grid cell size [m].
        device: Torch device.

    Returns:
        Tuple of ``(heightmap, x_min, y_min)`` where heightmap is ``[H, W]``
        with ``inf`` at missed cells, x_min/y_min are the world-space
        coordinates of cell ``[0, 0]``.
    """
    nx = max(int((x_range[1] - x_range[0]) / scale), 1)
    ny = max(int((y_range[1] - y_range[0]) / scale), 1)

    heightmap = torch.full((nx, ny), float("inf"), dtype=torch.float32, device=device)

    wp.launch(
        rasterize_grid_kernel,
        dim=(nx, ny),
        inputs=[
            wp_mesh.id,
            float(x_range[0]),
            float(y_range[0]),
            float(scale),
            100.0,
            1.0e6,
        ],
        outputs=[wp.from_torch(heightmap, dtype=wp.float32)],
        device=wp_mesh.device,
    )
    return heightmap, x_range[0], y_range[0]


def _morphological_validity(
    heightmap: torch.Tensor, mask: torch.Tensor, max_height_diff: float, z_range: tuple[float, float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute morphological validity and height-range maps via a Warp kernel.

    For each cell, reduces max and min over the height values under the
    boolean footprint ``mask`` and returns whether the reduction stays
    within ``max_height_diff`` and ``z_range``.

    Args:
        heightmap: ``[H, W]`` heightmap tensor (``inf`` for missed rays).
        mask: ``[K, K]`` boolean footprint kernel.
        max_height_diff: Maximum allowed height range within the footprint [m].
        z_range: ``(z_min, z_max)`` world-space bounds for valid heights.

    Returns:
        ``(valid, h_range)`` — boolean validity mask ``[H, W]`` and the
        per-cell footprint height range ``[H, W]`` (undefined on invalid
        cells; callers must gate reads on ``valid``).
    """
    H, W = heightmap.shape
    k = mask.shape[0]
    pad = k // 2
    device = heightmap.device

    hm_c = heightmap.contiguous()
    mask_u8 = mask.to(torch.uint8).contiguous()

    out_valid = torch.empty((H, W), dtype=torch.uint8, device=device)
    out_h_range = torch.empty((H, W), dtype=torch.float32, device=device)

    wp.launch(
        morph_validity_kernel,
        dim=(H, W),
        inputs=[
            wp.from_torch(hm_c, dtype=wp.float32),
            wp.from_torch(mask_u8, dtype=wp.uint8),
            float(max_height_diff),
            float(z_range[0]),
            float(z_range[1]),
            int(pad),
        ],
        outputs=[
            wp.from_torch(out_valid, dtype=wp.uint8),
            wp.from_torch(out_h_range, dtype=wp.float32),
        ],
        device=str(device),
    )
    return out_valid.to(torch.bool), out_h_range


def find_flat_patches_morphological(
    wp_mesh: wp.Mesh,
    origin: np.ndarray | torch.Tensor | tuple[float, float, float],
    cfg: patch_cfg.MorphologicalPatchSamplingCfg,
) -> torch.Tensor:
    """Find flat patches using deterministic morphological heightmap filtering.

    Instead of rejection sampling, this function:

    1. Rasterizes the mesh to a 2D heightmap with one batched ray-cast.
    2. Computes a validity mask via morphological max-min filtering using the
       configured robot footprint kernel.
    3. For rectangular footprints, tests multiple yaw angles and records which
       yaw produces the smallest height range at each cell.
    4. Samples ``num_patches`` from the valid region, optionally with
       farthest-point refinement for spatial coverage.

    Args:
        wp_mesh: The warp mesh to find patches on.
        origin: Sub-terrain origin in the mesh frame.
        cfg: Morphological sampling configuration.

    Returns:
        Tensor of shape ``(num_patches, 7)`` — ``[x, y, z, qx, qy, qz, qw]``
        in the mesh frame with origin subtracted (quaternion is absolute).
    """
    MORPH_TIMINGS.clear()
    device = wp.device_to_torch(wp_mesh.device)
    footprint = _resolve_footprint(cfg.footprint)

    with _morph_time("setup", device):
        if isinstance(origin, np.ndarray):
            origin_t = torch.from_numpy(origin).float().to(device)
        elif isinstance(origin, torch.Tensor):
            origin_t = origin.float().to(device)
        else:
            origin_t = torch.tensor(origin, dtype=torch.float32, device=device)

        # Compute mesh XY bounds on GPU once, then pull four scalars -- avoids
        # the per-invocation ``wp_mesh.points.numpy()`` transfer of the whole
        # vertex buffer.
        verts_xy = wp.to_torch(wp_mesh.points)[:, :2]
        bounds_min = verts_xy.amin(dim=0)
        bounds_max = verts_xy.amax(dim=0)
        mesh_xmin = float(bounds_min[0])
        mesh_xmax = float(bounds_max[0])
        mesh_ymin = float(bounds_min[1])
        mesh_ymax = float(bounds_max[1])

        ox, oy, oz = origin_t[0].item(), origin_t[1].item(), origin_t[2].item()
        x_range = (max(cfg.x_range[0] + ox, mesh_xmin), min(cfg.x_range[1] + ox, mesh_xmax))
        y_range = (max(cfg.y_range[0] + oy, mesh_ymin), min(cfg.y_range[1] + oy, mesh_ymax))
        z_range = (cfg.z_range[0] + oz, cfg.z_range[1] + oz)

        scale = cfg.horizontal_scale

    with _morph_time("rasterize", device):
        heightmap, hm_x0, hm_y0 = _rasterize_mesh(wp_mesh, x_range, y_range, scale, device)
        H, W = heightmap.shape

    is_rect = isinstance(footprint, patch_cfg.RectFootprintCfg)

    with _morph_time("validity", device):
        if is_rect:
            # test 8 discrete yaw angles in [0, pi) — rectangle has 180-deg symmetry
            num_yaw = 8
            yaw_angles = torch.linspace(0, math.pi, num_yaw + 1, device=device)[:num_yaw]
            rotated_masks = _build_rotated_rect_masks(footprint, scale, yaw_angles, device)

            best_range = torch.full((H, W), float("inf"), device=device)
            best_yaw_idx = torch.zeros((H, W), dtype=torch.long, device=device)
            combined_valid = torch.zeros((H, W), dtype=torch.bool, device=device)

            for yi, mask in enumerate(rotated_masks):
                valid_yi, h_range = _morphological_validity(heightmap, mask, cfg.max_height_diff, z_range)
                improved = valid_yi & (h_range < best_range)
                best_range[improved] = h_range[improved]
                best_yaw_idx[improved] = yi
                combined_valid |= valid_yi

            valid = combined_valid
            yaw_map = yaw_angles[best_yaw_idx]
        else:
            footprint_mask = _build_footprint_mask(footprint, scale, device)
            valid, _ = _morphological_validity(heightmap, footprint_mask, cfg.max_height_diff, z_range)
            yaw_map = torch.zeros((H, W), device=device)

        valid_coords = valid.nonzero(as_tuple=False)  # [K, 2]
        num_valid = valid_coords.shape[0]

    if num_valid < cfg.num_patches:
        total_cells = H * W
        valid_frac = num_valid / total_cells if total_cells > 0 else 0.0
        raise RuntimeError(
            f"Morphological patch sampling found only {num_valid} valid cells but"
            f" {cfg.num_patches} patches requested."
            f"\n\tGrid size: {H}x{W} ({total_cells} cells)"
            f"\n\tValid fraction: {valid_frac:.4f}"
            f"\n\tmax_height_diff: {cfg.max_height_diff}"
            f"\n\tfootprint: {footprint}"
            f"\n\tx_range: {x_range}, y_range: {y_range}, z_range: {z_range}"
            f"\n\tHint: lower horizontal_scale or relax max_height_diff to grow num_valid."
        )

    with _morph_time("candidates", device):
        n_candidates = min(int(cfg.num_patches * cfg.oversample_ratio), num_valid)
        perm = torch.randperm(num_valid, device=device)[:n_candidates]
        candidates_rc = valid_coords[perm]

        cand_x = hm_x0 + (candidates_rc[:, 0].float() + 0.5) * scale
        cand_y = hm_y0 + (candidates_rc[:, 1].float() + 0.5) * scale
        cand_z = heightmap[candidates_rc[:, 0], candidates_rc[:, 1]]
        cand_yaw = yaw_map[candidates_rc[:, 0], candidates_rc[:, 1]]
        cand_quat = _yaw_to_quat_xyzw(cand_yaw)
        cand_pos = torch.stack([cand_x, cand_y, cand_z], dim=-1)

    with _morph_time("fps", device):
        if cfg.oversample_ratio > 1.0 and n_candidates > cfg.num_patches:
            sel_idx = grid_bucket_downsample(cand_pos[:, :2], cfg.num_patches)
            pos = cand_pos[sel_idx]
            quat = cand_quat[sel_idx]
        else:
            pos = cand_pos[: cfg.num_patches]
            quat = cand_quat[: cfg.num_patches]

        result = torch.cat([pos - origin_t, quat], dim=-1)
    return result
