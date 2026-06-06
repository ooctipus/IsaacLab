# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Top-down terrain heightmap utility for 2D dashboards.

A small, dependency-light helper that raycasts the scene's ground mesh
straight down into a regular ``[H, W]`` grid so 2D plots (curriculum
spawn-scatter, trajectory recorder, ...) can paint a consistent
backdrop. Used by:

* :func:`.sampler_images.spawn_goal_scatter_image`
* :class:`...viz.trajectory_recorder.TrajectoryRecorder`

We keep the raycast in one place because (a) the heightmap is
expensive enough (~1M rays for the default 1024×1024 grid) to want
to compute once and reuse across panels, and (b) every consumer
needs the same ``(extent, border-cropped bounds, NaN-on-miss)``
contract or the panels will misalign.
"""

from __future__ import annotations

import numpy as np
import torch


def render_terrain_background(
    scene_terrain,
    *,
    device: torch.device | str,
    resolution: int = 1024,
) -> tuple[np.ndarray | None, tuple[float, float, float, float] | None]:
    """One-time top-down raycast → heightmap suitable for ``imshow``.

    Casts a ``resolution × resolution`` grid of downward rays from above
    ``terrain.terrain_mesh`` to recover the topmost surface, so
    overhanging features (beams, floating islands) shadow correctly
    without per-pixel max-z bookkeeping. Misses come back as ``NaN``
    so colormaps render them transparent.

    Args:
        scene_terrain: ``env.scene.terrain`` -- expected to expose
            ``terrain_mesh`` (with ``vertices`` and ``faces``) and
            ``cfg.terrain_generator.border_width``.
        device: Device to host the warp mesh + raycast on.
        resolution: Grid side length in pixels.

    Returns:
        ``(heightmap, extent)`` where ``heightmap`` is ``[H, W]`` float32
        with ``NaN`` for misses, and ``extent`` is
        ``(xmin, xmax, ymin, ymax)`` [m] in world frame matching
        :paramref:`matplotlib.axes.Axes.imshow.extent` semantics.
        ``(None, None)`` when the terrain has no mesh (e.g. plane-only
        scenes), so callers can fall back to a plain background.
    """
    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

    terrain_mesh = getattr(scene_terrain, "terrain_mesh", None)
    if terrain_mesh is None or terrain_mesh.vertices.shape[0] == 0:
        return None, None

    verts = np.asarray(terrain_mesh.vertices, dtype=np.float32)
    faces = np.asarray(terrain_mesh.faces, dtype=np.int32)
    xmin, xmax = float(verts[:, 0].min()), float(verts[:, 0].max())
    ymin, ymax = float(verts[:, 1].min()), float(verts[:, 1].max())
    zmax = float(verts[:, 2].max())

    # Crop the flat ``border_width`` perimeter -- no patches/spawn live
    # there and including it just compresses the active tile grid into a
    # smaller central region of the panel.
    border = float(getattr(scene_terrain.cfg.terrain_generator, "border_width", 0.0))
    if border > 0.0:
        xmin += border
        xmax -= border
        ymin += border
        ymax -= border

    H = W = int(resolution)
    xs = np.linspace(xmin, xmax, W, dtype=np.float32)
    ys = np.linspace(ymin, ymax, H, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    starts = np.stack(
        [
            grid_x.ravel(),
            grid_y.ravel(),
            np.full(H * W, zmax + 1.0, dtype=np.float32),
        ],
        axis=-1,
    )
    dirs = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (H * W, 1))
    starts_t = torch.from_numpy(starts).to(device)
    dirs_t = torch.from_numpy(dirs).to(device)

    wp_mesh = convert_to_warp_mesh(verts, faces, device=str(device))
    hits, _, _, _ = raycast_mesh(starts_t, dirs_t, wp_mesh, max_dist=zmax + 100.0)
    z = hits[:, 2].view(H, W).cpu().numpy()
    z = np.where(np.isfinite(z), z, np.nan)

    return z, (xmin, xmax, ymin, ymax)


def heightmap_to_rgb(
    heightmap: np.ndarray,
    *,
    cmap_name: str = "gray",
    miss_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Apply a matplotlib colormap to a heightmap, returning ``[H, W, 3] uint8``.

    Pure-numpy consumers (no matplotlib draw cycle) need a baked RGB
    canvas they can just blit per frame. We render through matplotlib's
    cmap LUT once -- not via ``imshow`` -- so the cost is purely the
    LUT lookup (microseconds), not a full figure.

    Args:
        heightmap: ``[H, W]`` float; ``NaN`` marks misses.
        cmap_name: matplotlib colormap name. Defaults to plain ``"gray"``
            (darker = lower terrain) -- vivid maps like ``"terrain"``
            wash out the dot overlay we paint on top.
        miss_color: RGB triplet [0..255] painted into NaN cells.

    Returns:
        ``[H, W, 3] uint8`` RGB suitable for direct numpy splatting.
    """
    import matplotlib

    valid = np.isfinite(heightmap)
    if not valid.any():
        H, W = heightmap.shape
        return np.broadcast_to(np.asarray(miss_color, dtype=np.uint8), (H, W, 3)).copy()

    z = heightmap.copy()
    z_min = float(np.nanmin(z))
    z_max = float(np.nanmax(z))
    span = max(z_max - z_min, 1e-6)
    z_norm = (z - z_min) / span
    z_norm = np.where(valid, z_norm, 0.0)

    cmap = matplotlib.colormaps[cmap_name]
    rgba = (cmap(z_norm)[..., :3] * 255.0).astype(np.uint8)
    rgba[~valid] = np.asarray(miss_color, dtype=np.uint8)
    return rgba
