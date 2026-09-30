# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for viewer env selection (Newton viewers and Kit partial USD visibility)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import warp as wp

VISUALIZER_INFINITE_PLANE_SIZE = 1000.0
"""Finite render size used for Newton planes encoded as infinite."""


def expand_infinite_plane_scale(
    geo_scale: tuple[float, ...], plane_size: float = VISUALIZER_INFINITE_PLANE_SIZE
) -> tuple[float, ...]:
    """Return a finite visual scale for Newton planes encoded with non-positive extents.

    Newton uses non-positive X/Y plane scale values to represent an effectively
    infinite plane. Newton GL renders those with a large finite mesh; web viewers
    also need a finite size, otherwise their world-extents heuristic can shrink
    the floor to just the actor bounds.
    """
    scale = tuple(float(value) for value in geo_scale)
    width = scale[0] if len(scale) > 0 else 0.0
    length = scale[1] if len(scale) > 1 else 0.0
    if width > 0.0 and length > 0.0:
        return scale
    tail = scale[2:] if len(scale) > 2 else ()
    return (
        width if width > 0.0 else float(plane_size),
        length if length > 0.0 else float(plane_size),
        *tail,
    )


def log_geo_with_expanded_plane_scale(
    super_log_geo: Callable[..., Any],
    plane_geo_type: int,
    name: str,
    geo_type: int,
    geo_scale: tuple[float, ...],
    geo_thickness: float,
    geo_is_solid: bool,
    geo_src=None,
    hidden: bool = False,
):
    """Log geometry after expanding Newton infinite-plane extents for web viewers."""
    if geo_type == plane_geo_type:
        geo_scale = expand_infinite_plane_scale(geo_scale)
    return super_log_geo(name, geo_type, geo_scale, geo_thickness, geo_is_solid, geo_src, hidden)


def resolve_visible_env_indices(
    env_ids: list[int] | None,
    max_visible_envs: int | None,
    num_envs: int,
) -> list[int] | None:
    """Resolve which environment indices stay visible.

    * Cap-only path (``env_ids`` is ``None``): contiguous ``0 .. min(cap, num_envs) - 1`` when ``max_visible_envs``
      is set; otherwise ``None`` (viewer shows all worlds). (Random cap-only selection is applied earlier by
      turning it into explicit ``env_ids``.)
    * Explicit path (``env_ids`` is a list): remove duplicate indices while preserving order, then keep only the
      first *cap* indices when ``max_visible_envs`` is set.

    Returns:
        Selected indices, or ``None`` when all environments should be visible (cap-only, no limit).
    """
    if env_ids is not None:
        out = list(dict.fromkeys(env_ids))
        if max_visible_envs is not None:
            out = out[: max(0, int(max_visible_envs))]
        return out
    if max_visible_envs is not None and num_envs > 0:
        n = min(int(max_visible_envs), num_envs)
        return list(range(n))
    return None


def log_state_particles(viewer, state) -> None:
    """Log the point pointer already ordered for the clone-built Newton model by SDP."""
    model = viewer.model
    if model is None or not model.particle_count:
        return

    points = state.particle_q
    if points is None:
        raise RuntimeError("Newton viewer received a particle model without an SDP point publication.")
    if len(points) != model.particle_count:
        raise RuntimeError(
            f"SDP published {len(points)} points for a Newton model with {model.particle_count} particles."
        )

    points = viewer._apply_layer_transform_to_points(points)
    colors = (
        wp.full(shape=len(points), value=wp.vec3(0.7, 0.6, 0.4), device=viewer.device) if viewer.model_changed else None
    )
    viewer.log_points(
        name=viewer._qualify("/model/particles"),
        points=points,
        radii=model.particle_radius,
        colors=colors,
        hidden=not viewer.show_particles or viewer._layer_force_hidden(),
    )


# TODO: Newton GL's checker floor (GeoType.PLANE, material.z=1.0) renders in the Newton GL
# OpenGL viewport but the streaming view seen in the browser shows composited Isaac Sim
# camera-sensor frames (RTX-rendered).  Patching Newton's GLSL checker_scale only affects
# the OpenGL window, not the camera images, so it has no visible effect from the user's
# perspective.  To fix the floor appearance in the streaming view, the Isaac Sim USD scene
# for each task needs an updated floor material/texture (see the Kuka Allegro env for a
# reference with a blue-grid floor).
