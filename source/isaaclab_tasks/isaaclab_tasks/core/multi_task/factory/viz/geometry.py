# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build-time geometry for the factory success grid.

The reset-state poses never move, so the per-state silhouettes are precomputed
once (where the IK pipeline + its meshes are still alive) and stashed on the
table as static polygons; the periodic logger only recolors and draws them.

Each silhouette is a convex ``k``-gon support polygon (the extreme projected
vertex per direction) rather than a true convex hull -- fully vectorized, no
SciPy, and visually indistinguishable for an overview at cell scale.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from ..retarget.model import _quat_xyzw_rot

if TYPE_CHECKING:
    from ..retarget.model import FactoryIKModel


def _support_kgon(xy: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Convex support polygon of ``xy`` for ``k`` directions.

    Args:
        xy: Projected vertices ``[..., V, 2]``.
        dirs: Unit support directions ``[k, 2]`` (angle-ordered).

    Returns:
        The support vertex per direction ``[..., k, 2]`` (a convex k-gon).
    """
    proj = xy @ dirs.t()  # [..., V, k]
    idx = proj.argmax(dim=-2)  # [..., k]
    return torch.gather(xy, -2, idx.unsqueeze(-1).expand(*idx.shape, 2))


def _quat_apply_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` ``[..., 3]`` by xyzw quaternion ``q`` ``[..., 4]`` (matched shapes)."""
    qv = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + w * t + torch.cross(qv, t, dim=-1)


def _posed_kgon(verts: torch.Tensor, pose: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Top-down support k-gon of ``verts`` posed by ``pose`` ``[M, 7]`` (pos + xyzw) -> ``[M, k, 2]``."""
    m, n_verts = pose.shape[0], verts.shape[0]
    q = pose[:, None, 3:7].expand(m, n_verts, 4)
    world = pose[:, None, :3] + _quat_apply_xyzw(q, verts[None].expand(m, n_verts, 3))
    return _support_kgon(world[..., :2], dirs)


def _body_local_verts(model) -> dict[int, np.ndarray]:
    """Per-body collider vertices in the body frame (all shapes of a body merged)."""
    shape_body = model.shape_body.numpy()
    shape_tf = wp.to_torch(model.shape_transform).cpu().numpy()  # [S, 7] pos + xyzw
    groups: dict[int, list[np.ndarray]] = {}
    for si in range(model.shape_count):
        src = model.shape_source[si] if si < len(model.shape_source) else None
        verts = getattr(src, "vertices", None) if src is not None else None
        if verts is None:
            continue
        verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
        body_local = _quat_xyzw_rot(shape_tf[si, 3:7], verts) + shape_tf[si, :3]
        groups.setdefault(int(shape_body[si]), []).append(body_local)
    return {body: np.concatenate(parts, axis=0) for body, parts in groups.items()}


def build_success_grid_geometry(
    model: FactoryIKModel,
    joint_q: torch.Tensor,
    nut_pose: torch.Tensor,
    bolt_pose: torch.Tensor,
    board_pose: torch.Tensor,
    board_index: torch.Tensor,
    k: int = 16,
) -> dict[str, object]:
    """Precompute static top-down silhouettes for the success grid.

    All inputs are aligned to the stored reset-state rows (one entry per state)
    and expressed in the pipeline base frame (robot base at the env origin).

    Args:
        model: The IK model (meshes + batched FK), alive at table-build time.
        joint_q: Per-state robot joint coordinates ``[N, nq]`` (Newton order).
        nut_pose: Per-state held-asset pose ``[N, 7]`` (pos + xyzw).
        bolt_pose: Per-state fixed-asset (bolt) pose ``[N, 7]``.
        board_pose: Per-state board pose ``[N, 7]``.
        board_index: Per-state board-configuration id ``[N]``.
        k: Silhouette support-direction count (polygon resolution).

    Returns:
        A dict of stash fields (numpy): ``viz_link_polys`` ``[N, n_bodies, k, 2]``,
        ``viz_nut_polys`` ``[N, k, 2]``, ``viz_board_polys`` / ``viz_bolt_polys``
        ``[n_boards, k, 2]``, ``viz_cell_of_state`` ``[N]`` (0..n_boards-1), and
        ``viz_n_boards``.
    """
    device = joint_q.device
    angles = torch.linspace(0.0, 2.0 * math.pi, k + 1, device=device)[:-1]
    dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # [k, 2]

    body_q = model.eval_fk(joint_q)  # [N, body_count, 7]
    body_local = _body_local_verts(model.model)
    link_polys = torch.stack(
        [_posed_kgon(torch.as_tensor(body_local[b], device=device), body_q[:, b, :], dirs) for b in sorted(body_local)],
        dim=1,
    )  # [N, n_bodies, k, 2]

    nut_polys = _posed_kgon(torch.as_tensor(model.held_verts, device=device), nut_pose, dirs)

    n_states = joint_q.shape[0]
    uniq, inverse = torch.unique(board_index, return_inverse=True)
    n_boards = int(uniq.numel())
    # first stored row per board configuration (its representative pose)
    rep = torch.full((n_boards,), n_states, dtype=torch.long, device=device)
    rep.scatter_reduce_(0, inverse, torch.arange(n_states, device=device), reduce="amin", include_self=True)
    board_polys = _posed_kgon(torch.as_tensor(model.board_verts, device=device), board_pose[rep], dirs)
    bolt_polys = _posed_kgon(torch.as_tensor(model.fixed_verts, device=device), bolt_pose[rep], dirs)

    return {
        "viz_link_polys": link_polys.cpu().numpy(),
        "viz_nut_polys": nut_polys.cpu().numpy(),
        "viz_board_polys": board_polys.cpu().numpy(),
        "viz_bolt_polys": bolt_polys.cpu().numpy(),
        "viz_cell_of_state": inverse.cpu().numpy().astype(np.int64),
        "viz_n_boards": n_boards,
    }
