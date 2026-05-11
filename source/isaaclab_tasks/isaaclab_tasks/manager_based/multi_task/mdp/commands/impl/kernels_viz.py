# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug-visualization for state kernels that track the BASE entity.

Two-function-per-kernel pattern (mirrors ``state_command``):

- ``markers_*()`` — pure-data factory returning a list of
  ``(prim_path, VisualizationMarkersCfg)`` pairs. Called once at
  ``_set_debug_vis_impl(True)`` time to create marker prims. No scene-
  state access; safe to call before articulation views are bound.
- ``viz_*()`` — per-step update returning a ``dict[marker_path, kwargs]``
  forwarded to :meth:`VisualizationMarkers.visualize` for each marker.
  Reads scene state (robot pose, current velocity) and computes per-env
  translations / orientations / scales.

Each base-tracking kernel registers TWO markers:

- ``*_goal``: green — the commanded target.
- ``*_current``: blue — the robot's actual live state.

Color/scale conventions match ``state_command`` so the viewer feels
consistent across single- and multi-task envs:

- vel arrow base scale ``(0.5, 0.5, 0.5)``; per-instance ``[|v|·3, 1, 1]``.
- pos cuboid 0.25 m, red.
- pose frame 0.4 m axes.

Velocity arrows are anchored at the robot's base position (lifted
``+_LIFT_Z`` so they sit above the body) and rendered in **world frame**.
The composer's body-frame rotation contract is irrelevant for viz — the
arrow's job is to show the *visual* direction the robot should/does move,
which is the world-frame velocity. Rotating by the full base quaternion
(my earlier impl) tied the arrow to the robot's roll/pitch oscillations
and made it tremble; world-frame rendering keeps it stable.

This module deliberately depends on :mod:`isaaclab.markers` and
:mod:`isaaclab.utils.math` — Kit-launched-time imports. The kernel
registry uses ``_lazy_attr`` wrappers so cfg construction stays pre-Kit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

if TYPE_CHECKING:
    from ..multi_task_command import MultiTaskCommand


_HIDE_Z = -100.0
"""Park inactive markers below the ground plane."""

_LIFT_Z = 0.5
"""Vertical offset above the robot root so markers sit above the body."""


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _hide_inactive(world_pos: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Park inactive rows at ``z = _HIDE_Z`` so the marker stays out of view."""
    hidden = world_pos.clone()
    hidden[:, 2] = _HIDE_Z
    return torch.where(active.unsqueeze(-1), world_pos, hidden)


def _quat_align_x_to_vec(vec: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Quaternion that rotates body-frame +x onto ``vec``, plus magnitudes.

    Returns ``(quat[N,4]_xyzw, magnitude[N])``. Near-zero vectors collapse
    to identity rotation.
    """
    mag = torch.linalg.vector_norm(vec, dim=-1)
    safe_mag = mag.clamp(min=eps)
    direction = vec / safe_mag.unsqueeze(-1)
    x = torch.zeros_like(direction)
    x[:, 0] = 1.0
    cos_t = (x * direction).sum(dim=-1).clamp(-1.0, 1.0)
    w = torch.sqrt(((1.0 + cos_t) / 2.0).clamp(min=0.0))
    sin_half = torch.sqrt(((1.0 - cos_t) / 2.0).clamp(min=0.0))
    axis = torch.linalg.cross(x, direction)
    axis_norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp(min=eps)
    axis = axis / axis_norm
    xyz = axis * sin_half.unsqueeze(-1)
    quat_xyzw = torch.cat([xyz, w.unsqueeze(-1)], dim=-1)
    identity = torch.zeros_like(quat_xyzw)
    identity[:, 3] = 1.0
    quat_xyzw = torch.where((1.0 - cos_t).unsqueeze(-1) < eps, identity, quat_xyzw)
    return quat_xyzw, mag


def _arrow_kwargs_from_world_vec(
    vec_w: torch.Tensor, base_pos_w: torch.Tensor, active: torch.Tensor, length_scale: float = 3.0
) -> dict:
    """Build ``visualize`` kwargs for an arrow at ``base_pos_w + lift`` along ``vec_w``."""
    pos = base_pos_w.clone()
    pos[:, 2] += _LIFT_Z
    pos = _hide_inactive(pos, active)
    arrow_quat, magnitude = _quat_align_x_to_vec(vec_w)
    scales = torch.ones((vec_w.shape[0], 3), device=vec_w.device)
    scales[:, 0] = (magnitude * length_scale).clamp(min=0.05)
    scales = torch.where(active.unsqueeze(-1), scales, torch.zeros_like(scales))
    return {"translations": pos, "orientations": arrow_quat, "scales": scales}


# ---------------------------------------------------------------------------
# BODY_POS — red cuboid at world target. No "current" marker (the robot
# itself shows the current pos).
# ---------------------------------------------------------------------------


_PATH_BODY_POS_GOAL = "/Visuals/MultiTaskCommand/body_pos_goal"


def markers_body_pos() -> list[tuple[str, object]]:
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkersCfg

    return [
        (
            _PATH_BODY_POS_GOAL,
            VisualizationMarkersCfg(
                prim_path=_PATH_BODY_POS_GOAL,
                markers={
                    "cuboid": sim_utils.CuboidCfg(
                        size=(0.25, 0.25, 0.25),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            ),
        )
    ]


def viz_body_pos(cmd: MultiTaskCommand, target_per_env: torch.Tensor, active: torch.Tensor) -> dict:
    env_origins = cmd._env.scene.env_origins
    world_pos = target_per_env + env_origins
    world_pos = world_pos.clone()
    world_pos[:, 2] += _LIFT_Z
    pos = _hide_inactive(world_pos, active)
    return {_PATH_BODY_POS_GOAL: {"translations": pos}}


# ---------------------------------------------------------------------------
# BODY_QUAT — red frame at robot world pos, oriented to target. No "current"
# (the robot's frame is the current orientation).
# ---------------------------------------------------------------------------


_PATH_BODY_QUAT_GOAL = "/Visuals/MultiTaskCommand/body_quat_goal"


def markers_body_quat() -> list[tuple[str, object]]:
    from isaaclab.markers import FRAME_MARKER_CFG

    cfg = FRAME_MARKER_CFG.replace(prim_path=_PATH_BODY_QUAT_GOAL)
    cfg.markers["frame"].scale = (0.4, 0.4, 0.4)
    return [(_PATH_BODY_QUAT_GOAL, cfg)]


def viz_body_quat(cmd: MultiTaskCommand, target_per_env: torch.Tensor, active: torch.Tensor) -> dict:
    robot = cmd._env.scene["robot"]
    base_pos_w = wp.to_torch(robot.data.root_pos_w).clone()
    base_pos_w[:, 2] += _LIFT_Z
    pos = _hide_inactive(base_pos_w, active)
    return {_PATH_BODY_QUAT_GOAL: {"translations": pos, "orientations": target_per_env}}


# ---------------------------------------------------------------------------
# BODY_LIN_VEL — green goal arrow + blue current arrow.
# ---------------------------------------------------------------------------


_PATH_BODY_LIN_VEL_GOAL = "/Visuals/MultiTaskCommand/body_lin_vel_goal"
_PATH_BODY_LIN_VEL_CURRENT = "/Visuals/MultiTaskCommand/body_lin_vel_current"


def markers_body_lin_vel() -> list[tuple[str, object]]:
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkersCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    arrow_usd = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd"
    return [
        (
            _PATH_BODY_LIN_VEL_GOAL,
            VisualizationMarkersCfg(
                prim_path=_PATH_BODY_LIN_VEL_GOAL,
                markers={
                    "arrow": sim_utils.UsdFileCfg(
                        usd_path=arrow_usd,
                        scale=(0.5, 0.5, 0.5),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                },
            ),
        ),
        (
            _PATH_BODY_LIN_VEL_CURRENT,
            VisualizationMarkersCfg(
                prim_path=_PATH_BODY_LIN_VEL_CURRENT,
                markers={
                    "arrow": sim_utils.UsdFileCfg(
                        usd_path=arrow_usd,
                        scale=(0.5, 0.5, 0.5),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                    ),
                },
            ),
        ),
    ]


def viz_body_lin_vel(cmd: MultiTaskCommand, target_per_env: torch.Tensor, active: torch.Tensor) -> dict:
    """Goal (green) = target velocity in WORLD frame, anchored at robot pos.
    Current (blue) = robot's actual world-frame linear velocity.

    Treating target as world-frame for viz keeps the arrow stable as the
    robot rolls/pitches — the composer's body-frame error semantics don't
    apply to visualization.
    """
    robot = cmd._env.scene["robot"]
    base_pos_w = wp.to_torch(robot.data.root_pos_w)
    # Goal arrow — target velocity rendered in world frame directly.
    goal_kwargs = _arrow_kwargs_from_world_vec(target_per_env, base_pos_w, active)
    # Current arrow — robot's actual world-frame velocity. Always shown
    # for envs where the goal is active so they can be visually compared.
    current_w = wp.to_torch(robot.data.root_lin_vel_w)
    current_kwargs = _arrow_kwargs_from_world_vec(current_w, base_pos_w, active)
    return {_PATH_BODY_LIN_VEL_GOAL: goal_kwargs, _PATH_BODY_LIN_VEL_CURRENT: current_kwargs}


# ---------------------------------------------------------------------------
# BODY_ANG_VEL — green goal + blue current. Same convention.
# ---------------------------------------------------------------------------


_PATH_BODY_ANG_VEL_GOAL = "/Visuals/MultiTaskCommand/body_ang_vel_goal"
_PATH_BODY_ANG_VEL_CURRENT = "/Visuals/MultiTaskCommand/body_ang_vel_current"


def markers_body_ang_vel() -> list[tuple[str, object]]:
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkersCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    arrow_usd = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd"
    return [
        (
            _PATH_BODY_ANG_VEL_GOAL,
            VisualizationMarkersCfg(
                prim_path=_PATH_BODY_ANG_VEL_GOAL,
                markers={
                    "arrow": sim_utils.UsdFileCfg(
                        usd_path=arrow_usd,
                        scale=(0.5, 0.5, 0.5),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                },
            ),
        ),
        (
            _PATH_BODY_ANG_VEL_CURRENT,
            VisualizationMarkersCfg(
                prim_path=_PATH_BODY_ANG_VEL_CURRENT,
                markers={
                    "arrow": sim_utils.UsdFileCfg(
                        usd_path=arrow_usd,
                        scale=(0.5, 0.5, 0.5),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                    ),
                },
            ),
        ),
    ]


def viz_body_ang_vel(cmd: MultiTaskCommand, target_per_env: torch.Tensor, active: torch.Tensor) -> dict:
    """Goal (green) = ω target axis in world frame, length ∝ |ω|.
    Current (blue) = robot's actual world-frame ω axis."""
    robot = cmd._env.scene["robot"]
    base_pos_w = wp.to_torch(robot.data.root_pos_w)
    # Smaller length scale than lin_vel — ang vel magnitudes (rad/s) often
    # exceed lin vel (m/s), so this keeps arrow length in a comparable range.
    goal_kwargs = _arrow_kwargs_from_world_vec(target_per_env, base_pos_w, active, length_scale=1.0)
    current_w = wp.to_torch(robot.data.root_ang_vel_w)
    current_kwargs = _arrow_kwargs_from_world_vec(current_w, base_pos_w, active, length_scale=1.0)
    return {_PATH_BODY_ANG_VEL_GOAL: goal_kwargs, _PATH_BODY_ANG_VEL_CURRENT: current_kwargs}
