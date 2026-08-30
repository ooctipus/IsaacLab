# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug-visualization registry for state kernels that track the BASE entity.

This module is imported lazily by :meth:`MultiTaskCommand._set_debug_vis_impl`
post-Kit (it depends on :mod:`isaaclab.sim` and :mod:`isaaclab.markers` which
need Kit to be loaded). Module-level cfg constants are direct, declarative
:class:`VisualizationMarkersCfg` literals; the :data:`VIZ_REGISTRY` table maps
state-kernel ids to their marker list + per-step update function.

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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ..kernel_ids import STATE_KERNEL_ID

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..multi_task_command import MultiTaskCommand


_HIDE_Z = -100.0
"""Park inactive markers below the ground plane."""

_LIFT_Z = 0.5
"""Vertical offset above the robot root so markers sit above the body."""


# ---------------------------------------------------------------------------
# Marker cfg literals — declarative, module-level.
# ---------------------------------------------------------------------------

_PATH_BODY_POS_GOAL = "/Visuals/MultiTaskCommand/body_pos_goal"
_PATH_BODY_QUAT_GOAL = "/Visuals/MultiTaskCommand/body_quat_goal"
_PATH_BODY_LIN_VEL_GOAL = "/Visuals/MultiTaskCommand/body_lin_vel_goal"
_PATH_BODY_LIN_VEL_CURRENT = "/Visuals/MultiTaskCommand/body_lin_vel_current"
_PATH_BODY_ANG_VEL_GOAL = "/Visuals/MultiTaskCommand/body_ang_vel_goal"
_PATH_BODY_ANG_VEL_CURRENT = "/Visuals/MultiTaskCommand/body_ang_vel_current"

# Cfg templates. Each variant differs only in prim_path; use .replace() at the
# call site. Goal markers are green, current markers are blue — colors baked
# into the template so the variation is purely path.
_ARROW_USD = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd"

_ARROW_TEMPLATE_GOAL = VisualizationMarkersCfg(
    prim_path="",  # set per variant via .replace()
    markers={
        "arrow": sim_utils.UsdFileCfg(
            usd_path=_ARROW_USD,
            scale=(0.5, 0.5, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    },
)
_ARROW_TEMPLATE_CURRENT = VisualizationMarkersCfg(
    prim_path="",
    markers={
        "arrow": sim_utils.UsdFileCfg(
            usd_path=_ARROW_USD,
            scale=(0.5, 0.5, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    },
)

_BODY_QUAT_GOAL_CFG = FRAME_MARKER_CFG.replace(prim_path=_PATH_BODY_QUAT_GOAL)
_BODY_QUAT_GOAL_CFG.markers["frame"].scale = (0.4, 0.4, 0.4)

_MARKERS_BODY_POS: list[tuple[str, VisualizationMarkersCfg]] = [
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
    ),
]
_MARKERS_BODY_QUAT: list[tuple[str, VisualizationMarkersCfg]] = [
    (_PATH_BODY_QUAT_GOAL, _BODY_QUAT_GOAL_CFG),
]
_MARKERS_BODY_LIN_VEL: list[tuple[str, VisualizationMarkersCfg]] = [
    (_PATH_BODY_LIN_VEL_GOAL, _ARROW_TEMPLATE_GOAL.replace(prim_path=_PATH_BODY_LIN_VEL_GOAL)),
    (_PATH_BODY_LIN_VEL_CURRENT, _ARROW_TEMPLATE_CURRENT.replace(prim_path=_PATH_BODY_LIN_VEL_CURRENT)),
]
_MARKERS_BODY_ANG_VEL: list[tuple[str, VisualizationMarkersCfg]] = [
    (_PATH_BODY_ANG_VEL_GOAL, _ARROW_TEMPLATE_GOAL.replace(prim_path=_PATH_BODY_ANG_VEL_GOAL)),
    (_PATH_BODY_ANG_VEL_CURRENT, _ARROW_TEMPLATE_CURRENT.replace(prim_path=_PATH_BODY_ANG_VEL_CURRENT)),
]


# ---------------------------------------------------------------------------
# Helpers shared by the per-step viz_fns.
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
# Per-step viz functions. Each takes the command, the kernel's per-env target
# slice, and the per-env active mask; returns ``dict[marker_path, kwargs]``.
# ---------------------------------------------------------------------------


def _viz_body_pos(cmd: MultiTaskCommand, target_per_env: torch.Tensor, active: torch.Tensor) -> dict:
    env_origins = cmd._env.scene.env_origins
    world_pos = target_per_env + env_origins
    world_pos = world_pos.clone()
    world_pos[:, 2] += _LIFT_Z
    pos = _hide_inactive(world_pos, active)
    return {_PATH_BODY_POS_GOAL: {"translations": pos}}


def _viz_body_quat(cmd: MultiTaskCommand, target_per_env: torch.Tensor, active: torch.Tensor) -> dict:
    robot = cmd._env.scene["robot"]
    base_pos_w = robot.data.root_pos_w.torch.clone()
    base_pos_w[:, 2] += _LIFT_Z
    pos = _hide_inactive(base_pos_w, active)
    return {_PATH_BODY_QUAT_GOAL: {"translations": pos, "orientations": target_per_env}}


def _viz_body_lin_vel(cmd: MultiTaskCommand, target_per_env: torch.Tensor, active: torch.Tensor) -> dict:
    """Goal (green) = target velocity in WORLD frame, anchored at robot pos.
    Current (blue) = robot's actual world-frame linear velocity.

    Treating target as world-frame for viz keeps the arrow stable as the
    robot rolls/pitches — the composer's body-frame error semantics don't
    apply to visualization.
    """
    robot = cmd._env.scene["robot"]
    base_pos_w = robot.data.root_pos_w.torch
    goal_kwargs = _arrow_kwargs_from_world_vec(target_per_env, base_pos_w, active)
    current_w = robot.data.root_lin_vel_w.torch
    current_kwargs = _arrow_kwargs_from_world_vec(current_w, base_pos_w, active)
    return {_PATH_BODY_LIN_VEL_GOAL: goal_kwargs, _PATH_BODY_LIN_VEL_CURRENT: current_kwargs}


def _viz_body_ang_vel(cmd: MultiTaskCommand, target_per_env: torch.Tensor, active: torch.Tensor) -> dict:
    """Goal (green) = ω target axis in world frame, length ∝ |ω|.
    Current (blue) = robot's actual world-frame ω axis."""
    robot = cmd._env.scene["robot"]
    base_pos_w = robot.data.root_pos_w.torch
    # Smaller length scale than lin_vel — ang vel magnitudes (rad/s) often
    # exceed lin vel (m/s), so this keeps arrow length in a comparable range.
    goal_kwargs = _arrow_kwargs_from_world_vec(target_per_env, base_pos_w, active, length_scale=1.0)
    current_w = robot.data.root_ang_vel_w.torch
    current_kwargs = _arrow_kwargs_from_world_vec(current_w, base_pos_w, active, length_scale=1.0)
    return {_PATH_BODY_ANG_VEL_GOAL: goal_kwargs, _PATH_BODY_ANG_VEL_CURRENT: current_kwargs}


# ---------------------------------------------------------------------------
# Viz registry: state-kernel id → markers + per-step update fn.
#
# Kernels without an honest spatial primitive (JOINT_*, BODY_POS_Z,
# CONTACT_*, JOINT_MECH_POWER) are simply absent from this table — the base
# command term iterates the registry, not the full STATE_KERNELS tuple.
# ---------------------------------------------------------------------------


class VizEntry(NamedTuple):
    """One state-kernel's viz binding."""

    markers: list[tuple[str, VisualizationMarkersCfg]]
    """``(prim_path, cfg)`` pairs — typically a goal + a current marker."""
    update_fn: Callable[[MultiTaskCommand, torch.Tensor, torch.Tensor], dict]
    """Per-step update returning ``{marker_path: visualize_kwargs}``."""


VIZ_REGISTRY: dict[int, VizEntry] = {
    int(STATE_KERNEL_ID.BODY_POS): VizEntry(_MARKERS_BODY_POS, _viz_body_pos),
    int(STATE_KERNEL_ID.BODY_QUAT): VizEntry(_MARKERS_BODY_QUAT, _viz_body_quat),
    int(STATE_KERNEL_ID.BODY_LIN_VEL): VizEntry(_MARKERS_BODY_LIN_VEL, _viz_body_lin_vel),
    int(STATE_KERNEL_ID.BODY_ANG_VEL): VizEntry(_MARKERS_BODY_ANG_VEL, _viz_body_ang_vel),
}
