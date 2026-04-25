# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain-agnostic termination terms shared across terrain and factory tasks.

Two functions and one base cfg live here:

- :func:`abnormal_robot_state` — joint-velocity limit watchdog. Fires when any
  joint of the asset exceeds twice its declared joint-vel limit. Indicates
  unstable physics from extreme actions and applies equally to manipulators
  and legged robots.
- :func:`out_of_bound` — env-origin-relative AABB containment check on a rigid
  asset's root position. Replaces the absolute-z ``root_height_below_minimum``
  used by terrain (which doesn't generalize to non-zero spawn heights) and
  generalizes the manipulation-side held-asset bounds check.
- :class:`BaseTerminationsCfg` — shared cfg with ``time_out`` + ``abnormal``
  defaults. Domain-specific cfgs (factory, terrain) extend this and add their
  own ``oob`` term with appropriate ``asset_cfg`` + ``in_bound_range``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.envs.mdp import time_out as _time_out
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Fire when any joint speed exceeds twice its declared limit.

    Catches unstable physics from extreme actions — applies to any articulated
    asset (manipulator arm, legged base, …).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    return (wp.to_torch(robot.data.joint_vel).abs() > (wp.to_torch(robot.data.joint_vel_limits) * 2)).any(dim=1)


def out_of_bound(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    in_bound_range: dict[str, tuple[float, float]] = {},
) -> torch.Tensor:
    """Fire when the asset's env-relative root position leaves the AABB.

    Args:
        env: The environment.
        asset_cfg: The asset to track. Defaults to the ``"robot"`` scene entity.
        in_bound_range: Per-axis ``(min, max)`` bounds in env-local frame. Axes
            absent from the dict default to ``(0.0, 0.0)`` — i.e. nothing
            allowed — so callers should specify every axis they care about.

    Note: env-origin-relative, not absolute-world. For terrain envs whose
    spawn z varies with the terrain mesh, this remains correct because the
    env origin tracks the spawn cell.
    """
    object: RigidObject = env.scene[asset_cfg.name]
    range_list = [in_bound_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)

    object_pos_local = wp.to_torch(object.data.root_pos_w) - env.scene.env_origins
    return ((object_pos_local < ranges[:, 0]) | (object_pos_local > ranges[:, 1])).any(dim=1)


@configclass
class BaseTerminationsCfg:
    """Shared termination defaults for terrain + factory tasks.

    Domain-specific cfgs add ``oob`` (with their own ``asset_cfg`` +
    ``in_bound_range``) plus any task-specific terms (``base_contact``,
    ``progress_context``, ``success``, …) by inheriting from this class.
    """

    time_out = DoneTerm(func=_time_out, time_out=True)
    """Episode-length timeout — fires when the env's step counter reaches
    ``max_episode_length``. ``time_out=True`` so rsl_rl bootstraps off it."""

    abnormal = DoneTerm(func=abnormal_robot_state)
    """Joint-velocity-limit watchdog. Catches diverged simulations."""
