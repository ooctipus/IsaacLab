# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain-agnostic termination terms shared across terrain and factory tasks.

Three terms and one base cfg live here:

- :func:`abnormal_robot_state` — joint-velocity limit watchdog. Fires when any
  joint of the asset exceeds twice its declared joint-vel limit. Indicates
  unstable physics from extreme actions and applies equally to manipulators
  and legged robots.
- :func:`out_of_bound` — env-origin-relative AABB containment check on a rigid
  asset's root position. Replaces the absolute-z ``root_height_below_minimum``
  used by terrain (which doesn't generalize to non-zero spawn heights) and
  generalizes the manipulation-side held-asset bounds check.
- :class:`illegal_contact_ratio` — contact-impact watchdog. Fires when a
  contact-sensor body's force exceeds ``ratio × total_bodyweight``. The
  threshold is computed at init from the articulation's per-body mass, so
  the same cfg works across robots without per-robot tuning.
- :class:`BaseTerminationsCfg` — shared cfg with ``time_out`` + ``abnormal``
  defaults. Domain-specific cfgs (factory, terrain) extend this and add their
  own ``oob`` term with appropriate ``asset_cfg`` + ``in_bound_range``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
import warp as wp

from isaaclab.envs.mdp import time_out as _time_out
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors.joint_wrench import JointWrenchSensor


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


class illegal_contact_ratio(ManagerTermBase):
    """Terminate when contact force exceeds ``threshold_ratio × total_bodyweight``.

    The threshold is resolved at construction from the articulation's
    per-body mass — ``ratio × Σ mᵢ × g`` — so the same cfg works across
    robots of different sizes without per-robot threshold presets.

    ``threshold_ratio = 3`` is the natural starting point: routine static
    contact (lying, kneeling, climbing) tops out around 1× bodyweight while
    shock impacts easily exceed 5-10×, so the middle band cleanly separates
    them.

    Domain-agnostic: usable by any task whose contact sensor's body subset
    should be impact-gated (locomotion non-foot bodies, manipulation tool
    shanks, …).

    Args (passed via :attr:`isaaclab.managers.TerminationTermCfg.params`):
        threshold_ratio: Multiple of total bodyweight that constitutes an
            impact.
        sensor_cfg: Contact sensor + body subset to monitor.
        asset_cfg: Articulation whose total mass defines bodyweight.
            Defaults to ``SceneEntityCfg("robot")``.

    Note: the per-env threshold is fixed at construction. Per-episode mass
    randomisation events (e.g. ``add_base_mass``) shift the true bodyweight
    by a few percent, well below the static-vs-impact margin, so the cached
    threshold remains a valid impact gate.
    """

    def __init__(self, cfg: DoneTerm, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        threshold_ratio = float(cfg.params["threshold_ratio"])
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._sensor = env.scene.sensors[sensor_cfg.name]
        self._body_ids = sensor_cfg.body_ids
        asset: Articulation = env.scene[asset_cfg.name]
        # [num_envs, 1] for broadcast against per-body force [num_envs, n_bodies].
        total_mass = wp.to_torch(asset.data.body_mass).sum(dim=-1)
        self._threshold = (threshold_ratio * total_mass * 9.81).unsqueeze(-1)
        # Manager will not pass kwargs back to ``__call__`` if cfg.params is empty.
        cfg.params = {}

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        net_forces = wp.to_torch(self._sensor.data.net_forces_w_history)
        max_force = torch.max(torch.linalg.norm(net_forces[:, :, self._body_ids], dim=-1), dim=1)[0]
        return torch.any(max_force > self._threshold, dim=1)


class joint_reaction_overload(ManagerTermBase):
    """Terminate when reported joint reaction force exceeds a body-weight multiple."""

    def __init__(self, cfg: DoneTerm, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        params = cast(dict[str, Any], cfg.params)
        force_ratio = float(params["force_ratio"])
        sensor_cfg: SceneEntityCfg = params.get("sensor_cfg", SceneEntityCfg("joint_wrench"))
        asset_cfg: SceneEntityCfg = params.get("asset_cfg", SceneEntityCfg("robot"))
        force_mode = str(params.get("force_mode", "off_axis"))
        if force_mode not in ("off_axis", "magnitude"):
            raise ValueError("joint_reaction_overload force_mode must be 'off_axis' or 'magnitude'.")
        self._sensor = cast("JointWrenchSensor", env.scene.sensors[sensor_cfg.name])
        if force_mode == "off_axis":
            axes_data = self._sensor.data.force_axes
            if axes_data is None:
                raise RuntimeError("joint_reaction_overload force_mode='off_axis' requires joint force axes.")
            self._force_axes = torch.nn.functional.normalize(axes_data.torch, dim=-1).unsqueeze(0)
            self._reduce_force = self._force_off_axis
        else:
            self._reduce_force = self._force_identity
        asset: Articulation = env.scene[asset_cfg.name]
        total_mass = asset.data.body_mass.torch.sum(dim=-1)
        self._threshold = (force_ratio * total_mass * 9.81).unsqueeze(-1)
        cfg.params = {}

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        force_data = self._sensor.data.force
        if force_data is None:
            raise RuntimeError("joint_reaction_overload requires an initialized JointWrenchSensor.")
        force = self._reduce_force(force_data.torch)
        max_force = torch.linalg.norm(force, dim=-1)
        return torch.any(max_force > self._threshold, dim=1)

    def _force_identity(self, force: torch.Tensor) -> torch.Tensor:
        return force

    def _force_off_axis(self, force: torch.Tensor) -> torch.Tensor:
        return force - torch.sum(force * self._force_axes, dim=-1, keepdim=True) * self._force_axes


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
