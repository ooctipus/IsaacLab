# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.manager_term_cfg import ObservationTermCfg


class bound_height_scan(ManagerTermBase):
    """Flat height-scan observation bound to the robot body."""

    cfg: ObservationTermCfg

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        if not hasattr(sensor, "bind_articulation"):
            return
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        if asset_cfg.body_names is None or len(asset_cfg.body_names) != 1:
            raise ValueError(
                "bound_height_scan: asset_cfg.body_names must list exactly one body to bind to;"
                f" got {asset_cfg.body_names!r}."
            )
        sensor.bind_articulation(env.scene[asset_cfg.name], asset_cfg.body_names[0])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        offset: float = 0.5,
    ) -> torch.Tensor:
        del env, asset_cfg
        sensor = self._env.scene.sensors[sensor_cfg.name]
        return sensor.data.pos_w.torch[:, 2].unsqueeze(1) - sensor.data.ray_hits_w.torch[..., 2] - offset


def gravity_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """World-frame gravity vector projected into the robot's base frame, with magnitude preserved.

    Companion to the standard :func:`~isaaclab.envs.mdp.projected_gravity` observation,
    which exposes the *unit* gravity direction. Under per-env gravity randomization
    (see :func:`~isaaclab.envs.mdp.randomize_physics_scene_gravity`) the unit
    direction conveys tilt only -- heavy and light gravity are indistinguishable
    after normalization. This observation preserves ``||g||`` so the policy can
    additionally adapt to the magnitude (e.g. heavier loading, more reaction
    force needed at the same posture).

    Args:
        env: :class:`~isaaclab.envs.ManagerBasedRLEnv` instance.
        asset_cfg: Robot articulation cfg. Defaults to ``SceneEntityCfg("robot")``.

    Returns:
        Gravity vector in the base frame [m/s^2], shape ``[num_envs, 3]``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    g_w = asset.data.GRAVITY_VEC_W.torch
    base_quat_w = asset.data.root_link_quat_w.torch
    return math_utils.quat_apply_inverse(base_quat_w, g_w)


def target_pos_env(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Commanded target position expressed in the per-env local frame [m].

    Built for CRL: the commanded goal must be an *absolute* reachable-pose slice
    (not a relative-state delta) so that Hindsight Experience Replay can relabel
    with reached poses from the same trajectory.

    The returned vector is the commanded world-position minus the env's terrain-
    spawn origin, keeping the coordinate range stable across the many parallel
    envs that live at different world locations.

    Args:
        env: :class:`ManagerBasedRLEnv` instance.
        command_name: Name of the :class:`~isaaclab_tasks.core.multi_task.mdp.commands.StateCommand` term. Defaults to
            ``"goal_point"`` matching the position task.

    Returns:
        Tensor of shape ``[num_envs, 3]`` with ``(x, y, z)`` targets [m] in the
        per-env local frame.
    """
    return env.command_manager.get_term(command_name).get_state("target_position")


def achieved_pos_env(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Currently achieved root position expressed in the per-env local frame [m].

    The HER-compatible companion to :func:`target_pos_env`: at any timestep this
    returns the agent's reached pose in the same coordinate frame as the
    commanded target. Sampling a future timestep's achieved pose gives CRL an
    automatically-correct relabeled goal.

    Args:
        env: :class:`ManagerBasedRLEnv` instance.
        command_name: Name of the :class:`~isaaclab_tasks.core.multi_task.mdp.commands.StateCommand` term.

    Returns:
        Tensor of shape ``[num_envs, 3]`` with the robot root position [m]
        relative to the terrain spawn origin for that env.
    """
    return env.command_manager.get_term(command_name).get_state("current_position")


def command_current_state(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Current state: root pose/vel (env-local position) + foot positions.

    Layout: ``[x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, foot_pos...]``
    where ``foot_pos`` is ``num_feet`` positions in env-local world frame
    (world minus env origin), flattened.

    Foot positions match the payload success criterion, so HER relabeling
    ``target ← future current`` produces a self-consistent goal whose error
    matches the actual reward signal.

    Args:
        env: :class:`ManagerBasedRLEnv` instance.
        command_name: Name of the :class:`~isaaclab_tasks.core.multi_task.mdp.commands.StateCommand` term.

    Returns:
        Tensor of shape ``[num_envs, 12 + 3 * num_feet]``.
    """
    return env.command_manager.get_term(command_name).get_state("current")


def command_std(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Per-env success thresholds of the currently-bound task as a policy observation.

    Returns the active payload's per-error-group thresholds so the policy can
    see the threshold alongside the raw command delta.

    Args:
        env: :class:`ManagerBasedRLEnv` instance.
        command_name: Name of the :class:`~isaaclab_tasks.core.multi_task.mdp.commands.StateCommand` term.

    Returns:
        Tensor of shape ``[num_envs, num_error_groups]``.
    """
    return env.command_manager.get_term(command_name).command_std


def command_target_state(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Target state: root pose/vel (env-local position) + foot positions.

    Layout matches :func:`command_current_state`. The foot portion is the
    commanded target foot positions in env-local world frame.

    Args:
        env: :class:`ManagerBasedRLEnv` instance.
        command_name: Name of the :class:`~isaaclab_tasks.core.multi_task.mdp.commands.StateCommand` term.

    Returns:
        Tensor of shape ``[num_envs, 12 + 3 * num_feet]``.
    """
    return env.command_manager.get_term(command_name).get_state("target")
