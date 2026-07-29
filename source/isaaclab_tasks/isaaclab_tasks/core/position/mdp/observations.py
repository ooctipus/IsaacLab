# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.manager_term_cfg import ObservationTermCfg


class bound_height_scan(ManagerTermBase):
    """Height-scan observation backed by :class:`FastTerrainScanner` with the scanner's pose
    read from the host articulation's GPU-resident body tensors.

    Why: the upstream :class:`isaaclab.sensors.RayCaster` (and ``FastTerrainScanner``'s
    unbound fallback) resolves the sensor world pose through ``FrameView`` →
    ``FabricFrameView``, which is hard-gated to ``cuda:0`` in
    ``isaaclab_physx/sim/views/fabric_frame_view.py``. On any other device the path falls
    back to a per-prim USD ``xform_cache`` loop in Python, which costs ~5 ms/step at 4096
    envs and silently kills distributed training throughput on rank ≥ 1.

    Binding the scanner to the host articulation routes the per-step pose read through
    ``articulation.data.body_pos_w/.body_quat_w`` (GPU tensors, no Fabric/USD touch), so
    both ranks step the simulation at the same rate.

    How to apply: pass ``asset_cfg`` with the host articulation name and a single
    ``body_names`` entry pointing at the scanner's parent body.
    """

    cfg: ObservationTermCfg

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        # Plain RayCaster doesn't expose bind_articulation; the unbound fallback is the
        # default behavior, so silently skipping here preserves backward compatibility.
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
        asset_cfg: SceneEntityCfg,  # noqa: ARG002  (used in __init__ only)
        offset: float = 0.5,
    ) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_cfg.name]
        return sensor.data.pos_w.torch[:, 2].unsqueeze(1) - sensor.data.ray_hits_w.torch[..., 2] - offset


def gravity_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """World-frame gravity vector projected into the robot's base frame, with magnitude preserved.

    Companion to the standard :func:`~isaaclab.envs.mdp.projected_gravity` observation,
    which exposes the *unit* gravity direction. Under per-env gravity randomization
    (see :class:`~isaaclab.envs.mdp.randomize_physics_scene_gravity`) the unit
    direction conveys tilt only — heavy and light gravity are indistinguishable
    after normalization. This observation preserves ``‖g‖`` so the policy can
    additionally adapt to the magnitude (e.g. heavier loading, more reaction
    force needed at the same posture).

    Args:
        env: :class:`ManagerBasedRLEnv` instance.
        asset_cfg: Robot articulation cfg. Defaults to ``SceneEntityCfg("robot")``.

    Returns:
        Gravity vector in the base frame [m/s\\ :sup:`2`], shape ``(num_envs, 3)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    g_w = asset.data.GRAVITY_VEC_W.torch
    base_quat_w = asset.data.root_link_quat_w.torch
    return math_utils.quat_apply_inverse(base_quat_w, g_w)


def time_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    if hasattr(env, "episode_length_buf"):
        life_left = (1 - env.episode_length_buf.float() / env.max_episode_length) * env.max_episode_length_s
    else:
        life_left = torch.ones(env.num_envs, device=env.device, dtype=torch.float) * env.max_episode_length_s
    return life_left.view(-1, 1)


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
        command_name: Name of the :class:`RelativeStateCommand` term. Defaults to
            ``"goal_point"`` matching the position task.

    Returns:
        Tensor of shape ``[num_envs, 3]`` with ``(x, y, z)`` targets [m] in the
        per-env local frame.
    """
    command_term = env.command_manager.get_term(command_name)
    env_origins = env.scene.terrain.env_origins  # [num_envs, 3]
    return command_term.cmd_buf[:, 0, :3] - env_origins


def achieved_pos_env(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Currently achieved root position expressed in the per-env local frame [m].

    The HER-compatible companion to :func:`target_pos_env`: at any timestep this
    returns the agent's reached pose in the same coordinate frame as the
    commanded target. Sampling a future timestep's achieved pose gives CRL an
    automatically-correct relabeled goal.

    Args:
        env: :class:`ManagerBasedRLEnv` instance.
        command_name: Name of the :class:`RelativeStateCommand` term.

    Returns:
        Tensor of shape ``[num_envs, 3]`` with the robot root position [m]
        relative to the terrain spawn origin for that env.
    """
    command_term = env.command_manager.get_term(command_name)
    env_origins = env.scene.terrain.env_origins  # [num_envs, 3]
    return command_term.cmd_buf[:, 2, :3] - env_origins
