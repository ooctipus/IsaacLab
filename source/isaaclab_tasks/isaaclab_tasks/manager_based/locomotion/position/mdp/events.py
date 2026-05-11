# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


def _resolve_attr(root, dotted_path: str):
    """Resolve ``"a.b.c"`` into ``root.a.b.c`` via cached ``operator.attrgetter``."""
    return reduce(getattr, dotted_path.split("."), root)


def reset_root_state_from_terrain(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_noise: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    spawn_pos_path: str = "scene.terrain.env_origins",
    asset_cfg: SceneEntityCfg = None,
):
    """Reset root state using a spawn-position buffer resolved by dotted attribute path.

    The position tensor is looked up at runtime from ``env.<spawn_pos_path>``.
    Orientation uses the asset default orientation plus sampled roll/pitch/yaw noise.

    Args:
        env: The environment instance.
        env_ids: Indices of environments to reset.
        pose_noise: Additive noise ranges for ``x``, ``y``, ``z``,
            ``roll``, ``pitch``, ``yaw`` [m or rad].
        velocity_range: Velocity randomization ranges for
            ``x``, ``y``, ``z``, ``roll``, ``pitch``, ``yaw`` [m/s or rad/s].
        spawn_pos_path: Dotted path from ``env`` to the ``[N, 3]`` position
            buffer. Defaults to ``"scene.terrain.env_origins"``.
        asset_cfg: Scene entity to reset. Defaults to ``"robot"``.
    """
    if asset_cfg is None:
        from isaaclab.managers import SceneEntityCfg

        asset_cfg = SceneEntityCfg("robot")

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    default_root_pose = wp.to_torch(asset.data.default_root_pose)[env_ids].clone()
    default_root_vel = wp.to_torch(asset.data.default_root_vel)[env_ids].clone()

    spawn_pos = _resolve_attr(env, spawn_pos_path)[env_ids]

    # position = default offset + spawn origin + noise
    pos_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    noise_ranges = torch.tensor([pose_noise.get(k, (0.0, 0.0)) for k in pos_keys], device=asset.device)
    noise = math_utils.sample_uniform(noise_ranges[:, 0], noise_ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = default_root_pose[:, 0:3] + spawn_pos + noise[:, 0:3]

    # orientation = default orientation * euler_noise(roll, pitch, yaw)
    noise_quat = math_utils.quat_from_euler_xyz(noise[:, 3], noise[:, 4], noise[:, 5])
    orientations = math_utils.quat_mul(default_root_pose[:, 3:7], noise_quat)

    # velocity
    vel_ranges = torch.tensor([velocity_range.get(k, (0.0, 0.0)) for k in pos_keys], device=asset.device)
    velocities = default_root_vel + math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=asset.device
    )

    asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)
