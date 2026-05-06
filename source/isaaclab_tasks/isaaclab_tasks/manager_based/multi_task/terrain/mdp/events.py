# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from functools import reduce
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase

from ..viz import TrajectoryRecorder

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
    spawn_quat_path: str = "scene.terrain.env_spawn_quats",
    asset_cfg: SceneEntityCfg = None,
):
    """Reset root state using spawn buffers resolved by dotted attribute paths.

    The *position* and *quaternion* tensors are looked up at runtime from
    ``env.<spawn_pos_path>`` and ``env.<spawn_quat_path>``.  This avoids
    hard-coding any particular buffer location and lets callers point to
    custom tensors via the event config.

    Args:
        env: The environment instance.
        env_ids: Indices of environments to reset.
        pose_noise: Additive noise ranges for ``x``, ``y``, ``z``,
            ``roll``, ``pitch``, ``yaw`` [m or rad].
        velocity_range: Velocity randomization ranges for
            ``x``, ``y``, ``z``, ``roll``, ``pitch``, ``yaw`` [m/s or rad/s].
        spawn_pos_path: Dotted path from ``env`` to the ``[N, 3]`` position
            buffer. Defaults to ``"scene.terrain.env_origins"``.
        spawn_quat_path: Dotted path from ``env`` to the ``[N, 4]`` quaternion
            (x,y,z,w) buffer. Defaults to ``"scene.terrain.env_spawn_quats"``.
        asset_cfg: Scene entity to reset. Defaults to ``"robot"``.
    """
    if asset_cfg is None:
        from isaaclab.managers import SceneEntityCfg

        asset_cfg = SceneEntityCfg("robot")

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    default_root_pose = wp.to_torch(asset.data.default_root_pose)[env_ids].clone()
    default_root_vel = wp.to_torch(asset.data.default_root_vel)[env_ids].clone()

    spawn_pos = _resolve_attr(env, spawn_pos_path)[env_ids]
    spawn_quat = _resolve_attr(env, spawn_quat_path)[env_ids]  # (x, y, z, w)

    # position = default offset + spawn origin + noise
    pos_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    noise_ranges = torch.tensor([pose_noise.get(k, (0.0, 0.0)) for k in pos_keys], device=asset.device)
    noise = math_utils.sample_uniform(noise_ranges[:, 0], noise_ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = default_root_pose[:, 0:3] + spawn_pos + noise[:, 0:3]

    # orientation = spawn_quat * euler_noise(roll, pitch, yaw)
    noise_quat = math_utils.quat_from_euler_xyz(noise[:, 3], noise[:, 4], noise[:, 5])
    orientations = math_utils.quat_mul(spawn_quat, noise_quat)

    # velocity
    vel_ranges = torch.tensor([velocity_range.get(k, (0.0, 0.0)) for k in pos_keys], device=asset.device)
    velocities = default_root_vel + math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=asset.device
    )

    asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)


class record_trajectory_video(ManagerTermBase):
    """Per-step event term that drives a :class:`TrajectoryRecorder`.

    Plug into the env cfg as an interval event firing every step::

        traj_video = EventTerm(
            func=mdp.record_trajectory_video,
            mode="interval",
            interval_range_s=(env.step_dt, env.step_dt),  # every step
            is_global_time=True,
            params={
                "command_name": "goal_point",
                "video_interval": 5000,
                "video_length": 200,
                "max_envs": 16,
            },
        )

    The recorder writes mp4s under ``{log_dir}/videos/trajectory/``;
    rsl_rl's logger globs ``*.mp4`` from ``log_dir`` at every iteration
    end and auto-uploads to W&B, so no logger plumbing is needed.

    Lightweight: per-step capture is two small ``[N_subset, 2]`` CPU
    copies plus an ``[N_subset]`` bool. Render runs at end-of-window.
    """

    def __init__(self, cfg: EventTermCfg, env) -> None:
        super().__init__(cfg, env)
        params = cfg.params
        log_dir = getattr(env.cfg, "log_dir", None) or os.getcwd()
        video_folder = params.get("video_folder") or os.path.join(log_dir, "videos", "trajectory")
        self.command_name: str = params.get("command_name", "goal_point")
        self.recorder = TrajectoryRecorder(
            video_folder=video_folder,
            video_interval=int(params.get("video_interval", 5000)),
            video_length=int(params.get("video_length", 200)),
            max_envs=int(params.get("max_envs", 16)),
            fps=int(params.get("fps", 30)),
            trail_steps=int(params.get("trail_steps", 30)),
        )

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor, **_: object) -> None:
        # Read the latest per-env state and pass it to the recorder.
        # This is a side-effect-only term -- it returns nothing.
        cmd = env.command_manager.get_term(self.command_name)
        robot = env.scene["robot"]

        robot_pos = robot.data.root_pos_w
        if not isinstance(robot_pos, torch.Tensor):
            robot_pos = wp.to_torch(robot_pos)
        env_origins = env.scene.env_origins
        if not isinstance(env_origins, torch.Tensor):
            env_origins = wp.to_torch(env_origins)

        # Targets: row 0 of the cmd buffer, columns 0:3 = position.
        target_pos = cmd.cmd_buf[:, 0, :2]
        # ``instant_success`` -- all active error groups below threshold this step.
        success = (cmd._err < cmd._reward_scales[cmd.cmd_ids.long()]).all(dim=1)

        robot_xy = (robot_pos[:, :2] - env_origins[:, :2]).detach().cpu().numpy()
        target_xy = (target_pos - env_origins[:, :2]).detach().cpu().numpy()
        success_np = success.detach().cpu().numpy()

        self.recorder.maybe_record(robot_xy, target_xy, success_np)
