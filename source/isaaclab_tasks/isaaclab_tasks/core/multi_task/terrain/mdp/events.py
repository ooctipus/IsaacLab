# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import EventTermCfg, ManagerTermBase

from ..viz import TrajectoryRecorder

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class record_trajectory_video(ManagerTermBase):
    """Per-step event term that drives a :class:`TrajectoryRecorder`.

    Plug into the env cfg as an interval event firing every step::

        traj_video = EventTerm(
            func=mdp.record_trajectory_video,
            mode="interval",
            interval_range_s=(0.0, 0.0),  # every step
            is_global_time=True,
            params={
                "command_name": "goal_point",
                "video_interval": 5000,
                "video_length": 200,
            },
        )

    The recorder writes a gif (or mp4 when ffmpeg is on PATH) under
    ``{log_dir}/videos/trajectory/``, uploads each one directly to W&B
    under ``Sampler/trajectory_video``, then renames mp4 outputs to
    ``.mp4.archived`` so rsl_rl's ``rglob("*.mp4")`` can't double-upload
    under its hardcoded ``"video"`` key (which would overwrite the
    standard 3D ``RecordVideo`` panel on every subsequent iteration).

    Lightweight: per-step capture is two small ``[N_subset, 2]`` CPU
    copies plus an ``[N_subset]`` bool. Render runs at end-of-window.
    """

    def __init__(self, cfg: EventTermCfg, env) -> None:
        super().__init__(cfg, env)
        params = cfg.params
        log_dir = getattr(env.cfg, "log_dir", None) or os.getcwd()
        video_folder = params.get("video_folder") or os.path.join(log_dir, "videos", "trajectory")
        self.command_name: str = params.get("command_name", "goal_point")
        # Bake the terrain heightmap once at term construction so every
        # window's render can blit a pre-baked RGB canvas (no raycast in
        # the hot path). Mirrors the spawn-scatter background.
        from ..viz import render_terrain_background

        bg_image, bg_extent = render_terrain_background(env.scene.terrain, device=env.device)
        self.recorder = TrajectoryRecorder(
            video_folder=video_folder,
            video_interval=int(params.get("video_interval", 5000)),
            video_length=int(params.get("video_length", 200)),
            fps=int(params.get("fps", 30)),
            background_image=bg_image,
            background_extent=bg_extent,
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        # Params are consumed by __init__ via cfg.params and stored on self;
        # they're listed here only because the manager's static validator
        # introspects __call__'s signature to match against cfg.params
        # (manager_base.py:360 -- a bare ``**kwargs`` shows up as a single
        # mandatory parameter named after the kwargs binding).
        command_name: str | None = None,
        video_interval: int | None = None,
        video_length: int | None = None,
        video_folder: str | None = None,
        fps: int | None = None,
    ) -> None:
        env_subset_np = self.recorder.capture_env_indices(num_envs=env.num_envs)
        if env_subset_np is None:
            return

        # Read only the sampled envs for the active recording window. The
        # idle path above is just a counter increment, so ordinary training
        # steps do not pay GPU->CPU copies for every environment.
        cmd = env.command_manager.get_term(self.command_name)
        robot = env.scene["robot"]

        robot_pos = robot.data.root_pos_w
        if not isinstance(robot_pos, torch.Tensor):
            robot_pos = wp.to_torch(robot_pos)
        env_origins = env.scene.env_origins
        if not isinstance(env_origins, torch.Tensor):
            env_origins = wp.to_torch(env_origins)

        target_pos = cmd.get_state("target_position")[:, :2]
        success = cmd.get_task_done()

        env_subset = torch.as_tensor(env_subset_np, device=robot_pos.device, dtype=torch.long)
        origins = env_origins.index_select(0, env_subset)[:, :2]
        robot_xy = (robot_pos.index_select(0, env_subset)[:, :2] - origins).detach().cpu().numpy()
        target_xy = target_pos.index_select(0, env_subset).detach().cpu().numpy()
        success_np = success.index_select(0, env_subset).detach().cpu().numpy()

        self.recorder.append_frame(robot_xy, target_xy, success_np)
