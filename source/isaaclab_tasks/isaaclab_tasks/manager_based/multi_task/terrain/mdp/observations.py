# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class vision_obs(ManagerTermBase):
    """Unified 2D vision observation for either :class:`TiledCamera` or :class:`RayCaster`.

    Returns ``(B, C, H, W)`` (channels-first, channels-last memory format) ready for
    :class:`rsl_rl.models.CNNModel`.

    Sensor handling:

    * :class:`TiledCamera` — reads ``sensor.data.output[data_types[0]]``. ``"rgb"`` is
      normalized to ``[0, 1]`` and per-image mean-centered; ``"distance_to_image_plane"``
      / ``"depth"`` are squashed via ``tanh(x/2)·2`` and per-image mean-centered.
    * :class:`RayCaster` (with :class:`GridPatternCfg`) — height value
      ``sensor_z − hit_z − offset`` is computed and reshaped onto its underlying grid as a
      single-channel "depth-like" image, then put through the same depth normalization.

    The normalization makes the network input well-conditioned: per-image mean-subtraction
    removes the absolute-height baseline (so the policy sees relative terrain shape, not
    "robot is high" / "robot is low"), and ``tanh`` softly bounds the residual.

    Args:
        sensor_cfg: Scene-entity cfg pointing to a :class:`TiledCamera` or
            :class:`RayCaster`. Defaults to ``SceneEntityCfg("tiled_camera")``.
        normalize: If ``True`` (default), apply type-appropriate normalization and return
            ``(B, C, H, W)`` channels-first. If ``False``, return raw ``(B, H, W, C)`` for
            visualization (e.g. :meth:`show_collage`).
        offset: Subtracted from each :class:`RayCaster` height value [m]. Ignored for
            cameras. Defaults to ``0.5``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params.get("sensor_cfg", SceneEntityCfg("tiled_camera"))
        self.sensor = env.scene.sensors[sensor_cfg.name]

        # Camera-style sensors expose ``data.output[type]``; check them first because
        # ``RayCasterCamera`` inherits from ``RayCaster`` and would otherwise be routed
        # into the grid path (where its ``PinholeCameraPatternCfg`` has no ``size`` /
        # ``resolution`` attributes).
        from isaaclab.sensors import RayCaster, RayCasterCamera, TiledCamera

        from isaaclab_tasks.manager_based.multi_task.sensors import FastTerrainScanner
        if isinstance(self.sensor, (TiledCamera, RayCasterCamera)):
            self._sensor_type = self.sensor.cfg.data_types[0]
            self._fetch = self._fetch_camera
            self._norm = self._depth_norm if self._sensor_type in ("distance_to_image_plane", "depth") else self._rgb_norm
        elif isinstance(self.sensor, (RayCaster, FastTerrainScanner)):
            pattern_cfg = self.sensor.cfg.pattern_cfg
            self._nx = round(pattern_cfg.size[0] / pattern_cfg.resolution) + 1
            self._ny = round(pattern_cfg.size[1] / pattern_cfg.resolution) + 1
            self._ordering = pattern_cfg.ordering
            self._fetch = self._fetch_raycaster
            self._norm = self._depth_norm
            if isinstance(self.sensor, FastTerrainScanner):
                asset_name = cfg.params.get("asset_name", "robot")
                body_name = self.sensor.cfg.prim_path.rsplit("/", 1)[0].rsplit("/", 1)[-1]
                self.sensor.bind_articulation(env.scene[asset_name], body_name)
        else:
            raise TypeError(
                f"vision_obs supports TiledCamera, RayCasterCamera, RayCaster, or FastTerrainScanner;"
                f" got {type(self.sensor).__name__}"
            )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        normalize: bool = True,
        offset: float = 0.5,
    ) -> torch.Tensor:
        images = self._fetch(offset)  # (B, H, W, C)
        torch.nan_to_num_(images, nan=1e6)
        if normalize:
            images = self._norm(images)
            images = images.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        return images

    def _fetch_camera(self, offset: float) -> torch.Tensor:
        return self.sensor.data.output[self._sensor_type]

    def _fetch_raycaster(self, offset: float) -> torch.Tensor:
        flat = self.sensor.data.pos_w.torch[:, 2].unsqueeze(1) - self.sensor.data.ray_hits_w.torch[..., 2] - offset
        # (B, num_rays) -> (B, H, W, 1) NHWC. ordering="xy" -> rows are constant-y -> (Ny, Nx).
        if self._ordering == "xy":
            return flat.view(self.num_envs, self._ny, self._nx, 1)
        return flat.view(self.num_envs, self._nx, self._ny, 1)

    def _rgb_norm(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float() / 255.0
        images = images - torch.mean(images, dim=(1, 2), keepdim=True)
        return images

    def _depth_norm(self, images: torch.Tensor) -> torch.Tensor:
        images = torch.tanh(images / 2) * 2
        images = images - torch.mean(images, dim=(1, 2), keepdim=True)
        return images

    def collage(self, offset: float = 0.5, save_path: str = "./collage.png"):
        """Save a turbo-colormapped collage of the raw sensor tiles to disk.

        Fetches unnormalized ``(B, H, W, C)`` images via :meth:`_fetch` and
        arranges them in a square grid.  Single-channel (depth / heightmap) tiles
        are mapped through the *turbo* colormap; multi-channel (RGB) tiles are
        kept as-is.

        Args:
            offset: Height offset [m] forwarded to :meth:`_fetch_raycaster`.
            save_path: Destination file path. ``~`` is expanded.
        """
        import os

        import matplotlib
        import numpy as np
        from PIL import Image

        images = self._fetch(offset)
        torch.nan_to_num_(images, nan=0.0)
        images = images.clamp(-10.0, 10.0)
        a = images.detach().cpu().numpy()
        n, h, w, c = a.shape
        s = int(np.ceil(np.sqrt(n)))
        canvas = np.full((s * h, s * w, 3), 255, np.uint8)
        turbo = matplotlib.colormaps["turbo"]
        for i in range(n):
            r, col = divmod(i, s)
            img = a[i]
            if c == 1:
                d = img[..., 0]
                finite = np.isfinite(d)
                if finite.any():
                    lo, hi = d[finite].min(), d[finite].max()
                else:
                    lo, hi = 0.0, 1.0
                d = np.clip(d, lo, hi)
                d = (d - lo) / (hi - lo + 1e-8)
                rgb = (turbo(d)[..., :3] * 255).astype(np.uint8)
            else:
                x = img if img.max() > 1 else img * 255
                rgb = np.clip(x, 0, 255).astype(np.uint8)
            canvas[r * h : (r + 1) * h, col * w : (col + 1) * w] = rgb
        save_path = os.path.expanduser(save_path)
        Image.fromarray(canvas).save(save_path)


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


def command_current_state(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Current state: root pose/vel (env-local position) + foot positions.

    Layout: ``[x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, foot_pos...]``
    where ``foot_pos`` is ``num_feet`` positions in env-local world frame
    (world minus env origin), flattened.

    Foot positions match the success criterion (see
    :meth:`RelativeStateCommand.compute_state_error`), so HER relabeling
    ``target ← future current`` produces a self-consistent goal whose
    error matches the actual reward signal.

    Args:
        env: :class:`ManagerBasedRLEnv` instance.
        command_name: Name of the :class:`RelativeStateCommand` term.

    Returns:
        Tensor of shape ``[num_envs, 12 + 3 * num_feet]``.
    """
    cmd = env.command_manager.get_term(command_name)
    buf = cmd.cmd_buf[:, 2]
    env_origins = env.scene.terrain.env_origins
    pos_local = buf[:, :3] - env_origins
    foot_pos_local = (cmd._current_foot_pos_w - env_origins[:, None, :]).flatten(1)
    return torch.cat([pos_local, buf[:, 3:12], foot_pos_local], dim=-1)


def command_std(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Per-env success thresholds of the currently-bound task as a policy observation.

    Returns the active task's ``[pos_std, rot_std, lin_vel_std, ang_vel_std, foot_pos_std]``
    so the policy can see the threshold alongside the raw command delta. Lets the
    network distinguish per-task subtasks (e.g. ``terrain_pose_cmd`` vs.
    ``terrain_pose_cmd_foot``) without losing the absolute error magnitude.

    Args:
        env: :class:`ManagerBasedRLEnv` instance.
        command_name: Name of the :class:`RelativeStateCommand` term.

    Returns:
        Tensor of shape ``[num_envs, 5]``.
    """
    return env.command_manager.get_term(command_name).command_std


def command_target_state(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Target state: root pose/vel (env-local position) + foot positions.

    Layout matches :func:`command_current_state`. The foot portion is the
    commanded target foot positions in env-local world frame (kept fresh
    by :meth:`RelativeStateCommand._compute_target_foot_pos_w` at resample).

    Args:
        env: :class:`ManagerBasedRLEnv` instance.
        command_name: Name of the :class:`RelativeStateCommand` term.

    Returns:
        Tensor of shape ``[num_envs, 12 + 3 * num_feet]``.
    """
    cmd = env.command_manager.get_term(command_name)
    buf = cmd.cmd_buf[:, 0]
    env_origins = env.scene.terrain.env_origins
    pos_local = buf[:, :3] - env_origins
    foot_pos_local = (cmd._target_foot_pos_w - env_origins[:, None, :]).flatten(1)
    return torch.cat([pos_local, buf[:, 3:12], foot_pos_local], dim=-1)
