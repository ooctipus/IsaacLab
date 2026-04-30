# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

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
        if isinstance(self.sensor, (TiledCamera, RayCasterCamera)):
            self._sensor_type = self.sensor.cfg.data_types[0]
            self._fetch = self._fetch_camera
            self._norm = self._depth_norm if self._sensor_type in ("distance_to_image_plane", "depth") else self._rgb_norm
        elif isinstance(self.sensor, RayCaster):
            pattern_cfg = self.sensor.cfg.pattern_cfg
            self._nx = round(pattern_cfg.size[0] / pattern_cfg.resolution) + 1
            self._ny = round(pattern_cfg.size[1] / pattern_cfg.resolution) + 1
            self._ordering = pattern_cfg.ordering
            self._fetch = self._fetch_raycaster
            self._norm = self._depth_norm
        else:
            raise TypeError(
                f"vision_obs supports TiledCamera, RayCasterCamera, or RayCaster; got "
                f"{type(self.sensor).__name__}"
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


def height_scan_2d(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan reshaped to ``(num_envs, 1, H, W)`` for 2D CNN encoders.

    Same height computation as :func:`isaaclab.envs.mdp.height_scan` (sensor z
    minus hit z minus ``offset`` [m]), but the flat ray output is reshaped onto
    its underlying grid using the sensor's
    :class:`~isaaclab.sensors.ray_caster.patterns.GridPatternCfg` size /
    resolution / ordering. Required by :class:`rsl_rl.models.CNNModel`, which
    rejects flat 1D inputs.

    Args:
        env: The environment.
        sensor_cfg: Scene-entity cfg for a :class:`RayCaster` whose pattern is a
            :class:`GridPatternCfg`.
        offset: Subtracted from each height value [m]. Defaults to ``0.5``.

    Returns:
        Tensor of shape ``[num_envs, 1, H, W]`` where ``H × W = num_rays``,
        ordered to match the sensor's ``GridPatternCfg.ordering``.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    pattern_cfg = sensor.cfg.pattern_cfg
    nx = round(pattern_cfg.size[0] / pattern_cfg.resolution) + 1
    ny = round(pattern_cfg.size[1] / pattern_cfg.resolution) + 1
    flat = sensor.data.pos_w.torch[:, 2].unsqueeze(1) - sensor.data.ray_hits_w.torch[..., 2] - offset
    # ordering="xy" -> inner loop over x -> rows are constant-y -> reshape (Ny, Nx).
    # ordering="yx" -> inner loop over y -> rows are constant-x -> reshape (Nx, Ny).
    if pattern_cfg.ordering == "xy":
        return flat.view(env.num_envs, 1, ny, nx)
    return flat.view(env.num_envs, 1, nx, ny)


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
    """Current state: root pose/vel (12D, env-local position) + joint positions.

    Layout: ``[x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, joint_pos...]``.

    Including joint positions ensures CRL (via HER relabeling) learns to
    match the full robot configuration, not just the root pose.

    Args:
        env: :class:`ManagerBasedRLEnv` instance.
        command_name: Name of the :class:`RelativeStateCommand` term.

    Returns:
        Tensor of shape ``[num_envs, 12 + num_joints]``.
    """
    import warp as wp

    cmd = env.command_manager.get_term(command_name)
    buf = cmd.cmd_buf[:, 2]
    pos_local = buf[:, :3] - env.scene.terrain.env_origins
    joint_pos = wp.to_torch(cmd.robot.data.joint_pos)
    return torch.cat([pos_local, buf[:, 3:12], joint_pos], dim=-1)


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
    """Target state: root pose/vel (12D, env-local position) + joint positions.

    Layout matches :func:`command_current_state`. The joint portion uses
    the robot's current joints as placeholder — HER replaces the entire
    target with a future ``current_state``, so the placeholder values
    are never used for training.

    Args:
        env: :class:`ManagerBasedRLEnv` instance.
        command_name: Name of the :class:`RelativeStateCommand` term.

    Returns:
        Tensor of shape ``[num_envs, 12 + num_joints]``.
    """
    import warp as wp

    cmd = env.command_manager.get_term(command_name)
    buf = cmd.cmd_buf[:, 0]
    pos_local = buf[:, :3] - env.scene.terrain.env_origins
    joint_pos = wp.to_torch(cmd.robot.data.joint_pos)
    return torch.cat([pos_local, buf[:, 3:12], joint_pos], dim=-1)
