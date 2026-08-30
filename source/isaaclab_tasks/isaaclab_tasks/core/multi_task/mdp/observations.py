# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain-agnostic observation terms shared across terrain and factory tasks.

Groups:

- **Vision obs** — ``vision_obs`` converts camera-style sensors and grid
  raycasters to CNN-shaped image tensors.
- **Multi-task command obs** — ``command_progress``, ``command_reach``,
  ``command_track``, ``command_active``. Wrap properties of any
  :class:`~.commands.MultiTaskCommand` so the policy can read its
  current task state. Domain-agnostic because the underlying command term
  is.
- **Frame-relative obs** — ``target_asset_pose_in_root_asset_frame``,
  ``asset_link_velocity_in_root_asset_frame``. Read pose/velocity of one
  scene asset relative to another. Pure rigid-body math.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from ..geom.pose_offset import Offset


# ---------------------------------------------------------------------------
# Vision observation.
# ---------------------------------------------------------------------------


class vision_obs(ManagerTermBase):
    """Unified 2D vision observation for camera-style sensors and grid raycasters.

    Args:
        sensor_cfg: Scene sensor to read.
        normalize: If ``True``, return normalized ``(B, C, H, W)`` images.
            If ``False``, return raw ``(B, H, W, C)`` images.
        offset: Height offset [m] subtracted from grid raycaster heights.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.sensor = env.scene.sensors[sensor_cfg.name]

        # Camera-style sensors expose ``data.output[type]``; check them first because
        # ``RayCasterCamera`` inherits from ``RayCaster`` and would otherwise be routed
        # into the grid path.
        from isaaclab.sensors import RayCaster, RayCasterCamera, TiledCamera

        from isaaclab_tasks.core.multi_task.sensors import FastTerrainScanner

        if isinstance(self.sensor, (TiledCamera, RayCasterCamera)):
            self._sensor_type = self.sensor.cfg.data_types[0]
            self._fetch = self._fetch_camera
            self._norm = (
                self._depth_norm if self._sensor_type in ("distance_to_image_plane", "depth") else self._rgb_norm
            )
        elif isinstance(self.sensor, (RayCaster, FastTerrainScanner)):
            pattern_cfg = self.sensor.cfg.pattern_cfg
            self._nx = round(pattern_cfg.size[0] / pattern_cfg.resolution) + 1
            self._ny = round(pattern_cfg.size[1] / pattern_cfg.resolution) + 1
            self._ordering = pattern_cfg.ordering
            self._fetch = self._fetch_raycaster
            self._norm = self._depth_norm
            if isinstance(self.sensor, FastTerrainScanner):
                self._bind_fast_terrain_scanner(env)
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

    def _bind_fast_terrain_scanner(self, env: ManagerBasedRLEnv) -> None:
        sensor_body_path = self.sensor.body_prim_path
        articulation = None
        articulation_path_len = -1
        for candidate in env.scene.articulations.values():
            prim_path = candidate.cfg.prim_path.rstrip("/")
            if sensor_body_path == prim_path or sensor_body_path.startswith(f"{prim_path}/"):
                if len(prim_path) > articulation_path_len:
                    articulation = candidate
                    articulation_path_len = len(prim_path)
        if articulation is None:
            raise RuntimeError(
                "FastTerrainScanner vision_obs requires the scanner prim path to be under an articulation prim path; "
                f"got scanner body prim path '{sensor_body_path}'."
            )
        body_name = sensor_body_path.rsplit("/", 1)[-1]
        self.sensor.bind_articulation(articulation, body_name)

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
        """Save a turbo-colormapped collage of the raw sensor tiles to disk."""
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


# ---------------------------------------------------------------------------
# Episode-progress observation.
# ---------------------------------------------------------------------------


def time_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fraction of the episode remaining ∈ ``[0, 1]``, shape ``[num_envs, 1]``.

    Bounded and episode-length-invariant — preferred over absolute-seconds
    forms because the policy's obs distribution stays stable across
    curriculum changes to ``episode_length_s``.
    """
    time_left_frac = 1 - env.episode_length_buf / env.max_episode_length
    return time_left_frac.view(env.num_envs, -1)


# ---------------------------------------------------------------------------
# Multi-task command observation accessors.
# ---------------------------------------------------------------------------


def command_progress(env, command_name: str = "goal_point"):
    """Scalar per-env task progress ∈ [0, 1], shape ``[num_envs, 1]``.

    Mean of the env's active-subtask activations — a task-normalized "how close am I?"
    signal with no reward-kernel parameters baked in.
    """
    return env.command_manager.get_term(command_name).progress.unsqueeze(-1)


def command_reach(env, command_name: str = "goal_point"):
    """Canonical state delta for instant ("reach") subtasks.

    Shape ``[num_envs, reach_canonical_width]``. Populated only by instant subtasks;
    tracking subtasks write to :func:`command_track`. Keeps the two semantic
    categories in separate obs tensors so the policy reads them positionally.
    """
    return env.command_manager.get_term(command_name).command_reach


def command_track(env, command_name: str = "goal_point"):
    """Canonical state delta for tracking subtasks.

    Shape ``[num_envs, track_canonical_width]``. Populated only by tracking
    subtasks. Same positional encoding as :func:`command_reach` but disjoint
    channels, so same-kernel reach + track subtasks coexist without aliasing.
    """
    return env.command_manager.get_term(command_name).command_track


def command_active(env, command_name: str = "goal_point"):
    """Per-channel active mask paired with :func:`command_reach` + :func:`command_track`.

    Shape ``[num_envs, reach_canonical_width + track_canonical_width]``. The
    layout mirrors ``cat([command_reach, command_track], dim=-1)`` slot-for-
    slot: column ``i`` of this mask gates column ``i`` of the concatenated
    delta. ``1.0`` iff the channel is populated by a live subtask of the
    env's current task; ``0.0`` otherwise (inactive channel, or joint-kernel
    subtask with no canonical projection).
    """
    return env.command_manager.get_term(command_name).command_active


# ---------------------------------------------------------------------------
# Frame-relative pose / velocity observations.
# ---------------------------------------------------------------------------


def target_asset_pose_in_root_asset_frame(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_asset_offset: Offset | None = None,
    root_asset_offset: Offset | None = None,
):
    """Pose of ``target_asset`` expressed in the root frame of ``root_asset``.

    Optional ``Offset`` cfgs let callers compose static frame offsets onto
    either side (e.g. observe an end-effector grasp point relative to a
    fixed-asset tip).

    Returns a ``[num_envs, 7]`` tensor — translation (3) + quaternion xyzw (4).
    """
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
    root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]

    target_body_idx = 0 if isinstance(target_asset_cfg.body_ids, slice) else target_asset_cfg.body_ids
    root_body_idx = 0 if isinstance(root_asset_cfg.body_ids, slice) else root_asset_cfg.body_ids

    target_pos = target_asset.data.body_link_pos_w.torch[:, target_body_idx].view(-1, 3)
    target_quat = target_asset.data.body_link_quat_w.torch[:, target_body_idx].view(-1, 4)
    root_pos = root_asset.data.body_link_pos_w.torch[:, root_body_idx].view(-1, 3)
    root_quat = root_asset.data.body_link_quat_w.torch[:, root_body_idx].view(-1, 4)

    if root_asset_offset is not None:
        root_pos, root_quat = root_asset_offset.combine(root_pos, root_quat)
    if target_asset_offset is not None:
        target_pos, target_quat = target_asset_offset.combine(target_pos, target_quat)

    target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(root_pos, root_quat, target_pos, target_quat)
    return torch.cat([target_pos_b, target_quat_b], dim=1)


def asset_link_velocity_in_root_asset_frame(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Linear + angular velocity of ``target_asset``'s link, expressed in the root frame.

    Returns a ``[num_envs, 6]`` tensor — linear (3) + angular (3) in body frame.
    """
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
    root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]

    target_body_idx = 0 if isinstance(target_asset_cfg.body_ids, slice) else target_asset_cfg.body_ids

    root_quat = root_asset.data.root_quat_w.torch
    lin_vel_w = target_asset.data.body_lin_vel_w.torch[:, target_body_idx].view(-1, 3)
    ang_vel_w = target_asset.data.body_ang_vel_w.torch[:, target_body_idx].view(-1, 3)

    lin_vel_b = math_utils.quat_apply_inverse(root_quat, lin_vel_w)
    ang_vel_b = math_utils.quat_apply_inverse(root_quat, ang_vel_w)

    return torch.cat([lin_vel_b, ang_vel_b], dim=1)
