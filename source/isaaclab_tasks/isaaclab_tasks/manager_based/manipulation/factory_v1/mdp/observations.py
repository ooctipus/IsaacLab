# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from .util import get_reset_state

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from ..assembly_keypoints import Offset

def get_state(env: ManagerBasedRLEnv, reset_assets: list[str]):
    return get_reset_state(env, slice(None), reset_assets, is_relative=True)


def time_left(env: ManagerBasedRLEnv):
    time_left_frac = 1 - env.episode_length_buf / env.max_episode_length
    return time_left_frac.view(env.num_envs, -1)


def target_asset_pose_in_root_asset_frame(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_asset_offset: Offset | None = None,
    root_asset_offset: Offset | None = None,
):
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
    root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]

    target_body_idx = 0 if isinstance(target_asset_cfg.body_ids, slice) else target_asset_cfg.body_ids
    root_body_idx = 0 if isinstance(root_asset_cfg.body_ids, slice) else root_asset_cfg.body_ids

    target_pos = wp.to_torch(target_asset.data.body_link_pos_w)[:, target_body_idx].view(-1, 3)
    target_quat = wp.to_torch(target_asset.data.body_link_quat_w)[:, target_body_idx].view(-1, 4)
    root_pos = wp.to_torch(root_asset.data.body_link_pos_w)[:, root_body_idx].view(-1, 3)
    root_quat = wp.to_torch(root_asset.data.body_link_quat_w)[:, root_body_idx].view(-1, 4)

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
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
    root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]

    target_body_idx = 0 if isinstance(target_asset_cfg.body_ids, slice) else target_asset_cfg.body_ids

    root_quat = wp.to_torch(root_asset.data.root_quat_w)
    lin_vel_w = wp.to_torch(target_asset.data.body_lin_vel_w)[:, target_body_idx].view(-1, 3)
    ang_vel_w = wp.to_torch(target_asset.data.body_ang_vel_w)[:, target_body_idx].view(-1, 3)

    lin_vel_b = math_utils.quat_apply_inverse(root_quat, lin_vel_w)
    ang_vel_b = math_utils.quat_apply_inverse(root_quat, ang_vel_w)

    return torch.cat([lin_vel_b, ang_vel_b], dim=1)
