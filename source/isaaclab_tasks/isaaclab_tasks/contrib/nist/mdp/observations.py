# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain-agnostic observation terms shared across terrain and factory tasks.

:func:`target_asset_pose_in_root_asset_frame` and
:func:`asset_link_velocity_in_root_asset_frame` read the pose and velocity of
one scene asset relative to another. Pure rigid-body math.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv

    from ..utils.pose_offset import Offset


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
    target_asset_offset: Offset | None = None,
) -> torch.Tensor:
    """Linear and angular velocity of a target keypoint expressed in the root frame.

    The optional target offset moves the linear velocity from the selected link origin
    to the offset point. It does not change angular velocity.

    Returns a ``[num_envs, 6]`` tensor containing linear velocity [m/s] followed by
    angular velocity [rad/s].
    """
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
    root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]

    target_body_idx = 0 if isinstance(target_asset_cfg.body_ids, slice) else target_asset_cfg.body_ids

    root_quat = root_asset.data.root_quat_w.torch
    lin_vel_w = target_asset.data.body_lin_vel_w.torch[:, target_body_idx].view(-1, 3)
    ang_vel_w = target_asset.data.body_ang_vel_w.torch[:, target_body_idx].view(-1, 3)
    if target_asset_offset is not None:
        target_quat_w = target_asset.data.body_link_quat_w.torch[:, target_body_idx].view(-1, 4)
        offset_pos_w = math_utils.quat_apply(
            target_quat_w, target_asset_offset.pos_t(lin_vel_w.device).expand(lin_vel_w.shape[0], -1)
        )
        lin_vel_w = lin_vel_w + torch.cross(ang_vel_w, offset_pos_w, dim=-1)

    lin_vel_b = math_utils.quat_apply_inverse(root_quat, lin_vel_w)
    ang_vel_b = math_utils.quat_apply_inverse(root_quat, ang_vel_w)

    return torch.cat([lin_vel_b, ang_vel_b], dim=1)
