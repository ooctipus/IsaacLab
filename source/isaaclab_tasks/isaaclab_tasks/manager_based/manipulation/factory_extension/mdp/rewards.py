# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from .data_cfg import AlignmentDataCfg
    from .data import AlignmentMetric


def progress_reward_prod(env: ManagerBasedRLEnv, alignment: AlignmentDataCfg) -> torch.Tensor:
    alignment_data: AlignmentMetric.AlignmentData = alignment.get(env.data_manager)
    pos_diff_rew = (1 - torch.tanh(alignment_data.pos_error / alignment_data.pos_std)).prod(dim=1)
    rot_diff_rew = (1 - torch.tanh(alignment_data.rot_error / alignment_data.rot_std)).prod(dim=1)
    return pos_diff_rew * rot_diff_rew


def progress_reward_l2(env: ManagerBasedRLEnv, alignment: AlignmentDataCfg) -> torch.Tensor:
    alignment_data: AlignmentMetric.AlignmentData = alignment.get(env.data_manager)
    pos_norm = torch.linalg.norm(alignment_data.pos_error / alignment_data.pos_std, dim=-1)
    rot_norm = torch.linalg.norm(alignment_data.rot_error / alignment_data.rot_std, dim=-1)
    return (1 - torch.tanh(pos_norm)) * (1 - torch.tanh(rot_norm))


def success_reward(env: ManagerBasedRLEnv, alignment: AlignmentDataCfg) -> torch.Tensor:
    alignment_data: AlignmentMetric.AlignmentData = alignment.get(env.data_manager)
    return torch.where(alignment_data.pos_aligned & alignment_data.rot_aligned, 1.0, 0.0)


def still_on_success(env: ManagerBasedRLEnv, alignment: AlignmentDataCfg) -> torch.Tensor:
    alignment_data: AlignmentMetric.AlignmentData = alignment.resolve(env.data_manager)
    success_mask = alignment_data.pos_aligned & alignment_data.rot_aligned
    robot: Articulation = env.scene["robot"]
    return torch.where(success_mask, torch.norm(robot.data.body_link_vel_w, p=1, dim=(1, 2)), 0)
