# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import action_cfg


class PCAAction(ActionTerm):

    cfg: action_cfg.PCAHandActionCfg

    def __init__(self, cfg: action_cfg.PCAHandActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        cfg.arm_joints_cfg.resolve(env.scene)
        cfg.pca_joints_cfg.resolve(env.scene)
        self.num_arm_joints = len(cfg.arm_joints_cfg.joint_ids)
        self.num_hand_joints = len(cfg.pca_joints_cfg.joint_ids)
        self.pca_matrix = torch.tensor(cfg.pca_matrix, device=self.device)
        self.arm_joint_actions = torch.zeros((self.num_envs, self.num_arm_joints), device=self.device)
        self._raw_actions = torch.zeros(
            (self.num_envs, self.pca_matrix.shape[0] + self.num_arm_joints), device=self.device
        )
        self._processed_actions = torch.zeros(
            (self.num_envs, self.num_arm_joints + self.num_hand_joints), device=self.device
        )
        self.hand_pca_high_limits = torch.tensor(cfg.hand_pca_maxs, device=self.device)
        self.hand_pca_low_limits = torch.tensor(cfg.hand_pca_mins, device=self.device)

    @property
    def action_dim(self) -> int:
        return self.pca_matrix.shape[0] + self.num_arm_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # relative
        self._raw_actions = self._raw_actions.clamp(min=-1, max=1)
        self._processed_actions[:, self.cfg.arm_joints_cfg.joint_ids] = (
            actions[:, : self.num_arm_joints] * 0.3 + self._asset.data.joint_pos[:, self.cfg.arm_joints_cfg.joint_ids]
        )
        limit_range = self.hand_pca_high_limits - self.hand_pca_low_limits
        hand_pca_target = (self._raw_actions[:, self.num_arm_joints :] + 1) / 2 * limit_range + self.hand_pca_low_limits
        self._processed_actions[:, self.cfg.pca_joints_cfg.joint_ids] = hand_pca_target @ self.pca_matrix

    def apply_actions(self):
        self._asset.set_joint_position_target(self.processed_actions)
