# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.assets import RigidObjectCollection, Articulation
    from . import actions_cfg


class RigidObjectCollectionAttachmentAction(ActionTerm):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.RigidObjectCollectionAttachmentActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.RigidObjectCollectionAttachmentActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        cfg.attach_to_asset_cfg.resolve(env.scene)
        cfg.asset_cfg.resolve(env.scene)
        self.asset: RigidObjectCollection = env.scene[cfg.asset_cfg.name]
        self.attach_to_asset: Articulation = env.scene[cfg.attach_to_asset_cfg.name]

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.empty((0,), device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        return torch.empty((0,), device=self.device)

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        target_pose = self.attach_to_asset.data.body_link_state_w[:, self.cfg.attach_to_asset_cfg.body_ids, :7]
        self.asset.write_object_pose_to_sim(target_pose, object_ids=self.cfg.asset_cfg.object_collection_ids)
