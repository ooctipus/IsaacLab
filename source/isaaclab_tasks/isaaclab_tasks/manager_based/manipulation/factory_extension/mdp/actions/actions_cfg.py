# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from .attachment_action import RigidObjectCollectionAttachmentAction


@configclass
class RigidObjectCollectionAttachmentActionCfg(ActionTermCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = RigidObjectCollectionAttachmentAction

    asset_cfg: SceneEntityCfg = MISSING

    attach_to_asset_cfg: SceneEntityCfg = MISSING
