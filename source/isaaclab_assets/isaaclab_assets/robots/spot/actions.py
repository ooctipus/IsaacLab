# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg


SPOT_JOINT_POSITION: JointPositionActionCfg = JointPositionActionCfg(
    asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True
)

ARM_DEFAULT_JOINT_POSITION: JointPositionActionCfg = JointPositionActionCfg(
    asset_name="arm", joint_names=[".*"], scale=0.2, use_default_offset=True
)
