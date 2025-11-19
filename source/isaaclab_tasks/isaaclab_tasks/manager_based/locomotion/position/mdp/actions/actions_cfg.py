# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs.mdp import JointActionCfg, ActionTerm
from isaaclab.utils import configclass

from .actions import DefaultJointPositionStaticAction


@configclass
class DefaultJointPositionStaticActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = DefaultJointPositionStaticAction

    use_default_offset: bool = True
