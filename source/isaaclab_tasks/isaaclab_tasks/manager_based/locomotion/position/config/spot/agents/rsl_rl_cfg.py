# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from ...rsl_rl_cfg import PositionLocomotionPPORunnerCfg


@configclass
class SpotPositionLocomotionPPORunnerCfg(PositionLocomotionPPORunnerCfg):
    experiment_name: str = "spot_position_command"
