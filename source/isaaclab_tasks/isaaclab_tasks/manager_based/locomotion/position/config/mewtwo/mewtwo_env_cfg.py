# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##
import isaaclab_assets.robots.mewtwo as mewtwo
from isaaclab.utils import configclass

from ... import position_env_cfg


@configclass
class MewtwoEnvMixin:
    def __post_init__(self: position_env_cfg.LocomotionPositionCommandEnvCfg):
        # Ensure parent classes run their setup first
        super().__post_init__()  # type: ignore
        self.scene.robot = mewtwo.MEWTWO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Pelvis"
        # self.events.add_base_mass.params["asset_cfg"].body_names = "Torso"
        del self.events.add_base_mass
        self.viewer.body_name = "Torso"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "^(?!.*(?:Toe|Thumb|Index|Pinky)).*$"
        self.terminations.success.params["robot_cfg"].joint_names = "^(?!.*(?:Toe*|Thumb*|Index*|Pinky*|Coccyx.*)).*$"

        self.terminations.log_gait.params["async_pairs"] = (("RightToe", "LeftToe"),)
        self.terminations.log_gait.params["sync_pairs"] = ()


@configclass
class MewtwoLocomotionPositionCommandEnvCfg(MewtwoEnvMixin, position_env_cfg.LocomotionPositionCommandEnvCfg):
    pass
