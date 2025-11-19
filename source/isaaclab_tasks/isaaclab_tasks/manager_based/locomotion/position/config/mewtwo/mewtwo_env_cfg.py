# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##
import isaaclab_assets.robots.mewtwo as mewtwo
from isaaclab.utils import configclass
from ...mdp import negative_y_exploration_reward
from ... import mdp as mdp
from ... import position_env_cfg


@configclass
class MewtwoActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["^(?!.*(Toe|Heel).* ).*$"], scale=0.2, use_default_offset=True
    )
    arm_pos = mdp.DefaultJointPositionStaticActionCfg(
        asset_name="robot", joint_names=[".*(Toe|Heel).*"], scale=1, use_default_offset=True
    )


@configclass
class MewtwoEnvMixin:
    actions: MewtwoActionsCfg = MewtwoActionsCfg()

    def __post_init__(self: position_env_cfg.LocomotionPositionCommandEnvCfg):
        # Ensure parent classes run their setup first
        super().__post_init__()  # type: ignore
        self.scene.robot = mewtwo.MEWTWO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Pelvis"
        # self.events.add_base_mass.params["asset_cfg"].body_names = "Torso"
        del self.events.add_base_mass
        self.viewer.body_name = "Torso"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "^(?!.*(?:Toe|Thumb|Index|Pinky|Coccyx.*)).*$"
        self.terminations.success.params["robot_cfg"].joint_names = "^(?!.*(?:Toe*|Thumb*|Index*|Pinky*|Coccyx.*)).*$"
        self.terminations.success.params["robot_cfg"].body_names = "^(?!.*(?:Toe*|Thumb*|Index*|Pinky*|Coccyx.*)).*$"
        self.rewards.explore.func = negative_y_exploration_reward
        if hasattr(self.terminations, "log_gait"):
            self.terminations.log_gait.params["async_pairs"] = (("RightToe", "LeftToe"),)
            self.terminations.log_gait.params["sync_pairs"] = ()


@configclass
class MewtwoLocomotionPositionCommandEnvCfg(MewtwoEnvMixin, position_env_cfg.LocomotionPositionCommandEnvCfg):
    pass
