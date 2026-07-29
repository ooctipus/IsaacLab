# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##
import os

from isaaclab.utils.configclass import configclass

import isaaclab_assets.robots.unitree as unitree
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

from ... import position_env_cfg


@configclass
class B2EnvMixin:
    def __post_init__(self: position_env_cfg.LocomotionPositionCommandEnvCfg):
        # Ensure parent classes run their setup first
        super().__post_init__()  # type: ignore
        self.scene.robot = unitree.B2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore
        self.scene.robot.spawn.usd_path = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Robots", "Unitree", "B2", "b2.usd")
        self.rewards.undesired_contact.params["sensor_cfg"].body_names = "^(?!.*(?:foot)).*$"
        self.rewards.foot_touchdown.params["sensor_cfg"].body_names = ".*foot"
        self.rewards.foot_touchdown.params["asset_cfg"].body_names = ".*foot"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        self.viewer.body_name = "base_link"
        self.commands.foot_sampled_commands.goal_point.foot_body_names = [".*foot"]
        pipeline_cfg = self.commands.foot_sampled_commands.goal_point.task_table.pipeline_cfg
        pipeline_cfg.foot_body_names = ".*foot"
        pipeline_cfg.lateral_hip_joint_pattern = ".*hip_joint"
        pipeline_cfg.joint_regularize_targets = {".*hip_joint": 0.0, ".*_calf_joint": -0.873}


@configclass
class B2LocomotionPositionCommandEnvCfg(B2EnvMixin, position_env_cfg.LocomotionPositionCommandEnvCfg):
    pass
