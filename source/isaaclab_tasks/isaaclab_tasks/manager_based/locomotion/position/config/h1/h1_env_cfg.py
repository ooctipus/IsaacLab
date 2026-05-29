# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

##
# Pre-defined configs
##
from isaaclab_assets import H1_CFG  # isort: skip

from ... import position_env_cfg


@configclass
class H1EnvMixin:
    def __post_init__(self: position_env_cfg.LocomotionPositionCommandEnvCfg):
        # Ensure parent classes run their setup first
        super().__post_init__()
        # overwrite as H1's body names for sensors
        self.scene.robot = H1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.usd_path = "https://uwlab-assets.s3.us-west-004.backblazeb2.com/Robots/Unitree/H1/h1.usd"
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True

        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        # overwrite as H1's body names for events
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "^(?!.*ankle_link).*$"
        self.viewer.body_name = "torso_link"
        self.commands.foot_sampled_commands.goal_point.foot_body_names = [".*ankle_link"]
        pipeline_cfg = self.commands.foot_sampled_commands.goal_point.task_table.pipeline_cfg
        pipeline_cfg.foot_body_names = ".*ankle_link"
        pipeline_cfg.lateral_hip_joint_pattern = ".*_hip_roll"
        pipeline_cfg.joint_regularize_targets = {
            ".*_hip_yaw": 0.0,
            ".*_hip_roll": 0.0,
            "torso": 0.0,
            ".*_shoulder_pitch": 0.28,
            ".*_shoulder_roll": 0.0,
            ".*_shoulder_yaw": 0.0,
            ".*_elbow": 0.52,
        }
        if hasattr(self.terminations, "log_gait"):
            self.terminations.log_gait.params["async_pairs"] = (("left_ankle_link", "right_ankle_link"),)
            self.terminations.log_gait.params["sync_pairs"] = ()


@configclass
class H1LocomotionPositionCommandEnvCfg(H1EnvMixin, position_env_cfg.LocomotionPositionCommandEnvCfg):
    pass
