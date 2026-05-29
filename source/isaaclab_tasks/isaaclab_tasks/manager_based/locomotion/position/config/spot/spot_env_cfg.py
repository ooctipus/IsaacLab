# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.spot import SPOT_CFG

from ... import position_env_cfg


@configclass
class SpotEnvMixin:
    def __post_init__(self: position_env_cfg.LocomotionPositionCommandEnvCfg):
        # Ensure parent classes run their setup first
        super().__post_init__()
        # overwrite as spot's body names for sensors
        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/body"

        # overwrite as spot's body names for events
        self.events.add_base_mass.params["asset_cfg"].body_names = "body"
        self.viewer.body_name = "body"
        self.commands.foot_sampled_commands.goal_point.foot_body_names = [".*_foot"]
        pipeline_cfg = self.commands.foot_sampled_commands.goal_point.task_table.pipeline_cfg
        pipeline_cfg.foot_body_names = ".*_foot"
        pipeline_cfg.lateral_hip_joint_pattern = ".*hip_x"
        pipeline_cfg.joint_regularize_targets = {".*hip_x": 0.0, ".*_kn": -1.5}
        if hasattr(self.terminations, "log_gait"):
            self.terminations.log_gait.params["async_pairs"] = (
                ("fl_foot", "fr_foot"),
                ("hr_foot", "hl_foot"),
                ("fl_foot", "hl_foot"),
                ("fr_foot", "hr_foot"),
            )
            self.terminations.log_gait.params["sync_pairs"] = (("fl_foot", "hr_foot"), ("fr_foot", "hl_foot"))


@configclass
class SpotLocomotionPositionCommandEnvCfg(SpotEnvMixin, position_env_cfg.LocomotionPositionCommandEnvCfg):
    pass
