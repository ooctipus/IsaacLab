# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##
from isaaclab.utils.configclass import configclass

import isaaclab_assets.robots.anymal as anymal
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_tasks.utils import preset
import isaaclab.sim as sim_utils
from ... import position_env_cfg


ANYDRIVE_3_SIMPLE_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    effort_limit_sim=80.0,
    velocity_limit_sim=7.5,
    effort_limit=80.0,
    velocity_limit=7.5,
    stiffness={".*": 40.0},
    damping={".*": 5.0},
    armature={".*": 0.001},
)

@configclass
class AnymalCEnvMixin:
    def __post_init__(self: position_env_cfg.LocomotionPositionCommandEnvCfg):
        # Ensure parent classes run their setup first
        super().__post_init__()  # type: ignore
        self.scene.robot = anymal.ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore
        self.scene.robot.spawn.usd_path = (
            "https://uwlab-assets.s3.us-west-004.backblazeb2.com/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
        )
        self.scene.robot.spawn.joint_drive_props=preset(
            implicit_actuator=sim_utils.JointDrivePropertiesCfg(
                drive_type="force",
                stiffness=40.0,
                damping=5.0,
                max_force=120.0,
                max_joint_velocity=7.5,
            ),
            default=None,
            lstm_actuator=None,
        )

        self.scene.robot.actuators = {"legs": preset(implicit_actuator = ANYDRIVE_3_SIMPLE_ACTUATOR_CFG, default=anymal.ANYDRIVE_3_LSTM_ACTUATOR_CFG, lstm_actuator= anymal.ANYDRIVE_3_LSTM_ACTUATOR_CFG)}
        pipeline_cfg = self.commands.foot_sampled_commands.goal_point.task_table.pipeline_cfg
        pipeline_cfg.foot_body_names = ".*FOOT.*"
        pipeline_cfg.lateral_hip_joint_pattern = ".*HAA"
        pipeline_cfg.joint_regularize_targets = {".*HAA": 0.0, ".*F_KFE": -0.8, ".*H_KFE": 0.8}
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

@configclass
class AnymalCLocomotionPositionCommandEnvCfg(AnymalCEnvMixin, position_env_cfg.LocomotionPositionCommandEnvCfg):
    pass
