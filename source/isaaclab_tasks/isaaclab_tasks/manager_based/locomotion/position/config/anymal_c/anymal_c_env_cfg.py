# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import preset

import isaaclab_assets.robots.anymal as anymal

from ... import position_env_cfg

ANYDRIVE_3_LSTM_ONNX_PATH = str(Path(__file__).resolve().parents[2] / "assets" / "anydrive_3_lstm.onnx")
ANYDRIVE_3_LSTM_ACTUATOR_CFG = anymal.ANYDRIVE_3_LSTM_ACTUATOR_CFG.replace(
    network_file=preset(
        default=anymal.ANYDRIVE_3_LSTM_ACTUATOR_CFG.network_file,
        newton_mjwarp=ANYDRIVE_3_LSTM_ONNX_PATH,
    )
)


ANYDRIVE_3_SIMPLE_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    effort_limit_sim=80.0,
    velocity_limit_sim=7.5,
    effort_limit=80.0,
    velocity_limit=7.5,
    stiffness={".*": 40.0},
    damping={".*": 5.0},
    # Reflected drivetrain inertia of the ANYdrive (N^2 * J_rotor, gear + rotor). The original 0.001
    # was ~0, which let the policy fling the legs ballistically for free. 0.15 matches the physics
    # estimate for a harmonic-drive quad and ETH's PACE sim2real fits for geared joints (arXiv
    # 2509.06342, geared hips ~0.14 kg*m^2). Requires a retrain to take effect.
    armature={".*": 0.15},
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
        self.scene.robot.spawn.joint_drive_props = preset(
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

        self.scene.robot.actuators = {
            "legs": preset(
                implicit_actuator=ANYDRIVE_3_SIMPLE_ACTUATOR_CFG,
                default=ANYDRIVE_3_LSTM_ACTUATOR_CFG,
                lstm_actuator=ANYDRIVE_3_LSTM_ACTUATOR_CFG,
            )
        }
        self.commands.foot_sampled_commands.goal_point.foot_body_names = [".*FOOT.*"]
        pipeline_cfg = self.commands.foot_sampled_commands.goal_point.task_table.pipeline_cfg
        pipeline_cfg.foot_body_names = ".*FOOT.*"
        pipeline_cfg.lateral_hip_joint_pattern = ".*HAA"
        pipeline_cfg.joint_regularize_targets = {".*HAA": 0.0, ".*F_KFE": -0.8, ".*H_KFE": 0.8}
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class AnymalCLocomotionPositionCommandEnvCfg(AnymalCEnvMixin, position_env_cfg.LocomotionPositionCommandEnvCfg):
    pass
