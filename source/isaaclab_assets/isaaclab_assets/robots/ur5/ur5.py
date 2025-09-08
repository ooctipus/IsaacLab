# Copyright (c) 2024-2025, The Octi Lab Project Developers.
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

"""Configuration for the UR5 robots.

The following configurations are available:

* :obj:`UR5_CFG`: Ur5 robot
"""

from isaaclab.utils.assets import LOCAL_ASSET_PATH_DIR
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

UR5_DEFAULT_JOINT_POS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": -1.5708,
    "finger_joint": 0.0,
    "right_outer.*": 0.0,
    "left_outer.*": 0.0,
    "left_inner_finger_knuckle_joint": 0.0,
    "right_inner_finger_knuckle_joint": 0.0,
    "left_inner_finger_joint": -0.785398,
    "right_inner_finger_joint": 0.785398,
}

UR5_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Robots/UniversalRobots/Ur5RobotiqGripper/ur5e_robotiq_gripper_d415_mount_v2.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=36, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), joint_pos=UR5_DEFAULT_JOINT_POS),
    soft_joint_pos_limit_factor=1,
)

IMPLICIT_UR5 = UR5_ARTICULATION.copy()  # type: ignore
IMPLICIT_UR5.actuators = {
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
        stiffness={
            "shoulder_pan_joint": 4.63,
            "shoulder_lift_joint": 5.41,
            "elbow_joint": 8.06,
            "wrist_1_joint": 7.28,
            "wrist_2_joint": 8.04,
            "wrist_3_joint": 7.18,
        },
        damping={
            "shoulder_pan_joint": 8.84,
            "shoulder_lift_joint": 6.47,
            "elbow_joint": 9.46,
            "wrist_1_joint": 2.80,
            "wrist_2_joint": 2.41,
            "wrist_3_joint": 1.90,
        },
        velocity_limit_sim=3.14,
        effort_limit_sim={
            "shoulder_pan_joint": 150.0,
            "shoulder_lift_joint": 150.0,
            "elbow_joint": 150.0,
            "wrist_1_joint": 28.0,
            "wrist_2_joint": 28.0,
            "wrist_3_joint": 28.0,
        },
    ),
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],
        stiffness=17,
        damping=5,
        # velocity_limit=2.27,
        effort_limit_sim=165,
    ),
    "inner_finger": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_joint"],
        stiffness=0.2,
        damping=0.02,
        # velocity_limit=5.3,
        effort_limit_sim=0.5,
    ),
}
