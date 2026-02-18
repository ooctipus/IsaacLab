# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Kuka-lbr-iiwa arm robots and Allegro Hand.

The following configurations are available:

* :obj:`KUKA_ALLEGRO_CFG`: Kuka Allegro with implicit actuator model.

Reference:

* https://www.kuka.com/en-us/products/robotics-systems/industrial-robots/lbr-iiwa
* https://www.wonikrobotics.com/robot-hand

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_LOCAL_DIR

##
# Shared Kuka Arm Actuator Configuration
##

KUKA_ARM_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=["iiwa7_joint_(1|2|3|4|5|6|7)"],
    effort_limit_sim=300.0,
    stiffness={
        "iiwa7_joint_(1|2|3|4)": 300.0,
        "iiwa7_joint_5": 100.0,
        "iiwa7_joint_6": 50.0,
        "iiwa7_joint_7": 25.0,
    },
    damping={
        "iiwa7_joint_(1|2|3|4)": 45.0,
        "iiwa7_joint_5": 20.0,
        "iiwa7_joint_6": 15.0,
        "iiwa7_joint_7": 15.0,
    },
    friction=1.0,
)

##
# Shared Spawn Configuration (for local USD with variants)
##

KUKA_ARTICULATION = sim_utils.UsdFileCfg(
    usd_path=f"{ISAACLAB_LOCAL_DIR}/Robots/KukaAllegro/kuka.usd",
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,
        retain_accelerations=True,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1000.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=32,
        solver_velocity_iteration_count=1,
        sleep_threshold=0.005,
        stabilization_threshold=0.0005,
    ),
    joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
)

##
# Robot Configurations
##

KUKA_ALLEGRO_CFG = ArticulationCfg(
    spawn=KUKA_ARTICULATION.replace(variants={"EE": "Allegro"}),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "iiwa7_joint_(1|2|7)": 0.0,
            "iiwa7_joint_3": 0.7854,
            "iiwa7_joint_4": 1.5708,
            "iiwa7_joint_(5|6)": -1.5708,
            "(index|middle|ring)_joint_0": 0.0,
            "(index|middle|ring)_joint_1": 0.3,
            "(index|middle|ring)_joint_2": 0.3,
            "(index|middle|ring)_joint_3": 0.3,
            "thumb_joint_0": 1.5,
            "thumb_joint_1": 0.60147215,
            "thumb_joint_2": 0.33795027,
            "thumb_joint_3": 0.60845138,
        },
    ),
    actuators={
        "kuka_arm": KUKA_ARM_ACTUATOR_CFG,
        "allegro_fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                "index_joint_(0|1|2|3)",
                "middle_joint_(0|1|2|3)",
                "ring_joint_(0|1|2|3)",
                "thumb_joint_(0|1|2|3)",
            ],
            effort_limit_sim=0.5,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


KUKA_SHARPA_CFG = ArticulationCfg(
    spawn=KUKA_ARTICULATION.replace(variants={"EE": "Sharpa"}),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "iiwa7_joint_(1|2|7)": 0.0,
            "iiwa7_joint_3": 0.7854,
            "iiwa7_joint_4": 1.5708,
            "iiwa7_joint_(5|6)": -1.5708,
            ".*(index|middle|ring|pinky|thumb).*": 0.0,
        },
    ),
    actuators={
        "kuka_arm": KUKA_ARM_ACTUATOR_CFG,
        "sharpa_fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*(index|middle|ring|pinky|thumb).*"],
            stiffness=None,
            damping=None,
            friction=0.01,
            armature=0.001,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


KUKA_INSPIRE_CFG = ArticulationCfg(
    spawn=KUKA_ARTICULATION.replace(variants={"EE": "Inspire"}),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "iiwa7_joint_(1|2|7)": 0.0,
            "iiwa7_joint_3": 0.7854,
            "iiwa7_joint_4": 1.5708,
            "iiwa7_joint_(5|6)": -1.5708,
            ".*(index|middle|ring|little|thumb).*": 0.0,
        },
    ),
    actuators={
        "kuka_arm": KUKA_ARM_ACTUATOR_CFG,
        "inspire_fingers": 
            ImplicitActuatorCfg(
                joint_names_expr=[".*(index|middle|ring|little|thumb).*"],
                stiffness=None,
                damping=None,
                friction=0.01,
                armature=0.001,
            ),
    },
    soft_joint_pos_limit_factor=1.0,
)
