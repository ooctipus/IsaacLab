# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Mewtwo robots."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration - Actuators.
##


MEWTWO_LEG_IMPLICIT_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[r".*(Torso|Waist|Coccyx_col1|Thigh|Calf).*"],
    effort_limit_sim=300,
    stiffness={
        r".*(Torso|Waist|Calf|Heel).*": 150,
        r".*(Thigh|Coccyx_col1).*": 200.0,
    },
    damping={r".*(Torso|Waist|Thigh|Coccyx_col1|Calf).*": 5.0},
)

MEWTWO_ARM_IMPLICIT_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[r".*(Brachium|Forearm).*"],
    effort_limit_sim=120,
    stiffness={r".*(Brachium|Forearm).*": 40},
    damping={r".*(Brachium|Forearm|Palm).*": 5.0},
)


MEWTWO_FEET_IMPLICIT_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[r".*(Toe|Heel).*"],
    effort_limit_sim=50,
    stiffness={r".*(Toe|Heel).*": 20},
    damping={r".*(Toe|Heel).*": 4.0},
)


MEWTWO_HAND_IMPLICIT_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[r".*(Palm|Distal|Proximal).*"],
    effort_limit_sim=30,
    stiffness={r".*(Palm|Distal|Proximal).*": 40.0},
    damping={r".*(Palm|Distal|Proximal).*": 1.0},
)

MEWTWO_HEAD_IMPLICIT_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[r".*(Neck|Head).*"],
    effort_limit_sim=100,
    stiffness={r".*(Neck|Head).*": 40.0},
    damping={r".*(Neck|Head).*": 5.0},
)

MEWTWO_TAIL_IMPLICIT_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[r".*Coccyx_(col2|col3|col4|col5).*"],
    effort_limit_sim=100,
    stiffness={r".*Coccyx_(col2|col3|col4|col5).*": 40.0},
    damping={r".*Coccyx_(col2|col3|col4|col5).*": 5.0},
)


##
# Configuration - Articulation.
##

MEWTWO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Robots", "Mewtwo", "mewtwo.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        scale=(0.5, 0.5, 0.5)
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            # 6-DOF / multi-DOF with explicit values
            "WaistJoint:0": 0.0,
            "WaistJoint:1": 0.0,
            "TorsoJoint:0": -0.22689280275926285,  # -13 deg
            "TorsoJoint:1": 0.0,
            "NeckJoint:0": 0.0,
            "NeckJoint:1": 0.0,
            # "Coccyx_col1Joint:0": 0.0,
            # "Coccyx_col1Joint:1": 0.2984513,
            # "Coccyx_col2Joint:0": 0.436332,
            # "Coccyx_col2Joint:1": -0.3054326,
            # "Coccyx_col3Joint:0": 0.3926991,  
            # "Coccyx_col3Joint:1": -0.4066617,
            # "Coccyx_col4Joint:0": 0.2530727415391778,
            # "Coccyx_col4Joint:1": -0.349066,
            # "Coccyx_col5Joint:0": 0.19198621771937624,
            # "Coccyx_col5Joint:1": 0.0,
            "HeadJoint": 0.0,
            r"(Left|Right)(Palm|Thigh|Calf|Heel|Toe|Forearm|Brachium).*": 0.0,
            r"(Left|Right)(Distal|Proximal).*": 0.0,
        },
    ),
    actuators={
        "head": MEWTWO_HEAD_IMPLICIT_ACTUATOR_CFG,
        "leg": MEWTWO_LEG_IMPLICIT_ACTUATOR_CFG,
        "feet": MEWTWO_FEET_IMPLICIT_ACTUATOR_CFG,
        "arm": MEWTWO_ARM_IMPLICIT_ACTUATOR_CFG,
        "hand": MEWTWO_HAND_IMPLICIT_ACTUATOR_CFG,
        # "tail": MEWTWO_TAIL_IMPLICIT_ACTUATOR_CFG
    },
    soft_joint_pos_limit_factor=0.95,
)
