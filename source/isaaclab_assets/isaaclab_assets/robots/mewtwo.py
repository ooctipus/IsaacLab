# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Mewtwo robots.

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetLSTMCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration - Actuators.
##

MEWTWO_IMPLICIT_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*"],
    effort_limit=80.0,
    velocity_limit=7.5,
    stiffness={".*": 150.0},
    damping={".*": 5.0},
)

MEWTWO_SIMPLE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,
    stiffness={".*": 40.0},
    damping={".*": 5.0},
)
"""Configuration for ANYdrive 3.x with DC actuator model."""


MEWTWO_LSTM_ACTUATOR_CFG = ActuatorNetLSTMCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt",
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,
)
"""Configuration for ANYdrive 3.0 (used on ANYmal-C) with LSTM actuator model."""


##
# Configuration - Articulation.
##

MEWTWO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/zhengyuz/Downloads/Mewtwo/mewtwo.usd",
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

            "Coccyx_col1Joint:0": 0.0,
            "Coccyx_col1Joint:1": 0.2984513,
            "Coccyx_col2Joint:0": 0.436332,
            "Coccyx_col2Joint:1": -0.3054326,
            "Coccyx_col3Joint:0": 0.3926991,
            "Coccyx_col3Joint:1": -0.4066617,
            "Coccyx_col4Joint:0": 0.2530727415391778,
            "Coccyx_col4Joint:1": -0.349066,
            "Coccyx_col5Joint:0": 0.19198621771937624,
            "Coccyx_col5Joint:1": 0.0,

            "HeadJoint": 0.0,

            # Catch-all zeroing patterns (NO overlapping explicit names!)
            # Zero all limbs/feet/forearms/brachiums etc.
            r"(Left|Right)(Palm|Thigh|Calf|Heel|Toe|Forearm|Brachium).*": 0.0,
            # Zero all proximal/distal finger joints
            r"(Left|Right)(Distal|Proximal).*": 0.0,
        },
    ),
    actuators={"legs": MEWTWO_IMPLICIT_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
