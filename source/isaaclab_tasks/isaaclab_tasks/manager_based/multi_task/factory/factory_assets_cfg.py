# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg

# This is where we will get the Robot that we want to use
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, LOCAL_ASSET_PATH_DIR

from .assembly_keypoints import NIST_BOARD_CFG

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"


ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG = sim_utils.RigidBodyPropertiesCfg(
    kinematic_enabled=True,
    solver_position_iteration_count=192,
    solver_velocity_iteration_count=1,
)

ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG = sim_utils.RigidBodyPropertiesCfg(
    solver_position_iteration_count=192,
    solver_velocity_iteration_count=1,
)

ASSEMBLY_SOCKET_COLLISION_PROPS_CFG = sim_utils.CollisionPropertiesCfg(contact_offset=0.0025, rest_offset=0.0)


ASSEMBLY_PLUG_COLLISION_PROPS_CFG = sim_utils.CollisionPropertiesCfg(contact_offset=0.0025, rest_offset=0.0)

GROUND_CFG = AssetBaseCfg(
    prim_path="/World/ground",
    spawn=sim_utils.GroundPlaneCfg(),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.868)),
)

DOMELIGHT_CFG = AssetBaseCfg(
    prim_path="/World/DomeLight",
    spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0),
)


FRANKA_PANDA_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/franka_mimic.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.00871,
            "panda_joint2": -0.10368,
            "panda_joint3": -0.00794,
            "panda_joint4": -1.49139,
            "panda_joint5": -0.00083,
            "panda_joint6": 1.38774,
            "panda_joint7": 0.0,
            "panda_finger_joint2": 0.04,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
    ),
    # Stiffness and dampness of the panda arm parts
    # will be set
    actuators={
        "panda_arm1": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            stiffness=80.0,
            damping=4.0,
            friction=0.0,
            armature=0.0,
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
        ),
        "panda_arm2": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            stiffness=80.0,
            damping=4.0,
            friction=0.0,
            armature=0.0,
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
        ),
        # Stiffness and dampness should be zero in order for these to not move
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint[1-2]"],
            effort_limit_sim=40.0,
            velocity_limit_sim=0.2,
            stiffness=7500.0,
            damping=173.0,
            friction=0.1,
            armature=0.0,
        ),
    },
)

# Table
TABLE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/Mounts/UWPatVention/pat_vention.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, -0.868), rot=(0.0, 0.0, -0.70711, 0.70711)),
)

x, y, z = NIST_BOARD_CFG.nist_board_center.pos
# NIST Board
NISTBOARD_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/NistBoard",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/Taskboard/nistboard.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        scale=(1.0, 1.0, 0.5),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.65 - x, 0.0 - y, 0.0206 - z), rot=(0.0, 1.0, 0.0, 0.0)),
)

##
# Assembly Tools
##


BOLT_M16_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/BOLT_M16",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/bolt_m16.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    ),
)


NUT_M16_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/NUT_M16",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/nut_m16.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    ),
)


BOLT_M12_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/BOLT_M12",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/bolt_m12.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    ),
)


NUT_M12_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/NUT_M12",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/nut_m12.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    ),
)


BOLT_M8_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/BOLT_M8",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/bolt_m8.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    ),
)


NUT_M8_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/NUT_M8",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/nut_m8.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    ),
)


BOLT_M4_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/BOLT_M4",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/bolt_m4.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    ),
)


NUT_M4_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/NUT_M4",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/nut_m4.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    ),
)

HOLE_16MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/HOLE_16MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/round_hole_16mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    ),
)


ROD_16MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/ROD_16MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/round_peg_16mm.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    ),
)


HOLE_12MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/HOLE_12MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/round_hole_12mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    ),
)


ROD_12MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/ROD_12MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/round_peg_12mm.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    ),
)


HOLE_8MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/HOLE_8MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/round_hole_8mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    ),
)


ROD_8MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/ROD_8MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/round_peg_8mm.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    ),
)


HOLE_4MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/HOLE_4MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/round_hole_4mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    ),
)


ROD_4MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/ROD_4MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/round_peg_4mm.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    ),
)


RECTANGULAR_HOLE_16MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_HOLE_16MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/rectangular_hole_16mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    ),
)


RECTANGULAR_PEG_16MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_PEG_16MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/rectangular_peg_16mm.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    ),
)


RECTANGULAR_HOLE_12MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_HOLE_12MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/rectangular_hole_12mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    ),
)


RECTANGULAR_PEG_12MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_PEG_12MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/rectangular_peg_12mm.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    ),
)


RECTANGULAR_HOLE_8MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_HOLE_8MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/rectangular_hole_8mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    ),
)


RECTANGULAR_PEG_8MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_PEG_8MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/rectangular_peg_8mm.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    ),
)


RECTANGULAR_HOLE_4MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_HOLE_4MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/rectangular_hole_4mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    ),
)


RECTANGULAR_PEG_4MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_PEG_4MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/rectangular_peg_4mm.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    ),
)


LARGE_GEAR_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/LARGE_GEAR",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/gear_large.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    ),
)


MEDIUM_GEAR_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/MEDIUM_GEAR",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/gear_medium.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.012),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    ),
)


SMALL_GEAR_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/SMALL_GEAR",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/gear_small.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    ),
)

GEAR_BASE_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/GEAR_BASE",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/gear_base.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    ),
)


USBA_PLUG_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/USB_A_PLUG",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/usb_a_plug.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    ),
)

USBA_SOCKET_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/USB_A_Socket",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/usb_a_socket.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.012),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    ),
)


WATERPROOF_SOCKET_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/WATERPROOF_SOCKET",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/waterproof_socket.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    ),
)

WATERPROOF_PLUG_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/WATERPROOF_PLUG",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/waterproof_plug.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    ),
)


DSUB_SOCKET_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/D_SUB_SOCKET",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/dsub_socket.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    ),
)


DSUB_PLUG_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/D_SUB_PLUG",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/dsub_plug.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    ),
)


BNC_SOCKET_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/BNC_SOCKET",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/bnc_socket.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    ),
)


BNC_PLUG_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/BNC_PLUG",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/bnc_plug.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    ),
)


RJ45_SOCKET_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RJ45_SOCKET",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/rj45_socket.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    ),
)


RJ45_PLUG_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RJ45_PLUG",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST2/rj45_plug.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    ),
)
