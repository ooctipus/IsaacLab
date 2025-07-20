# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, LOCAL_ASSET_PATH_DIR
from .tasks.assembly_object_key_points import NIST_BOARD_KEY_POINTS_CFG, Offset

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"


GROUND_CFG = AssetBaseCfg(
    prim_path="/World/ground",
    spawn=sim_utils.GroundPlaneCfg(),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.868)),
)


DOME_LIGHT_CFG = AssetBaseCfg(
    prim_path="/World/DomeLight",
    spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0),
)


FRANKA_PANDA_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        activate_contact_sensors=True,
        usd_path=f"{ASSET_DIR}/franka_mimic.usd",
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
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    # Stiffness and dampness of the panda arm parts
    # will be set
    actuators={
        "panda_arm1": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
            effort_limit=87,
            velocity_limit=124.6,
        ),
        "panda_arm2": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
            effort_limit=12,
            velocity_limit=149.5,
        ),
        # Stiffness and dampness should be zero in order for these to not move
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint[1-2]"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
)

ROBOT_ROOT_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RobotRoot",
    spawn=sim_utils.SphereCfg(
        radius=0.01,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=False,
        )
    )
)

PANDA_HAND_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/EndEffectorRoot",
    spawn=sim_utils.SphereCfg(
        radius=0.01,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=False,
        )
    )
)

# Table
TABLE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/Mounts/UWPatVention/pat_vention.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, -0.868), rot=(0.70711, 0.0, 0.0, -0.70711)),
)

# NIST Board
center_offset: Offset = list(NIST_BOARD_KEY_POINTS_CFG.nist_board_center.offsets)[0]  # type:ignore
x, y, z = center_offset.pos
NIST_BOARD_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/NistBoard",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/nistboard.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.000),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.85 - x, 0.0 - y, 0.02062 - z), rot=(0.0, 0.0, 1.0, 0.0)),
)

KIT_TRAY_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/KitTray",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/kit_tray.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 0.01), rot=(1.0, 0.0, 0.0, 0.0)),
)

##
# Assembly Tools
##


ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG = sim_utils.RigidBodyPropertiesCfg(
    kinematic_enabled=True,
    solver_position_iteration_count=192,
    solver_velocity_iteration_count=1,
)


ASSEMBLY_SOCKET_COLLISION_PROPS_CFG = sim_utils.CollisionPropertiesCfg(contact_offset=0.05, rest_offset=0.0)


ASSEMBLY_PLUG_COLLISION_PROPS_CFG = sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0)


ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG = sim_utils.RigidBodyPropertiesCfg(
    solver_position_iteration_count=192,
    solver_velocity_iteration_count=1,
)


BOLT_M16_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/BOLT_M16",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/bolt_m16.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    )
)


NUT_M16_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/NUT_M16",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/nut_m16.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    )
)


BOLT_M12_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/BOLT_M12",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/bolt_m12.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    )
)


NUT_M12_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/NUT_M12",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/nut_m12.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    )
)


BOLT_M8_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/BOLT_M8",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/bolt_m8.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    )
)


NUT_M8_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/NUT_M8",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/nut_m8.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    )
)


BOLT_M4_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/BOLT_M4",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/bolt_m4.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    )
)


NUT_M4_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/NUT_M4",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/nut_m4.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    )
)


HOLE_16MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/HOLE_16MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/round_hole_16mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    )
)


ROD_16MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/ROD_16MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/round_peg_16mm.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    )
)


HOLE_12MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/HOLE_12MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/round_hole_12mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    )
)


ROD_12MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/ROD_12MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/round_peg_12mm.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    )
)


HOLE_8MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/HOLE_8MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/round_hole_8mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    )
)


ROD_8MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/ROD_8MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/round_peg_8mm.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    )
)


HOLE_4MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/HOLE_4MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/round_hole_4mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    )
)


ROD_4MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/ROD_4MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/round_peg_4mm.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    )
)


RECTANGULAR_HOLE_16MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_HOLE_16MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/rectangular_hole_16mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    )
)


RECTANGULAR_PEG_16MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_PEG_16MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/rectangular_peg_16mm_tight.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    )
)


RECTANGULAR_HOLE_12MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_HOLE_12MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/rectangular_hole_12mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    )
)


RECTANGULAR_PEG_12MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_PEG_12MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/rectangular_peg_12mm_tight.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    )
)


RECTANGULAR_HOLE_8MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_HOLE_8MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/rectangular_hole_8mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    )
)


RECTANGULAR_PEG_8MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_PEG_8MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/rectangular_peg_8mm_tight.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    )
)


RECTANGULAR_HOLE_4MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_HOLE_4MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/rectangular_hole_4mm.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    )
)


RECTANGULAR_PEG_4MM_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RECTANGULAR_PEG_4MM",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/rectangular_peg_4mm_tight.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    )
)


LARGE_GEAR_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/LARGE_GEAR",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/gear_large.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    )
)


MEDIUM_GEAR_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/MEDIUM_GEAR",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/gear_medium.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.012),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    )
)


SMALL_GEAR_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/SMALL_GEAR",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/gear_small.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    )
)

GEAR_BASE_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/GEAR_BASE",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/gear_base.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    )
)


USBA_PLUG_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/USB_A_PLUG",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/usb_a_plug.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    )
)

USBA_SOCKET_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/USB_A_Socket",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/usb_a_socket.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.012),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    )
)


WATERPROOF_SOCKET_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/WATERPROOF_SOCKET",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/waterproof_socket.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    )
)

WATERPROOF_PLUG_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/WATERPROOF_PLUG",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/waterproof_plug.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    )
)


DSUB_SOCKET_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/D_SUB_SOCKET",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/dsub_socket.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    )
)


DSUB_PLUG_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/D_SUB_PLUG",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/dsub_plug.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG
    )
)


BNC_SOCKET_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/BNC_SOCKET",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/bnc_socket.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    )
)


BNC_PLUG_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/BNC_PLUG",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/bnc_plug.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    )
)


RJ45_SOCKET_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RJ45_SOCKET",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/rj45_socket.usd",
        rigid_props=ASSEMBLY_SOCKET_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_SOCKET_COLLISION_PROPS_CFG,
    )
)


RJ45_PLUG_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/RJ45_PLUG",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/NIST/rj45_plug.usd",
        rigid_props=ASSEMBLY_PLUG_RIGID_BODY_PROPS_CFG,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=ASSEMBLY_PLUG_COLLISION_PROPS_CFG,
    )
)
