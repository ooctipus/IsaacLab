# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Allegro Hand robots from Wonik Robotics.

The following configurations are available:

* :obj:`ALLEGRO_HAND_CFG`: Allegro Hand with implicit actuator model.

Reference:

* https://www.wonikrobotics.com/robot-hand

"""

import math

from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

ALLEGRO_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/WonikRobotics/AllegroHand/allegro_hand_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            NewtonArticulationCfg(self_collision_enabled=True),
        ],
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.283045, 0.683330, -0.621782, 0.257551),
        joint_pos={"^(?!thumb_joint_0).*": 0.0, "thumb_joint_0": 0.28},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            joint_effort_limit=0.5,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
            # Each joint's motor reflected through its gearbox, N^2 I_rotor: the datasheet's gear ratio N = 369
            # (Allegro Hand V4, Wonik Robotics) and a rotor of about 1 g cm^2 = 1e-7 kg m^2, the brushed DC motors of
            # 14-16 mm and about 2.5 W that give its 0.7 N m at 0.11 s/60 deg (Drake's Allegro example derives the
            # same rotor; its 1e-6 kg m^2 is a unit slip). Known to about a factor of 3; it is a hundred times the
            # finger links' own inertia, so the fingers move as their motors do.
            armature=0.0136,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Allegro Hand robot."""


ALLEGRO_FINGERTIP_BODY_NAMES: list[str] = [
    "index_link_3",
    "middle_link_3",
    "ring_link_3",
    "thumb_link_3",
]
"""Allegro Hand fingertip body names."""

ALLEGRO_ACTUATED_JOINT_NAMES: list[str] = [
    "index_joint_0",
    "middle_joint_0",
    "ring_joint_0",
    "thumb_joint_0",
    "index_joint_1",
    "index_joint_2",
    "index_joint_3",
    "middle_joint_1",
    "middle_joint_2",
    "middle_joint_3",
    "ring_joint_1",
    "ring_joint_2",
    "ring_joint_3",
    "thumb_joint_1",
    "thumb_joint_2",
    "thumb_joint_3",
]
"""Allegro Hand actuated joint names, in the Direct task's actuation order."""


ALLEGRO_HAND_FRICTION_REPLACEMENTS = {
    "(index|middle|ring|thumb)_joint_[0-3]": (ALLEGRO_HAND_CFG.actuators["fingers"].friction, 0.0, 0.0)
}
"""OVPhysX legacy friction replacements matching the robot actuator efforts, in SI units."""
