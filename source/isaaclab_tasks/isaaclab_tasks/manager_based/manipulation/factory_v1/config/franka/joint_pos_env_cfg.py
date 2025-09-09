# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import RewardTermCfg as RewTerm

from ...factory_assets_cfg import FRANKA_PANDA_CFG
from ...factory_env_base import FactoryBaseEnvCfg
from ...gearmesh_env_cfg import GearMeshEnvCfg
from ...nutthread_env_cfg import NutThreadEnvCfg
from ...peginsert_env_cfg import PegInsertEnvCfg
from ... import mdp


@configclass
class ActionCfg:
    arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale={
            "(?!panda_joint7).*": 0.02,
            "panda_joint7": 0.2,
        },
        use_zero_offset=True,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class FrankaFactoryEnvMixIn:
    actions: ActionCfg = ActionCfg()

    def __post_init__(self: FactoryBaseEnvCfg):
        super().__post_init__()
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["panda_arm1"].stiffness = 80.0
        self.scene.robot.actuators["panda_arm1"].damping = 4.0
        self.scene.robot.actuators["panda_arm2"].stiffness = 80.0
        self.scene.robot.actuators["panda_arm2"].damping = 4.0

        finger_tip_body_list = ["panda_leftfinger", "panda_rightfinger"]
        for link_name in finger_tip_body_list:
            setattr(
                self.scene, f"{link_name}_object_s", ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/" + link_name)
            )

        setattr(
            self.rewards, "good_finger_contact", RewTerm(
                func=mdp.gripper_firm_contact, weight=0.5, params={"threshold": 10.0},
            )
        )

        setattr(
            self.rewards, "bad_finger_contact", RewTerm(
                func=mdp.gripper_asymetric_contact_penalty, weight=-0.5, params={"threshold": 1.0},
            )
        )


@configclass
class FrankaNutThreadEnvCfg(FrankaFactoryEnvMixIn, NutThreadEnvCfg):
    pass


@configclass
class FrankaGearMeshEnvCfg(FrankaFactoryEnvMixIn, GearMeshEnvCfg):
    pass


@configclass
class FrankaPegInsertEnvCfg(FrankaFactoryEnvMixIn, PegInsertEnvCfg):
    pass