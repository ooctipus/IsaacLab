# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from ...factory_assets_cfg import FRANKA_PANDA_CFG
from ...factory_env_base import FactoryBaseEnvCfg
from ...gearmesh_env_cfg import GearMeshEnvCfg, GearMeshEnvSuccessTerminateCfg
from ...nutthread_env_cfg import NutThreadEnvCfg, NutThreadEnvSuccessTerminateCfg
from ...peginsert_env_cfg import PegInsertEnvCfg, PegInsertEnvSuccessTerminateCfg
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

        for link in ["panda_leftfinger", "panda_rightfinger"]:
            setattr(self.scene, f"{link}_object_s", ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/" + link))

        self.rewards.joint_effort.params["asset_cfg"] = SceneEntityCfg("robot", joint_names="(?!panda_joint7$|panda_finger_.*$).*")
        if hasattr(self.rewards, "reach_reward"):
            self.rewards.reach_reward.params["ee_cfg"] = SceneEntityCfg("robot", body_names="panda_fingertip_centered")

        gripper_penality = RewTerm(func=mdp.gripper_asymetric_contact_penalty, weight=-0.02, params={"threshold": 1.0})
        setattr(self.rewards, "bad_finger_contact", gripper_penality)


@configclass
class FrankaNutThreadEnvCfg(FrankaFactoryEnvMixIn, NutThreadEnvCfg):
    pass


@configclass
class FrankaGearMeshEnvCfg(FrankaFactoryEnvMixIn, GearMeshEnvCfg):
    pass


@configclass
class FrankaPegInsertEnvCfg(FrankaFactoryEnvMixIn, PegInsertEnvCfg):
    pass


@configclass
class FrankaNutThreadSuccessTerminateEnvCfg(FrankaFactoryEnvMixIn, NutThreadEnvSuccessTerminateCfg):
    pass


@configclass
class GearMeshEnvSuccessTerminateEnvCfg(FrankaFactoryEnvMixIn, GearMeshEnvSuccessTerminateCfg):
    pass


@configclass
class PegInsertEnvSuccessTerminateEnvCfg(FrankaFactoryEnvMixIn, PegInsertEnvSuccessTerminateCfg):
    pass
