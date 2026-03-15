# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import RewardTermCfg as RewTerm

from ...factory_assets_cfg import FRANKA_PANDA_CFG
from ...factory_env_base import FactoryBaseEnvCfg
from ...factory_presets import EndEffectorBodyCfg, GripperJointNamesCfg, IKJointNamesCfg, JointEffortNamesCfg
from ... import mdp

# Register Franka-specific robot presets
EndEffectorBodyCfg.franka = "panda_fingertip_centered"
EndEffectorBodyCfg.default = EndEffectorBodyCfg.franka

GripperJointNamesCfg.franka = ["panda_finger.*"]
GripperJointNamesCfg.default = GripperJointNamesCfg.franka

IKJointNamesCfg.franka = ["panda_joint.*"]
IKJointNamesCfg.default = IKJointNamesCfg.franka

JointEffortNamesCfg.franka = "(?!panda_joint7$|panda_finger_.*$).*"
JointEffortNamesCfg.default = JointEffortNamesCfg.franka


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

        gripper_penality = RewTerm(func=mdp.gripper_asymetric_contact_penalty, weight=-0.02, params={"threshold": 1.0})
        setattr(self.rewards, "bad_finger_contact", gripper_penality)


@configclass
class FrankaFactoryTaskEnvCfg(FrankaFactoryEnvMixIn, FactoryBaseEnvCfg):
    pass

