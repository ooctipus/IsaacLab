# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import RewardTermCfg as RewTerm

from ...factory_assets_cfg import FRANKA_PANDA_CFG
from ...factory_env_base import FactoryBaseEnvCfg, NutThreadSceneCfg, GearMeshSceneCfg, PegInsertSceneCfg
from ...factory_presets import EndEffectorBodyCfg, GripperJointNamesCfg, IKJointNamesCfg, JointEffortNamesCfg
from isaaclab_tasks.utils import PresetCfg
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
class FrankaSensorSceneCfg:
    panda_leftfinger_object_s = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger")
    panda_rightfinger_object_s = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger")

@configclass
class FrankaFactoryNutThreadSceneCfg(NutThreadSceneCfg, FrankaSensorSceneCfg):
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class FrankaFactoryGearMeshSceneCfg(GearMeshSceneCfg, FrankaSensorSceneCfg):
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class FrankaFactoryPegInsertSceneCfg(PegInsertSceneCfg, FrankaSensorSceneCfg):
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class FrankaFactorySceneCfg(PresetCfg):
    nut_thread = FrankaFactoryNutThreadSceneCfg(num_envs=2, env_spacing=2.0)
    gear_mesh = FrankaFactoryGearMeshSceneCfg(num_envs=2, env_spacing=2.0)
    peg_insert = FrankaFactoryPegInsertSceneCfg(num_envs=2, env_spacing=2.0)
    default = nut_thread


@configclass
class FrankaFactoryEnvMixIn:
    scene: FrankaFactorySceneCfg = FrankaFactorySceneCfg()
    actions: ActionCfg = ActionCfg()

    def __post_init__(self: FactoryBaseEnvCfg):
        super().__post_init__()
        gripper_penality = RewTerm(func=mdp.gripper_asymetric_contact_penalty, weight=-0.02, params={"threshold": 1.0})
        setattr(self.rewards, "bad_finger_contact", gripper_penality)


@configclass
class FrankaFactoryTaskEnvCfg(FrankaFactoryEnvMixIn, FactoryBaseEnvCfg):
    pass

