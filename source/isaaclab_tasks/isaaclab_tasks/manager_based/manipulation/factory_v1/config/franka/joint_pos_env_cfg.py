# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import RewardTermCfg as RewTerm

from ...factory_assets_cfg import FRANKA_PANDA_CFG
from ...factory_env_base import FactoryBaseEnvCfg
from ...assembly_keypoints import PANDA_HAND
from ...factory_presets import EndEffectorBodyCfg, GripperGraspOffsetCfg, GripperJointNamesCfg, IKJointNamesCfg, JointEffortNamesCfg
from ... import factory_scenes_cfg as scenes
from isaaclab_tasks.utils import PresetCfg
from ... import mdp

EndEffectorBodyCfg.franka = "panda_fingertip_centered"
EndEffectorBodyCfg.default = EndEffectorBodyCfg.franka

GripperJointNamesCfg.franka = ["panda_finger.*"]
GripperJointNamesCfg.default = GripperJointNamesCfg.franka

IKJointNamesCfg.franka = ["panda_joint.*"]
IKJointNamesCfg.default = IKJointNamesCfg.franka

GripperGraspOffsetCfg.franka = PANDA_HAND.gripper_center_grasp_point
GripperGraspOffsetCfg.default = GripperGraspOffsetCfg.franka

JointEffortNamesCfg.franka = "(?!panda_joint7$|panda_finger_.*$).*"
JointEffortNamesCfg.default = JointEffortNamesCfg.franka

_FRANKA_ROBOT = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


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


def _franka_scene(base: type) -> type:
    """Create a Franka scene class from a base scene class, adding robot + contact sensors."""
    return configclass(type(f"Franka{base.__name__}", (base, FrankaSensorSceneCfg), {"robot": _FRANKA_ROBOT}))


_NUM_ENVS = 2
_ENV_SPACING = 2.0


@configclass
class FrankaFactorySceneCfg(PresetCfg):
    # Nut threading
    nut_thread_m4 = _franka_scene(scenes.NutThreadM4SceneCfg)(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING)
    nut_thread_m8 = _franka_scene(scenes.NutThreadM8SceneCfg)(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING)
    nut_thread_m12 = _franka_scene(scenes.NutThreadM12SceneCfg)(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING)
    nut_thread_m16 = _franka_scene(scenes.NutThreadM16SceneCfg)(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING)

    # Gear mesh
    gear_mesh_small = _franka_scene(scenes.GearMeshSmallSceneCfg)(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING)
    gear_mesh_medium = _franka_scene(scenes.GearMeshMediumSceneCfg)(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING)
    gear_mesh_large = _franka_scene(scenes.GearMeshLargeSceneCfg)(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING)

    # Rod insert (round)
    rod_insert_4mm = _franka_scene(scenes.RodInsert4MMSceneCfg)(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING)
    rod_insert_8mm = _franka_scene(scenes.RodInsert8MMSceneCfg)(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING)
    rod_insert_12mm = _franka_scene(scenes.RodInsert12MMSceneCfg)(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING)
    rod_insert_16mm = _franka_scene(scenes.RodInsert16MMSceneCfg)(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING)

    # Peg insert (rectangular)
    peg_insert_4mm = _franka_scene(scenes.PegInsert4MMSceneCfg)(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING)
    peg_insert_8mm = _franka_scene(scenes.PegInsert8MMSceneCfg)(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING)
    peg_insert_12mm = _franka_scene(scenes.PegInsert12MMSceneCfg)(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING)
    peg_insert_16mm = _franka_scene(scenes.PegInsert16MMSceneCfg)(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING)

    # Connector insert
    usba = _franka_scene(scenes.ConnectorUSBASceneCfg)(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING)
    waterproof = _franka_scene(scenes.ConnectorWaterproofSceneCfg)(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING)
    bnc = _franka_scene(scenes.ConnectorBNCSceneCfg)(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING)
    dsub = _franka_scene(scenes.ConnectorDSUBSceneCfg)(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING)
    rj45 = _franka_scene(scenes.ConnectorRJ45SceneCfg)(num_envs=_NUM_ENVS, env_spacing=_ENV_SPACING)

    default = nut_thread_m16


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
