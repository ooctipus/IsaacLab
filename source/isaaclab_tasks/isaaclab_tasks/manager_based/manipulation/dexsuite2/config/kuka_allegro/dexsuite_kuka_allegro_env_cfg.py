# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ... import dexsuite_env_cfg as dexsuite
from ... import dexsuite_shelf_env_cfg as dexsuite_shelf
from ... import mdp


@configclass
class KukaAllegroSceneCfg(dexsuite.SceneCfg):
    """Kuka Allegro participant scene for Dexsuite Lifting/Reorientation"""

    def __post_init__(self: dexsuite.SceneCfg):
        super().__post_init__()
        self.robot = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        finger_tip_body_list = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
        for link_name in finger_tip_body_list:
            setattr(self, f"{link_name}_object_s", ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/ee_link/" + link_name,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
            ))


@configclass
class KukaAllegroFabricSceneCfg(KukaAllegroSceneCfg):
    """Kuka Allegro participant scene for Dexsuite Lifting/Reorientation"""

    def __post_init__(self: KukaAllegroSceneCfg):
        super().__post_init__()
        self.robot.init_state.joint_pos = {
            'iiwa7_joint_1': -0.85,
            'iiwa7_joint_2': 0.0,
            'iiwa7_joint_3': 0.76,
            'iiwa7_joint_4': 1.25,
            'iiwa7_joint_5': -1.76,
            'iiwa7_joint_6': 0.90,
            'iiwa7_joint_7': 0.64,
            '(index|middle|ring)_joint_0': 0.0,
            '(index|middle|ring)_joint_1': 0.3,
            '(index|middle|ring)_joint_2': 0.3,
            '(index|middle|ring)_joint_3': 0.3,
            'thumb_joint_0': 1.5,
            'thumb_joint_1': 0.60147215,
            'thumb_joint_2': 0.33795027,
            'thumb_joint_3': 0.60845138
        }


@configclass
class KukaAllegroObservationCfg(dexsuite.ObservationsCfg):
    """Kuka Allegro participant scene for Dexsuite Lifting/Reorientation"""

    def __post_init__(self: dexsuite.ObservationsCfg):
        super().__post_init__()
        finger_tip_body_list = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
        self.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )
        self.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["palm_link", ".*_tip"]


@configclass
class NonStackingObservationCfg(KukaAllegroObservationCfg):
    """Non-stacking observation configuration."""

    def __post_init__(self):
        super().__post_init__()
        for group in self.__dataclass_fields__.values():
            obs_group = getattr(self, group.name)
            obs_group.history_length = None


@configclass
class KukaAllegroRelJointPosActionCfg:
    joint_action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class KukaAllegroFabricActionCfg:
    joint_action = mdp.FabricActionCfg(asset_name="robot")


@configclass
class KukaAllegroReorientRewardCfg(dexsuite.RewardsCfg):

    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb.
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.5,
        params={"threshold": 1.0},
    )


@configclass
class KukaAllegroMixinCfg:
    scene: KukaAllegroSceneCfg = KukaAllegroSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    observations: KukaAllegroObservationCfg = KukaAllegroObservationCfg()
    rewards: KukaAllegroReorientRewardCfg = KukaAllegroReorientRewardCfg()
    actions: KukaAllegroRelJointPosActionCfg = KukaAllegroRelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "palm_link"
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"])
        self.variants.setdefault("observations", {}).update({
            "non_stacking": NonStackingObservationCfg()
        })


@configclass
class KukaAllegroFabricMixinCfg(KukaAllegroMixinCfg):
    scene: KukaAllegroFabricSceneCfg = KukaAllegroFabricSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    actions: KukaAllegroFabricActionCfg = KukaAllegroFabricActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        del self.events.reset_robot_wrist_joint
        # del self.events.variable_gravity
        # del self.curriculum.gravity_adr


@configclass
class DexsuiteKukaAllegroReorientEnvCfg(KukaAllegroMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroReorientEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    pass


@configclass
class DexsuiteKukaAllegroLiftEnvCfg(KukaAllegroMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroLiftEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass


@configclass
class DexsuiteKukaAllegroFabricLiftEnvCfg(KukaAllegroFabricMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroFabricLiftEnvCfg_PLAY(KukaAllegroFabricMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass


@configclass
class DexsuiteKukaAllegroShelvesReorientEnvCfg(KukaAllegroMixinCfg, dexsuite_shelf.DexSuiteShelvesReorientEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroShelvesPlaceEnvCfg(KukaAllegroMixinCfg, dexsuite_shelf.DexSuiteShelvesPlaceEnvCfg):
    pass
