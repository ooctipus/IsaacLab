# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.assets import ArticulationCfg
from isaaclab_assets.robots.kuka_allegro import KUKA_ALLEGRO_CFG  # isort: skip
from isaaclab.sensors import ContactSensorCfg

from ... import dexsuite_env_cfg as dexsuite
from ... import dexsuite_shelf_env_cfg as dexsuite_shelf
from ... import mdp


@configclass
class KukaAllegroRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)

@configclass
class KukaAllegroReorientRewardCfg(dexsuite.RewardsCfg):

    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb. 
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.5,
        params={"threshold": 1.0},
    )
    
    # bad_finger_contact = RewTerm(
    #     func=mdp.contacts_sum,
    #     weight=-0.5,
    #     params={"threshold": 5.0, "sensors":[
    #         "index_link_3_table_s", "middle_link_3_table_s", "ring_link_3_table_s", "thumb_link_3_table_s",
    #         "index_link_2_table_s", "middle_link_2_table_s", "ring_link_2_table_s", "thumb_link_2_table_s",
    #         "palm_link_table_s"
    #     ]},
    # )


@configclass
class KukaAllegroMixinCfg:
    rewards: KukaAllegroReorientRewardCfg = KukaAllegroReorientRewardCfg()
    actions: KukaAllegroRelJointPosActionCfg = KukaAllegroRelJointPosActionCfg()
    def __post_init__(self: dexsuite.DexSuiteReorientEnvCfg):
        super().__post_init__()
        self.observations.policy.contact = ObsTerm(func=mdp.fingers_contact_force_w)
        self.commands.object_pose.body_name = "palm_link"
        self.scene.robot = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot").replace(
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    "iiwa7_joint_(1|2|7)": 0.,
                    "iiwa7_joint_3": 0.7854,
                    "iiwa7_joint_4": 1.5708,
                    "iiwa7_joint_(5|6)": -1.5708,
                    '(index|middle|ring)_joint_0': 0.0,
                    '(index|middle|ring)_joint_1': 0.3,
                    '(index|middle|ring)_joint_2': 0.3,
                    '(index|middle|ring)_joint_3': 0.3,
                    'thumb_joint_0': 1.5,
                    'thumb_joint_1': 0.60147215,
                    'thumb_joint_2': 0.33795027,
                    'thumb_joint_3': 0.60845138
                },
            ),
            actuators={
                "kuka_allegro_actuators": KUKA_ALLEGRO_CFG.actuators["kuka_allegro_actuators"].replace(
                    friction={
                        "iiwa7_joint_(1|2|3|4|5|6|7)": 1.,
                        "index_joint_(0|1|2|3)": 0.01,
                        "middle_joint_(0|1|2|3)": 0.01,
                        "ring_joint_(0|1|2|3)": 0.01,
                        "thumb_joint_(0|1|2|3)": 0.01,
                    }
                )
            },
        )

        self.observations.policy.hand_tips_state_b.params["body_asset_cfg"].body_names = ["palm_link", ".*_tip"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"])
        self.scene.robot.spawn.activate_contact_sensors = True
        for link_name in [
            "index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3",
        ]:
            setattr(self.scene, f"{link_name}_object_s", ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + link_name, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"]
            ))

@configclass
class DexsuiteKukaAllegroReorientEnvCfg(KukaAllegroMixinCfg, dexsuite.DexSuiteReorientEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroReorientEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.DexSuiteReorientEnvCfg_PLAY):
    pass


@configclass
class DexsuiteKukaAllegroLiftEnvCfg(KukaAllegroMixinCfg, dexsuite.DexSuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroLiftEnvCfg_PLAY(KukaAllegroMixinCfg, dexsuite.DexSuiteLiftEnvCfg_PLAY):
    pass


@configclass
class DexsuiteKukaAllegroShelvesReorientEnvCfg(KukaAllegroMixinCfg, dexsuite_shelf.DexSuiteShelvesReorientEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroShelvesPlaceEnvCfg(KukaAllegroMixinCfg, dexsuite_shelf.DexSuiteShelvesPlaceEnvCfg):
    pass
