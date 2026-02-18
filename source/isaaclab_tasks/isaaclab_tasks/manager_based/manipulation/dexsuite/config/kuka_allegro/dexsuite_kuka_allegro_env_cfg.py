# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp


finger_bodies = ["index_link_3", "middle_link_3", "ring_link_3"]
thumb_body = "thumb_link_3"


@configclass
class KukaAllegroSceneCfg(dexsuite.SceneCfg):
    """Kuka Allegro participant scene for Dexsuite Lifting/Reorientation"""

    def __post_init__(self: dexsuite.SceneCfg):
        super().__post_init__()
        self.robot = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        for link_name in finger_bodies + [thumb_body]:
            setattr(
                self,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/object"],
                ),
            )


@configclass
class KukaAllegroObservationCfg(dexsuite.ObservationsCfg):
    """Kuka Allegro participant observations for Dexsuite Lifting/Reorientation"""

    def __post_init__(self: dexsuite.ObservationsCfg):
        super().__post_init__()
        self.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_bodies + [thumb_body]]}
        )
        self.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["palm_link", ".*_tip"]


@configclass
class KukaAllegroRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class KukaAllegroReorientRewardCfg(dexsuite.RewardsCfg):
    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb.
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.5,
        params={
            "threshold": 0.01,
            "thumb_name": f"{thumb_body}_object_s",
            "finger_names": [f"{body}_object_s" for body in finger_bodies],
        },
    )

    contact_count = RewTerm(
        func=mdp.contact_count,
        weight=1.0,
        params={
            "threshold": 0.01,
            "sensor_names": [f"{body}_object_s" for body in finger_bodies] + [f"{thumb_body}_object_s"]
        },
    )

    def __post_init__(self: dexsuite.RewardsCfg):
        super().__post_init__()
        finger_sensors = [f"{body}_object_s" for body in finger_bodies]
        thumb_sensor = f"{thumb_body}_object_s"
        self.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"])
        # Add contact sensor params for rewards that need them
        # fingers_to_object (object_ee_distance)
        self.fingers_to_object.params["thumb_name"] = thumb_sensor
        self.fingers_to_object.params["finger_names"] = finger_sensors
        self.fingers_to_object.params["contact_threshold"] = 0.01

        # position/orientation tracking
        self.position_tracking.params["thumb_name"] = thumb_sensor
        self.position_tracking.params["finger_names"] = finger_sensors
        if self.orientation_tracking:
            self.orientation_tracking.params["thumb_name"] = thumb_sensor
            self.orientation_tracking.params["finger_names"] = finger_sensors
        self.success.params["thumb_name"] = thumb_sensor
        self.success.params["finger_names"] = finger_sensors

@configclass
class KukaAllegroMixinCfg:
    scene: KukaAllegroSceneCfg = KukaAllegroSceneCfg(env_spacing=3.0, num_envs=4096, replicate_physics=False)
    rewards: KukaAllegroReorientRewardCfg = KukaAllegroReorientRewardCfg()
    actions: KukaAllegroRelJointPosActionCfg = KukaAllegroRelJointPosActionCfg()
    observations: KukaAllegroObservationCfg = KukaAllegroObservationCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "palm_link"


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
