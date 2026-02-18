# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots import KUKA_INSPIRE_CFG

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp


finger_bodies = ["right_index_2", "right_middle_2", "right_ring_2", "right_little_2"]
thumb_body = "right_thumb_4"


@configclass
class KukaInspireRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class KukaInspireReorientRewardCfg(dexsuite.RewardsCfg):
    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb.
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=1.0,
        params={
            "threshold": 0.01,
            "thumb_name": f"{thumb_body}_object_s",
            "finger_names": [f"{body}_object_s" for body in finger_bodies]
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

@configclass
class KukaInspireMixinCfg:
    rewards: KukaInspireReorientRewardCfg = KukaInspireReorientRewardCfg()
    actions: KukaInspireRelJointPosActionCfg = KukaInspireRelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "mount"
        self.scene.robot = KUKA_INSPIRE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Body names (used in prim_path)
        all_bodies = finger_bodies + [thumb_body]

        # Sensor names (body name + _object_s suffix)
        finger_sensors = [f"{body}_object_s" for body in finger_bodies]
        thumb_sensor = f"{thumb_body}_object_s"

        # Create contact sensors for each finger tip body
        for body_name in all_bodies:
            sensor_name = f"{body_name}_object_s"
            setattr(
                self.scene,
                sensor_name,
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/" + body_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )

        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": finger_sensors + [thumb_sensor]},
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )
        self.events.randomize_object_scale.params["scale_range"] = (0.75, 1.0)  # smaller than allegro
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["mount", ".*(index|middle|ring|little)_2", "right_thumb_4"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["mount", ".*(index|middle|ring|little)_2", "right_thumb_4"])
        # Add contact sensor params for rewards that need them
        # fingers_to_object (object_ee_distance)
        self.rewards.fingers_to_object.params["thumb_name"] = thumb_sensor
        self.rewards.fingers_to_object.params["finger_names"] = finger_sensors
        self.rewards.fingers_to_object.params["contact_threshold"] = 0.01

        # position/orientation tracking
        self.rewards.position_tracking.params["thumb_name"] = thumb_sensor
        self.rewards.position_tracking.params["finger_names"] = finger_sensors
        if self.rewards.orientation_tracking:
            self.rewards.orientation_tracking.params["thumb_name"] = thumb_sensor
            self.rewards.orientation_tracking.params["finger_names"] = finger_sensors
        self.rewards.success.params["thumb_name"] = thumb_sensor
        self.rewards.success.params["finger_names"] = finger_sensors

@configclass
class DexsuiteKukaInspireReorientEnvCfg(KukaInspireMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexsuiteKukaInspireReorientEnvCfg_PLAY(KukaInspireMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    pass


@configclass
class DexsuiteKukaInspireLiftEnvCfg(KukaInspireMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteKukaInspireLiftEnvCfg_PLAY(KukaInspireMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass
