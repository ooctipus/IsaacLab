# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots import KUKA_SHARPA_CFG

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp

thumb_body = "left_thumb_elastomer"
finger_names = ["left_index_elastomer", "left_middle_elastomer", "left_ring_elastomer", "left_pinky_elastomer"]


@configclass
class KukaSharpaRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class KukaSharpaReorientRewardCfg(dexsuite.RewardsCfg):
    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb.
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=1.0,
        params={
            "threshold": 0.01,
            "thumb_name": f"{thumb_body}_object_s",
            "finger_names": [f"{body}_object_s" for body in finger_names]
        },
    )

    contact_count = RewTerm(
        func=mdp.contact_count,
        weight=1.0,
        params={
            "threshold": 0.01,
            "sensor_names": [f"{body}_object_s" for body in finger_names] + [f"{thumb_body}_object_s"]
        },
    )


@configclass
class KukaSharpaMixinCfg:
    rewards: KukaSharpaReorientRewardCfg = KukaSharpaReorientRewardCfg()
    actions: KukaSharpaRelJointPosActionCfg = KukaSharpaRelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "mount"
        self.scene.robot = KUKA_SHARPA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Sensor names (body name + _object_s suffix)
        finger_sensors = [f"{body}_object_s" for body in finger_names]
        thumb_sensor = f"{thumb_body}_object_s"

        # Create contact sensors for each finger tip body
        for body_name in finger_names + [thumb_body]:
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
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["mount", ".*_elastomer"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["mount", ".*_elastomer"])

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
class DexsuiteKukaSharpaReorientEnvCfg(KukaSharpaMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexsuiteKukaSharpaReorientEnvCfg_PLAY(KukaSharpaMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    pass


@configclass
class DexsuiteKukaSharpaLiftEnvCfg(KukaSharpaMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteKukaSharpaLiftEnvCfg_PLAY(KukaSharpaMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass
