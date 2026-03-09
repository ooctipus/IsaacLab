# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots import KUKA_SHARPA_CFG

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp
from ..kuka_allegro.camera_cfg import (
    BaseTiledCameraCfg,
    DuoCameraObservationsCfg,
    SingleCameraObservationsCfg,
    StateObservationCfg,
    WristTiledCameraCfg,
)
from ..kuka_allegro.dexsuite_kuka_allegro_env_cfg import KukaAllegroPhysicsCfg

FINGER_BODIES = ["left_index_elastomer", "left_middle_elastomer", "left_ring_elastomer", "left_pinky_elastomer"]
THUMB_BODY = "left_thumb_elastomer"
THUMB_SENSOR = f"{THUMB_BODY}_object_s"
FINGER_SENSORS = [f"{body}_object_s" for body in FINGER_BODIES]


@configclass
class KukaSharpaSceneCfg(PresetCfg):
    @configclass
    class KukaSharpaSceneCfg(dexsuite.SceneCfg):
        """Kuka Sharpa participant scene for Dexsuite Lifting/Reorientation"""

        robot: ArticulationCfg = KUKA_SHARPA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        base_camera: TiledCameraCfg | None = None

        wrist_camera: TiledCameraCfg | None = None

        def __post_init__(self: dexsuite.SceneCfg):
            super().__post_init__()
            for body_name in FINGER_BODIES + [THUMB_BODY]:
                setattr(
                    self,
                    f"{body_name}_object_s",
                    ContactSensorCfg(
                        prim_path="{ENV_REGEX_NS}/Robot/ee_link/" + body_name,
                        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                    ),
                )

    default = KukaSharpaSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True)
    single_camera = default.replace(base_camera=BaseTiledCameraCfg())
    duo_camera = default.replace(base_camera=BaseTiledCameraCfg(), wrist_camera=WristTiledCameraCfg())


@configclass
class KukaSharpaRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class KukaSharpaReorientRewardCfg(dexsuite.RewardsCfg):
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=1.0,
        params={"threshold": 0.01, "thumb_name": THUMB_SENSOR, "finger_names": FINGER_SENSORS},
    )

    contact_count = RewTerm(
        func=mdp.contact_count,
        weight=1.0,
        params={
            "threshold": 0.01,
            "sensor_names": FINGER_SENSORS + [THUMB_SENSOR],
        },
    )

    def __post_init__(self: dexsuite.RewardsCfg):
        super().__post_init__()
        self.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["mount", ".*_elastomer"])
        self.fingers_to_object.params["thumb_name"] = THUMB_SENSOR
        self.fingers_to_object.params["finger_names"] = FINGER_SENSORS
        self.fingers_to_object.params["contact_threshold"] = 0.01
        self.position_tracking.params["thumb_name"] = THUMB_SENSOR
        self.position_tracking.params["finger_names"] = FINGER_SENSORS
        if self.orientation_tracking:
            self.orientation_tracking.params["thumb_name"] = THUMB_SENSOR
            self.orientation_tracking.params["finger_names"] = FINGER_SENSORS
        self.success.params["thumb_name"] = THUMB_SENSOR
        self.success.params["finger_names"] = FINGER_SENSORS


@configclass
class KukaSharpaObservationCfg(PresetCfg):
    default = StateObservationCfg()
    single_camera = SingleCameraObservationsCfg()
    duo_camera = DuoCameraObservationsCfg()


@configclass
class KukaSharpaEventCfg(PresetCfg):
    @configclass
    class KukaSharpaPhysxEventCfg(dexsuite.StartupEventCfg, dexsuite.EventCfg):
        pass

    default = KukaSharpaPhysxEventCfg()
    newton = dexsuite.EventCfg()
    physx = default


@configclass
class KukaSharpaMixinCfg:
    scene: KukaSharpaSceneCfg = KukaSharpaSceneCfg()
    rewards: KukaSharpaReorientRewardCfg = KukaSharpaReorientRewardCfg()
    observations: KukaSharpaObservationCfg = KukaSharpaObservationCfg()
    events: KukaSharpaEventCfg = KukaSharpaEventCfg()
    actions: KukaSharpaRelJointPosActionCfg = KukaSharpaRelJointPosActionCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = KukaAllegroPhysicsCfg()
        self.commands.object_pose.body_name = "mount"
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["mount", ".*_elastomer"]
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": FINGER_SENSORS + [THUMB_SENSOR]},
            clip=(-20.0, 20.0),
        )


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
