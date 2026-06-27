# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import MeshConeCfg, MeshSphereCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import preset

from isaaclab_assets.robots import FRANKA_PANDA_CFG
from isaaclab_assets.robots.franka import NEWTON_ACTUATOR_CFG, PHYSX_ACTUATOR_CFG

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp
from .camera_cfg import StateObservationCfg

FINGERTIP_LIST = ["panda_rightfinger", "panda_leftfinger"]
THUMB_SENSOR = "panda_leftfinger_object_s"
FINGER_SENSORS = [f"{name}_object_s" for name in FINGERTIP_LIST if name != THUMB_SENSOR.replace("_object_s", "")]


def _spawn_franka_with_finger_equality(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the Franka USD, then couple the two gripper fingers with a MuJoCo joint-equality.

    Newton/mjwarp honors ``MjcEqualityJointAPI`` (``finger_joint2 = finger_joint1``); PhysX ignores it
    and keeps its own coupling. Joint prims in the Panda USD are authored at articulation scope (not
    inside the instanceable link prims), so they are editable directly - no de-instancing needed
    (de-instancing breaks the finger contact sensors).
    """
    from pxr import Sdf, Usd  # noqa: PLC0415

    import isaaclab.sim as sim_utils  # noqa: PLC0415

    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for root in sim_utils.find_matching_prims(prim_path):
        joints = {p.GetName(): p for p in Usd.PrimRange(root) if p.GetName().startswith("panda_finger_joint")}
        j1, j2 = joints.get("panda_finger_joint1"), joints.get("panda_finger_joint2")
        if j1 is None or j2 is None:
            continue
        j2.AddAppliedSchema("MjcEqualityJointAPI")
        j2.CreateRelationship("mjc:target").SetTargets([j1.GetPath()])
        j2.CreateAttribute("mjc:coef0", Sdf.ValueTypeNames.Double).Set(0.0)
        j2.CreateAttribute("mjc:coef1", Sdf.ValueTypeNames.Double).Set(1.0)
    return prim


@configclass
class FrankaSceneCfg(dexsuite.SceneCfg):
    """Franka scene for the dexsuite lift/reorient tasks.

    The ``base_camera`` / ``wrist_camera`` slots are left unset (``None``) for the state task; the
    camera env config populates them (see ``dexsuite_franka_camera_env_cfg``).
    """

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    base_camera: CameraCfg | None = None
    wrist_camera: CameraCfg | None = None

    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.activate_contact_sensors = True
        # author a MuJoCo finger-coupling equality at spawn (read by Newton/mjwarp; PhysX ignores it)
        self.robot.spawn.func = _spawn_franka_with_finger_equality
        actuator_cfg = preset(physx=PHYSX_ACTUATOR_CFG, newton_mjwarp=NEWTON_ACTUATOR_CFG, default=PHYSX_ACTUATOR_CFG)
        self.robot.actuators = actuator_cfg
        self.robot.init_state.rot = (0.0, 0.0, 1.0, 0.0)
        for link_name in FINGERTIP_LIST:
            setattr(
                self,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )
        graspable_assets_cfg = []
        for assets_cfg in self.object.spawn.shapes.assets_cfg:
            if not isinstance(assets_cfg, (MeshSphereCfg, MeshConeCfg)):
                graspable_assets_cfg.append(assets_cfg)
        self.object.spawn.shapes.assets_cfg = graspable_assets_cfg


@configclass
class FrankaRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class FrankaReorientRewardCfg(dexsuite.RewardsCfg):
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.125,
        params={"threshold": 0.1, "thumb_name": THUMB_SENSOR, "finger_names": FINGER_SENSORS},
    )

    contact_count = RewTerm(
        func=mdp.contact_count,
        weight=0.25,
        params={"threshold": 0.01, "sensor_names": FINGER_SENSORS + [THUMB_SENSOR]},
    )

    def __post_init__(self):
        super().__post_init__()
        self.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=".*finger")
        self.fingers_to_object.params["thumb_name"] = THUMB_SENSOR
        self.fingers_to_object.params["finger_names"] = FINGER_SENSORS
        self.position_tracking.params["thumb_name"] = THUMB_SENSOR
        self.position_tracking.params["finger_names"] = FINGER_SENSORS
        if self.orientation_tracking:
            self.orientation_tracking.params["thumb_name"] = THUMB_SENSOR
            self.orientation_tracking.params["finger_names"] = FINGER_SENSORS
        self.success.params["thumb_name"] = THUMB_SENSOR
        self.success.params["finger_names"] = FINGER_SENSORS


@configclass
class FrankaMixinCfg:
    scene: FrankaSceneCfg = FrankaSceneCfg(num_envs=4096, env_spacing=0, replicate_physics=True)
    rewards: FrankaReorientRewardCfg = FrankaReorientRewardCfg()
    observations: StateObservationCfg = StateObservationCfg()
    actions: FrankaRelJointPosActionCfg = FrankaRelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "panda_hand"
        self.events.reset_robot_wrist_joint.params["asset_cfg"] = SceneEntityCfg("robot", joint_names="panda_joint7")
        # Franka base is rotated 180 deg about z, so the workspace mirrors to positive x.
        self.commands.object_pose.ranges.pos_x = (0.3, 0.7)


@configclass
class DexsuiteFrankaReorientEnvCfg(FrankaMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexsuiteFrankaReorientEnvCfg_PLAY(FrankaMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    pass


@configclass
class DexsuiteFrankaLiftEnvCfg(FrankaMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteFrankaLiftEnvCfg_PLAY(FrankaMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass
