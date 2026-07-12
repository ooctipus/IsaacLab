# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import MeshConeCfg, MeshSphereCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots import FRANKA_PANDA_CFG

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp
from .camera_cfg import StateObservationCfg

FINGERTIP_LIST = ["panda_rightfinger", "panda_leftfinger"]
THUMB_SENSOR = "panda_leftfinger_object_s"
FINGER_SENSORS = [f"{name}_object_s" for name in FINGERTIP_LIST if name != THUMB_SENSOR.replace("_object_s", "")]


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
        # the converted menagerie asset already authors the finger-coupling mimic in its
        # physics payload; re-enable _spawn_franka_with_finger_equality only for assets
        # that lack it (e.g. panda_instanceable)
        self.robot.init_state.rot = (0.0, 0.0, 1.0, 0.0)
        # keep action targets off the hard stops: the policy must not learn to ride
        # joint limits (a compliant-limit affordance that does not transfer across engines)
        for link_name in FINGERTIP_LIST:
            setattr(
                self,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Geometry/panda_link0/panda_link1/panda_link2/panda_link3/panda_link4/panda_link5/panda_link6/panda_link7/panda_hand/"
                    + link_name,
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
class FrankaEventCfg(dexsuite.EventCfg):
    """Franka-specific event configuration."""

    # gripper closing-speed randomization: with a relative-position action the drive
    # presses with the damping-independent stall force kp * action_scale (35 N) and
    # closes at kp * action_scale / kd, so derive the kd range from the intended speed
    # span (datasheet jaw speed 0.2 m/s down to a deliberately slow 0.01 m/s tail) — the
    # numbers stay correct if kp, the base kd, or the action scale change. The slow tail
    # forces the policy to gate lifting on the contact sensor instead of predicting
    # closure timing. Only the driven finger joint: finger_joint2 is passive (single-motor
    # hand); adding damping to the passive joint would drag the mimic asymmetrically.
    gripper_closing_speed = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint1"),
            "damping_distribution_params": (
                FRANKA_PANDA_CFG.actuators["panda_hand"].damping,
                FRANKA_PANDA_CFG.actuators["panda_hand"].stiffness * 0.1 / 0.01
                - FRANKA_PANDA_CFG.actuators["panda_hand"].damping,
            ),
            "operation": "add",
        },
    )

    def __post_init__(self):
        reset_terms = self.conditional_reset.params["terms"]
        criteria = self.conditional_reset.params["valid_criteria"]
        # the coupled finger pair is ONE mechanical DOF: independent per-joint draws write
        # equality-violating states the solver snaps shut at birth (measured: 0.7 m/s
        # passive-finger spikes in 58% of resets). Randomize the arm per-joint and the
        # gripper width with a single shared draw for the pair.
        reset_terms["reset_robot_wrist_joint"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names="panda_joint7")
        reset_terms["reset_robot_joints"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names="panda_joint.*")
        fingers = SceneEntityCfg("robot", joint_names="panda_finger_joint.*")
        reset_terms["reset_gripper_width"] = EventTerm(
            func="isaaclab_tasks.core.dexsuite.mdp.events:reset_joints_shared_offset",
            mode="reset",
            params={"position_range": [-0.04, 0.0], "asset_cfg": fingers},
        )
        # spawn-in-hand: the grasp center sits ~0.10 m along the hand z-axis (fingertip plane)
        to_target = reset_terms["reset_object_to_target"].params
        to_target["target_cfg"] = SceneEntityCfg("robot", body_names="panda_hand")
        to_target["pose_range"] = {"x": [-0.02, 0.02], "y": [-0.02, 0.02], "z": [0.08, 0.12]}
        # table/ground clearance on every link but the ground-mounted base: plain +-0.5 rad
        # draws put the wrist/hand inside the table in ~20% of draws, and the depenetration
        # slams panda_joint6 to 3-40x its velocity limit — the abnormal_robot floor.
        criteria["robot_table_clearance"].body_names = ["panda_link[1-7]", "panda_hand", ".*finger"]
        # keep the generic gain randomization off the fingers: the closing-speed term owns
        # their damping, and stacking the x2 scale on top cancels the grip force entirely.
        self.joint_stiffness_and_damping.params["asset_cfg"] = SceneEntityCfg("robot", joint_names="panda_joint.*")


@configclass
class FrankaMixinCfg:
    scene: FrankaSceneCfg = FrankaSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True)
    rewards: FrankaReorientRewardCfg = FrankaReorientRewardCfg()
    observations: StateObservationCfg = StateObservationCfg()
    actions: FrankaRelJointPosActionCfg = FrankaRelJointPosActionCfg()
    events: FrankaEventCfg = FrankaEventCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "panda_hand"
        # Franka base is rotated 180 deg about z, so the workspace mirrors to positive x.
        self.commands.object_pose.ranges.pos_x = (0.3, 0.7)
        self.terminations.abnormal_robot.params["asset_cfg"] = SceneEntityCfg("robot", joint_names="panda_joint.*")


@configclass
class DexsuiteFrankaReorientEnvCfg(FrankaMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexsuiteFrankaReorientEnvCfg_PLAY(FrankaMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        # deploy/eval at the datasheet gripper speed: no closing-speed randomization, and
        # the hand kd=175 caps closing at 0.2 m/s (the real hand's jaw-speed limit)
        self.events.gripper_closing_speed = None


@configclass
class DexsuiteFrankaLiftEnvCfg(FrankaMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteFrankaLiftEnvCfg_PLAY(FrankaMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        # deploy/eval at the datasheet gripper speed: no closing-speed randomization, and
        # the hand kd=175 caps closing at 0.2 m/s (the real hand's jaw-speed limit)
        self.events.gripper_closing_speed = None
