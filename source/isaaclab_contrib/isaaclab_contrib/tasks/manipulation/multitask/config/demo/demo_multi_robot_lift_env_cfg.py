# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-robot lift env: OpenArm + Franka, lifting a shared cube type.

Each robot type occupies its own env-id group.  A single
:class:`DiffIKAction` + :class:`BinaryGripperAction`
pair handles all groups.
``action_dim = 6 (IK) + 1 (gripper) = 7``.

The ``lift_object`` :class:`RigidObjectCfg` is listed in both groups'
:class:`InclusionSet`, so each env gets its own cube instance but
they share a single :class:`RigidObject` view.

Layout (2 groups, evenly split):
    Group 0:  OpenArm  -- Lift cube
    Group 1:  Franka   -- Lift cube
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.cloner import sequential
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBaseCfg as TermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import CloneCfg, InclusionSet, InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_contrib.tasks.manipulation.multitask import mdp
from isaaclab_contrib.tasks.manipulation.multitask.mdp.utils import PoseCommandRanges

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_assets.robots.openarm import OPENARM_UNI_HIGH_PD_CFG

from .demo_multi_robot_reach_env_cfg import MultitaskPhysicsCfg

_TABLE_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
)

_CUBE_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    scale=(0.8, 0.8, 0.8),
    rigid_props=RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    ),
)

_MARKER_CFG = FRAME_MARKER_CFG.copy()
_MARKER_CFG.markers["frame"].scale = (0.1, 0.1, 0.1)
_MARKER_CFG.prim_path = "/Visuals/FrameTransformer"


# ===========================================================
# Scene
# ===========================================================


@configclass
class MultiRobotLiftSceneCfg(InteractiveSceneCfg):
    """Two robot types lifting a shared cube type (one view, two groups)."""

    clone_cfg = CloneCfg(
        clone_strategy=sequential,
        clone_groups={
            "openarm_lift": InclusionSet(assets=["openarm_robot", "lift_object", "openarm_ee_frame"], weight=1),
            "franka_lift": InclusionSet(assets=["franka_robot", "lift_object", "franka_ee_frame"], weight=1),
        },
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
    )

    # ── Shared lift object (one view across both groups) ─────
    lift_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LiftObject",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 0.055), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_CUBE_SPAWN,
    )

    # ── OpenArm ──────────────────────────────────────────────
    openarm_robot = OPENARM_UNI_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/OpenArm_Robot",
    )
    openarm_ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/OpenArm_Robot/openarm_link0",
        debug_vis=False,
        visualizer_cfg=_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/OpenArm_Robot/openarm_ee_tcp",
                name="end_effector",
            ),
        ],
    )

    # ── Franka ───────────────────────────────────────────────
    franka_robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Franka_Robot",
    )
    franka_ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Franka_Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Franka_Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
            ),
        ],
    )


# ===========================================================
# Actions  (DiffIK + binary gripper, columns shared)
# ===========================================================

_IK_CTRL = DifferentialIKControllerCfg(
    command_type="pose",
    use_relative_mode=True,
    ik_method="dls",
)


@configclass
class MultiRobotLiftActionsCfg:
    """Shared-column actions for OpenArm + Franka, both doing DiffIK + gripper.

    action_dim = 6 (shared IK) + 1 (shared gripper) = 7.
    Each ScatteredActionTerm dispatches its shared columns to the right robot.
    """

    arm = mdp.ScatteredActionTermCfg(
        terms=[
            DifferentialInverseKinematicsActionCfg(
                asset_name="openarm_robot",
                joint_names=["openarm_joint.*"],
                body_name="openarm_hand",
                controller=_IK_CTRL,
                scale=0.5,
            ),
            DifferentialInverseKinematicsActionCfg(
                asset_name="franka_robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller=_IK_CTRL,
                scale=0.5,
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
            ),
        ]
    )
    gripper = mdp.ScatteredActionTermCfg(
        terms=[
            BinaryJointPositionActionCfg(
                asset_name="openarm_robot",
                joint_names=["openarm_finger_joint.*"],
                open_command_expr={"openarm_finger_joint.*": 0.044},
                close_command_expr={"openarm_finger_joint.*": 0.0},
            ),
            BinaryJointPositionActionCfg(
                asset_name="franka_robot",
                joint_names=["panda_finger.*"],
                open_command_expr={"panda_finger_.*": 0.04},
                close_command_expr={"panda_finger_.*": 0.0},
            ),
        ]
    )


# ===========================================================
# Commands
# ===========================================================


@configclass
class MultiRobotLiftCommandsCfg:
    """Per-group pose commands for lift goals."""

    openarm_lift_goal = mdp.PoseCommandCfg(
        asset_cfg=SceneEntityCfg("openarm_robot", body_names=["openarm_hand"], groups=["openarm_lift"]),
        ranges=PoseCommandRanges(
            pos_x=(0.2, 0.4),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.4),
            roll=(-math.pi / 6, math.pi / 6),
            pitch=(math.pi / 2, math.pi / 2),
            yaw=(-math.pi / 9, math.pi / 9),
        ),
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
    )
    franka_lift_goal = mdp.PoseCommandCfg(
        asset_cfg=SceneEntityCfg("franka_robot", body_names=["panda_hand"], groups=["franka_lift"]),
        ranges=PoseCommandRanges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.25, 0.25),
            pos_z=(0.25, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),
            yaw=(-3.14, 3.14),
        ),
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
    )


# ===========================================================
# Observations
# ===========================================================


# fmt: off
@configclass
class MultiRobotLiftObsCfg:
    """Proprioceptive + object observations with per-group :class:`SceneEntityCfg`."""

    @configclass
    class PolicyCfg(ObsGroup):
        # ── ee pose ──────────────────────
        ee_pose = ObsTerm(func=mdp.scatter_term, params={"terms": [
            TermCfg(func=mdp.ee_pose, params={"asset_cfg": SceneEntityCfg("openarm_robot", body_names=["openarm_hand"], groups=["openarm_lift"])}),
            TermCfg(func=mdp.ee_pose, params={"asset_cfg": SceneEntityCfg("franka_robot", body_names=["panda_hand"], groups=["franka_lift"])}),
        ]})

        # ── openarm_lift + franka_lift: object manipulation
        object_pos = ObsTerm(func=mdp.scatter_term, params={"terms": [
            TermCfg(func=mdp.object_pos_in_robot_frame, params={"robot_cfg": SceneEntityCfg("openarm_robot", groups=["openarm_lift"]), "object_cfg": SceneEntityCfg("lift_object", groups=["openarm_lift"])}),
            TermCfg(func=mdp.object_pos_in_robot_frame, params={"robot_cfg": SceneEntityCfg("franka_robot", groups=["franka_lift"]), "object_cfg": SceneEntityCfg("lift_object", groups=["franka_lift"])}),
        ]})
        target_object_pos = ObsTerm(func=mdp.scatter_term, params={"terms": [
            TermCfg(func=mdp.generated_commands, params={"asset_cfg": SceneEntityCfg("openarm_robot", groups=["openarm_lift"]), "command_name": "openarm_lift_goal"}),
            TermCfg(func=mdp.generated_commands, params={"asset_cfg": SceneEntityCfg("franka_robot", groups=["franka_lift"]), "command_name": "franka_lift_goal"}),
        ]})
        ee_object_pos_error = ObsTerm(func=mdp.scatter_term, params={"terms": [
            TermCfg(func=mdp.ee_object_pos_error, params={"robot_cfg": SceneEntityCfg("openarm_robot", groups=["openarm_lift"]), "object_cfg": SceneEntityCfg("lift_object", groups=["openarm_lift"]), "ee_frame_cfg": SceneEntityCfg("openarm_ee_frame", groups=["openarm_lift"])}),
            TermCfg(func=mdp.ee_object_pos_error, params={"robot_cfg": SceneEntityCfg("franka_robot", groups=["franka_lift"]), "object_cfg": SceneEntityCfg("lift_object", groups=["franka_lift"]), "ee_frame_cfg": SceneEntityCfg("franka_ee_frame", groups=["franka_lift"])}),
        ]})
        object_target_pos_error = ObsTerm(func=mdp.scatter_term, params={"terms": [
            TermCfg(func=mdp.object_target_pos_error, params={"robot_cfg": SceneEntityCfg("openarm_robot", groups=["openarm_lift"]), "object_cfg": SceneEntityCfg("lift_object", groups=["openarm_lift"]), "command_name": "openarm_lift_goal"}),
            TermCfg(func=mdp.object_target_pos_error, params={"robot_cfg": SceneEntityCfg("franka_robot", groups=["franka_lift"]), "object_cfg": SceneEntityCfg("lift_object", groups=["franka_lift"]), "command_name": "franka_lift_goal"}),
        ]})

        # ── last actions ──────────────────────
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ===========================================================
# Rewards
# ===========================================================


@configclass
class MultiRobotLiftRewardsCfg:
    """Lift rewards with composite terms over robot groups."""

    reaching_object = RewTerm(func=mdp.scatter_term, weight=1.0, params={"terms": [
        TermCfg(func=mdp.object_ee_distance, params={"std": 0.1, "object_cfg": SceneEntityCfg("lift_object", groups=["openarm_lift"]), "ee_frame_cfg": SceneEntityCfg("openarm_ee_frame", groups=["openarm_lift"])}),
        TermCfg(func=mdp.object_ee_distance, params={"std": 0.1, "object_cfg": SceneEntityCfg("lift_object", groups=["franka_lift"]), "ee_frame_cfg": SceneEntityCfg("franka_ee_frame", groups=["franka_lift"])}),
    ]})
    lifting_object = RewTerm(func=mdp.scatter_term, weight=15.0, params={"terms": [
        TermCfg(func=mdp.object_is_lifted, params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("lift_object", groups=["openarm_lift"])}),
        TermCfg(func=mdp.object_is_lifted, params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("lift_object", groups=["franka_lift"])}),
    ]})
    object_goal_tracking = RewTerm(func=mdp.scatter_term, weight=16.0, params={"terms": [
        TermCfg(func=mdp.object_goal_distance, params={
            "std": 0.3, "minimal_height": 0.04, "command_name": "openarm_lift_goal", "robot_cfg": SceneEntityCfg("openarm_robot", groups=["openarm_lift"]), "object_cfg": SceneEntityCfg("lift_object", groups=["openarm_lift"])}),
        TermCfg(func=mdp.object_goal_distance, params={
            "std": 0.3, "minimal_height": 0.04, "command_name": "franka_lift_goal", "robot_cfg": SceneEntityCfg("franka_robot", groups=["franka_lift"]), "object_cfg": SceneEntityCfg("lift_object", groups=["franka_lift"])}),
    ]})
    object_goal_tracking_fine = RewTerm(func=mdp.scatter_term, weight=5.0, params={"terms": [
        TermCfg(func=mdp.object_goal_distance, params={
            "std": 0.05, "minimal_height": 0.04, "command_name": "openarm_lift_goal", "robot_cfg": SceneEntityCfg("openarm_robot", groups=["openarm_lift"]), "object_cfg": SceneEntityCfg("lift_object", groups=["openarm_lift"])}),
        TermCfg(func=mdp.object_goal_distance, params={
            "std": 0.05, "minimal_height": 0.04, "command_name": "franka_lift_goal", "robot_cfg": SceneEntityCfg("franka_robot", groups=["franka_lift"]), "object_cfg": SceneEntityCfg("lift_object", groups=["franka_lift"])}),
    ]})

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.scatter_term, weight=-1e-4, params={"terms": [
        TermCfg(func=mdp.joint_vel_l2, params={
            "asset_cfg": SceneEntityCfg("openarm_robot", joint_names=["openarm_joint.*", "openarm_finger_joint.*"], groups=["openarm_lift"])}),
        TermCfg(func=mdp.joint_vel_l2, params={
            "asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_joint.*", "panda_finger.*"], groups=["franka_lift"])}),
    ]})


# ===========================================================
# Terminations
# ===========================================================
# fmt: on


@configclass
class MultiRobotLiftTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.object_height_below_minimum,
        params={
            "minimum_height": -0.05,
            "object_cfg": SceneEntityCfg("lift_object", groups=[".*"]),
        },
    )


# ===========================================================
# Events
# ===========================================================


@configclass
class MultiRobotLiftEventsCfg:
    """Reset events with per-robot terms and shared object reset."""

    openarm_reset_to_default = EventTerm(
        func=mdp.reset_to_default,
        mode="reset",
        params={
            "reset_joint_targets": True,
            "asset_cfgs": [
                SceneEntityCfg("openarm_robot", groups=["openarm_lift"]),
                SceneEntityCfg("lift_object", groups=["openarm_lift"]),
            ],
        },
    )
    franka_reset_to_default = EventTerm(
        func=mdp.reset_to_default,
        mode="reset",
        params={
            "reset_joint_targets": True,
            "asset_cfgs": [
                SceneEntityCfg("franka_robot", groups=["franka_lift"]),
                SceneEntityCfg("lift_object", groups=["franka_lift"]),
            ],
        },
    )
    openarm_reset_joints = EventTerm(
        func=mdp.reset_joints,
        mode="reset",
        params={
            "position_range": (0.5, 1.25),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg(
                "openarm_robot",
                joint_names=["openarm_joint.*", "openarm_finger_joint.*"],
                groups=["openarm_lift"],
            ),
        },
    )
    franka_reset_joints = EventTerm(
        func=mdp.reset_joints,
        mode="reset",
        params={
            "position_range": (0.5, 1.25),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg(
                "franka_robot",
                joint_names=["panda_joint.*", "panda_finger.*"],
                groups=["franka_lift"],
            ),
        },
    )
    reset_object = EventTerm(
        func=mdp.reset_object_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "object_cfg": SceneEntityCfg("lift_object", groups=[".*"]),
        },
    )


# ===========================================================
# Curriculum
# ===========================================================


@configclass
class MultiRobotLiftCurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )


# ===========================================================
# Top-level env config
# ===========================================================


@configclass
class MultiRobotLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Multi-robot lift: OpenArm + Franka, shared ``lift_object`` view.

    Group 0: OpenArm (7 arm DoF + 2 finger DoF)
    Group 1: Franka  (7 arm DoF + 2 finger DoF)

    Action dim: 6 (IK) + 1 (gripper) = 7.
    Columns shared across disjoint groups.

    The ``lift_object`` is included in both groups' :class:`InclusionSet`,
    so a single :class:`RigidObject` view spans all envs.
    """

    scene: MultiRobotLiftSceneCfg = MultiRobotLiftSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
    )

    actions: MultiRobotLiftActionsCfg = MultiRobotLiftActionsCfg()
    commands: MultiRobotLiftCommandsCfg = MultiRobotLiftCommandsCfg()
    observations: MultiRobotLiftObsCfg = MultiRobotLiftObsCfg()
    rewards: MultiRobotLiftRewardsCfg = MultiRobotLiftRewardsCfg()
    terminations: MultiRobotLiftTerminationsCfg = MultiRobotLiftTerminationsCfg()
    events: MultiRobotLiftEventsCfg = MultiRobotLiftEventsCfg()
    curriculum: MultiRobotLiftCurriculumCfg = MultiRobotLiftCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics = MultitaskPhysicsCfg()


@configclass
class MultiRobotLiftEnvCfg_PLAY(MultiRobotLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.observations.policy.enable_corruption = False
