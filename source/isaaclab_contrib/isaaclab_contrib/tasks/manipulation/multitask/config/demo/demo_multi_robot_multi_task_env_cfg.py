# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Multi-robot multi-task env: OpenArm-lift, Franka-cabinet, UR10-reach.

Three robot types, each performing a different manipulation task.
Each robot-task pair occupies its own env-id group.  Separate
action terms handle DiffIK, relative joint-pos, and grippers:
``action_dim = 6 + 7 + 6 + 1 + 1 = 21`` (OpenArm IK + Franka joints + UR10 joints
+ OpenArm gripper + Franka gripper).

MDP terms use :class:`~isaaclab.managers.SceneEntityCfg` ``groups`` so each
term applies only to the clone groups that own the relevant assets.

Layout (3 groups, evenly split):
    Group 0:  OpenArm  -- Lift cube       (DiffIK + gripper)
    Group 1:  Franka   -- Open cabinet    (rel joint-pos + gripper)
    Group 2:  UR10     -- Track 6D pose   (rel joint-pos)
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    RelativeJointPositionActionCfg,
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
from isaaclab_contrib.tasks.manipulation.multitask.mdp.commands_cfg import PoseCommandRanges

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_assets.robots.openarm import OPENARM_UNI_HIGH_PD_CFG
from isaaclab_assets.robots.universal_robots import UR10_CFG

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

_OPENARM_MARKER_CFG = FRAME_MARKER_CFG.copy()
_OPENARM_MARKER_CFG.markers["frame"].scale = (0.1, 0.1, 0.1)
_OPENARM_MARKER_CFG.prim_path = "/Visuals/OpenArmEEFrame"

_FRANKA_MARKER_CFG = FRAME_MARKER_CFG.copy()
_FRANKA_MARKER_CFG.markers["frame"].scale = (0.1, 0.1, 0.1)
_FRANKA_MARKER_CFG.prim_path = "/Visuals/FrankaEEFrame"

_CABINET_MARKER_CFG = FRAME_MARKER_CFG.copy()
_CABINET_MARKER_CFG.markers["frame"].scale = (0.1, 0.1, 0.1)
_CABINET_MARKER_CFG.prim_path = "/Visuals/CabinetFrame"

_IK_CTRL = DifferentialIKControllerCfg(
    command_type="pose",
    use_relative_mode=True,
    ik_method="dls",
)


# ===========================================================
# Scene
# ===========================================================


@configclass
class MultiRobotMultiTaskSceneCfg(InteractiveSceneCfg):
    """Three robot types, each performing a different task."""

    clone_cfg = CloneCfg(
        clone_groups={
            "openarm_lift": InclusionSet(
                assets=["openarm_table", "openarm_robot", "openarm_object", "openarm_ee_frame"], weight=1
            ),
            "franka_cabinet": InclusionSet(
                assets=["franka_robot", "cabinet", "franka_ee_frame", "cabinet_frame"], weight=1
            ),
            "ur10_reach": InclusionSet(assets=["ur10_table", "ur10_robot"], weight=1),
        }
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

    # ── OpenArm Lift ─────────────────────────────────────────
    openarm_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/OpenArmTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
    )
    openarm_robot = OPENARM_UNI_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/OpenArm_Robot",
    )
    openarm_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/OpenArm_Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.055), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_CUBE_SPAWN,
    )
    openarm_ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/OpenArm_Robot/openarm_link0",
        debug_vis=False,
        visualizer_cfg=_OPENARM_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/OpenArm_Robot/openarm_ee_tcp",
                name="end_effector",
            ),
        ],
    )

    # ── Franka Cabinet ───────────────────────────────────────
    franka_robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Franka_Robot",
    )
    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.4),
            rot=(0.0, 0.0, 1.0, 0.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )
    franka_ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Franka_Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=_FRANKA_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Franka_Robot/panda_hand",
                name="ee_tcp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Franka_Robot/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Franka_Robot/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
        ],
    )
    cabinet_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet/sektion",
        debug_vis=False,
        visualizer_cfg=_CABINET_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_top",
                name="drawer_handle_top",
                offset=OffsetCfg(
                    pos=(0.305, 0.0, 0.01),
                    rot=(0.5, -0.5, -0.5, 0.5),
                ),
            ),
        ],
    )

    # ── UR10 Reach ───────────────────────────────────────────
    ur10_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/UR10Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
    )
    ur10_robot = UR10_CFG.replace(
        prim_path="{ENV_REGEX_NS}/UR10_Robot",
    )


# ===========================================================
# Actions  (OpenArm: DiffIK, Franka/UR10: relative joint pos, concatenated)
# ===========================================================


@configclass
class MultiRobotMultiTaskActionsCfg:
    """Actions for three robot types with heterogeneous arm control.

    Arm dims differ across robots so each has its own column. All group-specific
    terms are wrapped in ScatteredActionTermCfg so each sub-term is initialised
    with the correct group size rather than the global env count.
    Grippers (OpenArm + Franka) share one column.
    action_dim = 6 (OpenArm IK) + 7 (Franka joints) + 6 (UR10 joints) + 1 (gripper) = 20.
    """

    openarm_arm = mdp.ScatteredActionTermCfg(
        dim=6,
        terms=[
            DifferentialInverseKinematicsActionCfg(
                asset_name="openarm_robot",
                joint_names=["openarm_joint.*"],
                body_name="openarm_hand",
                controller=_IK_CTRL,
                scale=0.5,
            ),
        ],
    )
    franka_joints = mdp.ScatteredActionTermCfg(
        dim=7,
        terms=[
            RelativeJointPositionActionCfg(
                asset_name="franka_robot",
                joint_names=["panda_joint.*"],
                scale=0.1,
            ),
        ],
    )
    ur10_joints = mdp.ScatteredActionTermCfg(
        dim=6,
        terms=[
            RelativeJointPositionActionCfg(
                asset_name="ur10_robot",
                joint_names=[".*"],
                scale=0.1,
            ),
        ],
    )
    gripper = mdp.ScatteredActionTermCfg(
        dim=1,
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
        ],
    )


# ===========================================================
# Commands
# ===========================================================


@configclass
class MultiRobotMultiTaskCommandsCfg:
    """Pose commands for lift and reach; cabinet has no pose command."""

    lift_goal = mdp.PoseCommandCfg(
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
        debug_vis=True,
    )
    reach_target = mdp.PoseCommandCfg(
        asset_cfg=SceneEntityCfg("ur10_robot", body_names=["ee_link"], groups=["ur10_reach"]),
        ranges=PoseCommandRanges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi / 2, math.pi / 2),
            yaw=(-3.14, 3.14),
        ),
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
    )


# ===========================================================
# Observations
# ===========================================================


# fmt: off
@configclass
class MultiRobotMultiTaskObsCfg:
    """Per-group observation terms; inactive groups contribute zeros from each term."""

    @configclass
    class PolicyCfg(ObsGroup):

        # ── shared across all groups ──────────────────────
        task_onehot = ObsTerm(func=mdp.multi_task_onehot)
        ee_pose = ObsTerm(func=mdp.scatter_term, params={"terms": [
            TermCfg(func=mdp.ee_pose, params={"asset_cfg": SceneEntityCfg("openarm_robot", body_names=["openarm_hand"], groups=["openarm_lift"])}),
            TermCfg(func=mdp.ee_pose, params={"asset_cfg": SceneEntityCfg("franka_robot", body_names=["panda_hand"], groups=["franka_cabinet"])}),
            TermCfg(func=mdp.ee_pose, params={"asset_cfg": SceneEntityCfg("ur10_robot", body_names=["ee_link"], groups=["ur10_reach"])}),
        ]})
        actions = ObsTerm(func=mdp.last_action)

        # ── openarm_lift: object manipulation ─────────────
        openarm_object_pos = ObsTerm(
            func=mdp.object_pos_in_robot_frame,
            params={"robot_cfg": SceneEntityCfg("openarm_robot", groups=["openarm_lift"]), "object_cfg": SceneEntityCfg("openarm_object", groups=["openarm_lift"])},
        )
        openarm_object_target_pos_error = ObsTerm(
            func=mdp.object_target_pos_error,
            params={
                "robot_cfg": SceneEntityCfg("openarm_robot", groups=["openarm_lift"]),
                "object_cfg": SceneEntityCfg("openarm_object", groups=["openarm_lift"]),
                "command_name": "lift_goal",
            },
        )
        openarm_ee_object_pos_error = ObsTerm(
            func=mdp.ee_object_pos_error,
            params={
                "robot_cfg": SceneEntityCfg("openarm_robot", groups=["openarm_lift"]),
                "object_cfg": SceneEntityCfg("openarm_object", groups=["openarm_lift"]),
                "ee_frame_cfg": SceneEntityCfg("openarm_ee_frame", groups=["openarm_lift"]),
            },
        )

        # ── franka_cabinet: drawer manipulation ───────────
        cabinet_joint_pos = mdp.MultiTaskObsTerm(
            dim=1, func=mdp.cabinet_joint_pos,
            params={"cabinet_asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"], groups=["franka_cabinet"])},
        )
        cabinet_joint_vel = mdp.MultiTaskObsTerm(
            dim=1, func=mdp.cabinet_joint_vel,
            params={"cabinet_asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"], groups=["franka_cabinet"])},
        )
        cabinet_handle_error = ObsTerm(
            func=mdp.cabinet_rel_ee_drawer_distance,
            params={"ee_frame_cfg": SceneEntityCfg("franka_ee_frame", groups=["franka_cabinet"]), "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame", groups=["franka_cabinet"])},
        )

        # ── openarm_lift + ur10_reach: pose tracking ──────
        commands = ObsTerm(func=mdp.scatter_term, params={"output_dim": 7, "terms": [
            TermCfg(func=mdp.generated_commands, params={"asset_cfg": SceneEntityCfg("openarm_robot", groups=["openarm_lift"]), "command_name": "lift_goal"}),
            TermCfg(func=mdp.generated_commands, params={"asset_cfg": SceneEntityCfg("ur10_robot", groups=["ur10_reach"]), "command_name": "reach_target"}),
        ]})
        ee_pos_error = ObsTerm(func=mdp.scatter_term, params={"output_dim": 3, "terms": [
            TermCfg(func=mdp.ee_pos_error, params={"asset_cfg": SceneEntityCfg("openarm_robot", body_names=["openarm_hand"], groups=["openarm_lift"]), "command_name": "lift_goal"}),
            TermCfg(func=mdp.ee_pos_error, params={"asset_cfg": SceneEntityCfg("ur10_robot", body_names=["ee_link"], groups=["ur10_reach"]), "command_name": "reach_target"}),
        ]})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ===========================================================
# Rewards
# ===========================================================


@configclass
class MultiRobotMultiTaskRewardsCfg:
    """Task-specific rewards with per-group :class:`SceneEntityCfg` references."""

    # ── OpenArm Lift ─────────────────────────────────
    lift_reaching_object = RewTerm(
        func=mdp.object_ee_distance, weight=1.0,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("openarm_object", groups=["openarm_lift"]), "ee_frame_cfg": SceneEntityCfg("openarm_ee_frame", groups=["openarm_lift"])},
    )
    lift_lifting_object = RewTerm(
        func=mdp.object_is_lifted, weight=15.0,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("openarm_object", groups=["openarm_lift"])},
    )
    lift_object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance, weight=16.0,
        params={
            "std": 0.3, "minimal_height": 0.04, "command_name": "lift_goal",
            "robot_cfg": SceneEntityCfg("openarm_robot", groups=["openarm_lift"]), "object_cfg": SceneEntityCfg("openarm_object", groups=["openarm_lift"]),
        },
    )
    lift_object_goal_tracking_fine = RewTerm(
        func=mdp.object_goal_distance, weight=5.0,
        params={
            "std": 0.05, "minimal_height": 0.04, "command_name": "lift_goal",
            "robot_cfg": SceneEntityCfg("openarm_robot", groups=["openarm_lift"]), "object_cfg": SceneEntityCfg("openarm_object", groups=["openarm_lift"]),
        },
    )

    # ── Franka Cabinet ───────────────────────────────
    cabinet_approach_ee_handle = RewTerm(
        func=mdp.cabinet_approach_ee_handle, weight=2.0,
        params={"threshold": 0.2, "ee_frame_cfg": SceneEntityCfg("franka_ee_frame", groups=["franka_cabinet"]), "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame", groups=["franka_cabinet"])},
    )
    cabinet_align_ee_handle = RewTerm(
        func=mdp.cabinet_align_ee_handle, weight=0.5,
        params={"ee_frame_cfg": SceneEntityCfg("franka_ee_frame", groups=["franka_cabinet"]), "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame", groups=["franka_cabinet"])},
    )
    cabinet_approach_gripper_handle = RewTerm(
        func=mdp.cabinet_approach_gripper_handle, weight=5.0,
        params={"offset": 0.04, "ee_frame_cfg": SceneEntityCfg("franka_ee_frame", groups=["franka_cabinet"]), "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame", groups=["franka_cabinet"])},
    )
    cabinet_align_grasp_around_handle = RewTerm(
        func=mdp.cabinet_align_grasp_around_handle, weight=0.125,
        params={"ee_frame_cfg": SceneEntityCfg("franka_ee_frame", groups=["franka_cabinet"]), "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame", groups=["franka_cabinet"])},
    )
    cabinet_grasp_handle = RewTerm(
        func=mdp.cabinet_grasp_handle, weight=0.5,
        params={
            "threshold": 0.03, "open_joint_pos": 0.04,
            "asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_finger.*"], groups=["franka_cabinet"]),
            "ee_frame_cfg": SceneEntityCfg("franka_ee_frame", groups=["franka_cabinet"]), "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame", groups=["franka_cabinet"]),
        },
    )
    cabinet_open_drawer_bonus = RewTerm(
        func=mdp.cabinet_open_drawer_bonus, weight=7.5,
        params={
            "ee_frame_cfg": SceneEntityCfg("franka_ee_frame", groups=["franka_cabinet"]), "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame", groups=["franka_cabinet"]),
            "cabinet_asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"], groups=["franka_cabinet"]),
        },
    )
    cabinet_multi_stage_open_drawer = RewTerm(
        func=mdp.cabinet_multi_stage_open_drawer, weight=1.0,
        params={
            "ee_frame_cfg": SceneEntityCfg("franka_ee_frame", groups=["franka_cabinet"]), "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame", groups=["franka_cabinet"]),
            "cabinet_asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"], groups=["franka_cabinet"]),
        },
    )

    # ── UR10 Reach ───────────────────────────────────
    reach_ee_pos_tracking = RewTerm(
        func=mdp.position_command_error, weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("ur10_robot", body_names=["ee_link"], groups=["ur10_reach"]), "command_name": "reach_target"},
    )
    reach_ee_pos_tracking_fine = RewTerm(
        func=mdp.position_command_error_tanh, weight=0.1,
        params={"std": 0.1, "asset_cfg": SceneEntityCfg("ur10_robot", body_names=["ee_link"], groups=["ur10_reach"]), "command_name": "reach_target"},
    )
    reach_ee_ori_tracking = RewTerm(
        func=mdp.orientation_command_error, weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("ur10_robot", body_names=["ee_link"], groups=["ur10_reach"]), "command_name": "reach_target"},
    )
    reach_ee_ori_tracking_fine = RewTerm(
        func=mdp.orientation_command_error_tanh, weight=0.1,
        params={"std": 0.1, "asset_cfg": SceneEntityCfg("ur10_robot", body_names=["ee_link"], groups=["ur10_reach"]), "command_name": "reach_target"},
    )

    # ── Global penalties ─────────────────────────────
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.scatter_term, weight=-1e-4,
        params={"output_dim": 0, "terms": [
            TermCfg(
                func=mdp.joint_vel_l2,
                params={"asset_cfg": SceneEntityCfg("openarm_robot", joint_names=["openarm_joint.*", "openarm_finger_joint.*"], groups=["openarm_lift"])},
            ),
            TermCfg(
                func=mdp.joint_vel_l2,
                params={"asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_joint.*", "panda_finger.*"], groups=["franka_cabinet"])},
            ),
            TermCfg(func=mdp.joint_vel_l2, params={"asset_cfg": SceneEntityCfg("ur10_robot", joint_names=[".*"], groups=["ur10_reach"])}),
        ]},
    )


# ===========================================================
# Terminations
# ===========================================================
# fmt: on


@configclass
class MultiRobotMultiTaskTerminationsCfg:
    """Task-specific terminations for the multi-robot multi-task demo."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    lift_object_dropping = DoneTerm(
        func=mdp.object_height_below_minimum,
        params={
            "minimum_height": -0.05,
            "object_cfg": SceneEntityCfg("openarm_object", groups=["openarm_lift"]),
        },
    )
    cabinet_success = DoneTerm(
        func=mdp.cabinet_drawer_opened,
        params={
            "threshold": 0.39,
            "cabinet_asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"], groups=["franka_cabinet"]),
        },
    )


# ===========================================================
# Events
# ===========================================================


@configclass
class MultiRobotMultiTaskEventsCfg:
    """Per-robot reset events for heterogeneous layouts."""

    openarm_reset_to_default = EventTerm(
        func=mdp.reset_to_default,
        mode="reset",
        params={
            "reset_joint_targets": True,
            "asset_cfgs": [
                SceneEntityCfg("openarm_robot", groups=["openarm_lift"]),
                SceneEntityCfg("openarm_object", groups=["openarm_lift"]),
            ],
        },
    )
    franka_reset_to_default = EventTerm(
        func=mdp.reset_to_default,
        mode="reset",
        params={
            "reset_joint_targets": True,
            "asset_cfgs": [SceneEntityCfg("franka_robot", groups=["franka_cabinet"])],
        },
    )
    ur10_reset_to_default = EventTerm(
        func=mdp.reset_to_default,
        mode="reset",
        params={
            "reset_joint_targets": True,
            "asset_cfgs": [SceneEntityCfg("ur10_robot", groups=["ur10_reach"])],
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
                groups=["franka_cabinet"],
            ),
        },
    )
    ur10_reset_joints = EventTerm(
        func=mdp.reset_joints,
        mode="reset",
        params={
            "position_range": (0.5, 1.25),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("ur10_robot", joint_names=[".*"], groups=["ur10_reach"]),
        },
    )
    reset_openarm_object = EventTerm(
        func=mdp.reset_object_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "object_cfg": SceneEntityCfg("openarm_object", groups=["openarm_lift"]),
        },
    )
    reset_cabinet = EventTerm(
        func=mdp.reset_to_default,
        mode="reset",
        params={"asset_cfgs": [SceneEntityCfg("cabinet", groups=["franka_cabinet"])]},
    )


# ===========================================================
# Curriculum
# ===========================================================


@configclass
class MultiRobotMultiTaskCurriculumCfg:
    """Gradually increase action-rate and joint-vel penalties."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 100000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 100000},
    )


# ===========================================================
# Top-level env config
# ===========================================================


@configclass
class MultiRobotMultiTaskEnvCfg(ManagerBasedRLEnvCfg):
    """Multi-robot multi-task: OpenArm-lift, Franka-cabinet, UR10-reach.

    Group 0: OpenArm (7 arm DoF + 2 finger DoF) -- lift a cube
    Group 1: Franka  (7 arm DoF + 2 finger DoF) -- open a cabinet drawer
    Group 2: UR10    (6 arm DoF)                -- track a 6D pose command

    Action dim: 6 (IK) + 7 (Franka joints) + 6 (UR10 joints) + 1 (OpenArm gripper)
    + 1 (Franka gripper) = 21.
    Joint-space actions use independent columns per robot.

    Observations, rewards, and events use ``groups`` on :class:`SceneEntityCfg`.
    """

    scene: MultiRobotMultiTaskSceneCfg = MultiRobotMultiTaskSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
    )

    actions: MultiRobotMultiTaskActionsCfg = MultiRobotMultiTaskActionsCfg()
    commands: MultiRobotMultiTaskCommandsCfg = MultiRobotMultiTaskCommandsCfg()
    observations: MultiRobotMultiTaskObsCfg = MultiRobotMultiTaskObsCfg()
    rewards: MultiRobotMultiTaskRewardsCfg = MultiRobotMultiTaskRewardsCfg()
    terminations: MultiRobotMultiTaskTerminationsCfg = MultiRobotMultiTaskTerminationsCfg()
    events: MultiRobotMultiTaskEventsCfg = MultiRobotMultiTaskEventsCfg()
    curriculum: MultiRobotMultiTaskCurriculumCfg = MultiRobotMultiTaskCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physics = MultitaskPhysicsCfg()


@configclass
class MultiRobotMultiTaskEnvCfg_PLAY(MultiRobotMultiTaskEnvCfg):
    """Play config that selectively disables task groups.

    Disabled groups keep ``weight=0`` (preserving
    :func:`~...mdp.obs.multi_task_onehot` dimensionality) but receive
    zero environments and no spawned assets.  See :func:`mdp.apply_task_filter`
    for the full disabling logic.
    """

    disabled_tasks: tuple[str, ...] = ()
    """Clone-group names to disable.  Choices: ``"openarm_lift"``, ``"franka_cabinet"``, ``"ur10_reach"``.
    Set to ``()`` to keep all tasks."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.observations.policy.enable_corruption = False
        mdp.apply_task_filter(self, set(self.disabled_tasks))
