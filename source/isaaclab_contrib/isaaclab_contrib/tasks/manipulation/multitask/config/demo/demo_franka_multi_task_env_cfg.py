# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Single-robot Franka multitask env with lift, cabinet, and reach groups.

This demo keeps one shared Franka articulation across all environments while
splitting environments into three task groups:

* ``lift``: lift a cube
* ``cabinet``: open a drawer
* ``reach``: track a pose command

All MDP terms use ``groups`` on :class:`SceneEntityCfg` to specify which
clone groups each term applies to.  Task-specific terms are zero-padded
for groups where they do not apply.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.cloner import cloner_strategies
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

_EE_MARKER_CFG = FRAME_MARKER_CFG.copy()
_EE_MARKER_CFG.markers["frame"].scale = (0.1, 0.1, 0.1)
_EE_MARKER_CFG.prim_path = "/Visuals/FrankaMultiTaskEEFrame"

_CABINET_MARKER_CFG = FRAME_MARKER_CFG.copy()
_CABINET_MARKER_CFG.markers["frame"].scale = (0.1, 0.1, 0.1)
_CABINET_MARKER_CFG.prim_path = "/Visuals/FrankaMultiTaskCabinetFrame"

_IK_CTRL = DifferentialIKControllerCfg(
    command_type="pose",
    use_relative_mode=True,
    ik_method="dls",
)


@configclass
class FrankaMultiTaskSceneCfg(InteractiveSceneCfg):
    """Single Franka scene with clone groups for lift, cabinet, and reach."""

    clone_cfg = CloneCfg(
        clone_strategy=cloner_strategies.sequential,
        clone_groups={
            "lift": InclusionSet(assets=["table", "lift_object"], weight=1),
            "cabinet": InclusionSet(assets=["cabinet", "cabinet_frame"], weight=1),
            "reach": InclusionSet(assets=["table"], weight=1),
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

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=_EE_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="ee_tcp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
        ],
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
    )
    lift_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LiftObject",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.055), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=_CUBE_SPAWN,
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
    cabinet_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet/sektion",
        debug_vis=True,
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


@configclass
class FrankaMultiTaskActionsCfg:
    """Shared Franka actions across all clone groups."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=_IK_CTRL,
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
    )
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class FrankaMultiTaskCommandsCfg:
    """Separate pose commands for lift and reach clone groups."""

    lift_goal = mdp.PoseCommandCfg(
        asset_cfg=SceneEntityCfg("robot", body_names=["panda_hand"], groups=["lift"]),
        ranges=PoseCommandRanges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.25, 0.25),
            pos_z=(0.25, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),
            yaw=(-3.14, 3.14),
        ),
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
    )
    reach_target = mdp.PoseCommandCfg(
        asset_cfg=SceneEntityCfg("robot", body_names=["panda_hand"], groups=["reach"]),
        ranges=PoseCommandRanges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),
            yaw=(-3.14, 3.14),
        ),
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
    )


# fmt: off
@configclass
class FrankaMultiTaskObservationsCfg:
    """Observation space for all clone groups.

    Every term is a :class:`ManagerTermBase` that internally
    iterates group entries and fills a single output tensor
    covering all environments.  Task-specific terms (e.g. object
    position for lift, cabinet joint state) are zero-padded for groups
    where they do not apply.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # ── shared across all groups ──────────────────────
        task_onehot = ObsTerm(func=mdp.multi_task_onehot)
        ee_pose = ObsTerm(
            func=mdp.ee_pose,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"], groups=[".*"]),
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        # ── lift: object manipulation ─────────────────────
        object_pos = ObsTerm(
            func=mdp.object_pos_in_robot_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot", groups=["lift"]),
                "object_cfg": SceneEntityCfg("lift_object", groups=["lift"]),
            },
        )
        object_target_pos_error = ObsTerm(
            func=mdp.object_target_pos_error,
            params={
                "robot_cfg": SceneEntityCfg("robot", groups=["lift"]),
                "object_cfg": SceneEntityCfg("lift_object", groups=["lift"]),
                "command_name": "lift_goal",
            },
        )
        ee_object_pos_error = ObsTerm(
            func=mdp.ee_object_pos_error,
            params={
                "robot_cfg": SceneEntityCfg("robot", groups=["lift"]),
                "object_cfg": SceneEntityCfg("lift_object", groups=["lift"]),
                "ee_frame_cfg": SceneEntityCfg("ee_frame", groups=["lift"]),
            },
        )

        # ── cabinet: drawer manipulation ──────────────────
        cabinet_joint_pos = ObsTerm(
            func=mdp.cabinet_joint_pos,
            params={
                "cabinet_asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"], groups=["cabinet"]),
            },
        )
        cabinet_joint_vel = ObsTerm(
            func=mdp.cabinet_joint_vel,
            params={
                "cabinet_asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"], groups=["cabinet"]),
            },
        )
        cabinet_handle_error = ObsTerm(
            func=mdp.cabinet_rel_ee_drawer_distance,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame", groups=["cabinet"]),
                "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame", groups=["cabinet"]),
            },
        )

        # ── lift + reach: pose tracking ───────────────────
        commands = ObsTerm(func=mdp.scatter_term, params={"terms": [
            TermCfg(func=mdp.generated_commands, params={
                "asset_cfg": SceneEntityCfg("robot", groups=["lift"]),
                "command_name": "lift_goal",
            }),
            TermCfg(func=mdp.generated_commands, params={
                "asset_cfg": SceneEntityCfg("robot", groups=["reach"]),
                "command_name": "reach_target",
            }),
        ]})
        ee_pos_error = ObsTerm(func=mdp.scatter_term, params={"terms": [
            TermCfg(func=mdp.ee_pos_error, params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"], groups=["lift"]),
                "command_name": "lift_goal",
            }),
            TermCfg(func=mdp.ee_pos_error, params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"], groups=["reach"]),
                "command_name": "reach_target",
            }),
        ]})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaMultiTaskRewardsCfg:
    """Rewards for lift, cabinet, and reach clone groups."""

    # Lift
    lift_reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        weight=1.0,
        params={
            "std": 0.1,
            "object_cfg": SceneEntityCfg("lift_object", groups=["lift"]),
            "ee_frame_cfg": SceneEntityCfg("ee_frame", groups=["lift"]),
        },
    )
    lift_lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        weight=15.0,
        params={
            "minimal_height": 0.04,
            "object_cfg": SceneEntityCfg("lift_object", groups=["lift"]),
        },
    )
    lift_object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        weight=16.0,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "lift_goal",
            "robot_cfg": SceneEntityCfg("robot", groups=["lift"]),
            "object_cfg": SceneEntityCfg("lift_object", groups=["lift"]),
        },
    )
    lift_object_goal_tracking_fine = RewTerm(
        func=mdp.object_goal_distance,
        weight=5.0,
        params={
            "std": 0.05,
            "minimal_height": 0.04,
            "command_name": "lift_goal",
            "robot_cfg": SceneEntityCfg("robot", groups=["lift"]),
            "object_cfg": SceneEntityCfg("lift_object", groups=["lift"]),
        },
    )

    # Cabinet
    cabinet_approach_ee_handle = RewTerm(
        func=mdp.cabinet_approach_ee_handle,
        weight=2.0,
        params={
            "threshold": 0.2,
            "ee_frame_cfg": SceneEntityCfg("ee_frame", groups=["cabinet"]),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame", groups=["cabinet"]),
        },
    )
    cabinet_align_ee_handle = RewTerm(
        func=mdp.cabinet_align_ee_handle,
        weight=0.5,
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame", groups=["cabinet"]),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame", groups=["cabinet"]),
        },
    )
    cabinet_approach_gripper_handle = RewTerm(
        func=mdp.cabinet_approach_gripper_handle,
        weight=5.0,
        params={
            "offset": 0.04,
            "ee_frame_cfg": SceneEntityCfg("ee_frame", groups=["cabinet"]),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame", groups=["cabinet"]),
        },
    )
    cabinet_align_grasp_around_handle = RewTerm(
        func=mdp.cabinet_align_grasp_around_handle,
        weight=0.125,
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame", groups=["cabinet"]),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame", groups=["cabinet"]),
        },
    )
    cabinet_grasp_handle = RewTerm(
        func=mdp.cabinet_grasp_handle,
        weight=0.5,
        params={
            "threshold": 0.03,
            "open_joint_pos": 0.04,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger.*"], groups=["cabinet"]),
            "ee_frame_cfg": SceneEntityCfg("ee_frame", groups=["cabinet"]),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame", groups=["cabinet"]),
        },
    )
    cabinet_open_drawer_bonus = RewTerm(
        func=mdp.cabinet_open_drawer_bonus,
        weight=7.5,
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame", groups=["cabinet"]),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame", groups=["cabinet"]),
            "cabinet_asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"], groups=["cabinet"]),
        },
    )
    cabinet_multi_stage_open_drawer = RewTerm(
        func=mdp.cabinet_multi_stage_open_drawer,
        weight=1.0,
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame", groups=["cabinet"]),
            "cabinet_frame_cfg": SceneEntityCfg("cabinet_frame", groups=["cabinet"]),
            "cabinet_asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"], groups=["cabinet"]),
        },
    )

    # Reach
    reach_ee_pos_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"], groups=["reach"]),
            "command_name": "reach_target",
        },
    )
    reach_ee_pos_tracking_fine = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={
            "std": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"], groups=["reach"]),
            "command_name": "reach_target",
        },
    )
    reach_ee_ori_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"], groups=["reach"]),
            "command_name": "reach_target",
        },
    )
    reach_ee_ori_tracking_fine = RewTerm(
        func=mdp.orientation_command_error_tanh,
        weight=0.1,
        params={
            "std": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"], groups=["reach"]),
            "command_name": "reach_target",
        },
    )

    # Global penalties
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*", "panda_finger.*"], groups=[".*"]),
        },
    )


# fmt: on
@configclass
class FrankaMultiTaskTerminationsCfg:
    """Terminations for the multitask Franka demo."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    lift_object_dropping = DoneTerm(
        func=mdp.object_height_below_minimum,
        params={
            "minimum_height": -0.05,
            "object_cfg": SceneEntityCfg("lift_object", groups=["lift"]),
        },
    )
    cabinet_success = DoneTerm(
        func=mdp.cabinet_drawer_opened,
        params={
            "threshold": 0.39,
            "cabinet_asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"], groups=["cabinet"]),
        },
    )


@configclass
class FrankaMultiTaskEventsCfg:
    """Reset events for shared-robot multitask layouts."""

    reset_to_default = EventTerm(
        func=mdp.reset_to_default,
        mode="reset",
        params={
            "reset_joint_targets": True,
            "asset_cfgs": [
                SceneEntityCfg("robot", groups=[".*"]),
                SceneEntityCfg("lift_object", groups=["lift"]),
            ],
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints,
        mode="reset",
        params={
            "position_range": (0.5, 1.25),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*", "panda_finger.*"], groups=[".*"]),
        },
    )
    reset_lift_object = EventTerm(
        func=mdp.reset_object_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "object_cfg": SceneEntityCfg("lift_object", groups=["lift"]),
        },
    )
    reset_cabinet = EventTerm(
        func=mdp.reset_to_default,
        mode="reset",
        params={
            "asset_cfgs": [SceneEntityCfg("cabinet", groups=["cabinet"])],
        },
    )


@configclass
class FrankaMultiTaskCurriculumCfg:
    """Gradually increase global action penalties."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 100000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 100000},
    )


@configclass
class FrankaMultiTaskEnvCfg(ManagerBasedRLEnvCfg):
    """Single Franka multitask RL env covering lift, cabinet, and reach.

    All MDP terms use ``groups`` on :class:`SceneEntityCfg` to specify
    which clone groups each term applies to.
    """

    scene: FrankaMultiTaskSceneCfg = FrankaMultiTaskSceneCfg(num_envs=4096, env_spacing=2.5)

    actions: FrankaMultiTaskActionsCfg = FrankaMultiTaskActionsCfg()
    commands: FrankaMultiTaskCommandsCfg = FrankaMultiTaskCommandsCfg()
    observations: FrankaMultiTaskObservationsCfg = FrankaMultiTaskObservationsCfg()
    rewards: FrankaMultiTaskRewardsCfg = FrankaMultiTaskRewardsCfg()
    terminations: FrankaMultiTaskTerminationsCfg = FrankaMultiTaskTerminationsCfg()
    events: FrankaMultiTaskEventsCfg = FrankaMultiTaskEventsCfg()
    curriculum: FrankaMultiTaskCurriculumCfg = FrankaMultiTaskCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physics = MultitaskPhysicsCfg()


@configclass
class FrankaMultiTaskEnvCfg_PLAY(FrankaMultiTaskEnvCfg):
    """Play config with fewer environments and no observation corruption."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.observations.policy.enable_corruption = False
