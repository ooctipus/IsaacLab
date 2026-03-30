# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-robot reach env: three robot types, all reach.

Each robot type occupies its own env-id group. Three differential IK action
terms (one per arm) apply to disjoint env rows.

Layout (3 groups, evenly split):
    Group 0:  OpenArm -- Reach (7 arm DoF)
    Group 1:  Franka  -- Reach (7 arm DoF)
    Group 2:  UR10    -- Reach (6 arm DoF)
"""

from __future__ import annotations

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBaseCfg as TermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import CloneCfg, InclusionSet, InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_contrib.tasks.manipulation.multitask import mdp
from isaaclab_contrib.tasks.manipulation.multitask.mdp.utils import PoseCommandRanges

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_assets.robots.openarm import OPENARM_UNI_HIGH_PD_CFG
from isaaclab_assets.robots.universal_robots import UR10_CFG

_TABLE_SPAWN = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
)


@configclass
class MultitaskPhysicsCfg(PresetCfg):
    default: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
        gpu_total_aggregate_pairs_capacity=2**18,
        friction_correlation_distance=0.00625,
    )
    physx: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
        gpu_total_aggregate_pairs_capacity=2**18,
        friction_correlation_distance=0.00625,
    )
    newton: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=60,
            nconmax=80,
            ls_iterations=20,
            cone="pyramidal",
            ls_parallel=True,
            integrator="implicitfast",
            impratio=1,
        ),
        num_substeps=1,
        debug_mode=False,
    )


@configclass
class MultiRobotReachSceneCfg(InteractiveSceneCfg):
    clone_cfg = CloneCfg(
        clone_groups={
            "openarm_reach": InclusionSet(assets=["openarm_robot"], weight=1),
            "franka_reach": InclusionSet(assets=["franka_robot"], weight=1),
            "ur10_reach": InclusionSet(assets=["ur10_robot"], weight=1),
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
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=_TABLE_SPAWN,
    )
    openarm_robot = OPENARM_UNI_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/OpenArm_Robot")
    franka_robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Franka_Robot")
    ur10_robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/UR10_Robot")


_IK_CTRL = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")


@configclass
class MultiRobotReachActionsCfg:
    openarm_arm = DifferentialInverseKinematicsActionCfg(
        asset_name="openarm_robot",
        joint_names=["openarm_joint.*"],
        body_name="openarm_hand",
        controller=_IK_CTRL,
        scale=0.5,
    )
    franka_arm = DifferentialInverseKinematicsActionCfg(
        asset_name="franka_robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=_IK_CTRL,
        scale=0.5,
    )
    ur10_arm = DifferentialInverseKinematicsActionCfg(
        asset_name="ur10_robot",
        joint_names=[".*"],
        body_name="ee_link",
        controller=_IK_CTRL,
        scale=0.5,
    )


@configclass
class MultiRobotReachCommandsCfg:
    openarm_ee_pose = mdp.PoseCommandCfg(
        resampling_time_range=(3.0, 3.0),
        debug_vis=True,
        asset_cfg=SceneEntityCfg("openarm_robot", body_names=["openarm_hand"], groups=["openarm_reach"]),
        ranges=PoseCommandRanges(
            pos_x=(0.25, 0.35),
            pos_y=(-0.2, 0.2),
            pos_z=(0.3, 0.4),
            roll=(-math.pi / 6, math.pi / 6),
            pitch=(math.pi / 2, math.pi / 2),
            yaw=(-math.pi / 9, math.pi / 9),
        ),
    )
    franka_ee_pose = mdp.PoseCommandCfg(
        resampling_time_range=(3.0, 3.0),
        debug_vis=True,
        asset_cfg=SceneEntityCfg("franka_robot", body_names=["panda_hand"], groups=["franka_reach"]),
        ranges=PoseCommandRanges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),
            yaw=(-3.14, 3.14),
        ),
    )
    ur10_ee_pose = mdp.PoseCommandCfg(
        resampling_time_range=(3.0, 3.0),
        debug_vis=True,
        asset_cfg=SceneEntityCfg("ur10_robot", body_names=["ee_link"], groups=["ur10_reach"]),
        ranges=PoseCommandRanges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi / 2, math.pi / 2),
            yaw=(-3.14, 3.14),
        ),
    )


# fmt: off
@configclass
class MultiRobotReachObsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        openarm_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("openarm_robot", joint_names=["openarm_joint.*"], groups=["openarm_reach"])},
        )
        franka_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_joint.*"], groups=["franka_reach"])},
        )
        ur10_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("ur10_robot", joint_names=[".*"], groups=["ur10_reach"])},
        )
        openarm_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("openarm_robot", joint_names=["openarm_joint.*"], groups=["openarm_reach"])},
        )
        franka_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_joint.*"], groups=["franka_reach"])},
        )
        ur10_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("ur10_robot", joint_names=[".*"], groups=["ur10_reach"])},
        )
        ee_pose = ObsTerm(func=mdp.scatter_term, params={"terms": [
            TermCfg(func=mdp.ee_pose, params={"asset_cfg": SceneEntityCfg("openarm_robot", body_names=["openarm_hand"], groups=["openarm_reach"])}),
            TermCfg(func=mdp.ee_pose, params={"asset_cfg": SceneEntityCfg("franka_robot", body_names=["panda_hand"], groups=["franka_reach"])}),
            TermCfg(func=mdp.ee_pose, params={"asset_cfg": SceneEntityCfg("ur10_robot", body_names=["ee_link"], groups=["ur10_reach"])}),
        ]})
        ee_command = ObsTerm(func=mdp.scatter_term, params={"terms": [
            TermCfg(func=mdp.generated_commands, params={"asset_cfg": SceneEntityCfg("openarm_robot", groups=["openarm_reach"]), "command_name": "openarm_ee_pose"}),
            TermCfg(func=mdp.generated_commands, params={"asset_cfg": SceneEntityCfg("franka_robot", groups=["franka_reach"]), "command_name": "franka_ee_pose"}),
            TermCfg(func=mdp.generated_commands, params={"asset_cfg": SceneEntityCfg("ur10_robot", groups=["ur10_reach"]), "command_name": "ur10_ee_pose"}),
        ]})
        ee_pos_error = ObsTerm(func=mdp.scatter_term, params={"terms": [
            TermCfg(func=mdp.ee_pos_error, params={"asset_cfg": SceneEntityCfg("openarm_robot", body_names=["openarm_hand"], groups=["openarm_reach"]), "command_name": "openarm_ee_pose"}),
            TermCfg(func=mdp.ee_pos_error, params={"asset_cfg": SceneEntityCfg("franka_robot", body_names=["panda_hand"], groups=["franka_reach"]), "command_name": "franka_ee_pose"}),
            TermCfg(func=mdp.ee_pos_error, params={"asset_cfg": SceneEntityCfg("ur10_robot", body_names=["ee_link"], groups=["ur10_reach"]), "command_name": "ur10_ee_pose"}),
        ]})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class MultiRobotReachRewardsCfg:
    ee_pos_tracking = RewTerm(func=mdp.scatter_term, weight=-0.2, params={"terms": [
        TermCfg(func=mdp.position_command_error, params={"asset_cfg": SceneEntityCfg("openarm_robot", body_names=["openarm_hand"], groups=["openarm_reach"]), "command_name": "openarm_ee_pose"}),
        TermCfg(func=mdp.position_command_error, params={"asset_cfg": SceneEntityCfg("franka_robot", body_names=["panda_hand"], groups=["franka_reach"]), "command_name": "franka_ee_pose"}),
        TermCfg(func=mdp.position_command_error, params={"asset_cfg": SceneEntityCfg("ur10_robot", body_names=["ee_link"], groups=["ur10_reach"]), "command_name": "ur10_ee_pose"}),
    ]})
    ee_pos_tracking_fine = RewTerm(func=mdp.scatter_term, weight=0.1, params={"terms": [
        TermCfg(func=mdp.position_command_error_tanh, params={"std": 0.1, "asset_cfg": SceneEntityCfg("openarm_robot", body_names=["openarm_hand"], groups=["openarm_reach"]), "command_name": "openarm_ee_pose"}),
        TermCfg(func=mdp.position_command_error_tanh, params={"std": 0.1, "asset_cfg": SceneEntityCfg("franka_robot", body_names=["panda_hand"], groups=["franka_reach"]), "command_name": "franka_ee_pose"}),
        TermCfg(func=mdp.position_command_error_tanh, params={"std": 0.1, "asset_cfg": SceneEntityCfg("ur10_robot", body_names=["ee_link"], groups=["ur10_reach"]), "command_name": "ur10_ee_pose"}),
    ]})
    ee_ori_tracking = RewTerm(func=mdp.scatter_term, weight=-0.1, params={"terms": [
        TermCfg(func=mdp.orientation_command_error, params={"asset_cfg": SceneEntityCfg("openarm_robot", body_names=["openarm_hand"], groups=["openarm_reach"]), "command_name": "openarm_ee_pose"}),
        TermCfg(func=mdp.orientation_command_error, params={"asset_cfg": SceneEntityCfg("franka_robot", body_names=["panda_hand"], groups=["franka_reach"]), "command_name": "franka_ee_pose"}),
        TermCfg(func=mdp.orientation_command_error, params={"asset_cfg": SceneEntityCfg("ur10_robot", body_names=["ee_link"], groups=["ur10_reach"]), "command_name": "ur10_ee_pose"}),
    ]})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(func=mdp.scatter_term, weight=-0.0001, params={"terms": [
        TermCfg(func=mdp.joint_vel_l2, params={"asset_cfg": SceneEntityCfg("openarm_robot", joint_names=["openarm_joint.*"], groups=["openarm_reach"])}),
        TermCfg(func=mdp.joint_vel_l2, params={"asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_joint.*"], groups=["franka_reach"])}),
        TermCfg(func=mdp.joint_vel_l2, params={"asset_cfg": SceneEntityCfg("ur10_robot", joint_names=[".*"], groups=["ur10_reach"])}),
    ]})
# fmt: on


@configclass
class MultiRobotReachTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class MultiRobotReachEventsCfg:
    openarm_reset_to_default = EventTerm(
        func=mdp.reset_to_default,
        mode="reset",
        params={
            "reset_joint_targets": True,
            "asset_cfgs": [SceneEntityCfg("openarm_robot", groups=["openarm_reach"])],
        },
    )
    franka_reset_to_default = EventTerm(
        func=mdp.reset_to_default,
        mode="reset",
        params={
            "reset_joint_targets": True,
            "asset_cfgs": [SceneEntityCfg("franka_robot", groups=["franka_reach"])],
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
            "asset_cfg": SceneEntityCfg("openarm_robot", joint_names=["openarm_joint.*"], groups=["openarm_reach"]),
        },
    )
    franka_reset_joints = EventTerm(
        func=mdp.reset_joints,
        mode="reset",
        params={
            "position_range": (0.5, 1.25),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("franka_robot", joint_names=["panda_joint.*"], groups=["franka_reach"]),
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


@configclass
class MultiRobotReachCurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.005, "num_steps": 12000},
    )


@configclass
class MultiRobotReachEnvCfg(ManagerBasedRLEnvCfg):
    scene: MultiRobotReachSceneCfg = MultiRobotReachSceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=True,
    )
    actions: MultiRobotReachActionsCfg = MultiRobotReachActionsCfg()
    commands: MultiRobotReachCommandsCfg = MultiRobotReachCommandsCfg()
    observations: MultiRobotReachObsCfg = MultiRobotReachObsCfg()
    rewards: MultiRobotReachRewardsCfg = MultiRobotReachRewardsCfg()
    terminations: MultiRobotReachTerminationsCfg = MultiRobotReachTerminationsCfg()
    events: MultiRobotReachEventsCfg = MultiRobotReachEventsCfg()
    curriculum: MultiRobotReachCurriculumCfg = MultiRobotReachCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 3
        self.episode_length_s = 6.0
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physics = MultitaskPhysicsCfg()


@configclass
class MultiRobotReachEnvCfg_PLAY(MultiRobotReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.observations.policy.enable_corruption = False
