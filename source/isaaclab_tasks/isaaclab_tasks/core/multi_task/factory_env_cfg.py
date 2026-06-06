# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_physx.physics import PhysxCfg

from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_tasks.utils import PresetCfg, preset

from .factory import mdp
from .factory.factory_presets import (
    EndEffectorBodyCfg,
    FactoryAssemblyProfileCfg,
    FixedAssetTipCfg,
    HeldAssetAlignOffsetCfg,
    HeldAssetTipCfg,
    JointEffortNamesCfg,
)
from .factory.factory_scenes_cfg import FactorySceneCfg
from .factory.mdp_presets import GripperAsymContactPenaltyCfg, RobotActionsCfg
from .factory.reset_env_cfg import RESET_STRATEGIES
from .mdp.terminations import BaseTerminationsCfg


@configclass
class FactoryObservationsCfg:
    """Observation specifications for Factory."""

    @configclass
    class SuccessEstimatorInputCfg(ObsGroup):
        state = ObsTerm(
            func=mdp.get_state, params={"reset_assets": ["nistboard", "fixed_asset", "held_asset", "robot"]}
        )

        time_left = ObsTerm(func=mdp.time_left)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PolicyCfg(ObsGroup):
        end_effector_vel_lin_ang_b = ObsTerm(
            func=mdp.asset_link_velocity_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names=EndEffectorBodyCfg()),  # type:ignore
                "root_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        end_effector_pose = ObsTerm(
            func=mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names=EndEffectorBodyCfg()),  # type:ignore
                "root_asset_cfg": SceneEntityCfg("robot"),
                "target_asset_offset": HeldAssetTipCfg(),
            },
        )

        held_asset_in_fixed_asset_frame: ObsTerm = ObsTerm(
            func=mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("held_asset"),
                "root_asset_cfg": SceneEntityCfg("fixed_asset"),
                "root_asset_offset": FixedAssetTipCfg(),
            },
        )

        fixed_asset_in_end_effector_frame: ObsTerm = ObsTerm(
            func=mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("fixed_asset"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names=EndEffectorBodyCfg()),  # type:ignore
                "target_asset_offset": FixedAssetTipCfg(),
            },
        )

        joint_pos = ObsTerm(func=mdp.joint_pos)

        prev_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()
    success: SuccessEstimatorInputCfg = SuccessEstimatorInputCfg()


@configclass
class FactoryEventCfg:
    """Events specifications for Factory"""

    # when nut dropped right above the bolt, it sometime can immediately success due to high speed it falls
    # down can can may training in early stage very finicky. we uses less aggressive gravity for training
    # and can make more aggressive later in the stage...

    # mode: startup
    held_asset_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("held_asset"),
        },
    )

    fixed_asset_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("fixed_asset"),
        },
    )

    robot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_strategies = RESET_STRATEGIES

    variable_gravity = preset(
        default=EventTerm(
            func=mdp.randomize_physics_scene_gravity,
            mode="reset",
            params={"operation": "abs", "gravity_distribution_params": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))},
        ),
        eval=None,
    )


@configclass
class TimeoutRewardsCfg:
    """Reward terms for the timeout-terminate formulation (success is not terminal)."""

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-1e-4)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-1e-4)
    joint_effort = RewTerm(
        func=mdp.joint_torques_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointEffortNamesCfg())},  # type:ignore
        weight=-1e-4,
    )
    early_termination = RewTerm(func=mdp.is_terminated_term, params={"term_keys": "abnormal"}, weight=-0.01)
    reach_reward = RewTerm(
        func=mdp.reach_reward,
        weight=0.1,
        params={
            "std": 1.0,
            "held_asset_cfg": SceneEntityCfg("held_asset"),
            "ee_cfg": SceneEntityCfg("robot", body_names=EndEffectorBodyCfg()),  # type:ignore
        },
    )
    progress_reward_fine = RewTerm(func=mdp.progress_reward, weight=0.1, params={"std": 0.005})
    success_reward = RewTerm(func=mdp.success_reward, weight=1.0)
    bad_finger_contact: RewTerm | None = GripperAsymContactPenaltyCfg()  # type: ignore


@configclass
class SuccessRewardsCfg:
    """Reward terms for the success-terminate formulation (success is terminal)."""

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-1e-4)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-1e-4)
    joint_effort = RewTerm(
        func=mdp.joint_torques_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointEffortNamesCfg())},  # type:ignore
        weight=-1e-4,
    )
    early_termination = RewTerm(func=mdp.is_terminated_term, params={"term_keys": "abnormal"}, weight=-0.01)
    success_reward = RewTerm(func=mdp.success_reward, weight=100.0)
    bad_finger_contact: RewTerm | None = GripperAsymContactPenaltyCfg()  # type: ignore


@configclass
class FactoryRewardsCfg(PresetCfg):
    """Reward configuration preset for Factory tasks."""

    timeout_terminate: TimeoutRewardsCfg = TimeoutRewardsCfg()
    success_terminate: SuccessRewardsCfg = SuccessRewardsCfg()
    default: SuccessRewardsCfg = success_terminate


_PROGRESS_CONTEXT = DoneTerm(
    func=mdp.progress_context,
    params={
        "success_threshold": 0.001,
        "held_asset_cfg": SceneEntityCfg("held_asset"),
        "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
        "held_asset_offset": HeldAssetAlignOffsetCfg(),
        "assembly_profile": FactoryAssemblyProfileCfg(),
    },
)

_OOB = DoneTerm(
    func=mdp.out_of_bound,
    params={
        "asset_cfg": SceneEntityCfg("held_asset"),
        "in_bound_range": {"x": (-0.0, 1.0), "y": (-0.675, 0.675), "z": (-0.05, 1.0)},
    },
)


@configclass
class TimeoutTerminationsCfg(BaseTerminationsCfg):
    """Termination terms for the timeout-terminate formulation."""

    oob = _OOB
    progress_context = _PROGRESS_CONTEXT


@configclass
class SuccessTerminationsCfg(BaseTerminationsCfg):
    """Termination terms for the success-terminate formulation."""

    oob = _OOB
    progress_context = _PROGRESS_CONTEXT
    success = DoneTerm(func=mdp.success_termination)


@configclass
class FactoryTerminationsCfg(PresetCfg):
    """Termination configuration preset for Factory tasks."""

    timeout_terminate: TimeoutTerminationsCfg = TimeoutTerminationsCfg()
    success_terminate: SuccessTerminationsCfg = SuccessTerminationsCfg()
    default: SuccessTerminationsCfg = success_terminate


@configclass
class FactoryCurriculumsCfg:
    difficulty_scheduler = CurrTerm(
        func=mdp.DifficultyScheduler,
        params={
            "max_difficulty": 10,
            "success_rate_callback": preset(
                default="env.event_manager.get_term_cfg('reset_strategies').func.monitor_success_rate",
                accumulator="env.event_manager.get_term_cfg('reset_strategies').func.monitor_success_rate",
                choice="env.event_manager.get_term_cfg('reset_strategies').func.terms['reset_strategies'].func.term_success_rate",
            ),
        },
    )

    gravity_adr = preset(
        default=CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "events.variable_gravity.params.gravity_distribution_params",
                "modify_fn": mdp.initial_final_interpolate_fn,
                "modify_params": {
                    "initial_value": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                    "final_value": ((0.0, 0.0, -9.81), (0.0, 0.0, -9.81)),
                    "difficulty_term_str": "difficulty_scheduler",
                },
            },
        ),
        eval=None,
    )


##
# Environment configuration
##

@configclass
class FactoryPhysicsCfg(PresetCfg):
    """Physics-backend preset for Factory tasks.

    Selected via ``presets=physx`` (default) or ``presets=newton_mjwarp``. The PhysX
    variant keeps Factory's contact-rich solver tuning; the Newton variant follows Newton's
    reference ``example_nut_bolt_sdf`` (MuJoCo/Newton solver path): few constraint iterations
    with many line-search iterations, ``impratio=1.0``, ``num_substeps=5``, a small global
    shape gap, and Newton's SDF collision pipeline (rather than MuJoCo's internal contacts).
    Capacity knobs (``njmax``/``nconmax``) are kept larger than the bare nut/bolt demo since
    Factory's scene also contains the robot, NIST board, and table.
    """

    default = PhysxCfg(
        solver_type=1,
        max_position_iteration_count=192,
        max_velocity_iteration_count=1,
        bounce_threshold_velocity=0.2,
        friction_offset_threshold=0.01,
        friction_correlation_distance=0.00625,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=2**23,
        gpu_collision_stack_size=2**32 - 1,
        gpu_max_num_partitions=1,
        gpu_found_lost_pairs_capacity=2**22,
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=1500,
            nconmax=300,
            impratio=1.0,
            cone="pyramidal",
            update_data_interval=2,
            ls_parallel=False,
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            broad_phase="sap",
            max_triangle_pairs=60_000_000,
            rigid_contact_max=5_000_000,
        ),
        default_shape_cfg=NewtonShapeCfg(margin=0.0, gap=0.001, ke=1e7, kd=1e4),
        num_substeps=4,
        debug_mode=False,
        use_cuda_graph=True,
    )
    physx = default


@configclass
class FactoryBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the base Factory environment."""

    scene: FactorySceneCfg = FactorySceneCfg()
    observations: FactoryObservationsCfg = FactoryObservationsCfg()
    events: FactoryEventCfg = FactoryEventCfg()
    terminations: FactoryTerminationsCfg = FactoryTerminationsCfg()
    rewards: FactoryRewardsCfg = FactoryRewardsCfg()
    curriculum: FactoryCurriculumsCfg = FactoryCurriculumsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(0.0, 0.8, 0.4), origin_type="asset_root", asset_name="held_asset")
    actions: RobotActionsCfg = RobotActionsCfg()  # type: ignore

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 8
        self.episode_length_s = 14.0
        # simulation settings
        self.sim.dt = 0.04 / self.decimation
        self.sim.render_interval = self.decimation
        # Select the physics backend from the active preset (``presets=physx`` default, or
        # ``presets=newton_mjwarp`` for the kitless Newton/MuJoCo path). Previously this hardcoded
        # ``PhysxCfg`` here, which silently overrode ``presets=newton_mjwarp`` and forced Kit to launch.
        self.sim.physics = FactoryPhysicsCfg()

        self.sim.physics_material.static_friction = 0.5
        self.sim.physics_material.dynamic_friction = 0.5

        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_dlssg = True
