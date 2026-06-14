# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.multi_task.curriculum import StateLayoutCfg, SuccessMonitorCfg
from isaaclab_tasks.utils import PresetCfg, preset

from .factory import mdp, mdp_presets
from .factory.factory_presets import (
    EndEffectorBodyCfg,
    FactoryAssemblyProfileCfg,
    FingerBodyNamesCfg,
    FixedAssetMapCfg,
    FixedAssetTipCfg,
    GripperBodyNamesCfg,
    HeldAssetAlignOffsetCfg,
    HeldAssetSymmetryCfg,
    HeldAssetTipCfg,
)
from .factory.factory_scenes_cfg import FactorySceneCfg
from .factory.mdp_presets import RobotActionsCfg
from .factory.reset_env_cfg import FACTORY_RESET_SAMPLER_PRESETS
from .factory.retarget import (
    BoardLibraryCfg,
    CollisionAvoidanceCfg,
    CollisionCheckCfg,
    FactoryIKPipelineCfg,
    FactoryRobotCfg,
    FingerPinObjectiveCfg,
    GraspSamplingCfg,
    IKSolveCfg,
    JointDefaultObjectiveCfg,
    JointLimitObjectiveCfg,
    JointWithinLimitCfg,
    PlacementSamplingCfg,
    ReachRowsCfg,
)
from .factory.viz.sampler_images import log_factory_board_grid


@configclass
class FactoryObservationsCfg:
    """Observation specifications for Factory."""

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

        reset_state_commands: ObsTerm = ObsTerm(func=mdp.generated_commands, params={"command_name": "reset_state"})

        fixed_asset_in_end_effector_frame: ObsTerm = ObsTerm(
            func=mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("fixed_asset"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names=EndEffectorBodyCfg()),  # type:ignore
                "target_asset_offset": FixedAssetTipCfg(),
            },
        )

        held_asset_in_end_effector_frame: ObsTerm = ObsTerm(
            func=mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("held_asset"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names=EndEffectorBodyCfg()),  # type:ignore
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

    variable_gravity = preset(
        default=EventTerm(
            func=mdp.randomize_physics_scene_gravity,
            mode="reset",
            params={"operation": "abs", "gravity_distribution_params": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))},
        ),
        eval=None,
    )


@configclass
class FactoryCommandsCfg:
    """Command specifications for Factory."""

    reset_state = mdp.StateCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        randomize_command_indices=False,
        states_relative=True,
        commands={
            "assembly_asset": mdp.FactoryAssemblyAssetCommandCfg(
                orientation_threshold=0.025,
                position_threshold=0.005,
                duration=(0.0, 1.0),
            )
        },
        payload=mdp.FactoryAssemblyPayloadCfg(
            reset_assets=["nistboard", "fixed_asset", "held_asset", "robot"],
            held_asset_cfg=SceneEntityCfg("held_asset"),
            fixed_asset_cfg=SceneEntityCfg("fixed_asset"),
            robot_cfg=SceneEntityCfg("robot"),
            symmetry=HeldAssetSymmetryCfg(),  # type: ignore[arg-type]
        ),
        task_table=mdp.FactoryResetStateTableCfg(
            pipeline_cfg=FactoryIKPipelineCfg(
                # identity: scene entities + robot presets -- USD paths, stance, and
                # device resolve from the live env at wiring time (no assumptions here)
                board=BoardLibraryCfg(  # stage 1: the WORLD cells (terrain-grid analog)
                    board_asset_cfg=SceneEntityCfg("nistboard"),
                    fixed_asset_cfg=SceneEntityCfg("fixed_asset"),
                    fixed_asset_map=FixedAssetMapCfg(),  # type: ignore[arg-type]
                    num_boards=64,
                    pose_range={
                        "x": (-0.1, 0.1),
                        "y": (-0.1, 0.1),
                        "z": (0.0, 0.1),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.8, 0.8),
                    },
                ),
                placement=PlacementSamplingCfg(  # stage 2: states WITHIN each configuration
                    held_asset_cfg=SceneEntityCfg("held_asset"),
                    assembly_profile=FactoryAssemblyProfileCfg(),  # type: ignore[arg-type]
                    align_offset=HeldAssetAlignOffsetCfg(),  # type: ignore[arg-type]
                    placements_per_board=64,  # total scales with the library (4 x candidates)
                    placement_weights={"on_bolt": 0.5, "on_table": 0.2, "in_air": 0.3},
                    assembly_bands={
                        "near_seated": (0.0, 0.33),
                        "mid_insertion": (0.33, 0.85),
                        "above_tip": (0.85, 1.6),
                    },
                    grasp=GraspSamplingCfg(  # per placement: antipodal pairs on the held mesh
                        grasps_per_placement=8,
                        ik_seeds_per_grasp=4,  # IK starting poses tried per grasp (19%->33% solved)
                        friction_mu=0.5,
                        aperture_range=(0.002, 0.08),
                        n_pairs_retained=512,
                    ),
                ),
                robot=FactoryRobotCfg(  # stage 3: place the robot on each candidate + accept
                    asset_cfg=SceneEntityCfg("robot"),
                    ee_body_name=EndEffectorBodyCfg(),  # type: ignore[arg-type]
                    finger_body_names=FingerBodyNamesCfg(),  # type: ignore[arg-type]
                    gripper_body_names=GripperBodyNamesCfg(),  # type: ignore[arg-type]
                    solve=IKSolveCfg(
                        iterations=250,
                        refine_iterations=40,
                        pos_tol=0.004,
                        objectives=[  # soft constraint terms; membership enables (mirror of criteria)
                            JointLimitObjectiveCfg(weight=10.0),
                            JointDefaultObjectiveCfg(weight=0.0005),  # gentle arm centering (0.05 kills mm-reach)
                            FingerPinObjectiveCfg(weight=10.0),
                            CollisionAvoidanceCfg(weight=20.0, margin=0.001, n_samples=48),
                        ],
                    ),
                    criteria=[
                        JointWithinLimitCfg(limit_ratio=0.9),  # not parked against a joint stop
                        CollisionCheckCfg(n_samples=240, max_pen=0.0005, self_max_pen=0.002, adjacency_hops=2),
                    ],
                    reach=ReachRowsCfg(per_grasp=1, standoff_range=(0.03, 0.15), clearance=0.005),
                ),
            ),
            rows_per_board=30,  # table size = this x board.num_boards
            targets_per_board=30,  # goals = spread subset of each board's rows (<= rows_per_board)
            # reject reset states whose nut spawns outside the oob box (else they
            # terminate on step 0). Keep in sync with SuccessTerminationsCfg.oob.
            nut_bounds={"x": (-0.0, 1.0), "y": (-0.675, 0.675), "z": (-0.05, 1.0)},
            stash_viz_geometry=True,  # precompute silhouettes for the success-grid image logger
        ),
    )


@configclass
class FactoryRewardsCfg(PresetCfg):
    """Reward configuration preset for Factory tasks."""

    timeout_terminate: mdp_presets.TimeoutRewardsCfg = mdp_presets.TimeoutRewardsCfg()
    success_terminate: mdp_presets.SuccessRewardsCfg = mdp_presets.SuccessRewardsCfg()
    default: mdp_presets.SuccessRewardsCfg = success_terminate


@configclass
class FactoryTerminationsCfg(PresetCfg):
    """Termination configuration preset for Factory tasks."""

    timeout_terminate: mdp_presets.TimeoutTerminationsCfg = mdp_presets.TimeoutTerminationsCfg()
    success_terminate: mdp_presets.SuccessTerminationsCfg = mdp_presets.SuccessTerminationsCfg()
    default: mdp_presets.SuccessTerminationsCfg = success_terminate


@configclass
class FactoryCurriculumsCfg:
    reset_sampler = CurrTerm(
        func=mdp.success_rate_sampler,
        params={
            "success_rates_bind": "env.command_manager.get_term('reset_state').success_rates",
            "sample_indices_bind": "env.command_manager.get_term('reset_state').cmd_indices",
            "layout": StateLayoutCfg(
                coords_bind="env.command_manager.get_term('reset_state').table.state_coords",
                spawn_index_bind="env.command_manager.get_term('reset_state').table.spawn_index",
                target_index_bind="env.command_manager.get_term('reset_state').table.target_index",
            ),
            "sampling": FACTORY_RESET_SAMPLER_PRESETS,
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=50),
            "success_bind": "env.command_manager.get_term('reset_state').get_task_done()",
            # periodic success-grid + tag-matrix images (wandb/extras). The grid is
            # rasterization-bound (not cacheable -- colors change every call), so the
            # logger draws 8 states/board at dpi 70 (~1.5 s total); the period is kept
            # large to stay negligible against training throughput.
            "sampler_visual_logger": log_factory_board_grid,
            "sampler_visual_log_period": 2000,
        },
    )

    difficulty_scheduler = CurrTerm(
        func=mdp.DifficultyScheduler,
        params={
            "max_difficulty": 10,
            "success_rate_callback": "env.command_manager.get_term('reset_state').success_rates",
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
        default_shape_cfg=NewtonShapeCfg(ke=1e7, kd=1e4),
        num_substeps=2,
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
    viewer: ViewerCfg = ViewerCfg(eye=((3.5, 1.0, 0.35)), lookat=(0.0, 1.0, 0.35))
    actions: RobotActionsCfg = RobotActionsCfg()  # type: ignore
    commands: FactoryCommandsCfg = FactoryCommandsCfg()

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
