# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPairingCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_physx.physics import PhysxCfg

from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.multi_task.curriculum import ObservationCache, StateLayoutCfg, SuccessMonitorCfg
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
    CollisionCheckCfg,
    FactoryApproachTargetGenerateCfg,
    FactoryAssemblyPoseGenerateCfg,
    FactoryFamilyCfg,
    FactoryFpsSelectionCfg,
    FactoryFreePoseGenerateCfg,
    FactoryGeometryCfg,
    FactoryGraspTargetGenerateCfg,
    FactoryHeldPoseBoundsCriterionCfg,
    FactoryIKSolveCfg,
    FactoryRobotCfg,
    FactoryRobotSeedGenerateCfg,
    FactorySupportPoseGenerateCfg,
    FactoryTargetErrorCriterionCfg,
    GraspSamplingCfg,
    JointWithinLimitCfg,
)
from .kinematics import NewtonKinematicsBuildCfg
from .kinematics.ik_objectives.cfg import (
    BodyPointsCfg,
    IKObjectiveJointDefaultCfg,
    IKObjectiveJointLimitCfg,
    IKObjectiveJointPinCfg,
    IKObjectivePositionCfg,
)


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
class FactoryResetAssetsCfg(PresetCfg):
    """Complete reset-entity tuple selected by the independent task variant."""

    default: tuple[str, ...] = ("robot", "nistboard", "fixed_asset", "held_asset")
    gear_mesh_small: tuple[str, ...] = default + ("medium_gear", "large_gear")
    gear_mesh_medium: tuple[str, ...] = default + ("small_gear", "large_gear")
    gear_mesh_large: tuple[str, ...] = default + ("small_gear", "medium_gear")


@configclass
class FactoryCommandsCfg:
    """Command specifications for Factory."""

    reset_state = mdp.StateCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        reset_assets=FactoryResetAssetsCfg(),  # type: ignore[arg-type]
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
            held_asset_cfg=SceneEntityCfg("held_asset"),
            fixed_asset_cfg=SceneEntityCfg("fixed_asset"),
            robot_cfg=SceneEntityCfg("robot"),
            symmetry=HeldAssetSymmetryCfg(),  # type: ignore[arg-type]
        ),
        task_table=mdp.FactoryResetStateTableCfg(
            kinematics=NewtonKinematicsBuildCfg(collapse_fixed_joints=False),
            geometry=FactoryGeometryCfg(
                held_asset_cfg=SceneEntityCfg("held_asset"),
                board=BoardLibraryCfg(
                    board_asset_cfg=SceneEntityCfg("nistboard"),
                    fixed_asset_cfg=SceneEntityCfg("fixed_asset"),
                    fixed_asset_map=FixedAssetMapCfg(),  # type: ignore[arg-type]
                    num_boards=16,
                    library_oversample=4.0,
                    oversample=10.0,
                    pose_range={
                        "x": (-0.1, 0.1),
                        "y": (-0.1, 0.1),
                        "z": (0.0, 0.1),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.8, 0.8),
                    },
                ),
                robot=FactoryRobotCfg(
                    asset_cfg=SceneEntityCfg("robot"),
                    ee_body_name=EndEffectorBodyCfg(),  # type: ignore[arg-type]
                    finger_body_names=FingerBodyNamesCfg(),  # type: ignore[arg-type]
                    gripper_body_names=GripperBodyNamesCfg(),  # type: ignore[arg-type]
                ),
            ),
            families=(
                FactoryFamilyCfg(
                    name="assembly_grasp",
                    fraction=0.25,
                    candidate_oversample=1280.0,
                    generate=(
                        FactoryAssemblyPoseGenerateCfg(
                            assembly_profile=FactoryAssemblyProfileCfg(),  # type: ignore[arg-type]
                            align_offset=HeldAssetAlignOffsetCfg(),  # type: ignore[arg-type]
                            assembly_bands={
                                "near_seated": (0.0, 0.33),
                                "mid_insertion": (0.33, 0.85),
                                "above_tip": (0.85, 1.6),
                            },
                        ),
                        FactoryGraspTargetGenerateCfg(
                            sampling=GraspSamplingCfg(
                                friction_mu=0.5,
                                aperture_range=(0.002, 0.08),
                                n_pairs_retained=512,
                            ),
                            grasps_per_placement=8,
                        ),
                        FactoryRobotSeedGenerateCfg(ik_seeds_per_grasp=4),
                    ),
                    solve=FactoryIKSolveCfg(
                        objectives=(
                            IKObjectivePositionCfg(
                                name="grasp",
                                current=BodyPointsCfg(
                                    asset="robot",
                                    bodies=FingerBodyNamesCfg(),  # type: ignore[arg-type]
                                ),
                                target_bind="generated.grasp_points",
                                weight=1.0,
                            ),
                            IKObjectiveJointLimitCfg(weight=10.0),
                            IKObjectiveJointDefaultCfg(weight=0.025),
                            IKObjectiveJointPinCfg(weight=10.0),
                        ),
                    ),
                    criteria=(
                        FactoryTargetErrorCriterionCfg(max_error_m=0.004),
                        FactoryHeldPoseBoundsCriterionCfg(
                            bounds={"x": (0.0, 1.0), "y": (-0.675, 0.675), "z": (-0.05, 1.0)}
                        ),
                        JointWithinLimitCfg(limit_ratio=0.8),
                        CollisionCheckCfg(n_samples=240, max_pen=0.0005, self_max_pen=0.002, adjacency_hops=2),
                    ),
                    selection=FactoryFpsSelectionCfg(position_frame="fixed_asset"),
                ),
                FactoryFamilyCfg(
                    name="assembly_approach",
                    fraction=0.25,
                    candidate_oversample=80.0,
                    generate=(
                        FactoryAssemblyPoseGenerateCfg(
                            assembly_profile=FactoryAssemblyProfileCfg(),  # type: ignore[arg-type]
                            align_offset=HeldAssetAlignOffsetCfg(),  # type: ignore[arg-type]
                            assembly_bands={
                                "near_seated": (0.0, 0.33),
                                "mid_insertion": (0.33, 0.85),
                                "above_tip": (0.85, 1.6),
                            },
                        ),
                        FactoryGraspTargetGenerateCfg(
                            sampling=GraspSamplingCfg(
                                friction_mu=0.5,
                                aperture_range=(0.002, 0.08),
                                n_pairs_retained=512,
                            ),
                            grasps_per_placement=8,
                        ),
                        FactoryRobotSeedGenerateCfg(ik_seeds_per_grasp=4),
                        FactoryApproachTargetGenerateCfg(standoff_range=(0.03, 0.15), clearance=0.005),
                    ),
                    solve=FactoryIKSolveCfg(
                        objectives=(
                            IKObjectivePositionCfg(
                                name="grasp",
                                current=BodyPointsCfg(
                                    asset="robot",
                                    bodies=FingerBodyNamesCfg(),  # type: ignore[arg-type]
                                ),
                                target_bind="generated.grasp_points",
                                weight=1.0,
                            ),
                            IKObjectiveJointLimitCfg(weight=10.0),
                            IKObjectiveJointDefaultCfg(weight=0.025),
                            IKObjectiveJointPinCfg(weight=10.0),
                        ),
                    ),
                    criteria=(
                        FactoryTargetErrorCriterionCfg(max_error_m=0.004),
                        FactoryHeldPoseBoundsCriterionCfg(
                            bounds={"x": (0.0, 1.0), "y": (-0.675, 0.675), "z": (-0.05, 1.0)}
                        ),
                        JointWithinLimitCfg(limit_ratio=0.8),
                        CollisionCheckCfg(n_samples=240, max_pen=0.0005, self_max_pen=0.002, adjacency_hops=2),
                    ),
                    selection=FactoryFpsSelectionCfg(position_frame="fixed_asset"),
                ),
                FactoryFamilyCfg(
                    name="support_grasp",
                    fraction=0.1,
                    candidate_oversample=1280.0,
                    generate=(
                        FactorySupportPoseGenerateCfg(
                            pose_range={"x": (0.25, 0.6), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)},
                            table_height=0.04,
                        ),
                        FactoryGraspTargetGenerateCfg(
                            sampling=GraspSamplingCfg(
                                friction_mu=0.5,
                                aperture_range=(0.002, 0.08),
                                n_pairs_retained=512,
                            ),
                            grasps_per_placement=8,
                        ),
                        FactoryRobotSeedGenerateCfg(ik_seeds_per_grasp=4),
                    ),
                    solve=FactoryIKSolveCfg(
                        objectives=(
                            IKObjectivePositionCfg(
                                name="grasp",
                                current=BodyPointsCfg(
                                    asset="robot",
                                    bodies=FingerBodyNamesCfg(),  # type: ignore[arg-type]
                                ),
                                target_bind="generated.grasp_points",
                                weight=1.0,
                            ),
                            IKObjectiveJointLimitCfg(weight=10.0),
                            IKObjectiveJointDefaultCfg(weight=0.025),
                            IKObjectiveJointPinCfg(weight=10.0),
                        ),
                    ),
                    criteria=(
                        FactoryTargetErrorCriterionCfg(max_error_m=0.004),
                        FactoryHeldPoseBoundsCriterionCfg(
                            bounds={"x": (0.0, 1.0), "y": (-0.675, 0.675), "z": (-0.05, 1.0)}
                        ),
                        JointWithinLimitCfg(limit_ratio=0.8),
                        CollisionCheckCfg(n_samples=240, max_pen=0.0005, self_max_pen=0.002, adjacency_hops=2),
                    ),
                    selection=FactoryFpsSelectionCfg(position_frame="world", position_axes=(0, 1)),
                ),
                FactoryFamilyCfg(
                    name="support_approach",
                    fraction=0.1,
                    candidate_oversample=80.0,
                    generate=(
                        FactorySupportPoseGenerateCfg(
                            tag="on_table",
                            pose_range={"x": (0.25, 0.6), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)},
                            table_height=0.04,
                        ),
                        FactoryGraspTargetGenerateCfg(
                            sampling=GraspSamplingCfg(
                                friction_mu=0.5,
                                aperture_range=(0.002, 0.08),
                                n_pairs_retained=512,
                            ),
                            grasps_per_placement=8,
                        ),
                        FactoryRobotSeedGenerateCfg(ik_seeds_per_grasp=4),
                        FactoryApproachTargetGenerateCfg(standoff_range=(0.03, 0.15), clearance=0.005),
                    ),
                    solve=FactoryIKSolveCfg(
                        objectives=(
                            IKObjectivePositionCfg(
                                name="grasp",
                                current=BodyPointsCfg(
                                    asset="robot",
                                    bodies=FingerBodyNamesCfg(),  # type: ignore[arg-type]
                                ),
                                target_bind="generated.grasp_points",
                                weight=1.0,
                            ),
                            IKObjectiveJointLimitCfg(weight=10.0),
                            IKObjectiveJointDefaultCfg(weight=0.025),
                            IKObjectiveJointPinCfg(weight=10.0),
                        ),
                    ),
                    criteria=(
                        FactoryTargetErrorCriterionCfg(max_error_m=0.004),
                        FactoryHeldPoseBoundsCriterionCfg(
                            bounds={"x": (0.0, 1.0), "y": (-0.675, 0.675), "z": (-0.05, 1.0)}
                        ),
                        JointWithinLimitCfg(limit_ratio=0.8),
                        CollisionCheckCfg(n_samples=240, max_pen=0.0005, self_max_pen=0.002, adjacency_hops=2),
                    ),
                    selection=FactoryFpsSelectionCfg(position_frame="world", position_axes=(0, 1)),
                ),
                FactoryFamilyCfg(
                    name="free_grasp",
                    fraction=0.3,
                    candidate_oversample=1280.0,
                    generate=(
                        FactoryFreePoseGenerateCfg(
                            tag="in_air",
                            pose_range={
                                "x": (-0.15, 0.5),
                                "y": (-0.5, 0.5),
                                "z": (0.015, 0.2),
                                "roll": (-1.57, 1.57),
                                "pitch": (-1.57, 1.57),
                                "yaw": (-3.14, 3.14),
                            },
                        ),
                        FactoryGraspTargetGenerateCfg(
                            sampling=GraspSamplingCfg(
                                friction_mu=0.5,
                                aperture_range=(0.002, 0.08),
                                n_pairs_retained=512,
                            ),
                            grasps_per_placement=8,
                        ),
                        FactoryRobotSeedGenerateCfg(ik_seeds_per_grasp=4),
                    ),
                    solve=FactoryIKSolveCfg(
                        objectives=(
                            IKObjectivePositionCfg(
                                name="grasp",
                                current=BodyPointsCfg(
                                    asset="robot",
                                    bodies=FingerBodyNamesCfg(),  # type: ignore[arg-type]
                                ),
                                target_bind="generated.grasp_points",
                                weight=1.0,
                            ),
                            IKObjectiveJointLimitCfg(weight=10.0),
                            IKObjectiveJointDefaultCfg(weight=0.025),
                            IKObjectiveJointPinCfg(weight=10.0),
                        ),
                    ),
                    criteria=(
                        FactoryTargetErrorCriterionCfg(max_error_m=0.004),
                        FactoryHeldPoseBoundsCriterionCfg(
                            bounds={"x": (0.0, 1.0), "y": (-0.675, 0.675), "z": (-0.05, 1.0)}
                        ),
                        JointWithinLimitCfg(limit_ratio=0.8),
                        CollisionCheckCfg(n_samples=240, max_pen=0.0005, self_max_pen=0.002, adjacency_hops=2),
                    ),
                    selection=FactoryFpsSelectionCfg(position_frame="world"),
                ),
            ),
            rows_per_board=50,  # table size = this x board.num_boards
            targets_per_board=50,  # goals = spread subset of each board's rows (<= rows_per_board)
            # eval-only slot filter: keep only specific (spawn_tag -> target_tag)
            # placement-tag pairs. ``seated_air`` evaluates just the seated<->in-air
            # transitions; default keeps every pair (training unchanged).
            allowed_tag_pairs=preset(
                default=None,
                seated_air=[("near_seated", "in_air"), ("in_air", "near_seated")],
            ),
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
        },
    )

    difficulty_scheduler = CurrTerm(
        func=mdp.DifficultyScheduler,
        params={
            "max_difficulty": 10,
            "success_rate_callback": "env.curriculum_manager.get_term('reset_sampler').success_rates",
        },
    )

    goal_observations = preset(
        default=None,
        successor=CurrTerm(
            func=ObservationCache,
            params={"observations_bind": ("materialize_state_command_target_observations(env, 'reset_state')")},
        ),
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
            nconmax=400,
            impratio=1.0,
            cone="pyramidal",
            update_data_interval=2,
            ls_parallel=False,
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            broad_phase="sap",
            max_triangle_pairs=60_000_000,
            rigid_contact_max=7_000_000,
        ),
        default_shape_cfg=NewtonShapeCfg(ke=1e7, kd=1e4),
        # Isolate the fine ``sdf`` threads into a private mate group (they collide
        # only with each other; the coarse ``hull`` carries world contacts), filter
        # the two solid hulls so they cannot wedge, and route convex/box colliders
        # through the planar-SDF kernel.
        collision_pairing=NewtonCollisionPairingCfg(
            mate=[(r"(?i)nut.*/sdf", r"(?i)bolt.*/sdf")],
            forbid=[(r"(?i)nut.*/hull", r"(?i)bolt.*/hull")],
            convex_sdf_resolution=64,
        ),
        num_substeps=8,
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
    viewer: ViewerCfg = ViewerCfg(eye=((2.5, 0.0, 0.35)), lookat=(0.0, 0.0, 0.35))
    actions: RobotActionsCfg = RobotActionsCfg()  # type: ignore
    commands: FactoryCommandsCfg = FactoryCommandsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
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
