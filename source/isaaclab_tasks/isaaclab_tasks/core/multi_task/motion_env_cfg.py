# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Readable composition root for unified motion imitation."""

from __future__ import annotations

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_newton.sim.schemas import MujocoCollisionPropertiesCfg, NewtonMaterialPropertiesCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysxContactSensorCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import SensorBaseCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas import CollisionBaseCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg

from isaaclab_tasks.utils import PresetCfg, preset

from isaaclab_assets.robots.smpl.smpl_constants import MUJOCO_BODY_NAMES

from . import mdp as multi_task_mdp
from .kinematics.ik_objectives.cfg import (
    IKObjectiveJointDefaultCfg,
    IKObjectiveJointPinCfg,
    IKObjectiveMeshCollisionCfg,
    IKObjectiveMeshNonpenetrationCfg,
)
from .kinematics.newton_kinematics_cfg import NewtonKinematicsBuildCfg
from .mdp.commands.state_command.state_command_cfg import StateCommandCfg
from .motion.data.source import MotionSourceCfg
from .motion.data.sources import open_cmu_humenv_smpl_source, open_lafan_g1_source
from .motion.data.sources.amass_smplh import open_amass_smplh_source
from .motion.data.sources.lafan_bvh import open_lafan_bvh_source
from .motion.data.sources.retarget_dump_v5 import CMU_RETARGET_DUMP_SOURCE, LAFAN_RETARGET_DUMP_SOURCE
from .motion.mdp.commands.commands_cfg import (
    MotionAnalyticCoordinatesGenerateCfg,
    MotionAnalyticFamilyCfg,
    MotionConstraintGeometryFeasibleCriterionCfg,
    MotionContactCriterionCfg,
    MotionContactObjectiveCfg,
    MotionExactCoordinatesGenerateCfg,
    MotionExactFamilyCfg,
    MotionGroundPenetrationCriterionCfg,
    MotionInnerSolveConvergedCriterionCfg,
    MotionRequiredRefinementConvergedCriterionCfg,
    MotionSourceDirectionPointObjectiveCfg,
    MotionSourceEvidenceGenerateCfg,
    MotionSourceFidelityCriterionCfg,
    MotionSourceGlobalPositionObjectiveCfg,
    MotionSourceRotationObjectiveCfg,
    MotionStatePayloadCfg,
    MotionTargetCoordinateCriterionCfg,
    MotionTargetCoordinateLimitsCriterionCfg,
    MotionTaskTableCfg,
    MotionTrajectoryFamilyCfg,
    MotionTrajectorySolveCfg,
)
from .motion.robots.g1 import observations as g1_observations
from .motion.robots.g1.actions_cfg import G1JointPositionActionCfg
from .motion.robots.g1.articulation import (
    G1_BEHAVIOR_BODY_NAMES,
    G1_BEHAVIOR_JOINT_NAMES,
    G1_MOTION_ARTICULATION_CFG,
)
from .motion.robots.g1.frames import G1_HEAD_PARENT_BODY_NAME
from .motion.robots.g1.reference import g1_frame_target, g1_source_projection
from .motion.robots.g1.reset import G1ReferenceAndLieDownReset
from .motion.robots.smpl import observations as smpl_observations
from .motion.robots.smpl.articulation import SMPL_MOTION_ARTICULATION_CFG
from .motion.robots.smpl.reference import smpl_frame_target, smpl_source_projection
from .motion.robots.smpl.reset import SmplHumEnvMocapAndFallReset


@configclass
class MotionGroundCfg(PresetCfg):
    """Ground contact implementation selected by backend and randomization policy."""

    default = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.PlaneCfg(
            size=(200.0, 200.0),
            collision_props=MujocoCollisionPropertiesCfg(
                collision_enabled=True,
                margin=0.001,
                solimp=(0.99, 0.99, 0.003, 0.5, 2.0),
                solref=(0.015, 1.0),
            ),
            physics_material=NewtonMaterialPropertiesCfg(
                static_friction=preset(default=0.7, randomization_physics_observation_pose_push=1.0),  # type: ignore[arg-type]
                dynamic_friction=preset(default=0.7, randomization_physics_observation_pose_push=1.0),  # type: ignore[arg-type]
                restitution=0.0,
                torsional_friction=0.005,
                rolling_friction=0.0001,
            ),
        ),
    )
    physx = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.PlaneCfg(
            size=(200.0, 200.0),
            collision_props=CollisionBaseCfg(collision_enabled=True),
            physics_material=RigidBodyMaterialBaseCfg(
                static_friction=preset(default=0.7, randomization_physics_observation_pose_push=1.0),  # type: ignore[arg-type]
                dynamic_friction=preset(default=0.7, randomization_physics_observation_pose_push=1.0),  # type: ignore[arg-type]
                restitution=0.0,
            ),
        ),
    )


@configclass
class MotionContactSensorBackendCfg(PresetCfg):
    """Contact-sensor implementation selected independently by backend."""

    default = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=0,
    )
    physx = PhysxContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=0,
    )


@configclass
class MotionContactSensorCfg(PresetCfg):
    """Enable physical auxiliary evidence, then select its sensor backend."""

    default = None
    evidence_physical_auxiliary = MotionContactSensorBackendCfg()


@configclass
class MotionRobotCfg(PresetCfg):
    """Robot articulations selected independently from the motion source."""

    default = SMPL_MOTION_ARTICULATION_CFG
    smpl = default
    g1 = G1_MOTION_ARTICULATION_CFG


@configclass
class MotionSceneCfg(InteractiveSceneCfg):
    """Flat scene shared by all motion configurations."""

    ground: AssetBaseCfg = MotionGroundCfg()  # type: ignore[assignment]
    dome_light = AssetBaseCfg(
        prim_path="/World/domeLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )
    robot: ArticulationCfg = MotionRobotCfg()  # type: ignore[assignment]
    contact_forces: SensorBaseCfg | None = MotionContactSensorCfg()  # type: ignore[assignment]


@configclass
class MotionActionsCfg(PresetCfg):
    """Action groups selected by robot."""

    @configclass
    class SmplCfg:
        """Native MuJoCo control evaluated by MJWarp."""

        control = multi_task_mdp.NativeMujocoControlActionCfg(asset_name="robot", action_width=69)

    @configclass
    class G1Cfg:
        """G1 joint-position action route."""

        joint_position = G1JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(G1_BEHAVIOR_JOINT_NAMES),
            preserve_order=True,
            default_joint_offset_range=preset(
                default=(0.0, 0.0), randomization_physics_observation_pose_push=(-0.02, 0.02)
            ),
        )

    default = SmplCfg()
    g1 = G1Cfg()


@configclass
class MotionObservationsCfg(PresetCfg):
    """Observation groups selected by robot."""

    @configclass
    class SmplCfg:
        """SMPL 358-wide observation route."""

        @configclass
        class PolicyCfg(ObsGroup):
            observation = ObsTerm(
                func=smpl_observations.smpl_humenv_body_observation,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=list(MUJOCO_BODY_NAMES),
                        preserve_order=True,
                    )
                },
            )

            def __post_init__(self) -> None:
                self.enable_corruption = False
                self.concatenate_terms = True

        policy: PolicyCfg = PolicyCfg()

    @configclass
    class G1Cfg:
        """G1 actor, privileged, and transition routes."""

        @configclass
        class JointPositionCfg(ObsGroup):
            value = ObsTerm(
                func=g1_observations.g1_joint_pos_rel,
                params={
                    "action_name": "joint_position",
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=list(G1_BEHAVIOR_JOINT_NAMES),
                        preserve_order=True,
                    ),
                },
                noise=preset(
                    default=None,
                    randomization_physics_observation_pose_push=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
                ),
            )

            def __post_init__(self) -> None:
                self.enable_corruption = True
                self.concatenate_terms = True

        @configclass
        class JointPositionUnnoisedCfg(ObsGroup):
            value = ObsTerm(
                func=isaaclab_mdp.joint_pos_rel,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=list(G1_BEHAVIOR_JOINT_NAMES),
                        preserve_order=True,
                    ),
                },
            )

            def __post_init__(self) -> None:
                self.enable_corruption = False
                self.concatenate_terms = True

        @configclass
        class JointVelocityCfg(ObsGroup):
            value = ObsTerm(
                func=isaaclab_mdp.joint_vel,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=list(G1_BEHAVIOR_JOINT_NAMES),
                        preserve_order=True,
                    )
                },
                noise=preset(
                    default=None,
                    randomization_physics_observation_pose_push=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
                ),
            )

            def __post_init__(self) -> None:
                self.enable_corruption = True
                self.concatenate_terms = True

        @configclass
        class ProjectedGravityCfg(ObsGroup):
            value = ObsTerm(
                func=isaaclab_mdp.projected_gravity,
                noise=preset(
                    default=None,
                    randomization_physics_observation_pose_push=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
                ),
            )

            def __post_init__(self) -> None:
                self.enable_corruption = True
                self.concatenate_terms = True

        @configclass
        class BaseAngularVelocityCfg(ObsGroup):
            value = ObsTerm(
                func=isaaclab_mdp.base_ang_vel,
                noise=preset(
                    default=None,
                    randomization_physics_observation_pose_push=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
                ),
                scale=0.25,
            )

            def __post_init__(self) -> None:
                self.enable_corruption = True
                self.concatenate_terms = True

        @configclass
        class LastActionCfg(ObsGroup):
            value = ObsTerm(
                func=isaaclab_mdp.last_action,
                params={"action_name": "joint_position", "processed": True},
            )

            def __post_init__(self) -> None:
                self.enable_corruption = False
                self.concatenate_terms = True

        @configclass
        class PrivilegedStateCfg(ObsGroup):
            value = ObsTerm(
                func=g1_observations.g1_bfm_privileged_body_observation,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        body_names=list(G1_BEHAVIOR_BODY_NAMES),
                        preserve_order=True,
                    ),
                    "parent_idx": G1_BEHAVIOR_BODY_NAMES.index(G1_HEAD_PARENT_BODY_NAME),
                },
            )

            def __post_init__(self) -> None:
                self.enable_corruption = False
                self.concatenate_terms = True

        @configclass
        class TransitionCfg(PresetCfg):
            """Optional physical auxiliary evidence selected independently from the robot."""

            @configclass
            class PhysicalAuxiliaryCfg(ObsGroup):
                penalty_torques = ObsTerm(
                    func="isaaclab_tasks.core.multi_task.motion.robots.g1.actions:controller_torques_l2",
                    params={"action_name": "joint_position"},
                )
                penalty_action_rate = ObsTerm(
                    func="isaaclab_tasks.core.multi_task.motion.robots.g1.actions:controller_action_rate_l2",
                    params={"action_name": "joint_position"},
                )
                limits_dof_pos = ObsTerm(
                    func=multi_task_mdp.joint_position_limits,
                    params={
                        "soft_ratio": 0.95,
                        "asset_cfg": SceneEntityCfg(
                            "robot",
                            joint_names=list(G1_BEHAVIOR_JOINT_NAMES),
                            preserve_order=True,
                        ),
                    },
                )
                limits_torque = ObsTerm(
                    func="isaaclab_tasks.core.multi_task.motion.robots.g1.actions:controller_torque_limits",
                    params={"action_name": "joint_position", "soft_ratio": 0.95},
                )
                penalty_undesired_contact = ObsTerm(
                    func=multi_task_mdp.contact_undesired,
                    params={
                        "threshold": 1.0,
                        "sensor_cfg": SceneEntityCfg(
                            "contact_forces",
                            body_names=["pelvis", ".*shoulder.*", ".*hip.*"],
                        ),
                    },
                )
                penalty_feet_ori = ObsTerm(
                    func=multi_task_mdp.body_orientation_contact,
                    params={
                        "threshold": 1.0,
                        "sensor_cfg": SceneEntityCfg(
                            "contact_forces",
                            body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                            preserve_order=True,
                        ),
                        "asset_cfg": SceneEntityCfg(
                            "robot",
                            body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                            preserve_order=True,
                        ),
                    },
                )
                penalty_ankle_roll = ObsTerm(
                    func=multi_task_mdp.joint_position_target_l2,
                    params={
                        "target": 0.0,
                        "asset_cfg": SceneEntityCfg(
                            "robot",
                            joint_names=["left_ankle_roll_joint", "right_ankle_roll_joint"],
                            preserve_order=True,
                        ),
                    },
                )
                penalty_slippage = ObsTerm(
                    func=multi_task_mdp.body_contact_velocity,
                    params={
                        "threshold": 1.0,
                        "sensor_cfg": SceneEntityCfg(
                            "contact_forces",
                            body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                            preserve_order=True,
                        ),
                        "asset_cfg": SceneEntityCfg(
                            "robot",
                            body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                            preserve_order=True,
                        ),
                    },
                )

                def __post_init__(self) -> None:
                    self.enable_corruption = False
                    self.concatenate_terms = False

            default = None
            evidence_physical_auxiliary = PhysicalAuxiliaryCfg()

        joint_position: JointPositionCfg = JointPositionCfg()
        joint_position_unnoised: JointPositionUnnoisedCfg = JointPositionUnnoisedCfg()
        joint_velocity: JointVelocityCfg = JointVelocityCfg()
        projected_gravity: ProjectedGravityCfg = ProjectedGravityCfg()
        base_angular_velocity: BaseAngularVelocityCfg = BaseAngularVelocityCfg()
        last_action: LastActionCfg = LastActionCfg()
        privileged_state: PrivilegedStateCfg = PrivilegedStateCfg()
        transition: TransitionCfg.PhysicalAuxiliaryCfg | None = TransitionCfg()  # type: ignore[assignment]

    default = SmplCfg()
    g1 = G1Cfg()


@configclass
class MotionSourcesCfg(PresetCfg):
    """Motion datasets selected independently from robot and backend."""

    humenv_cmu = MotionSourceCfg(
        identifier="cmu_humenv_smpl",
        purpose="oracle",
        open_source=open_cmu_humenv_smpl_source,
        format="one_hdf5_file_per_clip_with_group_ep_0",
        semantic_level="smpl_robot_state_and_observation",
        decoder_version="cmu_humenv_smpl_v1",
        source_fps=30.0,
        license="amass_cmu_and_smpl_registered_source_required",
        clip_directory="data_preparation/humenv_amass",
        train=MotionSourceCfg.SplitCfg(
            name="train",
            artifact="data_preparation/test_train_split/0-CMU_train_0.1.txt",
            artifact_sha256="99929805f4ab531a89bff89837d27c403625d4b4d89d1a4d381b88825548a996",
            source_content_sha256="fe17c0673e1f5d55d985ac135e36c895df81c2cac91417df0e966fd32eb3e6b6",
            clip_count=1_638,
            frame_count=730_307,
        ),
        evaluation=MotionSourceCfg.SplitCfg(
            name="test",
            artifact="data_preparation/test_train_split/0-CMU_test_0.1.txt",
            artifact_sha256="c9b77782f5c35e0a33b3daa18c110856554acbebed00c4b2877b836d53f9b1b7",
            source_content_sha256="2621cf6d60231a1a6c319d9ab1c44d66c13bb76de4c7bad7e7bdef2d57f0ed32",
            clip_count=182,
            frame_count=88_364,
        ),
    )
    cmu = MotionSourceCfg(
        identifier="amass_cmu_smplh",
        open_source=open_amass_smplh_source,
        format="concrete_clip_rows_over_amass_smplh_npz",
        semantic_level="smplh_pose_shape",
        decoder_version="amass_smplh_clip_rows_v2",
        source_fps=None,
        license="amass_cmu_and_smpl_registered_source_required",
        clip_directory="data_preparation/AMASS/datasets/CMU",
        dependencies=(
            MotionSourceCfg.DependencyCfg(
                name="smpl_female",
                artifact="data_preparation/AMASS/models/compact/SMPL_FEMALE.npz",
                artifact_sha256="821491737fe7e0dd01e4c8a522974c09ec3a47a460cd9b416f4d824d64dbfee4",
            ),
            MotionSourceCfg.DependencyCfg(
                name="smpl_male",
                artifact="data_preparation/AMASS/models/compact/SMPL_MALE.npz",
                artifact_sha256="7f2cfa91c256cc9901157ad952a45f0904a0483f97770a983e33e0b9d2e6417b",
            ),
            MotionSourceCfg.DependencyCfg(
                name="smpl_neutral",
                artifact="data_preparation/AMASS/models/compact/SMPL_NEUTRAL.npz",
                artifact_sha256="b5c2b6e0a50e58dfba892011ba59bf00c198bc1db8a69e4a6113f9c5009081df",
            ),
        ),
        train=MotionSourceCfg.SplitCfg(
            name="train",
            artifact="data_preparation/test_train_split/0-CMU_train_raw.csv",
            artifact_sha256="ce6cfc57492f48f35f1a7ee1e6f69c1a295d0bf3bc3a20eee215d1eb0c757233",
            source_content_sha256="0ed6a96f777e11fd1b5834e5072403b6a6a01d56f189dc9de64ea9020d69d488",
            clip_count=1_638,
            frame_count=2_705_926,
        ),
        evaluation=MotionSourceCfg.SplitCfg(
            name="test",
            artifact="data_preparation/test_train_split/0-CMU_test_raw.csv",
            artifact_sha256="3f3f854f219b73464b37b8786dc4a33259e3b4b0aeb9cfb7dcf59b2966143cc8",
            source_content_sha256="583ea9b431201ecee1c51e9a9c494c86d448ef2c729e4247446270eb8d12dbe9",
            clip_count=182,
            frame_count=319_508,
        ),
    )
    default = cmu

    bfm_lafan = MotionSourceCfg(
        identifier="lafan_g1_29dof",
        open_source=open_lafan_g1_source,
        purpose="oracle",
        format="joblib_pickle_mapping_clip_name_to_field_mapping",
        semantic_level="robot_pose_g1_not_canonical_lafan",
        decoder_version="lafan_g1_29dof_v1",
        source_fps=30.0,
        license="retargeted_lafan_redistribution_requires_provenance_review",
        clip_directory=None,
        train=MotionSourceCfg.SplitCfg(
            name="training",
            artifact="humanoidverse/data/lafan_29dof_10s-clipped.pkl",
            artifact_sha256="7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010",
            source_content_sha256="7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010",
            clip_count=862,
            frame_count=258_600,
        ),
        evaluation=MotionSourceCfg.SplitCfg(
            name="evaluation",
            artifact="humanoidverse/data/lafan_29dof.pkl",
            artifact_sha256="f3a0c2810363f5c50bf4146fa2db33c1ff5b90d00cb7c0bc2aa4622696375e11",
            source_content_sha256="f3a0c2810363f5c50bf4146fa2db33c1ff5b90d00cb7c0bc2aa4622696375e11",
            clip_count=40,
            frame_count=264_705,
        ),
    )
    lafan = MotionSourceCfg(
        identifier="lafan1_bvh_ground",
        open_source=open_lafan_bvh_source,
        format="concrete_clip_rows_over_official_lafan1_bvh_zip",
        semantic_level="source_skeleton_local_pose",
        decoder_version="lafan_bvh_clip_rows_v4",
        source_fps=None,
        license="ubisoft_laforge_lafan1_research_dataset",
        clip_directory=None,
        dependencies=(
            MotionSourceCfg.DependencyCfg(
                name="lafan_zip",
                artifact="lafan1.zip",
                artifact_sha256="ea918082b500a5d158e9d3aa39039df04cd42e25f5c02fe8f7e88e8e9365a977",
            ),
        ),
        train=MotionSourceCfg.SplitCfg(
            name="ground_training_windows",
            artifact="lafan_ground_train.csv",
            artifact_sha256="5d712e5442e396f64e22f598257c338cd493e05b7f6f1f8a600aa2ae3554f841",
            source_content_sha256="430cbae6ff52756ec3e196d3f3c68274cf629464f7f4166ceb3b79f42ecdb44f",
            clip_count=862,
            frame_count=258_600,
        ),
        evaluation=MotionSourceCfg.SplitCfg(
            name="ground_evaluation_clips",
            artifact="lafan_ground_evaluation.csv",
            artifact_sha256="ca8dfa15b6e426df4353e7d4a65101e472d90b04e61f662df5d54b55057d96e5",
            source_content_sha256="74ad40e43d0b173e7cef8fc71936808fe62717df130c25a8c3fb6b325272ff04",
            clip_count=40,
            frame_count=264_705,
        ),
    )

    # BFM campaign (bfm-env-20260805) A/B CONTROL arms: explicit-token clones of the two
    # oracle sources; oracle registrations and the builder's oracle refusal stay untouched.
    humenv_cmu_control = MotionSourceCfg(
        identifier="cmu_humenv_smpl",
        purpose="training-control",
        open_source=open_cmu_humenv_smpl_source,
        format="one_hdf5_file_per_clip_with_group_ep_0",
        semantic_level="smpl_robot_state_and_observation",
        decoder_version="cmu_humenv_smpl_v1",
        source_fps=30.0,
        license="amass_cmu_and_smpl_registered_source_required",
        clip_directory="data_preparation/humenv_amass",
        train=MotionSourceCfg.SplitCfg(
            name="train",
            artifact="data_preparation/test_train_split/0-CMU_train_0.1.txt",
            artifact_sha256="99929805f4ab531a89bff89837d27c403625d4b4d89d1a4d381b88825548a996",
            source_content_sha256="fe17c0673e1f5d55d985ac135e36c895df81c2cac91417df0e966fd32eb3e6b6",
            clip_count=1_638,
            frame_count=730_307,
        ),
        evaluation=MotionSourceCfg.SplitCfg(
            name="test",
            artifact="data_preparation/test_train_split/0-CMU_test_0.1.txt",
            artifact_sha256="c9b77782f5c35e0a33b3daa18c110856554acbebed00c4b2877b836d53f9b1b7",
            source_content_sha256="2621cf6d60231a1a6c319d9ab1c44d66c13bb76de4c7bad7e7bdef2d57f0ed32",
            clip_count=182,
            frame_count=88_364,
        ),
    )
    bfm_lafan_control = MotionSourceCfg(
        identifier="lafan_g1_29dof",
        purpose="training-control",
        open_source=open_lafan_g1_source,
        format="joblib_pickle_mapping_clip_name_to_field_mapping",
        semantic_level="robot_pose_g1_not_canonical_lafan",
        decoder_version="lafan_g1_29dof_v1",
        source_fps=30.0,
        license="retargeted_lafan_redistribution_requires_provenance_review",
        clip_directory=None,
        train=MotionSourceCfg.SplitCfg(
            name="training",
            artifact="humanoidverse/data/lafan_29dof_10s-clipped.pkl",
            artifact_sha256="7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010",
            source_content_sha256="7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010",
            clip_count=862,
            frame_count=258_600,
        ),
        evaluation=MotionSourceCfg.SplitCfg(
            name="evaluation",
            artifact="humanoidverse/data/lafan_29dof.pkl",
            artifact_sha256="f3a0c2810363f5c50bf4146fa2db33c1ff5b90d00cb7c0bc2aa4622696375e11",
            source_content_sha256="f3a0c2810363f5c50bf4146fa2db33c1ff5b90d00cb7c0bc2aa4622696375e11",
            clip_count=40,
            frame_count=264_705,
        ),
    )

    cmu_retarget = CMU_RETARGET_DUMP_SOURCE  # bfm-converter-20260805 our-data arms; pins live with the decoder
    lafan_retarget = LAFAN_RETARGET_DUMP_SOURCE


@configclass
class MotionTargetKinematicsCfg(PresetCfg):
    """Target robot semantics bound to the selected scene articulation."""

    default = MotionTaskTableCfg.TargetKinematicsCfg(
        target_factory=smpl_frame_target,
        source_projection_factory=smpl_source_projection,
        calibration=MotionTaskTableCfg.TargetKinematicsCfg.CalibrationCfg(
            artifact="data_preparation/AMASS/models/compact/SMPL_NEUTRAL.npz",
            artifact_sha256="b5c2b6e0a50e58dfba892011ba59bf00c198bc1db8a69e4a6113f9c5009081df",
        ),
        asset_cfg=SceneEntityCfg("robot"),
        kinematics=NewtonKinematicsBuildCfg(collapse_fixed_joints=False),
        physics_types=(NewtonCfg,),
        contact_patches=(
            MotionTaskTableCfg.TargetKinematicsCfg.ContactPatchCfg(
                channel="left_foot",
                body_name="L_Ankle",
                points_per_body=3,
                height_band_m=0.005,
            ),
            MotionTaskTableCfg.TargetKinematicsCfg.ContactPatchCfg(
                channel="right_foot",
                body_name="R_Ankle",
                points_per_body=3,
                height_band_m=0.005,
            ),
        ),
    )
    smpl = default
    g1 = MotionTaskTableCfg.TargetKinematicsCfg(
        target_factory=g1_frame_target,
        source_projection_factory=g1_source_projection,
        asset_cfg=SceneEntityCfg("robot"),
        kinematics=NewtonKinematicsBuildCfg(collapse_fixed_joints=False),
        physics_types=(NewtonCfg, PhysxCfg),
        supports_physical_evidence=True,
        supports_randomization=True,
        contact_patches=(
            MotionTaskTableCfg.TargetKinematicsCfg.ContactPatchCfg(
                channel="left_foot",
                body_name="left_ankle_roll_link",
                points_per_body=3,
                height_band_m=0.005,
            ),
            MotionTaskTableCfg.TargetKinematicsCfg.ContactPatchCfg(
                channel="right_foot",
                body_name="right_ankle_roll_link",
                points_per_body=3,
                height_band_m=0.005,
            ),
        ),
    )


@configclass
class MotionCommandsCfg:
    """Source-to-robot task table and reset-state command."""

    @configclass
    class PayloadCfg(PresetCfg):
        """Robot-specific reset-state sampling and decoding."""

        default = MotionStatePayloadCfg(
            robot_asset_name="robot",
            reset_transform_factory=SmplHumEnvMocapAndFallReset,
            reset_transform_binds={
                "seed": "env.cfg.seed",
                "live_joint_names": "tuple(payload.robot.joint_names)",
                "physics_dt_seconds": "env.physics_dt",
                "physics_steps_per_action": "round(env.step_dt / env.physics_dt)",
            },
            reset_transform_params={
                "random_actions_high_exclusive": 5,
                "fall_pool_size": 8192,
                "initial_root_height_m": 1.0,
                "initial_root_quaternion_component_range": (0.0, 1.0),
                "control_range": (-0.5, 0.5),
            },
            root_velocity_frame="link",
            reset_sources=(("reference", 0.8), ("fall", 0.2)),
        )
        g1 = MotionStatePayloadCfg(
            robot_asset_name="robot",
            reset_transform_factory=G1ReferenceAndLieDownReset,
            reset_transform_params={
                "lie_down_root_height_m": 0.5,
                "lie_down_roll_magnitude_rad": 0.5 * math.pi,
                "lie_down_negative_roll_probability": 0.5,
            },
            root_velocity_frame="center_of_mass",
            reset_sources=(("reference", 0.7), ("lie_down", 0.3)),
        )

    motion = StateCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        reset_assets=("robot",),
        randomize_command_indices=True,
        states_relative=True,
        commands={},
        task_table=MotionTaskTableCfg(
            source=MotionSourcesCfg(),  # type: ignore[arg-type]
            contact_channels=(
                MotionTaskTableCfg.ContactChannelCfg(name="left_foot", source_probe_roles=("left_ankle", "left_toe")),
                MotionTaskTableCfg.ContactChannelCfg(
                    name="right_foot", source_probe_roles=("right_ankle", "right_toe")
                ),
            ),
            target_kinematics=MotionTargetKinematicsCfg(),  # type: ignore[arg-type]
            families=(
                MotionExactFamilyCfg(
                    name="exact",
                    generate=(MotionExactCoordinatesGenerateCfg(),),
                    solve=None,
                    criteria=(MotionTargetCoordinateCriterionCfg(),),
                ),
                MotionAnalyticFamilyCfg(
                    name="analytic",
                    generate=(MotionAnalyticCoordinatesGenerateCfg(),),
                    solve=None,
                    criteria=(MotionTargetCoordinateCriterionCfg(),),
                ),
                MotionTrajectoryFamilyCfg(
                    name="trajectory",
                    generate=(MotionSourceEvidenceGenerateCfg(),),
                    solve=MotionTrajectorySolveCfg(
                        objectives=(
                            MotionSourceGlobalPositionObjectiveCfg(weight=1.0, root_weight=10.0),
                            MotionSourceRotationObjectiveCfg(),
                            MotionSourceDirectionPointObjectiveCfg(weight=0.25),
                            MotionContactObjectiveCfg(),
                            IKObjectiveJointDefaultCfg(weight=1.0),
                            IKObjectiveJointPinCfg(weight=1.0),
                            IKObjectiveMeshCollisionCfg(weight=5.0, margin=0.03, n_samples=4),
                            IKObjectiveMeshNonpenetrationCfg(tolerance_m=0.002, maximum_penetration_m=0.0, n_samples=4),
                        ),
                        contact=MotionTrajectorySolveCfg.ContactCfg(
                            enter_height_m=0.03,
                            exit_height_m=0.06,
                            enter_speed_mps=0.15,
                            exit_speed_mps=0.30,
                            persistence_seconds=0.08,
                        ),
                        dynamics=MotionTrajectorySolveCfg.DynamicsCfg(
                            friction_coefficient=0.7,
                            iterations=96,
                            effort_weight=1.0,
                            force_regularization=1.0e-6,
                        ),
                        source_position_velocity_weight=1.0e-4,
                        source_position_acceleration_weight=1.0e-8,
                        source_rotation_velocity_weight=1.0e-4,
                        source_rotation_acceleration_weight=1.0e-8,
                        joint_default_position_weight=2.5e-3,
                        joint_temporal_velocity_weight=1.0e-4,
                        joint_temporal_acceleration_weight=1.0e-8,
                        joint_temporal_jerk_weight=1.0e-8,
                        damping=1.0e-4,
                        krylov_max_iterations=128,
                        krylov_relative_tolerance=1.0e-4,
                        kkt_relative_tolerance=1.0e-4,
                    ),
                    criteria=(
                        MotionConstraintGeometryFeasibleCriterionCfg(),
                        MotionInnerSolveConvergedCriterionCfg(),
                        MotionRequiredRefinementConvergedCriterionCfg(),
                        MotionSourceFidelityCriterionCfg(),
                        MotionContactCriterionCfg(),
                        MotionTargetCoordinateCriterionCfg(),
                        MotionTargetCoordinateLimitsCriterionCfg(),
                        MotionGroundPenetrationCriterionCfg(),
                    ),
                ),
            ),
            task_row_mode=preset(
                default="source_frames", sampling_source_rows="source_frames", sampling_clip_time="clip_time_ranges"
            ),
            source_artifact_root="",
            target_artifact_root="",
            motion_split="train",
        ),
        payload=PayloadCfg(),  # type: ignore[arg-type]
    )


@configclass
class MotionEventsCfg(PresetCfg):
    """Optional physical randomization for motion imitation."""

    @configclass
    class EmptyCfg:
        """No startup randomization or interval disturbances."""

        pass

    @configclass
    class RandomizationCfg:
        """Physical, observation, pose, and push randomization events."""

        robot_material = EventTerm(
            func=isaaclab_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.5, 1.25),
                "dynamic_friction_range": (0.5, 1.25),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1024,
            },
        )
        body_mass = EventTerm(
            func=isaaclab_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.95, 1.05),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        torso_com = EventTerm(
            func=isaaclab_mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {axis: (-0.02, 0.02) for axis in "xyz"},
            },
        )
        push = EventTerm(
            func=multi_task_mdp.RootVelocityPushDiscrete,
            mode="interval",
            interval_range_s=(1.0, 1.0),
            is_global_time=True,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "interval_seconds_range": (1, 3),
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.0),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                },
            },
        )

    default = EmptyCfg()
    randomization_physics_observation_pose_push = RandomizationCfg()


@configclass
class MotionRewardsCfg:
    """Concrete empty reward manager; FB reward channels are learner-owned."""

    pass


@configclass
class MotionTerminationsCfg:
    """Shared timeout driven by the environment's single episode clock."""

    time_out = DoneTerm(func=isaaclab_mdp.time_out, time_out=True)


@configclass
class MotionCurriculumCfg:
    """Concrete empty curriculum manager."""

    pass


@configclass
class MotionPhysicsCfg(PresetCfg):
    """Physics choices validated against each selected native robot asset."""

    default = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            integrator="implicitfast",
            use_mujoco_contacts=True,
            enable_multiccd=False,
            enable_native_ccd=False,
            tolerance=1.0e-8,
        ),
        num_substeps=1,
    )
    newton_mjwarp = default
    physx = PhysxCfg(
        solver_type=1,
        bounce_threshold_velocity=0.5,
        gpu_max_rigid_patch_count=5 * 2**20,
        gpu_found_lost_pairs_capacity=2**25,
        gpu_total_aggregate_pairs_capacity=2**25,
    )


@configclass
class MotionTimingCfg(PresetCfg):
    """Simulation timing selected independently from robot and dataset."""

    default = SimulationCfg(dt=1.0 / 450.0, render_interval=15, physics=MotionPhysicsCfg())
    timing_sim450_control30_horizon300 = default
    timing_sim200_control50_horizon501 = SimulationCfg(dt=1.0 / 200.0, render_interval=4, physics=MotionPhysicsCfg())


@configclass
class MotionImitationEnvCfg(ManagerBasedRLEnvCfg):
    """Unified motion environment selected by robot, dataset, and backend tokens."""

    scene: MotionSceneCfg = MotionSceneCfg(num_envs=1024, env_spacing=3.0, replicate_physics=True)
    sim: SimulationCfg = MotionTimingCfg()  # type: ignore[assignment]
    actions: MotionActionsCfg = MotionActionsCfg()
    observations: MotionObservationsCfg = MotionObservationsCfg()
    commands: MotionCommandsCfg = MotionCommandsCfg()
    events: MotionEventsCfg = MotionEventsCfg()
    rewards: MotionRewardsCfg = MotionRewardsCfg()
    terminations: MotionTerminationsCfg = MotionTerminationsCfg()
    curriculum: MotionCurriculumCfg = MotionCurriculumCfg()
    decimation: int = preset(default=15, timing_sim200_control50_horizon501=4)  # type: ignore[assignment]
    episode_length_s: float = preset(default=300 / 30.0, timing_sim200_control50_horizon501=501 / 50.0)  # type: ignore[assignment]
    viewer: ViewerCfg = ViewerCfg(
        eye=(3.0, 3.0, 2.0),
        lookat=(0.0, 0.0, 1.0),
        origin_type="asset_root",
        asset_name="robot",
    )

    compute_final_obs: bool = True

    def validate_config(self) -> None:
        """Reject incompatible robot capabilities and physics backends."""
        if isinstance(self.actions, PresetCfg):
            return

        physics = self.sim.physics
        target = self.commands.motion.task_table.target_kinematics
        evidence_enabled = self.scene.contact_forces is not None
        randomization_enabled = isinstance(self.events, MotionEventsCfg.RandomizationCfg)
        if evidence_enabled and not target.supports_physical_evidence:
            raise ValueError("The selected robot target does not support physical auxiliary evidence.")
        if randomization_enabled and not target.supports_randomization:
            raise ValueError("The selected robot target does not support physical and observation randomization.")
        if target.supports_physical_evidence and evidence_enabled != (self.observations.transition is not None):
            raise ValueError(
                "Physical auxiliary evidence requires both its contact sensor and transition observation group."
            )
        if not isinstance(physics, target.physics_types):
            supported = ", ".join(physics_type.__name__ for physics_type in target.physics_types)
            raise ValueError(f"The selected robot target requires one of these physics types: {supported}.")
