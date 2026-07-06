# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import patterns
from isaaclab.sensors.joint_wrench import JointWrenchSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

# Multi-task-local fast height scanner — fresh impl that subclasses only ``SensorBase``,
# specialized for static terrain (no per-step mesh transform updates), with a single fused
# kernel that does sensor pose + world-frame ray transform + multi-mesh closest-hit raycast
# in one launch. Fully self-contained under ``multi_task/sensors/`` so a future rebase onto
# a new IsaacLab version requires no merge work in shared sensor code.
from isaaclab_tasks.core.multi_task.sensors import FastTerrainScannerCfg
from isaaclab_tasks.utils import PresetCfg, preset

from .kinematics import NewtonKinematicsBuildCfg
from .kinematics.ik_objectives.cfg import (
    BodyPointsCfg,
    EntityPositionCfg,
    EntityRotationCfg,
    IKObjectiveGravityTorqueCfg,
    IKObjectiveJointDefaultCfg,
    IKObjectiveJointLimitCfg,
    IKObjectiveMeshCollisionCfg,
    IKObjectivePositionCfg,
    IKObjectiveRotationCfg,
    IKObjectiveStabilityMarginCfg,
)
from .terrain import mdp, mdp_presets
from .terrain.mdp_presets.command_presets import CommandPayloadPresetCfg, CommandsPresetCfg
from .terrain.mdp_presets.robots.robot_presets import FootBodyNamesCfg, RetargetLateralHipJointPatternCfg
from .terrain.retarget.cfg import PatchSamplingCfg, SamplerCfg, SamplerSizingCfg
from .terrain.retarget.criteria_cfg import (
    CollisionCheckCfg,
    FootPositionErrorCfg,
    JointWithinLimitCfg,
    LateralHipLimitCfg,
    SolverCostOutlierCfg,
    SupportPolygonStabilityCfg,
)
from .terrain.retarget.feature_extractors import XYZYawFeatures


@configclass
class PositionEnvContactSensorCfg(PresetCfg):
    default = PhysXContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    newton_mjwarp = NewtonContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    physx = default


@configclass
class SceneCfg(InteractiveSceneCfg):
    """ "Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        use_terrain_origins=True,
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(10.0, 10.0),
            border_width=20.0,
            num_rows=10,
            num_cols=14,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=True,
            sub_terrains=mdp_presets.SubTerrainPresetCfg(),  # type: ignore
        ),
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # robots
    robot: ArticulationCfg = mdp_presets.RobotArticulationCfg()  # type: ignore

    # sensors
    height_scanner = FastTerrainScannerCfg(
        prim_path=mdp_presets.HeightScannerPrimPathCfg(),  # type: ignore
        offset=FastTerrainScannerCfg.OffsetCfg(pos=(0.5, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.075, size=(2.5, 1.5)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = PositionEnvContactSensorCfg()
    joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ActionsCfg:
    """Actions for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True)


@configclass
class CommandsCfg:
    """Position command variants and complete terrain-state construction."""

    goal_point = mdp.StateCommandCfg(
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        reset_assets=("robot",),
        randomize_command_indices=False,
        states_relative=False,
        commands=CommandsPresetCfg(),  # type: ignore[arg-type]
        task_table=mdp.TaskTableCfg(
            kinematics=NewtonKinematicsBuildCfg(collapse_fixed_joints=False),
            pool_spacing=0.5,
            families=(
                mdp.PositionTerrainStanceFamilyCfg(
                    generate=(
                        mdp.PositionTerrainStanceGenerateCfg(
                            sampler=SamplerCfg(
                                patch=PatchSamplingCfg(
                                    contact_radius=0.04,
                                    max_height_diff=0.03,
                                    horizontal_scale=0.01,
                                    oversample_ratio=5.0,
                                ),
                                sizing=SamplerSizingCfg(criteria_yield=0.10),
                                min_contacts=3,
                                terrain_snap_distance=0.2,
                                outward_snap_penalty=1.0,
                            ),
                            foot_body_names=FootBodyNamesCfg(),  # type: ignore[arg-type]
                        ),
                    ),
                    solve=mdp.PositionIKSolveCfg(
                        objectives=(
                            IKObjectivePositionCfg(
                                name="foot_targets",
                                current=BodyPointsCfg(asset="robot", bodies=FootBodyNamesCfg()),  # type: ignore[arg-type]
                                target_bind="generated.foot_targets",
                                weight=1.0,
                            ),
                            IKObjectivePositionCfg(
                                name="base_position",
                                current=EntityPositionCfg(asset="robot"),
                                target_bind="generated.base_position",
                                weight=0.05,
                            ),
                            IKObjectiveRotationCfg(
                                name="base_rotation",
                                current=EntityRotationCfg(asset="robot"),
                                target_bind="generated.base_rotation",
                                weight=0.5,
                            ),
                            IKObjectiveJointLimitCfg(weight=10.0),
                            IKObjectiveMeshCollisionCfg(weight=2.0, margin=0.05, n_samples=4),
                            IKObjectiveStabilityMarginCfg(weight=1.0),
                            IKObjectiveGravityTorqueCfg(weight=0.02),
                            IKObjectiveJointDefaultCfg(weight=0.5),
                        ),
                        max_iterations=200,
                    ),
                    criteria=(
                        CollisionCheckCfg(n_samples=16, max_pen=0.02),
                        JointWithinLimitCfg(limit_ratio=0.9),
                        LateralHipLimitCfg(
                            joint_pattern=RetargetLateralHipJointPatternCfg(),  # type: ignore[arg-type]
                            max_angle=1.05,
                        ),
                        SupportPolygonStabilityCfg(),
                        FootPositionErrorCfg(max_err=0.4, aggregate="sum"),
                        SolverCostOutlierCfg(threshold_multiplier=3.0),
                    ),
                    selection=mdp.PositionFpsSelectionCfg(features=XYZYawFeatures(yaw_scale=0.1)),
                ),
            ),
            pairing=mdp.PositionSameCellPairingCfg(max_spawns_per_cell=20, num_targets_per_cell=20),
        ),
        payload=CommandPayloadPresetCfg(),  # type: ignore[arg-type]
    )


@configclass
class EventsCfg:
    # startup
    physical_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.5),
            "dynamic_friction_range": (0.4, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=mdp_presets.BaseBodyNameCfg()),  # type: ignore
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )


@configclass
class PositionPhysicsCfg(PresetCfg):
    default = PhysxCfg(
        gpu_total_aggregate_pairs_capacity=2**25,
        gpu_found_lost_pairs_capacity=2**25,
        gpu_collision_stack_size=2**31,
        gpu_max_rigid_patch_count=5 * 2**20,
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=250,
            nconmax=100,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=1,
        debug_mode=False,
        default_shape_cfg=NewtonShapeCfg(margin=0.02),
    )
    physx = default


@configclass
class LocomotionPositionCommandEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=0.0)
    sim: SimulationCfg = SimulationCfg(physics=PositionPhysicsCfg())  # type: ignore
    observations: mdp_presets.ObservationsCfg = mdp_presets.ObservationsCfg()  # type: ignore
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: mdp_presets.RewardsCfg = mdp_presets.RewardsCfg()
    terminations: mdp_presets.TerminationsCfg = mdp_presets.TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: mdp_presets.CurriculumPresetCfg = mdp_presets.CurriculumPresetCfg()
    viewer: ViewerCfg = ViewerCfg(
        eye=(4.0 / 4, 7.0 / 4, 7.0 / 4),
        origin_type="asset_body",
        asset_name="robot",
        body_name=mdp_presets.BaseBodyNameCfg(),  # type: ignore
    )

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.use_newton_actuators = preset(default=False, newton_mjwarp=True)

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
