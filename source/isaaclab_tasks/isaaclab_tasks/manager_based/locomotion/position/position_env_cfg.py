# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.multi_task.curriculum import (
    BetaSamplingStrategyCfg,
    SamplerCfg,
    StateLayoutCfg,
    SuccessMonitorCfg,
)
from isaaclab_tasks.manager_based.multi_task.mdp.curriculums import success_rate_sampler
from isaaclab_tasks.manager_based.multi_task.mdp.terminations import illegal_contact_ratio
from isaaclab_tasks.manager_based.multi_task.sensors import FastTerrainScannerCfg
from isaaclab_tasks.manager_based.multi_task.terrain.viz.sampler_images import log_spawn_goal_sampler_images
from isaaclab_tasks.utils import PresetCfg

from . import mdp
from .mdp.commands_preset import CommandsCfg
from .mdp.curriculums import terrain_spawn_goal_pair_success_rate_levels
from .terrain_preset import SubTerrainPresetCfg


@configclass
class PositionEnvContactSensorCfg(PresetCfg):
    default = PhysXContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    newton = NewtonContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
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
            size=(10.0, 10.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=True,
            sub_terrains=SubTerrainPresetCfg(),  # type: ignore
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
    robot: ArticulationCfg = MISSING  # type: ignore

    # sensors
    # FastTerrainScanner replaces the upstream RayCaster because, when ``bound_height_scan``
    # binds it to the host articulation, its world pose is read from GPU-resident
    # ``body_pos_w``/``body_quat_w`` tensors. RayCaster always routes through Fabric, which
    # is hard-gated to cuda:0 (see ``fabric_frame_view.py``) and falls back to a 4096-prim
    # USD ``xform_cache`` loop on any other device — that fallback was the source of the
    # ~5 ms/step rank-1 slowdown observed in 2-GPU distributed training.
    height_scanner = FastTerrainScannerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=FastTerrainScannerCfg.OffsetCfg(pos=(0.5, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(2.5, 1.5)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = PositionEnvContactSensorCfg()


@configclass
class ActionsCfg:
    """Actions for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observations for the MDP (flat variant: ``height_scan`` lives in ``policy``)."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        last_actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.bound_height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            },
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class TaskCfg(ObsGroup):
        goal_point_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_point"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    task: TaskCfg = TaskCfg()


@configclass
class ObservationsEncoderCfg:
    """Observations for the MDP (encoder variant: ``height_scan`` in its own 1D group).

    Separates the flat ``height_scan`` into a dedicated group so it can be routed through a
    per-group MLP encoder (e.g. :class:`rsl_rl.models.MLPEncoderModel`) before being fused
    with the proprioceptive ``policy`` group at the main MLP head.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        last_actions = ObsTerm(func=mdp.last_action)

    @configclass
    class TaskCfg(ObsGroup):
        goal_point_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_point"})

    @configclass
    class HeightScanCfg(ObsGroup):
        height_scan = ObsTerm(
            func=mdp.bound_height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            },
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-1.0, 1.0),
        )

    policy: PolicyCfg = PolicyCfg()
    task: TaskCfg = TaskCfg()
    height_scan: HeightScanCfg = HeightScanCfg()


@configclass
class ObservationsPresetCfg(PresetCfg):
    """Selectable observation layouts for the position task.

    - ``flat`` (default): all proprioceptive observations and ``height_scan`` live in the
      ``policy`` group as a single concatenated vector.
    - ``encoder``: ``height_scan`` is moved to its own ``height_scan`` group, so an
      encoder model can route it through a dedicated sub-network before the main MLP head.
    """

    flat: ObservationsCfg = ObservationsCfg()
    encoder: ObservationsEncoderCfg = ObservationsEncoderCfg()
    # SimBa variants consume the same observation layout as the plain encoder.
    simba: ObservationsEncoderCfg = encoder
    simba_big: ObservationsEncoderCfg = encoder
    default: ObservationsEncoderCfg = encoder


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
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )


@configclass
class RewardsCfg:
    # task rewards
    success = RewTerm(func=mdp.command_success, weight=50.0)

    mech_work = RewTerm(func=mdp.mechanical_power, weight=-0.001)

    joint_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-0.01)

    foot_touchdown = RewTerm(
        func=mdp.foot_touchdown_impact,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*FOOT.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT.*"),
            "history_length": 3,
        },
    )

    undesired_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*(?:(FOOT))).*$"), "threshold": 1.0},
    )

    fail = RewTerm(
        func=mdp.is_terminated_term, params={"term_keys": ["drop", "base_contact", "abnormal_robot"]}, weight=-50.0
    )

    explore = RewTerm(func=mdp.exploration_reward, weight=0.3, params={"forward_only": True})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    drop = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -20})

    abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)

    base_contact = DoneTerm(
        func=illegal_contact_ratio,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            "threshold_ratio": 3.0,
        },
    )

    success = DoneTerm(func=mdp.success_terminate, time_out=False)


@configclass
class FootSampledCurriculumCfg:
    terrain_levels = CurrTerm(
        func=success_rate_sampler,
        params={
            "success_rates_bind": "env.command_manager.get_term('goal_point').success_rates",
            "sample_indices_bind": "env.command_manager.get_term('goal_point').cmd_indices",
            "layout": StateLayoutCfg(
                coords_bind="env.command_manager.get_term('goal_point').table.spawn_states[:, :2]",
                spawn_index_bind="env.command_manager.get_term('goal_point').table.spawn_index",
                target_index_bind="env.command_manager.get_term('goal_point').table.target_index",
                task_partition_bind="env.command_manager.get_term('goal_point').table.task_partition",
            ),
            "sampling": SamplerCfg(
                strategies=[BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0)],
                eps=1e-8,
            ),
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=20),
            "success_bind": "env.termination_manager.get_term('success')",
            "sampler_visual_logger": log_spawn_goal_sampler_images,
            "sampler_visual_log_period": 1000,
        },
    )
    remove_explore_reward = CurrTerm(func=mdp.skip_reward_term, params={"reward_term": "explore"})


@configclass
class FlatPatchCurriculumCfg:
    terrain_levels = CurrTerm(
        func=terrain_spawn_goal_pair_success_rate_levels,
        params={"kappa": 5.0, "temperature": 2.0, "target": 0.66, "success_term": "success"},
    )
    remove_explore_reward = CurrTerm(func=mdp.skip_reward_term, params={"reward_term": "explore"})


@configclass
class CurriculumCfg(PresetCfg):
    foot_sampled_commands: FootSampledCurriculumCfg = FootSampledCurriculumCfg()
    flat_patch_commands: FlatPatchCurriculumCfg = FlatPatchCurriculumCfg()
    default = foot_sampled_commands


@configclass
class PositionPhysicsCfg(PresetCfg):
    default = PhysxCfg(
        gpu_total_aggregate_pairs_capacity=2**25,
        gpu_found_lost_pairs_capacity=2**25,
        gpu_collision_stack_size=2**31,
        gpu_max_rigid_patch_count=5 * 2**20,
    )
    newton = NewtonCfg(
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
    observations: ObservationsPresetCfg = ObservationsPresetCfg()  # type: ignore
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    viewer: ViewerCfg = ViewerCfg(
        eye=(4.0 / 4, 7.0 / 4, 7.0 / 4), origin_type="asset_body", asset_name="robot", body_name="base"
    )

    def __post_init__(self):
        self.decimation = 8
        self.episode_length_s = 12.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
