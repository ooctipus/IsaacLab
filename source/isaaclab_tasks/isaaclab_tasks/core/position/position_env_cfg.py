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
from isaaclab.sensors.joint_wrench import JointWrenchSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.core.multi_task.curriculum import (
    BetaSamplingStrategyCfg,
    SamplerCfg,
    StateLayoutCfg,
    SuccessMonitorCfg,
    UniformSamplingStrategyCfg,
    ValueShiftSamplingStrategyCfg,
)
from isaaclab_tasks.core.multi_task.sensors import FastTerrainScannerCfg
from isaaclab_tasks.core.multi_task.terrain.viz.sampler_images import log_spawn_goal_sampler_images
from isaaclab_tasks.utils import PresetCfg, preset

from . import mdp
from .mdp.commands_preset import CommandsCfg
from .mdp.curriculums import terrain_spawn_goal_pair_success_rate_levels
from .terrain_preset import SubTerrainPresetCfg


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
            size=(10.0, 10.0),
            border_width=20.0,
            num_rows=10,
            num_cols=14,
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
class ObservationsCfg:
    """Observations for the MDP (flat variant: ``height_scan`` lives in ``policy``)."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        # Body-frame gravity vector with magnitude preserved [m/s^2]. Pairs with
        # ``proj_gravity`` (unit direction) so the policy can also condition on
        # ``‖g‖`` under per-env gravity randomization.
        gravity_b = ObsTerm(func=mdp.gravity_b, noise=Unoise(n_min=-0.5, n_max=0.5))
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
    """Observations for the MDP (encoder variant: ``height_scan`` in its own 2D group).

    Moves ``height_scan`` into a dedicated group emitted as a 2D ``(1, H, W)`` image by
    :func:`vision_obs` (which derives the grid resolution from the scanner's ``pattern_cfg``, so it
    adapts to per-robot scanner sizes). A single 2D layout serves both encoder kinds: an MLP encoder
    flattens it, a CNN encoder consumes it directly. Used by the ``encoder`` and all ``simba`` presets.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        gravity_b = ObsTerm(func=mdp.gravity_b, noise=Unoise(n_min=-0.5, n_max=0.5))
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        last_actions = ObsTerm(func=mdp.last_action)

    @configclass
    class TaskCfg(ObsGroup):
        goal_point_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_point"})

    @configclass
    class HeightScanCfg(ObsGroup):
        height_scan = ObsTerm(
            func=mdp.vision_obs,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
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
    - ``encoder`` / ``simba*``: ``height_scan`` is a dedicated 2D ``(1, H, W)`` group, routed through a
      per-group encoder (MLP flattens it, CNN consumes it directly) before the main residual head.
    """

    flat: ObservationsCfg = ObservationsCfg()
    encoder: ObservationsEncoderCfg = ObservationsEncoderCfg()
    # All SimBa variants share the single 2D ``height_scan`` layout; the model's encoder kind
    # (MLP vs CNN) is what differs, not the observation.
    simba: ObservationsEncoderCfg = encoder
    simba_big: ObservationsEncoderCfg = encoder
    simba_mlp: ObservationsEncoderCfg = encoder
    simba_mlp_big: ObservationsEncoderCfg = encoder
    simba_cnn: ObservationsEncoderCfg = encoder
    simba_cnn_big: ObservationsEncoderCfg = encoder
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

    # gravity_scale = EventTerm(
    #     func=mdp.randomize_physics_scene_gravity,
    #     mode="startup",
    #     params={
    #         "gravity_distribution_params": (
    #             [-0.5, -0.5, -9.81 * 1.25],
    #             [+0.5, +0.5, -9.81 * 0.75],
    #         ),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )


@configclass
class RewardsV1Cfg:
    # task rewards
    success = RewTerm(func=mdp.command_success, weight=5.0)

    mech_work = RewTerm(func=mdp.mechanical_power, weight=-0.000025)

    undesired_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.01,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*(?:(FOOT))).*$"), "threshold": 1.0},
    )


@configclass
class RewardsV2Cfg:
    reward_composer = RewTerm(
        func=mdp.reward_compose,
        weight=1.0,
        params={
            "success": RewTerm(func=mdp.command_success, weight=5.0),
            "quality": {
                "mech_work": RewTerm(func=mdp.mechanical_power, weight=-0.000025),
                "undesired_contact": RewTerm(
                    func=mdp.undesired_contacts,
                    weight=-0.01,
                    params={
                        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*(?:(FOOT))).*$"),
                        "threshold": 1.0,
                    },
                ),
            },
        },
    )


@configclass
class RewardsCfg(PresetCfg):
    rew_v1 = RewardsV1Cfg()
    rew_v2 = RewardsV2Cfg()
    default = rew_v1


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    drop = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -20})

    abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)

    base_contact = DoneTerm(
        func=mdp.illegal_contact_ratio,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            "threshold_ratio": 3.0,
        },
    )

    joint_reaction = DoneTerm(
        func=mdp.joint_reaction_overload,
        time_out=False,
        params={
            "sensor_cfg": SceneEntityCfg("joint_wrench"),
            "force_ratio": 6.0,
        },
    )

    success = DoneTerm(func=mdp.success_terminate, time_out=False)


def _make_sampler_presets(state_buffer_bind: str):
    """Build a sampler preset, parameterized by the per-task state-buffer bind.

    The ``state_buffer_bind`` only needs to evaluate to a ``[num_tasks, K]``
    tensor (the value-shift strategy uses ``shape[0]`` as the cache length).
    Different curricula expose this through different attributes on the
    ``goal_point`` term: foot-sampled uses ``table.params`` (multi-task
    :class:`RelativeStateCommand`), flat-patch uses ``spec.descretized_cmd``
    (locomotion/position :class:`RelativeStateCommand`).
    """
    return preset(
        uniform=SamplerCfg(
            strategies=[UniformSamplingStrategyCfg(weight=1.0)],
            eps=0.0,
        ),
        beta=SamplerCfg(
            strategies=[BetaSamplingStrategyCfg(target=0.66, kappa=2.5, weight=1.0, success_rate_bind="success_rates")],
            eps=1e-4,
        ),
        value_shift=SamplerCfg(
            strategies=[
                ValueShiftSamplingStrategyCfg(
                    weight=0.5,
                    state_buffer_bind=state_buffer_bind,
                    cmd_indices_bind="env.command_manager.get_term('goal_point').cmd_indices",
                    resample_command_fn_bind="env.command_manager.get_term('goal_point')._resample_command",
                    get_critic_obs_fn_bind="lambda: env.observation_manager.compute()",
                )
            ],
            eps=1e-4,
        ),
        beta_value_shift=SamplerCfg(
            strategies=[
                BetaSamplingStrategyCfg(target=0.66, kappa=2.5, weight=1.0, success_rate_bind="success_rates"),
                ValueShiftSamplingStrategyCfg(
                    weight=1.0,
                    state_buffer_bind=state_buffer_bind,
                    cmd_indices_bind="env.command_manager.get_term('goal_point').cmd_indices",
                    resample_command_fn_bind="env.command_manager.get_term('goal_point')._resample_command",
                    get_critic_obs_fn_bind="lambda: env.observation_manager.compute()",
                ),
            ],
            eps=1e-4,
        ),
        default=SamplerCfg(
            strategies=[BetaSamplingStrategyCfg(target=0.66, kappa=2.5, weight=1.0, success_rate_bind="success_rates")],
            eps=1e-4,
        ),
    )


FOOT_SAMPLED_SAMPLER_PRESETS = _make_sampler_presets(
    state_buffer_bind="env.command_manager.get_term('goal_point').table.task_partition",
)
FLAT_PATCH_SAMPLER_PRESETS = _make_sampler_presets(
    state_buffer_bind="env.command_manager.get_term('goal_point').spec.descretized_cmd",
)


@configclass
class FootSampledCurriculumCfg:
    terrain_levels = CurrTerm(
        func=mdp.success_rate_sampler,
        params={
            "success_rates_bind": "env.command_manager.get_term('goal_point').success_rates",
            "sample_indices_bind": "env.command_manager.get_term('goal_point').cmd_indices",
            "layout": StateLayoutCfg(
                coords_bind="env.command_manager.get_term('goal_point').table.spawn_states[:, :2]",
                spawn_index_bind="env.command_manager.get_term('goal_point').table.spawn_index",
                target_index_bind="env.command_manager.get_term('goal_point').table.target_index",
                task_partition_bind="env.command_manager.get_term('goal_point').table.task_partition",
            ),
            "sampling": FOOT_SAMPLED_SAMPLER_PRESETS,
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=100),
            "success_bind": "env.termination_manager.get_term('success')",
            "sampler_visual_logger": log_spawn_goal_sampler_images,
            "sampler_visual_log_period": 1000,
        },
    )


@configclass
class FlatPatchCurriculumCfg:
    terrain_levels = CurrTerm(
        func=terrain_spawn_goal_pair_success_rate_levels,
        params={
            "success_term": "success",
            "layout": StateLayoutCfg(
                coords_bind="env.command_manager.get_term('goal_point').spec.descretized_cmd[:, 0:6]",
                spawn_index_bind=(
                    "torch.arange("
                    "env.command_manager.get_term('goal_point').spec.num_descretized_cmd, "
                    "device=env.device)"
                ),
            ),
            "sampling": FLAT_PATCH_SAMPLER_PRESETS,
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=100),
        },
    )


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
