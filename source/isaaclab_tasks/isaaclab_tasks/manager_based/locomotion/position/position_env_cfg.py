# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp, terrains


@configclass
class SceneCfg(InteractiveSceneCfg):
    """ "Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
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
            sub_terrains={},
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
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.5, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, debug_vis=True
    )


@configclass
class ActionsCfg:
    """Actions for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class CommandsCfg:
    "Command specifications for the MDP."

    goal_point = mdp.TerrainBasedPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        simple_heading=False,
        debug_vis=True,
        ranges=mdp.TerrainBasedPose2dCommandCfg.Ranges(
            heading=(-3.14, 3.14),
        ),
    )


@configclass
class ObservationsCfg:
    """Observations for the MDP"""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        goal_point_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_point"})
        time_left = ObsTerm(func=mdp.time_left)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        last_actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    # startup
    physical_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.8, 0.8),
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

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:

    # task rewards
    success_reward = RewTerm(func=mdp.is_terminated_term, params={"term_keys": "success"}, weight=250)

    # penalties
    # action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)

    energy = RewTerm(func=mdp.work, weight=-0.5)

    failure_terminal = RewTerm(
        func=mdp.is_terminated_term, params={"term_keys": ["robot_drop", "base_contact"]}, weight=-25.0
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    robot_drop = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -20})

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )

    success = DoneTerm(func=mdp.success, params={"std": (0.4, 0.5)})


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)  # type: ignore


def make_terrain(terrain_dict):

    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
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
            sub_terrains=terrain_dict,
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


variants = {
    "scene.terrain": {
        "all": make_terrain({
            "gap": terrains.GAP,
            "pit": terrains.PIT,
            "extreme_stair": terrains.EXTREME_STAIR,
            "slope_inv": terrains.SLOPE_INV,
            "stepping_stone": terrains.STEPPING_STONE,
            "radiating_beam": terrains.RADIATING_BEAM,
        }),
        "eval": make_terrain({
            "gap": terrains.GAP.replace(gap_width_range=(1.0, 1.5)),
            "pit": terrains.PIT.replace(pit_depth_range=(0.8, 1.2)),
            "extreme_stair": terrains.EXTREME_STAIR.replace(step_height_range=(0.12, 0.2)),
            "slope_inv": terrains.SLOPE_INV.replace(slope_range=(0.6, 0.9)),
            "stepping_stone": terrains.STEPPING_STONE.replace(
                w_gap=(0.15, 0.26),
                w_stone=(0.4, 0.2),
                s_max=(0.080, 0.118),
                h_max=(0.075, 0.1)
            ),
            "radiating_beam": terrains.RADIATING_BEAM.replace(num_bars=(5, 1)),
        }),
        "gap": make_terrain({"gap": terrains.GAP}),
        "pit": make_terrain({"pit": terrains.PIT}),
        "extreme_stair": make_terrain({"extreme_stair": terrains.EXTREME_STAIR}),
        "slope_inv": make_terrain({"slope_inv": terrains.SLOPE_INV}),
        "square_pillar_obstacle": make_terrain({"square_pillar_obstacle": terrains.SQUARE_PILLAR_OBSTACLE}),
        "stepping_stone": make_terrain({"stepping_stone": terrains.STEPPING_STONE}),
        "radiating_beam": make_terrain({"radiating_beam": terrains.RADIATING_BEAM}),
    },
    "curriculum.terrain_levels": {
        "success_rate": CurrTerm(func=mdp.terrain_success_rate_levels),
        "success_rate_fine_grained": CurrTerm(func=mdp.terrain_spawn_goal_pair_success_rate_levels)
    },
}


@configclass
class LocomotionPositionCommandEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=10)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(4.0, 7.0, 7.0), origin_type="asset_body", asset_name="robot", body_name="base")
    variants = variants

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 6.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**25
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**25
        self.sim.physx.gpu_collision_stack_size = 2**31
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**20

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
