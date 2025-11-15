# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp
from .assembly_keypoints import KEYPOINTS_NISTBOARD
from . import reset_env_cfg as staging_cfg

from . import factory_assets_cfg as assets

"""
Base scene definition for Factory Tasks
"""


@configclass
class FactorySceneCfg(InteractiveSceneCfg):
    """Configuration for a factory task scene."""

    # Ground plane
    ground = assets.GROUND_CFG

    # Table
    table = assets.TABLE_CFG

    # NIST Board
    nistboard = assets.NISTBOARD_CFG

    # "FIXED ASSETS"
    bolt_m16: RigidObjectCfg = assets.BOLT_M16_CFG
    gear_base: ArticulationCfg = assets.GEAR_BASE_CFG
    hole_8mm: ArticulationCfg = assets.HOLE_8MM_CFG

    # "Moving Gears"
    small_gear: ArticulationCfg = assets.SMALL_GEAR_CFG
    large_gear: ArticulationCfg = assets.LARGE_GEAR_CFG

    # "HELD ASSETS"
    nut_m16: RigidObjectCfg = assets.NUT_M16_CFG
    medium_gear: ArticulationCfg = assets.MEDIUM_GEAR_CFG
    peg_8mm: ArticulationCfg = assets.PEG_8MM_CFG

    # Robot Override
    robot: ArticulationCfg = MISSING  # type: ignore

    # Lights
    dome_light = assets.DOMELIGHT_CFG


@configclass
class FactoryObservationsCfg:
    """Observation specifications for Factory."""

    @configclass
    class PolicyCfg(ObsGroup):
        end_effector_vel_lin_ang_b = ObsTerm(
            func=mdp.asset_link_velocity_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="end_effector"),
                "root_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        end_effector_pose = ObsTerm(
            func=mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="end_effector"),
                "root_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        held_asset_in_fixed_asset_frame: ObsTerm = ObsTerm(
            func=mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("held_asset"),
                "root_asset_cfg": SceneEntityCfg("fixed_asset"),
            },
        )

        fixed_asset_in_end_effector_frame: ObsTerm = ObsTerm(
            func=mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("fixed_asset"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="end_effector"),
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

    # mode: reset
    reset_env = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_board = EventTerm(
        func=mdp.reset_root_state_uniform_on_offset,
        mode="reset",
        params={
            "offset": KEYPOINTS_NISTBOARD.nist_board_center,
            "pose_range": {"x": (-0.00, 0.00), "y": (-0.05, 0.05), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("nistboard"),
        },
    )

    reset_fixed_asset = EventTerm(
        func=mdp.reset_fixed_assets,
        mode="reset",
        params={
            "asset_list": ["fixed_asset"],
        },
    )

    reset_strategies = EventTerm(
        func=mdp.TermChoice,
        mode="reset",
        params={
            "terms" : {
                "grasp_asset_in_air": staging_cfg.GRIPPER_GRASP_ASSET_IN_AIR,
                "start_fully_assembled": staging_cfg.FULL_ASSEMBLE_FIRST_THEN_GRIPPER_CLOSE,
                "start_assembled": staging_cfg.ASSEMBLE_FIRST_THEN_GRIPPER_CLOSE,
                "start_grasped_then_assembled": staging_cfg.GRIPPER_CLOSE_FIRST_THEN_ASSET_IN_GRIPPER
            },
            "sampling_strategy": "failure_rate"
        }
    )

    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "operation": "abs",
            "gravity_distribution_params": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        },
    )


@configclass
class FactoryRewardsCfg:
    """Reward terms for Factory"""

    # penalties
    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-1e-4)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-1e-4)

    joint_effort = RewTerm(func=mdp.joint_torques_l2, params={"asset_cfg": SceneEntityCfg("robot")}, weight=-1e-4)

    early_termination = RewTerm(func=mdp.is_terminated_term, params={"term_keys": "abnormal"}, weight=-0.01)

    reach_reward = RewTerm(func=mdp.reach_reward, weight=0.1, params={"std": 1.0})

    progress_reward_fine = RewTerm(func=mdp.progress_reward, weight=0.1, params={"std": 0.005})

    success_reward = RewTerm(func=mdp.success_reward, weight=1.0)


@configclass
class FactoryTerminationsCfg:
    """Termination terms for Factory."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    abnormal = DoneTerm(func=mdp.abnormal_robot_state)

    oob = DoneTerm(func=mdp.out_of_bound, params={
        "asset_cfg": SceneEntityCfg("held_asset"),
        "in_bound_range" : {"x": (-0.0, 1.0), "y": (-0.675, 0.675), "z": (-0.05, 1.0)}
    })

    progress_context = DoneTerm(
        func=mdp.progress_context,
        params={
            "success_threshold": 0.001,
            "held_asset_cfg": SceneEntityCfg("held_asset"),
            "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
            "held_asset_offset": MISSING,
            "fixed_asset_offset": MISSING,
        }
    )


@configclass
class FactoryCurriculumsCfg:

    difficulty_scheduler = CurrTerm(func=mdp.DifficultyScheduler, params={"max_difficulty": 10})

    gravity_adr = CurrTerm(
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
    )

##
# Environment configuration
##


@configclass
class FactoryBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the base Factory environment."""

    scene: FactorySceneCfg = FactorySceneCfg(num_envs=2, env_spacing=2.0)
    observations: FactoryObservationsCfg = FactoryObservationsCfg()
    events: FactoryEventCfg = FactoryEventCfg()
    terminations: FactoryTerminationsCfg = FactoryTerminationsCfg()
    rewards: FactoryRewardsCfg = FactoryRewardsCfg()
    curriculum: FactoryCurriculumsCfg = FactoryCurriculumsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(0.0, 0.25, 0.1), origin_type="asset_body", asset_name="robot", body_name="panda_fingertip_centered")
    actions = MISSING

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 12
        self.episode_length_s = 14.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192  # Important to avoid interpenetration.
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**32 - 1
        self.sim.physx.gpu_max_num_partitions = 1

        self.sim.physics_material.static_friction = 0.5
        self.sim.physics_material.dynamic_friction = 0.5

        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_dlssg = True


@configclass
class FactoryBaseSuccessTerminateEnvCfg(FactoryBaseEnvCfg):
    """Configuration for the base Factory environment."""
    # Post initialization
    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.success_reward.weight = 100.0
        delattr(self.rewards, "reach_reward")
        delattr(self.rewards, "progress_reward_fine")
        setattr(self.terminations, "success", DoneTerm(func=mdp.success_termination))


'''

add below code to InteractiveScene.py after filter_collsion to be a bit more efficient
from pxr import PhysxSchema, Usd, UsdPhysics, Sdf, UsdGeom

    self.setup_env_nut_bolt_filters(
        self.stage,
        physicsscene_path="/physicsScene",
        collision_root_path="/World/collisions",
        envs_root_path="/World/envs",
        ground_collision_prim_path="/World/ground/GroundPlane/CollisionPlane",
    )

    def setup_env_nut_bolt_filters(
        self,
        stage: Usd.Stage,
        physicsscene_path: str = "/physicsScene",
        collision_root_path: str = "/World/collisions",
        envs_root_path: str = "/World/envs",
        ground_collision_prim_path: str = "/World/ground/GroundPlane/CollisionPlane",
    ) -> None:
        """Setup general, per-env collision groups for nut/bolt + robot.

        Conceptual mapping:

        - G_ground                 -> 'ground_group'
        - G_robot_convex_i         -> '{env_name}_robot_convex'
        - G_static_convex_i        -> '{env_name}_static_convex'
        - G_nut_convex_i           -> '{env_name}_nut_convex'
        - G_bolt_convex_i          -> '{env_name}_bolt_convex'
        - G_nut_sdf_i              -> '{env_name}_nut_sdf'
        - G_bolt_sdf_i             -> '{env_name}_bolt_sdf'
        """

        # ------------------------------------------------------------------
        # Enable inverted collision-group filtering (general requirement)
        # ------------------------------------------------------------------
        physx_scene_prim = stage.GetPrimAtPath(physicsscene_path)
        if not physx_scene_prim:
            raise RuntimeError(f"PhysX scene '{physicsscene_path}' not found")
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physx_scene_prim)
        physx_scene.CreateInvertCollisionGroupFilterAttr().Set(True)

        with Usd.EditContext(stage, Usd.EditTarget(stage.GetRootLayer())):
            coll_root_path_sdf = Sdf.Path(collision_root_path)
            UsdGeom.Scope.Define(stage, coll_root_path_sdf)

            # ------------------------------------------------------------------
            # G_ground  (global ground group)
            # ------------------------------------------------------------------
            ground_group_path = coll_root_path_sdf.AppendChild("ground_group")
            ground_group = UsdPhysics.CollisionGroup.Define(stage, ground_group_path)
            coll_api = Usd.CollectionAPI.Apply(ground_group.GetPrim(), "colliders")
            ground_inc = coll_api.CreateIncludesRel()
            if stage.GetPrimAtPath(ground_collision_prim_path):
                ground_inc.AddTarget(ground_collision_prim_path)
            ground_filtered = ground_group.CreateFilteredGroupsRel()

            # ------------------------------------------------------------------
            # Per-env groups
            # ------------------------------------------------------------------
            envs_root = stage.GetPrimAtPath(envs_root_path)
            if not envs_root:
                raise RuntimeError(f"Env root '{envs_root_path}' not found")

            for env_prim in envs_root.GetChildren():
                env_name = env_prim.GetName()
                if not env_name.startswith("env_"):
                    continue
                env_path = env_prim.GetPath()

                # -------------------------------
                # Shape paths in this env (data)
                # -------------------------------
                bolt_convex = env_path.AppendChild("BoltAsset").AppendPath("factory_bolt_loose/Thread_convex")
                bolt_sdf    = env_path.AppendChild("BoltAsset").AppendPath("factory_bolt_loose/Thread_sdf")
                nut_convex  = env_path.AppendChild("NutAsset").AppendPath("factory_nut_loose/convex")
                nut_sdf     = env_path.AppendChild("NutAsset").AppendPath("factory_nut_loose/sdf")
                robot       = env_path.AppendChild("Robot")
                table       = env_path.AppendChild("Table")
                nist        = env_path.AppendChild("NistBoard")

                # ==============================================================
                # 1) CONVEX GROUPS  (role × convex) – one per logical role
                # ==============================================================

                # G_robot_convex_i -> '{env_name}_robot_convex'
                robot_group_path = coll_root_path_sdf.AppendChild(f"{env_name}_robot_convex")
                robot_group = UsdPhysics.CollisionGroup.Define(stage, robot_group_path)
                r_api = Usd.CollectionAPI.Apply(robot_group.GetPrim(), "colliders")
                r_inc = r_api.CreateIncludesRel()
                if stage.GetPrimAtPath(robot):
                    r_inc.AddTarget(robot)

                # G_static_convex_i -> '{env_name}_static_convex'
                static_group_path = coll_root_path_sdf.AppendChild(f"{env_name}_static_convex")
                static_group = UsdPhysics.CollisionGroup.Define(stage, static_group_path)
                s_api = Usd.CollectionAPI.Apply(static_group.GetPrim(), "colliders")
                s_inc = s_api.CreateIncludesRel()
                for p in (table, nist):
                    if stage.GetPrimAtPath(p):
                        s_inc.AddTarget(p)

                # G_nut_convex_i -> '{env_name}_nut_convex'
                nut_cvx_group_path = coll_root_path_sdf.AppendChild(f"{env_name}_nut_convex")
                nut_cvx_group = UsdPhysics.CollisionGroup.Define(stage, nut_cvx_group_path)
                ncvx_api = Usd.CollectionAPI.Apply(nut_cvx_group.GetPrim(), "colliders")
                ncvx_inc = ncvx_api.CreateIncludesRel()
                if stage.GetPrimAtPath(nut_convex):
                    ncvx_inc.AddTarget(nut_convex)

                # G_bolt_convex_i -> '{env_name}_bolt_convex'
                bolt_cvx_group_path = coll_root_path_sdf.AppendChild(f"{env_name}_bolt_convex")
                bolt_cvx_group = UsdPhysics.CollisionGroup.Define(stage, bolt_cvx_group_path)
                bcvx_api = Usd.CollectionAPI.Apply(bolt_cvx_group.GetPrim(), "colliders")
                bcvx_inc = bcvx_api.CreateIncludesRel()
                if stage.GetPrimAtPath(bolt_convex):
                    bcvx_inc.AddTarget(bolt_convex)

                # --------------------------------------------------------------
                # Convex filteredGroups wiring (this is the group↔group table)
                # --------------------------------------------------------------
                # Remember: with inverted filtering, collisions happen only if
                # BOTH groups whitelist each other.

                # G_robot_convex_i collides with: itself, static, nut_convex, bolt_convex, ground
                r_f = robot_group.CreateFilteredGroupsRel()
                r_f.AddTarget(robot_group_path)
                r_f.AddTarget(static_group_path)
                r_f.AddTarget(nut_cvx_group_path)
                r_f.AddTarget(bolt_cvx_group_path)
                r_f.AddTarget(ground_group_path)

                # G_static_convex_i collides with: itself, robot, nut_convex, bolt_convex, ground
                s_f = static_group.CreateFilteredGroupsRel()
                s_f.AddTarget(static_group_path)
                s_f.AddTarget(robot_group_path)
                s_f.AddTarget(nut_cvx_group_path)
                s_f.AddTarget(bolt_cvx_group_path)
                s_f.AddTarget(ground_group_path)

                # G_nut_convex_i collides with: itself, robot, static, ground
                # (NO bolt_convex here -> no convex nut–bolt)
                ncvx_f = nut_cvx_group.CreateFilteredGroupsRel()
                ncvx_f.AddTarget(nut_cvx_group_path)
                ncvx_f.AddTarget(robot_group_path)
                ncvx_f.AddTarget(static_group_path)
                ncvx_f.AddTarget(ground_group_path)

                # G_bolt_convex_i collides with: itself, robot, static, ground
                # (NO nut_convex here -> no convex nut–bolt)
                bcvx_f = bolt_cvx_group.CreateFilteredGroupsRel()
                bcvx_f.AddTarget(bolt_cvx_group_path)
                bcvx_f.AddTarget(robot_group_path)
                bcvx_f.AddTarget(static_group_path)
                bcvx_f.AddTarget(ground_group_path)

                # G_ground collides with all convex groups (all envs)
                ground_filtered.AddTarget(robot_group_path)
                ground_filtered.AddTarget(static_group_path)
                ground_filtered.AddTarget(nut_cvx_group_path)
                ground_filtered.AddTarget(bolt_cvx_group_path)

                # ==============================================================
                # 2) SDF GROUPS (nut/bolt threading only)
                # ==============================================================

                # G_nut_sdf_i -> '{env_name}_nut_sdf'
                nut_sdf_group_path = coll_root_path_sdf.AppendChild(f"{env_name}_nut_sdf")
                nut_sdf_group = UsdPhysics.CollisionGroup.Define(stage, nut_sdf_group_path)
                n_api = Usd.CollectionAPI.Apply(nut_sdf_group.GetPrim(), "colliders")
                n_inc = n_api.CreateIncludesRel()
                if stage.GetPrimAtPath(nut_sdf):
                    n_inc.AddTarget(nut_sdf)
                n_filtered = nut_sdf_group.CreateFilteredGroupsRel()

                # G_bolt_sdf_i -> '{env_name}_bolt_sdf'
                bolt_sdf_group_path = coll_root_path_sdf.AppendChild(f"{env_name}_bolt_sdf")
                bolt_sdf_group = UsdPhysics.CollisionGroup.Define(stage, bolt_sdf_group_path)
                b_api = Usd.CollectionAPI.Apply(bolt_sdf_group.GetPrim(), "colliders")
                b_inc = b_api.CreateIncludesRel()
                if stage.GetPrimAtPath(bolt_sdf):
                    b_inc.AddTarget(bolt_sdf)
                b_filtered = bolt_sdf_group.CreateFilteredGroupsRel()

                # SDF nut <-> SDF bolt only (threading)
                # No SDF vs convex or SDF vs ground unless you add it on purpose.
                n_filtered.AddTarget(bolt_sdf_group_path)
                b_filtered.AddTarget(nut_sdf_group_path)
'''