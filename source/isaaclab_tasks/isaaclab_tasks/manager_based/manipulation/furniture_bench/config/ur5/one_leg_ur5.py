from __future__ import annotations

from dataclasses import MISSING

import isaaclab_assets.robots.ur5 as ur5
from isaaclab.utils.assets import LOCAL_ASSET_PATH_DIR


import isaaclab.envs.mdp as orbit_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ... import assembly_data
from ... import mdp as task_mdp


@configclass
class ObjectSceneCfg(InteractiveSceneCfg):
    robot = ur5.IMPLICIT_UR5.replace(prim_path="{ENV_REGEX_NS}/Robot")

    leg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/SquareTableLeg1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/FurnitureBench/SquareLeg/square_leg.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0, disable_gravity=False
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.022)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    table_top: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/SquareTableTop",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/FurnitureBench/SquareTableTop/square_table_top.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0, kinematic_enabled=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Table
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, -0.881), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/Mounts/UWPatVention/pat_vention.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        ),
    )

    ur5_metal_support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/UR5MetalSupport",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0, -0.013), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSET_PATH_DIR}/Props/Mounts/UWPatVention2/Ur5MetalSupport/ur5plate.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        ),
    )

    # override ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.868)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=10000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for randomization."""

    # mode: startup (randomize dynamics)
    robot_material = EventTerm(
        func=orbit_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("robot"),
            "make_consistent": True,
        },
    )

    leg_material = EventTerm(
        func=orbit_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.2, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("leg"),
            "make_consistent": True,
        },
    )

    table_top_material = EventTerm(
        func=orbit_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.2, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("table_top"),
            "make_consistent": True,
        },
    )

    table_material = EventTerm(
        func=orbit_mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.6),
            "dynamic_friction_range": (0.2, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "asset_cfg": SceneEntityCfg("table"),
            "make_consistent": True,
        },
    )

    randomize_robot_mass = EventTerm(
        func=orbit_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_leg_mass = EventTerm(
        func=orbit_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("leg"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_table_top_mass = EventTerm(
        func=orbit_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("table_top"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_table_mass = EventTerm(
        func=orbit_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    randomize_robot_joint_parameters = EventTerm(
        func=task_mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*", "finger_joint"]),
            "friction_distribution_params": (0.0, 0.2),
            "armature_distribution_params": (0.0, 0.05),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    randomize_robot_actuator_parameters = EventTerm(
        func=task_mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*", "finger_joint"]),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # mode: reset
    reset_everything = EventTerm(
        func=orbit_mdp.reset_scene_to_default,
        mode="reset",
        params={},
    )

    reset_robot_position = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.01, 0.01),
                "y": (-0.059, -0.019),
                "z": (-0.01, 0.01),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfgs": {
                "robot": SceneEntityCfg("robot"),
                "ur5_metal_support": SceneEntityCfg("ur5_metal_support")
            }
        },
    )

@configclass
class TrainEventCfg(EventCfg):
    reset_from_init_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "datasets": [
                "furniture_datasets/insertion_init_states_dataset_preprocessed.pt",
                "furniture_datasets/assembledgrasped_init_states_dataset_preprocessed.pt"
            ],
            "probs": [0.5, 0.5], #[1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "failure_rate_sampling": False
        }
    )


@configclass
class EvalEventCfg(EventCfg):
    reset_from_init_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "datasets": [
                "furniture_datasets/insertion_init_states_dataset.hdf5",
                "furniture_datasets/assembledgrasped_init_states_dataset.hdf5",
            ],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "failure_rate_sampling": False
        }
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    pass


@configclass
class RelativeJointPositionActionCfg:
    actions: ur5.Ur5RelativeJointPositionAction = ur5.Ur5RelativeJointPositionAction()


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    task_command = task_mdp.TaskCommandCfg(
        asset_cfg=SceneEntityCfg("robot", body_names="body"),
        resampling_time_range=(1e6, 1e6),
        success_threshold=0.0015,
        held_asset_cfg=SceneEntityCfg("leg"),
        fixed_asset_cfg=SceneEntityCfg("table_top"),
        held_asset_offset=assembly_data.KEYPOINTS_TABLELEG.center_axis_bottom,
        fixed_asset_offset=assembly_data.KEYPOINTS_TABLETOPHOLE.hole0_leg_assembled_offset,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        prev_actions = ObsTerm(func=task_mdp.last_action)

        joint_pos = ObsTerm(func=task_mdp.joint_pos)

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "target_asset_offset": assembly_data.KEYPOINTS_ROBOTIQGRIPPER.offset,
            },
        )

        held_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("leg"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_offset": assembly_data.KEYPOINTS_ROBOTIQGRIPPER.offset,
            },
        )

        fixed_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("table_top"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "target_asset_offset": assembly_data.KEYPOINTS_TABLETOPHOLE.hole0_tip_offset,
            },
        )

        held_asset_in_fixed_asset_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("leg"),
                "root_asset_cfg": SceneEntityCfg("table_top"),
                "root_asset_offset": assembly_data.KEYPOINTS_TABLETOPHOLE.hole0_tip_offset,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations for policy group."""

        prev_actions = ObsTerm(func=task_mdp.last_action)

        joint_pos = ObsTerm(func=task_mdp.joint_pos)

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "target_asset_offset": assembly_data.KEYPOINTS_ROBOTIQGRIPPER.offset,
            },
        )

        held_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("leg"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_offset": assembly_data.KEYPOINTS_ROBOTIQGRIPPER.offset,
            },
        )

        fixed_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("table_top"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "target_asset_offset": assembly_data.KEYPOINTS_TABLETOPHOLE.hole0_tip_offset,
            },
        )

        held_asset_in_fixed_asset_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("leg"),
                "root_asset_cfg": SceneEntityCfg("table_top"),
                "root_asset_offset": assembly_data.KEYPOINTS_TABLETOPHOLE.hole0_tip_offset,
            },
        )

        # privileged observations

        joint_vel = ObsTerm(func=task_mdp.joint_vel)

        end_effector_vel_lin_ang_b = ObsTerm(
            func=task_mdp.asset_link_velocity_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        joint_effort = ObsTerm(func=task_mdp.joint_effort)

        joint_force = ObsTerm(func=task_mdp.joint_force, params={"asset_cfg": SceneEntityCfg("robot")})

        robot_material_properties = ObsTerm(func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("robot")})

        leg_material_properties = ObsTerm(func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("leg")})

        table_top_material_properties = ObsTerm(func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("table_top")})

        table_material_properties = ObsTerm(func=task_mdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("table")})

        robot_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("robot")})

        leg_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("leg")})

        table_top_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("table_top")})

        table_mass = ObsTerm(func=task_mdp.get_mass, params={"asset_cfg": SceneEntityCfg("table")})

        robot_joint_friction = ObsTerm(func=task_mdp.get_joint_friction, params={"asset_cfg": SceneEntityCfg("robot")})

        robot_joint_armature = ObsTerm(func=task_mdp.get_joint_armature, params={"asset_cfg": SceneEntityCfg("robot")})

        robot_joint_stiffness = ObsTerm(func=task_mdp.get_joint_stiffness, params={"asset_cfg": SceneEntityCfg("robot")})

        robot_joint_damping = ObsTerm(func=task_mdp.get_joint_damping, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class DataCollectionObservationsCfg:
    @configclass
    class DataCollectionPolicyCfg(ObsGroup):
        """Observations for data collection group."""

        # helper
        last_arm_joint_pos = ObsTerm(
            func=task_mdp.last_joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["shoulder.*", "elbow.*", "wrist.*"]
                ),
            }
        )

        # policy
        last_arm_action = ObsTerm(
            func=task_mdp.last_action,
            params={
                "action_name": "jointpos",
            }
        )

        last_gripper_action = ObsTerm(
            func=task_mdp.last_action,
            params={
                "action_name": "gripper",
            }
        )

        arm_joint_pos = ObsTerm(
            func=task_mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["shoulder.*", "elbow.*", "wrist.*"]
                ),
            }
        )

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "target_asset_offset": assembly_data.KEYPOINTS_ROBOTIQGRIPPER.offset,
            },
        )

        held_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("leg"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "root_asset_offset": assembly_data.KEYPOINTS_ROBOTIQGRIPPER.offset,
            },
        )

        fixed_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("table_top"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
                "target_asset_offset": assembly_data.KEYPOINTS_TABLETOPHOLE.hole0_tip_offset,
            },
        )

        held_asset_in_fixed_asset_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("leg"),
                "root_asset_cfg": SceneEntityCfg("table_top"),
                "root_asset_offset": assembly_data.KEYPOINTS_TABLETOPHOLE.hole0_tip_offset,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False
            self.history_length = 1

    # observation groups
    policy: DataCollectionPolicyCfg = DataCollectionPolicyCfg()

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)

    action_rate = RewTerm(func=task_mdp.action_rate_l2_clamped, weight=-1e-4)

    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2_clamped,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"])},
    )

    abnormal_robot = RewTerm(func=task_mdp.abnormal_robot_state, weight=-100.0)

    # progress rewards
    progress_context = RewTerm(
        func=task_mdp.ProgressContext,  # type: ignore
        weight=0.1,
        params={
            "held_asset_cfg": SceneEntityCfg("leg"),
            "fixed_asset_cfg": SceneEntityCfg("table_top"),
            "held_asset_offset": assembly_data.KEYPOINTS_TABLELEG.center_axis_bottom,
            "fixed_asset_offset": assembly_data.KEYPOINTS_TABLETOPHOLE.hole0_leg_assembled_offset,
        },
    )

    ee_held_distance = RewTerm(
        func=task_mdp.ee_held_asset_distance_tanh,
        weight=0.1,
        params={
            "root_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "target_asset_cfg": SceneEntityCfg("leg"),
            "root_asset_offset": assembly_data.KEYPOINTS_ROBOTIQGRIPPER.offset,
            "target_asset_offset": assembly_data.KEYPOINTS_TABLELEG.graspable,
            "std": 1.0,
            "failure_rate_weight": "env.reward_manager.get_term_cfg('progress_context').func.failure_rate",
        },
    )

    progress_reward = RewTerm(
        func=task_mdp.progress_dense,
        weight=0.1,
        params={
            "std": 1.0,
            "failure_rate_weight": "env.reward_manager.get_term_cfg('progress_context').func.failure_rate",
        }
    )

    success_reward = RewTerm(func=task_mdp.success_reward, weight=1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=orbit_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)


@configclass
class DataCollectionTerminationsCfg:

    time_out = DoneTerm(func=orbit_mdp.time_out, time_out=True)

    success = DoneTerm(func=task_mdp.consecutive_success_state, params={"num_consecutive_successes": 10})

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)


@configclass
class OneLegUr5(ManagerBasedRLEnvCfg):
    scene: ObjectSceneCfg = ObjectSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = MISSING  # type: ignore
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(3.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        self.decimation = 12
        self.episode_length_s = 4.0
        # simulation settings
        self.sim.dt = 1 / 120.0

        # Contact and solver settings
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.02
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005

        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**31

        # Render settings
        self.sim.render.enable_dlssg = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True


@configclass
class OneLegUr5IkAbs(OneLegUr5):
    actions: ur5.Ur5IkAbsoluteAction = ur5.Ur5IkAbsoluteAction()
    events: TrainEventCfg = TrainEventCfg()


@configclass
class OneLegUr5RelJointPosition(OneLegUr5):
    actions: ur5.Ur5RelativeJointPositionAction = ur5.Ur5RelativeJointPositionAction()
    events: TrainEventCfg = TrainEventCfg()


@configclass
class OneLegUr5EvalRelJointPosition(OneLegUr5):
    actions: ur5.Ur5RelativeJointPositionAction = ur5.Ur5RelativeJointPositionAction()
    events: EvalEventCfg = EvalEventCfg()


@configclass
class OneLegUr5EvalRelUnscaledJointPosition(OneLegUr5):
    actions: ur5.Ur5RelativeJointPositionActionUnscaled = ur5.Ur5RelativeJointPositionActionUnscaled()
    events: EvalEventCfg = EvalEventCfg()
    observations: DataCollectionObservationsCfg = DataCollectionObservationsCfg()
    terminations: DataCollectionTerminationsCfg = DataCollectionTerminationsCfg()


@configclass
class OneLegUr5RelJointPositionDataCollection(OneLegUr5):
    actions: ur5.Ur5RelativeJointPositionActionClipped = ur5.Ur5RelativeJointPositionActionClipped()
    events: TrainEventCfg = TrainEventCfg()
    observations: DataCollectionObservationsCfg = DataCollectionObservationsCfg()
    terminations: DataCollectionTerminationsCfg = DataCollectionTerminationsCfg()
