# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.assemble_trocar import mdp

from isaaclab_tasks.contrib.assemble_trocar.config import G1_29DOF_DEX3_CFG  # isort: skip
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg, MultiBackendSceneCfg, MultiBackendSimulationCfg

joint_names = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]
offset_dict = {
    "left_elbow_joint": -0.3,
    "right_elbow_joint": -0.3,
}

HEALTHCARE_S3 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/Healthcare/0.5.0/132c82d"
USD_ROOT = f"{HEALTHCARE_S3}/Props/LightWheel"
_ISAAC_RTX_CFG = IsaacRtxRendererCfg(
    global_settings=IsaacRtxRendererGlobalSettingsCfg(
        enable_translucency=True,
        carb_settings={"rtx.raytracing.fractionalCutoutOpacity": True},
        antialiasing_mode="DLAA",
    )
)
_RTX_RENDERER_CFG = MultiBackendRendererCfg(default=_ISAAC_RTX_CFG, isaacsim_rtx=_ISAAC_RTX_CFG)


@configclass
class AssembleTrocarSceneCfg(MultiBackendSceneCfg):
    """Scene configuration for the assemble_trocar task (robot + objects + lights)."""

    # humanoid robot configuration
    robot: ArticulationCfg = G1_29DOF_DEX3_CFG.replace(
        init_state=G1_29DOF_DEX3_CFG.init_state.replace(pos=(-1.84919, 1.94, 0.81168), rot=(0.0, 0.0, 0.0, 1.0))
    )
    front_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/d435_link/front_cam",
        update_period=0.02,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.5, focus_distance=400.0, horizontal_aperture=14.25, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        renderer_cfg=_RTX_RENDERER_CFG,
    )
    left_wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_camera_base_link/left_wrist_camera",
        update_period=0.02,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=400.0, horizontal_aperture=14.25, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.04012, -0.07441, 0.15711), rot=(0.00539, 0.86024, 0.0424, 0.50809), convention="ros"
        ),
        renderer_cfg=_RTX_RENDERER_CFG,
    )
    right_wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_camera_base_link/right_wrist_camera",
        update_period=0.02,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=400.0, horizontal_aperture=14.25, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.04012, 0.07441, 0.15711), rot=(0.00539, 0.86024, 0.0424, 0.50809), convention="ros"
        ),
        renderer_cfg=_RTX_RENDERER_CFG,
    )

    scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=UsdFileCfg(
            usd_path=f"{USD_ROOT}/scene03.usd",
        ),
    )

    trocar_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/trocar_1",
        spawn=UsdFileCfg(
            usd_path=f"{USD_ROOT}/Assets/Trocar002/Trocar002-xform-wo.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,
                rest_offset=-0.001,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-1.60202, 1.91362, 0.87183],
            rot=[-0.0, 0.70711, 0.70711, 0.0],
        ),
    )

    trocar_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/trocar_2",
        spawn=UsdFileCfg(
            usd_path=(
                f"{USD_ROOT}/Assets/"
                "DisposableLaparoscopicPunctureDevice001/"
                "DisposableLaparoscopicPunctureDevice005-xform.usd"
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            rot=[-0.71475, -0.000243, 0.05853, 0.69692], pos=[-1.50635, 1.90997, 0.8631]
        ),
    )
    trocar_tips = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/trocar_1",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/trocar_1/Trocar002/White_pos", name="trocar_1_tip"),
            FrameTransformerCfg.FrameCfg(
                prim_path=("{ENV_REGEX_NS}/trocar_2/DisposableLaparoscopicPunctureDevice001/Red_pos"),
                name="trocar_2_tip",
            ),
        ],
    )
    tray = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/surgical_tray",
        spawn=UsdFileCfg(
            usd_path=f"{USD_ROOT}/Assets/SurgicalTray001/SurgicalTray001.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=[-1.54919, 2.03365, 0.84554], rot=[0.0, 0.0, -0.70711, 0.70711]),
        actuators={},  # Empty dict for passive articulation (no motors)
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=1000.0,
        ),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """defines the action configuration related to robot control, using direct joint angle control"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=joint_names,
        scale=1.0,
        use_default_offset=False,
        offset=offset_dict,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """defines all available observation information"""

    @configclass
    class PolicyCfg(ObsGroup):
        """policy group observation configuration class
        defines all state observation values for policy decision
        inherit from ObsGroup base class
        """

        # robot joint state observation
        robot_joint_state = ObsTerm(func=mdp.get_robot_body_joint_states)
        # dex3 hand joint state observation
        robot_dex3_joint_state = ObsTerm(func=mdp.get_robot_dex3_joint_states)

        def __post_init__(self):
            """post initialization function
            set the basic attributes of the observation group
            """
            self.enable_corruption = False  # disable observation value corruption
            self.concatenate_terms = False  # disable observation item connection

    @configclass
    class CameraImagesCfg(ObsGroup):
        """Observations from the robot's cameras."""

        front_camera = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("front_camera"), "data_type": "rgb", "normalize": False},
        )
        left_wrist_camera = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("left_wrist_camera"), "data_type": "rgb", "normalize": False},
        )
        right_wrist_camera = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("right_wrist_camera"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.concatenate_terms = False

    # observation groups
    # create policy observation group instance
    policy: PolicyCfg = PolicyCfg()
    camera_images: CameraImagesCfg = CameraImagesCfg()


@configclass
class TerminationsCfg:
    """Termination conditions for the environment."""

    # Time out termination
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Task success termination (all stages completed)
    task_success = DoneTerm(
        func=mdp.task_success_termination,
        time_out=False,  # This is a success termination, not a failure
        params={
            "print_log": False,
            "success_stage": 4,
        },
    )
    object_drop = DoneTerm(
        func=mdp.object_drop_termination,
        time_out=True,  # Treat as timeout/failure
        params={
            "drop_height_threshold": 0.5,  # Objects below this Z height are considered dropped
            "asset_cfg1": SceneEntityCfg("trocar_1"),
            "asset_cfg2": SceneEntityCfg("trocar_2"),
        },
    )


@configclass
class RewardsCfg:
    """Reward configuration for sparse reward mode.

    Each stage gives 1.0 reward on completion -> Total reward for full task = 4.0
    This ensures clear reward signal for each stage transition.

    ``update_stage`` runs first (weight=0) to advance the task stage before any
    reward term reads it, removing implicit ordering dependencies.
    """

    # Stage machine — weight=0, runs before all reward terms to update task stage
    update_stage = RewTerm(
        func=mdp.update_task_stage,
        weight=0.0,
        params={
            "asset_cfg1": SceneEntityCfg("trocar_1"),
            "asset_cfg2": SceneEntityCfg("trocar_2"),
            "tip_frames_cfg": SceneEntityCfg("trocar_tips"),
            "table_height": 0.85483,
            "lift_threshold": 0.15,
            "tip_align_threshold": 0.015,
            "insertion_dist_threshold": 0.05,
            "insertion_angle_threshold": 0.15,
            "placement_x_min": -1.8,
            "placement_x_max": -1.4,
            "placement_y_min": 1.5,
            "placement_y_max": 1.8,
            "print_log": False,
        },
    )

    # Stage 0: Lift trocars
    lift_trocars = RewTerm(
        func=mdp.lift_trocars_reward,
        weight=1.0,
        params={
            "table_height": 0.85483,
            "lift_threshold": 0.15,
            "asset_cfg1": SceneEntityCfg("trocar_1"),
            "asset_cfg2": SceneEntityCfg("trocar_2"),
            "use_sparse_reward": True,
            "print_log": False,
        },
    )

    # Stage 1: Tip alignment (find hole)
    tip_alignment = RewTerm(
        func=mdp.trocar_tip_alignment_reward,
        weight=1.0,  # Give 1.0 reward when stage 1->2 completes
        params={
            "tip_dist_std": 0.02,  # Std for tip distance reward shaping
            "asset_cfg1": SceneEntityCfg("trocar_1"),
            "asset_cfg2": SceneEntityCfg("trocar_2"),
            "tip_frames_cfg": SceneEntityCfg("trocar_tips"),
            "use_sparse_reward": True,
            "print_log": False,
        },
    )

    # Stage 2: Insertion (push in)
    insert_trocars = RewTerm(
        func=mdp.trocar_insertion_reward,
        weight=1.0,  # Give 1.0 reward when stage 2->3 completes
        params={
            "angle_std": 0.2,  # Std for angle alignment reward
            "angle_threshold": 0.10,  # ~5.7 degrees tolerance for parallelism
            "center_dist_std": 0.05,  # Std for center distance reward
            "asset_cfg1": SceneEntityCfg("trocar_1"),
            "asset_cfg2": SceneEntityCfg("trocar_2"),
            "use_sparse_reward": True,
            "print_log": False,
        },
    )

    # Stage 3: Placement (place in tray)
    placement_trocars = RewTerm(
        func=mdp.trocar_placement_reward,
        weight=1.0,  # Give 1.0 reward when stage 3->4 completes
        params={
            "x_min": -1.8,
            "x_max": -1.4,
            "y_min": 1.5,
            "y_max": 1.8,
            "asset_cfg1": SceneEntityCfg("trocar_1"),
            "asset_cfg2": SceneEntityCfg("trocar_2"),
            "use_sparse_reward": True,
            "print_log": False,
        },
    )


@configclass
class EventCfg:
    """Event configuration for scene reset."""

    # Reset scene when episode terminates (timeout or success)
    reset_scene = EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset")

    # Reset task stage tracker when environment resets
    reset_task_stage = EventTermCfg(func=mdp.reset_task_stage, mode="reset")

    # Random rotation for tray and trocars
    reset_tray_random_rotation = EventTermCfg(
        func=mdp.reset_tray_with_random_rotation,
        mode="reset",
        params={
            "tray_cfg": SceneEntityCfg("tray"),
            "trocar_1_cfg": SceneEntityCfg("trocar_1"),
            "trocar_2_cfg": SceneEntityCfg("trocar_2"),
            "rotation_range": [0, 10],
        },
    )


@configclass
class G1AssembleTrocarEnvCfg(ManagerBasedRLEnvCfg):
    """Unitree G1 robot assemble trocar environment configuration class
    inherits from ManagerBasedRLEnvCfg, defines all configuration parameters for the entire environment
    """

    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg()
    # scene settings
    scene: AssembleTrocarSceneCfg = AssembleTrocarSceneCfg(
        num_envs=1,
        env_spacing=6.0,
        replicate_physics=True,
    )
    # basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    num_rerenders_on_reset: int = 1

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(bounce_threshold_velocity=0.01)


@configclass
class EventCfgFixTrayRotation(EventCfg):
    """Event configuration with a deterministic-but-different yaw per env index.

    This is useful for eval with many parallel envs:
      - env 0..N-1 get different yaw angles,
      - for a fixed global seed, the set of N angles is reproducible across runs/resets.

    Notes:
        - Determinism is tied to torch's global seed (set by env reset seed in IsaacLab).
        - Angle unit is degrees.
    """

    reset_tray_random_rotation = EventTermCfg(
        func=mdp.reset_tray_with_random_rotation,
        mode="reset",
        params={
            "tray_cfg": SceneEntityCfg("tray"),
            "trocar_1_cfg": SceneEntityCfg("trocar_1"),
            "trocar_2_cfg": SceneEntityCfg("trocar_2"),
            "rotation_range": [0, 10],
            "deterministic_per_env": True,
            # Use torch.initial_seed() by default to follow the env reset seed.
            "deterministic_seed": None,
        },
    )


@configclass
class G1AssembleTrocarEvalEnvCfg(G1AssembleTrocarEnvCfg):
    """Eval-friendly env cfg.

    This is currently an alias of `G1AssembleTrocarEnvCfg`, but registered under a
    separate Gym id for compatibility with RLinf configs.
    """

    # Override events to enforce deterministic per-env tray yaw on every reset.
    events: EventCfgFixTrayRotation = EventCfgFixTrayRotation()
