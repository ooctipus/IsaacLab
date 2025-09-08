from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as orbit_mdp

import isaaclab_assets.robots.ur5 as ur5

from ... import assembly_data
from ... import mdp as task_mdp

from .one_leg_ur5 import ObjectSceneCfg, EventCfg, OneLegUr5, DataCollectionTerminationsCfg


@configclass
class RGBTiledObjectSceneCfg(ObjectSceneCfg):
    # background
    curtain_left = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainLeft",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.68, 0.519), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 1.0, 1.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 0.0)
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            )
        ),
    )

    curtain_back = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainBack",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.15, 0.0, 0.519), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 1.3, 1.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 0.0)
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            )
        ),
    )

    curtain_right = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainRight",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.68, 0.519), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 1.0, 1.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 0.0)
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            )
        ),
    )

    front_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rgb_front_camera",
        update_period=0,
        height=480,
        width=640,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0052094, -0.00899991, 0.5105068),
            rot=(0.66004321, 0.25644884, 0.26165847, 0.65582909),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=13.50
        )
    )

    side_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rgb_side_camera",
        update_period=0,
        height=480,
        width=640,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.86820596, 0.5783852, 0.2649344),
            rot=(0.30984667, 0.21713063, 0.5323381, 0.75727503),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=21.8
        )
    )

    wrist_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera",
        update_period=0,
        height=480,
        width=640,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0182505, -0.00408447, -0.0689107),
            rot=(0.34254336, -0.61819255, -0.6160212, 0.347879),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.55
        )
    )


@configclass
class RGBTiledEventCfg(EventCfg):
    """Configuration for randomization."""

    # randomize camera pose
    randomize_front_camera = EventTerm(
        func=task_mdp.randomize_tiled_cameras,
        mode="reset",
        params={
            "camera_path_template": "/World/envs/env_{}/Robot/rgb_front_camera",
            # Base values from TiledCameraCfg
            "base_position": (1.0052094, -0.00899991, 0.5105068),
            "base_rotation": (0.66004321, 0.25644884, 0.26165847, 0.65582909),
            # Delta ranges for position (in meters)
            "position_deltas": {
                "x": (-0.04, 0.04),
                "y": (-0.04, 0.04),
                "z": (-0.04, 0.04)
            },
            # Delta ranges for euler angles (in degrees)
            "euler_deltas": {
                "pitch": (-2.0, 2.0),
                "yaw": (-2.0, 2.0),
                "roll": (-2.0, 2.0)
            }
        }
    )

    randomize_front_camera_focal_length = EventTerm(
        func=task_mdp.randomize_camera_focal_length,
        mode="reset",
        params={
            "camera_path_template": "/World/envs/env_{}/Robot/rgb_front_camera",
            "focal_length_range": (13.2, 13.8)  # Range from wide-angle to telephoto
        },
    )

    randomize_side_camera = EventTerm(
        func=task_mdp.randomize_tiled_cameras,
        mode="reset",
        params={
            "camera_path_template": "/World/envs/env_{}/Robot/rgb_side_camera",
            # Base values from TiledCameraCfg
            "base_position": (0.86820596, 0.5783852, 0.2649344),
            "base_rotation": (0.30984667, 0.21713063, 0.5323381, 0.75727503),
            # Delta ranges for position (in meters)
            "position_deltas": {
                "x": (-0.04, 0.04),
                "y": (-0.04, 0.04),
                "z": (-0.04, 0.04)
            },
            # Delta ranges for euler angles (in degrees)
            "euler_deltas": {
                "pitch": (-2.0, 2.0),
                "yaw": (-2.0, 2.0),
                "roll": (-2.0, 2.0)
            }
        }
    )

    randomize_side_camera_focal_length = EventTerm(
        func=task_mdp.randomize_camera_focal_length,
        mode="reset",
        params={
            "camera_path_template": "/World/envs/env_{}/Robot/rgb_side_camera",
            "focal_length_range": (21.5, 22.1)  # Range from wide-angle to telephoto
        },
    )

    randomize_wrist_camera = EventTerm(
        func=task_mdp.randomize_tiled_cameras,
        mode="reset",
        params={
            "camera_path_template": "/World/envs/env_{}/Robot/robotiq_base_link/rgb_wrist_camera",
            # Base values from TiledCameraCfg
            "base_position": (0.0182505, -0.00408447, -0.0689107),
            "base_rotation": (0.34254336, -0.61819255, -0.6160212, 0.347879),
            # Delta ranges for position (in meters)
            "position_deltas": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01)
            },
            # Delta ranges for euler angles (in degrees)
            "euler_deltas": {
                "pitch": (-2.5, 2.5),
                "yaw": (-2.5, 2.5),
                "roll": (-2.5, 2.5)
            }
        }
    )

    randomize_wrist_camera_focal_length = EventTerm(
        func=task_mdp.randomize_camera_focal_length,
        mode="reset",
        params={
            "camera_path_template": "/World/envs/env_{}/Robot/robotiq_base_link/rgb_wrist_camera",
            "focal_length_range": (23.05, 26.05)  # Range from wide-angle to telephoto
        },
    )

    # reset colors
    randomize_wrist_mount_color = EventTerm(
        func=orbit_mdp.randomize_visual_color,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "colors": {
                "r": (0.0, 1.0),  # Red component range
                "g": (0.0, 1.0),  # Green component range
                "b": (0.0, 1.0)   # Blue component range
            },
            "event_name": "randomize_wrist_mount_color_event",
            "mesh_name": "robotiq_base_link/visuals/D415_to_Robotiq_Mount"
        },
    )

    randomize_inner_finger_color = EventTerm(
        func=task_mdp.randomize_visual_color_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "colors": {
                "r": (0.0, 1.0),
                "g": (0.0, 1.0),
                "b": (0.0, 1.0)
            },
            "event_name": "randomize_inner_finger_color_event",
            "mesh_names": ["left_inner_finger/visuals/mesh_1", "right_inner_finger/visuals/mesh_1"]
        },
    )

    randomize_leg_color = EventTerm(
        func=orbit_mdp.randomize_visual_color,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("leg"),
            "colors": {
                "r": (0.0, 1.0),  # Red component range
                "g": (0.0, 1.0),  # Green component range
                "b": (0.0, 1.0)   # Blue component range
            },
            "event_name": "randomize_leg_color_event",
            "mesh_name": ""  # Empty string to target the whole asset
        },
    )

    randomize_table_top_color = EventTerm(
        func=orbit_mdp.randomize_visual_color,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("table_top"),
            "colors": {
                "r": (0.0, 1.0),  # Red component range
                "g": (0.0, 1.0),  # Green component range
                "b": (0.0, 1.0)   # Blue component range
            },
            "event_name": "randomize_table_top_color_event",
            "mesh_name": ""  # Empty string to target the whole asset
        },
    )

    randomize_table_color = EventTerm(
        func=orbit_mdp.randomize_visual_color,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "colors": {
                "r": (0.0, 1.0),
                "g": (0.0, 1.0),
                "b": (0.0, 1.0)
            },
            "event_name": "randomize_table_color_event",
            "mesh_name": "visuals/vention_mat"  # Empty string to target the whole asset
        },
    )

    randomize_curtain_left_colors = EventTerm(
        func=orbit_mdp.randomize_visual_color,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("curtain_left"),
            "colors": {
                "r": (0.0, 1.0),
                "g": (0.0, 1.0),
                "b": (0.0, 1.0)
            },
            "event_name": "randomize_curtain_left_color_event",
            "mesh_name": ""  # Empty string to target the whole asset
        },
    )

    randomize_curtain_back_colors = EventTerm(
        func=orbit_mdp.randomize_visual_color,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("curtain_back"),
            "colors": {
                "r": (0.0, 1.0),
                "g": (0.0, 1.0),
                "b": (0.0, 1.0)
            },
            "event_name": "randomize_curtain_left_color_event",
            "mesh_name": ""  # Empty string to target the whole asset
        },
    )

    randomize_curtain_right_colors = EventTerm(
        func=orbit_mdp.randomize_visual_color,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("curtain_right"),
            "colors": {
                "r": (0.0, 1.0),
                "g": (0.0, 1.0),
                "b": (0.0, 1.0)
            },
            "event_name": "randomize_curtain_left_color_event",
            "mesh_name": ""  # Empty string to target the whole asset
        },
    )

    # reset background
    randomize_sky_light = EventTerm(
        func=task_mdp.randomize_hdri,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "light_path": "/World/skyLight",
            "hdri_paths": [
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Clear/evening_road_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Clear/kloppenheim_02_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Clear/mealie_road_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Clear/noon_grass_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Clear/qwantani_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Clear/signal_hill_sunrise_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Clear/sunflowers_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Clear/syferfontein_18d_clear_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Clear/venice_sunset_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Clear/white_cliff_top_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/champagne_castle_1_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/sunflowers_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/table_mountain_1_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Evening/evening_road_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/adams_place_bridge_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/bathroom_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/en_suite_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/entrance_hall_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hotel_room_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/lebombo_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/small_empty_house_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/studio_small_04_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/surgery_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/vulture_hide_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/wooden_lounge_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/ZetoCG_com_WarehouseInterior2b.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Night/kloppenheim_02_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Night/moonlit_golf_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Storm/approaching_storm_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/studio_small_05_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/studio_small_07_4k.hdr"
            ],
            "intensity_range": (0.0, 10000.0)
        },
    )


@configclass
class RGBTiledTrainEventCfg(RGBTiledEventCfg):
    reset_from_init_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "datasets": [
                "furniture_datasets/reaching_init_states_rgb_dataset_preprocessed.pt",
                "furniture_datasets/grasped_init_states_rgb_dataset_preprocessed.pt",
                "furniture_datasets/insertion_init_states_rgb_dataset_preprocessed.pt",
                "furniture_datasets/assembled_grasped_init_states_rgb_dataset_preprocessed.pt"
            ],
            "probs": [0.85, 0.1, 0.05, 0.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "failure_rate_sampling": False
        }
    )


@configclass
class RGBTiledEvalEventCfg(RGBTiledEventCfg):
    reset_from_init_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "datasets": [
                "furniture_datasets/reaching_init_states_rgb_dataset_preprocessed.pt",
            ],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "failure_rate_sampling": False
        }
    )


@configclass
class RGBObservationsCfg:
    @configclass
    class RGBPolicyCfg(ObsGroup):
        """Observations for policy group."""

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

        front_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "rgb",
                "process_image": True,
                "output_size": (224, 224)
            },
        )

        side_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("side_camera"),
                "data_type": "rgb",
                "process_image": True,
                "output_size": (224, 224)
            },
        )

        wrist_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
                "process_image": True,
                "output_size": (224, 224)
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: RGBPolicyCfg = RGBPolicyCfg()


@configclass
class EvalRGBObservationsCfg:
    @configclass
    class EvalRGBPolicyCfg(ObsGroup):
        """Observations for policy group."""

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

        front_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "rgb",
                "process_image": True,
                "output_size": (224, 224)
            },
        )

        side_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("side_camera"),
                "data_type": "rgb",
                "process_image": True,
                "output_size": (224, 224)
            },
        )

        wrist_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
                "process_image": True,
                "output_size": (224, 224)
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: EvalRGBPolicyCfg = EvalRGBPolicyCfg()


@configclass
class DataCollectionObservationsCfg:
    @configclass
    class DataCollectionRGBPolicyCfg(ObsGroup):
        """Observations for policy group that combines base observations, RGB observations, and additional features."""
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

        last_processed_arm_action = ObsTerm(
            func=task_mdp.last_processed_action,
            params={
                "action_name": "jointpos",
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

        front_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "rgb",
                # Don't process image since we want save it as int8
                "process_image": False,
                "output_size": (224, 224)
            },
        )

        side_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("side_camera"),
                "data_type": "rgb",
                # Don't process image since we want save it as int8
                "process_image": False,
                "output_size": (224, 224)
            },
        )

        wrist_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
                # Don't process image since we want save it as int8
                "process_image": False,
                "output_size": (224, 224)
            },
        )

        # extra
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

    # observation groups
    policy: DataCollectionRGBPolicyCfg = DataCollectionRGBPolicyCfg()


@configclass
class RGBTiledCommandsCfg:
    """Command specifications for the MDP."""

    task_command = task_mdp.TaskCommandCfg(
        asset_cfg=SceneEntityCfg("robot", body_names="body"),
        resampling_time_range=(1e6, 1e6),
        success_threshold=0.003, # looser success threshold for RGB
        held_asset_cfg=SceneEntityCfg("leg"),
        fixed_asset_cfg=SceneEntityCfg("table_top"),
        held_asset_offset=assembly_data.KEYPOINTS_TABLELEG.center_axis_bottom,
        fixed_asset_offset=assembly_data.KEYPOINTS_TABLETOPHOLE.hole0_leg_assembled_offset,
    )


@configclass
class OneLegUr5RGB(OneLegUr5):
    scene: RGBTiledObjectSceneCfg = RGBTiledObjectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    commands: RGBTiledCommandsCfg = RGBTiledCommandsCfg()

@configclass
class OneLegUr5RGBRelJointPosition(OneLegUr5RGB):
    observations: RGBObservationsCfg = RGBObservationsCfg()
    actions: ur5.Ur5RelativeJointPositionAction = ur5.Ur5RelativeJointPositionAction()
    events: RGBTiledTrainEventCfg = RGBTiledTrainEventCfg()
    commands: RGBTiledCommandsCfg = RGBTiledCommandsCfg()

@configclass
class OneLegUr5DataCollectionRGBRelJointPosition(OneLegUr5RGB):
    observations: DataCollectionObservationsCfg = DataCollectionObservationsCfg()
    actions: ur5.Ur5RelativeJointPositionActionClipped = ur5.Ur5RelativeJointPositionActionClipped()
    terminations: DataCollectionTerminationsCfg = DataCollectionTerminationsCfg()
    events: RGBTiledTrainEventCfg = RGBTiledTrainEventCfg()
    commands: RGBTiledCommandsCfg = RGBTiledCommandsCfg()

@configclass
class OneLegUr5EvalRGBRelUnscaledJointPosition(OneLegUr5RGB):
    observations: EvalRGBObservationsCfg = EvalRGBObservationsCfg()
    actions: ur5.Ur5RelativeJointPositionActionUnscaled = ur5.Ur5RelativeJointPositionActionUnscaled()
    terminations: DataCollectionTerminationsCfg = DataCollectionTerminationsCfg()
    events: RGBTiledEvalEventCfg = RGBTiledEvalEventCfg()
    commands: RGBTiledCommandsCfg = RGBTiledCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 32.0

@configclass
class OneLegUr5EvalRGBAbsJointPosition(OneLegUr5RGB):
    observations: EvalRGBObservationsCfg = EvalRGBObservationsCfg()
    actions: ur5.Ur5JointPositionAction = ur5.Ur5JointPositionAction()
    terminations: DataCollectionTerminationsCfg = DataCollectionTerminationsCfg()
    events: RGBTiledEvalEventCfg = RGBTiledEvalEventCfg()
    commands: RGBTiledCommandsCfg = RGBTiledCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 32.0
