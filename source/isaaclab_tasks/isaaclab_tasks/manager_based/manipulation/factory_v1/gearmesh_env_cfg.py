# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .assembly_keypoints import KEYPOINTS_GEARBASE, KEYPOINTS_MEDIUMGEAR
from .factory_env_base import FactoryBaseEnvCfg, FactoryEventCfg, FactoryObservationsCfg, FactoryRewardsCfg


@configclass
class GearMeshObservationsCfg(FactoryObservationsCfg):
    def __post_init__(self):
        # policy
        self.policy.end_effector_vel_lin_ang_b.params["target_asset_cfg"].body_names = "panda_fingertip_centered"
        self.policy.end_effector_pose.params["target_asset_cfg"].body_names = "panda_fingertip_centered"
        self.policy.fixed_asset_in_end_effector_frame.params["target_asset_cfg"] = SceneEntityCfg("gear_base")
        self.policy.fixed_asset_in_end_effector_frame.params["root_asset_cfg"].body_names = "panda_fingertip_centered"
        self.policy.fixed_asset_in_end_effector_frame.params["target_asset_offset"] = (
            KEYPOINTS_GEARBASE.medium_gear_tip_offset
        )
        self.policy.held_asset_in_fixed_asset_frame.params["target_asset_cfg"] = SceneEntityCfg("medium_gear")
        self.policy.held_asset_in_fixed_asset_frame.params["root_asset_cfg"] = SceneEntityCfg("gear_base")
        self.policy.held_asset_in_fixed_asset_frame.params["root_asset_offset"] = (
            KEYPOINTS_GEARBASE.medium_gear_tip_offset
        )
        
        self.critic.end_effector_vel_lin_ang_b.params["target_asset_cfg"].body_names = "panda_fingertip_centered"
        self.critic.end_effector_pose.params["target_asset_cfg"].body_names = "panda_fingertip_centered"
        self.critic.fixed_asset_in_end_effector_frame.params["target_asset_cfg"] = SceneEntityCfg("gear_base")
        self.critic.fixed_asset_in_end_effector_frame.params["root_asset_cfg"].body_names = "panda_fingertip_centered"
        self.critic.fixed_asset_in_end_effector_frame.params["target_asset_offset"] = (
            KEYPOINTS_GEARBASE.medium_gear_tip_offset
        )
        self.critic.held_asset_in_fixed_asset_frame.params["target_asset_cfg"] = SceneEntityCfg("medium_gear")
        self.critic.held_asset_in_fixed_asset_frame.params["root_asset_cfg"] = SceneEntityCfg("gear_base")
        self.critic.held_asset_in_fixed_asset_frame.params["root_asset_offset"] = (
            KEYPOINTS_GEARBASE.medium_gear_tip_offset
        )


@configclass
class GearMeshEventCfg(FactoryEventCfg):
    def __post_init__(self):
        # For asset_material
        self.held_asset_material.params["asset_cfg"] = SceneEntityCfg("medium_gear")
        self.fixed_asset_material.params["asset_cfg"] = SceneEntityCfg("gear_base")

        # For reset_fixed_asset
        self.reset_fixed_asset.params["asset_list"] = ["gear_base", "large_gear", "small_gear"]

        scene_staging = self.staging.params["terms"]["scene_staging"].params["terms"]
        player_enters = self.staging.params["terms"]["player_enters"].params["terms"]
        task_assigning = self.staging.params["terms"]["task_assigning"].params["terms"]
        player_prepares_for_task = self.staging.params["terms"]["player_prepares_for_task"].params["terms"]
        
        scene_staging["reset_held_asset_on_fixed_asset"].params["held_asset_cfg"] = SceneEntityCfg("medium_gear")
        scene_staging["reset_held_asset_on_fixed_asset"].params["fixed_asset_cfg"] = SceneEntityCfg("gear_base")
        scene_staging["reset_held_asset_on_fixed_asset"].params["assembled_offset"] = KEYPOINTS_GEARBASE.medium_gear_assembled_bottom_offset
        scene_staging["reset_held_asset_on_fixed_asset"].params["entry_offset"] = KEYPOINTS_GEARBASE.medium_gear_tip_offset
        scene_staging["reset_held_asset_on_fixed_asset"].params["assembly_fraction_range"] = (0.3, 1.)  # 0.6 hits the nistboard
        scene_staging["reset_held_asset_on_fixed_asset"].params["assembly_ratio"] = (0., 0., 0.)
        scene_staging["reset_asset_on_table"].params["asset_cfg"] = SceneEntityCfg("medium_gear")
        scene_staging["reset_asset_in_air"].params["asset_cfg"] = SceneEntityCfg("medium_gear")
        
        task_assigning["reset_end_effector_around_fixed_asset"].params["fixed_asset_cfg"] = SceneEntityCfg("gear_base")
        task_assigning["reset_end_effector_around_fixed_asset"].params["fixed_asset_offset"] = KEYPOINTS_GEARBASE.medium_gear_tip_offset
        task_assigning["reset_end_effector_around_fixed_asset"].params["robot_ik_cfg"].joint_names = ["panda_joint.*"]
        task_assigning["reset_end_effector_around_fixed_asset"].params["robot_ik_cfg"].body_names = "panda_fingertip_centered"
        task_assigning["reset_end_effector_around_fixed_asset"].params["pose_range_b"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (0.035, 0.045),
            "roll": (3.141, 3.141),
            "yaw": (-0.785, 0.785),
        }
        
        
        move_and_grasb_asset = player_prepares_for_task["move_and_grasb_asset"].params["terms"]
        move_and_grasb_asset["reset_end_effector_around_held_asset"].params["fixed_asset_cfg"] = SceneEntityCfg("medium_gear")
        move_and_grasb_asset["reset_end_effector_around_held_asset"].params["fixed_asset_offset"] = KEYPOINTS_MEDIUMGEAR.grasp_point
        move_and_grasb_asset["reset_end_effector_around_held_asset"].params["robot_ik_cfg"].joint_names = ["panda_joint.*"]
        move_and_grasb_asset["reset_end_effector_around_held_asset"].params["robot_ik_cfg"].body_names = "panda_fingertip_centered"
        move_and_grasb_asset["reset_end_effector_around_held_asset"].params["pose_range_b"] = {
            "z": (0.0, 0.0),
            "roll": (3.141, 3.141),
            "yaw": (1.57, 2.09),
        }
        move_and_grasb_asset["grasp_held_asset"].params["robot_cfg"].body_names = "panda_fingertip_centered"
        move_and_grasb_asset["grasp_held_asset"].params["robot_cfg"].joint_names = "panda_finger_joint[1-2]"
        move_and_grasb_asset["grasp_held_asset"].params["held_asset_diameter"] = KEYPOINTS_MEDIUMGEAR.grasp_diameter
        
        move_held_asset_in_hand_then_grasp = player_prepares_for_task["move_held_asset_in_hand_then_grasp"].params["terms"]
        move_held_asset_in_hand_then_grasp["reset_held_asset_in_hand"].params["holding_body_cfg"].body_names = "panda_fingertip_centered"
        move_held_asset_in_hand_then_grasp["reset_held_asset_in_hand"].params["held_asset_cfg"] = SceneEntityCfg("medium_gear")
        move_held_asset_in_hand_then_grasp["reset_held_asset_in_hand"].params["held_asset_graspable_offset"] = KEYPOINTS_MEDIUMGEAR.grasp_point
        move_held_asset_in_hand_then_grasp["grasp_held_asset"].params["robot_cfg"].body_names = "panda_fingertip_centered"
        move_held_asset_in_hand_then_grasp["grasp_held_asset"].params["robot_cfg"].joint_names = "panda_finger_joint[1-2]"
        move_held_asset_in_hand_then_grasp["grasp_held_asset"].params["held_asset_diameter"] = KEYPOINTS_MEDIUMGEAR.grasp_diameter
        


@configclass
class GearMeshRewardsCfg(FactoryRewardsCfg):
    def __post_init__(self):
        # For progress_context
        self.progress_context.params["fixed_asset_cfg"] = SceneEntityCfg("gear_base")
        self.progress_context.params["held_asset_cfg"] = SceneEntityCfg("medium_gear")
        self.progress_context.params["held_asset_offset"] = KEYPOINTS_MEDIUMGEAR.center_axis_bottom
        self.progress_context.params["fixed_asset_offset"] = KEYPOINTS_GEARBASE.medium_gear_assembled_bottom_offset


@configclass
class GearMeshEnvCfg(FactoryBaseEnvCfg):
    """Configuration for the GearMesh environment."""

    observations: GearMeshObservationsCfg = GearMeshObservationsCfg()
    events: GearMeshEventCfg = GearMeshEventCfg()
    rewards: GearMeshRewardsCfg = GearMeshRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        for asset in ["bolt_m16", "hole_8mm", "nut_m16", "peg_8mm"]:
            delattr(self.scene, asset)
