# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .assembly_keypoints import KEYPOINTS_HOLE8MM, KEYPOINTS_PEG8MM
from .factory_env_base import FactoryBaseEnvCfg, FactoryEventCfg, FactoryObservationsCfg, FactoryRewardsCfg


@configclass
class PegInsertObservationsCfg(FactoryObservationsCfg):
    def __post_init__(self):
        # policy
        self.policy.end_effector_vel_lin_ang_b.params["target_asset_cfg"].body_names = "panda_fingertip_centered"
        self.policy.end_effector_pose.params["target_asset_cfg"].body_names = "panda_fingertip_centered"
        self.policy.fixed_asset_in_end_effector_frame.params["target_asset_cfg"] = SceneEntityCfg("hole_8mm")
        self.policy.fixed_asset_in_end_effector_frame.params["root_asset_cfg"].body_names = "panda_fingertip_centered"
        self.policy.fixed_asset_in_end_effector_frame.params["target_asset_offset"] = KEYPOINTS_HOLE8MM.hole_tip_offset
        self.policy.held_asset_in_fixed_asset_frame.params["target_asset_cfg"] = SceneEntityCfg("peg_8mm")
        self.policy.held_asset_in_fixed_asset_frame.params["root_asset_cfg"] = SceneEntityCfg("hole_8mm")
        self.policy.held_asset_in_fixed_asset_frame.params["root_asset_offset"] = KEYPOINTS_HOLE8MM.hole_tip_offset
        
        self.critic.end_effector_vel_lin_ang_b.params["target_asset_cfg"].body_names = "panda_fingertip_centered"
        self.critic.end_effector_pose.params["target_asset_cfg"].body_names = "panda_fingertip_centered"
        self.critic.fixed_asset_in_end_effector_frame.params["target_asset_cfg"] = SceneEntityCfg("hole_8mm")
        self.critic.fixed_asset_in_end_effector_frame.params["root_asset_cfg"].body_names = "panda_fingertip_centered"
        self.critic.fixed_asset_in_end_effector_frame.params["target_asset_offset"] = KEYPOINTS_HOLE8MM.hole_tip_offset
        self.critic.held_asset_in_fixed_asset_frame.params["target_asset_cfg"] = SceneEntityCfg("peg_8mm")
        self.critic.held_asset_in_fixed_asset_frame.params["root_asset_cfg"] = SceneEntityCfg("hole_8mm")
        self.critic.held_asset_in_fixed_asset_frame.params["root_asset_offset"] = KEYPOINTS_HOLE8MM.hole_tip_offset


@configclass
class PegInsertEventCfg(FactoryEventCfg):
    def __post_init__(self):
        # For asset_material
        self.held_asset_material.params["asset_cfg"] = SceneEntityCfg("peg_8mm")
        self.fixed_asset_material.params["asset_cfg"] = SceneEntityCfg("hole_8mm")

        # For reset_fixed_asset
        self.reset_fixed_asset.params["asset_list"] = ["hole_8mm"]

        scene_staging = self.staging.params["terms"]["scene_staging"].params["terms"]
        player_enters = self.staging.params["terms"]["player_enters"].params["terms"]
        task_assigning = self.staging.params["terms"]["task_assigning"].params["terms"]
        player_prepares_for_task = self.staging.params["terms"]["player_prepares_for_task"].params["terms"]
        
        scene_staging["reset_held_asset_on_fixed_asset"].params["held_asset_cfg"] = SceneEntityCfg("peg_8mm")
        scene_staging["reset_held_asset_on_fixed_asset"].params["fixed_asset_cfg"] = SceneEntityCfg("hole_8mm")
        scene_staging["reset_held_asset_on_fixed_asset"].params["assembled_offset"] = KEYPOINTS_HOLE8MM.inserted_peg_base_offset
        scene_staging["reset_held_asset_on_fixed_asset"].params["entry_offset"] = KEYPOINTS_HOLE8MM.hole_tip_offset
        scene_staging["reset_held_asset_on_fixed_asset"].params["assembly_fraction_range"] = (0.91, 1.0)  # 0.6 hits the nistboard
        scene_staging["reset_held_asset_on_fixed_asset"].params["assembly_ratio"] = (0., 0., 0.002 / 6.2832)
        scene_staging["reset_asset_on_table"].params["asset_cfg"] = SceneEntityCfg("peg_8mm")
        scene_staging["reset_asset_in_air"].params["asset_cfg"] = SceneEntityCfg("peg_8mm")
        
        task_assigning["reset_end_effector_around_fixed_asset"].params["fixed_asset_cfg"] = SceneEntityCfg("hole_8mm")
        task_assigning["reset_end_effector_around_fixed_asset"].params["fixed_asset_offset"] = KEYPOINTS_HOLE8MM.hole_tip_offset
        task_assigning["reset_end_effector_around_fixed_asset"].params["robot_ik_cfg"].joint_names = ["panda_joint.*"]
        task_assigning["reset_end_effector_around_fixed_asset"].params["robot_ik_cfg"].body_names = "panda_fingertip_centered"
        task_assigning["reset_end_effector_around_fixed_asset"].params["pose_range_b"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (0.015, 0.025),
            "roll": (3.141, 3.141),
            "yaw": (1.57, 2.09),
        }
        
                
        move_and_grasb_asset = player_prepares_for_task["move_and_grasb_asset"].params["terms"]
        move_and_grasb_asset["reset_end_effector_around_held_asset"].params["fixed_asset_cfg"] = SceneEntityCfg("peg_8mm")
        move_and_grasb_asset["reset_end_effector_around_held_asset"].params["fixed_asset_offset"] = KEYPOINTS_PEG8MM.center_axis_middle
        move_and_grasb_asset["reset_end_effector_around_held_asset"].params["robot_ik_cfg"].joint_names = ["panda_joint.*"]
        move_and_grasb_asset["reset_end_effector_around_held_asset"].params["robot_ik_cfg"].body_names = "panda_fingertip_centered"
        move_and_grasb_asset["reset_end_effector_around_held_asset"].params["pose_range_b"] = {
            "z": (-0.005, -0.005),
            "roll": (3.141, 3.141),
            "yaw": (1.57, 2.09),
        }
        move_and_grasb_asset["grasp_held_asset"].params["robot_cfg"].body_names = "panda_fingertip_centered"
        move_and_grasb_asset["grasp_held_asset"].params["robot_cfg"].joint_names = "panda_finger_joint[1-2]"
        move_and_grasb_asset["grasp_held_asset"].params["held_asset_diameter"] = KEYPOINTS_PEG8MM.grasp_diameter
        
        move_held_asset_in_hand_then_grasp = player_prepares_for_task["move_held_asset_in_hand_then_grasp"].params["terms"]
        move_held_asset_in_hand_then_grasp["reset_held_asset_in_hand"].params["holding_body_cfg"].body_names = "panda_fingertip_centered"
        move_held_asset_in_hand_then_grasp["reset_held_asset_in_hand"].params["held_asset_cfg"] = SceneEntityCfg("peg_8mm")
        move_held_asset_in_hand_then_grasp["reset_held_asset_in_hand"].params["held_asset_graspable_offset"] = KEYPOINTS_PEG8MM.grasp_point
        move_held_asset_in_hand_then_grasp["grasp_held_asset"].params["robot_cfg"].body_names = "panda_fingertip_centered"
        move_held_asset_in_hand_then_grasp["grasp_held_asset"].params["robot_cfg"].joint_names = "panda_finger_joint[1-2]"
        move_held_asset_in_hand_then_grasp["grasp_held_asset"].params["held_asset_diameter"] = KEYPOINTS_PEG8MM.grasp_diameter


@configclass
class PegInsertRewardsCfg(FactoryRewardsCfg):
    def __post_init__(self):
        # For progress_context
        self.progress_context.params["fixed_asset_cfg"] = SceneEntityCfg("hole_8mm")
        self.progress_context.params["held_asset_cfg"] = SceneEntityCfg("peg_8mm")
        self.progress_context.params["held_asset_offset"] = KEYPOINTS_PEG8MM.center_axis_bottom
        self.progress_context.params["fixed_asset_offset"] = KEYPOINTS_HOLE8MM.inserted_peg_base_offset


@configclass
class PegInsertEnvCfg(FactoryBaseEnvCfg):
    """Configuration for the PegInsert environment."""

    observations: PegInsertObservationsCfg = PegInsertObservationsCfg()
    events: PegInsertEventCfg = PegInsertEventCfg()
    rewards: PegInsertRewardsCfg = PegInsertRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        for asset in ["hole_8mm", "gear_base", "small_gear", "large_gear", "medium_gear", "peg_8mm"]:
            delattr(self.scene, asset)
