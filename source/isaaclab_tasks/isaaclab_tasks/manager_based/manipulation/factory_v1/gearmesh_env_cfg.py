# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .assembly_keypoints import KEYPOINTS_GEARBASE as KP_GEARBASE
from .assembly_keypoints import KEYPOINTS_MEDIUMGEAR as KP_MEDIUMGEAR
from .factory_env_base import FactoryBaseEnvCfg, FactoryBaseSuccessTerminateEnvCfg


@configclass
class GearMeshObservationsMixinCfg:
    def __post_init__(self: FactoryBaseEnvCfg):
        super().__post_init__()
        # policy
        policy = self.observations.policy
        policy.end_effector_vel_lin_ang_b.params["target_asset_cfg"].body_names = "panda_fingertip_centered"
        policy.end_effector_pose.params["target_asset_cfg"].body_names = "panda_fingertip_centered"
        policy.fixed_asset_in_end_effector_frame.params["target_asset_cfg"] = SceneEntityCfg("gear_base")
        policy.fixed_asset_in_end_effector_frame.params["root_asset_cfg"].body_names = "panda_fingertip_centered"
        policy.fixed_asset_in_end_effector_frame.params["target_asset_offset"] = KP_GEARBASE.medium_gear_tip_offset
        policy.held_asset_in_fixed_asset_frame.params["target_asset_cfg"] = SceneEntityCfg("medium_gear")
        policy.held_asset_in_fixed_asset_frame.params["root_asset_cfg"] = SceneEntityCfg("gear_base")
        policy.held_asset_in_fixed_asset_frame.params["root_asset_offset"] = KP_GEARBASE.medium_gear_tip_offset

        critic = self.observations.critic
        critic.end_effector_vel_lin_ang_b.params["target_asset_cfg"].body_names = "panda_fingertip_centered"
        critic.end_effector_pose.params["target_asset_cfg"].body_names = "panda_fingertip_centered"
        critic.fixed_asset_in_end_effector_frame.params["target_asset_cfg"] = SceneEntityCfg("gear_base")
        critic.fixed_asset_in_end_effector_frame.params["root_asset_cfg"].body_names = "panda_fingertip_centered"
        critic.fixed_asset_in_end_effector_frame.params["target_asset_offset"] = KP_GEARBASE.medium_gear_tip_offset
        critic.held_asset_in_fixed_asset_frame.params["target_asset_cfg"] = SceneEntityCfg("medium_gear")
        critic.held_asset_in_fixed_asset_frame.params["root_asset_cfg"] = SceneEntityCfg("gear_base")
        critic.held_asset_in_fixed_asset_frame.params["root_asset_offset"] = KP_GEARBASE.medium_gear_tip_offset


@configclass
class GearMeshEventMixinCfg:
    def __post_init__(self: FactoryBaseEnvCfg):
        super().__post_init__()
        # For asset_material
        events = self.events
        events.held_asset_material.params["asset_cfg"] = SceneEntityCfg("medium_gear")
        events.fixed_asset_material.params["asset_cfg"] = SceneEntityCfg("gear_base")

        # For reset_fixed_asset
        events.reset_fixed_asset.params["asset_list"] = ["gear_base", "large_gear", "small_gear"]

        if "start_assembled" in events.reset_strategies.params["terms"]:
            reset_s1: dict = events.reset_strategies.params["terms"]["start_assembled"].params["terms"]
            # For reset held_asset on fixed_asset
            reset_s1["reset_held_asset_on_fixed_asset"].params["held_asset_cfg"] = SceneEntityCfg("medium_gear")
            reset_s1["reset_held_asset_on_fixed_asset"].params["fixed_asset_cfg"] = SceneEntityCfg("gear_base")
            reset_s1["reset_held_asset_on_fixed_asset"].params["assembled_offset"] = KP_GEARBASE.medium_gear_assembled_bottom_offset
            reset_s1["reset_held_asset_on_fixed_asset"].params["entry_offset"] = KP_GEARBASE.medium_gear_tip_offset
            reset_s1["reset_held_asset_on_fixed_asset"].params["held_asset_align_offset"] = KP_MEDIUMGEAR.center_axis_bottom
            reset_s1["reset_held_asset_on_fixed_asset"].params["assembly_fraction_range"] = (0.3, 1.)
            reset_s1["reset_held_asset_on_fixed_asset"].params["assembly_ratio"] = (0., 0., 0.)

            reset_s1["reset_end_effector_around_held_asset"].params["fixed_asset_cfg"] = SceneEntityCfg("medium_gear")
            reset_s1["reset_end_effector_around_held_asset"].params["fixed_asset_offset"] = KP_MEDIUMGEAR.grasp_point
            reset_s1["reset_end_effector_around_held_asset"].params["robot_ik_cfg"].joint_names = ["panda_joint.*"]
            reset_s1["reset_end_effector_around_held_asset"].params["robot_ik_cfg"].body_names = "panda_fingertip_centered"
            reset_s1["reset_end_effector_around_held_asset"].params["pose_range_b"] = {
                "x": (-0.005, 0.005),
                "y": (-0.005, 0.005),
                "z": (-0.015, 0.025),
                "roll": (3.141 - 0.1, 3.141 + 0.1),
                "pitch": (-0.5, 0.5),
                "yaw": (-2.09, 2.09),
            }

            reset_s1["grasp_held_asset"].params["robot_cfg"].body_names = "panda_fingertip_centered"
            reset_s1["grasp_held_asset"].params["robot_cfg"].joint_names = "panda_finger_joint[1-2]"
            reset_s1["grasp_held_asset"].params["held_asset_diameter"] = KP_MEDIUMGEAR.grasp_diameter

        if "start_grasped_then_assembled" in events.reset_strategies.params["terms"]:
            reset_s2: dict = events.reset_strategies.params["terms"]["start_grasped_then_assembled"].params["terms"]
            # For reset_hand
            reset_s2["reset_end_effector_around_fixed_asset"].params["fixed_asset_cfg"] = SceneEntityCfg("gear_base")
            reset_s2["reset_end_effector_around_fixed_asset"].params["fixed_asset_offset"] = KP_GEARBASE.medium_gear_tip_offset
            reset_s2["reset_end_effector_around_fixed_asset"].params["robot_ik_cfg"].joint_names = ["panda_joint.*"]
            reset_s2["reset_end_effector_around_fixed_asset"].params["robot_ik_cfg"].body_names = "panda_fingertip_centered"
            reset_s2["reset_end_effector_around_fixed_asset"].params["pose_range_b"] = {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (0.035, 0.045),
                "roll": (3.141, 3.141),
                "pitch": (-0.5, 0.5),
                "yaw": (-2.09, 2.09),
            }

            # For reset_held_asset
            reset_s2["reset_held_asset_in_hand"].params["holding_body_cfg"].body_names = "panda_fingertip_centered"
            reset_s2["reset_held_asset_in_hand"].params["held_asset_cfg"] = SceneEntityCfg("medium_gear")
            reset_s2["reset_held_asset_in_hand"].params["held_asset_graspable_offset"] = KP_MEDIUMGEAR.grasp_point

            # For grasp_held_assset
            reset_s2["grasp_held_asset"].params["robot_cfg"].body_names = "panda_fingertip_centered"
            reset_s2["grasp_held_asset"].params["robot_cfg"].joint_names = "panda_finger_joint[1-2]"
            reset_s2["grasp_held_asset"].params["held_asset_diameter"] = KP_MEDIUMGEAR.grasp_diameter

        if "grasp_asset_in_air" in events.reset_strategies.params["terms"]:
            reset_s3: dict = events.reset_strategies.params["terms"]["grasp_asset_in_air"].params["terms"]
            reset_s3["reset_asset_in_air"].params["asset_cfg"] = SceneEntityCfg("medium_gear")
            reset_s3["reset_end_effector_around_held_asset"].params["fixed_asset_cfg"] = SceneEntityCfg("medium_gear")
            reset_s3["reset_end_effector_around_held_asset"].params["fixed_asset_offset"] = KP_MEDIUMGEAR.grasp_point
            reset_s3["reset_end_effector_around_held_asset"].params["robot_ik_cfg"].joint_names = ["panda_joint.*"]
            reset_s3["reset_end_effector_around_held_asset"].params["robot_ik_cfg"].body_names = "panda_fingertip_centered"
            reset_s3["reset_end_effector_around_held_asset"].params["pose_range_b"] = {
                "x": (-0.005, 0.005),
                "y": (-0.005, 0.005),
                "z": (-0.015, 0.025),
                "roll": (3.141 - 0.1, 3.141 + 0.1),
                "pitch": (-0.5, 0.5),
                "yaw": (-2.09, 2.09),
            }

            reset_s3["grasp_held_asset"].params["robot_cfg"].body_names = "panda_fingertip_centered"
            reset_s3["grasp_held_asset"].params["robot_cfg"].joint_names = "panda_finger_joint[1-2]"
            reset_s3["grasp_held_asset"].params["held_asset_diameter"] = KP_MEDIUMGEAR.grasp_diameter

        if "start_fully_assembled" in events.reset_strategies.params["terms"]:
            reset_s4: dict = events.reset_strategies.params["terms"]["start_fully_assembled"].params["terms"]
            reset_s4["reset_held_asset_on_fixed_asset"].params["held_asset_cfg"] = SceneEntityCfg("medium_gear")
            reset_s4["reset_held_asset_on_fixed_asset"].params["fixed_asset_cfg"] = SceneEntityCfg("gear_base")
            reset_s4["reset_held_asset_on_fixed_asset"].params["assembled_offset"] = KP_GEARBASE.medium_gear_assembled_bottom_offset
            reset_s4["reset_held_asset_on_fixed_asset"].params["entry_offset"] = KP_GEARBASE.medium_gear_tip_offset
            reset_s4["reset_held_asset_on_fixed_asset"].params["held_asset_align_offset"] = KP_MEDIUMGEAR.center_axis_bottom
            reset_s4["reset_held_asset_on_fixed_asset"].params["assembly_fraction_range"] = (0.1, 0.5)  # 0.6 hits the nistboard
            reset_s4["reset_held_asset_on_fixed_asset"].params["assembly_ratio"] = (0., 0., 0.)

            reset_s4["reset_end_effector_around_held_asset"].params["fixed_asset_cfg"] = SceneEntityCfg("medium_gear")
            reset_s4["reset_end_effector_around_held_asset"].params["fixed_asset_offset"] = KP_MEDIUMGEAR.grasp_point
            reset_s4["reset_end_effector_around_held_asset"].params["robot_ik_cfg"].joint_names = ["panda_joint.*"]
            reset_s4["reset_end_effector_around_held_asset"].params["robot_ik_cfg"].body_names = "panda_fingertip_centered"
            reset_s4["reset_end_effector_around_held_asset"].params["pose_range_b"] = {
                "x": (-0.005, 0.005),
                "y": (-0.005, 0.005),
                "z": (-0.015, 0.025),
                "roll": (3.141 - 0.1, 3.141 + 0.1),
                "pitch": (-0.5, 0.5),
                "yaw": (-2.09, 2.09),
            }

            reset_s4["grasp_held_asset"].params["robot_cfg"].body_names = "panda_fingertip_centered"
            reset_s4["grasp_held_asset"].params["robot_cfg"].joint_names = "panda_finger_joint[1-2]"
            reset_s4["grasp_held_asset"].params["held_asset_diameter"] = KP_MEDIUMGEAR.grasp_diameter


@configclass
class GearMeshTerminationsMixinCfg:
    def __post_init__(self: FactoryBaseEnvCfg):
        super().__post_init__()
        # For progress_context
        terminations = self.terminations
        terminations.progress_context.params["fixed_asset_cfg"] = SceneEntityCfg("gear_base")
        terminations.progress_context.params["held_asset_cfg"] = SceneEntityCfg("medium_gear")
        terminations.progress_context.params["held_asset_offset"] = KP_MEDIUMGEAR.center_axis_bottom
        terminations.progress_context.params["fixed_asset_offset"] = KP_GEARBASE.medium_gear_assembled_bottom_offset


@configclass
class GearMeshEnvCfg(
    GearMeshObservationsMixinCfg,
    GearMeshEventMixinCfg,
    GearMeshTerminationsMixinCfg,
    FactoryBaseEnvCfg
):
    """Configuration for the GearMesh environment."""
    def __post_init__(self):
        super().__post_init__()
        self.rewards.reach_reward.params["held_asset_cfg"] = SceneEntityCfg("medium_gear")
        self.terminations.oob.params["asset_cfg"] = SceneEntityCfg("nut_m16")
        for asset in ["bolt_m16", "hole_8mm", "nut_m16", "peg_8mm"]:
            delattr(self.scene, asset)


@configclass
class GearMeshSuccessTerminateEnvCfg(
    GearMeshObservationsMixinCfg,
    GearMeshEventMixinCfg,
    GearMeshTerminationsMixinCfg,
    FactoryBaseSuccessTerminateEnvCfg
):
    """Configuration for the GearMesh environment."""
    def __post_init__(self):
        super().__post_init__()
        self.terminations.oob.params["asset_cfg"] = SceneEntityCfg("medium_gear")
        for asset in ["bolt_m16", "hole_8mm", "nut_m16", "peg_8mm"]:
            delattr(self.scene, asset)
