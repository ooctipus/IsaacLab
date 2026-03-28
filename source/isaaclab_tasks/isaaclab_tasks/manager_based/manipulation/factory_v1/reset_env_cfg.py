# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from . import mdp
from .factory_presets import (
    AssembledOffsetCfg,
    AssemblyFractionFullCfg,
    AssemblyFractionPartialCfg,
    AssemblyRatioCfg,
    EndEffectorBodyCfg,
    EntryOffsetCfg,
    FixedAssetTipCfg,
    GraspedPoseRangeCfg,
    GripperGraspOffsetCfg,
    GripperJointNamesCfg,
    HeldAssetAlignOffsetCfg,
    HeldAssetGraspDiameterCfg,
    HeldAssetGraspMiddleCfg,
    HeldAssetGraspPointCfg,
    IKJointNamesCfg,
)


GRIPPER_GRASP_ASSET_IN_AIR = EventTerm(
    func=mdp.ChainedResetTerms,
    mode="reset",
    params={
        "terms" : {
            "reset_asset_in_air": EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.15, 0.5), "y": (-0.5, 0.5), "z": (0.015, 0.2),
                        "roll": (-1.57, 1.57), "pitch": (-1.57, 1.57), "yaw": (-3.14, 3.14)
                    },
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("held_asset")
                }
            ),
            "reset_end_effector_around_held_asset": EventTerm(
                func=mdp.reset_end_effector_around_asset,
                mode="reset",
                params={
                    "fixed_asset_cfg": SceneEntityCfg("held_asset"),
                    "fixed_asset_offset": HeldAssetGraspMiddleCfg(),
                    "pose_range_b": {
                        "x": (-0.005, 0.005),
                        "y": (-0.005, 0.005),
                        "z": (-0.015, 0.025),
                        "roll": (3.141 - 0.1, 3.141 + 0.1),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-2.09, 2.09),
                    },
                    "robot_ik_cfg": SceneEntityCfg("robot", joint_names=IKJointNamesCfg(), body_names=EndEffectorBodyCfg()),
                    "ik_iterations": (5, 30),
                }
            ),
            "grasp_held_asset": EventTerm(
                func=mdp.grasp_held_asset,
                mode="reset",
                params={
                    "robot_cfg": SceneEntityCfg("robot", joint_names=GripperJointNamesCfg(), body_names=EndEffectorBodyCfg()),
                    "held_asset_diameter": HeldAssetGraspDiameterCfg(),
                }
            ),
        }
    }
)

FULL_ASSEMBLE_FIRST_THEN_GRIPPER_CLOSE = EventTerm(
    func=mdp.ChainedResetTerms,
    mode="reset",
    params={
        "terms" : {
            "reset_held_asset_on_fixed_asset": EventTerm(
                func=mdp.reset_held_asset_on_fixed_asset,
                mode="reset",
                params={
                    "assembled_offset": AssembledOffsetCfg(),
                    "entry_offset": EntryOffsetCfg(),
                    "held_asset_align_offset": HeldAssetAlignOffsetCfg(),
                    "assembly_fraction_range": AssemblyFractionFullCfg(),
                    "assembly_ratio": AssemblyRatioCfg(),
                    "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
                    "held_asset_cfg": SceneEntityCfg("held_asset"),
                }
            ),
            "reset_end_effector_around_held_asset": EventTerm(
                func=mdp.reset_end_effector_around_asset,
                mode="reset",
                params={
                    "fixed_asset_cfg": SceneEntityCfg("held_asset"),
                    "fixed_asset_offset": HeldAssetGraspMiddleCfg(),
                    "pose_range_b": {
                        "x": (-0.005, 0.005),
                        "y": (-0.005, 0.005),
                        "z": (-0.015, 0.025),
                        "roll": (3.141 - 0.1, 3.141 + 0.1),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-2.09, 2.09),
                    },
                    "robot_ik_cfg": SceneEntityCfg("robot", joint_names=IKJointNamesCfg(), body_names=EndEffectorBodyCfg()),
                    "ik_iterations": (15, 25),
                }
            ),
            "grasp_held_asset": EventTerm(
                func=mdp.grasp_held_asset,
                mode="reset",
                params={
                    "robot_cfg": SceneEntityCfg("robot", joint_names=GripperJointNamesCfg(), body_names=EndEffectorBodyCfg()),
                    "held_asset_diameter": HeldAssetGraspDiameterCfg(),
                }
            ),
        }
    }
)

ASSEMBLE_FIRST_THEN_GRIPPER_CLOSE = EventTerm(
    func=mdp.ChainedResetTerms,
    mode="reset",
    params={
        "terms" : {
            "reset_held_asset_on_fixed_asset": EventTerm(
                func=mdp.reset_held_asset_on_fixed_asset,
                mode="reset",
                params={
                    "assembled_offset": AssembledOffsetCfg(),
                    "entry_offset": EntryOffsetCfg(),
                    "held_asset_align_offset": HeldAssetAlignOffsetCfg(),
                    "assembly_fraction_range": AssemblyFractionPartialCfg(),
                    "assembly_ratio": AssemblyRatioCfg(),
                    "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
                    "held_asset_cfg": SceneEntityCfg("held_asset"),
                }
            ),
            "reset_end_effector_around_held_asset": EventTerm(
                func=mdp.reset_end_effector_around_asset,
                mode="reset",
                params={
                    "fixed_asset_cfg": SceneEntityCfg("held_asset"),
                    "fixed_asset_offset": HeldAssetGraspMiddleCfg(),
                    "pose_range_b": {
                        "x": (-0.005, 0.005),
                        "y": (-0.005, 0.005),
                        "z": (-0.015, 0.025),
                        "roll": (3.141 - 0.1, 3.141 + 0.1),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-2.09, 2.09),
                    },
                    "robot_ik_cfg": SceneEntityCfg("robot", joint_names=IKJointNamesCfg(), body_names=EndEffectorBodyCfg()),
                    "ik_iterations": (15, 25),
                }
            ),
            "grasp_held_asset": EventTerm(
                func=mdp.grasp_held_asset,
                mode="reset",
                params={
                    "robot_cfg": SceneEntityCfg("robot", joint_names=GripperJointNamesCfg(), body_names=EndEffectorBodyCfg()),
                    "held_asset_diameter": HeldAssetGraspDiameterCfg(),
                }
            ),
        }
    }
)

GRIPPER_CLOSE_FIRST_THEN_ASSET_IN_GRIPPER = EventTerm(
    func=mdp.ChainedResetTerms,
    mode="reset",
    params={
        "terms" : {
            "reset_end_effector_around_fixed_asset": EventTerm(
                func=mdp.reset_end_effector_around_asset,
                mode="reset",
                params={
                    "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
                    "fixed_asset_offset": FixedAssetTipCfg(),
                    "pose_range_b": GraspedPoseRangeCfg(),
                    "robot_ik_cfg": SceneEntityCfg("robot", joint_names=IKJointNamesCfg(), body_names=EndEffectorBodyCfg()),
                    "ik_iterations": (10, 20)
                }
            ),
            "reset_held_asset_in_hand": EventTerm(
                func=mdp.reset_held_asset_in_gripper,
                mode="reset",
                params={
                    "holding_body_cfg": SceneEntityCfg("robot", body_names=EndEffectorBodyCfg()),
                    "held_asset_cfg": SceneEntityCfg("held_asset"),
                    "held_asset_graspable_offset": HeldAssetGraspPointCfg(),
                    "held_asset_inhand_range": {},
                    "gripper_grasp_offset": GripperGraspOffsetCfg(),
                }
            ),
            "grasp_held_asset": EventTerm(
                func=mdp.grasp_held_asset,
                mode="reset",
                params={
                    "robot_cfg": SceneEntityCfg("robot", joint_names=GripperJointNamesCfg(), body_names=EndEffectorBodyCfg()),
                    "held_asset_diameter": HeldAssetGraspDiameterCfg(),
                    "flexible_angle": False
                }
            ),
        }
    }
)
