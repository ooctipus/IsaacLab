# Copyright (c) 2024-2025, The Octi Lab Project Developers.
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    JointPositionActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.utils import configclass
from isaaclab.controllers.differential_ik_cfg import MultiConstraintDifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DefaultJointPositionStaticActionCfg,
    MultiConstraintsDifferentialInverseKinematicsActionCfg,
)

"""
UR5 GRIPPER ACTIONS
"""
UR5_JOINT_POSITION: JointPositionActionCfg = JointPositionActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    scale=1.0,
    use_default_offset=False,
)

UR5_RELATIVE_JOINT_POSITION: RelativeJointPositionActionCfg = RelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    scale={
        "(?!wrist_3_joint).*": 0.02,
        "wrist_3_joint": 0.2,
    },
    use_zero_offset=True,
    # clip={
    #     ".*joint.*": (-0.1, 0.1),
    # },
)

UR5_RELATIVE_JOINT_POSITION_CLIPPED: RelativeJointPositionActionCfg = RelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    scale={
        "(?!wrist_3_joint).*": 0.02,
        "wrist_3_joint": 0.2,
    },
    # clip={
    #     "(?!wrist_3_joint).*": (-0.1, 0.1),
    #     "wrist_3_joint": (-1.0, 1.0),
    # },
    use_zero_offset=True,
)

UR5_RELATIVE_JOINT_POSITION_UNSCALED: RelativeJointPositionActionCfg = RelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    scale=1.0,
    use_zero_offset=True,
)

UR5_MC_IKABSOLUTE_ARM = MultiConstraintsDifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    body_name=["robotiq_base_link"],
    controller=MultiConstraintDifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    ),
    scale=1,
)

UR5_MC_IKDELTA_ARM = MultiConstraintsDifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["joint.*"],
    body_name=["robotiq_base_link"],
    controller=MultiConstraintDifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    scale=0.5,
)

ROBOTIQ_GRIPPER_BINARY_ACTIONS = BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["finger_joint"],
    open_command_expr={"finger_joint": 0.0},
    close_command_expr={"finger_joint": 0.785398},
)

ROBOTIQ_COMPLIANT_JOINTS = DefaultJointPositionStaticActionCfg(
    asset_name="robot", joint_names=["left_inner_finger_joint", "right_inner_finger_joint"]
)

ROBOTIQ_MC_IK_ABSOLUTE = MultiConstraintsDifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["joint.*"],
    body_name=["left_inner_finger", "right_inner_finger"],
    controller=MultiConstraintDifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    ),
    scale=1,
)


@configclass
class Ur5IkAbsoluteAction:
    jointpos = UR5_MC_IKABSOLUTE_ARM
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
    compliant_joints = ROBOTIQ_COMPLIANT_JOINTS


@configclass
class Ur5McIkDeltaAction:
    jointpos = UR5_MC_IKDELTA_ARM
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
    compliant_joints = ROBOTIQ_COMPLIANT_JOINTS


@configclass
class Ur5JointPositionAction:
    jointpos = UR5_JOINT_POSITION
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
    compliant_joints = ROBOTIQ_COMPLIANT_JOINTS


@configclass
class Ur5RelativeJointPositionAction:
    jointpos = UR5_RELATIVE_JOINT_POSITION
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
    compliant_joints = ROBOTIQ_COMPLIANT_JOINTS


@configclass
class Ur5RelativeJointPositionActionClipped:
    jointpos = UR5_RELATIVE_JOINT_POSITION_CLIPPED
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
    compliant_joints = ROBOTIQ_COMPLIANT_JOINTS


@configclass
class Ur5RelativeJointPositionActionUnscaled:
    jointpos = UR5_RELATIVE_JOINT_POSITION_UNSCALED
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
    compliant_joints = ROBOTIQ_COMPLIANT_JOINTS
