# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Various action terms that can be used in the environment."""

from isaaclab.utils.lazy_imports import lazy_export

lazy_export(
    ("actions_cfg", [
        "JointActionCfg",
        "JointPositionActionCfg",
        "RelativeJointPositionActionCfg",
        "JointVelocityActionCfg",
        "JointEffortActionCfg",
        "JointPositionToLimitsActionCfg",
        "EMAJointPositionToLimitsActionCfg",
        "BinaryJointActionCfg",
        "BinaryJointPositionActionCfg",
        "BinaryJointVelocityActionCfg",
        "AbsBinaryJointPositionActionCfg",
        "NonHolonomicActionCfg",
        "DifferentialInverseKinematicsActionCfg",
        "OperationalSpaceControllerActionCfg",
        "SurfaceGripperBinaryActionCfg",
    ]),
    ("binary_joint_actions", ["BinaryJointAction", "BinaryJointPositionAction", "BinaryJointVelocityAction", "AbsBinaryJointPositionAction"]),
    ("joint_actions", ["JointAction", "JointPositionAction", "RelativeJointPositionAction", "JointVelocityAction", "JointEffortAction"]),
    ("joint_actions_to_limits", ["JointPositionToLimitsAction", "EMAJointPositionToLimitsAction"]),
    ("non_holonomic_actions", "NonHolonomicAction"),
    ("surface_gripper_actions", "SurfaceGripperBinaryAction"),
)
